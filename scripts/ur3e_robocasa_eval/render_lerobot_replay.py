#!/usr/bin/env python3
"""Render saved Franka / UR3e LeRobot replay results in one MuJoCo process.

This renderer intentionally handles one robot per invocation. Keeping a
single offscreen context alive avoids the OSMesa context interactions that can
occur when the full replay evaluator creates several environments in one
process.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robocasa.utils import env_utils
from robosuite.utils import transform_utils as T

from scripts.ur3e_robocasa_eval.replay_lerobot_expert_franka_ur3e import (
    _apply_common_scene_state,
    _extract_common_scene_state,
    _gripper_pad_geometry,
    _restore_expert_xml_state,
    _set_recorded_state,
    _source_gripper_mapping,
        _mapped_gripper_signal,
        _install_compatible_grasp_check,
)
import scripts.ur3e_robocasa_eval.replay_lerobot_expert_franka_ur3e as replay_module
from scripts.ur3e_robocasa_eval.run_xiaomi_ur3e_fixed_eval import (
    _resize_frame,
    align_initial_eef_to,
    register_ur3e,
)


def _load_expert(dataset_root: Path, episode_index: int) -> dict[str, Any]:
    import pandas as pd

    episode_dir = dataset_root / "extras" / f"episode_{episode_index:06d}"
    states_path = episode_dir / "states.npz"
    xml_path = episode_dir / "model.xml.gz"
    meta_path = episode_dir / "ep_meta.json"
    parquet_path = sorted(
        dataset_root.glob(f"data/*/episode_{episode_index:06d}.parquet")
    )[0]
    states = np.load(states_path)["states"].astype(np.float64)
    with gzip.open(xml_path, "rt", encoding="utf-8") as handle:
        model_xml = handle.read()
    ep_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    frame_table = pd.read_parquet(parquet_path)
    return {
        "states": states,
        "model_xml": model_xml,
        "ep_meta": ep_meta,
        "actions": len(frame_table),
    }


def _make_env(
    task: str,
    robot: str,
    ep_meta: dict[str, Any],
    seed: int,
    width: int,
    height: int,
) -> Any:
    reset_meta = dict(ep_meta)
    if robot != "PandaOmron":
        reset_meta.pop("init_robot_base_pos", None)
        reset_meta.pop("init_robot_base_ori", None)
    env = env_utils.create_env(
        task,
        robots=robot,
        split=None,
        obj_instance_split="target",
        layout_and_style_ids=[
            (int(ep_meta["layout_id"]), int(ep_meta["style_id"]))
        ],
        seed=seed,
        camera_names=[
            "robot0_agentview_left",
            "robot0_eye_in_hand",
        ],
        camera_widths=width,
        camera_heights=height,
        render_camera="robot0_agentview_left",
        control_freq=20,
        initialization_noise=None,
    )
    env.set_ep_meta(reset_meta)
    env.reset()
    env._replay_camera_width = int(width)
    env._replay_camera_height = int(height)
    return env


def _render(env: Any, camera_name: str, width: int, height: int) -> np.ndarray:
    render_width = int(getattr(env, "_replay_camera_width", width))
    render_height = int(getattr(env, "_replay_camera_height", height))
    frame = np.asarray(
        env.sim.render(
            height=render_height,
            width=render_width,
            camera_name=camera_name,
        ),
        dtype=np.uint8,
    )[::-1, :, :].copy()
    return _resize_frame(frame, width, height)


def _writer(path: Path, fps: int):
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    )


def _save_selected(
    output_dir: Path,
    step: int,
    agentview: np.ndarray,
    eye: np.ndarray,
) -> None:
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(agentview).save(output_dir / f"step_{step:06d}_agentview_left.png")
    Image.fromarray(eye).save(output_dir / f"step_{step:06d}_eye_in_hand.png")


def _render_franka(args: argparse.Namespace, expert: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(args.replay_json.read_text(encoding="utf-8"))
    env = _make_env(
        args.task,
        "PandaOmron",
        expert["ep_meta"],
        args.seed,
        args.camera_width,
        args.camera_height,
    )
    selected = set(np.linspace(0, len(result["rows"]) - 1, 5, dtype=int).tolist())
    writer = _writer(args.output_video, args.fps)
    try:
        _restore_expert_xml_state(env, expert["model_xml"], expert["states"][0])
        for index, row in enumerate(result["rows"]):
            _set_recorded_state(
                env,
                expert["states"][int(row["recorded_state_index"])],
            )
            agentview = _render(
                env,
                "robot0_agentview_left",
                args.video_width,
                args.video_height,
            )
            eye = _render(
                env,
                "robot0_eye_in_hand",
                args.video_width,
                args.video_height,
            )
            writer.append_data(agentview)
            if index in selected:
                _save_selected(
                    args.selected_dir,
                    int(row["step"]),
                    agentview,
                    eye,
                )
    finally:
        writer.close()
        env.close()
    return {"robot": "PandaOmron", "frames": len(result["rows"])}


def _render_ur3e(args: argparse.Namespace, expert: dict[str, Any]) -> dict[str, Any]:
    replay_module._UR3E_CONTROLLER_MODE = "joint_position"
    replay_module._UR3E_JOINT_POSITION_KP = float(args.joint_position_kp)
    replay_module._UR3E_JOINT_POSITION_DAMPING_RATIO = float(
        args.joint_position_damping_ratio
    )
    result = json.loads(args.replay_json.read_text(encoding="utf-8"))
    franka_result = json.loads(args.franka_json.read_text(encoding="utf-8"))
    source_env = _make_env(
        args.task,
        "PandaOmron",
        expert["ep_meta"],
        args.seed,
        64,
        64,
    )
    try:
        _restore_expert_xml_state(source_env, expert["model_xml"], expert["states"][0])
        scene_state = _extract_common_scene_state(source_env, expert["states"][0])
    finally:
        source_env.close()

    register_ur3e(args.base_z, args.base_y_offset)
    env = _make_env(
        args.task,
        "UR3eOfficialFixed",
        expert["ep_meta"],
        args.seed,
        args.camera_width,
        args.camera_height,
    )
    selected = set(np.linspace(0, len(result["rows"]) - 1, 5, dtype=int).tolist())
    writer = _writer(args.output_video, args.fps)
    eye_writer = (
        _writer(args.output_eye_video, args.fps)
        if args.output_eye_video is not None
        else None
    )
    try:
        _apply_common_scene_state(env, scene_state)
        body_collision = result.get("ur3e_body_collision") or {}
        replay_module._configure_ur3e_body_collision(
            env,
            enabled=bool(body_collision.get("enabled", False)),
        )
        alignment = result["alignment"]
        base_id = env.sim.model.body_name2id("robot0_base")
        env.sim.model.body_pos[base_id] = np.asarray(
            alignment["solved_base_position_world"],
            dtype=np.float64,
        )
        env.sim.forward()
        env.sim.data.qpos[env.robots[0]._ref_arm_joint_pos_indexes] = np.asarray(
            alignment["solved_arm_qpos"],
            dtype=np.float64,
        )
        env.sim.data.qvel[env.robots[0]._ref_arm_joint_vel_indexes] = 0.0
        env.sim.forward()
        env.robots[0].composite_controller.update_state()
        env.robots[0].composite_controller.reset()
        gripper_joint_indexes = np.asarray(
            env.robots[0]._ref_gripper_joint_pos_indexes["right"],
            dtype=np.int64,
        )
        env.sim.data.qpos[gripper_joint_indexes] = np.asarray(
            np.zeros(gripper_joint_indexes.size, dtype=np.float64),
            dtype=np.float64,
        )
        env.sim.data.qvel[
            np.asarray(
                env.robots[0]._ref_gripper_joint_vel_indexes["right"],
                dtype=np.int64,
            )
        ] = 0.0
        env.sim.forward()
        env.robots[0].gripper["right"].current_action = np.asarray(
            [-1.0], dtype=np.float64
        )
        gripper_mode = result.get("gripper_control_mode", "state_mapping")
        mapping = result.get("gripper_mapping") or franka_result.get(
            "gripper_mapping"
        ) or {"enabled": False}
        previous_gripper_action: float | None = None
        for index, row in enumerate(result["rows"]):
            signal = row.get("gripper_signal")
            if gripper_mode != "source_action" and signal is None:
                signal = _mapped_gripper_signal(
                    {
                        "gripper_opening_m": row.get(
                            "source_gripper_opening_m"
                        )
                    },
                    mapping,
                )
            if gripper_mode != "source_action" and signal is not None:
                env.robots[0].gripper["right"].current_action = np.asarray(
                    [float(np.clip(signal, -1.0, 1.0))],
                    dtype=np.float64,
                )
            elif (
                row.get("gripper_signal_source")
                in {
                    "legacy_binary_action_immediate_close",
                    "source_binary_action_immediate_close",
                }
                and float(row.get("gripper_action", 0.0)) > 0.0
                and (
                    previous_gripper_action is None
                    or previous_gripper_action <= 0.0
                )
            ):
                # Preserve the one-time tight-close transition recorded by
                # the isolated replay when rendering from its JSON.
                env.robots[0].gripper["right"].current_action = np.asarray(
                    [1.0],
                    dtype=np.float64,
                )
            if (
                result.get("joint_target_application") == "teleport"
                and row.get("q_target") is not None
            ):
                env.sim.data.qpos[
                    env.robots[0]._ref_arm_joint_pos_indexes
                ] = np.asarray(row["q_target"], dtype=np.float64)
                env.sim.data.qvel[
                    env.robots[0]._ref_arm_joint_vel_indexes
                ] = 0.0
                env.sim.forward()
                env.robots[0].composite_controller.update_state()
            action = np.asarray(row["action_7d"], dtype=np.float32)
            env.step(action)
            previous_gripper_action = float(row.get("gripper_action", 0.0))
            agentview = _render(
                env,
                "robot0_agentview_left",
                args.video_width,
                args.video_height,
            )
            eye = _render(
                env,
                "robot0_eye_in_hand",
                args.video_width,
                args.video_height,
            )
            writer.append_data(agentview)
            if eye_writer is not None:
                eye_writer.append_data(eye)
            if index in selected:
                _save_selected(
                    args.selected_dir,
                    int(row["step"]),
                    agentview,
                    eye,
                )
    finally:
        writer.close()
        if eye_writer is not None:
            eye_writer.close()
        env.close()
    return {"robot": "UR3eOfficialFixed", "frames": len(result["rows"])}


def _combine(args: argparse.Namespace) -> None:
    from PIL import Image, ImageDraw

    franka_dir = args.franka_selected_dir
    ur3e_dir = args.ur3e_selected_dir
    steps = sorted(
        int(path.stem.split("_")[1])
        for path in franka_dir.glob("step_*_agentview_left.png")
    )
    args.comparison_dir.mkdir(parents=True, exist_ok=True)
    for number, step in enumerate(steps):
        names = [
            franka_dir / f"step_{step:06d}_agentview_left.png",
            ur3e_dir / f"step_{step:06d}_agentview_left.png",
            franka_dir / f"step_{step:06d}_eye_in_hand.png",
            ur3e_dir / f"step_{step:06d}_eye_in_hand.png",
        ]
        images = [Image.open(path).convert("RGB") for path in names]
        width, height = images[0].size
        canvas = Image.new("RGB", (width * 2, height * 2 + 52), (245, 245, 245))
        draw = ImageDraw.Draw(canvas)
        labels = [
            "Franka agentview_left",
            "UR3e agentview_left",
            "Franka eye_in_hand",
            "UR3e eye_in_hand",
        ]
        positions = [
            (0, 52),
            (width, 52),
            (0, height + 52),
            (width, height + 52),
        ]
        for image, label, position in zip(images, labels, positions):
            canvas.paste(image, position)
            draw.text((position[0] + 8, position[1] + 8), label, fill=(255, 255, 0))
        canvas.save(args.comparison_dir / f"compare_{number:02d}_step_{step:06d}.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=["franka", "ur3e", "combine"], required=True)
    parser.add_argument("--task", default="ScrubCuttingBoard")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--episode-index", type=int, default=376)
    parser.add_argument("--seed", type=int, default=376)
    parser.add_argument("--replay-json", type=Path)
    parser.add_argument("--franka-json", type=Path)
    parser.add_argument("--output-video", type=Path)
    parser.add_argument(
        "--output-eye-video",
        type=Path,
        help="Optional separate MP4 containing the robot0_eye_in_hand view.",
    )
    parser.add_argument("--selected-dir", type=Path)
    parser.add_argument("--franka-selected-dir", type=Path)
    parser.add_argument("--ur3e-selected-dir", type=Path)
    parser.add_argument("--comparison-dir", type=Path)
    parser.add_argument("--base-z", type=float, default=0.92)
    parser.add_argument("--base-y-offset", type=float, default=0.0)
    parser.add_argument("--camera-width", type=int, default=256)
    parser.add_argument("--camera-height", type=int, default=256)
    parser.add_argument("--video-width", type=int, default=512)
    parser.add_argument("--video-height", type=int, default=512)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--joint-position-kp", type=float, default=150.0)
    parser.add_argument("--joint-position-damping-ratio", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _install_compatible_grasp_check()
    if args.robot == "combine":
        _combine(args)
        return
    if args.dataset_root is None:
        raise ValueError("--dataset-root is required for robot rendering")
    if args.replay_json is None or args.output_video is None or args.selected_dir is None:
        raise ValueError("--replay-json, --output-video and --selected-dir are required")
    expert = _load_expert(args.dataset_root, args.episode_index)
    if args.robot == "franka":
        result = _render_franka(args, expert)
    else:
        if args.franka_json is None:
            raise ValueError("--franka-json is required for UR3e rendering")
        result = _render_ur3e(args, expert)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
