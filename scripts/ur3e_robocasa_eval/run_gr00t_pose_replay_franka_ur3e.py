#!/usr/bin/env python3
"""Replay GR00T Franka EEF targets on a fixed-base UR3e.

The Franka rollout runs the original GR00T policy and PandaOmron controller.
For every executed action, this script records the actual OSC target pose
created by RoboSuite. The UR3e rollout then solves that world-frame target
pose with numerical IK and advances the scene with the matching gripper
command. This keeps the experiment separate from all normal evaluators.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robocasa.utils import env_utils
from robocasa.utils.dataset_registry_utils import get_task_horizon
from robosuite.utils import transform_utils as T

from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy

from scripts.long_horizon_controller.policy_adapters import Gr00tPolicyAdapter
from scripts.long_horizon_controller.run_composite_seen_eval import (
    DEFAULT_TASK_INSTRUCTIONS,
)
from scripts.ur3e_robocasa_eval.run_gr00t_franka_trajectory import (
    raw_to_policy_observation,
)
from scripts.ur3e_robocasa_eval.run_xiaomi_ur3e_fixed_eval import (
    _current_eef_pose,
    _resize_frame,
    align_initial_eef_to,
    register_ur3e,
)


DEFAULT_MODEL_PATH = (
    "/data/zjw/workspace/Isaac-GR00T/expdata/"
    "foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000"
)


def _first(value: Any) -> np.ndarray:
    array = np.asarray(value)
    while array.ndim > 1 and array.shape[0] == 1:
        array = array[0]
    return np.asarray(array)


def _grip_site_pose(env: Any) -> tuple[np.ndarray, np.ndarray]:
    """Read the grip-site pose used by RoboSuite's OSC controller."""
    robot = env.robots[0]
    site_id = robot.eef_site_id["right"]
    position = np.asarray(env.sim.data.site_xpos[site_id], dtype=np.float64).copy()
    rotation = np.asarray(
        env.sim.data.site_xmat[site_id], dtype=np.float64
    ).reshape(3, 3)
    quaternion = T.mat2quat(rotation)
    quaternion /= max(np.linalg.norm(quaternion), 1e-12)
    return position, quaternion


def _gripper_qpos(raw: dict[str, Any]) -> list[float]:
    return _first(raw["robot0_gripper_qpos"]).astype(np.float64).reshape(-1).tolist()


def _camera_frame(
    env: Any,
    camera_name: str,
    width: int,
    height: int,
) -> np.ndarray:
    frame = np.asarray(
        env.sim.render(height=height, width=width, camera_name=camera_name),
        dtype=np.uint8,
    )
    return frame[::-1, :, :].copy()


def _write_video_frame(
    writer: Any,
    env: Any,
    camera_name: str,
    width: int,
    height: int,
) -> None:
    writer.append_data(
        _resize_frame(
            _camera_frame(env, camera_name, width, height),
            width,
            height,
        )
    )


def _controller_target_pose(
    env: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Read the world-frame target created by Franka's OSC controller."""
    robot = env.robots[0]
    controller = robot.composite_controller.get_controller("right")
    if controller.goal_pos is None or controller.goal_ori is None:
        raise RuntimeError("Franka OSC controller did not create an EEF goal")
    if controller.origin_pos is None or controller.origin_ori is None:
        raise RuntimeError("Franka OSC controller has no base-frame origin")

    position = np.asarray(controller.origin_pos, dtype=np.float64) + np.asarray(
        controller.origin_ori, dtype=np.float64
    ).dot(np.asarray(controller.goal_pos, dtype=np.float64))
    rotation = np.asarray(controller.origin_ori, dtype=np.float64).dot(
        np.asarray(controller.goal_ori, dtype=np.float64)
    )
    quaternion = T.mat2quat(rotation)
    quaternion /= max(np.linalg.norm(quaternion), 1e-12)
    return position, quaternion


def _record_controller_target(
    env: Any,
    normalized_delta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read Franka's target and independently compute its scaled delta."""
    robot = env.robots[0]
    controller = robot.composite_controller.get_controller("right")
    position, quaternion = _controller_target_pose(env)
    scaled_delta = controller.scale_action(
        np.asarray(normalized_delta[:6], dtype=np.float64)
    )
    return position, quaternion, scaled_delta


def _make_env(
    task: str,
    robot: str,
    seed: int,
    split: str,
    layout_id: int | None,
    style_id: int | None,
    camera_width: int,
    camera_height: int,
) -> Any:
    kwargs: dict[str, Any] = {
        "camera_names": [
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ],
        "camera_widths": camera_width,
        "camera_heights": camera_height,
        "render_camera": "robot0_agentview_left",
        "control_freq": 20,
        "initialization_noise": None,
    }
    if layout_id is None:
        kwargs["split"] = split
    else:
        kwargs.update(
            split=None,
            obj_instance_split=split,
            layout_and_style_ids=[(layout_id, style_id)],
        )
    return env_utils.create_env(task, robots=robot, seed=seed, **kwargs)


def _run_franka(
    args: argparse.Namespace,
    policy: Gr00tPolicyAdapter,
) -> dict[str, Any]:
    env = _make_env(
        args.task,
        "PandaOmron",
        args.seed,
        args.split,
        args.layout_id,
        args.style_id,
        args.camera_width,
        args.camera_height,
    )
    import imageio.v2 as imageio

    writer = imageio.get_writer(
        args.output_root / "franka_gr00t.mp4",
        fps=20,
        codec="libx264",
    )
    eye_writer = imageio.get_writer(
        args.output_root / ".franka_eye_cache.mp4",
        fps=20,
        codec="libx264",
    )
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    horizon = args.max_episode_steps or get_task_horizon(args.task)
    instruction = DEFAULT_TASK_INSTRUCTIONS.get(args.task, args.task)
    success = False
    try:
        raw = env.reset()
        initial_position, initial_quaternion = _grip_site_pose(env)
        initial_gripper = _gripper_qpos(raw)
        if hasattr(policy, "reset"):
            policy.reset()

        for step in range(1, horizon + 1):
            observation = raw_to_policy_observation(raw)
            _, action_chunk = policy.act(observation, instruction)
            action_chunk = np.asarray(action_chunk, dtype=np.float32)
            if action_chunk.ndim != 2 or action_chunk.shape[1] < 7:
                raise ValueError(f"Unexpected GR00T action shape: {action_chunk.shape}")

            stop = False
            for action_row in action_chunk:
                if len(rows) >= horizon:
                    stop = True
                    break
                full_action = np.asarray(
                    action_row[: env.action_dim], dtype=np.float32
                )
                if full_action.shape != (env.action_dim,):
                    raise ValueError(
                        f"GR00T action has {action_row.shape[0]} dims, "
                        f"but Franka expects {env.action_dim}"
                    )
                normalized_delta = full_action[:7]
                raw, _, done, _ = env.step(full_action)
                target_position, target_quaternion, scaled_delta = (
                    _record_controller_target(env, normalized_delta)
                )
                actual_position, actual_quaternion = _grip_site_pose(env)
                rows.append(
                    {
                        "step": len(rows) + 1,
                        "action_12d": np.asarray(action_row[:12], dtype=np.float32).tolist(),
                        "eef_delta_normalized_6d": normalized_delta[:6].tolist(),
                        "eef_delta_scaled_6d": scaled_delta.tolist(),
                        "gripper_action": float(normalized_delta[6]),
                        "target_eef_pos_world": target_position.tolist(),
                        "target_eef_quat_world_xyzw": target_quaternion.tolist(),
                        "actual_eef_pos_world": actual_position.tolist(),
                        "actual_eef_quat_world_xyzw": actual_quaternion.tolist(),
                        "gripper_qpos": _gripper_qpos(raw),
                    }
                )
                _write_video_frame(
                    writer,
                    env,
                    "robot0_agentview_left",
                    args.video_width,
                    args.video_height,
                )
                _write_video_frame(
                    eye_writer,
                    env,
                    "robot0_eye_in_hand",
                    args.video_width,
                    args.video_height,
                )
                success = bool(env._check_success())
                if success or bool(done):
                    stop = True
                    break
            if stop:
                break
    finally:
        writer.close()
        eye_writer.close()
        env.close()

    return {
        "robot": "PandaOmron",
        "initial_eef_pos_world": initial_position.tolist(),
        "initial_eef_quat_world_xyzw": initial_quaternion.tolist(),
        "initial_gripper_qpos": initial_gripper,
        "rows": rows,
        "success": bool(success),
        "elapsed_sec": time.perf_counter() - started,
    }


def _run_ur3e(
    args: argparse.Namespace,
    franka: dict[str, Any],
) -> dict[str, Any]:
    env = _make_env(
        args.task,
        "UR3eOfficialFixed",
        args.seed,
        args.split,
        args.layout_id,
        args.style_id,
        args.camera_width,
        args.camera_height,
    )
    import imageio.v2 as imageio

    writer = imageio.get_writer(
        args.output_root / "ur3e_gr00t_pose_replay.mp4",
        fps=20,
        codec="libx264",
    )
    eye_writer = imageio.get_writer(
        args.output_root / ".ur3e_eye_cache.mp4",
        fps=20,
        codec="libx264",
    )
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    successes = False
    errors: list[str] = []
    try:
        raw = env.reset()
        franka_initial_position = np.asarray(
            franka["initial_eef_pos_world"], dtype=np.float64
        )
        franka_initial_quaternion = np.asarray(
            franka["initial_eef_quat_world_xyzw"], dtype=np.float64
        )
        alignment = align_initial_eef_to(
            env,
            franka_initial_position,
            franka_initial_quaternion,
            allow_base_translation=True,
            orientation_source="site",
        )
        raw = env._get_observations(force_update=True)
        initial_position, initial_quaternion = _grip_site_pose(env)
        initial_gripper = _gripper_qpos(raw)

        for source_row in franka["rows"]:
            target_position = np.asarray(
                source_row["target_eef_pos_world"], dtype=np.float64
            )
            target_quaternion = np.asarray(
                source_row["target_eef_quat_world_xyzw"], dtype=np.float64
            )
            try:
                ik = align_initial_eef_to(
                    env,
                    target_position,
                    target_quaternion,
                    allow_base_translation=False,
                    max_iterations=args.ik_iterations,
                    position_tolerance=args.ik_position_tolerance,
                    orientation_tolerance=args.ik_orientation_tolerance,
                    orientation_source="site",
                )
            except Exception as exc:
                errors.append(f"step {source_row['step']}: {type(exc).__name__}: {exc}")
                break

            # Hold the solved arm pose while applying the Franka gripper
            # command. The controller reset in align_initial_eef_to makes a
            # zero arm delta hold the current IK solution.
            arm_gripper_action = np.zeros(7, dtype=np.float32)
            arm_gripper_action[6] = np.float32(source_row["gripper_action"])
            raw, _, done, _ = env.step(arm_gripper_action)
            actual_position, actual_quaternion = _grip_site_pose(env)
            rows.append(
                {
                    "step": int(source_row["step"]),
                    "source_target_eef_pos_world": target_position.tolist(),
                    "source_target_eef_quat_world_xyzw": target_quaternion.tolist(),
                    "actual_eef_pos_world": actual_position.tolist(),
                    "actual_eef_quat_world_xyzw": actual_quaternion.tolist(),
                    "gripper_action": float(source_row["gripper_action"]),
                    "gripper_qpos": _gripper_qpos(raw),
                    "ik": ik,
                }
            )
            _write_video_frame(
                writer,
                env,
                "robot0_agentview_left",
                args.video_width,
                args.video_height,
            )
            _write_video_frame(
                eye_writer,
                env,
                "robot0_eye_in_hand",
                args.video_width,
                args.video_height,
            )
            successes = bool(env._check_success())
            if successes or bool(done):
                break
    finally:
        writer.close()
        eye_writer.close()
        env.close()

    return {
        "robot": "UR3eOfficialFixed",
        "initial_eef_pos_world": initial_position.tolist(),
        "initial_eef_quat_world_xyzw": initial_quaternion.tolist(),
        "initial_gripper_qpos": initial_gripper,
        "alignment": alignment,
        "rows": rows,
        "success": bool(successes),
        "errors": errors,
        "elapsed_sec": time.perf_counter() - started,
    }


def _quat_error_rad(target: np.ndarray, actual: np.ndarray) -> float:
    return float(
        np.linalg.norm(T.get_orientation_error(target.astype(float), actual.astype(float)))
    )


def _build_comparison_frames(
    args: argparse.Namespace,
    franka: dict[str, Any],
    ur3e: dict[str, Any],
) -> dict[str, Any]:
    """Render five synchronized 2x2 comparison images from both rollouts."""
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw

    out_dir = args.output_root / "comparison_frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    count = min(len(franka["rows"]), len(ur3e["rows"]))
    if count == 0:
        return {"count": 0, "frames": []}
    indices = sorted(set(np.linspace(0, count - 1, min(5, count), dtype=int).tolist()))
    readers = [
        imageio.get_reader(args.output_root / "franka_gr00t.mp4"),
        imageio.get_reader(args.output_root / "ur3e_gr00t_pose_replay.mp4"),
        imageio.get_reader(args.output_root / ".franka_eye_cache.mp4"),
        imageio.get_reader(args.output_root / ".ur3e_eye_cache.mp4"),
    ]
    frame_rows: list[dict[str, Any]] = []
    try:
        for frame_number, index in enumerate(indices):
            source_franka = franka["rows"][index]
            source_ur3e = ur3e["rows"][index]
            images = [
                np.asarray(readers[0].get_data(index), dtype=np.uint8),
                np.asarray(readers[1].get_data(index), dtype=np.uint8),
                np.asarray(readers[2].get_data(index), dtype=np.uint8),
                np.asarray(readers[3].get_data(index), dtype=np.uint8),
            ]

            # Order: top-left Franka agentview, top-right UR3e agentview,
            # bottom-left Franka eye-in-hand, bottom-right UR3e eye-in-hand.
            canvas = Image.new(
                "RGB",
                (args.video_width * 2, args.video_height * 2 + 64),
                color=(245, 245, 245),
            )
            draw = ImageDraw.Draw(canvas)
            labels = [
                "Franka agentview_left",
                "UR3e agentview_left",
                "Franka eye_in_hand",
                "UR3e eye_in_hand",
            ]
            positions = [
                (0, 64),
                (args.video_width, 64),
                (0, args.video_height + 64),
                (args.video_width, args.video_height + 64),
            ]
            for image, label, position in zip(images, labels, positions):
                image_obj = Image.fromarray(image).convert("RGB").resize(
                    (args.video_width, args.video_height)
                )
                canvas.paste(image_obj, position)
                draw.text((position[0] + 10, position[1] + 10), label, fill=(255, 255, 0))
            path = out_dir / f"compare_{frame_number:02d}_step_{index + 1:06d}.png"
            canvas.save(path)

            target = np.asarray(source_franka["target_eef_pos_world"], dtype=np.float64)
            actual = np.asarray(source_ur3e["actual_eef_pos_world"], dtype=np.float64)
            target_q = np.asarray(
                source_franka["target_eef_quat_world_xyzw"], dtype=np.float64
            )
            actual_q = np.asarray(
                source_ur3e["actual_eef_quat_world_xyzw"], dtype=np.float64
            )
            frame_rows.append(
                {
                    "step": int(index + 1),
                    "path": str(path),
                    "ur3e_target_position_error_m": float(np.linalg.norm(actual - target)),
                    "ur3e_target_orientation_error_rad": _quat_error_rad(target_q, actual_q),
                }
            )
    finally:
        for reader in readers:
            reader.close()
    return {"count": len(frame_rows), "frames": frame_rows}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="PreSoakPan")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--split", default="target", choices=["target", "pretrain"])
    parser.add_argument("--data-config", default="panda_omron")
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--layout-id", type=int, default=3)
    parser.add_argument("--style-id", type=int, default=3)
    parser.add_argument("--base-z", type=float, default=0.92)
    parser.add_argument("--base-y-offset", type=float, default=0.0)
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument("--max-episode-steps", type=int, default=1600)
    parser.add_argument("--camera-width", type=int, default=256)
    parser.add_argument("--camera-height", type=int, default=256)
    parser.add_argument("--video-width", type=int, default=512)
    parser.add_argument("--video-height", type=int, default=512)
    parser.add_argument("--ik-iterations", type=int, default=60)
    parser.add_argument("--ik-position-tolerance", type=float, default=0.003)
    parser.add_argument("--ik-orientation-tolerance", type=float, default=0.03)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "expdata/ur3e_robocasa_fixed_gr00t_smoke/"
            "gr00t_pose_replay_presoak_seed1000"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.data_config not in DATA_CONFIG_MAP:
        raise ValueError(f"Unknown data config: {args.data_config}")
    if (args.layout_id is None) != (args.style_id is None):
        raise ValueError("--layout-id and --style-id must be provided together")
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "frame_cache").mkdir(parents=True, exist_ok=True)

    import torch

    register_ur3e(args.base_z, args.base_y_offset)
    data_config = DATA_CONFIG_MAP[args.data_config]
    modality_config = data_config.modality_config()
    model = Gr00tPolicy(
        model_path=args.model_path,
        modality_config=modality_config,
        modality_transform=data_config.transform(),
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    policy = Gr00tPolicyAdapter(
        policy=model,
        action_keys=modality_config["action"].modality_keys,
    )
    franka = _run_franka(args, policy)
    (args.output_root / "franka_rollout.json").write_text(
        json.dumps(franka, indent=2), encoding="utf-8"
    )
    ur3e = _run_ur3e(args, franka)
    (args.output_root / "ur3e_rollout.json").write_text(
        json.dumps(ur3e, indent=2), encoding="utf-8"
    )
    comparison = _build_comparison_frames(args, franka, ur3e)
    for cache_path in (
        args.output_root / ".franka_eye_cache.mp4",
        args.output_root / ".ur3e_eye_cache.mp4",
    ):
        cache_path.unlink(missing_ok=True)
    payload = {
        "task": args.task,
        "seed": args.seed,
        "layout_id": args.layout_id,
        "style_id": args.style_id,
        "model_path": args.model_path,
        "franka": {
            "steps": len(franka["rows"]),
            "success": franka["success"],
        },
        "ur3e": {
            "steps": len(ur3e["rows"]),
            "success": ur3e["success"],
            "alignment": ur3e.get("alignment"),
        },
        "comparison": comparison,
    }
    (args.output_root / "replay_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
