#!/usr/bin/env python3
"""Replay one recorded Franka action sequence on Franka and fixed-base UR3e.

The script creates the same RoboCasa scene twice, replays the exact recorded
12D action sequence on PandaOmron and the first 7D arm/gripper dimensions on
UR3e, and compares the resulting end-effector trajectories.
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

from scripts.ur3e_robocasa_eval.run_xiaomi_ur3e_fixed_eval import (
    _current_eef_pose,
    _resize_frame,
    align_initial_eef_to,
    configure_review_camera,
    register_ur3e,
)


def _first(value: Any) -> np.ndarray:
    array = np.asarray(value)
    while array.ndim > 1 and array.shape[0] == 1:
        array = array[0]
    return np.asarray(array)


def _pose_from_raw(raw: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    position = _first(raw["robot0_eef_pos"]).astype(np.float64)
    quaternion = _first(raw["robot0_eef_quat"]).astype(np.float64)
    quaternion /= max(np.linalg.norm(quaternion), 1e-12)
    return position, quaternion


def _quat_to_mat(quaternion_xyzw: np.ndarray) -> np.ndarray:
    return T.quat2mat(np.asarray(quaternion_xyzw, dtype=np.float64))


def _mat_to_quat(rotation: np.ndarray) -> np.ndarray:
    quaternion = T.mat2quat(np.asarray(rotation, dtype=np.float64))
    quaternion /= max(np.linalg.norm(quaternion), 1e-12)
    return quaternion


def _gripper_summary(raw: dict[str, Any]) -> dict[str, Any]:
    values = _first(raw["robot0_gripper_qpos"]).astype(np.float64).reshape(-1)
    return {
        "qpos": values.tolist(),
        "mean": float(np.mean(values)) if values.size else 0.0,
        "min": float(np.min(values)) if values.size else 0.0,
        "max": float(np.max(values)) if values.size else 0.0,
    }


def _scene(env: Any) -> dict[str, Any]:
    return {
        "layout_id": int(env.layout_id),
        "style_id": (
            int(env.style_id)
            if isinstance(env.style_id, (int, np.integer))
            else str(env.style_id)
        ),
    }


def _load_actions(path: Path) -> np.ndarray:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Trajectory is empty or invalid: {path}")

    actions: list[np.ndarray] = []
    for index, row in enumerate(rows, start=1):
        if "predicted_action_12d" in row:
            action = np.asarray(row["predicted_action_12d"], dtype=np.float32)
        else:
            action = np.asarray(row["action"], dtype=np.float32)
        if action.shape != (12,):
            raise ValueError(
                f"Expected a 12D Franka action at row {index}, got {action.shape}"
            )
        actions.append(action)
    return np.stack(actions, axis=0)


def _create_env(
    task: str,
    robot: str,
    seed: int,
    split: str,
    layout_id: int,
    style_id: int,
    camera_width: int,
    camera_height: int,
) -> Any:
    return env_utils.create_env(
        task,
        robots=robot,
        split=None,
        obj_instance_split=split,
        layout_and_style_ids=[(layout_id, style_id)],
        seed=seed,
        camera_names=[
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ],
        camera_widths=camera_width,
        camera_heights=camera_height,
        render_camera="robot0_agentview_left",
        control_freq=20,
        initialization_noise=None,
    )


def _make_writer(path: Path, fps: int):
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(path, fps=fps, codec="libx264")


def replay_franka(
    env: Any,
    actions: np.ndarray,
    args: argparse.Namespace,
    video_path: Path | None,
) -> dict[str, Any]:
    writer = _make_writer(video_path, 20) if video_path else None
    raw = env.reset()
    initial_position, initial_quaternion = _pose_from_raw(raw)
    initial_gripper = _gripper_summary(raw)
    rows: list[dict[str, Any]] = []
    success = False
    try:
        for step_index, action in enumerate(actions, start=1):
            raw, _, done, _ = env.step(action)
            position, quaternion = _pose_from_raw(raw)
            rows.append(
                {
                    "step": step_index,
                    "action_12d": action.tolist(),
                    "eef_pos_world": position.tolist(),
                    "eef_quat_world_xyzw": quaternion.tolist(),
                    "gripper": _gripper_summary(raw),
                }
            )
            if writer is not None:
                frame = np.asarray(
                    env.sim.render(
                        height=args.video_height,
                        width=args.video_width,
                        camera_name="robot0_agentview_left",
                    ),
                    dtype=np.uint8,
                )[::-1, :, :].copy()
                writer.append_data(
                    _resize_frame(frame, args.video_width, args.video_height)
                )
            success = bool(env._check_success())
            if success or bool(done):
                break
    finally:
        if writer is not None:
            writer.close()

    return {
        "initial_position_world": initial_position.tolist(),
        "initial_quaternion_world_xyzw": initial_quaternion.tolist(),
        "initial_gripper": initial_gripper,
        "rows": rows,
        "success": success,
    }


def replay_ur3e(
    env: Any,
    actions: np.ndarray,
    franka_initial_position: np.ndarray,
    franka_initial_quaternion: np.ndarray,
    args: argparse.Namespace,
    video_path: Path | None,
) -> dict[str, Any]:
    writer = _make_writer(video_path, 20) if video_path else None
    raw = env.reset()
    ur3e_reset_position, ur3e_reset_quaternion = _pose_from_raw(raw)
    alignment = align_initial_eef_to(
        env,
        franka_initial_position,
        franka_initial_quaternion,
    )
    raw = env._get_observations(force_update=True)
    ur3e_initial_position, ur3e_initial_quaternion = _pose_from_raw(raw)
    configure_review_camera(env, args.task)

    rows: list[dict[str, Any]] = []
    success = False
    try:
        for step_index, action in enumerate(actions, start=1):
            ur3e_action = np.asarray(action[:7], dtype=np.float32)
            raw, _, done, _ = env.step(ur3e_action)
            position, quaternion = _pose_from_raw(raw)
            rows.append(
                {
                    "step": step_index,
                    "action_7d": ur3e_action.tolist(),
                    "eef_pos_world": position.tolist(),
                    "eef_quat_world_xyzw": quaternion.tolist(),
                    "gripper": _gripper_summary(raw),
                }
            )
            if writer is not None:
                frame = np.asarray(
                    env.sim.render(
                        height=args.video_height,
                        width=args.video_width,
                        camera_name="robot0_review_camera",
                    ),
                    dtype=np.uint8,
                )[::-1, :, :].copy()
                writer.append_data(
                    _resize_frame(frame, args.video_width, args.video_height)
                )
            success = bool(env._check_success())
            if success or bool(done):
                break
    finally:
        if writer is not None:
            writer.close()

    return {
        "reset_position_world": ur3e_reset_position.tolist(),
        "reset_quaternion_world_xyzw": ur3e_reset_quaternion.tolist(),
        "initial_position_world": ur3e_initial_position.tolist(),
        "initial_quaternion_world_xyzw": ur3e_initial_quaternion.tolist(),
        "initial_gripper": _gripper_summary(raw),
        "alignment": alignment,
        "rows": rows,
        "success": success,
    }


def _map_franka_pose_to_ur3e_target(
    franka_rows: list[dict[str, Any]],
    franka_initial_position: np.ndarray,
    franka_initial_quaternion: np.ndarray,
    ur3e_initial_position: np.ndarray,
    ur3e_initial_quaternion: np.ndarray,
) -> list[dict[str, Any]]:
    """Map Franka relative EEF motion into the aligned UR3e world frame."""
    franka_initial_rotation = _quat_to_mat(franka_initial_quaternion)
    ur3e_initial_rotation = _quat_to_mat(ur3e_initial_quaternion)
    targets: list[dict[str, Any]] = []
    for row in franka_rows:
        franka_position = np.asarray(row["eef_pos_world"], dtype=np.float64)
        franka_rotation = _quat_to_mat(
            np.asarray(row["eef_quat_world_xyzw"], dtype=np.float64)
        )
        relative_position_delta = franka_initial_rotation.T @ (
            franka_position - franka_initial_position
        )
        target_position = ur3e_initial_position + ur3e_initial_rotation @ relative_position_delta
        relative_rotation_delta = franka_initial_rotation.T @ franka_rotation
        target_rotation = ur3e_initial_rotation @ relative_rotation_delta
        targets.append(
            {
                "step": int(row["step"]),
                "target_eef_pos_world": target_position.tolist(),
                "target_eef_quat_world_xyzw": _mat_to_quat(target_rotation).tolist(),
            }
        )
    return targets


def compare_rollouts(
    franka: dict[str, Any],
    ur3e: dict[str, Any],
) -> dict[str, Any]:
    franka_rows = franka["rows"]
    ur3e_rows = ur3e["rows"]
    count = min(len(franka_rows), len(ur3e_rows))
    if count == 0:
        raise ValueError("No replay steps were executed")

    franka_initial_position = np.asarray(franka["initial_position_world"], dtype=np.float64)
    franka_initial_quaternion = np.asarray(
        franka["initial_quaternion_world_xyzw"], dtype=np.float64
    )
    ur3e_initial_position = np.asarray(ur3e["initial_position_world"], dtype=np.float64)
    ur3e_initial_quaternion = np.asarray(
        ur3e["initial_quaternion_world_xyzw"], dtype=np.float64
    )
    targets = _map_franka_pose_to_ur3e_target(
        franka_rows[:count],
        franka_initial_position,
        franka_initial_quaternion,
        ur3e_initial_position,
        ur3e_initial_quaternion,
    )

    franka_positions = np.asarray(
        [row["eef_pos_world"] for row in franka_rows[:count]], dtype=np.float64
    )
    ur3e_positions = np.asarray(
        [row["eef_pos_world"] for row in ur3e_rows[:count]], dtype=np.float64
    )
    target_positions = np.asarray(
        [row["target_eef_pos_world"] for row in targets], dtype=np.float64
    )
    position_error = np.linalg.norm(ur3e_positions - target_positions, axis=1)
    trajectory_delta = (
        (franka_positions - franka_initial_position)
        - (ur3e_positions - ur3e_initial_position)
    )
    trajectory_error = np.linalg.norm(trajectory_delta, axis=1)

    target_rotations = [
        _quat_to_mat(np.asarray(row["target_eef_quat_world_xyzw"], dtype=np.float64))
        for row in targets
    ]
    ur3e_rotations = [
        _quat_to_mat(np.asarray(row["eef_quat_world_xyzw"], dtype=np.float64))
        for row in ur3e_rows[:count]
    ]
    orientation_errors = np.asarray(
        [
            np.linalg.norm(
                T.get_orientation_error(
                    _mat_to_quat(target_rotation),
                    _mat_to_quat(actual_rotation),
                )
            )
            for target_rotation, actual_rotation in zip(
                target_rotations, ur3e_rotations
            )
        ],
        dtype=np.float64,
    )

    franka_gripper_mean = np.asarray(
        [row["gripper"]["mean"] for row in franka_rows[:count]], dtype=np.float64
    )
    ur3e_gripper_mean = np.asarray(
        [row["gripper"]["mean"] for row in ur3e_rows[:count]], dtype=np.float64
    )
    gripper_command = np.asarray(
        [row["action_7d"][6] for row in ur3e_rows[:count]], dtype=np.float64
    )

    return {
        "num_steps_compared": count,
        "franka_replay_success": bool(franka["success"]),
        "ur3e_replay_success": bool(ur3e["success"]),
        "initial_eef_alignment": ur3e["alignment"],
        "eef_target_position_error_m": {
            "mean": float(np.mean(position_error)),
            "rmse": float(np.sqrt(np.mean(position_error**2))),
            "max": float(np.max(position_error)),
            "final": float(position_error[-1]),
        },
        "eef_target_orientation_error_rad": {
            "mean": float(np.mean(orientation_errors)),
            "rmse": float(np.sqrt(np.mean(orientation_errors**2))),
            "max": float(np.max(orientation_errors)),
            "final": float(orientation_errors[-1]),
        },
        "eef_relative_trajectory_error_m": {
            "mean": float(np.mean(trajectory_error)),
            "rmse": float(np.sqrt(np.mean(trajectory_error**2))),
            "max": float(np.max(trajectory_error)),
            "final": float(trajectory_error[-1]),
        },
        "gripper": {
            "command_mean_abs": float(np.mean(np.abs(gripper_command))),
            "franka_mean_range": [
                float(np.min(franka_gripper_mean)),
                float(np.max(franka_gripper_mean)),
            ],
            "ur3e_mean_range": [
                float(np.min(ur3e_gripper_mean)),
                float(np.max(ur3e_gripper_mean)),
            ],
            "mean_absolute_mean_qpos_difference": float(
                np.mean(np.abs(franka_gripper_mean - ur3e_gripper_mean))
            ),
            "final_mean_qpos_difference": float(
                abs(franka_gripper_mean[-1] - ur3e_gripper_mean[-1])
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--task", default="PreSoakPan")
    parser.add_argument("--split", default="target", choices=["target", "pretrain"])
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--layout-id", type=int, required=True)
    parser.add_argument("--style-id", type=int, required=True)
    parser.add_argument("--base-z", type=float, default=0.92)
    parser.add_argument("--base-y-offset", type=float, default=0.0)
    parser.add_argument("--camera-width", type=int, default=128)
    parser.add_argument("--camera-height", type=int, default=128)
    parser.add_argument("--video-width", type=int, default=512)
    parser.add_argument("--video-height", type=int, default=512)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--save-video", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    actions = _load_actions(args.trajectory)
    args.output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    franka_env = _create_env(
        args.task,
        "PandaOmron",
        args.seed,
        args.split,
        args.layout_id,
        args.style_id,
        args.camera_width,
        args.camera_height,
    )
    try:
        franka_video = (
            args.output_root / "franka_replay.mp4" if args.save_video else None
        )
        franka = replay_franka(franka_env, actions, args, franka_video)
    finally:
        franka_env.close()

    register_ur3e(args.base_z, args.base_y_offset)
    ur3e_env = _create_env(
        args.task,
        "UR3eOfficialFixed",
        args.seed,
        args.split,
        args.layout_id,
        args.style_id,
        args.camera_width,
        args.camera_height,
    )
    try:
        ur3e_video = (
            args.output_root / "ur3e_replay.mp4" if args.save_video else None
        )
        ur3e = replay_ur3e(
            ur3e_env,
            actions,
            np.asarray(franka["initial_position_world"], dtype=np.float64),
            np.asarray(franka["initial_quaternion_world_xyzw"], dtype=np.float64),
            args,
            ur3e_video,
        )
    finally:
        ur3e_env.close()

    comparison = compare_rollouts(franka, ur3e)
    payload = {
        "task": args.task,
        "seed": args.seed,
        "layout_id": args.layout_id,
        "style_id": args.style_id,
        "trajectory_source": str(args.trajectory),
        "elapsed_sec": time.perf_counter() - started,
        "franka": {
            "initial_position_world": franka["initial_position_world"],
            "initial_quaternion_world_xyzw": franka[
                "initial_quaternion_world_xyzw"
            ],
            "steps": len(franka["rows"]),
            "success": franka["success"],
        },
        "ur3e": {
            "initial_position_world": ur3e["initial_position_world"],
            "initial_quaternion_world_xyzw": ur3e[
                "initial_quaternion_world_xyzw"
            ],
            "steps": len(ur3e["rows"]),
            "success": ur3e["success"],
        },
        "comparison": comparison,
    }
    (args.output_root / "replay_comparison.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    (args.output_root / "franka_replay.json").write_text(
        json.dumps(franka, indent=2),
        encoding="utf-8",
    )
    (args.output_root / "ur3e_replay.json").write_text(
        json.dumps(ur3e, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
