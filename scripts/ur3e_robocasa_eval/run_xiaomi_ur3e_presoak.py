#!/usr/bin/env python3
"""Evaluate Xiaomi Robotics-1 on RoboCasa PreSoakPan with a fixed UR3e."""

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
from robosuite.utils.transform_utils import euler2mat, mat2quat
from robosuite.robots import ROBOT_CLASS_MAPPING
from robosuite.robots.fixed_base_robot import FixedBaseRobot

from scripts.long_horizon_controller.run_composite_seen_eval import (
    DEFAULT_TASK_INSTRUCTIONS,
)
from scripts.long_horizon_controller.xiaomi_policy_adapter import XiaomiPolicyAdapter
from scripts.ur3e_robocasa_eval.ur3e_robot import UR3e


TASK_NAME = "PreSoakPan"
DEFAULT_MODEL_PATH = (
    "/data/zjw/workspace/Isaac-GR00T/expdata/"
    "Xiaomi-Robotics-1-RoboCasa365"
)


def register_ur3e() -> None:
    """Register UR3e only in this process; existing evaluations are untouched."""
    ROBOT_CLASS_MAPPING["UR3e"] = FixedBaseRobot
    # UR3e is shorter than PandaOmron. Raise its fixed base so sink objects are
    # inside the arm workspace. This is the requested suspended-base setup.
    env_utils._ROBOT_POS_OFFSETS["UR3e"] = [0.0, 0.0, 0.43]
    # Kitchen's stock reset helper assumes a PandaOmron mobile base. Replace
    # only that helper in this process with a fixed-body pose update.
    env_utils.set_robot_base = _set_fixed_robot_base


def _set_fixed_robot_base(
    env: Any,
    anchor_pos: np.ndarray,
    anchor_ori: np.ndarray,
    rot_dev: float,
    pos_dev_x: float,
    pos_dev_y: float,
) -> np.ndarray:
    del rot_dev, pos_dev_x, pos_dev_y
    body_id = env.sim.model.body_name2id("robot0_base")
    env.sim.model.body_pos[body_id] = np.asarray(anchor_pos, dtype=np.float64)
    # MuJoCo body quaternions use wxyz. RoboSuite's mat2quat returns xyzw.
    env.sim.model.body_quat[body_id] = mat2quat(
        euler2mat(np.asarray(anchor_ori, dtype=np.float64))
    )[[3, 0, 1, 2]]
    env.sim.forward()
    return np.asarray(anchor_pos, dtype=np.float64)


def raw_to_policy_observation(raw: dict[str, Any], env: Any) -> dict[str, Any]:
    """Translate raw RoboSuite observations to the Xiaomi RoboCasa365 schema."""
    required = [
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "robot0_agentview_left_image",
        "robot0_agentview_right_image",
        "robot0_eye_in_hand_image",
    ]
    missing = [key for key in required if key not in raw]
    if missing:
        raise KeyError(
            f"UR3e observation is missing {missing}; available keys include "
            f"{sorted(raw.keys())[:40]}"
        )
    base_pos = np.asarray(env.sim.data.get_body_xpos("robot0_base"), dtype=np.float32)
    base_quat = T.convert_quat(
        np.asarray(env.sim.data.get_body_xquat("robot0_base"), dtype=np.float32),
        to="xyzw",
    )
    base_pose = T.pose2mat((base_pos, base_quat))
    eef_pose = T.pose2mat(
        (
            np.asarray(raw["robot0_eef_pos"], dtype=np.float32),
            np.asarray(raw["robot0_eef_quat"], dtype=np.float32),
        )
    )
    relative_pose = T.pose_inv(base_pose).dot(eef_pose)
    relative_quat = T.mat2quat(relative_pose[:3, :3])
    gripper_qpos = np.asarray(raw["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
    if gripper_qpos.size >= 4:
        # RoboCasa365 uses two finger coordinates. Robotiq85 exposes six
        # coupled joints; the first and fourth are the left/right primary
        # finger joints and preserve the opening state expected by the model.
        gripper_state = gripper_qpos[[0, 3]]
    else:
        gripper_state = np.pad(gripper_qpos, (0, max(0, 2 - gripper_qpos.size)))[
            :2
        ]
    def official_camera_frame(key: str) -> np.ndarray:
        # RoboSuite's raw offscreen buffer uses the opposite vertical
        # convention. RoboCasa's Gym wrapper flips it before exposing model
        # observations, so the direct UR3e path must do the same.
        return np.asarray(raw[key], dtype=np.uint8)[::-1, :, :].copy()

    return {
        "state.end_effector_position_relative": relative_pose[:3, 3],
        "state.end_effector_rotation_relative": relative_quat,
        "state.gripper_qpos": gripper_state,
        "state.base_position": np.zeros(3, dtype=np.float32),
        "state.base_rotation": np.array([0, 0, 0, 1], dtype=np.float32),
        "video.robot0_agentview_left": official_camera_frame(
            "robot0_agentview_left_image"
        ),
        "video.robot0_agentview_right": official_camera_frame(
            "robot0_agentview_right_image"
        ),
        "video.robot0_eye_in_hand": official_camera_frame(
            "robot0_eye_in_hand_image"
        ),
    }


def _to_scalar_or_vector(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim > 1 and array.shape[0] == 1:
        array = array[0]
    return array.reshape(-1)


def run_episode(
    args: argparse.Namespace,
    episode_index: int,
    policy: Any,
) -> dict[str, Any]:
    seed = args.seed_base + episode_index
    episode_dir = (
        Path(args.output_root)
        / TASK_NAME
        / "episodes"
        / f"episode_{episode_index:03d}_seed_{seed}"
    )
    result_path = episode_dir / "result.json"
    if result_path.exists() and not args.overwrite:
        return json.loads(result_path.read_text(encoding="utf-8"))
    episode_dir.mkdir(parents=True, exist_ok=True)

    horizon = args.max_episode_steps or get_task_horizon(TASK_NAME)
    env = None
    writer = None
    started = time.perf_counter()
    steps = 0
    success = False
    error = None
    try:
        env = env_utils.create_env(
            TASK_NAME,
            robots="UR3e",
            split=args.split,
            seed=seed,
            camera_names=[
                "robot0_agentview_left",
                "robot0_agentview_right",
                "robot0_eye_in_hand",
            ],
            camera_widths=args.camera_width,
            camera_heights=args.camera_height,
            render_camera="robot0_agentview_left",
            control_freq=20,
        )
        raw_observation = env.reset()
        if hasattr(policy, "reset"):
            policy.reset()
        instruction = DEFAULT_TASK_INSTRUCTIONS[TASK_NAME]

        if args.save_video:
            import imageio.v2 as imageio

            writer = imageio.get_writer(
                episode_dir / f"{TASK_NAME}_ur3e_seed_{seed}.mp4",
                fps=20,
            )

        while steps < horizon:
            observation = raw_to_policy_observation(raw_observation, env)
            _, action_chunk = policy.act(observation, instruction)
            if action_chunk.ndim != 2 or action_chunk.shape[1] < 7:
                raise ValueError(f"Unexpected Xiaomi action chunk shape: {action_chunk.shape}")

            for action_row in action_chunk:
                if steps >= horizon:
                    break
                # Xiaomi RoboCasa365: EE pos(3), EE rot(3), gripper(1),
                # base(4), control mode(1). UR3e consumes only the arm+gripper.
                env_action = np.concatenate(
                    [
                        np.asarray(action_row[:6], dtype=np.float32),
                        np.asarray(action_row[6:7], dtype=np.float32),
                    ]
                )
                raw_observation, _, done, _ = env.step(env_action)
                steps += 1
                if writer is not None:
                    # Use the same camera-observation path that feeds Xiaomi.
                    # Direct sim.render() can return a stale/blank EGL buffer
                    # for this custom fixed-base robot.
                    camera_key = f"{args.video_camera}_image"
                    frame = np.asarray(
                        raw_observation[camera_key],
                        dtype=np.uint8,
                    )[::-1, :, :].copy()
                    if frame.shape[:2] != (args.video_height, args.video_width):
                        from PIL import Image

                        frame = np.asarray(
                            Image.fromarray(frame).resize(
                                (args.video_width, args.video_height),
                                Image.Resampling.BILINEAR,
                            ),
                            dtype=np.uint8,
                        )
                    writer.append_data(frame)
                success = bool(env._check_success())
                if success or bool(done):
                    break
            if success:
                break
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if writer is not None:
            writer.close()
        if env is not None:
            env.close()

    result = {
        "task_name": TASK_NAME,
        "robot": "UR3e",
        "policy": "Xiaomi-Robotics-1-RoboCasa365",
        "episode_index": episode_index,
        "seed": seed,
        "env_success": bool(success and error is None),
        "steps": steps,
        "elapsed_sec": time.perf_counter() - started,
        "error": error,
        "output_dir": str(episode_dir),
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--split", default="target", choices=["target", "pretrain"])
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument(
        "--output-root",
        default=(
            "/data/zjw/workspace/Isaac-GR00T/expdata/"
            "ur3e_robocasa_presoak_xiaomi_10eps"
        ),
    )
    parser.add_argument("--history-length", type=int, default=4)
    parser.add_argument("--n-action-steps", type=int, default=16)
    parser.add_argument("--num-diffusion-steps", type=int, default=5)
    parser.add_argument("--max-episode-steps", type=int, default=0)
    parser.add_argument("--camera-width", type=int, default=128)
    parser.add_argument("--camera-height", type=int, default=128)
    parser.add_argument("--video-width", type=int, default=512)
    parser.add_argument("--video-height", type=int, default=512)
    parser.add_argument(
        "--video-camera",
        choices=[
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ],
        default="robot0_eye_in_hand",
        help="Camera observation to encode into the review video.",
    )
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    register_ur3e()
    policy = XiaomiPolicyAdapter(
        model_path=args.model_path,
        history_length=args.history_length,
        action_steps=args.n_action_steps,
        num_diffusion_steps=args.num_diffusion_steps,
    )
    rows = []
    for episode_index in range(args.n_episodes):
        print(
            f"[ur3e-presoak] {TASK_NAME} episode "
            f"{episode_index + 1}/{args.n_episodes}",
            flush=True,
        )
        row = run_episode(args, episode_index, policy)
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)

    successes = sum(bool(row["env_success"]) for row in rows)
    summary = {
        "task_name": TASK_NAME,
        "robot": "UR3e",
        "policy": "Xiaomi-Robotics-1-RoboCasa365",
        "num_episodes": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows) if rows else 0.0,
        "seed_base": args.seed_base,
        "results": rows,
    }
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "results.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
