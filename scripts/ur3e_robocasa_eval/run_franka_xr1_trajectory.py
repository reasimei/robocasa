#!/usr/bin/env python3
"""Run XR-1 on the original RoboCasa PandaOmron and save EE trajectories.

This is an isolated comparison runner. It does not alter the existing Xiaomi,
RoboCasa, or UR3e evaluation scripts.
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

from scripts.long_horizon_controller.run_composite_seen_eval import (
    DEFAULT_TASK_INSTRUCTIONS,
)
from scripts.long_horizon_controller.xiaomi_policy_adapter import XiaomiPolicyAdapter


DEFAULT_MODEL_PATH = (
    "/data/zjw/workspace/Isaac-GR00T/expdata/"
    "Xiaomi-Robotics-1-RoboCasa365"
)
TASK_NAME = "PreSoakPan"


def _first(value: Any) -> np.ndarray:
    array = np.asarray(value)
    while array.ndim > 1 and array.shape[0] == 1:
        array = array[0]
    return np.asarray(array)


def _camera_frame(raw: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray(raw[key], dtype=np.uint8)[::-1, :, :].copy()


def raw_to_policy_observation(raw: dict[str, Any]) -> dict[str, Any]:
    """Build exactly the RoboCasa365 observation schema used by the adapter."""
    required = [
        "robot0_base_to_eef_pos",
        "robot0_base_to_eef_quat",
        "robot0_gripper_qpos",
        "robot0_agentview_left_image",
        "robot0_agentview_right_image",
        "robot0_eye_in_hand_image",
    ]
    missing = [key for key in required if key not in raw]
    if missing:
        raise KeyError(f"PandaOmron observation is missing {missing}")

    return {
        "state.end_effector_position_relative": _first(
            raw["robot0_base_to_eef_pos"]
        ).astype(np.float32),
        "state.end_effector_rotation_relative": _first(
            raw["robot0_base_to_eef_quat"]
        ).astype(np.float32),
        "state.gripper_qpos": _first(raw["robot0_gripper_qpos"]).astype(np.float32),
        "state.base_position": np.zeros(3, dtype=np.float32),
        "state.base_rotation": np.array([0, 0, 0, 1], dtype=np.float32),
        "video.robot0_agentview_left": _camera_frame(
            raw, "robot0_agentview_left_image"
        ),
        "video.robot0_agentview_right": _camera_frame(
            raw, "robot0_agentview_right_image"
        ),
        "video.robot0_eye_in_hand": _camera_frame(
            raw, "robot0_eye_in_hand_image"
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = (
        Path(args.output_root)
        / TASK_NAME
        / "episodes"
        / f"episode_000_seed_{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    if result_path.exists() and not args.overwrite:
        return json.loads(result_path.read_text(encoding="utf-8"))

    model = XiaomiPolicyAdapter(
        model_path=args.model_path,
        history_length=args.history_length,
        action_steps=args.n_action_steps,
        num_diffusion_steps=args.num_diffusion_steps,
    )
    env = None
    writer = None
    trajectory: list[dict[str, Any]] = []
    started = time.perf_counter()
    steps = 0
    success = False
    error = None
    scene = None

    try:
        env_kwargs = dict(
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
        if args.layout_id is not None:
            env_kwargs.update(
                split=None,
                obj_instance_split=args.split,
                layout_and_style_ids=[(args.layout_id, args.style_id)],
            )
        else:
            env_kwargs["split"] = args.split
        env = env_utils.create_env(
            TASK_NAME,
            robots="PandaOmron",
            seed=args.seed,
            **env_kwargs,
        )
        raw_observation = env.reset()
        scene = {
            "layout_id": int(env.layout_id),
            "style_id": (
                int(env.style_id)
                if isinstance(env.style_id, (int, np.integer))
                else str(env.style_id)
            ),
        }
        model.reset()
        instruction = DEFAULT_TASK_INSTRUCTIONS[TASK_NAME]

        if args.save_video:
            import imageio.v2 as imageio

            writer = imageio.get_writer(
                output_dir / f"{TASK_NAME}_franka_xr1_seed_{args.seed}.mp4",
                fps=20,
                codec="libx264",
            )

        horizon = args.max_episode_steps or get_task_horizon(TASK_NAME)
        while steps < horizon:
            observation = raw_to_policy_observation(raw_observation)
            _, action_chunk = model.act(observation, instruction)
            action_chunk = np.asarray(action_chunk)
            if action_chunk.ndim != 2 or action_chunk.shape[1] < env.action_dim:
                raise ValueError(f"Unexpected XR-1 action chunk shape: {action_chunk.shape}")

            for action_row in action_chunk:
                if steps >= horizon:
                    break
                # PandaOmron consumes the full RoboCasa365 12D action:
                # EE position (3), EE rotation (3), gripper (1), base (4),
                # and control mode (1). UR3e consumes only the first 7 dims.
                env_action = np.asarray(
                    action_row[: env.action_dim], dtype=np.float32
                )
                raw_observation, _, done, _ = env.step(env_action)
                steps += 1
                trajectory.append(
                    {
                        "step": steps,
                        "eef_pos_world": np.asarray(
                            raw_observation["robot0_eef_pos"], dtype=np.float32
                        ).tolist(),
                        "eef_quat_world_xyzw": np.asarray(
                            raw_observation["robot0_eef_quat"], dtype=np.float32
                        ).tolist(),
                        "eef_pos_base": np.asarray(
                            raw_observation["robot0_base_to_eef_pos"],
                            dtype=np.float32,
                        ).tolist(),
                        "eef_quat_base_xyzw": np.asarray(
                            raw_observation["robot0_base_to_eef_quat"],
                            dtype=np.float32,
                        ).tolist(),
                        "action": env_action.tolist(),
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
                    )[:: -1, :, :].copy()
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
        "robot": "PandaOmron",
        "policy": "Xiaomi-Robotics-1-RoboCasa365 (XR-1)",
        "seed": args.seed,
        "scene": scene,
        "env_success": bool(success and error is None),
        "steps": steps,
        "elapsed_sec": time.perf_counter() - started,
        "error": error,
        "output_dir": str(output_dir),
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output_dir / "eef_trajectory.json").write_text(
        json.dumps(trajectory, indent=2),
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--split", default="target", choices=["pretrain", "target"])
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--layout-id",
        type=int,
        default=None,
        help="Fix the RoboCasa kitchen layout. Must be used with --style-id.",
    )
    parser.add_argument(
        "--style-id",
        type=int,
        default=None,
        help="Fix the RoboCasa kitchen style. Must be used with --layout-id.",
    )
    parser.add_argument(
        "--output-root",
        default=(
            "/data/zjw/workspace/Isaac-GR00T/expdata/"
            "ur3e_official_stack_full/PreSoakPan/franka_xr1"
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
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if (args.layout_id is None) != (args.style_id is None):
        raise ValueError("--layout-id and --style-id must be provided together")
    print(json.dumps(run(args), indent=2), flush=True)
