#!/usr/bin/env python3
"""Run the GR00T RoboCasa policy on the original PandaOmron robot.

This is an isolated paired-control baseline for comparing the same GR00T
checkpoint and scene against the fixed-base UR3e evaluator.
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

from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy

from scripts.long_horizon_controller.policy_adapters import Gr00tPolicyAdapter
from scripts.long_horizon_controller.run_composite_seen_eval import (
    DEFAULT_TASK_INSTRUCTIONS,
)
from scripts.ur3e_robocasa_eval.run_xiaomi_ur3e_fixed_eval import _resize_frame


DEFAULT_MODEL_PATH = (
    "/data/zjw/workspace/Isaac-GR00T/expdata/"
    "foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000"
)


def raw_to_policy_observation(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert raw PandaOmron observations to the trained GR00T schema."""
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

    def first(value: Any) -> np.ndarray:
        array = np.asarray(value)
        while array.ndim > 1 and array.shape[0] == 1:
            array = array[0]
        return np.asarray(array)

    def camera(key: str) -> np.ndarray:
        return np.asarray(raw[key], dtype=np.uint8)[::-1, :, :].copy()

    return {
        "state.end_effector_position_relative": first(
            raw["robot0_base_to_eef_pos"]
        ).astype(np.float32),
        "state.end_effector_rotation_relative": first(
            raw["robot0_base_to_eef_quat"]
        ).astype(np.float32),
        "state.gripper_qpos": first(raw["robot0_gripper_qpos"]).astype(np.float32),
        "state.base_position": np.zeros(3, dtype=np.float32),
        "state.base_rotation": np.array([0, 0, 0, 1], dtype=np.float32),
        "video.robot0_agentview_left": camera("robot0_agentview_left_image"),
        "video.robot0_agentview_right": camera("robot0_agentview_right_image"),
        "video.robot0_eye_in_hand": camera("robot0_eye_in_hand_image"),
    }


def run_episode(
    args: argparse.Namespace,
    task_name: str,
    policy: Gr00tPolicyAdapter,
) -> dict[str, Any]:
    seed = args.seed
    episode_dir = (
        Path(args.output_root)
        / task_name
        / "episodes"
        / f"episode_000_seed_{seed}"
    )
    result_path = episode_dir / "result.json"
    if result_path.exists() and not args.overwrite:
        return json.loads(result_path.read_text(encoding="utf-8"))
    episode_dir.mkdir(parents=True, exist_ok=True)

    env = None
    writer = None
    trajectory: list[dict[str, Any]] = []
    steps = 0
    success = False
    error: str | None = None
    scene: dict[str, Any] | None = None
    started = time.perf_counter()

    try:
        env_kwargs: dict[str, Any] = {
            "camera_names": [
                "robot0_agentview_left",
                "robot0_agentview_right",
                "robot0_eye_in_hand",
            ],
            "camera_widths": args.camera_width,
            "camera_heights": args.camera_height,
            "render_camera": "robot0_agentview_left",
            "control_freq": 20,
            "initialization_noise": None,
        }
        if args.layout_id is not None:
            env_kwargs.update(
                split=None,
                obj_instance_split=args.split,
                layout_and_style_ids=[(args.layout_id, args.style_id)],
            )
        else:
            env_kwargs["split"] = args.split

        env = env_utils.create_env(
            task_name,
            robots="PandaOmron",
            seed=seed,
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
        policy.last_aux_output = None
        instruction = DEFAULT_TASK_INSTRUCTIONS.get(task_name, task_name)

        if args.save_video:
            import imageio.v2 as imageio

            writer = imageio.get_writer(
                episode_dir / f"{task_name}_gr00t_franka_seed_{seed}.mp4",
                fps=20,
                codec="libx264",
            )

        horizon = args.max_episode_steps or get_task_horizon(task_name)
        while steps < horizon:
            observation = raw_to_policy_observation(raw_observation)
            _, action_chunk = policy.act(observation, instruction)
            action_chunk = np.asarray(action_chunk, dtype=np.float32)
            if action_chunk.ndim != 2 or action_chunk.shape[1] < env.action_dim:
                raise ValueError(
                    f"Unexpected GR00T action chunk shape: {action_chunk.shape}; "
                    f"expected at least {env.action_dim} dimensions"
                )

            for action_row in action_chunk:
                if steps >= horizon:
                    break
                env_action = np.asarray(
                    action_row[: env.action_dim], dtype=np.float32
                )
                raw_observation, _, done, _ = env.step(env_action)
                steps += 1
                trajectory.append(
                    {
                        "step": steps,
                        "predicted_action_12d": np.asarray(
                            action_row[:12], dtype=np.float32
                        ).tolist(),
                        "predicted_base_motion": np.asarray(
                            action_row[7:11], dtype=np.float32
                        ).tolist(),
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
                    writer.append_data(
                        _resize_frame(frame, args.video_width, args.video_height)
                    )
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
        "task_name": task_name,
        "robot": "PandaOmron",
        "policy": "GR00T",
        "model_path": args.model_path,
        "data_config": args.data_config,
        "embodiment_tag": args.embodiment_tag,
        "scene": scene,
        "episode_index": 0,
        "seed": seed,
        "env_success": bool(success and error is None),
        "steps": steps,
        "elapsed_sec": time.perf_counter() - started,
        "error": error,
        "output_dir": str(episode_dir),
    }
    (episode_dir / "eef_trajectory.json").write_text(
        json.dumps(trajectory, indent=2),
        encoding="utf-8",
    )
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="PreSoakPan")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--split", default="target", choices=["target", "pretrain"])
    parser.add_argument("--data-config", default="panda_omron")
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--output-root",
        default=(
            "/data/zjw/workspace/Isaac-GR00T/expdata/"
            "ur3e_robocasa_fixed_gr00t_smoke/franka"
        ),
    )
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument("--max-episode-steps", type=int, default=0)
    parser.add_argument("--camera-width", type=int, default=256)
    parser.add_argument("--camera-height", type=int, default=256)
    parser.add_argument("--video-width", type=int, default=512)
    parser.add_argument("--video-height", type=int, default=512)
    parser.add_argument("--layout-id", type=int, default=None)
    parser.add_argument("--style-id", type=int, default=None)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.layout_id is None) != (args.style_id is None):
        raise ValueError("--layout-id and --style-id must be provided together")
    if args.data_config not in DATA_CONFIG_MAP:
        raise ValueError(f"Unknown data config: {args.data_config}")

    import torch

    data_config = DATA_CONFIG_MAP[args.data_config]
    modality_config = data_config.modality_config()
    policy = Gr00tPolicy(
        model_path=args.model_path,
        modality_config=modality_config,
        modality_transform=data_config.transform(),
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    policy_adapter = Gr00tPolicyAdapter(
        policy=policy,
        action_keys=modality_config["action"].modality_keys,
    )
    row = run_episode(args, args.task, policy_adapter)
    print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
