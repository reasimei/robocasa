#!/usr/bin/env python3
"""Run the local Xiaomi Robotics-1 VLA directly on RoboCasa."""

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

from gr00t.eval.simulation import MultiStepConfig, SimulationConfig, VideoConfig
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
from robocasa.utils.dataset_registry_utils import get_task_horizon

from scripts.long_horizon_controller.robocasa_adapter import RobocasaVectorEnvAdapter
from scripts.long_horizon_controller.run_composite_seen_eval import DEFAULT_TASK_INSTRUCTIONS
from scripts.long_horizon_controller.xiaomi_policy_adapter import XiaomiPolicyAdapter


DEFAULT_MODEL_PATH = (
    "/data/zjw/workspace/Isaac-GR00T/expdata/"
    "Xiaomi-Robotics-1-RoboCasa365"
)


def is_cuda_oom(exc: BaseException) -> bool:
    if exc.__class__.__name__ == "OutOfMemoryError":
        return True
    message = str(exc).lower()
    return "cuda out of memory" in message or "out of memory" in message and "cuda" in message


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--task", default="KettleBoiling")
    parser.add_argument("--task-set", default="")
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--split", default="target", choices=["pretrain", "target"])
    parser.add_argument("--n-episodes", type=int, default=1)
    parser.add_argument(
        "--seed-base",
        type=int,
        default=1000,
        help="Episode seeds are seed_base + episode_index for paired evaluations.",
    )
    parser.add_argument("--output-root", default=(
        "/data/zjw/workspace/Isaac-GR00T/expdata/long_horizon_controller/"
        "xiaomi_robocasa_smoke"
    ))
    parser.add_argument("--n-action-steps", type=int, default=16)
    parser.add_argument("--max-episode-steps", type=int, default=0)
    parser.add_argument("--history-length", type=int, default=4)
    parser.add_argument(
        "--history-interval-steps",
        type=int,
        default=2,
        help="Low-level simulator step interval between Xiaomi history frames.",
    )
    parser.add_argument("--num-diffusion-steps", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run_episode(args: argparse.Namespace, task_name: str, index: int, model: XiaomiPolicyAdapter) -> dict:
    import torch

    episode_dir = Path(args.output_root) / "evals" / args.split / task_name / "episodes" / f"episode_{index:03d}"
    result_path = episode_dir / "result.json"
    if result_path.exists() and not args.overwrite:
        return json.loads(result_path.read_text(encoding="utf-8"))
    episode_dir.mkdir(parents=True, exist_ok=True)
    seed = args.seed_base + index
    torch.manual_seed(seed)
    max_steps = args.max_episode_steps or get_task_horizon(task_name)
    config = SimulationConfig(
        env_name=f"robocasa/{task_name}",
        split=args.split,
        n_episodes=1,
        n_envs=1,
        video=VideoConfig(video_dir=str(episode_dir / "videos")),
        multistep=MultiStepConfig(
            video_delta_indices=np.arange(
                -(args.history_length - 1) * args.history_interval_steps,
                1,
                args.history_interval_steps,
            ),
            state_delta_indices=np.arange(
                -(args.history_length - 1) * args.history_interval_steps,
                1,
                args.history_interval_steps,
            ),
            n_action_steps=args.n_action_steps,
            max_episode_steps=max_steps,
        ),
    )
    env = RobocasaVectorEnvAdapter(
        simulation_config=config,
        vlm_image_key="video.robot0_agentview_left",
    )
    model.reset()
    observation, _ = env._env.reset(seed=seed)
    instruction = DEFAULT_TASK_INSTRUCTIONS[task_name]
    steps = 0
    started = time.perf_counter()
    try:
        while steps < max_steps:
            action, _ = model.act(observation, instruction)
            observation, _, done, info = env.step(action)
            steps += args.n_action_steps
            if done or bool(_unwrap_base_env(env)._check_success()):
                break
        base_env = _unwrap_base_env(env)
        final_success = bool(base_env._check_success())
    except Exception as exc:
        if is_cuda_oom(exc):
            print(
                "[xiaomi-robocasa] WARNING: CUDA out of memory at "
                f"{task_name}/episode_{index:03d}; stopping evaluation immediately. "
                "No failure result was written. Free GPU memory and resume with "
                "the same command/output directory.",
                file=sys.stderr,
                flush=True,
            )
            raise RuntimeError(
                f"CUDA OOM at {task_name}/episode_{index:03d}; evaluation stopped."
            ) from exc
        raise
    finally:
        env.close()
    result = {
        "task_name": task_name,
        "episode_index": index,
        "seed": seed,
        "env_success": final_success,
        "steps": steps,
        "elapsed_sec": time.perf_counter() - started,
        "output_dir": str(episode_dir),
    }
    result_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result


def _unwrap_base_env(adapter: RobocasaVectorEnvAdapter) -> Any:
    current = adapter._env.envs[0]
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "_check_success") and hasattr(current, "sim"):
            return current
        current = getattr(current, "env", None)
    raise RuntimeError("Could not locate the RoboCasa base environment.")


def main() -> None:
    args = parse_args()
    if args.tasks:
        task_names = args.tasks
    elif args.task_set:
        task_names = list(TASK_SET_REGISTRY[args.task_set])
    else:
        task_names = [args.task]
    missing = [task for task in task_names if task not in DEFAULT_TASK_INSTRUCTIONS]
    if missing:
        raise ValueError(f"Missing default instruction for {missing}")

    model = XiaomiPolicyAdapter(
        model_path=args.model_path,
        history_length=args.history_length,
        action_steps=args.n_action_steps,
        num_diffusion_steps=args.num_diffusion_steps,
    )
    rows = []
    for task_name in task_names:
        for episode_index in range(args.n_episodes):
            print(
                f"[xiaomi-robocasa] {task_name} episode "
                f"{episode_index + 1}/{args.n_episodes}",
                flush=True,
            )
            rows.append(run_episode(args, task_name, episode_index, model))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "num_episodes": len(rows),
        "success_rate": float(np.mean([row["env_success"] for row in rows])) if rows else 0.0,
        "seed_base": args.seed_base,
        "results": rows,
    }
    (output_root / "results.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
