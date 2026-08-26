#!/usr/bin/env python3
"""Evaluate a stage-AdaLN Xiaomi policy on KettleBoiling with oracle stages.

The full KettleBoiling task instruction is unchanged for every policy call.
Only the trained AdaLN branch receives the current GT stage text.  Success is
always the RoboCasa simulator success signal, over fixed seeds 1000-1049.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.long_horizon_controller import run_kettle_oracle_split_eval as kettle
from scripts.long_horizon_controller.schemas import TaskPlan, plan_from_dict
from scripts.long_horizon_stage_adaln.model import adapter_training_args
from scripts.long_horizon_stage_adaln.policy_adapter import StageAdaLNXiaomiPolicyAdapter


DEFAULT_MODEL_PATH = (
    "/data/zjw/workspace/Isaac-GR00T/expdata/Xiaomi-Robotics-1-RoboCasa365"
)
DEFAULT_ADAPTER_CHECKPOINT = (
    "/data/zjw/workspace/Isaac-GR00T/expdata/long_horizon_stage_adaln/"
    "target_composite_adapter_full/checkpoint-6000.pt"
)
DEFAULT_PLAN_PATH = (
    "/data/zjw/workspace/Isaac-GR00T/expdata/long_horizon_controller/"
    "composite_seen_plan_cache_llama70b/KettleBoiling/plan.json"
)
TASK_NAME = "KettleBoiling"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter-checkpoint",
        default=DEFAULT_ADAPTER_CHECKPOINT,
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--plan-path", default=DEFAULT_PLAN_PATH)
    parser.add_argument(
        "--output-root",
        default=(
            "/data/zjw/workspace/Isaac-GR00T/expdata/long_horizon_stage_adaln/"
            "kettle_oracle_stage_adaln_checkpoint6000"
        ),
    )
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--split", choices=["pretrain", "target"], default="target")
    parser.add_argument("--n-action-steps", type=int, default=16)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--xiaomi-history-length", type=int, default=4)
    parser.add_argument("--xiaomi-history-interval-steps", type=int, default=2)
    parser.add_argument("--xiaomi-num-diffusion-steps", type=int, default=5)
    parser.add_argument(
        "--stage-condition-format",
        choices=["full", "subtask_only"],
        default="full",
        help="Text condition injected into DiT AdaLN.",
    )
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def stage_text(
    stage_index: int,
    labels: dict[str, Any],
    condition_format: str,
) -> str:
    if stage_index == 0:
        full_text = (
            "Atomic skill: PickPlaceCounterToStove. Stage: pick. "
            "Current subtask: pick up the kettle from the counter"
        )
        return (
            "pick up the kettle from the counter"
            if condition_format == "subtask_only"
            else full_text
        )
    if stage_index == 1:
        full_text = (
            "Atomic skill: PickPlaceCounterToStove. Stage: place. "
            "Current subtask: place the kettle on the stove burner"
        )
        return (
            "place the kettle on the stove burner"
            if condition_format == "subtask_only"
            else full_text
        )
    burner = labels.get("burner_location") or "left"
    burner_name = str(burner).replace("_", "-")
    full_text = (
        "Atomic skill: TurnOnStove. Stage: execute. "
        f"Current subtask: turn on the {burner_name} burner where the kettle is placed"
    )
    if condition_format == "subtask_only":
        return f"turn on the {burner_name} burner where the kettle is placed"
    return full_text


def run_episode(
    args: argparse.Namespace,
    plan: TaskPlan,
    policy: StageAdaLNXiaomiPolicyAdapter,
    env: Any,
    episode_index: int,
) -> dict[str, Any]:
    import torch

    seed = args.seed_base + episode_index
    torch.manual_seed(seed)
    policy.reset()
    observation, _ = env._env.reset(seed=seed)
    base_env = kettle.unwrap_kettle_env(env)
    subtask_index = 0
    transitions: list[dict[str, Any]] = []
    label_trace: list[dict[str, Any]] = []
    env_success = False
    done = False
    policy_calls = 0
    simulator_steps = 0

    while not done and simulator_steps < args.max_episode_steps:
        labels_before = kettle.kettle_oracle_labels(base_env)
        active_subtask = plan.subtasks[min(subtask_index, len(plan.subtasks) - 1)]
        current_stage_text = stage_text(
            subtask_index,
            labels_before,
            args.stage_condition_format,
        )
        action, _ = policy.act(
            observation,
            plan.task_instruction,
            current_stage_text,
        )
        observation, _, done, info = env.step(action)
        policy_calls += 1
        simulator_steps = min(
            args.max_episode_steps,
            simulator_steps + args.n_action_steps,
        )
        env_success = bool(info.get("success", False)) or env_success
        labels = kettle.kettle_oracle_labels(base_env)
        label_trace.append(
            {
                "simulator_steps": simulator_steps,
                "policy_call": policy_calls,
                "active_subtask_id": active_subtask.subtask_id,
                "stage_text": current_stage_text,
                **labels,
            }
        )
        while (
            subtask_index < len(plan.subtasks) - 1
            and bool(labels[plan.subtasks[subtask_index].subtask_id])
        ):
            completed = plan.subtasks[subtask_index]
            subtask_index += 1
            transitions.append(
                {
                    "from_subtask_id": completed.subtask_id,
                    "to_subtask_id": plan.subtasks[subtask_index].subtask_id,
                    "simulator_steps": simulator_steps,
                    "policy_call": policy_calls,
                    "oracle_labels": labels,
                }
            )
        if env_success:
            break

    final_labels = kettle.kettle_oracle_labels(base_env)
    return {
        "episode_index": episode_index,
        "seed": seed,
        "mode": "stage_adaln_oracle",
        "policy_backend": "xiaomi_stage_adaln",
        "env_success": bool(env_success or final_labels["env_check_success"]),
        "done": bool(done),
        "policy_calls": policy_calls,
        "simulator_steps": simulator_steps,
        "final_subtask_index": subtask_index,
        "transitions": transitions,
        "final_oracle_labels": final_labels,
        "label_trace": label_trace,
    }


def summary(
    results: list[dict[str, Any]],
    args: argparse.Namespace,
    plan: TaskPlan,
) -> dict[str, Any]:
    success = np.asarray([bool(item["env_success"]) for item in results], dtype=float)
    rate = float(success.mean()) if len(success) else 0.0
    stderr = math.sqrt(rate * (1.0 - rate) / len(success)) if len(success) else 0.0
    return {
        "task_name": TASK_NAME,
        "method": "oracle_stage_text_adaln",
        "adapter_checkpoint": args.adapter_checkpoint,
        "stage_condition_format": args.stage_condition_format,
        "full_task_prompt_unchanged": True,
        "n_episodes": len(results),
        "successes": success.astype(bool).tolist(),
        "success_rate": rate,
        "success_rate_standard_error": stderr,
        "success_rate_95ci_normal": [
            max(0.0, rate - 1.96 * stderr),
            min(1.0, rate + 1.96 * stderr),
        ],
        "seed_base": args.seed_base,
        "episode_seeds": [item["seed"] for item in results],
        "task_instruction": plan.task_instruction,
        "subtasks": [item.subtask_id for item in plan.subtasks],
        "episode_results": results,
    }


def main() -> None:
    args = parse_args()
    args.policy_backend = "xiaomi"
    args.mode = "stage_adaln_oracle"
    args.subtask_prompt_format = "not_used_full_prompt_kept"
    if args.n_episodes < 1:
        raise ValueError("--n-episodes must be positive.")
    if not Path(args.adapter_checkpoint).is_file():
        raise FileNotFoundError(f"Adapter checkpoint not found: {args.adapter_checkpoint}")
    training_args = adapter_training_args(args.adapter_checkpoint)
    checkpoint_format = training_args.get("stage_condition_format")
    if checkpoint_format and checkpoint_format != args.stage_condition_format:
        raise ValueError(
            "Adapter was trained with "
            f"--stage-condition-format {checkpoint_format!r}, but evaluation "
            f"requested {args.stage_condition_format!r}."
        )

    plan = plan_from_dict(json.loads(Path(args.plan_path).read_text(encoding="utf-8")))
    expected = ["pick_kettle", "place_kettle_on_stove", "turn_on_burner"]
    if [item.subtask_id for item in plan.subtasks] != expected:
        raise ValueError(f"Kettle plan must have stage IDs {expected}.")

    output_root = Path(args.output_root)
    episodes_dir = output_root / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    kettle.write_json(output_root / "config.json", vars(args))
    plan.save(output_root / "plan.json")
    policy = StageAdaLNXiaomiPolicyAdapter(
        model_path=args.model_path,
        adapter_checkpoint=args.adapter_checkpoint,
        history_length=args.xiaomi_history_length,
        action_steps=args.n_action_steps,
        num_diffusion_steps=args.xiaomi_num_diffusion_steps,
    )

    start = time.perf_counter()
    results: list[dict[str, Any]] = []
    for episode_index in range(args.n_episodes):
        episode_dir = episodes_dir / f"episode_{episode_index:03d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        result_path = episode_dir / "result.json"
        if result_path.exists() and not args.overwrite:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            results.append(result)
            continue
        print(
            f"[stage-adaln-kettle] episode {episode_index + 1}/{args.n_episodes} "
            f"(seed={args.seed_base + episode_index})",
            flush=True,
        )
        env = None
        try:
            env = kettle.make_env(args, episode_dir)
            result = run_episode(args, plan, policy, env, episode_index)
        except Exception as exc:
            result = {
                "episode_index": episode_index,
                "seed": args.seed_base + episode_index,
                "env_success": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            print(f"[stage-adaln-kettle] ERROR: {result['error']}", flush=True)
        finally:
            if env is not None:
                env.close()
        if args.video:
            try:
                video_path = kettle.annotate_episode_video(args, plan, result, episode_dir)
                if video_path is not None:
                    result["video_path"] = str(video_path.relative_to(episode_dir))
            except Exception as exc:
                result["video_error"] = f"{type(exc).__name__}: {exc}"
        kettle.write_json(result_path, result)
        results.append(result)
        running = summary(results, args, plan)
        kettle.write_json(output_root / "summary.json", running)
        print(
            f"[stage-adaln-kettle] success={result['env_success']} "
            f"running_sr={running['success_rate']:.3f}",
            flush=True,
        )

    payload = summary(results, args, plan)
    payload["elapsed_sec"] = time.perf_counter() - start
    kettle.write_json(output_root / "summary.json", payload)
    print(
        f"[stage-adaln-kettle] complete n={len(results)} "
        f"success_rate={payload['success_rate']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
