#!/usr/bin/env python3
"""Run a GR00T RoboCasa composite_seen evaluation with fixed-task or Oracle prompts."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gr00t.eval.simulation import MultiStepConfig, SimulationConfig, VideoConfig
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
from robocasa.utils.dataset_registry_utils import get_task_horizon

from scripts.long_horizon_controller.composite_seen_oracle import (
    COMPOSITE_SEEN_TASKS,
    get_stage_catalog,
    get_stage_labels,
)
from scripts.long_horizon_controller.policy_adapters import Gr00tPolicyAdapter
from scripts.long_horizon_controller.robocasa_adapter import RobocasaVectorEnvAdapter
from scripts.long_horizon_controller.run_composite_seen_eval import DEFAULT_TASK_INSTRUCTIONS
from scripts.long_horizon_controller.schemas import SubtaskSpec, TaskPlan
from scripts.long_horizon_controller.xiaomi_policy_adapter import XiaomiPolicyAdapter


DEFAULT_MODEL_PATH = (
    "/data/zjw/workspace/Isaac-GR00T/expdata/foundation_model_learning/"
    "target_posttraining/composite_seen/checkpoint-60000"
)
DEFAULT_XIAOMI_MODEL_PATH = (
    "/data/zjw/workspace/Isaac-GR00T/expdata/Xiaomi-Robotics-1-RoboCasa365"
)
DEFAULT_MANIFEST = (
    "/data/zjw/workspace/Isaac-GR00T/expdata/long_horizon_stage_adaln/"
    "target_composite_manifest.json"
)


def make_plan(task: str, manifest_path: str) -> tuple[TaskPlan, str]:
    stages, source = get_stage_catalog(task, manifest_path)
    subtasks = [
        SubtaskSpec(
            instruction=stage.instruction,
            expected_start_state="Simulator Oracle: previous stage is complete.",
            expected_finish_state=f"Simulator Oracle: stage {index} is complete.",
            max_duration_sec=60.0,
            subtask_id=stage.subtask_id,
            notes=f"atomic_skill={stage.atomic_skill}; stage={stage.stage}",
        )
        for index, stage in enumerate(stages)
    ]
    return TaskPlan(
        task_instruction=DEFAULT_TASK_INSTRUCTIONS[task],
        subtasks=subtasks,
        planner_model="simulator_oracle",
        raw_response="",
    ), source


def instruction_for_subtask(
    task_instruction: str,
    subtask: SubtaskSpec,
    subtask_index: int,
    prompt_format: str,
) -> str:
    if prompt_format == "subtask_only":
        return subtask.instruction
    if prompt_format == "natural":
        return f"{task_instruction}\nNow focus on: {subtask.instruction}"
    prefixes = ("First", "Next", "Finally")
    prefix = prefixes[min(subtask_index, len(prefixes) - 1)]
    text = subtask.instruction.strip()
    if prompt_format == "step_sentence":
        if text:
            text = text[0].lower() + text[1:]
        return f"{task_instruction} {prefix}, {text}"
    return f"Overall task: {task_instruction}\nCurrent subtask: {subtask.instruction}"


def _live_obj_lang(env: Any, name: str, fallback: str) -> str:
    try:
        value = str(env.get_obj_lang(name)).strip()
        return value or fallback
    except Exception:
        return fallback


def _article(value: str) -> str:
    return "an" if value[:1].lower() in "aeiou" else "a"


def ground_stage_instruction(task: str, instruction: str, env: Any) -> str:
    """Ground catalog text to the objects and task references in this episode."""
    text = instruction
    if task == "GetToastedBread":
        text = text.replace("bread", _live_obj_lang(env, "obj", "bread"))
    elif task == "KettleBoiling":
        text = text.replace(
            "turn on the left burner where the kettle is placed",
            "turn on the burner where the kettle is placed",
        )
    elif task == "PackIdenticalLunches":
        meat = _live_obj_lang(env, "meat0", "chicken drumstick")
        vegetable = _live_obj_lang(env, "vegetable0", "eggplant")
        text = text.replace("chicken drumstick", meat)
        text = text.replace("eggplant", vegetable)
    elif task == "PrepareCoffee":
        text = text.replace("mug", _live_obj_lang(env, "obj", "mug"))
    elif task == "SearingMeat":
        meat = _live_obj_lang(env, "meat", "chicken drumstick")
        text = text.replace("chicken drumstick", meat)
        knob = getattr(env, "knob", None)
        if knob:
            text = re.sub(
                r"\bthe rear-right burner\b",
                f"the {str(knob).replace('_', ' ')} burner",
                text,
            )
    elif task == "SetUpCuttingStation":
        meat = _live_obj_lang(env, "meat", "chicken drumstick")
        text = text.replace("chicken drumstick", meat)
    elif task == "SteamInMicrowave":
        vegetable = _live_obj_lang(env, "vegetable", "onion")
        text = text.replace("onion", vegetable)
    elif task == "StirVegetables":
        veg1 = _live_obj_lang(env, "veg1", "corn")
        veg2 = _live_obj_lang(env, "veg2", "tomato")
        text = text.replace("corn", veg1).replace("tomato", veg2)
    elif task == "StoreLeftoversInBowl":
        vegetable = _live_obj_lang(env, "vegetable", "eggplant")
        text = text.replace("eggplant", vegetable)

    text = re.sub(
        r"\ba (an? )?([aeiouAEIOU])",
        lambda match: f"{_article(match.group(2))} {match.group(2)}",
        text,
    )
    return text


def ground_plan(task: str, plan: TaskPlan, env: Any) -> tuple[TaskPlan, dict[str, Any]]:
    """Create the episode-specific plan used for policy prompts."""
    grounding: dict[str, Any] = {
        "mode": "live_episode",
        "task": task,
        "objects": {},
        "target_knob": getattr(env, "knob", None),
    }
    object_names = {
        "GetToastedBread": ("obj",),
        "PackIdenticalLunches": ("meat0", "vegetable0"),
        "PrepareCoffee": ("obj",),
        "SearingMeat": ("meat",),
        "SetUpCuttingStation": ("meat",),
        "SteamInMicrowave": ("vegetable",),
        "StirVegetables": ("veg1", "veg2"),
        "StoreLeftoversInBowl": ("vegetable",),
    }.get(task, ())
    for name in object_names:
        grounding["objects"][name] = _live_obj_lang(env, name, name.replace("_", " "))

    grounded_subtasks = [
        replace(
            subtask,
            instruction=ground_stage_instruction(task, subtask.instruction, env),
        )
        for subtask in plan.subtasks
    ]
    return replace(plan, subtasks=grounded_subtasks), grounding


def unwrap_base_env(adapter: RobocasaVectorEnvAdapter) -> Any:
    current = adapter._env.envs[0]
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "_check_success") and hasattr(current, "sim"):
            return current
        current = getattr(current, "env", None)
    raise RuntimeError("Could not locate the RoboCasa base environment.")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def latch_stage_label(
    stage_ids: list[str],
    raw_labels: dict[str, bool],
    latched_labels: dict[str, bool],
    stage_index: int,
) -> dict[str, bool]:
    """Latch only the currently active stage, preserving sequential order."""
    if 0 <= stage_index < len(stage_ids):
        stage_id = stage_ids[stage_index]
        latched_labels[stage_id] = bool(
            latched_labels.get(stage_id, False) or raw_labels.get(stage_id, False)
        )
    return dict(latched_labels)


def make_policy(args: argparse.Namespace) -> Any:
    if args.policy_backend == "xiaomi":
        return XiaomiPolicyAdapter(
            model_path=args.model_path,
            history_length=args.xiaomi_history_length,
            action_steps=args.n_action_steps,
            num_diffusion_steps=args.xiaomi_num_diffusion_steps,
        )

    data_config = DATA_CONFIG_MAP[args.data_config]
    modality_config = data_config.modality_config()
    policy = Gr00tPolicy(
        model_path=args.model_path,
        modality_config=modality_config,
        modality_transform=data_config.transform(),
        embodiment_tag=args.embodiment_tag,
        denoising_steps=4,
    )
    return Gr00tPolicyAdapter(
        policy=policy,
        action_keys=modality_config["action"].modality_keys,
        aux_head_path="",
    )


def run_episode(
    args: argparse.Namespace,
    task: str,
    plan: TaskPlan,
    source: str,
    policy: Gr00tPolicyAdapter,
    episode_index: int,
    episode_dir: Path,
) -> dict[str, Any]:
    import torch

    seed = args.seed_base + episode_index
    max_steps = args.max_episode_steps or get_task_horizon(task)
    if args.policy_backend == "xiaomi":
        history_offsets = np.arange(
            -(args.xiaomi_history_length - 1) * args.xiaomi_history_interval_steps,
            1,
            args.xiaomi_history_interval_steps,
        )
        video_delta_indices = history_offsets
        state_delta_indices = history_offsets
    else:
        video_delta_indices = np.array([0])
        state_delta_indices = np.array([0])
    config = SimulationConfig(
        env_name=f"robocasa/{task}",
        split=args.split,
        n_episodes=1,
        n_envs=1,
        video=VideoConfig(video_dir=str(episode_dir / "videos") if args.video else None),
        multistep=MultiStepConfig(
            video_delta_indices=video_delta_indices,
            state_delta_indices=state_delta_indices,
            n_action_steps=args.n_action_steps,
            max_episode_steps=max_steps,
        ),
    )
    env = RobocasaVectorEnvAdapter(simulation_config=config)
    try:
        if hasattr(policy, "reset"):
            policy.reset()
        torch.manual_seed(seed)
        observation, _ = env._env.reset(seed=seed)
        base_env = unwrap_base_env(env)
        stages, _ = get_stage_catalog(task, args.manifest_path)
        grounded_plan, live_grounding = ground_plan(task, plan, base_env)
        current_index = 0
        transitions: list[dict[str, Any]] = []
        trace: list[dict[str, Any]] = []
        policy_calls = 0
        simulator_steps = 0
        done = False
        env_success = False
        stage_ids = [stage.subtask_id for stage in stages]
        latched_labels = {stage_id: False for stage_id in stage_ids}

        raw_labels, diagnostics = get_stage_labels(task, base_env, stages)
        labels = dict(latched_labels)
        while not done and simulator_steps < max_steps:
            if args.mode == "full_task":
                active_index = None
                active_id = "full_task"
                instruction = grounded_plan.task_instruction
            else:
                active_index = min(current_index, len(grounded_plan.subtasks) - 1)
                active = grounded_plan.subtasks[active_index]
                active_id = active.subtask_id
                instruction = instruction_for_subtask(
                    grounded_plan.task_instruction,
                    active,
                    active_index,
                    args.prompt_format,
                )
            action, _ = policy.act(observation, instruction)
            observation, _, done, info = env.step(action)
            policy_calls += 1
            simulator_steps = min(max_steps, simulator_steps + args.n_action_steps)
            env_success = bool(info.get("success", False)) or env_success

            if args.mode == "oracle_split":
                raw_labels, diagnostics = get_stage_labels(task, base_env, stages)
                labels = latch_stage_label(
                    stage_ids, raw_labels, latched_labels, current_index
                )
            trace.append(
                {
                    "policy_call": policy_calls,
                    "simulator_steps": simulator_steps,
                    "active_subtask_index": active_index,
                    "active_subtask_id": active_id,
                    "instruction": instruction,
                    "labels": labels,
                    "raw_labels": raw_labels,
                    "latched_labels": labels,
                    "diagnostics": diagnostics,
                    "env_success": bool(info.get("success", False)),
                }
            )

            if args.mode == "oracle_split":
                while (
                    current_index < len(plan.subtasks) - 1
                    and labels[plan.subtasks[current_index].subtask_id]
                ):
                    completed = plan.subtasks[current_index]
                    current_index += 1
                    transitions.append(
                        {
                            "from_subtask_id": completed.subtask_id,
                            "to_subtask_id": plan.subtasks[current_index].subtask_id,
                            "policy_call": policy_calls,
                            "simulator_steps": simulator_steps,
                            "oracle_labels": labels,
                            "raw_oracle_labels": raw_labels,
                        }
                    )

                    labels = latch_stage_label(
                        stage_ids, raw_labels, latched_labels, current_index
                    )

            if env_success:
                break

        if args.mode == "oracle_split":
            raw_labels, diagnostics = get_stage_labels(task, base_env, stages)
            labels = latch_stage_label(
                stage_ids, raw_labels, latched_labels, current_index
            )
            env_check_success = bool(raw_labels.get(stages[-1].subtask_id, False))
            env_success = bool(env_success or env_check_success)
        else:
            raw_labels = {}
            diagnostics = {}
            labels = {}

        return {
            "episode_index": episode_index,
            "seed": seed,
            "task_name": task,
            "mode": args.mode,
            "policy_backend": args.policy_backend,
            "prompt_format": (
                args.prompt_format if args.mode == "oracle_split" else "full_task"
            ),
            "stage_source": source,
            "live_grounding": live_grounding,
            "grounded_subtasks": [
                {
                    "index": index,
                    "subtask_id": subtask.subtask_id,
                    "instruction": subtask.instruction,
                }
                for index, subtask in enumerate(grounded_plan.subtasks)
            ],
            "env_success": env_success,
            "done": bool(done),
            "policy_calls": policy_calls,
            "simulator_steps": simulator_steps,
            "final_subtask_index": (
                current_index if args.mode == "oracle_split" else None
            ),
            "transitions": transitions,
            "final_labels": labels,
            "final_raw_labels": raw_labels,
            "final_latched_labels": labels,
            "final_diagnostics": diagnostics,
            "label_trace": trace,
        }
    finally:
        env.close()


def summarize(
    task: str,
    plan: TaskPlan,
    source: str,
    mode: str,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    values = np.asarray([bool(item.get("env_success", False)) for item in results], dtype=float)
    rate = float(values.mean()) if len(values) else 0.0
    stderr = math.sqrt(rate * (1.0 - rate) / len(values)) if len(values) else 0.0
    return {
        "task_name": task,
        "mode": mode,
        "stage_source": source,
        "prompt_format": plan.raw_response or ("full_task" if mode == "full_task" else ""),
        "n_episodes": len(results),
        "successes": values.astype(bool).tolist(),
        "success_rate": rate,
        "success_rate_standard_error": stderr,
        "success_rate_95ci_normal": [
            max(0.0, rate - 1.96 * stderr),
            min(1.0, rate + 1.96 * stderr),
        ],
        "task_instruction": plan.task_instruction,
        "subtasks": [
            {
                "index": index,
                "subtask_id": subtask.subtask_id,
                "instruction": subtask.instruction,
                "notes": subtask.notes,
            }
            for index, subtask in enumerate(plan.subtasks)
        ],
        "episode_results": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default=(
            "/data/zjw/workspace/Isaac-GR00T/expdata/long_horizon_controller/"
            "composite_seen_gr00t_oracle_split_smoke"
        ),
    )
    parser.add_argument("--tasks", nargs="*", default=list(COMPOSITE_SEEN_TASKS))
    parser.add_argument(
        "--mode",
        choices=["full_task", "oracle_split"],
        default="oracle_split",
        help="full_task keeps the complete task instruction throughout; oracle_split switches at simulator labels.",
    )
    parser.add_argument(
        "--policy-backend",
        choices=["gr00t", "xiaomi"],
        default="gr00t",
        help="VLA policy used with the same simulator Oracle controller.",
    )
    parser.add_argument("--split", choices=["pretrain", "target"], default="target")
    parser.add_argument("--n-episodes", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST)
    parser.add_argument("--data-config", default="panda_omron")
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--n-action-steps", type=int, default=16)
    parser.add_argument("--max-episode-steps", type=int, default=0)
    parser.add_argument("--xiaomi-history-length", type=int, default=4)
    parser.add_argument("--xiaomi-history-interval-steps", type=int, default=2)
    parser.add_argument("--xiaomi-num-diffusion-steps", type=int, default=5)
    parser.add_argument(
        "--prompt-format",
        choices=["structured", "natural", "step_sentence", "subtask_only"],
        default="structured",
    )
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unknown = [task for task in args.tasks if task not in COMPOSITE_SEEN_TASKS]
    if unknown:
        raise ValueError(f"Unsupported tasks: {unknown}")
    if args.n_episodes < 1:
        raise ValueError("--n-episodes must be positive")
    if args.policy_backend == "xiaomi" and args.model_path == DEFAULT_MODEL_PATH:
        args.model_path = DEFAULT_XIAOMI_MODEL_PATH

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    policy = make_policy(args)
    batch_rows: list[dict[str, Any]] = []
    start = time.perf_counter()

    for task in args.tasks:
        task_dir = output_root / task
        task_dir.mkdir(parents=True, exist_ok=True)
        plan, source = make_plan(task, args.manifest_path)
        plan.raw_response = (
            "full_task"
            if args.mode == "full_task"
            else f"prompt_format={args.prompt_format}"
        )
        write_json(
            task_dir / "plan.json",
            {
                "task_instruction": plan.task_instruction,
                "planner_model": plan.planner_model,
                "stage_source": source,
                "subtasks": [
                    {
                        "subtask_id": item.subtask_id,
                        "instruction": item.instruction,
                        "expected_start_state": item.expected_start_state,
                        "expected_finish_state": item.expected_finish_state,
                        "max_duration_sec": item.max_duration_sec,
                        "notes": item.notes,
                    }
                    for item in plan.subtasks
                ],
            },
        )
        results: list[dict[str, Any]] = []
        for episode_index in range(args.n_episodes):
            episode_dir = task_dir / "episodes" / f"episode_{episode_index:03d}"
            result_path = episode_dir / "result.json"
            if result_path.exists() and not args.overwrite:
                result = json.loads(result_path.read_text(encoding="utf-8"))
                results.append(result)
                print(
                    f"[{args.mode}] {task} episode {episode_index}: existing",
                    flush=True,
                )
                continue
            print(
                f"[{args.mode}] running {task} episode {episode_index + 1}/{args.n_episodes} "
                f"seed={args.seed_base + episode_index}",
                flush=True,
            )
            try:
                result = run_episode(
                    args,
                    task,
                    plan,
                    source,
                    policy,
                    episode_index,
                    episode_dir,
                )
            except Exception as exc:
                result = {
                    "episode_index": episode_index,
                    "seed": args.seed_base + episode_index,
                    "task_name": task,
                    "mode": args.mode,
                    "policy_backend": args.policy_backend,
                    "env_success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
                print(f"[{args.mode}] ERROR {task}: {result['error']}", flush=True)
            write_json(result_path, result)
            results.append(result)
            print(
                f"[{args.mode}] {task} episode {episode_index}: "
                f"success={result.get('env_success', False)} "
                f"transitions={len(result.get('transitions', []))}",
                flush=True,
            )

        summary = summarize(task, plan, source, args.mode, results)
        summary["prompt_format"] = (
            args.prompt_format if args.mode == "oracle_split" else "full_task"
        )
        summary["n_action_steps"] = args.n_action_steps
        summary["seed_base"] = args.seed_base
        write_json(task_dir / "summary.json", summary)
        batch_rows.append(
            {
                "task_name": task,
                "env_success": summary["success_rate"] > 0.0,
                "success_rate": summary["success_rate"],
                "status": "completed",
                "output_dir": str(task_dir),
                "num_stages": len(plan.subtasks),
            }
        )

    write_json(
        output_root / f"{args.mode}_eval_results.json",
        {
            "mode": args.mode,
            "policy_backend": args.policy_backend,
            "model_path": args.model_path,
            "split": args.split,
            "prompt_format": (
                args.prompt_format if args.mode == "oracle_split" else "full_task"
            ),
            "n_episodes": args.n_episodes,
            "seed_base": args.seed_base,
            "elapsed_sec": time.perf_counter() - start,
            "num_tasks": len(batch_rows),
            "mean_task_success_rate": (
                float(np.mean([row["success_rate"] for row in batch_rows]))
                if batch_rows
                else 0.0
            ),
            "results": batch_rows,
        },
    )
    print(f"[{args.mode}] complete output={output_root}", flush=True)


if __name__ == "__main__":
    main()
