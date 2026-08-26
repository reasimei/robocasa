#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
from robocasa.utils.dataset_registry_utils import get_task_horizon

from gr00t.eval.simulation import MultiStepConfig, SimulationConfig, VideoConfig
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy

from scripts.long_horizon_controller.controller import ControllerConfig, LongHorizonController
from scripts.long_horizon_controller.fast_monitor import ActionEntropyMonitor
from scripts.long_horizon_controller.policy_adapters import Gr00tPolicyAdapter
from scripts.long_horizon_controller.robocasa_adapter import RobocasaVectorEnvAdapter
from scripts.long_horizon_controller.run_composite_seen_eval import (
    DEFAULT_TASK_INSTRUCTIONS,
    MODEL_PATH,
    annotate_video,
    clear_episode_artifacts,
    load_episode_result,
    load_or_create_plan,
    save_batch_results,
    save_episode_stats,
    save_stats,
    task_has_enough_episodes,
)
from scripts.long_horizon_controller.planner import OllamaPlanner, OpenAICompatiblePlanner, StaticPlanner
from scripts.long_horizon_controller.schemas import FastMonitorConfig, FastTrigger, SubtaskSpec, TaskPlan


AUX_HEAD_PATH = (
    "/data/zjw/workspace/Isaac-GR00T/expdata/aux_progress/"
    "atomic_retry_3class_history_retry_boost_run2/checkpoint-11000"
)


class FastAuxOnlyController(LongHorizonController):
    """Controller variant that uses CC + auxiliary head only, without VLM."""

    def __init__(
        self,
        *,
        aux_retry_confidence_threshold: float,
        aux_success_confidence_threshold: float,
        aux_decision_mode: str,
        fast_retry_rollback_chunks: int,
        allow_timeout_gate: bool,
        **kwargs: Any,
    ) -> None:
        super().__init__(vlm_verifier=None, aux_fusion=None, **kwargs)
        self.aux_retry_confidence_threshold = float(aux_retry_confidence_threshold)
        self.aux_success_confidence_threshold = float(aux_success_confidence_threshold)
        self.aux_decision_mode = str(aux_decision_mode)
        self.fast_retry_rollback_chunks = int(fast_retry_rollback_chunks)
        self.allow_timeout_gate = bool(allow_timeout_gate)

    def _aux_prob(self, aux_output: dict[str, Any] | None, state_name: str) -> float | None:
        if not aux_output:
            return None
        probs = aux_output.get("probs")
        if isinstance(probs, dict) and state_name in probs:
            return float(probs[state_name])
        if str(aux_output.get("state", "")) == state_name and aux_output.get("confidence") is not None:
            return float(aux_output["confidence"])
        return None

    def _aux_decision(self, aux_output: dict[str, Any] | None) -> tuple[str, float | None, dict[str, Any]]:
        if not aux_output:
            return "progress", None, {}

        state = str(aux_output.get("state", ""))
        confidence = aux_output.get("confidence")
        confidence_f = float(confidence) if confidence is not None else None
        retry_prob = self._aux_prob(aux_output, "retry")
        success_prob = self._aux_prob(aux_output, "success")

        if self.aux_decision_mode == "argmax":
            if state == "retry" and confidence_f is not None and confidence_f >= self.aux_retry_confidence_threshold:
                return "retry", confidence_f, {"retry_prob": retry_prob, "success_prob": success_prob}
            if state == "success" and confidence_f is not None and confidence_f >= self.aux_success_confidence_threshold:
                return "success", confidence_f, {"retry_prob": retry_prob, "success_prob": success_prob}
            return "progress", confidence_f, {"retry_prob": retry_prob, "success_prob": success_prob}

        if self.aux_decision_mode != "prob":
            raise ValueError(f"Unsupported aux_decision_mode={self.aux_decision_mode!r}")

        # Retry is prioritized because a high success probability was the main
        # source of premature transitions in the previous rollout logs.
        if retry_prob is not None and retry_prob >= self.aux_retry_confidence_threshold:
            return "retry", retry_prob, {"retry_prob": retry_prob, "success_prob": success_prob}
        if success_prob is not None and success_prob >= self.aux_success_confidence_threshold:
            return "success", success_prob, {"retry_prob": retry_prob, "success_prob": success_prob}
        return "progress", confidence_f, {"retry_prob": retry_prob, "success_prob": success_prob}

    def _run_impl(self) -> None:
        observation = self.env.reset()
        global_step = 0
        current_index = 0
        env_success = False
        last_env_success = False

        while self.queue and global_step < self.config.max_total_steps:
            subtask = self.queue.pop(0)
            self.active_action_history = []
            self.fast_monitor.reset()
            local_step = 0
            fast_trigger_count = 0
            self.log_event(global_step, current_index, subtask, "start_subtask", asdict(subtask))

            while global_step < self.config.max_total_steps:
                elapsed = local_step * self.config.step_dt_sec
                instruction = self.policy_instruction(subtask)
                action, action_chunk = self.policy.act(observation, instruction)
                action_snapshot = self._snapshot_action(action)
                action_entry = {
                    "step_index": global_step,
                    "subtask_id": subtask.subtask_id,
                    "action": action_snapshot,
                }
                self.action_history.append(action_entry)
                self.active_action_history.append(action_entry)

                observation, reward, done, info = self.env.step(action)
                del reward
                if isinstance(info, dict) and "success" in info:
                    last_env_success = bool(info.get("success"))
                    env_success = last_env_success
                global_step += 1
                local_step += 1

                cc_signal = self.fast_monitor.update(
                    action_chunk=action_chunk,
                    elapsed_sec=elapsed,
                    timeout_sec=subtask.max_duration_sec,
                )
                aux_output = getattr(self.policy, "last_aux_output", None)
                if aux_output is None and isinstance(info, dict):
                    aux_output = info.get("aux_head")

                cc_gate = cc_signal.trigger == FastTrigger.SUSPECT_TRANSITION
                timeout_gate = self.allow_timeout_gate and cc_signal.trigger == FastTrigger.TIMEOUT
                if cc_gate or timeout_gate:
                    fast_trigger_count += 1
                    aux_decision, aux_confidence, aux_extra = self._aux_decision(aux_output)
                    gate_payload = {
                        "cc_trigger": cc_signal.trigger.value,
                        "cc_score": cc_signal.score,
                        "aux_decision": aux_decision,
                        "aux_state": None if not aux_output else aux_output.get("state"),
                        "aux_confidence": aux_confidence,
                        "aux_output": aux_output or {},
                        "aux_decision_mode": self.aux_decision_mode,
                        "retry_threshold": self.aux_retry_confidence_threshold,
                        "success_threshold": self.aux_success_confidence_threshold,
                        "fast_trigger_count": fast_trigger_count,
                        **aux_extra,
                    }
                    self.log_event(global_step, current_index, subtask, "fast_gate", gate_payload)

                    if aux_decision == "success":
                        self.completed.append(subtask)
                        self.log_event(
                            global_step,
                            current_index,
                            subtask,
                            "fast_signal",
                            {
                                "trigger": FastTrigger.SUSPECT_COMPLETE.value,
                                "score": cc_signal.score,
                                "aux_state": gate_payload["aux_state"],
                                "aux_confidence": aux_confidence,
                                "elapsed_sec": cc_signal.elapsed_sec,
                                "reason": "cc transition/timeout and auxiliary head predicts success",
                                **aux_extra,
                            },
                        )
                        self.log_event(global_step, current_index, subtask, "advance_subtask")
                        current_index += 1
                        break

                    if aux_decision == "retry":
                        rollback_start = global_step
                        (
                            observation,
                            global_step,
                            env_success,
                            rollback_count,
                            rollback_done,
                        ) = self._rollback(
                            observation=observation,
                            global_step=global_step,
                            current_index=current_index,
                            subtask=subtask,
                            requested_chunks=self.fast_retry_rollback_chunks,
                            env_success=env_success,
                        )
                        self.log_event(
                            global_step,
                            current_index,
                            subtask,
                            "fast_signal",
                            {
                                "trigger": FastTrigger.SUSPECT_FAIL.value,
                                "score": cc_signal.score,
                                "aux_state": gate_payload["aux_state"],
                                "aux_confidence": aux_confidence,
                                "elapsed_sec": cc_signal.elapsed_sec,
                                "reason": "cc transition/timeout and auxiliary head predicts retry",
                                "requested_rollback_chunks": self.fast_retry_rollback_chunks,
                                "executed_rollback_chunks": rollback_count,
                                **aux_extra,
                            },
                        )
                        if rollback_count > 0:
                            self.queue = [subtask] + self.queue
                            self.log_event(
                                global_step,
                                current_index,
                                subtask,
                                "rollback_retry",
                                {
                                    "recovery_mode": "fast_aux_only",
                                    "requested_rollback_chunks": self.fast_retry_rollback_chunks,
                                    "executed_rollback_chunks": rollback_count,
                                    "rollback_start_step": rollback_start,
                                    "rollback_end_step": global_step,
                                    "rollback_done": rollback_done,
                                },
                            )
                            break

                        self.log_event(
                            global_step,
                            current_index,
                            subtask,
                            "rollback_unavailable",
                            {
                                "recovery_mode": "fast_aux_only",
                                "requested_rollback_chunks": self.fast_retry_rollback_chunks,
                                "available_action_chunks": len(self.active_action_history),
                                "reason": "environment adapter does not support rollback or history is empty",
                            },
                        )
                        self.queue = [subtask] + self.queue
                        break

                if done:
                    env_success = last_env_success
                    self.log_event(
                        global_step,
                        current_index,
                        subtask,
                        "env_done",
                        {"success": env_success},
                    )
                    if env_success and subtask not in self.completed:
                        self.completed.append(subtask)
                    self.queue.clear()
                    break

        controller_completed_plan = len(self.completed) >= len(self.plan.subtasks)
        if env_success or controller_completed_plan:
            finished_subtask = self.completed[-1] if self.completed else SubtaskSpec(
                instruction="",
                expected_start_state="",
                expected_finish_state="",
                max_duration_sec=0.0,
                subtask_id="plan",
            )
            self.log_event(
                global_step,
                current_index,
                finished_subtask,
                "finish_plan",
                {
                    "env_success": env_success,
                    "controller_completed_plan": controller_completed_plan,
                },
            )


def make_planner(args: argparse.Namespace):
    if args.planner == "api":
        return OpenAICompatiblePlanner(timeout_sec=args.llm_timeout_sec)
    if args.planner == "ollama":
        return OllamaPlanner(
            model=args.ollama_model,
            base_url=args.ollama_base_url,
            timeout_sec=args.llm_timeout_sec,
            num_predict=args.ollama_num_predict,
            num_gpu=args.ollama_num_gpu,
        )
    if args.planner == "static":
        return StaticPlanner(max_duration_sec=30.0)
    raise ValueError(f"Unsupported planner: {args.planner}")


def extract_env_success(events_path: Path) -> bool:
    if not events_path.exists():
        return False
    events = json.loads(events_path.read_text(encoding="utf-8"))
    for event in reversed(events):
        if event.get("event_type") == "finish_plan":
            payload = event.get("payload", {}) or {}
            if "env_success" in payload:
                return bool(payload["env_success"])
        if event.get("event_type") == "env_done":
            payload = event.get("payload", {}) or {}
            if "success" in payload:
                return bool(payload["success"])
    return False


def run_one_episode(
    args: argparse.Namespace,
    task_name: str,
    plan: TaskPlan,
    policy_adapter: Gr00tPolicyAdapter,
    episode_dir: Path,
) -> dict[str, Any]:
    max_episode_steps = args.max_episode_steps or get_task_horizon(task_name)
    simulation_config = SimulationConfig(
        env_name=f"robocasa/{task_name}",
        split=args.split,
        n_episodes=1,
        n_envs=1,
        video=VideoConfig(video_dir=str(episode_dir / "videos")),
        multistep=MultiStepConfig(
            n_action_steps=args.n_action_steps,
            max_episode_steps=max_episode_steps,
        ),
    )
    env = RobocasaVectorEnvAdapter(
        simulation_config=simulation_config,
        vlm_image_key=args.vlm_image_key,
    )
    try:
        controller = FastAuxOnlyController(
            plan=plan,
            policy=policy_adapter,
            env=env,
            fast_monitor=ActionEntropyMonitor(
                FastMonitorConfig(
                    complete_score_threshold=args.complete_score_threshold,
                    min_steps_before_trigger=args.min_steps_before_trigger,
                    cooldown_steps=args.cooldown_steps,
                )
            ),
            config=ControllerConfig(
                output_dir=str(episode_dir),
                max_total_steps=max_episode_steps,
                step_dt_sec=args.step_dt_sec,
                max_rollback_chunks=args.max_rollback_chunks,
            ),
            aux_retry_confidence_threshold=args.aux_retry_confidence_threshold,
            aux_success_confidence_threshold=args.aux_success_confidence_threshold,
            aux_decision_mode=args.aux_decision_mode,
            fast_retry_rollback_chunks=args.fast_retry_rollback_chunks,
            allow_timeout_gate=args.allow_timeout_gate,
        )
        controller.run()
    finally:
        env.close()

    env_success = extract_env_success(episode_dir / "controller_events.json")
    annotated_video = annotate_video(task_name, env_success, episode_dir) if args.annotate_videos else ""
    return {
        "env_success": env_success,
        "output_dir": str(episode_dir),
        "annotated_video": annotated_video,
    }


def run_one_task(
    args: argparse.Namespace,
    task_name: str,
    task_instruction: str,
    plan: TaskPlan | None,
    policy_adapter: Gr00tPolicyAdapter,
) -> dict[str, Any]:
    output_root = Path(args.output_root)
    env_dir = output_root / "evals" / args.split / task_name
    stats_path = env_dir / "stats.json"
    if stats_path.exists() and not args.overwrite:
        data = json.loads(stats_path.read_text(encoding="utf-8"))
        completed_episodes = int(data.get("num_episodes", 0))
        if completed_episodes >= args.n_episodes:
            success_rate = float(data.get("success_rate", 0.0))
            return {
                "task_name": task_name,
                "task_instruction": task_instruction,
                "success_rate": success_rate,
                "env_success": success_rate > 0.0,
                "status": "skipped_existing",
                "output_dir": str(env_dir),
            }
        print(
            f"[fast-aux-only] {task_name}: found {completed_episodes}/"
            f"{args.n_episodes} completed episodes; resuming",
            flush=True,
        )

    env_dir.mkdir(parents=True, exist_ok=True)
    if plan is None:
        raise RuntimeError(f"No plan was prepared for {task_name}.")
    plan.save(env_dir / "plan.json")

    start = time.perf_counter()
    episode_results = []
    for episode_idx in range(args.n_episodes):
        episode_dir = env_dir / "episodes" / f"episode_{episode_idx:03d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        if not args.overwrite:
            existing_result = load_episode_result(episode_dir, episode_idx)
            if existing_result is not None:
                print(
                    f"[fast-aux-only] {task_name} episode "
                    f"{episode_idx + 1}/{args.n_episodes} already exists; skipping",
                    flush=True,
                )
                episode_results.append(existing_result)
                save_stats(env_dir, episode_results)
                continue
        print(
            f"[fast-aux-only] {task_name} episode "
            f"{episode_idx + 1}/{args.n_episodes}",
            flush=True,
        )
        if args.overwrite:
            clear_episode_artifacts(episode_dir)
        episode_result = run_one_episode(
            args=args,
            task_name=task_name,
            plan=plan,
            policy_adapter=policy_adapter,
            episode_dir=episode_dir,
        )
        episode_result["episode_index"] = episode_idx
        episode_result["status"] = "completed"
        save_episode_stats(episode_dir, episode_result)
        episode_results.append(episode_result)
        save_stats(env_dir, episode_results)

    success_values = [float(result["env_success"]) for result in episode_results]
    success_rate = float(np.mean(success_values)) if success_values else 0.0
    save_stats(env_dir, episode_results)
    return {
        "task_name": task_name,
        "task_instruction": task_instruction,
        "env_success": success_rate > 0.0,
        "success_rate": success_rate,
        "status": "completed",
        "output_dir": str(env_dir),
        "episode_results": episode_results,
        "elapsed_sec": time.perf_counter() - start,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default=(
            "/data/zjw/workspace/Isaac-GR00T/expdata/long_horizon_controller/"
            "composite_seen_fast_aux_only_aux11000_s090_r045"
        ),
    )
    parser.add_argument("--task-set", default="composite_seen")
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--n-episodes", type=int, default=1)
    parser.add_argument("--split", default="target", choices=["pretrain", "target"])
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--aux-head-path", default=AUX_HEAD_PATH)
    parser.add_argument("--data-config", default="panda_omron")
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--planner", default="ollama", choices=["api", "ollama", "static"])
    parser.add_argument("--planner-fallback-static", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--plan-cache-dir", default="")
    parser.add_argument("--overwrite-plan", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--ollama-model", default="llama3.1:70b")
    parser.add_argument("--ollama-base-url", default="http://localhost:11434")
    parser.add_argument("--llm-timeout-sec", type=float, default=300.0)
    parser.add_argument("--ollama-num-predict", type=int, default=768)
    parser.add_argument("--ollama-num-gpu", type=int, default=33)
    parser.add_argument(
        "--vlm-image-key",
        default="video.robot0_agentview_left,video.robot0_eye_in_hand",
        help="Kept for video annotation/camera compatibility; no VLM is called.",
    )
    parser.add_argument("--n-action-steps", type=int, default=16)
    parser.add_argument("--max-episode-steps", type=int, default=0)
    parser.add_argument("--max-rollback-chunks", type=int, default=8)
    parser.add_argument("--fast-retry-rollback-chunks", type=int, default=2)
    parser.add_argument(
        "--step-dt-sec",
        type=float,
        default=0.8,
        help="Wall-clock duration represented by one action chunk; 16 actions * 0.05s by default.",
    )
    parser.add_argument("--complete-score-threshold", type=float, default=0.35)
    parser.add_argument("--min-steps-before-trigger", type=int, default=8)
    parser.add_argument("--cooldown-steps", type=int, default=8)
    parser.add_argument("--aux-retry-confidence-threshold", type=float, default=0.45)
    parser.add_argument("--aux-success-confidence-threshold", type=float, default=0.9)
    parser.add_argument("--aux-decision-mode", choices=["prob", "argmax"], default="prob")
    parser.add_argument(
        "--allow-timeout-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow timeout to query aux without a CC transition; disabled for strict CC+aux tests.",
    )
    parser.add_argument("--annotate-videos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_names = args.tasks or list(TASK_SET_REGISTRY[args.task_set])
    if args.max_tasks:
        task_names = task_names[: args.max_tasks]

    missing = [name for name in task_names if name not in DEFAULT_TASK_INSTRUCTIONS]
    if missing:
        raise ValueError(f"No default task instruction for: {missing}")

    planner = make_planner(args)
    output_root = Path(args.output_root)
    plan_cache_dir = Path(args.plan_cache_dir) if args.plan_cache_dir else None
    plans: dict[str, TaskPlan] = {}
    for task_name in task_names:
        if not args.plan_only and task_has_enough_episodes(args, task_name):
            continue
        env_dir = output_root / "evals" / args.split / task_name
        plans[task_name] = load_or_create_plan(
            planner=planner,
            task_name=task_name,
            task_instruction=DEFAULT_TASK_INSTRUCTIONS[task_name],
            env_dir=env_dir,
            plan_cache_dir=plan_cache_dir,
            overwrite_plan=args.overwrite_plan,
            fallback_static=args.planner_fallback_static,
        )
    if args.plan_only:
        rows = [
            {
                "task_name": task_name,
                "task_instruction": DEFAULT_TASK_INSTRUCTIONS[task_name],
                "status": "planned",
                "env_success": False,
                "success_rate": 0.0,
                "output_dir": str(output_root / "evals" / args.split / task_name),
                "plan_path": str(output_root / "evals" / args.split / task_name / "plan.json"),
            }
            for task_name in task_names
        ]
        save_batch_results(output_root, rows)
        print(f"Saved fast-aux-only composite plans to {output_root}", flush=True)
        return

    data_config = DATA_CONFIG_MAP[args.data_config]
    modality_config = data_config.modality_config()
    policy = Gr00tPolicy(
        model_path=args.model_path,
        modality_config=modality_config,
        modality_transform=data_config.transform(),
        embodiment_tag=args.embodiment_tag,
        denoising_steps=4,
    )
    policy_adapter = Gr00tPolicyAdapter(
        policy=policy,
        action_keys=modality_config["action"].modality_keys,
        aux_head_path=args.aux_head_path,
    )

    rows: list[dict[str, Any]] = []
    for task_name in task_names:
        print(f"[fast-aux-only] Running {task_name}", flush=True)
        try:
            row = run_one_task(
                args=args,
                task_name=task_name,
                task_instruction=DEFAULT_TASK_INSTRUCTIONS[task_name],
                plan=plans.get(task_name),
                policy_adapter=policy_adapter,
            )
        except Exception as exc:
            row = {
                "task_name": task_name,
                "task_instruction": DEFAULT_TASK_INSTRUCTIONS[task_name],
                "env_success": False,
                "status": "error",
                "error": str(exc),
                "output_dir": str(output_root / "evals" / args.split / task_name),
            }
            print(f"[fast-aux-only] ERROR {task_name}: {exc}", flush=True)
        rows.append(row)
        save_batch_results(output_root, rows)
        print(
            f"[fast-aux-only] {task_name}: status={row['status']} "
            f"success_rate={row.get('success_rate', float(row['env_success']))}",
            flush=True,
        )

    save_batch_results(output_root, rows)
    print(f"Saved fast-aux-only composite eval outputs to {output_root}", flush=True)


if __name__ == "__main__":
    main()
