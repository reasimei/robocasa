#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
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
from scripts.long_horizon_controller.fast_monitor import ActionEntropyMonitor, AuxHeadFusionMonitor
from scripts.long_horizon_controller.planner import OllamaPlanner, OpenAICompatiblePlanner, StaticPlanner
from scripts.long_horizon_controller.policy_adapters import Gr00tPolicyAdapter
from scripts.long_horizon_controller.robocasa_adapter import RobocasaVectorEnvAdapter
from scripts.long_horizon_controller.schemas import FastMonitorConfig, TaskPlan, VLMStatus, plan_from_dict
from scripts.long_horizon_controller.vlm_verifier import DEFAULT_QWEN3_VL_PATH, DryRunVerifier, LocalQwenVLVerifier, OllamaVLVerifier
from scripts.long_horizon_controller.xiaomi_policy_adapter import XiaomiPolicyAdapter


MODEL_PATH = (
    "/data/zjw/workspace/Isaac-GR00T/expdata/foundation_model_learning/"
    "target_posttraining/composite_seen/checkpoint-60000"
)
AUX_HEAD_PATH = (
    "/data/zjw/workspace/Isaac-GR00T/expdata/aux_progress/"
    "atomic_retry_3class_history_run1/checkpoint-8000"
)


DEFAULT_TASK_INSTRUCTIONS = {
    "DeliverStraw": "Take a straw from the drawer in front and place it inside the glass cup on the dining counter.",
    "GetToastedBread": "Start the toaster. Once the lever pops up, take the bread to the plate on the dining counter.",
    "KettleBoiling": "Pick the kettle from the counter and place it on a stove burner. Then turn the burner on.",
    "LoadDishwasher": "Pick up the cup and bowl from the counter, place them in the dishwasher, and close the dishwasher door.",
    "PackIdenticalLunches": "Place one vegetable and one meat in each tupperware on the nearby counter, to pack two identical lunches.",
    "PreSoakPan": "Pick the pan and sponge and place them into the sink. Then turn on the water.",
    "PrepareCoffee": "Pick the mug from the cabinet, place it under the coffee machine dispenser, and press the start button.",
    "RinseSinkBasin": "Turn on the sink and manuever the spout to wash all locations of the sink basin.",
    "ScrubCuttingBoard": (
        "Pick up the sponge from the counter and clean the cutting board by briefly "
        "scrubbing or pressing down on the cutting board. Once finished, release the sponge."
    ),
    "SearingMeat": (
        "Grab the pan from the cabinet and place it on a stove burner. "
        "Then place the meat on the stove and turn the burner on."
    ),
    "SetUpCuttingStation": (
        "Pick up the knife from the drawer and place it on the cutting board. "
        "Then place the meat from the plate to the cutting board."
    ),
    "StackBowlsCabinet": (
        "Pick up the bowls on the counter and stack them on top of one another in the open cabinet. "
        "Place the smaller bowl on top of the larger bowl."
    ),
    "SteamInMicrowave": (
        "Pick the vegetable from the sink and place it in the bowl. "
        "Then pick the bowl and place it in the microwave. "
        "Then close the microwave door and press the start button."
    ),
    "StirVegetables": "Put the vegetables in the pot. Retrieve the spatula and lightly stir the vegetables in the pot.",
    "StoreLeftoversInBowl": (
        "Pick the chicken drumstick and vegetable from their plates and place them in the bowl. "
        "Then put the bowl in the fridge."
    ),
    "WashLettuce": "Wash the lettuce in the sink by running water over it.",
}


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


def make_verifier(args: argparse.Namespace):
    if args.verifier == "qwen_vl":
        return LocalQwenVLVerifier(model_path=args.vlm_model_path)
    if args.verifier == "ollama_vl":
        return OllamaVLVerifier(
            model=args.vlm_ollama_model,
            base_url=args.vlm_ollama_base_url,
            timeout_sec=args.vlm_timeout_sec,
            num_predict=args.vlm_num_predict,
            keep_alive=args.vlm_ollama_keep_alive,
        )
    if args.verifier == "dry":
        return DryRunVerifier(default_status=VLMStatus(args.dry_vlm_status))
    raise ValueError(f"Unsupported verifier: {args.verifier}")


def load_or_create_plan(
    planner: Any,
    task_name: str,
    task_instruction: str,
    env_dir: Path,
    plan_cache_dir: Path | None,
    overwrite_plan: bool,
    fallback_static: bool = False,
) -> TaskPlan:
    cache_path = plan_cache_dir / task_name / "plan.json" if plan_cache_dir else None
    env_plan_path = env_dir / "plan.json"
    for candidate in (env_plan_path, cache_path):
        if candidate and candidate.exists() and not overwrite_plan:
            return plan_from_dict(json.loads(candidate.read_text(encoding="utf-8")))

    try:
        plan = planner.plan(task_instruction)
    except Exception as exc:
        if not fallback_static:
            raise
        print(
            f"[long-horizon-eval] WARNING {task_name}: planner failed ({exc}); "
            "falling back to a one-step static plan.",
            flush=True,
        )
        plan = StaticPlanner(max_duration_sec=30.0).plan(task_instruction)
    env_plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan.save(env_plan_path)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        plan.save(cache_path)
    return plan


def task_has_enough_episodes(args: argparse.Namespace, task_name: str) -> bool:
    if args.overwrite:
        return False
    stats_path = Path(args.output_root) / "evals" / args.split / task_name / "stats.json"
    if not stats_path.exists():
        return False
    try:
        data = json.loads(stats_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return int(data.get("num_episodes", 0)) >= args.n_episodes


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


def latest_raw_video(video_dir: Path) -> Path | None:
    if not video_dir.exists():
        return None
    candidates = [
        path
        for path in video_dir.glob("*.mp4")
        if "annotated" not in path.name and "envsuccess" not in path.name
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def annotate_video(task_name: str, env_success: bool, env_dir: Path) -> str:
    video_dir = env_dir / "videos"
    raw_video = latest_raw_video(video_dir)
    events_path = env_dir / "controller_events.json"
    if raw_video is None or not events_path.exists():
        return ""
    output = video_dir / f"{task_name}_envsuccess_{str(env_success).lower()}_annotated_bars.mp4"
    command = [
        sys.executable,
        "scripts/long_horizon_controller/annotate_rollout_video.py",
        "--video",
        str(raw_video),
        "--events",
        str(events_path),
        "--output",
        str(output),
    ]
    last_error = ""
    for attempt in range(3):
        if attempt:
            time.sleep(2.0)
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[2],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0:
            return str(output)
        last_error = (completed.stderr or completed.stdout or "").strip()
    print(
        f"[long-horizon-eval] WARNING {task_name}: video annotation failed; "
        f"keeping episode stats. {last_error[-500:]}",
        flush=True,
    )
    if output.exists():
        output.unlink()
    return ""


def save_stats(env_dir: Path, episode_results: list[dict[str, Any]]) -> None:
    successes = [float(result["env_success"]) for result in episode_results]
    stats = {
        "num_episodes": len(episode_results),
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "successes": [bool(result["env_success"]) for result in episode_results],
        "episode_results": episode_results,
    }
    (env_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


def save_episode_stats(episode_dir: Path, episode_result: dict[str, Any]) -> None:
    stats = {
        "num_episodes": 1,
        "success_rate": float(episode_result["env_success"]),
        "env_success": bool(episode_result["env_success"]),
        "episode_index": int(episode_result["episode_index"]),
        "annotated_video": episode_result.get("annotated_video", ""),
    }
    (episode_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


def clear_episode_artifacts(episode_dir: Path) -> None:
    """Remove outputs from an explicitly requested episode overwrite."""
    for name in (
        "controller_events.json",
        "action_history.json",
        "stats.json",
    ):
        path = episode_dir / name
        if path.exists():
            path.unlink()
    for directory_name in ("videos", "vlm_frames"):
        directory = episode_dir / directory_name
        if not directory.exists():
            continue
        for path in directory.iterdir():
            if path.is_file() or path.is_symlink():
                path.unlink()


def load_episode_result(episode_dir: Path, episode_idx: int) -> dict[str, Any] | None:
    stats_path = episode_dir / "stats.json"
    if not stats_path.exists():
        return None
    data = json.loads(stats_path.read_text(encoding="utf-8"))
    if "env_success" in data:
        env_success = bool(data["env_success"])
    elif "success_rate" in data:
        env_success = float(data["success_rate"]) > 0.0
    else:
        return None
    return {
        "env_success": env_success,
        "output_dir": str(episode_dir),
        "annotated_video": str(data.get("annotated_video", "")),
        "episode_index": episode_idx,
        "status": "skipped_existing",
    }


def save_batch_results(output_root: Path, rows: list[dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    completed = [row for row in rows if row.get("status") == "completed"]
    success_values = [float(row.get("success_rate", float(row["env_success"]))) for row in completed]
    payload = {
        "num_tasks": len(rows),
        "num_completed": len(completed),
        "success_rate": float(np.mean(success_values)) if success_values else 0.0,
        "results": rows,
    }
    (output_root / "long_horizon_eval_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_one_episode(
    args: argparse.Namespace,
    task_name: str,
    plan: TaskPlan,
    verifier: Any,
    policy_adapter: Any,
    episode_dir: Path,
) -> dict[str, Any]:
    max_episode_steps = args.max_episode_steps or get_task_horizon(task_name)
    video_delta_indices = np.array([0])
    state_delta_indices = np.array([0])
    if args.policy_backend == "xiaomi":
        video_delta_indices = np.arange(
            -(args.xiaomi_history_length - 1) * args.xiaomi_history_interval_steps,
            1,
            args.xiaomi_history_interval_steps,
        )
        state_delta_indices = video_delta_indices.copy()
    simulation_config = SimulationConfig(
        env_name=f"robocasa/{task_name}",
        split=args.split,
        n_episodes=1,
        n_envs=1,
        video=VideoConfig(video_dir=str(episode_dir / "videos")),
        multistep=MultiStepConfig(
            video_delta_indices=video_delta_indices,
            state_delta_indices=state_delta_indices,
            n_action_steps=args.n_action_steps,
            max_episode_steps=max_episode_steps,
        ),
    )
    env = RobocasaVectorEnvAdapter(
        simulation_config=simulation_config,
        vlm_image_key=args.vlm_image_key,
    )
    try:
        if hasattr(policy_adapter, "reset"):
            policy_adapter.reset()
        controller = LongHorizonController(
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
            aux_fusion=AuxHeadFusionMonitor(
                retry_confidence_threshold=args.aux_retry_confidence_threshold,
                success_confidence_threshold=args.aux_success_confidence_threshold,
                cooldown_steps=args.aux_cooldown_steps,
            ),
            vlm_verifier=verifier,
            config=ControllerConfig(
                output_dir=str(episode_dir),
                max_total_steps=max_episode_steps,
                step_dt_sec=args.step_dt_sec,
                vlm_history_frames=args.vlm_history_frames,
                vlm_history_interval_sec=args.vlm_history_interval_sec,
                save_vlm_frames=args.save_vlm_frames,
                max_rollback_chunks=args.max_rollback_chunks,
            ),
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
    verifier: Any,
    policy_adapter: Any,
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
            f"[long-horizon-eval] {task_name}: found {completed_episodes}/"
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
                    f"[long-horizon-eval] {task_name} episode "
                    f"{episode_idx + 1}/{args.n_episodes} already exists; skipping",
                    flush=True,
                )
                episode_results.append(existing_result)
                save_stats(env_dir, episode_results)
                continue
        print(
            f"[long-horizon-eval] {task_name} episode "
            f"{episode_idx + 1}/{args.n_episodes}",
            flush=True,
        )
        if args.overwrite:
            clear_episode_artifacts(episode_dir)
        episode_result = run_one_episode(
            args=args,
            task_name=task_name,
            plan=plan,
            verifier=verifier,
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
    parser.add_argument("--output-root", default="/data/zjw/workspace/Isaac-GR00T/expdata/long_horizon_controller/composite_seen_full")
    parser.add_argument("--task-set", default="composite_seen")
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--n-episodes", type=int, default=1)
    parser.add_argument("--split", default="target", choices=["pretrain", "target"])
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument(
        "--policy-backend",
        default="gr00t",
        choices=["gr00t", "xiaomi"],
        help="Base VLA implementation used by the long-horizon controller.",
    )
    parser.add_argument("--aux-head-path", default=AUX_HEAD_PATH)
    parser.add_argument(
        "--xiaomi-history-length",
        type=int,
        default=4,
        help="Number of observation frames per camera sent to Xiaomi VLA.",
    )
    parser.add_argument(
        "--xiaomi-history-interval-steps",
        type=int,
        default=2,
        help="Low-level simulator step interval between Xiaomi history frames.",
    )
    parser.add_argument(
        "--xiaomi-num-diffusion-steps",
        type=int,
        default=5,
        help="Number of Xiaomi action-head integration steps.",
    )
    parser.add_argument("--data-config", default="panda_omron")
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--planner", default="ollama", choices=["api", "ollama", "static"])
    parser.add_argument("--planner-fallback-static", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--plan-cache-dir", default="")
    parser.add_argument("--overwrite-plan", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--verifier", default="ollama_vl", choices=["qwen_vl", "ollama_vl", "dry"])
    parser.add_argument("--dry-vlm-status", default="complete")
    parser.add_argument("--vlm-model-path", default=DEFAULT_QWEN3_VL_PATH)
    parser.add_argument("--vlm-ollama-model", default="qwen3-vl:8b")
    parser.add_argument("--vlm-ollama-base-url", default="http://localhost:11434")
    parser.add_argument("--vlm-ollama-keep-alive", default="30m")
    parser.add_argument("--vlm-timeout-sec", type=float, default=120.0)
    parser.add_argument("--vlm-num-predict", type=int, default=1024)
    parser.add_argument(
        "--vlm-history-frames",
        type=int,
        default=2,
        help="Temporal snapshots sent to VLM; default is 1s-before plus current.",
    )
    parser.add_argument("--vlm-history-interval-sec", type=float, default=1.0)
    parser.add_argument("--save-vlm-frames", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ollama-model", default="llama3.1:70b")
    parser.add_argument("--ollama-base-url", default="http://localhost:11434")
    parser.add_argument("--llm-timeout-sec", type=float, default=300.0)
    parser.add_argument("--ollama-num-predict", type=int, default=768)
    parser.add_argument("--ollama-num-gpu", type=int, default=33)
    parser.add_argument(
        "--vlm-image-key",
        default="video.robot0_agentview_left,video.robot0_eye_in_hand",
        help="Comma-separated VLM camera keys, ordered as agentview then eye-in-hand.",
    )
    parser.add_argument("--n-action-steps", type=int, default=16)
    parser.add_argument("--max-episode-steps", type=int, default=0)
    parser.add_argument(
        "--max-rollback-chunks",
        type=int,
        default=8,
        help="Maximum policy action chunks that rollback_retry may reverse.",
    )
    parser.add_argument("--step-dt-sec", type=float, default=0.05)
    parser.add_argument("--complete-score-threshold", type=float, default=0.35)
    parser.add_argument("--min-steps-before-trigger", type=int, default=8)
    parser.add_argument("--cooldown-steps", type=int, default=8)
    parser.add_argument("--aux-retry-confidence-threshold", type=float, default=0.6)
    parser.add_argument("--aux-success-confidence-threshold", type=float, default=0.7)
    parser.add_argument("--aux-cooldown-steps", type=int, default=8)
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
        print(f"Saved long-horizon composite plans to {output_root}", flush=True)
        return

    verifier = make_verifier(args)

    if args.policy_backend == "xiaomi":
        if args.aux_head_path:
            print(
                "[long-horizon-eval] INFO Xiaomi backend does not use the GR00T "
                "auxiliary checkpoint; ignoring --aux-head-path.",
                flush=True,
            )
        policy_adapter = XiaomiPolicyAdapter(
            model_path=args.model_path,
            history_length=args.xiaomi_history_length,
            action_steps=args.n_action_steps,
            num_diffusion_steps=args.xiaomi_num_diffusion_steps,
        )
    else:
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
        print(f"[long-horizon-eval] Running {task_name}", flush=True)
        try:
            row = run_one_task(
                args=args,
                task_name=task_name,
                task_instruction=DEFAULT_TASK_INSTRUCTIONS[task_name],
                plan=plans.get(task_name),
                verifier=verifier,
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
            print(f"[long-horizon-eval] ERROR {task_name}: {exc}", flush=True)
        rows.append(row)
        save_batch_results(output_root, rows)
        print(
            f"[long-horizon-eval] {task_name}: status={row['status']} "
            f"success_rate={row.get('success_rate', float(row['env_success']))}",
            flush=True,
        )

    save_batch_results(output_root, rows)
    print(f"Saved long-horizon composite eval outputs to {output_root}", flush=True)


if __name__ == "__main__":
    main()
