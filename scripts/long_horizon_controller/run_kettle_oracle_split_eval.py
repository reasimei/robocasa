#!/usr/bin/env python3
"""Compare KettleBoiling full-task control against oracle subtask switching.

The oracle reads simulator state directly. It is intentionally separate from the
VLM controller so this experiment measures the value of correct subtask timing,
not verifier accuracy.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import robocasa.utils.object_utils as OU

from gr00t.eval.simulation import MultiStepConfig, SimulationConfig, VideoConfig
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy
from scripts.long_horizon_controller.policy_adapters import Gr00tPolicyAdapter
from scripts.long_horizon_controller.robocasa_adapter import RobocasaVectorEnvAdapter
from scripts.long_horizon_controller.schemas import SubtaskSpec, TaskPlan, plan_from_dict
from scripts.long_horizon_controller.xiaomi_policy_adapter import XiaomiPolicyAdapter


DEFAULT_MODEL_PATH = (
    "/data/zjw/workspace/Isaac-GR00T/expdata/foundation_model_learning/"
    "target_posttraining/composite_seen/checkpoint-60000"
)
DEFAULT_XIAOMI_MODEL_PATH = (
    "/data/zjw/workspace/Isaac-GR00T/expdata/"
    "Xiaomi-Robotics-1-RoboCasa365"
)
DEFAULT_PLAN_PATH = (
    "/data/zjw/workspace/Isaac-GR00T/expdata/long_horizon_controller/"
    "composite_seen_plan_cache_llama70b/KettleBoiling/plan.json"
)
TASK_NAME = "KettleBoiling"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=["full_task", "oracle_split"],
        help="full_task keeps the complete instruction throughout; oracle_split switches at simulator labels.",
    )
    parser.add_argument(
        "--output-root",
        default=(
            "/data/zjw/workspace/Isaac-GR00T/expdata/long_horizon_controller/"
            "kettle_oracle_split"
        ),
    )
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--split", choices=["pretrain", "target"], default="target")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--policy-backend",
        choices=["gr00t", "xiaomi"],
        default="gr00t",
        help="Base VLA policy used for the oracle split comparison.",
    )
    parser.add_argument("--plan-path", default=DEFAULT_PLAN_PATH)
    parser.add_argument("--data-config", default="panda_omron")
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--n-action-steps", type=int, default=16)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--xiaomi-history-length", type=int, default=4)
    parser.add_argument("--xiaomi-history-interval-steps", type=int, default=2)
    parser.add_argument("--xiaomi-num-diffusion-steps", type=int, default=5)
    parser.add_argument(
        "--subtask-prompt-format",
        choices=["structured", "natural", "step_sentence", "subtask_only"],
        default="structured",
        help="How oracle_split combines the full task and current subtask instruction.",
    )
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def instruction_for_subtask(
    plan: TaskPlan,
    subtask: SubtaskSpec,
    subtask_index: int,
    prompt_format: str,
) -> str:
    if prompt_format == "natural":
        return (
            f"{plan.task_instruction}\n"
            f"Now focus on: {subtask.instruction}"
        )
    if prompt_format == "step_sentence":
        prefixes = ("First", "Next", "Finally")
        prefix = prefixes[min(subtask_index, len(prefixes) - 1)]
        subtask_instruction = subtask.instruction.strip()
        if subtask_instruction:
            subtask_instruction = subtask_instruction[0].lower() + subtask_instruction[1:]
        return f"{plan.task_instruction} {prefix}, {subtask_instruction}"
    if prompt_format == "subtask_only":
        return f"{subtask.instruction}"
    return (
        f"Overall task: {plan.task_instruction}\n"
        f"Current subtask: {subtask.instruction}"
    )


def unwrap_kettle_env(adapter: RobocasaVectorEnvAdapter) -> Any:
    """Find the base KettleBoiling environment beneath vector and gym wrappers."""
    current = adapter._env.envs[0]
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if all(hasattr(current, attr) for attr in ("sim", "stove", "objects", "obj_body_id")):
            return current
        current = getattr(current, "env", None)
    raise RuntimeError("Could not locate the base KettleBoiling environment through wrappers.")


def kettle_on_burner(env: Any) -> tuple[bool, str | None, float | None]:
    """Match KettleBoiling's stove-contact and 0.15 m burner-site criterion."""
    kettle = env.objects["obj"]
    kettle_pos = np.asarray(env.sim.data.body_xpos[env.obj_body_id[kettle.name]])[:2]
    if not OU.check_obj_fixture_contact(env, "obj", env.stove):
        return False, None, None

    closest_location: str | None = None
    closest_distance: float | None = None
    for location, site in env.stove.burner_sites.items():
        if site is None:
            continue
        burner_pos = np.asarray(env.sim.data.get_site_xpos(site.get("name")))[:2]
        distance = float(np.linalg.norm(burner_pos - kettle_pos))
        if closest_distance is None or distance < closest_distance:
            closest_location = location
            closest_distance = distance
    return bool(closest_distance is not None and closest_distance < 0.15), closest_location, closest_distance


def kettle_oracle_labels(env: Any) -> dict[str, Any]:
    """Return monotonic task-stage labels from simulator state.

    `pick_kettle` uses Robocasa's contact-and-closed-gripper grasp predicate.
    Placement mirrors the task's own fixture-contact and burner-distance check.
    The final stage is exactly the environment success predicate.
    """
    grasped = bool(OU.check_obj_grasped(env, "obj"))
    on_burner, burner_location, burner_distance = kettle_on_burner(env)
    gripper_far = bool(OU.gripper_obj_far(env, "obj"))
    burner_on = bool(
        on_burner
        and burner_location is not None
        and env.stove.is_burner_on(env=env, burner_loc=burner_location)
    )
    placed = bool(on_burner and gripper_far)
    final_success = bool(placed and burner_on)
    return {
        "pick_kettle": grasped,
        "place_kettle_on_stove": placed,
        "turn_on_burner": final_success,
        "grasped": grasped,
        "kettle_on_burner": on_burner,
        "burner_location": burner_location,
        "burner_distance": burner_distance,
        "gripper_far": gripper_far,
        "burner_on": burner_on,
        "env_check_success": bool(env._check_success()),
    }


def make_env(args: argparse.Namespace, video_root: Path) -> RobocasaVectorEnvAdapter:
    video_dir = str(video_root / "videos") if args.video else None
    video_delta_indices = np.array([0])
    state_delta_indices = np.array([0])
    if args.policy_backend == "xiaomi":
        video_delta_indices = np.arange(
            -(args.xiaomi_history_length - 1) * args.xiaomi_history_interval_steps,
            1,
            args.xiaomi_history_interval_steps,
        )
        state_delta_indices = video_delta_indices.copy()
    config = SimulationConfig(
        env_name=f"robocasa/{TASK_NAME}",
        split=args.split,
        n_episodes=1,
        n_envs=1,
        video=VideoConfig(video_dir=video_dir),
        multistep=MultiStepConfig(
            video_delta_indices=video_delta_indices,
            state_delta_indices=state_delta_indices,
            n_action_steps=args.n_action_steps,
            max_episode_steps=args.max_episode_steps,
        ),
    )
    return RobocasaVectorEnvAdapter(simulation_config=config)


def _trim_video_text(
    text: str,
    max_width: int,
    font: int,
    scale: float,
    thickness: int,
) -> str:
    import cv2

    if cv2.getTextSize(text, font, scale, thickness)[0][0] <= max_width:
        return text
    suffix = "..."
    trimmed = text
    while trimmed:
        candidate = trimmed + suffix
        if cv2.getTextSize(candidate, font, scale, thickness)[0][0] <= max_width:
            return candidate
        trimmed = trimmed[:-1]
    return suffix


def annotate_episode_video(
    args: argparse.Namespace,
    plan: TaskPlan,
    result: dict[str, Any],
    episode_dir: Path,
) -> Path | None:
    """Create one playable episode video with the active subtask in the top-left."""
    import cv2

    video_dir = episode_dir / "videos"
    if not video_dir.exists():
        return None

    output_path = video_dir / (
        f"{TASK_NAME}_episode_{int(result['episode_index']):03d}_"
        f"envsuccess_{str(bool(result['env_success'])).lower()}_subtask.mp4"
    )
    raw_paths = [
        path
        for path in video_dir.glob("*.mp4")
        if path != output_path and "subtask" not in path.stem
    ]
    if not raw_paths:
        return output_path if output_path.exists() else None
    source_path = max(raw_paths, key=lambda path: path.stat().st_mtime)

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open recorded video: {source_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Recorded video has invalid dimensions: {source_path}")

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.42
    thickness = 1
    pad = 8
    line_height = 18
    max_text_width = width - 2 * pad - 16
    task_by_id = {subtask.subtask_id: subtask for subtask in plan.subtasks}

    segments: list[tuple[int, int, list[str]]] = []
    if args.mode == "full_task":
        segments.append(
            (
                0,
                2**31 - 1,
                ["subtask: full_task", plan.task_instruction],
            )
        )
    else:
        steps_per_render = 2
        for trace in result.get("label_trace", []):
            policy_call = int(trace["policy_call"])
            start_frame = int(
                (policy_call - 1) * args.n_action_steps // steps_per_render
            )
            end_frame = int(
                policy_call * args.n_action_steps // steps_per_render
            ) - 1
            subtask_id = str(trace.get("active_subtask_id", "unknown"))
            subtask = task_by_id.get(subtask_id)
            instruction = subtask.instruction if subtask is not None else ""
            segments.append(
                (
                    start_frame,
                    end_frame,
                    [f"subtask: {subtask_id}", instruction],
                )
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps}",
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-profile:v",
        "baseline",
        "-level",
        "3.0",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    writer = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert writer.stdin is not None

    segment_index = 0
    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            while (
                segment_index + 1 < len(segments)
                and frame_index > segments[segment_index][1]
            ):
                segment_index += 1
            lines: list[str] = []
            if segments and segments[segment_index][0] <= frame_index:
                lines = segments[segment_index][2]
            if lines:
                text_lines = [
                    _trim_video_text(
                        line,
                        max_text_width,
                        font,
                        scale,
                        thickness,
                    )
                    for line in lines
                    if line
                ]
                box_height = len(text_lines) * line_height + 2 * pad
                box_width = width - 8
                cv2.rectangle(
                    frame,
                    (4, 4),
                    (box_width, 4 + box_height),
                    (0, 0, 0),
                    -1,
                )
                cv2.rectangle(
                    frame,
                    (4, 4),
                    (box_width, 4 + box_height),
                    (0, 220, 255),
                    1,
                )
                for line_index, line in enumerate(text_lines):
                    baseline = 4 + pad + 13 + line_index * line_height
                    cv2.putText(
                        frame,
                        line,
                        (4 + pad, baseline),
                        font,
                        scale,
                        (255, 255, 255),
                        thickness,
                        cv2.LINE_AA,
                    )
            writer.stdin.write(frame.tobytes())
            frame_index += 1
    finally:
        cap.release()
        writer.stdin.close()
        stderr = (
            writer.stderr.read().decode("utf-8", errors="replace")
            if writer.stderr
            else ""
        )
        return_code = writer.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed with code {return_code}:\n{stderr}")

    source_path.unlink(missing_ok=True)
    return output_path


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def existing_episode_result(episode_dir: Path) -> dict[str, Any] | None:
    path = episode_dir / "result.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def run_episode(
    args: argparse.Namespace,
    plan: TaskPlan,
    policy: Any,
    env: RobocasaVectorEnvAdapter,
    episode_index: int,
) -> dict[str, Any]:
    seed = args.seed_base + episode_index
    if args.policy_backend == "xiaomi":
        import torch

        torch.manual_seed(seed)
    if hasattr(policy, "reset"):
        policy.reset()
    observation, _ = env._env.reset(seed=seed)
    base_env = unwrap_kettle_env(env)
    subtask_index = 0
    first_true_steps: dict[str, int] = {}
    transitions: list[dict[str, Any]] = []
    label_trace: list[dict[str, Any]] = []
    env_success = False
    done = False
    policy_calls = 0
    simulator_steps = 0

    initial_labels = kettle_oracle_labels(base_env)
    for subtask_id in ("pick_kettle", "place_kettle_on_stove", "turn_on_burner"):
        if initial_labels[subtask_id]:
            first_true_steps[subtask_id] = 0

    while not done and simulator_steps < args.max_episode_steps:
        if args.mode == "full_task":
            instruction = plan.task_instruction
            active_subtask_id = "full_task"
        else:
            active_subtask = plan.subtasks[min(subtask_index, len(plan.subtasks) - 1)]
            instruction = instruction_for_subtask(
                plan,
                active_subtask,
                subtask_index,
                args.subtask_prompt_format,
            )
            active_subtask_id = active_subtask.subtask_id

        action, _ = policy.act(observation, instruction)
        observation, _, done, info = env.step(action)
        policy_calls += 1
        simulator_steps = min(
            args.max_episode_steps,
            simulator_steps + args.n_action_steps,
        )
        env_success = bool(info.get("success", False)) or env_success

        labels = kettle_oracle_labels(base_env)
        for subtask_id in ("pick_kettle", "place_kettle_on_stove", "turn_on_burner"):
            if labels[subtask_id] and subtask_id not in first_true_steps:
                first_true_steps[subtask_id] = simulator_steps
        label_trace.append(
            {
                "simulator_steps": simulator_steps,
                "policy_call": policy_calls,
                "active_subtask_id": active_subtask_id,
                **labels,
            }
        )

        if args.mode == "oracle_split":
            while (
                subtask_index < len(plan.subtasks) - 1
                and bool(labels[plan.subtasks[subtask_index].subtask_id])
            ):
                completed_subtask = plan.subtasks[subtask_index]
                subtask_index += 1
                transitions.append(
                    {
                        "from_subtask_id": completed_subtask.subtask_id,
                        "to_subtask_id": plan.subtasks[subtask_index].subtask_id,
                        "simulator_steps": simulator_steps,
                        "policy_call": policy_calls,
                        "oracle_labels": labels,
                    }
                )

        if env_success:
            break

    final_labels = kettle_oracle_labels(base_env)
    env_success = bool(env_success or final_labels["env_check_success"])
    return {
        "episode_index": episode_index,
        "seed": seed,
        "mode": args.mode,
        "policy_backend": args.policy_backend,
        "subtask_prompt_format": args.subtask_prompt_format,
        "env_success": env_success,
        "done": bool(done),
        "policy_calls": policy_calls,
        "simulator_steps": simulator_steps,
        "final_subtask_index": subtask_index if args.mode == "oracle_split" else None,
        "first_oracle_true_simulator_steps": first_true_steps,
        "transitions": transitions,
        "final_oracle_labels": final_labels,
        "label_trace": label_trace,
    }


def summarize(results: list[dict[str, Any]], args: argparse.Namespace, plan: TaskPlan) -> dict[str, Any]:
    successes = np.asarray([bool(result["env_success"]) for result in results], dtype=float)
    rate = float(successes.mean()) if len(successes) else 0.0
    stderr = math.sqrt(rate * (1.0 - rate) / len(successes)) if len(successes) else 0.0
    return {
        "task_name": TASK_NAME,
        "mode": args.mode,
        "policy_backend": args.policy_backend,
        "subtask_prompt_format": args.subtask_prompt_format,
        "n_episodes": len(results),
        "successes": successes.astype(bool).tolist(),
        "success_rate": rate,
        "success_rate_standard_error": stderr,
        "success_rate_95ci_normal": [
            max(0.0, rate - 1.96 * stderr),
            min(1.0, rate + 1.96 * stderr),
        ],
        "seed_base": args.seed_base,
        "episode_seeds": [result["seed"] for result in results],
        "max_episode_steps": args.max_episode_steps,
        "n_action_steps": args.n_action_steps,
        "task_instruction": plan.task_instruction,
        "subtasks": [subtask.subtask_id for subtask in plan.subtasks],
        "episode_results": results,
    }


def main() -> None:
    args = parse_args()
    if args.n_episodes < 1:
        raise ValueError("--n-episodes must be positive.")
    if args.max_episode_steps < args.n_action_steps:
        raise ValueError("--max-episode-steps must be >= --n-action-steps.")

    plan = plan_from_dict(json.loads(Path(args.plan_path).read_text(encoding="utf-8")))
    expected_ids = ["pick_kettle", "place_kettle_on_stove", "turn_on_burner"]
    actual_ids = [subtask.subtask_id for subtask in plan.subtasks]
    if actual_ids != expected_ids:
        raise ValueError(
            f"This oracle expects KettleBoiling subtasks {expected_ids}, got {actual_ids}. "
            "Pass a plan with exactly these task-stage IDs."
        )

    mode_dir = Path(args.output_root) / args.mode
    episodes_dir = mode_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    plan.save(mode_dir / "plan.json")
    write_json(
        mode_dir / "config.json",
        {
            **vars(args),
            "oracle_definition": {
                "pick_kettle": "OU.check_obj_grasped(env, 'obj')",
                "place_kettle_on_stove": (
                    "kettle contacts stove, is within 0.15 m of a burner site, "
                    "and OU.gripper_obj_far(env, 'obj')"
                ),
                "turn_on_burner": (
                    "place_kettle_on_stove plus the matched burner is on; "
                    "equivalent to KettleBoiling._check_success()"
                ),
            },
        },
    )

    if args.policy_backend == "xiaomi" and args.model_path == DEFAULT_MODEL_PATH:
        args.model_path = DEFAULT_XIAOMI_MODEL_PATH
    policy_adapter = make_policy(args)

    start = time.perf_counter()
    results: list[dict[str, Any]] = []
    for episode_index in range(args.n_episodes):
        episode_dir = episodes_dir / f"episode_{episode_index:03d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        result_path = episode_dir / "result.json"
        existing = existing_episode_result(episode_dir)
        if existing is not None and not args.overwrite:
            print(
                f"[kettle-oracle] {args.mode} episode {episode_index + 1}/{args.n_episodes} "
                "already exists; skipping",
                flush=True,
            )
            results.append(existing)
            write_json(mode_dir / "summary.json", summarize(results, args, plan))
            continue

        print(
            f"[kettle-oracle] {args.mode} episode {episode_index + 1}/{args.n_episodes} "
            f"(seed={args.seed_base + episode_index})",
            flush=True,
        )
        env = None
        try:
            env = make_env(args, episode_dir)
            result = run_episode(args, plan, policy_adapter, env, episode_index)
        except Exception as exc:
            result = {
                "episode_index": episode_index,
                "seed": args.seed_base + episode_index,
                "mode": args.mode,
                "policy_backend": args.policy_backend,
                "subtask_prompt_format": args.subtask_prompt_format,
                "env_success": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            print(f"[kettle-oracle] ERROR episode {episode_index}: {result['error']}", flush=True)
        finally:
            if env is not None:
                env.close()

        if args.video:
            try:
                video_path = annotate_episode_video(args, plan, result, episode_dir)
                if video_path is not None:
                    result["video_path"] = str(video_path.relative_to(episode_dir))
            except Exception as exc:
                result["video_error"] = f"{type(exc).__name__}: {exc}"
                print(
                    f"[kettle-oracle] VIDEO ERROR episode {episode_index}: "
                    f"{result['video_error']}",
                    flush=True,
                )
        write_json(result_path, result)
        results.append(result)
        summary = summarize(results, args, plan)
        write_json(mode_dir / "summary.json", summary)
        print(
            f"[kettle-oracle] {args.mode} episode {episode_index + 1}: "
            f"success={result['env_success']} running_sr={summary['success_rate']:.3f}",
            flush=True,
        )

    summary = summarize(results, args, plan)
    summary["elapsed_sec"] = time.perf_counter() - start
    write_json(mode_dir / "summary.json", summary)
    print(
        f"[kettle-oracle] complete mode={args.mode} n={len(results)} "
        f"success_rate={summary['success_rate']:.3f} "
        f"95ci={summary['success_rate_95ci_normal']} output={mode_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
