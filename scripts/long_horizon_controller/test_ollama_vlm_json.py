#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
from PIL import Image

from .schemas import plan_from_dict
from .vlm_verifier import OllamaVLVerifier


DEFAULT_EVAL_ROOT = Path(
    "expdata/long_horizon_controller/"
    "composite_seen_full_lhc_aux8000_qwen3vl8b_1eps/evals/target"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test whether an Ollama VLM verifier returns strict structured JSON."
    )
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--task", default="PreSoakPan")
    parser.add_argument("--episode", default="episode_000")
    parser.add_argument("--plan", type=Path, default=None)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--model", default="qwen2.5vl:7b")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--subtask-index", type=int, default=0)
    parser.add_argument(
        "--follow-subtasks",
        action="store_true",
        help="After a complete decision, evaluate later steps with the next plan subtask.",
    )
    parser.add_argument("--history-frames", type=int, default=4)
    parser.add_argument(
        "--controller-steps",
        nargs="*",
        type=int,
        default=None,
        help="Controller steps to replay. If omitted, test the last video frames.",
    )
    parser.add_argument("--n-action-steps", type=int, default=16)
    parser.add_argument("--steps-per-render", type=int, default=2)
    parser.add_argument("--step-dt-sec", type=float, default=0.05)
    parser.add_argument("--history-interval-sec", type=float, default=1.0)
    parser.add_argument("--num-predict", type=int, default=256)
    parser.add_argument("--timeout-sec", type=float, default=180.0)
    parser.add_argument(
        "--format-mode",
        choices=("schema", "json"),
        default="schema",
        help="Use strict JSON schema or Ollama's generic JSON grammar.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def resolve_plan_path(args: argparse.Namespace) -> Path:
    if args.plan:
        return args.plan
    return args.eval_root / args.task / "plan.json"


def resolve_video_path(args: argparse.Namespace) -> Path:
    if args.video:
        return args.video
    video_dir = args.eval_root / args.task / "episodes" / args.episode / "videos"
    candidates = sorted(
        path for path in video_dir.glob("*.mp4") if "annotated" not in path.name
    )
    if not candidates:
        raise FileNotFoundError(f"No non-annotated mp4 found under {video_dir}")
    return candidates[0]


def sample_video_frames(video_path: Path, count: int) -> list[Image.Image]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise RuntimeError(f"Video has no readable frames: {video_path}")
        count = max(1, min(count, total_frames))
        start = max(0, total_frames - count)
        indices = list(range(start, total_frames))
        frames: list[Image.Image] = []
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame_bgr = cap.read()
            if not ok:
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        if not frames:
            raise RuntimeError(f"Failed to decode sampled frames from {video_path}")
        return frames
    finally:
        cap.release()


def sample_controller_step_frames(
    video_path: Path,
    controller_step: int,
    count: int,
    n_action_steps: int,
    steps_per_render: int,
    step_dt_sec: float,
    interval_sec: float,
) -> tuple[list[Image.Image], list[float]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise RuntimeError(f"Video has no readable frames: {video_path}")
        frame_scale = n_action_steps / max(steps_per_render, 1)
        target_steps = [
            controller_step - offset * max(1, round(interval_sec / step_dt_sec))
            for offset in range(count - 1, 0, -1)
        ]
        target_steps.append(controller_step)
        images: list[Image.Image] = []
        timestamps: list[float] = []
        for target_step in target_steps:
            if target_step < 0:
                continue
            frame_index = min(
                total_frames - 1,
                max(0, int(round(target_step * frame_scale))),
            )
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame_bgr = cap.read()
            if not ok:
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            images.append(Image.fromarray(frame_rgb))
            timestamps.append(round((target_step - controller_step) * step_dt_sec, 3))
        if not images:
            raise RuntimeError(
                f"Failed to decode controller step {controller_step} from {video_path}"
            )
        return images, timestamps
    finally:
        cap.release()


def main() -> None:
    args = parse_args()
    plan_path = resolve_plan_path(args)
    video_path = resolve_video_path(args)
    plan = plan_from_dict(json.loads(plan_path.read_text(encoding="utf-8")))
    if args.subtask_index < 0 or args.subtask_index >= len(plan.subtasks):
        raise IndexError(
            f"--subtask-index {args.subtask_index} is outside plan with "
            f"{len(plan.subtasks)} subtasks."
        )
    verifier = OllamaVLVerifier(
        model=args.model,
        base_url=args.base_url,
        timeout_sec=args.timeout_sec,
        temperature=0.0,
        num_predict=args.num_predict,
        format_mode=args.format_mode,
    )
    controller_steps = args.controller_steps
    if not controller_steps:
        controller_steps = [None]
    decisions: list[dict[str, Any]] = []
    active_subtask_index = args.subtask_index
    for controller_step in controller_steps:
        current = plan.subtasks[active_subtask_index]
        next_subtask = (
            plan.subtasks[active_subtask_index + 1]
            if active_subtask_index + 1 < len(plan.subtasks)
            else None
        )
        setattr(current, "task_instruction", plan.task_instruction)
        if next_subtask:
            setattr(next_subtask, "task_instruction", plan.task_instruction)
        if controller_step is None:
            frames = sample_video_frames(video_path, args.history_frames)
            timestamps = None
        else:
            frames, timestamps = sample_controller_step_frames(
                video_path=video_path,
                controller_step=controller_step,
                count=args.history_frames,
                n_action_steps=args.n_action_steps,
                steps_per_render=args.steps_per_render,
                step_dt_sec=args.step_dt_sec,
                interval_sec=args.history_interval_sec,
            )
        setattr(current, "vlm_image_timestamps_sec", timestamps)
        setattr(current, "vlm_history_interval_sec", args.history_interval_sec)
        setattr(
            current,
            "controller_context",
            (
                f"Offline replay of controller step {controller_step}; "
                "evaluate the current subtask only."
            ),
        )
        decision = verifier.verify(frames, current, next_subtask)
        decisions.append(
            {
                "controller_step": controller_step,
                "subtask_index": active_subtask_index,
                "subtask_id": current.subtask_id,
                "subtask_instruction": current.instruction,
                "num_images": len(frames),
                "image_timestamps_sec": timestamps,
                "status": decision.status.value,
                "failure_type": decision.failure_type,
                "finish_state_satisfied": decision.finish_state_satisfied,
                "next_start_plausible": decision.next_start_plausible,
                "confidence": decision.confidence,
                "rationale": decision.rationale,
                "should_advance": decision.should_advance,
                "recovery_subtasks": [asdict(item) for item in decision.recovery_subtasks],
                "timings": decision.timings,
                "raw_response": decision.raw_response,
                "strict_json_parse_ok": "unparseable JSON" not in decision.rationale,
            }
        )
        if args.follow_subtasks and decision.status.value == "complete":
            if active_subtask_index + 1 < len(plan.subtasks):
                active_subtask_index += 1

    result: dict[str, Any] = {
        "model": args.model,
        "plan_path": str(plan_path),
        "video_path": str(video_path),
        "task_instruction": plan.task_instruction,
        "initial_subtask_index": args.subtask_index,
        "follow_subtasks": args.follow_subtasks,
        "decisions": decisions,
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
