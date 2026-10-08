#!/usr/bin/env python3
"""Materialize labeled VLM cases from the auxiliary-head training manifests.

The positive manifest supplies successful demonstrations. Middle frames are
`in_progress`, terminal frames are `complete`. Retry records are rendered as
temporal regressions/repetitions and labeled `failed` (the controller's VLM
status corresponding to the auxiliary-head `retry` class).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import cv2
from PIL import Image


CAMERA_KEYS = ("robot0_agentview_left", "robot0_eye_in_hand")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--positive-manifest",
        type=Path,
        default=Path("expdata/aux_progress/atomic_positive_manifest.json"),
    )
    parser.add_argument(
        "--retry-manifest",
        type=Path,
        default=Path("expdata/aux_progress/atomic_retry_manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("expdata/long_horizon_controller/vlm_benchmark/aux_cases_10tasks_50.jsonl"),
    )
    parser.add_argument("--num-tasks", type=int, default=10)
    parser.add_argument("--cases-per-task", type=int, default=5)
    parser.add_argument("--complete-per-task", type=int, default=1)
    parser.add_argument("--progress-per-task", type=int, default=2)
    parser.add_argument("--retry-per-task", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--history-stride", type=int, default=10)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_episode_tasks(lerobot_root: Path) -> dict[int, str]:
    path = lerobot_root / "meta" / "episodes.jsonl"
    result: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        tasks = item.get("tasks") or []
        if tasks:
            result[int(item["episode_index"])] = str(tasks[0])
    return result


def video_path(lerobot_root: Path, camera_key: str, episode_index: int) -> Path:
    info = load_json(lerobot_root / "meta" / "info.json")
    chunk_size = int(info.get("chunks_size", 1000))
    chunk = episode_index // chunk_size
    path = (
        lerobot_root
        / "videos"
        / f"chunk-{chunk:03d}"
        / f"observation.images.{camera_key}"
        / f"episode_{episode_index:06d}.mp4"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def read_frame(path: Path, frame_index: int):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_index = max(0, min(int(frame_index), max(total - 1, 0)))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Cannot read frame {frame_index} from {path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()


def save_dual_view(
    root: Path,
    case_id: str,
    lerobot_root: Path,
    episode_index: int,
    historical_step: int,
    current_step: int,
) -> list[str]:
    output_dir = root / case_id.replace("/", "_")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for image_index, (step, camera_key) in enumerate(
        [
            (historical_step, CAMERA_KEYS[0]),
            (historical_step, CAMERA_KEYS[1]),
            (current_step, CAMERA_KEYS[0]),
            (current_step, CAMERA_KEYS[1]),
        ]
    ):
        frame = read_frame(video_path(lerobot_root, camera_key, episode_index), step)
        path = output_dir / f"image_{image_index:02d}.png"
        Image.fromarray(frame).save(path)
        paths.append(str(path))
    return paths


def make_subtask(instruction: str) -> dict[str, Any]:
    return {
        "instruction": instruction,
        "expected_start_state": "The robot and the objects are in the demonstrated initial state.",
        "expected_finish_state": f"The atomic manipulation is visibly complete: {instruction}",
        "max_duration_sec": 30.0,
        "subtask_id": "atomic_task",
    }


def main() -> None:
    args = parse_args()
    if (
        sum((args.complete_per_task, args.progress_per_task, args.retry_per_task))
        != args.cases_per_task
    ):
        raise ValueError("complete/progress/retry counts must sum to --cases-per-task")
    rng = random.Random(args.seed)
    positive = load_json(args.positive_manifest)
    retry = load_json(args.retry_manifest)
    datasets = positive["datasets"]
    by_task: dict[str, dict[str, Any]] = {}
    by_dataset_index: dict[int, dict[str, Any]] = {}
    for dataset_index, record in enumerate(datasets):
        task = str(record["task_name"])
        root = Path(record["lerobot_root"])
        episode_tasks = load_episode_tasks(root)
        dataset_record = {
            "dataset_index": dataset_index,
            "root": root,
            "episodes": record["episodes"],
            "episode_tasks": episode_tasks,
        }
        by_task[task] = dataset_record
        by_dataset_index[dataset_index] = dataset_record
    task_names = sorted(by_task)
    if args.num_tasks > len(task_names):
        raise ValueError(f"Only {len(task_names)} atomic tasks are available")
    selected_tasks = rng.sample(task_names, args.num_tasks)
    retry_by_task: dict[str, list[dict[str, Any]]] = {task: [] for task in selected_tasks}
    for example in retry.get("retry_examples", []):
        if example["source_task_name"] in retry_by_task:
            retry_by_task[example["source_task_name"]].append(example)
    for examples in retry_by_task.values():
        rng.shuffle(examples)

    output_root = args.output.parent / (args.output.stem + "_frames")
    rows: list[dict[str, Any]] = []
    for task in selected_tasks:
        record = by_task[task]
        episodes = list(record["episodes"])
        rng.shuffle(episodes)
        instruction = next(iter(record["episode_tasks"].values()), task)
        chosen = episodes[: args.complete_per_task + args.progress_per_task]
        if len(chosen) < args.complete_per_task + args.progress_per_task:
            raise ValueError(f"Not enough positive episodes for {task}")
        for index, episode in enumerate(chosen):
            episode_index = int(episode["episode_index"])
            length = int(episode["length"])
            if index < args.complete_per_task:
                current = max(length - 1, 0)
                historical = max(0, current - args.history_stride)
                status = "complete"
                bucket = "complete"
            else:
                fraction = 0.45 + 0.2 * (index - args.complete_per_task)
                current = min(max(int(length * fraction), 0), max(length - 1, 0))
                historical = max(0, current - args.history_stride)
                status = "in_progress"
                bucket = "progress"
            case_id = f"aux/{task}/{episode_index:06d}/{bucket}"
            rows.append(
                {
                    "case_id": case_id,
                    "task_instruction": instruction,
                    "current_subtask": make_subtask(instruction),
                    "next_subtask": None,
                    "images": save_dual_view(
                        output_root, case_id, record["root"], episode_index, historical, current
                    ),
                    "expected_status": status,
                    "auxiliary_label": "success" if status == "complete" else "progress",
                    "sampling_bucket": bucket,
                    "source_dataset": str(record["root"]),
                    "source_episode_index": episode_index,
                    "source_step_index": current,
                }
            )
        retry_examples = retry_by_task[task]
        if len(retry_examples) < args.retry_per_task:
            raise ValueError(f"Not enough retry examples for {task}")
        for example in retry_examples[: args.retry_per_task]:
            episode_index = int(example["source_episode_index"])
            current = int(example["source_step_index"])
            length = int(example["source_episode_length"])
            retry_type = str(example["retry_type"])
            if retry_type == "repeat":
                historical = current
            elif retry_type in {"reverse", "backtrack"}:
                historical = min(length - 1, current + args.history_stride)
            else:
                historical = max(0, current - args.history_stride)
            example_instruction = instruction
            language_source_task = None
            if retry_type == "mismatch":
                language_dataset_index = example.get("language_source_dataset_index")
                language_episode_index = example.get("language_source_episode_index")
                if language_dataset_index is None or language_episode_index is None:
                    raise ValueError(
                        f"Mismatch example is missing language source: {example['sample_id']}"
                    )
                language_record = by_dataset_index[int(language_dataset_index)]
                language_source_task = language_record["episode_tasks"].get(
                    int(language_episode_index)
                )
                if language_source_task is None:
                    language_source_task = next(
                        iter(language_record["episode_tasks"].values()),
                        str(example.get("source_task_name", "")),
                    )
                example_instruction = str(language_source_task)
            case_id = f"aux/{task}/{episode_index:06d}/retry_{retry_type}_{current:04d}"
            rows.append(
                {
                    "case_id": case_id,
                    "task_instruction": example_instruction,
                    "current_subtask": make_subtask(example_instruction),
                    "next_subtask": None,
                    "images": save_dual_view(
                        output_root,
                        case_id,
                        Path(example["source_lerobot_root"]),
                        episode_index,
                        historical,
                        current,
                    ),
                    "expected_status": "failed",
                    "auxiliary_label": "retry",
                    "retry_type": retry_type,
                    "sampling_bucket": "retry",
                    "source_dataset": example["source_lerobot_root"],
                    "source_episode_index": episode_index,
                    "source_step_index": current,
                    "mismatch_language_instruction": language_source_task,
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    print(f"Wrote {len(rows)} labeled cases to {args.output}")
    print(f"Frames saved under {output_root}")


if __name__ == "__main__":
    main()
