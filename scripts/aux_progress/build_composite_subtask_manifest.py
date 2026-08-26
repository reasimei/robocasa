#!/usr/bin/env python3
"""
Build a subtask manifest for composite Robocasa demonstrations.

The chunk-consistency inference script writes one summary per task/episode.  This
script turns those summaries into dense, non-overlapping subtask segments.  The
number of segments is taken from the cached Llama plan:

    num_boundaries = len(plan["subtasks"]) - 1

The summary already contains the top candidate peaks.  We select the highest
scoring candidates, sort the selected frame indices in time, and use them as
boundaries in [0, episode_length].  No data is copied or rewritten.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import tyro


@dataclass
class SubtaskRecord:
    subtask_index: int
    subtask_id: str
    instruction: str
    start_frame: int
    end_frame: int
    length: int
    boundary_before_frame: int | None
    boundary_after_frame: int | None


@dataclass
class EpisodeRecord:
    episode_index: int
    length: int
    subtasks: list[SubtaskRecord]
    selected_peaks: list[dict[str, Any]]
    summary_path: str


@dataclass
class DatasetRecord:
    task_name: str
    dated_dir: str
    lerobot_root: str
    num_episodes: int
    num_frames: int
    num_subtasks: int
    episodes: list[EpisodeRecord]


@dataclass
class Args:
    composite_root: str = "/data/zjw/workspace/robocasa/datasets/v1.0/target/composite"
    plan_cache_dir: str = (
        "/data/zjw/workspace/Isaac-GR00T/expdata/"
        "long_horizon_controller/composite_seen_plan_cache_llama70b"
    )
    consistency_root: str = (
        "/data/zjw/workspace/Isaac-GR00T/expdata/"
        "chunk_consistency/composite_seen_ckpt60000"
    )
    output_path: str = (
        "/data/zjw/workspace/Isaac-GR00T/expdata/aux_progress/"
        "composite_subtask_manifest.json"
    )
    tasks: tuple[str, ...] = ()
    allow_missing_episodes: bool = False
    min_segment_frames: int = 16


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_lerobot_root(composite_root: Path, task_name: str) -> Path:
    candidates = sorted((composite_root / task_name).glob("*/lerobot"))
    if not candidates:
        raise FileNotFoundError(f"No LeRobot dataset found for task {task_name} under {composite_root}")
    return candidates[-1]


def load_episode_records(root: Path) -> list[dict[str, Any]]:
    path = root / "meta" / "episodes.jsonl"
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def plan_subtasks(plan_path: Path) -> list[dict[str, Any]]:
    plan = load_json(plan_path)
    subtasks = plan.get("subtasks")
    if not isinstance(subtasks, list) or not subtasks:
        raise ValueError(f"Plan has no non-empty subtasks list: {plan_path}")
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(subtasks):
        if not isinstance(item, dict):
            raise ValueError(f"Invalid subtask at index {index} in {plan_path}")
        instruction = str(item.get("instruction", "")).strip()
        if not instruction:
            raise ValueError(f"Subtask {index} has no instruction in {plan_path}")
        normalized.append(
            {
                "subtask_index": index,
                "subtask_id": str(item.get("subtask_id", f"subtask_{index + 1}")),
                "instruction": instruction,
            }
        )
    return normalized


def summary_path(consistency_root: Path, task_name: str, episode_index: int) -> Path:
    return (
        consistency_root
        / task_name
        / f"episode_{int(episode_index):06d}"
        / "chunk_consistency_summary.json"
    )


def select_boundaries(
    peaks: Iterable[dict[str, Any]],
    num_boundaries: int,
    episode_length: int,
    min_segment_frames: int,
) -> tuple[list[int], list[dict[str, Any]]]:
    if num_boundaries <= 0:
        return [], []

    candidates: list[dict[str, Any]] = []
    for peak in peaks:
        if "frame" not in peak or "score" not in peak:
            continue
        frame = int(peak["frame"])
        if 0 < frame < episode_length:
            candidates.append(
                {
                    "frame": frame,
                    "sec": float(peak.get("sec", 0.0)),
                    "score": float(peak["score"]),
                }
            )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    if len(candidates) < num_boundaries:
        raise ValueError(
            f"Need {num_boundaries} boundaries for {episode_length=}, "
            f"but summary has only {len(candidates)} usable peaks."
        )

    selected: list[dict[str, Any]] = []
    for candidate in candidates:
        if all(abs(candidate["frame"] - item["frame"]) >= min_segment_frames for item in selected):
            selected.append(candidate)
            if len(selected) == num_boundaries:
                break
    if len(selected) < num_boundaries:
        raise ValueError(
            f"Could not select {num_boundaries} peaks with min_segment_frames="
            f"{min_segment_frames}; candidates={candidates}"
        )

    boundaries = sorted(int(item["frame"]) for item in selected)
    edges = [0, *boundaries, int(episode_length)]
    if any((right - left) < min_segment_frames for left, right in zip(edges, edges[1:])):
        raise ValueError(
            f"Selected boundaries create a segment shorter than {min_segment_frames}: "
            f"edges={edges}"
        )
    selected = sorted(selected, key=lambda item: int(item["frame"]))
    return boundaries, selected


def build_episode_record(
    task_subtasks: list[dict[str, Any]],
    episode: dict[str, Any],
    summary_file: Path,
    min_segment_frames: int,
) -> EpisodeRecord:
    episode_index = int(episode["episode_index"])
    episode_length = int(episode["length"])
    summary = load_json(summary_file)
    if int(summary.get("episode_length_frames", episode_length)) != episode_length:
        raise ValueError(
            f"Episode length mismatch for {summary_file}: "
            f"dataset={episode_length}, summary={summary.get('episode_length_frames')}"
        )

    boundaries, selected_peaks = select_boundaries(
        summary.get("peaks") or [],
        num_boundaries=len(task_subtasks) - 1,
        episode_length=episode_length,
        min_segment_frames=min_segment_frames,
    )
    edges = [0, *boundaries, episode_length]
    subtasks: list[SubtaskRecord] = []
    for index, (subtask, start_frame, end_frame) in enumerate(
        zip(task_subtasks, edges, edges[1:])
    ):
        subtasks.append(
            SubtaskRecord(
                subtask_index=index,
                subtask_id=subtask["subtask_id"],
                instruction=subtask["instruction"],
                start_frame=int(start_frame),
                end_frame=int(end_frame),
                length=int(end_frame - start_frame),
                boundary_before_frame=None if index == 0 else int(start_frame),
                boundary_after_frame=None if index + 1 == len(task_subtasks) else int(end_frame),
            )
        )
    return EpisodeRecord(
        episode_index=episode_index,
        length=episode_length,
        subtasks=subtasks,
        selected_peaks=selected_peaks,
        summary_path=str(summary_file),
    )


def iter_task_names(args: Args, composite_root: Path, plan_cache_dir: Path) -> list[str]:
    if args.tasks:
        return list(args.tasks)
    task_names: list[str] = []
    for path in sorted(composite_root.iterdir()):
        if not path.is_dir() or not (plan_cache_dir / path.name / "plan.json").is_file():
            continue
        if list(path.glob("*/lerobot")):
            task_names.append(path.name)
    return task_names


def main(args: Args) -> None:
    composite_root = Path(args.composite_root)
    plan_cache_dir = Path(args.plan_cache_dir)
    consistency_root = Path(args.consistency_root)
    if not composite_root.is_dir():
        raise FileNotFoundError(f"Composite root not found: {composite_root}")
    if not plan_cache_dir.is_dir():
        raise FileNotFoundError(f"Plan cache not found: {plan_cache_dir}")
    if args.min_segment_frames < 1:
        raise ValueError("--min-segment-frames must be positive")

    dataset_records: list[DatasetRecord] = []
    missing: list[str] = []
    for task_name in iter_task_names(args, composite_root, plan_cache_dir):
        plan_path = plan_cache_dir / task_name / "plan.json"
        if not plan_path.is_file():
            raise FileNotFoundError(f"Missing plan for task {task_name}: {plan_path}")
        root = find_lerobot_root(composite_root, task_name)
        task_subtasks = plan_subtasks(plan_path)
        episodes: list[EpisodeRecord] = []
        for episode in load_episode_records(root):
            episode_index = int(episode["episode_index"])
            current_summary = summary_path(consistency_root, task_name, episode_index)
            if not current_summary.is_file():
                missing.append(f"{task_name}/episode_{episode_index:06d}")
                continue
            episodes.append(
                build_episode_record(
                    task_subtasks=task_subtasks,
                    episode=episode,
                    summary_file=current_summary,
                    min_segment_frames=int(args.min_segment_frames),
                )
            )

        if not episodes:
            continue
        dataset_records.append(
            DatasetRecord(
                task_name=task_name,
                dated_dir=str(root.parent),
                lerobot_root=str(root),
                num_episodes=len(episodes),
                num_frames=sum(item.length for item in episodes),
                num_subtasks=len(task_subtasks),
                episodes=episodes,
            )
        )

    if missing and not args.allow_missing_episodes:
        preview = ", ".join(missing[:8])
        suffix = " ..." if len(missing) > 8 else ""
        raise FileNotFoundError(
            f"Missing {len(missing)} consistency summaries, for example: {preview}{suffix}. "
            "Run chunk-consistency inference for all downloaded episodes or pass "
            "--allow-missing-episodes for a partial manifest."
        )
    if not dataset_records:
        raise RuntimeError("No composite datasets with usable consistency summaries were found.")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_segments = sum(
        len(episode.subtasks)
        for dataset in dataset_records
        for episode in dataset.episodes
    )
    payload = {
        "format_version": 1,
        "source": "composite_chunk_consistency",
        "composite_root": str(composite_root),
        "plan_cache_dir": str(plan_cache_dir),
        "consistency_root": str(consistency_root),
        "allow_missing_episodes": bool(args.allow_missing_episodes),
        "missing_episodes": missing,
        "num_datasets": len(dataset_records),
        "num_episodes": sum(item.num_episodes for item in dataset_records),
        "num_frames": sum(item.num_frames for item in dataset_records),
        "num_subtask_segments": total_segments,
        "datasets": [
            {
                **asdict(dataset),
                "episodes": [
                    {
                        **asdict(episode),
                        "subtasks": [asdict(subtask) for subtask in episode.subtasks],
                    }
                    for episode in dataset.episodes
                ],
            }
            for dataset in dataset_records
        ],
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Wrote composite manifest to {output_path}")
    print(
        f"Datasets={payload['num_datasets']} episodes={payload['num_episodes']} "
        f"frames={payload['num_frames']} subtask_segments={payload['num_subtask_segments']}"
    )
    if missing:
        print(f"Skipped missing summaries: {len(missing)}")


if __name__ == "__main__":
    main(tyro.cli(Args))
