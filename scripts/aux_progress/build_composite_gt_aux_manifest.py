#!/usr/bin/env python3
"""
Build composite auxiliary-training data from Robocasa per-frame GT annotations.

This is intentionally independent from the older chunk-consistency manifest
pipeline.  It reads:

  annotation.human.subtask
  annotation.human.subtask_name
  annotation.human.subtask_stage
  subtask_idx

from each episode parquet and maps the integer annotation ids through
meta/tasks.jsonl.  Consecutive frames with the same subtask_idx become one
ground-truth segment.  The output contains:

1. positive segment records for progress/success supervision;
2. synthetic retry references built inside GT subtask segments.

The source parquet files are never modified or copied.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq
import tyro
from tqdm import tqdm


GT_COLUMNS = (
    "frame_index",
    "annotation.human.task_description",
    "annotation.human.subtask",
    "annotation.human.subtask_name",
    "annotation.human.subtask_stage",
    "subtask_idx",
)


@dataclass
class SubtaskSegment:
    segment_index: int
    subtask_idx: int
    subtask_id: str
    subtask_annotation_id: int
    atomic_skill_annotation_id: int
    stage_annotation_id: int
    atomic_skill: str
    stage: str
    instruction: str
    start_frame: int
    end_frame: int
    length: int


@dataclass
class EpisodeRecord:
    episode_index: int
    length: int
    task_instruction: str
    subtasks: list[SubtaskSegment]


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
class RetryRecord:
    sample_id: str
    retry_type: str
    source_dataset_index: int
    source_task_name: str
    source_episode_index: int
    source_episode_length: int
    source_subtask_segment_index: int
    source_subtask_idx: int
    source_subtask_start_frame: int
    source_subtask_length: int
    source_step_index: int
    synthetic_sequence_index: int
    synthetic_sequence_length: int
    signed_progress_target: float | None = None
    replacement_instruction: str | None = None
    replacement_subtask_id: str | None = None
    note: str = ""


@dataclass
class Args:
    composite_root: str = "/data/zjw/workspace/robocasa/datasets/v1.0/target/composite"
    output_path: str = (
        "/data/zjw/workspace/Isaac-GR00T/expdata/aux_progress/"
        "composite_gt_aux_manifest.json"
    )
    tasks: tuple[str, ...] = ()
    max_episodes_per_task: int = -1
    skip_missing_annotations: bool = True
    skip_invalid_episodes: bool = False
    reverse_step_stride: int = 4
    repeat_anchor_stride: int = 8
    repeat_copies: int = 3
    mismatch_step_stride: int = 4
    backtrack_turn_fraction: float = 0.75
    backtrack_step_stride: int = 4
    max_retry_samples_per_type: int = 20000
    seed: int = 42
    progress: bool = True


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_lerobot_root(composite_root: Path, task_name: str) -> Path:
    candidates = sorted((composite_root / task_name).glob("*/lerobot"))
    if not candidates:
        raise FileNotFoundError(f"No LeRobot dataset found for task {task_name}")
    return candidates[-1]


def load_tasks(root: Path) -> dict[int, str]:
    path = root / "meta" / "tasks.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        return {
            int(record["task_index"]): str(record["task"])
            for record in (json.loads(line) for line in handle if line.strip())
        }


def episode_records(root: Path) -> list[dict[str, Any]]:
    path = root / "meta" / "episodes.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def episode_parquet_path(root: Path, episode_index: int) -> Path:
    info = load_json(root / "meta" / "info.json")
    chunk = int(episode_index) // int(info["chunks_size"])
    return root / info["data_path"].format(
        episode_chunk=chunk,
        episode_index=int(episode_index),
    )


def as_int(value: Any) -> int:
    if value is None:
        raise ValueError("GT annotation contains null")
    return int(value)


def normalize_frame_rows(table: Any) -> list[dict[str, Any]]:
    columns = {name: table[name].to_pylist() for name in table.column_names}
    size = table.num_rows
    return [
        {name: values[index] for name, values in columns.items()}
        for index in range(size)
    ]


def read_gt_episode(
    root: Path,
    task_name: str,
    episode: dict[str, Any],
    task_map: dict[int, str],
) -> EpisodeRecord:
    episode_index = int(episode["episode_index"])
    expected_length = int(episode["length"])
    path = episode_parquet_path(root, episode_index)
    parquet = pq.ParquetFile(path)
    missing = [column for column in GT_COLUMNS if column not in parquet.schema_arrow.names]
    if missing:
        raise ValueError(f"{path} is missing GT columns: {missing}")

    table = parquet.read(columns=list(GT_COLUMNS))
    rows = normalize_frame_rows(table)
    if len(rows) != expected_length:
        raise ValueError(
            f"Length mismatch for {path}: metadata={expected_length}, parquet={len(rows)}"
        )

    frame_indices = [int(row["frame_index"]) for row in rows]
    if frame_indices != list(range(expected_length)):
        raise ValueError(
            f"Unexpected frame_index sequence for {path}: "
            f"first={frame_indices[:5]} last={frame_indices[-5:]}"
        )

    task_instruction = task_map[as_int(rows[0]["annotation.human.task_description"])]
    segments: list[SubtaskSegment] = []
    run_start = 0
    for position in range(1, expected_length + 1):
        previous_idx = as_int(rows[run_start]["subtask_idx"])
        current_idx = (
            as_int(rows[position]["subtask_idx"])
            if position < expected_length
            else None
        )
        if position < expected_length and current_idx == previous_idx:
            continue

        row = rows[run_start]
        subtask_annotation_id = as_int(row["annotation.human.subtask"])
        skill_id = as_int(row["annotation.human.subtask_name"])
        stage_id = as_int(row["annotation.human.subtask_stage"])
        segments.append(
            SubtaskSegment(
                segment_index=len(segments),
                subtask_idx=previous_idx,
                subtask_id=f"{task_name}:annotation_{subtask_annotation_id}",
                subtask_annotation_id=subtask_annotation_id,
                atomic_skill_annotation_id=skill_id,
                stage_annotation_id=stage_id,
                atomic_skill=task_map[skill_id],
                stage=task_map[stage_id],
                instruction=task_map[subtask_annotation_id],
                start_frame=run_start,
                end_frame=position,
                length=position - run_start,
            )
        )
        run_start = position

    if not segments:
        raise ValueError(f"No GT subtask segments found in {path}")
    return EpisodeRecord(
        episode_index=episode_index,
        length=expected_length,
        task_instruction=task_instruction,
        subtasks=segments,
    )


class ReservoirSampler:
    def __init__(self, limit: int, seed: int):
        self.limit = max(int(limit), 0)
        self.rng = random.Random(seed)
        self.seen = 0
        self.items: list[RetryRecord] = []

    def add(self, item: RetryRecord) -> None:
        self.seen += 1
        if self.limit <= 0:
            return
        if len(self.items) < self.limit:
            self.items.append(item)
            return
        slot = self.rng.randint(0, self.seen - 1)
        if slot < self.limit:
            self.items[slot] = item


def positions(length: int, stride: int) -> list[int]:
    stride = max(int(stride), 1)
    output = list(range(0, int(length), stride))
    if not output:
        output = [0]
    if output[-1] != int(length) - 1:
        output.append(int(length) - 1)
    return output


def retry_id(
    retry_type: str,
    dataset_index: int,
    episode_index: int,
    segment_index: int,
    sequence_index: int,
) -> str:
    return f"{retry_type}:{dataset_index}:{episode_index}:{segment_index}:{sequence_index}"


def make_retry_examples(
    datasets: list[DatasetRecord],
    max_samples_per_type: int,
    args: Args,
) -> tuple[list[RetryRecord], dict[str, int]]:
    samplers = {
        "reverse": ReservoirSampler(max_samples_per_type, args.seed + 11),
        "repeat": ReservoirSampler(max_samples_per_type, args.seed + 23),
        "mismatch": ReservoirSampler(max_samples_per_type, args.seed + 37),
        "backtrack": ReservoirSampler(max_samples_per_type, args.seed + 53),
    }
    refs: list[tuple[str, str, str]] = []
    for dataset in datasets:
        for episode in dataset.episodes:
            for segment in episode.subtasks:
                refs.append((dataset.task_name, segment.subtask_id, segment.instruction))

    def add_segment_sample(
        sampler: ReservoirSampler,
        retry_type: str,
        dataset_index: int,
        dataset: DatasetRecord,
        episode: EpisodeRecord,
        segment: SubtaskSegment,
        source_step: int,
        sequence_index: int,
        sequence_length: int,
        signed_progress: float | None,
        replacement: tuple[str, str] | None,
        note: str,
    ) -> None:
        sampler.add(
            RetryRecord(
                sample_id=retry_id(
                    retry_type,
                    dataset_index,
                    episode.episode_index,
                    segment.segment_index,
                    sequence_index,
                ),
                retry_type=retry_type,
                source_dataset_index=dataset_index,
                source_task_name=dataset.task_name,
                source_episode_index=episode.episode_index,
                source_episode_length=episode.length,
                source_subtask_segment_index=segment.segment_index,
                source_subtask_idx=segment.subtask_idx,
                source_subtask_start_frame=segment.start_frame,
                source_subtask_length=segment.length,
                source_step_index=source_step,
                synthetic_sequence_index=sequence_index,
                synthetic_sequence_length=sequence_length,
                signed_progress_target=signed_progress,
                replacement_instruction=replacement[1] if replacement else None,
                replacement_subtask_id=replacement[0] if replacement else None,
                note=note,
            )
        )

    mismatch_counter = 0
    for dataset_index, dataset in enumerate(datasets):
        for episode in dataset.episodes:
            for segment in episode.subtasks:
                segment_positions = positions(segment.length, args.reverse_step_stride)
                reversed_positions = list(reversed(segment_positions))
                total = len(reversed_positions)
                for sequence_index, local_step in enumerate(reversed_positions):
                    signed = 0.0 if total <= 1 else -sequence_index / float(total - 1)
                    add_segment_sample(
                        samplers["reverse"],
                        "reverse",
                        dataset_index,
                        dataset,
                        episode,
                        segment,
                        segment.start_frame + local_step,
                        sequence_index,
                        total,
                        signed,
                        None,
                        "Reversed source positions inside a GT subtask segment.",
                    )

                anchors = positions(segment.length, args.repeat_anchor_stride)
                for anchor in anchors:
                    for sequence_index in range(max(int(args.repeat_copies), 1)):
                        add_segment_sample(
                            samplers["repeat"],
                            "repeat",
                            dataset_index,
                            dataset,
                            episode,
                            segment,
                            segment.start_frame + anchor,
                            sequence_index,
                            max(int(args.repeat_copies), 1),
                            0.0,
                            None,
                            "Repeated one observation to simulate stagnation.",
                        )

                mismatch_positions = positions(segment.length, args.mismatch_step_stride)
                for sequence_index, local_step in enumerate(mismatch_positions):
                    if len(refs) < 2:
                        replacement = None
                    else:
                        ref = refs[(mismatch_counter + 1) % len(refs)]
                        if ref[1] == segment.subtask_id:
                            ref = refs[(mismatch_counter + 2) % len(refs)]
                        replacement = (ref[1], ref[2])
                    mismatch_counter += 1
                    add_segment_sample(
                        samplers["mismatch"],
                        "mismatch",
                        dataset_index,
                        dataset,
                        episode,
                        segment,
                        segment.start_frame + local_step,
                        sequence_index,
                        len(mismatch_positions),
                        0.0,
                        replacement,
                        "Kept the observation but replaced it with another GT subtask instruction.",
                    )

                if segment.length > 1:
                    turn = max(
                        1,
                        min(
                            int(round(args.backtrack_turn_fraction * (segment.length - 1))),
                            segment.length - 1,
                        ),
                    )
                    backtrack_positions = list(
                        range(
                            turn - 1,
                            -1,
                            -max(int(args.backtrack_step_stride), 1),
                        )
                    )
                    if not backtrack_positions or backtrack_positions[-1] != 0:
                        backtrack_positions.append(0)
                    total = len(backtrack_positions)
                    for sequence_index, local_step in enumerate(backtrack_positions):
                        signed = 0.0 if total <= 1 else -sequence_index / float(total - 1)
                        add_segment_sample(
                            samplers["backtrack"],
                            "backtrack",
                            dataset_index,
                            dataset,
                            episode,
                            segment,
                            segment.start_frame + local_step,
                            sequence_index,
                            total,
                            signed,
                            None,
                            f"Backtracked inside the GT segment after synthetic turn={turn}.",
                        )

    examples = [
        item
        for retry_type in ("reverse", "repeat", "mismatch", "backtrack")
        for item in samplers[retry_type].items
    ]
    counts = {retry_type: len(samplers[retry_type].items) for retry_type in samplers}
    seen = {f"{retry_type}_seen": sampler.seen for retry_type, sampler in samplers.items()}
    counts.update(seen)
    return examples, counts


def iter_task_names(args: Args, root: Path) -> list[str]:
    if args.tasks:
        return list(args.tasks)
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def main(args: Args) -> None:
    root = Path(args.composite_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Composite root not found: {root}")

    datasets: list[DatasetRecord] = []
    skipped_tasks: dict[str, str] = {}
    skipped_episodes: dict[str, str] = {}
    task_names = iter_task_names(args, root)
    task_iter = tqdm(task_names, desc="GT tasks", unit="task", disable=not args.progress)
    for task_name in task_iter:
        try:
            lerobot_root = find_lerobot_root(root, task_name)
            task_map = load_tasks(lerobot_root)
            records: list[EpisodeRecord] = []
            episodes = episode_records(lerobot_root)
            if args.max_episodes_per_task > 0:
                episodes = episodes[: args.max_episodes_per_task]
            for episode in tqdm(
                episodes,
                desc=task_name,
                unit="episode",
                leave=False,
                disable=not args.progress,
            ):
                try:
                    records.append(read_gt_episode(lerobot_root, task_name, episode, task_map))
                except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
                    message = str(exc)
                    if "missing GT columns" in message and args.skip_missing_annotations:
                        skipped_tasks[task_name] = message
                        records = []
                        break
                    if args.skip_invalid_episodes:
                        skipped_episodes[f"{task_name}/{episode['episode_index']}"] = message
                        continue
                    raise
            if records:
                datasets.append(
                    DatasetRecord(
                        task_name=task_name,
                        dated_dir=str(lerobot_root.parent),
                        lerobot_root=str(lerobot_root),
                        num_episodes=len(records),
                        num_frames=sum(record.length for record in records),
                        num_subtasks=sum(len(record.subtasks) for record in records),
                        episodes=records,
                    )
                )
        except (FileNotFoundError, KeyError, ValueError) as exc:
            if args.skip_missing_annotations:
                skipped_tasks[task_name] = str(exc)
                continue
            raise

    if not datasets:
        raise RuntimeError("No composite datasets with valid GT subtask annotations were found.")

    retry_examples, retry_counts = make_retry_examples(
        datasets,
        args.max_retry_samples_per_type,
        args,
    )
    payload = {
        "format_version": 1,
        "source": "robocasa_per_frame_subtask_ground_truth",
        "composite_root": str(root),
        "gt_columns": list(GT_COLUMNS),
        "num_datasets": len(datasets),
        "num_episodes": sum(dataset.num_episodes for dataset in datasets),
        "num_frames": sum(dataset.num_frames for dataset in datasets),
        "num_subtasks": sum(dataset.num_subtasks for dataset in datasets),
        "skipped_tasks": skipped_tasks,
        "skipped_episodes": skipped_episodes,
        "retry_counts": retry_counts,
        "retry_examples": [asdict(item) for item in retry_examples],
        "generation_config": {
            "reverse_step_stride": int(args.reverse_step_stride),
            "repeat_anchor_stride": int(args.repeat_anchor_stride),
            "repeat_copies": int(args.repeat_copies),
            "mismatch_step_stride": int(args.mismatch_step_stride),
            "backtrack_turn_fraction": float(args.backtrack_turn_fraction),
            "backtrack_step_stride": int(args.backtrack_step_stride),
            "max_retry_samples_per_type": int(args.max_retry_samples_per_type),
            "seed": int(args.seed),
        },
        "datasets": [
            {
                **asdict(dataset),
                "episodes": [
                    {
                        **asdict(episode),
                        "subtasks": [asdict(segment) for segment in episode.subtasks],
                    }
                    for episode in dataset.episodes
                ],
            }
            for dataset in datasets
        ],
    }
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote GT auxiliary manifest: {output}")
    print(
        f"datasets={payload['num_datasets']} episodes={payload['num_episodes']} "
        f"frames={payload['num_frames']} subtasks={payload['num_subtasks']} "
        f"retry_examples={len(retry_examples)}"
    )
    if skipped_tasks:
        print(f"Skipped tasks without usable GT annotations: {len(skipped_tasks)}")
    if skipped_episodes:
        print(f"Skipped invalid episodes: {len(skipped_episodes)}")


if __name__ == "__main__":
    main(tyro.cli(Args))
