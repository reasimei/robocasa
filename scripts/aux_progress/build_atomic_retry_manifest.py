#!/usr/bin/env python3
"""
Build a synthetic retry manifest from the positive Robocasa atomic manifest.

This script does not copy image/video data. It emits lightweight sample references that
describe how to synthesize retry examples for four failure modes:

1. reverse
2. repeat
3. mismatch
4. backtrack

Each retry sample points back to one source positive episode step, plus optional language
replacement references for mismatch examples. A later dataset wrapper can use this manifest
to materialize the actual retry training samples on the fly.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import tyro


RETRY_LABEL = "retry"
RETRY_LABEL_ID = 2
STATE_LABEL_MAP = {"progress": 0, "success": 1, RETRY_LABEL: RETRY_LABEL_ID}


@dataclass
class EpisodeRecord:
    episode_index: int
    length: int


@dataclass
class DatasetRecord:
    task_name: str
    dated_dir: str
    lerobot_root: str
    num_episodes: int
    num_frames: int
    episodes: list[EpisodeRecord]


@dataclass
class RetrySampleRecord:
    sample_id: str
    retry_type: str
    label: str
    label_id: int
    source_dataset_index: int
    source_task_name: str
    source_lerobot_root: str
    source_episode_index: int
    source_episode_length: int
    source_step_index: int
    synthetic_sequence_index: int
    synthetic_sequence_length: int
    signed_progress_target: float | None = None
    language_source_dataset_index: int | None = None
    language_source_episode_index: int | None = None
    language_source_step_index: int | None = None
    note: str | None = None


@dataclass
class Args:
    positive_manifest_path: str = "/data/zjw/workspace/Isaac-GR00T/expdata/aux_progress/atomic_positive_manifest.json"
    output_path: str = "/data/zjw/workspace/Isaac-GR00T/expdata/aux_progress/atomic_retry_manifest.json"
    reverse_step_stride: int = 1
    repeat_anchor_stride: int = 8
    repeat_copies: int = 3
    mismatch_step_stride: int = 4
    backtrack_turn_fraction: float = 0.75
    backtrack_step_stride: int = 1
    max_retry_samples_per_type: int = 20000
    seed: int = 42


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_dataset_records(manifest: dict[str, Any]) -> Iterable[tuple[int, dict[str, Any]]]:
    for dataset_index, record in enumerate(manifest["datasets"]):
        yield dataset_index, record


def build_language_refs(manifest: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    refs_by_task: dict[str, list[dict[str, Any]]] = {}
    for dataset_index, record in iter_dataset_records(manifest):
        task_name = record["task_name"]
        refs_by_task.setdefault(task_name, [])
        for episode in record["episodes"]:
            refs_by_task[task_name].append(
                {
                    "dataset_index": dataset_index,
                    "task_name": task_name,
                    "lerobot_root": record["lerobot_root"],
                    "episode_index": int(episode["episode_index"]),
                    "episode_length": int(episode["length"]),
                    "step_index": 0,
                }
            )
    task_names = sorted(refs_by_task)
    return refs_by_task, task_names


def make_sample_id(
    retry_type: str,
    dataset_index: int,
    episode_index: int,
    source_step_index: int,
    synthetic_sequence_index: int,
) -> str:
    return f"{retry_type}:{dataset_index}:{episode_index}:{source_step_index}:{synthetic_sequence_index}"


def downsample_positions(length: int, stride: int) -> list[int]:
    stride = max(int(stride), 1)
    positions = list(range(0, length, stride))
    if not positions:
        positions = [0]
    if positions[-1] != length - 1:
        positions.append(length - 1)
    return positions


class ReservoirSampler:
    def __init__(self, max_items: int, seed: int):
        self.max_items = max(int(max_items), 0)
        self.rng = random.Random(seed)
        self.items: list[RetrySampleRecord] = []
        self.seen = 0

    def add(self, item: RetrySampleRecord) -> None:
        self.seen += 1
        if self.max_items <= 0:
            return
        if len(self.items) < self.max_items:
            self.items.append(item)
            return
        slot = self.rng.randint(0, self.seen - 1)
        if slot < self.max_items:
            self.items[slot] = item


def choose_language_source(
    ref: dict[str, Any],
    language_refs_by_task: dict[str, list[dict[str, Any]]],
    task_names: list[str],
    start_index: int,
) -> dict[str, Any]:
    if len(task_names) < 2:
        raise ValueError("Cannot build mismatch samples without at least two tasks.")

    source_task = ref["task_name"]
    source_task_pos = task_names.index(source_task)
    candidate_task = task_names[(source_task_pos + 1 + start_index) % len(task_names)]
    if candidate_task == source_task:
        candidate_task = task_names[(source_task_pos + 1) % len(task_names)]
    candidates = language_refs_by_task[candidate_task]
    return candidates[start_index % len(candidates)]


def build_reverse_examples(
    manifest: dict[str, Any],
    reverse_step_stride: int,
    sampler: ReservoirSampler,
) -> None:
    for dataset_index, record in iter_dataset_records(manifest):
        for episode in record["episodes"]:
            episode_index = int(episode["episode_index"])
            episode_length = int(episode["length"])
            positions = downsample_positions(episode_length, reverse_step_stride)
            reversed_positions = list(reversed(positions))
            total = len(reversed_positions)
            for synthetic_sequence_index, source_step_index in enumerate(reversed_positions):
                signed_progress = 0.0 if total <= 1 else -synthetic_sequence_index / float(total - 1)
                sampler.add(
                    RetrySampleRecord(
                        sample_id=make_sample_id(
                            "reverse",
                            dataset_index,
                            episode_index,
                            source_step_index,
                            synthetic_sequence_index,
                        ),
                        retry_type="reverse",
                        label=RETRY_LABEL,
                        label_id=RETRY_LABEL_ID,
                        source_dataset_index=dataset_index,
                        source_task_name=record["task_name"],
                        source_lerobot_root=record["lerobot_root"],
                        source_episode_index=episode_index,
                        source_episode_length=episode_length,
                        source_step_index=source_step_index,
                        synthetic_sequence_index=synthetic_sequence_index,
                        synthetic_sequence_length=total,
                        signed_progress_target=signed_progress,
                        note="Reversed successful trajectory order.",
                    )
                )


def build_repeat_examples(
    manifest: dict[str, Any],
    repeat_anchor_stride: int,
    repeat_copies: int,
    sampler: ReservoirSampler,
) -> None:
    repeat_copies = max(int(repeat_copies), 1)
    for dataset_index, record in iter_dataset_records(manifest):
        for episode in record["episodes"]:
            episode_index = int(episode["episode_index"])
            episode_length = int(episode["length"])
            anchors = downsample_positions(episode_length, repeat_anchor_stride)
            for anchor_step_index in anchors:
                for repeat_index in range(repeat_copies):
                    sampler.add(
                        RetrySampleRecord(
                            sample_id=make_sample_id(
                                "repeat",
                                dataset_index,
                                episode_index,
                                anchor_step_index,
                                repeat_index,
                            ),
                            retry_type="repeat",
                            label=RETRY_LABEL,
                            label_id=RETRY_LABEL_ID,
                            source_dataset_index=dataset_index,
                            source_task_name=record["task_name"],
                            source_lerobot_root=record["lerobot_root"],
                            source_episode_index=episode_index,
                            source_episode_length=episode_length,
                            source_step_index=anchor_step_index,
                            synthetic_sequence_index=repeat_index,
                            synthetic_sequence_length=repeat_copies,
                            signed_progress_target=0.0,
                            note="Repeated the same observation to simulate stagnation.",
                        )
                    )


def build_mismatch_examples(
    manifest: dict[str, Any],
    mismatch_step_stride: int,
    language_refs_by_task: dict[str, list[dict[str, Any]]],
    task_names: list[str],
    sampler: ReservoirSampler,
) -> None:
    global_ref_index = 0
    for dataset_index, record in iter_dataset_records(manifest):
        for episode in record["episodes"]:
            episode_index = int(episode["episode_index"])
            episode_length = int(episode["length"])
            source_positions = downsample_positions(episode_length, mismatch_step_stride)
            for synthetic_sequence_index, source_step_index in enumerate(source_positions):
                ref = {
                    "dataset_index": dataset_index,
                    "task_name": record["task_name"],
                    "lerobot_root": record["lerobot_root"],
                    "episode_index": episode_index,
                    "episode_length": episode_length,
                    "step_index": source_step_index,
                }
                language_source = choose_language_source(
                    ref,
                    language_refs_by_task,
                    task_names,
                    global_ref_index,
                )
                global_ref_index += 17
                sampler.add(
                    RetrySampleRecord(
                        sample_id=make_sample_id(
                            "mismatch",
                            dataset_index,
                            episode_index,
                            source_step_index,
                            synthetic_sequence_index,
                        ),
                        retry_type="mismatch",
                        label=RETRY_LABEL,
                        label_id=RETRY_LABEL_ID,
                        source_dataset_index=dataset_index,
                        source_task_name=record["task_name"],
                        source_lerobot_root=record["lerobot_root"],
                        source_episode_index=episode_index,
                        source_episode_length=episode_length,
                        source_step_index=source_step_index,
                        synthetic_sequence_index=synthetic_sequence_index,
                        synthetic_sequence_length=len(source_positions),
                        language_source_dataset_index=int(language_source["dataset_index"]),
                        language_source_episode_index=int(language_source["episode_index"]),
                        language_source_step_index=int(language_source["step_index"]),
                        note="Paired source observation with a different task instruction.",
                    )
                )


def build_backtrack_examples(
    manifest: dict[str, Any],
    backtrack_turn_fraction: float,
    backtrack_step_stride: int,
    sampler: ReservoirSampler,
) -> None:
    for dataset_index, record in iter_dataset_records(manifest):
        for episode in record["episodes"]:
            episode_index = int(episode["episode_index"])
            episode_length = int(episode["length"])
            if episode_length <= 1:
                continue

            turn_step = int(round(backtrack_turn_fraction * (episode_length - 1)))
            turn_step = max(1, min(turn_step, episode_length - 1))
            return_positions = list(range(turn_step - 1, -1, -max(int(backtrack_step_stride), 1)))
            if not return_positions or return_positions[-1] != 0:
                return_positions.append(0)
            total = len(return_positions)

            for synthetic_sequence_index, source_step_index in enumerate(return_positions):
                signed_progress = 0.0 if total <= 1 else -synthetic_sequence_index / float(total - 1)
                sampler.add(
                    RetrySampleRecord(
                        sample_id=make_sample_id(
                            "backtrack",
                            dataset_index,
                            episode_index,
                            source_step_index,
                            synthetic_sequence_index,
                        ),
                        retry_type="backtrack",
                        label=RETRY_LABEL,
                        label_id=RETRY_LABEL_ID,
                        source_dataset_index=dataset_index,
                        source_task_name=record["task_name"],
                        source_lerobot_root=record["lerobot_root"],
                        source_episode_index=episode_index,
                        source_episode_length=episode_length,
                        source_step_index=source_step_index,
                        synthetic_sequence_index=synthetic_sequence_index,
                        synthetic_sequence_length=total,
                        signed_progress_target=signed_progress,
                        note=f"Backtracked after synthetic turn step {turn_step}.",
                    )
                )


def main(args: Args) -> None:
    positive_manifest_path = Path(args.positive_manifest_path)
    if not positive_manifest_path.is_file():
        raise FileNotFoundError(f"Positive manifest not found: {positive_manifest_path}")

    manifest = load_manifest(positive_manifest_path)
    language_refs_by_task, task_names = build_language_refs(manifest)

    reverse_sampler = ReservoirSampler(args.max_retry_samples_per_type, args.seed + 11)
    repeat_sampler = ReservoirSampler(args.max_retry_samples_per_type, args.seed + 23)
    mismatch_sampler = ReservoirSampler(args.max_retry_samples_per_type, args.seed + 37)
    backtrack_sampler = ReservoirSampler(args.max_retry_samples_per_type, args.seed + 53)

    build_reverse_examples(manifest, args.reverse_step_stride, reverse_sampler)
    build_repeat_examples(manifest, args.repeat_anchor_stride, args.repeat_copies, repeat_sampler)
    build_mismatch_examples(
        manifest,
        args.mismatch_step_stride,
        language_refs_by_task,
        task_names,
        mismatch_sampler,
    )
    build_backtrack_examples(
        manifest, args.backtrack_turn_fraction, args.backtrack_step_stride, backtrack_sampler
    )

    retry_examples = (
        reverse_sampler.items
        + repeat_sampler.items
        + mismatch_sampler.items
        + backtrack_sampler.items
    )

    retry_counts: dict[str, int] = {}
    for example in retry_examples:
        retry_counts[example.retry_type] = retry_counts.get(example.retry_type, 0) + 1

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "source_positive_manifest_path": str(positive_manifest_path),
        "label_map": STATE_LABEL_MAP,
        "num_source_samples": int(manifest.get("total_frames", 0)),
        "num_language_source_tasks": len(task_names),
        "num_language_source_episodes": sum(len(refs) for refs in language_refs_by_task.values()),
        "num_retry_samples": len(retry_examples),
        "retry_counts": retry_counts,
        "retry_seen_counts": {
            "reverse": reverse_sampler.seen,
            "repeat": repeat_sampler.seen,
            "mismatch": mismatch_sampler.seen,
            "backtrack": backtrack_sampler.seen,
        },
        "max_retry_samples_per_type": args.max_retry_samples_per_type,
        "retry_examples": [asdict(example) for example in retry_examples],
        "generation_config": {
            "reverse_step_stride": max(int(args.reverse_step_stride), 1),
            "repeat_anchor_stride": max(int(args.repeat_anchor_stride), 1),
            "repeat_copies": max(int(args.repeat_copies), 1),
            "mismatch_step_stride": max(int(args.mismatch_step_stride), 1),
            "backtrack_turn_fraction": float(args.backtrack_turn_fraction),
            "backtrack_step_stride": max(int(args.backtrack_step_stride), 1),
            "max_retry_samples_per_type": max(int(args.max_retry_samples_per_type), 0),
            "seed": int(args.seed),
        },
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Wrote retry manifest to {output_path}")
    print(f"Source samples: {int(manifest.get('total_frames', 0))}")
    print(f"Language source tasks: {len(task_names)}")
    print(f"Language source episodes: {sum(len(refs) for refs in language_refs_by_task.values())}")
    print(f"Retry samples: {len(retry_examples)}")
    print(f"Retry counts: {retry_counts}")


if __name__ == "__main__":
    main(tyro.cli(Args))
