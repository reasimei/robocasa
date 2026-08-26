#!/usr/bin/env python3
"""
Build a manifest over Robocasa atomic success-only datasets for positive-only auxiliary supervision.

The manifest records each task dataset's LeRobot root and episode lengths, so the training script can
construct dense progress labels online without rewriting the original dataset.
Output json中
"episode_index": 12, 表示这是第12个成功演示
"length": 430，表示这个成功演示的长度为430帧
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import tyro


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
class Args:
    atomic_root: str = "/data/zjw/workspace/robocasa/datasets/v1.0/target/atomic"
    output_path: str = "/data/zjw/workspace/Isaac-GR00T/expdata/aux_progress/atomic_positive_manifest.json"


def iter_dataset_dirs(atomic_root: Path) -> Iterable[tuple[str, Path]]:
    for task_dir in sorted(p for p in atomic_root.iterdir() if p.is_dir()):
        dated_dirs = sorted(p for p in task_dir.iterdir() if p.is_dir())
        for dated_dir in dated_dirs:
            lerobot_root = dated_dir / "lerobot"
            if lerobot_root.is_dir():
                yield task_dir.name, dated_dir


def load_episode_records(episode_path: Path) -> list[EpisodeRecord]:
    episodes: list[EpisodeRecord] = []
    with episode_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            episodes.append(EpisodeRecord(episode_index=int(raw["episode_index"]), length=int(raw["length"])))
    return episodes


def main(args: Args) -> None:
    atomic_root = Path(args.atomic_root)
    if not atomic_root.is_dir():
        raise FileNotFoundError(f"Atomic dataset root not found: {atomic_root}")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_records: list[DatasetRecord] = []
    total_episodes = 0
    total_frames = 0

    for task_name, dated_dir in iter_dataset_dirs(atomic_root):
        lerobot_root = dated_dir / "lerobot"
        episode_path = lerobot_root / "meta" / "episodes.jsonl"
        if not episode_path.is_file():
            raise FileNotFoundError(f"Missing episodes file: {episode_path}")

        episodes = load_episode_records(episode_path)
        num_episodes = len(episodes)
        num_frames = sum(ep.length for ep in episodes)
        total_episodes += num_episodes
        total_frames += num_frames

        dataset_records.append(
            DatasetRecord(
                task_name=task_name,
                dated_dir=str(dated_dir),
                lerobot_root=str(lerobot_root),
                num_episodes=num_episodes,
                num_frames=num_frames,
                episodes=episodes,
            )
        )

    payload = {
        "atomic_root": str(atomic_root),
        "num_datasets": len(dataset_records),
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "datasets": [
            {
                **asdict(record),
                "episodes": [asdict(ep) for ep in record.episodes],
            }
            for record in dataset_records
        ],
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Wrote manifest with {len(dataset_records)} datasets to {output_path}")
    print(f"Total episodes: {total_episodes}")
    print(f"Total frames: {total_frames}")


if __name__ == "__main__":
    main(tyro.cli(Args))
