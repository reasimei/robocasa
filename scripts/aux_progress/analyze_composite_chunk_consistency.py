#!/usr/bin/env python3
"""
Inspect whether expert composite actions reveal subtask handoff points.

Important distinction:
For expert actions, the strict GR00T chunk-consistency overlap metric is degenerate:
chunk_t[1:] and chunk_{t+1}[:-1] refer to the same ground-truth future actions, so the
distance is exactly zero. This script also computes an expert-only proxy based on action
window drift to propose video timestamps worth inspecting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tyro


@dataclass
class Args:
    composite_root: str = "/data/zjw/workspace/robocasa/datasets/v1.0/target/composite"
    tasks: tuple[str, ...] = (
        "ArrangeTea",
        "DeliverStraw",
        "GatherTableware",
        "CategorizeCondiments",
        "GarnishPancake",
    )
    episodes: tuple[int, ...] = (0,)
    horizon: int = 50
    top_k: int = 5
    min_separation_sec: float = 3.0
    smooth_window: int = 21
    ignore_edge_sec: float = 2.0
    video_key: str = "observation.images.robot0_agentview_left"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_lerobot_root(composite_root: Path, task_name: str) -> Path:
    candidates = sorted((composite_root / task_name).glob("*/lerobot"))
    if not candidates:
        raise FileNotFoundError(f"No LeRobot dataset found for task {task_name} under {composite_root}")
    return candidates[-1]


def episode_chunk(episode_index: int, chunk_size: int) -> int:
    return int(episode_index) // int(chunk_size)


def load_episode_dataframe(root: Path, episode_index: int) -> pd.DataFrame:
    info = load_json(root / "meta" / "info.json")
    chunk = episode_chunk(episode_index, int(info["chunks_size"]))
    parquet_path = root / info["data_path"].format(
        episode_chunk=chunk,
        episode_index=episode_index,
    )
    return pd.read_parquet(parquet_path)


def stack_actions(df: pd.DataFrame) -> np.ndarray:
    return np.stack(df["action"].to_numpy()).astype(np.float32)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    window = max(int(window), 1)
    if window <= 1:
        return values
    kernel = np.ones(window, dtype=np.float32) / float(window)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def strict_expert_overlap_consistency(actions: np.ndarray, horizon: int) -> np.ndarray:
    n = actions.shape[0]
    max_t = max(n - horizon, 0)
    values = np.zeros(max_t, dtype=np.float32)
    for t in range(max_t):
        chunk_t = actions[t : t + horizon]
        chunk_next = actions[t + 1 : t + 1 + horizon]
        values[t] = float(np.sqrt(np.mean((chunk_t[1:] - chunk_next[:-1]) ** 2)))
    return values


def expert_window_drift_proxy(actions: np.ndarray, horizon: int, smooth_window: int) -> np.ndarray:
    n = actions.shape[0]
    if n <= horizon:
        return np.zeros(0, dtype=np.float32)
    scale = actions.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    normalized = (actions - actions.mean(axis=0)) / scale

    max_t = n - horizon
    values = np.zeros(max_t, dtype=np.float32)
    for t in range(max_t):
        left = normalized[t : t + horizon - 1]
        right = normalized[t + 1 : t + horizon]
        values[t] = float(np.sqrt(np.mean((right - left) ** 2)))
    return moving_average(values, smooth_window)


def pick_peaks(
    values: np.ndarray,
    fps: float,
    top_k: int,
    min_separation_sec: float,
    ignore_edge_sec: float,
) -> list[int]:
    if values.size == 0:
        return []
    edge = max(int(round(ignore_edge_sec * fps)), 0)
    valid = np.ones(values.shape[0], dtype=bool)
    valid[:edge] = False
    valid[max(values.shape[0] - edge, 0) :] = False
    candidate_indices = np.where(valid)[0]
    order = candidate_indices[np.argsort(values[candidate_indices])[::-1]]
    min_sep = max(int(round(min_separation_sec * fps)), 1)
    peaks: list[int] = []
    for idx in order:
        idx = int(idx)
        if all(abs(idx - prev) >= min_sep for prev in peaks):
            peaks.append(idx)
            if len(peaks) >= top_k:
                break
    return sorted(peaks)


def get_episode_record(root: Path, episode_index: int) -> dict[str, Any]:
    with (root / "meta" / "episodes.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if int(record["episode_index"]) == int(episode_index):
                return record
    raise KeyError(f"Episode {episode_index} not found in {root}")


def video_path(root: Path, episode_index: int, video_key: str) -> Path:
    info = load_json(root / "meta" / "info.json")
    chunk = episode_chunk(episode_index, int(info["chunks_size"]))
    return root / info["video_path"].format(
        episode_chunk=chunk,
        video_key=video_key,
        episode_index=episode_index,
    )


def main(args: Args) -> None:
    composite_root = Path(args.composite_root)
    print(
        "NOTE: strict expert overlap consistency should be zero because overlapping GT chunks are identical.",
        flush=True,
    )
    for task_name in args.tasks:
        root = find_lerobot_root(composite_root, task_name)
        info = load_json(root / "meta" / "info.json")
        fps = float(info["fps"])
        for episode_index in args.episodes:
            record = get_episode_record(root, episode_index)
            df = load_episode_dataframe(root, episode_index)
            actions = stack_actions(df)
            strict = strict_expert_overlap_consistency(actions, args.horizon)
            proxy = expert_window_drift_proxy(actions, args.horizon, args.smooth_window)
            peaks = pick_peaks(proxy, fps, args.top_k, args.min_separation_sec, args.ignore_edge_sec)

            reward_values = np.asarray(df["next.reward"].to_numpy(), dtype=np.float32)
            first_reward = np.where(reward_values > 0.5)[0]
            first_reward_index = int(first_reward[0]) if len(first_reward) else None

            print()
            print(f"TASK {task_name} episode={episode_index}")
            print(f"instruction: {record.get('tasks', [''])[0]}")
            print(f"length_frames={len(df)} fps={fps:g} duration_sec={len(df) / fps:.2f}")
            print(f"video: {video_path(root, episode_index, args.video_key)}")
            print(
                "available_gt_columns="
                f"{[c for c in df.columns if 'subtask' in c.lower() or 'stage' in c.lower() or 'skill' in c.lower()]}"
            )
            print(
                f"strict_overlap_max={float(strict.max()) if strict.size else 0.0:.8f} "
                f"strict_overlap_mean={float(strict.mean()) if strict.size else 0.0:.8f}"
            )
            if first_reward_index is not None:
                print(f"first_reward_frame={first_reward_index} first_reward_sec={first_reward_index / fps:.2f}")
            else:
                print("first_reward_frame=None")
            print("proxy_peak_candidates:")
            for rank, idx in enumerate(peaks, start=1):
                print(
                    f"  #{rank}: frame={idx} sec={idx / fps:.2f} "
                    f"proxy={float(proxy[idx]):.5f}"
                )


if __name__ == "__main__":
    main(tyro.cli(Args))
