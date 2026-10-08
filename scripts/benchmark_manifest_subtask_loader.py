#!/usr/bin/env python3
"""Benchmark the experimental manifest-backed loader without changing GR00T's loader."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from gr00t.data.subtask_manifest_dataset import ManifestSubtaskDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--samples", type=int, default=32)
    args = parser.parse_args()

    ds = ManifestSubtaskDataset(args.manifest, args.camera)
    count = min(args.samples, len(ds))
    t0 = time.perf_counter()
    for i in range(count):
        ds.get_state_action(i)
    parquet_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for i in range(count):
        ds.read_video_frames(i, [0])
    video_s = time.perf_counter() - t0
    ds.close()
    print(f"virtual_episodes={len(ds)} samples={count}")
    print(f"parquet_only_seconds={parquet_s:.3f} samples_per_second={count / parquet_s:.2f}")
    print(f"video_random_seconds={video_s:.3f} samples_per_second={count / video_s:.2f}")
    print(f"video_overhead_ratio={video_s / parquet_s:.2f}x")


if __name__ == "__main__":
    main()
