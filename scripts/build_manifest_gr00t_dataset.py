#!/usr/bin/env python3
"""Build a GR00T-compatible lightweight dataset from a subtask manifest.

Low-dimensional rows are copied per virtual subtask episode. Video files are
symlinks to the original source videos; timestamps remain global so GR00T's
timestamp-based reader selects the correct source frames.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    # Read the manifest before removing an existing output directory. The
    # all-task preparer stores manifest.json under the output metadata folder.
    manifest = json.loads(args.manifest.read_text())
    if args.output.exists():
        if not args.overwrite:
            raise SystemExit(f"Output exists: {args.output}; use --overwrite")
        shutil.rmtree(args.output)
    source = Path(manifest["source_root"])
    info = json.loads((source / "meta" / "info.json").read_text())
    features = info["features"]
    cameras = [k for k, v in features.items() if v.get("dtype") == "video"]
    chunks_size = int(info.get("chunks_size", 1000))
    data_pattern = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    video_pattern = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"

    (args.output / "meta").mkdir(parents=True)
    (args.output / "meta" / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "data" / "chunk-000").mkdir(parents=True)
    (args.output / "videos").mkdir(parents=True)
    out_info = dict(info)
    out_info["total_episodes"] = len(manifest["segments"])
    out_info["total_frames"] = sum(
        int(s["source_frame_end_exclusive"]) - int(s["source_frame_start"])
        for s in manifest["segments"]
    )
    # All virtual episodes are written into chunk-000. Keep the metadata
    # consistent so GR00T never derives chunk-001, chunk-002, etc.
    out_info["chunks_size"] = max(1, len(manifest["segments"]))
    out_info["data_path"] = data_pattern
    out_info["video_path"] = video_pattern
    (args.output / "meta" / "info.json").write_text(json.dumps(out_info, indent=2) + "\n")

    stats = json.loads((source / "meta" / "stats.json").read_text())
    # LeRobot v3 stats omit q01/q99, while GR00T validates them. Min/max are
    # conservative fallbacks that avoid rescanning thousands of parquet files.
    for value in stats.values():
        if isinstance(value, dict):
            value.setdefault("q01", value.get("min"))
            value.setdefault("q99", value.get("max"))
            dimensions = next(
                (len(value[key]) for key in ("mean", "std", "min", "max", "q01", "q99")
                 if isinstance(value.get(key), list)),
                None,
            )
            if dimensions is not None:
                count = value.get("count", [0])
                if isinstance(count, list):
                    count_value = count[0] if count else 0
                else:
                    count_value = count
                value["count"] = [count_value] * dimensions
    (args.output / "meta" / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    modality = Path("/data/zjw/workspace/robocasa/robocasa/models/assets/groot_dataset_assets/PandaOmron_modality.json")
    shutil.copy2(modality, args.output / "meta" / "modality.json")

    source_data = pd.read_parquet(source / "data" / "chunk-000" / "file-000.parquet")
    tasks = []
    task_to_index = {}
    episodes = []
    for output_episode, segment in enumerate(manifest["segments"]):
        start = int(segment["source_frame_start"])
        end = int(segment["source_frame_end_exclusive"])
        rows = source_data.iloc[start:end].copy()
        rows["episode_index"] = output_episode
        rows["index"] = range(len(rows))
        task = str(segment["task"])
        if task not in task_to_index:
            task_to_index[task] = len(task_to_index)
            tasks.append({"task_index": task_to_index[task], "task": task})
        # GR00T resolves annotation.human.task_description through task_index.
        # Reindex every frame to the virtual subtask instruction.
        rows["task_index"] = task_to_index[task]
        rows.to_parquet(args.output / "data" / "chunk-000" / f"episode_{output_episode:06d}.parquet", index=False)
        episodes.append({"episode_index": output_episode, "tasks": [task_to_index[task]], "length": len(rows)})

        source_episode = int(segment["source_episode"])
        source_chunk = source_episode // chunks_size
        for camera in cameras:
            src_video = source / "videos" / camera / f"chunk-{source_chunk:03d}" / "file-000.mp4"
            dst_video = args.output / "videos" / "chunk-000" / camera / f"episode_{output_episode:06d}.mp4"
            dst_video.parent.mkdir(parents=True, exist_ok=True)
            dst_video.symlink_to(src_video)

    (args.output / "meta" / "episodes.jsonl").write_text("".join(json.dumps(x) + "\n" for x in episodes))
    (args.output / "meta" / "tasks.jsonl").write_text("".join(json.dumps(x) + "\n" for x in tasks))
    print(f"built {len(episodes)} virtual episodes, {out_info['total_frames']} frames")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
