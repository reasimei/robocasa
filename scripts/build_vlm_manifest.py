#!/usr/bin/env python3
"""Create lightweight subtask manifests directly from VLM staging annotations."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/data/zjw/workspace/lerobot-annotate/scripts")
from split_lerobot_by_subtasks import parse_staging_segments, strip_label_prefix  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", type=Path, required=True)
    ap.add_argument("--annotation-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    info = json.loads((args.source_root / "meta" / "info.json").read_text())
    episodes = pd.read_parquet(args.source_root / "meta/episodes/chunk-000/file-000.parquet")
    data_path = args.source_root / "data/chunk-000/file-000.parquet"
    data_schema = pd.read_parquet(data_path, engine="pyarrow").columns
    wanted = ["timestamp"]
    for key in (
        "subtask_idx",
        "annotation.human.subtask",
        "annotation.human.subtask_name",
        "annotation.human.subtask_stage",
    ):
        if key in data_schema:
            wanted.append(key)
    data = pd.read_parquet(data_path, columns=wanted)
    task_table = pd.read_parquet(args.source_root / "meta/tasks.parquet")
    task_names = list(task_table.index) if task_table.index.name == "task" else []
    if not task_names and "task" in task_table.columns:
        task_names = task_table["task"].tolist()
    segments = []
    for row in episodes.itertuples():
        ep = int(row.episode_index)
        first, last = int(row.dataset_from_index), int(row.dataset_to_index)
        timestamps = np.asarray(data.iloc[first:last]["timestamp"], dtype=np.float64)
        raw = parse_staging_segments(args.annotation_root / ".annotate_staging", ep)
        if raw:
            starts = [max(0, min(len(timestamps) - 1, int(np.searchsorted(timestamps, float(x["start"]), side="left")))) for x in raw]
            labeled = [(starts[i], starts[i + 1] if i + 1 < len(starts) else len(timestamps), x["label"]) for i, x in enumerate(raw)]
        else:
            # RoboCasa's released target data stores all of these per-frame
            # fields directly in its parquet rows:
            # - annotation.human.subtask: natural-language subtask instruction
            # - annotation.human.subtask_name: atomic-skill class name
            # - annotation.human.subtask_stage: pick/place/navigate/etc.
            #
            # GR00T must train on the first field. The atomic-skill name is
            # retained only as manifest metadata for traceability.
            required = {"subtask_idx", "annotation.human.subtask"}
            if not required.issubset(data.columns):
                continue
            rows = data.iloc[first:last]
            values = rows["subtask_idx"].to_numpy()
            instructions = rows["annotation.human.subtask"].to_numpy()
            atomic_skills = (
                rows["annotation.human.subtask_name"].to_numpy()
                if "annotation.human.subtask_name" in rows.columns
                else None
            )
            stages = (
                rows["annotation.human.subtask_stage"].to_numpy()
                if "annotation.human.subtask_stage" in rows.columns
                else None
            )
            labeled = []
            start = 0
            for i in range(1, len(rows) + 1):
                if i == len(rows) or values[i] != values[start]:
                    instruction_index = int(instructions[start])
                    label = (
                        task_names[instruction_index]
                        if 0 <= instruction_index < len(task_names)
                        else f"subtask_{int(values[start])}"
                    )
                    atomic_skill = None
                    if atomic_skills is not None:
                        atomic_index = int(atomic_skills[start])
                        if 0 <= atomic_index < len(task_names):
                            atomic_skill = task_names[atomic_index]
                    stage = None
                    if stages is not None:
                        stage_index = int(stages[start])
                        if 0 <= stage_index < len(task_names):
                            stage = task_names[stage_index]
                    labeled.append((start, i, label, atomic_skill, stage))
                    start = i
        for item in labeled:
            start, end, label = item[:3]
            if end <= start:
                continue
            segment = {
                "output_episode": len(segments),
                "source_episode": ep,
                "source_frame_start": start,
                "source_frame_end_exclusive": end,
                "start": float(timestamps[start]),
                "end": float(timestamps[end - 1]),
                "task": strip_label_prefix(label),
            }
            if len(item) > 3 and item[3]:
                segment["atomic_skill"] = item[3]
            if len(item) > 4 and item[4]:
                segment["stage"] = item[4]
            segments.append(segment)
    payload = {"source_root": str(args.source_root), "annotation_root": str(args.annotation_root), "segments": segments}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {len(segments)} segments to {args.output}")


if __name__ == "__main__":
    main()
