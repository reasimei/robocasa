#!/usr/bin/env python3
"""Prepare lightweight GR00T datasets for all fully annotated RoboCasa tasks."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, default=Path("/data/zjw/workspace/robocasa/datasets/v3.0"))
    ap.add_argument("--annotation-root", type=Path, default=Path("/data/zjw/workspace/robocasa/datasets/v3.0_vlm_annotations"))
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--source-split", nargs="+", default=["pretrain", "target"])
    ap.add_argument("--rebuild", action="store_true", help="Rebuild existing lightweight datasets")
    args = ap.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    manifest_builder = here / "build_vlm_manifest.py"
    dataset_builder = here / "build_manifest_gr00t_dataset.py"
    prepared = 0
    skipped = 0
    for split in args.source_split:
        for source in sorted((args.dataset_root / split / "composite").glob("*/*/lerobot")):
            rel = source.relative_to(args.dataset_root)
            annotation = args.annotation_root / rel
            info = json.loads((source / "meta/info.json").read_text())
            total = int(info["total_episodes"])
            features = info.get("features", {})
            native = all(k in features for k in ("subtask_idx", "annotation.human.subtask_name"))
            staging = annotation / ".annotate_staging"
            done = len(list(staging.glob("episode_*/plan.jsonl"))) if staging.exists() else 0
            if not native and done < total:
                skipped += 1
                print(f"[skip] incomplete annotations {rel}: staging={done}/{total}", flush=True)
                continue
            out = args.output_root / rel
            manifest = out / "meta" / "manifest.json"
            if args.rebuild or not manifest.exists():
                manifest.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run([sys.executable, str(manifest_builder), "--source-root", str(source), "--annotation-root", str(annotation), "--output", str(manifest)], check=True)
            if args.rebuild or not (out / "meta" / "episodes.jsonl").exists():
                # A prior interrupted run may have left only manifest.json or
                # partial files. Rebuild that task atomically with overwrite.
                subprocess.run([sys.executable, str(dataset_builder), "--manifest", str(manifest), "--output", str(out), "--overwrite"], check=True)
            prepared += 1
            print(f"[ready] {out}", flush=True)
    print(f"prepared={prepared} skipped_incomplete={skipped}")


if __name__ == "__main__":
    main()
