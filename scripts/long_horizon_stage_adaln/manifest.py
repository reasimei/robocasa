#!/usr/bin/env python3
"""Build a non-target RoboCasa training manifest.

The manifest deliberately excludes the composite tasks used by the target
benchmark.  It accepts multiple dataset roots because the public RoboCasa
download separates pretrain and target data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TARGET_COMPOSITE_TASKS = (
    "BreadSelection",
    "GetToastedBread",
    "HeatKebabSandwich",
    "KettleBoiling",
    "LoadDishwasher",
    "MakeIceLemonade",
    "PackIdenticalLunches",
    "PanTransfer",
    "PortionHotDogs",
    "PreSoakPan",
    "PrepareCoffee",
    "RecycleBottlesByType",
    "RinseSinkBasin",
    "ScrubCuttingBoard",
    "SearingMeat",
    "SeparateFreezerRack",
    "SetUpCuttingStation",
    "StackBowlsCabinet",
    "SteamInMicrowave",
    "StirVegetables",
    "StoreLeftoversInBowl",
    "WaffleReheat",
    "WashFruitColander",
    "WashLettuce",
    "WeighIngredients",
)

GT_COLUMNS = {
    "annotation.human.subtask",
    "annotation.human.subtask_name",
    "annotation.human.subtask_stage",
    "subtask_idx",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        action="append",
        dest="source_roots",
        default=None,
        help="Dataset root(s) to scan. Repeat this option for multiple roots.",
    )
    parser.add_argument(
        "--output",
        default=(
            "/data/zjw/workspace/Isaac-GR00T/expdata/"
            "long_horizon_stage_adaln/non_target_manifest.json"
        ),
    )
    parser.add_argument(
        "--include-atomic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include atomic datasets if they contain the GT fields.",
    )
    parser.add_argument(
        "--allow-target-tasks",
        action="store_true",
        help=(
            "Allow the listed target composite tasks into the training manifest. "
            "Use only for supervised target-task adaptation experiments."
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_lerobot_roots(root: Path) -> list[Path]:
    return sorted(path for path in root.glob("*/*/lerobot") if path.is_dir())


def has_gt_annotations(root: Path) -> bool:
    info = load_json(root / "meta" / "info.json")
    features = info.get("features", {})
    return GT_COLUMNS.issubset(features)


def episodes(root: Path) -> list[dict[str, Any]]:
    path = root / "meta" / "episodes.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    skipped: dict[str, str] = {}
    target_names = set(TARGET_COMPOSITE_TASKS)
    source_roots = args.source_roots or [
        "/data/zjw/workspace/robocasa/datasets/v1.0/pretrain/composite",
        "/data/zjw/workspace/robocasa/datasets/v1.0/pretrain/atomic",
    ]

    for source_root_name in source_roots:
        source_root = Path(source_root_name)
        if not source_root.is_dir():
            skipped[str(source_root)] = "source root does not exist"
            continue
        for lerobot_root in find_lerobot_roots(source_root):
            task_name = lerobot_root.parent.parent.name
            is_atomic = source_root.name == "atomic"
            if task_name in target_names and not args.allow_target_tasks:
                skipped[task_name] = "excluded target composite task"
                continue
            if is_atomic and not args.include_atomic:
                skipped[task_name] = "atomic dataset excluded by default"
                continue
            try:
                if not has_gt_annotations(lerobot_root):
                    skipped[task_name] = "missing per-frame GT subtask columns"
                    continue
                records.append(
                    {
                        "task_name": task_name,
                        "dataset_root": str(lerobot_root),
                        "dataset_split": source_root.parent.name,
                        "dataset_kind": source_root.name,
                        "episodes": episodes(lerobot_root),
                    }
                )
            except (OSError, KeyError, json.JSONDecodeError) as exc:
                skipped[str(lerobot_root)] = f"{type(exc).__name__}: {exc}"

    if not records:
        raise RuntimeError(
            "No non-target datasets with GT annotations were found. "
            "Download the RoboCasa pretrain composite datasets and rerun this "
            "script. The target composite datasets were intentionally excluded."
        )

    return {
        "format_version": 1,
        "purpose": "xiaomi_stage_adaln_oracle_training",
        "excluded_target_composite_tasks": list(TARGET_COMPOSITE_TASKS),
        "allow_target_tasks": bool(args.allow_target_tasks),
        "datasets": records,
        "skipped": skipped,
        "num_datasets": len(records),
        "num_episodes": sum(len(record["episodes"]) for record in records),
    }


def main() -> None:
    args = parse_args()
    payload = build_manifest(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "num_datasets": payload["num_datasets"],
        "num_episodes": payload["num_episodes"],
        "excluded_tasks": len(payload["excluded_target_composite_tasks"]),
    }, indent=2))


if __name__ == "__main__":
    main()
