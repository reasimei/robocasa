#!/usr/bin/env python3
"""Analyze stage-level effects from paired Oracle rollout result.json files.

The preferred comparison is scale=0.00 versus a conditioned scale using the
same evaluator, task, episode index, and seed.  The report uses simulator
stage predicates recorded in ``stage_trace``:

* reached: the policy was active on the stage
* completed: the stage's latched predicate became true
* completion_step: first step where the stage became latched
* failure_stage: active stage when the episode ended unsuccessfully

This is rollout-level attribution.  It cannot prove action-space causality
without per-chunk baseline/conditioned action predictions; that optional
diagnostic is reported separately when absent.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-root",
        required=True,
        help="Usually scale_0p00, the exact baseline path.",
    )
    parser.add_argument(
        "--conditioned-root",
        required=True,
        help="Conditioned adapter rollout root, or a root containing part roots.",
    )
    parser.add_argument("--reference-label", default="scale_0.00")
    parser.add_argument("--conditioned-label", default="conditioned")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def result_files(root: Path) -> list[Path]:
    return sorted(root.glob("evals/target/*/episodes/episode_*/result.json"))


def load_results(root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for path in result_files(root):
        row = json.loads(path.read_text(encoding="utf-8"))
        key = (str(row["task_name"]), int(row["episode_index"]))
        rows[key] = row
    return rows


def load_results_from_roots(roots: list[Path]) -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for root in roots:
        rows.update(load_results(root))
    return rows


def stage_stats(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    stages = row.get("stages", [])
    trace = row.get("stage_trace", [])
    output: dict[str, dict[str, Any]] = {}
    for stage in stages:
        stage_id = str(stage.get("subtask_id", ""))
        output[stage_id] = {
            "index": int(stage.get("index", len(output))),
            "instruction": str(stage.get("instruction", "")),
            "atomic_skill": str(stage.get("atomic_skill", "")),
            "stage": str(stage.get("stage", "")),
            "reached": False,
            "completed": False,
            "completion_step": None,
            "last_active_step": None,
        }

    for item in trace:
        stage_id = str(item.get("subtask_id", ""))
        if stage_id not in output:
            continue
        stats = output[stage_id]
        stats["reached"] = True
        stats["last_active_step"] = int(item.get("step", 0))
        latched = item.get("latched_labels", {})
        if bool(latched.get(stage_id, False)) and not stats["completed"]:
            stats["completed"] = True
            stats["completion_step"] = int(item.get("step", 0))
    return output


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def pct(values: list[bool]) -> float | None:
    return 100.0 * sum(bool(value) for value in values) / len(values) if values else None


def bootstrap_low(values: list[float], rounds: int = 2000, seed: int = 17) -> float | None:
    if len(values) < 2:
        return None
    import numpy as np

    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=np.float64)
    samples = np.asarray(
        [rng.choice(array, size=len(array), replace=True).mean() for _ in range(rounds)]
    )
    return float(np.quantile(samples, 0.05))


def compare_stage(
    ref_rows: dict[tuple[str, int], dict[str, Any]],
    cond_rows: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    common = sorted(set(ref_rows) & set(cond_rows))
    records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key in common:
        ref = stage_stats(ref_rows[key])
        cond = stage_stats(cond_rows[key])
        for stage_id in sorted(set(ref) & set(cond), key=lambda item: ref[item]["index"]):
            r = ref[stage_id]
            c = cond[stage_id]
            records[stage_id].append(
                {
                    "task_name": key[0],
                    "episode_index": key[1],
                    "instruction": r["instruction"],
                    "atomic_skill": r["atomic_skill"],
                    "stage": r["stage"],
                    "reference_reached": r["reached"],
                    "conditioned_reached": c["reached"],
                    "reference_completed": r["completed"],
                    "conditioned_completed": c["completed"],
                    "reference_completion_step": r["completion_step"],
                    "conditioned_completion_step": c["completion_step"],
                    "completion_delta": (
                        c["completion_step"] - r["completion_step"]
                        if r["completion_step"] is not None
                        and c["completion_step"] is not None
                        else None
                    ),
                }
            )

    report: dict[str, Any] = {
        "num_common_episodes": len(common),
        "stage_count": len(records),
        "stages": {},
        "paired_episode_outcomes": [],
    }
    for key in common:
        ref_success = bool(ref_rows[key].get("env_success", False))
        cond_success = bool(cond_rows[key].get("env_success", False))
        report["paired_episode_outcomes"].append(
            {
                "task_name": key[0],
                "episode_index": key[1],
                "seed_reference": ref_rows[key].get("seed"),
                "seed_conditioned": cond_rows[key].get("seed"),
                "reference_success": ref_success,
                "conditioned_success": cond_success,
                "outcome": (
                    "both_success"
                    if ref_success and cond_success
                    else "conditioned_only"
                    if cond_success and not ref_success
                    else "reference_only"
                    if ref_success and not cond_success
                    else "both_fail"
                ),
            }
        )

    for stage_id, rows in records.items():
        ref_completed = [bool(row["reference_completed"]) for row in rows]
        cond_completed = [bool(row["conditioned_completed"]) for row in rows]
        deltas = [
            float(row["completion_delta"])
            for row in rows
            if row["completion_delta"] is not None
        ]
        improved = [
            bool(row["conditioned_completed"]) and not bool(row["reference_completed"])
            for row in rows
        ]
        damaged = [
            bool(row["reference_completed"]) and not bool(row["conditioned_completed"])
            for row in rows
        ]
        first = rows[0]
        report["stages"][stage_id] = {
            "instruction": first["instruction"],
            "atomic_skill": first["atomic_skill"],
            "stage": first["stage"],
            "num_pairs": len(rows),
            "reference_completed_rate": pct(ref_completed),
            "conditioned_completed_rate": pct(cond_completed),
            "completion_rate_delta_pp": (
                pct(cond_completed) - pct(ref_completed)
                if pct(cond_completed) is not None and pct(ref_completed) is not None
                else None
            ),
            "conditioned_only_pairs": sum(improved),
            "reference_only_pairs": sum(damaged),
            "mean_completion_step_delta": mean(deltas),
            "completion_step_delta_p05": bootstrap_low(
                deltas,
                seed=17 + int(first["stage"] == "navigate"),
            ),
            "interpretation": (
                "helpful"
                if sum(improved) > sum(damaged)
                else "harmful"
                if sum(damaged) > sum(improved)
                else "neutral"
            ),
            "pairs": rows,
        }

    outcomes = report["paired_episode_outcomes"]
    report["overall"] = {
        "reference_successes": sum(row["reference_success"] for row in outcomes),
        "conditioned_successes": sum(row["conditioned_success"] for row in outcomes),
        "conditioned_only": sum(row["outcome"] == "conditioned_only" for row in outcomes),
        "reference_only": sum(row["outcome"] == "reference_only" for row in outcomes),
        "both_success": sum(row["outcome"] == "both_success" for row in outcomes),
        "both_fail": sum(row["outcome"] == "both_fail" for row in outcomes),
    }
    return report


def main() -> None:
    args = parse_args()
    reference_root = Path(args.reference_root)
    conditioned_root = Path(args.conditioned_root)
    ref_rows = load_results(reference_root)
    conditioned_parts = sorted(conditioned_root.parent.glob(conditioned_root.name + "_part*"))
    if conditioned_parts:
        cond_rows = load_results_from_roots(conditioned_parts)
        conditioned_roots = [str(path) for path in conditioned_parts]
    else:
        cond_rows = load_results(conditioned_root)
        conditioned_roots = [str(conditioned_root)]
    report = compare_stage(ref_rows, cond_rows)
    report["reference_root"] = str(reference_root)
    report["conditioned_roots"] = conditioned_roots
    report["reference_label"] = args.reference_label
    report["conditioned_label"] = args.conditioned_label
    report["action_level_diagnostics_available"] = False
    report["action_level_diagnostics_note"] = (
        "These result.json files contain stage predicates and outcomes, but do "
        "not contain paired baseline/conditioned action predictions. The report "
        "therefore attributes effects at stage completion/failure level only."
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        json.dumps(
            {
                "common_episodes": report["num_common_episodes"],
                "overall": report["overall"],
                "output": str(output),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    for stage_id, row in report["stages"].items():
        print(
            f"{row['stage']:12s} {row['atomic_skill']:32s} "
            f"ref={row['reference_completed_rate']:.1f}% "
            f"cond={row['conditioned_completed_rate']:.1f}% "
            f"delta={row['completion_rate_delta_pp']:+.1f}pp "
            f"cond_only={row['conditioned_only_pairs']} "
            f"ref_only={row['reference_only_pairs']} "
            f"{row['interpretation']}"
        )


if __name__ == "__main__":
    main()
