#!/usr/bin/env python3
"""Create a compact, reached-aware summary from stage rollout analysis JSON.

The input is produced by ``analyze_stage_rollouts.py``.  Completion rates are
reported both over all paired episodes and, more usefully for later stages,
over episodes where the stage was reached by the corresponding policy.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-pairs", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=15)
    return parser.parse_args()


def rate(numerator: int, denominator: int) -> float | None:
    return 100.0 * numerator / denominator if denominator else None


def rounded(value: float | None) -> float | None:
    return round(value, 3) if value is not None else None


def summarize_rows(rows: list[dict[str, Any]], key_fields: tuple[str, ...]) -> dict[str, Any]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(field, "")) for field in key_fields)].append(row)

    output: list[dict[str, Any]] = []
    for key, group in groups.items():
        ref_completed = sum(bool(row["reference_completed"]) for row in group)
        cond_completed = sum(bool(row["conditioned_completed"]) for row in group)
        ref_reached = sum(bool(row["reference_reached"]) for row in group)
        cond_reached = sum(bool(row["conditioned_reached"]) for row in group)
        cond_only = sum(
            bool(row["conditioned_completed"]) and not bool(row["reference_completed"])
            for row in group
        )
        ref_only = sum(
            bool(row["reference_completed"]) and not bool(row["conditioned_completed"])
            for row in group
        )
        completion_deltas = [
            float(row["completion_delta"])
            for row in group
            if row["completion_delta"] is not None
        ]
        ref_reached_completed = sum(
            bool(row["reference_reached"]) and bool(row["reference_completed"])
            for row in group
        )
        cond_reached_completed = sum(
            bool(row["conditioned_reached"]) and bool(row["conditioned_completed"])
            for row in group
        )
        item = dict(zip(key_fields, key))
        item.update(
            {
                "num_pairs": len(group),
                "reference_reached": ref_reached,
                "conditioned_reached": cond_reached,
                "reference_completed": ref_completed,
                "conditioned_completed": cond_completed,
                "reference_completion_rate_all_pp": rounded(
                    rate(ref_completed, len(group))
                ),
                "conditioned_completion_rate_all_pp": rounded(
                    rate(cond_completed, len(group))
                ),
                "completion_rate_delta_all_pp": rounded(
                    rate(cond_completed, len(group))
                    - rate(ref_completed, len(group))
                ),
                "reference_completion_rate_reached_pp": rounded(
                    rate(ref_reached_completed, ref_reached)
                ),
                "conditioned_completion_rate_reached_pp": rounded(
                    rate(cond_reached_completed, cond_reached)
                ),
                "completion_rate_delta_reached_pp": rounded(
                    (
                        rate(cond_reached_completed, cond_reached)
                        - rate(ref_reached_completed, ref_reached)
                    )
                    if cond_reached and ref_reached
                    else None
                ),
                "conditioned_only_pairs": cond_only,
                "reference_only_pairs": ref_only,
                "net_pair_delta": cond_only - ref_only,
                "mean_completion_step_delta": rounded(
                    sum(completion_deltas) / len(completion_deltas)
                    if completion_deltas
                    else None
                ),
            }
        )
        item["interpretation"] = (
            "helpful"
            if item["net_pair_delta"] > 0
            else "harmful"
            if item["net_pair_delta"] < 0
            else "neutral"
        )
        output.append(item)
    return output


def sort_key(row: dict[str, Any]) -> tuple[float, int]:
    delta = row.get("completion_rate_delta_reached_pp")
    if delta is None:
        delta = row.get("completion_rate_delta_all_pp")
    return (float(delta or 0.0), int(row["num_pairs"]))


def main() -> None:
    args = parse_args()
    analysis = json.loads(Path(args.analysis).read_text(encoding="utf-8"))
    raw_stages = analysis.get("stages", {})
    rows: list[dict[str, Any]] = []
    for stage_id, stage in raw_stages.items():
        for pair in stage.get("pairs", []):
            row = dict(pair)
            row["stage_id"] = stage_id
            rows.append(row)

    task_stage_rows = summarize_rows(
        rows,
        ("task_name", "stage", "atomic_skill", "stage_id"),
    )
    task_rows = summarize_rows(rows, ("task_name",))
    stage_type_rows = summarize_rows(rows, ("stage",))
    atomic_skill_rows = summarize_rows(rows, ("atomic_skill",))

    def eligible(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [item for item in items if item["num_pairs"] >= args.min_pairs]

    task_stage_rows = eligible(task_stage_rows)
    task_rows = eligible(task_rows)
    stage_type_rows = eligible(stage_type_rows)
    atomic_skill_rows = eligible(atomic_skill_rows)

    helpful = sorted(
        [row for row in task_stage_rows if row["net_pair_delta"] > 0],
        key=sort_key,
        reverse=True,
    )[: args.top_k]
    harmful = sorted(
        [row for row in task_stage_rows if row["net_pair_delta"] < 0],
        key=sort_key,
    )[: args.top_k]

    report = {
        "source_analysis": str(args.analysis),
        "min_pairs": args.min_pairs,
        "top_k": args.top_k,
        "overall": analysis.get("overall", {}),
        "task_stage": task_stage_rows,
        "by_task": task_rows,
        "by_stage_type": sorted(stage_type_rows, key=sort_key, reverse=True),
        "by_atomic_skill": sorted(atomic_skill_rows, key=sort_key, reverse=True),
        "most_helpful_task_stages": helpful,
        "most_harmful_task_stages": harmful,
        "interpretation_note": (
            "Reached-aware rates use each policy's own reached count. "
            "They are descriptive rollout attribution, not paired action-level "
            "causal estimates."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"wrote {output}")
    print("\nBy stage type:")
    for row in report["by_stage_type"]:
        print(
            f"{row['stage']:10s} n={row['num_pairs']:3d} "
            f"ref={row['reference_completion_rate_reached_pp']}% "
            f"cond={row['conditioned_completion_rate_reached_pp']}% "
            f"delta={row['completion_rate_delta_reached_pp']!s:>7}pp "
            f"cond-only={row['conditioned_only_pairs']:2d} "
            f"ref-only={row['reference_only_pairs']:2d}"
        )
    print("\nMost helpful task stages:")
    for row in helpful:
        print(
            f"{row['task_name']:24s} {row['stage_id']:48s} "
            f"{row['stage']:10s} delta={row['completion_rate_delta_reached_pp']!s}pp "
            f"net={row['net_pair_delta']:+d} n={row['num_pairs']}"
        )
    print("\nMost harmful task stages:")
    for row in harmful:
        print(
            f"{row['task_name']:24s} {row['stage_id']:48s} "
            f"{row['stage']:10s} delta={row['completion_rate_delta_reached_pp']!s}pp "
            f"net={row['net_pair_delta']:+d} n={row['num_pairs']}"
        )


if __name__ == "__main__":
    main()
