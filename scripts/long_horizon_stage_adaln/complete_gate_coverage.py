#!/usr/bin/env python3
"""Add every target-composite GT atomic skill to a gate report.

This creates a structurally complete gate without inventing calibration
statistics. Skills absent from the input report are marked ``unseen`` and
the evaluator falls back to their coarse stage gate.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def key(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).lower()


def metadata_skill_names(manifest: dict) -> dict[str, dict[str, object]]:
    """Collect canonical atomic-skill labels without opening parquet files."""
    task_names = {str(item["task_name"]) for item in manifest.get("datasets", [])}
    ignored = {
        "done",
        "task complete",
        "pick",
        "place",
        "navigate",
        "execute",
        "press",
        "wait",
        "tilt",
    }
    result: dict[str, dict[str, object]] = {}
    for record in manifest.get("datasets", []):
        task_file = Path(record["dataset_root"]) / "meta" / "tasks.jsonl"
        if not task_file.is_file():
            continue
        for line in task_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            text = str(json.loads(line).get("task", "")).strip()
            normalized = key(text)
            # Skill labels in RoboCasa metadata are canonical names such as
            # PickPlaceCounterToStove. Exclude task names and natural-language
            # instructions; the latter contain spaces.
            if (
                not text
                or normalized in ignored
                or text in task_names
                or " " in text
                or not text[0].isupper()
            ):
                continue
            result.setdefault(
                normalized,
                {
                    "display_name": text,
                    "stage_kinds": [],
                    "tasks": [],
                },
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = json.loads(Path(args.input).read_text(encoding="utf-8"))
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    skills = report.setdefault("skills", {})
    skill_sources = metadata_skill_names(manifest)

    for skill_key, source in sorted(skill_sources.items()):
        if skill_key in skills:
            skills[skill_key].setdefault("display_name", source["display_name"])
            continue
        skills[skill_key] = {
            "display_name": source["display_name"],
            "stage_kinds": sorted(source["stage_kinds"]),
            "tasks": sorted(source["tasks"]),
            "num_samples": 0,
            "mean_improvement": None,
            "relative_improvement": None,
            "win_rate": None,
            "bootstrap_ci_low": None,
            "recommended_scale": 0.0,
            "enabled": False,
            "status": "unseen",
            "fallback_to_stage": True,
        }

    report["skill_coverage"] = {
        "num_gt_atomic_skills": len(skill_sources),
        "num_reported_skills": len(skills),
        "num_calibrated_skills": sum(
            row.get("status", "calibrated") != "unseen"
            for row in skills.values()
            if isinstance(row, dict)
        ),
        "num_unseen_skills": sum(
            row.get("status") == "unseen"
            for row in skills.values()
            if isinstance(row, dict)
        ),
        "note": (
            "unseen skills have no action-space calibration samples; the "
            "evaluator falls back to the coarse stage gate."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "gt_atomic_skills": len(skill_sources),
                "reported_skills": len(skills),
                "unseen_skills": report["skill_coverage"]["num_unseen_skills"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
