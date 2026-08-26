"""Load RoboCasa per-frame GT stage text for stage-AdaLN evaluation."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_TARGET_MANIFEST = (
    "/data/zjw/workspace/Isaac-GR00T/expdata/long_horizon_stage_adaln/"
    "target_composite_manifest.json"
)


@dataclass(frozen=True)
class GTStageSpec:
    index: int
    subtask_id: str
    instruction: str
    atomic_skill: str
    stage: str
    source: str

    def prompt(self) -> str:
        return (
            f"Atomic skill: {self.atomic_skill}. Stage: {self.stage}. "
            f"Current subtask: {self.instruction}"
        )

    def condition_text(self, condition_format: str = "full") -> str:
        if condition_format == "subtask_only":
            return self.instruction
        if condition_format == "full":
            return self.prompt()
        raise ValueError(
            f"Unknown stage condition format {condition_format!r}; "
            "expected 'full' or 'subtask_only'."
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# DeliverStraw's 20250813 target dataset has no per-frame GT fields.  Keep a
# transparent fallback so it can still be evaluated with a simulator oracle.
DELIVER_STRAW_FALLBACK = (
    GTStageSpec(
        0,
        "open_drawer",
        "Open the drawer in front.",
        "OpenDrawer",
        "execute",
        "manual_fallback_no_gt_annotations",
    ),
    GTStageSpec(
        1,
        "pick_straw_from_drawer",
        "Pick up a straw from the drawer.",
        "PickPlaceDrawerToCounter",
        "pick",
        "manual_fallback_no_gt_annotations",
    ),
    GTStageSpec(
        2,
        "move_to_dining_counter",
        "Move to the dining counter with the glass cup.",
        "NavigateKitchen",
        "navigate",
        "manual_fallback_no_gt_annotations",
    ),
    GTStageSpec(
        3,
        "place_straw_in_glass_cup",
        "Place the straw inside the glass cup on the dining counter.",
        "PickPlaceDrawerToCounter",
        "place",
        "manual_fallback_no_gt_annotations",
    ),
)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "stage"


def _load_task_table(path: Path) -> dict[int, str]:
    return {
        int(item["task_index"]): str(item["task"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for item in [json.loads(line)]
    }


def _read_episode(root: Path, episode_index: int) -> list[dict[str, int]]:
    import pyarrow.parquet as pq

    matches = list(root.glob(f"data/chunk-*/episode_{episode_index:06d}.parquet"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one parquet for episode {episode_index} under {root / 'data'}, "
            f"found {len(matches)}."
        )
    path = matches[0]
    table = pq.read_table(
        path,
        columns=[
            "subtask_idx",
            "annotation.human.subtask",
            "annotation.human.subtask_name",
            "annotation.human.subtask_stage",
        ],
    )
    columns = table.to_pydict()
    rows: list[dict[str, int]] = []
    previous = None
    for index in range(len(columns["subtask_idx"])):
        subtask_idx = int(columns["subtask_idx"][index])
        if subtask_idx == previous:
            continue
        previous = subtask_idx
        rows.append(
            {
                "subtask_idx": subtask_idx,
                "subtask": int(columns["annotation.human.subtask"][index]),
                "subtask_name": int(columns["annotation.human.subtask_name"][index]),
                "subtask_stage": int(columns["annotation.human.subtask_stage"][index]),
            }
        )
    return rows


def _matching_episode_index(
    record: dict[str, Any],
    task_instruction: str | None,
) -> int:
    if task_instruction:
        normalized = task_instruction.strip()
        for episode in record.get("episodes", []):
            descriptions = [str(item).strip() for item in episode.get("tasks", [])]
            if normalized in descriptions:
                return int(episode["episode_index"])
    return int(record["episodes"][0]["episode_index"])


def load_gt_stage_catalog(
    task: str,
    manifest_path: str | Path = DEFAULT_TARGET_MANIFEST,
    task_instruction: str | None = None,
) -> tuple[list[GTStageSpec], str]:
    """Return canonical GT stages and their source name.

    The target manifest records the exact RoboCasa task-table strings used by
    adapter training.  The final ``task complete`` marker is not an action
    stage and is intentionally omitted.
    """

    if task == "DeliverStraw":
        return list(DELIVER_STRAW_FALLBACK), "manual_fallback_no_gt_annotations"

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    record = next(
        (item for item in manifest.get("datasets", []) if item["task_name"] == task),
        None,
    )
    if record is None:
        raise FileNotFoundError(
            f"No GT dataset record for {task} in {manifest_path}. "
            "DeliverStraw is the only known fallback."
        )

    root = Path(record["dataset_root"])
    episode_index = _matching_episode_index(record, task_instruction)
    task_table = _load_task_table(root / "meta" / "tasks.jsonl")
    stages: list[GTStageSpec] = []
    for position, row in enumerate(_read_episode(root, episode_index)):
        instruction = task_table[row["subtask"]]
        atomic_skill = task_table[row["subtask_name"]]
        stage = task_table[row["subtask_stage"]]
        if stage == "done" or instruction.strip().lower() == "task complete":
            continue
        stages.append(
            GTStageSpec(
                index=len(stages),
                subtask_id=f"gt_{position:02d}_{_slug(instruction)[:48]}",
                instruction=instruction,
                atomic_skill=atomic_skill,
                stage=stage,
                source=f"robocasa_gt:{root}:episode_{episode_index:06d}",
            )
        )
    if not stages:
        raise ValueError(f"No action stages found for {task} in {root}")
    return stages, f"robocasa_gt:episode_{episode_index:06d}"
