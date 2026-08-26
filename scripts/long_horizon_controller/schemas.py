#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal


class FastTrigger(str, Enum):
    NONE = "none"
    SUSPECT_TRANSITION = "suspect_transition"
    SUSPECT_COMPLETE = "suspect_complete"
    SUSPECT_FAIL = "suspect_fail"
    TIMEOUT = "timeout"


class VLMStatus(str, Enum):
    COMPLETE = "complete"
    IN_PROGRESS = "in_progress"
    FAILED = "failed"


class RecoveryMode(str, Enum):
    NONE = "none"
    INSERT_RECOVERY = "insert_recovery"
    ROLLBACK_RETRY = "rollback_retry"


@dataclass
class SubtaskSpec:
    instruction: str
    expected_start_state: str
    expected_finish_state: str
    max_duration_sec: float
    subtask_id: str = ""
    notes: str = ""


@dataclass
class TaskPlan:
    task_instruction: str
    subtasks: list[SubtaskSpec]
    planner_model: str = ""
    raw_response: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(), encoding="utf-8")


@dataclass
class FastMonitorConfig:
    complete_score_threshold: float = 0.35
    min_steps_before_trigger: int = 8
    ema_alpha: float = 0.25
    cooldown_steps: int = 8
    action_step_delta: int = 1


@dataclass
class FastSignal:
    trigger: FastTrigger
    score: float | None = None
    aux_state: str | None = None
    aux_confidence: float | None = None
    elapsed_sec: float = 0.0
    reason: str = ""


@dataclass
class VLMDecision:
    status: VLMStatus
    confidence: float
    rationale: str
    finish_state_satisfied: bool | None = None
    next_start_plausible: bool | None = None
    confidence_label: str = ""
    failure_type: str = "unknown"
    recovery_mode: RecoveryMode = RecoveryMode.NONE
    rollback_steps: int = 0
    recovery_subtasks: list[SubtaskSpec] = field(default_factory=list)
    should_advance: bool = False
    raw_response: str = ""
    timings: dict[str, float] = field(default_factory=dict)


@dataclass
class ControllerEvent:
    step_index: int
    subtask_index: int
    subtask_id: str
    event_type: Literal[
        "start_subtask",
        "fast_signal",
        "vlm_decision",
        "advance_subtask",
        "insert_recovery",
        "rollback_retry",
        "rollback_unavailable",
        "rollback_step",
        "env_done",
        "finish_plan",
    ]
    payload: dict[str, Any] = field(default_factory=dict)


def subtask_from_dict(data: dict[str, Any], fallback_id: str = "") -> SubtaskSpec:
    return SubtaskSpec(
        instruction=str(data.get("instruction", "")),
        expected_start_state=str(data.get("expected_start_state", "")),
        expected_finish_state=str(data.get("expected_finish_state", "")),
        max_duration_sec=float(data.get("max_duration_sec", data.get("max_time_sec", 30.0))),
        subtask_id=str(data.get("subtask_id", fallback_id)),
        notes=str(data.get("notes", "")),
    )


def plan_from_json_payload(
    task_instruction: str,
    payload: dict[str, Any],
    planner_model: str = "",
    raw_response: str = "",
) -> TaskPlan:
    raw_subtasks = payload.get("subtasks", [])
    if not isinstance(raw_subtasks, list) or not raw_subtasks:
        raise ValueError(f"Planner response must contain a non-empty subtasks list: {payload}")
    subtasks = [
        subtask_from_dict(item, fallback_id=f"subtask_{idx + 1}")
        for idx, item in enumerate(raw_subtasks)
    ]
    for idx, subtask in enumerate(subtasks, start=1):
        if not subtask.subtask_id:
            subtask.subtask_id = f"subtask_{idx}"
    return TaskPlan(
        task_instruction=task_instruction,
        subtasks=subtasks,
        planner_model=planner_model,
        raw_response=raw_response,
    )


def plan_from_dict(data: dict[str, Any]) -> TaskPlan:
    raw_subtasks = data.get("subtasks", [])
    if not isinstance(raw_subtasks, list) or not raw_subtasks:
        raise ValueError(f"Plan data must contain a non-empty subtasks list: {data}")
    subtasks = [
        subtask_from_dict(item, fallback_id=f"subtask_{idx + 1}")
        for idx, item in enumerate(raw_subtasks)
    ]
    for idx, subtask in enumerate(subtasks, start=1):
        if not subtask.subtask_id:
            subtask.subtask_id = f"subtask_{idx}"
    return TaskPlan(
        task_instruction=str(data.get("task_instruction", "")),
        subtasks=subtasks,
        planner_model=str(data.get("planner_model", "")),
        raw_response=str(data.get("raw_response", "")),
    )
