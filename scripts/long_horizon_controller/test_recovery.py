#!/usr/bin/env python3
"""Focused recovery tests that do not require Robocasa, a GPU, or Ollama."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from .controller import ControllerConfig, LongHorizonController
from .fast_monitor import ActionEntropyMonitor
from .robocasa_adapter import RobocasaVectorEnvAdapter
from .robocasa_adapter import _first_env_value
from .schemas import FastMonitorConfig, RecoveryMode, SubtaskSpec, TaskPlan, VLMDecision, VLMStatus
from .policy_adapters import MockPolicyAdapter
from .vlm_verifier import decision_from_text


class _RollbackEnv:
    supports_action_rollback = True
    vlm_image_labels = ("camera",)

    def __init__(self) -> None:
        self.actions: list[dict[str, np.ndarray]] = []
        self.step_index = 0

    def reset(self) -> dict[str, np.ndarray]:
        self.step_index = 0
        self.actions.clear()
        return {"image": np.zeros((8, 8, 3), dtype=np.uint8)}

    def get_vlm_image(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        return observation["image"]

    def step(self, action: dict[str, np.ndarray]):
        self.actions.append(action)
        self.step_index += 1
        return {"image": np.zeros((8, 8, 3), dtype=np.uint8)}, 0.0, False, {}

    def rollback_action(self, action: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {"mock_action": -np.asarray(action["mock_action"]).copy()}


class _Verifier:
    supports_image_history = False

    def __init__(self) -> None:
        self.calls = 0

    def verify(self, image, current_subtask, next_subtask=None):
        del image, current_subtask, next_subtask
        self.calls += 1
        if self.calls == 1:
            return VLMDecision(
                status=VLMStatus.FAILED,
                confidence=1.0,
                rationale="recent motion should be reversed",
                failure_type="wrong_pose",
                recovery_mode=RecoveryMode.ROLLBACK_RETRY,
                rollback_steps=1,
            )
        return VLMDecision(
            status=VLMStatus.COMPLETE,
            confidence=1.0,
            rationale="finish state visible",
            finish_state_satisfied=True,
            next_start_plausible=True,
            should_advance=True,
        )


def test_recovery_json_modes() -> None:
    rollback = decision_from_text(
        '{"recovery_mode":"rollback_retry","rollback_steps":2,'
        '"rationale":"reverse the recent pose","recovery_subtasks":[]}',
        require_status=False,
    )
    assert rollback.recovery_mode == RecoveryMode.ROLLBACK_RETRY
    assert rollback.rollback_steps == 2
    assert not rollback.recovery_subtasks

    inserted = decision_from_text(
        '{"recovery_mode":"insert_recovery","rollback_steps":0,'
        '"rationale":"re-grasp","recovery_subtasks":[{"instruction":"re-grasp the cup",'
        '"expected_start_state":"cup on table","expected_finish_state":"cup held",'
        '"max_duration_sec":10}]}',
        require_status=False,
    )
    assert inserted.recovery_mode == RecoveryMode.INSERT_RECOVERY
    assert len(inserted.recovery_subtasks) == 1


def test_vector_info_scalar_shapes() -> None:
    assert _first_env_value(False) is False
    assert _first_env_value(np.asarray([True])) is True
    assert _first_env_value(np.asarray([[True]])) is True
    assert _first_env_value(np.asarray([0.25])) == 0.25


def test_rollback_action_handles_singleton_batch() -> None:
    action = {
        "action.end_effector_position": np.arange(6, dtype=np.float32).reshape(1, 3, 2),
        "action.gripper_close": np.asarray([[1, 0, 1]], dtype=np.float32),
    }
    reversed_action = RobocasaVectorEnvAdapter.rollback_action(
        object.__new__(RobocasaVectorEnvAdapter), action
    )
    np.testing.assert_array_equal(
        reversed_action["action.end_effector_position"],
        np.asarray([[-4, -5], [-2, -3], [0, -1]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        reversed_action["action.gripper_close"],
        np.asarray([[1, 0, 1]], dtype=np.float32),
    )


def test_controller_rolls_back_and_retries_current_subtask() -> None:
    subtask = SubtaskSpec("move the cup", "cup on table", "cup at target", 30.0, "move")
    env = _RollbackEnv()
    with tempfile.TemporaryDirectory(prefix="lhc_recovery_test_") as tmp:
        controller = LongHorizonController(
            plan=TaskPlan("move the cup", [subtask]),
            policy=MockPolicyAdapter(trigger_step=1000),
            env=env,
            fast_monitor=ActionEntropyMonitor(
                FastMonitorConfig(
                    complete_score_threshold=0.0,
                    min_steps_before_trigger=2,
                    cooldown_steps=0,
                )
            ),
            vlm_verifier=_Verifier(),
            config=ControllerConfig(
                output_dir=tmp,
                max_total_steps=6,
                max_rollback_chunks=2,
                save_vlm_frames=False,
            ),
        )
        controller.run()
        events = json.loads((Path(tmp) / "controller_events.json").read_text())

    event_types = [event["event_type"] for event in events]
    assert "rollback_step" in event_types
    assert "rollback_retry" in event_types
    assert event_types.count("start_subtask") == 2
    assert event_types.count("advance_subtask") == 1


if __name__ == "__main__":
    test_recovery_json_modes()
    test_vector_info_scalar_shapes()
    test_rollback_action_handles_singleton_batch()
    test_controller_rolls_back_and_retries_current_subtask()
    print("recovery tests passed")
