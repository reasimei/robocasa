#!/usr/bin/env python3
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol
import time

from .fast_monitor import ActionEntropyMonitor, AuxHeadFusionMonitor
from .policy_adapters import PolicyAdapter
from .schemas import ControllerEvent, FastTrigger, SubtaskSpec, TaskPlan, VLMStatus


class EnvironmentAdapter(Protocol):
    def reset(self) -> dict[str, Any]:
        ...

    def step(self, action: dict[str, Any]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        ...

    def get_vlm_image(self, observation: dict[str, Any]) -> Any:
        ...


@dataclass
class ControllerConfig:
    output_dir: str = "/data/zjw/workspace/Isaac-GR00T/expdata/long_horizon_controller"
    max_total_steps: int = 5000
    step_dt_sec: float = 0.05
    # Two temporal snapshots: one historical snapshot and the current snapshot.
    # With the default two cameras this produces four VLM images.
    vlm_history_frames: int = 2
    vlm_history_interval_sec: float = 1.0
    save_vlm_frames: bool = False
    max_rollback_chunks: int = 8


@dataclass
class LongHorizonController:
    plan: TaskPlan
    policy: PolicyAdapter
    env: EnvironmentAdapter
    fast_monitor: ActionEntropyMonitor
    vlm_verifier: Any
    aux_fusion: AuxHeadFusionMonitor | None = None
    config: ControllerConfig = field(default_factory=ControllerConfig)

    def __post_init__(self) -> None:
        self.events: list[ControllerEvent] = []
        self.queue: list[SubtaskSpec] = list(self.plan.subtasks)
        self.completed: list[SubtaskSpec] = []
        self.action_history: list[dict[str, Any]] = []
        self.active_action_history: list[dict[str, Any]] = []

    def log_event(
        self,
        step_index: int,
        subtask_index: int,
        subtask: SubtaskSpec,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.events.append(
            ControllerEvent(
                step_index=step_index,
                subtask_index=subtask_index,
                subtask_id=subtask.subtask_id,
                event_type=event_type,  # type: ignore[arg-type]
                payload=payload or {},
            )
        )

    def save_events(self) -> Path:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "controller_events.json"
        import json

        path.write_text(
            json.dumps([asdict(event) for event in self.events], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    @staticmethod
    def _snapshot_action(action: dict[str, Any]) -> dict[str, Any]:
        snapshot: dict[str, Any] = {}
        for key, value in action.items():
            try:
                import torch

                if torch.is_tensor(value):
                    value = value.detach().cpu().numpy()
            except ImportError:
                pass
            try:
                import numpy as np

                snapshot[key] = np.asarray(value).copy()
            except Exception:
                snapshot[key] = value
        return snapshot

    @staticmethod
    def _jsonable(value: Any) -> Any:
        try:
            import numpy as np

            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
        except ImportError:
            pass
        if isinstance(value, dict):
            return {str(key): LongHorizonController._jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [LongHorizonController._jsonable(item) for item in value]
        return value

    def save_action_history(self) -> Path:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "action_history.json"
        payload = [
            {
                "step_index": item["step_index"],
                "subtask_id": item["subtask_id"],
                "action": self._jsonable(item["action"]),
            }
            for item in self.action_history
        ]
        import json

        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    def _rollback(
        self,
        *,
        observation: dict[str, Any],
        global_step: int,
        current_index: int,
        subtask: SubtaskSpec,
        requested_chunks: int,
        env_success: bool,
    ) -> tuple[dict[str, Any], int, bool, int, bool]:
        """Reverse recent action chunks and return the resulting controller state."""
        rollback_fn = getattr(self.env, "rollback_action", None)
        supports_rollback = bool(getattr(self.env, "supports_action_rollback", False))
        if not supports_rollback or not callable(rollback_fn):
            return observation, global_step, env_success, 0, False

        available = len(self.active_action_history)
        count = min(
            max(1, int(requested_chunks)),
            max(0, int(self.config.max_rollback_chunks)),
            available,
        )
        if count <= 0:
            return observation, global_step, env_success, 0, False

        executed = 0
        done = False
        for entry in reversed(self.active_action_history[-count:]):
            rollback_action = rollback_fn(entry["action"])
            observation, reward, done, info = self.env.step(rollback_action)
            del reward
            global_step += 1
            executed += 1
            if isinstance(info, dict) and "success" in info:
                env_success = bool(info.get("success")) or env_success
            self.log_event(
                global_step,
                current_index,
                subtask,
                "rollback_step",
                {
                    "source_step_index": entry["step_index"],
                    "rollback_step_index": global_step,
                    "action_keys": sorted(entry["action"].keys()),
                },
            )
            if done:
                break

        return observation, global_step, env_success, executed, done

    def policy_instruction(self, subtask: SubtaskSpec) -> str:
        if not self.plan.task_instruction:
            return subtask.instruction
        return (
            f"Overall task: {self.plan.task_instruction}\n"
            f"Current subtask: {subtask.instruction}"
        )

    def _snapshot_vlm_image(self, image: Any) -> Any:
        if isinstance(image, (list, tuple)):
            return [self._snapshot_vlm_image(item) for item in image]
        try:
            import numpy as np

            return np.asarray(image).copy()
        except Exception:
            if hasattr(image, "copy"):
                return image.copy()
            return image

    def _vlm_history_images(
        self,
        image_history: deque[tuple[float, Any]],
        current_time_sec: float,
        current_image: Any,
    ) -> tuple[list[Any], list[float], list[str]]:
        """Select snapshots and flatten each camera pair in temporal order."""
        frame_count = max(1, int(self.config.vlm_history_frames))
        interval_sec = float(self.config.vlm_history_interval_sec)

        history = list(image_history)
        selected: list[tuple[float, Any]] = []
        if frame_count > 1 and interval_sec > 0.0:
            for offset_index in range(frame_count - 1, 0, -1):
                target_time = current_time_sec - offset_index * interval_sec
                candidates = [item for item in history if item[0] <= target_time + 1e-9]
                if candidates:
                    selected.append(candidates[-1])
        selected.append((current_time_sec, current_image))

        deduplicated: list[tuple[float, Any]] = []
        for item in selected:
            if deduplicated and abs(item[0] - deduplicated[-1][0]) < 1e-9:
                continue
            deduplicated.append(item)

        camera_labels = tuple(
            getattr(self.env, "vlm_image_labels", ())
            or ("camera_view",)
        )
        images: list[Any] = []
        timestamps: list[float] = []
        labels: list[str] = []
        for time_sec, bundle in deduplicated:
            bundle_images = list(bundle) if isinstance(bundle, (list, tuple)) else [bundle]
            bundle_labels = (
                list(camera_labels)
                if len(camera_labels) == len(bundle_images)
                else [f"camera_{index}" for index in range(len(bundle_images))]
            )
            offset = round(time_sec - current_time_sec, 3)
            images.extend(bundle_images)
            timestamps.extend([offset] * len(bundle_images))
            labels.extend(bundle_labels)
        return images, timestamps, labels

    def _save_vlm_frames(self, images: Any, step_index: int) -> list[str]:
        if not self.config.save_vlm_frames:
            return []
        try:
            import numpy as np
            from PIL import Image

            frame_dir = Path(self.config.output_dir) / "vlm_frames"
            frame_dir.mkdir(parents=True, exist_ok=True)
            image_list = list(images) if isinstance(images, (list, tuple)) else [images]
            paths: list[str] = []
            for image_index, image in enumerate(image_list):
                array = np.asarray(image)
                while array.ndim > 3:
                    array = array[-1]
                if array.dtype != np.uint8:
                    array = np.clip(array, 0, 255).astype(np.uint8)
                path = frame_dir / f"step_{step_index:06d}_image_{image_index:02d}.png"
                Image.fromarray(array).convert("RGB").save(path)
                paths.append(str(path))
            return paths
        except Exception as exc:
            self.log_event(
                step_index,
                -1,
                SubtaskSpec("", "", "", 0.0, subtask_id="frame_save"),
                "fast_signal",
                {"vlm_frame_save_error": str(exc)},
            )
            return []

    def run(self) -> None:
        """Run the controller and persist partial state even on failure."""
        try:
            self._run_impl()
        finally:
            # VLM, environment, or policy errors can happen after many
            # actions. Keep the partial event/action history for diagnosis.
            self.save_events()
            self.save_action_history()

    def _run_impl(self) -> None:
        observation = self.env.reset()
        global_step = 0
        current_index = 0
        env_success = False
        last_env_success = False
        history_capacity = max(
            2,
            int(
                self.config.vlm_history_frames
                * max(1.0, self.config.vlm_history_interval_sec)
                / max(self.config.step_dt_sec, 1e-6)
            )
            + 4,
        )
        image_history: deque[tuple[float, Any]] = deque(maxlen=history_capacity)
        try:
            image_history.append(
                (0.0, self._snapshot_vlm_image(self.env.get_vlm_image(observation)))
            )
        except Exception:
            image_history.clear()

        while self.queue and global_step < self.config.max_total_steps:
            subtask = self.queue.pop(0)
            self.active_action_history = []
            self.fast_monitor.reset()
            if self.aux_fusion is not None and hasattr(self.aux_fusion, "reset"):
                self.aux_fusion.reset()
            image_history.clear()
            try:
                image_history.append(
                    (
                        global_step * self.config.step_dt_sec,
                        self._snapshot_vlm_image(self.env.get_vlm_image(observation)),
                    )
                )
            except Exception:
                image_history.clear()
            local_step = 0
            fast_trigger_count = 0
            vlm_in_progress_count = 0
            self.log_event(global_step, current_index, subtask, "start_subtask", asdict(subtask))

            while global_step < self.config.max_total_steps:
                elapsed = local_step * self.config.step_dt_sec
                instruction = self.policy_instruction(subtask)
                action, action_chunk = self.policy.act(observation, instruction)
                action_snapshot = self._snapshot_action(action)
                action_entry = {
                    "step_index": global_step,
                    "subtask_id": subtask.subtask_id,
                    "action": action_snapshot,
                }
                self.action_history.append(action_entry)
                self.active_action_history.append(action_entry)
                observation, reward, done, info = self.env.step(action)
                del reward
                if isinstance(info, dict) and "success" in info:
                    last_env_success = bool(info.get("success"))
                    env_success = last_env_success
                global_step += 1
                local_step += 1
                try:
                    image_history.append(
                        (
                            global_step * self.config.step_dt_sec,
                            self._snapshot_vlm_image(self.env.get_vlm_image(observation)),
                        )
                    )
                except Exception:
                    image_history.clear()

                fast_signal = self.fast_monitor.update(
                    action_chunk=action_chunk,
                    elapsed_sec=elapsed,
                    timeout_sec=subtask.max_duration_sec,
                )
                aux_output = getattr(self.policy, "last_aux_output", None)
                if aux_output is None and isinstance(info, dict):
                    aux_output = info.get("aux_head")
                if self.aux_fusion is not None:
                    fast_signal = self.aux_fusion.fuse(fast_signal, aux_output)

                if fast_signal.trigger != FastTrigger.NONE:
                    fast_trigger_count += 1
                    self.log_event(
                        global_step,
                        current_index,
                        subtask,
                        "fast_signal",
                        asdict(fast_signal),
                    )
                    next_subtask = self.queue[0] if self.queue else None
                    current_image = self._snapshot_vlm_image(
                        self.env.get_vlm_image(observation)
                    )
                    image = current_image
                    setattr(subtask, "task_instruction", self.plan.task_instruction)
                    setattr(
                        subtask,
                        "controller_context",
                        (
                            f"Current subtask fast-system trigger #{fast_trigger_count}. "
                            f"Previous VLM in_progress decisions for this subtask: "
                            f"{vlm_in_progress_count}. Latest fast trigger: "
                            f"{fast_signal.trigger.value}; reason: {fast_signal.reason}. "
                            f"Recent action chunks available for rollback: "
                            f"{len(self.active_action_history)}. "
                            f"If rollback_retry is selected, rollback_steps counts "
                            f"policy action chunks, newest first, and must be <= "
                            f"{self.config.max_rollback_chunks}."
                        ),
                    )
                    if (
                        getattr(self.vlm_verifier, "supports_image_history", False)
                        and image_history
                    ):
                        image, image_timestamps, image_labels = self._vlm_history_images(
                            image_history=image_history,
                            current_time_sec=global_step * self.config.step_dt_sec,
                            current_image=current_image,
                        )
                        setattr(subtask, "vlm_image_timestamps_sec", image_timestamps)
                        setattr(subtask, "vlm_image_labels", image_labels)
                        setattr(
                            subtask,
                            "vlm_history_interval_sec",
                            self.config.vlm_history_interval_sec,
                        )
                    vlm_frame_paths = self._save_vlm_frames(image, global_step)
                    verify_start = time.perf_counter()
                    decision = self.vlm_verifier.verify(image, subtask, next_subtask)
                    verify_elapsed = time.perf_counter() - verify_start
                    decision_payload = asdict(decision)
                    decision_payload["elapsed_sec"] = verify_elapsed
                    if vlm_frame_paths:
                        decision_payload["vlm_frame_paths"] = vlm_frame_paths
                    if getattr(subtask, "vlm_image_labels", None):
                        decision_payload["vlm_image_labels"] = getattr(
                            subtask, "vlm_image_labels"
                        )
                    self.log_event(
                        global_step,
                        current_index,
                        subtask,
                        "vlm_decision",
                        decision_payload,
                    )

                    if decision.status == VLMStatus.COMPLETE:
                        vlm_in_progress_count = 0
                        self.completed.append(subtask)
                        self.log_event(global_step, current_index, subtask, "advance_subtask")
                        current_index += 1
                        break

                    if (
                        decision.status == VLMStatus.FAILED
                        and decision.recovery_mode.value == "rollback_retry"
                    ):
                        rollback_start = global_step
                        (
                            observation,
                            global_step,
                            env_success,
                            rollback_count,
                            rollback_done,
                        ) = self._rollback(
                            observation=observation,
                            global_step=global_step,
                            current_index=current_index,
                            subtask=subtask,
                            requested_chunks=decision.rollback_steps or 1,
                            env_success=env_success,
                        )
                        if rollback_count > 0:
                            vlm_in_progress_count = 0
                            self.queue = [subtask] + self.queue
                            self.log_event(
                                global_step,
                                current_index,
                                subtask,
                                "rollback_retry",
                                {
                                    "recovery_mode": decision.recovery_mode.value,
                                    "requested_rollback_chunks": decision.rollback_steps or 1,
                                    "executed_rollback_chunks": rollback_count,
                                    "rollback_start_step": rollback_start,
                                    "rollback_end_step": global_step,
                                    "rollback_done": rollback_done,
                                },
                            )
                            break

                        self.log_event(
                            global_step,
                            current_index,
                            subtask,
                            "rollback_unavailable",
                            {
                                "recovery_mode": decision.recovery_mode.value,
                                "requested_rollback_chunks": decision.rollback_steps or 1,
                                "available_action_chunks": len(self.active_action_history),
                                "reason": "environment adapter does not support rollback or history is empty",
                            },
                        )
                        if not decision.recovery_subtasks:
                            self.queue = [subtask] + self.queue
                            break

                    if decision.status == VLMStatus.FAILED and decision.recovery_subtasks:
                        vlm_in_progress_count = 0
                        self.queue = list(decision.recovery_subtasks) + [subtask] + self.queue
                        self.log_event(
                            global_step,
                            current_index,
                            subtask,
                            "insert_recovery",
                            {
                                "recovery_mode": decision.recovery_mode.value,
                                "num_recovery_subtasks": len(decision.recovery_subtasks),
                                "rollback_steps": decision.rollback_steps,
                            },
                        )
                        break

                    if decision.status == VLMStatus.IN_PROGRESS:
                        vlm_in_progress_count += 1
                        self.fast_monitor.reset()
                        local_step = 0

                if done:
                    env_success = last_env_success
                    self.log_event(
                        global_step,
                        current_index,
                        subtask,
                        "env_done",
                        {"success": env_success},
                    )
                    if env_success and subtask not in self.completed:
                        self.completed.append(subtask)
                    self.queue.clear()
                    break

        controller_completed_plan = len(self.completed) >= len(self.plan.subtasks)
        if env_success or controller_completed_plan:
            finished_subtask = self.completed[-1] if self.completed else SubtaskSpec(
                instruction="",
                expected_start_state="",
                expected_finish_state="",
                max_duration_sec=0.0,
                subtask_id="plan",
            )
            self.log_event(
                global_step,
                current_index,
                finished_subtask,
                "finish_plan",
                {
                    "env_success": env_success,
                    "controller_completed_plan": controller_completed_plan,
                },
            )
