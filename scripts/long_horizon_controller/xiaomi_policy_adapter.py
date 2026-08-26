#!/usr/bin/env python3
"""Xiaomi Robotics-1 policy adapter for the local RoboCasa365 checkpoint."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image


CAMERA_KEYS = (
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
)


def _first_array(value: Any) -> np.ndarray:
    array = np.asarray(value)
    while array.ndim > 1 and array.shape[0] == 1:
        array = array[0]
    return np.asarray(array, dtype=np.float32).reshape(-1)


def _first_image(value: Any) -> np.ndarray:
    array = np.asarray(value)
    while array.ndim > 3:
        array = array[0] if array.shape[0] == 1 else array[-1]
    if array.ndim != 3:
        raise ValueError(f"Expected an RGB image, got shape {array.shape}")
    return np.asarray(array, dtype=np.uint8)


def _quat_xyzw_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    norm = np.linalg.norm(quaternion)
    if norm < 1e-12:
        return np.zeros(3, dtype=np.float32)
    quaternion = quaternion / norm
    if quaternion[3] < 0:
        quaternion = -quaternion
    xyz = quaternion[:3]
    sin_half = np.linalg.norm(xyz)
    if sin_half < 1e-12:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arctan2(sin_half, np.clip(quaternion[3], -1.0, 1.0))
    return (xyz / sin_half * angle).astype(np.float32)


def _center_crop(image: np.ndarray, crop_ratio: float) -> Image.Image:
    pil_image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    if crop_ratio >= 1.0:
        return pil_image
    width, height = pil_image.size
    crop_width = max(1, int(width * crop_ratio))
    crop_height = max(1, int(height * crop_ratio))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    cropped = pil_image.crop((left, top, left + crop_width, top + crop_height))
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    return cropped.resize((width, height), resampling)


@dataclass
class XiaomiPolicyAdapter:
    model_path: str
    device: str = "cuda"
    dtype: Any = None
    history_length: int = 4
    action_steps: int = 16
    num_diffusion_steps: int = 5
    clip_actions: bool = True
    crop_ratio: float = 0.95

    def __post_init__(self) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.dtype = self.dtype or torch.bfloat16
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_fast=False,
        )
        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=self.dtype,
            attn_implementation="eager",
        ).to(self.device)
        self.model.eval()
        self._images: dict[str, deque[np.ndarray]] = {
            key: deque(maxlen=self.history_length) for key in CAMERA_KEYS
        }
        self._states: deque[np.ndarray] = deque(maxlen=self.history_length)

    def reset(self) -> None:
        for frames in self._images.values():
            frames.clear()
        self._states.clear()

    def _state_vector(
        self,
        eef_pos: np.ndarray,
        eef_quat_xyzw: np.ndarray,
        gripper_qpos: np.ndarray,
        base_pos: np.ndarray,
        base_quat_xyzw: np.ndarray,
    ) -> np.ndarray:
        # Match Xiaomi's official RoboCasa365 evaluator: EE-first 14D state
        # with both quaternions converted to axis-angle, then pad to 60D.
        raw_state = np.concatenate(
            [
                np.asarray(eef_pos, dtype=np.float32).reshape(-1),
                _quat_xyzw_to_axis_angle(eef_quat_xyzw),
                np.asarray(gripper_qpos, dtype=np.float32).reshape(-1),
                np.asarray(base_pos, dtype=np.float32).reshape(-1),
                _quat_xyzw_to_axis_angle(base_quat_xyzw),
            ],
            axis=0,
        ).astype(np.float32)
        if raw_state.shape != (14,):
            raise ValueError(f"Expected 14D official RoboCasa365 state, got {raw_state.shape}")
        state = np.zeros(60, dtype=np.float32)
        state[: raw_state.size] = raw_state
        return state

    def _image_history(self, value: Any) -> list[np.ndarray]:
        array = np.asarray(value)
        if array.ndim == 5 and array.shape[0] == 1:
            array = array[0]
        if array.ndim == 4:
            return [np.asarray(frame, dtype=np.uint8) for frame in array[-self.history_length:]]
        return [_first_image(array)]

    def _state_history(self, observation: dict[str, Any]) -> np.ndarray:
        history: dict[str, np.ndarray] = {}
        for key in (
            "state.end_effector_position_relative",
            "state.end_effector_rotation_relative",
            "state.gripper_qpos",
            "state.base_position",
            "state.base_rotation",
        ):
            value = np.asarray(observation[key], dtype=np.float32)
            if value.ndim >= 3 and value.shape[0] == 1:
                value = value[0]
            if value.ndim == 1:
                value = value[None, :]
            history[key] = value[-self.history_length:]
        length = min(piece.shape[0] for piece in history.values())
        return np.stack(
            [
                self._state_vector(
                    history["state.end_effector_position_relative"][-length + index],
                    history["state.end_effector_rotation_relative"][-length + index],
                    history["state.gripper_qpos"][-length + index],
                    history["state.base_position"][-length + index],
                    history["state.base_rotation"][-length + index],
                )
                for index in range(length)
            ],
            axis=0,
        )

    def _append_latest_observation(self, observation: dict[str, Any]) -> None:
        for key in CAMERA_KEYS:
            if key not in observation:
                raise KeyError(f"Missing Xiaomi camera input {key!r}")
            self._images[key].append(self._image_history(observation[key])[-1])
        self._states.append(self._state_history(observation)[-1])
        while len(self._states) < self.history_length:
            self._states.appendleft(self._states[0].copy())
        for key in CAMERA_KEYS:
            while len(self._images[key]) < self.history_length:
                self._images[key].appendleft(self._images[key][0].copy())

    def _observation_history(self, observation: dict[str, Any]) -> tuple[list[np.ndarray], np.ndarray]:
        image_histories = {
            key: self._image_history(observation[key])
            for key in CAMERA_KEYS
            if key in observation
        }
        state_history = self._state_history(observation)
        if (
            len(image_histories) == len(CAMERA_KEYS)
            and state_history.shape[0] >= self.history_length
            and all(len(history) >= self.history_length for history in image_histories.values())
        ):
            videos = [
                np.stack(image_histories[key][-self.history_length:], axis=0)
                for key in CAMERA_KEYS
            ]
            return videos, state_history[-self.history_length:][None, ...]

        self._append_latest_observation(observation)
        videos = [
            np.stack(list(self._images[key]), axis=0)
            for key in CAMERA_KEYS
        ]
        state = np.stack(list(self._states), axis=0)[None, ...]
        return videos, state

    def _prompt(self, instruction: str) -> str:
        marker = (
            f"{self.processor.vision_start_token}"
            f"{self.processor.video_token}"
            f"{self.processor.vision_end_token}"
        )
        return (
            "<|im_start|>user\n"
            f"Left camera: {marker}"
            f"\nRight camera: {marker}"
            f"\nWrist camera: {marker}"
            f"\n\nGenerate robot actions for the task:\n{instruction} /no_cot"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<cot></cot>"
            "<|im_end|>\n"
        )

    def _crop_videos(self, videos: list[np.ndarray]) -> list[np.ndarray]:
        return [
            np.stack(
                [np.asarray(_center_crop(frame, self.crop_ratio), dtype=np.uint8) for frame in video],
                axis=0,
            )
            for video in videos
        ]

    def act(
        self,
        observation: dict[str, Any],
        instruction: str,
    ) -> tuple[dict[str, Any], np.ndarray]:
        videos, state = self._observation_history(observation)
        inputs = self.processor(
            videos=self._crop_videos(videos),
            text=self._prompt(instruction),
            return_tensors="pt",
            state=state,
            robot_type="robocasa365",
        )
        inputs = {
            key: value.to(self.device)
            if hasattr(value, "to")
            else value
            for key, value in inputs.items()
        }
        with self.torch.inference_mode():
            output = self.model(
                **inputs,
                num_steps=self.num_diffusion_steps,
            )
        actions = self.processor.decode_action(
            output.actions,
            "robocasa365",
        )[0].float().detach().cpu().numpy()
        actions = np.asarray(actions[: self.action_steps], dtype=np.float32)
        if actions.shape != (self.action_steps, 60):
            raise ValueError(f"Unexpected Xiaomi action shape: {actions.shape}")
        if self.clip_actions:
            actions[:, :12] = np.clip(actions[:, :12], -1.0, 1.0)

        # Match Xiaomi's official RoboCasa365 evaluator. The first 12 dims are
        # in robocasa.utils.env_utils.convert_action order:
        # eef_pos, eef_rot, gripper, base, control_mode.
        action = {
            # SyncVectorEnv expects a leading environment dimension.
            "action.end_effector_position": actions[None, :, 0:3],
            "action.end_effector_rotation": actions[None, :, 3:6],
            "action.gripper_close": actions[None, :, 6:7],
            "action.base_motion": actions[None, :, 7:11],
            "action.control_mode": actions[None, :, 11:12],
        }
        action_chunk = np.concatenate(
            [
                actions[:, 0:3],
                actions[:, 3:6],
                actions[:, 6:7],
                actions[:, 7:11],
                actions[:, 11:12],
            ],
            axis=-1,
        )
        return action, action_chunk
