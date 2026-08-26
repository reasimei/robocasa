#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class MockEnvironment:
    max_steps: int = 200
    image_size: int = 256

    def __post_init__(self) -> None:
        self.step_index = 0

    def reset(self) -> dict[str, Any]:
        self.step_index = 0
        return self._obs()

    def step(self, action: dict[str, Any]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        del action
        self.step_index += 1
        done = self.step_index >= self.max_steps
        return self._obs(), 0.0, done, {}

    def get_vlm_image(self, observation: dict[str, Any]) -> np.ndarray:
        return observation["video.robot0_agentview_left"][-1]

    def _obs(self) -> dict[str, Any]:
        image = np.zeros((1, self.image_size, self.image_size, 3), dtype=np.uint8)
        image[..., 0] = (self.step_index * 5) % 255
        image[..., 1] = 80
        image[..., 2] = 120
        return {
            "video.robot0_agentview_left": image,
            "video.robot0_agentview_right": image.copy(),
            "video.robot0_eye_in_hand": image.copy(),
            "state.end_effector_position_relative": np.zeros((1, 3), dtype=np.float32),
            "state.end_effector_rotation_relative": np.asarray([[0, 0, 0, 1]], dtype=np.float32),
            "state.gripper_qpos": np.zeros((1, 2), dtype=np.float32),
            "state.base_position": np.zeros((1, 3), dtype=np.float32),
            "state.base_rotation": np.asarray([[0, 0, 0, 1]], dtype=np.float32),
            "annotation.human.task_description": ["mock task"],
        }
