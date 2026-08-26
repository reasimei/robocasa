#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _first_env_image(value: Any) -> np.ndarray:
    array = np.asarray(value)
    while array.ndim > 3:
        if array.shape[0] == 1:
            array = array[0]
        else:
            array = array[-1]
    return array


def _first_env_value(value: Any, default: Any = None) -> Any:
    """Read the first value from scalar, (N,), or (N, 1) vector info."""
    array = np.asarray(value)
    if array.size == 0:
        return default
    return array.reshape(-1)[0].item()


@dataclass
class RobocasaVectorEnvAdapter:
    """Adapter for the GR00T Robocasa vector env used by scripts/run_eval.py.

    This controller currently owns one environment instance at a time, because the
    subtask queue, VLM verifier, and recovery insertion are episode-specific.
    """

    simulation_config: Any
    vlm_image_key: str = (
        "video.robot0_agentview_left,video.robot0_eye_in_hand"
    )
    host: str = "localhost"
    port: int = 0

    def __post_init__(self) -> None:
        if int(self.simulation_config.n_envs) != 1:
            raise ValueError("RobocasaVectorEnvAdapter currently supports n_envs=1 only.")
        self.vlm_image_keys = tuple(
            key.strip() for key in self.vlm_image_key.split(",") if key.strip()
        )
        if not self.vlm_image_keys:
            raise ValueError("vlm_image_key must contain at least one observation key.")
        self.vlm_image_labels = tuple(self._camera_label(key) for key in self.vlm_image_keys)
        from gr00t.eval.simulation import SimulationInferenceClient

        self._client = SimulationInferenceClient(host=self.host, port=self.port)
        self._env = self._client.setup_environment(self.simulation_config)
        # Robocasa's default PandaOmron arm controller is delta-controlled
        # (OSC_POSE); this is used by the controller to gate rollback mode.
        self.supports_action_rollback = True

    @staticmethod
    def _camera_label(key: str) -> str:
        labels = {
            "video.robot0_agentview_left": "agentview_left",
            "video.robot0_agentview_right": "agentview_right",
            "video.robot0_eye_in_hand": "eye_in_hand",
        }
        return labels.get(key, key.rsplit(".", 1)[-1])

    def reset(self) -> dict[str, Any]:
        observation, _ = self._env.reset()
        return observation

    def step(self, action: dict[str, Any]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        observation, rewards, terminations, truncations, infos = self._env.step(action)
        done = bool(
            _first_env_value(terminations, False)
            or _first_env_value(truncations, False)
        )
        info = {
            "success": bool(_first_env_value(infos["success"], False))
            if "success" in infos
            else False,
            "raw_env_info": infos,
        }
        return observation, float(_first_env_value(rewards, 0.0)), done, info

    def rollback_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Create a conservative reverse action for one policy action chunk.

        The PandaOmron arm and mobile-base velocity inputs are delta controls.
        Gripper and control-mode values are state-like commands, so they are
        held at the most recent value while the motion channels are reversed.
        """
        rollback: dict[str, Any] = {}
        for key, value in action.items():
            array = np.asarray(value).copy()
            if array.ndim == 0:
                rollback[key] = array
                continue

            # The vector env consumes action chunks as (T, D). A policy adapter
            # normally removes the batch dimension, but accept a singleton
            # (1, T, D) here as well so rollback cannot reverse the batch axis.
            had_batch_dim = array.ndim >= 3 and array.shape[0] == 1
            time_major = array[0] if had_batch_dim else array
            if time_major.ndim == 1:
                time_major = time_major[None, :]
            reversed_array = np.flip(time_major, axis=0).copy()
            if key in {
                "action.end_effector_position",
                "action.end_effector_rotation",
                "action.base_motion",
            }:
                reversed_array *= -1.0
            elif key in {
                "action.gripper_close",
                "action.control_mode",
            }:
                # These are binary/state commands rather than delta controls.
                # Keeping the latest command avoids opening a grasp during a
                # pose rollback.
                reversed_array[...] = time_major[-1]
            rollback[key] = reversed_array
        return rollback

    def get_vlm_image(self, observation: dict[str, Any]) -> Any:
        missing = [key for key in self.vlm_image_keys if key not in observation]
        if missing:
            raise KeyError(
                f"Observation does not contain VLM image keys {missing!r}. "
                f"Available keys: {sorted(observation.keys())}"
            )
        images = [_first_env_image(observation[key]) for key in self.vlm_image_keys]
        return images[0] if len(images) == 1 else images

    def close(self) -> None:
        self._env.close()
