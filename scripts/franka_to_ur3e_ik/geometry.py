"""Pose and gripper conversion utilities.

末端位姿和夹爪状态转换工具。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robosuite.utils import transform_utils as T


def normalize_quaternion(quaternion_xyzw: np.ndarray) -> np.ndarray:
    """Normalize an xyzw quaternion / 归一化 xyzw 四元数。"""
    value = np.asarray(quaternion_xyzw, dtype=np.float64).reshape(4)
    return value / max(float(np.linalg.norm(value)), 1e-12)


def pose_matrix(position: np.ndarray, quaternion_xyzw: np.ndarray) -> np.ndarray:
    """Build a homogeneous transform from position and xyzw quaternion."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = T.quat2mat(normalize_quaternion(quaternion_xyzw))
    transform[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    return transform


def pose_from_matrix(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return position and xyzw quaternion from a homogeneous transform."""
    matrix = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    return matrix[:3, 3].copy(), normalize_quaternion(T.mat2quat(matrix[:3, :3]))


def map_relative_pose(
    source_initial_position: np.ndarray,
    source_initial_quaternion: np.ndarray,
    source_position: np.ndarray,
    source_quaternion: np.ndarray,
    target_initial_position: np.ndarray,
    target_initial_quaternion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map a source EEF pose by preserving motion in the initial EEF frame.

    计算：
        T_target = T_target0 * inv(T_source0) * T_source

    This keeps the learned / recorded local motion while allowing the UR3e
    to have a different initial mounting pose.
    """
    source_initial = pose_matrix(source_initial_position, source_initial_quaternion)
    source_current = pose_matrix(source_position, source_quaternion)
    target_initial = pose_matrix(target_initial_position, target_initial_quaternion)
    target_current = target_initial @ np.linalg.inv(source_initial) @ source_current
    return pose_from_matrix(target_current)


@dataclass(frozen=True)
class GripperMapping:
    """Linear Panda opening to Robotiq normalized signal mapping.

    Panda opening 0.0 m -> Robotiq close signal +1.
    Panda opening 0.08 m -> Robotiq open signal -1.
    """

    panda_closed_opening_m: float = 0.0
    panda_open_opening_m: float = 0.08
    robotiq_closed_signal: float = 1.0
    robotiq_open_signal: float = -1.0

    def to_robotiq_signal(self, panda_opening_m: float) -> float:
        """Convert Panda finger separation to a Robotiq control signal."""
        fraction = np.clip(
            (float(panda_opening_m) - self.panda_closed_opening_m)
            / max(self.panda_open_opening_m - self.panda_closed_opening_m, 1e-8),
            0.0,
            1.0,
        )
        return float(
            self.robotiq_closed_signal
            + fraction * (self.robotiq_open_signal - self.robotiq_closed_signal)
        )


def panda_opening_from_qpos(gripper_qpos: np.ndarray) -> float | None:
    """Read PandaOmron finger opening from its two slide joints."""
    values = np.asarray(gripper_qpos, dtype=np.float64).reshape(-1)
    if values.size < 2:
        return None
    # The second Panda finger joint has the opposite sign convention.
    # Panda 第二个手指关节符号相反，因此开口是 q1 - q2。
    return float(np.clip(values[0] - values[1], 0.0, 0.08))


def orientation_error_rad(
    target_quaternion_xyzw: np.ndarray,
    actual_quaternion_xyzw: np.ndarray,
) -> float:
    """Return the angle-axis orientation error in radians."""
    error = T.get_orientation_error(
        normalize_quaternion(target_quaternion_xyzw),
        normalize_quaternion(actual_quaternion_xyzw),
    )
    return float(np.linalg.norm(error))

