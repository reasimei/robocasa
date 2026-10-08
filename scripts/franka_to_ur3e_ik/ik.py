"""Numerical IK for a fixed-base UR3e.

固定底座 UR3e 的数值 IK。初始对齐允许调整底座，轨迹开始后底座完全锁定。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from robosuite.utils import transform_utils as T

from .geometry import normalize_quaternion


@dataclass(frozen=True)
class IKConfig:
    """Damped least-squares IK configuration."""

    max_iterations: int = 60
    damping: float = 0.04
    step_scale: float = 0.8
    max_joint_step_rad: float = 0.15
    position_tolerance_m: float = 0.0015
    orientation_tolerance_rad: float = 0.02


def _site_pose(env: Any) -> tuple[np.ndarray, np.ndarray]:
    """Read Robotiq grip-site pose in world coordinates."""
    robot = env.robots[0]
    site_id = robot.eef_site_id["right"]
    position = np.asarray(env.sim.data.site_xpos[site_id], dtype=np.float64).copy()
    rotation = np.asarray(env.sim.data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
    return position, normalize_quaternion(T.mat2quat(rotation))


def _joint_indexes(env: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    robot = env.robots[0]
    qpos = np.asarray(robot._ref_arm_joint_pos_indexes, dtype=np.int64)
    qvel = np.asarray(robot._ref_arm_joint_vel_indexes, dtype=np.int64)
    joints = np.asarray(robot._ref_arm_joint_indexes, dtype=np.int64)
    if qpos.size != 6 or qvel.size != 6:
        raise RuntimeError(f"Expected six UR3e joints, got qpos={qpos}, qvel={qvel}")
    return qpos, qvel, joints


def solve_grip_site_ik(
    env: Any,
    target_position: np.ndarray,
    target_quaternion_xyzw: np.ndarray,
    *,
    initial_q: np.ndarray | None = None,
    config: IKConfig = IKConfig(),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve one grip-site pose without changing the persistent sim state."""
    qpos_idx, qvel_idx, joint_idx = _joint_indexes(env)
    target_position = np.asarray(target_position, dtype=np.float64).reshape(3)
    target_quaternion = normalize_quaternion(target_quaternion_xyzw)
    original_qpos = np.asarray(env.sim.data.qpos, dtype=np.float64).copy()
    original_qvel = np.asarray(env.sim.data.qvel, dtype=np.float64).copy()
    original_qacc = np.asarray(env.sim.data.qacc, dtype=np.float64).copy()
    q = (
        np.asarray(initial_q, dtype=np.float64).reshape(6).copy()
        if initial_q is not None
        else np.asarray(env.sim.data.qpos[qpos_idx], dtype=np.float64).copy()
    )
    q_reference = q.copy()
    ranges = np.asarray(env.sim.model.jnt_range[joint_idx], dtype=np.float64)
    best_q = q.copy()
    best_score = float("inf")
    best_position_error = float("inf")
    best_orientation_error = float("inf")
    converged = False
    iterations = 0

    def errors() -> tuple[np.ndarray, np.ndarray]:
        position, quaternion = _site_pose(env)
        return (
            target_position - position,
            T.get_orientation_error(target_quaternion, quaternion),
        )

    try:
        env.sim.data.qpos[qpos_idx] = q
        env.sim.data.qvel[qvel_idx] = 0.0
        env.sim.forward()
        for iterations in range(1, config.max_iterations + 1):
            position_error, orientation_error = errors()
            position_norm = float(np.linalg.norm(position_error))
            orientation_norm = float(np.linalg.norm(orientation_error))
            score = (
                position_norm / max(config.position_tolerance_m, 1e-9)
                + orientation_norm / max(config.orientation_tolerance_rad, 1e-9)
            )
            if score < best_score:
                best_score = score
                best_q = q.copy()
                best_position_error = position_norm
                best_orientation_error = orientation_norm
            if (
                position_norm <= config.position_tolerance_m
                and orientation_norm <= config.orientation_tolerance_rad
            ):
                converged = True
                break

            site_name = env.sim.model.site_id2name(env.robots[0].eef_site_id["right"])
            jacobian = np.vstack(
                (
                    env.sim.data.get_site_jacp(site_name).reshape(3, -1)[:, qvel_idx],
                    env.sim.data.get_site_jacr(site_name).reshape(3, -1)[:, qvel_idx],
                )
            )
            error = np.concatenate((position_error, orientation_error))
            lhs = jacobian.T @ jacobian + (config.damping**2) * np.eye(6)
            dq = np.linalg.solve(lhs, jacobian.T @ error) * config.step_scale
            dq_norm = float(np.linalg.norm(dq))
            if dq_norm > config.max_joint_step_rad:
                dq *= config.max_joint_step_rad / dq_norm
            q = q + dq
            for index, (lower, upper) in enumerate(ranges):
                if lower < upper:
                    q[index] = np.clip(q[index], lower + 1e-5, upper - 1e-5)
            env.sim.data.qpos[qpos_idx] = q
            env.sim.data.qvel[qvel_idx] = 0.0
            env.sim.forward()

        q = best_q
        env.sim.data.qpos[qpos_idx] = q
        env.sim.data.qvel[qvel_idx] = 0.0
        env.sim.forward()
        final_position_error, final_orientation_error = errors()
        info = {
            "iterations": iterations,
            "converged": bool(
                converged
                or (
                    np.linalg.norm(final_position_error) <= config.position_tolerance_m
                    and np.linalg.norm(final_orientation_error)
                    <= config.orientation_tolerance_rad
                )
            ),
            "position_error_m": float(np.linalg.norm(final_position_error)),
            "orientation_error_rad": float(np.linalg.norm(final_orientation_error)),
            "best_position_error_m": best_position_error,
            "best_orientation_error_rad": best_orientation_error,
            "joint_delta_norm_rad": float(np.linalg.norm(q - q_reference)),
        }
        return q.copy(), info
    finally:
        env.sim.data.qpos[:] = original_qpos
        env.sim.data.qvel[:] = original_qvel
        env.sim.data.qacc[:] = original_qacc
        env.sim.forward()


def align_initial_pose(
    env: Any,
    target_position: np.ndarray,
    target_quaternion_xyzw: np.ndarray,
    *,
    config: IKConfig = IKConfig(max_iterations=160),
    allow_base_xy: bool = True,
    base_regularization: float = 5.0,
    max_base_step_m: float = 0.03,
) -> dict[str, Any]:
    """Align UR3e initially while optionally moving only its XY base offset.

    只在初始阶段将 XY 底座位移作为 IK 自由度；返回后调用方必须锁住底座。
    """
    robot = env.robots[0]
    qpos_idx, qvel_idx, _ = _joint_indexes(env)
    site_name = env.sim.model.site_id2name(robot.eef_site_id["right"])
    base_id = env.sim.model.body_name2id("robot0_base")
    base_start = np.asarray(env.sim.model.body_pos[base_id], dtype=np.float64).copy()
    q = np.asarray(env.sim.data.qpos[qpos_idx], dtype=np.float64).copy()
    target_position = np.asarray(target_position, dtype=np.float64).reshape(3)
    target_quaternion = normalize_quaternion(target_quaternion_xyzw)
    base_axes = np.array([0, 1], dtype=np.int64) if allow_base_xy else np.array([], dtype=np.int64)
    last_q = q.copy()
    last_base = base_start.copy()
    last_score = float("inf")
    iterations = 0

    def read_errors() -> tuple[np.ndarray, np.ndarray]:
        current_position, current_quaternion = _site_pose(env)
        return (
            target_position - current_position,
            T.get_orientation_error(target_quaternion, current_quaternion),
        )

    for iterations in range(1, config.max_iterations + 1):
        position_error, orientation_error = read_errors()
        if (
            np.linalg.norm(position_error) <= config.position_tolerance_m
            and np.linalg.norm(orientation_error) <= config.orientation_tolerance_rad
        ):
            break
        arm_jacobian = np.vstack(
            (
                env.sim.data.get_site_jacp(site_name).reshape(3, -1)[:, qvel_idx],
                env.sim.data.get_site_jacr(site_name).reshape(3, -1)[:, qvel_idx],
            )
        )
        if base_axes.size:
            base_jacobian = np.vstack(
                (
                    np.eye(3, dtype=np.float64)[:, base_axes],
                    np.zeros((3, base_axes.size), dtype=np.float64),
                )
            )
            jacobian = np.column_stack((arm_jacobian, base_jacobian))
            regularizer = np.diag(
                np.concatenate(
                    (
                        np.full(6, config.damping**2),
                        np.full(base_axes.size, (config.damping * base_regularization) ** 2),
                    )
                )
            )
        else:
            jacobian = arm_jacobian
            regularizer = config.damping**2 * np.eye(6)
        delta = np.linalg.solve(
            jacobian.T @ jacobian + regularizer,
            jacobian.T @ np.concatenate((position_error, orientation_error)),
        ) * config.step_scale
        dq = delta[:6]
        dq_norm = float(np.linalg.norm(dq))
        if dq_norm > config.max_joint_step_rad:
            dq *= config.max_joint_step_rad / dq_norm
        base_delta = np.zeros(3, dtype=np.float64)
        if base_axes.size:
            base_delta[base_axes] = delta[6:]
            base_norm = float(np.linalg.norm(base_delta))
            if base_norm > max_base_step_m:
                base_delta *= max_base_step_m / base_norm
        q = q + dq
        ranges = np.asarray(env.sim.model.jnt_range[robot._ref_arm_joint_indexes], dtype=np.float64)
        for index, (lower, upper) in enumerate(ranges):
            if lower < upper:
                q[index] = np.clip(q[index], lower + 1e-5, upper - 1e-5)
        env.sim.data.qpos[qpos_idx] = q
        env.sim.data.qvel[qvel_idx] = 0.0
        env.sim.model.body_pos[base_id] = np.asarray(env.sim.model.body_pos[base_id]) + base_delta
        env.sim.forward()
        position_error, orientation_error = read_errors()
        score = float(
            np.linalg.norm(position_error) / max(config.position_tolerance_m, 1e-9)
            + np.linalg.norm(orientation_error) / max(config.orientation_tolerance_rad, 1e-9)
        )
        if score > last_score * 1.05:
            q = last_q
            env.sim.data.qpos[qpos_idx] = q
            env.sim.model.body_pos[base_id] = last_base
            env.sim.forward()
            break
        last_q = q.copy()
        last_base = np.asarray(env.sim.model.body_pos[base_id], dtype=np.float64).copy()
        last_score = score

    position, quaternion = _site_pose(env)
    position_error, orientation_error = read_errors()
    # Refresh controller state after direct qpos changes.
    robot.composite_controller.update_state()
    robot.composite_controller.reset()
    return {
        "target_position_world": target_position.tolist(),
        "target_quaternion_world_xyzw": target_quaternion.tolist(),
        "solved_position_world": position.tolist(),
        "solved_quaternion_world_xyzw": quaternion.tolist(),
        "position_error_m": float(np.linalg.norm(position_error)),
        "orientation_error_rad": float(np.linalg.norm(orientation_error)),
        "iterations": iterations,
        "base_start_world": base_start.tolist(),
        "base_final_world": np.asarray(env.sim.model.body_pos[base_id]).tolist(),
        "base_translation_world": (
            np.asarray(env.sim.model.body_pos[base_id]) - base_start
        ).tolist(),
        "base_was_movable_only_during_initial_alignment": True,
    }

