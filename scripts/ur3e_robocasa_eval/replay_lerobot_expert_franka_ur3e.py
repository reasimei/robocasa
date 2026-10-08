#!/usr/bin/env python3
"""Replay one RoboCasa LeRobot expert episode on Franka and UR3e.

The Franka side is restored from the episode's own model.xml.gz and initial
MuJoCo state, then replays the recorded actions. The UR3e side uses the same
RoboCasa scene metadata, copies all common non-robot joint states, aligns its
grip site to the Franka grip site, and follows the recorded Franka grip-site
poses with numerical IK.

This is an isolated migration experiment. It does not change any existing
RoboCasa, GR00T, or Xiaomi evaluator.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robocasa.utils import env_utils
from robocasa.utils.lerobot_utils import reorder_lerobot_action
import robocasa.utils.object_utils as object_utils
from robosuite.utils import transform_utils as T

from scripts.ur3e_robocasa_eval.run_xiaomi_ur3e_fixed_eval import (
    _resize_frame,
    align_initial_eef_to,
    register_ur3e,
)


DEFAULT_DATASET_ROOT = Path(
    "/data/zjw/workspace/robocasa/datasets/v1.0/target/composite/"
    "PreSoakPan/20250809/lerobot"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/zjw/workspace/Isaac-GR00T/expdata/ur3e_robocasa_expert_replay/"
    "PreSoakPan_episode_000237_grip_site"
)


_ORIGINAL_CHECK_OBJ_GRASPED = object_utils.check_obj_grasped
_ORIGINAL_LOAD_CONTROLLER_CONFIG = env_utils.load_composite_controller_config
_UR3E_CONTROLLER_MODE = "osc_eef_delta"
_UR3E_JOINT_POSITION_KP = 150.0
_UR3E_JOINT_POSITION_DAMPING_RATIO = 1.0


def _configure_ur3e_body_collision(
    env: Any,
    *,
    enabled: bool,
) -> dict[str, Any]:
    """Configure UR3e-link collision while preserving Robotiq contacts.

    The isolated migration model uses the official link meshes as both visual
    and collision geometry.  Use a separate collision bit for UR3e links so
    they collide with the gripper and scene objects, but not with one another.
    This avoids both visible/physical gripper-through-link motion and the
    self-collision deadlocks that result from enabling every link pair.
    """
    changed: list[dict[str, Any]] = []
    preserved: list[str] = []
    for geom_id in range(int(env.sim.model.ngeom)):
        geom_name = env.sim.model.geom_id2name(geom_id) or ""
        body_id = int(env.sim.model.geom_bodyid[geom_id])
        body_name = env.sim.model.body_id2name(body_id) or ""
        is_gripper = (
            geom_name.startswith("gripper0_")
            or body_name.startswith("gripper0_")
            or "finger" in geom_name.lower()
            or "knuckle" in geom_name.lower()
            or "hand_collision" in geom_name.lower()
        )
        is_ur3e_link = body_name.startswith("robot0_") and not is_gripper
        if is_ur3e_link:
            old = (
                int(env.sim.model.geom_contype[geom_id]),
                int(env.sim.model.geom_conaffinity[geom_id]),
            )
            # MuJoCo collision test is
            # (contype_a & conaffinity_b) or (contype_b & conaffinity_a).
            # Link (2, 1) and gripper/object (1, 1/2) collide across groups,
            # while two UR3e links (2, 1) do not collide with each other.
            new = (2, 1) if enabled else (0, 0)
            env.sim.model.geom_contype[geom_id] = new[0]
            env.sim.model.geom_conaffinity[geom_id] = new[1]
            if old != new:
                changed.append(
                    {
                        "geom_id": geom_id,
                        "geom_name": geom_name,
                        "body_name": body_name,
                        "old_contype": old[0],
                        "old_conaffinity": old[1],
                        "new_contype": new[0],
                        "new_conaffinity": new[1],
                    }
                )
        elif is_gripper and (
            int(env.sim.model.geom_contype[geom_id]) != 0
            or int(env.sim.model.geom_conaffinity[geom_id]) != 0
        ):
            preserved.append(geom_name)
    env.sim.forward()
    return {
        "enabled": bool(enabled),
        "policy": (
            "ur3e_links_contype_2_conaffinity_1; "
            "preserve gripper contacts; disable ur3e-link self-collision"
            if enabled
            else "all ur3e-link collision disabled"
        ),
        "changed_link_geom_count": len(changed),
        "changed_link_geoms": changed,
        "preserved_gripper_collision_geom_count": len(preserved),
        "preserved_gripper_collision_geoms": preserved,
    }


def _joint_position_controller_config() -> dict[str, Any]:
    return {
        "type": "BASIC",
        "body_parts": {
            "right": {
                "type": "JOINT_POSITION",
                "input_max": 1.0,
                "input_min": -1.0,
                "output_max": 0.05,
                "output_min": -0.05,
                "kp": _UR3E_JOINT_POSITION_KP,
                "damping_ratio": _UR3E_JOINT_POSITION_DAMPING_RATIO,
                "impedance_mode": "fixed",
                "kp_limits": [0, 300],
                "damping_ratio_limits": [0, 10],
                "qpos_limits": None,
                "interpolation": None,
                "input_type": "absolute",
                "gripper": {"type": "GRIP"},
            }
        }
    }


def _load_controller_config_for_replay(
    controller: str | None = None,
    robot: str | None = None,
) -> dict[str, Any]:
    if (
        controller is None
        and robot == "UR3eOfficialFixed"
        and _UR3E_CONTROLLER_MODE == "joint_position"
    ):
        return copy.deepcopy(_joint_position_controller_config())
    return _ORIGINAL_LOAD_CONTROLLER_CONFIG(controller=controller, robot=robot)


env_utils.load_composite_controller_config = _load_controller_config_for_replay


def _qpos_indices_for_joints(env: Any, joint_names: list[str]) -> list[int]:
    """Return scalar MuJoCo qpos indices for the named hinge joints."""
    indices: list[int] = []
    for joint_name in joint_names:
        try:
            address = env.sim.model.get_joint_qpos_addr(joint_name)
        except Exception:
            continue
        if isinstance(address, (tuple, list, np.ndarray)):
            indices.extend(int(index) for index in np.asarray(address).reshape(-1))
        else:
            indices.append(int(address))
    return indices


def _robotiq_obj_grasped(env: Any, obj_name: str, threshold: float) -> bool:
    """Check a Robotiq85 grasp without assuming Panda finger joint names."""
    obj = env.objects[obj_name]
    robot = env.robots[0]
    gripper = robot.gripper.get("right")
    if gripper is None:
        raise AttributeError("Gripper dictionary does not contain a 'right' key.")

    # The two actuated Robotiq joints are mirrored by tendons. The remaining
    # four joints are passive followers and may have negative angles, so they
    # are not suitable for the Panda-style ``qpos < threshold`` test.
    joint_names = [
        name
        for name in getattr(robot, "gripper_joints", {}).get("right", [])
        if name.endswith(("finger_joint", "right_outer_knuckle_joint"))
    ]
    qpos_indices = _qpos_indices_for_joints(env, joint_names)
    if not qpos_indices:
        # This is only a defensive fallback for an unfamiliar gripper model.
        # Contact remains meaningful, while the normal Panda path below keeps
        # its original strict joint test.
        return bool(env.check_contact(gripper, obj))

    qpos = np.asarray(env.sim.data.qpos[qpos_indices], dtype=np.float64)
    joint_ranges = []
    for joint_name in joint_names:
        try:
            joint_id = env.sim.model.joint_name2id(joint_name)
            joint_ranges.append(
                np.asarray(env.sim.model.jnt_range[joint_id], dtype=np.float64)
            )
        except Exception:
            pass

    # For Robotiq85, open is approximately 0 rad and closing increases the
    # actuated outer-knuckle angles. A small positive threshold also permits
    # an object to stop the fingers before the no-load 0.8-rad limit.
    closed = bool(np.any(qpos > max(float(threshold), 0.01)))
    in_contact = bool(env.check_contact(gripper, obj))
    return in_contact and closed


def _check_obj_grasped_compatible(
    env: Any,
    obj_name: str,
    threshold: float = 0.035,
) -> bool:
    """Keep RoboCasa's Panda check and add the UR3e Robotiq85 variant."""
    robot = env.robots[0]
    right_joint_names = getattr(robot, "gripper_joints", {}).get("right", [])
    panda_joint_names = {
        "gripper0_right_finger_joint1",
        "gripper0_right_finger_joint2",
    }
    if panda_joint_names.issubset(set(right_joint_names)):
        return _ORIGINAL_CHECK_OBJ_GRASPED(env, obj_name, threshold=threshold)
    return _robotiq_obj_grasped(env, obj_name, threshold)


def _install_compatible_grasp_check() -> None:
    """Patch only this evaluator process; do not modify RoboCasa on disk."""
    object_utils.check_obj_grasped = _check_obj_grasped_compatible


def _first(value: Any) -> np.ndarray:
    array = np.asarray(value)
    while array.ndim > 1 and array.shape[0] == 1:
        array = array[0]
    return np.asarray(array)


def _grip_site_pose(env: Any) -> tuple[np.ndarray, np.ndarray]:
    robot = env.robots[0]
    site_id = robot.eef_site_id["right"]
    position = np.asarray(env.sim.data.site_xpos[site_id], dtype=np.float64).copy()
    rotation = np.asarray(env.sim.data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
    quaternion = T.mat2quat(rotation)
    quaternion /= max(np.linalg.norm(quaternion), 1e-12)
    return position, quaternion


def _gripper_summary(raw: dict[str, Any]) -> dict[str, Any]:
    values = _first(raw["robot0_gripper_qpos"]).astype(np.float64).reshape(-1)
    return {
        "qpos": values.tolist(),
        "mean": float(np.mean(values)) if values.size else 0.0,
        "min": float(np.min(values)) if values.size else 0.0,
        "max": float(np.max(values)) if values.size else 0.0,
    }


def _geom_id_with_suffix(env: Any, suffixes: tuple[str, ...]) -> int | None:
    """Find a merged-model geom by its raw gripper suffix."""
    for geom_id in range(int(env.sim.model.ngeom)):
        name = env.sim.model.geom_id2name(geom_id) or ""
        if any(name.endswith(suffix) for suffix in suffixes):
            return geom_id
    return None


def _gripper_pad_geometry(env: Any) -> dict[str, Any]:
    """Return finger-pad centers and separation in world coordinates."""
    left_id = _geom_id_with_suffix(
        env,
        ("finger1_pad_collision", "left_fingerpad_collision"),
    )
    right_id = _geom_id_with_suffix(
        env,
        ("finger2_pad_collision", "right_fingerpad_collision"),
    )
    if left_id is None or right_id is None:
        return {
            "available": False,
            "left_pad_geom": None,
            "right_pad_geom": None,
            "left_pad_pos_world": None,
            "right_pad_pos_world": None,
            "midpoint_world": None,
            "separation_m": None,
        }
    left = np.asarray(env.sim.data.geom_xpos[left_id], dtype=np.float64).copy()
    right = np.asarray(env.sim.data.geom_xpos[right_id], dtype=np.float64).copy()
    return {
        "available": True,
        "left_pad_geom": env.sim.model.geom_id2name(left_id),
        "right_pad_geom": env.sim.model.geom_id2name(right_id),
        "left_pad_pos_world": left.tolist(),
        "right_pad_pos_world": right.tolist(),
        "midpoint_world": ((left + right) / 2.0).tolist(),
        "separation_m": float(np.linalg.norm(left - right)),
    }


def _gripper_pad_local_offset(env: Any) -> np.ndarray | None:
    """Return the pad-midpoint offset expressed in grip-site coordinates."""
    pad_geometry = _gripper_pad_geometry(env)
    midpoint = pad_geometry.get("midpoint_world")
    if midpoint is None:
        return None
    site_position, site_quaternion = _grip_site_pose(env)
    site_rotation = T.quat2mat(site_quaternion)
    return site_rotation.T.dot(
        np.asarray(midpoint, dtype=np.float64) - site_position
    )


def _target_site_for_source_pad_midpoint(
    source_row: dict[str, Any],
    target_quaternion: np.ndarray,
    target_pad_local_offset: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Shift a UR3e site target so its pad midpoint matches the source pad."""
    nominal_position = np.asarray(
        source_row["grip_site_pos_world"], dtype=np.float64
    )
    source_pad = source_row.get("grip_pad_geometry") or {}
    source_midpoint = source_pad.get("midpoint_world")
    if source_midpoint is None or target_pad_local_offset is None:
        return nominal_position, None
    target_rotation = T.quat2mat(
        np.asarray(target_quaternion, dtype=np.float64)
        / max(np.linalg.norm(target_quaternion), 1e-12)
    )
    adjusted = np.asarray(source_midpoint, dtype=np.float64) - target_rotation.dot(
        np.asarray(target_pad_local_offset, dtype=np.float64)
    )
    return adjusted, adjusted - nominal_position


def _set_ur3e_gripper_signal(env: Any, signal: float) -> float:
    """Set the Robotiq abstract signal as an absolute per-frame target.

    RoboSuite's Robotiq85 GRIP input is stateful: its ``format_action`` adds
    +/- ``speed`` to an internal signal. Setting that internal signal before
    sending a zero action gives the controller an absolute target, avoiding
    the incorrect direct reuse of Panda's binary command.
    """
    gripper = env.robots[0].gripper["right"]
    target = float(np.clip(signal, -1.0, 1.0))
    gripper.current_action = np.asarray([target], dtype=np.float64)
    return 0.0


def _teleport_link_gripper_collisions(
    env: Any,
    *,
    max_penetration_m: float,
) -> list[dict[str, Any]]:
    """Return deep UR3e-link / Robotiq-gripper contacts at the current qpos.

    Robot-object contact is deliberately excluded: the Robotiq pads must
    contact the sponge during a valid grasp. The only contacts rejected here
    are non-adjacent UR3e link geometry penetrating the mounted gripper.
    """
    violations: list[dict[str, Any]] = []
    for contact_index in range(int(env.sim.data.ncon)):
        contact = env.sim.data.contact[contact_index]
        geom1 = env.sim.model.geom_id2name(int(contact.geom1)) or ""
        geom2 = env.sim.model.geom_id2name(int(contact.geom2)) or ""
        is_link_gripper_pair = (
            (geom1.startswith("robot0_") and geom2.startswith("gripper0_"))
            or (geom2.startswith("robot0_") and geom1.startswith("gripper0_"))
        )
        penetration = max(0.0, -float(contact.dist))
        if is_link_gripper_pair and penetration > max_penetration_m:
            violations.append(
                {
                    "geom1": geom1,
                    "geom2": geom2,
                    "dist": float(contact.dist),
                    "penetration_m": penetration,
                }
            )
    return violations


def _robot_body_object_penetrations(
    env: Any,
    body_names: list[str],
) -> dict[tuple[str, str], float]:
    """Return robot/object penetration by canonical contact-geom pair."""
    wanted_body_ids: set[int] = set()
    for body_name in body_names:
        try:
            wanted_body_ids.add(int(env.sim.model.body_name2id(body_name)))
        except Exception:
            continue

    penetrations: dict[tuple[str, str], float] = {}
    for contact_index in range(int(env.sim.data.ncon)):
        contact = env.sim.data.contact[contact_index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        body1 = int(env.sim.model.geom_bodyid[geom1])
        body2 = int(env.sim.model.geom_bodyid[geom2])
        if body1 in wanted_body_ids:
            object_geom, robot_geom = geom1, geom2
        elif body2 in wanted_body_ids:
            object_geom, robot_geom = geom2, geom1
        else:
            continue

        robot_body = int(env.sim.model.geom_bodyid[robot_geom])
        robot_body_name = env.sim.model.body_id2name(robot_body) or ""
        if not (
            robot_body_name.startswith("robot0_")
            or robot_body_name.startswith("gripper0_")
        ):
            continue

        object_geom_name = env.sim.model.geom_id2name(object_geom) or ""
        robot_geom_name = env.sim.model.geom_id2name(robot_geom) or ""
        pair = tuple(sorted((object_geom_name, robot_geom_name)))
        penetration = max(0.0, -float(contact.dist))
        penetrations[pair] = max(penetrations.get(pair, 0.0), penetration)
    return penetrations


def _teleport_scene_collision_violations(
    env: Any,
    *,
    body_names: list[str],
    before_penetrations: dict[tuple[str, str], float],
    max_penetration_m: float,
    max_penetration_increase_m: float,
) -> list[dict[str, Any]]:
    """Reject an IK teleport that creates a deep new robot/object contact."""
    after_penetrations = _robot_body_object_penetrations(env, body_names)
    violations: list[dict[str, Any]] = []
    for pair, after in after_penetrations.items():
        before = float(before_penetrations.get(pair, 0.0))
        increase = after - before
        # Finger-pad contact is allowed for grasping, but a large new
        # penetration is still rejected because it produces an impulse.
        if (
            after > max_penetration_m
            and increase > max_penetration_increase_m
        ):
            violations.append(
                {
                    "geom_pair": list(pair),
                    "before_penetration_m": before,
                    "after_penetration_m": after,
                    "increase_m": increase,
                }
            )
    return violations


def _apply_joint_target_without_arm_dynamics(
    env: Any,
    q_target: np.ndarray,
    *,
    max_self_penetration_m: float,
    monitored_body_names: list[str] | None = None,
    max_allowed_penetration_m: float = 0.0015,
    max_penetration_increase_m: float = 0.00025,
) -> dict[str, Any]:
    """Place the UR3e arm at an IK target before the next physics step.

    This is an intentional diagnostic mode for separating pose conversion
    error from finite-rate arm tracking.  It does not advance simulation time
    and does not alter the gripper qpos; the following ``env.step`` still
    advances physics and lets the Robotiq fingers contact the object. A target
    that deeply intersects a UR3e link with the mounted Robotiq gripper is
    restored and rejected before it can become a rendered / physical step.
    """
    robot = env.robots[0]
    qpos_indexes = np.asarray(robot._ref_arm_joint_pos_indexes, dtype=np.int64)
    qvel_indexes = np.asarray(robot._ref_arm_joint_vel_indexes, dtype=np.int64)
    q_target = np.asarray(q_target, dtype=np.float64).reshape(6)
    original_qpos = np.asarray(env.sim.data.qpos[qpos_indexes], dtype=np.float64).copy()
    original_qvel = np.asarray(env.sim.data.qvel[qvel_indexes], dtype=np.float64).copy()
    before_scene_penetrations = _robot_body_object_penetrations(
        env,
        list(monitored_body_names or []),
    )
    env.sim.data.qpos[qpos_indexes] = q_target
    env.sim.data.qvel[qvel_indexes] = 0.0
    env.sim.forward()
    self_violations = _teleport_link_gripper_collisions(
        env,
        max_penetration_m=max_self_penetration_m,
    )
    scene_violations = _teleport_scene_collision_violations(
        env,
        body_names=list(monitored_body_names or []),
        before_penetrations=before_scene_penetrations,
        max_penetration_m=max_allowed_penetration_m,
        max_penetration_increase_m=max_penetration_increase_m,
    )
    violations = self_violations + scene_violations
    if violations:
        env.sim.data.qpos[qpos_indexes] = original_qpos
        env.sim.data.qvel[qvel_indexes] = original_qvel
        env.sim.forward()
        robot.composite_controller.update_state()
        return {
            "applied": False,
            "self_collision_violations": self_violations,
            "scene_collision_violations": scene_violations,
            "max_self_penetration_m": float(max_self_penetration_m),
            "max_allowed_penetration_m": float(max_allowed_penetration_m),
            "max_penetration_increase_m": float(max_penetration_increase_m),
        }
    robot.composite_controller.update_state()
    return {
        "applied": True,
        "self_collision_violations": [],
        "scene_collision_violations": [],
        "max_self_penetration_m": float(max_self_penetration_m),
        "max_allowed_penetration_m": float(max_allowed_penetration_m),
        "max_penetration_increase_m": float(max_penetration_increase_m),
    }


def _panda_gripper_opening_m(row: dict[str, Any]) -> float | None:
    """Return Panda finger separation from the two slide-joint positions."""
    values = np.asarray(
        row.get("gripper", {}).get("qpos", []),
        dtype=np.float64,
    ).reshape(-1)
    if values.size < 2:
        return None
    # Panda finger_joint2 has the opposite sign convention.
    return float(np.clip(values[0] - values[1], 0.0, 0.08))


def _source_gripper_mapping(rows: list[dict[str, Any]]) -> dict[str, Any]:
    openings = np.asarray(
        [
            row["gripper_opening_m"]
            for row in rows
            if row.get("gripper_opening_m") is not None
        ],
        dtype=np.float64,
    )
    if openings.size == 0:
        return {
            "enabled": False,
            "reason": "source Panda finger joint positions unavailable",
        }
    # Panda's two slide joints have ranges [0, 0.04] and [-0.04, 0].
    # Use the physical range instead of trajectory extrema, so an episode
    # starting half-open is not incorrectly treated as fully closed.
    closed_opening = 0.0
    open_opening = 0.08
    return {
        "enabled": True,
        "source_closed_opening_m": closed_opening,
        "source_open_opening_m": open_opening,
        "source_observed_min_opening_m": float(np.min(openings)),
        "source_observed_max_opening_m": float(np.max(openings)),
        "mapping": (
            "Panda finger_joint1 - finger_joint2 opening in [0, 0.08] m "
            "-> absolute Robotiq signal [-1=open, +1=close]"
        ),
    }


def _mapped_gripper_signal(
    row: dict[str, Any],
    mapping: dict[str, Any],
) -> float | None:
    # Accept replay files written before the Panda-joint opening mapping was
    # introduced. Those files used source finger-pad distance extrema.
    if (
        mapping.get("enabled", False)
        and "source_closed_opening_m" not in mapping
        and "source_open_separation_m" in mapping
    ):
        return None
    opening = row.get("gripper_opening_m")
    if opening is None:
        opening = _panda_gripper_opening_m(row)
    if opening is None or not mapping.get("enabled", False):
        return None
    closed_opening = float(mapping["source_closed_opening_m"])
    open_opening = float(mapping["source_open_opening_m"])
    opening_fraction = np.clip(
        (float(opening) - closed_opening)
        / max(open_opening - closed_opening, 1e-6),
        0.0,
        1.0,
    )
    closure = 1.0 - opening_fraction
    return float(2.0 * closure - 1.0)


def _quat_error_rad(target: np.ndarray, actual: np.ndarray) -> float:
    return float(
        np.linalg.norm(
            T.get_orientation_error(
                np.asarray(target, dtype=np.float64),
                np.asarray(actual, dtype=np.float64),
            )
        )
    )


def _solve_grip_site_ik(
    env: Any,
    target_position: np.ndarray,
    target_quaternion: np.ndarray,
    *,
    initial_q: np.ndarray | None = None,
    max_iterations: int = 80,
    damping: float = 0.04,
    step_scale: float = 0.8,
    max_joint_step: float = 0.15,
    max_total_joint_step: float | None = None,
    position_tolerance: float = 0.0015,
    orientation_tolerance: float = 0.02,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve a grip-site pose into a UR3e arm joint target.

    The simulator state is used only for forward kinematics and Jacobians.
    The original state is restored before returning, so the solved qpos is
    later applied through the normal joint-position controller.
    """
    robot = env.robots[0]
    site_id = robot.eef_site_id["right"]
    qpos_indexes = np.asarray(robot._ref_arm_joint_pos_indexes, dtype=np.int64)
    qvel_indexes = np.asarray(robot._ref_arm_joint_vel_indexes, dtype=np.int64)
    joint_indexes = np.asarray(robot._ref_arm_joint_indexes, dtype=np.int64)
    if qpos_indexes.size != 6 or qvel_indexes.size != 6:
        raise RuntimeError(
            f"Expected a 6-DoF UR3e arm, got qpos={qpos_indexes}, qvel={qvel_indexes}"
        )

    target_position = np.asarray(target_position, dtype=np.float64)
    target_quaternion = np.asarray(target_quaternion, dtype=np.float64)
    target_quaternion /= max(np.linalg.norm(target_quaternion), 1e-12)

    original_qpos = np.asarray(env.sim.data.qpos, dtype=np.float64).copy()
    original_qvel = np.asarray(env.sim.data.qvel, dtype=np.float64).copy()
    original_qacc = np.asarray(env.sim.data.qacc, dtype=np.float64).copy()
    current_q = np.asarray(env.sim.data.qpos[qpos_indexes], dtype=np.float64).copy()
    q = (
        current_q.copy()
        if initial_q is None
        else np.asarray(initial_q, dtype=np.float64).reshape(6).copy()
    )
    q_reference = q.copy()
    joint_types = np.asarray(env.sim.model.jnt_type[joint_indexes], dtype=np.int64)
    converged = False
    iterations = 0
    total_step_limited = False
    best_q = q.copy()
    best_score = float("inf")
    best_position_error = float("inf")
    best_orientation_error = float("inf")

    def errors() -> tuple[np.ndarray, np.ndarray]:
        position, quaternion = _grip_site_pose(env)
        return (
            target_position - position,
            T.get_orientation_error(target_quaternion, quaternion),
        )

    def unwrap_hinge_joints(values: np.ndarray) -> np.ndarray:
        """Keep equivalent hinge angles on the branch nearest the current q."""
        values = np.asarray(values, dtype=np.float64).copy()
        hinge_mask = joint_types == 3
        values[hinge_mask] += 2.0 * np.pi * np.round(
            (q_reference[hinge_mask] - values[hinge_mask]) / (2.0 * np.pi)
        )
        return values

    try:
        def update_best() -> None:
            nonlocal best_q, best_score
            nonlocal best_position_error, best_orientation_error
            position_error, orientation_error = errors()
            score = float(
                np.linalg.norm(position_error)
                / max(position_tolerance, 1e-6)
                + np.linalg.norm(orientation_error)
                / max(orientation_tolerance, 1e-6)
            )
            if score < best_score:
                best_score = score
                best_q = q.copy()
                best_position_error = float(np.linalg.norm(position_error))
                best_orientation_error = float(np.linalg.norm(orientation_error))

        env.sim.data.qpos[qpos_indexes] = q
        env.sim.data.qvel[qvel_indexes] = 0.0
        env.sim.data.qacc[qvel_indexes] = 0.0
        env.sim.forward()
        update_best()
        for iterations in range(1, max_iterations + 1):
            position_error, orientation_error = errors()
            if (
                np.linalg.norm(position_error) <= position_tolerance
                and np.linalg.norm(orientation_error) <= orientation_tolerance
            ):
                converged = True
                break

            jacobian_position = env.sim.data.get_site_jacp(
                env.sim.model.site_id2name(site_id)
            ).reshape((3, -1))[:, qvel_indexes]
            jacobian_orientation = env.sim.data.get_site_jacr(
                env.sim.model.site_id2name(site_id)
            ).reshape((3, -1))[:, qvel_indexes]
            jacobian = np.vstack((jacobian_position, jacobian_orientation))
            error = np.concatenate((position_error, orientation_error))
            lhs = jacobian.T @ jacobian + (damping**2) * np.eye(6)
            dq = np.linalg.solve(lhs, jacobian.T @ error) * step_scale
            dq_norm = np.linalg.norm(dq)
            if dq_norm > max_joint_step:
                dq *= max_joint_step / dq_norm

            q = unwrap_hinge_joints(q + dq)
            if (
                max_total_joint_step is not None
                and np.linalg.norm(q - original_qpos[qpos_indexes]) > max_total_joint_step
            ):
                direction = q - original_qpos[qpos_indexes]
                direction_norm = np.linalg.norm(direction)
                if direction_norm > 1e-12:
                    q = original_qpos[qpos_indexes] + (
                        direction * (max_total_joint_step / direction_norm)
                    )
                total_step_limited = True
                break
            ranges = np.asarray(
                env.sim.model.jnt_range[joint_indexes],
                dtype=np.float64,
            )
            for index, (lower, upper) in enumerate(ranges):
                if lower < upper:
                    q[index] = np.clip(q[index], lower + 1e-5, upper - 1e-5)
            env.sim.data.qpos[qpos_indexes] = q
            env.sim.data.qvel[qvel_indexes] = 0.0
            env.sim.data.qacc[qvel_indexes] = 0.0
            env.sim.forward()
            update_best()

        q = best_q.copy()
        env.sim.data.qpos[qpos_indexes] = q
        env.sim.data.qvel[qvel_indexes] = 0.0
        env.sim.data.qacc[qvel_indexes] = 0.0
        env.sim.forward()
        final_position_error, final_orientation_error = errors()
        result = {
            "iterations": iterations,
            "converged": bool(
                converged
                or (
                    np.linalg.norm(final_position_error) <= position_tolerance
                    and np.linalg.norm(final_orientation_error)
                    <= orientation_tolerance
                )
            ),
            "final_position_error_m": float(np.linalg.norm(final_position_error)),
            "final_orientation_error_rad": float(
                np.linalg.norm(final_orientation_error)
            ),
            "total_joint_step_limited": bool(total_step_limited),
            "total_joint_step_rad": float(
                np.linalg.norm(q - q_reference)
            ),
            "initial_q_source": "previous_target" if initial_q is not None else "actual_q",
            "best_score": float(best_score),
        }
        return unwrap_hinge_joints(q), result
    finally:
        env.sim.data.qpos[:] = original_qpos
        env.sim.data.qvel[:] = original_qvel
        env.sim.data.qacc[:] = original_qacc
        env.sim.forward()


def _eef_delta_action(
    env: Any,
    target_position: np.ndarray,
    target_quaternion: np.ndarray,
    gripper_action: float,
    *,
    max_position_command_m: float = 0.025,
    max_orientation_command_rad: float = 0.25,
) -> np.ndarray:
    """Convert a world-frame grip-site target into a bounded OSC delta.

    The old replay solved IK by writing ``sim.data.qpos`` directly. That
    bypassed the arm controller and could teleport collision geometry into the
    pan. This function keeps the normal RoboSuite OSC controller in the loop.
    """
    robot = env.robots[0]
    controller = robot.composite_controller.get_controller("right")
    _, origin_rotation = (
        robot.composite_controller.get_controller_base_pose("right")
    )

    current_grip_position, current_grip_quaternion = _grip_site_pose(env)
    current_grip_rotation = T.quat2mat(current_grip_quaternion)
    target_rotation = T.quat2mat(
        np.asarray(target_quaternion, dtype=np.float64)
        / max(np.linalg.norm(target_quaternion), 1e-12)
    )
    origin_rotation = np.asarray(origin_rotation, dtype=np.float64).reshape(3, 3)

    # RoboSuite's OSC ``ref_name`` is ``gripper0_right_grip_site``. The
    # controller regulates that site's pose directly; ``right_center`` is
    # only the origin used to express the delta action. Therefore both the
    # current pose and the recorded target must stay in grip-site coordinates.
    delta_position_base = origin_rotation.T.dot(
        np.asarray(target_position, dtype=np.float64) - current_grip_position
    )
    current_rotation_base = origin_rotation.T.dot(current_grip_rotation)
    target_rotation_base = origin_rotation.T.dot(target_rotation)
    delta_rotation_base = target_rotation_base.dot(current_rotation_base.T)
    delta_quaternion = T.mat2quat(delta_rotation_base)
    if delta_quaternion[3] < 0:
        delta_quaternion = -delta_quaternion
    delta_orientation_base = T.quat2axisangle(delta_quaternion)

    controller_position_limit = np.asarray(
        controller.output_max[:3], dtype=np.float64
    )
    controller_orientation_limit = np.asarray(
        controller.output_max[3:6], dtype=np.float64
    )
    position_limit = np.minimum(
        controller_position_limit,
        float(max_position_command_m),
    )
    orientation_limit = np.minimum(
        controller_orientation_limit,
        float(max_orientation_command_rad),
    )
    position_limit = np.maximum(position_limit, 1e-6)
    orientation_limit = np.maximum(orientation_limit, 1e-6)

    arm_action = np.concatenate(
        (
            np.clip(
                delta_position_base / controller_position_limit,
                -position_limit / controller_position_limit,
                position_limit / controller_position_limit,
            ),
            np.clip(
                delta_orientation_base / controller_orientation_limit,
                -orientation_limit / controller_orientation_limit,
                orientation_limit / controller_orientation_limit,
            ),
        )
    )
    return np.concatenate(
        (arm_action.astype(np.float32), np.asarray([gripper_action], dtype=np.float32))
    )


def _body_position(env: Any, body_name: str) -> np.ndarray | None:
    try:
        return np.asarray(
            env.sim.data.get_body_xpos(body_name), dtype=np.float64
        ).copy()
    except Exception:
        return None


def _resolve_monitored_body_names(
    env: Any,
    primary_body_name: str | None = None,
) -> list[str]:
    """Return movable object bodies whose explosive motion must be guarded."""
    names: list[str] = []
    if primary_body_name:
        names.append(primary_body_name)
    object_ids = getattr(env, "obj_body_id", {}) or {}
    for key in ("cutting_board", "sponge", "obj", "object", "pan", "cup", "glass"):
        body_id = object_ids.get(key)
        if body_id is None:
            continue
        try:
            name = str(env.sim.model.body_id2name(int(body_id)))
        except Exception:
            continue
        if name and name not in names:
            names.append(name)
    return names


def _gripper_arm_qpos(env: Any) -> np.ndarray:
    """Read the six Robotiq joint coordinates used by the merged model."""
    indexes = np.asarray(
        env.robots[0]._ref_gripper_joint_pos_indexes["right"],
        dtype=np.int64,
    )
    return np.asarray(env.sim.data.qpos[indexes], dtype=np.float64).copy()


def _gripper_joint_limit_violation(
    env: Any,
    qpos: np.ndarray,
    *,
    tolerance_rad: float = 0.02,
) -> float:
    """Return the largest Robotiq joint-limit violation in radians.

    Use the exact MuJoCo qpos address for each named joint. Robotiq85 contains
    passive tendon-driven joints, so relying on an implicit flat ordering can
    compare a joint value against a different joint's range.
    """
    joint_names = list(
        getattr(env.robots[0], "gripper_joints", {}).get("right", [])
    )
    qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
    qpos_by_address: dict[int, float] = {}
    qpos_indexes = np.asarray(
        env.robots[0]._ref_gripper_joint_pos_indexes["right"],
        dtype=np.int64,
    ).reshape(-1)
    for index, address in enumerate(qpos_indexes):
        if index < qpos.size:
            qpos_by_address[int(address)] = float(qpos[index])

    violations: list[float] = []
    actuated_joint_ids = {
        int(joint_id)
        for joint_id in np.asarray(env.sim.model.actuator_trnid)[:, 0]
        if int(joint_id) >= 0
    }
    for name in joint_names:
        try:
            joint_id = int(env.sim.model.joint_name2id(name))
            qadr = env.sim.model.get_joint_qpos_addr(name)
        except Exception:
            continue
        # The four inner finger / knuckle joints are tendon followers. Their
        # coordinates are not independently position-actuated and may move
        # slightly outside their standalone XML range under contact.
        if joint_id not in actuated_joint_ids:
            continue
        if isinstance(qadr, (tuple, list, np.ndarray)):
            addresses = np.asarray(qadr, dtype=np.int64).reshape(-1)
        else:
            addresses = np.asarray([int(qadr)], dtype=np.int64)
        if addresses.size != 1 or int(addresses[0]) not in qpos_by_address:
            continue
        value = qpos_by_address[int(addresses[0])]
        lower, upper = np.asarray(
            env.sim.model.jnt_range[joint_id],
            dtype=np.float64,
        )
        if lower >= upper:
            continue
        violations.append(
            max(
                0.0,
                float(lower - value - tolerance_rad),
                float(value - upper - tolerance_rad),
            )
        )
    return max(violations, default=0.0)


def _safe_env_step(
    env: Any,
    action: np.ndarray,
    *,
    object_body_name: str | None = None,
    monitored_body_names: list[str] | None = None,
    max_object_step_m: float = 0.01,
    max_gripper_qpos_step_rad: float = 0.35,
    max_gripper_qvel_rad_s: float = 100.0,
    max_gripper_limit_violation_rad: float = 0.02,
    rollback_state: np.ndarray | None = None,
) -> tuple[dict[str, Any], Any, bool, dict[str, Any]]:
    """Advance one action and rollback numerically explosive contacts."""
    saved_state = (
        np.asarray(rollback_state, dtype=np.float64).copy()
        if rollback_state is not None
        else env.sim.get_state().flatten()
    )
    body_names = list(monitored_body_names or [])
    if object_body_name and object_body_name not in body_names:
        body_names.insert(0, object_body_name)
    before_positions = {
        name: _body_position(env, name)
        for name in body_names
    }
    before_gripper_qpos = _gripper_arm_qpos(env)
    before_gripper_action = np.asarray(
        env.robots[0].gripper["right"].current_action,
        dtype=np.float64,
    ).copy()
    raw, reward, done, info = env.step(action)
    after_positions = {
        name: _body_position(env, name)
        for name in body_names
    }
    body_steps = {
        name: float(np.linalg.norm(after_positions[name] - before_positions[name]))
        for name in body_names
        if before_positions[name] is not None and after_positions[name] is not None
    }
    object_step_m = float(body_steps.get(object_body_name, 0.0))
    after_gripper_qpos = _gripper_arm_qpos(env)
    gripper_qpos_step = float(
        np.linalg.norm(after_gripper_qpos - before_gripper_qpos)
    )
    gripper_qvel_indexes = np.asarray(
        env.robots[0]._ref_gripper_joint_vel_indexes["right"],
        dtype=np.int64,
    )
    gripper_qvel = np.asarray(
        env.sim.data.qvel[gripper_qvel_indexes],
        dtype=np.float64,
    )
    gripper_limit_violation = _gripper_joint_limit_violation(
        env,
        after_gripper_qpos,
        tolerance_rad=max_gripper_limit_violation_rad,
    )
    finite_state = bool(
        np.all(np.isfinite(env.sim.data.qpos))
        and np.all(np.isfinite(env.sim.data.qvel))
    )
    body_violations = {
        name: step
        for name, step in body_steps.items()
        if step > max_object_step_m
    }
    rollback_reasons: list[str] = []
    if body_violations:
        rollback_reasons.append("movable_body_step")
    if not finite_state:
        rollback_reasons.append("nonfinite_sim_state")
    if gripper_qpos_step > max_gripper_qpos_step_rad:
        rollback_reasons.append("gripper_qpos_jump")
    if np.max(np.abs(gripper_qvel), initial=0.0) > max_gripper_qvel_rad_s:
        rollback_reasons.append("gripper_qvel_spike")
    if gripper_limit_violation > 0.0:
        rollback_reasons.append("gripper_joint_limit")
    rolled_back = bool(rollback_reasons)
    if rolled_back:
        env.sim.set_state_from_flattened(saved_state)
        env.sim.forward()
        robot = env.robots[0]
        robot.gripper["right"].current_action = before_gripper_action
        robot.composite_controller.update_state()
        robot.composite_controller.reset()
        raw = env._get_observations(force_update=True)
        info = dict(info or {})
        info["physics_rollback"] = True
        info["object_step_m_before_rollback"] = object_step_m
        info["monitored_body_steps_m"] = body_steps
        info["movable_body_violations_m"] = body_violations
        info["gripper_qpos_step_rad"] = gripper_qpos_step
        info["gripper_qvel_max_rad_s"] = float(
            np.max(np.abs(gripper_qvel), initial=0.0)
        )
        info["gripper_limit_violation_rad"] = gripper_limit_violation
        info["physics_rollback_reasons"] = rollback_reasons
    else:
        info = dict(info or {})
        info["physics_rollback"] = False
        info["object_step_m"] = object_step_m
        info["monitored_body_steps_m"] = body_steps
        info["gripper_qpos_step_rad"] = gripper_qpos_step
        info["gripper_qvel_max_rad_s"] = float(
            np.max(np.abs(gripper_qvel), initial=0.0)
        )
        info["gripper_limit_violation_rad"] = gripper_limit_violation
    return raw, reward, done, info


def _render(env: Any, camera_name: str, width: int, height: int) -> np.ndarray:
    # Render at the dimensions used when the RoboSuite offscreen context was
    # created, then enlarge only the encoded/comparison image. This avoids
    # repeatedly resizing an OSMesa framebuffer during a long rollout.
    render_width = int(getattr(env, "_replay_camera_width", width))
    render_height = int(getattr(env, "_replay_camera_height", height))
    frame = np.asarray(
        env.sim.render(
            height=render_height,
            width=render_width,
            camera_name=camera_name,
        ),
        dtype=np.uint8,
    )
    return _resize_frame(frame[::-1, :, :].copy(), width, height)


def _make_writer(path: Path, fps: int = 20):
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    )


class _NullWriter:
    def append_data(self, frame: np.ndarray) -> None:
        del frame

    def close(self) -> None:
        pass


def _writer(path: Path, args: argparse.Namespace):
    return _NullWriter() if args.no_video else _make_writer(path)


def _episode_parquet(dataset_root: Path, episode_index: int) -> Path:
    matches = sorted(
        dataset_root.glob(f"data/*/episode_{episode_index:06d}.parquet")
    )
    if not matches:
        raise FileNotFoundError(
            f"No parquet found for episode {episode_index} under {dataset_root}"
        )
    return matches[0]


def _load_expert(dataset_root: Path, episode_index: int) -> dict[str, Any]:
    import pandas as pd

    episode_dir = dataset_root / "extras" / f"episode_{episode_index:06d}"
    states_path = episode_dir / "states.npz"
    xml_path = episode_dir / "model.xml.gz"
    meta_path = episode_dir / "ep_meta.json"
    parquet_path = _episode_parquet(dataset_root, episode_index)
    for path in (states_path, xml_path, meta_path, parquet_path):
        if not path.exists():
            raise FileNotFoundError(path)

    states = np.load(states_path)["states"].astype(np.float64)
    with gzip.open(xml_path, "rt", encoding="utf-8") as handle:
        model_xml = handle.read()
    ep_meta = json.loads(meta_path.read_text(encoding="utf-8"))

    frame_table = pd.read_parquet(parquet_path)
    lerobot_actions = np.stack(frame_table["action"].to_list()).astype(np.float32)
    actions = reorder_lerobot_action(lerobot_actions, dataset_root).astype(np.float32)
    if states.shape[0] != actions.shape[0]:
        raise ValueError(
            f"Expert state/action length mismatch: states={states.shape}, "
            f"actions={actions.shape}"
        )
    if actions.shape[1] != 12:
        raise ValueError(f"Expected 12D RoboCasa actions, got {actions.shape}")

    return {
        "states": states,
        "model_xml": model_xml,
        "ep_meta": ep_meta,
        "actions": actions,
        "lerobot_actions": lerobot_actions,
        "frame_table": frame_table,
        "parquet_path": str(parquet_path),
    }


def _make_env(
    task: str,
    robot: str,
    ep_meta: dict[str, Any],
    seed: int,
    camera_width: int,
    camera_height: int,
    render: bool = True,
) -> Any:
    reset_meta = dict(ep_meta)
    if robot != "PandaOmron":
        # Kitchen._reset_internal restores these fields through the mobile
        # base joints. UR3e has no mobile base; its fixed base is positioned by
        # register_ur3e() and the subsequent grip-site IK alignment.
        reset_meta.pop("init_robot_base_pos", None)
        reset_meta.pop("init_robot_base_ori", None)
    camera_names = (
        [
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ]
        if render
        else []
    )
    env = env_utils.create_env(
        task,
        robots=robot,
        split=None,
        obj_instance_split="target",
        layout_and_style_ids=[
            (int(ep_meta["layout_id"]), int(ep_meta["style_id"]))
        ],
        seed=seed,
        camera_names=camera_names,
        camera_widths=camera_width,
        camera_heights=camera_height,
        render_camera="robot0_agentview_left",
        control_freq=20,
        initialization_noise=None,
    )
    # set_ep_meta must happen before reset, so object configs and the original
    # robot base pose are used when RoboCasa constructs this episode.
    env.set_ep_meta(reset_meta)
    env.reset()
    env._replay_camera_width = int(camera_width)
    env._replay_camera_height = int(camera_height)
    return env


def _restore_expert_xml_state(
    env: Any,
    model_xml: str,
    state: np.ndarray,
) -> None:
    edited_xml = env.edit_model_xml(model_xml)
    env.reset_from_xml_string(edited_xml)
    if len(state) != len(env.sim.get_state().flatten()):
        raise ValueError(
            "Episode model/state mismatch after XML restore: "
            f"state={len(state)}, sim={len(env.sim.get_state().flatten())}"
        )
    env.sim.set_state_from_flattened(state)
    env.sim.forward()
    if hasattr(env, "update_sites"):
        env.update_sites()
    if hasattr(env, "update_state"):
        env.update_state()


def _joint_widths(sim: Any, joint_id: int) -> tuple[int, int]:
    joint_type = int(sim.model.jnt_type[joint_id])
    # MuJoCo: free=0, ball=1, slide=2, hinge=3.
    if joint_type == 0:
        return 7, 6
    if joint_type == 1:
        return 4, 3
    return 1, 1


def _extract_common_scene_state(
    source_env: Any,
    source_state: np.ndarray,
) -> dict[str, Any]:
    """Extract furniture/object joint values by name from the expert model."""
    source_sim = source_env.sim
    source_qpos_n = int(source_sim.model.nq)
    source_qpos = source_state[1 : 1 + source_qpos_n]
    source_qvel = source_state[1 + source_qpos_n : 1 + source_qpos_n + source_sim.model.nv]
    joints: dict[str, dict[str, list[float]]] = {}
    for source_joint_id in range(source_sim.model.njnt):
        name = source_sim.model.joint_id2name(source_joint_id)
        if not name or name.startswith(("robot0_", "gripper0_", "mobilebase0_")):
            continue
        source_qadr = int(source_sim.model.jnt_qposadr[source_joint_id])
        source_vadr = int(source_sim.model.jnt_dofadr[source_joint_id])
        qwidth, vwidth = _joint_widths(source_sim, source_joint_id)
        joints[name] = {
            "qpos": source_qpos[source_qadr : source_qadr + qwidth].tolist(),
            "qvel": source_qvel[source_vadr : source_vadr + vwidth].tolist(),
        }
    return {"time": float(source_state[0]), "joints": joints}


def _apply_common_scene_state(
    target_env: Any,
    scene_state: dict[str, Any],
) -> dict[str, Any]:
    """Apply extracted furniture/object joint values to the UR3e model."""
    target_sim = target_env.sim
    copied: list[str] = []
    skipped: list[str] = []
    for name, values in scene_state["joints"].items():
        try:
            target_joint_id = target_sim.model.joint_name2id(name)
        except Exception:
            skipped.append(name)
            continue
        qwidth, vwidth = _joint_widths(target_sim, target_joint_id)
        qpos = np.asarray(values["qpos"], dtype=np.float64)
        qvel = np.asarray(values["qvel"], dtype=np.float64)
        if qpos.size != qwidth or qvel.size != vwidth:
            skipped.append(name)
            continue
        qadr = int(target_sim.model.jnt_qposadr[target_joint_id])
        vadr = int(target_sim.model.jnt_dofadr[target_joint_id])
        target_sim.data.qpos[qadr : qadr + qwidth] = qpos
        target_sim.data.qvel[vadr : vadr + vwidth] = qvel
        copied.append(name)

    target_sim.data.time = float(scene_state["time"])
    target_sim.forward()
    if hasattr(target_env, "update_state"):
        target_env.update_state()
    return {
        "copied_joint_count": len(copied),
        "copied_joint_names": copied,
        "skipped_joint_count": len(skipped),
        "skipped_joint_names": skipped,
    }


def _capture_selected(
    selected: set[int],
    step_index: int,
    env: Any,
    args: argparse.Namespace,
) -> dict[str, np.ndarray] | None:
    if step_index not in selected:
        return None
    return {
        "agentview_left": _render(
            env,
            "robot0_agentview_left",
            args.video_width,
            args.video_height,
        ),
        "eye_in_hand": _render(
            env,
            "robot0_eye_in_hand",
            args.video_width,
            args.video_height,
        ),
    }


def _set_recorded_state(env: Any, state: np.ndarray) -> None:
    """Load one recorded state and refresh derived RoboCasa state."""
    env.sim.set_state_from_flattened(np.asarray(state, dtype=np.float64))
    env.sim.forward()
    if hasattr(env, "update_sites"):
        env.update_sites()
    if hasattr(env, "update_state"):
        env.update_state()


def _resolve_primary_object_body_name(
    env: Any,
    requested_name: str | None = None,
) -> str | None:
    """Resolve the primary movable object body used by contact monitoring."""
    if requested_name:
        try:
            env.sim.model.body_name2id(requested_name)
        except Exception as exc:
            raise ValueError(
                f"Requested object body {requested_name!r} does not exist in "
                f"the current {type(env).__name__} model"
            ) from exc
        return requested_name

    object_ids = getattr(env, "obj_body_id", {}) or {}
    for key in ("obj", "object", "sponge", "pan", "cup", "glass"):
        if key not in object_ids:
            continue
        try:
            return str(env.sim.model.body_id2name(int(object_ids[key])))
        except Exception:
            pass

    # Composite tasks often expose semantic object names only (for example
    # ``sponge`` in ScrubCuttingBoard). Prefer the first registered movable
    # object before falling back to a body-name heuristic.
    for key in getattr(env, "objects", {}) or {}:
        if key not in object_ids:
            continue
        try:
            return str(env.sim.model.body_id2name(int(object_ids[key])))
        except Exception:
            pass

    for body_id in object_ids.values():
        try:
            body_name = str(env.sim.model.body_id2name(int(body_id)))
        except Exception:
            continue
        if body_name.startswith(("obj", "sponge", "pan", "cup", "glass")):
            return body_name
    return None


def _replay_franka_recorded_states(
    env: Any,
    expert: dict[str, Any],
    args: argparse.Namespace,
    selected: set[int],
) -> dict[str, Any]:
    """Render exact post-action expert states and expose them as targets.

    The converted dataset stores state ``i`` before action ``i``. For a
    synchronized comparison, action ``i`` is paired with recorded state
    ``i + 1`` on both robots. The initial state is retained separately for
    base / grip-site alignment.
    """
    states = expert["states"]
    actions = expert["actions"]
    frame_table = expert["frame_table"]
    rewards = np.asarray(frame_table["next.reward"], dtype=np.float64)
    dones = np.asarray(frame_table["next.done"], dtype=bool)
    writer = _writer(args.output_root / "franka_expert_replay.mp4", args)
    rows: list[dict[str, Any]] = []
    selected_frames: dict[int, dict[str, np.ndarray]] = {}
    env_success = False
    initial_state = states[0]
    _set_recorded_state(env, initial_state)
    initial_raw = env._get_observations(force_update=True)
    initial_position, initial_quaternion = _grip_site_pose(env)
    initial_gripper = _gripper_summary(initial_raw)
    initial_pad_geometry = _gripper_pad_geometry(env)
    try:
        for index, action in enumerate(actions):
            recorded_state_index = min(index + 1, len(states) - 1)
            state = states[recorded_state_index]
            _set_recorded_state(env, state)
            raw = env._get_observations(force_update=True)
            position, quaternion = _grip_site_pose(env)
            pad_geometry = _gripper_pad_geometry(env)
            state_success = bool(env._check_success())
            env_success = state_success or env_success
            if not args.no_render:
                writer.append_data(
                    _render(
                        env,
                        "robot0_agentview_left",
                        args.video_width,
                        args.video_height,
                    )
                )
                frame_pair = _capture_selected(selected, index, env, args)
                if frame_pair is not None:
                    selected_frames[index] = frame_pair
            rows.append(
                {
                    "step": index + 1,
                    "recorded_state_index": recorded_state_index,
                    "recorded_state_time": float(state[0]),
                    "action_hdf5_12d": action.tolist(),
                    "action_lerobot_12d": expert["lerobot_actions"][index].tolist(),
                    "grip_site_pos_world": position.tolist(),
                    "grip_site_quat_world_xyzw": quaternion.tolist(),
                    "gripper": _gripper_summary(raw),
                    "gripper_opening_m": _panda_gripper_opening_m(
                        {"gripper": _gripper_summary(raw)}
                    ),
                    "grip_pad_geometry": pad_geometry,
                    "dataset_reward": float(rewards[index]),
                    "dataset_done": bool(dones[index]),
                    "env_check_success": state_success,
                }
            )
    finally:
        writer.close()

    dataset_success = bool(np.any(dones))
    result = {
        "robot": "PandaOmron",
        "source_mode": "recorded_states",
        "target_pose_offset": 0,
        "steps": len(rows),
        "success": bool(dataset_success or env_success),
        "dataset_success": dataset_success,
        "env_check_success": bool(env_success),
        "dataset_done_indices": np.flatnonzero(dones).astype(int).tolist(),
        "dataset_reward_sum": float(np.sum(rewards)),
        "initial_grip_site_pos_world": initial_position.tolist(),
        "initial_grip_site_quat_world_xyzw": initial_quaternion.tolist(),
        "initial_gripper": initial_gripper,
        "initial_grip_pad_geometry": initial_pad_geometry,
        "gripper_mapping": None,
        "rows": rows,
    }
    result["gripper_mapping"] = _source_gripper_mapping(rows)
    (args.output_root / "franka_replay.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return {
        "result": result,
        "selected_frames": selected_frames,
        "target_pose_offset": 0,
    }


def _replay_franka(
    env: Any,
    expert: dict[str, Any],
    args: argparse.Namespace,
    selected: set[int],
) -> dict[str, Any]:
    import json

    actions = expert["actions"]
    states = expert["states"]
    writer = _writer(args.output_root / "franka_expert_replay.mp4", args)
    rows: list[dict[str, Any]] = []
    selected_frames: dict[int, dict[str, np.ndarray]] = {}
    source_state_errors: list[float] = []
    success = False
    try:
        initial_position, initial_quaternion = _grip_site_pose(env)
        initial_raw = env._get_observations(force_update=True)
        initial_gripper = _gripper_summary(initial_raw)
        for index, action in enumerate(actions):
            raw, _, done, _ = env.step(action)
            position, quaternion = _grip_site_pose(env)
            actual_state = env.sim.get_state().flatten()
            if index + 1 < len(states):
                state_error = float(np.linalg.norm(actual_state - states[index + 1]))
            else:
                # The converted LeRobot export stores one state per action and
                # drops the final post-action state. There is no exact target
                # state for the last action.
                state_error = None
            source_state_errors.append(state_error)
            if not args.no_render:
                writer.append_data(
                    _render(
                        env,
                        "robot0_agentview_left",
                        args.video_width,
                        args.video_height,
                    )
                )
                frame_pair = _capture_selected(selected, index, env, args)
                if frame_pair is not None:
                    selected_frames[index] = frame_pair
            rows.append(
                {
                    "step": index + 1,
                    "action_hdf5_12d": action.tolist(),
                    "action_lerobot_12d": expert["lerobot_actions"][index].tolist(),
                    "grip_site_pos_world": position.tolist(),
                    "grip_site_quat_world_xyzw": quaternion.tolist(),
                    "gripper": _gripper_summary(raw),
                    "state_error_to_next_recorded_state": state_error,
                    "done": bool(done),
                }
            )
            success = bool(env._check_success()) or success
    finally:
        writer.close()

    result = {
        "robot": "PandaOmron",
        "steps": len(rows),
        "success": bool(success),
        "initial_grip_site_pos_world": initial_position.tolist(),
        "initial_grip_site_quat_world_xyzw": initial_quaternion.tolist(),
        "initial_gripper": initial_gripper,
        "max_state_error": (
            max(error for error in source_state_errors if error is not None)
            if any(error is not None for error in source_state_errors)
            else None
        ),
        "mean_state_error": (
            float(np.mean([error for error in source_state_errors if error is not None]))
            if any(error is not None for error in source_state_errors)
            else None
        ),
        "rows": rows,
    }
    (args.output_root / "franka_replay.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return {"result": result, "selected_frames": selected_frames}


def _replay_ur3e(
    env: Any,
    source_env: Any,
    expert: dict[str, Any],
    franka: dict[str, Any],
    args: argparse.Namespace,
    selected: set[int],
) -> dict[str, Any]:
    import json

    writer = _writer(args.output_root / "ur3e_grip_site_ik_replay.mp4", args)
    rows: list[dict[str, Any]] = []
    selected_frames: dict[int, dict[str, np.ndarray]] = {}
    success = False
    previous_q_target: np.ndarray | None = None
    previous_gripper_action: float | None = None
    source_rows = franka["result"]["rows"]
    target_pose_offset = int(franka.get("target_pose_offset", 0))
    ur3e_body_collision = getattr(
        env,
        "_ur3e_body_collision_info",
        {"enabled": False, "reason": "not requested"},
    )
    object_body_name = _resolve_primary_object_body_name(
        env,
        args.object_body_name,
    )
    monitored_body_names = _resolve_monitored_body_names(
        env,
        object_body_name,
    )
    print(
        f"[ur3e-replay] collision monitor object body: "
        f"{object_body_name or 'disabled'}; monitored bodies: "
        f"{monitored_body_names or 'none'}",
        flush=True,
    )
    try:
        source_initial_position = np.asarray(
            franka["result"]["initial_grip_site_pos_world"], dtype=np.float64
        )
        source_initial_quaternion = np.asarray(
            franka["result"]["initial_grip_site_quat_world_xyzw"],
            dtype=np.float64,
        )
        # The imported Robotiq85 ``init_qpos`` is not consistent with this
        # merged XML: its active and passive tendon coordinates can start
        # outside their ranges. A zero vector is the XML-consistent open
        # configuration: both actuated joints and both fixed-tendon equations
        # are satisfied before the first physics step.
        gripper_joint_indexes = np.asarray(
            env.robots[0]._ref_gripper_joint_pos_indexes["right"],
            dtype=np.int64,
        )
        env.sim.data.qpos[gripper_joint_indexes] = 0.0
        env.sim.data.qvel[
            np.asarray(
                env.robots[0]._ref_gripper_joint_vel_indexes["right"],
                dtype=np.int64,
            )
        ] = 0.0
        env.sim.forward()
        env._replay_gripper_reference_qpos = _gripper_arm_qpos(env)
        env.robots[0].gripper["right"].current_action = np.asarray(
            [-1.0], dtype=np.float64
        )
        target_pad_local_offset = (
            _gripper_pad_local_offset(env)
            if args.align_to_pad_midpoint
            else None
        )
        initial_source_row = source_rows[0]
        if args.align_to_pad_midpoint:
            source_initial_position, initial_pad_alignment_offset = (
                _target_site_for_source_pad_midpoint(
                    initial_source_row,
                    source_initial_quaternion,
                    target_pad_local_offset,
                )
            )
        else:
            initial_pad_alignment_offset = None
        alignment = align_initial_eef_to(
            env,
            source_initial_position,
            source_initial_quaternion,
            allow_base_translation=True,
            base_translation_axes=(True, True, False),
            orientation_source="site",
            max_iterations=args.ik_iterations,
            position_tolerance=args.ik_position_tolerance,
            orientation_tolerance=args.ik_orientation_tolerance,
        )
        initial_link_gripper_violations = _teleport_link_gripper_collisions(
            env,
            max_penetration_m=args.teleport_max_self_penetration_m,
        )
        if initial_link_gripper_violations:
            raise RuntimeError(
                "Initial UR3e grip-site alignment creates a deep UR3e-link / "
                "Robotiq-gripper intersection: "
                f"{initial_link_gripper_violations}"
            )
        raw = env._get_observations(force_update=True)
        initial_position, initial_quaternion = _grip_site_pose(env)
        initial_gripper = _gripper_summary(raw)
        gripper_mapping = franka["result"].get("gripper_mapping") or {
            "enabled": False,
            "reason": "no source gripper mapping in Franka replay JSON",
        }
        initial_ur3e_pad_geometry = _gripper_pad_geometry(env)

        for index, source_row in enumerate(source_rows):
            if index % 20 == 0:
                print(
                    f"[ur3e-replay] step {index + 1}/{len(source_rows)}",
                    flush=True,
                )
            target_row = source_rows[
                min(index + target_pose_offset, len(source_rows) - 1)
            ]
            target_position = np.asarray(
                target_row["grip_site_pos_world"], dtype=np.float64
            )
            target_quaternion = np.asarray(
                target_row["grip_site_quat_world_xyzw"], dtype=np.float64
            )
            nominal_target_position = target_position.copy()
            if args.align_to_pad_midpoint:
                # The Robotiq pad midpoint changes slightly with finger
                # opening. Use the current physical opening rather than a
                # fixed offset measured at reset.
                current_pad_local_offset = _gripper_pad_local_offset(env)
                target_position, pad_alignment_offset = (
                    _target_site_for_source_pad_midpoint(
                        target_row,
                        target_quaternion,
                        current_pad_local_offset,
                    )
                )
            else:
                pad_alignment_offset = None
            mapped_signal = _mapped_gripper_signal(
                target_row,
                gripper_mapping,
            )
            use_source_action = args.gripper_control_mode == "source_action"
            if mapped_signal is None or use_source_action:
                # Compatibility fallback for old Franka JSON files.
                gripper_action = float(expert["actions"][index, 6])
                gripper_signal_source = "source_binary_action"
                # The Robotiq controller normally integrates a binary command
                # by ``speed`` (0.2) per environment step.  In legacy replay
                # this delays the first closing frame while the arm is already
                # moving.  Optionally jump directly to the maximum internal
                # close signal only on the open -> close transition.
                if (
                    args.immediate_close_transition
                    and gripper_action > 0.0
                    and (
                        previous_gripper_action is None
                        or previous_gripper_action <= 0.0
                    )
                ):
                    env.robots[0].gripper["right"].current_action = np.asarray(
                        [float(args.close_signal)],
                        dtype=np.float64,
                    )
                    gripper_signal_source = "source_binary_action_immediate_close"
            else:
                gripper_action = _set_ur3e_gripper_signal(env, mapped_signal)
                gripper_signal_source = "franka_pad_separation"
            ik_info = None
            q_target = None
            teleport_info = None
            if args.replay_controller == "joint_position":
                q_target, ik_info = _solve_grip_site_ik(
                    env,
                    target_position,
                    target_quaternion,
                    initial_q=previous_q_target,
                    max_iterations=args.trajectory_ik_iterations,
                    max_joint_step=args.trajectory_ik_max_joint_step,
                    max_total_joint_step=args.trajectory_ik_max_total_joint_step,
                    position_tolerance=args.ik_position_tolerance,
                    orientation_tolerance=args.ik_orientation_tolerance,
                )
                if args.joint_target_application == "teleport":
                    teleport_info = _apply_joint_target_without_arm_dynamics(
                        env,
                        q_target,
                        max_self_penetration_m=args.teleport_max_self_penetration_m,
                    )
                    if not teleport_info["applied"]:
                        # Keep the arm at its last dynamically valid pose.
                        # Sending the rejected IK target to JOINT_POSITION
                        # would still move it into the invalid configuration.
                        q_target = np.asarray(
                            env.sim.data.qpos[
                                env.robots[0]._ref_arm_joint_pos_indexes
                            ],
                            dtype=np.float64,
                        ).copy()
                action = np.concatenate(
                    (
                        np.asarray(q_target, dtype=np.float32),
                        np.asarray([gripper_action], dtype=np.float32),
                    )
                )
                previous_q_target = np.asarray(q_target, dtype=np.float64).copy()
            else:
                action = _eef_delta_action(
                    env,
                    target_position,
                    target_quaternion,
                    gripper_action,
                    max_position_command_m=args.max_position_command_m,
                    max_orientation_command_rad=args.max_orientation_command_rad,
            )
            previous_gripper_action = gripper_action
            raw, _, done, step_info = _safe_env_step(
                env,
                action,
                object_body_name=object_body_name,
                monitored_body_names=monitored_body_names,
                max_object_step_m=args.max_object_step_m,
                max_gripper_qpos_step_rad=args.max_gripper_qpos_step_rad,
                max_gripper_qvel_rad_s=args.max_gripper_qvel_rad_s,
                max_gripper_limit_violation_rad=args.max_gripper_limit_violation_rad,
            )
            actual_position, actual_quaternion = _grip_site_pose(env)
            if not args.no_render:
                writer.append_data(
                    _render(
                        env,
                        "robot0_agentview_left",
                        args.video_width,
                        args.video_height,
                    )
                )
                frame_pair = _capture_selected(selected, index, env, args)
                if frame_pair is not None:
                    selected_frames[index] = frame_pair
            position_error = float(np.linalg.norm(actual_position - target_position))
            orientation_error = _quat_error_rad(target_quaternion, actual_quaternion)
            actual_qpos = np.asarray(
                env.sim.data.qpos[env.robots[0]._ref_arm_joint_pos_indexes],
                dtype=np.float64,
            )
            rows.append(
                {
                    "step": index + 1,
                    "target_recorded_state_index": min(
                        index + target_pose_offset, len(source_rows) - 1
                    ),
                    "nominal_source_grip_site_pos_world": nominal_target_position.tolist(),
                    "target_grip_site_pos_world": target_position.tolist(),
                    "target_grip_site_quat_world_xyzw": target_quaternion.tolist(),
                    "actual_grip_site_pos_world": actual_position.tolist(),
                    "actual_grip_site_quat_world_xyzw": actual_quaternion.tolist(),
                    "position_error_m": position_error,
                    "orientation_error_rad": orientation_error,
                    "gripper_action": gripper_action,
                    "gripper_signal": (
                        float(
                            np.asarray(
                                env.robots[0].gripper["right"].current_action,
                                dtype=np.float64,
                            ).reshape(-1)[0]
                        )
                        if np.asarray(
                            env.robots[0].gripper["right"].current_action
                        ).size
                        else None
                    ),
                    "mapped_source_gripper_signal": (
                        float(mapped_signal) if mapped_signal is not None else None
                    ),
                    "gripper_signal_source": gripper_signal_source,
                    "source_grip_pad_geometry": target_row.get(
                        "grip_pad_geometry", {}
                    ),
                    "pad_midpoint_alignment_offset_world": (
                        np.asarray(pad_alignment_offset, dtype=np.float64).tolist()
                        if pad_alignment_offset is not None
                        else None
                    ),
                    "source_gripper_opening_m": target_row.get(
                        "gripper_opening_m"
                    ),
                    "actual_grip_pad_geometry": _gripper_pad_geometry(env),
                    "gripper": _gripper_summary(raw),
                    "control_mode": args.replay_controller,
                    "action_7d": action.tolist(),
                    "q_target": (
                        np.asarray(q_target, dtype=np.float64).tolist()
                        if q_target is not None
                        else None
                    ),
                    "actual_arm_qpos": actual_qpos.tolist(),
                    "joint_position_error_rad": (
                        float(
                            np.linalg.norm(
                                actual_qpos - np.asarray(q_target, dtype=np.float64)
                            )
                        )
                        if q_target is not None
                        else None
                    ),
                    "trajectory_ik": ik_info,
                    "teleport": teleport_info,
                    "physics_rollback": bool(step_info.get("physics_rollback")),
                    "object_step_m": float(
                        step_info.get(
                            "object_step_m",
                            step_info.get("object_step_m_before_rollback", 0.0),
                        )
                    ),
                    "monitored_body_steps_m": step_info.get(
                        "monitored_body_steps_m", {}
                    ),
                    "movable_body_violations_m": step_info.get(
                        "movable_body_violations_m", {}
                    ),
                    "gripper_qpos_step_rad": float(
                        step_info.get("gripper_qpos_step_rad", 0.0)
                    ),
                    "gripper_qvel_max_rad_s": float(
                        step_info.get("gripper_qvel_max_rad_s", 0.0)
                    ),
                    "gripper_limit_violation_rad": float(
                        step_info.get("gripper_limit_violation_rad", 0.0)
                    ),
                    "physics_rollback_reasons": step_info.get(
                        "physics_rollback_reasons", []
                    ),
                    "done": bool(done),
                }
            )
            success = bool(env._check_success()) or success

        result = {
            "robot": "UR3eOfficialFixed",
            "control_mode": args.replay_controller,
            "joint_target_application": args.joint_target_application,
            "steps": len(rows),
            "success": bool(success),
            "initial_grip_site_pos_world": initial_position.tolist(),
            "initial_grip_site_quat_world_xyzw": initial_quaternion.tolist(),
            "initial_gripper": initial_gripper,
            "initial_grip_pad_geometry": initial_ur3e_pad_geometry,
            "gripper_mapping": gripper_mapping,
            "alignment": alignment,
            "initial_link_gripper_collision_violations": initial_link_gripper_violations,
            "align_to_pad_midpoint": bool(args.align_to_pad_midpoint),
            "gripper_control_mode": args.gripper_control_mode,
            "initial_pad_midpoint_alignment_offset_world": (
                np.asarray(initial_pad_alignment_offset, dtype=np.float64).tolist()
                if initial_pad_alignment_offset is not None
                else None
            ),
            "target_pad_local_offset_in_site": (
                np.asarray(target_pad_local_offset, dtype=np.float64).tolist()
                if target_pad_local_offset is not None
                else None
            ),
            "ur3e_body_collision": ur3e_body_collision,
            "monitored_body_names": monitored_body_names,
            "position_error_m": {
                "mean": float(np.mean([row["position_error_m"] for row in rows])),
                "max": float(np.max([row["position_error_m"] for row in rows])),
                "final": float(rows[-1]["position_error_m"]),
            },
            "orientation_error_rad": {
                "mean": float(np.mean([row["orientation_error_rad"] for row in rows])),
                "max": float(np.max([row["orientation_error_rad"] for row in rows])),
                "final": float(rows[-1]["orientation_error_rad"]),
            },
            "grip_pad_separation_error_m": _grip_pad_separation_error_stats(rows),
            "rows": rows,
        }
    finally:
        writer.close()

    (args.output_root / "ur3e_replay.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return {"result": result, "selected_frames": selected_frames}


def _grip_pad_separation_error_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = []
    for row in rows:
        source = row.get("source_grip_pad_geometry", {})
        actual = row.get("actual_grip_pad_geometry", {})
        if source.get("separation_m") is None or actual.get("separation_m") is None:
            continue
        errors.append(
            abs(float(actual["separation_m"]) - float(source["separation_m"]))
        )
    if not errors:
        return {"available": False}
    values = np.asarray(errors, dtype=np.float64)
    return {
        "available": True,
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
        "final": float(values[-1]),
    }


def _write_comparisons(
    args: argparse.Namespace,
    franka_frames: dict[int, dict[str, np.ndarray]],
    ur3e_frames: dict[int, dict[str, np.ndarray]],
    count: int,
) -> list[dict[str, Any]]:
    from PIL import Image, ImageDraw

    comparison_dir = args.output_root / "comparison_frames"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    indices = sorted(set(np.linspace(0, count - 1, min(5, count), dtype=int).tolist()))
    rows: list[dict[str, Any]] = []
    labels = [
        "Franka agentview_left",
        "UR3e agentview_left",
        "Franka eye_in_hand",
        "UR3e eye_in_hand",
    ]
    for number, index in enumerate(indices):
        images = [
            franka_frames[index]["agentview_left"],
            ur3e_frames[index]["agentview_left"],
            franka_frames[index]["eye_in_hand"],
            ur3e_frames[index]["eye_in_hand"],
        ]
        canvas = Image.new(
            "RGB",
            (args.video_width * 2, args.video_height * 2 + 52),
            color=(245, 245, 245),
        )
        draw = ImageDraw.Draw(canvas)
        positions = [
            (0, 52),
            (args.video_width, 52),
            (0, args.video_height + 52),
            (args.video_width, args.video_height + 52),
        ]
        for image, label, position in zip(images, labels, positions):
            canvas.paste(Image.fromarray(image).convert("RGB"), position)
            draw.text((position[0] + 8, position[1] + 8), label, fill=(255, 255, 0))
        path = comparison_dir / f"compare_{number:02d}_step_{index + 1:06d}.png"
        canvas.save(path)
        rows.append({"step": index + 1, "path": str(path)})
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--episode-index", type=int, default=237)
    parser.add_argument("--task", default="PreSoakPan")
    parser.add_argument("--seed", type=int, default=237)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--reuse-franka-replay-json",
        type=Path,
        default=None,
        help="Reuse an existing Franka target-pose JSON and skip Franka rendering.",
    )
    parser.add_argument("--base-z", type=float, default=0.92)
    parser.add_argument("--base-y-offset", type=float, default=0.0)
    parser.add_argument("--camera-width", type=int, default=256)
    parser.add_argument("--camera-height", type=int, default=256)
    parser.add_argument("--video-width", type=int, default=512)
    parser.add_argument("--video-height", type=int, default=512)
    parser.add_argument("--ik-iterations", type=int, default=80)
    parser.add_argument("--ik-position-tolerance", type=float, default=0.0015)
    parser.add_argument("--ik-orientation-tolerance", type=float, default=0.015)
    parser.add_argument(
        "--replay-controller",
        choices=["osc_eef_delta", "joint_position"],
        default="osc_eef_delta",
        help="Controller used to execute the converted grip-site trajectory.",
    )
    parser.add_argument(
        "--trajectory-ik-iterations",
        type=int,
        default=40,
        help="Maximum numerical IK iterations for each replay target.",
    )
    parser.add_argument(
        "--trajectory-ik-max-joint-step",
        type=float,
        default=0.15,
        help="Maximum joint-space step per trajectory IK iteration.",
    )
    parser.add_argument(
        "--trajectory-ik-max-total-joint-step",
        type=float,
        default=None,
        help="Maximum total joint-space change from the current state for one target.",
    )
    parser.add_argument(
        "--joint-position-kp",
        type=float,
        default=150.0,
        help="UR3e JOINT_POSITION proportional gain for the isolated replay.",
    )
    parser.add_argument(
        "--joint-position-damping-ratio",
        type=float,
        default=1.0,
        help="UR3e JOINT_POSITION damping ratio for the isolated replay.",
    )
    parser.add_argument(
        "--joint-target-application",
        choices=["controller", "teleport"],
        default="controller",
        help=(
            "How to execute an IK joint target. controller uses the normal "
            "finite-rate JOINT_POSITION controller; teleport writes the arm "
            "qpos immediately before one physical env.step for a diagnostic "
            "that removes arm tracking lag."
        ),
    )
    parser.add_argument(
        "--teleport-max-self-penetration-m",
        type=float,
        default=0.0005,
        help=(
            "Reject a teleport IK target if a UR3e link penetrates the "
            "mounted Robotiq gripper by more than this distance."
        ),
    )
    parser.add_argument(
        "--immediate-close-transition",
        action="store_true",
        help=(
            "For legacy binary gripper replay, set the Robotiq internal "
            "signal directly to --close-signal on the first open-to-close "
            "transition instead of ramping by the gripper speed."
        ),
    )
    parser.add_argument(
        "--close-signal",
        type=float,
        default=1.0,
        help="Robotiq internal signal used by --immediate-close-transition.",
    )
    parser.add_argument(
        "--align-to-pad-midpoint",
        action="store_true",
        help=(
            "Shift UR3e grip-site targets so the Robotiq pad midpoint follows "
            "the Franka pad midpoint, compensating for different gripper geometry."
        ),
    )
    parser.add_argument(
        "--gripper-control-mode",
        choices=["state_mapping", "source_action"],
        default="state_mapping",
        help=(
            "Use the mapped source Panda opening or replay the recorded binary "
            "gripper action. source_action preserves the close command timing "
            "and lets Robotiq close at its native per-step speed."
        ),
    )
    parser.add_argument(
        "--disable-ur3e-body-collision",
        action="store_true",
        help=(
            "Disable collision for all UR3e link geoms. Robotiq gripper "
            "collision remains enabled. By default UR3e link collision is "
            "enabled with link-link self-collision filtered out."
        ),
    )
    parser.add_argument(
        "--keep-ur3e-body-collision",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--ik-max-joint-step",
        type=float,
        default=0.04,
        help="Maximum joint-space step per IK iteration during UR3e replay.",
    )
    parser.add_argument(
        "--ik-max-total-joint-step",
        type=float,
        default=0.25,
        help="Maximum total joint-space movement while solving one target.",
    )
    parser.add_argument(
        "--max-allowed-penetration",
        type=float,
        default=0.0015,
        help="Reject an IK candidate if gripper/pan penetration exceeds this value.",
    )
    parser.add_argument(
        "--max-penetration-increase",
        type=float,
        default=0.00025,
        help="Reject a candidate that increases existing gripper/pan penetration.",
    )
    parser.add_argument(
        "--max-position-command-m",
        type=float,
        default=0.025,
        help="Maximum EEF position delta sent to OSC per replay frame.",
    )
    parser.add_argument(
        "--max-orientation-command-rad",
        type=float,
        default=0.25,
        help="Maximum EEF orientation delta sent to OSC per replay frame.",
    )
    parser.add_argument(
        "--max-object-step-m",
        type=float,
        default=0.01,
        help=(
            "Rollback a replay step if any monitored movable object moves "
            "farther than this."
        ),
    )
    parser.add_argument(
        "--max-gripper-qpos-step-rad",
        type=float,
        default=0.35,
        help=(
            "Rollback if the six Robotiq joint coordinates change by more "
            "than this in one physics step."
        ),
    )
    parser.add_argument(
        "--max-gripper-qvel-rad-s",
        type=float,
        default=100.0,
        help="Rollback if a Robotiq joint velocity exceeds this value.",
    )
    parser.add_argument(
        "--max-gripper-limit-violation-rad",
        type=float,
        default=0.02,
        help=(
            "Allowed Robotiq joint-limit overshoot before rollback. The "
            "default catches the large contact-induced qpos explosion."
        ),
    )
    parser.add_argument(
        "--object-body-name",
        default=None,
        help=(
            "Override the movable object body used by collision monitoring. "
            "By default it is resolved from env.obj_body_id['obj']."
        ),
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip continuous MP4 encoding; retain selected comparison frames.",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Skip all rendering; use for controller-only diagnostics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global _UR3E_CONTROLLER_MODE
    global _UR3E_JOINT_POSITION_KP, _UR3E_JOINT_POSITION_DAMPING_RATIO
    _UR3E_CONTROLLER_MODE = args.replay_controller
    _UR3E_JOINT_POSITION_KP = float(args.joint_position_kp)
    _UR3E_JOINT_POSITION_DAMPING_RATIO = float(
        args.joint_position_damping_ratio
    )
    if args.keep_ur3e_body_collision and args.disable_ur3e_body_collision:
        raise ValueError(
            "--keep-ur3e-body-collision and --disable-ur3e-body-collision "
            "are mutually exclusive"
        )
    _install_compatible_grasp_check()
    args.output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    expert = _load_expert(args.dataset_root, args.episode_index)
    count = len(expert["actions"])
    selected = set(np.linspace(0, count - 1, min(5, count), dtype=int).tolist())

    if args.reuse_franka_replay_json is not None:
        replay_path = args.reuse_franka_replay_json
        if not replay_path.exists():
            raise FileNotFoundError(replay_path)
        franka_result = json.loads(replay_path.read_text(encoding="utf-8"))
        # The saved Franka replay JSON already stores the post-action target
        # row (recorded_state_index=i+1). Do not shift it a second time.
        franka = {
            "result": franka_result,
            "selected_frames": {},
            "target_pose_offset": 0,
        }
        scene = {
            "layout_id": int(expert["ep_meta"]["layout_id"]),
            "style_id": int(expert["ep_meta"]["style_id"]),
            "reused_franka_replay_json": str(replay_path),
        }
    else:
        franka_env = _make_env(
            args.task,
            "PandaOmron",
            expert["ep_meta"],
            args.seed,
            args.camera_width,
            args.camera_height,
            render=not args.no_render,
        )
        try:
            _restore_expert_xml_state(
                franka_env,
                expert["model_xml"],
                expert["states"][0],
            )
            franka = _replay_franka_recorded_states(
                franka_env,
                expert,
                args,
                selected,
            )
            scene = {
                "layout_id": int(franka_env.layout_id),
                "style_id": int(franka_env.style_id),
            }
        finally:
            franka_env.close()

    # Extract the exact furniture/object joint state before creating UR3e.
    # Closing the source context first also avoids OSMesa context conflicts.
    source_env = _make_env(
        args.task,
        "PandaOmron",
        expert["ep_meta"],
        args.seed,
        args.camera_width,
        args.camera_height,
        render=False,
    )
    try:
        _restore_expert_xml_state(
            source_env,
            expert["model_xml"],
            expert["states"][0],
        )
        scene_state = _extract_common_scene_state(source_env, expert["states"][0])
    finally:
        source_env.close()

    register_ur3e(args.base_z, args.base_y_offset)
    ur3e_env = _make_env(
        args.task,
        "UR3eOfficialFixed",
        expert["ep_meta"],
        args.seed,
        args.camera_width,
        args.camera_height,
        render=not args.no_render,
    )
    try:
        scene_copy = _apply_common_scene_state(ur3e_env, scene_state)
        ur3e_body_collision = _configure_ur3e_body_collision(
            ur3e_env,
            enabled=not args.disable_ur3e_body_collision,
        )
        if args.keep_ur3e_body_collision:
            ur3e_body_collision["legacy_keep_flag"] = True
        ur3e_env._ur3e_body_collision_info = ur3e_body_collision
        ur3e = _replay_ur3e(
            ur3e_env,
            ur3e_env,
            expert,
            franka,
            args,
            selected,
        )
    finally:
        ur3e_env.close()

    comparison = []
    if franka["selected_frames"]:
        comparison = _write_comparisons(
            args,
            franka["selected_frames"],
            ur3e["selected_frames"],
            count,
        )
    summary = {
        "task": args.task,
        "episode_index": args.episode_index,
        "dataset_root": str(args.dataset_root),
        "parquet_path": expert["parquet_path"],
        "scene": scene,
        "scene_state_copy": scene_copy,
        "franka": {
            "steps": franka["result"]["steps"],
            "success": franka["result"]["success"],
        },
        "ur3e": {
            "steps": ur3e["result"]["steps"],
            "success": ur3e["result"]["success"],
            "position_error_m": ur3e["result"]["position_error_m"],
            "orientation_error_rad": ur3e["result"]["orientation_error_rad"],
        },
        "comparison_frames": comparison,
        "elapsed_sec": time.perf_counter() - started,
    }
    (args.output_root / "replay_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
