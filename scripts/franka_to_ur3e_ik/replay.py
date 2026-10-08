"""End-to-end RoboCasa expert replay from PandaOmron to UR3e.

端到端回放：读取 PandaOmron 专家状态，通过相对 EEF 位姿和 IK 驱动 UR3e。
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from robocasa.utils import env_utils
import robocasa.utils.object_utils as object_utils
from robosuite.robots import ROBOT_CLASS_MAPPING
from robosuite.robots.fixed_base_robot import FixedBaseRobot
from robosuite.utils import transform_utils as T

from .dataset import ExpertEpisode
from .geometry import (
    GripperMapping,
    map_relative_pose,
    orientation_error_rad,
    panda_opening_from_qpos,
)
from .ik import IKConfig, align_initial_pose, solve_grip_site_ik
from .robot import UR3eFrankaReplay


ROBOT_NAME = "UR3eFrankaReplay"
_UR3E_BASE_OFFSET = np.zeros(3, dtype=np.float64)
_UR3E_CONTROLLER_CONFIG: dict[str, Any] | None = None
_ORIGINAL_GRASP_CHECK = object_utils.check_obj_grasped


def register_ur3e(base_offset_xyz: np.ndarray) -> None:
    """Register the local UR3e model and its fixed installation offset."""
    global _UR3E_BASE_OFFSET
    _UR3E_BASE_OFFSET = np.asarray(base_offset_xyz, dtype=np.float64).reshape(3)
    ROBOT_CLASS_MAPPING[ROBOT_NAME] = FixedBaseRobot
    # RoboCasa applies the registered Z offset before calling set_robot_base.
    # Keep only XY here; adding Z again would place the robot twice as high.
    # RoboCasa 会先应用注册的 Z 高度，再调用 set_robot_base；这里不能重复加 Z。
    env_utils._ROBOT_POS_OFFSETS[ROBOT_NAME] = [
        0.0,
        0.0,
        float(_UR3E_BASE_OFFSET[2]),
    ]
    env_utils.set_robot_base = _set_fixed_robot_base


def _set_fixed_robot_base(
    env: Any,
    anchor_pos: np.ndarray,
    anchor_ori: np.ndarray,
    rot_dev: float,
    pos_dev_x: float,
    pos_dev_y: float,
) -> np.ndarray:
    """Place the base once during reset; later IK never calls this function."""
    del rot_dev, pos_dev_x, pos_dev_y
    body_id = env.sim.model.body_name2id("robot0_base")
    position = np.asarray(anchor_pos, dtype=np.float64).copy()
    position[:2] += _UR3E_BASE_OFFSET[:2]
    env.sim.model.body_pos[body_id] = position
    # RoboSuite uses xyzw, while MuJoCo body_quat uses wxyz.
    env.sim.model.body_quat[body_id] = T.mat2quat(
        T.euler2mat(np.asarray(anchor_ori, dtype=np.float64))
    )[[3, 0, 1, 2]]
    env.sim.forward()
    return position


def _joint_position_controller_config() -> dict[str, Any]:
    """Absolute UR3e joint-position controller plus Robotiq GRIP."""
    return {
        "type": "BASIC",
        "body_parts": {
            "right": {
                "type": "JOINT_POSITION",
                "input_max": 1.0,
                "input_min": -1.0,
                "output_max": 0.05,
                "output_min": -0.05,
                "kp": 150.0,
                "damping_ratio": 1.0,
                "impedance_mode": "fixed",
                "kp_limits": [0.0, 300.0],
                "damping_ratio_limits": [0.0, 10.0],
                "qpos_limits": None,
                "interpolation": None,
                "input_type": "absolute",
                "gripper": {"type": "GRIP"},
            }
        },
    }


_ORIGINAL_CONTROLLER_CONFIG = env_utils.load_composite_controller_config


def _controller_config(controller: str | None = None, robot: str | None = None):
    # RoboCasa calls this helper with controller=None during create_env().
    # RoboCasa 在 create_env() 中会传入 controller=None，因此这里按 robot 名称拦截。
    if robot == ROBOT_NAME:
        return copy.deepcopy(_UR3E_CONTROLLER_CONFIG or _joint_position_controller_config())
    return _ORIGINAL_CONTROLLER_CONFIG(controller=controller, robot=robot)


env_utils.load_composite_controller_config = _controller_config


def _robotiq_grasp_check(env: Any, object_name: str, threshold: float = 0.035) -> bool:
    """Robotiq-compatible grasp check / 兼容 Robotiq 的抓取判定。"""
    robot = env.robots[0]
    gripper = robot.gripper["right"]
    actuated_names = [
        name
        for name in getattr(robot, "gripper_joints", {}).get("right", [])
        if name.endswith(("finger_joint", "outer_knuckle_joint"))
    ]
    qpos_indices: list[int] = []
    for name in actuated_names:
        try:
            address = env.sim.model.get_joint_qpos_addr(name)
        except Exception:
            continue
        if isinstance(address, (tuple, list, np.ndarray)):
            qpos_indices.extend(int(value) for value in np.asarray(address).reshape(-1))
        else:
            qpos_indices.append(int(address))
    closed = bool(qpos_indices) and bool(
        np.any(np.asarray(env.sim.data.qpos[qpos_indices], dtype=np.float64) > max(threshold, 0.01))
    )
    return bool(closed and env.check_contact(gripper, env.objects[object_name]))


def _compatible_grasp_check(env: Any, object_name: str, threshold: float = 0.035) -> bool:
    """Dispatch Panda and Robotiq grasp checks / 分发不同夹爪的抓取判定。"""
    names = set(getattr(env.robots[0], "gripper_joints", {}).get("right", []))
    if {"gripper0_right_finger_joint1", "gripper0_right_finger_joint2"} <= names:
        return _ORIGINAL_GRASP_CHECK(env, object_name, threshold=threshold)
    return _robotiq_grasp_check(env, object_name, threshold)


def _install_compatible_grasp_check() -> None:
    """Patch only this Python process / 只修改当前 Python 进程。"""
    object_utils.check_obj_grasped = _compatible_grasp_check


def _first(value: Any) -> np.ndarray:
    array = np.asarray(value)
    while array.ndim > 1 and array.shape[0] == 1:
        array = array[0]
    return np.asarray(array)


def _gripper_qpos(raw: dict[str, Any]) -> np.ndarray:
    return _first(raw["robot0_gripper_qpos"]).astype(np.float64).reshape(-1)


def _set_gripper_target(env: Any, signal: float) -> None:
    """Set Robotiq's stateful internal target before a zero gripper action."""
    gripper = env.robots[0].gripper["right"]
    gripper.current_action = np.asarray([float(np.clip(signal, -1.0, 1.0))])


def _grip_site_pose(env: Any) -> tuple[np.ndarray, np.ndarray]:
    """Read grip-site pose / 读取 grip-site 位姿。"""
    site_id = env.robots[0].eef_site_id["right"]
    position = np.asarray(env.sim.data.site_xpos[site_id], dtype=np.float64).copy()
    rotation = np.asarray(env.sim.data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
    return position, T.mat2quat(rotation)


def _geom_id_with_suffix(env: Any, suffixes: tuple[str, ...]) -> int | None:
    """Find a geometry by its raw XML suffix / 按 XML 后缀查找几何体。"""
    for geom_id in range(int(env.sim.model.ngeom)):
        name = env.sim.model.geom_id2name(geom_id) or ""
        if any(name.endswith(suffix) for suffix in suffixes):
            return geom_id
    return None


def _gripper_pad_geometry(env: Any) -> dict[str, Any]:
    """Read both finger-pad centers in world coordinates.

    读取两侧指尖 pad 的世界坐标。Franka 和 Robotiq 的 grip-site 几何
    不同，所以轨迹对齐应优先使用 pad 中点。
    """
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


def _gripper_pad_separation(env: Any) -> float | None:
    """Read the current Robotiq finger-pad separation / 读取当前指尖间距。"""
    value = _gripper_pad_geometry(env).get("separation_m")
    return float(value) if value is not None else None


def _gripper_pad_local_offset(env: Any) -> np.ndarray | None:
    """Return pad midpoint offset in grip-site coordinates.

    返回 pad 中点在 grip-site 局部坐标系下的偏移；Robotiq 开合时该偏移
    会有小幅变化，因此每帧重新测量。
    """
    geometry = _gripper_pad_geometry(env)
    midpoint = geometry.get("midpoint_world")
    if midpoint is None:
        return None
    site_position, site_quaternion = _grip_site_pose(env)
    return T.quat2mat(site_quaternion).T.dot(
        np.asarray(midpoint, dtype=np.float64) - site_position
    )


def _target_site_for_source_pad_motion(
    source_row: dict[str, Any],
    source_initial_rotation: np.ndarray,
    target_initial_rotation: np.ndarray,
    source_initial_pad_midpoint: np.ndarray | None,
    target_initial_pad_midpoint: np.ndarray | None,
    nominal_target_position: np.ndarray,
    target_quaternion: np.ndarray,
    target_pad_local_offset: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Convert source pad motion into a Robotiq grip-site target.

    将 Franka pad 中点相对于初始 pad 的运动映射到 Robotiq，再反算
    Robotiq grip-site 位置。这样不会把两种夹爪不同的安装偏移重复叠加。
    """
    if (
        source_initial_pad_midpoint is None
        or target_initial_pad_midpoint is None
        or target_pad_local_offset is None
    ):
        return np.asarray(nominal_target_position, dtype=np.float64), None
    source_pad = (source_row.get("grip_pad_geometry") or {}).get("midpoint_world")
    if source_pad is None:
        return np.asarray(nominal_target_position, dtype=np.float64), None
    source_pad_delta = source_initial_rotation.T.dot(
        np.asarray(source_pad, dtype=np.float64) - source_initial_pad_midpoint
    )
    # map_relative_pose preserves the source motion in the source initial
    # EEF frame, so the corresponding target displacement uses the target
    # initial EEF frame rather than the changing current orientation.
    # map_relative_pose 在初始 EEF 坐标系中保持相对运动，因此这里应使用
    # 目标初始末端坐标系，而不是每一帧变化的当前姿态。
    desired_pad_midpoint = np.asarray(target_initial_pad_midpoint) + target_initial_rotation.dot(
        source_pad_delta
    )
    adjusted = desired_pad_midpoint - target_initial_rotation.dot(
        np.asarray(target_pad_local_offset, dtype=np.float64)
    )
    return adjusted, adjusted - np.asarray(nominal_target_position, dtype=np.float64)


def _target_gripper_qpos(env: Any) -> np.ndarray:
    """Read all Robotiq gripper qpos / 读取 Robotiq 全部夹爪关节角。"""
    indexes = np.asarray(
        env.robots[0]._ref_gripper_joint_pos_indexes["right"],
        dtype=np.int64,
    )
    return np.asarray(env.sim.data.qpos[indexes], dtype=np.float64).copy()


def _body_position(env: Any, body_name: str | None) -> np.ndarray | None:
    """Read one body position if it exists / 读取物体位置。"""
    if not body_name:
        return None
    try:
        return np.asarray(
            env.sim.data.get_body_xpos(body_name), dtype=np.float64
        ).copy()
    except Exception:
        return None


def _resolve_sponge_body_name(env: Any) -> str | None:
    """Resolve the ScrubCuttingBoard sponge body / 定位百洁布 body。"""
    for candidate in ("sponge_main", "sponge"):
        try:
            env.sim.model.body_name2id(candidate)
            return candidate
        except Exception:
            continue
    object_ids = getattr(env, "obj_body_id", {}) or {}
    for key, body_id in object_ids.items():
        if "sponge" not in str(key).lower():
            continue
        try:
            return str(env.sim.model.body_id2name(int(body_id)))
        except Exception:
            pass
    return None


def _free_joint_for_body(env: Any, body_name: str | None) -> tuple[int, int, int] | None:
    """Find a free joint and its qpos/qvel addresses / 查找物体自由关节。"""
    if body_name is None:
        return None
    try:
        body_id = int(env.sim.model.body_name2id(body_name))
    except Exception:
        return None
    for joint_id in range(int(env.sim.model.njnt)):
        if int(env.sim.model.jnt_bodyid[joint_id]) != body_id:
            continue
        if int(env.sim.model.jnt_type[joint_id]) != 0:
            continue
        return (
            int(env.sim.model.jnt_qposadr[joint_id]),
            int(env.sim.model.jnt_dofadr[joint_id]),
            joint_id,
        )
    return None


def _body_pose_matrix(env: Any, body_name: str) -> np.ndarray:
    """Read one body's world transform / 读取物体世界变换矩阵。"""
    position = np.asarray(
        env.sim.data.get_body_xpos(body_name), dtype=np.float64
    )
    # MuJoCo body quaternions use wxyz.
    quaternion_wxyz = np.asarray(
        env.sim.data.get_body_xquat(body_name), dtype=np.float64
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = T.quat2mat(quaternion_wxyz[[1, 2, 3, 0]])
    transform[:3, 3] = position
    return transform


def _set_free_body_pose(
    env: Any,
    body_name: str,
    transform: np.ndarray,
) -> None:
    """Write a free body's pose and zero its velocity.

    写入自由物体位姿并清零速度。用于 teleport 回放中的抓取保持。
    """
    addresses = _free_joint_for_body(env, body_name)
    if addresses is None:
        return
    qpos_address, qvel_address, _ = addresses
    matrix = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    quaternion_xyzw = T.mat2quat(matrix[:3, :3])
    env.sim.data.qpos[qpos_address : qpos_address + 3] = matrix[:3, 3]
    env.sim.data.qpos[qpos_address + 3 : qpos_address + 7] = quaternion_xyzw[
        [3, 0, 1, 2]
    ]
    env.sim.data.qvel[qvel_address : qvel_address + 6] = 0.0
    env.sim.forward()


def _capture_grasp_lock(env: Any, body_name: str | None) -> dict[str, Any] | None:
    """Capture object pose relative to the UR3e grip site.

    记录百洁布相对 UR3e grip-site 的位姿，后续末端运动时保持该相对
    变换，从而避免 teleport 离散运动把物体从接触中弹落。
    """
    if body_name is None or _free_joint_for_body(env, body_name) is None:
        return None
    site_position, site_quaternion = _grip_site_pose(env)
    site_transform = np.eye(4, dtype=np.float64)
    site_transform[:3, :3] = T.quat2mat(site_quaternion)
    site_transform[:3, 3] = site_position
    object_transform = _body_pose_matrix(env, body_name)
    return {
        "body_name": body_name,
        "site_to_object": (
            np.linalg.inv(site_transform) @ object_transform
        ).tolist(),
    }


def _apply_grasp_lock(env: Any, grasp_lock: dict[str, Any] | None) -> None:
    """Apply the saved relative object pose / 应用抓取保持相对位姿。"""
    if grasp_lock is None:
        return
    body_name = str(grasp_lock["body_name"])
    site_position, site_quaternion = _grip_site_pose(env)
    site_transform = np.eye(4, dtype=np.float64)
    site_transform[:3, :3] = T.quat2mat(site_quaternion)
    site_transform[:3, 3] = site_position
    object_transform = site_transform @ np.asarray(
        grasp_lock["site_to_object"], dtype=np.float64
    )
    _set_free_body_pose(env, body_name, object_transform)


def _geom_world_z_extent(env: Any, geom_id: int) -> tuple[float, float] | None:
    """Return a collision geom's world-space Z extent.

    计算碰撞几何体在世界坐标系中的 Z 范围。这里支持 RoboCasa
    主要使用的 box 和 mesh 两类几何体。
    """
    geom_type = int(env.sim.model.geom_type[geom_id])
    geom_position = np.asarray(env.sim.data.geom_xpos[geom_id], dtype=np.float64)
    geom_rotation = np.asarray(
        env.sim.data.geom_xmat[geom_id], dtype=np.float64
    ).reshape(3, 3)
    if geom_type == 6:  # MuJoCo mjGEOM_BOX / MuJoCo 盒体
        half_size = np.asarray(env.sim.model.geom_size[geom_id], dtype=np.float64)
        half_height = float(np.sum(np.abs(geom_rotation[2]) * half_size))
        return (
            float(geom_position[2] - half_height),
            float(geom_position[2] + half_height),
        )
    if geom_type == 7:  # MuJoCo mjGEOM_MESH / MuJoCo 网格
        mesh_id = int(env.sim.model.geom_dataid[geom_id])
        vertex_start = int(env.sim.model.mesh_vertadr[mesh_id])
        vertex_count = int(env.sim.model.mesh_vertnum[mesh_id])
        vertices = np.asarray(env.sim.model.mesh_vert)[
            vertex_start : vertex_start + vertex_count
        ]
        world_vertices = (geom_rotation @ vertices.T).T + geom_position
        return float(np.min(world_vertices[:, 2])), float(np.max(world_vertices[:, 2]))
    return None


def _body_collision_z_extent(
    env: Any,
    body_name: str | None,
) -> tuple[float, float] | None:
    """Return the union Z extent of one body's collision geoms.

    返回一个 body 的所有碰撞几何体的 Z 范围，用于判断物体是否穿过砧板。
    """
    if body_name is None:
        return None
    try:
        body_id = int(env.sim.model.body_name2id(body_name))
    except Exception:
        return None
    extents: list[tuple[float, float]] = []
    for geom_id in range(int(env.sim.model.ngeom)):
        if int(env.sim.model.geom_bodyid[geom_id]) != body_id:
            continue
        if int(env.sim.model.geom_contype[geom_id]) == 0:
            continue
        extent = _geom_world_z_extent(env, geom_id)
        if extent is not None:
            extents.append(extent)
    if not extents:
        return None
    return (
        min(extent[0] for extent in extents),
        max(extent[1] for extent in extents),
    )


def _gripper_collision_z_extent(env: Any) -> tuple[float, float] | None:
    """Return the union Z extent of Robotiq collision geoms.

    返回 Robotiq 夹爪所有碰撞几何体的 Z 范围。抓取锁激活后，这个范围
    也必须位于砧板上表面之上，避免手指本体穿过砧板。
    """
    extents: list[tuple[float, float]] = []
    for geom_id in range(int(env.sim.model.ngeom)):
        body_id = int(env.sim.model.geom_bodyid[geom_id])
        body_name = env.sim.model.body_id2name(body_id) or ""
        geom_name = env.sim.model.geom_id2name(geom_id) or ""
        if not (
            body_name.startswith("gripper0_")
            or geom_name.startswith("gripper0_")
        ):
            continue
        if int(env.sim.model.geom_contype[geom_id]) == 0:
            continue
        extent = _geom_world_z_extent(env, geom_id)
        if extent is not None:
            extents.append(extent)
    if not extents:
        return None
    return (
        min(extent[0] for extent in extents),
        max(extent[1] for extent in extents),
    )


def _resolve_cutting_board_body_name(env: Any) -> str | None:
    """Resolve the cutting-board body / 定位砧板 body。"""
    for candidate in ("cutting_board_main", "cutting_board"):
        try:
            env.sim.model.body_name2id(candidate)
            return candidate
        except Exception:
            continue
    object_ids = getattr(env, "obj_body_id", {}) or {}
    for key, body_id in object_ids.items():
        if "cutting" not in str(key).lower() and "board" not in str(key).lower():
            continue
        try:
            return str(env.sim.model.body_id2name(int(body_id)))
        except Exception:
            pass
    return None


def _gripper_board_penetration(
    env: Any,
    board_body_name: str | None,
) -> float:
    """Return the deepest gripper-to-board contact penetration.

    返回夹爪碰撞几何体与砧板之间的最大接触穿透深度。百洁布与砧板的
    接触不在这里判定，因为擦拭时百洁布允许接触砧板，但不能穿过砧板。
    """
    if board_body_name is None:
        return 0.0
    try:
        board_body_id = int(env.sim.model.body_name2id(board_body_name))
    except Exception:
        return 0.0
    deepest = 0.0
    for contact_index in range(int(env.sim.data.ncon)):
        contact = env.sim.data.contact[contact_index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        body1 = int(env.sim.model.geom_bodyid[geom1])
        body2 = int(env.sim.model.geom_bodyid[geom2])
        if body1 == board_body_id:
            other_geom = geom2
        elif body2 == board_body_id:
            other_geom = geom1
        else:
            continue
        other_body = env.sim.model.body_id2name(
            int(env.sim.model.geom_bodyid[other_geom])
        ) or ""
        if not other_body.startswith("gripper0_"):
            continue
        deepest = max(deepest, max(0.0, -float(contact.dist)))
    return float(deepest)


def _project_locked_target_above_board(
    env: Any,
    target_position: np.ndarray,
    target_quaternion: np.ndarray,
    q_target: np.ndarray,
    grasp_lock: dict[str, Any] | None,
    *,
    board_body_name: str | None,
    sponge_body_name: str | None,
    ik_config: IKConfig,
    previous_q: np.ndarray | None,
    projection_enabled: bool = True,
    clearance_m: float = 0.001,
    max_passes: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Project a teleport target out of the cutting board.

    对 teleport 回放的目标做碰撞投影：

    - 百洁布最低碰撞点低于砧板顶面时，整体沿 +Z 抬升；
    - 夹爪与砧板有深度穿透时，同样抬升；
    - 每次抬升后重新求 IK，保证夹爪和百洁布仍保持刚性抓取关系。

    这不是对百洁布单独“瞬移修正”，而是修正 EEF 目标后再把百洁布
    按抓取锁的相对变换带上去。
    """
    if not projection_enabled or grasp_lock is None or board_body_name is None:
        return (
            np.asarray(target_position, dtype=np.float64),
            np.asarray(target_quaternion, dtype=np.float64),
            np.asarray(q_target, dtype=np.float64),
            {
                "applied": False,
                "translation_world": [0.0, 0.0, 0.0],
                "passes": 0,
                "board_top_z_m": None,
                "sponge_bottom_z_m": None,
                "gripper_board_penetration_m": 0.0,
            },
        )

    board_extent = _body_collision_z_extent(env, board_body_name)
    if board_extent is None:
        return (
            np.asarray(target_position, dtype=np.float64),
            np.asarray(target_quaternion, dtype=np.float64),
            np.asarray(q_target, dtype=np.float64),
            {
                "applied": False,
                "translation_world": [0.0, 0.0, 0.0],
                "passes": 0,
                "board_top_z_m": None,
                "sponge_bottom_z_m": None,
                "gripper_board_penetration_m": 0.0,
            },
        )

    projected_position = np.asarray(target_position, dtype=np.float64).copy()
    projected_q = np.asarray(q_target, dtype=np.float64).copy()
    total_translation = np.zeros(3, dtype=np.float64)
    max_sponge_bottom = None
    max_gripper_penetration = 0.0
    max_gripper_bottom = None
    passes = 0
    board_top_z = float(board_extent[1])

    for passes in range(1, max_passes + 1):
        # Evaluate the candidate pose in the simulator. The IK solver restores
        # the previous state before returning, so this temporary placement is
        # safe and will be applied again by the caller.
        # 在模拟器中评估候选姿态。IK 求解器返回前会恢复原状态，因此这里的
        # 临时写入不会改变回放状态，最终姿态由主循环再次写入。
        _apply_joint_target_directly(env, projected_q)
        _apply_grasp_lock(env, grasp_lock)
        sponge_extent = _body_collision_z_extent(env, sponge_body_name)
        sponge_bottom = sponge_extent[0] if sponge_extent is not None else None
        gripper_penetration = _gripper_board_penetration(env, board_body_name)
        gripper_extent = _gripper_collision_z_extent(env)
        gripper_bottom = (
            gripper_extent[0] if gripper_extent is not None else None
        )
        max_sponge_bottom = sponge_bottom
        max_gripper_bottom = gripper_bottom
        max_gripper_penetration = max(
            max_gripper_penetration,
            gripper_penetration,
        )
        sponge_lift = (
            max(0.0, board_top_z + clearance_m - float(sponge_bottom))
            if sponge_bottom is not None
            else 0.0
        )
        gripper_lift = max(
            0.0,
            (
                board_top_z + clearance_m - float(gripper_bottom)
                if gripper_bottom is not None
                else float(gripper_penetration) + clearance_m
            ),
        )
        lift = max(sponge_lift, gripper_lift)
        if lift <= 1e-6:
            break

        projected_position[2] += lift
        total_translation[2] += lift
        projected_q, _ = solve_grip_site_ik(
            env,
            projected_position,
            target_quaternion,
            initial_q=previous_q if previous_q is not None else projected_q,
            config=ik_config,
        )

    # Leave the simulator at the projected candidate so the subsequent
    # physics step starts from exactly the pose recorded in the diagnostics.
    # 让模拟器停在最终候选姿态，后续物理步和诊断都从同一个状态开始。
    _apply_joint_target_directly(env, projected_q)
    _apply_grasp_lock(env, grasp_lock)
    final_sponge_extent = _body_collision_z_extent(env, sponge_body_name)
    final_gripper_penetration = _gripper_board_penetration(env, board_body_name)
    return (
        projected_position,
        np.asarray(target_quaternion, dtype=np.float64),
        projected_q,
        {
            "applied": bool(np.linalg.norm(total_translation) > 1e-9),
            "translation_world": total_translation.tolist(),
            "passes": int(passes),
            "board_top_z_m": board_top_z,
            "sponge_bottom_z_m": (
                float(final_sponge_extent[0])
                if final_sponge_extent is not None
                else None
            ),
            "gripper_bottom_z_m": (
                float(_gripper_collision_z_extent(env)[0])
                if _gripper_collision_z_extent(env) is not None
                else None
            ),
            "gripper_board_penetration_m": float(final_gripper_penetration),
            "max_observed_gripper_board_penetration_m": float(
                max_gripper_penetration
            ),
            "max_observed_gripper_bottom_z_m": (
                float(max_gripper_bottom)
                if max_gripper_bottom is not None
                else None
            ),
        },
    )


def _project_gripper_target_above_board(
    env: Any,
    target_position: np.ndarray,
    target_quaternion: np.ndarray,
    q_target: np.ndarray,
    *,
    board_body_name: str | None,
    ik_config: IKConfig,
    previous_q: np.ndarray | None,
    clearance_m: float = 0.001,
    max_passes: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Keep a released Robotiq gripper above the cutting board.

    释放百洁布后，仍需防止夹爪本体穿过砧板。这里不再修改百洁布，
    只对 EEF 目标做 +Z 投影并重新求 IK。
    """
    if board_body_name is None:
        return (
            np.asarray(target_position, dtype=np.float64),
            np.asarray(target_quaternion, dtype=np.float64),
            np.asarray(q_target, dtype=np.float64),
            {
                "applied": False,
                "translation_world": [0.0, 0.0, 0.0],
                "passes": 0,
                "board_top_z_m": None,
                "gripper_bottom_z_m": None,
            },
        )
    board_extent = _body_collision_z_extent(env, board_body_name)
    if board_extent is None:
        return (
            np.asarray(target_position, dtype=np.float64),
            np.asarray(target_quaternion, dtype=np.float64),
            np.asarray(q_target, dtype=np.float64),
            {
                "applied": False,
                "translation_world": [0.0, 0.0, 0.0],
                "passes": 0,
                "board_top_z_m": None,
                "gripper_bottom_z_m": None,
            },
        )

    projected_position = np.asarray(target_position, dtype=np.float64).copy()
    projected_q = np.asarray(q_target, dtype=np.float64).copy()
    total_translation = np.zeros(3, dtype=np.float64)
    board_top_z = float(board_extent[1])
    passes = 0
    for passes in range(1, max_passes + 1):
        _apply_joint_target_directly(env, projected_q)
        gripper_extent = _gripper_collision_z_extent(env)
        gripper_bottom = (
            gripper_extent[0] if gripper_extent is not None else None
        )
        if gripper_bottom is None:
            break
        lift = max(0.0, board_top_z + clearance_m - float(gripper_bottom))
        if lift <= 1e-6:
            break
        projected_position[2] += lift
        total_translation[2] += lift
        projected_q, _ = solve_grip_site_ik(
            env,
            projected_position,
            target_quaternion,
            initial_q=previous_q if previous_q is not None else projected_q,
            config=ik_config,
        )

    _apply_joint_target_directly(env, projected_q)
    final_extent = _gripper_collision_z_extent(env)
    return (
        projected_position,
        np.asarray(target_quaternion, dtype=np.float64),
        projected_q,
        {
            "applied": bool(np.linalg.norm(total_translation) > 1e-9),
            "translation_world": total_translation.tolist(),
            "passes": int(passes),
            "board_top_z_m": board_top_z,
            "gripper_bottom_z_m": (
                float(final_extent[0]) if final_extent is not None else None
            ),
        },
    )


def _grasp_candidate(
    env: Any,
    body_name: str | None,
    *,
    gripper_signal: float,
    max_site_distance_m: float = 0.065,
) -> bool:
    """Check whether the closing gripper is close enough to lock the sponge."""
    if body_name is None or gripper_signal < 0.6:
        return False
    object_position = _body_position(env, body_name)
    site_position, _ = _grip_site_pose(env)
    if object_position is None:
        return False
    return bool(np.linalg.norm(object_position - site_position) <= max_site_distance_m)


def _apply_joint_target_directly(env: Any, q_target: np.ndarray) -> None:
    """Apply an IK pose before one physics step / 直接写入 IK 关节目标。"""
    robot = env.robots[0]
    qpos_idx = np.asarray(robot._ref_arm_joint_pos_indexes, dtype=np.int64)
    qvel_idx = np.asarray(robot._ref_arm_joint_vel_indexes, dtype=np.int64)
    env.sim.data.qpos[qpos_idx] = np.asarray(q_target, dtype=np.float64).reshape(6)
    env.sim.data.qvel[qvel_idx] = 0.0
    env.sim.forward()
    robot.composite_controller.update_state()


def _apply_gripper_qpos_directly(env: Any, q_target: np.ndarray) -> None:
    """Apply a complete Robotiq joint pose / 直接写入 Robotiq 全部关节位姿。"""
    robot = env.robots[0]
    qpos_idx = np.asarray(
        robot._ref_gripper_joint_pos_indexes["right"],
        dtype=np.int64,
    )
    qvel_idx = np.asarray(
        robot._ref_gripper_joint_vel_indexes["right"],
        dtype=np.int64,
    )
    values = np.asarray(q_target, dtype=np.float64).reshape(-1)
    if values.size != qpos_idx.size:
        raise ValueError(
            f"Expected {qpos_idx.size} Robotiq qpos values, got {values.size}"
        )
    env.sim.data.qpos[qpos_idx] = values
    env.sim.data.qvel[qvel_idx] = 0.0
    env.sim.forward()
    robot.composite_controller.update_state()


def _robotiq_open_qpos(env: Any) -> np.ndarray:
    """Return the model's stable fully-open Robotiq pose.

    返回模型定义的稳定全开位姿。使用 gripper.init_qpos 比把六个关节
    全部设为零更可靠，因为 Robotiq 的被动关节由 tendon 约束关联。
    """
    values = np.asarray(
        env.robots[0].gripper["right"].init_qpos,
        dtype=np.float64,
    ).reshape(-1)
    expected = len(
        np.asarray(
            env.robots[0]._ref_gripper_joint_pos_indexes["right"],
            dtype=np.int64,
        )
    )
    if values.size != expected:
        raise ValueError(
            f"Robotiq init_qpos size {values.size} does not match {expected}"
        )
    return values


def _step_with_gripper_guard(
    env: Any,
    action: np.ndarray,
    *,
    max_gripper_qpos_step_rad: float = 0.25,
) -> tuple[dict[str, Any], Any, bool, dict[str, Any]]:
    """Step physics and rollback an explosive Robotiq update.

    teleport 模式下机械臂是运动学写入，而 Robotiq 仍由 MuJoCo 动力学
    推进；如果 tendon/碰撞求解产生异常大跳变，恢复到本帧之前的稳定状态。
    """
    saved_state = env.sim.get_state().flatten()
    gripper = env.robots[0].gripper["right"]
    saved_gripper_action = np.asarray(
        gripper.current_action, dtype=np.float64
    ).copy()
    before_qpos = _target_gripper_qpos(env)
    raw, reward, done, info = env.step(action)
    after_qpos = _target_gripper_qpos(env)
    qpos_step = float(np.linalg.norm(after_qpos - before_qpos))
    qvel_indexes = np.asarray(
        env.robots[0]._ref_gripper_joint_vel_indexes["right"],
        dtype=np.int64,
    )
    qvel_max = float(
        np.max(np.abs(np.asarray(env.sim.data.qvel[qvel_indexes])), initial=0.0)
    )
    rollback = (
        not np.all(np.isfinite(after_qpos))
        or qpos_step > max_gripper_qpos_step_rad
        or qvel_max > 100.0
    )
    info = dict(info or {})
    info["gripper_qpos_step_rad"] = qpos_step
    info["gripper_qvel_max_rad_s"] = qvel_max
    if rollback:
        env.sim.set_state_from_flattened(saved_state)
        env.sim.forward()
        gripper.current_action = saved_gripper_action
        env.robots[0].composite_controller.update_state()
        raw = env._get_observations(force_update=True)
        info["gripper_physics_rollback"] = True
        info["gripper_qpos_step_rad_before_rollback"] = qpos_step
        info["gripper_qvel_max_rad_s_before_rollback"] = qvel_max
        info["gripper_physics_rollback_reason"] = (
            "nonfinite_qpos"
            if not np.all(np.isfinite(after_qpos))
            else "qpos_step"
            if qpos_step > max_gripper_qpos_step_rad
            else "qvel_spike"
        )
    else:
        info["gripper_physics_rollback"] = False
    return raw, reward, done, info


def _make_env(
    task_name: str,
    robot_name: str,
    episode: ExpertEpisode,
    seed: int,
    camera_width: int,
    camera_height: int,
    render: bool,
) -> Any:
    """Create a deterministic RoboCasa scene matching the expert episode."""
    meta = dict(episode.ep_meta)
    meta.pop("init_robot_base_pos", None)
    meta.pop("init_robot_base_ori", None)
    kwargs: dict[str, Any] = {
        "split": None,
        "obj_instance_split": "target",
        "layout_and_style_ids": [(int(meta["layout_id"]), int(meta["style_id"]))],
        "seed": seed,
        "camera_names": (
            ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]
            if render
            else []
        ),
        "camera_widths": camera_width,
        "camera_heights": camera_height,
        "render_camera": "robot0_agentview_left",
        "control_freq": 20,
        "initialization_noise": None,
    }
    env = env_utils.create_env(task_name, robots=robot_name, **kwargs)
    env.set_ep_meta(meta)
    env.reset()
    return env


def _restore_state(env: Any, state: np.ndarray) -> None:
    """Restore one exact MuJoCo state / 恢复一帧精确 MuJoCo 状态。"""
    env.sim.set_state_from_flattened(np.asarray(state, dtype=np.float64))
    env.sim.forward()
    if hasattr(env, "update_sites"):
        env.update_sites()
    if hasattr(env, "update_state"):
        env.update_state()


def _restore_expert_xml_state(
    env: Any,
    model_xml: str,
    state: np.ndarray,
) -> None:
    """Restore the expert's exact Panda model and state."""
    edited_xml = env.edit_model_xml(model_xml)
    env.reset_from_xml_string(edited_xml)
    expected_size = len(env.sim.get_state().flatten())
    if expected_size != len(state):
        raise ValueError(
            "Expert model/state size mismatch: "
            f"model_state={expected_size}, recorded_state={len(state)}"
        )
    _restore_state(env, state)


def _extract_scene_state(source_env: Any, state: np.ndarray) -> dict[str, Any]:
    """Extract non-robot qpos/qvel values so both scenes share object motion."""
    sim = source_env.sim
    nq, nv = int(sim.model.nq), int(sim.model.nv)
    qpos = state[1 : 1 + nq]
    qvel = state[1 + nq : 1 + nq + nv]
    joints: dict[str, dict[str, list[float]]] = {}
    for joint_id in range(int(sim.model.njnt)):
        name = sim.model.joint_id2name(joint_id)
        if not name or name.startswith(("robot0_", "gripper0_", "mobilebase0_")):
            continue
        qwidth = 7 if sim.model.jnt_type[joint_id] == 0 else 4 if sim.model.jnt_type[joint_id] == 1 else 1
        vwidth = 6 if sim.model.jnt_type[joint_id] == 0 else 3 if sim.model.jnt_type[joint_id] == 1 else 1
        qadr = int(sim.model.jnt_qposadr[joint_id])
        vadr = int(sim.model.jnt_dofadr[joint_id])
        joints[name] = {
            "qpos": qpos[qadr : qadr + qwidth].tolist(),
            "qvel": qvel[vadr : vadr + vwidth].tolist(),
        }
    return {"time": float(state[0]), "joints": joints}


def _apply_scene_state(env: Any, scene_state: dict[str, Any]) -> dict[str, Any]:
    copied: list[str] = []
    skipped: list[str] = []
    for name, values in scene_state["joints"].items():
        try:
            joint_id = env.sim.model.joint_name2id(name)
        except Exception:
            skipped.append(name)
            continue
        qwidth = 7 if env.sim.model.jnt_type[joint_id] == 0 else 4 if env.sim.model.jnt_type[joint_id] == 1 else 1
        vwidth = 6 if env.sim.model.jnt_type[joint_id] == 0 else 3 if env.sim.model.jnt_type[joint_id] == 1 else 1
        qadr = int(env.sim.model.jnt_qposadr[joint_id])
        vadr = int(env.sim.model.jnt_dofadr[joint_id])
        qpos = np.asarray(values["qpos"], dtype=np.float64)
        qvel = np.asarray(values["qvel"], dtype=np.float64)
        if qpos.size != qwidth or qvel.size != vwidth:
            skipped.append(name)
            continue
        env.sim.data.qpos[qadr : qadr + qwidth] = qpos
        env.sim.data.qvel[vadr : vadr + vwidth] = qvel
        copied.append(name)
    env.sim.data.time = float(scene_state["time"])
    env.sim.forward()
    return {"copied_joint_count": len(copied), "skipped_joint_count": len(skipped)}


def _franka_rows(source_env: Any, episode: ExpertEpisode) -> dict[str, Any]:
    """Read exact post-action Franka EEF poses from recorded states."""
    rows: list[dict[str, Any]] = []
    _restore_state(source_env, episode.states[0])
    # reset_from_xml_string() can rebuild the MuJoCo model and invalidate old
    # numeric site IDs. Always query the ID after the exact expert model is in
    # place. reset_from_xml_string() 可能重建模型，旧 site id 会失效。
    robot = source_env.robots[0]
    site_id = robot.eef_site_id["right"]
    initial_raw = source_env._get_observations(force_update=True)
    initial_position = np.asarray(source_env.sim.data.site_xpos[site_id], dtype=np.float64).copy()
    initial_rotation = np.asarray(source_env.sim.data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
    initial_quaternion = T.mat2quat(initial_rotation)
    initial_opening = panda_opening_from_qpos(_gripper_qpos(initial_raw))
    initial_pad_geometry = _gripper_pad_geometry(source_env)
    for index in range(len(episode.actions_hdf5_order)):
        state_index = min(index + 1, len(episode.states) - 1)
        _restore_state(source_env, episode.states[state_index])
        raw = source_env._get_observations(force_update=True)
        position = np.asarray(source_env.sim.data.site_xpos[site_id], dtype=np.float64).copy()
        rotation = np.asarray(source_env.sim.data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
        quaternion = T.mat2quat(rotation)
        opening = panda_opening_from_qpos(_gripper_qpos(raw))
        pad_geometry = _gripper_pad_geometry(source_env)
        rows.append(
            {
                "step": index + 1,
                "state_index": state_index,
                "position_world": position.tolist(),
                "quaternion_world_xyzw": quaternion.tolist(),
                "panda_gripper_qpos": _gripper_qpos(raw).tolist(),
                "panda_opening_m": opening,
                "source_gripper_action": float(episode.actions_hdf5_order[index, 6]),
                "grip_pad_geometry": pad_geometry,
                "reward": float(episode.rewards[index]),
                "done": bool(episode.dones[index]),
            }
        )
    return {
        "initial_position_world": initial_position.tolist(),
        "initial_quaternion_world_xyzw": initial_quaternion.tolist(),
        "initial_opening_m": initial_opening,
        "initial_pad_geometry": initial_pad_geometry,
        "rows": rows,
    }


def run_replay(
    task_name: str,
    episode: ExpertEpisode,
    *,
    seed: int,
    base_offset_xyz: np.ndarray,
    output_dir: Path,
    camera_width: int = 128,
    camera_height: int = 128,
    ik_config: IKConfig = IKConfig(),
    save_video: bool = False,
    max_steps: int | None = None,
    execution_mode: str = "teleport",
    align_to_pad_midpoint: bool = True,
    gripper_control_mode: str = "source_action",
) -> dict[str, Any]:
    """Run the complete initial-alignment + fixed-base IK replay."""
    output_dir.mkdir(parents=True, exist_ok=True)
    # Create the Panda source scene before installing the UR3e-specific
    # set_robot_base callback. That callback is global in RoboCasa and would
    # otherwise also alter the source mobile Panda scene.
    # RoboCasa 的 set_robot_base 是全局函数，必须先创建 Franka 场景，
    # 否则 UR3e 的底座回调会误作用到移动 Franka 场景。
    source_env = _make_env(task_name, "PandaOmron", episode, seed, camera_width, camera_height, False)
    franka = None
    scene_state = None
    try:
        _restore_expert_xml_state(source_env, episode.model_xml, episode.states[0])
        scene_state = _extract_scene_state(source_env, episode.states[0])
        franka = _franka_rows(source_env, episode)
    finally:
        source_env.close()

    register_ur3e(base_offset_xyz)
    _install_compatible_grasp_check()
    target_env = _make_env(task_name, ROBOT_NAME, episode, seed, camera_width, camera_height, save_video)
    video_writers: dict[str, Any] = {}
    video_paths: dict[str, str] = {}
    if save_video:
        import imageio.v2 as imageio

        # Keep the original filename for the main view, and add a separate
        # eye-in-hand video for inspecting the tool motion.
        # 主视角保留原文件名，同时单独保存末端相机视角，便于观察工具运动。
        writer_options = {
            "fps": 20,
            "codec": "libx264",
            "ffmpeg_params": ["-pix_fmt", "yuv420p"],
        }
        video_paths = {
            "agentview_left": str(output_dir / "ur3e_replay.mp4"),
            "eye_in_hand": str(output_dir / "ur3e_eye_in_hand_replay.mp4"),
        }
        video_writers = {
            camera_name: imageio.get_writer(path, **writer_options)
            for camera_name, path in video_paths.items()
        }
    try:
        scene_copy = _apply_scene_state(target_env, scene_state)
        target_robot = target_env.robots[0]
        target_site_id = target_robot.eef_site_id["right"]
        # Keep the qpos produced by RoboCasa reset. It satisfies the Robotiq
        # tendon constraints; writing six zeros here causes a large first-step
        # passive-joint jump.
        # 保留 RoboCasa reset 生成的夹爪 qpos。它满足 tendon 约束；把六个
        # 关节全部写成 0 会导致第一帧被动关节发生很大的跳变。
        gripper_qvel_idx = np.asarray(
            target_robot._ref_gripper_joint_vel_indexes["right"], dtype=np.int64
        )
        target_env.sim.data.qvel[gripper_qvel_idx] = 0.0
        target_env.sim.forward()
        _set_gripper_target(target_env, -1.0)
        # Let the Robotiq passive joints settle to the open command before
        # recording the first frame. Otherwise the reset pose can create one
        # visible 0.3--0.4 rad jump even when the gripper is not touching an
        # object.
        # 在记录第一帧前让 Robotiq 被动关节先稳定到张开状态，避免 reset
        # 姿态与张开指令之间产生可见的 0.3--0.4 rad 瞬时跳变。
        for _ in range(5):
            settle_arm_q = np.asarray(
                target_env.sim.data.qpos[target_robot._ref_arm_joint_pos_indexes],
                dtype=np.float64,
            ).copy()
            target_env.step(
                np.concatenate(
                    (
                        settle_arm_q.astype(np.float32),
                        np.asarray([-1.0], dtype=np.float32),
                    )
                )
            )
            _apply_joint_target_directly(target_env, settle_arm_q)

        target_initial_position = np.asarray(target_env.sim.data.site_xpos[target_site_id], dtype=np.float64).copy()
        target_initial_rotation = np.asarray(target_env.sim.data.site_xmat[target_site_id], dtype=np.float64).reshape(3, 3)
        target_initial_quaternion = T.mat2quat(target_initial_rotation)
        source_initial_site_position = np.asarray(
            franka["initial_position_world"], dtype=np.float64
        )
        source_initial_quaternion = np.asarray(franka["initial_quaternion_world_xyzw"], dtype=np.float64)
        source_initial_rotation = T.quat2mat(source_initial_quaternion)
        source_initial_pad_midpoint = (
            np.asarray(
                (franka.get("initial_pad_geometry") or {}).get("midpoint_world"),
                dtype=np.float64,
            )
            if (franka.get("initial_pad_geometry") or {}).get("midpoint_world") is not None
            else None
        )
        target_pad_local_offset = (
            _gripper_pad_local_offset(target_env)
            if align_to_pad_midpoint
            else None
        )
        alignment_target_position = source_initial_site_position.copy()
        if align_to_pad_midpoint and source_initial_pad_midpoint is not None:
            # Align the Robotiq pad midpoint to the Franka pad midpoint.
            # 初始阶段对齐两种夹爪的 pad 中点，而不是不同的 grip-site。
            alignment_target_position = source_initial_pad_midpoint - source_initial_rotation.dot(
                np.asarray(target_pad_local_offset, dtype=np.float64)
            )

        alignment = align_initial_pose(
            target_env,
            alignment_target_position,
            source_initial_quaternion,
            config=ik_config,
            allow_base_xy=True,
        )
        # Use the achieved pose as the frame origin. This avoids introducing a
        # one-time jump if the initial alignment stopped within tolerance.
        # 使用实际对齐结果作为坐标原点，避免初始残差造成第一帧跳变。
        initial_target_position = np.asarray(
            target_env.sim.data.site_xpos[target_site_id], dtype=np.float64
        ).copy()
        initial_target_rotation = np.asarray(
            target_env.sim.data.site_xmat[target_site_id], dtype=np.float64
        ).reshape(3, 3)
        initial_target_quaternion = T.mat2quat(initial_target_rotation)
        initial_target_pad_geometry = _gripper_pad_geometry(target_env)
        target_initial_pad_midpoint = (
            np.asarray(
                initial_target_pad_geometry["midpoint_world"],
                dtype=np.float64,
            )
            if initial_target_pad_geometry.get("midpoint_world") is not None
            else None
        )
        # The base is now fixed. Every following solve only changes arm qpos.
        locked_base_position = np.asarray(
            target_env.sim.model.body_pos[target_env.sim.model.body_name2id("robot0_base")],
            dtype=np.float64,
        ).copy()
        base_body_id = target_env.sim.model.body_name2id("robot0_base")

        mapping = GripperMapping()
        rows: list[dict[str, Any]] = []
        previous_q: np.ndarray | None = None
        position_errors: list[float] = []
        orientation_errors: list[float] = []
        env_success = False
        max_base_drift_m = 0.0
        sponge_body_name = _resolve_sponge_body_name(target_env)
        cutting_board_body_name = _resolve_cutting_board_body_name(target_env)
        gripper_hold_command: float | None = None
        grasp_lock: dict[str, Any] | None = None
        grasp_lock_activation_step: int | None = None
        grasp_lock_release_step: int | None = None
        collision_projection_latched = False
        collision_projection_activation_step: int | None = None
        release_phase = False
        release_phase_start_step: int | None = None
        release_phase_end_step: int | None = None
        release_phase_frame_count = 0
        release_hold_q: np.ndarray | None = None
        release_start_gripper_q: np.ndarray | None = None
        release_open_gripper_q: np.ndarray | None = None
        last_applied_q: np.ndarray | None = None
        release_open_separation_m = 0.078
        # Use a fixed transition length so the lift is not shortened by
        # Robotiq's measured pad separation reaching the threshold early.
        # 使用固定过渡帧数，避免 Robotiq 指尖间距过早达到阈值而缩短抬升。
        release_max_frames = 16
        release_lift_active = False
        release_lift_start_step: int | None = None
        release_lift_height_m: float | None = None
        release_lift_clearance_m = 0.12
        release_lift_frame_count = 0
        release_lift_ramp_frames = 16
        release_lift_start_position: np.ndarray | None = None
        release_lift_start_quaternion: np.ndarray | None = None
        release_lift_target_position: np.ndarray | None = None
        release_sponge_pose: np.ndarray | None = None
        replay_rows = franka["rows"]
        if max_steps is not None and max_steps > 0:
            replay_rows = replay_rows[:max_steps]
        for index, source_row in enumerate(replay_rows):
            source_position = np.asarray(source_row["position_world"], dtype=np.float64)
            source_quaternion = np.asarray(source_row["quaternion_world_xyzw"], dtype=np.float64)
            target_position, target_quaternion = map_relative_pose(
                source_initial_site_position,
                source_initial_quaternion,
                source_position,
                source_quaternion,
                initial_target_position,
                initial_target_quaternion,
            )
            nominal_target_position = target_position.copy()
            if align_to_pad_midpoint:
                target_position, pad_alignment_offset = _target_site_for_source_pad_motion(
                    source_row,
                    source_initial_rotation,
                    T.quat2mat(initial_target_quaternion),
                    source_initial_pad_midpoint,
                    target_initial_pad_midpoint,
                    nominal_target_position,
                    target_quaternion,
                    _gripper_pad_local_offset(target_env),
                )
            else:
                pad_alignment_offset = None
            q_target, ik_info = solve_grip_site_ik(
                target_env,
                target_position,
                target_quaternion,
                initial_q=previous_q,
                config=ik_config,
            )
            previous_q = q_target.copy()
            panda_opening = source_row["panda_opening_m"]
            mapped_signal = (
                mapping.to_robotiq_signal(panda_opening)
                if panda_opening is not None
                else -1.0
            )
            source_gripper_action = float(source_row["source_gripper_action"])
            if gripper_control_mode == "source_action":
                # Preserve the expert's open/close timing. Robotiq's native
                # controller integrates this binary command at its own speed.
                # 保留专家轨迹的开合时序，避免用 Panda 开口量产生延迟。
                desired_gripper_action = float(np.sign(source_gripper_action))
                if (
                    gripper_hold_command is not None
                    and desired_gripper_action == gripper_hold_command
                ):
                    # Keep the last stable internal target after a rollback.
                    # 物理回滚后保持上一个稳定的内部目标，避免同一异常重复触发。
                    gripper_action = 0.0
                else:
                    gripper_hold_command = None
                    gripper_action = desired_gripper_action
                gripper_signal = float(
                    np.asarray(
                        target_env.robots[0].gripper["right"].current_action,
                        dtype=np.float64,
                    ).reshape(-1)[0]
                )
            elif gripper_control_mode == "state_mapping":
                gripper_signal = mapped_signal
                _set_gripper_target(target_env, gripper_signal)
                gripper_action = 0.0
            else:
                raise ValueError(
                    f"Unknown gripper_control_mode={gripper_control_mode!r}"
                )
            # The Robotiq contact model can lose the sponge when the arm is
            # teleported several centimeters between physics steps. Once the
            # closing fingers have reached the sponge, switch to an explicit
            # kinematic grasp lock until the expert opens the gripper.
            # teleport 模式下机械臂每帧可能跳过数厘米，Robotiq 接触模型会把
            # 百洁布弹落。闭合夹爪到达百洁布后建立显式保持，直到专家释放。
            if gripper_control_mode == "source_action":
                if (
                    source_gripper_action < 0.0
                    and grasp_lock is not None
                    and not release_phase
                ):
                    # Open and lift at the same time. Detach the sponge at its
                    # current board pose so it is not carried upward.
                    # 张开夹爪的同时平滑抬起机械臂；百洁布保持在当前砧板
                    # 位姿后再脱离抓取锁，避免被带到空中。
                    release_phase = True
                    release_phase_start_step = index + 1
                    release_phase_frame_count = 0
                    release_hold_q = (
                        last_applied_q.copy()
                        if last_applied_q is not None
                        else q_target.copy()
                    )
                    release_start_gripper_q = _target_gripper_qpos(target_env)
                    release_open_gripper_q = _robotiq_open_qpos(target_env)
                    collision_projection_latched = False
                    release_lift_frame_count = 0
                    release_lift_start_position, release_lift_start_quaternion = (
                        _grip_site_pose(target_env)
                    )
                    release_sponge_pose = (
                        _body_pose_matrix(target_env, sponge_body_name)
                        if sponge_body_name is not None
                        else None
                    )

                    # Estimate the height needed by the fully open gripper.
                    # 估计完全张开夹爪需要的最小安全抬升高度。
                    release_lift_distance = release_lift_clearance_m
                    board_extent = _body_collision_z_extent(
                        target_env,
                        cutting_board_body_name,
                    )
                    _apply_gripper_qpos_directly(
                        target_env,
                        release_open_gripper_q,
                    )
                    gripper_extent = _gripper_collision_z_extent(target_env)
                    if board_extent is not None and gripper_extent is not None:
                        release_lift_distance = max(
                            release_lift_distance,
                            float(board_extent[1] + 0.003 - gripper_extent[0]),
                        )
                    release_lift_target_position = (
                        release_lift_start_position.copy()
                    )
                    release_lift_target_position[2] += release_lift_distance
                    release_lift_height_m = float(
                        release_lift_target_position[2]
                    )
                    # Restore the pre-release finger state; the smooth
                    # interpolation below will open it frame by frame.
                    # 恢复释放前的指尖状态，下面逐帧平滑张开。
                    _apply_gripper_qpos_directly(
                        target_env,
                        release_start_gripper_q,
                    )
                    grasp_lock = None
                    grasp_lock_release_step = index + 1
                elif (
                    execution_mode == "teleport"
                    and grasp_lock is None
                    and not release_phase
                    and source_gripper_action > 0.0
                    and _grasp_candidate(
                        target_env,
                        sponge_body_name,
                        gripper_signal=gripper_signal,
                    )
                ):
                    grasp_lock = _capture_grasp_lock(target_env, sponge_body_name)
                    if grasp_lock is not None:
                        grasp_lock_activation_step = index + 1
            if release_phase:
                # Interpolate upward while opening, instead of making one
                # large safety lift after the gripper has opened.
                # 在夹爪张开的同时插值向上抬升，避免完全张开后一次性
                # 进行大幅安全抬升。
                release_alpha = min(
                    1.0,
                    float(release_phase_frame_count + 1)
                    / float(release_max_frames),
                )
                target_position = (
                    release_lift_start_position
                    + release_alpha
                    * (
                        release_lift_target_position
                        - release_lift_start_position
                    )
                )
                target_quaternion = release_lift_start_quaternion.copy()
                q_target, ik_info = solve_grip_site_ik(
                    target_env,
                    target_position,
                    target_quaternion,
                    initial_q=previous_q,
                    config=ik_config,
                )
                previous_q = q_target.copy()
                gripper_action = 0.0
            elif release_lift_active:
                # Continue from the final point of the smooth release lift.
                # 平滑释放抬升完成后保持安全位姿，避免 IK 回到原轨迹。
                target_position = release_lift_target_position.copy()
                target_quaternion = release_lift_start_quaternion.copy()
                q_target, ik_info = solve_grip_site_ik(
                    target_env,
                    target_position,
                    target_quaternion,
                    initial_q=previous_q,
                    config=ik_config,
                )
                previous_q = q_target.copy()
                # The fingers are already open. Keep the internal Robotiq
                # target open and do not send another close/open impulse.
                # 夹爪已经完全张开；保持内部目标为张开，不再发送重复动作，
                # 避免释放后的物理步触发无意义的关节跳变。
                _set_gripper_target(target_env, -1.0)
                gripper_action = 0.0
            if (
                grasp_lock is not None
                and not collision_projection_latched
                and not release_phase
            ):
                # Do not lift the sponge during the initial side pickup. Once
                # the expert has raised it above the board, keep the
                # non-penetration constraint latched for the wiping motion.
                # 初始从砧板侧面取百洁布时不要突然抬高；专家把它抬到砧板
                # 上方后，再锁定防穿透约束，覆盖后续擦拭阶段。
                board_extent = _body_collision_z_extent(
                    target_env,
                    cutting_board_body_name,
                )
                _apply_joint_target_directly(target_env, q_target)
                _apply_grasp_lock(target_env, grasp_lock)
                sponge_center = _body_position(target_env, sponge_body_name)
                if (
                    board_extent is not None
                    and sponge_center is not None
                    and sponge_center[2] > float(board_extent[1]) + 0.005
                ):
                    collision_projection_latched = True
                    collision_projection_activation_step = index + 1
            collision_projection = {
                "applied": False,
                "translation_world": [0.0, 0.0, 0.0],
                "passes": 0,
                "board_top_z_m": None,
                "sponge_bottom_z_m": None,
                "gripper_board_penetration_m": 0.0,
                "projection_latched": collision_projection_latched,
            }
            release_projection = {
                "applied": False,
                "translation_world": [0.0, 0.0, 0.0],
                "passes": 0,
                "board_top_z_m": None,
                "gripper_bottom_z_m": None,
            }
            if (
                execution_mode == "teleport"
                and grasp_lock is not None
                and not release_phase
            ):
                (
                    target_position,
                    target_quaternion,
                    q_target,
                    collision_projection,
                ) = _project_locked_target_above_board(
                    target_env,
                    target_position,
                    target_quaternion,
                    q_target,
                    grasp_lock,
                    board_body_name=cutting_board_body_name,
                    sponge_body_name=sponge_body_name,
                    ik_config=ik_config,
                    previous_q=previous_q,
                    projection_enabled=collision_projection_latched,
                )
                # Projection changes the EEF target, so use the projected
                # solution as the seed for the next frame.
                # 投影会改变 EEF 目标，因此下一帧从修正后的关节解继续求解。
                previous_q = q_target.copy()
            elif execution_mode == "teleport" and release_lift_active:
                (
                    target_position,
                    target_quaternion,
                    q_target,
                    release_projection,
                ) = _project_gripper_target_above_board(
                    target_env,
                    target_position,
                    target_quaternion,
                    q_target,
                    board_body_name=cutting_board_body_name,
                    ik_config=ik_config,
                    previous_q=previous_q,
                )
                previous_q = q_target.copy()
            if execution_mode == "teleport":
                # This isolates IK reachability from controller tracking lag.
                # 该模式用于先隔离 IK 可达性，不把控制器跟踪滞后混入结果。
                _apply_joint_target_directly(target_env, q_target)
                _apply_grasp_lock(target_env, grasp_lock)
            if release_phase:
                # Interpolate the complete six-joint Robotiq pose. This avoids
                # contact impulses caused by opening all passive joints at once.
                # 对 Robotiq 六个关节做平滑插值，避免一次性张开全部被动
                # 关节时产生接触冲击。
                release_alpha = min(
                    1.0,
                    float(release_phase_frame_count + 1)
                    / float(release_max_frames),
                )
                release_q = (
                    release_start_gripper_q
                    + release_alpha
                    * (release_open_gripper_q - release_start_gripper_q)
                )
                _apply_gripper_qpos_directly(target_env, release_q)
                _set_gripper_target(target_env, -1.0)
            elif release_lift_active:
                # Keep the released gripper fully open during the lift.
                # 释放后提起阶段保持夹爪全开，避免物理接触把手指重新拉回。
                _apply_gripper_qpos_directly(
                    target_env,
                    _robotiq_open_qpos(target_env),
                )
                _set_gripper_target(target_env, -1.0)
            action = np.concatenate(
                (
                    q_target.astype(np.float32),
                    np.asarray([gripper_action], dtype=np.float32),
                )
            )
            before_sponge_position = _body_position(target_env, sponge_body_name)
            raw, _, done, info = _step_with_gripper_guard(
                target_env,
                action,
            )
            if info.get("gripper_physics_rollback"):
                gripper_hold_command = desired_gripper_action if gripper_control_mode == "source_action" else None
            if execution_mode == "teleport":
                # Re-apply the kinematic IK pose after the physical substep.
                # Without this, arm gravity/contact dynamics can obscure the
                # geometric IK-reachability result.
                # teleport 模式在物理步后再次固定机械臂姿态，避免重力/接触
                # 动力学掩盖“几何 IK 是否可达”的结论。
                _apply_joint_target_directly(target_env, q_target)
                _apply_grasp_lock(target_env, grasp_lock)
                if release_lift_active:
                    _apply_gripper_qpos_directly(
                        target_env,
                        _robotiq_open_qpos(target_env),
                    )
            if release_sponge_pose is not None:
                # Keep the released sponge on the board during the short
                # opening/lifting transition.
                # 在短暂的张开/抬升过渡期间，让百洁布稳定留在砧板上。
                _set_free_body_pose(
                    target_env,
                    sponge_body_name,
                    release_sponge_pose,
                )
            if release_phase:
                release_phase_frame_count += 1
                release_lift_frame_count += 1
                release_alpha = min(
                    1.0,
                    float(release_phase_frame_count)
                    / float(release_max_frames),
                )
                release_q = (
                    release_start_gripper_q
                    + release_alpha
                    * (release_open_gripper_q - release_start_gripper_q)
                )
                _apply_gripper_qpos_directly(target_env, release_q)
                pad_separation = _gripper_pad_separation(target_env)
                if (
                    release_phase_frame_count >= release_max_frames
                ):
                    release_phase = False
                    release_phase_end_step = index + 1
                    release_hold_q = None
                    release_start_gripper_q = None
                    release_open_gripper_q = None
                    release_lift_active = True
                    release_lift_start_step = index + 2
                    release_lift_frame_count = release_lift_ramp_frames
                    release_sponge_pose = None
            current_base_position = np.asarray(
                target_env.sim.model.body_pos[base_body_id], dtype=np.float64
            )
            max_base_drift_m = max(
                max_base_drift_m,
                float(np.linalg.norm(current_base_position - locked_base_position)),
            )
            actual_position = np.asarray(target_env.sim.data.site_xpos[target_site_id], dtype=np.float64).copy()
            actual_rotation = np.asarray(target_env.sim.data.site_xmat[target_site_id], dtype=np.float64).reshape(3, 3)
            actual_quaternion = T.mat2quat(actual_rotation)
            position_error = float(np.linalg.norm(actual_position - target_position))
            orientation_error = orientation_error_rad(target_quaternion, actual_quaternion)
            actual_gripper_qpos = _target_gripper_qpos(target_env)
            actual_sponge_position = _body_position(target_env, sponge_body_name)
            gripper_qpos_step = float(info.get("gripper_qpos_step_rad", 0.0))
            sponge_step = (
                float(
                    np.linalg.norm(
                        actual_sponge_position - before_sponge_position
                    )
                )
                if actual_sponge_position is not None
                and before_sponge_position is not None
                else None
            )
            sponge_velocity = (
                float(
                    np.linalg.norm(
                        np.asarray(
                            target_env.sim.data.get_body_xvelp(sponge_body_name)
                        )
                    )
                )
                if sponge_body_name is not None
                else None
            )
            position_errors.append(position_error)
            orientation_errors.append(orientation_error)
            env_success = bool(target_env._check_success()) or env_success
            rows.append(
                {
                    "step": index + 1,
                    "target_position_world": target_position.tolist(),
                    "nominal_target_position_world": nominal_target_position.tolist(),
                    "target_quaternion_world_xyzw": target_quaternion.tolist(),
                    "actual_position_world": actual_position.tolist(),
                    "actual_quaternion_world_xyzw": actual_quaternion.tolist(),
                    "position_error_m": position_error,
                    "orientation_error_rad": orientation_error,
                    "q_target": q_target.tolist(),
                    "actual_arm_qpos": np.asarray(target_env.sim.data.qpos[target_robot._ref_arm_joint_pos_indexes]).tolist(),
                    "panda_opening_m": panda_opening,
                    "robotiq_signal": gripper_signal,
                    "mapped_robotiq_signal": mapped_signal,
                    "source_gripper_action": source_gripper_action,
                    "gripper_action": gripper_action,
                    "gripper_control_mode": gripper_control_mode,
                    "gripper_qpos": actual_gripper_qpos.tolist(),
                    "gripper_qpos_step_rad": gripper_qpos_step,
                    "gripper_qpos_step_applied_rad": (
                        0.0
                        if info.get("gripper_physics_rollback")
                        else gripper_qpos_step
                    ),
                    "gripper_physics_rollback": bool(
                        info.get("gripper_physics_rollback", False)
                    ),
                    "gripper_physics_rollback_reason": info.get(
                        "gripper_physics_rollback_reason"
                    ),
                    "grasp_lock_active": grasp_lock is not None,
                    "grasp_lock_activation_step": grasp_lock_activation_step,
                    "grasp_lock_release_step": grasp_lock_release_step,
                    "release_phase": release_phase,
                    "release_phase_start_step": release_phase_start_step,
                    "release_phase_end_step": release_phase_end_step,
                    "release_phase_frame_count": release_phase_frame_count,
                    "release_pad_separation_m": _gripper_pad_separation(target_env),
                    "release_open_separation_threshold_m": release_open_separation_m,
                    "release_lift_active": release_lift_active,
                    "release_lift_start_step": release_lift_start_step,
                    "release_lift_height_m": release_lift_height_m,
                    "release_projection": release_projection,
                    "sponge_body_name": sponge_body_name,
                    "cutting_board_body_name": cutting_board_body_name,
                    "sponge_position_world": (
                        actual_sponge_position.tolist()
                        if actual_sponge_position is not None
                        else None
                    ),
                    "sponge_step_m": sponge_step,
                    "sponge_speed_m_s": sponge_velocity,
                    "collision_projection": collision_projection,
                    "collision_projection_activation_step": (
                        collision_projection_activation_step
                    ),
                    "pad_midpoint_alignment_offset_world": (
                        np.asarray(pad_alignment_offset, dtype=np.float64).tolist()
                        if pad_alignment_offset is not None
                        else None
                    ),
                    "source_grip_pad_geometry": source_row.get(
                        "grip_pad_geometry", {}
                    ),
                    "actual_grip_pad_geometry": _gripper_pad_geometry(target_env),
                    "ik": ik_info,
                    "execution_mode": execution_mode,
                    "done": bool(done),
                    "step_info_keys": sorted(info.keys()) if isinstance(info, dict) else [],
                }
            )
            last_applied_q = q_target.copy()
            if video_writers:
                # Render both cameras at the requested output resolution.
                # 按指定分辨率同时渲染两个相机，并分别写入视频。
                for camera_name, writer in video_writers.items():
                    frame = np.asarray(
                        target_env.sim.render(
                            height=camera_height,
                            width=camera_width,
                            camera_name=(
                                "robot0_agentview_left"
                                if camera_name == "agentview_left"
                                else "robot0_eye_in_hand"
                            ),
                        ),
                        dtype=np.uint8,
                    )[::-1].copy()
                    writer.append_data(frame)

        result = {
            "task": task_name,
            "episode_index": episode.episode_index,
            "robot": ROBOT_NAME,
            "steps": len(rows),
            "franka_expert_success": bool(np.any(episode.dones)),
            "ur3e_env_success": bool(env_success),
            "scene_state_copy": scene_copy,
            "initial_alignment": alignment,
            "locked_base_position_world": locked_base_position.tolist(),
            "max_base_drift_after_initial_alignment_m": max_base_drift_m,
            "base_moved_after_initial_alignment": bool(max_base_drift_m > 1e-12),
            "execution_mode": execution_mode,
            "align_to_pad_midpoint": bool(align_to_pad_midpoint),
            "gripper_control_mode": gripper_control_mode,
            "video_paths": video_paths,
            "initial_target_pad_geometry": initial_target_pad_geometry,
            "target_pad_local_offset_in_site": (
                np.asarray(target_pad_local_offset, dtype=np.float64).tolist()
                if target_pad_local_offset is not None
                else None
            ),
            "sponge_body_name": sponge_body_name,
            "cutting_board_body_name": cutting_board_body_name,
            "release_phase_start_step": release_phase_start_step,
            "release_phase_end_step": release_phase_end_step,
            "release_open_separation_threshold_m": release_open_separation_m,
            "release_lift_start_step": release_lift_start_step,
            "release_lift_height_m": release_lift_height_m,
            "collision_projection_count": int(
                sum(
                    bool(row["collision_projection"].get("applied", False))
                    for row in rows
                )
            ),
            "max_collision_projection_m": float(
                max(
                    np.linalg.norm(
                        np.asarray(
                            row["collision_projection"]["translation_world"],
                            dtype=np.float64,
                        )
                    )
                    for row in rows
                )
            ),
            "diagnostics": {
                "max_gripper_qpos_step_rad": float(
                    max(row["gripper_qpos_step_rad"] for row in rows)
                ),
                "max_applied_gripper_qpos_step_rad": float(
                    max(row["gripper_qpos_step_applied_rad"] for row in rows)
                ),
                "mean_gripper_qpos_step_rad": float(
                    np.mean([row["gripper_qpos_step_rad"] for row in rows])
                ),
                "gripper_physics_rollback_count": int(
                    sum(row["gripper_physics_rollback"] for row in rows)
                ),
                "max_sponge_step_m": float(
                    max(
                        row["sponge_step_m"]
                        for row in rows
                        if row["sponge_step_m"] is not None
                    )
                )
                if any(row["sponge_step_m"] is not None for row in rows)
                else None,
                "max_pad_midpoint_error_m": float(
                    max(
                        np.linalg.norm(
                            np.asarray(
                                row["actual_grip_pad_geometry"]["midpoint_world"],
                                dtype=np.float64,
                            )
                            - np.asarray(
                                row["source_grip_pad_geometry"]["midpoint_world"],
                                dtype=np.float64,
                            )
                        )
                        for row in rows
                        if row["actual_grip_pad_geometry"].get("midpoint_world")
                        is not None
                        and row["source_grip_pad_geometry"].get("midpoint_world")
                        is not None
                    )
                )
                if any(
                    row["actual_grip_pad_geometry"].get("midpoint_world") is not None
                    and row["source_grip_pad_geometry"].get("midpoint_world")
                    is not None
                    for row in rows
                )
                else None,
            },
            "gripper_mapping": {
                "panda_opening_range_m": [0.0, 0.08],
                "robotiq_signal_range": [-1.0, 1.0],
                "description": "Panda opening [0.08=open, 0.0=closed] -> Robotiq [-1=open, +1=closed]",
            },
            "position_error_m": {
                "mean": float(np.mean(position_errors)),
                "max": float(np.max(position_errors)),
                "final": float(position_errors[-1]),
            },
            "orientation_error_rad": {
                "mean": float(np.mean(orientation_errors)),
                "max": float(np.max(orientation_errors)),
                "final": float(orientation_errors[-1]),
            },
            "franka": franka,
            "rows": rows,
        }
    finally:
        for writer in video_writers.values():
            writer.close()
        target_env.close()

    (output_dir / "replay.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
