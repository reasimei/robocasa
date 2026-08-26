#!/usr/bin/env python3
"""Run Xiaomi RoboCasa365 with a fixed, elevated UR3e.

This file is intentionally separate from the original PreSoakPan smoke test.
It places the UR3e base on the counter-height plane and records a third-person
camera so the whole arm is visible in the review video.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robocasa.utils import env_utils
from robocasa.utils.dataset_registry_utils import get_task_horizon
from robosuite.robots import ROBOT_CLASS_MAPPING
from robosuite.robots.fixed_base_robot import FixedBaseRobot
from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils import transform_utils as T
from robosuite.utils.transform_utils import euler2mat, mat2quat

from scripts.long_horizon_controller.run_composite_seen_eval import (
    DEFAULT_TASK_INSTRUCTIONS,
)
from scripts.long_horizon_controller.xiaomi_policy_adapter import XiaomiPolicyAdapter
from scripts.ur3e_robocasa_eval.ur3e_official_robot import UR3eOfficialFixed


DEFAULT_MODEL_PATH = (
    "/data/zjw/workspace/Isaac-GR00T/expdata/"
    "Xiaomi-Robotics-1-RoboCasa365"
)
DEFAULT_TASKS = ("PreSoakPan", "StackBowlsCabinet")
_FIXED_BASE_Y_OFFSET = 0.0


class UR3eFixed(ManipulatorModel):
    """UR3e model with a separate world-space review camera.

    The original ``UR3e`` registration and XML are left untouched.  The
    camera is only for the new evaluator's videos; Xiaomi still receives the
    same three RoboCasa observation cameras as before.
    """

    arms = ["right"]

    def __init__(self, idn=0):
        source = Path(__file__).with_name("ur3e.xml")
        tree = ET.parse(source)
        worldbody = tree.getroot().find("worldbody")
        if worldbody is None:
            raise RuntimeError(f"Missing worldbody in {source}")
        # RoboSuite's offscreen renderer enables visual mesh group 1.  The
        # standalone UR3e XML intentionally has no group attributes, which
        # puts these custom visual links in group 0 and hides the arm.
        for geom in worldbody.iter("geom"):
            name = geom.get("name", "")
            if name.endswith("_vis") or name == "base_col":
                geom.set("group", "1")
        # The lightweight standalone arm does not have the full UR controller
        # gravity compensation model. Keep its links at the deterministic
        # reset posture while the fixed-base evaluator is idling between
        # policy actions.
        for body in worldbody.iter("body"):
            if body.get("name") not in {"base", "fixed_base_link"}:
                body.set("gravcomp", "1")
        worldbody.append(
            ET.Element(
                "camera",
                {
                    "name": "review_camera",
                    "mode": "fixed",
                    "pos": "3.05 -3.25 1.9",
                    "quat": "1 0 0 0",
                    "fovy": "60",
                },
            )
        )
        temp_name = f"ur3e_fixed_review_{os.getpid()}_{idn}.xml"
        temp_path = Path(tempfile.gettempdir()) / temp_name
        tree.write(temp_path, encoding="utf-8", xml_declaration=True)
        super().__init__(str(temp_path), idn=idn)

    @property
    def default_base(self):
        return "NullMount"

    @property
    def default_gripper(self):
        return {"right": "Robotiq85Gripper"}

    @property
    def default_controller_config(self):
        return {"right": "default_ur5e"}

    @property
    def init_qpos(self):
        return np.array([-0.35, -1.35, 1.75, -1.95, -1.57, -1.55], dtype=np.float64)

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.4, 0.0, 0),
            "empty": (-0.4, 0.0, 0),
            "table": lambda table_length: (-0.4 - table_length / 2, 0, 0),
        }

    @property
    def top_offset(self):
        return np.array((0.0, 0.0, 0.65), dtype=np.float64)

    @property
    def _horizontal_radius(self):
        return 0.45

    @property
    def arm_type(self):
        return "single"


def register_ur3e(base_z: float, base_y_offset: float) -> None:
    """Register and place only the custom UR3e in this process."""
    global _FIXED_BASE_Y_OFFSET
    _FIXED_BASE_Y_OFFSET = float(base_y_offset)
    ROBOT_CLASS_MAPPING["UR3eOfficialFixed"] = FixedBaseRobot
    env_utils._ROBOT_POS_OFFSETS["UR3eOfficialFixed"] = [0.0, 0.0, float(base_z)]
    env_utils.set_robot_base = _set_fixed_robot_base


def _set_fixed_robot_base(
    env: Any,
    anchor_pos: np.ndarray,
    anchor_ori: np.ndarray,
    rot_dev: float,
    pos_dev_x: float,
    pos_dev_y: float,
) -> np.ndarray:
    del rot_dev, pos_dev_x, pos_dev_y
    body_id = env.sim.model.body_name2id("robot0_base")
    fixed_pos = np.asarray(anchor_pos, dtype=np.float64).copy()
    fixed_pos[1] += _FIXED_BASE_Y_OFFSET
    env.sim.model.body_pos[body_id] = fixed_pos
    # RoboSuite returns xyzw; MuJoCo stores wxyz.
    env.sim.model.body_quat[body_id] = mat2quat(
        euler2mat(np.asarray(anchor_ori, dtype=np.float64))
    )[[3, 0, 1, 2]]
    env.sim.forward()
    return fixed_pos


def raw_to_policy_observation(raw: dict[str, Any], env: Any) -> dict[str, Any]:
    """Translate raw RoboSuite observations to the Xiaomi RoboCasa365 schema."""
    base_pos = np.asarray(env.sim.data.get_body_xpos("robot0_base"), dtype=np.float32)
    base_quat = T.convert_quat(
        np.asarray(env.sim.data.get_body_xquat("robot0_base"), dtype=np.float32),
        to="xyzw",
    )
    base_pose = T.pose2mat((base_pos, base_quat))
    eef_pose = T.pose2mat(
        (
            np.asarray(raw["robot0_eef_pos"], dtype=np.float32),
            np.asarray(raw["robot0_eef_quat"], dtype=np.float32),
        )
    )
    relative_pose = T.pose_inv(base_pose).dot(eef_pose)
    gripper_qpos = np.asarray(raw["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
    if gripper_qpos.size >= 4:
        gripper_state = gripper_qpos[[0, 3]]
    else:
        gripper_state = np.pad(gripper_qpos, (0, max(0, 2 - gripper_qpos.size)))[:2]

    def camera_frame(key: str) -> np.ndarray:
        return np.asarray(raw[key], dtype=np.uint8)[::-1, :, :].copy()

    return {
        "state.end_effector_position_relative": relative_pose[:3, 3],
        "state.end_effector_rotation_relative": T.mat2quat(relative_pose[:3, :3]),
        "state.gripper_qpos": gripper_state,
        "state.base_position": np.zeros(3, dtype=np.float32),
        "state.base_rotation": np.array([0, 0, 0, 1], dtype=np.float32),
        "video.robot0_agentview_left": camera_frame("robot0_agentview_left_image"),
        "video.robot0_agentview_right": camera_frame("robot0_agentview_right_image"),
        "video.robot0_eye_in_hand": camera_frame("robot0_eye_in_hand_image"),
    }


def _resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame.shape[:2] == (height, width):
        return frame
    from PIL import Image

    return np.asarray(
        Image.fromarray(frame).resize((width, height), Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )


def _look_at_quat_wxyz(camera_pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return MuJoCo's wxyz camera quaternion looking along local -Z."""
    forward = np.asarray(target, dtype=np.float64) - np.asarray(camera_pos, dtype=np.float64)
    forward /= max(np.linalg.norm(forward), 1e-12)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    right /= max(np.linalg.norm(right), 1e-12)
    up = np.cross(right, forward)
    rotation = np.column_stack((right, up, -forward))
    quat_xyzw = T.mat2quat(rotation)
    return quat_xyzw[[3, 0, 1, 2]]


def configure_review_camera(env: Any, task_name: str) -> None:
    """Place a stable third-person camera in front of the active work area."""
    if task_name == "StackBowlsCabinet":
        # Keep the camera in front of the cabinet and slightly to its side.
        # The previous oblique position looked directly into the cabinet side
        # panel and hid the entire fixed arm behind it.
        target = np.array([3.05, -1.05, 1.10], dtype=np.float64)
        camera_pos = np.array([4.70, -2.65, 1.45], dtype=np.float64)
    elif task_name == "PreSoakPan":
        target = np.asarray(env.sink.pos, dtype=np.float64)
        target[2] = 0.95
        # The kitchen origin changes with layout/style. The old negative-x
        # offset put layout 4/style 4 inside a cabinet. Use a far-side
        # diagonal view that stays outside the counter geometry.
        camera_pos = target + np.array([2.20, -2.80, 1.60])
    else:
        target = np.asarray(env.sim.data.get_body_xpos("robot0_base"), dtype=np.float64)
        target[2] += 0.25
        camera_pos = target + np.array([0.0, -2.8, 0.75])

    camera_name = "robot0_review_camera"
    camera_id = env.sim.model.camera_name2id(camera_name)
    env.sim.model.cam_pos[camera_id] = camera_pos
    env.sim.model.cam_quat[camera_id] = _look_at_quat_wxyz(camera_pos, target)
    env.sim.forward()


def _load_target_eef_pose(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the first recorded world-frame EEF pose from a trajectory JSON."""
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"EEF trajectory is empty or invalid: {path}")
    row = rows[0]
    try:
        position = np.asarray(row["eef_pos_world"], dtype=np.float64)
        quaternion = np.asarray(row["eef_quat_world_xyzw"], dtype=np.float64)
    except KeyError as exc:
        raise KeyError(f"Missing {exc.args[0]!r} in {path}") from exc
    if position.shape != (3,) or quaternion.shape != (4,):
        raise ValueError(
            f"Invalid EEF pose shapes in {path}: "
            f"position={position.shape}, quaternion={quaternion.shape}"
        )
    quaternion /= max(np.linalg.norm(quaternion), 1e-12)
    return position, quaternion


def _current_eef_pose(env: Any) -> tuple[np.ndarray, np.ndarray]:
    """Read the same EEF position / orientation fields used in the rollout."""
    robot = env.robots[0]
    arm = "right"
    site_id = robot.eef_site_id[arm]
    body_name = robot.robot_model.eef_name[arm]
    position = np.asarray(env.sim.data.site_xpos[site_id], dtype=np.float64)
    quaternion = T.convert_quat(
        np.asarray(env.sim.data.get_body_xquat(body_name), dtype=np.float64),
        to="xyzw",
    )
    quaternion /= max(np.linalg.norm(quaternion), 1e-12)
    return position, quaternion


def align_initial_eef_to(
    env: Any,
    target_position: np.ndarray,
    target_quaternion_xyzw: np.ndarray,
    *,
    max_iterations: int = 800,
    position_tolerance: float = 0.002,
    orientation_tolerance: float = 0.02,
    damping: float = 0.04,
    step_scale: float = 0.7,
    allow_base_translation: bool = True,
    base_translation_weight: float = 0.02,
    max_base_step: float = 0.03,
) -> dict[str, Any]:
    """Align the UR3e EEF with damped least-squares IK.

    The position Jacobian is taken at the Robotiq grip site, while the
    orientation is taken from the UR3e EEF body. This matches RoboSuite's
    ``robot0_eef_pos`` and ``robot0_eef_quat`` observation conventions.

    When explicitly aligning to a Franka pose, the fixed base may also be
    translated in world XYZ. The base translation has an identity Jacobian
    for EEF position and zero Jacobian for orientation. A regularizer keeps
    the solved base close to its nominal installation pose.
    """
    robot = env.robots[0]
    arm = "right"
    eef_body = robot.robot_model.eef_name[arm]
    eef_site = env.sim.model.site_id2name(robot.eef_site_id[arm])
    qpos_indexes = np.asarray(robot._ref_arm_joint_pos_indexes, dtype=np.int64)
    qvel_indexes = np.asarray(robot._ref_arm_joint_vel_indexes, dtype=np.int64)
    joint_indexes = np.asarray(robot._ref_arm_joint_indexes, dtype=np.int64)
    if qpos_indexes.size != 6 or qvel_indexes.size != 6:
        raise RuntimeError(
            f"Expected a 6-DoF UR3e arm, got qpos={qpos_indexes}, qvel={qvel_indexes}"
        )

    target_position = np.asarray(target_position, dtype=np.float64)
    target_quaternion_xyzw = np.asarray(target_quaternion_xyzw, dtype=np.float64)
    target_quaternion_xyzw /= max(np.linalg.norm(target_quaternion_xyzw), 1e-12)
    base_body_id = env.sim.model.body_name2id("robot0_base")
    start_base_position = np.asarray(
        env.sim.model.body_pos[base_body_id], dtype=np.float64
    ).copy()

    def errors() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        current_position, current_quaternion = _current_eef_pose(env)
        position_error = target_position - current_position
        orientation_error = T.get_orientation_error(
            target_quaternion_xyzw, current_quaternion
        )
        return (
            current_position,
            current_quaternion,
            position_error,
            orientation_error,
        )

    start_qpos = np.asarray(env.sim.data.qpos[qpos_indexes], dtype=np.float64).copy()
    _, _, start_position_error, start_orientation_error = errors()
    iterations = 0
    converged = False

    for iterations in range(1, max_iterations + 1):
        (
            _current_position,
            _current_quaternion,
            position_error,
            orientation_error,
        ) = errors()
        if (
            np.linalg.norm(position_error) <= position_tolerance
            and np.linalg.norm(orientation_error) <= orientation_tolerance
        ):
            converged = True
            break

        jacobian_position = env.sim.data.get_site_jacp(eef_site).reshape(
            (3, -1)
        )[:, qvel_indexes]
        jacobian_orientation = env.sim.data.get_body_jacr(eef_body).reshape(
            (3, -1)
        )[:, qvel_indexes]
        arm_jacobian = np.vstack((jacobian_position, jacobian_orientation))
        task_error = np.concatenate((position_error, orientation_error))

        if allow_base_translation:
            # The base position is a world-frame root-body translation.
            jacobian = np.column_stack(
                (arm_jacobian, np.vstack((np.eye(3), np.zeros((3, 3)))))
            )
            regularizer = np.diag(
                np.concatenate(
                    (
                        np.full(6, damping**2, dtype=np.float64),
                        np.full(
                            3,
                            (damping * base_translation_weight) ** 2,
                            dtype=np.float64,
                        ),
                    )
                )
            )
        else:
            jacobian = arm_jacobian
            regularizer = (damping**2) * np.eye(6)

        # Regularized least-squares in the augmented arm-plus-base space.
        lhs = jacobian.T @ jacobian + regularizer
        delta = np.linalg.solve(lhs, jacobian.T @ task_error)
        delta *= step_scale
        delta_q = delta[:6]
        delta_norm = np.linalg.norm(delta_q)
        if delta_norm > 0.15:
            delta_q *= 0.15 / delta_norm

        if allow_base_translation:
            delta_base = delta[6:]
            base_norm = np.linalg.norm(delta_base)
            if base_norm > max_base_step:
                delta_base *= max_base_step / base_norm
        else:
            delta_base = np.zeros(3, dtype=np.float64)

        qpos = np.asarray(env.sim.data.qpos[qpos_indexes], dtype=np.float64)
        qpos += delta_q
        ranges = np.asarray(env.sim.model.jnt_range[joint_indexes], dtype=np.float64)
        for index, (lower, upper) in enumerate(ranges):
            if lower < upper:
                qpos[index] = np.clip(qpos[index], lower + 1e-5, upper - 1e-5)
        env.sim.data.qpos[qpos_indexes] = qpos
        env.sim.data.qvel[qvel_indexes] = 0.0
        if allow_base_translation:
            env.sim.model.body_pos[base_body_id] = (
                np.asarray(env.sim.model.body_pos[base_body_id], dtype=np.float64)
                + delta_base
            )
        env.sim.forward()

    (
        solved_position,
        solved_quaternion,
        final_position_error,
        final_orientation_error,
    ) = errors()
    converged = bool(
        converged
        or (
            np.linalg.norm(final_position_error) <= position_tolerance
            and np.linalg.norm(final_orientation_error) <= orientation_tolerance
        )
    )

    # Refresh controller state after manually changing qpos. reset() here only
    # clears controller goals; it does not restore the robot's qpos.
    robot.composite_controller.update_state()
    robot.composite_controller.reset()

    return {
        "target_position_world": target_position.tolist(),
        "target_quaternion_world_xyzw": target_quaternion_xyzw.tolist(),
        "start_position_error_m": float(np.linalg.norm(start_position_error)),
        "start_orientation_error_rad": float(np.linalg.norm(start_orientation_error)),
        "solved_position_world": solved_position.tolist(),
        "solved_quaternion_world_xyzw": solved_quaternion.tolist(),
        "final_position_error_m": float(np.linalg.norm(final_position_error)),
        "final_orientation_error_rad": float(np.linalg.norm(final_orientation_error)),
        "solved_arm_qpos": np.asarray(env.sim.data.qpos[qpos_indexes]).tolist(),
        "start_base_position_world": start_base_position.tolist(),
        "solved_base_position_world": np.asarray(
            env.sim.model.body_pos[base_body_id], dtype=np.float64
        ).tolist(),
        "base_translation_world": (
            np.asarray(env.sim.model.body_pos[base_body_id], dtype=np.float64)
            - start_base_position
        ).tolist(),
        "allow_base_translation": bool(allow_base_translation),
        "iterations": iterations,
        "converged": converged,
        "eef_body": eef_body,
        "eef_site": eef_site,
    }


def run_episode(
    args: argparse.Namespace,
    task_name: str,
    episode_index: int,
    policy: XiaomiPolicyAdapter,
) -> dict[str, Any]:
    seed = args.seed_base + episode_index
    episode_dir = (
        Path(args.output_root)
        / task_name
        / "episodes"
        / f"episode_{episode_index:03d}_seed_{seed}"
    )
    result_path = episode_dir / "result.json"
    if result_path.exists() and not args.overwrite:
        return json.loads(result_path.read_text(encoding="utf-8"))
    episode_dir.mkdir(parents=True, exist_ok=True)

    horizon = args.max_episode_steps or get_task_horizon(task_name)
    env = None
    writer = None
    steps = 0
    success = False
    error = None
    alignment: dict[str, Any] | None = None
    scene: dict[str, Any] | None = None
    trajectory: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        env_kwargs = dict(
            camera_names=[
                "robot0_agentview_left",
                "robot0_agentview_right",
                "robot0_eye_in_hand",
            ],
            camera_widths=args.camera_width,
            camera_heights=args.camera_height,
            render_camera="robot0_agentview_left",
            control_freq=20,
            # Keep the legacy UR3e evaluator's reset behavior unchanged.
            # Alignment mode uses the exact nominal posture as its IK seed.
            initialization_noise=(
                None if args.align_initial_eef_to is not None else "default"
            ),
        )
        if args.layout_id is not None:
            env_kwargs.update(
                split=None,
                obj_instance_split=args.split,
                layout_and_style_ids=[(args.layout_id, args.style_id)],
            )
        else:
            env_kwargs["split"] = args.split
        env = env_utils.create_env(
            task_name,
            robots="UR3eOfficialFixed",
            seed=seed,
            **env_kwargs,
        )
        raw_observation = env.reset()
        scene = {
            "layout_id": int(env.layout_id),
            "style_id": (
                int(env.style_id)
                if isinstance(env.style_id, (int, np.integer))
                else str(env.style_id)
            ),
        }
        if args.align_initial_eef_to is not None:
            target_position, target_quaternion = _load_target_eef_pose(
                args.align_initial_eef_to
            )
            alignment = align_initial_eef_to(
                env,
                target_position,
                target_quaternion,
            )
            (episode_dir / "initial_alignment.json").write_text(
                json.dumps(alignment, indent=2),
                encoding="utf-8",
            )
            # The manually solved qpos is now reflected in the first policy
            # observation. Re-read it instead of using the pre-IK reset frame.
            raw_observation = env._get_observations(force_update=True)
        configure_review_camera(env, task_name)
        policy.reset()
        instruction = DEFAULT_TASK_INSTRUCTIONS.get(task_name)
        if instruction is None:
            instruction = env.get_ep_meta().get("lang", task_name)

        if args.save_video:
            import imageio.v2 as imageio

            video_path = episode_dir / f"{task_name}_ur3e_seed_{seed}.mp4"
            writer = imageio.get_writer(video_path, fps=20, codec="libx264")

        while steps < horizon:
            observation = raw_to_policy_observation(raw_observation, env)
            _, action_chunk = policy.act(observation, instruction)
            action_chunk = np.asarray(action_chunk)
            if action_chunk.ndim != 2 or action_chunk.shape[1] < 7:
                raise ValueError(f"Unexpected Xiaomi action chunk shape: {action_chunk.shape}")

            for action_row in action_chunk:
                if steps >= horizon:
                    break
                env_action = np.asarray(action_row[:7], dtype=np.float32)
                raw_observation, _, done, _ = env.step(env_action)
                steps += 1
                trajectory.append(
                    {
                        "step": steps,
                        "eef_pos_world": np.asarray(
                            raw_observation["robot0_eef_pos"], dtype=np.float32
                        ).tolist(),
                        "eef_quat_world_xyzw": np.asarray(
                            raw_observation["robot0_eef_quat"], dtype=np.float32
                        ).tolist(),
                        "eef_pos_base": np.asarray(
                            raw_observation.get(
                                "robot0_base_to_eef_pos",
                                raw_observation["robot0_eef_pos"],
                            ),
                            dtype=np.float32,
                        ).tolist(),
                        "eef_quat_base_xyzw": np.asarray(
                            raw_observation.get(
                                "robot0_base_to_eef_quat",
                                raw_observation["robot0_eef_quat"],
                            ),
                            dtype=np.float32,
                        ).tolist(),
                        "action": env_action.tolist(),
                    }
                )
                if writer is not None:
                    frame = np.asarray(
                        env.sim.render(
                            height=args.video_height,
                            width=args.video_width,
                            camera_name="robot0_review_camera",
                        ),
                        dtype=np.uint8,
                    )[::-1, :, :].copy()
                    writer.append_data(
                        _resize_frame(frame, args.video_width, args.video_height)
                    )
                success = bool(env._check_success())
                if success or bool(done):
                    break
            if success:
                break
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if writer is not None:
            writer.close()
        if env is not None:
            env.close()

    result = {
        "task_name": task_name,
        "robot": "UR3eOfficialFixed",
        "policy": "Xiaomi-Robotics-1-RoboCasa365",
        "fixed_base_height": args.base_z,
        "fixed_base_y_offset": args.base_y_offset,
        "scene": scene,
        "align_initial_eef_to": (
            str(args.align_initial_eef_to)
            if args.align_initial_eef_to is not None
            else None
        ),
        "initial_alignment": alignment,
        "episode_index": episode_index,
        "seed": seed,
        "env_success": bool(success and error is None),
        "steps": steps,
        "elapsed_sec": time.perf_counter() - started,
        "error": error,
        "output_dir": str(episode_dir),
    }
    (episode_dir / "eef_trajectory.json").write_text(
        json.dumps(trajectory, indent=2),
        encoding="utf-8",
    )
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--split", default="target", choices=["target", "pretrain"])
    parser.add_argument("--n-episodes", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument(
        "--output-root",
        default=(
            "/data/zjw/workspace/Isaac-GR00T/expdata/"
            "ur3e_robocasa_fixed_counter_height"
        ),
    )
    parser.add_argument("--history-length", type=int, default=4)
    parser.add_argument("--n-action-steps", type=int, default=16)
    parser.add_argument("--num-diffusion-steps", type=int, default=5)
    parser.add_argument("--max-episode-steps", type=int, default=0)
    parser.add_argument("--camera-width", type=int, default=128)
    parser.add_argument("--camera-height", type=int, default=128)
    parser.add_argument("--video-width", type=int, default=512)
    parser.add_argument("--video-height", type=int, default=512)
    parser.add_argument("--base-z", type=float, default=0.92)
    parser.add_argument(
        "--layout-id",
        type=int,
        default=None,
        help="Fix the RoboCasa kitchen layout. Must be used with --style-id.",
    )
    parser.add_argument(
        "--style-id",
        type=int,
        default=None,
        help="Fix the RoboCasa kitchen style. Must be used with --layout-id.",
    )
    parser.add_argument(
        "--base-y-offset",
        type=float,
        default=-0.6,
        help="Additional world-y offset for the isolated fixed-base robot.",
    )
    parser.add_argument(
        "--align-initial-eef-to",
        type=Path,
        default=None,
        help=(
            "Trajectory JSON whose first EEF pose is used as the UR3e "
            "initial world-frame IK target."
        ),
    )
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.layout_id is None) != (args.style_id is None):
        raise ValueError("--layout-id and --style-id must be provided together")
    register_ur3e(args.base_z, args.base_y_offset)
    policy = XiaomiPolicyAdapter(
        model_path=args.model_path,
        history_length=args.history_length,
        action_steps=args.n_action_steps,
        num_diffusion_steps=args.num_diffusion_steps,
    )
    all_rows: list[dict[str, Any]] = []
    for task_name in args.tasks:
        for episode_index in range(args.n_episodes):
            print(
                f"[ur3e-fixed] {task_name} episode "
                f"{episode_index + 1}/{args.n_episodes}",
                flush=True,
            )
            row = run_episode(args, task_name, episode_index, policy)
            all_rows.append(row)
            print(json.dumps(row, indent=2), flush=True)

    by_task: dict[str, dict[str, Any]] = {}
    for task_name in args.tasks:
        rows = [row for row in all_rows if row["task_name"] == task_name]
        successes = sum(bool(row["env_success"]) for row in rows)
        by_task[task_name] = {
            "num_episodes": len(rows),
            "successes": successes,
            "success_rate": successes / len(rows) if rows else 0.0,
        }
    summary = {
        "robot": "UR3eOfficialFixed",
        "policy": "Xiaomi-Robotics-1-RoboCasa365",
        "fixed_base_height": args.base_z,
        "fixed_base_y_offset": args.base_y_offset,
        "results_by_task": by_task,
        "results": all_rows,
    }
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
