#!/usr/bin/env python3
"""Diagnose object motion around the UR3e grasp in an expert replay.

This intentionally runs only a short window. It uses the same grip-site IK
replay as replay_lerobot_expert_franka_ur3e.py and records the pan pose,
velocity, robot pose error, gripper state, and contact pairs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robocasa.utils import env_utils
from robosuite.utils import transform_utils as T

from scripts.ur3e_robocasa_eval.replay_lerobot_expert_franka_ur3e import (
    _eef_delta_action,
    _extract_common_scene_state,
    _grip_site_pose,
    _gripper_summary,
    _load_expert,
    _make_env,
    _make_writer,
    _render,
    _restore_expert_xml_state,
    _apply_common_scene_state,
    _safe_env_step,
)
from scripts.ur3e_robocasa_eval.run_xiaomi_ur3e_fixed_eval import (
    align_initial_eef_to,
    register_ur3e,
)


def _joint_info(env, joint_name: str):
    joint_id = env.sim.model.joint_name2id(joint_name)
    qadr = int(env.sim.model.jnt_qposadr[joint_id])
    vadr = int(env.sim.model.jnt_dofadr[joint_id])
    return joint_id, qadr, vadr


def _object_state(env, joint_name: str) -> dict:
    joint_id, qadr, vadr = _joint_info(env, joint_name)
    qwidth = 7 if int(env.sim.model.jnt_type[joint_id]) == 0 else 1
    vwidth = 6 if int(env.sim.model.jnt_type[joint_id]) == 0 else 1
    body_name = env.sim.model.jnt_bodyid[joint_id]
    body_name = env.sim.model.body_id2name(int(body_name))
    return {
        "joint": joint_name,
        "qpos": np.asarray(
            env.sim.data.qpos[qadr : qadr + qwidth], dtype=np.float64
        ).tolist(),
        "qvel": np.asarray(
            env.sim.data.qvel[vadr : vadr + vwidth], dtype=np.float64
        ).tolist(),
        "body": body_name,
        "body_pos": np.asarray(
            env.sim.data.get_body_xpos(body_name), dtype=np.float64
        ).tolist(),
        "body_quat_wxyz": np.asarray(
            env.sim.data.get_body_xquat(body_name), dtype=np.float64
        ).tolist(),
    }


def _contacts(env) -> list[dict]:
    contacts = []
    for index in range(int(env.sim.data.ncon)):
        contact = env.sim.data.contact[index]
        geom1 = env.sim.model.geom_id2name(int(contact.geom1))
        geom2 = env.sim.model.geom_id2name(int(contact.geom2))
        names = {geom1 or "", geom2 or ""}
        if any("obj1" in name or "gripper0_right" in name for name in names):
            contacts.append(
                {
                    "geom1": geom1,
                    "geom2": geom2,
                    "dist": float(contact.dist),
                    "includemargin": float(contact.includemargin),
                    "friction": np.asarray(contact.friction).tolist(),
                }
            )
    return contacts


def _quat_error(target, actual) -> float:
    return float(
        np.linalg.norm(
            T.get_orientation_error(
                np.asarray(target, dtype=np.float64),
                np.asarray(actual, dtype=np.float64),
            )
        )
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            "/data/zjw/workspace/robocasa/datasets/v1.0/target/composite/"
            "PreSoakPan/20250809/lerobot"
        ),
    )
    parser.add_argument("--episode-index", type=int, default=237)
    parser.add_argument("--start-step", type=int, default=90)
    parser.add_argument("--end-step", type=int, default=160)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/data/zjw/workspace/Isaac-GR00T/expdata/"
            "ur3e_robocasa_expert_replay/"
            "PreSoakPan_episode_000237_grip_site_test3/drift_diagnostic"
        ),
    )
    parser.add_argument("--base-z", type=float, default=0.92)
    parser.add_argument("--base-y-offset", type=float, default=0.0)
    parser.add_argument("--camera-width", type=int, default=256)
    parser.add_argument("--camera-height", type=int, default=256)
    parser.add_argument("--video-width", type=int, default=512)
    parser.add_argument("--video-height", type=int, default=512)
    parser.add_argument("--ik-iterations", type=int, default=80, help=argparse.SUPPRESS)
    parser.add_argument("--max-position-command-m", type=float, default=0.025)
    parser.add_argument("--max-orientation-command-rad", type=float, default=0.25)
    parser.add_argument("--max-object-step-m", type=float, default=0.01)
    parser.add_argument(
        "--alignment-orientation-weight",
        type=float,
        default=0.0,
        help="Weight of source orientation during diagnostic pre-alignment.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    expert = _load_expert(args.dataset_root, args.episode_index)

    source_env = _make_env(
        "PreSoakPan",
        "PandaOmron",
        expert["ep_meta"],
        args.episode_index,
        args.camera_width,
        args.camera_height,
    )
    try:
        _restore_expert_xml_state(
            source_env, expert["model_xml"], expert["states"][0]
        )
        scene_state = _extract_common_scene_state(
            source_env, expert["states"][0]
        )
    finally:
        source_env.close()

    register_ur3e(args.base_z, args.base_y_offset)
    env = _make_env(
        "PreSoakPan",
        "UR3eOfficialFixed",
        expert["ep_meta"],
        args.episode_index,
        args.camera_width,
        args.camera_height,
    )
    writer = _make_writer(
        args.output_root / "ur3e_grasp_diagnostic.mp4",
        fps=20,
    )
    try:
        _apply_common_scene_state(env, scene_state)
        replay_json = args.output_root.parent / "franka_replay.json"
        if not replay_json.exists():
            raise FileNotFoundError(
                f"Expected Franka target poses at {replay_json}"
            )
        source_rows = json.loads(replay_json.read_text(encoding="utf-8"))["rows"]
        # The expert's early trajectory includes mobile-base navigation. A
        # fixed-base UR3e cannot reproduce that part, so initialize directly
        # at the pose immediately before the diagnostic window and replay
        # only the grasp segment.
        alignment_index = max(args.start_step - 2, 0)
        target_position = np.asarray(
            source_rows[alignment_index]["grip_site_pos_world"], dtype=np.float64
        )
        target_quaternion = np.asarray(
            source_rows[alignment_index]["grip_site_quat_world_xyzw"],
            dtype=np.float64,
        )
        alignment = align_initial_eef_to(
            env,
            target_position,
            target_quaternion,
            allow_base_translation=True,
            base_translation_axes=(True, True, False),
            orientation_source="site",
            orientation_weight=args.alignment_orientation_weight,
            max_iterations=args.ik_iterations,
            position_tolerance=0.0015,
            orientation_tolerance=0.015,
        )
        raw = env._get_observations(force_update=True)
        rows = []
        pan_joint = "obj1_joint0"
        first_index = max(args.start_step - 1, 0)
        last_index = min(args.end_step, len(expert["actions"]))
        for index in range(first_index, last_index):
            action = expert["actions"][index]
            step = index + 1

            source_row = source_rows[index]
            target_position = np.asarray(
                source_row["grip_site_pos_world"], dtype=np.float64
            )
            target_quaternion = np.asarray(
                source_row["grip_site_quat_world_xyzw"], dtype=np.float64
            )
            before = _object_state(env, pan_joint)
            before_contacts = _contacts(env)
            arm_action = _eef_delta_action(
                env,
                target_position,
                target_quaternion,
                float(action[6]),
                max_position_command_m=args.max_position_command_m,
                max_orientation_command_rad=args.max_orientation_command_rad,
            )
            raw, _, done, step_info = _safe_env_step(
                env,
                arm_action,
                object_body_name="obj1_main",
                max_object_step_m=args.max_object_step_m,
            )
            writer.append_data(
                _render(
                    env,
                    "robot0_agentview_left",
                    args.video_width,
                    args.video_height,
                )
            )
            after = _object_state(env, pan_joint)
            after_contacts = _contacts(env)
            actual_position, actual_quaternion = _grip_site_pose(env)
            if step >= args.start_step:
                rows.append(
                    {
                        "step": step,
                        "time_sec": step / 20.0,
                        "gripper_action": float(action[6]),
                        "gripper": _gripper_summary(raw),
                        "eef_position_error_m": float(
                            np.linalg.norm(actual_position - target_position)
                        ),
                        "target_grip_site_pos_world": np.asarray(
                            target_position, dtype=np.float64
                        ).tolist(),
                        "actual_grip_site_pos_world": np.asarray(
                            actual_position, dtype=np.float64
                        ).tolist(),
                        "eef_orientation_error_rad": _quat_error(
                            target_quaternion, actual_quaternion
                        ),
                        "control_mode": "osc_eef_delta",
                        "action_7d": arm_action.tolist(),
                        "physics_rollback": bool(step_info.get("physics_rollback")),
                        "object_step_m": float(
                            step_info.get(
                                "object_step_m",
                                step_info.get("object_step_m_before_rollback", 0.0),
                            )
                        ),
                        "pan_before": before,
                        "pan_after": after,
                        "pan_delta_body_pos_m": float(
                            np.linalg.norm(
                                np.asarray(after["body_pos"])
                                - np.asarray(before["body_pos"])
                            )
                        ),
                        "pan_delta_qvel_norm": float(
                            np.linalg.norm(np.asarray(after["qvel"]))
                        ),
                        "contacts_before": before_contacts,
                        "contacts_after": after_contacts,
                        "done": bool(done),
                    }
                )
    finally:
        writer.close()
        env.close()

    result = {
        "task": "PreSoakPan",
        "episode_index": args.episode_index,
        "window_steps": [args.start_step, args.end_step],
        "alignment": alignment,
        "rows": rows,
    }
    (args.output_root / "drift_diagnostic.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
