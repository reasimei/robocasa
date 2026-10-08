#!/usr/bin/env python3
"""Save Franka and UR3e reset-pose images and alignment data.

This is an isolated diagnostic for the UR3e migration experiments. It creates
the same RoboCasa scene with PandaOmron and UR3e, aligns the UR3e EEF to the
actual Franka reset EEF pose, and saves images plus all relevant transforms.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robocasa.utils import env_utils
from robosuite.utils import transform_utils as T

from scripts.ur3e_robocasa_eval.run_xiaomi_ur3e_fixed_eval import (
    _current_eef_pose,
    _resize_frame,
    align_initial_eef_to,
    configure_review_camera,
    register_ur3e,
)


def _first(value: Any) -> np.ndarray:
    array = np.asarray(value)
    while array.ndim > 1 and array.shape[0] == 1:
        array = array[0]
    return np.asarray(array)


def _pose_from_raw(raw: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    position = _first(raw["robot0_eef_pos"]).astype(np.float64)
    quaternion = _first(raw["robot0_eef_quat"]).astype(np.float64)
    quaternion /= max(np.linalg.norm(quaternion), 1e-12)
    return position, quaternion


def _qpos_by_name(env: Any) -> dict[str, float]:
    values: dict[str, float] = {}
    for joint_id in range(env.sim.model.njnt):
        name = env.sim.model.joint_id2name(joint_id)
        if name is not None:
            values[name] = float(env.sim.data.qpos[env.sim.model.jnt_qposadr[joint_id]])
    return values


def _base_data(env: Any) -> dict[str, Any]:
    body_id = env.sim.model.body_name2id("robot0_base")
    return {
        "position_world": np.asarray(
            env.sim.data.xpos[body_id], dtype=np.float64
        ).tolist(),
        "quaternion_world_wxyz": np.asarray(
            env.sim.data.xquat[body_id], dtype=np.float64
        ).tolist(),
        "model_position_world": np.asarray(
            env.sim.model.body_pos[body_id], dtype=np.float64
        ).tolist(),
        "model_quaternion_parent_wxyz": np.asarray(
            env.sim.model.body_quat[body_id], dtype=np.float64
        ).tolist(),
    }


def _robot_data(env: Any, raw: dict[str, Any]) -> dict[str, Any]:
    position, quaternion = _pose_from_raw(raw)
    return {
        "eef_position_world": position.tolist(),
        "eef_quaternion_world_xyzw": quaternion.tolist(),
        "gripper_qpos": _first(raw["robot0_gripper_qpos"]).astype(np.float64).tolist(),
        "qpos_by_joint": _qpos_by_name(env),
        "base": _base_data(env),
    }


def _camera_frame(env: Any, camera_name: str, width: int, height: int) -> np.ndarray:
    frame = np.asarray(
        env.sim.render(height=height, width=width, camera_name=camera_name),
        dtype=np.uint8,
    )
    return frame[::-1, :, :].copy()


def _save_png(path: Path, frame: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(path)


def _create_env(
    task: str,
    robot: str,
    seed: int,
    split: str,
    layout_id: int,
    style_id: int,
    width: int,
    height: int,
) -> Any:
    return env_utils.create_env(
        task,
        robots=robot,
        split=None,
        obj_instance_split=split,
        layout_and_style_ids=[(layout_id, style_id)],
        seed=seed,
        camera_names=[
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ],
        camera_widths=width,
        camera_heights=height,
        render_camera="robot0_agentview_left",
        control_freq=20,
        initialization_noise=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="PreSoakPan")
    parser.add_argument("--split", default="target", choices=["target", "pretrain"])
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--layout-id", type=int, default=3)
    parser.add_argument("--style-id", type=int, default=3)
    parser.add_argument("--base-z", type=float, default=0.92)
    parser.add_argument("--base-y-offset", type=float, default=0.0)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "expdata/ur3e_robocasa_fixed_gr00t_smoke/"
            "initial_pose_alignment_presoak_seed1000"
        ),
    )
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    register_ur3e(args.base_z, args.base_y_offset)
    payload: dict[str, Any] = {
        "task": args.task,
        "split": args.split,
        "seed": args.seed,
        "layout_id": args.layout_id,
        "style_id": args.style_id,
        "camera_size": [args.width, args.height],
        "franka": {},
        "ur3e": {},
        "alignment": None,
        "images": {},
    }

    franka_env = _create_env(
        args.task,
        "PandaOmron",
        args.seed,
        args.split,
        args.layout_id,
        args.style_id,
        args.width,
        args.height,
    )
    try:
        franka_raw = franka_env.reset()
        franka_position, franka_quaternion = _pose_from_raw(franka_raw)
        payload["franka"] = _robot_data(franka_env, franka_raw)
        franka_left = _camera_frame(
            franka_env, "robot0_agentview_left", args.width, args.height
        )
        franka_hand = _camera_frame(
            franka_env, "robot0_eye_in_hand", args.width, args.height
        )
        franka_left_path = args.output_root / "franka_initial_agentview_left.png"
        franka_hand_path = args.output_root / "franka_initial_eye_in_hand.png"
        _save_png(franka_left_path, franka_left)
        _save_png(franka_hand_path, franka_hand)
        payload["images"]["franka_agentview_left"] = str(franka_left_path)
        payload["images"]["franka_eye_in_hand"] = str(franka_hand_path)
    finally:
        franka_env.close()

    ur3e_env = _create_env(
        args.task,
        "UR3eOfficialFixed",
        args.seed,
        args.split,
        args.layout_id,
        args.style_id,
        args.width,
        args.height,
    )
    try:
        ur3e_raw = ur3e_env.reset()
        ur3e_reset_data = _robot_data(ur3e_env, ur3e_raw)
        alignment = align_initial_eef_to(
            ur3e_env,
            franka_position,
            franka_quaternion,
            allow_base_translation=True,
        )
        ur3e_raw = ur3e_env._get_observations(force_update=True)
        configure_review_camera(ur3e_env, args.task)
        payload["alignment"] = alignment
        ur3e_aligned_data = _robot_data(ur3e_env, ur3e_raw)
        payload["ur3e"] = {
            "reset": ur3e_reset_data,
            "aligned": ur3e_aligned_data,
            "eef_position_error_to_franka_m": float(
                np.linalg.norm(
                    np.asarray(alignment["solved_position_world"])
                    - np.asarray(payload["franka"]["eef_position_world"])
                )
            ),
            "eef_orientation_error_to_franka_rad": float(
                np.linalg.norm(
                    T.get_orientation_error(
                        np.asarray(payload["franka"]["eef_quaternion_world_xyzw"]),
                        np.asarray(
                            ur3e_aligned_data["eef_quaternion_world_xyzw"]
                        ),
                    )
                )
            ),
        }
        ur3e_review = _camera_frame(
            ur3e_env, "robot0_review_camera", args.width, args.height
        )
        ur3e_left = _camera_frame(
            ur3e_env, "robot0_agentview_left", args.width, args.height
        )
        ur3e_hand = _camera_frame(
            ur3e_env, "robot0_eye_in_hand", args.width, args.height
        )
        ur3e_review_path = args.output_root / "ur3e_aligned_review_camera.png"
        ur3e_left_path = args.output_root / "ur3e_aligned_agentview_left.png"
        ur3e_hand_path = args.output_root / "ur3e_aligned_eye_in_hand.png"
        _save_png(ur3e_review_path, ur3e_review)
        _save_png(ur3e_left_path, ur3e_left)
        _save_png(ur3e_hand_path, ur3e_hand)
        payload["images"]["ur3e_aligned_review_camera"] = str(ur3e_review_path)
        payload["images"]["ur3e_aligned_agentview_left"] = str(ur3e_left_path)
        payload["images"]["ur3e_aligned_eye_in_hand"] = str(ur3e_hand_path)

        # The review camera is the comparable third-person view. Make a
        # side-by-side image from the two primary views for quick inspection.
        from PIL import Image, ImageDraw

        franka_view = Image.fromarray(franka_left).convert("RGB")
        ur3e_view = Image.fromarray(ur3e_review).convert("RGB")
        montage = Image.new(
            "RGB",
            (args.width * 2, args.height + 40),
            color=(245, 245, 245),
        )
        montage.paste(franka_view, (0, 40))
        montage.paste(ur3e_view, (args.width, 40))
        draw = ImageDraw.Draw(montage)
        draw.text((12, 12), "Franka reset EEF", fill=(0, 0, 0))
        draw.text((args.width + 12, 12), "UR3e aligned EEF", fill=(0, 0, 0))
        montage_path = args.output_root / "initial_pose_side_by_side.png"
        montage.save(montage_path)
        payload["images"]["side_by_side"] = str(montage_path)
    finally:
        ur3e_env.close()

    data_path = args.output_root / "initial_pose_data.json"
    data_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    print(f"DATA={data_path}", flush=True)


if __name__ == "__main__":
    main()
