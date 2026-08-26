#!/usr/bin/env python3
"""Replay a saved UR3e action trajectory with a corrected review camera."""

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

from scripts.ur3e_robocasa_eval.run_xiaomi_ur3e_fixed_eval import (
    configure_review_camera,
    register_ur3e,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--task", default="PreSoakPan")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--layout-id", type=int, default=4)
    parser.add_argument("--style-id", type=int, default=4)
    parser.add_argument("--base-z", type=float, default=0.92)
    parser.add_argument("--base-y-offset", type=float, default=0.0)
    parser.add_argument("--camera-width", type=int, default=512)
    parser.add_argument("--camera-height", type=int, default=512)
    parser.add_argument("--fps", type=int, default=20)
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _set_alignment_qpos(env: Any, alignment: dict[str, Any]) -> None:
    qpos = np.asarray(alignment["solved_arm_qpos"], dtype=np.float64)
    robot = env.robots[0]
    if qpos.shape != (6,):
        raise ValueError(f"Expected 6 UR3e arm joints, got {qpos.shape}")
    env.sim.data.qpos[robot._ref_arm_joint_pos_indexes] = qpos
    env.sim.data.qvel[robot._ref_arm_joint_vel_indexes] = 0.0
    env.sim.forward()
    robot.composite_controller.update_state()
    robot.composite_controller.reset()


def main() -> None:
    args = parse_args()
    rows = _load_json(args.trajectory)
    alignment = _load_json(args.alignment)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Trajectory is empty: {args.trajectory}")

    register_ur3e(args.base_z, args.base_y_offset)
    env = env_utils.create_env(
        args.task,
        robots="UR3eOfficialFixed",
        split=None,
        obj_instance_split="target",
        layout_and_style_ids=[(args.layout_id, args.style_id)],
        seed=args.seed,
        camera_names=[
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ],
        camera_widths=128,
        camera_heights=128,
        render_camera="robot0_agentview_left",
        control_freq=20,
        initialization_noise=None,
    )
    try:
        env.reset()
        _set_alignment_qpos(env, alignment)
        configure_review_camera(env, args.task)

        import imageio.v2 as imageio

        args.output_video.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(
            args.output_video,
            fps=args.fps,
            codec="libx264",
        ) as writer:
            for index, row in enumerate(rows, start=1):
                action = np.asarray(row["action"], dtype=np.float32)
                if action.shape != (7,):
                    raise ValueError(
                        f"Expected 7D UR3e action at row {index}, got {action.shape}"
                    )
                env.step(action)
                frame = np.asarray(
                    env.sim.render(
                        height=args.camera_height,
                        width=args.camera_width,
                        camera_name="robot0_review_camera",
                    ),
                    dtype=np.uint8,
                )[::-1, :, :].copy()
                writer.append_data(frame)
    finally:
        env.close()

    print(
        json.dumps(
            {
                "output_video": str(args.output_video),
                "frames": len(rows),
                "layout_id": args.layout_id,
                "style_id": args.style_id,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
