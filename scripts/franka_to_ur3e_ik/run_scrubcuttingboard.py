#!/usr/bin/env python3
"""Replay ScrubCuttingBoard expert episode 376 on a fixed-base UR3e.

将 ScrubCuttingBoard 第 376 个专家 episode 通过 IK 映射到固定底座 UR3e。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .dataset import load_expert_episode
from .ik import IKConfig
from .replay import run_replay


DEFAULT_DATASET_ROOT = Path(
    "/data/zjw/workspace/robocasa/datasets/v1.0/target/composite/"
    "ScrubCuttingBoard/20250816/lerobot"
)
DEFAULT_OUTPUT_DIR = Path(
    "/data/zjw/workspace/Isaac-GR00T/expdata/"
    "franka_to_ur3e_ik/ScrubCuttingBoard_episode_000376"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay RoboCasa expert states on a fixed-base UR3e."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--episode-index", type=int, default=376)
    parser.add_argument("--task", default="ScrubCuttingBoard")
    parser.add_argument("--seed", type=int, default=376)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--base-offset",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.92],
        metavar=("DX", "DY", "DZ"),
        help="Initial UR3e base offset from RoboCasa anchor, in meters.",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=512,
        help="Output video width in pixels; default is 512.",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=512,
        help="Output video height in pixels; default is 512.",
    )
    parser.add_argument("--ik-iterations", type=int, default=60)
    parser.add_argument("--ik-damping", type=float, default=0.04)
    parser.add_argument("--ik-step-scale", type=float, default=0.8)
    parser.add_argument("--ik-max-joint-step", type=float, default=0.15)
    parser.add_argument("--ik-position-tolerance", type=float, default=0.0015)
    parser.add_argument("--ik-orientation-tolerance", type=float, default=0.02)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Replay only the first N steps; 0 means the complete episode.",
    )
    parser.add_argument(
        "--execution-mode",
        choices=["teleport", "controller"],
        default="teleport",
        help=(
            "teleport applies IK qpos directly before physics; controller "
            "uses the finite-rate joint-position controller."
        ),
    )
    parser.add_argument(
        "--gripper-control-mode",
        choices=["source_action", "state_mapping"],
        default="source_action",
        help=(
            "source_action preserves the expert open/close timing; "
            "state_mapping uses Panda opening to set an absolute Robotiq target."
        ),
    )
    parser.add_argument(
        "--no-pad-midpoint-alignment",
        action="store_true",
        help="Disable Franka/Robotiq finger-pad midpoint alignment.",
    )
    parser.add_argument("--save-video", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episode = load_expert_episode(args.dataset_root, args.episode_index)
    config = IKConfig(
        max_iterations=args.ik_iterations,
        damping=args.ik_damping,
        step_scale=args.ik_step_scale,
        max_joint_step_rad=args.ik_max_joint_step,
        position_tolerance_m=args.ik_position_tolerance,
        orientation_tolerance_rad=args.ik_orientation_tolerance,
    )
    result = run_replay(
        args.task,
        episode,
        seed=args.seed,
        base_offset_xyz=np.asarray(args.base_offset, dtype=np.float64),
        output_dir=args.output_dir,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        ik_config=config,
        save_video=args.save_video,
        max_steps=args.max_steps or None,
        execution_mode=args.execution_mode,
        align_to_pad_midpoint=not args.no_pad_midpoint_alignment,
        gripper_control_mode=args.gripper_control_mode,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
