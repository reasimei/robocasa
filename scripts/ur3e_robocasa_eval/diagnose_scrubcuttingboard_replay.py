#!/usr/bin/env python3
"""Inspect the ScrubCuttingBoard push-to-grasp window.

This diagnostic is intentionally read-only with respect to the replay output.
It compares the recorded Franka state and the current UR3e replay JSON around
the grasp transition, and reports active collision geoms in the UR3e model.
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

from scripts.ur3e_robocasa_eval.replay_lerobot_expert_franka_ur3e import (
    _grip_site_pose,
    _load_expert,
    _make_env,
    _restore_expert_xml_state,
)


def _body_position(env, body_name: str) -> list[float] | None:
    try:
        return np.asarray(
            env.sim.data.get_body_xpos(body_name), dtype=np.float64
        ).round(6).tolist()
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--franka-json", type=Path, required=True)
    parser.add_argument("--ur3e-json", type=Path, required=True)
    parser.add_argument("--start-step", type=int, default=130)
    parser.add_argument("--end-step", type=int, default=210)
    parser.add_argument("--task", default="ScrubCuttingBoard")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expert = _load_expert(args.dataset_root, args.episode_index)
    franka = json.loads(args.franka_json.read_text(encoding="utf-8"))
    ur3e = json.loads(args.ur3e_json.read_text(encoding="utf-8"))

    env = _make_env(
        args.task,
        "PandaOmron",
        expert["ep_meta"],
        args.episode_index,
        64,
        64,
        render=False,
    )
    try:
        _restore_expert_xml_state(env, expert["model_xml"], expert["states"][0])
        print("active_robot_collision_geoms:")
        for geom_id in range(int(env.sim.model.ngeom)):
            name = env.sim.model.geom_id2name(geom_id) or ""
            body_id = int(env.sim.model.geom_bodyid[geom_id])
            body = env.sim.model.body_id2name(body_id) or ""
            contype = int(env.sim.model.geom_contype[geom_id])
            conaffinity = int(env.sim.model.geom_conaffinity[geom_id])
            if (name.startswith("robot0_") or body.startswith("robot0_")) and (
                contype != 0 or conaffinity != 0
            ):
                print(f"  {name} body={body} contype={contype} conaffinity={conaffinity}")

        print("expert_window:")
        for step in range(args.start_step, min(args.end_step, len(franka["rows"])) + 1):
            state_index = min(step, len(expert["states"]) - 1)
            env.sim.set_state_from_flattened(expert["states"][state_index])
            env.sim.forward()
            row = franka["rows"][step - 1]
            print(
                json.dumps(
                    {
                        "step": step,
                        "grip_site": row.get("grip_site_pos_world"),
                        "sponge": _body_position(env, "sponge_main"),
                        "sponge_joint": _body_position(env, "sponge"),
                        "gripper_action": (
                            expert["actions"][step - 1, 6].item()
                            if step - 1 < len(expert["actions"])
                            else None
                        ),
                    },
                    separators=(",", ":"),
                )
            )
    finally:
        env.close()

    print("ur3e_window:")
    for row in ur3e["rows"]:
        if args.start_step <= row["step"] <= args.end_step:
            print(
                json.dumps(
                    {
                        "step": row["step"],
                        "target_grip_site": row.get("target_grip_site_pos_world"),
                        "actual_grip_site": row.get("actual_grip_site_pos_world"),
                        "position_error_m": row.get("position_error_m"),
                        "object_step_m": row.get("object_step_m"),
                        "gripper_action": row.get("gripper_action"),
                        "pad_midpoint": row.get("actual_grip_pad_geometry", {}).get(
                            "midpoint_world"
                        ),
                        "pad_separation_m": row.get("actual_grip_pad_geometry", {}).get(
                            "separation_m"
                        ),
                    },
                    separators=(",", ":"),
                )
            )


if __name__ == "__main__":
    main()
