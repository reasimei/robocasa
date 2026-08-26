#!/usr/bin/env python3
"""Compare saved EE trajectories from the isolated UR3e and Franka tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    positions = np.asarray([row["eef_pos_world"] for row in rows], dtype=np.float64)
    actions = np.asarray([row["action"] for row in rows], dtype=np.float64)
    return positions, actions


def _corr(a: np.ndarray, b: np.ndarray) -> list[float | None]:
    values: list[float | None] = []
    for index in range(a.shape[1]):
        if np.std(a[:, index]) < 1e-12 or np.std(b[:, index]) < 1e-12:
            values.append(None)
        else:
            values.append(float(np.corrcoef(a[:, index], b[:, index])[0, 1]))
    return values


def compare(
    ur3e_path: Path,
    franka_path: Path,
    output_path: Path,
) -> dict:
    ur3e_pos, ur3e_actions = _load(ur3e_path)
    franka_pos, franka_actions = _load(franka_path)
    length = min(len(ur3e_pos), len(franka_pos))
    ur3e_pos = ur3e_pos[:length]
    franka_pos = franka_pos[:length]
    ur3e_actions = ur3e_actions[:length]
    franka_actions = franka_actions[:length]

    ur3e_delta = ur3e_pos - ur3e_pos[0]
    franka_delta = franka_pos - franka_pos[0]
    aligned_delta = ur3e_delta - franka_delta
    position_error = np.linalg.norm(aligned_delta, axis=1)
    first_seven_action_error = ur3e_actions[:, :7] - franka_actions[:, :7]

    report = {
        "num_steps_compared": length,
        "ur3e": {
            "trajectory": str(ur3e_path),
            "initial_eef_world": ur3e_pos[0].tolist(),
            "final_eef_world": ur3e_pos[-1].tolist(),
            "displacement_world": (ur3e_pos[-1] - ur3e_pos[0]).tolist(),
        },
        "franka_pandaomron": {
            "trajectory": str(franka_path),
            "initial_eef_world": franka_pos[0].tolist(),
            "final_eef_world": franka_pos[-1].tolist(),
            "displacement_world": (franka_pos[-1] - franka_pos[0]).tolist(),
        },
        "initial_position_aligned_world_comparison": {
            "coordinate_rmse_m": float(np.sqrt(np.mean(aligned_delta**2))),
            "mean_euclidean_error_m": float(np.mean(position_error)),
            "max_euclidean_error_m": float(np.max(position_error)),
            "coordinate_correlation_xyz": _corr(ur3e_delta, franka_delta),
        },
        "first_seven_action_comparison": {
            "rmse_normalized_action": float(
                np.sqrt(np.mean(first_seven_action_error**2))
            ),
            "mean_absolute_error_normalized_action": float(
                np.mean(np.abs(first_seven_action_error))
            ),
            "coordinate_correlation": _corr(
                ur3e_actions[:, :7], franka_actions[:, :7]
            ),
        },
        "interpretation": (
            "The trajectories are not the same. The Franka/PandaOmron run "
            "uses the full 12D action including mobile-base motion, while the "
            "fixed UR3e run consumes only the first 7 action dimensions."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ur3e", type=Path, required=True)
    parser.add_argument("--franka", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            compare(args.ur3e, args.franka, args.output),
            indent=2,
        )
    )
