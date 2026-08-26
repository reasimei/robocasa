#!/usr/bin/env python3
"""
Run GR00T closed-loop in Robocasa and measure online action-chunk consistency.

Unlike run_composite_chunk_consistency.py, this script does not replay expert
observations. It executes GR00T's predicted actions in the simulator, then compares
consecutive GR00T action chunks from the resulting closed-loop trajectory.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import tyro

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gr00t.eval.simulation import MultiStepConfig, SimulationConfig, SimulationInferenceClient, VideoConfig
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy
from robocasa.utils.dataset_registry_utils import get_task_horizon

from scripts.aux_progress.run_composite_chunk_consistency import (
    concat_action_dict,
    consistency_rmse,
    online_threshold_triggers,
)


@dataclass
class Args:
    model_path: str = "/data/zjw/workspace/Isaac-GR00T/expdata/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000"
    env_name: str = "DeliverStraw"
    env_names: tuple[str, ...] = ()
    num_trials: int = 1
    seed: int = 42
    split: str = "target"
    output_dir: str = "/data/zjw/workspace/Isaac-GR00T/expdata/chunk_consistency/closed_loop_ckpt60000"
    data_config: str = "panda_omron"
    embodiment_tag: str = "new_embodiment"
    video_key: str = "video.robot0_agentview_left"
    max_steps: int = -1
    stop_after_success_steps: int = 50
    n_action_steps: int = 1
    denoising_steps: int = 4
    online_threshold: float = 0.2
    online_cooldown_sec: float = 3.0
    ignore_edge_sec: float = 2.0
    online_ema_alpha: float = 1.0
    fps: float = 20.0


def display_env_name(env_name: str) -> str:
    return env_name.split("/", 1)[1] if env_name.startswith("robocasa/") else env_name


def first_env_image(observation: dict[str, Any], video_key: str) -> np.ndarray:
    if video_key not in observation:
        raise KeyError(f"{video_key!r} not found in observation. Available keys: {sorted(observation.keys())}")
    array = np.asarray(observation[video_key])
    while array.ndim > 3:
        if array.shape[0] == 1:
            array = array[0]
        else:
            array = array[-1]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def write_scores_csv(path: Path, scores: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "sec", "score", "success"])
        writer.writeheader()
        for item in scores:
            writer.writerow(item)


def write_annotated_video(
    path: Path,
    frames: list[np.ndarray],
    scores: list[dict[str, Any]],
    triggers: list[dict[str, Any]],
    fps: float,
) -> None:
    if not frames:
        return
    height, width = frames[0].shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps}",
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    writer = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert writer.stdin is not None
    trigger_steps = sorted(int(item["step"]) for item in triggers)
    scores_by_step = {int(item["step"]): item for item in scores}
    try:
        for step, rgb in enumerate(frames):
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            subtask_index = 1 + sum(trigger_step <= step for trigger_step in trigger_steps)
            score = scores_by_step.get(step, {}).get("score")
            label = f"Subtask {subtask_index}"
            if score is not None:
                label += f"  cc={float(score):.3f}"
            cv2.rectangle(frame, (8, height - 38), (190, height - 8), (0, 0, 0), -1)
            cv2.rectangle(frame, (8, height - 38), (190, height - 8), (255, 255, 255), 1)
            cv2.putText(frame, label, (14, height - 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            writer.stdin.write(frame.tobytes())
    finally:
        writer.stdin.close()
        stderr = writer.stderr.read().decode("utf-8", errors="replace") if writer.stderr else ""
        return_code = writer.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed with code {return_code}:\n{stderr}")


def run_trial(
    args: Args,
    policy: Gr00tPolicy,
    action_keys: list[str],
    env_name: str,
    trial_index: int,
) -> dict[str, Any]:
    short_env_name = display_env_name(env_name)
    robocasa_env_name = env_name if env_name.startswith("robocasa/") else f"robocasa/{env_name}"
    horizon = get_task_horizon(short_env_name)
    max_steps = horizon if args.max_steps <= 0 else min(int(args.max_steps), int(horizon))
    trial_dir = Path(args.output_dir) / short_env_name / f"trial_{trial_index:03d}"
    config = SimulationConfig(
        env_name=robocasa_env_name,
        split=args.split,
        n_episodes=1,
        n_envs=1,
        video=VideoConfig(video_dir=str(trial_dir / "robocasa_video")),
        multistep=MultiStepConfig(n_action_steps=args.n_action_steps, max_episode_steps=max_steps),
    )
    client = SimulationInferenceClient(host="localhost", port=0)
    env = client.setup_environment(config)

    scores: list[dict[str, Any]] = []
    frames: list[np.ndarray] = []
    prev_chunk: np.ndarray | None = None
    success = False
    first_success_step: int | None = None
    reset_seed = int(args.seed) + int(trial_index)
    observation, _ = env.reset(seed=reset_seed)
    try:
        for step in range(max_steps):
            frames.append(first_env_image(observation, args.video_key))
            action = policy.get_action(observation)
            chunk = concat_action_dict(action, action_keys)
            score = None
            if prev_chunk is not None:
                score = consistency_rmse(prev_chunk, chunk, step_delta=1)
            scores.append(
                {
                    "step": int(step),
                    "sec": float(step / args.fps),
                    "score": score,
                    "success": bool(success),
                }
            )
            prev_chunk = chunk

            observation, rewards, terminations, truncations, infos = env.step(action)
            del rewards
            success = success or bool(infos["success"][0][0]) if "success" in infos else success
            if success and first_success_step is None:
                first_success_step = int(step)
            if (step + 1) % 25 == 0:
                print(
                    f"[{short_env_name} trial {trial_index}] "
                    f"closed-loop step {step + 1}/{max_steps} success={success}",
                    flush=True,
                )
            if bool(terminations[0] or truncations[0]):
                break
            if (
                first_success_step is not None
                and args.stop_after_success_steps >= 0
                and int(step) - first_success_step >= int(args.stop_after_success_steps)
            ):
                print(
                    f"[{short_env_name} trial {trial_index}] stopping "
                    f"{args.stop_after_success_steps} steps after success",
                    flush=True,
                )
                break
    finally:
        env.close()

    trigger_input = [
        {"frame": item["step"], "sec": item["sec"], "score": item["score"]}
        for item in scores
    ]
    triggers = online_threshold_triggers(
        trigger_input,
        threshold=args.online_threshold,
        cooldown_sec=args.online_cooldown_sec,
        ignore_edge_sec=args.ignore_edge_sec,
        ema_alpha=args.online_ema_alpha,
        rising_edge_only=True,
    )
    for trigger in triggers:
        trigger["step"] = int(trigger.pop("frame"))

    scores_csv = trial_dir / "closed_loop_chunk_consistency_scores.csv"
    summary_json = trial_dir / "closed_loop_chunk_consistency_summary.json"
    annotated_mp4 = trial_dir / "closed_loop_chunk_consistency_annotated.mp4"
    write_scores_csv(scores_csv, scores)
    write_annotated_video(annotated_mp4, frames, scores, triggers, args.fps)
    summary = {
        "env_name": short_env_name,
        "trial_index": int(trial_index),
        "seed": reset_seed,
        "model_path": args.model_path,
        "split": args.split,
        "n_action_steps": args.n_action_steps,
        "score_definition": "closed-loop online self-consistency RMSE(prev_groot_chunk[1:] - curr_groot_chunk[:-1]); GR00T actions are executed in the simulator",
        "num_steps": len(scores),
        "success": bool(success),
        "first_success_step": first_success_step,
        "first_success_sec": None if first_success_step is None else float(first_success_step / args.fps),
        "stop_after_success_steps": int(args.stop_after_success_steps),
        "online_threshold": args.online_threshold,
        "online_cooldown_sec": args.online_cooldown_sec,
        "ignore_edge_sec": args.ignore_edge_sec,
        "online_ema_alpha": args.online_ema_alpha,
        "online_triggers": triggers,
        "scores_csv": str(scores_csv),
        "annotated_mp4": str(annotated_mp4),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary: {summary_json}", flush=True)
    print(f"Saved annotated MP4: {annotated_mp4}", flush=True)
    print("Online triggers:")
    for trigger in triggers:
        print(f"  step={trigger['step']} sec={trigger['sec']:.2f} score={trigger['score']:.5f}", flush=True)
    return summary


def main(args: Args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run closed-loop GR00T chunk-consistency inference.")

    cfg = DATA_CONFIG_MAP[args.data_config]
    modality_config = cfg.modality_config()
    policy = Gr00tPolicy(
        model_path=args.model_path,
        modality_config=modality_config,
        modality_transform=cfg.transform(),
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
    )
    action_keys = modality_config["action"].modality_keys

    env_names = args.env_names or (args.env_name,)
    summaries: list[dict[str, Any]] = []
    for env_name in env_names:
        for trial_index in range(int(args.num_trials)):
            summaries.append(run_trial(args, policy, action_keys, env_name, trial_index))

    index_path = Path(args.output_dir) / "closed_loop_chunk_consistency_index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"Saved index: {index_path}", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
