#!/usr/bin/env python3
"""
Run a GR00T checkpoint on composite expert observations and measure action-chunk consistency.

The script uses teacher-forced observations from Robocasa composite demonstration episodes:
for each sampled frame t, it asks the policy for an action chunk, compares the overlapping
part with the previous sampled prediction, and saves scores. Videos are optional.
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import tyro
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy


@dataclass
class Args:
    model_path: str = "/data/zjw/workspace/Isaac-GR00T/expdata/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000"
    composite_root: str = "/data/zjw/workspace/robocasa/datasets/v1.0/target/composite"
    plan_cache_dir: str = (
        "/data/zjw/workspace/Isaac-GR00T/expdata/"
        "long_horizon_controller/composite_seen_plan_cache_llama70b"
    )
    output_dir: str = "/data/zjw/workspace/Isaac-GR00T/expdata/chunk_consistency/composite_seen_ckpt60000"
    tasks: tuple[str, ...] = ()
    episodes: tuple[int, ...] = (0,)
    all_episodes: bool = False
    skip_existing: bool = False
    resume: bool = False
    progress: bool = True
    save_source_video: bool = True
    save_annotated_video: bool = True
    data_config: str = "panda_omron"
    embodiment_tag: str = "new_embodiment"
    video_backend: str = "opencv"
    video_key: str = "observation.images.robot0_agentview_left"
    frame_stride: int = 4
    max_frames: int = -1
    top_k: int = 8
    online_threshold: float = 0.2
    online_cooldown_sec: float = 3.0
    online_ema_alpha: float = 1.0
    online_rising_edge_only: bool = True
    min_peak_separation_sec: float = 3.0
    merge_peak_distance_sec: float = 3.0
    ignore_edge_sec: float = 2.0
    denoising_steps: int = 4


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_lerobot_root(composite_root: Path, task_name: str) -> Path:
    candidates = sorted((composite_root / task_name).glob("*/lerobot"))
    if not candidates:
        raise FileNotFoundError(f"No LeRobot dataset found for task {task_name} under {composite_root}")
    return candidates[-1]


def discover_tasks(composite_root: Path, plan_cache_dir: Path) -> list[str]:
    """Discover tasks that have both a cached plan and an extracted LeRobot dataset."""
    if not plan_cache_dir.is_dir():
        raise FileNotFoundError(f"Plan cache not found: {plan_cache_dir}")
    task_names: list[str] = []
    for plan_path in sorted(plan_cache_dir.glob("*/plan.json")):
        task_name = plan_path.parent.name
        try:
            find_lerobot_root(composite_root, task_name)
        except FileNotFoundError:
            continue
        task_names.append(task_name)
    return task_names


def episode_chunk(episode_index: int, chunk_size: int) -> int:
    return int(episode_index) // int(chunk_size)


def episode_record(root: Path, episode_index: int) -> dict[str, Any]:
    with (root / "meta" / "episodes.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if int(record["episode_index"]) == int(episode_index):
                return record
    raise KeyError(f"Episode {episode_index} not found in {root}")


def episode_indices(root: Path) -> list[int]:
    return [
        int(record["episode_index"])
        for record in (
            json.loads(line)
            for line in (root / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    ]


def episode_video_path(root: Path, episode_index: int, video_key: str) -> Path:
    info = load_json(root / "meta" / "info.json")
    chunk = episode_chunk(episode_index, int(info["chunks_size"]))
    return root / info["video_path"].format(
        episode_chunk=chunk,
        video_key=video_key,
        episode_index=episode_index,
    )


def episode_parquet_path(root: Path, episode_index: int) -> Path:
    info = load_json(root / "meta" / "info.json")
    chunk = episode_chunk(episode_index, int(info["chunks_size"]))
    return root / info["data_path"].format(
        episode_chunk=chunk,
        episode_index=episode_index,
    )


def concat_action_dict(action: dict[str, Any], action_keys: list[str]) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for key in action_keys:
        value = action[key]
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        array = np.asarray(value, dtype=np.float32)
        if array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        if array.ndim == 1:
            array = array[None, :]
        chunks.append(array)
    return np.concatenate(chunks, axis=-1)


def consistency_rmse(prev_chunk: np.ndarray, curr_chunk: np.ndarray, step_delta: int) -> float | None:
    if step_delta <= 0:
        return None
    horizon = min(prev_chunk.shape[0], curr_chunk.shape[0])
    if step_delta >= horizon:
        return None
    prev_overlap = prev_chunk[step_delta:horizon]
    curr_overlap = curr_chunk[: horizon - step_delta]
    return float(np.sqrt(np.mean((prev_overlap - curr_overlap) ** 2)))


def pick_peaks(
    scores: list[dict[str, Any]],
    fps: float,
    top_k: int,
    min_peak_separation_sec: float,
    ignore_edge_sec: float,
    episode_length: int,
) -> list[dict[str, Any]]:
    candidates = [
        item
        for item in scores
        if item["score"] is not None
        and item["frame"] >= int(round(ignore_edge_sec * fps))
        and item["frame"] <= episode_length - int(round(ignore_edge_sec * fps))
    ]
    candidates.sort(key=lambda item: float(item["score"]), reverse=True)
    min_sep = max(int(round(min_peak_separation_sec * fps)), 1)
    peaks: list[dict[str, Any]] = []
    for item in candidates:
        if all(abs(int(item["frame"]) - int(prev["frame"])) >= min_sep for prev in peaks):
            peaks.append(item)
            if len(peaks) >= top_k:
                break
    return sorted(peaks, key=lambda item: int(item["frame"]))


def merge_close_peaks(peaks: list[dict[str, Any]], fps: float, merge_distance_sec: float) -> list[dict[str, Any]]:
    if not peaks:
        return []
    merge_distance_frames = max(int(round(float(merge_distance_sec) * fps)), 0)
    sorted_peaks = sorted(peaks, key=lambda item: int(item["frame"]))
    groups: list[list[dict[str, Any]]] = [[sorted_peaks[0]]]
    for peak in sorted_peaks[1:]:
        if int(peak["frame"]) - int(groups[-1][-1]["frame"]) < merge_distance_frames:
            groups[-1].append(peak)
        else:
            groups.append([peak])

    merged: list[dict[str, Any]] = []
    for group in groups:
        if len(group) == 1:
            merged.append(group[0])
            continue
        start_frame = int(group[0]["frame"])
        end_frame = int(group[-1]["frame"])
        middle_frame = int(round((start_frame + end_frame) / 2.0))
        merged.append(
            {
                "frame": middle_frame,
                "sec": float(middle_frame / fps),
                "score": float(max(float(item["score"]) for item in group)),
                "merged_from": [
                    {
                        "frame": int(item["frame"]),
                        "sec": float(item["sec"]),
                        "score": float(item["score"]),
                    }
                    for item in group
                ],
            }
        )
    return merged


def online_threshold_triggers(
    scores: list[dict[str, Any]],
    threshold: float,
    cooldown_sec: float,
    ignore_edge_sec: float,
    ema_alpha: float = 1.0,
    rising_edge_only: bool = True,
) -> list[dict[str, Any]]:
    """Streaming-compatible trigger rule using only current and past scores."""
    triggers: list[dict[str, Any]] = []
    next_allowed_sec = float(ignore_edge_sec)
    ema_score: float | None = None
    was_above = False
    alpha = float(np.clip(ema_alpha, 0.0, 1.0))
    for item in scores:
        raw_score = item["score"]
        sec = float(item["sec"])
        if raw_score is None:
            continue
        raw_score = float(raw_score)
        if ema_score is None:
            ema_score = raw_score
        else:
            ema_score = alpha * raw_score + (1.0 - alpha) * ema_score

        is_above = ema_score >= float(threshold)
        is_rising_edge = is_above and not was_above
        was_above = is_above
        if sec < next_allowed_sec:
            continue
        if is_above and (is_rising_edge or not rising_edge_only):
            trigger = dict(item)
            trigger["raw_score"] = raw_score
            trigger["score"] = float(ema_score)
            trigger["threshold"] = float(threshold)
            trigger["cooldown_sec"] = float(cooldown_sec)
            trigger["ema_alpha"] = alpha
            trigger["rising_edge_only"] = bool(rising_edge_only)
            triggers.append(trigger)
            next_allowed_sec = sec + float(cooldown_sec)
    return triggers


def write_scores_csv(path: Path, scores: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame", "sec", "score"])
        writer.writeheader()
        for item in scores:
            writer.writerow(item)


def write_triggers_csv(path: Path, triggers: list[dict[str, Any]]) -> None:
    fieldnames = [
        "frame",
        "sec",
        "raw_score",
        "score",
        "threshold",
        "cooldown_sec",
        "ema_alpha",
        "rising_edge_only",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in triggers:
            writer.writerow({key: item.get(key) for key in fieldnames})


def make_annotated_video(
    source_video: Path,
    output_video: Path,
    scores: list[dict[str, Any]],
    transitions: list[dict[str, Any]],
    instruction: str,
) -> None:
    del scores, instruction
    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_video.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_cmd = [
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
        str(output_video),
    ]
    writer = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert writer.stdin is not None

    transition_frames = sorted(int(item["frame"]) for item in transitions)
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            subtask_index = 1 + sum(frame <= frame_idx for frame in transition_frames)
            label = f"Subtask {subtask_index}"
            box_left = 8
            box_bottom = height - 8
            box_top = height - 34
            box_right = 102
            cv2.rectangle(frame, (box_left, box_top), (box_right, box_bottom), (0, 0, 0), -1)
            cv2.rectangle(frame, (box_left, box_top), (box_right, box_bottom), (255, 255, 255), 1)
            cv2.putText(
                frame,
                label,
                (14, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            writer.stdin.write(frame.tobytes())
            frame_idx += 1
    finally:
        cap.release()
        writer.stdin.close()
        stderr = writer.stderr.read().decode("utf-8", errors="replace") if writer.stderr else ""
        return_code = writer.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed with code {return_code}:\n{stderr}")


def analyze_episode(
    args: Args,
    policy: Gr00tPolicy,
    task_name: str,
    episode_index: int,
) -> dict[str, Any]:
    root = find_lerobot_root(Path(args.composite_root), task_name)
    info = load_json(root / "meta" / "info.json")
    fps = float(info["fps"])
    record = episode_record(root, episode_index)
    instruction = record.get("tasks", [""])[0]

    cfg = DATA_CONFIG_MAP[args.data_config]
    transform = cfg.transform()
    transform.eval()
    dataset = LeRobotSingleDataset(
        dataset_path=root,
        modality_configs=cfg.modality_config(),
        transforms=transform,
        embodiment_tag=EmbodimentTag(args.embodiment_tag),
        video_backend=args.video_backend,
    )

    parquet_path = episode_parquet_path(root, episode_index)
    df = pd.read_parquet(parquet_path, columns=["frame_index"])
    episode_length = len(df)
    max_frame = episode_length if args.max_frames <= 0 else min(episode_length, args.max_frames)
    frame_stride = max(int(args.frame_stride), 1)
    frame_indices = list(range(0, max_frame, frame_stride))
    if frame_indices[-1] != max_frame - 1:
        frame_indices.append(max_frame - 1)

    action_keys = cfg.modality_config()["action"].modality_keys
    scores: list[dict[str, Any]] = []
    prev_chunk: np.ndarray | None = None
    prev_frame: int | None = None

    for i, frame_idx in enumerate(frame_indices):
        raw_obs = dataset.get_step_data(int(episode_index), int(frame_idx))
        action = policy.get_action(raw_obs)
        chunk = concat_action_dict(action, action_keys)
        score = None
        if prev_chunk is not None and prev_frame is not None:
            score = consistency_rmse(prev_chunk, chunk, int(frame_idx - prev_frame))
        scores.append(
            {
                "frame": int(frame_idx),
                "sec": float(frame_idx / fps),
                "score": score,
            }
        )
        prev_chunk = chunk
        prev_frame = int(frame_idx)
        if (i + 1) % 25 == 0 or (i + 1) == len(frame_indices):
            print(
                f"[{task_name} ep{episode_index}] predicted {i + 1}/{len(frame_indices)} chunks",
                flush=True,
            )

    peaks = pick_peaks(
        scores=scores,
        fps=fps,
        top_k=args.top_k,
        min_peak_separation_sec=args.min_peak_separation_sec,
        ignore_edge_sec=args.ignore_edge_sec,
        episode_length=episode_length,
    )
    peaks = merge_close_peaks(peaks, fps, args.merge_peak_distance_sec)
    online_triggers = online_threshold_triggers(
        scores=scores,
        threshold=args.online_threshold,
        cooldown_sec=args.online_cooldown_sec,
        ignore_edge_sec=args.ignore_edge_sec,
        ema_alpha=args.online_ema_alpha,
        rising_edge_only=args.online_rising_edge_only,
    )

    task_out = Path(args.output_dir) / task_name / f"episode_{episode_index:06d}"
    task_out.mkdir(parents=True, exist_ok=True)
    scores_csv = task_out / "chunk_consistency_scores.csv"
    online_csv = task_out / "chunk_consistency_online_triggers.csv"
    scores_json = task_out / "chunk_consistency_summary.json"
    source_video = episode_video_path(root, episode_index, args.video_key)
    copied_video = task_out / "source_episode.mp4"
    annotated_video = task_out / "chunk_consistency_annotated.mp4"
    write_scores_csv(scores_csv, scores)
    write_triggers_csv(online_csv, online_triggers)
    if args.save_source_video:
        shutil.copy2(source_video, copied_video)
    if args.save_annotated_video:
        make_annotated_video(source_video, annotated_video, scores, online_triggers, instruction)

    summary = {
        "task_name": task_name,
        "episode_index": episode_index,
        "instruction": instruction,
        "model_path": args.model_path,
        "lerobot_root": str(root),
        "source_video": str(source_video),
        "copied_video": str(copied_video) if args.save_source_video else None,
        "annotated_video": str(annotated_video) if args.save_annotated_video else None,
        "scores_csv": str(scores_csv),
        "online_triggers_csv": str(online_csv),
        "fps": fps,
        "episode_length_frames": episode_length,
        "duration_sec": episode_length / fps,
        "frame_stride": frame_stride,
        "num_predictions": len(frame_indices),
        "action_horizon": int(policy.model.action_head.config.action_horizon),
        "score_definition": "online self-consistency RMSE(prev_groot_chunk[delta:] - curr_groot_chunk[:-delta]) over unnormalized real action dims; no expert action is used in the score",
        "peaks": peaks,
        "online_threshold": float(args.online_threshold),
        "online_cooldown_sec": float(args.online_cooldown_sec),
        "online_ema_alpha": float(args.online_ema_alpha),
        "online_rising_edge_only": bool(args.online_rising_edge_only),
        "online_triggers": online_triggers,
    }
    with scores_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved summary: {scores_json}", flush=True)
    if args.save_source_video:
        print(f"Saved source MP4: {copied_video}", flush=True)
    if args.save_annotated_video:
        print(f"Saved annotated MP4: {annotated_video}", flush=True)
    print("Peaks:")
    for peak in peaks:
        print(f"  frame={peak['frame']} sec={peak['sec']:.2f} score={peak['score']:.5f}", flush=True)
    print(f"Online threshold triggers (threshold={args.online_threshold:g}):")
    for trigger in online_triggers:
        print(
            f"  frame={trigger['frame']} sec={trigger['sec']:.2f} "
            f"score={trigger['score']:.5f}",
            flush=True,
        )
    return summary


def main(args: Args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run GR00T chunk-consistency inference.")

    cfg = DATA_CONFIG_MAP[args.data_config]
    policy = Gr00tPolicy(
        model_path=args.model_path,
        modality_config=cfg.modality_config(),
        modality_transform=cfg.transform(),
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
    )

    task_names = list(args.tasks) if args.tasks else discover_tasks(
        Path(args.composite_root),
        Path(args.plan_cache_dir),
    )
    if not task_names:
        raise RuntimeError(
            "No composite tasks with both a plan cache entry and extracted LeRobot dataset were found."
        )
    print(f"Tasks to process ({len(task_names)}): {', '.join(task_names)}", flush=True)

    jobs: list[tuple[str, int]] = []
    for task_name in task_names:
        root = find_lerobot_root(Path(args.composite_root), task_name)
        current_episodes = episode_indices(root) if args.all_episodes else list(args.episodes)
        jobs.extend((task_name, int(episode_index)) for episode_index in current_episodes)

    resume_enabled = bool(args.resume or args.skip_existing)
    summaries: list[dict[str, Any]] = []
    pending_jobs: list[tuple[str, int]] = []
    skipped = 0
    for task_name, episode_index in jobs:
        existing_summary = (
            Path(args.output_dir)
            / task_name
            / f"episode_{int(episode_index):06d}"
            / "chunk_consistency_summary.json"
        )
        if resume_enabled and existing_summary.is_file():
            summaries.append(load_json(existing_summary))
            skipped += 1
        else:
            pending_jobs.append((task_name, episode_index))

    print(
        f"Episodes total={len(jobs)} pending={len(pending_jobs)} skipped={skipped} "
        f"resume={resume_enabled}",
        flush=True,
    )

    progress_bar = tqdm(
        pending_jobs,
        desc="composite episodes",
        unit="episode",
        disable=not args.progress,
    )
    for task_name, episode_index in progress_bar:
        progress_bar.set_postfix(task=task_name, episode=episode_index)
        summaries.append(analyze_episode(args, policy, task_name, episode_index))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "chunk_consistency_index.json"
    with index_path.open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)
    print(f"Saved index: {index_path}", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
