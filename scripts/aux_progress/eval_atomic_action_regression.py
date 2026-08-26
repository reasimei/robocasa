#!/usr/bin/env python3
"""
Offline evaluation for the action-regression variant that predicts success/progress as action dimensions.
"""

from __future__ import annotations

import copy
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import tyro
from torch.utils.data import ConcatDataset, DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.model.policy import Gr00tPolicy
from gr00t.model.transforms import DefaultDataCollator
from gr00t.utils.eval import convert_nested_float64_to_float32
from scripts.aux_progress.train_atomic_action_regression import Args as TrainArgs
from scripts.aux_progress.train_atomic_action_regression import (
    ActionRegressionAtomicDataset,
    build_datasets,
    index_subset_for_episode_ids,
    load_manifest,
    split_episode_indices,
)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class Args:
    model_path: str
    manifest_path: str = "/data/zjw/workspace/Isaac-GR00T/expdata/aux_progress/atomic_positive_manifest.json"
    data_config: str = "panda_omron"
    embodiment_tag: str = "new_embodiment"
    video_backend: str = "opencv"
    batch_size: int = 8
    dataloader_num_workers: int = 0
    progress_gamma: float = 1.5
    success_tail_fraction: float = 0.1
    success_tail_min_steps: int = 3
    train_split: float = 0.9
    seed: int = 42
    split: str = "val"
    max_batches: int = -1
    subset_mode: str = "random"
    subset_seed: int = 42
    success_threshold: float = 0.5
    log_every: int = 100
    denoising_steps: int = 4
    max_episodes: int = -1
    episode_subset_mode: str = "sequential"
    episode_subset_seed: int = 42
    max_episode_steps: int = -1


def build_eval_dataset(args: Args):
    cfg = DATA_CONFIG_MAP[args.data_config]
    manifest = load_manifest(Path(args.manifest_path))
    train_parts = []
    val_parts = []

    for record in manifest["datasets"]:
        train_transform = copy.deepcopy(cfg.transform())
        val_transform = copy.deepcopy(cfg.transform())
        train_transform.train()
        # Eval needs action targets present in each sample, so keep the transform in training mode
        # while still evaluating on the held-out split.
        val_transform.train()

        lerobot_root = record["lerobot_root"]
        episode_lengths = {int(ep["episode_index"]): int(ep["length"]) for ep in record["episodes"]}
        train_ids, val_ids = split_episode_indices(record["episodes"], args.train_split, args.seed)

        base_train = LeRobotSingleDataset(
            dataset_path=lerobot_root,
            modality_configs=cfg.modality_config(),
            transforms=train_transform,
            embodiment_tag=EmbodimentTag(args.embodiment_tag),
            video_backend=args.video_backend,
        )
        base_val = LeRobotSingleDataset(
            dataset_path=lerobot_root,
            modality_configs=cfg.modality_config(),
            transforms=val_transform,
            embodiment_tag=EmbodimentTag(args.embodiment_tag),
            video_backend=args.video_backend,
        )

        train_indices = index_subset_for_episode_ids(base_train, train_ids)
        val_indices = index_subset_for_episode_ids(base_val, val_ids)
        action_offsets = list(cfg.action_indices)

        train_parts.append(
            ActionRegressionAtomicDataset(
                base_dataset=base_train,
                episode_lengths=episode_lengths,
                indices=train_indices,
                action_offsets=action_offsets,
                progress_gamma=args.progress_gamma,
                success_tail_fraction=args.success_tail_fraction,
                success_tail_min_steps=args.success_tail_min_steps,
                success_oversample_factor=1,
            )
        )
        val_parts.append(
            ActionRegressionAtomicDataset(
                base_dataset=base_val,
                episode_lengths=episode_lengths,
                indices=val_indices,
                action_offsets=action_offsets,
                progress_gamma=args.progress_gamma,
                success_tail_fraction=args.success_tail_fraction,
                success_tail_min_steps=args.success_tail_min_steps,
                success_oversample_factor=1,
            )
        )

    train_dataset = ConcatDataset(train_parts)
    val_dataset = ConcatDataset(val_parts)
    if args.split == "train":
        return train_dataset
    if args.split == "val":
        return val_dataset
    raise ValueError(f"Unsupported split: {args.split}")


def maybe_build_subset(dataset, args: Args):
    if args.max_batches <= 0:
        return dataset, None
    if args.subset_mode not in {"sequential", "random"}:
        raise ValueError(f"Unsupported subset_mode: {args.subset_mode}")

    requested_samples = args.max_batches * args.batch_size
    if requested_samples <= 0 or requested_samples >= len(dataset):
        return dataset, {
            "mode": args.subset_mode,
            "requested_samples": requested_samples,
            "actual_samples": len(dataset),
            "subset_seed": args.subset_seed,
        }

    if args.subset_mode == "sequential":
        subset_indices = list(range(requested_samples))
    else:
        rng = np.random.default_rng(args.subset_seed)
        subset_indices = rng.choice(len(dataset), size=requested_samples, replace=False).tolist()

    return Subset(dataset, subset_indices), {
        "mode": args.subset_mode,
        "requested_samples": requested_samples,
        "actual_samples": len(subset_indices),
        "subset_seed": args.subset_seed,
    }


def compute_binary_stats(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float | list[list[int]]]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def get_action_modality_keys(data_config_name: str) -> list[str]:
    cfg = DATA_CONFIG_MAP[data_config_name]
    keys: list[str] = []
    for key in cfg.action_keys:
        if not key.startswith("action."):
            raise ValueError(f"Unexpected action key format: {key}")
        keys.append(key.split("action.", 1)[1])
    return keys


def build_policy_eval_episodes(args: Args) -> list[dict[str, Any]]:
    manifest = load_manifest(Path(args.manifest_path))
    episodes: list[dict[str, Any]] = []
    for record in manifest["datasets"]:
        train_ids, val_ids = split_episode_indices(record["episodes"], args.train_split, args.seed)
        selected_ids = train_ids if args.split == "train" else val_ids
        for episode in record["episodes"]:
            episode_index = int(episode["episode_index"])
            if episode_index not in selected_ids:
                continue
            episodes.append(
                {
                    "lerobot_root": record["lerobot_root"],
                    "episode_index": episode_index,
                    "length": int(episode["length"]),
                }
            )

    if args.max_episodes > 0 and args.max_episodes < len(episodes):
        if args.episode_subset_mode not in {"sequential", "random"}:
            raise ValueError(f"Unsupported episode_subset_mode: {args.episode_subset_mode}")
        if args.episode_subset_mode == "sequential":
            episodes = episodes[: args.max_episodes]
        else:
            rng = np.random.default_rng(args.episode_subset_seed)
            indices = rng.choice(len(episodes), size=args.max_episodes, replace=False)
            episodes = [episodes[int(i)] for i in indices]
    return episodes


def calc_action_mse_for_episode(
    policy: Gr00tPolicy,
    dataset: LeRobotSingleDataset,
    trajectory_id: int,
    modality_keys: list[str],
    steps: int,
    action_horizon: int,
) -> float:
    gt_action_across_time: list[np.ndarray] = []
    pred_action_across_time: list[np.ndarray] = []

    for step_count in range(steps):
        data_point = convert_nested_float64_to_float32(dataset.get_step_data(trajectory_id, step_count))
        concat_gt_action = np.concatenate(
            [data_point[f"action.{key}"][0] for key in modality_keys],
            axis=0,
        )
        gt_action_across_time.append(concat_gt_action)

        if step_count % action_horizon == 0:
            action_chunk = policy.get_action(data_point)
            for chunk_step in range(action_horizon):
                concat_pred_action = np.concatenate(
                    [np.atleast_1d(action_chunk[f"action.{key}"][chunk_step]) for key in modality_keys],
                    axis=0,
                )
                pred_action_across_time.append(concat_pred_action)

    gt_action = np.array(gt_action_across_time)
    pred_action = np.array(pred_action_across_time)[:steps]
    if gt_action.shape != pred_action.shape:
        raise ValueError(
            f"Action shape mismatch for trajectory {trajectory_id}: "
            f"{gt_action.shape=} vs {pred_action.shape=}"
        )
    if np.isnan(pred_action).any():
        raise ValueError(f"Pred action has NaN for trajectory {trajectory_id}")
    return float(np.mean((gt_action - pred_action) ** 2))


def evaluate_actions_official_style(args: Args) -> dict[str, Any]:
    data_config = DATA_CONFIG_MAP[args.data_config]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()
    policy = Gr00tPolicy(
        model_path=args.model_path,
        modality_config=modality_config,
        modality_transform=modality_transform,
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    action_horizon = len(data_config.action_indices)
    action_modality_keys = get_action_modality_keys(args.data_config)
    episodes = build_policy_eval_episodes(args)
    dataset_cache: dict[str, LeRobotSingleDataset] = {}
    episode_mse: list[float] = []
    steps_per_episode: list[int] = []

    for episode in episodes:
        dataset_path = episode["lerobot_root"]
        if dataset_path not in dataset_cache:
            dataset_cache[dataset_path] = LeRobotSingleDataset(
                dataset_path=dataset_path,
                modality_configs=policy.get_modality_config(),
                video_backend=args.video_backend,
                video_backend_kwargs=None,
                transforms=None,
                embodiment_tag=args.embodiment_tag,
            )
        dataset = dataset_cache[dataset_path]
        steps = int(episode["length"])
        if args.max_episode_steps > 0:
            steps = min(steps, args.max_episode_steps)
        steps = max(1, steps)
        episode_mse.append(
            calc_action_mse_for_episode(
                policy=policy,
                dataset=dataset,
                trajectory_id=int(episode["episode_index"]),
                modality_keys=action_modality_keys,
                steps=steps,
                action_horizon=action_horizon,
            )
        )
        steps_per_episode.append(steps)

    if not episode_mse:
        return {
            "official_action_num_episodes": 0,
            "official_action_action_horizon": action_horizon,
            "official_action_modality_keys": action_modality_keys,
        }

    mse_array = np.asarray(episode_mse, dtype=np.float64)
    steps_array = np.asarray(steps_per_episode, dtype=np.float64)
    return {
        "official_action_num_episodes": len(episode_mse),
        "official_action_mean_mse": float(np.mean(mse_array)),
        "official_action_std_mse": float(np.std(mse_array)),
        "official_action_min_mse": float(np.min(mse_array)),
        "official_action_max_mse": float(np.max(mse_array)),
        "official_action_mean_steps": float(np.mean(steps_array)),
        "official_action_action_horizon": action_horizon,
        "official_action_modality_keys": action_modality_keys,
    }


def main(args: Args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GR00T action-regression evaluation.")
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except RuntimeError:
        pass

    full_dataset = build_eval_dataset(args)
    dataset, subset_info = maybe_build_subset(full_dataset, args)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        pin_memory=False,
        persistent_workers=args.dataloader_num_workers > 0,
        collate_fn=DefaultDataCollator(),
    )
    total_batches = len(dataloader)

    model = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path=args.model_path,
        tune_llm=False,
        tune_visual=False,
        tune_projector=False,
        tune_diffusion_model=False,
    ).to("cuda")
    model.eval()

    print(
        f"Using CUDA device {torch.cuda.current_device()}: "
        f"{torch.cuda.get_device_name(torch.cuda.current_device())}"
    )
    print(
        f"Evaluating split={args.split} with {len(dataset)} samples "
        f"(full_dataset={len(full_dataset)}), batch_size={args.batch_size}, "
        f"total_batches={total_batches}, subset_mode={args.subset_mode}, "
        f"max_batches={args.max_batches}",
        flush=True,
    )
    if subset_info is not None:
        print(
            f"Subset details: mode={subset_info['mode']}, "
            f"requested_samples={subset_info['requested_samples']}, "
            f"actual_samples={subset_info['actual_samples']}, "
            f"subset_seed={subset_info['subset_seed']}",
            flush=True,
        )

    all_success_true: List[np.ndarray] = []
    all_success_pred: List[np.ndarray] = []
    all_progress_true: List[np.ndarray] = []
    all_progress_pred: List[np.ndarray] = []

    start_time = time.time()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for batch_idx, batch in enumerate(dataloader):
            outputs = model.get_action(batch)
            pred_actions = outputs["action_pred"].detach().cpu()
            target_actions = batch["action"].detach().cpu()
            success_indices = batch["success_action_index"].long()
            progress_indices = batch["progress_action_index"].long()
            success_idx = success_indices[:, None, None].expand(-1, pred_actions.shape[1], 1)
            progress_idx = progress_indices[:, None, None].expand(-1, pred_actions.shape[1], 1)

            pred_success = torch.gather(pred_actions, 2, success_idx).squeeze(-1).numpy()
            true_success = torch.gather(target_actions, 2, success_idx).squeeze(-1).numpy()
            pred_progress = torch.gather(pred_actions, 2, progress_idx).squeeze(-1).numpy()
            true_progress = torch.gather(target_actions, 2, progress_idx).squeeze(-1).numpy()

            # The diffusion head emits unconstrained reals. Interpret the semantic action channels
            # as bounded regression outputs in [0, 1].
            pred_success = sigmoid_np(pred_success)
            pred_progress = sigmoid_np(pred_progress)

            all_success_true.append(true_success)
            all_success_pred.append(pred_success)
            all_progress_true.append(true_progress)
            all_progress_pred.append(pred_progress)

            if args.log_every > 0 and ((batch_idx + 1) % args.log_every == 0 or (batch_idx + 1) == total_batches):
                elapsed = time.time() - start_time
                done = batch_idx + 1
                avg_sec_per_batch = elapsed / max(done, 1)
                eta_sec = max(total_batches - done, 0) * avg_sec_per_batch
                print(
                    f"[eval] batch {done}/{total_batches} "
                    f"({done / max(total_batches, 1):.2%}) "
                    f"elapsed={elapsed / 60.0:.1f}m eta={eta_sec / 60.0:.1f}m",
                    flush=True,
                )

    success_true = np.concatenate(all_success_true, axis=0)
    success_pred = np.concatenate(all_success_pred, axis=0)
    progress_true = np.concatenate(all_progress_true, axis=0)
    progress_pred = np.concatenate(all_progress_pred, axis=0)

    current_success_true = success_true[:, 0]
    current_success_pred = success_pred[:, 0]
    current_progress_true = progress_true[:, 0]
    current_progress_pred = progress_pred[:, 0]

    current_success_binary = (current_success_pred >= args.success_threshold).astype(np.int64)
    success_stats = compute_binary_stats(current_success_true.astype(np.int64), current_success_binary)
    official_action_metrics = evaluate_actions_official_style(args)

    metrics = {
        "split": args.split,
        "num_samples": int(success_true.shape[0]),
        "success_threshold": args.success_threshold,
        "current_success_mae": float(np.mean(np.abs(current_success_pred - current_success_true))),
        "full_horizon_success_mae": float(np.mean(np.abs(success_pred - success_true))),
        "current_progress_mae": float(np.mean(np.abs(current_progress_pred - current_progress_true))),
        "full_horizon_progress_mae": float(np.mean(np.abs(progress_pred - progress_true))),
        "current_progress_rmse": float(np.sqrt(np.mean((current_progress_pred - current_progress_true) ** 2))),
        "full_horizon_progress_rmse": float(np.sqrt(np.mean((progress_pred - progress_true) ** 2))),
        "current_success_thresholded": success_stats,
        "success_pred_stats": {
            "mean": float(np.mean(current_success_pred)),
            "std": float(np.std(current_success_pred)),
            "min": float(np.min(current_success_pred)),
            "max": float(np.max(current_success_pred)),
        },
        "progress_pred_stats": {
            "mean": float(np.mean(current_progress_pred)),
            "std": float(np.std(current_progress_pred)),
            "min": float(np.min(current_progress_pred)),
            "max": float(np.max(current_progress_pred)),
        },
    }
    metrics.update(official_action_metrics)

    print(json.dumps(metrics, indent=2))

    metrics_path = Path(args.model_path) / f"action_regression_eval_{args.split}.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
