#!/usr/bin/env python3
"""
Offline evaluation for the positive-only auxiliary heads on Robocasa atomic datasets.

This script evaluates:
1. progress regression quality
2. state classification quality for {progress, success}

It reuses the dataset wrapper from training, loads the frozen GR00T backbone, and restores
the saved auxiliary heads from an output directory that contains:
  - aux_heads.pt
  - aux_config.json
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import tyro
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.aux_progress.train_atomic_positive_aux import (
    AUX_CONTEXT_NONE,
    Args as TrainArgs,
    AtomicAuxCollator,
    GR00TAuxiliaryModel,
    STATE_CLASS_NAMES,
    STATE_PROGRESS,
    STATE_RETRY,
    STATE_SUCCESS,
    build_datasets,
    parse_observation_history_offsets,
)


@dataclass
class Args:
    aux_dir: str
    manifest_path: str = "/data/zjw/workspace/Isaac-GR00T/expdata/aux_progress/atomic_positive_manifest.json"
    retry_manifest_path: str = "/data/zjw/workspace/Isaac-GR00T/expdata/aux_progress/atomic_retry_manifest.json"
    composite_manifest_path: str = ""
    checkpoint_path: str = "/data/zjw/workspace/Isaac-GR00T/expdata/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000"
    data_config: str = "panda_omron"
    embodiment_tag: str = "new_embodiment"
    video_backend: str = "opencv"
    batch_size: int = 16
    dataloader_num_workers: int = 0
    progress_gamma: float = 1.5
    success_tail_fraction: float = 0.1
    success_tail_min_steps: int = 3
    retry_progress_default: float = 0.0
    retry_manifest_max_samples: int = -1
    state_label_smoothing: float = 0.0
    observation_history_offsets: str = "auto"
    synthetic_retry_history: bool = True
    aux_context_mode: str = "auto"
    train_split: float = 0.9
    seed: int = 42
    split: str = "val"
    max_batches: int = -1
    subset_mode: str = "sequential"
    subset_seed: int = 42
    success_threshold: float = 0.2
    log_every: int = 100


def load_aux_config(aux_dir: Path) -> dict:
    config_path = aux_dir / "aux_config.json"
    if not config_path.is_file():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_aux_state(aux_dir: Path) -> Dict[str, torch.Tensor]:
    ckpt_path = aux_dir / "aux_heads.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing auxiliary checkpoint: {ckpt_path}")
    return torch.load(ckpt_path, map_location="cpu")


def resolve_aux_runtime_args(args: Args, aux_config: dict) -> tuple[str, str, float, bool]:
    aux_context_mode = args.aux_context_mode
    if aux_context_mode == "auto":
        aux_context_mode = str(aux_config.get("aux_context_mode", AUX_CONTEXT_NONE))

    observation_history_offsets = args.observation_history_offsets
    if observation_history_offsets == "auto":
        configured_offsets = aux_config.get("observation_history_offsets", [])
        if configured_offsets:
            observation_history_offsets = ",".join(str(item) for item in configured_offsets)
        else:
            observation_history_offsets = ""

    state_label_smoothing = args.state_label_smoothing
    if "state_label_smoothing" in aux_config:
        state_label_smoothing = float(aux_config["state_label_smoothing"])

    synthetic_retry_history = bool(aux_config.get("synthetic_retry_history", args.synthetic_retry_history))

    return aux_context_mode, observation_history_offsets, state_label_smoothing, synthetic_retry_history


def build_eval_dataset(args: Args, observation_history_offsets: str, synthetic_retry_history: bool):
    train_args = TrainArgs(
        manifest_path=args.manifest_path,
        retry_manifest_path=args.retry_manifest_path,
        composite_manifest_path=args.composite_manifest_path,
        checkpoint_path=args.checkpoint_path,
        data_config=args.data_config,
        embodiment_tag=args.embodiment_tag,
        video_backend=args.video_backend,
        batch_size=args.batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
        progress_gamma=args.progress_gamma,
        success_tail_fraction=args.success_tail_fraction,
        success_tail_min_steps=args.success_tail_min_steps,
        retry_progress_default=args.retry_progress_default,
        retry_manifest_max_samples=args.retry_manifest_max_samples,
        observation_history_offsets=observation_history_offsets,
        synthetic_retry_history=synthetic_retry_history,
        train_split=args.train_split,
        seed=args.seed,
    )
    train_dataset, val_dataset = build_datasets(train_args)
    if args.split == "train":
        return train_dataset
    if args.split == "val":
        return val_dataset
    raise ValueError(f"Unsupported split: {args.split}")


def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    f1s: List[float] = []
    for class_id in range(num_classes):
        tp = np.sum((y_true == class_id) & (y_pred == class_id))
        fp = np.sum((y_true != class_id) & (y_pred == class_id))
        fn = np.sum((y_true == class_id) & (y_pred != class_id))
        if tp == 0 and fp == 0 and fn == 0:
            f1s.append(0.0)
            continue
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if precision + recall == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * precision * recall / (precision + recall))
    return float(np.mean(f1s))


def compute_binary_precision_recall_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_class: int,
) -> tuple[float, float, float]:
    tp = int(np.sum((y_true == positive_class) & (y_pred == positive_class)))
    fp = int(np.sum((y_true != positive_class) & (y_pred == positive_class)))
    fn = int(np.sum((y_true == positive_class) & (y_pred != positive_class)))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return float(precision), float(recall), float(f1)


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> list[list[int]]:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_class in range(num_classes):
        for pred_class in range(num_classes):
            matrix[true_class, pred_class] = int(
                np.sum((y_true == true_class) & (y_pred == pred_class))
            )
    return matrix.tolist()


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


def main(args: Args) -> None:
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except RuntimeError:
        pass

    aux_dir = Path(args.aux_dir)
    aux_config = load_aux_config(aux_dir)
    (
        aux_context_mode,
        observation_history_offsets,
        state_label_smoothing,
        synthetic_retry_history,
    ) = resolve_aux_runtime_args(
        args,
        aux_config,
    )
    full_dataset = build_eval_dataset(args, observation_history_offsets, synthetic_retry_history)
    dataset, subset_info = maybe_build_subset(full_dataset, args)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        pin_memory=False,
        persistent_workers=args.dataloader_num_workers > 0,
        collate_fn=AtomicAuxCollator(),
    )
    total_batches = len(dataloader)
    effective_max_batches = min(args.max_batches, total_batches) if args.max_batches > 0 else total_batches

    model = GR00TAuxiliaryModel(
        args.checkpoint_path,
        state_label_smoothing=state_label_smoothing,
        aux_context_mode=aux_context_mode,
        observation_history_offsets=parse_observation_history_offsets(observation_history_offsets),
        synthetic_retry_history=synthetic_retry_history,
    )
    missing, unexpected = model.load_state_dict(load_aux_state(aux_dir), strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys in auxiliary checkpoint: {unexpected}")
    allowed_missing = {"state_loss_fn.weight"}
    missing = [
        name
        for name in missing
        if not name.startswith("base_model.") and name not in allowed_missing
    ]
    if missing:
        raise RuntimeError(f"Missing keys when loading auxiliary checkpoint: {missing}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for auxiliary evaluation because the Eagle backbone uses flash-attn.")
    device = torch.device("cuda")
    model = model.to(device)
    print(
        f"Using CUDA device {torch.cuda.current_device()}: "
        f"{torch.cuda.get_device_name(torch.cuda.current_device())}"
    )
    print(
        f"Evaluating split={args.split} with {len(dataset)} samples "
        f"(full_dataset={len(full_dataset)}), "
        f"batch_size={args.batch_size}, total_batches={total_batches}, "
        f"max_batches={args.max_batches}, subset_mode={args.subset_mode}",
        flush=True,
    )
    print(
        f"Aux runtime config: aux_context_mode={aux_context_mode}, "
        f"observation_history_offsets={observation_history_offsets or '[0]'}, "
        f"synthetic_retry_history={synthetic_retry_history}",
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
    model.eval()

    all_progress_true: List[np.ndarray] = []
    all_progress_pred: List[np.ndarray] = []
    all_state_true: List[np.ndarray] = []
    all_state_pred: List[np.ndarray] = []
    all_state_success_prob: List[np.ndarray] = []
    all_state_retry_prob: List[np.ndarray] = []

    start_time = time.time()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            progress_true = batch["progress_target"].numpy()
            state_true = batch["state_target"].numpy()

            for key, value in batch.items():
                if torch.is_tensor(value):
                    batch[key] = value.to(device)

            outputs = model(batch)
            progress_pred = outputs["progress_pred"].detach().cpu().numpy()
            state_logits = outputs["state_logits"]
            state_probs = torch.softmax(state_logits, dim=-1).detach().cpu().numpy()
            success_prob = state_probs[:, STATE_SUCCESS]
            retry_prob = state_probs[:, STATE_RETRY]
            state_pred = np.argmax(state_probs, axis=-1).astype(np.int64)

            all_progress_true.append(progress_true)
            all_progress_pred.append(progress_pred)
            all_state_true.append(state_true)
            all_state_pred.append(state_pred)
            all_state_success_prob.append(success_prob)
            all_state_retry_prob.append(retry_prob)

            if args.log_every > 0 and ((batch_idx + 1) % args.log_every == 0 or (batch_idx + 1) == total_batches):
                elapsed = time.time() - start_time
                done = batch_idx + 1
                avg_sec_per_batch = elapsed / max(done, 1)
                remaining = max(effective_max_batches - done, 0)
                eta_sec = remaining * avg_sec_per_batch
                print(
                    f"[eval] batch {done}/{effective_max_batches} "
                    f"({done / max(effective_max_batches, 1):.2%}) "
                    f"elapsed={elapsed / 60.0:.1f}m "
                    f"eta={eta_sec / 60.0:.1f}m",
                    flush=True,
                )

    progress_true = np.concatenate(all_progress_true, axis=0)
    progress_pred = np.concatenate(all_progress_pred, axis=0)
    state_true = np.concatenate(all_state_true, axis=0)
    state_pred = np.concatenate(all_state_pred, axis=0)
    state_success_prob = np.concatenate(all_state_success_prob, axis=0)
    state_retry_prob = np.concatenate(all_state_retry_prob, axis=0)

    progress_mae = float(np.mean(np.abs(progress_pred - progress_true)))
    progress_rmse = float(np.sqrt(np.mean((progress_pred - progress_true) ** 2)))
    state_acc = float(np.mean(state_pred == state_true))
    num_classes = len(STATE_CLASS_NAMES)
    macro_f1 = compute_macro_f1(state_true, state_pred, num_classes=num_classes)
    success_precision, success_recall, success_f1 = compute_binary_precision_recall_f1(
        state_true,
        state_pred,
        positive_class=STATE_SUCCESS,
    )
    retry_precision, retry_recall, retry_f1 = compute_binary_precision_recall_f1(
        state_true,
        state_pred,
        positive_class=STATE_RETRY,
    )
    confusion_matrix = compute_confusion_matrix(state_true, state_pred, num_classes=num_classes)

    per_class_acc = {}
    for class_id, class_name in enumerate(STATE_CLASS_NAMES):
        mask = state_true == class_id
        per_class_acc[class_name] = float(np.mean(state_pred[mask] == state_true[mask])) if np.any(mask) else 0.0

    metrics = {
        "split": args.split,
        "num_samples": int(progress_true.shape[0]),
        "decision": "argmax",
        "success_threshold": args.success_threshold,
        "progress_mae": progress_mae,
        "progress_rmse": progress_rmse,
        "state_accuracy": state_acc,
        "state_macro_f1": macro_f1,
        "success_precision": success_precision,
        "success_recall": success_recall,
        "success_f1": success_f1,
        "retry_precision": retry_precision,
        "retry_recall": retry_recall,
        "retry_f1": retry_f1,
        "state_per_class_accuracy": per_class_acc,
        "state_confusion_matrix": {
            "labels": STATE_CLASS_NAMES,
            "rows_are_true": True,
            "cols_are_pred": True,
            "matrix": confusion_matrix,
        },
        "state_support": {
            class_name: int(np.sum(state_true == class_id))
            for class_id, class_name in enumerate(STATE_CLASS_NAMES)
        },
        "success_prob_stats": {
            "mean": float(np.mean(state_success_prob)),
            "std": float(np.std(state_success_prob)),
            "min": float(np.min(state_success_prob)),
            "max": float(np.max(state_success_prob)),
        },
        "retry_prob_stats": {
            "mean": float(np.mean(state_retry_prob)),
            "std": float(np.std(state_retry_prob)),
            "min": float(np.min(state_retry_prob)),
            "max": float(np.max(state_retry_prob)),
        },
    }

    print(json.dumps(metrics, indent=2))

    metrics_path = aux_dir / f"eval_{args.split}.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
