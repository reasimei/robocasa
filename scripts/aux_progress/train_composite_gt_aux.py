#!/usr/bin/env python3
"""
Train the three-class auxiliary head with Robocasa GT composite subtasks.

This is a new training entry point.  It does not modify or replace the older
chunk-consistency scripts.  By default it mixes:

  - the existing atomic positive/retry pipeline;
  - GT composite progress samples;
  - GT composite success samples;
  - synthetic retry samples created inside GT composite subtask segments.

The frozen GR00T backbone and auxiliary head architecture are reused from the
existing implementation, while the composite data construction is independent.
"""

from __future__ import annotations

import bisect
import copy
import json
import math
import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import tyro
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from transformers import Trainer, TrainingArguments, set_seed

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from scripts.aux_progress.train_atomic_positive_aux import (
    AUX_CONTEXT_NONE,
    AtomicAuxCollator,
    Args as LegacyArgs,
    AuxTrainer,
    GR00TAuxiliaryModel,
    STATE_PROGRESS,
    STATE_RETRY,
    STATE_SUCCESS,
    WeightedAuxDataset,
    build_data_config,
    build_datasets as build_atomic_datasets,
    ensure_hf_resume_checkpoint,
    get_step_data_with_observation_offsets,
    parse_observation_history_offsets,
    retry_observation_offsets,
)


@dataclass
class Args:
    gt_manifest_path: str = (
        "/data/zjw/workspace/Isaac-GR00T/expdata/aux_progress/"
        "composite_gt_aux_manifest.json"
    )
    atomic_manifest_path: str = (
        "/data/zjw/workspace/Isaac-GR00T/expdata/aux_progress/"
        "atomic_positive_manifest.json"
    )
    atomic_retry_manifest_path: str = (
        "/data/zjw/workspace/Isaac-GR00T/expdata/aux_progress/"
        "atomic_retry_manifest.json"
    )
    checkpoint_path: str = (
        "/data/zjw/workspace/Isaac-GR00T/expdata/foundation_model_learning/"
        "target_posttraining/composite_seen/checkpoint-60000"
    )
    output_dir: str = (
        "/data/zjw/workspace/Isaac-GR00T/expdata/aux_progress/"
        "composite_gt_3class_run1"
    )
    data_config: str = "panda_omron"
    embodiment_tag: str = "new_embodiment"
    video_backend: str = "opencv"
    batch_size: int = 32
    max_steps: int = 20000
    save_steps: int = 500
    save_total_limit: int = 10
    resume_from_checkpoint: str = ""
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.05
    dataloader_num_workers: int = 0
    bf16: bool = True
    num_gpus: int = 1
    progress_gamma: float = 1.5
    success_tail_fraction: float = 0.1
    success_tail_min_steps: int = 3
    observation_history_offsets: str = "-8,-4,0"
    synthetic_retry_history: bool = True
    aux_context_mode: str = "state_delta"
    train_split: float = 0.9
    train_epoch_size: int = 200000
    atomic_epoch_size: int = 100000
    atomic_sample_weight: float = 1.0
    gt_progress_sample_weight: float = 1.0
    gt_success_sample_weight: float = 1.0
    gt_retry_sample_weight: float = 0.5
    progress_class_loss_weight: float = 1.0
    success_class_loss_weight: float = 1.0
    retry_class_loss_weight: float = 0.7
    state_label_smoothing: float = 0.02
    seed: int = 42
    report_to: str = "tensorboard"
    wandb_project: str = "robocasa-aux-progress"
    wandb_entity: str = ""
    wandb_mode: str = "online"
    eval_batch_size: int = 32
    eval_max_batches: int = 2048
    eval_subset_seed: int = 123


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def split_episode_ids(
    episodes: Sequence[dict[str, Any]],
    train_split: float,
    seed: int,
) -> tuple[set[int], set[int]]:
    ids = [int(episode["episode_index"]) for episode in episodes]
    rng = random.Random(seed)
    rng.shuffle(ids)
    if len(ids) <= 1:
        return set(ids), set(ids)
    count = max(1, int(len(ids) * train_split))
    train_ids = set(ids[:count])
    val_ids = set(ids[count:]) or {ids[0]}
    return train_ids, val_ids


def replace_language(raw: dict[str, Any], instruction: str) -> None:
    for key, value in list(raw.items()):
        if not key.startswith("annotation."):
            continue
        if isinstance(value, list):
            raw[key] = [instruction for _ in value]
        else:
            raw[key] = [instruction]


class GTCompositePositiveDataset(Dataset):
    def __init__(
        self,
        base_dataset: LeRobotSingleDataset,
        episode_segments: dict[int, list[dict[str, Any]]],
        episode_ids: set[int],
        progress_gamma: float,
        success_tail_fraction: float,
        success_tail_min_steps: int,
        target_state: int,
    ):
        self.base_dataset = base_dataset
        self.progress_gamma = progress_gamma
        self.success_tail_fraction = success_tail_fraction
        self.success_tail_min_steps = success_tail_min_steps
        self.index_records: list[tuple[int, dict[str, Any]]] = []

        starts_by_episode = {
            int(episode_id): [int(segment["start_frame"]) for segment in segments]
            for episode_id, segments in episode_segments.items()
        }
        for base_index, (episode_id, step_index) in enumerate(base_dataset.all_steps):
            episode_id = int(episode_id)
            step_index = int(step_index)
            if episode_id not in episode_ids:
                continue
            segments = episode_segments[episode_id]
            starts = starts_by_episode[episode_id]
            segment_index = bisect.bisect_right(starts, step_index) - 1
            segment_index = max(0, min(segment_index, len(segments) - 1))
            segment = segments[segment_index]
            local_step = step_index - int(segment["start_frame"])
            segment_length = int(segment["length"])
            success_tail = max(
                int(success_tail_min_steps),
                int(math.ceil(success_tail_fraction * segment_length)),
            )
            success_start = max(0, segment_length - success_tail)
            state = STATE_SUCCESS if local_step >= success_start else STATE_PROGRESS
            if state == target_state:
                self.index_records.append((base_index, segment))

    def __len__(self) -> int:
        return len(self.index_records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        base_index, segment = self.index_records[index]
        episode_id, step_index = self.base_dataset.all_steps[base_index]
        segment_start = int(segment["start_frame"])
        segment_length = int(segment["length"])
        local_step = int(step_index) - segment_start
        raw = self.base_dataset.get_step_data(int(episode_id), int(step_index))
        replace_language(raw, str(segment["instruction"]))
        sample = self.base_dataset.transforms(raw)

        if segment_length <= 1:
            progress = 1.0
        else:
            progress = np.clip(
                (local_step / float(segment_length - 1)) ** self.progress_gamma,
                0.0,
                1.0,
            )
        sample["progress_target"] = np.asarray(progress, dtype=np.float32)
        sample["state_target"] = np.asarray(
            STATE_SUCCESS
            if local_step
            >= max(
                0,
                segment_length
                - max(
                    self.success_tail_min_steps,
                    int(math.ceil(self.success_tail_fraction * segment_length)),
                ),
            )
            else STATE_PROGRESS,
            dtype=np.int64,
        )
        sample["episode_index"] = np.asarray(int(episode_id), dtype=np.int64)
        sample["step_index"] = np.asarray(local_step, dtype=np.int64)
        sample["episode_length"] = np.asarray(segment_length, dtype=np.int64)
        sample["retry_type_id"] = np.asarray(-1, dtype=np.int64)
        return sample


class GTCompositeRetryDataset(Dataset):
    def __init__(
        self,
        base_datasets_by_index: dict[int, LeRobotSingleDataset],
        retry_examples: Sequence[dict[str, Any]],
        observation_history_offsets: Sequence[int],
        synthetic_retry_history: bool,
    ):
        self.base_datasets_by_index = base_datasets_by_index
        self.retry_examples = list(retry_examples)
        self.observation_history_offsets = list(observation_history_offsets)
        self.synthetic_retry_history = bool(synthetic_retry_history)

    def __len__(self) -> int:
        return len(self.retry_examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.retry_examples[index]
        dataset = self.base_datasets_by_index[int(example["source_dataset_index"])]
        episode_id = int(example["source_episode_index"])
        step_index = int(example["source_step_index"])
        retry_type = str(example["retry_type"])
        offsets = retry_observation_offsets(
            retry_type,
            self.observation_history_offsets,
            self.synthetic_retry_history,
        )
        raw = get_step_data_with_observation_offsets(
            dataset,
            episode_id,
            step_index,
            offsets,
        )
        replacement = example.get("replacement_instruction")
        if replacement:
            replace_language(raw, str(replacement))
        sample = dataset.transforms(raw)

        sample["progress_target"] = np.asarray(
            float(example.get("signed_progress_target") or 0.0),
            dtype=np.float32,
        )
        sample["state_target"] = np.asarray(STATE_RETRY, dtype=np.int64)
        sample["episode_index"] = np.asarray(episode_id, dtype=np.int64)
        sample["step_index"] = np.asarray(
            step_index - int(example["source_subtask_start_frame"]),
            dtype=np.int64,
        )
        sample["episode_length"] = np.asarray(
            int(example["source_subtask_length"]),
            dtype=np.int64,
        )
        retry_type_ids = {"reverse": 0, "repeat": 1, "mismatch": 2, "backtrack": 3}
        sample["retry_type_id"] = np.asarray(
            retry_type_ids.get(retry_type, -1),
            dtype=np.int64,
        )
        return sample


class GTBestEvalAuxTrainer(AuxTrainer):
    """Evaluate the current GT validation set after each saved checkpoint."""

    def __init__(
        self,
        *args,
        eval_batch_size: int = 32,
        eval_max_batches: int = 2048,
        eval_subset_seed: int = 123,
        **kwargs,
    ):
        self.gt_eval_batch_size = max(int(eval_batch_size), 1)
        self.gt_eval_max_batches = int(eval_max_batches)
        self.gt_eval_subset_seed = int(eval_subset_seed)
        super().__init__(*args, **kwargs)

    def _gt_eval_dataset(self) -> Dataset:
        if self.eval_dataset is None:
            raise RuntimeError("GT checkpoint evaluation requires an eval_dataset.")
        if self.gt_eval_max_batches <= 0:
            return self.eval_dataset

        max_samples = self.gt_eval_max_batches * self.gt_eval_batch_size
        if len(self.eval_dataset) <= max_samples:
            return self.eval_dataset

        rng = random.Random(self.gt_eval_subset_seed)
        indices = rng.sample(range(len(self.eval_dataset)), max_samples)
        return Subset(self.eval_dataset, indices)

    def _evaluate_gt_validation(self) -> dict[str, Any]:
        eval_dataset = self._gt_eval_dataset()
        loader = DataLoader(
            eval_dataset,
            batch_size=self.gt_eval_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )

        model = self.model
        was_training = model.training
        model.eval()
        confusion = torch.zeros((3, 3), dtype=torch.int64)
        progress_abs_error = 0.0
        num_samples = 0
        num_batches = 0
        try:
            with torch.no_grad():
                for batch in loader:
                    batch = self._prepare_inputs(batch)
                    outputs = model(batch)
                    targets = batch["state_target"].detach().view(-1).long()
                    predictions = outputs["state_logits"].detach().argmax(dim=-1).long()
                    flat = targets * 3 + predictions
                    confusion += torch.bincount(flat, minlength=9).cpu().reshape(3, 3)

                    progress_abs_error += float(
                        torch.abs(
                            outputs["progress_pred"].detach().view(-1)
                            - batch["progress_target"].detach().view(-1)
                        )
                        .sum()
                        .item()
                    )
                    num_samples += int(targets.numel())
                    num_batches += 1
        finally:
            if was_training:
                model.train()

        per_class: dict[str, float] = {}
        for class_index, class_name in enumerate(
            ("progress", "success", "retry")
        ):
            class_total = int(confusion[class_index].sum().item())
            per_class[class_name] = (
                float(confusion[class_index, class_index].item()) / class_total
                if class_total > 0
                else 0.0
            )

        total_correct = int(torch.diag(confusion).sum().item())
        state_accuracy = (
            float(total_correct) / num_samples if num_samples > 0 else 0.0
        )
        mean_per_class_accuracy = sum(per_class.values()) / len(per_class)
        return {
            "state_accuracy": state_accuracy,
            "state_per_class_accuracy": per_class,
            "state_per_class_accuracy_mean": mean_per_class_accuracy,
            "state_per_class_accuracy_min": min(per_class.values()),
            "progress_mae": progress_abs_error / num_samples
            if num_samples > 0
            else 0.0,
            "num_samples": num_samples,
            "num_batches": num_batches,
            "confusion_matrix": confusion.tolist(),
            "eval_max_batches": self.gt_eval_max_batches,
            "eval_subset_seed": self.gt_eval_subset_seed,
        }

    def _run_external_eval(self, checkpoint_dir: Path) -> None:
        checkpoint_key = str(checkpoint_dir.resolve())
        if checkpoint_key in self._launched_eval_checkpoints:
            return

        metrics = self._evaluate_gt_validation()
        payload = {
            "global_step": int(self.state.global_step),
            "checkpoint": checkpoint_dir.name,
            **metrics,
        }
        with (checkpoint_dir / "eval_val.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

        self.log(
            {
                f"eval/{key}": value
                for key, value in metrics.items()
                if isinstance(value, (int, float))
            }
        )
        self._launched_eval_checkpoints.add(checkpoint_key)


def build_gt_datasets(args: Args) -> tuple[Dataset, Dataset]:
    manifest = load_manifest(Path(args.gt_manifest_path))
    offsets = parse_observation_history_offsets(args.observation_history_offsets)
    cfg = build_data_config(args.data_config, offsets)
    train_progress_parts: list[Dataset] = []
    train_success_parts: list[Dataset] = []
    val_progress_parts: list[Dataset] = []
    val_success_parts: list[Dataset] = []
    train_bases: dict[int, LeRobotSingleDataset] = {}
    val_bases: dict[int, LeRobotSingleDataset] = {}
    train_retry_examples: list[dict[str, Any]] = []
    val_retry_examples: list[dict[str, Any]] = []

    for dataset_index, record in enumerate(manifest["datasets"]):
        train_ids, val_ids = split_episode_ids(
            record["episodes"],
            args.train_split,
            args.seed,
        )
        episode_segments = {
            int(episode["episode_index"]): list(episode["subtasks"])
            for episode in record["episodes"]
        }
        train_transform = copy.deepcopy(cfg.transform())
        val_transform = copy.deepcopy(cfg.transform())
        train_transform.train()
        val_transform.eval()
        train_base = LeRobotSingleDataset(
            dataset_path=record["lerobot_root"],
            modality_configs=cfg.modality_config(),
            transforms=train_transform,
            embodiment_tag=EmbodimentTag(args.embodiment_tag),
            video_backend=args.video_backend,
        )
        val_base = LeRobotSingleDataset(
            dataset_path=record["lerobot_root"],
            modality_configs=cfg.modality_config(),
            transforms=val_transform,
            embodiment_tag=EmbodimentTag(args.embodiment_tag),
            video_backend=args.video_backend,
        )
        train_bases[dataset_index] = train_base
        val_bases[dataset_index] = val_base

        train_progress_parts.append(
            GTCompositePositiveDataset(
                train_base,
                episode_segments,
                train_ids,
                args.progress_gamma,
                args.success_tail_fraction,
                args.success_tail_min_steps,
                STATE_PROGRESS,
            )
        )
        train_success_parts.append(
            GTCompositePositiveDataset(
                train_base,
                episode_segments,
                train_ids,
                args.progress_gamma,
                args.success_tail_fraction,
                args.success_tail_min_steps,
                STATE_SUCCESS,
            )
        )
        val_progress_parts.append(
            GTCompositePositiveDataset(
                val_base,
                episode_segments,
                val_ids,
                args.progress_gamma,
                args.success_tail_fraction,
                args.success_tail_min_steps,
                STATE_PROGRESS,
            )
        )
        val_success_parts.append(
            GTCompositePositiveDataset(
                val_base,
                episode_segments,
                val_ids,
                args.progress_gamma,
                args.success_tail_fraction,
                args.success_tail_min_steps,
                STATE_SUCCESS,
            )
        )

        for example in manifest.get("retry_examples", []):
            if int(example["source_dataset_index"]) != dataset_index:
                continue
            source_episode = int(example["source_episode_index"])
            if source_episode in train_ids:
                train_retry_examples.append(example)
            elif source_episode in val_ids:
                val_retry_examples.append(example)

    train_progress = ConcatDataset(train_progress_parts)
    train_success = ConcatDataset(train_success_parts)
    val_progress = ConcatDataset(val_progress_parts)
    val_success = ConcatDataset(val_success_parts)
    train_parts: list[Dataset] = [train_progress, train_success]
    train_weights: list[float] = [
        args.gt_progress_sample_weight,
        args.gt_success_sample_weight,
    ]
    val_parts: list[Dataset] = [val_progress, val_success]

    if train_retry_examples:
        train_retry = GTCompositeRetryDataset(
            train_bases,
            train_retry_examples,
            offsets,
            args.synthetic_retry_history,
        )
        train_parts.append(train_retry)
        train_weights.append(args.gt_retry_sample_weight)
    if val_retry_examples:
        val_parts.append(
            GTCompositeRetryDataset(
                val_bases,
                val_retry_examples,
                offsets,
                args.synthetic_retry_history,
            )
        )

    inferred_size = sum(len(part) for part in train_parts)
    train_dataset = WeightedAuxDataset(
        train_parts,
        train_weights,
        args.train_epoch_size if args.train_epoch_size > 0 else inferred_size,
        args.seed,
    )
    val_dataset = ConcatDataset(val_parts)
    print(
        "GT composite dataset sizes: "
        f"progress_train={len(train_progress)} "
        f"success_train={len(train_success)} "
        f"retry_train={len(train_retry_examples)} "
        f"progress_val={len(val_progress)} "
        f"success_val={len(val_success)} "
        f"retry_val={len(val_retry_examples)} "
        f"train_epoch_size={len(train_dataset)}",
        flush=True,
    )
    return train_dataset, val_dataset


def build_training_datasets(args: Args) -> tuple[Dataset, Dataset]:
    gt_train, gt_val = build_gt_datasets(args)
    if not args.atomic_manifest_path:
        return gt_train, gt_val

    legacy_args = LegacyArgs(
        manifest_path=args.atomic_manifest_path,
        retry_manifest_path=args.atomic_retry_manifest_path,
        checkpoint_path=args.checkpoint_path,
        data_config=args.data_config,
        embodiment_tag=args.embodiment_tag,
        video_backend=args.video_backend,
        batch_size=args.batch_size,
        train_epoch_size=args.atomic_epoch_size,
        progress_gamma=args.progress_gamma,
        success_tail_fraction=args.success_tail_fraction,
        success_tail_min_steps=args.success_tail_min_steps,
        observation_history_offsets=args.observation_history_offsets,
        synthetic_retry_history=args.synthetic_retry_history,
        aux_context_mode=args.aux_context_mode,
        train_split=args.train_split,
        seed=args.seed,
    )
    atomic_train, atomic_val = build_atomic_datasets(legacy_args)
    train = WeightedAuxDataset(
        [atomic_train, gt_train],
        [args.atomic_sample_weight, 1.0],
        args.train_epoch_size,
        args.seed + 101,
    )
    val = ConcatDataset([atomic_val, gt_val])
    print(
        f"Mixed train datasets: atomic_epoch={len(atomic_train)} "
        f"gt_epoch={len(gt_train)} mixed_epoch={len(train)} val={len(val)}",
        flush=True,
    )
    return train, val


def main(args: Args) -> None:
    set_seed(args.seed)
    if args.save_steps <= 0:
        raise ValueError(f"save_steps must be positive, got {args.save_steps}")
    if args.save_total_limit <= 0:
        raise ValueError(
            f"save_total_limit must be positive when retaining best checkpoints, "
            f"got {args.save_total_limit}"
        )
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except RuntimeError:
        pass
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-visible GPU is required for auxiliary training.")

    train_dataset, val_dataset = build_training_datasets(args)
    model = GR00TAuxiliaryModel(
        args.checkpoint_path,
        state_loss_weights=[
            args.progress_class_loss_weight,
            args.success_class_loss_weight,
            args.retry_class_loss_weight,
        ],
        state_label_smoothing=args.state_label_smoothing,
        aux_context_mode=args.aux_context_mode,
        observation_history_offsets=parse_observation_history_offsets(
            args.observation_history_offsets
        ),
        synthetic_retry_history=args.synthetic_retry_history,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "gt_training_config.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "gt_manifest_path": args.gt_manifest_path,
                "gt_manifest": load_manifest(Path(args.gt_manifest_path)).get(
                    "num_episodes", 0
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        if args.wandb_entity:
            os.environ.setdefault("WANDB_ENTITY", args.wandb_entity)
        os.environ.setdefault("WANDB_MODE", args.wandb_mode)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=output_dir.name,
        remove_unused_columns=False,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=False,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_steps=args.max_steps,
        num_train_epochs=100,
        save_steps=args.save_steps,
        save_strategy="steps",
        # GTBestEvalAuxTrainer prunes after evaluating the just-saved checkpoint.
        # Disable Trainer's chronological rotation so it cannot delete a checkpoint
        # before its validation score has been compared against previous saves.
        save_total_limit=None,
        logging_steps=10,
        report_to=[args.report_to] if args.report_to != "none" else [],
        bf16=args.bf16,
        tf32=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        seed=args.seed,
        dataloader_drop_last=False,
    )
    trainer = GTBestEvalAuxTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=AtomicAuxCollator(),
        external_eval_gpu=-1,
        external_eval_sync=True,
        checkpoint_retention_strategy="best_eval",
        best_checkpoint_metric="state_per_class_accuracy_mean",
        best_checkpoint_mode="max",
        best_checkpoint_keep_n=args.save_total_limit,
        best_checkpoint_keep_unevaluated=0,
        eval_batch_size=args.eval_batch_size,
        eval_max_batches=args.eval_max_batches,
        eval_subset_seed=args.eval_subset_seed,
        manifest_path=args.atomic_manifest_path,
        retry_manifest_path=args.atomic_retry_manifest_path,
        base_checkpoint_path=args.checkpoint_path,
        data_config_name=args.data_config,
        embodiment_tag=args.embodiment_tag,
        video_backend=args.video_backend,
        progress_gamma=args.progress_gamma,
        success_tail_fraction=args.success_tail_fraction,
        success_tail_min_steps=args.success_tail_min_steps,
        observation_history_offsets=args.observation_history_offsets,
        synthetic_retry_history=args.synthetic_retry_history,
        aux_context_mode=args.aux_context_mode,
        train_split=args.train_split,
        seed=args.seed,
    )
    resume = (
        ensure_hf_resume_checkpoint(args.resume_from_checkpoint)
        if args.resume_from_checkpoint
        else None
    )
    trainer.train(resume_from_checkpoint=resume)
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    trainer._run_external_eval(final_dir)
    print(f"Saved and evaluated final GT auxiliary checkpoint at {final_dir}")


if __name__ == "__main__":
    main(tyro.cli(Args))
