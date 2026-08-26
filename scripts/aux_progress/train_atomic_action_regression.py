#!/usr/bin/env python3
"""
Train GR00T with success/progress injected into the action tensor itself.

This follows the CycleVLA-style idea more closely than the auxiliary-head baseline:
1. `success` is treated as a scalar regression target in {0, 1}
2. `progress` is treated as a scalar regression target in [0, 1]
3. Both values are appended to the existing padded action tensor and supervised by the
   standard GR00T flow-matching action loss.

Implementation note:
GR00T already pads actions to `max_action_dim=32` and masks valid action channels with
`action_mask`. Instead of rebuilding the action head to change its shape, we activate two
previously masked channels in the padded action tensor. This keeps the checkpoint loading
path simple while still making success/progress true action outputs of the model.

Success and progress are both trained as regression targets:
- success target is binary {0, 1}
- progress target is continuous [0, 1]

Because the GR00T diffusion head itself emits unconstrained continuous values, downstream
evaluation / inference should map these two decoded channels back into [0, 1] before they
are interpreted as semantic signals.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import tyro
from torch.utils.data import ConcatDataset, Dataset
from transformers import TrainingArguments, set_seed

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.experiment.trainer import DualBrainTrainer
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.model.transforms import DefaultDataCollator
from gr00t.utils.experiment import CheckpointFormatCallback, safe_save_model_for_hf_trainer


@dataclass
class Args:
    manifest_path: str = "/data/zjw/workspace/Isaac-GR00T/expdata/aux_progress/atomic_positive_manifest.json"
    checkpoint_path: str = "/data/zjw/workspace/Isaac-GR00T/expdata/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000"
    output_dir: str = "/data/zjw/workspace/Isaac-GR00T/expdata/action_regression/atomic_success_progress"
    data_config: str = "panda_omron"
    embodiment_tag: str = "new_embodiment"
    video_backend: str = "opencv"
    batch_size: int = 32
    max_steps: int = 20000
    save_steps: int = 500
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.05
    dataloader_num_workers: int = 0
    bf16: bool = True
    progress_gamma: float = 1.5
    success_tail_fraction: float = 0.1
    success_tail_min_steps: int = 3
    train_split: float = 0.9
    seed: int = 42
    report_to: str = "tensorboard"
    wandb_project: str = "robocasa-aux-progress"
    wandb_entity: str = ""
    wandb_mode: str = "online"
    resume: bool = False
    tune_llm: bool = False
    tune_visual: bool = False
    tune_projector: bool = True
    tune_diffusion_model: bool = True
    success_oversample_factor: int = 1


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def split_episode_indices(
    episodes: Sequence[dict[str, Any]],
    train_split: float,
    seed: int,
) -> tuple[set[int], set[int]]:
    episode_ids = [int(ep["episode_index"]) for ep in episodes]
    rng = random.Random(seed)
    rng.shuffle(episode_ids)
    if len(episode_ids) == 1:
        return set(episode_ids), set(episode_ids)
    train_count = max(1, int(len(episode_ids) * train_split))
    train_ids = set(episode_ids[:train_count])
    val_ids = set(episode_ids[train_count:])
    if not val_ids:
        val_ids = set(list(train_ids)[:1])
    return train_ids, val_ids


def index_subset_for_episode_ids(dataset: LeRobotSingleDataset, episode_ids: set[int]) -> list[int]:
    indices: list[int] = []
    for idx, (trajectory_id, _) in enumerate(dataset.all_steps):
        if int(trajectory_id) in episode_ids:
            indices.append(idx)
    return indices


def clone_array(value: Any):
    if isinstance(value, torch.Tensor):
        return value.clone()
    return np.array(value, copy=True)


class ActionRegressionAtomicDataset(Dataset):
    """
    Wrap a transformed LeRobot dataset and activate two extra padded action channels:
    - success in {0, 1}
    - progress in [0, 1]

    The extra dimensions are injected after the real action channels and supervised by the
    stock GR00T flow-matching loss via `action_mask`.
    """

    def __init__(
        self,
        base_dataset: LeRobotSingleDataset,
        episode_lengths: Dict[int, int],
        indices: Sequence[int],
        action_offsets: Sequence[int],
        progress_gamma: float,
        success_tail_fraction: float,
        success_tail_min_steps: int,
        success_oversample_factor: int = 1,
    ):
        self.base_dataset = base_dataset
        self.episode_lengths = episode_lengths
        self.progress_gamma = progress_gamma
        self.success_tail_fraction = success_tail_fraction
        self.success_tail_min_steps = success_tail_min_steps
        self.action_offsets = list(action_offsets)
        self.max_action_offset = max(self.action_offsets) if self.action_offsets else 0
        self.indices = self._expand_indices(indices, success_oversample_factor)

    def _expand_indices(self, indices: Sequence[int], success_oversample_factor: int) -> list[int]:
        factor = max(int(success_oversample_factor), 1)
        expanded: list[int] = []
        for base_idx in indices:
            expanded.append(int(base_idx))
            if factor == 1:
                continue
            trajectory_id, step_idx = self.base_dataset.all_steps[int(base_idx)]
            episode_length = self.episode_lengths[int(trajectory_id)]
            if self.window_has_success(int(step_idx), episode_length):
                expanded.extend([int(base_idx)] * (factor - 1))
        return expanded

    def __len__(self) -> int:
        return len(self.indices)

    def compute_progress(self, step_idx: int, episode_length: int) -> float:
        if episode_length <= 1:
            return 1.0
        normalized = step_idx / float(episode_length - 1)
        return float(np.clip(normalized ** self.progress_gamma, 0.0, 1.0))

    def compute_success(self, step_idx: int, episode_length: int) -> float:
        success_tail = max(
            self.success_tail_min_steps,
            int(math.ceil(self.success_tail_fraction * episode_length)),
        )
        success_start = max(0, episode_length - success_tail)
        return 1.0 if step_idx >= success_start else 0.0

    def window_has_success(self, step_idx: int, episode_length: int) -> bool:
        last_window_step = min(step_idx + self.max_action_offset, episode_length - 1)
        return self.compute_success(last_window_step, episode_length) > 0.5

    def __getitem__(self, index: int) -> dict[str, Any]:
        base_idx = self.indices[index]
        trajectory_id, step_idx = self.base_dataset.all_steps[base_idx]
        episode_length = self.episode_lengths[int(trajectory_id)]

        sample = self.base_dataset[base_idx]
        action = clone_array(sample["action"])
        action_mask = clone_array(sample["action_mask"])

        base_action_dim = int(np.asarray(action_mask[0]).astype(np.int64).sum())
        success_action_index = base_action_dim
        progress_action_index = base_action_dim + 1

        if progress_action_index >= action.shape[-1]:
            raise ValueError(
                f"Not enough padded action channels to append success/progress: "
                f"{base_action_dim=}, padded_dim={action.shape[-1]}"
            )

        for t, offset in enumerate(self.action_offsets[: action.shape[0]]):
            absolute_step = min(step_idx + int(offset), episode_length - 1)
            action[t, success_action_index] = self.compute_success(absolute_step, episode_length)
            action[t, progress_action_index] = self.compute_progress(absolute_step, episode_length)

        action_mask[:, success_action_index : progress_action_index + 1] = True

        sample["action"] = action
        sample["action_mask"] = action_mask
        sample["episode_index"] = np.asarray(int(trajectory_id), dtype=np.int64)
        sample["step_index"] = np.asarray(int(step_idx), dtype=np.int64)
        sample["episode_length"] = np.asarray(int(episode_length), dtype=np.int64)
        sample["base_action_dim"] = np.asarray(base_action_dim, dtype=np.int64)
        sample["success_action_index"] = np.asarray(success_action_index, dtype=np.int64)
        sample["progress_action_index"] = np.asarray(progress_action_index, dtype=np.int64)
        return sample


def build_single_dataset(
    record: dict[str, Any],
    data_config_name: str,
    embodiment_tag: str,
    video_backend: str,
    train_split: float,
    seed: int,
    progress_gamma: float,
    success_tail_fraction: float,
    success_tail_min_steps: int,
    success_oversample_factor: int,
) -> tuple[ActionRegressionAtomicDataset, ActionRegressionAtomicDataset]:
    cfg = DATA_CONFIG_MAP[data_config_name]
    train_transform = copy.deepcopy(cfg.transform())
    val_transform = copy.deepcopy(cfg.transform())
    train_transform.train()
    val_transform.eval()

    lerobot_root = record["lerobot_root"]
    episode_lengths = {int(ep["episode_index"]): int(ep["length"]) for ep in record["episodes"]}
    train_ids, val_ids = split_episode_indices(record["episodes"], train_split, seed)

    base_train = LeRobotSingleDataset(
        dataset_path=lerobot_root,
        modality_configs=cfg.modality_config(),
        transforms=train_transform,
        embodiment_tag=EmbodimentTag(embodiment_tag),
        video_backend=video_backend,
    )
    base_val = LeRobotSingleDataset(
        dataset_path=lerobot_root,
        modality_configs=cfg.modality_config(),
        transforms=val_transform,
        embodiment_tag=EmbodimentTag(embodiment_tag),
        video_backend=video_backend,
    )

    train_indices = index_subset_for_episode_ids(base_train, train_ids)
    val_indices = index_subset_for_episode_ids(base_val, val_ids)
    action_offsets = list(cfg.action_indices)

    train_dataset = ActionRegressionAtomicDataset(
        base_dataset=base_train,
        episode_lengths=episode_lengths,
        indices=train_indices,
        action_offsets=action_offsets,
        progress_gamma=progress_gamma,
        success_tail_fraction=success_tail_fraction,
        success_tail_min_steps=success_tail_min_steps,
        success_oversample_factor=success_oversample_factor,
    )
    val_dataset = ActionRegressionAtomicDataset(
        base_dataset=base_val,
        episode_lengths=episode_lengths,
        indices=val_indices,
        action_offsets=action_offsets,
        progress_gamma=progress_gamma,
        success_tail_fraction=success_tail_fraction,
        success_tail_min_steps=success_tail_min_steps,
        success_oversample_factor=1,
    )
    return train_dataset, val_dataset


def build_datasets(args: Args) -> tuple[Dataset, Dataset]:
    manifest = load_manifest(Path(args.manifest_path))
    train_parts: list[Dataset] = []
    val_parts: list[Dataset] = []

    for record in manifest["datasets"]:
        train_ds, val_ds = build_single_dataset(
            record=record,
            data_config_name=args.data_config,
            embodiment_tag=args.embodiment_tag,
            video_backend=args.video_backend,
            train_split=args.train_split,
            seed=args.seed,
            progress_gamma=args.progress_gamma,
            success_tail_fraction=args.success_tail_fraction,
            success_tail_min_steps=args.success_tail_min_steps,
            success_oversample_factor=args.success_oversample_factor,
        )
        train_parts.append(train_ds)
        val_parts.append(val_ds)

    return ConcatDataset(train_parts), ConcatDataset(val_parts)


def maybe_recreate_action_head_for_horizon(model: GR00T_N1_5, action_horizon: int, args: Args) -> GR00T_N1_5:
    if action_horizon == model.action_head.config.action_horizon:
        return model

    print(
        f"Recreating action head with action_horizon {action_horizon} "
        f"(was {model.action_head.config.action_horizon})"
    )
    new_action_head_config = model.action_head.config
    new_action_head_config.action_horizon = action_horizon

    from gr00t.model.action_head.flow_matching_action_head import FlowmatchingActionHead

    new_action_head = FlowmatchingActionHead(new_action_head_config)
    new_action_head.load_state_dict(model.action_head.state_dict(), strict=False)
    model.action_head = new_action_head
    model.config.action_horizon = action_horizon
    model.action_horizon = action_horizon
    model.config.action_head_cfg["action_horizon"] = action_horizon
    model.action_head.set_trainable_parameters(
        tune_projector=args.tune_projector,
        tune_diffusion_model=args.tune_diffusion_model,
    )
    return model


def write_experiment_metadata(
    output_dir: Path,
    train_dataset: Dataset,
    args: Args,
) -> Path:
    exp_cfg_dir = output_dir / "experiment_cfg"
    exp_cfg_dir.mkdir(parents=True, exist_ok=True)

    metadata_payload: dict[str, Any] = {
        "manifest_path": args.manifest_path,
        "checkpoint_path": args.checkpoint_path,
        "data_config": args.data_config,
        "embodiment_tag": args.embodiment_tag,
        "progress_gamma": args.progress_gamma,
        "success_tail_fraction": args.success_tail_fraction,
        "success_tail_min_steps": args.success_tail_min_steps,
        "success_oversample_factor": args.success_oversample_factor,
    }

    first_sample = train_dataset[0]
    base_action_dim = int(first_sample["base_action_dim"])
    metadata_payload["action_regression"] = {
        "base_action_dim": base_action_dim,
        "success_action_index": int(first_sample["success_action_index"]),
        "progress_action_index": int(first_sample["progress_action_index"]),
        "padded_action_dim": int(first_sample["action"].shape[-1]),
        "action_horizon": int(first_sample["action"].shape[0]),
    }

    with (exp_cfg_dir / "action_regression_config.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata_payload, handle, indent=2)

    source_metadata_path = Path(args.checkpoint_path) / "experiment_cfg" / "metadata.json"
    if source_metadata_path.exists():
        shutil.copy2(source_metadata_path, exp_cfg_dir / "metadata.json")
    else:
        print(f"Warning: metadata.json not found at {source_metadata_path}, skipping copy.")

    return exp_cfg_dir


def main(args: Args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for GR00T action regression training.")

    set_seed(args.seed)
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except RuntimeError:
        pass

    train_dataset, val_dataset = build_datasets(args)
    del val_dataset  # offline evaluation is handled by the dedicated eval script

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exp_cfg_dir = write_experiment_metadata(output_dir, train_dataset, args)

    if args.report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        if args.wandb_entity:
            os.environ.setdefault("WANDB_ENTITY", args.wandb_entity)
        os.environ.setdefault("WANDB_MODE", args.wandb_mode)

    data_cfg = DATA_CONFIG_MAP[args.data_config]
    model = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path=args.checkpoint_path,
        tune_llm=args.tune_llm,
        tune_visual=args.tune_visual,
        tune_projector=args.tune_projector,
        tune_diffusion_model=args.tune_diffusion_model,
    )
    model = maybe_recreate_action_head_for_horizon(model, len(data_cfg.action_indices), args)
    model.compute_dtype = "bfloat16" if args.bf16 else "float32"
    model.config.compute_dtype = model.compute_dtype

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=output_dir.name,
        remove_unused_columns=False,
        bf16=args.bf16,
        tf32=True,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=args.dataloader_num_workers > 0,
        optim="adamw_torch",
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10,
        num_train_epochs=300,
        max_steps=args.max_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=20,
        report_to=[args.report_to] if args.report_to != "none" else [],
        seed=args.seed,
        do_eval=False,
        ddp_find_unused_parameters=False,
    )

    trainer = DualBrainTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DefaultDataCollator(),
        compute_dtype=torch.bfloat16 if args.bf16 else torch.float32,
    )
    trainer.add_callback(CheckpointFormatCallback(run_name=output_dir.name, exp_cfg_dir=exp_cfg_dir))

    print(
        f"Train dataset length: {len(train_dataset)}\n"
        f"GPU memory before training: {torch.cuda.memory_allocated() / 1024 / 1024 / 1024:.2f} GB",
        flush=True,
    )

    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=str(output_dir / "final"))


if __name__ == "__main__":
    main(tyro.cli(Args))
