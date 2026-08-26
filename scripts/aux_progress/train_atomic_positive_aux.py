#!/usr/bin/env python3
"""
Auxiliary state supervision for GR00T on Robocasa atomic success and synthetic retry samples.

This script:
1. Loads a frozen GR00T checkpoint.
2. Wraps Robocasa atomic LeRobot datasets with dense progress/success labels.
3. Optionally mixes in synthetic retry samples from a retry manifest.
4. Trains two lightweight heads on top of frozen backbone features:
   - progress_head: regress signed progress
   - state_head: classify {progress, success, retry}

The original action head is not updated.
"""

from __future__ import annotations

import copy
import bisect
import json
import math
import os
import random
import shutil
import subprocess
import sys
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
import tyro
from torch import nn
from torch.utils.data import ConcatDataset, Dataset
from transformers import BatchFeature, Trainer, TrainingArguments, set_seed
from transformers.trainer_utils import SaveStrategy

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.model.transforms import DefaultDataCollator


STATE_PROGRESS = 0
STATE_SUCCESS = 1
STATE_RETRY = 2
STATE_CLASS_NAMES = ["progress", "success", "retry"]
HF_WEIGHTS_NAME = "pytorch_model.bin"
OBSERVATION_MODALITIES = {"video", "state", "language"}
AUX_CONTEXT_NONE = "none"
AUX_CONTEXT_STATE_DELTA = "state_delta"
AUX_CONTEXT_STATE_ACTION_DELTA = "state_action_delta"
AUX_CONTEXT_MODES = {
    AUX_CONTEXT_NONE,
    AUX_CONTEXT_STATE_DELTA,
    AUX_CONTEXT_STATE_ACTION_DELTA,
}
CHECKPOINT_RETENTION_RECENT = "recent"
CHECKPOINT_RETENTION_BEST_EVAL = "best_eval"
CHECKPOINT_RETENTION_STRATEGIES = {
    CHECKPOINT_RETENTION_RECENT,
    CHECKPOINT_RETENTION_BEST_EVAL,
}
BEST_CHECKPOINT_MODES = {"max", "min"}


@dataclass
class Args:
    manifest_path: str = "/data/zjw/workspace/Isaac-GR00T/expdata/aux_progress/atomic_positive_manifest.json"
    retry_manifest_path: str = "/data/zjw/workspace/Isaac-GR00T/expdata/aux_progress/atomic_retry_manifest.json"
    composite_manifest_path: str = ""
    checkpoint_path: str = "/data/zjw/workspace/Isaac-GR00T/expdata/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000"
    output_dir: str = "/data/zjw/workspace/Isaac-GR00T/expdata/aux_progress/atomic_positive_only"
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
    progress_sample_weight: float = 1.0
    success_sample_weight: float = 0.8
    retry_sample_weight: float = 0.5
    composite_sample_weight: float = 1.0
    train_epoch_size: int = 200000
    retry_progress_default: float = 0.0
    retry_manifest_max_samples: int = -1
    retry_types: str = ""
    progress_class_loss_weight: float = 1.0
    success_class_loss_weight: float = 1.0
    retry_class_loss_weight: float = 0.7
    state_label_smoothing: float = 0.02
    observation_history_offsets: str = ""
    synthetic_retry_history: bool = True
    aux_context_mode: str = AUX_CONTEXT_NONE
    checkpoint_retention_strategy: str = CHECKPOINT_RETENTION_RECENT
    best_checkpoint_metric: str = "state_per_class_accuracy_mean"
    best_checkpoint_mode: str = "max"
    best_checkpoint_keep_unevaluated: int = 2
    external_eval_sync: bool = False
    train_split: float = 0.9
    seed: int = 42
    report_to: str = "tensorboard"
    wandb_project: str = "robocasa-aux-progress"
    wandb_entity: str = ""
    wandb_mode: str = "online"
    wandb_log_model: str = "checkpoint"
    inline_eval: bool = False
    external_eval_gpu: int = 1
    external_eval_batch_size: int = 2
    external_eval_num_workers: int = 0
    external_eval_split: str = "val"
    external_eval_max_batches: int = 4096
    external_eval_subset_mode: str = "random"
    external_eval_subset_seed: int = 42


class PositiveAtomicAuxDataset(Dataset):
    """
    A lightweight wrapper that keeps the original LeRobot sample intact and adds online labels.

    Labels are constructed from episode length only:
    - progress_target in [0, 1] using an ease-in curve
    - state_target in {progress, success} with a small terminal success band
    """

    def __init__(
        self,
        base_dataset: LeRobotSingleDataset,
        episode_lengths: Dict[int, int],
        indices: Sequence[int],
        progress_gamma: float,
        success_tail_fraction: float,
        success_tail_min_steps: int,
    ):
        self.base_dataset = base_dataset
        self.episode_lengths = episode_lengths
        self.indices = list(indices)
        self.progress_gamma = progress_gamma
        self.success_tail_fraction = success_tail_fraction
        self.success_tail_min_steps = success_tail_min_steps

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        base_idx = self.indices[index]
        trajectory_id, step_idx = self.base_dataset.all_steps[base_idx]
        episode_length = self.episode_lengths[int(trajectory_id)]
        sample = self.base_dataset[base_idx]

        progress_target = self.compute_progress(step_idx, episode_length)
        state_target = self.compute_state(step_idx, episode_length)

        sample["progress_target"] = np.asarray(progress_target, dtype=np.float32)
        sample["state_target"] = np.asarray(state_target, dtype=np.int64)
        sample["episode_index"] = np.asarray(int(trajectory_id), dtype=np.int64)
        sample["step_index"] = np.asarray(int(step_idx), dtype=np.int64)
        sample["episode_length"] = np.asarray(int(episode_length), dtype=np.int64)
        sample["retry_type_id"] = np.asarray(-1, dtype=np.int64)
        return sample

    def compute_progress(self, step_idx: int, episode_length: int) -> float:
        if episode_length <= 1:
            return 1.0
        normalized = step_idx / float(episode_length - 1)
        return float(np.clip(normalized ** self.progress_gamma, 0.0, 1.0))

    def compute_state(self, step_idx: int, episode_length: int) -> int:
        success_tail = max(self.success_tail_min_steps, int(math.ceil(self.success_tail_fraction * episode_length)))
        success_start = max(0, episode_length - success_tail)
        return STATE_SUCCESS if step_idx >= success_start else STATE_PROGRESS


class CompositeSubtaskAuxDataset(Dataset):
    """
    Positive auxiliary supervision over automatically segmented composite episodes.

    Each item belongs to exactly one plan subtask segment.  The original full-task
    language annotation is replaced with that subtask's instruction before applying
    the normal GR00T transforms.
    """

    def __init__(
        self,
        base_dataset: LeRobotSingleDataset,
        index_records: Sequence[tuple[int, dict[str, Any]]],
        progress_gamma: float,
        success_tail_fraction: float,
        success_tail_min_steps: int,
    ):
        self.base_dataset = base_dataset
        self.index_records = [(int(base_index), segment) for base_index, segment in index_records]
        self.progress_gamma = progress_gamma
        self.success_tail_fraction = success_tail_fraction
        self.success_tail_min_steps = success_tail_min_steps

    def __len__(self) -> int:
        return len(self.index_records)

    def compute_progress(self, local_step: int, segment_length: int) -> float:
        if segment_length <= 1:
            return 1.0
        normalized = local_step / float(segment_length - 1)
        return float(np.clip(normalized ** self.progress_gamma, 0.0, 1.0))

    def compute_state(self, local_step: int, segment_length: int) -> int:
        success_tail = max(
            self.success_tail_min_steps,
            int(math.ceil(self.success_tail_fraction * segment_length)),
        )
        success_start = max(0, segment_length - success_tail)
        return STATE_SUCCESS if local_step >= success_start else STATE_PROGRESS

    @staticmethod
    def replace_language(raw: dict[str, Any], instruction: str) -> None:
        for key, value in list(raw.items()):
            if not key.startswith("annotation."):
                continue
            if isinstance(value, list):
                raw[key] = [instruction for _ in value]
            else:
                raw[key] = [instruction]

    def __getitem__(self, index: int) -> dict[str, Any]:
        base_index, segment = self.index_records[index]
        trajectory_id, step_idx = self.base_dataset.all_steps[base_index]
        start_frame = int(segment["start_frame"])
        segment_length = int(segment["length"])
        local_step = int(step_idx) - start_frame
        raw = self.base_dataset.get_step_data(int(trajectory_id), int(step_idx))
        self.replace_language(raw, str(segment["instruction"]))
        sample = self.base_dataset.transforms(raw)

        sample["progress_target"] = np.asarray(
            self.compute_progress(local_step, segment_length),
            dtype=np.float32,
        )
        sample["state_target"] = np.asarray(
            self.compute_state(local_step, segment_length),
            dtype=np.int64,
        )
        sample["episode_index"] = np.asarray(int(trajectory_id), dtype=np.int64)
        sample["step_index"] = np.asarray(local_step, dtype=np.int64)
        sample["episode_length"] = np.asarray(segment_length, dtype=np.int64)
        sample["retry_type_id"] = np.asarray(-1, dtype=np.int64)
        return sample


def filter_positive_indices_by_state(
    base_dataset: LeRobotSingleDataset,
    episode_lengths: Dict[int, int],
    indices: Sequence[int],
    target_state: int,
    success_tail_fraction: float,
    success_tail_min_steps: int,
) -> list[int]:
    filtered: list[int] = []
    for base_idx in indices:
        trajectory_id, step_idx = base_dataset.all_steps[int(base_idx)]
        episode_length = episode_lengths[int(trajectory_id)]
        success_tail = max(
            success_tail_min_steps,
            int(math.ceil(success_tail_fraction * episode_length)),
        )
        success_start = max(0, episode_length - success_tail)
        state = STATE_SUCCESS if int(step_idx) >= success_start else STATE_PROGRESS
        if state == target_state:
            filtered.append(int(base_idx))
    return filtered


def set_retry_type_progress_target(
    retry_type: str,
    signed_progress_target: float | None,
    retry_progress_default: float,
) -> float:
    if signed_progress_target is not None:
        return float(signed_progress_target)
    if retry_type == "repeat":
        return 0.0
    return float(retry_progress_default)


def parse_observation_history_offsets(offsets: str) -> list[int]:
    if not offsets.strip():
        return []
    parsed = [int(item.strip()) for item in offsets.split(",") if item.strip()]
    if not parsed:
        return []
    if parsed[-1] != 0:
        raise ValueError(
            "--observation-history-offsets should be ordered from history to current "
            "and end with 0, for example: -8,-4,0"
        )
    return parsed


def validate_aux_context_mode(aux_context_mode: str) -> str:
    if aux_context_mode not in AUX_CONTEXT_MODES:
        raise ValueError(
            f"Unsupported aux_context_mode={aux_context_mode!r}; "
            f"expected one of {sorted(AUX_CONTEXT_MODES)}"
        )
    return aux_context_mode


def validate_checkpoint_retention_strategy(strategy: str) -> str:
    if strategy not in CHECKPOINT_RETENTION_STRATEGIES:
        raise ValueError(
            f"Unsupported checkpoint_retention_strategy={strategy!r}; "
            f"expected one of {sorted(CHECKPOINT_RETENTION_STRATEGIES)}"
        )
    return strategy


def validate_best_checkpoint_mode(mode: str) -> str:
    if mode not in BEST_CHECKPOINT_MODES:
        raise ValueError(
            f"Unsupported best_checkpoint_mode={mode!r}; expected one of {sorted(BEST_CHECKPOINT_MODES)}"
        )
    return mode


def build_data_config(data_config_name: str, observation_history_offsets: Sequence[int]):
    cfg = copy.deepcopy(DATA_CONFIG_MAP[data_config_name])
    if observation_history_offsets:
        if not hasattr(cfg, "observation_indices"):
            raise ValueError(
                f"Data config {data_config_name!r} does not expose observation_indices, "
                "so observation history cannot be injected."
            )
        cfg.observation_indices = list(observation_history_offsets)
    return cfg


def get_step_data_with_observation_offsets(
    dataset: LeRobotSingleDataset,
    trajectory_id: int,
    base_index: int,
    observation_offsets: Sequence[int],
) -> dict[str, Any]:
    if not observation_offsets:
        return dataset.get_step_data(trajectory_id, base_index)

    old_offsets: dict[str, np.ndarray] = {}
    try:
        for modality in dataset.modality_keys:
            for key in dataset.modality_keys[modality]:
                if modality in OBSERVATION_MODALITIES:
                    old_offsets[key] = dataset._delta_indices[key].copy()
                    dataset._delta_indices[key] = np.asarray(observation_offsets, dtype=np.int64)
        return dataset.get_step_data(trajectory_id, base_index)
    finally:
        for key, value in old_offsets.items():
            dataset._delta_indices[key] = value


def retry_observation_offsets(
    retry_type: str,
    observation_history_offsets: Sequence[int],
    synthetic_retry_history: bool,
) -> list[int]:
    offsets = list(observation_history_offsets)
    if not offsets or not synthetic_retry_history:
        return offsets
    if retry_type == "repeat":
        return [0 for _ in offsets]
    if retry_type in {"reverse", "backtrack"}:
        return [-offset for offset in offsets]
    return offsets


def get_eval_checkpoint_metric(eval_payload: dict[str, Any], metric_name: str) -> float | None:
    value = eval_payload.get(metric_name)
    if isinstance(value, (int, float)):
        return float(value)

    if metric_name == "state_per_class_accuracy_mean":
        per_class = eval_payload.get("state_per_class_accuracy")
        if isinstance(per_class, dict) and per_class:
            values = [float(v) for v in per_class.values()]
            return float(sum(values) / len(values))
    elif metric_name == "state_per_class_accuracy_min":
        per_class = eval_payload.get("state_per_class_accuracy")
        if isinstance(per_class, dict) and per_class:
            values = [float(v) for v in per_class.values()]
            return float(min(values))

    if "." in metric_name:
        current: Any = eval_payload
        for part in metric_name.split("."):
            if not isinstance(current, dict) or part not in current:
                return None
            current = current[part]
        if isinstance(current, (int, float)):
            return float(current)
    return None


def ensure_hf_resume_checkpoint(resume_from_checkpoint: str) -> str:
    checkpoint_dir = Path(resume_from_checkpoint)
    if not checkpoint_dir.is_dir():
        return resume_from_checkpoint

    hf_weights_path = checkpoint_dir / HF_WEIGHTS_NAME
    aux_weights_path = checkpoint_dir / "aux_heads.pt"
    if hf_weights_path.is_file() or not aux_weights_path.is_file():
        return resume_from_checkpoint

    shutil.copyfile(aux_weights_path, hf_weights_path)
    print(
        f"Created {hf_weights_path} from aux_heads.pt so HuggingFace Trainer can resume.",
        flush=True,
    )
    return resume_from_checkpoint


def allow_legacy_numpy_rng_state_load() -> None:
    try:
        from numpy.core.multiarray import _reconstruct

        torch.serialization.add_safe_globals(
            [
                _reconstruct,
                np.ndarray,
                np.dtype,
                type(np.dtype(np.uint32)),
            ]
        )
    except Exception as exc:
        warnings.warn(
            f"Failed to allowlist numpy RNG state globals for checkpoint resume: {exc}",
            RuntimeWarning,
        )


RETRY_TYPE_TO_ID = {
    "reverse": 0,
    "repeat": 1,
    "mismatch": 2,
    "backtrack": 3,
}


class RetryAtomicAuxDataset(Dataset):
    """
    Materializes retry examples from a lightweight retry manifest.

    The manifest stores source episode/step references only. This dataset keeps data loading
    lazy and applies the existing GR00T transforms after optional language replacement.
    """

    def __init__(
        self,
        base_datasets_by_index: dict[int, LeRobotSingleDataset],
        retry_examples: Sequence[dict[str, Any]],
        retry_progress_default: float,
        observation_history_offsets: Sequence[int],
        synthetic_retry_history: bool,
    ):
        self.base_datasets_by_index = base_datasets_by_index
        self.retry_examples = list(retry_examples)
        self.retry_progress_default = retry_progress_default
        self.observation_history_offsets = list(observation_history_offsets)
        self.synthetic_retry_history = bool(synthetic_retry_history)
        self._language_cache: dict[tuple[int, int, int], list[str]] = {}

    def __len__(self) -> int:
        return len(self.retry_examples)

    def _get_language(self, dataset_index: int, episode_index: int, step_index: int) -> list[str]:
        cache_key = (int(dataset_index), int(episode_index), int(step_index))
        if cache_key not in self._language_cache:
            dataset = self.base_datasets_by_index[int(dataset_index)]
            raw = dataset.get_step_data(int(episode_index), int(step_index))
            language_values: list[str] = []
            for key, value in raw.items():
                if key.startswith("annotation."):
                    if isinstance(value, list):
                        language_values = [str(item) for item in value]
                    else:
                        language_values = [str(value)]
                    break
            self._language_cache[cache_key] = language_values
        return self._language_cache[cache_key]

    def _replace_language(self, raw: dict[str, Any], replacement: list[str]) -> None:
        if not replacement:
            return
        for key in list(raw.keys()):
            if key.startswith("annotation."):
                raw[key] = list(replacement)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.retry_examples[index]
        dataset = self.base_datasets_by_index[int(example["source_dataset_index"])]
        episode_index = int(example["source_episode_index"])
        step_index = int(example["source_step_index"])
        episode_length = int(example["source_episode_length"])

        retry_type = str(example["retry_type"])
        raw = get_step_data_with_observation_offsets(
            dataset,
            episode_index,
            step_index,
            retry_observation_offsets(
                retry_type,
                self.observation_history_offsets,
                self.synthetic_retry_history,
            ),
        )
        if retry_type == "mismatch":
            language_source_dataset_index = example.get("language_source_dataset_index")
            language_source_episode_index = example.get("language_source_episode_index")
            language_source_step_index = example.get("language_source_step_index")
            if language_source_dataset_index is not None:
                replacement = self._get_language(
                    int(language_source_dataset_index),
                    int(language_source_episode_index),
                    int(language_source_step_index),
                )
                self._replace_language(raw, replacement)

        sample = dataset.transforms(raw)
        sample["progress_target"] = np.asarray(
            set_retry_type_progress_target(
                retry_type=retry_type,
                signed_progress_target=example.get("signed_progress_target"),
                retry_progress_default=self.retry_progress_default,
            ),
            dtype=np.float32,
        )
        sample["state_target"] = np.asarray(STATE_RETRY, dtype=np.int64)
        sample["episode_index"] = np.asarray(episode_index, dtype=np.int64)
        sample["step_index"] = np.asarray(step_index, dtype=np.int64)
        sample["episode_length"] = np.asarray(episode_length, dtype=np.int64)
        sample["retry_type_id"] = np.asarray(RETRY_TYPE_TO_ID.get(retry_type, -1), dtype=np.int64)
        return sample


class WeightedAuxDataset(Dataset):
    """
    Deterministic ratio-controlled sampler over progress/success/retry datasets.

    This keeps the effective epoch size bounded and avoids materializing a huge mixed
    dataset. Each index deterministically chooses a bucket and then a sample within it.
    """

    def __init__(
        self,
        datasets: Sequence[Dataset],
        weights: Sequence[float],
        epoch_size: int,
        seed: int,
    ):
        if len(datasets) != len(weights):
            raise ValueError(f"{len(datasets)=} must match {len(weights)=}")
        filtered = [
            (dataset, float(weight))
            for dataset, weight in zip(datasets, weights)
            if len(dataset) > 0 and float(weight) > 0.0
        ]
        if not filtered:
            raise ValueError("WeightedAuxDataset requires at least one non-empty dataset with positive weight.")
        self.datasets = [dataset for dataset, _ in filtered]
        raw_weights = np.asarray([weight for _, weight in filtered], dtype=np.float64)
        self.probs = raw_weights / raw_weights.sum()
        self.cumulative = np.cumsum(self.probs)
        self.epoch_size = max(int(epoch_size), 1)
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = random.Random(self.seed + int(index))
        bucket = int(np.searchsorted(self.cumulative, rng.random(), side="right"))
        bucket = min(bucket, len(self.datasets) - 1)
        dataset = self.datasets[bucket]
        sample_index = rng.randrange(len(dataset))
        return dataset[sample_index]


class AtomicAuxCollator(DefaultDataCollator):
    """
    Extends the stock GR00T collator with scalar auxiliary targets.
    """

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        progress_targets = np.asarray([float(elem.pop("progress_target")) for elem in features], dtype=np.float32)
        state_targets = np.asarray([int(elem.pop("state_target")) for elem in features], dtype=np.int64)
        episode_index = np.asarray([int(elem.pop("episode_index")) for elem in features], dtype=np.int64)
        step_index = np.asarray([int(elem.pop("step_index")) for elem in features], dtype=np.int64)
        episode_length = np.asarray([int(elem.pop("episode_length")) for elem in features], dtype=np.int64)
        retry_type_id = np.asarray([int(elem.pop("retry_type_id", -1)) for elem in features], dtype=np.int64)

        batch = super().__call__(features)
        batch["progress_target"] = torch.from_numpy(progress_targets)
        batch["state_target"] = torch.from_numpy(state_targets)
        batch["episode_index"] = torch.from_numpy(episode_index)
        batch["step_index"] = torch.from_numpy(step_index)
        batch["episode_length"] = torch.from_numpy(episode_length)
        batch["retry_type_id"] = torch.from_numpy(retry_type_id)
        return batch


class GR00TAuxiliaryModel(nn.Module):
    """
    Frozen GR00T backbone/action model plus lightweight auxiliary heads.

    The base model is only used to generate multimodal features. Auxiliary heads consume the
    masked mean of backbone token embeddings.
    """

    _keys_to_ignore_on_save = None

    def __init__(
        self,
        checkpoint_path: str,
        state_loss_weights: Sequence[float] | None = None,
        state_label_smoothing: float = 0.0,
        aux_context_mode: str = AUX_CONTEXT_NONE,
        observation_history_offsets: Sequence[int] | None = None,
        synthetic_retry_history: bool = True,
    ):
        super().__init__()
        self.aux_context_mode = validate_aux_context_mode(aux_context_mode)
        self.observation_history_offsets = list(observation_history_offsets or [])
        self.synthetic_retry_history = bool(synthetic_retry_history)
        if not torch.cuda.is_available():
            raise RuntimeError(
                "GR00T auxiliary training/evaluation requires a CUDA-visible GPU. "
                "The Eagle backbone initializes flash attention during model load."
            )
        self.base_model = GR00T_N1_5.from_pretrained(
            pretrained_model_name_or_path=checkpoint_path,
            tune_llm=False,
            tune_visual=False,
            tune_projector=False,
            tune_diffusion_model=False,
        )
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False

        hidden_size = int(self.base_model.config.hidden_size)
        head_input_dim = hidden_size + self._aux_context_dim()
        self.progress_head = nn.Sequential(
            nn.LayerNorm(head_input_dim),
            nn.Linear(head_input_dim, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
        )
        self.state_head = nn.Sequential(
            nn.LayerNorm(head_input_dim),
            nn.Linear(head_input_dim, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, len(STATE_CLASS_NAMES)),
        )

        self.progress_loss_fn = nn.SmoothL1Loss()
        if state_loss_weights is None:
            state_loss_weights = [1.0] * len(STATE_CLASS_NAMES)
        if len(state_loss_weights) != len(STATE_CLASS_NAMES):
            raise ValueError(f"{state_loss_weights=} must have {len(STATE_CLASS_NAMES)} values.")
        self.state_loss_fn = nn.CrossEntropyLoss(
            weight=torch.tensor(list(state_loss_weights), dtype=torch.float32),
            label_smoothing=float(state_label_smoothing),
        )

    def _aux_context_dim(self) -> int:
        if self.aux_context_mode == AUX_CONTEXT_NONE:
            return 0
        state_dim = int(getattr(self.base_model.action_head.config, "max_state_dim", 64))
        if self.aux_context_mode == AUX_CONTEXT_STATE_DELTA:
            return state_dim * 2
        action_dim = int(self.base_model.config.action_dim)
        return state_dim * 2 + action_dim * 2

    def _masked_mean(self, features: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.to(dtype=features.dtype).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (features * mask).sum(dim=1) / denom

    def _head_dtype(self) -> torch.dtype:
        # DataParallel replicas do not expose parameters through parameters().
        return self.progress_head[0].weight.dtype

    def _build_aux_context(self, inputs: Dict[str, Any]) -> torch.Tensor | None:
        if self.aux_context_mode == AUX_CONTEXT_NONE:
            return None

        state = inputs["state"].to(self.base_model.device)
        state_mask = inputs.get("state_mask")
        if state_mask is not None:
            state = state * state_mask.to(self.base_model.device, dtype=state.dtype)
        state_first = state[:, 0]
        state_last = state[:, -1]
        parts = [state_last, state_last - state_first]

        if self.aux_context_mode == AUX_CONTEXT_STATE_ACTION_DELTA:
            action = inputs["action"].to(self.base_model.device)
            action_mask = inputs.get("action_mask")
            if action_mask is not None:
                action = action * action_mask.to(self.base_model.device, dtype=action.dtype)
            action_first = action[:, 0]
            action_last = action[:, -1]
            parts.extend([action_first, action_last - action_first])

        return torch.cat(parts, dim=-1)

    def forward(self, inputs: Dict[str, Any] | None = None, **kwargs) -> BatchFeature:
        # Trainer uses `model(inputs)` in our custom training path, but switches to
        # `model(**inputs)` during eval / prediction. Support both call styles here.
        if inputs is None:
            inputs = kwargs
        elif kwargs:
            merged_inputs = dict(inputs)
            merged_inputs.update(kwargs)
            inputs = merged_inputs

        progress_target = inputs["progress_target"].to(self.base_model.device)
        state_target = inputs["state_target"].to(self.base_model.device)

        with torch.no_grad():
            backbone_inputs, _ = self.base_model.prepare_input(inputs)
            tensor_inputs = [
                value for value in backbone_inputs.values() if torch.is_tensor(value)
            ]
            if tensor_inputs and tensor_inputs[0].device.type != "cuda":
                raise RuntimeError(
                    f"Backbone inputs are on {tensor_inputs[0].device}, not CUDA. "
                    "This usually means the script is running without a visible GPU or "
                    "an older eval script is still being executed."
                )
            backbone_outputs = self.base_model.backbone(backbone_inputs)
            pooled = self._masked_mean(
                backbone_outputs["backbone_features"],
                backbone_outputs["backbone_attention_mask"],
            )

        pooled = pooled.to(dtype=self._head_dtype())
        aux_context = self._build_aux_context(inputs)
        if aux_context is not None:
            aux_context = aux_context.to(device=pooled.device, dtype=pooled.dtype)
            pooled = torch.cat([pooled, aux_context], dim=-1)
        progress_logits = self.progress_head(pooled).squeeze(-1)
        state_logits = self.state_head(pooled)

        progress_loss = self.progress_loss_fn(progress_logits, progress_target)
        state_loss = self.state_loss_fn(state_logits, state_target)
        loss = progress_loss + state_loss

        preds = torch.argmax(state_logits, dim=-1)
        state_acc = (preds == state_target).to(torch.float32).mean()

        return BatchFeature(
            data={
                "loss": loss,
                "progress_loss": progress_loss.detach(),
                "state_loss": state_loss.detach(),
                "state_accuracy": state_acc.detach(),
                "progress_pred": progress_logits.detach(),
                "state_logits": state_logits.detach(),
            }
        )

    def save_pretrained(self, save_directory: str, state_dict: Dict[str, torch.Tensor] | None = None):
        """
        Save only the lightweight auxiliary parameters plus the reference base checkpoint path.
        """
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        source_state = state_dict if state_dict is not None else self.state_dict()
        state_dict = {
            key: value.detach().cpu()
            for key, value in source_state.items()
            if key.startswith("progress_head.")
            or key.startswith("state_head.")
            or key == "state_loss_fn.weight"
        }

        torch.save(state_dict, save_path / "aux_heads.pt")
        torch.save(state_dict, save_path / HF_WEIGHTS_NAME)
        with (save_path / "aux_config.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "base_checkpoint_path": str(self.base_model.local_model_path),
                    "hidden_size": int(self.base_model.config.hidden_size),
                    "head_input_dim": int(self.progress_head[0].normalized_shape[0]),
                    "state_classes": STATE_CLASS_NAMES,
                    "state_label_smoothing": float(self.state_loss_fn.label_smoothing),
                    "aux_context_mode": self.aux_context_mode,
                    "observation_history_offsets": self.observation_history_offsets,
                    "synthetic_retry_history": self.synthetic_retry_history,
                },
                handle,
                indent=2,
            )


class AuxTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.external_eval_gpu = kwargs.pop("external_eval_gpu", None)
        self.external_eval_batch_size = kwargs.pop("external_eval_batch_size", 2)
        self.external_eval_num_workers = kwargs.pop("external_eval_num_workers", 0)
        self.external_eval_split = kwargs.pop("external_eval_split", "val")
        self.external_eval_max_batches = kwargs.pop("external_eval_max_batches", 4096)
        self.external_eval_subset_mode = kwargs.pop("external_eval_subset_mode", "random")
        self.external_eval_subset_seed = kwargs.pop("external_eval_subset_seed", 42)
        self.manifest_path = kwargs.pop("manifest_path", None)
        self.retry_manifest_path = kwargs.pop("retry_manifest_path", None)
        self.composite_manifest_path = kwargs.pop("composite_manifest_path", "")
        self.base_checkpoint_path = kwargs.pop("base_checkpoint_path", None)
        self.data_config_name = kwargs.pop("data_config_name", "panda_omron")
        self.embodiment_tag = kwargs.pop("embodiment_tag", "new_embodiment")
        self.video_backend = kwargs.pop("video_backend", "opencv")
        self.progress_gamma = kwargs.pop("progress_gamma", 1.5)
        self.success_tail_fraction = kwargs.pop("success_tail_fraction", 0.1)
        self.success_tail_min_steps = kwargs.pop("success_tail_min_steps", 3)
        self.retry_progress_default = kwargs.pop("retry_progress_default", 0.0)
        self.retry_manifest_max_samples = kwargs.pop("retry_manifest_max_samples", -1)
        self.state_label_smoothing = kwargs.pop("state_label_smoothing", 0.0)
        self.observation_history_offsets = kwargs.pop("observation_history_offsets", "")
        self.synthetic_retry_history = kwargs.pop("synthetic_retry_history", True)
        self.aux_context_mode = kwargs.pop("aux_context_mode", AUX_CONTEXT_NONE)
        self.checkpoint_retention_strategy = kwargs.pop(
            "checkpoint_retention_strategy", CHECKPOINT_RETENTION_RECENT
        )
        self.best_checkpoint_metric = kwargs.pop("best_checkpoint_metric", "state_per_class_accuracy_mean")
        self.best_checkpoint_mode = kwargs.pop("best_checkpoint_mode", "max")
        self.best_checkpoint_keep_n = kwargs.pop("best_checkpoint_keep_n", 10)
        self.best_checkpoint_keep_unevaluated = kwargs.pop("best_checkpoint_keep_unevaluated", 2)
        self.external_eval_sync = kwargs.pop("external_eval_sync", False)
        self.train_split = kwargs.pop("train_split", 0.9)
        self.seed = kwargs.pop("seed", 42)
        self._launched_eval_checkpoints: set[str] = set()
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(inputs)
        loss = outputs["loss"]
        self._last_aux_metrics = {
            key: float(outputs[key].mean().item())
            for key in ("progress_loss", "state_loss", "state_accuracy")
            if key in outputs
        }
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float], start_time: float | None = None) -> None:
        if hasattr(self, "_last_aux_metrics"):
            logs.update(getattr(self, "_last_aux_metrics"))
        super().log(logs, start_time)

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False):
        save_dir = output_dir or self.args.output_dir
        if self.args.should_save:
            self.model.save_pretrained(save_dir)

    def _build_external_eval_command(self, checkpoint_dir: Path) -> list[str]:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent / "eval_atomic_positive_aux.py"),
            "--aux-dir",
            str(checkpoint_dir),
            "--manifest-path",
            str(self.manifest_path),
            "--retry-manifest-path",
            str(self.retry_manifest_path),
            "--checkpoint-path",
            str(self.base_checkpoint_path),
            "--data-config",
            str(self.data_config_name),
            "--embodiment-tag",
            str(self.embodiment_tag),
            "--video-backend",
            str(self.video_backend),
            "--batch-size",
            str(self.external_eval_batch_size),
            "--dataloader-num-workers",
            str(self.external_eval_num_workers),
            "--progress-gamma",
            str(self.progress_gamma),
            "--success-tail-fraction",
            str(self.success_tail_fraction),
            "--success-tail-min-steps",
            str(self.success_tail_min_steps),
            "--retry-progress-default",
            str(self.retry_progress_default),
            "--retry-manifest-max-samples",
            str(self.retry_manifest_max_samples),
            "--state-label-smoothing",
            str(self.state_label_smoothing),
            f"--observation-history-offsets={self.observation_history_offsets}",
            "--aux-context-mode",
            str(self.aux_context_mode),
            "--train-split",
            str(self.train_split),
            "--seed",
            str(self.seed),
            "--split",
            str(self.external_eval_split),
            "--max-batches",
            str(self.external_eval_max_batches),
            "--subset-mode",
            str(self.external_eval_subset_mode),
            "--subset-seed",
            str(self.external_eval_subset_seed),
        ]
        if self.composite_manifest_path:
            cmd.extend(["--composite-manifest-path", str(self.composite_manifest_path)])
        cmd.append("--synthetic-retry-history" if self.synthetic_retry_history else "--no-synthetic-retry-history")
        return cmd

    def _run_external_eval(self, checkpoint_dir: Path) -> None:
        if self.external_eval_gpu is None or self.external_eval_gpu < 0:
            return
        checkpoint_key = str(checkpoint_dir.resolve())
        if checkpoint_key in self._launched_eval_checkpoints:
            return

        cmd = self._build_external_eval_command(checkpoint_dir)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.external_eval_gpu)
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        log_path = checkpoint_dir / f"external_eval_gpu{self.external_eval_gpu}.log"
        with log_path.open("w", encoding="utf-8") as handle:
            handle.write("COMMAND:\n")
            handle.write(" ".join(cmd))
            handle.write("\n\n")
            handle.flush()
            if self.external_eval_sync:
                subprocess.run(
                    cmd,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    env=env,
                    check=False,
                )
            else:
                subprocess.Popen(
                    cmd,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    env=env,
                    start_new_session=True,
                )
        self._launched_eval_checkpoints.add(checkpoint_key)

    def _load_eval_payload(self, checkpoint_dir: Path) -> dict[str, Any] | None:
        eval_path = checkpoint_dir / "eval_val.json"
        if not eval_path.is_file():
            return None
        with eval_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _collect_checkpoint_scores(self) -> tuple[list[tuple[float, Path]], list[Path]]:
        output_dir = Path(self.args.output_dir)
        scored: list[tuple[float, Path]] = []
        unscored: list[Path] = []
        for checkpoint_dir in sorted(output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1])):
            if not checkpoint_dir.is_dir():
                continue
            payload = self._load_eval_payload(checkpoint_dir)
            if payload is None:
                unscored.append(checkpoint_dir)
                continue
            metric = get_eval_checkpoint_metric(payload, self.best_checkpoint_metric)
            if metric is None:
                unscored.append(checkpoint_dir)
                continue
            scored.append((metric, checkpoint_dir))
        return scored, unscored

    def _prune_checkpoints_by_eval(self) -> None:
        if self.checkpoint_retention_strategy != CHECKPOINT_RETENTION_BEST_EVAL:
            return
        keep_n = max(int(self.best_checkpoint_keep_n), 0)
        if keep_n <= 0:
            return

        scored, unscored = self._collect_checkpoint_scores()
        reverse = self.best_checkpoint_mode == "max"
        scored = sorted(scored, key=lambda item: item[0], reverse=reverse)

        keep: set[Path] = set()
        for _, checkpoint_dir in scored[:keep_n]:
            keep.add(checkpoint_dir.resolve())

        if self.external_eval_sync:
            keep_unscored = max(int(self.best_checkpoint_keep_unevaluated), 0)
            if keep_unscored > 0:
                for checkpoint_dir in unscored[-keep_unscored:]:
                    keep.add(checkpoint_dir.resolve())
        else:
            keep_unscored = len(unscored)
            for checkpoint_dir in unscored:
                keep.add(checkpoint_dir.resolve())

        for _, checkpoint_dir in scored:
            resolved = checkpoint_dir.resolve()
            if resolved not in keep:
                shutil.rmtree(checkpoint_dir, ignore_errors=True)
        for checkpoint_dir in unscored:
            resolved = checkpoint_dir.resolve()
            if resolved not in keep:
                shutil.rmtree(checkpoint_dir, ignore_errors=True)

        summary_path = Path(self.args.output_dir) / "best_eval_checkpoint_summary.json"
        summary = {
            "metric": self.best_checkpoint_metric,
            "mode": self.best_checkpoint_mode,
            "keep_n": keep_n,
            "keep_unscored": keep_unscored,
            "scored": [
                {"checkpoint": checkpoint_dir.name, "metric": metric}
                for metric, checkpoint_dir in sorted(scored, key=lambda item: item[0], reverse=reverse)[:keep_n]
            ],
            "unscored_kept": [path.name for path in unscored[-keep_unscored:]] if keep_unscored > 0 else [],
        }
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

    def _maybe_log_save_evaluate(
        self,
        tr_loss,
        grad_norm,
        model,
        trial,
        epoch,
        ignore_keys_for_eval,
        start_time,
        learning_rate=None,
    ):
        # Keep the stock logging behavior.
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            logs: dict[str, float] = {}
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
            tr_loss -= tr_loss

            logs["loss"] = round(
                tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged),
                4,
            )
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            logs["learning_rate"] = learning_rate if learning_rate is not None else self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()
            self.log(logs, start_time)

        # For step-based saving, checkpoint first so eval failures don't waste the run.
        if self.control.should_save and self.args.save_strategy == SaveStrategy.STEPS:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
            checkpoint_dir = Path(self.args.output_dir) / f"checkpoint-{self.state.global_step}"
            self._run_external_eval(checkpoint_dir)
            if self.external_eval_sync:
                self._prune_checkpoints_by_eval()

        metrics = None
        if self.control.should_evaluate:
            try:
                metrics = self._evaluate(trial, ignore_keys_for_eval)
                is_new_best_metric = self._determine_best_metric(metrics=metrics, trial=trial)
                if self.args.save_strategy == SaveStrategy.BEST:
                    self.control.should_save = is_new_best_metric
            except Exception as exc:
                warnings.warn(
                    f"AuxTrainer evaluation failed at step {self.state.global_step}: {exc}\n"
                    f"{traceback.format_exc()}",
                    RuntimeWarning,
                )
                error_path = Path(self.args.output_dir) / "eval_errors.log"
                with error_path.open("a", encoding="utf-8") as handle:
                    handle.write(f"[step {self.state.global_step}] {exc}\n")
                    handle.write(traceback.format_exc())
                    handle.write("\n")
                self.control.should_evaluate = False

        # For BEST / EPOCH strategies, preserve the default post-eval save behavior.
        if self.control.should_save and self.args.save_strategy != SaveStrategy.STEPS:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
            checkpoint_dir = Path(self.args.output_dir) / f"checkpoint-{self.state.global_step}"
            self._run_external_eval(checkpoint_dir)
            if self.external_eval_sync:
                self._prune_checkpoints_by_eval()


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def split_episode_indices(episodes: Sequence[dict[str, Any]], train_split: float, seed: int) -> tuple[set[int], set[int]]:
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


def build_composite_index_records(
    dataset: LeRobotSingleDataset,
    episode_segments: dict[int, list[dict[str, Any]]],
    episode_ids: set[int],
    success_tail_fraction: float,
    success_tail_min_steps: int,
) -> tuple[list[tuple[int, dict[str, Any]]], list[tuple[int, dict[str, Any]]]]:
    """
    Map base dataset steps to their composite segment and split them by state label.

    Episodes are split before this function is called, so train and validation never
    share frames from the same demonstration.
    """
    progress_records: list[tuple[int, dict[str, Any]]] = []
    success_records: list[tuple[int, dict[str, Any]]] = []
    starts_by_episode: dict[int, list[int]] = {}
    for episode_index, segments in episode_segments.items():
        starts_by_episode[int(episode_index)] = [
            int(segment["start_frame"]) for segment in segments
        ]

    for base_index, (trajectory_id, step_idx) in enumerate(dataset.all_steps):
        trajectory_id = int(trajectory_id)
        step_idx = int(step_idx)
        if trajectory_id not in episode_ids:
            continue
        segments = episode_segments.get(trajectory_id)
        if not segments:
            raise ValueError(f"No composite segments for episode {trajectory_id}")
        starts = starts_by_episode[trajectory_id]
        segment_index = bisect.bisect_right(starts, step_idx) - 1
        segment_index = max(0, min(segment_index, len(segments) - 1))
        segment = segments[segment_index]
        segment_start = int(segment["start_frame"])
        segment_length = int(segment["length"])
        local_step = step_idx - segment_start
        success_tail = max(
            int(success_tail_min_steps),
            int(math.ceil(success_tail_fraction * segment_length)),
        )
        success_start = max(0, segment_length - success_tail)
        if local_step >= success_start:
            success_records.append((base_index, segment))
        else:
            progress_records.append((base_index, segment))
    return progress_records, success_records


def build_datasets(args: Args) -> tuple[Dataset, Dataset]:
    manifest = load_manifest(Path(args.manifest_path))
    composite_manifest: dict[str, Any] | None = None
    if args.composite_manifest_path.strip():
        composite_manifest_path = Path(args.composite_manifest_path)
        if not composite_manifest_path.is_file():
            raise FileNotFoundError(f"Composite manifest not found: {composite_manifest_path}")
        composite_manifest = load_manifest(composite_manifest_path)
    observation_history_offsets = parse_observation_history_offsets(args.observation_history_offsets)
    retry_manifest_path = Path(args.retry_manifest_path) if args.retry_manifest_path else None
    retry_manifest: dict[str, Any] | None = None
    retry_by_source_dataset: dict[int, list[dict[str, Any]]] = {}
    if retry_manifest_path is not None and retry_manifest_path.is_file():
        retry_manifest = load_manifest(retry_manifest_path)
        retry_examples = retry_manifest.get("retry_examples", [])
        if args.retry_types.strip():
            allowed_retry_types = {
                item.strip()
                for item in args.retry_types.split(",")
                if item.strip()
            }
            retry_examples = [
                example
                for example in retry_examples
                if str(example.get("retry_type", "")) in allowed_retry_types
            ]
        if args.retry_manifest_max_samples > 0:
            retry_examples = retry_examples[: args.retry_manifest_max_samples]
        for example in retry_examples:
            dataset_index = int(example["source_dataset_index"])
            retry_by_source_dataset.setdefault(dataset_index, []).append(example)
    elif retry_manifest_path is not None:
        print(f"Retry manifest not found at {retry_manifest_path}; training positive-only labels.")

    cfg = build_data_config(args.data_config, observation_history_offsets)
    train_base_by_index: dict[int, LeRobotSingleDataset] = {}
    val_base_by_index: dict[int, LeRobotSingleDataset] = {}
    progress_train_parts: list[Dataset] = []
    success_train_parts: list[Dataset] = []
    progress_val_parts: list[Dataset] = []
    success_val_parts: list[Dataset] = []
    composite_progress_train_parts: list[Dataset] = []
    composite_success_train_parts: list[Dataset] = []
    composite_progress_val_parts: list[Dataset] = []
    composite_success_val_parts: list[Dataset] = []
    retry_train_examples: list[dict[str, Any]] = []
    retry_val_examples: list[dict[str, Any]] = []

    for dataset_index, record in enumerate(manifest["datasets"]):
        train_transform = copy.deepcopy(cfg.transform())
        val_transform = copy.deepcopy(cfg.transform())
        train_transform.train()
        val_transform.eval()

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
        train_base_by_index[dataset_index] = base_train
        val_base_by_index[dataset_index] = base_val

        train_indices = index_subset_for_episode_ids(base_train, train_ids)
        val_indices = index_subset_for_episode_ids(base_val, val_ids)
        progress_train_indices = filter_positive_indices_by_state(
            base_train,
            episode_lengths,
            train_indices,
            STATE_PROGRESS,
            args.success_tail_fraction,
            args.success_tail_min_steps,
        )
        success_train_indices = filter_positive_indices_by_state(
            base_train,
            episode_lengths,
            train_indices,
            STATE_SUCCESS,
            args.success_tail_fraction,
            args.success_tail_min_steps,
        )
        progress_val_indices = filter_positive_indices_by_state(
            base_val,
            episode_lengths,
            val_indices,
            STATE_PROGRESS,
            args.success_tail_fraction,
            args.success_tail_min_steps,
        )
        success_val_indices = filter_positive_indices_by_state(
            base_val,
            episode_lengths,
            val_indices,
            STATE_SUCCESS,
            args.success_tail_fraction,
            args.success_tail_min_steps,
        )

        progress_train_parts.append(
            PositiveAtomicAuxDataset(
                base_dataset=base_train,
                episode_lengths=episode_lengths,
                indices=progress_train_indices,
                progress_gamma=args.progress_gamma,
                success_tail_fraction=args.success_tail_fraction,
                success_tail_min_steps=args.success_tail_min_steps,
            )
        )
        success_train_parts.append(
            PositiveAtomicAuxDataset(
                base_dataset=base_train,
                episode_lengths=episode_lengths,
                indices=success_train_indices,
                progress_gamma=args.progress_gamma,
                success_tail_fraction=args.success_tail_fraction,
                success_tail_min_steps=args.success_tail_min_steps,
            )
        )
        progress_val_parts.append(
            PositiveAtomicAuxDataset(
                base_dataset=base_val,
                episode_lengths=episode_lengths,
                indices=progress_val_indices,
                progress_gamma=args.progress_gamma,
                success_tail_fraction=args.success_tail_fraction,
                success_tail_min_steps=args.success_tail_min_steps,
            )
        )
        success_val_parts.append(
            PositiveAtomicAuxDataset(
                base_dataset=base_val,
                episode_lengths=episode_lengths,
                indices=success_val_indices,
                progress_gamma=args.progress_gamma,
                success_tail_fraction=args.success_tail_fraction,
                success_tail_min_steps=args.success_tail_min_steps,
            )
        )

        for example in retry_by_source_dataset.get(dataset_index, []):
            source_episode_index = int(example["source_episode_index"])
            if source_episode_index in train_ids:
                retry_train_examples.append(example)
            elif source_episode_index in val_ids:
                retry_val_examples.append(example)

    if composite_manifest is not None:
        for record in composite_manifest.get("datasets", []):
            train_transform = copy.deepcopy(cfg.transform())
            val_transform = copy.deepcopy(cfg.transform())
            train_transform.train()
            val_transform.eval()

            episode_lengths = {
                int(episode["episode_index"]): int(episode["length"])
                for episode in record["episodes"]
            }
            train_ids, val_ids = split_episode_indices(
                record["episodes"],
                args.train_split,
                args.seed,
            )
            base_train = LeRobotSingleDataset(
                dataset_path=record["lerobot_root"],
                modality_configs=cfg.modality_config(),
                transforms=train_transform,
                embodiment_tag=EmbodimentTag(args.embodiment_tag),
                video_backend=args.video_backend,
            )
            base_val = LeRobotSingleDataset(
                dataset_path=record["lerobot_root"],
                modality_configs=cfg.modality_config(),
                transforms=val_transform,
                embodiment_tag=EmbodimentTag(args.embodiment_tag),
                video_backend=args.video_backend,
            )
            del episode_lengths

            episode_segments = {
                int(episode["episode_index"]): list(episode["subtasks"])
                for episode in record["episodes"]
            }
            progress_train_records, success_train_records = build_composite_index_records(
                base_train,
                episode_segments,
                train_ids,
                args.success_tail_fraction,
                args.success_tail_min_steps,
            )
            progress_val_records, success_val_records = build_composite_index_records(
                base_val,
                episode_segments,
                val_ids,
                args.success_tail_fraction,
                args.success_tail_min_steps,
            )
            composite_progress_train_parts.append(
                CompositeSubtaskAuxDataset(
                    base_dataset=base_train,
                    index_records=progress_train_records,
                    progress_gamma=args.progress_gamma,
                    success_tail_fraction=args.success_tail_fraction,
                    success_tail_min_steps=args.success_tail_min_steps,
                )
            )
            composite_success_train_parts.append(
                CompositeSubtaskAuxDataset(
                    base_dataset=base_train,
                    index_records=success_train_records,
                    progress_gamma=args.progress_gamma,
                    success_tail_fraction=args.success_tail_fraction,
                    success_tail_min_steps=args.success_tail_min_steps,
                )
            )
            composite_progress_val_parts.append(
                CompositeSubtaskAuxDataset(
                    base_dataset=base_val,
                    index_records=progress_val_records,
                    progress_gamma=args.progress_gamma,
                    success_tail_fraction=args.success_tail_fraction,
                    success_tail_min_steps=args.success_tail_min_steps,
                )
            )
            composite_success_val_parts.append(
                CompositeSubtaskAuxDataset(
                    base_dataset=base_val,
                    index_records=success_val_records,
                    progress_gamma=args.progress_gamma,
                    success_tail_fraction=args.success_tail_fraction,
                    success_tail_min_steps=args.success_tail_min_steps,
                )
            )

    progress_train = ConcatDataset(progress_train_parts)
    success_train = ConcatDataset(success_train_parts)
    progress_val = ConcatDataset(progress_val_parts)
    success_val = ConcatDataset(success_val_parts)

    train_buckets: list[Dataset] = [progress_train, success_train]
    train_weights: list[float] = [args.progress_sample_weight, args.success_sample_weight]
    val_parts: list[Dataset] = [progress_val, success_val]

    composite_progress_train = ConcatDataset(composite_progress_train_parts) if composite_progress_train_parts else None
    composite_success_train = ConcatDataset(composite_success_train_parts) if composite_success_train_parts else None
    composite_progress_val = ConcatDataset(composite_progress_val_parts) if composite_progress_val_parts else None
    composite_success_val = ConcatDataset(composite_success_val_parts) if composite_success_val_parts else None
    if composite_progress_train is not None:
        train_buckets.append(composite_progress_train)
        train_weights.append(args.progress_sample_weight * args.composite_sample_weight)
        if composite_progress_val is not None:
            val_parts.append(composite_progress_val)
    if composite_success_train is not None:
        train_buckets.append(composite_success_train)
        train_weights.append(args.success_sample_weight * args.composite_sample_weight)
        if composite_success_val is not None:
            val_parts.append(composite_success_val)

    if retry_train_examples:
        retry_train = RetryAtomicAuxDataset(
            base_datasets_by_index=train_base_by_index,
            retry_examples=retry_train_examples,
            retry_progress_default=args.retry_progress_default,
            observation_history_offsets=observation_history_offsets,
            synthetic_retry_history=args.synthetic_retry_history,
        )
        train_buckets.append(retry_train)
        train_weights.append(args.retry_sample_weight)
    if retry_val_examples:
        retry_val = RetryAtomicAuxDataset(
            base_datasets_by_index=val_base_by_index,
            retry_examples=retry_val_examples,
            retry_progress_default=args.retry_progress_default,
            observation_history_offsets=observation_history_offsets,
            synthetic_retry_history=args.synthetic_retry_history,
        )
        val_parts.append(retry_val)

    inferred_epoch_size = (
        len(progress_train)
        + len(success_train)
        + (len(composite_progress_train) if composite_progress_train is not None else 0)
        + (len(composite_success_train) if composite_success_train is not None else 0)
        + len(retry_train_examples)
    )
    train_epoch_size = args.train_epoch_size if args.train_epoch_size > 0 else inferred_epoch_size
    train_dataset = WeightedAuxDataset(
        datasets=train_buckets,
        weights=train_weights,
        epoch_size=train_epoch_size,
        seed=args.seed,
    )
    val_dataset = ConcatDataset(val_parts)

    print(
        "Aux dataset sizes: "
        f"progress_train={len(progress_train)}, success_train={len(success_train)}, "
        f"composite_progress_train={len(composite_progress_train) if composite_progress_train is not None else 0}, "
        f"composite_success_train={len(composite_success_train) if composite_success_train is not None else 0}, "
        f"retry_train={len(retry_train_examples)}, train_epoch_size={len(train_dataset)}, "
        f"progress_val={len(progress_val)}, success_val={len(success_val)}, "
        f"composite_progress_val={len(composite_progress_val) if composite_progress_val is not None else 0}, "
        f"composite_success_val={len(composite_success_val) if composite_success_val is not None else 0}, "
        f"retry_val={len(retry_val_examples)}, val={len(val_dataset)}",
        flush=True,
    )
    print(
        "Aux sampling weights: "
        f"progress={args.progress_sample_weight}, success={args.success_sample_weight}, "
        f"retry={args.retry_sample_weight if retry_train_examples else 0.0}, "
        f"composite={args.composite_sample_weight if composite_manifest is not None else 0.0}",
        flush=True,
    )
    print(
        "Aux history config: "
        f"observation_history_offsets={observation_history_offsets or [0]}, "
        f"synthetic_retry_history={args.synthetic_retry_history}, "
        f"aux_context_mode={args.aux_context_mode}",
        flush=True,
    )
    if retry_manifest is not None:
        print(
            f"Loaded retry manifest with counts={retry_manifest.get('retry_counts', {})} "
            f"from {retry_manifest_path}",
            flush=True,
        )
    if composite_manifest is not None:
        print(
            f"Loaded composite manifest with datasets={composite_manifest.get('num_datasets', 0)} "
            f"episodes={composite_manifest.get('num_episodes', 0)} "
            f"from {args.composite_manifest_path}",
            flush=True,
        )

    return train_dataset, val_dataset


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    del predictions, labels
    return {}


def main(args: Args) -> None:
    set_seed(args.seed)
    observation_history_offsets = parse_observation_history_offsets(args.observation_history_offsets)
    validate_aux_context_mode(args.aux_context_mode)
    validate_checkpoint_retention_strategy(args.checkpoint_retention_strategy)
    validate_best_checkpoint_mode(args.best_checkpoint_mode)
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except RuntimeError:
        pass

    train_dataset, val_dataset = build_datasets(args)
    model = GR00TAuxiliaryModel(
        args.checkpoint_path,
        state_loss_weights=[
            args.progress_class_loss_weight,
            args.success_class_loss_weight,
            args.retry_class_loss_weight,
        ],
        state_label_smoothing=args.state_label_smoothing,
        aux_context_mode=args.aux_context_mode,
        observation_history_offsets=observation_history_offsets,
        synthetic_retry_history=args.synthetic_retry_history,
    )
    collator = AtomicAuxCollator()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        if args.wandb_entity:
            os.environ.setdefault("WANDB_ENTITY", args.wandb_entity)
        os.environ.setdefault("WANDB_MODE", args.wandb_mode)
        os.environ.setdefault("WANDB_LOG_MODEL", args.wandb_log_model)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=output_dir.name,
        remove_unused_columns=False,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=args.dataloader_num_workers > 0,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_steps=args.max_steps,
        num_train_epochs=100,
        save_steps=args.save_steps,
        save_strategy="steps",
        save_total_limit=None if args.checkpoint_retention_strategy == CHECKPOINT_RETENTION_BEST_EVAL else args.save_total_limit,
        do_eval=args.inline_eval,
        eval_strategy="steps" if args.inline_eval else "no",
        eval_steps=args.save_steps if args.inline_eval else None,
        logging_steps=10,
        report_to=[args.report_to] if args.report_to != "none" else [],
        bf16=args.bf16,
        tf32=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        seed=args.seed,
        dataloader_drop_last=False,
    )

    trainer = AuxTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if args.inline_eval else None,
        data_collator=collator,
        external_eval_gpu=args.external_eval_gpu,
        external_eval_batch_size=args.external_eval_batch_size,
        external_eval_num_workers=args.external_eval_num_workers,
        external_eval_split=args.external_eval_split,
        external_eval_max_batches=args.external_eval_max_batches,
        external_eval_subset_mode=args.external_eval_subset_mode,
        external_eval_subset_seed=args.external_eval_subset_seed,
        manifest_path=args.manifest_path,
        retry_manifest_path=args.retry_manifest_path,
        composite_manifest_path=args.composite_manifest_path,
        base_checkpoint_path=args.checkpoint_path,
        data_config_name=args.data_config,
        embodiment_tag=args.embodiment_tag,
        video_backend=args.video_backend,
        progress_gamma=args.progress_gamma,
        success_tail_fraction=args.success_tail_fraction,
        success_tail_min_steps=args.success_tail_min_steps,
        retry_progress_default=args.retry_progress_default,
        retry_manifest_max_samples=args.retry_manifest_max_samples,
        state_label_smoothing=args.state_label_smoothing,
        observation_history_offsets=args.observation_history_offsets,
        synthetic_retry_history=args.synthetic_retry_history,
        aux_context_mode=args.aux_context_mode,
        checkpoint_retention_strategy=args.checkpoint_retention_strategy,
        best_checkpoint_metric=args.best_checkpoint_metric,
        best_checkpoint_mode=args.best_checkpoint_mode,
        best_checkpoint_keep_n=args.save_total_limit,
        best_checkpoint_keep_unevaluated=args.best_checkpoint_keep_unevaluated,
        external_eval_sync=args.external_eval_sync,
        train_split=args.train_split,
        seed=args.seed,
    )
    resume_from_checkpoint = (
        ensure_hf_resume_checkpoint(args.resume_from_checkpoint)
        if args.resume_from_checkpoint
        else None
    )
    if resume_from_checkpoint:
        allow_legacy_numpy_rng_state_load()
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer._prune_checkpoints_by_eval()
    trainer.save_model(str(output_dir / "final"))


if __name__ == "__main__":
    config = tyro.cli(Args)
    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if available_gpus == 0:
        raise RuntimeError(
            "No CUDA GPU is visible in the current environment. "
            "Use a GPU-enabled environment to run auxiliary GR00T training."
        )
    if config.num_gpus > 1 and available_gpus > 1 and os.environ.get("IS_TORCHRUN", "0") != "1":
        raise NotImplementedError("Multi-GPU launch is not wired for the auxiliary trainer yet. Start with num_gpus=1.")
    main(config)
