#!/usr/bin/env python3
"""Train a frozen-Xiaomi auxiliary progress/state head on RoboCasa GT stages.

The Xiaomi VLA remains frozen.  Only two small heads are optimized:

* progress_head: scalar progress regression in [0, 1]
* state_head: {progress, success, retry} classification

The first version trains composite GT positive examples.  Retry examples can
be added later once their synthetic observation history is materialized with
the same Xiaomi preprocessing.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.long_horizon_controller.xiaomi_aux_runtime import (  # noqa: E402
    STATE_CLASS_NAMES,
    XiaomiAuxiliaryHeads,
    masked_mean,
)
from scripts.long_horizon_controller.xiaomi_policy_adapter import (  # noqa: E402
    CAMERA_KEYS,
    _center_crop,
)


STATE_PROGRESS = 0
STATE_SUCCESS = 1
STATE_RETRY = 2
STATE_KEYS = (
    "state.base_position",
    "state.base_rotation",
    "state.end_effector_position_relative",
    "state.end_effector_rotation_relative",
    "state.gripper_qpos",
)


@dataclass
class SampleRef:
    dataset_index: int
    episode_index: int
    frame_index: int
    stage_index: int
    stage_start: int
    stage_length: int
    task_name: str
    task_instruction: str
    stage_instruction: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=(
            "/data/zjw/workspace/Isaac-GR00T/expdata/aux_progress/"
            "composite_gt_aux_manifest.json"
        ),
    )
    parser.add_argument(
        "--model-path",
        default="/data/zjw/workspace/Isaac-GR00T/expdata/Xiaomi-Robotics-1-RoboCasa365",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "/data/zjw/workspace/Isaac-GR00T/expdata/aux_progress/"
            "xiaomi_composite_gt_aux_run1"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--history-frames", type=int, default=4)
    parser.add_argument("--history-interval-steps", type=int, default=2)
    parser.add_argument("--sample-stride", type=int, default=4)
    parser.add_argument("--max-samples-per-stage", type=int, default=64)
    parser.add_argument("--train-split", type=float, default=0.9)
    parser.add_argument("--progress-gamma", type=float, default=1.5)
    parser.add_argument("--success-tail-fraction", type=float, default=0.1)
    parser.add_argument("--success-tail-min-steps", type=int, default=3)
    parser.add_argument("--head-hidden-dim", type=int, default=1280)
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Optional task whitelist. Defaults to every task in the manifest.",
    )
    parser.add_argument("--max-train-samples", type=int, default=-1)
    parser.add_argument("--max-val-samples", type=int, default=-1)
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--val-batches", type=int, default=128)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def quat_xyzw_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    norm = np.linalg.norm(quaternion)
    if norm < 1e-12:
        return np.zeros(3, dtype=np.float32)
    quaternion = quaternion / norm
    if quaternion[3] < 0:
        quaternion = -quaternion
    xyz = quaternion[:3]
    sin_half = np.linalg.norm(xyz)
    if sin_half < 1e-12:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arctan2(sin_half, np.clip(quaternion[3], -1.0, 1.0))
    return (xyz / sin_half * angle).astype(np.float32)


def state_to_xiaomi(raw: dict[str, Any]) -> np.ndarray:
    length = len(np.asarray(raw[STATE_KEYS[0]]))
    rows: list[np.ndarray] = []
    for index in range(length):
        values = np.concatenate(
            [
                np.asarray(raw["state.end_effector_position_relative"][index]),
                quat_xyzw_to_axis_angle(
                    raw["state.end_effector_rotation_relative"][index]
                ),
                np.asarray(raw["state.gripper_qpos"][index]),
                np.asarray(raw["state.base_position"][index]),
                quat_xyzw_to_axis_angle(raw["state.base_rotation"][index]),
            ]
        ).astype(np.float32)
        if values.shape != (14,):
            raise ValueError(f"Expected 14D RoboCasa state, got {values.shape}")
        row = np.zeros(60, dtype=np.float32)
        row[:14] = values
        rows.append(row)
    return np.stack(rows, axis=0)[None, ...]


def xiaomi_prompt(processor: Any, instruction: str) -> str:
    marker = (
        f"{processor.vision_start_token}"
        f"{processor.video_token}"
        f"{processor.vision_end_token}"
    )
    return (
        "<|im_start|>user\n"
        f"Left camera: {marker}\n"
        f"Right camera: {marker}\n"
        f"Wrist camera: {marker}\n\n"
        "Generate robot actions for the task:\n"
        f"{instruction} /no_cot"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<cot></cot>"
        "<|im_end|>\n"
    )


class XiaomiStageDataset(Dataset):
    """Index stage-consistent frames while keeping dataset IO lazy."""

    def __init__(self, manifest: dict[str, Any], args: argparse.Namespace, split: str):
        from robocasa.utils.groot_utils.groot_dataset import (
            LeRobotSingleDataset,
            ModalityConfig,
        )

        self.args = args
        self.split = split
        self.datasets: list[Any] = []
        self.refs: list[SampleRef] = []
        offsets = [
            -args.history_interval_steps * (args.history_frames - 1 - index)
            for index in range(args.history_frames)
        ]
        modality = {
            "video": ModalityConfig(
                delta_indices=offsets,
                modality_keys=list(CAMERA_KEYS),
            ),
            "state": ModalityConfig(
                delta_indices=offsets,
                modality_keys=list(STATE_KEYS),
            ),
        }
        wanted = set(args.tasks or [])
        rng = random.Random(args.seed)

        for dataset_index, record in enumerate(manifest.get("datasets", [])):
            task_name = str(record["task_name"])
            if wanted and task_name not in wanted:
                continue
            episodes = list(record.get("episodes", []))
            episode_ids = [int(item["episode_index"]) for item in episodes]
            rng.shuffle(episode_ids)
            cut = max(1, int(len(episode_ids) * args.train_split))
            selected = set(
                episode_ids[:cut] if split == "train" else episode_ids[cut:]
            )
            if split == "val" and not selected:
                selected = {episode_ids[0]}

            dataset = LeRobotSingleDataset(
                dataset_path=record.get("lerobot_root") or record.get("dataset_root"),
                modality_configs=modality,
                embodiment_tag="new_embodiment",
                video_backend="opencv",
            )
            self.datasets.append(dataset)
            local_dataset_index = len(self.datasets) - 1
            for episode in episodes:
                episode_index = int(episode["episode_index"])
                if episode_index not in selected:
                    continue
                for stage_index, stage in enumerate(episode.get("subtasks", [])):
                    start = int(stage["start_frame"])
                    end = int(stage["end_frame"])
                    length = int(stage["length"])
                    first = max(
                        start + args.history_interval_steps * (args.history_frames - 1),
                        start,
                    )
                    positions = list(
                        range(first, max(first, end), max(args.sample_stride, 1))
                    )
                    if len(positions) > args.max_samples_per_stage > 0:
                        positions = rng.sample(positions, args.max_samples_per_stage)
                        positions.sort()
                    for frame_index in positions:
                        self.refs.append(
                            SampleRef(
                                dataset_index=local_dataset_index,
                                episode_index=episode_index,
                                frame_index=frame_index,
                                stage_index=stage_index,
                                stage_start=start,
                                stage_length=length,
                                task_name=task_name,
                                task_instruction=str(
                                    episode.get("task_instruction", task_name)
                                ),
                                stage_instruction=str(stage["instruction"]),
                            )
                        )
        if not self.refs:
            raise RuntimeError(
                f"No Xiaomi auxiliary {split} samples found. "
                "Check manifest, GT columns, and split settings."
            )

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        ref = self.refs[index]
        raw = self.datasets[ref.dataset_index].get_step_data(
            ref.episode_index,
            ref.frame_index,
        )
        videos = [
            np.stack(
                [
                    np.asarray(_center_crop(frame, 0.95), dtype=np.uint8)
                    for frame in np.asarray(raw[key])
                ],
                axis=0,
            )
            for key in CAMERA_KEYS
        ]
        local_step = ref.frame_index - ref.stage_start
        if ref.stage_length <= 1:
            progress = 1.0
        else:
            progress = float(
                np.clip(
                    (local_step / float(ref.stage_length - 1))
                    ** self.args.progress_gamma,
                    0.0,
                    1.0,
                )
            )
        success_tail = max(
            self.args.success_tail_min_steps,
            int(math.ceil(self.args.success_tail_fraction * ref.stage_length)),
        )
        state = (
            STATE_SUCCESS
            if local_step >= max(0, ref.stage_length - success_tail)
            else STATE_PROGRESS
        )
        instruction = (
            f"Overall task: {ref.task_instruction}\n"
            f"Current subtask: {ref.stage_instruction}"
        )
        return {
            "videos": videos,
            "state": state_to_xiaomi(raw),
            "instruction": instruction,
            "progress_target": np.float32(progress),
            "state_target": np.int64(state),
            "ref": ref,
        }


def make_inputs(processor: Any, sample: dict[str, Any], device: str) -> dict[str, Any]:
    inputs = processor(
        videos=sample["videos"],
        text=xiaomi_prompt(processor, sample["instruction"]),
        return_tensors="pt",
        state=sample["state"],
        robot_type="robocasa365",
    )
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


def vlm_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in inputs.items()
        if key not in {"state", "action_mask"}
    }


def evaluate(
    model: Any,
    processor: Any,
    heads: XiaomiAuxiliaryHeads,
    dataset: XiaomiStageDataset,
    device: str,
    max_batches: int,
) -> dict[str, float]:
    was_training = heads.training
    heads.eval()
    total = 0
    correct = 0
    progress_error = 0.0
    with torch.inference_mode():
        for index in range(min(len(dataset), max_batches)):
            sample = dataset[index]
            inputs = make_inputs(processor, sample, device)
            outputs = model.vlm(**vlm_inputs(inputs), use_cache=True)
            pooled = masked_mean(outputs.last_hidden_state, inputs["attention_mask"])
            progress, logits = heads(pooled)
            target_state = torch.tensor(
                [int(sample["state_target"])], device=device, dtype=torch.long
            )
            target_progress = torch.tensor(
                [float(sample["progress_target"])], device=device, dtype=torch.float32
            )
            correct += int(logits.argmax(dim=-1).eq(target_state).sum().item())
            progress_error += float(
                torch.abs(progress.float() - target_progress).sum().item()
            )
            total += 1
    if was_training:
        heads.train()
    return {
        "val_state_accuracy": correct / max(total, 1),
        "val_progress_mae": progress_error / max(total, 1),
        "val_samples": float(total),
    }


def save_checkpoint(
    output_dir: Path,
    step: int,
    heads: XiaomiAuxiliaryHeads,
    args: argparse.Namespace,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "aux_heads.pt"
    torch.save(
        {key: value.detach().cpu() for key, value in heads.state_dict().items()},
        weights_path,
    )
    config = {
        "base_model_path": args.model_path,
        "feature_type": "vlm_last_hidden_masked_mean",
        "feature_dim": 2560,
        "head_hidden_dim": args.head_hidden_dim,
        "state_classes": list(STATE_CLASS_NAMES),
        "observation_history_offsets": [
            -args.history_interval_steps * (args.history_frames - 1 - index)
            for index in range(args.history_frames)
        ],
        "history_length": args.history_frames,
        "history_interval_steps": args.history_interval_steps,
        "prompt_format": "structured",
        "crop_ratio": 0.95,
        "global_step": step,
        "args": vars(args),
    }
    (output_dir / "aux_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return weights_path


def main() -> None:
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError(
            "The first implementation supports batch-size=1 because Xiaomi's "
            "three-camera processor layout is single-example specific. Use "
            "--gradient-accumulation-steps for a larger effective batch."
        )
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for Xiaomi auxiliary training.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    train_dataset = XiaomiStageDataset(manifest, args, "train")
    val_dataset = XiaomiStageDataset(manifest, args, "val")
    print(
        f"Xiaomi auxiliary samples: train={len(train_dataset)} "
        f"val={len(val_dataset)}",
        flush=True,
    )
    if args.dry_run:
        sample = train_dataset[0]
        print(
            json.dumps(
                {
                    "instruction": sample["instruction"],
                    "video_shapes": [list(video.shape) for video in sample["videos"]],
                    "state_shape": list(sample["state"].shape),
                    "progress_target": float(sample["progress_target"]),
                    "state_target": int(sample["state_target"]),
                },
                indent=2,
            ),
            flush=True,
        )
        return

    from transformers import AutoModel, AutoProcessor

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=False,
    )
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation="eager",
    ).to(args.device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    heads = XiaomiAuxiliaryHeads(
        input_dim=2560,
        hidden_dim=args.head_hidden_dim,
    ).to(args.device, dtype=dtype)
    optimizer = torch.optim.AdamW(
        heads.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    state_loss = nn.CrossEntropyLoss()
    progress_loss = nn.SmoothL1Loss()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_args.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    rng = random.Random(args.seed + 1)
    max_train = (
        len(train_dataset)
        if args.max_train_samples <= 0
        else min(len(train_dataset), args.max_train_samples)
    )
    train_indices = list(range(max_train))
    for step in range(1, args.steps + 1):
        index = train_indices[rng.randrange(len(train_indices))]
        sample = train_dataset[index]
        inputs = make_inputs(processor, sample, args.device)
        with torch.inference_mode():
            outputs = model.vlm(**vlm_inputs(inputs), use_cache=True)
            pooled = masked_mean(outputs.last_hidden_state, inputs["attention_mask"])
        # VLM is frozen and runs in inference_mode; clone before attaching the
        # trainable heads so autograd does not receive an inference tensor.
        pooled = pooled.detach().clone().to(dtype=dtype)
        progress_pred, state_logits = heads(pooled)
        target_progress = torch.tensor(
            [float(sample["progress_target"])],
            device=args.device,
            dtype=progress_pred.dtype,
        )
        target_state = torch.tensor(
            [int(sample["state_target"])],
            device=args.device,
            dtype=torch.long,
        )
        loss = (
            progress_loss(progress_pred.float(), target_progress.float())
            + state_loss(state_logits.float(), target_state)
        ) / max(args.gradient_accumulation_steps, 1)
        loss.backward()
        if step % max(args.gradient_accumulation_steps, 1) == 0:
            torch.nn.utils.clip_grad_norm_(heads.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if step == 1 or step % 20 == 0:
            print(
                f"step={step}/{args.steps} loss={loss.item():.6f} "
                f"sample_state={STATE_CLASS_NAMES[int(sample['state_target'])]}",
                flush=True,
            )
        if args.val_every > 0 and step % args.val_every == 0:
            metrics = evaluate(
                model,
                processor,
                heads,
                val_dataset,
                args.device,
                args.val_batches,
            )
            print(
                " ".join(f"{key}={value:.6f}" for key, value in metrics.items()),
                flush=True,
            )
        if args.save_every > 0 and step % args.save_every == 0:
            checkpoint_dir = output_dir / f"checkpoint-{step}"
            save_checkpoint(checkpoint_dir, step, heads, args)
            print(f"saved {checkpoint_dir}", flush=True)

    final_dir = output_dir / "final"
    save_checkpoint(final_dir, args.steps, heads, args)
    print(f"saved {final_dir}", flush=True)


if __name__ == "__main__":
    main()
