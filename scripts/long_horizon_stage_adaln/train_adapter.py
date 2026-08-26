#!/usr/bin/env python3
"""Train the isolated Xiaomi stage-text AdaLN adapter.

This script intentionally trains only ``stage_adaln``.  The Xiaomi VLM and
DiT remain frozen.  Training examples are sampled from a manifest produced by
``manifest.py`` and action chunks crossing a GT subtask boundary are skipped.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.long_horizon_stage_adaln.model import (  # noqa: E402
    adapter_state_dict,
    attach_stage_adaln,
    encode_text_condition,
)


CAMERA_KEYS = (
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
)
STATE_KEYS = (
    "state.base_position",
    "state.base_rotation",
    "state.end_effector_position_relative",
    "state.end_effector_rotation_relative",
    "state.gripper_qpos",
)
ACTION_KEYS = (
    "action.base_motion",
    "action.control_mode",
    "action.end_effector_position",
    "action.end_effector_rotation",
    "action.gripper_close",
)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=(
            "/data/zjw/workspace/Isaac-GR00T/expdata/"
            "long_horizon_stage_adaln/non_target_manifest.json"
        ),
    )
    parser.add_argument(
        "--model-path",
        default="/data/zjw/workspace/Isaac-GR00T/expdata/Xiaomi-Robotics-1-RoboCasa365",
    )
    parser.add_argument(
        "--output-root",
        default=(
            "/data/zjw/workspace/Isaac-GR00T/expdata/"
            "long_horizon_stage_adaln/adapter_training"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--history-frames", type=int, default=4)
    # Keep this aligned with eval_kettle_oracle.py's
    # --xiaomi-history-interval-steps default.
    parser.add_argument("--history-interval-frames", type=int, default=2)
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--max-samples-per-episode", type=int, default=128)
    parser.add_argument("--max-video-size", type=int, default=224)
    parser.add_argument(
        "--stage-condition-format",
        choices=["full", "subtask_only"],
        default="full",
        help=(
            "Text encoded for the DiT AdaLN condition. 'subtask_only' uses "
            "the raw RoboCasa GT subtask sentence without a prefix."
        ),
    )
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


def state_to_xiaomi(state: dict[str, np.ndarray]) -> np.ndarray:
    length = len(state["state.end_effector_position_relative"])
    rows: list[np.ndarray] = []
    for index in range(length):
        values = np.concatenate(
            [
                np.asarray(state["state.end_effector_position_relative"][index]),
                quat_xyzw_to_axis_angle(
                    state["state.end_effector_rotation_relative"][index]
                ),
                np.asarray(state["state.gripper_qpos"][index]),
                np.asarray(state["state.base_position"][index]),
                quat_xyzw_to_axis_angle(state["state.base_rotation"][index]),
            ]
        ).astype(np.float32)
        if values.shape != (14,):
            raise ValueError(f"Expected 14D RoboCasa state, got {values.shape}")
        output = np.zeros(60, dtype=np.float32)
        output[:14] = values
        rows.append(output)
    output = np.stack(rows, axis=0)
    return output


def action_to_xiaomi(action: dict[str, np.ndarray]) -> np.ndarray:
    """Convert LeRobot action modality order to Xiaomi's first 12 channels."""
    base = np.asarray(action["action.base_motion"])
    control = np.asarray(action["action.control_mode"])
    eef_pos = np.asarray(action["action.end_effector_position"])
    eef_rot = np.asarray(action["action.end_effector_rotation"])
    grip = np.asarray(action["action.gripper_close"])
    values = np.concatenate([eef_pos, eef_rot, grip, base, control], axis=-1)
    if values.shape[-1] != 12:
        raise ValueError(f"Expected 12D action, got {values.shape}")
    output = np.zeros((*values.shape[:-1], 60), dtype=np.float32)
    output[..., :12] = values
    return output


def resize_frame(frame: np.ndarray, size: int) -> np.ndarray:
    import cv2

    frame = np.asarray(frame, dtype=np.uint8)
    if frame.shape[:2] == (size, size):
        return frame
    return cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)


class RoboCasaStageDataset:
    def __init__(self, manifest: dict[str, Any], args: argparse.Namespace) -> None:
        from robocasa.utils.groot_utils.groot_dataset import (
            LeRobotSingleDataset,
            ModalityConfig,
        )

        self.args = args
        self.datasets: list[Any] = []
        self.dataset_task_names: list[str] = []
        self.samples: list[tuple[int, int]] = []
        self.episode_lengths: list[int] = []
        self.task_texts: list[str] = []
        modality = {
            "video": ModalityConfig(
                delta_indices=[
                    -args.history_interval_frames * (args.history_frames - 1 - i)
                    for i in range(args.history_frames)
                ],
                modality_keys=list(CAMERA_KEYS),
            ),
            "state": ModalityConfig(
                delta_indices=[
                    -args.history_interval_frames * (args.history_frames - 1 - i)
                    for i in range(args.history_frames)
                ],
                modality_keys=list(STATE_KEYS),
            ),
            "action": ModalityConfig(
                delta_indices=list(range(args.action_horizon)),
                modality_keys=list(ACTION_KEYS),
            ),
        }

        for dataset_record in manifest["datasets"]:
            dataset = LeRobotSingleDataset(
                dataset_path=dataset_record["dataset_root"],
                modality_configs=modality,
                embodiment_tag="new_embodiment",
                video_backend="opencv",
            )
            dataset_index = len(self.datasets)
            self.datasets.append(dataset)
            self.dataset_task_names.append(str(dataset_record["task_name"]))
            for episode in dataset_record["episodes"]:
                episode_index = int(episode["episode_index"])
                length = int(episode["length"])
                self.episode_lengths.append(length)
                frame_count = 0
                table = dataset.get_trajectory_data(episode_index)
                if "subtask_idx" not in table.columns:
                    continue
                labels = table["subtask_idx"].to_numpy()
                for base in range(
                    args.history_interval_frames * (args.history_frames - 1),
                    max(
                        0,
                        length - args.action_horizon + 1,
                    ),
                ):
                    label_window = labels[base : base + args.action_horizon]
                    if len(label_window) != args.action_horizon:
                        continue
                    if np.all(label_window == label_window[0]):
                        self.samples.append((dataset_index, episode_index * 10_000_000 + base))
                        frame_count += 1
                        if frame_count >= args.max_samples_per_episode:
                            break

        if not self.samples:
            raise RuntimeError(
                "No valid stage-consistent action chunks were found. "
                "Check the manifest, GT fields, and dataset source."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def get(self, index: int) -> dict[str, Any]:
        dataset_index, packed = self.samples[index]
        episode_index, base_index = divmod(packed, 10_000_000)
        raw = self.datasets[dataset_index].get_step_data(episode_index, base_index)
        episode_table = self.datasets[dataset_index].get_trajectory_data(episode_index)
        ann = episode_table.iloc[base_index]
        task_table = self.datasets[dataset_index].tasks
        task_instruction = str(
            task_table.loc[int(ann["annotation.human.task_description"])]["task"]
        )
        stage_name = str(task_table.loc[int(ann["annotation.human.subtask_stage"])]["task"])
        skill_name = str(task_table.loc[int(ann["annotation.human.subtask_name"])]["task"])
        subtask_instruction = str(
            task_table.loc[int(ann["annotation.human.subtask"])]["task"]
        )
        stage_text = (
            f"Atomic skill: {skill_name}. Stage: {stage_name}. "
            f"Current subtask: {subtask_instruction}"
        )
        if self.args.stage_condition_format == "subtask_only":
            stage_text = subtask_instruction
        videos = [
            np.stack(
                [resize_frame(frame, self.args.max_video_size) for frame in raw[key]],
                axis=0,
            )
            for key in CAMERA_KEYS
        ]
        states = state_to_xiaomi(
            {key: np.asarray(raw[key]) for key in STATE_KEYS}
        )
        actions = action_to_xiaomi(
            {key: np.asarray(raw[key]) for key in ACTION_KEYS}
        )
        return {
            "task_name": self.dataset_task_names[dataset_index],
            "task_instruction": task_instruction,
            "stage_text": stage_text,
            "stage_name": stage_name,
            "atomic_skill": skill_name,
            "videos": videos,
            "state": states[None, ...],
            "action": actions,
            "dataset_index": dataset_index,
            "episode_index": episode_index,
            "base_index": base_index,
        }


def full_prompt(processor: Any, instruction: str) -> str:
    marker = (
        f"{processor.vision_start_token}"
        f"{processor.video_token}"
        f"{processor.vision_end_token}"
    )
    return (
        "<|im_start|>user\n"
        f"Left camera: {marker}\nRight camera: {marker}\n"
        f"Wrist camera: {marker}\n\nGenerate robot actions for the task:\n"
        f"{instruction} /no_cot<|im_end|>\n"
        "<|im_start|>assistant\n<cot></cot><|im_end|>\n"
    )


def prepare_action_forward(
    model: Any,
    inputs: dict[str, Any],
    target_action: torch.Tensor,
    stage_hidden: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    vlm_inputs = {
        key: value
        for key, value in inputs.items()
        if key not in {"state", "action_mask"}
    }
    with torch.no_grad():
        vlm_outputs = model.vlm(**vlm_inputs, use_cache=True)
    action_mask = inputs["action_mask"]
    state = inputs["state"]
    action_length = target_action.shape[1]
    state_length = state.shape[1]
    batch_size = target_action.shape[0]
    dit_query_length = action_length + state_length + 1
    position_ids = (
        torch.arange(
            0,
            dit_query_length,
            device=target_action.device,
        ).view(1, 1, -1).repeat(3, batch_size, 1)
        + vlm_outputs.position_ids.max(dim=-1)[0][..., None]
        + 1
    )
    position_embeds = model.rotary_emb(action_mask, position_ids)
    dit_mask = torch.tril(
        torch.ones(
            (batch_size, dit_query_length, dit_query_length),
            device=target_action.device,
        ),
        diagonal=0,
    )
    cache_mask = vlm_outputs.attention_mask[:, None, :].expand(
        -1,
        dit_query_length,
        -1,
    )
    attn_mask = torch.cat([cache_mask, dit_mask], dim=-1)[:, None].bool()
    state_embed = model.state_projector(state)
    noise = torch.randn_like(target_action)
    t = torch.rand(
        (batch_size, 1, 1),
        device=target_action.device,
        dtype=target_action.dtype,
    )
    noisy = (1.0 - t) * noise + t * target_action
    model._stage_adaln_condition = stage_hidden
    prediction = model.dit_forward(
        noisy_action=noisy,
        t=t,
        action_mask=action_mask,
        state_embed=state_embed,
        position_embeds=position_embeds,
        past_key_values=vlm_outputs.past_key_values,
        attn_mask=attn_mask,
    )
    return prediction, target_action - noise


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}. Run manifest.py first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset = RoboCasaStageDataset(manifest, args)
    print(f"stage-adaln samples={len(dataset)} datasets={len(dataset.datasets)}")
    if args.dry_run:
        sample = dataset.get(0)
        print(
            json.dumps(
                {
                    "task_instruction": sample["task_instruction"],
                    "stage_text": sample["stage_text"],
                    "state_shape": list(sample["state"].shape),
                    "action_shape": list(sample["action"].shape),
                    "video_shapes": [list(video.shape) for video in sample["videos"]],
                },
                indent=2,
            )
        )
        return

    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=False,
    )
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(args.device)
    model.eval()
    attach_stage_adaln(model)
    optimizer = torch.optim.AdamW(
        model.stage_adaln.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    action_config = processor.action_config["robocasa365"]
    mean = action_config["mean"].to(args.device)
    std = action_config["std"].to(args.device)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for step in range(1, args.steps + 1):
        sample = dataset.get(random.randrange(len(dataset)))
        text = full_prompt(processor, sample["task_instruction"])
        inputs = processor(
            videos=sample["videos"],
            text=text,
            return_tensors="pt",
            state=sample["state"],
            robot_type="robocasa365",
        )
        inputs = {
            key: value.to(args.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        target = (torch.from_numpy(sample["action"])[None].to(args.device) - mean) / std.clamp_min(1e-6)
        stage_hidden = encode_text_condition(
            model,
            processor.tokenizer,
            [sample["stage_text"]],
            args.device,
        ).to(dtype=target.dtype)
        prediction, velocity = prepare_action_forward(
            model,
            inputs,
            target.to(dtype=torch.bfloat16),
            stage_hidden.to(dtype=torch.bfloat16),
        )
        mask = inputs["action_mask"].expand_as(prediction)
        loss = ((prediction.float() - velocity.float()) ** 2 * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.stage_adaln.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % 50 == 0:
            print(f"step={step}/{args.steps} loss={loss.item():.6f}")
        if step % args.save_every == 0 or step == args.steps:
            checkpoint = output_root / f"checkpoint-{step}.pt"
            torch.save(
                {
                    "adapter": adapter_state_dict(model),
                    "step": step,
                    "manifest": str(manifest_path),
                    "model_path": args.model_path,
                    "args": vars(args),
                },
                checkpoint,
            )
            print(f"saved {checkpoint}")


if __name__ == "__main__":
    main()
