#!/usr/bin/env python3
"""Train a baseline-aware stage residual adapter.

The frozen Xiaomi policy is evaluated and the stage-conditioned policy is
evaluated on the same noisy diffusion example.  The adapter is penalized when
it is worse than the baseline and regularized to keep its residual small.
This makes "stage information is useful" an offline action-space criterion.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.long_horizon_stage_adaln.benefit_model import (
    STAGE_KIND_ORDER,
    adapter_state_dict,
    attach_benefit_adaln,
    load_adapter_state,
    stage_kind_index,
)
from scripts.long_horizon_stage_adaln.model import encode_text_condition
from scripts.long_horizon_stage_adaln.train_adapter import (
    RoboCasaStageDataset,
    full_prompt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=(
            "/data/zjw/workspace/Isaac-GR00T/expdata/"
            "long_horizon_stage_adaln/target_composite_manifest.json"
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
            "long_horizon_stage_adaln/benefit_adapter_target_composite"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--history-frames", type=int, default=4)
    parser.add_argument("--history-interval-frames", type=int, default=2)
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--max-samples-per-episode", type=int, default=128)
    parser.add_argument("--max-video-size", type=int, default=224)
    parser.add_argument(
        "--stage-condition-format",
        choices=["full", "subtask_only"],
        default="subtask_only",
    )
    parser.add_argument("--non-degradation-weight", type=float, default=2.0)
    parser.add_argument("--residual-weight", type=float, default=0.10)
    parser.add_argument("--gate-regularization", type=float, default=0.001)
    parser.add_argument("--stage-dropout", type=float, default=0.15)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_context(model: Any, inputs: dict[str, Any]) -> dict[str, Any]:
    vlm_inputs = {
        key: value
        for key, value in inputs.items()
        if key not in {"state", "action_mask", "action"}
    }
    with torch.no_grad():
        vlm_outputs = model.vlm(**vlm_inputs, use_cache=True)
    action_mask = inputs["action_mask"]
    state = inputs["state"]
    action_length = inputs["action"].shape[1]
    state_length = state.shape[1]
    batch_size = action_length * 0 + state.shape[0]
    query_length = action_length + state_length + 1
    position_ids = (
        torch.arange(0, query_length, device=action_mask.device)
        .view(1, 1, -1)
        .repeat(3, batch_size, 1)
        + vlm_outputs.position_ids.max(dim=-1)[0][..., None]
        + 1
    )
    position_embeds = model.rotary_emb(action_mask, position_ids)
    dit_mask = torch.tril(
        torch.ones(
            (batch_size, query_length, query_length),
            device=action_mask.device,
        ),
        diagonal=0,
    )
    cache_mask = vlm_outputs.attention_mask[:, None, :].expand(
        -1,
        query_length,
        -1,
    )
    attn_mask = torch.cat([cache_mask, dit_mask], dim=-1)[:, None].bool()
    with torch.no_grad():
        state_embed = model.state_projector(state)
    return {
        "action_mask": action_mask,
        "state_embed": state_embed,
        "position_embeds": position_embeds,
        "past_key_values": vlm_outputs.past_key_values,
        "attn_mask": attn_mask,
    }


def forward_at(
    model: Any,
    context: dict[str, Any],
    noisy: torch.Tensor,
    timestep: torch.Tensor,
    stage_hidden: torch.Tensor | None,
    stage_ids: torch.Tensor | None,
) -> torch.Tensor:
    model._benefit_stage_condition = stage_hidden
    model._benefit_stage_kind_ids = stage_ids
    try:
        return model.dit_forward(
            noisy_action=noisy,
            t=timestep,
            action_mask=context["action_mask"],
            state_embed=context["state_embed"],
            position_embeds=context["position_embeds"],
            past_key_values=context["past_key_values"],
            attn_mask=context["attn_mask"],
        )
    finally:
        model._benefit_stage_condition = None
        model._benefit_stage_kind_ids = None


def masked_mse(value: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.expand_as(value).float()
    return ((value.float() - target.float()).pow(2) * expanded).sum() / expanded.sum().clamp_min(1.0)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset = RoboCasaStageDataset(manifest, args)
    print(f"benefit-adapter samples={len(dataset)} datasets={len(dataset.datasets)}")
    if args.dry_run:
        sample = dataset.get(0)
        print(json.dumps({
            "task_name": sample["task_name"],
            "stage_name": sample["stage_name"],
            "stage_text": sample["stage_text"],
            "state_shape": list(sample["state"].shape),
            "action_shape": list(sample["action"].shape),
        }, indent=2))
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
    attach_benefit_adaln(model)
    optimizer = torch.optim.AdamW(
        model.stage_adaln.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    action_config = processor.action_config["robocasa365"]
    mean = action_config["mean"].to(args.device)
    std = action_config["std"].to(args.device).clamp_min(1e-6)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for step in range(1, args.steps + 1):
        sample = dataset.get(random.randrange(len(dataset)))
        inputs = processor(
            videos=sample["videos"],
            text=full_prompt(processor, sample["task_instruction"]),
            return_tensors="pt",
            state=sample["state"],
            robot_type="robocasa365",
        )
        inputs = {
            key: value.to(args.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        target = (
            torch.from_numpy(sample["action"])[None].to(args.device) - mean
        ) / std
        target = target.to(dtype=torch.bfloat16)
        inputs["action"] = target
        stage_hidden = encode_text_condition(
            model,
            processor.tokenizer,
            [sample["stage_text"]],
            args.device,
        ).to(dtype=target.dtype)
        stage_ids = stage_kind_index(sample["stage_name"], args.device)
        if random.random() < args.stage_dropout:
            stage_hidden = None

        context = build_context(model, inputs)
        noise = torch.randn_like(target)
        timestep = torch.rand(
            (target.shape[0], 1, 1),
            device=target.device,
            dtype=target.dtype,
        )
        noisy = (1.0 - timestep) * noise + timestep * target
        velocity = target - noise

        with torch.no_grad():
            base_prediction = forward_at(
                model, context, noisy, timestep, None, None
            )
            base_error = masked_mse(
                base_prediction,
                velocity,
                context["action_mask"],
            )
        conditioned_prediction = forward_at(
            model,
            context,
            noisy,
            timestep,
            stage_hidden,
            stage_ids,
        )
        conditioned_error = masked_mse(
            conditioned_prediction,
            velocity,
            context["action_mask"],
        )
        residual_error = masked_mse(
            conditioned_prediction,
            base_prediction.detach(),
            context["action_mask"],
        )
        non_degradation = torch.relu(conditioned_error - base_error.detach())
        gate = torch.sigmoid(model.stage_adaln.stage_gate_logits).mean()
        loss = (
            conditioned_error
            + args.non_degradation_weight * non_degradation
            + args.residual_weight * residual_error
            + args.gate_regularization * gate
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.stage_adaln.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % 50 == 0:
            improvement = float((base_error - conditioned_error).detach().cpu())
            print(
                f"step={step}/{args.steps} loss={loss.item():.6f} "
                f"base={base_error.item():.6f} cond={conditioned_error.item():.6f} "
                f"improvement={improvement:.6f} gate={gate.item():.4f}",
                flush=True,
            )
        if step % args.save_every == 0 or step == args.steps:
            checkpoint = output_root / f"checkpoint-{step}.pt"
            torch.save(
                {
                    "adapter": adapter_state_dict(model),
                    "step": step,
                    "manifest": str(manifest_path),
                    "model_path": args.model_path,
                    "stage_kind_order": STAGE_KIND_ORDER,
                    "args": vars(args),
                },
                checkpoint,
            )
            print(f"saved {checkpoint}", flush=True)


if __name__ == "__main__":
    main()
