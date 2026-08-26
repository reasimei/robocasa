#!/usr/bin/env python3
"""Stage-text AdaLN adapter for Xiaomi RoboCasa365.

The base Xiaomi checkpoint is loaded unchanged.  The adapter is attached at
runtime and only its parameters are saved, keeping this experiment isolated
from the original checkpoint and evaluator.
"""

from __future__ import annotations

from types import MethodType
from typing import Any

import torch
from torch import nn


class StageAdaLNAdapter(nn.Module):
    def __init__(
        self,
        text_hidden_size: int = 2560,
        dit_hidden_size: int = 1024,
        bottleneck: int = 1024,
    ) -> None:
        super().__init__()
        self.text_hidden_size = text_hidden_size
        self.dit_hidden_size = dit_hidden_size
        self.net = nn.Sequential(
            nn.LayerNorm(text_hidden_size),
            nn.Linear(text_hidden_size, bottleneck, bias=False),
            nn.GELU(approximate="tanh"),
            nn.Linear(bottleneck, 6 * dit_hidden_size, bias=False),
        )
        # Separate gates for shift/scale/residual-gate modulation groups.
        # sigmoid(-4) keeps the initial stage condition small while allowing
        # gradients to reach the gate during training.
        self.stage_gate_logits = nn.Parameter(torch.full((1, 6, 1), -4.0))
        # Exact baseline at initialization.  The last layer becomes trainable
        # immediately, while the frozen Xiaomi model is never modified.
        nn.init.zeros_(self.net[-1].weight)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        dtype = hidden.dtype
        hidden = hidden.float()
        delta = self.net(hidden).view(hidden.shape[0], 6, self.dit_hidden_size)
        gate = torch.sigmoid(self.stage_gate_logits).to(delta.dtype)
        return (delta * gate).to(dtype)


def masked_mean(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    weights = attention_mask.to(hidden.dtype).unsqueeze(-1)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


@torch.no_grad()
def encode_text_condition(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    device: str,
) -> torch.Tensor:
    tokens = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    )
    tokens = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in tokens.items()
    }
    output = model.vlm.language_model(
        input_ids=tokens["input_ids"],
        attention_mask=tokens["attention_mask"],
        use_cache=False,
    )
    return masked_mean(output.last_hidden_state, tokens["attention_mask"])


def _stage_dit_forward(
    self: Any,
    noisy_action: torch.Tensor,
    t: torch.Tensor,
    action_mask: torch.Tensor,
    state_embed: torch.Tensor,
    position_embeds: torch.Tensor,
    past_key_values: Any,
    attn_mask: torch.Tensor,
) -> torch.Tensor:
    t_embeds = self.t_embedder(t[:, 0, 0] * 1000)
    t_embeds = self.t_projector(t_embeds).view(t_embeds.shape[0], 6, -1)
    stage_delta = getattr(self, "_stage_adaln_condition", None)
    if stage_delta is not None:
        condition_scale = float(getattr(self, "_stage_adaln_condition_scale", 1.0))
        t_embeds = t_embeds + condition_scale * self.stage_adaln(stage_delta)

    noisy_action = noisy_action * action_mask
    noisy_action = self.action_projector(noisy_action)
    sink = self.sink.weight[None].repeat(state_embed.shape[0], 1, 1)
    hidden_states = torch.cat([sink, state_embed, noisy_action], dim=1).contiguous()
    hidden_states = self.dit(
        hidden_states,
        past_key_values,
        attn_mask,
        position_embeds,
        t_embeds,
    )
    return self.action_output_layer(hidden_states[:, -noisy_action.shape[1] :, :])


def attach_stage_adaln(
    model: Any,
    text_hidden_size: int = 2560,
    dit_hidden_size: int = 1024,
    bottleneck: int = 1024,
) -> Any:
    model.stage_adaln = StageAdaLNAdapter(
        text_hidden_size=text_hidden_size,
        dit_hidden_size=dit_hidden_size,
        bottleneck=bottleneck,
    ).to(next(model.parameters()).device)
    model.dit_forward = MethodType(_stage_dit_forward, model)
    model._stage_adaln_condition = None
    model._stage_adaln_condition_scale = 1.0
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.stage_adaln.parameters():
        parameter.requires_grad_(True)
    return model


def adapter_state_dict(model: Any) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu()
        for key, value in model.stage_adaln.state_dict().items()
    }


def load_adapter_state(model: Any, checkpoint: str) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload.get("adapter", payload)
    model.stage_adaln.load_state_dict(state)


def adapter_training_args(checkpoint: str) -> dict[str, Any]:
    """Read the saved training arguments without constructing the base model."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    args = payload.get("args", {})
    return args if isinstance(args, dict) else {}
