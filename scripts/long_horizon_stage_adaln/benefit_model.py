#!/usr/bin/env python3
"""Baseline-aware stage residual adapter.

This variant keeps the Xiaomi policy as the reference behavior.  The stage
condition is a bounded residual on the DiT timestep embedding, and its
learned gate is indexed by the semantic RoboCasa stage type.
"""

from __future__ import annotations

from types import MethodType
from typing import Any

import torch
from torch import nn


STAGE_KIND_ORDER = (
    "pick",
    "place",
    "navigate",
    "execute",
    "open",
    "close",
    "pour",
    "turn_on",
    "unknown",
)
STAGE_KIND_TO_INDEX = {name: index for index, name in enumerate(STAGE_KIND_ORDER)}


def normalize_stage_kind(value: str | None) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    if text in STAGE_KIND_TO_INDEX:
        return text
    aliases = {
        "navigation": "navigate",
        "move": "navigate",
        "moving": "navigate",
        "grasp": "pick",
        "pickup": "pick",
        "placing": "place",
        "put": "place",
        "interaction": "execute",
        "interact": "execute",
        "activate": "turn_on",
        "switch_on": "turn_on",
    }
    return aliases.get(text, "unknown")


def stage_kind_index(value: str | None, device: str | torch.device) -> torch.Tensor:
    return torch.tensor(
        [STAGE_KIND_TO_INDEX[normalize_stage_kind(value)]],
        dtype=torch.long,
        device=device,
    )


class BenefitStageAdaLNAdapter(nn.Module):
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
        # Zero output gives an exact Xiaomi baseline at initialization.
        nn.init.zeros_(self.net[-1].weight)
        # The gate starts nearly closed.  It can open only when the
        # baseline-aware action loss finds a useful residual.
        self.stage_gate_logits = nn.Parameter(
            torch.full((len(STAGE_KIND_ORDER), 6, 1), -5.0)
        )

    def forward(
        self,
        hidden: torch.Tensor,
        stage_kind_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dtype = hidden.dtype
        hidden = hidden.float()
        delta = self.net(hidden).view(hidden.shape[0], 6, self.dit_hidden_size)
        delta = delta.to(dtype)
        if stage_kind_ids is None:
            stage_kind_ids = torch.full(
                (hidden.shape[0],),
                STAGE_KIND_TO_INDEX["unknown"],
                dtype=torch.long,
                device=hidden.device,
            )
        stage_kind_ids = stage_kind_ids.reshape(-1).to(device=hidden.device)
        if stage_kind_ids.numel() == 1 and hidden.shape[0] != 1:
            stage_kind_ids = stage_kind_ids.expand(hidden.shape[0])
        gate = torch.sigmoid(self.stage_gate_logits[stage_kind_ids]).to(delta.dtype)
        return delta * gate


def _benefit_dit_forward(
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
    stage_hidden = getattr(self, "_benefit_stage_condition", None)
    if stage_hidden is not None:
        stage_ids = getattr(self, "_benefit_stage_kind_ids", None)
        residual = self.stage_adaln(stage_hidden, stage_ids)
        scale = float(getattr(self, "_benefit_condition_scale", 1.0))
        # Match the residual RMS to the timestep embedding, then apply a
        # small external scale selected by offline calibration.
        t_rms = t_embeds.float().pow(2).mean(dim=(-1, -2), keepdim=True).sqrt()
        residual_rms = (
            residual.float().pow(2).mean(dim=(-1, -2), keepdim=True).sqrt().detach()
        )
        # Keep the residual normalization bounded at the zero-initialized
        # point; otherwise the first gradient can be amplified by 1e5.
        residual_denom = torch.maximum(residual_rms, 0.1 * t_rms)
        residual = residual * (t_rms / residual_denom).to(residual.dtype)
        t_embeds = t_embeds + scale * residual

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


def attach_benefit_adaln(
    model: Any,
    text_hidden_size: int = 2560,
    dit_hidden_size: int = 1024,
    bottleneck: int = 1024,
) -> Any:
    model.stage_adaln = BenefitStageAdaLNAdapter(
        text_hidden_size=text_hidden_size,
        dit_hidden_size=dit_hidden_size,
        bottleneck=bottleneck,
    ).to(next(model.parameters()).device)
    model.dit_forward = MethodType(_benefit_dit_forward, model)
    model._benefit_stage_condition = None
    model._benefit_stage_kind_ids = None
    model._benefit_condition_scale = 1.0
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
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    args = payload.get("args", {})
    return args if isinstance(args, dict) else {}
