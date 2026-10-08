#!/usr/bin/env python3
"""Runtime for a Xiaomi Robotics-1 auxiliary progress/state head."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn


STATE_CLASS_NAMES = ("progress", "success", "retry")


def masked_mean(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.to(dtype=hidden.dtype).unsqueeze(-1)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


class XiaomiAuxiliaryHeads(nn.Module):
    """Small heads consuming the frozen Xiaomi VLM pooled representation."""

    def __init__(
        self,
        input_dim: int = 2560,
        hidden_dim: int = 1280,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.progress_head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.state_head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, pooled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.progress_head(pooled).squeeze(-1),
            self.state_head(pooled),
        )


class XiaomiAuxHeadRuntime:
    """Load and run heads on VLM outputs from the current Xiaomi action query."""

    def __init__(self, model: Any, aux_head_path: str = "") -> None:
        self.model = model
        self.aux_head_path = str(aux_head_path)
        self.enabled = bool(self.aux_head_path)
        self.config: dict[str, Any] = {}
        self.heads: XiaomiAuxiliaryHeads | None = None
        if self.enabled:
            self._load(self.aux_head_path)

    def _load(self, aux_head_path: str) -> None:
        path = Path(aux_head_path)
        config_path = path / "aux_config.json"
        weights_path = path / "aux_heads.pt"
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing Xiaomi auxiliary config: {config_path}")
        if not weights_path.is_file():
            raise FileNotFoundError(f"Missing Xiaomi auxiliary weights: {weights_path}")

        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        input_dim = int(self.config.get("feature_dim", 2560))
        hidden_dim = int(self.config.get("head_hidden_dim", max(input_dim // 2, 1)))
        state_classes = self.config.get("state_classes", list(STATE_CLASS_NAMES))
        if tuple(state_classes) != STATE_CLASS_NAMES:
            raise ValueError(
                "Xiaomi auxiliary runtime expects state classes "
                f"{list(STATE_CLASS_NAMES)}, got {state_classes!r}"
            )
        self.heads = XiaomiAuxiliaryHeads(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=len(state_classes),
        )
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        if "heads" in state:
            state = state["heads"]
        self.heads.load_state_dict(state)
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        self.heads.to(device=device, dtype=dtype).eval()

    def predict(
        self,
        vlm_outputs: Any,
        attention_mask: torch.Tensor,
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        if self.heads is None:
            raise RuntimeError("Xiaomi auxiliary heads were not loaded.")
        hidden = vlm_outputs.last_hidden_state
        pooled = masked_mean(hidden, attention_mask)
        with torch.inference_mode():
            progress, logits = self.heads(pooled)
            probs = torch.softmax(logits, dim=-1)

        confidence, state_index = probs[0].max(dim=-1)
        state_name = STATE_CLASS_NAMES[int(state_index.detach().cpu())]
        values = probs[0].detach().float().cpu().tolist()
        return {
            "state": state_name,
            "confidence": float(confidence.detach().float().cpu()),
            "progress": float(progress[0].detach().float().cpu()),
            "probs": {
                name: float(values[index])
                for index, name in enumerate(STATE_CLASS_NAMES)
            },
        }

