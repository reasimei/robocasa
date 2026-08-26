"""Xiaomi RoboCasa365 policy with a trained stage-text AdaLN adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from scripts.long_horizon_controller.xiaomi_policy_adapter import XiaomiPolicyAdapter
from scripts.long_horizon_stage_adaln.model import (
    attach_stage_adaln,
    encode_text_condition,
    load_adapter_state,
)


@dataclass
class StageAdaLNXiaomiPolicyAdapter(XiaomiPolicyAdapter):
    adapter_checkpoint: str = ""
    condition_scale: float = 1.0

    def __post_init__(self) -> None:
        if not self.adapter_checkpoint:
            raise ValueError("adapter_checkpoint is required for stage AdaLN evaluation.")
        super().__post_init__()
        attach_stage_adaln(self.model)
        load_adapter_state(self.model, self.adapter_checkpoint)
        if self.condition_scale < 0.0:
            raise ValueError("condition_scale must be non-negative")
        self.model._stage_adaln_condition_scale = float(self.condition_scale)
        self.model.eval()

    def act(
        self,
        observation: dict[str, Any],
        instruction: str,
        stage_text: str,
        stage_kind: str | None = None,
    ):
        # Keep scale=0 as an exact baseline path. This avoids an extra text
        # forward and bypasses the patched DiT branch during diagnostics.
        if self.condition_scale == 0.0:
            return super().act(observation, instruction)

        stage_hidden = encode_text_condition(
            self.model,
            self.processor.tokenizer,
            [stage_text],
            self.device,
        ).to(dtype=self.dtype)
        self.model._stage_adaln_condition = stage_hidden
        try:
            return super().act(observation, instruction)
        finally:
            self.model._stage_adaln_condition = None
