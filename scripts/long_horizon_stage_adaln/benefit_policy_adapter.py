#!/usr/bin/env python3
"""Xiaomi policy with an offline-calibrated stage residual gate."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.long_horizon_controller.xiaomi_policy_adapter import XiaomiPolicyAdapter
from scripts.long_horizon_stage_adaln.benefit_model import (
    STAGE_KIND_TO_INDEX,
    attach_benefit_adaln,
    load_adapter_state,
    normalize_stage_kind,
    stage_kind_index,
)
from scripts.long_horizon_stage_adaln.model import encode_text_condition


@dataclass
class BenefitGatedXiaomiPolicyAdapter(XiaomiPolicyAdapter):
    adapter_checkpoint: str = ""
    gate_config: str = ""

    def __post_init__(self) -> None:
        if not self.adapter_checkpoint:
            raise ValueError("adapter_checkpoint is required.")
        super().__post_init__()
        attach_benefit_adaln(self.model)
        load_adapter_state(self.model, self.adapter_checkpoint)
        self.stage_scales = {
            stage: 0.0 for stage in STAGE_KIND_TO_INDEX
        }
        self.skill_scales: dict[str, float] = {}
        if self.gate_config:
            payload = json.loads(Path(self.gate_config).read_text(encoding="utf-8"))
            for stage, row in payload.get("stages", {}).items():
                self.stage_scales[normalize_stage_kind(stage)] = float(
                    row.get("recommended_scale", 0.0)
                )
            for skill, row in payload.get("skills", {}).items():
                if isinstance(row, dict):
                    scale = row.get("recommended_scale", 0.0)
                    status = row.get("status", "calibrated")
                    if status == "unseen":
                        continue
                else:
                    scale = row
                self.skill_scales[str(skill).strip().lower()] = float(scale)
        self.model.eval()

    def act(
        self,
        observation: dict[str, Any],
        instruction: str,
        stage_text: str,
        stage_kind: str | None = None,
        atomic_skill: str | None = None,
    ):
        normalized = normalize_stage_kind(stage_kind)
        skill_name = str(atomic_skill or "").strip().lower()
        if skill_name in self.skill_scales:
            scale = float(self.skill_scales[skill_name])
        else:
            scale = float(self.stage_scales.get(normalized, 0.0))
        if scale <= 0.0:
            return super().act(observation, instruction)
        stage_hidden = encode_text_condition(
            self.model,
            self.processor.tokenizer,
            [stage_text],
            self.device,
        ).to(dtype=self.dtype)
        self.model._benefit_stage_condition = stage_hidden
        self.model._benefit_stage_kind_ids = stage_kind_index(
            normalized,
            self.device,
        )
        self.model._benefit_condition_scale = scale
        try:
            return super().act(observation, instruction)
        finally:
            self.model._benefit_stage_condition = None
            self.model._benefit_stage_kind_ids = None
            self.model._benefit_condition_scale = 1.0
