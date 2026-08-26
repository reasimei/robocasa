#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class PolicyAdapter(Protocol):
    def act(self, observation: dict[str, Any], instruction: str) -> tuple[dict[str, Any], np.ndarray]:
        """Return environment action dict and action chunk for fast monitoring."""


def concat_action_dict(action: dict[str, Any], action_keys: list[str]) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for key in action_keys:
        value = action[key]
        try:
            import torch

            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()
        except ImportError:
            pass
        array = np.asarray(value, dtype=np.float32)
        if array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        if array.ndim == 1:
            array = array[None, :]
        chunks.append(array)
    return np.concatenate(chunks, axis=-1)


@dataclass
class Gr00tPolicyAdapter:
    policy: Any
    action_keys: list[str]
    annotation_key: str = "annotation.human.task_description"
    aux_head_path: str = ""

    def __post_init__(self) -> None:
        self.aux_head = Gr00tAuxHeadRuntime(self.policy, self.annotation_key, self.aux_head_path)
        self.last_aux_output: dict[str, Any] | None = None

    def act(self, observation: dict[str, Any], instruction: str) -> tuple[dict[str, Any], np.ndarray]:
        obs = dict(observation)
        obs[self.annotation_key] = [instruction]
        action = self.policy.get_action(obs)
        action_chunk = concat_action_dict(action, self.action_keys)
        self.last_aux_output = self.aux_head.predict(obs) if self.aux_head.enabled else None
        return action, action_chunk


class Gr00tAuxHeadRuntime:
    """Runs the frozen auxiliary state/progress heads on the live GR00T backbone."""

    def __init__(self, policy: Any, annotation_key: str, aux_head_path: str = ""):
        self.policy = policy
        self.annotation_key = annotation_key
        self.aux_head_path = aux_head_path
        self.enabled = bool(aux_head_path)
        self.config: dict[str, Any] = {}
        self.state_classes: list[str] = ["progress", "success", "retry"]
        self.progress_head = None
        self.state_head = None
        if self.enabled:
            self._load(aux_head_path)

    def _load(self, aux_head_path: str) -> None:
        import torch
        from torch import nn

        path = Path(aux_head_path)
        config_path = path / "aux_config.json"
        weights_path = path / "aux_heads.pt"
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing auxiliary config: {config_path}")
        if not weights_path.is_file():
            raise FileNotFoundError(f"Missing auxiliary checkpoint: {weights_path}")

        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.state_classes = [str(item) for item in self.config.get("state_classes", self.state_classes)]
        hidden_size = int(self.config.get("hidden_size", self.policy.model.config.hidden_size))
        head_input_dim = int(self.config.get("head_input_dim", hidden_size))
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
            nn.Linear(hidden_size // 2, len(self.state_classes)),
        )
        state = torch.load(weights_path, map_location="cpu")
        self.progress_head.load_state_dict(
            {key.removeprefix("progress_head."): value for key, value in state.items() if key.startswith("progress_head.")}
        )
        self.state_head.load_state_dict(
            {key.removeprefix("state_head."): value for key, value in state.items() if key.startswith("state_head.")}
        )
        device = next(self.policy.model.parameters()).device
        dtype = next(self.policy.model.parameters()).dtype
        self.progress_head.to(device=device, dtype=dtype).eval()
        self.state_head.to(device=device, dtype=dtype).eval()

    def predict(self, observation: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled:
            return {}
        import torch

        assert self.progress_head is not None and self.state_head is not None
        obs = dict(observation)
        is_batch = self.policy._check_state_is_batched(obs)
        if not is_batch:
            from gr00t.model.policy import unsqueeze_dict_values

            obs = unsqueeze_dict_values(obs)
        for key, value in obs.items():
            if not isinstance(value, np.ndarray):
                obs[key] = np.array(value)

        normalized_input = self.policy.apply_transforms(obs)
        device = next(self.policy.model.parameters()).device
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=next(self.policy.model.parameters()).dtype):
            backbone_inputs, _ = self.policy.model.prepare_input(normalized_input)
            backbone_outputs = self.policy.model.backbone(backbone_inputs)
            features = backbone_outputs["backbone_features"]
            mask = backbone_outputs["backbone_attention_mask"].to(dtype=features.dtype).unsqueeze(-1)
            pooled = (features * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            pooled = self._append_aux_context(normalized_input, pooled, device)
            progress = self.progress_head(pooled).squeeze(-1)
            logits = self.state_head(pooled)
            probs = torch.softmax(logits, dim=-1)

        confidence, state_idx = probs[0].max(dim=-1)
        state_name = self.state_classes[int(state_idx.detach().cpu())]
        prob_values = probs[0].detach().float().cpu().numpy().tolist()
        return {
            "state": state_name,
            "confidence": float(confidence.detach().float().cpu()),
            "progress": float(progress[0].detach().float().cpu()),
            "probs": {
                name: float(prob_values[idx])
                for idx, name in enumerate(self.state_classes)
            },
        }

    def _append_aux_context(self, normalized_input: dict[str, Any], pooled: Any, device: Any) -> Any:
        import torch

        mode = str(self.config.get("aux_context_mode", "none"))
        if mode == "none":
            return pooled
        state = normalized_input["state"].to(device)
        state_mask = normalized_input.get("state_mask")
        if state_mask is not None:
            state = state * state_mask.to(device=device, dtype=state.dtype)
        parts = [state[:, -1], state[:, -1] - state[:, 0]]
        if mode == "state_action_delta":
            action = normalized_input["action"].to(device)
            action_mask = normalized_input.get("action_mask")
            if action_mask is not None:
                action = action * action_mask.to(device=device, dtype=action.dtype)
            parts.extend([action[:, 0], action[:, -1] - action[:, 0]])
        elif mode != "state_delta":
            raise ValueError(f"Unsupported aux_context_mode={mode!r}")
        return torch.cat([pooled, *[part.to(dtype=pooled.dtype) for part in parts]], dim=-1)


@dataclass
class MockPolicyAdapter:
    action_dim: int = 12
    horizon: int = 16
    trigger_step: int = 6
    trigger_scale: float = 0.25

    def __post_init__(self) -> None:
        self.step = 0

    def act(self, observation: dict[str, Any], instruction: str) -> tuple[dict[str, Any], np.ndarray]:
        del observation, instruction
        self.step += 1
        chunk = np.zeros((self.horizon, self.action_dim), dtype=np.float32)
        chunk[:, 0] = np.sin((np.arange(self.horizon) + self.step) / 5.0)
        if self.step == self.trigger_step:
            chunk[:, 1] = self.trigger_scale
        return {"mock_action": chunk}, chunk
