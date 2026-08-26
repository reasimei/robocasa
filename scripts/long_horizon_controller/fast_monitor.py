#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .schemas import FastMonitorConfig, FastSignal, FastTrigger


def action_chunk_rmse(prev_chunk: np.ndarray, curr_chunk: np.ndarray, step_delta: int) -> float | None:
    step_delta = int(step_delta)
    if step_delta <= 0:
        return None
    horizon = min(prev_chunk.shape[0], curr_chunk.shape[0])
    if step_delta >= horizon:
        return None
    prev_overlap = prev_chunk[step_delta:horizon]
    curr_overlap = curr_chunk[: horizon - step_delta]
    return float(np.sqrt(np.mean((prev_overlap - curr_overlap) ** 2)))


@dataclass
class ActionEntropyMonitor:
    config: FastMonitorConfig

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.prev_chunk: np.ndarray | None = None
        self.step_count = 0
        self.cooldown = 0
        self.ema_score: float | None = None

    def update(
        self,
        action_chunk: np.ndarray,
        elapsed_sec: float,
        timeout_sec: float,
        step_delta: int | None = None,
    ) -> FastSignal:
        self.step_count += 1
        step_delta = self.config.action_step_delta if step_delta is None else int(step_delta)
        score = None
        if self.prev_chunk is not None:
            score = action_chunk_rmse(self.prev_chunk, action_chunk, step_delta)
            if score is not None:
                if self.ema_score is None:
                    self.ema_score = score
                else:
                    alpha = float(self.config.ema_alpha)
                    self.ema_score = alpha * score + (1.0 - alpha) * self.ema_score
        self.prev_chunk = np.asarray(action_chunk, dtype=np.float32)

        if elapsed_sec >= timeout_sec:
            return FastSignal(
                trigger=FastTrigger.TIMEOUT,
                score=self.ema_score,
                elapsed_sec=elapsed_sec,
                reason=f"elapsed_sec={elapsed_sec:.2f} >= timeout_sec={timeout_sec:.2f}",
            )

        if self.cooldown > 0:
            self.cooldown -= 1
            return FastSignal(
                trigger=FastTrigger.NONE,
                score=self.ema_score,
                elapsed_sec=elapsed_sec,
                reason="cooldown",
            )

        if self.step_count < self.config.min_steps_before_trigger or self.ema_score is None:
            return FastSignal(
                trigger=FastTrigger.NONE,
                score=self.ema_score,
                elapsed_sec=elapsed_sec,
                reason="warming_up",
            )

        if self.ema_score >= self.config.complete_score_threshold:
            self.cooldown = self.config.cooldown_steps
            return FastSignal(
                trigger=FastTrigger.SUSPECT_TRANSITION,
                score=self.ema_score,
                elapsed_sec=elapsed_sec,
                reason=(
                    f"chunk consistency score {self.ema_score:.4f} >= transition threshold; "
                    "VLM must decide success vs retry"
                ),
            )

        return FastSignal(
            trigger=FastTrigger.NONE,
            score=self.ema_score,
            elapsed_sec=elapsed_sec,
            reason="progress: chunk consistency below transition threshold",
        )


@dataclass
class AuxHeadFusionMonitor:
    """Optional fusion layer for auxiliary state-head output."""

    retry_confidence_threshold: float = 0.6
    success_confidence_threshold: float = 0.7
    cooldown_steps: int = 8

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.cooldown = 0

    def fuse(self, entropy_signal: FastSignal, aux_output: dict[str, Any] | None = None) -> FastSignal:
        if self.cooldown > 0:
            self.cooldown -= 1
            return entropy_signal
        if not aux_output:
            return entropy_signal

        state = str(aux_output.get("state", ""))
        confidence = aux_output.get("confidence")
        confidence_f = float(confidence) if confidence is not None else None
        if state == "retry" and confidence_f is not None and confidence_f >= self.retry_confidence_threshold:
            self.cooldown = self.cooldown_steps
            return FastSignal(
                trigger=FastTrigger.SUSPECT_FAIL,
                score=entropy_signal.score,
                aux_state=state,
                aux_confidence=confidence_f,
                elapsed_sec=entropy_signal.elapsed_sec,
                reason="auxiliary head predicts retry",
            )
        if state == "success" and confidence_f is not None and confidence_f >= self.success_confidence_threshold:
            self.cooldown = self.cooldown_steps
            return FastSignal(
                trigger=FastTrigger.SUSPECT_COMPLETE,
                score=entropy_signal.score,
                aux_state=state,
                aux_confidence=confidence_f,
                elapsed_sec=entropy_signal.elapsed_sec,
                reason="auxiliary head predicts success",
            )
        return entropy_signal
