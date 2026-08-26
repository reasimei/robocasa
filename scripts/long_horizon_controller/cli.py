#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass

import tyro

from .controller import ControllerConfig, LongHorizonController
from .fast_monitor import ActionEntropyMonitor, AuxHeadFusionMonitor
from .mock_env import MockEnvironment
from .planner import OllamaPlanner, OpenAICompatiblePlanner, StaticPlanner
from .policy_adapters import MockPolicyAdapter
from .schemas import FastMonitorConfig, VLMStatus
from .vlm_verifier import DEFAULT_QWEN3_VL_PATH, DryRunVerifier, LocalQwenVLVerifier, OllamaVLVerifier


@dataclass
class Args:
    task: str = "Pick the kettle from the counter and place it on the tray, then place the mug on the tray."
    output_dir: str = "/data/zjw/workspace/Isaac-GR00T/expdata/long_horizon_controller/dry_run"
    planner: str = "static"  # static | api | ollama
    verifier: str = "dry"  # dry | qwen_vl | ollama_vl
    dry_vlm_status: str = "complete"
    vlm_model_path: str = DEFAULT_QWEN3_VL_PATH
    vlm_ollama_model: str = "qwen3-vl:8b"
    vlm_ollama_base_url: str = "http://localhost:11434"
    vlm_ollama_keep_alive: str = "30m"
    vlm_timeout_sec: float = 120.0
    vlm_num_predict: int = 1024
    vlm_history_frames: int = 4
    ollama_model: str = "llama3.1:70b"
    ollama_base_url: str = "http://localhost:11434"
    llm_timeout_sec: float = 180.0
    ollama_num_predict: int = 512
    ollama_num_gpu: int = 33
    max_total_steps: int = 80
    max_static_subtask_sec: float = 1.0


def main(args: Args) -> None:
    if args.planner == "api":
        planner = OpenAICompatiblePlanner(timeout_sec=args.llm_timeout_sec)
    elif args.planner == "ollama":
        planner = OllamaPlanner(
            model=args.ollama_model,
            base_url=args.ollama_base_url,
            timeout_sec=args.llm_timeout_sec,
            num_predict=args.ollama_num_predict,
            num_gpu=args.ollama_num_gpu,
        )
    elif args.planner == "static":
        planner = StaticPlanner(max_duration_sec=args.max_static_subtask_sec)
    else:
        raise ValueError(f"Unsupported planner: {args.planner}")

    plan = planner.plan(args.task)
    plan.save(f"{args.output_dir}/plan.json")

    if args.verifier == "qwen_vl":
        verifier = LocalQwenVLVerifier(model_path=args.vlm_model_path)
    elif args.verifier == "ollama_vl":
        verifier = OllamaVLVerifier(
            model=args.vlm_ollama_model,
            base_url=args.vlm_ollama_base_url,
            timeout_sec=args.vlm_timeout_sec,
            num_predict=args.vlm_num_predict,
            keep_alive=args.vlm_ollama_keep_alive,
        )
    elif args.verifier == "dry":
        verifier = DryRunVerifier(default_status=VLMStatus(args.dry_vlm_status))
    else:
        raise ValueError(f"Unsupported verifier: {args.verifier}")

    controller = LongHorizonController(
        plan=plan,
        policy=MockPolicyAdapter(),
        env=MockEnvironment(),
        fast_monitor=ActionEntropyMonitor(
            FastMonitorConfig(
                complete_score_threshold=0.01,
                min_steps_before_trigger=3,
                cooldown_steps=2,
            )
        ),
        aux_fusion=AuxHeadFusionMonitor(),
        vlm_verifier=verifier,
        config=ControllerConfig(
            output_dir=args.output_dir,
            max_total_steps=args.max_total_steps,
            vlm_history_frames=args.vlm_history_frames,
        ),
    )
    controller.run()
    print(f"Saved dry-run plan and controller events to {args.output_dir}")


if __name__ == "__main__":
    main(tyro.cli(Args))
