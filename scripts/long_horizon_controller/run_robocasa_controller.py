#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import tyro

from .controller import ControllerConfig, LongHorizonController
from .fast_monitor import ActionEntropyMonitor, AuxHeadFusionMonitor
from .planner import OllamaPlanner, OpenAICompatiblePlanner, StaticPlanner
from .policy_adapters import Gr00tPolicyAdapter
from .robocasa_adapter import RobocasaVectorEnvAdapter
from .schemas import FastMonitorConfig, VLMStatus, plan_from_dict
from .vlm_verifier import DEFAULT_QWEN3_VL_PATH, DryRunVerifier, LocalQwenVLVerifier, OllamaVLVerifier


@dataclass
class Args:
    task: str
    env_name: str
    plan_json_path: str = ""
    model_path: str = (
        "/data/zjw/workspace/Isaac-GR00T/expdata/foundation_model_learning/"
        "target_posttraining/composite_seen/checkpoint-60000"
    )
    aux_head_path: str = (
        "/data/zjw/workspace/Isaac-GR00T/expdata/aux_progress/"
        "atomic_retry_3class_history_run1/checkpoint-8000"
    )
    output_dir: str = "/data/zjw/workspace/Isaac-GR00T/expdata/long_horizon_controller/robocasa_run"
    split: str = "target"
    data_config: str = "panda_omron"
    embodiment_tag: str = "new_embodiment"
    planner: str = "api"  # api | ollama | static
    verifier: str = "dry"
    dry_vlm_status: str = "complete"
    vlm_model_path: str = DEFAULT_QWEN3_VL_PATH
    vlm_ollama_model: str = "qwen3-vl:8b"
    vlm_ollama_base_url: str = "http://localhost:11434"
    vlm_ollama_keep_alive: str = "30m"
    vlm_timeout_sec: float = 120.0
    vlm_num_predict: int = 1024
    vlm_history_frames: int = 2
    ollama_model: str = "llama3.1:70b"
    ollama_base_url: str = "http://localhost:11434"
    llm_timeout_sec: float = 180.0
    ollama_num_predict: int = 512
    ollama_num_gpu: int = 33
    vlm_image_key: str = (
        "video.robot0_agentview_left,video.robot0_eye_in_hand"
    )
    n_action_steps: int = 16
    max_episode_steps: int = 1440
    step_dt_sec: float = 0.05
    complete_score_threshold: float = 0.35
    min_steps_before_trigger: int = 8
    cooldown_steps: int = 8
    aux_retry_confidence_threshold: float = 0.6
    aux_success_confidence_threshold: float = 0.7
    aux_cooldown_steps: int = 8


def _make_planner(args: Args):
    if args.planner == "api":
        return OpenAICompatiblePlanner(timeout_sec=args.llm_timeout_sec)
    if args.planner == "ollama":
        return OllamaPlanner(
            model=args.ollama_model,
            base_url=args.ollama_base_url,
            timeout_sec=args.llm_timeout_sec,
            num_predict=args.ollama_num_predict,
            num_gpu=args.ollama_num_gpu,
        )
    if args.planner == "static":
        return StaticPlanner(max_duration_sec=30.0)
    raise ValueError(f"Unsupported planner: {args.planner}")


def _make_verifier(args: Args):
    if args.verifier == "qwen_vl":
        return LocalQwenVLVerifier(model_path=args.vlm_model_path)
    if args.verifier == "ollama_vl":
        return OllamaVLVerifier(
            model=args.vlm_ollama_model,
            base_url=args.vlm_ollama_base_url,
            timeout_sec=args.vlm_timeout_sec,
            num_predict=args.vlm_num_predict,
            keep_alive=args.vlm_ollama_keep_alive,
        )
    if args.verifier == "dry":
        return DryRunVerifier(default_status=VLMStatus(args.dry_vlm_status))
    raise ValueError(f"Unsupported verifier: {args.verifier}")


def main(args: Args) -> None:
    from gr00t.eval.simulation import MultiStepConfig, SimulationConfig, VideoConfig
    from gr00t.experiment.data_config import DATA_CONFIG_MAP
    from gr00t.model.policy import Gr00tPolicy

    if args.plan_json_path:
        plan = plan_from_dict(json.loads(Path(args.plan_json_path).read_text(encoding="utf-8")))
    else:
        planner = _make_planner(args)
        plan = planner.plan(args.task)
    plan.save(f"{args.output_dir}/plan.json")

    data_config = DATA_CONFIG_MAP[args.data_config]
    modality_config = data_config.modality_config()
    policy = Gr00tPolicy(
        model_path=args.model_path,
        modality_config=modality_config,
        modality_transform=data_config.transform(),
        embodiment_tag=args.embodiment_tag,
        denoising_steps=4,
    )
    policy_adapter = Gr00tPolicyAdapter(
        policy=policy,
        action_keys=modality_config["action"].modality_keys,
        aux_head_path=args.aux_head_path,
    )

    env_name = args.env_name if args.env_name.startswith("robocasa/") else f"robocasa/{args.env_name}"
    simulation_config = SimulationConfig(
        env_name=env_name,
        split=args.split,
        n_episodes=1,
        n_envs=1,
        video=VideoConfig(video_dir=f"{args.output_dir}/videos"),
        multistep=MultiStepConfig(
            n_action_steps=args.n_action_steps,
            max_episode_steps=args.max_episode_steps,
        ),
    )
    env = RobocasaVectorEnvAdapter(
        simulation_config=simulation_config,
        vlm_image_key=args.vlm_image_key,
    )
    try:
        controller = LongHorizonController(
            plan=plan,
            policy=policy_adapter,
            env=env,
            fast_monitor=ActionEntropyMonitor(
                FastMonitorConfig(
                    complete_score_threshold=args.complete_score_threshold,
                    min_steps_before_trigger=args.min_steps_before_trigger,
                    cooldown_steps=args.cooldown_steps,
                )
            ),
            aux_fusion=AuxHeadFusionMonitor(
                retry_confidence_threshold=args.aux_retry_confidence_threshold,
                success_confidence_threshold=args.aux_success_confidence_threshold,
                cooldown_steps=args.aux_cooldown_steps,
            ),
            vlm_verifier=_make_verifier(args),
            config=ControllerConfig(
                output_dir=args.output_dir,
                max_total_steps=args.max_episode_steps,
                step_dt_sec=args.step_dt_sec,
                vlm_history_frames=args.vlm_history_frames,
            ),
        )
        controller.run()
    finally:
        env.close()
    print(f"Saved long-horizon controller outputs to {args.output_dir}")


if __name__ == "__main__":
    main(tyro.cli(Args))
