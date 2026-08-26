#!/usr/bin/env python3
"""Calibrate stage-condition gates from held-out action-space evidence.

No simulator rollout is used here.  For each held-out dataset sample, the
baseline and stage-conditioned DiT see identical observations, noise and
diffusion timestep.  A stage is enabled only when its paired action error
improvement is positive with a conservative bootstrap lower bound.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.long_horizon_stage_adaln.benefit_model import (
    STAGE_KIND_ORDER,
    attach_benefit_adaln,
    load_adapter_state,
    normalize_stage_kind,
    stage_kind_index,
)
from scripts.long_horizon_stage_adaln.model import encode_text_condition
from scripts.long_horizon_stage_adaln.train_adapter import (
    RoboCasaStageDataset,
    full_prompt,
)
from scripts.long_horizon_stage_adaln.train_benefit_adapter import (
    build_context,
    forward_at,
    masked_mse,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--holdout-mod", type=int, default=5)
    parser.add_argument("--holdout-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=19)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--stage-condition-format", choices=["full", "subtask_only"], default="subtask_only")
    parser.add_argument("--history-frames", type=int, default=4)
    parser.add_argument("--history-interval-frames", type=int, default=2)
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--max-samples-per-episode", type=int, default=128)
    parser.add_argument("--max-video-size", type=int, default=224)
    return parser.parse_args()


def bootstrap_low(
    values: list[float],
    count: int,
    seed: int,
) -> float | None:
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=np.float64)
    means = np.empty(count, dtype=np.float64)
    for index in range(count):
        means[index] = rng.choice(array, size=array.size, replace=True).mean()
    return float(np.quantile(means, 0.05))


def skill_key(value: str | None) -> str:
    """Stable JSON key for an atomic-skill label from RoboCasa metadata."""
    text = str(value or "").strip()
    return re.sub(r"\s+", " ", text).lower()


def summarize_records(
    rows: list[dict[str, float]],
    args: argparse.Namespace,
    seed_offset: int,
) -> dict[str, Any]:
    improvements = [row["improvement"] for row in rows]
    mean_improvement = float(np.mean(improvements)) if improvements else 0.0
    win_rate = (
        float(np.mean(np.asarray(improvements) > args.margin))
        if improvements
        else 0.0
    )
    ci_low = bootstrap_low(
        improvements,
        args.bootstrap_samples,
        args.bootstrap_seed + seed_offset,
    )
    relative = (
        float(
            np.mean(
                [
                    row["improvement"] / max(row["base_error"], 1e-8)
                    for row in rows
                ]
            )
        )
        if rows
        else 0.0
    )
    if ci_low is not None and ci_low > args.margin and win_rate >= 0.60:
        recommended_scale = 1.0
    elif mean_improvement > args.margin and win_rate >= 0.55:
        recommended_scale = 0.25
    else:
        recommended_scale = 0.0
    return {
        "num_samples": len(rows),
        "mean_improvement": mean_improvement,
        "relative_improvement": relative,
        "win_rate": win_rate,
        "bootstrap_ci_low": ci_low,
        "recommended_scale": recommended_scale,
        "enabled": bool(recommended_scale > 0.0),
        "status": "calibrated" if rows else "unseen",
        "low_sample_warning": bool(rows and len(rows) < 2),
    }


def build_stratified_indices(
    dataset: Any,
    candidate_indices: np.ndarray,
    max_samples: int,
    seed: int,
    max_episodes_per_dataset: int = 32,
) -> tuple[np.ndarray, dict[str, int]]:
    """Select a holdout subset while covering every observed stage/skill.

    Dataset samples are grouped by task and episode in the manifest.  We read
    each episode table once, then reserve one candidate for every observed
    normalized stage and atomic skill before filling the remaining budget
    uniformly.  This avoids silently producing a gate with zero samples for a
    rare but real skill.
    """
    from collections import defaultdict

    episode_to_indices: dict[tuple[int, int], list[int]] = defaultdict(list)
    dataset_episode_order: dict[int, list[int]] = defaultdict(list)
    seen_episode: set[tuple[int, int]] = set()
    for sample_index in candidate_indices.tolist():
        dataset_index, packed = dataset.samples[int(sample_index)]
        episode_index, _ = divmod(packed, 10_000_000)
        episode_key = (dataset_index, episode_index)
        episode_to_indices[episode_key].append(int(sample_index))
        if episode_key not in seen_episode:
            dataset_episode_order[dataset_index].append(episode_index)
            seen_episode.add(episode_key)

    # A task's stage vocabulary is normally present in its first few
    # trajectories.  Limiting parquet reads keeps calibration startup bounded.
    coverage_episodes = [
        (dataset_index, episode_index)
        for dataset_index in sorted(dataset_episode_order)
        for episode_index in dataset_episode_order[dataset_index][
            :max_episodes_per_dataset
        ]
    ]
    group_to_indices: dict[str, list[int]] = defaultdict(list)
    loaded_episodes = 0
    for dataset_index, episode_index in coverage_episodes:
        table = dataset.datasets[dataset_index].get_trajectory_data(episode_index)
        task_table = dataset.datasets[dataset_index].tasks
        for sample_index in episode_to_indices[(dataset_index, episode_index)]:
            _, packed = dataset.samples[sample_index]
            _, base_index = divmod(packed, 10_000_000)
            ann = table.iloc[base_index]
            stage_name = str(
                task_table.loc[int(ann["annotation.human.subtask_stage"])]["task"]
            ).strip()
            atomic_skill = str(
                task_table.loc[int(ann["annotation.human.subtask_name"])]["task"]
            ).strip()
            group_to_indices[f"stage:{normalize_stage_kind(stage_name)}"].append(
                sample_index
            )
            group_to_indices[f"skill:{skill_key(atomic_skill)}"].append(
                sample_index
            )
        loaded_episodes += 1

    candidate_set = set(int(index) for index in candidate_indices.tolist())
    rng = np.random.default_rng(seed)
    required: list[int] = []
    for group in sorted(group_to_indices):
        choices = group_to_indices[group]
        required.append(int(choices[int(rng.integers(len(choices)))]))
    required = list(dict.fromkeys(required))

    if len(required) > max_samples:
        # Keep stage coverage first, then use the remaining budget for skills.
        stage_required = [
            int(group_to_indices[group][int(rng.integers(len(group_to_indices[group])))])
            for group in sorted(group_to_indices)
            if group.startswith("stage:")
        ]
        required = list(dict.fromkeys(stage_required))[:max_samples]

    remaining = np.asarray(
        sorted(candidate_set.difference(required)),
        dtype=np.int64,
    )
    budget = max(0, max_samples - len(required))
    if budget < len(remaining):
        remaining = rng.choice(remaining, size=budget, replace=False)
    selected = np.asarray(required + remaining.tolist(), dtype=np.int64)
    rng.shuffle(selected)
    return selected, {
        "candidate_episodes_scanned": len(episode_to_indices),
        "coverage_episodes_loaded": loaded_episodes,
        "observed_groups": len(group_to_indices),
        "required_coverage_samples": len(required),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_args = argparse.Namespace(
        history_frames=args.history_frames,
        history_interval_frames=args.history_interval_frames,
        action_horizon=args.action_horizon,
        max_samples_per_episode=args.max_samples_per_episode,
        max_video_size=args.max_video_size,
        stage_condition_format=args.stage_condition_format,
        manifest=str(manifest_path),
    )
    dataset = RoboCasaStageDataset(manifest, dataset_args)

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
    load_adapter_state(model, args.adapter_checkpoint)
    action_config = processor.action_config["robocasa365"]
    mean = action_config["mean"].to(args.device)
    std = action_config["std"].to(args.device).clamp_min(1e-6)

    records: dict[str, list[dict[str, float]]] = defaultdict(list)
    raw_stage_records: dict[str, list[dict[str, float]]] = defaultdict(list)
    skill_records: dict[str, list[dict[str, float]]] = defaultdict(list)
    skill_display_names: dict[str, str] = {}
    # Do not take the first ``max_samples`` chunks in dataset order.  The
    # manifest is grouped by task, so that would make calibration depend on
    # whichever tasks happen to appear first and can omit later stage kinds.
    candidate_indices = np.arange(
        args.holdout_index,
        len(dataset),
        args.holdout_mod,
        dtype=np.int64,
    )
    sample_count = min(args.max_samples, len(candidate_indices))
    selected_indices, sampling_info = build_stratified_indices(
        dataset,
        candidate_indices,
        sample_count,
        args.seed,
    )
    selected = 0
    with torch.inference_mode():
        for sample_index in selected_indices.tolist():
            sample = dataset.get(sample_index)
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
            context = build_context(model, inputs)
            generator = torch.Generator(device=args.device).manual_seed(
                args.seed + sample_index
            )
            noise = torch.randn(
                target.shape,
                generator=generator,
                device=target.device,
                dtype=target.dtype,
            )
            timestep = torch.rand(
                (target.shape[0], 1, 1),
                generator=generator,
                device=target.device,
                dtype=target.dtype,
            )
            noisy = (1.0 - timestep) * noise + timestep * target
            velocity = target - noise
            base = forward_at(model, context, noisy, timestep, None, None)
            stage_hidden = encode_text_condition(
                model,
                processor.tokenizer,
                [sample["stage_text"]],
                args.device,
            ).to(dtype=target.dtype)
            stage_ids = stage_kind_index(sample["stage_name"], args.device)
            conditioned = forward_at(
                model,
                context,
                noisy,
                timestep,
                stage_hidden,
                stage_ids,
            )
            base_error = float(
                masked_mse(base, velocity, context["action_mask"]).cpu()
            )
            conditioned_error = float(
                masked_mse(conditioned, velocity, context["action_mask"]).cpu()
            )
            stage_kind = normalize_stage_kind(sample["stage_name"])
            records[stage_kind].append(
                {
                    "base_error": base_error,
                    "conditioned_error": conditioned_error,
                    "improvement": base_error - conditioned_error,
                }
            )
            raw_stage = str(sample["stage_name"]).strip()
            raw_stage_records[raw_stage].append(
                {
                    "base_error": base_error,
                    "conditioned_error": conditioned_error,
                    "improvement": base_error - conditioned_error,
                }
            )
            atomic_skill = str(sample["atomic_skill"]).strip()
            atomic_skill_key = skill_key(atomic_skill)
            skill_display_names.setdefault(atomic_skill_key, atomic_skill)
            skill_records[atomic_skill_key].append(
                {
                    "base_error": base_error,
                    "conditioned_error": conditioned_error,
                    "improvement": base_error - conditioned_error,
                }
            )
            selected += 1
            if selected % 25 == 0:
                print(f"calibration samples={selected}", flush=True)

    report: dict[str, Any] = {
        "format_version": 1,
        "criterion": {
            "description": "paired held-out diffusion action error",
            "holdout_mod": args.holdout_mod,
            "holdout_index": args.holdout_index,
            "sampling": "uniform_without_replacement_over_holdout_indices",
            "candidate_samples": int(len(candidate_indices)),
            "sampling_info": sampling_info,
            "margin": args.margin,
            "bootstrap_low_quantile": 0.05,
        },
        "adapter_checkpoint": args.adapter_checkpoint,
        "stage_kind_order": STAGE_KIND_ORDER,
        "num_samples": selected,
        "stages": {},
        "raw_stages": {},
        "skills": {},
    }
    for stage_kind in STAGE_KIND_ORDER:
        rows = records.get(stage_kind, [])
        report["stages"][stage_kind] = summarize_records(
            rows,
            args,
            STAGE_KIND_ORDER.index(stage_kind),
        )

    for index, raw_stage in enumerate(sorted(raw_stage_records)):
        report["raw_stages"][raw_stage] = summarize_records(
            raw_stage_records[raw_stage],
            args,
            100 + index,
        )

    for index, atomic_skill_key in enumerate(sorted(skill_records)):
        row = summarize_records(
            skill_records[atomic_skill_key],
            args,
            1000 + index,
        )
        row["display_name"] = skill_display_names[atomic_skill_key]
        report["skills"][atomic_skill_key] = row

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
