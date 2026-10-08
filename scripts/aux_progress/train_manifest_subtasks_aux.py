#!/usr/bin/env python3
"""
Train the GR00T auxiliary progress/state heads on pre-split subtask episodes.

The datasets under ``gr00t_manifest_subtasks`` are different from the older
composite GT manifest:

* each subtask is already stored as one LeRobot episode;
* ``episodes.jsonl`` points to its instruction through ``tasks.jsonl``;
* some pretrain exports keep ``annotation.human.task_description`` at zero,
  so the task text must be resolved from ``episodes.tasks`` explicitly.

This script converts those episodes into the manifest format consumed by
``train_composite_gt_aux.py`` and then reuses that trainer.  It therefore
keeps the existing auxiliary-head architecture, observation history handling,
synthetic retry generation, checkpoint evaluation, and best-checkpoint
retention behavior.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import tyro

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.aux_progress.build_composite_gt_aux_manifest import (  # noqa: E402
    DatasetRecord,
    EpisodeRecord,
    SubtaskSegment,
    asdict as manifest_asdict,
    make_retry_examples,
)
from scripts.aux_progress.build_composite_gt_aux_manifest import Args as RetryArgs  # noqa: E402
from scripts.aux_progress.train_composite_gt_aux import (  # noqa: E402
    Args as CompositeTrainArgs,
    main as train_composite_main,
)


@dataclass
class Args:
    # Dataset discovery.
    dataset_root: str = (
        "/data/zjw/workspace/robocasa/datasets/"
        "gr00t_manifest_subtasks"
    )
    dataset_split: str = "all"
    tasks: str = ""
    dataset_paths: tuple[str, ...] = ()
    min_episode_length: int = 4
    max_episodes_per_dataset: int = -1
    skip_invalid_datasets: bool = False

    # Generated manifest.
    generated_manifest_path: str = (
        "/data/zjw/workspace/Isaac-GR00T/expdata/aux_progress/"
        "gr00t_manifest_subtasks_aux_manifest.json"
    )
    max_retry_samples_per_type: int = 20000
    reverse_step_stride: int = 4
    repeat_anchor_stride: int = 8
    repeat_copies: int = 3
    mismatch_step_stride: int = 4
    backtrack_turn_fraction: float = 0.75
    backtrack_step_stride: int = 4
    retry_seed: int = 42
    manifest_only: bool = False
    use_existing_manifest: bool = False

    # Frozen GR00T checkpoint and auxiliary training.
    checkpoint_path: str = (
        "/data/zjw/workspace/Isaac-GR00T/expdata/"
        "foundation_model_learning/target_posttraining/composite_seen/"
        "checkpoint-60000"
    )
    output_dir: str = (
        "/data/zjw/workspace/Isaac-GR00T/expdata/aux_progress/"
        "gr00t_manifest_subtasks_aux_run1"
    )
    atomic_manifest_path: str = ""
    atomic_retry_manifest_path: str = ""
    resume_from_checkpoint: str = ""
    data_config: str = "panda_omron"
    embodiment_tag: str = "new_embodiment"
    video_backend: str = "opencv"
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_steps: int = 20000
    save_steps: int = 500
    save_total_limit: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.05
    dataloader_num_workers: int = 0
    bf16: bool = True
    num_gpus: int = 1

    # Auxiliary labels and sampling.
    progress_gamma: float = 1.5
    success_tail_fraction: float = 0.1
    success_tail_min_steps: int = 3
    observation_history_offsets: str = "-8,-4,0"
    synthetic_retry_history: bool = True
    aux_context_mode: str = "state_delta"
    train_split: float = 0.9
    train_epoch_size: int = 200000
    subtask_progress_sample_weight: float = 1.0
    subtask_success_sample_weight: float = 1.0
    subtask_retry_sample_weight: float = 0.5
    atomic_sample_weight: float = 1.0
    atomic_epoch_size: int = 100000
    progress_class_loss_weight: float = 1.0
    success_class_loss_weight: float = 1.0
    retry_class_loss_weight: float = 0.7
    state_label_smoothing: float = 0.02
    seed: int = 42

    # Logging and validation.
    report_to: str = "wandb"
    wandb_project: str = "robocasa-aux-progress"
    wandb_entity: str = ""
    wandb_mode: str = "online"
    eval_batch_size: int = 32
    eval_max_batches: int = 2048
    eval_subset_seed: int = 123


def _load_json_lines(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _resolve_dataset_paths(args: Args) -> list[Path]:
    explicit = [Path(item).expanduser().resolve() for item in args.dataset_paths]
    if explicit:
        candidates = explicit
    else:
        root = Path(args.dataset_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist: {root}")
        candidates = sorted(root.glob("*/composite/*/*/lerobot"))

    if args.dataset_split not in {"all", "pretrain", "target"}:
        raise ValueError(
            f"dataset_split must be 'all', 'pretrain', or 'target', got "
            f"{args.dataset_split!r}"
        )
    if args.dataset_split != "all":
        candidates = [
            path
            for path in candidates
            if args.dataset_split in path.parts
        ]

    requested_tasks = {
        item.strip()
        for item in args.tasks.split(",")
        if item.strip()
    }
    if requested_tasks:
        candidates = [
            path
            for path in candidates
            if path.parents[1].name in requested_tasks
        ]

    deduplicated: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if path not in seen:
            deduplicated.append(path)
            seen.add(path)
    if not deduplicated:
        raise RuntimeError("No subtask LeRobot datasets matched the requested filters.")
    return deduplicated


def _validate_dataset_root(path: Path) -> None:
    required = (
        path / "meta" / "info.json",
        path / "meta" / "tasks.jsonl",
        path / "meta" / "episodes.jsonl",
        path / "meta" / "modality.json",
    )
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        raise ValueError(f"{path} is missing required metadata: {missing}")
    if not list((path / "data").glob("chunk-*/episode_*.parquet")):
        raise ValueError(f"{path} has no episode parquet files")


def _dataset_record(
    path: Path,
    dataset_index: int,
    args: Args,
) -> tuple[DatasetRecord, list[str]]:
    _validate_dataset_root(path)
    task_name = path.parents[1].name
    split_name = path.parents[3].name
    task_rows = _load_json_lines(path / "meta" / "tasks.jsonl")
    task_map = {
        int(row["task_index"]): str(row["task"]).strip()
        for row in task_rows
    }
    if not task_map or any(not instruction for instruction in task_map.values()):
        raise ValueError(f"{path} has empty or missing task instructions")

    source_episodes = _load_json_lines(path / "meta" / "episodes.jsonl")
    if args.max_episodes_per_dataset > 0:
        source_episodes = source_episodes[: args.max_episodes_per_dataset]

    episodes: list[EpisodeRecord] = []
    skipped: list[str] = []
    for episode in source_episodes:
        episode_index = int(episode["episode_index"])
        length = int(episode["length"])
        if length < args.min_episode_length:
            skipped.append(
                f"{task_name}/{episode_index}:length={length}"
            )
            continue

        task_ids = [int(item) for item in episode.get("tasks", [])]
        if len(task_ids) != 1:
            raise ValueError(
                f"{path} episode {episode_index} has tasks={task_ids}; "
                "expected exactly one pre-split subtask."
            )
        task_id = task_ids[0]
        if task_id not in task_map:
            raise ValueError(
                f"{path} episode {episode_index} references unknown "
                f"task_index={task_id}"
            )

        instruction = task_map[task_id]
        segment = SubtaskSegment(
            segment_index=0,
            subtask_idx=task_id,
            subtask_id=f"{task_name}:episode_{episode_index}:task_{task_id}",
            subtask_annotation_id=task_id,
            atomic_skill_annotation_id=task_id,
            stage_annotation_id=task_id,
            atomic_skill=instruction,
            stage="",
            instruction=instruction,
            start_frame=0,
            end_frame=length,
            length=length,
        )
        episodes.append(
            EpisodeRecord(
                episode_index=episode_index,
                length=length,
                task_instruction=instruction,
                subtasks=[segment],
            )
        )

    if not episodes:
        raise ValueError(
            f"{path} has no episodes after min_episode_length="
            f"{args.min_episode_length}"
        )

    return (
        DatasetRecord(
            task_name=task_name,
            dated_dir=str(path.parent),
            lerobot_root=str(path),
            num_episodes=len(episodes),
            num_frames=sum(item.length for item in episodes),
            num_subtasks=len(episodes),
            episodes=episodes,
        ),
        skipped,
    )


def _build_manifest(args: Args) -> dict[str, Any]:
    records: list[DatasetRecord] = []
    skipped_datasets: dict[str, str] = {}
    skipped_episodes: list[str] = []

    for path in _resolve_dataset_paths(args):
        try:
            record, skipped = _dataset_record(path, len(records), args)
            records.append(record)
            skipped_episodes.extend(skipped)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            if not args.skip_invalid_datasets:
                raise
            skipped_datasets[str(path)] = str(exc)

    if not records:
        raise RuntimeError("No valid subtask datasets remained after validation.")

    retry_args = RetryArgs(
        reverse_step_stride=args.reverse_step_stride,
        repeat_anchor_stride=args.repeat_anchor_stride,
        repeat_copies=args.repeat_copies,
        mismatch_step_stride=args.mismatch_step_stride,
        backtrack_turn_fraction=args.backtrack_turn_fraction,
        backtrack_step_stride=args.backtrack_step_stride,
        max_retry_samples_per_type=args.max_retry_samples_per_type,
        seed=args.retry_seed,
        progress=False,
    )
    retry_records, retry_counts = make_retry_examples(
        records,
        args.max_retry_samples_per_type,
        retry_args,
    )

    # The old retry dataset only replaces language when a retry is a mismatch.
    # Explicitly attach the source subtask instruction to other retry types too.
    source_instruction: dict[tuple[int, int, int], tuple[str, str]] = {}
    for dataset_index, record in enumerate(records):
        for episode in record.episodes:
            for segment in episode.subtasks:
                source_instruction[
                    (
                        dataset_index,
                        int(episode.episode_index),
                        int(segment.segment_index),
                    )
                ] = (segment.subtask_id, segment.instruction)
    for retry in retry_records:
        if retry.replacement_instruction is None:
            key = (
                int(retry.source_dataset_index),
                int(retry.source_episode_index),
                int(retry.source_subtask_segment_index),
            )
            subtask_id, instruction = source_instruction[key]
            retry.replacement_subtask_id = subtask_id
            retry.replacement_instruction = instruction

    payload = {
        "format_version": 2,
        "source": "robocasa_gr00t_manifest_subtasks",
        "dataset_root": str(Path(args.dataset_root).expanduser().resolve()),
        "dataset_split": args.dataset_split,
        "min_episode_length": int(args.min_episode_length),
        "num_datasets": len(records),
        "num_episodes": sum(item.num_episodes for item in records),
        "num_frames": sum(item.num_frames for item in records),
        "num_subtasks": sum(item.num_subtasks for item in records),
        "skipped_datasets": skipped_datasets,
        "skipped_short_episodes": skipped_episodes,
        "retry_counts": retry_counts,
        "retry_examples": [manifest_asdict(item) for item in retry_records],
        "generation_config": {
            "reverse_step_stride": int(args.reverse_step_stride),
            "repeat_anchor_stride": int(args.repeat_anchor_stride),
            "repeat_copies": int(args.repeat_copies),
            "mismatch_step_stride": int(args.mismatch_step_stride),
            "backtrack_turn_fraction": float(args.backtrack_turn_fraction),
            "backtrack_step_stride": int(args.backtrack_step_stride),
            "max_retry_samples_per_type": int(args.max_retry_samples_per_type),
            "seed": int(args.retry_seed),
        },
        "datasets": [
            {
                **asdict(record),
                "episodes": [
                    {
                        **asdict(episode),
                        "subtasks": [
                            asdict(segment)
                            for segment in episode.subtasks
                        ],
                    }
                    for episode in record.episodes
                ],
            }
            for record in records
        ],
    }
    output = Path(args.generated_manifest_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(
        f"Generated {output}: datasets={payload['num_datasets']} "
        f"episodes={payload['num_episodes']} frames={payload['num_frames']} "
        f"subtasks={payload['num_subtasks']} "
        f"retry_examples={len(retry_records)} "
        f"skipped_short={len(skipped_episodes)}",
        flush=True,
    )
    if skipped_datasets:
        print(
            f"Skipped invalid datasets: {len(skipped_datasets)}",
            flush=True,
        )
    return payload


def _training_args(args: Args) -> CompositeTrainArgs:
    return CompositeTrainArgs(
        gt_manifest_path=str(Path(args.generated_manifest_path).expanduser().resolve()),
        atomic_manifest_path=args.atomic_manifest_path,
        atomic_retry_manifest_path=args.atomic_retry_manifest_path,
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        data_config=args.data_config,
        embodiment_tag=args.embodiment_tag,
        video_backend=args.video_backend,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        resume_from_checkpoint=args.resume_from_checkpoint,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        dataloader_num_workers=args.dataloader_num_workers,
        bf16=args.bf16,
        num_gpus=args.num_gpus,
        progress_gamma=args.progress_gamma,
        success_tail_fraction=args.success_tail_fraction,
        success_tail_min_steps=args.success_tail_min_steps,
        observation_history_offsets=args.observation_history_offsets,
        synthetic_retry_history=args.synthetic_retry_history,
        aux_context_mode=args.aux_context_mode,
        train_split=args.train_split,
        train_epoch_size=args.train_epoch_size,
        atomic_epoch_size=args.atomic_epoch_size,
        atomic_sample_weight=args.atomic_sample_weight,
        gt_progress_sample_weight=args.subtask_progress_sample_weight,
        gt_success_sample_weight=args.subtask_success_sample_weight,
        gt_retry_sample_weight=args.subtask_retry_sample_weight,
        progress_class_loss_weight=args.progress_class_loss_weight,
        success_class_loss_weight=args.success_class_loss_weight,
        retry_class_loss_weight=args.retry_class_loss_weight,
        state_label_smoothing=args.state_label_smoothing,
        seed=args.seed,
        report_to=args.report_to,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
        eval_batch_size=args.eval_batch_size,
        eval_max_batches=args.eval_max_batches,
        eval_subset_seed=args.eval_subset_seed,
    )


def main(args: Args) -> None:
    manifest_path = Path(args.generated_manifest_path).expanduser().resolve()
    if args.use_existing_manifest and manifest_path.is_file():
        print(f"Using existing generated manifest: {manifest_path}", flush=True)
    else:
        _build_manifest(args)
    if args.manifest_only:
        return
    train_composite_main(_training_args(args))


if __name__ == "__main__":
    main(tyro.cli(Args))
