#!/usr/bin/env python3
"""Fine-tune GR00T on a manifest-built lightweight dataset.

The production GR00T loader and finetune script remain untouched. This wrapper
registers one or more prepared dataset directories under a temporary soup key
and delegates the actual model/trainer setup to ``scripts/gr00t_finetune.py``.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from robocasa.utils.dataset_registry import DATASET_SOUP_REGISTRY

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gr00t_finetune import ArgsConfig, main as finetune_main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", default=[], help="Prepared GR00T dataset directory; repeatable")
    ap.add_argument("--dataset-root", action="append", default=[], help="Root containing prepared task datasets")
    ap.add_argument("--base-model-path", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--data-config", default="panda_omron")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--num-gpus", type=int, default=1)
    ap.add_argument("--dataloader-num-workers", type=int, default=4)
    ap.add_argument("--wandb-project", default="robocasa-manifest-subtasks")
    ap.add_argument("--wandb-run-name", default="manifest-subtask-finetune")
    ap.add_argument("--wandb-mode", default="online", choices=("online", "offline", "disabled"))
    ap.add_argument("--tune-llm", action="store_true")
    ap.add_argument("--tune-visual", action="store_true")
    ap.add_argument("--tune-diffusion-model", action="store_true", help="Train the large diffusion action head; default is frozen for 24GB GPUs")
    args = ap.parse_args()

    paths = [str(Path(p).expanduser().resolve()) for p in args.dataset]
    for root in args.dataset_root:
        paths.extend(str(p.resolve()) for p in sorted(Path(root).expanduser().glob("*/composite/*/*/lerobot")) if (p / "meta" / "modality.json").exists())
    if not paths:
        ap.error("provide --dataset or --dataset-root")
    for path in paths:
        if not Path(path).is_dir():
            raise FileNotFoundError(path)
    soup_key = "__manifest_subtasks_runtime__"
    DATASET_SOUP_REGISTRY[soup_key] = [{"path": path, "filter_key": None} for path in paths]

    os.environ["WANDB_PROJECT"] = args.wandb_project
    os.environ["WANDB_RUN_NAME"] = args.wandb_run_name
    os.environ["WANDB_MODE"] = args.wandb_mode
    config = ArgsConfig(
        dataset_soup=soup_key,
        output_dir=args.output_dir,
        run_name=args.wandb_run_name,
        data_config=args.data_config,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        num_gpus=args.num_gpus,
        base_model_path=args.base_model_path,
        dataloader_num_workers=args.dataloader_num_workers,
        tune_llm=args.tune_llm,
        tune_visual=args.tune_visual,
        tune_diffusion_model=args.tune_diffusion_model,
        report_to="wandb" if args.wandb_mode != "disabled" else "tensorboard",
        embodiment_tag="new_embodiment",
        video_backend="opencv",
    )
    finetune_main(config)


if __name__ == "__main__":
    main()
