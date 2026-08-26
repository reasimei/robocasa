# Xiaomi Stage AdaLN Oracle Experiment

This directory is deliberately separate from `scripts/long_horizon_controller`.

The experiment keeps the complete task instruction in Xiaomi's normal VLM
input.  The current RoboCasa GT subtask text is encoded by the same Qwen3-VL
text backbone and injected as an additional AdaLN condition in the Xiaomi DiT.
Only the new adapter is trainable.

The target composite tasks listed by the experiment are excluded by
`manifest.py`.  The source must therefore contain RoboCasa pretrain datasets;
the local target-only download is not a valid source.

The adapter is zero-initialized, so its untrained checkpoint is exactly the
original Xiaomi policy.  The eventual KettleBoiling evaluation uses simulator
oracle stage labels and seeds 1000 through 1049.

## Benefit-Gated Adapter

`train_benefit_adapter.py` trains a conservative residual adapter.  Its loss
compares the conditioned DiT with the frozen baseline on the same noisy action
sample and penalizes the condition when it is worse than baseline.

`calibrate_benefit_gate.py` then uses held-out action-space comparisons to
produce a per-stage `recommended_scale`.  This is an offline criterion; no
rollout success signal is used to decide whether a stage condition is enabled.

Example:

```bash
conda run -n robocasa env \
  CUDA_VISIBLE_DEVICES=0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  NO_ALBUMENTATIONS_UPDATE=1 \
  python scripts/long_horizon_stage_adaln/train_benefit_adapter.py \
  --manifest expdata/long_horizon_stage_adaln/target_composite_manifest.json \
  --model-path expdata/Xiaomi-Robotics-1-RoboCasa365 \
  --output-root expdata/long_horizon_stage_adaln/benefit_adapter_target_composite \
  --stage-condition-format subtask_only \
  --steps 10000 \
  --save-every 1000
```

After training, calibrate on a held-out subset:

```bash
conda run -n robocasa env \
  CUDA_VISIBLE_DEVICES=0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  NO_ALBUMENTATIONS_UPDATE=1 \
  python scripts/long_horizon_stage_adaln/calibrate_benefit_gate.py \
  --manifest expdata/long_horizon_stage_adaln/target_composite_manifest.json \
  --model-path expdata/Xiaomi-Robotics-1-RoboCasa365 \
  --adapter-checkpoint expdata/long_horizon_stage_adaln/benefit_adapter_target_composite/checkpoint-10000.pt \
  --output expdata/long_horizon_stage_adaln/benefit_adapter_target_composite/gate.json \
  --stage-condition-format subtask_only
```

Use the calibrated gate with `eval_composite_seen_oracle.py`:

```bash
python scripts/long_horizon_stage_adaln/eval_composite_seen_oracle.py \
  --adapter-variant benefit_gated \
  --adapter-checkpoint expdata/long_horizon_stage_adaln/benefit_adapter_target_composite/checkpoint-10000.pt \
  --benefit-gate-config expdata/long_horizon_stage_adaln/benefit_adapter_target_composite/gate.json \
  --stage-condition-format subtask_only
```
