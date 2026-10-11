# Isaac-GR00T Long-Horizon Control

This README documents the current implementation. The original file is preserved as README.original.backup.md.

## Pipeline

The controller combines a VLA policy, lightweight auxiliary heads, a fast action monitor, and a slower VLM verifier. A planner creates SubtaskSpec items. controller.py executes one item at a time, sends its instruction to the policy, executes action chunks, records history, and invokes the fast monitor. A fast trigger requests VLM verification. VLM returns complete, in_progress, or failed. Complete advances the queue; failed invokes recovery.

Main files: scripts/long_horizon_controller/controller.py, planner.py, schemas.py, run_composite_seen_eval.py, and run_robocasa_controller.py.

## Auxiliary heads

Online runtime is in policy_adapters.py (Gr00tAuxHeadRuntime and Gr00tPolicyAdapter) and xiaomi_aux_runtime.py. The frozen policy backbone produces a pooled representation; small heads predict progress, success, and retry. fast_monitor.py fuses these predictions with fast action signals.

Data and training scripts are under scripts/aux_progress/: build_*_manifest.py, train_manifest_subtasks_aux.py, train_atomic_positive_aux.py, train_composite_gt_aux.py, and train_xiaomi_aux.py. Training freezes the GR00T/Xiaomi backbone and updates only the lightweight state/progress heads. Checkpoints contain aux_heads.pt and aux_config.json, including observation-history offsets used again online.

## Action entropy and fast triggering

fast_monitor.py implements FastTrigger values SUSPECT_TRANSITION, SUSPECT_COMPLETE, SUSPECT_FAIL, and TIMEOUT. It analyzes action chunks, local progress, timing, and auxiliary predictions. controller.py applies minimum-step and cooldown gates before asking the VLM. The fast signal is only a VLM gate; final completion is visual VLM confirmation. run_composite_seen_fast_aux_only_eval.py is the fast/auxiliary ablation entry point.

## VLM confirmation

vlm_verifier.py contains VLM_PROMPT_TEMPLATE and RECOVERY_PROMPT_TEMPLATE. It compares current and historical agentview/eye-in-hand frames and checks finish_state_satisfied and next_start_plausible. qwen_vl_worker.py provides local Qwen-VL execution; benchmark_vlm.py and test_ollama_vlm_json.py provide evaluation and protocol tests. VLM verifies state and does not control continuous actions.

## Subtask progression

schemas.py defines SubtaskSpec with instruction, expected_start_state, expected_finish_state, max_duration_sec, and notes. controller.py owns self.queue. On complete it appends to completed and increments current_index. Each new subtask clears active_action_history, fast-monitor state, and image history. Events go to controller_events.json; actions go to action_history.json. planner.py creates the initial subtask plan.

## Failure recovery

VLM must choose one mode.

### rollback_retry

For failures caused by recent actions or pose: controller.py::_rollback() takes the newest actions in active_action_history, reverses their order, and calls env.rollback_action(). The failed subtask is put back at the queue head. --max-rollback-chunks defaults to 8. The environment must expose supports_action_rollback=True and rollback_action(). This is action-level open-loop recovery, not a full simulator snapshot restore.

### insert_recovery

For dropped objects, wrong object/target, unstable grasp, blocked motion, or active repositioning: VLM returns 1-3 SubtaskSpec recovery items. controller.py inserts them before the failed subtask, then retries the failed subtask. Recovery logic and JSON parsing are in vlm_verifier.py; queue mutation and event logging are in controller.py; tests are in test_recovery.py.

## XR-1 special task embedding and adapter

xiaomi_policy_adapter.py loads the Xiaomi checkpoint/processor, builds the official XR-1 observation schema (agentview_left, agentview_right, eye_in_hand, end-effector/gripper/base state, and history), and re-encodes every subtask instruction as the model language/task embedding. It converts the resulting action chunk back to environment actions. Therefore inserted recovery subtasks become new XR-1 language-conditioned goals.

XR-1 evaluation entries are run_franka_xr1_trajectory.py, run_xiaomi_ur3e_fixed_eval.py, and run_xiaomi_ur3e_presoak.py.

## Adapter training

policy_adapters.py and xiaomi_policy_adapter.py are inference adapters: they convert observation/action schemas and do not train the backbone. Auxiliary adapter training is:

1. Build labeled subtask/stage manifests with build_*_manifest.py.
2. Train frozen-backbone heads with train_manifest_subtasks_aux.py, train_atomic_positive_aux.py, train_composite_gt_aux.py, or train_xiaomi_aux.py.
3. Save aux_heads.pt and aux_config.json.
4. Pass --aux-head-path to long-horizon evaluation; the adapter predicts auxiliary signals on every act() call.

## Common commands

    python -m scripts.long_horizon_controller.run_composite_seen_eval --model-path CHECKPOINT --aux-head-path AUX_DIR --output-root OUTPUT

    python -m scripts.long_horizon_controller.run_robocasa_controller --model-path CHECKPOINT --aux-head-path AUX_DIR

    python scripts/ur3e_robocasa_eval/run_xiaomi_ur3e_fixed_eval.py --model-path CHECKPOINT --aux-head-path AUX_DIR

    python -m pytest scripts/long_horizon_controller/test_recovery.py

## Debug outputs

controller_events.json records fast signals, VLM decisions, advancement, rollback, and inserted recovery. action_history.json stores action chunks. VLM frames/videos show visual evidence. aux_heads.pt and aux_config.json are auxiliary checkpoints.
