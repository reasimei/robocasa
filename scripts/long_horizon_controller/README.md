# Long-Horizon Controller

This folder contains a modular fast/slow controller for long-horizon Robocasa tasks.

整体目标：在Robocasa365的benchmark上尽可能提高机械臂长程操作的成功率

具体方案如下：

高层规划任务：分解为子任务至少需要包含
{语言指令，期望开始时和完成时的状态，最大执行时间(用于后续VLM判断)}

子任务衔接：“快慢系统”
高频/低成本：每步都跑辅助头或动作熵监控（几乎无开销），作为“快系统”
低频/高可靠：只在快系统触发“疑似完成/失败”时，才调用一次 VLM 做确认（“慢系统”），VLM同时用当前任务的期望完成状态和下一任务的期望初始状态为prompt看是否完成。

子任务执行：已有VLA策略
失败恢复：VLM检测出失败同时给出重规划的子任务，将给出的恢复子任务插队到VLA的输入

其中“快系统”作为一个高频低成本和动作一起输出的任务状态检测器，如何训练？
任务状态：{成功，进展，retry} 【离散{1,0,-1}；连续[-1,1]；离散+连续……】

辅助头：参考SeqVLA或CycleVLA，加上retry需要额外的失败数据，训练策略可能也要调整；数据：Success和progress的数据直接从专家数据来，最后一帧1-success，之前帧按时间顺序压缩到0-1-progress；Retry的数据？①reverse（开门→关门，记录为0~-1负进展）②repeat（模拟停滞）③mismatch（配错误的语言指令）④backtrack（成功轨迹123456→12321）；训练：两个头，冻结动作，只训分类头；课程学习，先用success和progress全量微调，再加入retry冻结动作。
动作熵：VLA输出时对动作的确定程度/变化程度，不确定/变化大—可能需要retry或者成功要切换下一任务，使用groot模型：定义Chunk consistency. GR00T 预测 action chunk（比如 50 步 future actions）。在时刻 t 预测的 chunk[t:t+50] 和 t+1 时刻预测的 chunk[t+1:t+51] 有 49 步重叠，算重叠部分的 L2 差异（只需要缓存上一步的 chunk）

最后辅助头和动作熵信号融合&EMA平滑
trigger_vlm = suspect_complete OR suspect_fail OR timeout（最大时间兜底）

## Modules

1. High-level planner
   - File: `planner.py`
   - Input: full task instruction
   - Output: `TaskPlan` with subtasks:
     - `instruction`
     - `expected_start_state`
     - `expected_finish_state`
     - `max_duration_sec`

2. Fast/slow subtask transition
   - Fast system: `fast_monitor.py`
     - `ActionEntropyMonitor` computes action chunk consistency.
     - `AuxHeadFusionMonitor` can fuse auxiliary-head `{progress, success, retry}` output.
   - Slow system: `vlm_verifier.py`
     - `LocalQwenVLVerifier` loads the local Qwen3-VL checkpoint.
     - Verifies current finish state and next start state.
     - Returns `complete`, `in_progress`, or `failed`.

3. VLA execution
   - File: `policy_adapters.py`
   - `Gr00tPolicyAdapter` wraps an existing `Gr00tPolicy`.
   - `MockPolicyAdapter` supports dry-run tests without GPU.

4. Failure recovery
   - File: `controller.py`
   - On VLM failure, the recovery prompt must choose exactly one mode:
     - `insert_recovery`: insert the returned `recovery_subtasks` before retrying the current subtask.
     - `rollback_retry`: reverse the requested number of recent policy action chunks, then retry the current subtask.
   - Rollback actions are stored in `action_history.json`; each rollback is recorded as
     `rollback_step` and `rollback_retry` in `controller_events.json`.

## LLM Planner

The high-level planner supports two LLM backends:

- `--planner api`: OpenAI-compatible `/chat/completions` endpoint through Python stdlib.
- `--planner ollama`: local Ollama `/api/chat` endpoint, also through Python stdlib.

For API mode, set:

```bash
export OPENAI_API_KEY=...
export OPENAI_MODEL=...
export OPENAI_BASE_URL=https://api.openai.com/v1
```

Then run with `--planner api`.

For Ollama mode on this server, use the local llama3.1 70B model:

```bash
export OLLAMA_MODEL=llama3.1:70b
export OLLAMA_BASE_URL=http://localhost:11434
```

Then run with `--planner ollama`. You can also pass `--ollama-model`, `--ollama-base-url`,
`--llm-timeout-sec`, `--ollama-num-predict`, and `--ollama-num-gpu` on the CLI.

## Local VLM

Default local VLM path:

```text
/data/zjw/.cache/huggingface/hub/models--unsloth--Qwen3-VL-8B-Instruct-unsloth-bnb-4bit/snapshots/b5b904c3fcdc7541adf2a2bb219b0ed95288c794
```

Use `--verifier qwen_vl` to load it. The dry-run verifier is the default for smoke tests.
In the current `robocasa` environment, `transformers==4.51.3` does not recognize
`model_type=qwen3_vl`; upgrade `transformers` before running the real local VLM verifier.

## Dry Run

From the repository root:

```bash
conda run -n robocasa python -m scripts.long_horizon_controller.cli \
  --planner static \
  --verifier dry \
  --dry-vlm-status complete \
  --output-dir expdata/long_horizon_controller/dry_run
```

Outputs:

```text
expdata/long_horizon_controller/dry_run/plan.json
expdata/long_horizon_controller/dry_run/controller_events.json
```

Minimal Ollama planner smoke test, still using the dry verifier and mock policy/env:

```bash
conda run -n robocasa python -m scripts.long_horizon_controller.cli \
  --planner ollama \
  --ollama-model llama3.1:70b \
  --llm-timeout-sec 600 \
  --ollama-num-predict 256 \
  --ollama-num-gpu 33 \
  --verifier dry \
  --dry-vlm-status complete \
  --output-dir expdata/long_horizon_controller/ollama_smoke
```

## Integrating With GR00T

For a real single-episode Robocasa rollout, use:

```bash
conda run -n robocasa python -m scripts.long_horizon_controller.run_robocasa_controller \
  --task "Pick the kettle from the counter and place it on the tray, then place the mug on the tray." \
  --env-name <RobocasaEnvName> \
  --model-path /data/zjw/workspace/Isaac-GR00T/expdata/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000 \
  --planner ollama \
  --ollama-model llama3.1:70b \
  --ollama-num-gpu 33 \
  --verifier dry \
  --output-dir expdata/long_horizon_controller/robocasa_run
```

This runner saves:

```text
expdata/long_horizon_controller/robocasa_run/plan.json
expdata/long_horizon_controller/robocasa_run/controller_events.json
expdata/long_horizon_controller/robocasa_run/videos/
```

The runner currently supports `n_envs=1`, because every episode has its own subtask
queue, VLM calls, and recovery insertion.

To wire the controller manually, create a `Gr00tPolicy`, then wrap it:

```python
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy
from scripts.long_horizon_controller.policy_adapters import Gr00tPolicyAdapter

cfg = DATA_CONFIG_MAP["panda_omron"]
policy = Gr00tPolicy(
    model_path="/path/to/checkpoint",
    modality_config=cfg.modality_config(),
    modality_transform=cfg.transform(),
    embodiment_tag="new_embodiment",
    denoising_steps=4,
)
adapter = Gr00tPolicyAdapter(
    policy=policy,
    action_keys=cfg.modality_config()["action"].modality_keys,
)
```

The environment adapter must implement:

```python
reset() -> observation
step(action) -> observation, reward, done, info
get_vlm_image(observation) -> image
```
