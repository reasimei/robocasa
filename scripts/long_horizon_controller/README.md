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

## VLM Benchmark

`benchmark_vlm.py` compares verifier backends on identical saved VLM frames and
uses the production verifier prompt and JSON schema. Model arguments are
`ollama:MODEL` for local Ollama or `api:MODEL` for an OpenAI-compatible API.

First export a reproducible case set from saved rollout events. The generated
JSONL is intentionally unlabeled: add `expected_status` (`complete`,
`in_progress`, or `failed`) after visually reviewing each case before using it
to compare decision quality.

```bash
conda run -n robocasa python -m scripts.long_horizon_controller.benchmark_vlm \
  --models ollama:qwen3-vl:8b \
  --eval-root expdata/long_horizon_controller/composite_seen_full_lhc_aux11000_qwen25vl7b_dualview/evals/target \
  --max-cases 60 \
  --export-cases expdata/long_horizon_controller/vlm_benchmark/cases.jsonl
```

With reviewed labels, compare local Qwen with a GPT model enabled for the API
key. `gpt-5.2` is an example; choose a model name available to the account.

```bash
export OPENAI_API_KEY=...
conda run -n robocasa python -m scripts.long_horizon_controller.benchmark_vlm \
  --models ollama:qwen3-vl:8b api:gpt-5.2 \
  --cases expdata/long_horizon_controller/vlm_benchmark/cases.jsonl \
  --output expdata/long_horizon_controller/vlm_benchmark/qwen_vs_gpt52.json
```

The output contains per-case raw responses, parsed decisions, request latency,
available token usage, JSON parse rate, and, when labels are present, status
accuracy, per-status precision/recall/F1, and macro F1. Use
`--api-endpoint chat_completions` for an OpenAI-compatible endpoint that does
not implement the Responses API.

The benchmark also supports the local Hugging Face Qwen checkpoint directly:

```bash
conda run -n robocasa python -m scripts.long_horizon_controller.benchmark_vlm \
  --models 'local:/data/zjw/.cache/huggingface/hub/models--unsloth--Qwen3-VL-8B-Instruct-unsloth-bnb-4bit/snapshots/b5b904c3fcdc7541adf2a2bb219b0ed95288c794' \
  --cases expdata/long_horizon_controller/vlm_benchmark/cases_10tasks_50.jsonl \
  --output expdata/long_horizon_controller/vlm_benchmark/qwen3vl_local.json
```

`local:PATH` loads the checkpoint with `LocalQwenVLVerifier`. It is different
from `ollama:MODEL`: Ollama cannot use a Hugging Face snapshot path as its model
name. To test an Ollama model, import/register it in Ollama first and pass its
actual tag, for example `ollama:qwen3-vl:8b`.

To create a diverse 50-case set, use `--num-tasks 10`. The sampler requests one
`complete`, two `in_progress`, and two `retry` cases per task. Existing rollout
events do not contain enough `in_progress` cases for all tasks, so fallback rows
are marked with `sampling_fallback: true`; inspect or relabel them before
reporting accuracy. `reference_status` is the old rollout verifier result and
is useful for stratification or agreement only, not independent ground truth.

For ground-truth labels derived from the auxiliary-head data, build cases with:

```bash
conda run -n robocasa python scripts/long_horizon_controller/build_aux_vlm_cases.py \
  --num-tasks 10 --cases-per-task 5 \
  --complete-per-task 1 --progress-per-task 2 --retry-per-task 2 \
  --output expdata/long_horizon_controller/vlm_benchmark/aux_cases_10tasks_50.jsonl
```

This labels terminal positive frames as `complete`, middle positive frames as
`in_progress`, and synthetic retry frames as `failed`.

```bash
conda run -n robocasa python -m scripts.long_horizon_controller.benchmark_vlm \
  --models api:gpt-5.6-sol \
  --num-tasks 10 --cases-per-task 5 \
  --complete-per-task 1 --progress-per-task 2 --retry-per-task 2 \
  --export-cases expdata/long_horizon_controller/vlm_benchmark/cases_10tasks_50.jsonl

conda run -n robocasa python -m scripts.long_horizon_controller.benchmark_vlm \
  --models api:gpt-5.6-sol \
  --cases expdata/long_horizon_controller/vlm_benchmark/cases_10tasks_50.jsonl \
  --output expdata/long_horizon_controller/vlm_benchmark/gpt56_10tasks_50.json
```

The benchmark atomically checkpoints the output after every completed case. To
continue an interrupted run, rerun the identical command with the same
`--output` path and add `--resume`. Successfully parsed model/case pairs are
skipped; failed requests are retried.

```bash
conda run -n robocasa python -m scripts.long_horizon_controller.benchmark_vlm \
  --models ollama:qwen3-vl:8b api:gpt-5.2 \
  --cases expdata/long_horizon_controller/vlm_benchmark/cases.jsonl \
  --output expdata/long_horizon_controller/vlm_benchmark/qwen_vs_gpt52.json \
  --resume
```

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
