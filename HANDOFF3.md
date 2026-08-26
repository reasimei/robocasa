# HANDOFF3

更新时间：2026-08-20

## 1. Overall Goal

在 Robocasa365 composite_seen benchmark 上测试长时域控制框架，目标是提高已有 GR00T VLA/Xiaomi robotics-1（XR-1） 策略在composite任务上的成功率。希望不大幅度改变现有的VLA策略，而是进行系统级的改进能泛用到各种策略上。

整体框架：

1. 高层 planner 使用 Llama 将完整任务拆成多个原子子任务。
2. 每个子任务包含：
   - `instruction`
   - `expected_start_state`
   - `expected_finish_state`
   - `max_duration_sec`
   - `subtask_id`
3. 已有 GR00T VLA/XR-1 执行动作。
4. 快系统每个 policy action chunk 运行：
   - action chunk consistency / action entropy
   - auxiliary progress/state head
5. 快系统触发时调用慢系统 VLM verifier。
6. VLM 判断当前子任务是否完成，或者失败后选择恢复方式：
   - 插入新的恢复子任务
   - 回滚最近动作后重试当前子任务
7. 用 Robocasa 环境的 `info["success"]` 判断最终任务成功，不能使用 controller 自己的 `finish_plan` 作为环境成功判断。
8. 子任务的切换方式，GROOT模型使用的就是改变语言输入（使用原来的完整语言+current subgoal形式）避免输入分布偏移过大，并且用Oracle条件测试了确实能够对成功率有一些改善
9. 但是用这个方法对Xiaomi，用仿真Oracle切分子任务成功率下降【用不同的给子任务prompt：overall+current从52%下降到40%，subtask_only下降到约11%】
目前XR-1切任务的流程如下：
 - 保持完整任务语言输入不变
 - 子任务作为额外条件，取qwen text hidden经过一个小adapter和原本的时间步t经过gate（减少子任务的权重，减少对原分布的影响）一起映射到DiT的6x1024的AdaLN（自适应归一化层）条件（PS：这个adaln条件是本来就有的，根据流匹配的时间步t算出两个参数：缩放scale和平移shift，然后用这两个参数去调整DiT网络内部神经元的激活值（可以理解为神经元传递的信号强度）将当前处于哪个生成阶段这个关键信息，高效地融入到了动作生成的全过程中）
 - adapter需要训练且零初始化
 - adapter设计：
   输入：GT 切出来的子任务文本，先经过 Xiaomi 里的 Qwen 文本 backbone，取 last_hidden_state，再做 mask mean pooling，得到一个 text embedding。
   输出：把这个 embedding 映射成 6 x 1024 的 AdaLN delta，经过gate加到原本的 time AdaLN 上，再送进 DiT。
   损失：不是对齐 text embedding，本质上还是对齐 DiT 的动作扩散输出。代码里是用噪声构造 velocity target，然后做 MSE。
但是这使用Oracle条件的效果目前还是不如纯XR-1，即想改进的baseline，所以xr-1还没有接到快慢系统那一套上。

代码根目录：

```text
/data/zjw/workspace/Isaac-GR00T
```

主要实现目录：

```text
快慢系统相关：
/data/zjw/workspace/Isaac-GR00T/scripts/long_horizon_controller

xr-1相关：
/data/zjw/workspace/Isaac-GR00T/scripts/long_horizon_stage_adaln
```

## 2. Important Models

GR00T policy：

```text
/data/zjw/workspace/Isaac-GR00T/expdata/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000
```

当前最近使用的 auxiliary checkpoint：

```text
/data/zjw/workspace/Isaac-GR00T/expdata/aux_progress/atomic_retry_3class_history_retry_boost_run2/checkpoint-11000
```


Planner：

```text
Ollama llama3.1:70b
```

VLM verifier：

```text
Ollama qwen2.5vl:7b
```

Ollama 必须提前启动，并确保模型已经存在：

```bash
/data/zjw/bin/ollama serve
/data/zjw/bin/ollama list
```

## 3. Plan Cache

Llama plan cache：

```text
/data/zjw/workspace/Isaac-GR00T/expdata/long_horizon_controller/composite_seen_plan_cache_llama70b
```

每个任务的 plan 文件例如：

```text
composite_seen_plan_cache_llama70b/DeliverStraw/plan.json
```

运行时真正使用的是：

```json
plan["subtasks"]
```

`raw_response` 只保存原始 Llama 输出用于追溯，不会在执行时重新解析，也不会覆盖已经修改的 `subtasks`。

plan 读取优先级：

```text
<output-root>/evals/target/<Task>/plan.json
    优先于
<plan-cache-dir>/<Task>/plan.json
```

因此，如果手动修改了 plan cache：

- 使用新的 `--output-root`；或者
- 删除旧 output root 中对应任务的 `evals/target/<Task>/plan.json`

不要只使用 `--overwrite` 期待重新读取 cache。`--overwrite` 主要覆盖 episode 结果。`--overwrite-plan` 会重新调用 Llama 生成 plan，不适合加载手动修改的 cache。

## 4. Main Evaluation Command

建议使用新 output root，避免旧视频和旧 plan 干扰：

```bash
cd /data/zjw/workspace/Isaac-GR00T

conda run -n robocasa env \
  CUDA_VISIBLE_DEVICES=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  NO_ALBUMENTATIONS_UPDATE=1 \
  python scripts/long_horizon_controller/run_composite_seen_eval.py \
  --output-root expdata/long_horizon_controller/composite_seen_aux11000_qwen25vl7b_recovery_rerun \
  --task-set composite_seen \
  --split target \
  --n-episodes 10 \
  --model-path expdata/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000 \
  --aux-head-path expdata/aux_progress/atomic_retry_3class_history_retry_boost_run2/checkpoint-11000 \
  --planner ollama \
  --ollama-model llama3.1:70b \
  --plan-cache-dir expdata/long_horizon_controller/composite_seen_plan_cache_llama70b \
  --verifier ollama_vl \
  --vlm-ollama-model qwen2.5vl:7b \
  --vlm-history-frames 2 \
  --vlm-history-interval-sec 1.0 \
  --vlm-num-predict 256 \
  --max-episode-steps 3000 \
  --vlm-timeout-sec 180 \
  --max-rollback-chunks 4 \
  --annotate-videos \
  --overwrite
```

`--n-episodes 10` 表示每个 composite_seen 任务跑 10 个 episode，不是整个任务集总共跑 10 个 episode。

## 5. Image Input To VLM

默认 camera keys：

```text
video.robot0_agentview_left
video.robot0_eye_in_hand
```

当前设置：

```text
--vlm-history-frames 2
--vlm-history-interval-sec 1.0
```

因此 VLM 收到 4 张图：

1. 1 秒前的 `agentview_left`
2. 1 秒前的 `eye_in_hand`
3. 当前的 `agentview_left`
4. 当前的 `eye_in_hand`

VLM prompt 会说明：

- 同一时间的两个 camera 是互补视角。
- 第一组是历史，第二组是当前。
- 用历史/当前对判断进展。
- 用当前的两个视角判断状态。

环境在等待同步 Ollama VLM 回复期间是阻塞的。

## 6. Fast System

实现：

```text
scripts/long_horizon_controller/fast_monitor.py
```

`ActionEntropyMonitor` 实际计算的是 action chunk consistency：

- 当前 action chunk 与上一次 chunk 的重叠部分计算 RMSE。
- 使用 EMA 平滑。
- consistency 高于阈值时触发慢系统。
- timeout 也会触发慢系统。

辅助头融合：

- `state=success` 且置信度高于阈值，触发 `suspect_complete`
- `state=retry` 且置信度高于阈值，触发 `suspect_fail`
- `progress` 不触发 VLM

典型事件：

```json
{
  "event_type": "fast_signal",
  "payload": {
    "trigger": "suspect_complete|suspect_fail|timeout",
    "score": 0.1,
    "aux_state": "success|retry|progress"
  }
}
```

## 7. VLM State Decision

实现：

```text
scripts/long_horizon_controller/vlm_verifier.py
```

Ollama 使用 structured output：

- `VLM_DECISION_JSON_SCHEMA`
- `RECOVERY_JSON_SCHEMA`

主 VLM prompt 必须先独立判断两个条件：

```text
Condition A:
只检查当前子任务 expected_finish_state

Condition B:
只检查下一子任务 expected_start_state
```

判定规则：

```text
complete:
  finish_state_satisfied=true
  AND next_start_plausible=true

in_progress:
  当前 finish 未满足，但仍在可信地向当前 finish 前进

failed:
  当前 finish 未满足，且动作不再可信或需要恢复
```

特别注意：

- 下一子任务的 `expected_finish_state` 不参与当前判断。
- `not/no/without/absent/away from` 是显式否定条件。
- “不在目标位置”可能是下一子任务的正确开始条件，不能自动当成失败。

之前出现过的错误：

```text
Robot is holding a straw, but it is not at the dining counter.
finish_state_satisfied=false
```

对于 `pick_straw_from_drawer` 来说，这通常是 VLM 把下一子任务的负条件误判成失败，而不是看不到吸管。prompt 已经改为通用的 A/B 条件判定，不包含具体任务名。

## 8. Recovery Logic

恢复 prompt 要求 VLM 严格二选一：

### `insert_recovery`

适用于：

- 物体掉落
- 抓错物体
- 目标错误
- 需要重新抓取
- 需要主动移动到新的恢复状态

VLM 输出 1 到 3 个恢复子任务，controller 将它们插入当前子任务之前：

```text
recovery_subtasks + current_subtask + remaining_queue
```

### `rollback_retry`

适用于：

- 最近动作导致姿态偏离
- 反向执行近期动作可能回到可用状态
- 不需要主动重新抓取或重新定位

VLM 输出：

```json
{
  "recovery_mode": "rollback_retry",
  "rollback_steps": 2,
  "recovery_subtasks": []
}
```

controller 会：

1. 记录过的动作块按时间倒序取出。
2. EEF position、EEF rotation、base motion 取负并反转时间顺序。
3. gripper 和 control mode 保持最近状态，避免回滚时意外松爪。
4. 将当前子任务重新放回 queue。
5. 重新执行当前子任务。

默认最多回滚 8 个 action chunks；建议实验先使用：

```text
--max-rollback-chunks 2
```

当前 action chunk 通常包含 `n_action_steps=16` 个仿真动作步。回滚是 delta action 的开环近似，不是仿真状态的精确恢复。

事件：

```text
rollback_step
rollback_retry
rollback_unavailable
insert_recovery
```

## 9. Output Files

每个任务/episode 目录通常包含：

```text
controller_events.json
action_history.json
plan.json
stats.json
videos/
vlm_frames/    # 如果启用 --save-vlm-frames
```

`controller_events.json` 记录：

- `start_subtask`
- `fast_signal`
- `vlm_decision`
- `advance_subtask`
- `insert_recovery`
- `rollback_step`
- `rollback_retry`
- `env_done`
- `finish_plan`

最终成功必须看：

```text
env_done.payload.success
```

或者环境成功状态提取结果，不能只看 `finish_plan.controller_completed_plan`。
