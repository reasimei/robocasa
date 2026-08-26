# HANDOFF

## 1. 任务是什么
在Robocasa365的benchmark上尽可能提高机械臂长程操作的成功率
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

以上是总体目标，当前已经完成了 辅助头 success和progress的训练 正在测试训练的结果


## 2. 关键目录 / 文件

### 数据 / checkpoint

- Robocasa atomic success 数据：
  - `/data/zjw/workspace/robocasa/datasets/v1.0/target/atomic`

- 基础 GROOT checkpoint：
  - `/data/zjw/workspace/Isaac-GR00T/expdata/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000`

- manifest：
  - `/data/zjw/workspace/Isaac-GR00T/expdata/aux_progress/atomic_positive_manifest.json`

### auxiliary-head 脚本
   - 不改 GROOT 原动作头
   - 冻结 backbone / 动作头
   - 额外训练：
     - `progress` 回归头
     - `state={progress, success}` 分类头

- manifest 构造：
  - `/data/zjw/workspace/Isaac-GR00T/scripts/aux_progress/build_atomic_positive_manifest.py`

- 训练：
  - `/data/zjw/workspace/Isaac-GR00T/scripts/aux_progress/train_atomic_positive_aux.py`

- 评估：
  - `/data/zjw/workspace/Isaac-GR00T/scripts/aux_progress/eval_atomic_positive_aux.py`

- 目前看来 `/data/zjw/workspace/Isaac-GR00T/expdata/aux_progress/atomic_positive_only_run4/checkpoint-6500` 比较好

### action-regression 脚本
   - 不再单独挂分类头
   - 把 `success` 和 `progress` 直接作为 **动作张量中的新增维度** 来训练
   - 两者都按回归理解，但：
     - `success` 真值是 `0/1`
     - `progress` 真值是 `[0,1]`
- 训练：
  - `/data/zjw/workspace/Isaac-GR00T/scripts/aux_progress/train_atomic_action_regression.py`

- 评估：
  - `/data/zjw/workspace/Isaac-GR00T/scripts/aux_progress/eval_atomic_action_regression.py`

- 还在测试，好像对原来动作有些损害，一直在漂移，加了动作输出2维之后能否还用原来的指令测试？ `python scripts/run_eval.py     --model_path expdata/action_regression/atomic_success_progress_run3/checkpoint-53000   --task_set composite_seen   --split target  --n_envs 1`


## 3. 下一步计划（推荐顺序）
帮我调查加了动作输出2维之后训出来的模型能否还用原来的指令测试？ `python scripts/run_eval.py     --model_path expdata/action_regression/atomic_success_progress_run3/checkpoint-53000   --task_set composite_seen   --split target  --n_envs 1` 
- 如果不能新写一个run_eval2.py保持原来的动作输出测试，先不管新加的2维，再进行测试
- 如果就是用run_eval测试没问题，那么表示现在微调的策略把原来的动作输出弄坏了，可能需要用回auxiliary-head 脚本，不改变动作头，冻结动作额外训练




