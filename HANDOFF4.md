# HANDOFF4

更新时间：2026-08-25

## 1. Overall Goal

在 RoboCasa365 `composite_seen` benchmark 上提高长时域操作成功率。当前主要研究两条路线：

1. **GR00T 长程控制系统**
   - Llama 高层规划。
   - 子任务队列和 VLM verifier。
   - 快系统：action chunk consistency / auxiliary progress-retry head。
   - 慢系统：Qwen VLM 判断子任务完成、失败恢复、回滚重试或插入恢复子任务。

2. **Xiaomi Robotics-1 / XR-1**
   - 使用完整任务语言作为原始 VLA 输入。
   - GT 子任务文本经过 Qwen text backbone 和小 adapter。
   - adapter 输出 `6 x 1024` 的 AdaLN delta。
   - 通过 gate 加到 DiT 原有 timestep AdaLN 条件。
   - Oracle 根据 RoboCasa 仿真状态切换子任务。

当前新增的独立实验：

3. **UR3e 迁移验证**
   - 用官方 UR3e mesh 在 RoboCasa 仿真中固定底座。
   - 使用 Xiaomi Robotics-1 输出的 EE action 前 7 维。
   - 将 UR3e 初始 EEF 位姿通过 MuJoCo Jacobian IK 对齐到 Franka/PandaOmron 的初始 EEF。
   - 对比相同 `seed/layout/style` 下 Franka 和 UR3e 的轨迹与成功率。
   - 为什么输出末端位姿，用IK不行？

代码根目录：

```text
/data/zjw/workspace/Isaac-GR00T
```

## 2. Main Models

GR00T checkpoint：

```text
/data/zjw/workspace/Isaac-GR00T/expdata/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000
```

Xiaomi Robotics-1 checkpoint：

```text
/data/zjw/workspace/Isaac-GR00T/expdata/Xiaomi-Robotics-1-RoboCasa365
```

Auxiliary checkpoint：

```text
/data/zjw/workspace/Isaac-GR00T/expdata/aux_progress/atomic_retry_3class_history_retry_boost_run2/checkpoint-11000
```

当前常用 planner：

```text
Ollama llama3.1:70b
```

当前常用 verifier：

```text
Ollama qwen2.5vl:7b
```

启动 Ollama：

```bash
/data/zjw/bin/ollama serve
/data/zjw/bin/ollama list
```

## 3. GR00T Long-Horizon Controller

主要目录：

```text
/data/zjw/workspace/Isaac-GR00T/scripts/long_horizon_controller
```

主要入口：

```text
scripts/long_horizon_controller/run_composite_seen_eval.py
```

Llama plan cache：

```text
/data/zjw/workspace/Isaac-GR00T/expdata/long_horizon_controller/composite_seen_plan_cache_llama70b
```

每个任务的 plan 例如：

```text
composite_seen_plan_cache_llama70b/DeliverStraw/plan.json
```

运行时主要使用：

```json
plan["subtasks"]
```

`raw_response` 只是原始 Llama 输出记录，运行时不会重新使用它覆盖已修改的 `subtasks`。

plan 读取优先级：

```text
<output-root>/evals/target/<Task>/plan.json
    优先于
<plan-cache-dir>/<Task>/plan.json
```

如果手动修改了 plan cache，建议使用新的 `--output-root`，或删除旧 output root 里的任务 plan。

## 4. GR00T Main Command

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

`--n-episodes 10` 是每个任务 10 个 episode。

最终成功必须依据 RoboCasa 环境的成功值：

```text
env_done.payload.success
```

不能只看 controller 的 `finish_plan`。

## 5. GR00T Fast/Slow System

快系统：

```text
scripts/long_horizon_controller/fast_monitor.py
```

当前 action consistency 逻辑：

- 当前 action chunk 与上一 chunk 的重叠部分计算 RMSE。
- 用 EMA 平滑。
- consistency 异常时触发慢系统。
- 超时也触发慢系统。

辅助头逻辑：

- `success` 且置信度足够高：触发 `suspect_complete`。
- `retry` 且置信度足够高：触发 `suspect_fail`。
- `progress` 不触发 VLM。

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

VLM verifier：

```text
scripts/long_horizon_controller/vlm_verifier.py
```

当前使用 Ollama structured output schema：

```text
VLM_DECISION_JSON_SCHEMA
RECOVERY_JSON_SCHEMA
```

VLM 输入为两个视角、当前和历史帧。典型设置：

```text
--vlm-history-frames 2
--vlm-history-interval-sec 1.0
```

实际输入顺序：

1. 1 秒前 agentview_left
2. 1 秒前 eye_in_hand
3. 当前 agentview_left
4. 当前 eye_in_hand

VLM prompt 已要求严格 JSON、简洁回答，并分别判断：

```text
Condition A:
当前子任务 expected_finish_state 是否满足

Condition B:
下一子任务 expected_start_state 是否合理
```

当前完成切换要求：

```text
finish_state_satisfied == true
AND
next_start_plausible == true
```

不能把下一子任务的 expected_finish_state 用来判断当前子任务。

Recovery 二选一：

1. `insert_recovery`
   - VLM 生成 1 到 3 个恢复子任务，插入当前子任务之前。
2. `rollback_retry`
   - 回滚最近若干 action chunks，再重新执行当前子任务。

回滚只是 delta action 的开环近似，不是精确恢复仿真状态。

## 6. Xiaomi Stage AdaLN 结果

主要目录：

```text
/data/zjw/workspace/Isaac-GR00T/scripts/long_horizon_stage_adaln
/data/zjw/workspace/Isaac-GR00T/expdata/long_horizon_stage_adaln
```

当前 adapter checkpoint：

```text
/data/zjw/workspace/Isaac-GR00T/expdata/long_horizon_stage_adaln/target_composite_adapter_gate_h2_subtask_only/checkpoint-10000.pt
```

之前的 stage AdaLN 对比结果显示，直接注入子任务条件并不稳定：

```text
纯 Xiaomi baseline       41.25%
scale=0.10                40.00%
scale=0.25                38.75%
scale=0.50                30.62%
scale=1.00                33.75%
动态 gate                 36.875%
```

因此后续如果继续优化，建议先分析：

- 哪些 task/stage 的 adapter 有帮助。
- 哪些 stage 的条件注入会破坏原始动作分布。
- 是否需要更小的 gate、skill-level gate 或只在特定 stage 激活。

## 7. UR3e Isolated Evaluation

所有 UR3e 新实验都在以下目录中，不能修改原始 RoboCasa、GR00T、XR-1 测试：

```text
/data/zjw/workspace/Isaac-GR00T/scripts/ur3e_robocasa_eval
```

重要文件：

```text
run_xiaomi_ur3e_fixed_eval.py
run_franka_xr1_trajectory.py
ur3e_official_robot.py
ur3e_official.xml
compare_eef_trajectories.py
replay_ur3e_video.py
```

UR3e 官方 mesh 来源：

```text
/data/zjw/workspace/ur_description/meshes/ur3e/
```

当前 UR3e 使用：

```text
base-z = 0.92
base-y-offset = 0.0
```

`base-y-offset=0.0` 时，UR3e 底座在当前布局中更接近 Franka 的工作区域；旧的 `-0.3` 或 `-0.6` 会让工作空间和相机位置不合适。

## 8. UR3e Initial EEF Alignment

`run_xiaomi_ur3e_fixed_eval.py` 已增加：

```text
--align-initial-eef-to <franka trajectory json>
```

它会：

1. 读取 Franka trajectory 第一帧的：
   - `eef_pos_world`
   - `eef_quat_world_xyzw`
2. 使用 MuJoCo position/orientation Jacobian 做阻尼最小二乘 IK。
3. 设置 UR3e 六个 arm joint。
4. 刷新 composite controller。
5. 写出：

```text
initial_alignment.json
```

最近一次固定场景对齐结果：

```text
layout_id = 4
style_id = 4
position error ≈ 1.05 mm
orientation error ≈ 0.0006 rad
```

## 9. Fair Franka/UR3e Comparison

成功的原始 XR-1 baseline 视频：

```text
/data/zjw/workspace/Isaac-GR00T/expdata/long_horizon_stage_adaln/composite_seen_xiaomi_baseline_20eps/evals/target/PreSoakPan/episodes/episode_000/videos/2d1546fa-7f7a-4491-a140-b35b6bcda2ca.mp4
```

对应结果：

```json
{
  "seed": 1000,
  "env_success": true,
  "steps": 1328
}
```

从官方式 Xiaomi runner 重新构造该 episode 得到：

```text
layout_id = 4
style_id = 4
```

UR3e 当前固定场景实验：

```text
output:
/data/zjw/workspace/Isaac-GR00T/expdata/ur3e_official_stack_full/PreSoakPan/aligned_to_franka_xr1_seed1000_layout4_style4
```

结果：

```text
seed=1000
layout_id=4
style_id=4
env_success=false
steps=1600
```

固定场景后，Franka 和 UR3e 的非机器人物体位置几乎完全相同，最大差异约：

```text
8.5e-9 m
```

所以后续 UR3e 失败不能再归因于厨房场景随机不同。

主要原因：

1. Franka/PandaOmron 执行完整 12D action。
2. UR3e 只执行前 7D action。
3. Franka 有移动底盘，UR3e 是固定底座。
4. Xiaomi policy 是在 PandaOmron/XR-1 的运动学和动作分布上训练的。
5. 同一 EE delta 在 Franka 和 UR3e 上经过不同 controller、关节限位和运动学后，实际轨迹不同。

当前固定场景轨迹比较：

```text
UR3e displacement:
[-0.046, -0.150, -0.061] m

Franka displacement:
[0.187, 0.357, 0.309] m
```

因此“相同 seed + 相同 layout/style + 相同初始 EEF”仍然不能保证两个机器人执行相同轨迹。

## 10. UR3e Video Rendering Bug And Fix

固定到 `layout=4/style=4` 后，旧的 `PreSoakPan` review camera：

```python
camera_pos = sink + [-1.30, -2.00, 0.55]
```

会进入柜体/墙面，导致视频只显示灰色几何体。这是渲染相机问题，不是 action 输出问题。

已修改 `configure_review_camera()`：

```python
target = sink + [0, 0, 0.95]
camera_pos = sink + [2.20, -2.80, 1.60]
```

新增轨迹重放工具：

```text
scripts/ur3e_robocasa_eval/replay_ur3e_video.py
```

它读取已经保存的：

```text
eef_trajectory.json
initial_alignment.json
```

重新设置 IK 初始关节并重放 7D action，不重新加载 Xiaomi、不重新推理。

修正后视频：

```text
/data/zjw/workspace/Isaac-GR00T/expdata/ur3e_official_stack_full/PreSoakPan/aligned_to_franka_xr1_seed1000_layout4_style4/PreSoakPan/episodes/episode_000_seed_1000/PreSoakPan_ur3e_seed_1000_fixed_camera.mp4
```

视频已验证：

```text
512 x 512
1600 frames
80 seconds
H.264
```

重放命令：

```bash
cd /data/zjw/workspace/Isaac-GR00T

conda run -n robocasa env \
  MUJOCO_GL=osmesa \
  PYOPENGL_PLATFORM=osmesa \
  python scripts/ur3e_robocasa_eval/replay_ur3e_video.py \
  --trajectory \
  expdata/ur3e_official_stack_full/PreSoakPan/aligned_to_franka_xr1_seed1000_layout4_style4/PreSoakPan/episodes/episode_000_seed_1000/eef_trajectory.json \
  --alignment \
  expdata/ur3e_official_stack_full/PreSoakPan/aligned_to_franka_xr1_seed1000_layout4_style4/PreSoakPan/episodes/episode_000_seed_1000/initial_alignment.json \
  --output-video \
  expdata/ur3e_official_stack_full/PreSoakPan/aligned_to_franka_xr1_seed1000_layout4_style4/PreSoakPan/episodes/episode_000_seed_1000/PreSoakPan_ur3e_seed_1000_fixed_camera.mp4 \
  --task PreSoakPan \
  --seed 1000 \
  --layout-id 4 \
  --style-id 4 \
  --base-z 0.92 \
  --base-y-offset 0.0 \
  --fps 20
```

## 11. Important Caveats For Next Session

1. 原始成功视频来自：

```text
scripts/long_horizon_controller/run_xiaomi_robocasa_eval.py
```

它使用 `RobocasaVectorEnvAdapter`，而 UR3e/Franka 隔离 runner 使用手动 `env.step()` 重放 action chunk；两者执行封装不完全相同。

2. 旧的 `franka_xr1` trajectory 没有记录 layout/style 元数据。当前通过 `seed=1000` 重建时得到 `(4,4)`，但以后必须显式传：

```text
--layout-id 4 --style-id 4
```

3. 对比时必须同时固定：

```text
seed
layout_id
style_id
obj_instance_split
initialization_noise
robot base pose
camera configuration
```

4. `vlm_lm_head.weight` 未初始化的 warning 与 Xiaomi action generation checkpoint 使用无关；当前 XR-1 inference 仍可以运行。

5. OSMesa 运行中可能出现：

```text
OpenGL error 0x501 in or before mjr_makeContext
```

但如果生成的 mp4 帧数、时长和抽帧内容正常，则通常只是渲染上下文 warning。固定相机后的视频已经正常。

## 12. Recommended Next Steps

优先级建议：

1. 使用显式固定 `(layout=4, style=4)` 的 Franka runner 重新跑一遍，并保存 scene metadata。
2. 让 Franka 和 UR3e 使用相同的 observation/action chunk wrapper，而不是一个用 vector adapter、一个手动 step。
3. 对 UR3e 做 action adapter：
   - 保留完整模型输出；
   - 对前 7 维做尺度/坐标变换；
   - 明确处理 PandaOmron 与 UR3e 的 EE frame 差异。
4. 如果目标是让 UR3e 成功，不应只截取 Franka action 前 7 维；需要使用 UR3e 数据微调或训练一个 EE-to-UR3e action adapter。
5. 继续 GR00T/XR-1 子任务实验前，先确认 baseline 与 condition 使用同一 runner、同一 scene、同一 seed。
