# Isaac-GR00T 长程机器人操作代码说明

本文档说明当前仓库的长程操作实现。英文版本保存在 README.english.backup.md，最初版本保存在 README.original.backup.md。

## 总体流程

系统由 VLA 策略、辅助头、快速动作监控器和较慢但可靠的 VLM 验证器组成。planner 将完整任务拆成 SubtaskSpec；controller 逐个执行子任务，把 instruction 送入策略模型，执行 action chunk 并记录动作历史。fast monitor 发现疑似阶段转换、完成、失败或超时后触发 VLM。VLM 返回 complete、in_progress 或 failed：complete 推进队列，failed 进入恢复，in_progress 继续当前子任务。

核心文件：

- scripts/long_horizon_controller/controller.py：主循环、队列、动作历史、推进和恢复。
- scripts/long_horizon_controller/planner.py：任务拆解。
- scripts/long_horizon_controller/schemas.py：TaskPlan、SubtaskSpec、VLMDecision、RecoveryMode。
- scripts/long_horizon_controller/run_composite_seen_eval.py：综合长程评测。
- scripts/long_horizon_controller/run_robocasa_controller.py：RoboCasa 入口。

## 辅助头

辅助头运行在冻结的策略 backbone 上，预测 progress、success、retry，不直接替代动作策略。

在线实现：

- scripts/long_horizon_controller/policy_adapters.py：Gr00tAuxHeadRuntime 从 GR00T backbone 提取 pooled hidden representation；Gr00tPolicyAdapter.act() 同时产生动作和辅助预测。
- scripts/long_horizon_controller/xiaomi_aux_runtime.py：Xiaomi/XR-1 辅助头运行时。
- scripts/long_horizon_controller/fast_monitor.py：融合辅助头和快速动作信号。

数据和训练脚本位于 scripts/aux_progress/：

- build_composite_subtask_manifest.py
- build_composite_gt_aux_manifest.py
- build_atomic_positive_manifest.py
- train_manifest_subtasks_aux.py
- train_atomic_positive_aux.py
- train_composite_gt_aux.py
- train_xiaomi_aux.py

训练时冻结 GR00T/Xiaomi backbone，只更新轻量 progress/state heads。产物为 aux_heads.pt 和 aux_config.json。在线 adapter 根据 aux_config.json 恢复相同的 observation history，保证训练和推理输入一致。

## 动作熵与快速触发

动作不确定性/动作熵属于快环，用于低成本发现动作停滞、异常或疑似完成。

- scripts/long_horizon_controller/fast_monitor.py：FastTrigger 包含 SUSPECT_TRANSITION、SUSPECT_COMPLETE、SUSPECT_FAIL、TIMEOUT；根据 action chunk 变化、局部进度和时间生成 fast signal，并融合辅助头输出。
- scripts/long_horizon_controller/controller.py：每次策略调用后更新 fast monitor；满足最小步数、冷却时间和触发条件时才请求 VLM。
- scripts/long_horizon_controller/run_composite_seen_fast_aux_only_eval.py：fast/auxiliary 消融入口。

快速信号只负责门控 VLM，最终完成判断由 VLM 的视觉确认负责。

## VLM 确认

scripts/long_horizon_controller/vlm_verifier.py 是慢环状态验证器：

- VLM_PROMPT_TEMPLATE 输出 complete、in_progress、failed。
- 同时检查 finish_state_satisfied 和 next_start_plausible。
- 支持 agentview、eye-in-hand，以及历史帧和当前帧的时间比较。
- 失败后使用 RECOVERY_PROMPT_TEMPLATE 生成恢复决策。
- qwen_vl_worker.py 提供本地 Qwen-VL worker。
- benchmark_vlm.py 和 test_ollama_vlm_json.py 用于 benchmark 和 JSON 协议测试。

VLM 只负责视觉状态确认，不负责连续动作控制，也不根据后续任务猜测当前任务已经完成。

## 子任务推进

schemas.py 中的 SubtaskSpec 包含：

- instruction
- expected_start_state
- expected_finish_state
- max_duration_sec
- notes

controller.py 的推进逻辑：

- self.queue 保存待执行子任务。
- complete 后将当前子任务加入 completed，递增 current_index，开始下一个。
- 新子任务开始时清空 active_action_history、fast monitor 和 VLM 图像历史。
- in_progress 重置局部动作计数并继续当前子任务。
- controller_events.json 记录事件，action_history.json 记录动作。

planner.py 负责生成初始子任务计划。

## 失败恢复

VLM 对失败严格二选一。

### rollback_retry

适合最近动作或姿态导致失败，且反向 action 足以恢复：

- VLM 输出 rollback_steps。
- controller.py::_rollback() 从当前子任务的 active_action_history 取最近动作，倒序调用环境 rollback_action()。
- 回滚后把当前子任务放回队列头部。
- --max-rollback-chunks 默认值为 8。
- 环境必须提供 supports_action_rollback=True 和 rollback_action()。
- 这是 action-level 的开环近似恢复，不是完整仿真状态恢复。

### insert_recovery

适合物体掉落、抓错物体、目标错误、抓取不稳、阻塞或需要主动重新定位：

- VLM 生成 1--3 个恢复 SubtaskSpec。
- controller 将队列改为：恢复子任务 -> 当前失败子任务 -> 原剩余队列。
- 恢复子任务完成后，重新执行原失败子任务。

恢复 prompt 和 JSON 解析在 vlm_verifier.py；队列修改、回滚、重试和事件记录在 controller.py；测试在 test_recovery.py。

## XR-1 特殊子任务 embedding 与 adapter

XR-1/Xiaomi 使用自己的视觉语言动作模型输入格式。

scripts/long_horizon_controller/xiaomi_policy_adapter.py：

- 加载 Xiaomi checkpoint 和 processor。
- 组织 agentview_left、agentview_right、eye_in_hand 三路相机。
- 组织末端、夹爪、底盘状态和历史窗口。
- 每个子任务的 instruction 都会重新通过 processor 编码为语言/任务 embedding。
- 将模型输出的 action chunk 映射回环境动作。

XR-1 评测入口：

- scripts/ur3e_robocasa_eval/run_franka_xr1_trajectory.py
- scripts/ur3e_robocasa_eval/run_xiaomi_ur3e_fixed_eval.py
- scripts/ur3e_robocasa_eval/run_xiaomi_ur3e_presoak.py

恢复子任务会重新进入 adapter，因此可以作为 XR-1 的独立语言条件动作目标执行。

## Adapter 怎么训练

这里的 adapter 特指 XR-1/Xiaomi 的子任务 embedding 对齐模块，不是辅助头。由于 XR-1 VLA 直接切换子任务语言时动作跟随效果较差，当前方法把当前 RoboCasa 子任务文本单独编码成 stage embedding，再注入 Xiaomi DiT 的 AdaLN 条件；额外训练的 adapter 学习这个 embedding 如何修正 VLA 动作输出。

实现目录：scripts/long_horizon_stage_adaln。

- model.py：StageAdaLNAdapter 将 encode_text_condition() 得到的文本 embedding 映射成 6 组 AdaLN 调制量，加到 DiT timestep embedding；最后一层零初始化，stage gate 初始较小。
- attach_stage_adaln()：冻结原始 Xiaomi VLM、DiT 和 action head，只开放 stage_adaln 参数。
- policy_adapter.py：StageAdaLNXiaomiPolicyAdapter 推理时编码 stage_text，临时注入 embedding，再生成动作；condition_scale=0 可作为原始策略基线。
- manifest.py：生成训练 manifest 并过滤目标任务。
- train_adapter.py：读取 RoboCasa 图像、状态、动作、完整任务文本和 GT 子任务文本；默认 history_frames=4、history_interval_frames=2、action_horizon=16；跨子任务边界的 action chunk 会跳过。

训练过程：完整任务文本仍作为 Xiaomi 原有 VLA 条件；当前子任务文本编码成 stage embedding；embedding 注入 DiT；在视觉、状态和 noisy action 输入下预测动作噪声/动作流；用真实 action chunk 计算损失；只更新 stage_adaln，原始 VLA 全部冻结。入口是 scripts/long_horizon_stage_adaln/train_adapter.py，主要参数包括 --manifest、--model-path、--output-root、--steps、--lr、--history-frames、--action-horizon 和 --stage-condition-format。checkpoint 只保存 adapter 权重和训练参数。

train_benefit_adapter.py 是保守版本：比较带子任务条件的 DiT 与冻结 baseline，当条件版本变差时惩罚该条件；calibrate_benefit_gate.py 用 held-out action-space 比较产生每阶段 recommended_scale；eval_composite_seen_oracle.py 使用该 gate。辅助头训练单独写在“辅助头”部分，不属于这里的 adapter。

## 旧的 UR3e 映射实验

scripts/ur3e_robocasa_eval 下的代码属于早期 Franka 到 UR3e 的仿真轨迹映射实验，与当前子任务 embedding + XR-1 AdaLN adapter 方法无关，可以忽略。

## 常用入口

    python -m scripts.long_horizon_controller.run_composite_seen_eval --model-path <checkpoint> --aux-head-path <aux-checkpoint> --output-root <output-dir>

    python -m scripts.long_horizon_controller.run_robocasa_controller --model-path <checkpoint> --aux-head-path <aux-checkpoint>

    python scripts/ur3e_robocasa_eval/run_xiaomi_ur3e_fixed_eval.py --model-path <xiaomi-checkpoint> --aux-head-path <aux-checkpoint>

    python -m pytest scripts/long_horizon_controller/test_recovery.py

## 输出与调试文件

- controller_events.json：fast signal、VLM decision、子任务推进、回滚和插入恢复。
- action_history.json：策略动作历史。
- VLM 帧和视频：检查视觉判断依据。
- aux_heads.pt、aux_config.json：辅助头 checkpoint。

建议按 controller_events.json 的时间顺序，同时查看 VLM 帧、动作历史和辅助头输出。
