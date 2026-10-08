# Franka to UR3e IK Replay

Franka 到固定底座 UR3e 的第一阶段验证代码。

## 实验逻辑

1. 读取 RoboCasa LeRobot 专家 episode 的 `states.npz`、`model.xml.gz` 和 12 维动作。
2. 使用专家的精确 MuJoCo 状态提取 Franka 的 grip-site 轨迹。
3. 创建相同 RoboCasa 场景，并复制家具/物体关节状态。
4. 初始对齐时允许 UR3e 底座在 XY 平面移动。
5. 初始对齐完成后锁定底座，只对每一帧目标 EEF pose 做数值 IK。
6. 默认保留专家轨迹的二值夹爪开合时序，或可选择将 Panda 手指开口
   映射为 Robotiq 2F-85 的归一化控制信号。

默认 `teleport` 模式先直接应用 IK 解，再推进一帧夹爪/场景物理，用来判断
UR3e 的几何可达性。`--execution-mode controller` 使用有限速率的关节位置
控制器，更接近真实执行，但会额外引入控制跟踪误差。保存视频时默认输出
`512x512` 分辨率，并同时生成主视角和 eye-in-hand 视角。

初始对齐和后续轨迹默认使用两侧指尖 pad 中点，而不是直接使用 grip-site。
这是因为 Franka Panda 和 Robotiq 2F-85 的 grip-site 到实际接触面的偏移不同。
回放开始前会让 Robotiq 被动关节短暂稳定，避免 reset qpos 与张开命令之间产生
可见的夹爪跳变。`replay.json` 还会记录夹爪 qpos、pad 中点误差和
`sponge_main` 的逐帧位移。

抓取建立后，回放器会先允许专家完成“从侧面拿起百洁布”的动作；当百洁布
中心高于砧板后，启用砧板防穿透投影。若百洁布或 Robotiq 碰撞几何的最低点
低于砧板顶面，系统会沿世界坐标 `+Z` 抬升 EEF 目标，并重新求 IK，再按抓取
锁的相对位姿带动百洁布。这样百洁布和夹爪一起移动，不会通过直接单独修改
百洁布位置而破坏两侧夹持关系。每帧的修正量记录在 `replay.json` 的
`collision_projection` 字段中。

释放阶段也采用显式状态机：收到专家的张开信号后，先保持最后一个安全 EEF
姿态，并将 Robotiq 六个关节平滑插值到全开位姿；确认指尖间距达到阈值后，
才解除百洁布抓取锁并抬起机械臂。释放后的夹爪保持全开，且末端高度不会低于
砧板安全高度，避免“夹爪尚未松开就下压”造成穿模。

## 运行

```bash
cd /data/zjw/workspace/Isaac-GR00T
conda run -n robocasa python -m scripts.franka_to_ur3e_ik.run_scrubcuttingboard \
  --episode-index 376 \
  --save-video
```

如需测试旧的 Panda 开口量映射：

```bash
conda run -n robocasa python -m scripts.franka_to_ur3e_ik.run_scrubcuttingboard \
  --episode-index 376 \
  --gripper-control-mode state_mapping
```

无显示服务器时可加 `MUJOCO_GL=osmesa`。保存视频需要可用的 EGL 或 OSMesa
图形后端。

默认数据集和输出目录就是 `ScrubCuttingBoard` 的 episode 376。
视频文件为 `ur3e_replay.mp4` 和 `ur3e_eye_in_hand_replay.mp4`。

## 重要约定

- `base-offset DX DY DZ` 只影响 reset 时的初始安装位置。
- 轨迹开始后，底座不会再作为 IK 自由度。
- 当前验证目标是 EEF 轨迹复现；Robotiq 与 Panda 的指尖几何不同，因此夹爪开口映射是独立的一项指标。
- 默认 `source_action` 会保留专家 `-1=open / +1=close` 的时序。
- `--no-pad-midpoint-alignment` 可关闭 pad 中点补偿，用于对比实验。
- 输出 `replay.json` 包含每帧 target/actual pose、IK 误差、关节目标和夹爪信号。
