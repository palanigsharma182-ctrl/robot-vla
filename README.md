# qwen-vla-v0.1

本项目第一版目标是在单张 RTX 4090 24GB 上建立一条可训练、可闭环验证的
Franka Panda 双相机 VLA 链路：

```text
4-step External/front RGB + Wrist RGB + Language
  + Proprioception + TCP/Wrist-camera pose + F_L/F_R + Controller state
  -> Frozen Qwen3.5-2B
  -> QwenVLAAdapter
  -> SmolVLA-style standalone Action Expert
  -> Rectified Flow Action Chunk [16,8]
  -> 执行前 4 步
  -> 使用新观测重新规划
```

旧的单相机 ResNet-18 + DistilBERT + Smooth L1 确定性 Baseline 已经删除，不再作为
可运行实现或兼容目标。V1 模型架构标识为 `qwen_vla_late_fusion_v1`；最小可部署状态 amendment
使用显式不兼容的 `qwen_vla_temporal_state_fusion_v2`。两者共用 `robot-vla-trajectory/v2`
容器，但 V2 observation 必须由额外的版本化数组和统计身份声明，不能从旧数据推断。

## 当前状态

已经完成：

- `qwen-vla-v0.1` 的架构、数据、Flow Matching、训练和执行语义决策；
- 大陆 Ubuntu 22.04 RTX 4090 VM 的 CUDA 12.8、Vulkan、Qwen Processor、
  ManiSkill 和 SAPIEN 环境验证；
- 可复现的 GPU VM 构建与镜像清单脚本；
- ManiSkill Franka `RobotSpec`、Task/Skill 版本身份和双指夹爪状态映射；
- `robot-vla-trajectory/v2` manifest、双相机数组结构、严格校验和 LRU Store；
- ProprioStats/Normalizer、物理 Action/模型 Action/ManiSkill Action Adapter；
- 双图 Action Chunk Dataset、Episode 尾部 mask、Qwen Processor Collator；
- Task → Episode → timestep 平衡采样及可记录的技能重采样权重；
- 固定 Prompt/Processor、Qwen 架构校验与精确 revision BF16 权重加载工厂；
- Frozen Qwen Context、2048→720 Adapter、16 层 standalone Action Expert；
- masked Rectified Flow 训练目标、10 步 Euler 采样和 Cross-Attention K/V 复用；
- Stage 1 AdamW/warmup/cosine、BF16、有效 Action 加权的 Gradient Accumulation；
- 固定 seed 验证，以及包含完整契约、训练状态和 RNG 的 latest/periodic/best Checkpoint；
- 在线 Runtime、独立可追踪采样 seed、D017 前 4 步滚动执行和失败 hold；
- Action 标签/执行统一为 commanded-target 增量，并跨 Replan 保存 command reference；
- Observation V2：TCP 完整位姿、OpenGL→OpenCV→base 相机变换、四步双图、`F_L/F_R`、
  时间/validity 和 controller state，以及 train-only force stats/checkpoint identity；
- E013 的厘米级闭环精调架构：低频 VLA + 20 Hz 三头 U-Net、显式 base-plane 几何、逐轴不确定性和
  shadow-only metric residual 控制契约；正式目标为 `p50<=12 mm/p90<=20 mm`，工程底线为
  `p50<=15 mm/p90<=25 mm`，当前仍只是工程 scaffold，尚无闭环效果证据；
- ManiSkill 单环境 Franka `pd_joint_delta_pos` Controller Adapter；
- reach/grasp/lift/transport/place Outcome Predicate、原子技能状态机和完整组合任务；
- 目标双相机可见、必须松爪并稳定放置的 `RobotVLAPickCubeToRegion-v1` 环境；
- MPlib 专家双相机采集、原子 NPZ/manifest writer、scene 级 split 和完整数据审计；
- 第一批 30 条、5996 控制步的可信数据（24/3/3）及 train-only ProprioStats；
- 固定 Qwen revision 下载、RTX 4090 显存测量和五技能小数据过拟合验证；
- Stage 1 正式训练入口、源码树 revision、实验身份、指标、Checkpoint 和断点恢复。
- 推理态 Checkpoint 契约验证、逐控制步 Outcome Predicate 闭环 Rollout；
- test/unseen seed 分组、Flow seed 追踪、原子技能通过率、Wilson 区间和失败分类；
- 可恢复的闭环评估 CLI，以及按实际 periodic 权重重新选择候选 Checkpoint 的工具；
- 显式且保持 scene split 不变的可信数据集扩充门槛；
- 220 条可信轨迹的 v0.4 数据集，以及 grasp/release、contact、速度突变、pickup/place 事件标签；
- 完整 16-step BC/Flow loss 加前 4 个实际执行步关键事件 loss，并完成 A–F 权重消融；
- 最新 Chunk 占主导的 temporal ensemble，以及异常清空历史并重新请求 VLA 的控制链路；
- 固定 `lambda_event=0.25`、从随机初始化开始的 100-epoch 正式训练和完整产物审计；
- 正式模型 25 个原子闭环、20 unseen 完整闭环及三组控制层消融。

第一版正式结果：

- 正式 best 为 epoch 98，验证总/base/event loss 分别为 `0.03085/0.02318/0.03287`；
- 原子技能为 reach `0/5`、grasp `5/5`、lift `5/5`、transport `2/5`、place `4/5`，总计 `16/25`；
- 默认 ensemble+replan 的 20 unseen 完整成功仍为 `0/20`，阶段通过为 `5/4/3/1/0`；
- newest-only 只有 reach `1/20`，证明 temporal ensemble 明显改善前半程稳定性；本批 seed 未触发
  anomaly replan，因此重规划预算的行为收益尚未被这组消融覆盖。

第一版工程链路和预定实验已经完成，但不能把它描述为 manipulation 成功：完整任务成功率仍为 0，
主要瓶颈是 reach 泛化，其次是 transport/release 的连续组合。后续迭代应继续从数据覆盖和监督分布
解决这些学习问题，而不是加入 stable-grasp、release-hold 或 settle 等任务语义状态机。

后续 E008 对 Qwen Layer 12 做了受控空间与闭环诊断：线性 probe 的 test median world-XY error
从 Layer 24 的 `0.1245 m` 降到 `0.0253 m`，Reach-only 从 `1/5` 提高到 `2/5`；但五技能联合训练
的原子总成功仍为 `16/25`，20 unseen 完整成功仍为 `0/20`。完整阶段由 Layer 24 的
`5/4/3/1/0` 变为 Layer 12 的 `9/3/2/0/0`，说明几何改善没有稳定传递到 Grasp/Transport
交接。E009 随后对 11 个 periodic/best checkpoint 做了 66-Episode screening 和 60-Episode 独立
confirmation：epoch 100 把 Reach 从 epoch 98 的 `0/10` 提到 `3/10`，但 Transport 从 `7/10`
降到 `2/10`，没有单一 checkpoint 通过 promotion 门槛。这说明聚合 checkpoint 在不同技能间存在
行为冲突，不能靠替换 `best.pt` 解决。E010 随后用 34 个严格配对的 raw Gradient Gram 做机制归因：
e098-best/e100 的 train Reach–Transport median cosine 为 `+0.164/+0.173`，独立 val 为
`-0.094/+0.441`，均未通过预注册的两阶段负冲突门槛；五技能所有 train pair median 也全部为正。
因此当前不直接增加多动作头或 PCGrad/CAGrad。下一步先投影 epoch 98→100 的真实 checkpoint 位移，
并补充 event-conditioned 与 handoff boundary probe，再决定是否需要训练目标或架构分支。当前仍不把
Layer 12 升级为默认 Context。
完整配置、结果和限制见 [E008](docs/experiments.md#e008--qwen-layer-12-空间表示reach-与五技能组合诊断)、
[E009](docs/results/e009/README.md)、[E010](docs/results/e010/README.md) 与
[D024](docs/decisions.md#d024--不直接以-layer-12-替换最终层先诊断技能交接再评估语义-key--几何-value)。

## 快速开始

基础开发和单元测试使用 Python 3.10 及以上版本：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest -q
```

需要加载固定 Qwen3.5 权重并运行 ManiSkill 闭环时安装完整可选依赖：

```bash
python -m pip install -e ".[dev,qwen-vla,sim]"
```

仓库只包含源码、测试、环境构建脚本和可复现实验记录，不包含 Qwen 权重、可信训练数据、训练
Checkpoint 或闭环产物。Qwen、ManiSkill、SAPIEN 及其他第三方组件继续受各自许可证约束。
这是研究与工程原型，不应未经独立安全验证直接控制真实机器人。

## 固定契约

- 机器人：ManiSkill Franka Panda，7DoF Arm + 平行双指夹爪；
- 相机顺序：external/front 在前，wrist 在后，不得交换；
- 状态：`q[7] + dq[7] + g[1]`，共 15 维；
- V2 历史：最近 4 个连续控制步，oldest-to-newest；Episode 开头零 padding，不复制首帧；
- V2 位姿：`base_from_tcp` 与 `base_from_wrist_camera_cv`，模型使用 position + Rotation-6D；
- V2 接触：左右 finger–cube pairwise force magnitude `F_L/F_R`；它是仿真近似，不是真实应变片；
- 动作：`command_target_delta_q[7] + gripper_target[1]`，共 8 维；controller correction 是独立的
  `commanded_target - actual_q`；
- 精密层动作：`commanded TCP target delta [base dx/dy/dz meter + dyaw radian]`；与 VLA joint Action
  隔离，Motion Head 第一阶段只允许 shadow；
- 控制频率：20 Hz；Action Horizon：16；每次执行前 4 步；
- Qwen：`Qwen/Qwen3.5-2B`，固定 revision
  `15852e8c16360a2fea060d615a32b45270f8a8fc`；
- Prompt：V1 为 `qwen-vla-prompt/v1`，V2 为 `qwen-vla-prompt/v2-history4`；
- Dataset：`robot-vla-trajectory/v2`；
- 模型：`qwen_vla_late_fusion_v1`；
- 第一阶段冻结 Qwen，只训练 Adapter、State/Action Token、Action Expert 和输出头。

最小可部署状态、坐标公式、Action 等式、迁移与验证门禁见
[`docs/minimum_deployable_state.md`](docs/minimum_deployable_state.md)。其他边界见
[`docs/architecture.md`](docs/architecture.md) 和 [`docs/decisions.md`](docs/decisions.md)。
厘米级闭环精调层的当前实现边界和未完成门禁见
[`docs/e013_precision_execution.md`](docs/e013_precision_execution.md)。
E013 之后候选的 `qwen-vla-v1.0` 分层子任务、阶段控制与渐进验证事项见
[`docs/roadmap.md`](docs/roadmap.md)；该 Roadmap 仅是计划，不属于当前实现或效果结论。

## 环境验证

大陆 RTX 4090 完整虚拟机使用 `infra/gpu_vm` 的 CUDA 12.8 profile。服务器上执行：

```bash
cd /path/to/robot-vla

/opt/robot-vla/env/bin/python \
  infra/gpu_container/verify_runtime.py \
  --runtime-profile vm-cu128 \
  --qwen-config \
  --maniskill
```

这条检查验证 PyTorch/CUDA、BF16、固定 Qwen 配置、NVIDIA Vulkan 以及 ManiSkill
`PickCube-v1` RGB 环境和一帧离屏渲染，但不会下载 Qwen 模型权重。

## 开发顺序

新实现按依赖方向渐进落地：

```text
RobotSpec / TaskSpec
  -> trajectory/v2 schema + validator
  -> Franka Observation / Action Adapter
  -> Qwen Prompt / Processor contract
  -> Qwen Context Encoder + Adapter
  -> Flow Matching primitives
  -> standalone Action Expert
  -> Dataset / Collator / training
  -> Checkpoint / Runtime
  -> ManiSkill rollout / closed-loop evaluation
```

每一层先用小型 Tensor 和合成轨迹验证 shape、mask、时间方向与单位，再接入下一层；在真实
或仿真闭环成功率出现之前，离线 Loss 和 smoke test 不能被解释为 manipulation 成功。

## 闭环评估入口

如果一次旧训练只按周期保存权重，先从实际存在的 periodic Checkpoint 中选择验证 loss 最低者：

```bash
python -m robot_vla.cli.select_stage1_checkpoint \
  --run /path/to/stage1-run \
  --output /path/to/stage1-run/periodic-selection.json
```

正式评估默认运行全部 test 轨迹和 20 个不在 Dataset manifest 中的新 seed；每完成一个 Episode
都会原子追加结果并更新汇总，进程中断后可使用 `--resume`：

```bash
python -m robot_vla.cli.evaluate_maniskill \
  --data /path/to/trusted-dataset \
  --model-cache /path/to/huggingface-cache \
  --checkpoint /path/to/selected-checkpoint.pt \
  --output /path/to/rollout-run \
  --inference-strategy temporal-ensemble \
  --unseen-seed-start 10000 \
  --unseen-episodes 20
```

`--inference-strategy` 可显式选择 `newest-only`、`temporal-ensemble` 或实验性的 `rtc`；RTC 额外支持
`--rtc-execution-horizon 4 --rtc-max-guidance-weight 10.0`。`summary.json` 分别保存
test/unseen/overall 的完整任务成功率、95% Wilson 区间、条件交接率、阶段耗时、五个原子技能通过率
和失败计数；`episodes.jsonl` 保留每次 Replan 的 Flow sampling seed、策略/RTC 诊断、最终物理
Predicate 和失败阶段。只有这组闭环结果可以作为 manipulation 效果证据。

## 许可证

本项目以 [Apache License 2.0](LICENSE) 开源。Qwen、ManiSkill、SAPIEN 及其他第三方依赖不因
本仓库许可证而重新授权，使用时应同时遵守其各自的许可证和使用条款。
