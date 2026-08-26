# 系统架构

## 文档边界

本文记录项目中相对稳定的系统边界、主要模块、数据流和依赖关系。

项目已经固定并实现 `qwen-vla-v0.1` 的核心数据、策略、训练、Checkpoint、Runtime、安全执行、
ManiSkill 可信数据采集、Outcome Predicate 和闭环失败分析边界；旧的单相机确定性 Baseline 已
删除。正式闭环实验和扩充数据重训仍在进行，World Model 与候选比较 Planner 仍处于长期目标
架构阶段。因此，下文区分**已经接受的契约**、**实际已实现能力**和长期目标，不能把离线 loss
或 smoke test 解释为任务已经成功。

实验配置、临时实现细节和单次结果不记录在这里，分别放入 [experiments.md](experiments.md) 和代码本身；重要架构取舍记录在 [decisions.md](decisions.md)。

## 项目目标

项目面向 VLA、世界模型和机器人仿真的端到端学习与工程实践，目标主链路为：

```text
VLA
  -> World Model
  -> Evaluator
  -> Planner
  -> Simulator
  -> Data / Training Loop
```

系统最终关注闭环任务成功率，以及新增模块为什么能够改善或损害成功率，而不只关注离线模型指标。

## 主要模块

| 模块 | 主要职责 | 实现边界 |
| --- | --- | --- |
| Task Definition | 定义任务、成功、失败和进度语义 | TaskSpec、五技能状态机和逐控制步 Outcome Predicate 已实现 |
| Observation Adapter | 将仿真器或数据集的原始观测转换为模型使用的统一表示 | Franka 15 维状态、双图 Processor 与 Runtime 已实现 |
| VLA | 根据任务和观测生成动作或 Action Chunk 候选 | Frozen Qwen Late Fusion + Flow Expert 已实现 |
| Action Adapter | 将模型动作表示转换为控制器或仿真器动作 | 物理/模型/ManiSkill 映射与 D017 Executor 已实现 |
| World Model | 根据当前状态和动作预测未来状态或结果 | 项目核心，Dynamics 需要自行实现和验证 |
| Evaluator | 对预测或真实轨迹判断成功、失败、进度、风险或质量 | 项目核心，评价语义必须与任务定义一致 |
| Planner | 基于候选动作、世界模型预测和评价结果进行选择、重试或重规划 | 项目核心，策略逻辑需要显式实现 |
| Simulator / Controller | 执行动作并产生下一时刻观测和环境反馈 | 必须释放的桌面环境、MPlib 采集器和滚动控制已实现 |
| Dataset Pipeline | 采集、对齐、校验并组织轨迹数据 | trajectory/v2、双相机采集、审计、Dataset/Collator/Sampler 已实现 |
| Training Loop | 训练或微调 VLA、World Model、Evaluator 等可学习模块 | VLA Stage 1 训练/验证/Checkpoint 已实现 |
| Evaluation / Failure Analysis | 运行基线与候选系统，统计任务成功率并归因失败 | test/unseen seed Rollout、原子技能指标和失败分类已实现；正式结果待运行 |

## 在线闭环数据流

下面是概念级数据流；名称表示跨模块语义，不代表已经存在同名代码类型。

```text
Task Definition ------------------------------------------------------+
       |                                                              |
       v                                                              v
Simulator -- Raw Observation --> Observation Adapter --> VLA --> Action Chunk 候选
    ^                                                       |          |
    |                                                       v          |
    |                                                 World Model      |
    |                                                       |          |
    |                                               Predicted Rollout  |
    |                                                       v          |
    |                                                   Evaluator <----+
    |                                                       |
    |                                                  Evaluation
    |                                                       v
    +-- Controller <-- Action Adapter <-- Plan Decision <-- Planner
```

每次仿真步或控制周期产生的观测、动作、评价和环境反馈组成 Transition；连续 Transition 构成 Trajectory，并进入数据与评估链路。VLA v0.1 固定为 20 Hz 控制、执行 4 步后重新观测；未来 Planner 是否比较多个候选以及 World Model 的预测粒度仍未确定。

## 数据与训练流

```text
Simulator / Rollout
    -> Transition / Trajectory 采集
    -> 时间对齐与有效性检查
    -> Dataset Pipeline
    -> 训练 / 微调 VLA、World Model、Evaluator
    -> 离线指标与闭环任务评估
    -> Failure Analysis
    -> 新数据采集或下一轮训练
```

这条链路必须保留从训练样本回溯到原始轨迹、任务定义和评价结果的能力。数据过滤、Label 生成或失败样本重采样都属于会影响训练语义的操作，应当可追踪。

## 关键接口边界

以下是需要稳定下来的概念接口。当前状态明确区分 VLA v0.1 已实现部分与完整目标系统的待定部分。

| 概念接口 | 生产者 -> 消费者 | 必须表达的语义 | 当前状态 |
| --- | --- | --- | --- |
| Task Specification | Task Definition -> VLA / Evaluator / Planner | 任务目标、成功、失败和进度定义 | TaskSpec、Skill 映射和 pick-place Predicate 已实现 |
| Raw Observation | Simulator / Dataset -> Observation Adapter | 原始模态、采样时间、来源和坐标信息 | trajectory/v2 双相机时间戳、校验和 Store 已实现 |
| Model Observation | Observation Adapter -> VLA / World Model | 对齐后的模型输入及有效性信息 | 双 RGB/语言 Processor、Collator 和 15 维状态适配已实现 |
| Action Chunk | VLA / Planner -> World Model / Action Adapter | 一段有序动作及其时间语义 | `[B,16,8]` Flow 生成、temporal ensemble、反归一化与执行前缀已实现 |
| Predicted Rollout | World Model -> Evaluator / Planner | 在给定状态和动作下预测的未来结果及有效范围 | 状态表示与预测 Horizon 待定 |
| Evaluation | Evaluator -> Planner / Evaluation Pipeline | 成功、失败、进度或其他决策依据 | 真实 Rollout 的任务/技能成功率、Wilson 区间和失败分类已实现 |
| Plan Decision | Planner -> Action Adapter | 执行、重试、重规划或终止决策及选中动作 | 任务语义 Planner 待定；数值/安全/跟踪异常后的清空历史与 VLA 重规划已实现 |
| Transition / Trajectory | Simulator / Controller -> Dataset / Evaluation | 时间对齐的观测、动作、反馈、任务和终止信息 | trajectory/v2、可信采集器和在线评估 Episode 日志已实现 |

## Tensor 与数据的主要流向

`qwen-vla-v0.1` 已实现的直接策略流向如下：

```text
External RGB + Wrist RGB + Language
  -> qwen-vla-prompt/v1 + Qwen Processor
  -> Frozen Qwen3.5-2B final hidden states [B,N_context,2048]
  -> QwenVLAAdapter [B,N_context,720]
Proprio physical [B,15] -> train-split statistics -> normalized [B,15]
Normalized Action Chunk [B,16,8] + Flow Time + Noise
  -> State/Action/Time Tokens
  -> 16-layer standalone Action Expert
  -> Flow Velocity [B,16,8]
  -> 10-step Euler integration -> normalized Action Chunk [B,16,8]
  -> Action Adapter -> physical Action Chunk [B,16,8]
  -> execute first 4 steps -> re-observe/replan
```

Proprio 顺序为 `7 q + 7 dq + gripper opening ratio`。物理 Action 顺序为
`7 delta_q(rad/control-step) + normalized gripper target`，关节增量上限默认每步 0.05 rad，
并受更严格的机器人速度和位置限制约束。控制频率为 20 Hz，Action Horizon 为 16，默认
5 Hz 重规划。这些属于 `robot-vla-trajectory/v2` 受控契约。

Stage 1 的训练目标保留全部 16 个有效 Action step 的 masked Rectified Flow/BC loss，并只对
实际执行前 4 步中由 GT 专家轨迹检测到的关键事件增加损失：

```text
L = L_base + lambda_event * L_event
critical_mask = event_mask & action_mask & [1,1,1,1,0,...,0]
```

在线执行按全局控制时刻保存重叠 Chunk proposal；默认以 `recency_decay=0.5` 加权，最新到更旧
Chunk 的未归一化权重为 `1/0.5/0.25/0.125`，四个 proposal 并存时最新预测占约 53.3%。融合在
归一化 Action 空间完成，再交给既有 Action Adapter。推理、安全或跟踪异常会清空旧 proposal，
使用新观测重新请求 VLA；控制层不加入 stable-grasp、release-hold 或 settle 等任务语义状态机。

完整目标系统仍遵循以下语义流向：

```text
仿真原始数据
  -> Observation Adapter
  -> 模型 Tensor / Tensor Tree
  -> VLA
  -> Action Chunk Tensor
  -> World Model 预测表示
  -> Evaluator 输出
  -> Planner 决策
  -> Action Adapter
  -> 仿真器控制输入
```

`qwen-vla-v0.1` 的 Action Chunk 已固定为 `[B,16,8]`；World Model、Planner 及未来其他机器人
形态的动作表示仍未确定，不能自动沿用这一 shape。

时间戳、采样频率、Padding/Mask 和 Episode 边界应与 Tensor 一起流动，不能在 Adapter 或 Dataset Pipeline 中丢失。

## 模块依赖关系

- VLA 依赖稳定的 Task Specification、Model Observation 和 Action 定义。
- World Model 依赖与真实执行一致的状态、动作、时间步长和坐标语义。
- Evaluator 依赖明确的 Task、Success、Failure 和 Progress 定义。
- Planner 依赖 VLA 候选、World Model 预测、Evaluator 输出及可执行的控制约束。
- Dataset Pipeline 同时依赖 Observation Adapter、Action Adapter、时间对齐规则和任务 Label。
- Training Loop 依赖可追溯的数据版本、模型配置和评价标准。
- Evaluation / Failure Analysis 依赖固定基线、任务分布、随机种子策略和成功率定义。

PyTorch、Transformers、LoRA、ManiSkill、Isaac Lab、预训练视觉/视频 Encoder、物理引擎、IK 和标准训练工具属于优先复用的基础设施；项目核心模块不能因此退化为不可检查的黑盒调用。

## 整体架构约束

- 默认目标硬件为单张 RTX 4090 24GB。
- 默认优先 BF16、Frozen Backbone、LoRA、Gradient Accumulation、小型可训练模块、分阶段训练和按需加载。
- 不默认依赖多卡或 80GB GPU。
- Observation / Action、Action Horizon、Task、Dataset、Loss、训练目标、Planner 和 Evaluation 属于跨模块契约，不得静默修改。
- 重要架构选择及其替代方案记录在 [decisions.md](decisions.md)。

## 待实现后补全

当对应代码首次落地时，应逐项补充而不是预先猜测：

- World Model、Evaluator 和 Planner 的实际包与接口；
- 具体仿真器/控制器坐标系、关节名称及夹爪映射；
- 历史观测，以及 temporal ensemble 之外的动作平滑策略；
- World Model/Planner 引入后的预测风险、候选比较和 Plan Decision 语义。

新实现继续使用以下代码边界：`robot_vla.contracts`（稳定语义）、`adapters`（归一化与
Franka 映射）、`data`（trajectory/v2 和训练窗口）、`model`（Qwen Context、Adapter 与
Action Expert）、`training`（Flow Matching 和训练循环）、`runtime` 与 `execution`（在线接入）。
其中 `contracts`、`adapters`、`data`、`model`、`training`、`runtime`、`execution`、ManiSkill
可信采集和真实 Rollout Evaluation 已完成第一版实现；220 条事件数据、100-epoch 正式训练和
三组控制消融也已完成。当前行为瓶颈是未见 seed 的 reach 泛化和后续技能组合，而不是链路缺失；
World Model 与候选比较 Planner 不属于 `qwen-vla-v0.1` 的完成条件。
