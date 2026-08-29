# 技术决策记录

本文只记录会持续影响系统边界、数据契约、训练目标或评估方式的重要决策。小型实现调整、代码整理和单次实验配置不记录在这里。

新决策使用递增编号 `D001`、`D002`……。如果决策被替代，保留原记录，将状态改为 `superseded`，并链接替代它的新决策；不要删除历史记录。

## D001 — 复用成熟基础设施，自研核心系统

**Decision:**

PyTorch、Transformers、LoRA、ManiSkill、Isaac Lab、预训练视觉/视频 Encoder、物理引擎、IK 和标准训练工具优先复用成熟开源实现。Observation / Action Adapter、时间对齐、Action Chunk 执行、Task / Success / Failure / Progress、World Model Dynamics、Evaluator、Planner / Retry / Replan、Dataset Pipeline、Evaluation / Failure Analysis 和 Closed-loop Training 作为项目核心，由项目自行实现和理解。

**Reason:**

项目目标是掌握机器人学习系统的关键工程方法。复用通用基础设施可以把有限精力放在端到端系统的差异化部分，同时避免把开源模型当作无法解释的黑盒 API。

**Alternatives considered:**

- 从零实现 Transformer、物理引擎或 CUDA 等底层能力：投入过大，偏离项目目标。
- 将完整 VLA 或机器人系统仅作为黑盒服务调用：开发更快，但无法建立对数据、动作和闭环行为的理解。

**Status:** active

## D002 — 采用显式的 VLA 到闭环训练模块链路

**Decision:**

目标系统按 `VLA -> World Model -> Evaluator -> Planner -> Simulator -> Data / Training Loop` 组织，并保留 Observation / Action Adapter、任务定义、数据管线和失败分析等显式边界。

**Reason:**

显式边界便于独立验证数据流、Tensor shape、动作语义和各模块对任务成功率的实际贡献，也支持 Planner 的重试与重规划以及后续闭环训练。

**Alternatives considered:**

- 单体端到端模型直接从观测输出控制：链路更短，但难以隔离 Dynamics、评价和规划问题。
- 只做离线模型训练，不接仿真闭环：实现简单，但无法验证真实任务成功率和误差累积。

**Status:** active

## D003 — 以单张 RTX 4090 24GB 为默认资源约束

**Decision:**

设计、训练和评估默认应能在单张 RTX 4090 24GB 上运行。优先使用 BF16、Frozen Backbone、LoRA、Gradient Accumulation、小型可训练模块、分阶段训练和按需加载，不默认依赖多卡或 80GB GPU。

**Reason:**

这是项目的实际目标硬件。把显存和算力约束前置，可以避免形成无法在目标环境复现的架构和实验流程。

**Alternatives considered:**

- 默认全参数微调大型 Backbone：实现直接，但显存和训练成本可能超出目标硬件。
- 默认面向多卡或 80GB GPU：可支持更大模型，但降低项目的本地可复现性。

**Status:** active

## D004 — 将跨模块语义视为受控架构契约

**Decision:**

模型架构、Observation / Action 定义、Action Horizon、Task 定义、Dataset 格式、Loss、训练目标、Planner 逻辑和 Evaluation 标准不得静默修改。需要变更时，先记录问题、可选方案、权衡和推荐方案，由用户确认重要方向；确认后的长期决策再追加到本文。

**Reason:**

这些定义跨越数据、模型、控制和评估模块。局部修改可能让已有数据、Checkpoint、指标或控制行为失去可比性，且错误往往不会立即暴露。

**Alternatives considered:**

- 各模块按当前实现需要自行演进接口：短期速度更快，但容易造成语义漂移和不可复现实验。
- 只依赖代码评审发现契约变化：缺少长期、集中且可追溯的决策记录。

**Status:** active

## D005 — 以闭环任务成功率和可解释增益作为最终实验标准

**Decision:**

每个实验必须回答明确问题并保留简单 Baseline。模块价值最终通过机器人任务成功率是否真实提升及其原因来判断；离线 Loss 或代理指标不能单独作为最终结论。

**Reason:**

端到端机器人系统可能出现离线指标改善但闭环行为退化的情况。保留 Baseline、闭环评估和失败分析，才能判断新增复杂度是否值得。

**Alternatives considered:**

- 只比较训练 Loss 或单一离线指标：成本较低，但不能覆盖控制误差累积和任务级失败。
- 每次实验使用不同基线或评价口径：可能更容易展示提升，但结果不可比较。

**Status:** active

## D006 — VLA v0.1 使用关节增量连续动作块契约

**Decision:**

首版机器人固定为 7DoF Arm + 单一连续夹爪自由度。Proprioception 为
`7 q + 7 dq + gripper width + gripper velocity` 共 16 维；单步物理动作为
`7 delta_q(rad/control-step) + gripper target[-1,1]` 共 8 维。策略在 20 Hz 下输出
16 步动作块，每次执行前 4 步后重新观测。关节增量默认限制为每步 0.05 rad，模型只学习
`[-1,1]` 归一化动作，物理转换集中在 Action Adapter。

**Reason:**

关节增量不依赖首版尚未确定的末端坐标系和 IK 实现，能够先建立 RGB、语言、状态到连续
控制的闭环。16 步为策略提供 0.8 秒技能片段，4 步滚动执行将开环区间限制在 0.2 秒。
显式 Adapter 防止单位、限幅和归一化在数据、模型与控制器之间漂移。

**Alternatives considered:**

- 末端位姿增量加夹爪：更容易跨同构机械臂表达 manipulation，但必须先固定坐标系、旋转表示、IK 和不可达处理。
- 单步动作：接口简单，但不能直接学习动作块的短时协调。
- 一次执行完整 16 步：推理频率低，但视觉反馈不足，闭环误差风险更高。

**Status:** superseded by D008

## D007 — VLA v0.1 采用冻结 Backbone 的确定性 Transformer Chunk Policy

**Decision:**

视觉使用冻结的预训练 ResNet-18 空间特征，语言使用冻结的 DistilBERT token 特征，
proprioception 使用可训练 MLP。三类 token 经可训练 Transformer Encoder 融合，16 个 learned
action query 经 Transformer Decoder 并行生成 `[B,16,8]` 连续动作。训练目标为带 Episode
尾部 mask 的 Smooth L1 行为克隆损失。

**Reason:**

这是一条容易检查 shape 和数据语义、且适合单张 24GB 4090 的非自回归基线。冻结 Backbone
把首轮训练集中在模态融合与控制映射，并保留以后与 ACT/CVAE、Diffusion Policy、LoRA
或更强视觉表征进行受控对比的空间。

**Alternatives considered:**

- ACT/CVAE：更适合多峰演示，但首版同时引入潜变量和 KL 调参，不利于定位数据与控制问题。
- Diffusion Policy：连续多峰建模更强，但训练、采样和闭环延迟复杂度更高。
- 动作 token 化自回归模型：便于复用语言模型，但会引入量化误差，不符合首版直接连续输出目标。

**Status:** superseded by D009

## D008 — VLA v0.1 固定双相机、15 维状态与关节增量动作契约

**Decision:**

首版机器人继续固定为 7DoF Arm + 单一连续夹爪自由度。视觉观测改为两路同步 RGB：
`external/front camera` 提供全局工作空间和目标关系，`wrist camera` 提供末端附近的局部
几何与抓取对齐。两路图像在数据、Processor 和 Prompt 中保持固定顺序和显式相机角色，
不得静默交换。

Proprioception 改为 `q[7] + dq[7] + g[1]` 共 15 维，其中 `q` 为关节位置，`dq` 为关节
速度，`g` 为双指夹爪抽象后的单一连续开口状态。原始物理值先使用仅由训练集拟合的逐维
统计量归一化，再由小型 MLP 投影为一个 `[B,1,D_expert]` State Token；不把连续状态
离散化或转换成 Qwen 文本 token。`g` 的最终物理单位和双指关节到开口状态的换算，必须在
接入具体 gripper/controller 时由 Observation Adapter 明确，不在模型内部猜测。

动作侧暂时继承 D006 的其余契约：单步物理动作为
`7 delta_q(rad/control-step) + gripper target[-1,1]` 共 8 维；策略在 20 Hz 下生成
16 步连续 Action Chunk，每次执行前 4 步后重新观测；关节增量默认限制为每步 0.05 rad，
模型只学习 `[-1,1]` 归一化动作，物理转换集中在 Action Adapter。

双相机数据必须分别保留采样时间和有效性，后续数据 schema 需要从单相机 v1 升级；不能用
一个模糊时间戳掩盖 external、wrist、proprioception 和 action label 的错位。

**Reason:**

External camera 与 wrist camera 分别覆盖组合 manipulation 所需的全局任务进度和局部抓取
几何。对于固定的单一 7DoF 机器人，15 维连续状态没有需要单独 Transformer 建模的序列或
空间结构，归一化后投影成一个全局 State Token 是最小且容易验证的实现。保留现有关节增量
动作、Horizon 和滚动执行定义，可以把本次架构变化限制在观测、VLM 和策略训练目标。

**Alternatives considered:**

- 只使用 external camera：全局关系清楚，但精细抓取容易受分辨率与遮挡影响。
- 只使用 wrist camera：局部对齐清楚，但缺少全局目标、transport 和 place 语义。
- 保留 16 维并加入 gripper velocity：信息更完整，但简单 manipulation 首版尚无证据证明必要。
- 将 `q`、`dq`、`g` 拆成多组或 per-joint token：更适合多 embodiment，但增加固定单机器人首版的复杂度。

**Implementation status:** implemented RobotSpec、15 维状态和 8 维动作契约。

**Status:** active

## D009 — VLA v0.1 使用 Qwen3.5 post-training 全序列表示和重训 Action Expert

**Decision:**

首版 VLM 采用官方 post-training 模型 `Qwen/Qwen3.5-2B`，Stage 1 使用 BF16 并冻结
Qwen 参数。External/front 与 wrist 两张图及语言指令通过模型原生 Processor 和 Chat
Template 在一次 multimodal prefill 中联合编码；控制链路不生成文本，不使用 LM Head 的
词表 logits，也不在每个 Flow Integration step 重复运行 Qwen。

根据该模型官方配置，Vision Encoder 深度为 24、hidden size 为 1024，并投影到 2048；
多模态文本主干有 24 层、hidden size 为 2048，其中每 4 层包含一次 Full Attention，
最后一层为 Full Attention。第一版取第 24 层后的**全部多模态 token hidden states**
`[B,N_context,2048]` 作为 Qwen 上下文：不只取序列最后一个 token，不取 vocabulary logits，
也不拼接全部中间层。模型结构依据官方
[`config.json`](https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/config.json)，核对日期为
2026-08-24；后续更换模型仓库或 Revision 时必须重新校验这些接口。

`QwenVLAAdapter` 使用 Norm + 可训练 MLP 将上下文从 2048 投影到
`D_expert`，保留逐 token 表示和 Context Mask。Action Expert 采用 SmolVLA-style 架构但
从头训练，输入 Qwen Context、15 维状态得到的 State Token、带噪 Action Chunk 和 Flow
Time，使用 masked Flow Matching 预测 `[B,H_action,8]` Flow Velocity，并通过多步 Flow
Integration 生成连续 Action Chunk。Qwen Context 在同一个 Chunk 的所有 Flow step 之间复用。

当前冻结的 ResNet-18 + DistilBERT + Smooth L1 确定性 Chunk Policy 保留为简单 Baseline，
不再作为首版目标模型。中间层表示仅作为后续受控消融：优先比较第 20 层和第 24 层；只有
证据显示中间层改善局部几何、最终层改善语言组合时，才考虑少量层融合或 Context Resampler。

**Reason:**

Qwen post-training 表示更适合双相机语义、指令遵循和原子技能组合；保留最终层完整 token
序列，能让 Action Expert 对 external、wrist 和 language 的局部信息直接做条件注意，而不会
像单一 pooled/last token 那样过早压缩空间细节。Action Expert 从头训练消除了对原
SmolVLM hidden-state 分布和动作契约的权重兼容依赖。冻结 Qwen、绕过 LM logits、每次重规划
只计算一次 Context，符合单张 RTX 4090 24GB 的首版资源与延迟约束。

**Alternatives considered:**

- 使用 Qwen Base 版本：通用预训练表示更中性，但首版更看重 post-training 的多图和指令语义。
- 只取序列最后一个 token：计算便宜，但会把两路相机和语言过早压缩成单一向量。
- 使用第 20 层或其他中间层：可能保留更多局部几何，作为后续消融而非首版默认。
- 拼接全部 24 层：显存、接口和优化复杂度过高，且难以解释层贡献。
- 直接复用 SmolVLA Action Expert 权重：上游表示与动作语义不匹配，首版改为只复用架构并重训。
- 继续使用确定性 Smooth L1 Chunk Decoder：推理更快，但对多峰连续动作的表达能力较弱。

**Implementation status:** implemented frozen full-sequence Qwen Context and standalone Expert path.

**Status:** active

## D010 — QwenVLAAdapter 使用 Late Fusion 和模块化 SmolVLA-style Expert

**Decision:**

第一版采用方案 A：以 Late Fusion 连接冻结的 Qwen3.5-2B 和独立、模块化的
SmolVLA-style Action Expert。该实现不是官方 SmolVLA 的逐层 VLM—Expert 耦合实现，必须
准确称为 **SmolVLA-style standalone Action Expert**，不直接复用或声称兼容原
`SmolVLMWithExpertModel`。

`QwenVLAAdapter` 的固定接口为：

```text
Input tokens:  [B,N_context,2048]
Input mask:    [B,N_context] bool

Output tokens: [B,N_context,720]
Output mask:   [B,N_context] bool
```

Adapter 对每个 Context Token 独立执行带残差的两层 MLP 投影：

```text
x = RMSNorm(h_qwen)

skip = Linear(2048, 720)(x)

main = Linear(2048, 1440)(x)
main = SiLU(main)
main = Linear(1440, 720)(main)

context = RMSNorm(skip + main)
context = context * context_mask.unsqueeze(-1)
```

Adapter 保留完整 token 数量、原始 token 顺序和 Context Mask，`dropout=0`。第一版不增加新的
位置 Embedding，因为 Qwen 输出已经包含位置语义；不增加 Camera Embedding，而是依赖固定的
external/front、wrist 图像顺序和 Prompt 中的明确相机角色标签；不使用 Context Resampler，
也不融合多个 Qwen 中间层。Qwen 参数冻结，Adapter 参数参与训练。

Action Expert 的 hidden size 固定为 720，共 16 层，交替排列 8 个非因果 Self-Attention 层和
8 个对固定 Qwen Context 执行注意力的 Cross-Attention 层。Expert 输入序列为：

```text
[state_token, action_0, ..., action_H-1]
```

其中，15 维归一化 Proprioception 由 D008 中的小型 MLP 编码为 State Token
`[B,1,720]`；Noisy Action 与 Flow Time 联合编码为 `[B,H,720]`，首版 `H=16`、
Action dim 为 8。所有 Self-Attention 使用非因果 Mask；最终只读取 Action Token 位置，经
输出头预测 Flow Velocity `[B,16,8]`，State Token 位置不产生动作输出。训练使用 masked
Flow Matching，动作 Padding 不计入损失。

推理默认使用 10 个 Flow Integration steps，后续只在独立延迟/成功率实验中比较 4、6、10
步。同一个 Action Chunk 的所有 Flow steps 只运行一次 Qwen 和 Adapter，并复用 Adapter
Context；各 Cross-Attention 层由 Context 产生的 K/V 也应缓存复用。

上述 Expert 尺寸参考官方 SmolVLA 配置：SmolVLM hidden size 为 960，
`expert_width_multiplier=0.75`，因此 Expert hidden size 为 720；官方配置使用 16 个 VLM
层、非正数 `num_expert_layers` 表示 Expert 同为 16 层，并配置 `attention_mode=cross_attn`、
`self_attn_every_n_layers=2` 和 10 个 Flow steps。实现时只借鉴这些结构超参数，不继承官方
Expert 权重，也不假设 Qwen Context 与 SmolVLM Context 的分布兼容。参考资料：
[`smolvla_base/config.json`](https://huggingface.co/lerobot/smolvla_base/blob/main/config.json)、
[`modeling_smolvla.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/smolvla/modeling_smolvla.py)、
[`smolvlm_with_expert.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/smolvla/smolvlm_with_expert.py)；
核对日期为 2026-08-24。

**Reason:**

方案 A 把 Qwen 作为一次性多模态 Context Encoder，把动作生成完整封装在独立 Expert 内。
逐 token Late Fusion 保留双相机和语言的局部信息，同时通过 2048 到 720 的投影控制 Expert
宽度和显存。模块边界、Mask、缓存和单元测试均较清晰，适合先在单张 RTX 4090 24GB 上建立
可训练、可推理的第一版；后续替换 Context 压缩或更深耦合方案时，也无需改变数据和动作契约。

**Alternatives considered:**

- 方案 B：由 Qwen Final Context 直接生成 16 层 Expert K/V Cache。它更接近官方 SmolVLA，
  但层间耦合、RoPE、Mask 和 Cache 接口更复杂，首版暂缓。
- 方案 C：桥接 Qwen 第 4、8、12、16、20、24 层，形成多阶段层级融合。它有研究价值，但首版
  的显存、训练稳定性和调试成本过高。
- 方案 D：使用 Learned Context Resampler 将上下文压缩为 64 或 128 个 token。只有性能分析
  证明 Context 长度造成不可接受的延迟或显存开销后再采用，避免首版过早丢失视觉细节。
- 只取最后一个 token 或 LM logits：会过早压缩多模态信息，或引入与连续控制无关的词表接口，
  不采用。

**Implementation status:** implemented Adapter、16-layer Expert and Context K/V reuse.

**Status:** active

## D011 — 固定 qwen-vla-v0.1 的版本身份和完成边界

**Decision:**

新的第一版目标模型统一命名为 `qwen-vla-v0.1`，模型架构标识为
`qwen_vla_late_fusion_v1`。当前已经实现的单相机 ResNet-18 + DistilBERT + 确定性
Action Chunk Policy 改称 `deterministic-baseline-v0`，只作为历史工程 Baseline，不与新模型
共享模型版本身份。双相机数据使用新的 `robot-vla-trajectory/v2` Schema，不能继续写成旧的
`robot-vla-trajectory/v1`。

`qwen-vla-v0.1` 的完成边界固定为：

```text
当前 external/front RGB
+ 当前 wrist RGB
+ 当前语言指令
+ 当前 15 维 Proprioception
  -> Frozen Qwen3.5-2B
  -> QwenVLAAdapter
  -> SmolVLA-style standalone Action Expert
  -> 单个连续 Action Chunk [16,8]
  -> Receding-horizon 执行前 4 步
  -> 使用新观测重新规划
```

第一版明确不包含历史观测、Temporal Ensemble、多候选 Action Chunk、Real-Time Chunking、
World Model、在线 Evaluator/Planner、自动重试、长时规划和 Qwen LoRA。这些能力保留为后续版本，
不作为 `qwen-vla-v0.1` 的完成条件。

Checkpoint 和运行产物必须记录 `model_arch`、Dataset Schema、RobotSpec、Qwen model ID 与精确
Revision、Prompt 版本、Processor 配置、Normalization statistics 版本和代码版本；不允许只用
模糊的 `v1` 文件名判断兼容性。

**Reason:**

旧实现和新目标此前都被称为 v0.1，容易让配置、数据和 Checkpoint 静默混用。固定独立的模型、
数据和 Prompt 身份，可以保留旧 Baseline，又让新架构的完成范围保持可控。

**Alternatives considered:**

- 原地覆盖旧 v0.1：文件更少，但会破坏已有数据和 Checkpoint 的语义可追溯性。
- 第一版同时加入 World Model 和 Planner：目标更完整，但会妨碍先验证 VLA 本身的数据、控制和
  训练链路。

**Implementation status:** core v0.1 data/model/training/runtime scope implemented; training/evaluation pending.

**Status:** active

## D012 — 首版使用 ManiSkill Franka，并以可判定状态变化定义原子技能

**Decision:**

`qwen-vla-v0.1` 首先面向 ManiSkill 中的 Franka Panda 7DoF Arm + 平行双指夹爪建立数据、
训练和闭环评估链路。真实机器人以后通过独立 Observation / Action / Controller Adapter 接入，
不得假设仿真关节限制、夹爪方向和控制接口可以直接用于真机。

RobotSpec 必须固定 7 个关节的名称、顺序、位置范围、速度范围和单控制周期增量范围。原始状态
语义为：

```text
q[7]:  关节位置，单位 rad
dq[7]: 关节速度，单位 rad/s
g[1]:  夹爪统一开口比例，0 表示全闭，1 表示全开
```

具体夹爪关节值或开口宽度先由 Observation Adapter 映射到 `g in [0,1]`；`q`、`dq` 和 `g`
随后都使用仅由训练集计算的逐维统计量归一化，再输入 State MLP。测试集、验证集和在线数据不得
重新拟合统计量。原始 Dataset 保留上述可解释物理量和统一 `g`，不把模型归一化后的数值冒充
物理状态。

原子技能的命名与覆盖参考
[`CALVIN`](https://github.com/mees/calvin)、
[`LeRobot`](https://github.com/huggingface/lerobot) 和
[`ManiSkill`](https://github.com/haosulab/ManiSkill)，但项目使用自己的显式 TaskSpec 和
Outcome Predicate。一个原子技能不是单个 actuator command，而是满足以下条件的一组相对完整
动作：

1. 有明确的起始前提；
2. 内部可以包含多个 20 Hz 控制步骤；
3. 完成后环境产生清楚、稳定、可由程序判断的状态变化；
4. 成功、失败和未完成可以用 Outcome Predicate 区分。

第一版组合任务固定为 `pick-and-place`，初始规范技能序列为：

```text
reach -> grasp -> lift -> transport -> place
```

对应 Predicate 至少表达：末端进入目标邻域、形成稳定抓取、物体离开支撑面并超过高度阈值、
持物进入目标区域、物体释放后稳定落在目标区域。`place` 在第一版包含 release 和 settle；只有
后续数据证明需要独立分析时，才把 release 拆成单独技能。

训练数据必须包含完整组合轨迹和完整组合语言监督。原子技能及其分段标签用于数据覆盖、采样、
进度统计和失败分析，不作为额外模型输入。零样本技能组合可以作为探索性实验，但不作为第一版
验收条件。

**Reason:**

具体 embodiment 才能消除关节顺序、夹爪映射和控制限制的歧义。用环境状态变化而不是动作长度或
自然语言名称划分原子技能，可以让 CALVIN、LeRobot、ManiSkill 中不同粒度的任务最终落到同一套
可测试语义。完整组合轨迹监督也更符合从头训练 Action Expert 的数据需求。

**Alternatives considered:**

- 继续使用抽象的任意 7DoF 机器人：模型代码可以先运行，但数据、控制和闭环成功语义无法固定。
- 只训练原子技能并把零样本组合设为第一版目标：研究价值高，但难以区分组合失败来自表示、数据
  还是低层控制。
- 按固定帧数切分原子技能：实现简单，但切分点不对应可判定的环境状态。

**Implementation status:** TaskSpec and ManiSkill Franka adapters implemented; predicates pending.

**Status:** active

## D013 — 使用双相机对齐的 robot-vla-trajectory/v2

**Decision:**

模型训练使用控制周期对齐后的 `robot-vla-trajectory/v2`。每个 Episode 以共同控制索引 `T`
组织，至少包含：

```text
rgb_external        uint8   [T,H_external,W_external,3]
rgb_wrist           uint8   [T,H_wrist,W_wrist,3]

timestamp_external  float64 [T]
timestamp_wrist     float64 [T]
timestamp_proprio   float64 [T]
timestamp_action    float64 [T]

proprio              float32 [T,15]
action                float32 [T,8]

external_valid        bool [T]
wrist_valid           bool [T]
proprio_valid         bool [T]

terminated            bool [T]
truncated             bool [T]
success               bool [T]
skill_id               int16 [T]
```

`skill_id` 的整数到技能名称映射和各技能 Outcome Predicate 版本写入 Manifest/TaskSpec；未知或
未标注阶段使用保留值，不通过可变长字符串或 Pickle 写入 NPZ。相机内参、external 相机外参、
wrist 相机到末端的固定变换、标定版本和环境随机化参数作为 Episode 元数据保留。

索引 `t` 的语义固定为：external、wrist 和 proprio 是 `action[t]` 开始执行前可用的因果观测；
`terminated[t]`、`truncated[t]` 和 `success[t]` 描述执行该 Transition 后的环境结果。在 ManiSkill
中三类观测必须来自同一个 Simulator Tick。未来接入异步真机流时，只允许选择不晚于
`timestamp_action[t]` 的最新观测，并在 Robot/Dataset 配置中固定最大陈旧时间和两相机最大
时间差；不允许使用未来帧，也不允许静默复制上一帧填补缺失相机。

只要任一路相机或 proprio 无效，该控制索引就不能成为训练样本起点。Episode 尾部不足 16 步
的 Action Chunk 使用零值填充和 `action_mask`；无效 Action Token 既不参与 Flow Loss，也不能
成为有效 Expert Token 的 Attention K/V。

状态进入模型前使用训练集逐维 Mean/Std；标准差设置数值下限，归一化值默认裁剪到 `[-5,5]`。
其中 `g` 在拟合统计量前已经由机器人 Adapter 统一为 `[0,1]` 开口比例。动作不使用 Dataset
Mean/Std，而是按照 D008/D017 的固定物理增量限制和夹爪范围映射到 `[-1,1]`。

Dataset Split 必须按完整 Episode、场景和任务组划分。同一个场景随机化实例、同一条轨迹的窗口
和等价语言副本不能跨 Split。训练采样先在 Task/Episode 层平衡，再在 Episode 内采 timestep；
允许依据 `skill_id` 对 grasp/place 等短阶段做显式重采样，但采样权重必须记录在训练配置中。

**Reason:**

双相机、状态和动作的轻微错位会产生外观合理但监督错误的训练样本。共同控制索引配合独立时间戳
兼顾训练效率和可追溯性；阶段标签和任务平衡可以避免长 Episode 的静止或 transport 帧淹没短暂
的抓取与释放动作。

**Alternatives considered:**

- 继续扩展单相机 `trajectory/v1`：缺少明确的双相机时间和有效性语义，拒绝。
- 只保存一个共享 timestamp：无法发现相机与动作标签错位，拒绝。
- 对缺失相机复制上一帧：会把未知陈旧程度的数据伪装成同步数据，拒绝。
- 所有 timestep 全局均匀采样：实现最简单，但长 Episode 和高频 no-op 状态会主导训练。

**Implementation status:** trajectory/v2、Dataset、Collator and balanced Sampler implemented.

**Status:** active

## D014 — 固定 Qwen 双图 Prompt、Processor 预算和缺失模态策略

**Decision:**

Qwen 输入协议版本固定为 `qwen-vla-prompt/v1`。使用官方 Processor 和 Chat Template，消息
内容固定为：

```text
System:
You control a 7-DoF robot arm with a two-finger gripper.
Encode the observation and instruction for continuous robot control.

User:
External/front camera:
<external image>

Wrist camera:
<wrist image>

Robot instruction:
<instruction>
```

Processor 调用固定 `add_generation_prompt=False`，因此输入中不存在 assistant generation
header，也不生成任何回答 token。External/front 图像必须位于 wrist 图像之前，文字相机标签、
图像顺序和大小写属于 Prompt 契约，不能由 Dataset 或 Runtime 自行变更。

每张图默认最多产生 256 个视觉 token，Instruction 最多 64 个 token；加上特殊 token 后的总
Context 长度必须由 Collator 显式检查。实现时根据固定 Qwen Revision 的 patch/merge 配置将
256-token 上限转换成 Processor 的 `min_pixels/max_pixels`，而不是沿用旧 ResNet 的硬编码
`224x224`。Resize 保持宽高比并使用官方 Processor 语义；首版不使用水平翻转、随机旋转或其他
未同步变换 Action Label 的几何增强。

任一路相机缺失、无效或解码失败时，离线样本拒绝进入训练，在线 Runtime 拒绝推理并返回显式
错误；第一版不使用黑图、零图或上一帧进行替代。Instruction 超过 64 token 时数据校验失败，
不静默截断。

Qwen model ID 使用 D009 的 `Qwen/Qwen3.5-2B` post-training 模型；代码迁移时必须锁定精确
Hugging Face commit Revision 和兼容的 Transformers 版本。模型以 `output_hidden_states=True`
运行，使用经过最后一层及最终 Norm 的 `hidden_states[-1]` 和对应有效 token mask，保留图像、
文字及特殊 token；不得按 token 类型手工裁掉上下文。Qwen 始终处于 `eval()`，Stage 1 在
`torch.no_grad()` 下执行。

**Reason:**

Post-training VLM 的 hidden states 依赖消息模板、图像顺序和视觉 token 化。固定 Prompt、
Revision 和 token 预算可以让 Checkpoint 可复现，也防止两张高分辨率图使完整 Context 的显存和
Cross-Attention 延迟无上限增长。缺失相机直接拒绝比伪造输入更容易发现采集与在线系统故障。

**Alternatives considered:**

- `add_generation_prompt=True`：符合生成式对话入口，但本项目不生成回答，且会额外引入 assistant
  header；第一版不采用。
- 缺失相机时输入黑图或上一帧：可维持接口 shape，但会掩盖数据故障和观测陈旧性。
- 对超长 Instruction 静默截断：可能删掉目标或约束语义，拒绝。
- 两张图不限制 visual tokens：保留细节最多，但不满足单卡显存和可预测延迟要求。

**Implementation status:** fixed Prompt/Processor、budget checks and online rejection implemented.

**Status:** active

## D015 — 固定 State、Action、Flow Time Token 和 Expert Block

**Decision:**

15 维状态使用以下可训练 State MLP：

```text
state = RMSNorm(
    Linear(256,720)(
        SiLU(
            Linear(15,256)(normalized_proprio)
        )
    )
)
state_token = state + learned_state_type_embedding
```

State MLP 和 Expert 的 dropout 均为 0。Action Token 同时编码带噪动作、Flow Time 和 Action
Chunk 内的位置：

```text
action_emb = Linear(8,720)(noisy_action)
time_emb = SinCosEmbedding(flow_time, 720, min_period=0.004, max_period=4.0)

action_time = Linear(720,720)(
    SiLU(
        Linear(1440,720)(concat(action_emb, time_emb))
    )
)

action_token[h] = action_time[h] + learned_action_slot_embedding[h]
```

`learned_action_slot_embedding` 固定为 `[1,16,720]`，显式区分 `action_0` 到 `action_15`；
Flow Time 表示去噪积分时刻，Action Slot 表示物理控制先后，两者不得混用。Expert 输入为
`[state_token, action_token_0, ..., action_token_15]`。

Standalone Expert 使用 16 个 Pre-RMSNorm Transformer Block，hidden size 为 720，Gated-SiLU
FFN intermediate size 为 2048，RMSNorm epsilon 为 `1e-5`，dropout 为 0。Attention 默认沿用
SmolVLA/SmolLM2 风格的 15 个 Query Heads、5 个 K/V Heads 和 64 head dimension；所有这些值
必须进入版本化 `ExpertConfig`，不得根据框架默认值推断。

0-based 偶数层 `0,2,...,14` 使用 Expert Sequence 内的非因果 Self-Attention，奇数层
`1,3,...,15` 使用对固定 Qwen Adapter Context 的 Cross-Attention。State 和所有有效 Action
Token 在 Self-Attention 中双向可见；Cross-Attention 的 Query 包含 State 和有效 Action Token，
K/V 只来自有效 Context。Standalone Expert 使用显式 State Type/Action Slot Embedding；不再给
已经包含位置语义的 Qwen Context 添加第二套位置 Embedding。

每个 Cross-Attention Block 有独立的 Context K/V 投影。推理同一个 Chunk 时缓存这些固定 K/V；
Expert Self-Attention K/V 不缓存，因为每个 Flow step 的 Noisy Action 都会变化。输出经过最终
RMSNorm 和 `Linear(720,8)`，只读取 16 个 Action Token 位置；输出头结果转为 FP32 Flow
Velocity `[B,16,8]`。

**Reason:**

一个 State Token 足以表达固定单机器人 15 维状态。显式 Action Slot Embedding 解决非因果
Attention 对动作先后顺序不敏感的问题，而独立 Flow Time Embedding 表达当前去噪阶段。复用
SmolVLA block 尺寸同时保持 standalone Cross-Attention，使实现可解释且不重新引入官方逐层
VLM—Expert 耦合。

**Alternatives considered:**

- 不使用 Action Slot Embedding：模型无法稳定区分动作块中的物理时间位置，拒绝。
- 把 `q`、`dq`、`g` 拆成多个 State Token：有利于多 embodiment，但第一版固定单一 Franka，
  暂不增加序列长度。
- 缓存 Expert Self-Attention K/V：Noisy Action 在每个 Flow step 都变化，缓存会产生错误结果。
- 使用官方 `SmolVLMWithExpertModel`：与 D010 的 standalone 边界冲突，不采用。

**Implementation status:** State/Action/Time tokens and 16-layer Expert implemented.

**Status:** active

## D016 — 采用 SmolVLA 方向一致的 Rectified Flow 契约

**Decision:**

令 `a` 为归一化真实 Action Chunk，`epsilon` 为独立标准高斯噪声。训练采样和插值固定为：

```text
a       in [-1,1] with shape [B,16,8]
epsilon ~ Normal(0,I), float32
t       ~ Beta(1.5,1.0) * 0.999 + 0.001, float32

x_t = t * epsilon + (1-t) * a
u_t = epsilon - a
```

Action Expert 预测 `v_theta(x_t,t,context,state)`，训练目标为：

```text
loss = masked_mean((v_theta - u_t) ** 2)
```

Mask 同时覆盖 Episode 尾部 Padding 和无效动作，不把 padding 后的零值算入分母。8 个动作维度
第一版等权，因为进入 Flow 前均已映射到统一 `[-1,1]` 范围。

推理从 `x_1 ~ Normal(0,I)` 开始，使用 Forward Euler 从 `t=1` 积分到 `t=0`：

```text
num_steps = 10
dt = -1 / num_steps
t_k = 1 + k * dt
x <- x + dt * v_theta(x,t_k,context,state)
```

默认 10 步；4、6、10 步只作为后续延迟/成功率消融。最终 `x_0` 在进入 Action Adapter 前只做
一次 `[-1,1]` clamp，不在训练的 `x_t` 或 Velocity 路径中使用 `tanh`。Qwen和 Expert 主干
使用 BF16；噪声、Flow Time、MSE reduction、Velocity 输出和 Euler 状态使用 FP32。

训练每个样本只采一个 `t`，不运行 10 步积分。评估使用独立可复现的 Generator；在线每次
Replan 可以采新噪声，但必须把 seed 或 Generator state 写入 Rollout 日志。

**Reason:**

Flow Matching 的时间方向、目标速度和积分符号如果未固定，训练 Loss 仍可能下降但推理会沿反方向
积分。采用 SmolVLA 当前的明确约定可以直接对照上游实现，并用解析样例测试训练插值和 Euler
方向。FP32 Flow 状态降低小步积分和 Loss reduction 的数值误差。

**Alternatives considered:**

- 使用均匀 `t`：更简单，但首版选择与 SmolVLA 一致的 Beta 时间分布。
- 从 `t=0` 噪声积分到 `t=1` 动作：数学上可重新定义，但会与当前目标速度和上游实现方向相反。
- 每步对中间动作做 clamp 或 tanh：可能破坏速度场积分，只在最终输出执行一次 clamp。
- 使用 Heun 或高阶 ODE Solver：可能减少步数，但第一版先使用可复现的 Euler Baseline。

**Implementation status:** Flow target/loss、10-step Euler and deterministic Generators implemented.

**Status:** active

## D017 — 每个 Replan 以真实观测重置关节增量基准

**Decision:**

模型输出的每个 `delta_q[k]` 是一个控制周期的增量。对一次 Replan 得到的 Chunk，执行语义为：

```text
q_base = latest_valid_observed_q
q_cmd[0] = q_base + delta_q[0]
q_cmd[k] = q_cmd[k-1] + delta_q[k],  k = 1 ... K-1
K = 4
```

Chunk 内相对上一条命令累加，以保持动作序列连续；但每次 Replan 必须丢弃旧 Chunk 的未执行尾部
和旧 `q_cmd` 积分基准，重新读取最新有效真实关节位置并设置新的 `q_base`。禁止跨 Chunk 继续按
历史命令积分，从而避免跟踪误差、碰撞响应或伺服误差长期累积。Dataset 中的 `delta_q` Label
必须使用与该执行语义一致的相邻控制目标增量。

夹爪 Action 是绝对开口目标：

```text
gripper_action = -1 -> g_target = 0 -> fully closed
gripper_action = +1 -> g_target = 1 -> fully open
g_target = (gripper_action + 1) / 2
```

双指关节或实际宽度到 `g` 的正反向换算由 Franka Action/Observation Adapter 唯一定义。

每关节执行限制为：

```text
abs(delta_q_j) <= min(configured_step_limit_j,
                      robot_velocity_limit_j / control_hz)
```

D008 的 `0.05 rad/control-step` 是配置上限，不得覆盖更严格的 Franka 速度、位置、碰撞或环境
限制。Controller 在发送命令前执行最终限位和安全检查。

新 Chunk 可用时直接丢弃上一 Chunk 未执行的 12 步，第一版不做 Temporal Ensemble。如果新鲜
真实状态、任一路相机、推理结果或安全检查无效，则停止当前 Chunk，命令机械臂保持当前状态并
保留当前夹爪开口，不自动继续陈旧 Chunk，也不自动松开物体；同时返回显式失败供 Rollout 记录。

**Reason:**

Chunk 内累加符合关节增量轨迹语义，但跨 Chunk 使用命令位置作为基准会让真实跟踪误差不断进入
下一段动作。每次 Replan 回到真实观测，可以把误差限制在最长 4 个执行步骤内，同时保留 20 Hz
底层控制的连续性。

**Alternatives considered:**

- 每一步都相对 Replan 时的同一个 `q_base`：后续增量会被解释成绝对偏移，与训练动作序列语义
  不一致。
- 跨 Replan 沿旧 `q_cmd` 累加：命令连续，但真实跟踪误差可能长期漂移，拒绝。
- 推理失败时继续执行旧 Chunk：短期动作更平滑，但会在缺少新观测时扩大风险。
- 安全停止时自动全开夹爪：可能导致持有物掉落，第一版保持当前开口并交给上层处理。

**Implementation status:** Action Adapter、receding-horizon Executor and ManiSkill Controller implemented.

**Status:** active

## D018 — Stage 1 冻结 Qwen 并在线训练 Adapter 和 Expert

**Decision:**

`qwen-vla-v0.1` 只包含 Stage 1。参数训练边界固定为：

```text
Frozen / eval / no_grad:
- Qwen Vision Encoder
- Qwen multimodal/text backbone
- Qwen LM Head 不调用

Trainable:
- QwenVLAAdapter
- State MLP 和 State Type Embedding
- Action Input / Flow Time MLP 和 Action Slot Embedding
- 16-layer standalone Action Expert
- Flow Velocity Output Head
```

第一版不启用 Qwen LoRA。只有 Stage 1 已完成闭环训练、基础任务能够运行，且受控实验表明瓶颈
来自 Qwen 表示而不是数据覆盖、控制语义或 Expert 优化时，才新增 Stage 2 决策。

默认训练 Profile 采用 AdamW：

```text
learning_rate = 1e-4
betas = (0.9, 0.95)
eps = 1e-8
weight_decay = 1e-10
max_grad_norm = 10

warmup_steps = 1_000
cosine_decay_steps = 30_000
decay_learning_rate = 2.5e-6
```

使用 BF16 autocast 和 Gradient Accumulation；物理 batch size 与 accumulation steps 根据 24GB
显存测量确定，但有效 batch size、随机种子和实际优化步数必须写入实验记录。第一版关闭 EMA。
每个训练样本只采一个 Flow Time 和一个 Noise；验证集使用固定 seed 集合，避免随机噪声改变
Checkpoint 排名。

Qwen Context 在训练时在线计算，不预先保存 hidden-state 数据集。这样 Processor、Prompt 和
图像输入仍由同一版本化链路产生，也避免大规模 Context Cache 与 Qwen Revision 失配。若后续
吞吐测量证明需要离线 Cache，必须单独记录 Cache Key，其中至少包含图像哈希、Qwen Revision、
Processor、Prompt 和 dtype。

第一版图像训练 Profile 只使用 D014 的确定性官方 Resize/Pad，不默认启用随机几何增强；轻微
光照增强只能作为独立实验。训练采样遵循 D013 的 Task/Episode 平衡。Checkpoint 保存模型、
Optimizer、Scheduler、Scaler 状态、完整配置、RobotSpec、Prompt/Processor 版本、Normalization
statistics 和 RNG state；保存 latest、周期性 Checkpoint 及固定验证 seed 下 masked Flow Loss
最低的 best。闭环成功率仍按 D005 独立评估，不能用 best 离线 Loss 直接宣称任务成功。

**Reason:**

冻结 Qwen 将显存和优化重点集中在新的跨模态投影与动作生成模块，也使失败更容易归因。在线计算
Context 保证输入增强和版本语义一致；固定上游 SmolVLA 风格优化 Profile 为从头训练 Expert 提供
可复现起点，同时保留以后通过实验调整超参数的空间。

**Alternatives considered:**

- 第一版同时训练 Qwen LoRA：可能改善控制相关表示，但会增加显存和归因难度，推迟到 Stage 2。
- 预计算全部 Qwen Context：可提升 Expert 训练吞吐，但存储成本高，并冻结 Prompt、Processor 和
  图像增强结果。
- 默认启用强几何增强：可能增强视觉泛化，但如果不同步变换 Action Label，会破坏控制语义。
- 只按离线 Flow Loss 选择和报告模型：方便但不能代替闭环成功率，拒绝。

**Implementation status:** Trainer、Scheduler、validation and resumable Checkpoint implemented; full Qwen profile pending.

**Status:** active

## D019 — 删除旧确定性 Baseline，不保留运行时兼容层

**Decision:**

删除单相机 ResNet-18 + DistilBERT + Smooth L1 确定性 Action Chunk Baseline 的代码、配置、
训练入口、Checkpoint Runtime、trajectory/v1 数据实现和专属测试。项目不再发布
`deterministic-baseline-v0`，也不保留旧模块的兼容导入、旧 CLI 或双架构配置分支。

新代码只面向 `qwen-vla-v0.1`、`qwen_vla_late_fusion_v1` 和
`robot-vla-trajectory/v2`。旧 Checkpoint 与 trajectory/v1 数据不声明兼容；如果未来确实需要
复现实验，应从独立的历史代码快照恢复，而不是在主实现中重新加入条件分支。

本决策只取代 D009 中“保留旧确定性策略作为简单 Baseline”和 D011 中“发布
`deterministic-baseline-v0`”的部分；D009–D018 对 Qwen、双相机、Franka、Flow Matching、
训练和执行的其余决策继续有效。

**Reason:**

旧实现硬编码单相机、16 维 proprioception、trajectory/v1、224×224 ResNet 输入和 Smooth L1
训练目标，与新架构的双相机、15 维状态、trajectory/v2、Qwen Processor 和 Flow Matching
契约冲突。保留两套可运行架构会扩大配置、测试和 Checkpoint 兼容面，并增加误用旧入口的风险。
在新架构尚未开始编码时彻底移除旧实现，可以让后续每个模块只服务一个受控契约。

**Alternatives considered:**

- 保留旧代码并通过 `model_arch` 分支选择：便于对照，但每个数据、训练和 Runtime 入口都要维护
  双重语义，拒绝。
- 只删除模型层，保留 trajectory/v1 和旧测试：表面改动较小，但会继续强化已被取代的 16 维
  单相机契约，拒绝。
- 将旧实现移动到主包内的 `legacy` 目录：仍会形成隐性兼容责任；需要复现时使用独立历史快照。

**Implementation status:** 旧 Baseline、trajectory/v1、旧入口和兼容层已删除。

**Status:** active

## D020 — 模型 Action 严格拒绝，执行器跟踪修正显式饱和

**Decision:**

模型输出的归一化 Action 和物理 `delta_q` 继续按 D008/D017 严格校验，越界时拒绝执行，不做
静默裁剪。执行器把 Chunk 内累计目标转换为控制器命令时，如果仿真关节未完全跟踪前一目标，
仅对执行器内部计算的 `target_q - actual_q` tracking correction 按每关节有效单步上限做显式
饱和：

```text
applied_correction = clip(
    requested_correction,
    -effective_joint_delta_limit,
    +effective_joint_delta_limit,
)
```

Rollout 必须记录 `tracking_correction_saturation_count`、请求修正绝对最大值和实际修正绝对最大值。
饱和不能记作模型 Action 合法性提升，也不能把原本失败的任务记作成功。E002 使用旧执行版本的正式
结果保持不变，不回填或篡改。

**Reason:**

E002 的 4 个安全拒绝 seed 都在 Chunk 第 4 个执行步由第 4 关节的控制跟踪滞后触发；模型原始
Action Chunk 和夹爪目标均未越界。把执行器内部跟踪误差误报为模型危险输出会污染失败归因；放宽
模型契约又会真正扩大可接受动作范围。显式饱和并记录两侧数值能同时保持模型安全边界和控制可用性。

**Alternatives considered:**

- 放宽 `0.05 rad` 模型 Action 上限：会改变数据和动作契约，且不能解决错误归因，拒绝。
- 静默裁剪所有模型 Action：会掩盖策略越界，拒绝。
- 保留旧行为并将 tracking correction 记为模型安全拒绝：诊断不准确，拒绝。

**Implementation status:** implemented and covered by adapter、chunk executor and rollout tests.

**Status:** active

## D021 — 恢复数据保持完整成功契约，并显式配置瓶颈阶段采样

**Decision:**

恢复数据继续使用 `robot-vla-trajectory/v2` 的完整成功 Episode 契约；不把中间失败后缀、未完成
轨迹或手工伪造 Predicate 写入可信训练集。`trusted-pick-place-recovery/v1` 在完整专家轨迹内插入
可恢复扰动，覆盖 reach 绕行、偏位 grasp 后重抓、lift 前释放后重抓、transport 绕行和目标外
place 后重抓。每条恢复轨迹最终仍完整成功，`skill_id` 单调连续覆盖 0–4，并记录：

```text
recovery_profile
recovery_contract_version
recovery_evidence.disturbance_end_step
recovery_evidence.successful_recovery_end_step
```

恢复数据小规模 A/B 后暂不修改 Frozen Qwen、Adapter、Expert、Flow Matching 或 Action 表示。
Stage 1 的 Episode 内阶段权重改为显式 CLI 参数；为了复现 E001/E002，默认仍为：

```text
reach/grasp/lift/transport/place = 1/3/1.5/1/1.5
```

下一轮瓶颈采样实验推荐显式使用：

```text
--skill-weights 1.5 1 1 1.5 2
```

该参数和值必须写入 `experiment.json` 和 Checkpoint training config。它是待验证的训练目标实验，
不能在验证前称为正式提升。

**Reason:**

30-epoch 同预算 A/B 中，20 条恢复轨迹把 unseen 闭环阶段通过数从
`2/0/0/0/0` 提升到 `10/8/4/1/0`，说明现有模型容量能够利用恢复数据，尚无更换核心架构的
证据。但独立原子评估中 reach、transport、place 仍分别为 `0/5、0/5、0/5`，而 grasp/lift 为
`5/5、5/5`；继续对 grasp/lift 使用旧的额外权重与当前瓶颈不一致。新权重在 v0.3 train split
的 100,000 次确定性抽样中，把 reach/transport/place 合计占比从约 68.9% 提高到约 87.1%。

**Alternatives considered:**

- 立即更换 Qwen 层、Expert 架构或 Action 表示：变量过多，且数据 A/B 已产生明确阶段收益，暂缓。
- 放宽 trajectory/v2 审计以接收失败后缀：会把不完整 Episode 伪装成可信完整任务，拒绝。
- 直接替换旧默认权重：会让历史命令的语义静默变化，拒绝；使用显式参数。
- 立即修改 Flow loss：尚未证明问题来自各阶段内部的 loss 形状，先验证采样预算迁移。

**Implementation status:** recovery collector/audit and explicit Stage 1 skill weights implemented;
bottleneck sampling profile pending controlled training.

**Status:** active

## D022 — 保留完整 16-step BC/Flow 目标，并只对前 4 个执行步增加关键事件损失

**Decision:**

Stage 1 保留全部 16 个有效 Action step 上的原始 masked Rectified Flow/BC loss：

```text
L = L_base + lambda_event * L_event
critical_mask = event_mask & action_mask & exec_mask
exec_mask = [1, 1, 1, 1, 0, ..., 0]
```

`L_event` 只在 `critical_mask` 标记的 Action 元素上归一化；没有关键事件的 batch 不产生额外
损失。`event_mask` 从可信 GT 专家轨迹离线确定性检测，第一版包括 grasp/release command、contact、
linear/angular velocity jump，以及有可靠 object/support 状态时的 pickup/place。事件标签不得来自
模型预测，也不得把无效的 Episode 尾部 padding 标成关键事件。

固定高事件权重会让共享网络长期被全局梯度裁剪主导，因此训练支持把事件权重从接近 0 线性增加到
目标值：

```text
lambda_train(step) = lambda_target * min((step + 1) / warmup_steps, 1)
```

验证始终使用固定的 `lambda_target`，不能随训练 warmup 改变；否则不同 epoch 的验证总损失不可比，
并可能错误选择早期低权重 Checkpoint。`event_loss_weight_start/end`、base/event loss 和关键步数量
必须写入 epoch metrics。`warmup_steps=0` 保留旧命令的固定权重语义。

**Reason:**

机器人每次只执行新 Chunk 的前 4 步，直接影响闭环行为的 grasp、release、contact 和状态变化应在
这 4 步获得额外监督，但不能丢弃后 12 步提供的完整动作趋势。v0.4 的采样曝光分析还表明事件信号
高度集中于 grasp/place，几乎不给 reach/transport 直接监督；保留 `L_base` 是避免关键事件目标
破坏基础运动能力的必要条件。固定 `lambda=2` 的受控实验虽学到部分 release/place，却使 reach/lift
明显退化，并在大部分训练阶段触发梯度裁剪，因此较高目标权重必须渐进引入，所有权重都要通过
统一闭环评估选择。

**Alternatives considered:**

- 只在事件 step 训练：会丢失非事件动作与未来 12 步的运动监督，拒绝。
- 给全部 16 步复制事件权重：事件发生时刻与模型实际执行前缀不一致，拒绝。
- 固定使用较大的 `lambda_event`：已观察到共享网络被梯度裁剪主导，拒绝作为默认正式方案。
- 按训练中的 warmup 权重计算 validation：会改变 Checkpoint 排名标尺，拒绝。

**Implementation status:** event detection、Dataset/Collator mask、双损失、线性事件权重 warmup、
固定目标权重 validation 和指标记录已实现并有针对性测试。E006 完成 A–F 消融并选择固定
`lambda_event=0.25`；E007 已完成该配置的独立 100-epoch 正式训练和统一闭环评估。

**Status:** active

## D023 — 重叠 Chunk 使用最新预测占主导的 temporal ensemble，异常清空历史并重规划

**Decision:**

Runtime 按全局控制步对齐仍覆盖未来时刻的重叠 Action Chunk，并对同一时刻的 proposal 做指数加权：

```text
weight(age) = recency_decay ** age
recency_decay = 0.5
```

因此从最新到更旧 Chunk 的未归一化权重为 `1、0.5、0.25、0.125`；四个 proposal 同时存在时，
最新 Chunk 的归一化权重约为 53.3%，必须始终占主导。ensemble 在归一化 Action 空间完成，随后才
通过既有 Action Adapter 转成物理关节动作。Rollout 记录 buffer size、每步 proposal 数、最新
权重和最大 proposal spread。

推理异常、Action 安全异常或执行器要求重规划时，立即清空旧 Chunk 缓冲并用新观测请求 VLA
重新规划；连续异常重规划次数受显式预算限制。控制层只处理数值、安全和跟踪异常，不加入任务语义
状态机：不使用 stable-grasp gate 阻止 lift，不在 close→open 后强制保持 open，也不强制 settle。
这些 grasp/release 时序应由数据和 VLA 学习。

统一控制消融固定为：

```text
newest-only:     temporal ensemble off, max anomaly replans = 0
ensemble-only:   temporal ensemble on,  recency decay = 0.5, max anomaly replans = 0
ensemble+replan: temporal ensemble on,  recency decay = 0.5, max anomaly replans = 3
```

**Reason:**

每 4 个控制步重新规划而 Action Horizon 为 16，同一未来时刻天然存在多个重叠预测。按全局时刻融合
可降低 Chunk 边界抖动；让最新观测生成的 Chunk 权重最大，又避免历史预测压过当前场景状态。
异常后继续混入旧 proposal 会把已判无效的计划带回执行链，因此必须清空历史。语义 gate 虽能在
特定任务上掩盖 grasp/release 错误，却会把 VLA 学习问题变成手写确定性控制流程，与第一版目标冲突。

**Alternatives considered:**

- 只执行最新 Chunk：边界不连续，保留为消融而非默认方案。
- 让旧 Chunk 权重更高：不能及时响应新观测，拒绝。
- 异常后保留历史 ensemble：会重复混入导致异常的旧计划，拒绝。
- 增加 stable grasp、release hold 或 settle 状态机：改变学习边界并掩盖策略失败，拒绝。

**Implementation status:** temporal ensemble、trace、异常清空/重规划、CLI 配置和三组控制消融入口
已实现并有单元/集成测试。E007 的正式 20 unseen 消融中，ensemble 把阶段通过从 newest-only 的
`1/0/0/0/0` 提高到 `5/4/3/1/0`；两组 ensemble 均未触发异常，因此 replan 预算的净收益在该批
seed 上没有被激活，不能虚构为已改善成功率。

**Status:** active

## D024 — 不直接以 Layer 12 替换最终层，先诊断技能交接，再评估语义 Key / 几何 Value

**Decision:**

保持 Layer 24 作为当前 `qwen-vla-v0.1` 默认 Context，不把纯 Layer 12 直接升级为默认架构。
Layer 12 保留为受控诊断和候选几何来源。E009/E010 已完成两项低成本归因；进入新的长训练前继续
完成实际更新和 handoff 归因：

1. 已对 periodic checkpoint 做 Reach/Transport 闭环 sweep，并确认不同技能的最佳 epoch 分离；
2. 已用严格配对的 train/val 梯度 Gram 检查五技能、模块和 base/event；结果没有确认稳定负梯度
   冲突，不支持直接增加多头、后层分支或 PCGrad/CAGrad；
3. 对 e098-best→e100 实际参数位移做逐模块一阶投影，并补充 guaranteed-critical event batch；
4. 对 Reach 首次通过到 Grasp 的交接状态做显式 probe，比较策略自产状态与专家准备状态的相对位姿、
   速度、夹爪开度和双相机观测分布。

如果这些归因继续确认 Layer 12 的几何收益真实存在，但纯 Layer 12 的语义寻址或技能交接退化，则下一
候选架构采用同 token 位置对齐的分离注意力：Layer 24 投影为 semantic Key，Layer 12 投影为
geometry Value，Action/Proprio token 作为 Query。该候选必须重新经过独立原子和 20 unseen 完整
闭环，不能只凭 probe 或 validation loss 晋升。

Oracle TCP→物体相对几何只作为诊断上界，不进入生产 Observation；Runtime 仍不添加 stable-grasp、
release-hold、settle 等任务语义状态机。

**Reason:**

E008 的线性 probe 中，Layer 12 test median world-XY error 为 `0.0253 m`，Layer 24 为
`0.1245 m`；Reach-only 闭环也从 Layer 24 的 `1/5、0.0980 m` 改善到 Layer 12 的
`2/5、0.0628 m`，而 Oracle 为 `4/5、0.0397 m`。这些证据说明 Layer 12 确实保留更多几何信息，
但仍未达到 Oracle 上界。

同一 Layer 12 进入五技能联合训练后，独立原子结果为 `0/5、5/5、5/5、1/5、5/5`，总计仍是
`16/25`；20 unseen 完整阶段为 `9/3/2/0/0`，完整仍为 `0/20`。相对 Layer 24 的
`5/4/3/1/0`，Reach 通过增加但 Reach 后 Grasp 条件成功率从 `4/5` 降为 `3/9`。这说明完整瓶颈
不是单一“最终层没有位置”，而是几何、语义寻址、聚合 checkpoint 目标和技能 handoff 的共同问题。

独立 Grasp 为 5/5 而策略自产 Reach 状态后的 Grasp 只有 3/9，已经提供明确的分布失配证据；在
解释该差异前直接增加模型复杂度会重新混合变量，违背当前的归因原则。

E009 又提供了 checkpoint 维度的独立证据。在相同 10 个 confirmation seed 上，epoch 98 为 Reach
`0/10`、Transport `7/10`，epoch 100 为 Reach `3/10`、Transport `2/10`。epoch 100 的 Reach
residual 降低 55.8%，但 Transport 少成功 5 条且 residual 增至 6.17 倍；epoch 90 与 epoch 98
都是 `0/10 + 7/10`，也没有 residual promotion。没有单一 periodic checkpoint 同时保住两项技能，
所以问题不是简单选错 `best.pt`，而是聚合训练轨迹上的技能行为折中。该结果仍不能单独证明具体梯度
机制，handoff probe 继续保持为下一项必要归因。

E010 进一步否定了一个更具体但过强的机制解释。e098-best/e100 的 train Reach–Transport
`all_trainable` median cosine 为 `+0.164/+0.173`，独立 val 为 `-0.094/+0.441`，均未同时满足
`<=-0.10` 和预注册负 repeat 计数；三个 checkpoint 的五技能 train pair median 全部为正。
e098-best 的 Velocity head 虽为 `-0.120`，但只有 `3/5` 为负，e100 则为 `+0.480、0/5`，因此没有
输出头定位，也没有 broad/late/Adapter 冲突标签。Reach/Transport 的 event gradient 在选定前 4 步
均为零，within-skill base/event 机制不可识别；同时 Grasp 的中位梯度范数约为 Transport 的 2.7 倍，
使“更新幅度/采样暴露”成为新的候选而非结论。E010 说明闭环行为交换不能由当前 checkpoint 上稳定的
per-batch 负 cosine 直接解释，下一步必须检查真实 checkpoint 位移或状态条件边界。

**Alternatives considered:**

- 直接把 Layer 12 设为默认 Context：E008 完整成功仍为 0/20，且 Transport/后续阶段退化，拒绝。
- 继续相同配置增加训练 epoch：现有 100 epochs 已收敛，且聚合 loss 可能偏向事件技能，不能解决
  checkpoint 多目标冲突，暂不采用。
- 把 Oracle 相对几何作为正式 Observation：可以提高 Reach，但改变传感契约并引入仿真 GT，拒绝。
- 立即融合多层或加入复杂 Context Resampler：checkpoint 选择已在 E009 排查，但 handoff 数据问题
  尚未完成归因，继续推迟。
- 立即拆多动作头或加入 PCGrad/CAGrad：E010 没有确认输出头或广泛负梯度冲突；在已测大部分 batch
  上负 dot 不存在，拒绝在缺少 actual-update 证据时增加训练复杂度。
- 用手写技能状态机修复 Grasp/Transport：会掩盖 VLA 组合能力，继续拒绝。

**Implementation status:** Layer 12/24 同前向读取、独立 Layer 12 Context、空间 probe、Oracle Reach
和语义 Key / 几何 Value 诊断组件已在 E008 实验工作树中验证；E009 periodic checkpoint sweep 与
E010 gradient conflict probe 均已完成并发布完整原始结果。checkpoint-delta、event-conditioned、
handoff probe 和正式 Key/Value 组合闭环尚未运行。

**Status:** active

## 新决策模板

```markdown
## DXXX — 简短标题

**Decision:**

...

**Reason:**

...

**Alternatives considered:**

- ...

**Status:** proposed | active | superseded | rejected
```
