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

## D025 — Local DAgger 保留完整成功轨迹，但只监督同一 Session 内 Expert takeover 后的局部窗口

**Decision:**

E012 的 Local DAgger trajectory 从 frozen Policy roll-in 开始，在指定技能 boundary 的同一个
CollectionSession 内切换为可信 Expert，并由 Expert 一直执行到完整 Pick-and-Place 成功。第一版不放宽
`trajectory/v2` 正式 audit 对最终成功和五技能完整连续覆盖的要求，也不把 reset 后的原子 recovery 与
Policy 前缀拼成一条轨迹。

每条 Local DAgger trajectory 必须保存 `boundary_detection_step`、`expert_takeover_step`、半开区间
`training_window_start/end`、逐 Action 的 `action_source` 与 `expert_supervision_mask`。Policy roll-in
Action 只用于产生 policy-induced state 和诊断，禁止作为 BC/Flow target；当前单时刻 Dataset 实际使用
的是 takeover 当下由 roll-in 产生的双图/proprio，不把更早帧作为模型 history。第一版
训练窗口为 takeover 后 64 个 control steps，并且 Dataset 只允许完整 16-step Chunk 落在窗口且全部
Action 都由 Expert 生成；即使当前 Flow target 会把 mask 外 slot 置零，也不使用“Policy slot loss mask
为 0、Expert slot loss mask 为 1”的 mixed-source Chunk，以保持第一版契约简单且 fail closed。

Local DAgger metadata 或 mask 缺失时 fail closed。训练 loss 仍显式与 Expert supervision mask 求交，
Sampler 按 source 配额先选择 base-D0/RG/GL，再选择 Episode/timestep 并记录实际 exposure。`D1` 训练和推理
继续使用 `D0` 的 ProprioStats；新增数据上重算的 stats 只作诊断。warm start 新增只加载 Adapter/Expert
权重并重置 optimizer/scheduler/scaler/RNG 的 `--init-checkpoint`；现有 `--resume` 只用于同一训练身份的
中断恢复。

E012 的主因果对照是从同一 `pi_0` 初始化、相同额外 optimizer steps 的 `pi_dagger` 与 `pi_replay`，
不是直接比较训练后的 DAgger 与未继续训练的历史 checkpoint。正式 promotion 至少需要两个 paired
training repeats 方向一致，并同时报告无条件阶段完成数、共同 predecessor seeds、atomic 和系统安全
guardrail；条件率不能单独作为提升证据。

**Reason:**

现有 audit 会拒绝只运行到 Lift/Transport 前几步的 partial recovery；让 Expert 在同一 episode 中完成
整条任务可以保持可信轨迹契约和控制器连续性。与此同时，直接让现有 sliding-window Dataset 读取整条
Policy+Expert trajectory 会把 Policy 错误动作蒸馏回模型。显式 source/mask、局部完整 Chunk 和 loss
防线能够保留 takeover 时刻的 policy-induced Observation 分布，同时确保所有优化 target 都是 Expert
action。

额外训练步数本身可能改变已有 checkpoint，且 `--resume` 会继承旧 optimizer、scheduler 和 RNG 状态；
因此 replay control 与独立权重初始化语义是识别 Local DAgger 数据净作用的必要条件。冻结 D0
ProprioStats 则避免把输入标准化变化混入数据 exposure intervention。

**Alternatives considered:**

- 放宽正式 audit 接受短 partial recovery：会改变可信 Dataset 的完整成功契约，第一版拒绝。
- reset 到标准原子技能起点后采 Expert 数据：不再是 Policy 自己到达的 boundary distribution，拒绝。
- 把 takeover 前 Policy action 也作为 target：会自我蒸馏错误动作，拒绝。
- 允许一个 Chunk 横跨 Policy/Expert 后只 mask 直接 loss：当前实现可以把 mask 外 slot 置零，但会增加
  index/mask 语义和测试分支；第一版为保持 fail-closed 契约而拒绝。
- 只比较 `pi_dagger` 与历史 `pi_0`：无法隔离额外训练步数，拒绝作为主因果比较。
- 在 `D1` 上重新拟合 ProprioStats：同时改变输入归一化，不能识别纯数据 exposure 效果，拒绝。

**Implementation status:** additive Local DAgger provenance、live takeover collector、snapshot ring/round-trip、
trajectory/writer validator、Expert-only Dataset/Collator/loss 防线、source-first sampler、独立
`--init-checkpoint`、paired training verifier、checkpoint selection 与 paired evaluation analyzer 均已实现并通过
针对性测试。Legacy E012a 的 GL 容量 gate 失败按当时 stop rule 保留；后续 D026 amended protocol 独立通过
collection / D1 / union audit，repeat-1 replay/DAgger 训练和 checkpoint validation 已完成。两臂均无 eligible
checkpoint，因此按 D027 在 selection gate 停止，未运行 Stage A/B。

**Status:** active

## D026 — Amended GL 使用分段 Action 预算，并在 rollout 前冻结完整候选池

**Decision:**

E012 的 amended Grasp→Lift collection 保持 D025 的同一 `CollectionSession`、完整五技能 success/audit、
takeover 后 Expert-only boundary-local supervision 与 paired clean Expert 契约，只把主 trajectory 的 Action
预算改为：

```text
protocol:                 segmented-300-180-480
Policy roll-in budget:    300 actual environment actions
Expert recovery budget:   180 actual environment actions
environment hard limit:   480 actual environment actions
success deadline:         strictly before environment truncation
paired clean Expert:      legacy-300
eligible gate:            20
selection:                14 high-risk + 6 low-risk
```

正式 candidate 必须先写入同父目录的隐藏 staging dataset；只有 snapshot、paired clean Expert、risk、
`TrajectoryStore` 重新加载与完整 audit 全部通过后，才能原子发布到 canonical dataset。正常 rejection、
Python exception 或 record 写入失败必须回滚未提交的 staging/canonical 产物。`SIGKILL`、断电或文件系统
故障若发生在 dataset rename、staging marker 删除与 record 原子写之间，可能留下 partial canonical
dataset；此时 resume 必须因缺 record、残留 marker 或 partial candidate fail closed，禁止进入 selection / D1，
并要求人工审计清理。D1 builder 只消费 formal runner 显式发布的 canonical `accepted + selected` record
manifest，对 status/config/source/checkpoint/D0/audit/selection 任一漂移 fail closed；禁止扫描任意 NPZ 目录
推断训练输入。

正式 pool 必须在第一次 rollout 前一次性冻结连续 seed range、总 candidate 数和完整 config identity，
resume 只能接受 exact-equal identity；已有 `status=error`、partial candidate、receipt/record 漂移或不同
config 都必须阻断。不得根据中途观察到的 eligible 数量临时续采，否则会改变预注册 population 与
capacity interpretation。runner 固定要求从 amended formal 预留起点 `30200` 开始，并禁止 seed end 越过
checkpoint validation 起点 `31000`；experiment identity 还必须冻结 Qwen model/revision、关键 Python
package、CUDA 与 GPU runtime 信息，不能只记录 model cache 路径。

冻结 D0 的历史 dataset identity 使用加入 Local DAgger metadata 之前的 canonical projection。当前
`TrajectoryMeta.to_dict()` 会给 220 条 clean trajectory 自动补入 `local_dagger: null`，因此同一批未改写
文件在当前 projection 下得到不同 hash。Amended formal runner 不得把新的 hash 回写为 D0 身份，也不得
全局删除 D1 所需的 Local DAgger provenance；它必须在 CUDA 初始化和任何 formal artifact 写入前，以只读
方式同时验证：历史 projection `bc024…`、当前 projection `bb066…`、raw manifest/audit/proprio stats、
220 个 manifest-referenced NPZ 的实际 SHA、split/count/step 和无 symlink/path escape。只有两个 projection
及所有 leaf receipt 同时精确匹配，才能把带版本号的 compatibility receipt 写入 `experiment.json`。
Candidate record 还必须绑定其实际读取的 proprio-stats raw/semantic SHA 与 checkpoint 内嵌 stats，避免只在
pool 启动时验证一次路径内容。

Pool size 尚未冻结。现有规划 evidence 明确排除原来的固定 120 条作为低风险方案：planning point rate
为 `(10 legacy eligible + 5 recovered legacy TimeLimit) / 100 = 15%`，120 条在 Jeffreys Beta-binomial
posterior predictive 下达到 20 条 gate 的概率约 40.1%。200 条约 90.9%，220 条约 94.7%；严格达到 95%
的最小值为 223，因此当前严格低风险候选是按 20 条批量向上取整为 240 条。Owner 确认前不冻结 seed
end、不启动正式 amended rollout。

**Reason:**

旧 `legacy-300` 将 Policy roll-in 与 Expert recovery 共用同一 episode time limit，16 条 formal GL
trajectory 已到达 takeover，却在完整成功前被统一 TimeLimit 截断。固定 16-seed counterfactual 显示，
`segmented-300-180-480` 能恢复其中 5 条完整 eligible，并使另 1 条完成行为但被 snapshot gate 正确拒绝；
16/16 prefix metadata aligned、0 engineering error、0 hard deadline。它支持把 Policy 与 Expert 的预算
语义分开，但不支持放宽 audit 或把 counterfactual artifact 作为训练数据。

同一 counterfactual 也说明容量才是下一项主要执行风险：5/16 是旧 TimeLimit 条件恢复率，不能作为新
seed 总体率。使用 15/100 的 planning estimate 时，120 条 pool 的期望 eligible 仅为 18，本身低于 gate。
固定较大的完整 pool 可以在不事后选样或续采的情况下提高 gate assurance；保留 staged publish 和
canonical record manifest 则关闭历史 rejected record 旁残留 orphan NPZ 被误纳入 D1 的污染路径。

**Alternatives considered:**

- 继续使用 `legacy-300`：会把 Policy roll-in 长度和 Expert recovery 可用预算混为同一变量，已由 16 条
  post-takeover TimeLimit 证据否决。
- 只提高 environment hard limit，不设置独立 Expert cap：无法区分 Policy rollout 与 Expert recovery
  成本，也不能形成可审计的阶段性停止原因，拒绝。
- 放宽完整成功或 snapshot/paired gate 来增加 eligible：会改变可信 trajectory 契约，并让 seed 30181
  等 artifact 绕过已冻结审计，拒绝。
- 让 runner 扫描目录中的 NPZ：通用 trajectory artifact 不含自描述禁训练标记，且历史已有 rejected
  record 旁 orphan artifact，拒绝。
- 先跑 120 条、不足再续采：使总 pool size 依赖中途 outcome，破坏固定 population 与容量模型，拒绝。
- 依据 5/16 条件恢复率把正式总体率估成 31.25%：counterfactual 只选择旧 TimeLimit 子群，存在明确
  selection bias，拒绝。

**Implementation status:** segmented action accounting、hard-deadline 三信号、三条 smoke、16-seed
counterfactual、staged publish、canonical selected-record manifest、seed registry guard、exact-prefix resume、
Qwen/runtime identity、immutable receipt、D0 双 projection content verifier 和独立 artifact/statistics audit 已
实现。真实 frozen D0 的只读 verifier 已精确匹配 `220` 条、`48,922` steps 及全部 leaf/root identity；随后
owner-frozen amended formal pool、D1 build、D0+D1 union audit 与 repeat-1 训练均通过各自门禁。该进展不
回写 legacy formal，也不改变 smoke/counterfactual 禁止训练的用途。最终 promotion 在 D027 的 checkpoint
selection guardrail 停止。

**Status:** active

## D027 — Checkpoint selection 无 eligible 候选时停止 E012 promotion

**Decision:**

E012 repeat-1 只允许对每个训练臂的 epoch 10、20、30 运行预注册 lexicographic selection。先执行 system、
safety、tracking、anomaly、atomic Grasp/Lift/Place 与 full Reach guardrail；只有通过全部排除检查的候选才可
进入无条件 Lift、Grasp、mean completed skills、固定 D0 validation loss 与较早 epoch 的 ranking。若任一臂
`selected=null`，则不存在合法 Stage A pair，必须保留结果并停止：

```text
Stage A:                  not run
repeat 2:                 not trained
Stage B:                  not run
matched-state selected-pair diagnostics: not run
```

不得因为观察到 replay e20 或 Dagger e30 的正向 Grasp/Lift validation 信号而手工选模；不得加入 epoch 24、
Dagger epoch 9、`best.pt` 或 validation-loss 临时候选；不得调参后复用 `31000..31024`、`32000..` 或
`32100..` seeds。Checkpoint validation 只能作为 model-selection evidence，不能写成 Stage A/B 效果。

**Reason:**

正式 315-Episode validation 中，两臂的六个候选全部违反至少一个 guardrail。Replay e20 的 full
Reach/Grasp/Lift net wins 为 `+8/+9/+6`，但 atomic Place 为 `-3`；Dagger e30 为 `+2/+3/+4`，但 atomic
Place 为 `-2`，并新增 anomaly 和 tracking saturation。两臂的正式 receipt 均为
`selection_gate_passed=false`、`selected=null`、eligible ranking 为空。允许正向中间技能信号覆盖预注册
guardrail，会把已观察的 model-selection 数据用于事后改写 estimand，并失去 replay-controlled promotion
协议的可解释性。

这项 stop 不把 training audit 判为失败：两臂各完成 30 epochs、122,880 examples、1,920 optimizer steps，
paired verifier 和 Expert-only exposure audit 均通过。它只说明当前预注册 checkpoint 集合没有能同时保住
Reach/Place/运行 guardrail 的可 promotion 候选。因此合法结论是“promotion 在 checkpoint selection
停止”，不是“Local DAgger 改善”或“Local DAgger 必然无效”。

**Alternatives considered:**

- 选择 replay e20 或 Dagger e30：会忽略 atomic Place、anomaly/tracking 的显式排除规则，拒绝。
- 依据 epoch-24 validation loss 或运行时 `best.pt` 增加候选：候选集已在结果前冻结，拒绝。
- 直接进入 Stage A 再决定是否保留：Stage A 的输入本身要求合法 selected pair，不能用后续 gate 修补前置
  selection 失败，拒绝。
- 仅做 matched-state diagnostics：当前预注册 diagnostics 比较 selected replay/DAgger pair；自行挑模型会
  改变机制问题和样本选择，记为 not run。

**Implementation status:** 两份正式 selection receipt 已冻结并独立复算，SHA-256 分别为
`0fdc195552e742b017d71da57974a98ff626c018d289a1f5ffa891f74e1ee838` 与
`84ba2e7435438d65ebd1fb926cda21fbf69f92093021cf68cc4f0abc579586f6`；脱敏 compact summary 和 portable
technical report 已生成。Stage A/repeat 2/Stage B/matched-state selected-pair diagnostics 均未启动。

**Status:** active

## D028 — 最小可部署状态使用显式 Observation V2，Action 以 commanded target 为唯一标签语义

**Decision:**

保留 Expert 历史标签 `a_t = r_t - r_(t-1)`，其中 `r` 是 commanded joint target。Runtime 不再在每次
Replan 时从 actual `q_t` 重新解释 Action；它跨 Replan 保存最后一次成功 command reference，先积分标签得到
新 target，再以 `target - actual` 计算 controller correction。reset、hold、failure、tracking saturation 和
anomaly 清空 reference。

新建 `robot-vla-observation/v2` / `qwen_vla_temporal_state_fusion_v2`，固定输入为最近四个连续控制步的
双相机图像、proprio、base-frame TCP pose、OpenCV optical wrist-camera pose、左右 finger–cube pairwise
contact-force magnitude、时间/validity，以及 current controller state。Episode 起点使用前缀零 padding；
当前六模态必须完整，历史缺失模态必须零化并显式标记。位姿在模型中用 position + Rotation-6D，在数据
审计中保留可复算的 SE(3) 来源。

`F_L/F_R` 只从 train split 的有效值拟合版本化 `log1p / positive-P95` 稳健尺度；零接触不去中心。该
FingerForceStats 是 V2 checkpoint identity 的一部分。旧 D0 只有 aggregate force，不能复制成左右值；V2
Dataset 对缺失状态 fail closed。V1 schema/checkpoint 继续可读，但不得与 V2 policy、stats 或 prompt 混用。

**Reason:**

旧 Runtime 把 command-relative Expert label 当成 actual-relative correction，跟踪滞后时会让相同 Action 在
训练和执行指向不同 target。与此同时，单帧双图与关节状态无法显式观察末端几何、腕部视角运动、接触
不对称和历史速度线索，controller lag 也与策略动作混在一起。先修 correctness，再以独立版本增加最小状态，
可以避免把控制 bug、伪造数据或 checkpoint 兼容问题误归因为模型能力。

**Alternatives considered:**

- 把 Expert 标签改成 `target - actual`：会重写已采集监督语义，并把 controller dynamics 混入策略目标，拒绝。
- 每次 Replan 从 actual q 重新锚定：正是现有不一致，拒绝。
- 用旧 aggregate contact force 同时填 `F_L/F_R`：无法恢复接触不对称，属于伪造观测，拒绝。
- 用首帧复制填满四步历史：制造不存在的静止历史，并改变 Episode 起点分布，拒绝。
- 原地扩展 V1 checkpoint：状态 token、visual time embedding 和 force stats 都改变参数/输入身份，拒绝隐式兼容。

**Implementation status:** command-reference executor、V2 schema/history/coordinate helpers、Dataset/Collator、
Temporal Expert、八图 Processor、Runtime、ManiSkill adapter、train/eval CLI、force stats、checkpoint identity、
TCP FK orientation diagnostic 和 fail-closed tests 已实现。当前轻依赖回归为 `313 passed, 18 skipped`；真实
PyTorch/Qwen/ManiSkill GPU smoke、新 D0-V2 audit 和正式 paired training 尚未运行。

**Status:** active

## D029 — 2 mm 精定位从 VLA Action 中拆出，三头 U-Net Motion Head 先 shadow

**Decision:**

E013 在正式数据采集和训练开始前改为两时间尺度架构。低频 Qwen/VLA 继续负责指令、object/goal、技能、
粗 ROI 和 coarse approach；进入 fine alignment 后由高频 precision layer 独占位置控制权。Precision layer
直接读取腕部原始高分辨率 ROI，以三头 U-Net 输出稠密关键点/mask、base-frame 四维 TCP metric residual
和逐关键点/逐轴不确定性；目标像素先经过冻结 OpenCV 相机模型与平面几何生成 geometry delta。

新的笛卡尔动作语义固定为
`commanded-tcp-target-delta/base-frame/m-rad/v1`，与 D028 的 commanded joint-target delta 明确隔离。
Residual Head 最后一层零初始化；第一阶段只允许 `shadow`，正式候选命令等于 clipped geometry delta。
只有独立 shadow/calibration gate 通过后，才能在新实验身份中启用每轴硬限幅的 bounded residual。
四帧第一版在网络外对关键点、camera pose 和 timestamp 做状态估计；不把双相机八图继续送入 Qwen 或
直接堆进 U-Net。`F_L/F_R` 只在 contact mode 参与接触、偏载和滑脱控制。

**Reason:**

E008 Layer 12 的 world-XY median/p90 为 `25.3/38.8 mm`，距离 2 mm p90 目标约 19 倍；Layer 24 更差。
增加 Qwen 层、KV 或帧数不能恢复视觉 patch/merge 已丢失的空间带宽，也不能消除相机标定、TCP、控制
跟踪和接触误差。当前任务是已知桌面平面和受限物体几何，专用高分辨率 dense perception + 显式几何
可以分别验证 pixel、world、controller 和 final placement 误差。Shadow residual 保留学习系统偏差的能力，
同时避免再次把 Action label、坐标变换和控制动态混进不可审计的端到端 head。

**Alternatives considered:**

- 继续 sweep Qwen Layer 12/24 或使用 24 层 KV：没有度量坐标约束，且现有证据离 2 mm 太远，拒绝作为
  主精定位路径。
- 四帧双相机八图继续进入 Frozen Qwen：增加计算但不改变原始空间采样，旧 E013 在执行前 supersede。
- U-Net 直接输出不受限 Cartesian/Joint Action：重新混合感知、标签语义、IK 和控制误差，拒绝。
- 第一版立即启用 learned residual：没有 shadow calibration 和闭环证据，拒绝。
- 只使用单一 confidence sigmoid：无法区分遮挡、像素、投影和 motion uncertainty，拒绝。

**Implementation status:** `robot_vla.precision` 已加入动作契约、OpenCV base-plane 几何、三头 U-Net、
soft-argmax、heteroscedastic loss 和 fail-closed shadow/bounded-residual 仲裁；轻依赖合成测试已通过。
当前本机缺少 PyTorch，模型 forward/backward 测试只可静态编译并明确 skip；ManiSkill camera receipt、
oracle/HSV/RGB-only 数据、训练、Cartesian IK、四帧 filter、force controller 和 2 mm 闭环验证均未完成。

**Status:** superseded by D030 before formal E013 data collection or evaluation

## D030 — E013 以厘米级闭环精调为正式目标，工程可用档为底线

**Decision:**

保留 D029 的两时间尺度架构、三头 U-Net、显式几何、shadow residual、控制权互斥和 `F_L/F_R`
contact 边界，但取消系统级 `p90 <= 2 mm` 的项目成败要求。E013 按同一最终 object→goal world-XY
误差分三档：

- engineering floor：p50 `<=15 mm` 且 p90 `<=25 mm`，是可接受底线；
- recommended portfolio target：p50 `<=12 mm`、p90 `<=20 mm`，且 within-20-mm rate `>=90%`；
- optional stretch：p50 `<=10 mm` 且 p90 `<=15 mm`，只作附加结果，不阻断项目完成。

正式评估至少使用 100 个预注册 unseen paired Episode。任务失败有可测最终位置时必须进入误差统计；
invalid projection、system/safety/tracking failure、控制器重叠或 stale-observation command 任一非零都阻断
promotion。精调环最低要求为有效 `20 Hz` 和端到端 `p95 <=50 ms`；30 Hz 是可选性能目标，不要求
60 Hz。E008 Layer-12 `25.3/38.8 mm` p50/p90 是线性空间 probe 的固定诊断参考，不是最终放置
baseline；正式相对改善必须由相同 unseen paired seeds 上实际运行的 coarse-only control 复算，不在结果后
替换 control 或误用 probe 代理。

**Reason:**

个位数毫米系统精度会把主要工作转化为计量级相机/TCP/手眼标定、机械回差与柔性补偿、高带宽实时控制
和外部真值测量，超出当前求职项目的合理范围，也不是 π0.5 类基础操作系统通常用来证明语义泛化和长程
任务能力的核心指标。`12/20 mm` 推荐档仍是严格的厘米级系统目标；相对效果必须与正式 coarse-only
control 配对计算，并能充分展示粗到精控制、Observation V2、动作语义、动态相机几何、时间同步、
uncertainty gate 和接触反馈。工程底线与推荐目标分开报告，避免为追求单一数字删除失败样本或临时
放宽安全门禁。

**Alternatives considered:**

- 继续以 2 mm 为硬门槛：硬件、标定和真实测量成本过高，且会掩盖层级 VLA/闭环执行的求职价值，拒绝。
- 完全删除精调层：无法验证相对正式 coarse-only control 的厘米级闭环改善，也失去控制语义和多速率
  系统的核心工程贡献，拒绝。
- 只报告平均误差或挑选成功 Episode：会隐藏长尾和系统失败，拒绝。
- 因目标放宽而恢复旧八图 Layer-12 方案：旧 smoke 不是最终闭环，并且不解决控制权、相机几何或接触，
  拒绝。

**Implementation status:** RGB-only deployable/privileged Dataset、四时刻 Provider、formal-training
checkpoint、confidence calibration、held-out perception 与 20 Hz no-actuation observer 已实现并正式运行。
步骤 1–8 通过；100-seed 步骤 9 只形成 95 个 pair，并出现 5 个 Expert/collector rejection 与 7 次
`>50 ms` deadline miss，因此 promotion 按冻结 gate 停止。Cartesian IK、force controller、Motion Head
bounded residual、Precision actuator 和 final-placement 效果仍未实现或验证，不能把 offline GT-plane
held-out XY 误差写成目标档位已达到。

**Status:** active

## D031 — v1.0 将层级执行与 E013 精度归因分离

**Decision:**

`qwen-vla-v1.0` 只在 E013 通过正式闭环门禁后开始效果实验，并以其冻结 Observation V2、状态估计、
Precision、Force、Geometry 和 Controller 契约为不可变 parent。v1.0 使用一个共享 Action Expert、四个
宏观子任务和一个横切恢复状态；确定性 Subtask Executive 是 phase transition 的唯一提交者，Qwen 只
提供 schema 化 semantic proposal，Safety 保留动作否决权。第一正式 treatment 只引入 subtask/phase
condition、显式 Executive、关键不可逆动作门控和有限恢复；control 使用参数量匹配的 null condition，
两臂从同一 E013 checkpoint 派生并配对训练/评估。

开发按契约回放、shadow、Expert condition、不可逆门控、完整 Executive、独立 paired evaluation 逐级
推进。Runtime 不得读取仿真隐藏 GT；旧数据不得补造缺失的力、相机或 phase 字段。完整范围、Gate、
资源和发布条件记录在 [`roadmap.md`](roadmap.md)。

**Reason:**

E008–E012 已显示空间表示改善、checkpoint 选择和 boundary recovery 都不能自动转化为完整任务提升，
主要瓶颈集中在状态可观测性和阶段交接。若同时修改 E013 精调层、Qwen layer、数据和 hierarchy，就无法
区分收益来源。四个宏观子任务保留接触与控制约束的真实边界，又避免把固定 Pick-and-Place 扩张成多个
独立 policy 或伪装成 π0.5 规模基础模型。

**Alternatives considered:**

- 把 v1.0 合并进 E013：会混淆 Observation/precision 与 hierarchy 归因，拒绝。
- 为 Reach/Grasp/Lift/Transport/Place 分别训练网络：数据量和维护成本上升，且增加 handoff，拒绝。
- 让 Qwen 在控制循环中直接决定 phase 或 release：延迟、随机性和安全权责不可审计，拒绝。
- 同时解冻 Qwen、加入 24-layer KV 或学习式力控：超出单一变量和求职项目范围，进入 Future Work。

**Implementation status:** 项目计划与 Gate 已定义；`robot_vla.executive` 已实现 P0 的 semantic schema、
Plan Compiler、四时刻 wrist detection/camera pose/time 到 base-frame track/velocity 的确定性融合、可部署
State Estimate→Predicate/Snapshot adapter、shadow-only 状态机、有限恢复和 JSONL ledger replay，并有轻依赖
负例测试。`QwenVLAReplanLoop` 已有默认关闭、Action 执行后调用的 observer hook；真实 detector/outcome
monitor 尚未标定，也尚未接入 Qwen proposal、Action Expert condition、Precision/Force owner 或 actuator；
未训练或运行正式闭环实验。

**Status:** active

## D032 — Shadow Executive 在当前 Action 执行后观察，replan cadence 不冒充 20 Hz 控制

**Decision:**

Observation V2 Window 保留原始 float64 frame/modality timestamp；四时刻融合禁止用 newest timestamp 减
float32 age 重建采集时间。Precision decoded output 通过固定 adapter 形成 object/goal wrist detection，track
confidence 定义为 keypoint visibility 与 projection validity 的最小值；peak、entropy 和 pixel sigma 原样
进入诊断，未标定前不任意加权。

`QwenVLAReplanLoop` 的 `shadow_observer` 默认是 `None`，Observer 自身还需显式 `enabled=true`。启用时先
完成原有推理、ensemble/RTC 和当前 Action Chunk 执行，再把执行前 Observation 交给 Shadow Executive。
Observer 的 decision、`requires_action_reset` 或异常都不调用 actuator/executor；意外异常只写入独立
`shadow_error`。每 Episode reset 同时清空 observer 的内存 ledger，但调用方必须先冻结上一 Episode 记录。

当前 hook cadence 固定为 `replan-boundary/pre-execution-observation/v1`，通常只有 5 Hz。它可以验证接口、
ledger、错误隔离和当前 Chunk 的 Action parity，但不能测量正式 20 Hz phase delay/stability。同步 observer
还会记录 latency；在异步隔离或时延门禁通过前，不把它加入正式 treatment rollout。

**Reason:**

在原有 Action 生成或执行前同步运行 detector/Executive，会改变当前命令延迟，把 instrumentation 影响混入
hierarchy 归因。只保存 age 又会在 Episode 起点因 float32 舍入得到负时间，并掩盖图像—相机位姿错配。
保留原始时间并在 Action 后观察，可以先验证数据链和 fail-closed 语义，同时明确暴露下一次 Replan 延迟。

**Alternatives considered:**

- 从 `control_step / control_hz` 推算绝对时间：真实硬件、重连和异步传感器下不成立，拒绝。
- Observer 在当前 Action 前同步运行：可能改变当前 Chunk latency 和行为，拒绝作为 action-parity smoke。
- 直接让 shadow decision reset/hold：会提前把 P1 观察升级成控制 treatment，拒绝。
- 把 replan-boundary stability 写成 20 Hz 结果：cadence 不同，拒绝。

**Implementation status:** 原始 timestamp、Precision detection adapter、episode-local Observer、ledger/error/
latency record 与 `QwenVLAReplanLoop` 默认关闭 hook 已实现。RGB→冻结 U-Net 的 replay/shadow-only Provider
接线已实现，但没有训练/标定 checkpoint，也未在当前环境运行 Torch forward；20 Hz control-tick observer、
异步资源隔离和正式 shadow rollout 尚未实现或验证。

**Status:** active

## D033 — Precision Provider 要求显式 deployable geometry，四帧顺序推理不获得控制权

**Decision:**

新增 `PrecisionDetectionProvider`，默认 `enabled=false`，execution mode 固定为
`replay-shadow-only/no-actuation/v1`。它只消费 Observation V2，按有效 history 的 oldest→newest 顺序、
batch 1 运行 wrist RGB；padding 和缺失 wrist modality 返回 `None`，不会复制或补造图像。每个检测绑定
该帧原始 wrist timestamp，不能使用当前帧时间解释历史图像。

三头 U-Net forward 还需要结构化 frame state 和 `geometric_motion`。Provider 使用与 V2 Dataset/Runtime
相同的 proprio 与 `F_L/F_R` normalization，并要求 geometric-motion callback 返回带 timestamp、固定 Cartesian
semantics 和 `deployable_estimator` provenance 的输入；禁止用全零占位绕过接口，也拒绝 evaluator GT 和
图像—geometry 时间漂移。U-Net 按单帧 current-frame 语义训练，因此每个历史行使用该时刻的真实物理
state，但 forward 内的 frame-age feature 固定为 0；真实 freshness 只由原始 timestamp 和外部 estimator
计算，避免把相对最新窗口的 age 当成模型训练输入。Provider 不提供内部默认零向量；合法 callback 在
目标已经对齐时计算出的真实零 motion 仍然允许。

Torch wrapper 在构造时切换到 frozen eval，记录 checkpoint SHA、实际 parameter-state SHA、model-config
SHA、keypoint schema、预处理和设备类型；Provider 再绑定 RobotSpec、stats file SHA、实际 normalizer
semantic SHA、geometry provider ID 和自身 config SHA。Episode reset 会重新核对 tensor/config identity。
逐帧记录 timestamp、status、latency 和 confidence evidence，并提供 canonical JSONL/digest；不保存 RGB、
完整 state、NPZ 或权重。
Provider 只返回 `WristKeypointDetection | None`，没有 Action、executor 或 actuator 接口。

**Reason:**

Localization feature 虽是 image-only，当前 projection-validity 与 uncertainty head 仍显式依赖 state 和
geometry。用零 geometry 完成 forward 会产生看似合法但语义错误的 confidence，破坏最直接归因。显式
callback 和双重 identity 能先验证 RGB→检测→四帧 estimator 数据流，又不提前引入 controller treatment。

**Alternatives considered:**

- 固定传入零 geometric motion：projection-validity 输入与训练语义不一致，拒绝。
- padding 复制最近图像：制造不存在的历史并伪造速度，拒绝。
- 把四帧合成 batch 4：当前要求逐帧 latency、顺序错误定位和 batch-1 部署一致性，首版拒绝。
- Provider 直接生成 geometry command：会越过 Shadow Executive/Precision owner gate，拒绝。
- 每个正式记录保存 RGB/state：超出脱敏审计所需，增加数据泄漏风险，拒绝。

**Implementation status:** 轻依赖 Provider、identity/record contract、Torch lazy wrapper、负例和时间顺序
测试已实现。当前环境没有 PyTorch，真实 wrapper forward 只加入依赖型测试，尚未实际运行；训练 checkpoint、
calibration、20 Hz latency 和 ManiSkill shadow rollout 仍未完成。

**Status:** active

## D034 — E018 只授权仿真 development 的受限主动 front-camera 三阶段闭环

**Decision:**

用户已授权启动 E018-P1 三阶段主动视觉闭环，并指定由决策 Agent 持续负责实验组织、Gate、归因和回退，
由工程 Agent 负责具体实现与运行。当前授权严格限定为 `simulation / development-only / no-test`，不是正式
预注册、fresh test-once、真实硬件部署或 actuator promotion。三阶段依次为：

1. 动态 external observation、受限触发、owner、latch、SafeHold 和相机状态机；
2. front provider 资格、信息增益、pending candidate、HOME 验证和 Object Memory 两阶段提交；
3. 全阶段负例、故障注入、replay 以及匹配预算的 Passive/Active development 对照。

首版只恢复 object 的抓取前 navigation/pregrasp state，来源阶段只允许：

```text
ACQUIRE_TRACK
STABILIZE_PREGRASP
```

每个 Episode 最多一次 `HOME -> frozen alternate -> HOME`。进入 `FINAL_APPROACH`、发生 object contact、发送
gripper-close、形成 grasp candidate/grasp verified 或出现 object-maybe-moved 证据后，Episode-local active
window 永久关闭。Goal Memory 保持冻结；active Object Memory 不能单独授权 `FINAL_APPROACH`、contact、close、
lift 或 release。

Stage 1 只有有限 GO：可以实现独立 dynamic-external sidecar、纯 trigger/latch/state machine、shadow/replay 和
ManiSkill capability probe；可以沿用已通过 G0C v2 的无接触 camera motion parent。front provider 未完成逐
viewpoint qualification 前，不得把真实 front output 用作 live trigger、Memory write 或 manipulation resume。
Stage 1 的 live Object Memory write count 必须为零，fresh test read count 必须为零。

冻结的 Observation V2 不原地扩维或改变 external-pose 语义。active capture 使用独立版本化 contract，RGB 与
pose 保留各自 timestamp；视觉几何只接受与该 RGB 对应的
`sensor_param.base_camera.cam2world_gl` actual pose，经显式 OpenGL→OpenCV 和 world→base 转换。commanded pose
和 `camera.get_local_pose()` 只能用于命令/跟踪审计，不能冒充图像的 actual extrinsic。运动或 alternate-view
frame 不进入正常 VLA 的四帧历史；返回 HOME 后，必须重新取得四个连续、有效且均为 HOME 的 Observation V2
frame，才允许生成新的 manipulation Action Chunk。

external camera owner 与既有 arm/gripper `ControllerOwner` 分域。主动观察期间 effective arm owner 固定为
`SAFE_HOLD`，gripper 固定 open，camera 使用唯一 lease；camera controller 不获得 arm/TCP command interface。
首版 active phase 由 E018-only supervisor/context/ledger 承载，以保持默认 20-phase plan 和 P0 identity 不变。
若后续必须并入正式 Executive graph，使用新的 versioned plan/compiler identity，不静默改写 v1。

主动 request 前必须同时证明：当前 wrist object measurement 不可用、qualified HOME-front measurement 不可用、
Object Memory navigation state 不可用、来源 phase 合法、单调 latch 仍开放、attempt budget 未耗尽、安全前提
通过且失败属于 viewpoint-resolvable reason。单帧失败、invalid sensor/pose、provider identity mismatch、unsafe
arm/camera state 或 unknown reason 只能 Passive Reobserve/SafeHold，不能触发相机运动。

active request 接受时必须清除 temporal ensemble proposal、RTC overlap、executor previous command/action
reference，并记录严格递增的 Action generation。不得用整个 Episode 的 `reset()` 冒充 mid-Episode
invalidation，也不得复用 active 前的 Action Chunk。当前同步 executor 没有 20 Hz 中途打断能力，因此首版
trigger 明确发生在 replan boundary；不能把该结果描述为已经实现控制 Tick 级 chunk interrupt。

alternate settled window 只产生不可变 pending candidate。只有 actual HOME、arm q/TCP hold、open gripper、
no-contact、latch、candidate age/provenance、source invariant 和四帧 HOME barrier 全部复核通过后，Stage 2 才
允许一次原子 Object Memory commit。任一检查失败时丢弃 pending candidate、保留提交前 Memory、不恢复旧
Action，并进入 SafeHold/Abort。

单条技术路线的 no-go 不等于三阶段项目提前结束。每次 no-go 必须：

1. 冻结失败 config、source、provider、seed、ledger、receipt 和原始结果；
2. 区分实现错误、接口能力缺失、provider 不合格、viewpoint 无增益、安全失败和研究假设负结果；
3. 回退到最近一个 identity 完整且已通过的 checkpoint；
4. 在相同用户目标和安全边界内选择替代路线并下发新的工程任务；
5. schema、provider、schedule 或实验变量改变时创建新 config/experiment identity；
6. 同时报告失败路线与最终路线，不删除失败 seed、不放宽零容忍门禁；
7. 持续推进剩余独立工作，并在依赖修复后重新进入被阻断阶段。

合法完成有两种：安全与身份门禁全过且 Active 达到冻结 improvement gate 的正结果；或预定替代路线、paired
development 和失败分层全部完成后仍未达到 improvement gate 的证据化负结果。没有 qualified provider 时只能
形成 provider-qualification 负结果，不能冒充 Active-vs-Passive 的因果负结论；需要以独立上游 provider
实验冻结新 parent 后再继续 Stage 2。

以下变化必须建立新 Decision/Experiment/config identity，并在需要新增权限、明显资源预算或改变研究问题时
返回用户确认：

- 修改 Observation V2/model input、Dataset schema、label、正式 success/failure 或 test-once；
- 训练或微调 front provider，或用 Active 结果反向调整 perception threshold；
- 将 object-only 扩为 goal、held-object、contact 后或 manipulation 中主动观察；
- 将单次冻结 primitive 扩为多次运动、alternate-to-alternate、连续优化或 learned NBV；
- 改变相机/机械臂碰撞包络、让 camera/Memory/Executive 获得新的 arm/TCP actuator 权限；
- 超出已批准 simulation development 数据、GPU、时间或外部服务范围。

三阶段结束前，任何结果都不得声明正式任务成功率提升、通用主动视觉、真实硬件安全或部署资格。所有安全、
identity、provider 和 test 边界保持 fail closed；持续推进不能作为绕过 Gate 的理由。

**Reason:**

G0B/G0C 已证明 10 个低位相机姿态的静态和动态运动可行，但没有证明动态 external schema、front provider、
信息增益、Memory commit 或 manipulation resume。把这些能力一次接入会混淆运动、感知、Memory 和任务收益，
也容易让 alternate frame、旧 Action reference 或未验证的外参进入闭环。三阶段和两阶段提交把控制可行性、
感知资格与因果效果拆开，同时允许在路线失败后继续形成正结果或可解释负结果。

**Alternatives considered:**

- 直接把 active phase 加入默认 Executive v1：会改变冻结 plan/identity，并把尚未获得 actuator 权的 shadow
  Executive 提前升级，首轮拒绝。
- 复用 Observation V2 的固定 external calibration：动态相机后几何语义错误，拒绝。
- 直接把 wrist checkpoint 用于所有 front viewpoint：camera role/OOD 未资格化，拒绝。
- alternate view 一看到 object 就立即写 Memory 或继续抓取：无法在 HOME/arm hold 失败时回滚，也会绕过
  current contact evidence，拒绝。
- gate 失败即结束整个项目：无法区分单路线失败与研究假设负结果，拒绝。
- 为得到正结果放宽安全阈值、筛掉失败 seed 或增加 test 次数：破坏归因和 test-once，拒绝。

**Implementation status:** 用户已批准上述 simulation development 范围；三阶段实施计划和回退规则已形成。
G0B/G0C development parent 已存在；Stage 1 dynamic-external contract、E018 supervisor、mid-Episode Action
invalidation、Stage 2 provider/commit 和 Stage 3 paired evaluation 尚待按 Gate 实现与验证。当前不含正式 test
或真实硬件授权。

**Status:** active

## D035 — E018 推进期间授权决策 Agent 代决正式 B 级实验 Gate

**Decision:**

用户在 E018 三阶段持续推进期间明确授权：当用户离线或休息时，正式 B 级实验协议可以由决策 Agent 通过
Decision Gate 直接冻结、放行、否决或选择回退路线，不需要等待用户逐项确认。该授权覆盖受控实验设计，
包括 development 数据量、seed/split、校准方法、sampling、threshold/candidate window、ablation、公平对照、
资源预算和停止条件，也覆盖隔离 candidate 路径中的模型/架构/输入/loss/data strategy、offline 训练、
checkpoint 持久化、正式 offline evaluation 和 no-actuation shadow。D034 当前 active-reobserve 主实验仍保持
`simulation / development-only / no-test`；如需训练或消费新的 fresh held-out/test，必须由新的 B 级
Experiment/Config identity 在执行前冻结数据、选择规则、预算和 test-once，并且不得复用已消费的 E016 test。

每个 B 级 Gate 必须先记录并冻结：

1. hypothesis、control/treatment 和预期信息增益；
2. 自变量、冻结变量、data/config/source/provider identity；
3. train/validation/test 使用边界和泄漏审计；
4. 资源预算、停止条件、success/safety/promotion gate；
5. 允许的结论、失败语义、证据保留和下一条回退路线。

工程 Agent 对实现、常规代码自审、针对性测试、GPU 运行和 artifact 完整性负责。决策 Agent 保持独立，
只对 Dataset/label/sampling、坐标/时间、Memory write、安全门禁、checkpoint/result identity 和实验归因做
高风险逻辑审查与抽样检测；不逐行替代工程审查，也不重复整套常规测试。R2 reviewer PASS 不能替代正式实验
或安全证据。

该授权不覆盖 A 级变化。将候选 front provider、learned memory、active perception 或 held/contact memory
晋升为 canonical 主数据流，改变稳定 Observation/Action、坐标、时间、canonical Dataset schema/Label、
正式评估/公开 claim，放宽 fail-closed/privileged-data 边界，或让 camera/Memory/Executive 获得新的 actuator
控制权，仍必须暂停对应晋升并等待用户明确决定。隔离的 offline/no-actuation 候选训练本身属于 B 级，不能
再被一概当作 A 级阻塞。遇到真正的 A 级边界时，两个 Agent 必须先完成所有不依赖该授权的安全、合同、诊断
和 evidence 工作，不得把等待 A 级决定解释为整个三阶段项目停止。

**Reason:**

E018 包含多次可能的 protocol-invalid、parent-health no-go 和受控回退。如果每个 B 级细节都等待用户在线确认，
会造成 GPU 空转和项目中断；让独立决策 Agent 在预先批准的安全边界内冻结 B Gate，可以保持连续推进，同时
保留清晰的研究归因。把 A 级权限保留给用户，避免实验便利性被误用为核心架构、训练或控制授权。

**Alternatives considered:**

- 用户离线时暂停全部实验：会让已租用 GPU 和已冻结的安全路线无谓中断，拒绝。
- 让工程 Agent同时决定协议并实现：缺少 Dataset/label、安全和结论层的独立审查，拒绝。
- 把候选晋升、稳定合同或 actuator 等 A 级事项也交给决策 Agent：超出用户明确授权，也会改变核心研究
  所有权，拒绝。

**Status:** active

## D036 — 冻结 G2B-CAL-v2 协议性负结果，并以全新 free-static 数据训练隔离 front provider

**Decision:**

依据 D035 的 B 级代决授权，冻结 `E018-P1-G2B-CAL-v2` 为
`protocol-invalid / data-lifecycle-mismatch / negative-completion`。执行绑定 exact commit
`07821cb21d903454064599b395670bb476f0d8f7`、config canonical SHA
`7289767f783034a33ae4143d754697e6e05c90f03af6e833faa26c1a2fae109a`、source identity
`97443a6160c7f1a855556a1f54b17296480ab3f1a3a17fdbc7fb8c974501bdda`、prediction ledger
`0db758b1c0564c47d17774f87886a3570af18b0f7959cd5f8fd3a6790b5d8bee`、cohort audit
`3e9e8bf56ecabb3c7b1f1aec1307e6084b95ec1af9280d554c5ed27ed5000f08`、receipt internal/raw
`6b548874e90983a607b8d97770cc82a8c59d127a830268232b76db6d8aadbca4` /
`1541a043d210799906167d8931485e3eca3a5a2308afbb76de32334fa3edf4b5`。4154 个 prediction 和 skill
counts 均匹配；完整 1135-frame skill-0 cohort 中，仅 `pick-place-seed-134013/timestep 0` 同时出现
`is_grasped=true`、`F_L=27.81941795349121 N`、`F_R=27.870399475097656 N` 三个 violation，其余
finite/z/force-valid/raw-gripper 均零违规。按冻结规则整 cohort protocol-invalid；selection/fit 未执行，
calibration gate `evaluated=false/passed=null`，scoring ledger 为空，test/training/Memory/camera/arm/
manipulation counts 均为零。禁止启动 G2B preflight/full-50、删除该帧、改写 E016 artifact 或放宽 invariant。

只读诊断显示该 NPZ 与 label source SHA、shape/dtype/timestamp/skill/pose 均一致；t0 cube 仍在 task plane、
gripper open、TCP-object 距离 193 mm，t1 grasp/双力同时归零。ManiSkill reset 在 actor/velocity reset 后
直接读取 contact impulse、没有 physics step，因此高置信为上一个 rejected episode 遗留的 reset-first-frame
contact cache transient，而非真实抓持、数组错位或文件损坏。这个解释不追认 CAL-v2，也不授权 post-hoc
排帧。

否决方案 A：在已经读取的 E016 validation 上新增 deployable predicate 并删除 t0。它会在看到 privileged
lifecycle 异常后改变 cohort，破坏冻结协议；即使第三次 covariance 校准通过，也不能修正 G2A 诊断性 front
mean p90 约 141 mm 的 camera-role domain shift。选择方案 B：建立独立
`E018-P1-G2C-FRONT-PROVIDER-ADAPTATION-DEVELOPMENT/v1`，用全新 seed 隔离、专门
free-static/pregrasp 的 native contract + front RGB 数据训练 front provider，再用各自冻结的
model-selection、calibration 和 qualification split 验证。

G2C hypothesis：在 G0C 已通过运动门禁的十个离散 non-HOME front 位姿上，front-domain supervised
adaptation 能使至少一个 alternate 达到 object world-XYZ p90 `<=0.005 m`，并在冻结 uncertainty/write gate
后保持零 unsafe/catastrophic accepted。Control 为冻结的 E016-P1 selected epoch-12 wrist checkpoint 经原
role-substitution adapter 在新数据上的 baseline，只作诊断且永不进入 checkpoint selection。Treatment
候选池在读取 model-validation label 前固定为：

- `W`：同一 `PrecisionThreeHeadUNet` 从 E016 epoch-12 warm start；
- `S`：同架构 random initialization。

两臂使用相同数据、loss、optimizer 与 epoch 预算，只改变 initialization。沿用 E016 corrected-observability
tensor shape 和监督；motion head 保持 frozen-zero/shadow-only。Goal 输出最多作为既有辅助通道参与同 loss，
但在本实验没有 Goal Memory、qualification 或任何 consumer；只有 object measurement 可以获得候选 provider
identity。Qwen、Observation V2、canonical Dataset schema/Label 均不修改。

G2C 数据先使用 `G2C-DATA/v1` 冻结 seed/schema/lifecycle，DATA receipt 通过后再由其 manifest/file SHA
机械绑定 immutable `G2C-TRAIN/v1`；这两个 identity 不得合并成运行后才补 config 的单一 artifact。11 个
front pose 固定为 HOME 加 G2A/G0C 的 10 个低位平移/旋转 pose。split 固定为：

```text
train:                  76001..76400, 400 seeds × 11 = 4400 eligible frames
model-selection val:    76501..76600, 100 seeds × 11 = 1100 frames
per-view calibration:   76601..76650,  50 seeds × 11 =  550 frames
one-shot qualification: 76701..76750,  50 seeds × 11 =  550 scored frames
test:                   none
```

必须审计它们与所有 canonical manifest split、E016、G0/G0B/G0C/G1A、G2A `75001..75050` 及其他已登记
development seeds 完全不相交。不得用 E016 val/test、G2A output、Active-vs-Passive 结果或 qualification
label 训练/调参。

为处理已知 reset contact-cache 语义，每个新 episode reset 后固定执行 5 个 20 Hz SafeHold-open warmup
steps；raw reset-return observation 单独写 `reset_diagnostic`，永不成为训练、selection、calibration 或
qualification eligible row。生命周期 invariant 只对 warmup 后且明确标为 eligible 的 capture 执行，这一
规则必须在采集前冻结，不是事后删帧。train/validation/calibration 可使用 static-render-only pose
configuration，但每 pose 必须取得 same-observation actual external pose、intrinsic 和独立 RGB/pose
timestamp，并只保留冻结 settle rule 后的一帧。

任一 split 的任一 eligible capture 若不满足 finite object position、`z=0.02±1e-5 m`、not-grasped、
finger-force-valid、`F_L/F_R<=0.01 N`、raw gripper opening `>=0.95`、arm/TCP hold、no contact、
pose/RGB skew 和 geometry identity，则整个 split `protocol-invalid`；不补 seed、不删除 row。privileged
masks/positions/observability 只写独立 label sidecar；validation/calibration/qualification 都必须先生成
deployable prediction ledger、file fsync + parent fsync、冻结 hash 并销毁 model/data context，再打开对应
label arrays。

两 treatment 的训练策略冻结为 AdamW、BF16、batch size 32、20 epochs、learning rate `3e-4`、weight
decay `1e-4`、gradient clip `1.0`、cosine annealing eta-min 5%、`num_workers=0`、无 spatial
augmentation；candidate initialization/run seeds W=`18021`、S=`18022`，两臂共同使用
`sampler_seed=18020` 和完全相同的逐 epoch shuffle 顺序。初始化 RNG 与 sampler RNG 必须分离；不得让
W/S 的 candidate seed 驱动数据顺序。候选 epoch 固定 `{5,10,15,20}`，必须全部 checkpoint/hash 冻结后
才运行 model-validation。

checkpoint eligible 要求至少一个 G0C motion-qualified non-HOME alternate 同时满足 object visibility
precision `>=0.95`、recall `>=0.90`、observable-positive support `N>=30`、observable world-XYZ p90
`<=0.005 m`、max `<=0.020 m`、finite/valid geometry。某 viewpoint 的 support 小于 30 时只将该 viewpoint
标记为 `ineligible_reason=insufficient_observable_positive_support`，不是整轮 protocol-invalid；若因此没有
任何 non-HOME eligible viewpoint，则 `selected=null` 并收口为 protocol-valid model-selection negative。
support 计数与冻结 label/ledger 不一致或指标实现异常才属于 protocol-invalid。排序固定为：

1. eligible alternate 数量降序；
2. 最佳 alternate p90 升序；
3. 对应 max 升序；
4. validation loss 升序；
5. 更早 epoch；
6. 最终 W 优先于 S。

无 eligible 候选时 selected checkpoint 必须为 null，不得选择“最不坏”、增加 epoch 或读取 qualification 后
重选。

只有 selected checkpoint SHA 冻结后，才在独立 calibration split 对每个 viewpoint 分别拟合 scalar XY
Mahalanobis conformal covariance：`alpha=0.05`、target coverage `0.95`、chi-square `5.991`、
`k=ceil((N+1)*0.95)`、support `>=30`、scale `max(1,q/5.991)`、maximum calibrated position std
`<=0.020 m`。同一 calibration split 按冻结的 E018 object write-score semantics 逐 viewpoint 选择
“accepted-safe coverage 最大且 unsafe accepted=0”的 threshold；要求 coverage `>=0.10`，tie 取更保守的
较高 threshold。confidence、write acceptance 或 prediction error 不得用于 checkpoint selection；
qualification 之后不得修改 covariance scale、visibility/projection threshold、write threshold 或 viewpoint
pool。

Qualification 只运行 selected checkpoint 和已冻结 calibration/threshold 一次。它允许且只允许 isolated
ManiSkill simulator 中复用 G0C-v2 external RenderCamera route；`runtime/canonical camera-controller
actuation=0`、physical-camera actuation=0。父 identity 必须绑定 G0C config SHA
`c93bbfd48b6d9bc2fc75b5b87e4ded7161efebd7eda50cd81cc2ded47810e965` 和 receipt SHA
`bf8232b620cd5ff8de8c0007391252b8829c3ebbac320a7d5a60507beaca258e`。每个 qualification seed 对十个
non-HOME alternate 各执行恰好一条独立 env reset 的 `HOME→alternate→HOME` roundtrip，禁止 retry、seed
replacement、alternate-to-alternate，共 `50×10=500` routes，且每 route/episode 仅一次 active attempt。

每 route 精确继承：initial camera pose set 1；HOME warmup 5 ticks；一个 HOME anchor；outbound 40 ticks/
2.0 s；alternate settle 4；collect 3；return 40 ticks/2.0 s；HOME verify 4。固定总计
`camera_pose_set_count=97×500=48,500`，其中 moving interpolation command count
`80×500=40,000`；`simulator_safe_hold_open_step_count=96×500=48,000`；ledger frames
`92×500=46,000`。HOME provider 评分只取每 seed 固定首条 `LEFT_LOW__CENTER` route 的 HOME anchor，共
50；每 alternate 只取该 route 第 3 个/最后一个 settled COLLECT frame，共 500；总 scored=550。其余帧只作
运动/时序/安全 audit，不能扩大统计 N。

不得把 simulator camera commands 或 environment steps 记为零；必须另记 `runtime/canonical camera
actuation=0`、`physical camera actuation=0`、`nonzero arm-motion command=0`、`gripper-close
command=0`、`manipulation progression=0`、object contact=0、Memory R/W=0、test array reads=0。
SafeHold-open command 本身按 48,000 如实记录。

运动门槛精确继承 G0C：linear velocity `<=0.31 m/s`、linear acceleration `<=0.70 m/s²`、angular
velocity `<=0.75 rad/s`、angular acceleration `<=2.5 rad/s²`、camera tracking
`<=1e-5 m/1e-4 rad`、settled velocity `<=0.002 m/s/0.01 rad/s` 连续至少 3 ticks、HOME tolerance
`<=1e-5 m/1e-4 rad`、arm drift `<=1e-5 rad`、TCP drift `<=1e-5 m/1e-4 rad`、finger contact
`<=0.01 N`。任一路线出现 safety/identity/lifecycle/termination/truncation 失败，整个 qualification
protocol-invalid；不删除路线或重跑。

最终逐 viewpoint qualification 沿用 G2A front 门槛：visibility P/R `>=0.95/0.90`、observable XYZ
p90/max `<=0.005/0.020 m`、unsafe accepted=0、catastrophic accepted=0、accepted-safe coverage
`>=0.10`、covariance-95 coverage `>=0.90` 且 N `>=30`、maximum std `<=0.020 m`。至少一个
non-HOME alternate 通过才允许从 qualified 集按 frozen shortlist tier、coverage 降序、p90/max 升序、
`|cov95-0.95|` 升序、recall 降序和 frozen pose order 选择 PRIMARY。G2A 的 native-wrist parent-health gate
不作为新 front-trained provider 的 blocker；冻结 E016 Control 结果仅用于诊断，不支持相机域因果 claim。

资源预算：先允许 4-seed、无持久 checkpoint 的工程 smoke；Dataset/label/sampling/selection implementation
必须在 full-data 前完成独立 R2 抽样。正式上限为两 treatment、总 40 model-epochs、RTX 6000 Ada GPU
execution `<=10 h`、data+artifact `<=20 GB`、qualification once。数据 receipt SHA 必须在训练前写入
frozen config；output 已存在拒绝覆盖。任何 schema/seed overlap、eligible lifecycle violation、
prediction-before-label 破坏、nonfinite、权限计数非零、无 eligible checkpoint、无可校准 viewpoint、无
qualified alternate 或预算超限，都冻结当前 parent 的 negative/protocol-invalid receipt，不能现场改
threshold、加 epoch、补 seed 或切模型。

G2C PASS 只允许声明：“至少一个冻结离散 alternate 获得 simulation development-only front object provider
资格，可以作为新 parent 进入 E018 Stage 2 information-gain/pending-candidate/Object-Memory no-test 实现与
验证。”它不证明主动视角优于 Passive、不证明任务成功率、canonical 主闭环、actuator 或真实硬件安全。
若 W/S 均失败，冻结完整 control/candidate/data/ledger/checkpoint/receipt，并在同一用户目标下自动建立下一
独立 B Gate；优先比较 deployable object-mask centroid decoder 或更小 object-only provider，整个 E018 不
结束，但不得复用 qualification 调模型。任何 canonical 晋升、稳定 schema/label 更改、fresh formal test 或
actuator 权限仍属 A 级，必须返回用户。

CAL-v2 已达到 `DRIVE_VERIFIED`：远端唯一目录已 immutable copy/check，completion marker 后复核
24 matching/0 differences；worker source 保留，本机副本待补。G2C 的 canonical dataset、selected
checkpoint 和不可复现实验证据同样必须最终达到 Drive 与本机双验证副本，未达到对应 release gate 不得释放
唯一源。

**Reason:**

CAL-v2 精确证明旧 E016 validation 不满足为 pregrasp covariance calibration 冻结的 lifecycle，而不是证明
calibration 算法失败。事后排除异常行会污染协议，并且 covariance-only 路线无法解决 front 均值定位的数量级
误差。全新 reset-warmup、free-static、seed-disjoint 数据把数据生命周期、camera-domain learning、
per-view calibration 与动态 qualification 分开；两种固定 initialization 以低成本提高找到可用 baseline 的
概率，同时保持同一模型家族和输出合同。qualification 复用 G0C 有时延路线，避免用 static image 资格冒充
motion-settled provider 证据；精确拆分 48,500 camera pose set 与零 manipulation 权限，消除“仿真 camera
motion”和“canonical actuator=0”的语义冲突。

**Alternatives considered:**

- 对 E016 val 事后删掉 seed134013/t0：读取 privileged 异常后改变 cohort，拒绝。
- 第三次只改 covariance CAL：不能修复约 141 mm front 均值误差，拒绝。
- 直接使用 simulator GT 或 segmentation oracle 作为 provider：绕过 deployable 感知，拒绝。
- 只做 static qualification：不能支持 motion-settled capture，拒绝。
- qualification 允许仿真 camera route 却仍记 camera/environment steps 为 0：证据语义错误，拒绝。
- 一次引入更大 backbone、camera-pose-conditioned 模型或 learned NBV：在同架构 front adaptation 尚未
  验证前增加归因和成本，推迟到新 parent。

**Implementation status:** CAL-v2 已在 exact clean commit 运行并获得完整可验证 negative completion；Drive
已验证，本机副本待补。G2C Gate 已冻结；工程只先实现最小 collector/training/verifier 和 4-seed smoke，在
full-data 前返回 R2。

**Status:** active

## D037 — 修正 G2C 静态时间合同，并以 W-KV0 取代不稳定的原 W 正式候选

**Decision:**

依据 D035 的 B 级代决授权和 D036 engineering smoke 的 R2 抽查，D036 的数据、loss、预算和安全边界保持
不变，但在 full DATA 前冻结以下两项协议修正。

第一，`_capture_static_view()` 不得再用 `5/20 + sample_index*1e-6` 为 no-environment-step 的静态视角
构造伪时间。reset 后 5 个 20 Hz SafeHold-open step 产生的真实 simulation-control-time 是 0.25 s；如果
随后 11 个 static-render viewpoint 没有 environment step，它们可以且应当保留相同的真实 timestamp。采集
顺序只由 `sample_index`、`capture_sequence` 和 viewpoint order 表达，不能伪造时间差。RGB 与 actual pose
仍必须来自同一 observation，并分别保存绑定该 observation 的真实 timestamp。任何合成、回填、单调扰动或
把 capture order 冒充 simulation time 的行为都使 split protocol-invalid。

第二，每个 seed 的 reset diagnostic 明确保留两个 phase，各恰好一条：

```text
raw-reset-return-before-warmup/v1
post-five-safe-hold-open-warmup/v1
```

两条都必须是 diagnostic-only、`eligible=false`，并且 training/selection/calibration/qualification 使用计数
全为零。summary、receipt 和 verifier 必须分别冻结并校验 raw、post-warmup 和 total 三个计数：

```text
4-seed smoke: raw=4, post=4, total=8
full DATA:    raw=550, post=550, total=1100
```

缺 seed、重复 phase、phase 名称/role 漂移、使用标志非零或三项计数不一致都 fail closed。保留
post-warmup diagnostic 不改变 eligible capture 数；full DATA 仍为 train 4400、model-validation 1100 和
calibration 550，共 6050 eligible rows。

D036 的原 `W` warm-start 在 front-domain smoke 中仅保留为诊断，不再进入正式 candidate pool。exact
smoke 显示其 keypoint log-variance p50 为约 -11.53，weighted uncertainty loss 约 1918.63，总 loss 约
1929.09，pre-clip gradient norm 约 137486.66；同期 `S` 总 loss 约 12.01、weighted uncertainty loss 约
0.0047。虽然结果 finite 且 gradient clip 生效，但原 W 的 front-domain 更新会被继承的极端过置信 NLL
主导，不能直接放行 20-epoch 正式训练。

正式 warm-start treatment 改为 `W-KV0`：

1. 加载冻结的 E016 epoch-12 checkpoint 并验证原 checkpoint、parameter、provenance 和 model config SHA；
2. 只把 `uncertainty_head` 最后一个 Linear 中全部 keypoint-logvariance output rows 的 weight 与 bias
   确定性置零；
3. 保留 encoder、decoder、localization、mask、state、visibility、projection 和其余 uncertainty rows 的
   warm-start 参数；
4. Motion Head 继续 frozen-zero，且置零前后 motion parameter SHA 必须相同；
5. 记录被置零 row indices、置零规则版本、原/新 parameter SHA、原/新 keypoint-variance-row SHA；
6. 初始化后的 front smoke keypoint log-variance 必须精确为 0，loss/grad 必须 finite。

`S` 保持原 random initialization。正式 candidate pool 仍只有两个 treatment：`W-KV0` 与 `S`，不是增加
第三候选；control 与原 W 诊断都不能进入 selection。D036 冻结的 loss、optimizer、learning rate、weight
decay、20 epochs、batch size、BF16、candidate epochs、shared `sampler_seed=18020`、W/S run RNG、
checkpoint eligibility、calibration、qualification、预算和允许结论全部不变。W-KV0 只改变 warm-start
candidate 的确定性初始化边界，不放宽指标或安全 Gate。

工程必须在新的 source commit、config/source identity 和不覆盖旧 artifact 的新 run ID 上重跑同一个
4-seed smoke。旧 `be2ad36` smoke 保留为 `superseded-by-D037` 的工程证据，不是 provider negative。新
smoke 必须同时验证真实 timestamp、reset 三计数、W-KV0 row reset、W-KV0/S sampler order、finite
loss/grad、motion hash、prediction-before-label 和全部零权限计数。快速 R2 复核通过后，只能先放行 full
DATA；正式 TRAIN 仍需等待 canonical DATA receipt 和机械绑定的 TRAIN config。

**Reason:**

静态 RenderCamera 顺序需要可审计，但 capture 顺序和物理时间是两个不同字段。人工给未推进仿真的 observation
增加微秒会伪造 RGB/pose 时间语义，并可能让后续历史、age 或 skew 检查把同一时刻误认为连续物理观测。
保留真实相同 timestamp，同时用 sequence 表达顺序，既符合仿真事实，也不会丢失采集排序。

原 W 的 localization/mask 基础优于 S，但 inherited wrist-domain log-variance 对 front error 极端过置信，
使 NLL 比其他 loss 大约三个数量级。仅确定性清零 keypoint-logvariance output rows 是最小修改：它保留可迁移
的视觉与 visibility/projection 权重，去除已被 smoke 反证的不适用 uncertainty prior，同时不更改 loss、
数据、训练预算或最终 per-view conformal calibration。

**Alternatives considered:**

- 保留 `sample_index*1e-6` 作为“近似时间”：与实际 simulation time 不一致，拒绝。
- 为每个 static viewpoint 增加 environment step：会改变 free-static 数据生命周期和 object/robot state，
  不是修复时间字段所必需，拒绝。
- 删除 post-warmup diagnostic：会失去对 reset transient 已被 warmup 清除的直接审计证据，拒绝。
- 原 W 直接训练并依赖 gradient clipping：有限但由 uncertainty NLL 主导，20-epoch 行为不可解释，拒绝。
- 修改 uncertainty loss weight、clamp 或先冻结若干 epoch：同时改变 loss/schedule，归因范围更大，首选方案
  不采用。
- 重置整个 uncertainty head：会不必要地丢弃 visibility/projection 等可能可迁移参数，拒绝。

**Implementation status:** B 级修正已冻结；工程正在生成新 source identity 和替代 smoke。full DATA 与
正式 TRAIN 尚未授权。

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
