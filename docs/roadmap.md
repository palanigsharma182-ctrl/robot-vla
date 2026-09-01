# qwen-vla-v1.0 项目计划书

> 分层子任务执行、可审计阶段控制与受门控精密操作

| 字段 | 内容 |
| --- | --- |
| Plan ID | `QVLA-V1.0-PLAN-001` |
| 状态 | `engineering / P0`；主体契约、Plan Compiler、shadow Executive、ledger/replay 已实现，尚未接 Runtime、训练或运行正式闭环实验 |
| Owner | Project owner |
| 最后更新 | 2026-09-01 |
| 前置依赖 | E013 完成正式 GPU、ManiSkill 和至少 100 个 paired unseen Episode 闭环验收 |
| 计划基线 | E013 通过门禁后冻结的 Observation V2、精调执行层、Action/Controller 契约 |
| 目标版本 | `qwen-vla-v1.0`；是否成为默认 Runtime 取决于独立 promotion gate |

## Executive Summary

`qwen-vla-v1.0` 要解决的不是“再换一个更大的视觉骨干”，而是让一个连续 Pick-and-Place 系统明确知道
当前在做什么、何时可以切换阶段、谁拥有控制权，以及一次抓取、抬升或释放为何被批准或拒绝。它在
冻结 E013 最小可部署状态与厘米级精调能力的基础上，引入 Qwen 语义提议、确定性 Plan Compiler、
Subtask Executive、显式 Phase Controller 和共享 Action Expert 条件化，形成可以逐 Tick 回放的分层执行链。

该版本的求职价值是展示完整的机器人学习工程闭环：多模态状态契约、层级策略设计、控制器仲裁、
Expert-only supervision、配对实验、失败门禁和可复现实验发布。它不尝试按论文规模复刻 π0.5，不以
220 条单任务轨迹从头训练“机器人基础模型”，也不把 2 mm、60 Hz 或通用多任务能力设为当前成败标准。
核心问题只有一个：在 E013 能力和预算固定时，显式 hierarchy 是否能安全、可审计地改善阶段交接和
完整任务成功。

### 当前 P0 实现边界

`robot_vla.executive` 已实现以下轻依赖主体框架：

- 严格 `qwen-vla-semantic-plan/v1` schema，只接受冻结的单物体 Pick-and-Place 顺序；
- 稳定 SHA-256 plan identity、四个宏观子任务、17 个正常 phase 和 3 个恢复 phase；
- `history_length=4`、Observation V2、object/goal base-frame track 与 velocity、`F_L/F_R`、抓取/支撑/
  稳定置信度的 State Estimator 输出契约；
- 连续 Tick、required modality、entry/exit/invariant Predicate、多 Tick stability、timeout 接口、有限恢复
  和 retry budget；
- close/lift/release 每 Tick 重新授权、单 controller owner、transition 后 action/reference reset 要求；
- evaluator GT provenance 拒绝、默认 `shadow_only=True`、逐 Tick JSONL ledger、稳定 digest 和重新执行
  一致性检查。

这只证明 P0 状态机和审计链可以运行，不表示 E013 已完成，也不表示 hierarchy 改善了任何闭环指标。
当前 Executive 尚未接入 Qwen proposal、现有 `QwenVLAReplanLoop`、Action Expert condition、Precision、
Force Guard 或 ManiSkill actuator；State Estimator 目前只有输出契约，四帧滤波算法仍属于 E013 未完成项。

## 版本定位与证据边界

**Version positioning:** `1.0` 表示一次不兼容的架构主版本，而不是已经达到生产部署成熟度。该版本
同时改变任务执行权责、Runtime 持久状态、模型条件输入、Dataset/transition ledger、Checkpoint identity
和不可逆动作门控，不能作为 `v0.x` 的透明增量加载。正式命名仍须以工程门禁和独立闭环实验通过为条件。

本计划面向 E013 完成后的下一版本。目标是在不把连续运动切碎成大量独立策略的前提下，引入
可审计的中层执行语义，明确区分“语义意图、状态切换、连续动作和安全否决”。它不得并入 E013
的最小状态或厘米级精调归因，否则无法判断收益来自 Observation、精调层还是新的任务执行结构。

Engineering release 与 research promotion 是两个不同结论：接口、回放和安全门禁通过，可以说明
`v1.0` 架构可运行；只有独立 paired 闭环实验通过，才可以把 hierarchy 设为默认 Runtime 并声称改善。

### 当前基线

| 证据 | 已知事实 | 对 v1.0 的约束 |
| --- | --- | --- |
| E008 Layer-12 spatial probe | decoded world-XY localization `p50=25.3 mm`、`p90=38.8 mm` | 只作表示诊断参考，不是 final-placement baseline，不能用于计算闭环相对改善 |
| E012 Local DAgger | repeat-1 训练和 checkpoint validation 完成，但 replay/Dagger 均无 eligible promotion checkpoint | 不继承“DAgger 已改善”的结论；E012 产物保持只读 |
| E013 precision execution | 只有工程 scaffold；尚无正式训练、GPU smoke 或闭环效果证据 | 未通过 E013 G0 前不得启动 v1.0 正式训练或效果实验 |
| v1.0 正式 control | 以冻结 E013 checkpoint 为不可变 parent，派生使用 null hierarchy condition、无显式 Executive 的配对 control child checkpoint | 这是 hierarchy 相对效果的唯一 control，不以 probe、validation loss 或历史 checkpoint 代理；未改写 E013 parent |

E013 推荐求职档仍是 final-placement `p50<=12 mm`、`p90<=20 mm`、within-20-mm rate `>=90%`；
工程可用底线是 `p50<=15 mm`、`p90<=25 mm`。E013 只达到工程底线时仍可进入 v1.0，但 v1.0 不得
通过重新调精调层、Qwen layer、force threshold 或相机几何来获得 hierarchy 收益。

## 范围冻结

### 本版本包含

- 单物体、静态桌面、已知 Pick-and-Place 任务族；先在 ManiSkill 完成可部署接口约束的受控验证；
- 一个共享 Action Expert，通过 `subtask_id` / `phase_id` 条件化，不复制成多个独立 policy；
- 一个确定性、schema 驱动的 Executive 作为第一正式 baseline；
- Qwen 只在 Episode 目标建立或显式 replan 时提供语义 proposal，先 shadow、后受控接入；
- E013 已冻结的四帧双相机、TCP pose、动态相机位姿、`F_L/F_R`、精调与控制器契约；
- 关键不可逆动作门控、有限恢复、逐 Tick transition ledger 和 paired formal evaluation。

### 本版本明确不包含

- 2 mm 系统精度、60 Hz VLA、24-layer KV 堆叠或同时抽取更多 Qwen hidden layers；
- 全量 Qwen 解冻、从头训练基础模型、独立 skill network 或端到端联合训练全部模块；
- 学习式力控、插孔/装配、多物体长时序、多机器人迁移或真实机器人生产部署声明；
- 用仿真隐藏 GT 决定 Runtime transition，或给旧数据补造不存在的力/相机/状态字段；
- 在看到正式结果后新增 checkpoint、调整 gate 或复用 evaluation seeds 调参。

### MVP cut line

为防止再次同时推进过多模块，v1.0 MVP 只包含：确定性 Plan Compiler、四个宏观子任务加恢复状态、
共享 Expert 的轻量 condition、close/lift/release 三个关键门控、单 controller owner、transition ledger 和
一次完整 paired evaluation。Qwen proposal 在 P1 先 shadow；只有 schema、agreement 和安全门禁通过后才
进入 Episode-level 目标建立，仍不负责逐 phase 决策。

学习式 task graph、学习式 transition classifier、动态长时序 replan、多物体任务和新型 force policy 全部
进入 Future Work。P0–P3 可以形成可演示的 portfolio MVP，但没有 P4/P5 不能命名为正式 `v1.0.0`，也不能
声称 hierarchy 提升了任务成功率。

## 目标架构

下一版本采用以下权责边界：

```text
VLM / Semantic Planner
  -> 提议目标对象、目标区域和下一候选子任务
Subtask Executive
  -> 根据任务图、可部署 Predicate 和连续稳定证据批准或拒绝切换
Phase Controller
  -> 管理当前子任务内显式物理阶段
Shared Action Expert
  -> 在 subtask/phase 条件下生成连续 Action Chunk
Safety / Controller
  -> 允许、限幅、拒绝或中断具体动作；不可逆操作保留最终否决权
Outcome Monitor
  -> 提供完成、失败和恢复证据，不接受策略自报成功
```

VLM 只负责语义提议，不直接提交安全关键的状态切换。对于当前固定 Pick-and-Place，任务图可以在
Episode 开始时生成一次；正常 phase transition 不重复调用 VLM。Subtask Executive 是切换状态的唯一
权威，Safety 可以随时否决动作，但不负责选择语义任务。

### 系统架构与运行时数据流

```text
four-frame stereo RGB + TCP/camera pose + F_L/F_R + gripper/controller state
                                |
                                v
                  Observation V2 + time/validity
                                |
              +-----------------+------------------+
              |                                    |
              v                                    v
  deployable State Estimator              Qwen Semantic Planner
  track/velocity/contact/confidence        object/goal/task proposal
              |                                    |
              +-----------------+------------------+
                                v
                    deterministic Plan Compiler
                    schema + allowed task graph
                                |
                                v
               Subtask Executive + Phase Controller
               proposal -> predicate check -> commit/reject
                                |
               subtask/phase/age/target condition
                                v
                       Shared Action Expert
                       coarse Action Chunk
                                |
                                v
                    Controller Ownership Arbiter
          Action Chunk | Precision | Force Guard | Safe Hold
                                |
                                v
               Safety limits + commanded-target adapter
                                |
                                v
                         environment / robot
                                |
          Outcome Predicate + failure + transition ledger
                                +---------------------> Executive
```

每个 Control Tick 只能有一个运动控制 owner。`Action Chunk` 负责自由空间和宏观运动，E013 `Precision`
负责目标附近的高频小幅修正，`Force Guard` 只执行已定义的接触/夹爪保护，`Safe Hold` 在观测无效、
安全拒绝或恢复期间接管。Owner 切换必须原子地清空旧 chunk、旧 command reference 和过期 target；禁止
两个控制器叠加 TCP delta。各环频率继承冻结的 E013 配置，v1.0 不把改频率作为 hierarchy treatment。

Qwen 输出必须先通过版本化 schema 和 Plan Compiler，不直接进入 Action Adapter。固定单任务时，Qwen
通常只在 Episode 开始调用一次；只有 Executive 明确请求语义 replan 才再次调用。正常 20 Hz 级控制、
phase transition 和 safety decision 都不依赖在线生成自然语言。

### 子任务划分

第一版只规划四个正常子任务和一个横切恢复状态：

1. `ApproachAndAlign`：建立有效目标跟踪并形成低速、姿态合理的 pre-grasp 状态；
2. `AcquireAndVerify`：进入接触、闭合夹爪并确认稳定双指抓取；
3. `TransferHeldObject`：在保持抓取约束下完成抬升、运输和 pre-place 对齐；
4. `DepositAndVerify`：放低、确认支撑、受门控释放、撤离并验证稳定放置；
5. `RecoverOrHold`：在观测失效、抓取丢失、tracking saturation、异常或安全拒绝后安全保持、
   重新观测、有限重试或终止。

现有 `Reach/Grasp/Lift/Transport/Place` 和更细的接触、释放、稳定事件继续作为 Outcome Predicate、
数据标签和失败分析维度；它们不自动升级为独立网络或十个平级中层子任务。只有控制目标、物理约束、
可观测终止条件或恢复路径发生实质变化时，才拆出显式 phase。

### 子任务内部阶段

```text
ApproachAndAlign
  AcquireTrack -> CoarseApproach -> FineAlign -> StabilizePregrasp

AcquireAndVerify
  FinalApproach -> CloseUntilContact -> SeatAndBalance -> VerifyGrasp

TransferHeldObject
  LiftClearance -> MoveToGoal -> AlignForDeposit -> StabilizeHeld

DepositAndVerify
  LowerToSupport -> ConfirmSupport -> Release -> Retract -> VerifySettled

RecoverOrHold
  SafeHold -> Reobserve -> Diagnose -> Retry / Replan / Abort
```

Phase 是控制模式，不是新的自然语言任务。连续、可逆且约束相同的细微运动由 Action Expert 隐式处理；
自由空间到接触、抓取确认到抬升、运输到放低、放低到释放、释放到稳定验证以及正常执行到恢复等
物理语义变化必须显式表达。

### 状态切换规则

候选状态 `z'` 只有在以下条件同时成立时才提交：

```text
Exit(current_subtask_or_phase)
and Entry(z')
and StableForKControlTicks
and RequiredModalitiesValid
and TransitionAllowedByTaskGraph
and not UnsafeOrAnomalous
```

具体要求包括：

- 使用连续多帧证据，不根据单帧 VLM 分类立即切换；
- 接触、抓取和稳定判断使用 enter/exit hysteresis，避免阈值附近抖动；
- modality 无效或时间不同步时进入 Hold/Reobserve，不以零值冒充真实状态；
- phase/subtask timeout、失败原因和 retry budget 必须显式记录；
- 剩余 Action Chunk 在 reset、hold、安全拒绝、tracking saturation 和 anomaly 后失效；
- `gripper close`、开始 lift 和 release 等关键动作使用额外前置条件；
- release 必须同时通过 Executive 的任务条件和 Safety 的物理条件，不能只由 Action Expert决定。

阈值、稳定 Tick 数和 retry budget 在传感器标定与 shadow measurement 前保持待定，禁止凭经验写成
正式常量。任务评分用的 `Reach <= 4 cm` 只表示进入物体邻域，不等价于高质量抓取前置状态。

### 可部署状态估计前置项

Executive 不得读取仿真器隐藏的 `object_pose`、`is_grasped` 或成功标志控制策略。下一版本需要先定义并
验证可部署的 State Estimator，至少提供：

```text
object_pose_or_track_in_base + confidence/validity
goal_pose_or_track_in_base + confidence/validity
object_relative_velocity + confidence/validity
grasp_confidence
support_contact_or_settle_confidence
```

这些估计只能来自部署时可获得的四帧双相机、TCP pose、相机位姿、`F_L/F_R`、夹爪状态和 controller
state。仿真 GT 只可用于监督与评测估计误差。若没有可校准的 object/goal tracking，Executive 不能以
厘米距离规则伪装成可部署系统。

### 模型与数据边界

- 继续使用一个共享 Action Expert，不为每个 subtask/phase 训练独立网络；
- Action Expert可增加 `subtask_id`、`phase_id`、phase/subtask age 和目标参数条件；
- phase 标签由 Expert command provenance、可观测 Predicate、`F_L/F_R`、gripper command、位姿和
  时序证据生成，不按轨迹百分比切分；
- 边界证据不确定时保留 Action supervision，但 mask phase-classification supervision，并记录
  `transition_ambiguous=true`；
- Runtime 不接受旧 trajectory 静默补造 phase/state-estimator 字段；新数据、checkpoint、prompt 和
  Executive identity 必须版本隔离；
- VLM、Executive、Action Expert、Safety 和 Outcome Monitor 的 proposal/commit/reject/complete
  记录必须能够逐 Tick 回放。

## 数据计划

### 数据分层与来源

| 数据层 | 用途 | 允许来源 | 禁止事项 |
| --- | --- | --- | --- |
| Observation/Action | 共享 Expert 的行为克隆与闭环回放 | 时间同步的 Expert trajectory、Observation V2、真实 Expert commanded-target | Policy rollout 生成的 target 不能冒充 Expert label |
| Subtask/phase | condition 与 shadow transition 评估 | Expert command provenance、可部署 Predicate、时序规则和人工抽查 | 按轨迹百分比硬切 phase；在边界不确定时强造确定标签 |
| State-estimator | object/goal track、速度、抓取/支撑置信度训练和标定 | train-only 仿真 GT 或独立标注，与部署输入分离 | Runtime 读取 hidden GT；把 GT identity 混入 policy Observation |
| Formal evaluation | paired control/treatment 与 guardrail audit | 未见过的预注册环境 seed、独立 Flow seed | 使用训练/调参 seed，或因失败删除 Episode |

数据准备按以下顺序进行：

1. 对 E013 产生的 V2 trajectory 做 schema、时间戳、单位、坐标系、validity、控制器 owner 和 Action
   commanded-target 语义审计；
2. 从 Expert command provenance 和可部署事件证据生成 subtask/phase 候选标签；
3. 对 transition 前后窗口抽样人工复核，统计 coverage、class imbalance、边界歧义和不可能转换；
4. `transition_ambiguous=true` 的窗口继续参加 Action supervision，但从 phase 分类/transition loss 中 mask；
5. 按 trajectory/environment identity 划分 train/validation，禁止同一 Episode 的相邻窗口跨 split；
6. 冻结 portable manifest，记录 schema、source、split、sample count、单位/坐标约定和 SHA-256。

现有 220 条轨迹只在字段真实存在且通过 V2 审计时复用。缺失 `F_L/F_R`、动态相机位姿、validity、
controller owner 或 phase provenance 的旧轨迹不得用默认零值或 simulator state 回填后宣称为新数据；需要这些
字段的训练必须重新采集 Expert-only trajectory。原始 RGB、NPZ、视频和完整轨迹只保存在本地数据存储，
GitHub 只发布 schema、脱敏 manifest、聚合统计和小型可审计样例。

## 训练计划

正式层级归因只允许一个自变量：是否给共享 Action Expert 和 Runtime 提供显式 subtask/phase 结构。
E013 checkpoint 作为不可变 parent；两臂各自从它创建 child run，训练不得覆盖 parent。Observation、State
Estimator、Precision、Force、Geometry 和 Controller Adapter 全部冻结，只有预注册的 Action Expert
参数范围和参数量匹配的 condition adapter 可以更新。

| 项目 | Control | Treatment |
| --- | --- | --- |
| 初始化 | 同一个 E013 合格 checkpoint | 同一个 E013 合格 checkpoint |
| Qwen | 冻结 | 冻结；只产生 schema 化 semantic proposal |
| Observation/Action | 同一 V2 输入、同一 commanded-target 语义 | 完全相同 |
| 层级条件 | 固定 `null_subtask` / `null_phase` token | learned subtask/phase embedding、age 和目标参数 |
| Action Expert | 同一共享网络与可训练参数范围 | 同一共享网络与可训练参数范围 |
| 精调/Force/Geometry | 冻结 | 冻结 |
| 训练预算 | 同 sample exposure、microbatch、optimizer steps、schedule 和 seed | 完全配对 |
| checkpoint 候选 | 训练前预注册 | 相同 epoch/step 候选；不按正式 rollout 结果增补 |

第一正式实现只在 Action Expert 输入端加入小型 condition embedding/adapter，不全量解冻 Qwen，不拆分
独立 skill head。若 control 和 treatment 的张量形状必须不同，使用参数量匹配的 null-condition adapter，
并在报告中披露差异；不得把额外网络容量误写成 hierarchy 收益。

训练顺序为：接口/shape smoke → 小型 debug split 过拟合 → shadow 数据审计 → 冻结配置的 paired
control/treatment 训练 → 预注册 checkpoint 选择 → 全新 seeds 正式闭环。Debug、checkpoint selection 和
formal evaluation 使用互不重叠的 seed ledger。任何训练重启都必须从完整 trainer/RNG/sampler state 恢复；
若硬件浮点轨迹不能 bitwise 复现，保留结构和 exposure 证据并明确披露，不把近似续训说成位级一致。

## 渐进实现计划

### P0 — 契约与离线回放

- 定义 `SubtaskSpec`、`PhaseSpec`、允许转换、entry/exit/failure、timeout 和 retry 语义；
- 定义 State Estimator 输出、置信度、validity 和时间同步契约；
- 定义 transition ledger，记录 proposal、predicate snapshot、commit/reject reason 和上游 identity；
- 用 Expert trajectory 离线重放，检查 phase 标签覆盖、歧义和不可能转换。

### P1 — Shadow Executive

- Executive 只观察和记录，不改变现有策略 Action；
- 测量 phase/subtask confusion、提前切换、遗漏切换、边界抖动和 transition delay p50/p90；
- 对 release false-positive、隐藏 GT 泄漏、invalid-state 继续执行和无限恢复循环设置零容忍门禁。

### P2 — 共享 Expert 条件化

- 在同一个 Action Expert中加入 subtask/phase condition；
- 保持 Observation、Action、sample exposure、optimizer steps 和 evaluation seeds 可配对；
- 先验证条件化是否改变动作分布和交接质量，不同时启用完整 Executive 门控。

### P3 — 关键不可逆动作门控

- 先门控夹爪闭合、稳定抓取后的 lift、目标区域内的 release；
- tracking/anomaly/safety failure 统一清空 Action Chunk 和 command reference 后进入 `RecoverOrHold`；
- 门控拒绝必须产生可审计原因，不能静默裁剪或跳过 Episode。

### P4 — 完整 Executive 与有限恢复

- 启用完整 subtask/phase graph、timeout、hysteresis、retry budget 和恢复边；
- 恢复必须从重新观测后的可验证状态开始，禁止从 Action Expert猜测的隐状态继续；
- 超出 retry budget 或状态不可恢复时明确 Abort。

### P5 — 独立预注册闭环实验

- 以 E013 通过门禁后的冻结 Observation/精调层为共同基线；
- control 为无显式 Executive，treatment 为分层 Executive；
- 严格配对 Dataset、checkpoint initialization、训练预算、环境 seed 和 Flow seed；
- 先检查 system/safety/tracking/anomaly、误释放和恢复循环，再比较完整成功、五技能无条件完成数、
  mean completed skills 和交接状态质量；
- 未通过 guardrail 时保留结果并停止，不调参复用正式 seeds。

## Gate Matrix

Gate 按顺序执行；前一 Gate 未通过时，后续阶段只能做不会污染正式证据的离线工程工作。

| Gate | 进入条件 | 通过标准 | 失败动作 |
| --- | --- | --- | --- |
| G0 — E013 dependency | E013 代码 scaffold 完成 | 至少 100 个 paired unseen Episode；达到 `15/25 mm` 工程底线或更高档；有效控制 `>=20 Hz`、latency `p95<=50 ms`；system/safety/tracking/controller-overlap/stale-command 为 0；Dataset/checkpoint/config SHA 冻结 | 不启动 v1.0 正式训练；先完成 E013，不把 hierarchy 当精度补丁 |
| G1 — contracts/replay | P0 schema 与 task graph 完成 | 合法 ledger 100% 可解析；非法转换、缺失 modality、过期 command 和 hidden-GT dependency 的负例均被拒绝；Expert replay 无不可能转换 | 修契约或数据；不得接入控制器 |
| G2 — shadow | G1 通过，Executive 不控制 Action | false release proposal、未授权 lift、hidden-GT read、无限 recovery loop 均为 0；transition delay/confusion/ambiguity 完整报告；正式 hysteresis/timeout 在 actuator 接入前冻结 | 保持 shadow，补状态估计或修 Predicate |
| G3 — conditional Expert | V2 数据和 phase audit 通过 | control/treatment 初始化、sample exposure、optimizer steps、seed 和候选 checkpoint 配对；无 NaN/Inf；张量与 checkpoint identity 可验证；不使用 ambiguous phase label 监督 | 保留失败产物，新实验身份重训；不改正式 seed |
| G4 — irreversible gates | G2/G3 通过 | close/lift/release 负例中未授权动作 0；controller overlap 0；owner 切换后 stale chunk/command 0；每次拒绝有 reason code | 停在 Safe Hold；修门控，不能放宽条件绕过 |
| G5 — full Executive | G4 通过 | graph 外转换 0；retry 不超过预算；Abort/Hold 可达；所有 failure path 可回放；Runtime 不依赖 evaluator GT | 保持功能 off-by-default，不进入正式 rollout |
| G6 — paired formal evaluation | 配置、候选、指标和 seeds 已预注册 | 全部 Episode 完整；guardrail 先通过；placement 保持 E013 已达档位；满足下述 research promotion rule 才默认启用 hierarchy | 发布负结果并停止 promotion；正式 seeds 永不复用调参 |
| G7 — release/publication | G6 审计完成 | 源码、schema、配置、聚合结果、限制和 hashes 一致；敏感/大型产物未上传；本地备份完成 restore check | 不打正式 tag、不释放唯一数据副本 |

## 正式评估设计

### 对照与处理

- **Control:** 从冻结 E013 合格 parent 派生的配对 child checkpoint；同一个共享 Action Expert 使用 null
  hierarchy condition；Runtime 无显式 Subtask Executive，保留原有 Outcome/Safety 审计；
- **Treatment:** 同一 E013 初始化、Observation、Action、精调层、控制频率和训练预算；启用 subtask/phase
  condition、Executive、Phase Controller、不可逆动作门控和有限恢复；
- **Anchor:** 未继续训练的 E013 checkpoint 可用同一 seeds 运行，检查 child training 是否整体漂移，但只作
  次要诊断；hierarchy 主效应仍由参数量和训练预算配对的 Control/Treatment 计算；
- **Pairing:** 每一对使用相同环境 seed、初始状态、任务目标和 Flow sampling seed；执行顺序交错或随机化，
  避免温启动/顺序偏差；
- **Sample size:** 每臂至少 100 个 full-chain unseen Episode；另外每臂对 Reach/Grasp/Lift/Transport/Place
  各做至少 20 个 paired atomic guardrail Episode。准确 seed 区间和运行顺序在 P5 首次执行前写入版本化配置，
  不在本计划阶段提前伪造，也不在看到结果后修改。

### 指标优先级

1. **零容忍完整性：** hidden-GT dependency、数据/seed 泄漏、非零退出、NaN/Inf、controller overlap、
   stale command、结果行缺失；
2. **系统与安全：** system/safety/tracking failure、anomaly saturation、未授权 close/lift/release、
   false release、无限 recovery loop 和 retry-budget violation；
3. **主要任务指标：** full success、五技能无条件完成数、mean completed skills；
4. **机制指标：** `P(Grasp|Reach)`、`P(Lift|Grasp)`、`P(Transport|Lift)`、共同 predecessor 上的 paired
   wins/losses、transition delay p50/p90、handoff pose/velocity/contact/confidence quality；
5. **精度守恒：** 所有可测 Episode 的 final-placement XY p50/p90、within-20-mm rate 和 bootstrap 95% CI；
   任务失败但 final position 可测时仍进入误差统计。

先做 guardrail 审计，再计算效果。Research promotion 必须同时满足：

- treatment 不新增 system/safety/tracking/anomaly、false release、overlap 或 stale-command failure；
- atomic Grasp/Lift/Place 不出现 paired 净回归，Reach 下降不超过预注册容忍度；
- final-placement 仍处于 E013 已达到的同一档或更高档；
- full success 和 mean completed skills 的 point estimate 都优于 control；
- 至少一个主要任务指标的 paired bootstrap 95% CI 下界大于 0，另一个不得显示负向显著回归。

这条规则决定是否把 hierarchy 设为默认，不决定是否如实发布工程成果。若效果为零或负值，仍发布完整
配置、聚合结果和失败机制，但保持该功能 off-by-default，不增加候选、不调阈值、不复用正式 seeds。

## 时间与资源计划

### 预计工期

| 阶段 | 工作内容 | 专注开发日 |
| --- | --- | ---: |
| P0 | 契约、task graph、数据 schema、离线 replay | 2–3 |
| P1 | Shadow Executive、transition 指标和阈值标定 | 2–3 |
| P2 | condition adapter、paired train pipeline、GPU 训练 | 3–5 + GPU wall time |
| P3 | close/lift/release gate、owner handoff 与负例测试 | 2–3 |
| P4 | 完整 Executive、有限恢复和 closed-loop smoke | 2–4 |
| P5 | 预注册、paired formal rollout、独立分析 | 3–5 |
| Release | 脱敏、hash/manifest、报告、PR/tag | 1–2 |

合计约 `15–25` 个专注开发日，兼职节奏约 `4–6` 周，不含 E013 前置工作、GPU 排队和因 V2 数据不合格
而重新采集 Expert trajectory 的时间。若需要补采带真实 `F_L/F_R` 与完整相机位姿的新数据，应作为独立
数据里程碑排期，不能压缩为“自动转换旧数据”。

### 计算资源

- **RTX 4060 Laptop:** P0/P1、schema/replay、单元测试、少量 Episode smoke、文档与结果分析；不把它作为
  正式大批量 Action Expert 训练的时间承诺基础；
- **首选租用环境:** 已验证的软件镜像和 RTX 4090 24 GB，用于 P2 paired training、P4 smoke 与 P5 formal
  rollout，优先保证环境身份和可恢复性；
- **RTX 5090:** 只有在同一镜像、CUDA/PyTorch/ManiSkill smoke、checkpoint load 和数值范围核验通过，且
  单位有效 GPU 小时价格更优时使用。它可能缩短网络训练，但不能显著缩短数据审计、仿真串行 rollout、
  工程开发和报告，因此不是项目依赖；
- 所有正式 GPU run 都需要 side receipt、完整 exit code、GPU 型号级信息、软件环境 manifest 和 checkpoint
  hashes；公开材料不记录物理 GPU UUID。

## 风险登记

| 风险 | 触发证据 | 影响 | 预防/处置 |
| --- | --- | --- | --- |
| E013 未达到工程底线 | 正式 p50/p90 或 guardrail 失败 | hierarchy 建在不合格基础上，归因无效 | G0 阻断；先修 Observation/precision，不启动 v1.0 正式实验 |
| State Estimator 不可部署 | 依赖 hidden GT、track/force validity 差 | Executive 规则在真实输入下失效 | shadow 测量置信度/延迟；无有效状态时 Hold，不以零填充 |
| phase label ambiguity | 边界人工一致率低、抖动高 | condition 学到伪时序 | provenance 优先、ambiguous mask、边界窗口单独报告 |
| skill conditioning 回归 | atomic skill 或 Action 分布显著退化 | 层级收益来自容量/遗忘混淆 | 同初始化/预算/null adapter 配对；G3/G6 原子门禁 |
| controller handoff 污染 | owner overlap、stale chunk/reference | 可能造成突跳和安全失败 | 单 owner invariant；切换原子清空；负例与 ledger audit |
| false release | 未确认支撑就张爪 | 不可逆任务失败 | Executive 与 Safety 双门控；false release 零容忍 |
| 220 条单任务数据规划多样性不足 | planner proposal 单一、恢复覆盖低 | 不能支持通用 π0.5 式结论 | 限定固定任务族；不从头训练 planner；结论只覆盖当前任务 |
| 正式 seed 泄漏 | seed 出现在 train/debug/selection ledger | 效果不可置信 | 独立 seed registry；P5 前冻结；泄漏即新实验身份 |
| GPU/恢复数值漂移 | 环境 hash 变化或续训非 bitwise | 配对性和可复现性下降 | 镜像 smoke、完整 full-state checkpoint；披露 CUDA 限制 |
| 项目范围再次膨胀 | 同期加入新层、力控、任务或机器人 | 工期和归因失控 | 非范围列表与 Change Control；每次只允许一个正式自变量 |

## 交付物与存储策略

### GitHub 发布

- Executive/Phase/Plan Compiler/ledger 的源码、类型契约、schema、单元测试和最小可运行示例；
- 冻结配置、脱敏 Dataset manifest、checkpoint/evaluation identity 和 SHA-256；
- 聚合指标、置信区间、失败分类、机制分析、实验限制和可复现命令；
- 架构图、数据流、Decision Record、正式计划书和结果报告；
- 只包含小型、脱敏、可审计的 JSON/CSV 示例，不包含可还原原始轨迹的内容。

### 本地长期保存

- 原始/处理后 NPZ、RGB、视频、完整 Expert trajectory、模型权重、optimizer/full-state checkpoint；
- stdout/stderr、逐 Tick 完整 ledger、未脱敏失败证据和可恢复训练目录；
- 一个 portable manifest：相对路径、文件大小、SHA-256、生成实验 ID、依赖关系和保留级别。

释放租用实例前必须完成两份本地副本或一份本地副本加可靠对象存储副本，校验全部 hashes，并随机执行
Dataset manifest、正式 checkpoint 和聚合结果的 restore/read test。GitHub 不上传 NPZ、RGB、视频、
权重、stdout/stderr、凭据、物理 GPU UUID、敏感绝对路径或原始中断证据。

## Definition of Done

### Engineering release floor

- G0–G5 全部通过，接口、状态图、关键动作门控、有限恢复和回放可用；
- G6 正式 paired evaluation 完整执行，结果无污染且可独立复算；
- G7 发布与本地备份完成；公开文档明确模拟验证、可部署接口与真实硬件部署之间的边界；
- 即使效果 gate 未通过，也必须有负结果报告，hierarchy 保持 off-by-default。

### 推荐求职档

- Engineering release floor 全部满足；
- research promotion rule 通过，至少一个主要 paired 指标的 95% CI 支持正向收益；
- 用 matched-state/handoff diagnostics 解释改善发生在何处，并公开典型失败而不是只展示成功视频；
- final placement 不低于 E013 推荐档；若 E013 只有工程档，则只能声称 hierarchy 改善任务执行，不能声称
  已达到推荐精度档。

### 版本命名

- G0–G5 通过、G6 尚未完成：只允许 `v1.0.0-rc.N`；
- G6/G7 完成但 promotion 失败：可以发布 `v1.0.0` 的实验性 hierarchy 接口与负结果，但默认关闭，标题
  不得声称性能提升；
- G6/G7 和 promotion 均通过：发布 `v1.0.0`，可将 hierarchy 设为受配置控制的默认路径；
- 单元测试、离线 replay、shadow agreement 或挑选成功 Episode 均不能单独满足 Done。

## Change Control

- P5 formal seed 第一次执行前，冻结 task graph、Predicate、threshold、retry budget、control/treatment、
  checkpoint 候选、数据身份、样本量、指标和分析代码 hash；
- formal seed 一旦使用，任何 threshold、graph、Dataset、checkpoint、prompt、Qwen layer、精调层、Force
  或执行频率变化都必须创建新实验 ID 和全新 seeds，不能覆盖或续写原结果；
- E012、E013 的正式产物始终只读。v1.0 通过引用 hash 依赖它们，不回写其 Dataset、checkpoint 或结论；
- P0–P4 的设计变更记录在 Decision Record；若改变本计划的核心研究问题，提升 Plan revision 并重新审核
  P5 preregistration；
- 新功能请求默认进入 Future Work。只有解决当前 Gate 的最小修改才能进入 v1.0 scope。

## 非目标与结论边界

- 本计划不把十个细阶段训练为十个独立 policy；
- 不让 VLM 在 20 Hz 控制循环中自由决定 phase；
- 不用规则状态机注入仿真 GT 来掩盖视觉或状态估计不足；
- 不在 E013 结果前声称分层 Executive 能改善成功率；
- 不以单元测试、shadow agreement 或离线 phase accuracy 代替正式闭环效果；
- 不改变当前 E012/E013 冻结产物或回写其结论。

只有 P0–P4 的工程门禁通过并完成独立预注册 P5 后，才能决定该结构是否成为
`qwen-vla-v1.0` 的正式契约；在此之前它只是默认关闭的下一版本候选工程 scaffold。
