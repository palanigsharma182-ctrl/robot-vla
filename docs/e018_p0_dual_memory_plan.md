# E018-P0 抓取前 Dual Memory 实验计划书

> 状态：`proposed / planning-only`
> 本文件是工程与实验计划，不是正式预注册。正式运行前仍需提交冻结的 JSON config、数据身份、
> seed 范围、数值门槛、执行顺序和 test-once 规则。

> 2026-09-05 更新：首轮 E016 recorded-validation development replay 已完成。安全机制通过，但所有
> 41 个真实不可观察帧均为 Memory 初始化前的冷启动帧，且当前静态数据无法识别最大安全 age；因此尚不具备
> promotion 或正式预注册资格。详见
> [`E018-P0 development findings`](e018_p0_development_findings_20260905.md)。

## 1. 研究问题

在不训练新模型、不移动相机、不接入 actuator 的条件下，能否在现有显式 base-frame Goal Memory
之外增加一个 **position-only、pregrasp-only Object Memory**，安全保持抓取前短时不可观察的物体位置，
提高 `AcquireTrack` 所需状态的可用率，同时不让历史物体状态单独授权 `FinalApproach`、夹爪闭合或其他
接触动作？

E018-P0 只回答 Memory 本身是否成立。主动 front/external camera 运动属于后续
[`E018-P1`](e018_p1_active_front_reobservation_plan.md)，不得混入本实验 treatment。

## 2. Parent 与证据边界

- E016-P1 selected epoch 12 checkpoint、模型输出语义和公开结果保持只读；父 checkpoint SHA-256 为
  `c0be8769f75e4b991daa3a71878df72d2e2aaad70b5139abb4190512d6110552`。
- E016-P1 Goal Memory、write-score 与冻结规则保持对照项，不因 Object Memory 结果回写。
- E017-P0 保持正式 failed，不复用失败 epoch，不把 E018 结果描述为 E017 训练成功。
- 当前 E015/E016 已消费的 test-once 只能作为历史聚合背景，不能用于 E018 threshold、age、
  covariance、candidate-window 或 promotion 选择。
- Runtime 不得读取 simulator hidden GT；GT object position、mask、contact 和运动状态只用于离线监督、
  审计和正式评估。
- E018-P0 全程 no-actuation、no-camera-motion，`safe_for_actuator_promotion=false`。

## 3. 单一实验改动

Control 保持现有 Goal Memory，不提供跨帧 Object Memory。Treatment 只增加：

```text
reliable current object measurement
  -> candidate consistency gate
  -> position-only Object Memory in robot base frame
  -> hold across short pre-contact visual gaps
  -> phase-specific state availability
```

除 Object Memory 及其审计/状态解析外，以下项目保持不变：

- Precision U-Net 权重与解码；
- object/goal 当前帧 detection；
- Goal Memory；
- Observation V2；
- Qwen Context、Action Expert 和 Action Chunk；
- controller、Outcome Predicate 和任务成功定义；
- front camera 固定标定与 wrist camera 动态位姿语义。

## 4. 明确范围

### 4.1 本实验包含

- 目标实体仅为 `object_center`；
- 状态仅为 base-frame XYZ position 和 `3x3` position covariance；
- 抓取前无接触的 `FREE_STATIC` 假设；
- 当前测量、候选窗口、Memory 状态和失效原因的可审计记录；
- Object/Goal 独立 validity、age、covariance 和 provenance；
- `AcquireTrack` 中的短时状态 fallback；
- `StabilizePregrasp` 中对当前直接 object measurement 的强制要求；
- 离线 replay、shadow 状态解析和 fresh held-out test-once 评估。

### 4.2 本实验不包含

- object orientation 或完整 6DoF pose；
- 抓住物体后的 TCP-relative propagation；
- slip estimation、release 后 Object Memory 或动态运动模型；
- Kalman filter、learned memory、Transformer memory 或 World Model；
- wrist/front 主动运动；
- front Precision Provider 或跨相机 measurement fusion；
- 使用 Object Memory 单独授权 `FinalApproach`、`CloseUntilContact`、lift 或 release；
- 修改 E016/E017 checkpoint、训练数据或 test 结论；
- 真实机器人 actuator。

## 5. 状态与数据契约

### 5.1 Object Measurement

正式实现前应冻结一个版本化 `ObjectMeasurement`，至少包含：

```text
timestamp_s
position_base_m[3] | null
covariance_base_m2[3,3] | null
confidence
observable
projection_valid
geometry_valid
write_gate_passed
source_camera
source_model_identity
frame_semantics = position/robot-base/m/v1
```

要求：

- measurement 必须由部署可得 wrist RGB、同 tick camera/TCP pose 和冻结模型产生；
- position 与 covariance 必须同时存在或同时缺失；
- `write_gate_passed=true` 要求 observable、geometry valid、position/covariance 完整；
- model/checkpoint、camera role、时间戳和坐标系必须进入 provenance；
- 运动中、模态无效或时间不一致的帧不得成为 Memory candidate。

### 5.2 Object Memory Mode

E018-P0 只允许：

```text
UNINITIALIZED
FREE_STATIC
INVALID
```

- `UNINITIALIZED`：从未提交可靠 measurement；不得携带 position、covariance 或 source。
- `FREE_STATIC`：物体尚未接触，持有最近通过门禁的 base-frame position；`observable_now` 与
  `valid` 必须分开。
- `INVALID`：曾有状态但 age、covariance、innovation、接触、phase 或 provenance 门禁已失败；
  必须保留互异的 `invalid_reasons`。

`HELD_RELATIVE` 与 `RELEASE_PENDING` 作为 Future Work，不得用 `FREE_STATIC` 冒充。

### 5.3 Memory 状态

正式 `ObjectState` 至少记录：

```text
episode_id
mode
position_base_m[3] | null
covariance_base_m2[3,3] | null
measurement_confidence
last_observed_timestamp_s | null
state_timestamp_s
observable_now
valid
accepted_update_count
source
invalid_reasons[]
```

Memory 是状态估计，不是“上一帧复制”。所有历史值必须带 age、covariance、source 和明确 validity。

## 6. 写入、保持与失效规则

### 6.1 Candidate gate

单帧通过网络门禁后，先进入 candidate window，不立即承诺 live Object Memory。Candidate 至少检查：

- 连续稳定帧的 base-frame position 一致；
- covariance 有限且不超过冻结上限；
- 与现有 valid Memory 的 innovation 不超过冻结上限；
- object motion 与 `FREE_STATIC` 假设一致；
- gripper open；
- `F_L/F_R` 没有 object contact；
- controller tracking 正常；
- RGB、camera pose、TCP pose 和 control tick 时间一致；
- source model/config identity 未漂移。

Candidate-window 长度、position consistency、最大 covariance、最大 age 和最大 innovation 属于待冻结参数；
只能在 train/development 与 fresh validation 上选择，不能查看 fresh test 后修改。

### 6.2 Update policy

首版使用简单、可审计的策略：

```text
verified reliable candidate -> replace FREE_STATIC state
temporarily unobservable     -> hold previous state and grow/retain covariance by frozen rule
any invalidation event       -> INVALID
episode reset                -> UNINITIALIZED
```

不在 P0 中引入学习式 dynamics 或事后平滑 test 轨迹。

### 6.3 强制失效

以下任一事件必须使 `FREE_STATIC` 失效或失去控制授权：

- age 或 covariance 超限；
- non-finite state、source identity 漂移或 timestamp 回退；
- innovation 超限或可部署估计显示 object 发生运动；
- 意外 finger/object contact；
- gripper close command 或进入接触窗口；
- `grasp_candidate` / `grasp_verified`；
- camera/TCP geometry 或 controller tracking 异常；
- Episode reset。

不得因失效会降低 coverage 而延长 age、忽略接触或保留 stale state。

## 7. Phase-specific 使用权限

Memory validity 与控制授权必须分开。

### 7.1 `AcquireTrack`

用于粗粒度状态可用性：

```text
object_available_for_navigation =
    current_object_measurement_usable
    OR object_memory.valid

goal_available =
    current_goal_measurement_usable
    OR goal_memory.valid
```

Object Memory 可以桥接短时遮挡，但不得把 `observable_now=false` 写成当前直接视觉证据。

### 7.2 `StabilizePregrasp`

离开预抓取阶段前必须具有当前、fresh、稳定的 object measurement：

```text
object_ready_for_contact =
    current_object_measurement_usable
    AND current_object_measurement_fresh
    AND current_object_measurement_stable
    AND pregrasp_geometry_valid
```

`object_memory.valid` 可以提供诊断和重新观察的粗目标，但不能替代这个条件。

### 7.3 `FinalApproach` 及以后

- P0 Object Memory 不授权任何接触或抓持动作；
- 第一次进入 `FinalApproach`、检测接触或发送 close command 后，`FREE_STATIC` 状态不得继续作为
  静态物体控制依据；
- 缺少当前 object evidence 时只能 Hold/Abort，不得使用 stale Object Memory 推进任务。

## 8. 实验设计

### 8.1 Control 与 Treatment

```text
Control A: existing Goal Memory + current-frame object measurement
Treatment B: existing Goal Memory + current-frame object measurement + Object Memory
```

两臂必须使用完全相同的：

- trajectory、scene、timestep 和模型 prediction；
- checkpoint/config/data identity；
- Goal Memory rules；
- observation validity 与时间顺序；
- phase labels 和状态需求。

本实验不通过重新运行模型或增加额外图像给 Treatment 获得优势，只比较同一 prediction stream 上的
状态保持差异。

### 8.2 渐进执行

1. **Contract tests**：schema、坐标、时间、covariance、reset 和非法模式转换负例。
2. **Synthetic replay**：静态 object、短遮挡、age/covariance/innovation/contact 边界。
3. **Recorded train/validation replay**：只用于工程调试和参数候选，不形成正式 held-out 结论。
4. **Fresh validation**：冻结 candidate、age、covariance、innovation 和 phase-use 规则。
5. **Fresh test-once**：规则全部冻结后创建 claim，再执行一次 no-actuation paired replay。

正式数据 seed、数量、采集容器和运行顺序在预注册 config 中冻结。本计划不提前复用已消费 seed。

## 9. 指标与 Gate

### 9.1 状态覆盖

- current object measurement coverage；
- Object Memory coverage；
- memory valid while current object unobservable；
- uninitialized、stale、covariance-invalid 和 innovation-conflict count；
- `AcquireTrack` required-state availability；
- 相对 Control 的 paired coverage improvement。

### 9.2 误差与校准

- current/memory object world-XYZ p50、p90、max；
- 不可观察期间 memory error；
- 超过预注册 safe/catastrophic error 的 count；
- covariance coverage/calibration；
- accepted update 中的 unsafe count。

### 9.3 零容忍安全项

- Episode reset leakage；
- timestamp/source/coordinate identity mismatch；
- invalid memory 被标记为 available；
- unexpected-contact 后继续使用 `FREE_STATIC`；
- memory-only `FinalApproach` authorization；
- memory-only close/lift/release authorization；
- hidden-GT runtime dependency；
- test-before-claim 或 test 用于参数选择。

以上任一非零均阻断 E018-P1。

### 9.4 Promotion gate

只有同时满足以下条件才允许把冻结的 Dual Memory 作为 E018-P1 parent：

- 所有 contract/replay 负例按预期 fail closed；
- fresh validation 与 test-once 均无 unsafe accepted Object Memory；
- reset leakage、post-contact static-memory use 和 memory-only contact authorization 全为零；
- Object Memory 在预注册主要指标上相对 Control 提高抓取前状态可用性；
- 误差、coverage、失效原因和所有参数 identity 完整发布；
- 结果无论正负都保持 no-actuation。

若失败，冻结失败 receipt；不得通过删除困难 Episode、延长 memory age 或放宽误差门槛追认通过。

## 10. 交付物

- `ObjectMeasurement`、`ObjectMemoryConfig`、`ObjectState` 和 `ObjectMemory` 版本化契约；
- current-vs-memory phase-specific resolver；
- reset/contact/phase invalidation 与 replay ledger；
- 单元测试、属性/负例测试和 synthetic replay fixture；
- formal JSON config 与 identity verifier；
- fresh validation calibration receipt；
- test-once claim、private result 与 GitHub 脱敏 summary；
- 明确的 failure boundary 和 E018-P1 eligibility receipt。

## 11. 待正式预注册冻结的参数

- fresh validation/test seed 范围、Episode 数和场景分层；
- candidate stable-frame 数；
- object measurement confidence/write gate；
- position consistency、maximum innovation 和 covariance；
- `FREE_STATIC` maximum unobserved age 和 covariance growth；
- safe/catastrophic object error；
- object-static velocity/contact 条件；
- validation selection、test-once 与 promotion 的精确数值门槛。

这些值必须来自 train/development、工程 smoke 或 fresh validation；不能由 fresh test 反向选择。

## 12. 允许的结论

通过后最多可以声明：

> 在抓取前无接触条件下，position-only Object Memory 能在同一冻结 perception stream 上安全桥接短时
> object 不可观察，并提高 phase-required state availability；它不单独授权接触动作。

不得声明：

- 已实现完整 object pose memory；
- 已支持 held-object/slip/release memory；
- 已改善真实机器人抓取成功率；
- 已证明主动视觉有效；
- 已允许 actuator promotion。
