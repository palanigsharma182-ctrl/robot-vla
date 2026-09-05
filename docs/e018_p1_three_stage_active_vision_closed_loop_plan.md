# E018-P1 三阶段主动视觉闭环实施计划书

> 状态：`implementation-ready / development-only`
> 日期：2026-09-05
> 上位计划：[`E018-P1 受限 Front Active Reobserve 实验计划书`](e018_p1_active_front_reobservation_plan.md)
> Memory parent：[`E018-P0 抓取前 Dual Memory 实验计划书`](e018_p0_dual_memory_plan.md)
> 运动 parent：[`E018-P1 G0C rotated-motion findings`](e018_p1_g0c_rotated_motion_findings_20260905.md)
> 授权与推进决策：[`D034 — E018 仿真 development 三阶段闭环`](decisions.md)
> 当前 provider Gate：[`D036`](decisions.md)；独立协议见
> [`E018-P1-G2C Front Provider Adaptation`](e018_p1_g2c_front_provider_adaptation_plan.md)

> 2026-09-05 执行状态：Stage 1 dynamic observation 与 supervisor/replay 已通过 development-only Gate。
> G2A 证明原 E016 wrist provider 不能直接替代 front provider；G2B-CAL-v1/v2 已分别按数据生命周期不匹配
> 和 reset-first-frame contact-cache transient 冻结为协议性负结果。G2C static calibration 已通过；D047
> dynamic qualification preflight 的 capture 完成，但 public verifier 暴露了 raw pose 与 canonical SO(3)
> pose 的 representation-boundary bug，已冻结为不含 provider claim 的工程负结果。当前按 D047A 只修复该
> verifier 并以相同 noncanonical seed/view 做一次新 identity smoke；正式 `76701..76750` qualification 仍
> HOLD。在 G2C 至少一个 non-HOME viewpoint 正式通过前，Stage 2 live Object Memory commit 和
> Active-vs-Passive 对照保持关闭。

本文件把 E018-P1 的总实验协议收敛成三个连续、可编码、可测试的实施阶段：

1. 接通受限主动观察控制闭环；
2. 验证信息增益并安全更新 Object Memory；
3. 完成安全回归与 development-only 对照实验。

这三个阶段是实施工作包，不替代上位计划的 G0-G9 promotion gate。尤其不能因为软件闭环能够运行，就跳过
动态外参、front provider qualification、fresh validation 或 test-once 冻结规则。

## 1. 目标与完成定义

首版只回答一个最小问题：

> 在 `ACQUIRE_TRACK` 或 `STABILIZE_PREGRASP` 的无接触窗口中，当当前 wrist、当前 HOME front 和冻结
> Object Memory 都不能提供 phase-required object state 时，系统能否保持机械臂不动，执行一次预注册
> front-camera 平移/旋转，使用新的实际相机外参采集可靠 object measurement，返回 HOME，然后从原来源阶段
> 重新开始稳定性累计与规划？

首版闭环完成必须同时满足：

```text
dual uncertainty detected
  -> active request accepted only in an allowed source phase
  -> stale Action Chunk/controller references invalidated
  -> arm/gripper SafeHold
  -> one deterministic camera primitive
  -> actual-pose settle and synchronized capture
  -> provider-qualified candidate and information-gain gate
  -> camera returns HOME and arm hold is verified
  -> four fresh HOME-only Observation V2 frames are accumulated
  -> atomic Object Memory commit, if and only if every gate passes
  -> source invariants rechecked
  -> a new Action Chunk is generated
  -> original source phase resumes
```

以下情况不算闭环完成：

- 只让相机移动，但没有使用新图像；
- 使用 commanded pose 代替 actual pose；
- alternate view 能看到物体，但相机未验证返回 HOME；
- 直接复用主动观察前的 Action Chunk；
- 仅凭 Object Memory 进入 `FINAL_APPROACH` 或授权接触；
- 在禁止阶段拒绝动作的逻辑只存在于 runner，而没有可回放 ledger 证据；
- runtime 使用 simulator GT visibility、GT object pose 或测试集结果选择视角。

## 2. 已验证基础与当前缺口

| 能力 | 当前事实 | 本计划中的处理 |
|---|---|---|
| 相机离散动作空间 | 25 个静态位姿已筛选，10 个低位姿通过静态门禁 | 保留 10 个 development 候选，不继续扩容 |
| 相机动态运动 | 10 个低位姿、40/40 路线通过 G0C v2 | 继承运动限制和 v2 identity，不重做无关轨迹实验 |
| 相机写入时机 | `measurement_write_eligible()` 只允许 settled `COLLECT` | 作为不可绕过的第一层写入门禁 |
| Object candidate | 已有连续帧、timestamp、spread、covariance 和 provenance 检查 | 复用，不另造宽松 candidate 判定 |
| Object Memory | 已有 pregrasp-only、contact fail-closed 和 navigation/contact-ready 区分 | P1 不改变 P0 接触授权边界 |
| Executive | 已有 `ACQUIRE_TRACK`、`STABILIZE_PREGRASP`、`FINAL_APPROACH` 和被动 `REOBSERVE` | 首版使用 E018-only supervisor/context，不修改全局 20-phase graph |
| 执行器 owner | 当前 `ControllerOwner` 只表达 arm/gripper 控制权 | 单独增加 external-camera owner，不能让 camera owner 冒充 TCP owner |
| External pose | 数据层 `CameraCalibration.world_from_external` 是 Episode 固定值 | 增加版本化逐帧动态 external-pose sidecar |
| Observation V2 | 模型输入逐帧只包含 wrist camera pose | 不原地改变冻结的 42D 模型状态；active capture 独立保存，HOME 后重建四帧历史 |
| Front provider | 现有 precision checkpoint 明确绑定 hand camera | qualification 通过前只能 shadow/capture，不能写 live Memory |

由此可见，当前关键路径不是继续增加相机位姿，而是：

```text
dynamic external observation contract
  -> qualified front measurement provider
  -> phase-safe active controller
  -> candidate/information gate
  -> atomic Memory commit
  -> HOME verification and closed-loop resume
```

## 3. 首版范围与非目标

### 3.1 首版包含

- `ACQUIRE_TRACK` 和 `STABILIZE_PREGRASP` 两个来源阶段；
- object-only active recovery；Goal Memory 保持冻结，不在本切片中改变；
- 当前 wrist、当前 HOME front、Object Memory 全部不可用后的主动触发；
- 10 个已通过 G0C 的低位 development candidate pool；
- 一个由 config 决定的确定性 viewpoint schedule；
- 每 Episode 最多一次 active attempt；
- `hold -> move -> settle -> collect -> validate -> return-home -> verify -> commit -> resume`；
- matched-time/frame Passive HOME 对照；
- 全量 ledger、identity、timeout、reset 和 fail-closed 负例。

### 3.2 首版明确不包含

- 学习式 Next-Best-View；
- 连续 XYZ/yaw/pitch 优化；
- runtime 基于 GT 可见性选择视角；
- 两次以上相机移动；
- alternate-to-alternate 自由路径；
- 相机停在 alternate view 时继续执行 manipulation policy；
- contact 后、抓持中、transport 或 release 阶段的主动运动；
- 修改冻结的 VLA/precision checkpoint；
- 使用 active result 反向调 front provider；
- 正式 test-once。

如果首版单次确定性视角相对 Passive 没有可靠增益，应先冻结负结果和失败分层；不得直接扩大动作空间或改成
在线 NBV 来掩盖失败。

## 4. 关键架构决策

### 4.1 不原地改变冻结 Observation V2

`ObservationV2Frame` 与模型的 42D 状态契约已经冻结。直接加入 external camera pose 会改变训练/运行输入，
无法把结果归因于主动视角。因此首版新增版本化 sidecar：

```text
ActiveExternalObservationFrame
  request_id
  frame_id
  rgb_external[H,W,3]
  rgb_timestamp_s
  pose_timestamp_s
  intrinsic_external[3,3]
  base_from_external_camera_cv[4,4]
  commanded_world_from_external_camera_gl[4,4]
  actual_world_from_external_camera_gl[4,4]
  pose_valid
  tracking_error_position_m
  tracking_error_orientation_rad
  motion_state
  viewpoint_primitive_id
  settled
  provider_identity
  calibration_identity
  schema_version
```

active capture pipeline 与 `ObservationV2History` 分离：motion、alternate settle 和 alternate collect 帧都不得
append 到冻结模型使用的四帧历史。正常 policy 在 active window 中暂停；相机返回并验证 HOME 后，使用一个
HOME-only barrier 累计四个全新连续 Observation V2 frame，满足后才允许重新推理。active provider 只使用
sidecar 的 actual pose 完成 camera-to-base 变换；不得把 commanded pose 写成 actual pose。

不得调用会重置整个 Episode/control-step/shadow 状态的 `QwenVLAReplanLoop.reset()` 来模拟主动观察中断。
需要一个显式 mid-Episode action-history invalidation receipt：清空 ensemble、RTC overlap、executor command/action
reference，保留 episode/control-step identity，并让 action generation 严格递增。首版触发点固定在 replan
boundary，不声称能在任意 20 Hz Action Chunk 中间安全抢占。

### 4.2 camera owner 与 arm owner 分域

保留现有 `ControllerOwner` 的 arm/gripper 含义，新增独立枚举：

```text
ExternalCameraControllerOwner.NONE
ExternalCameraControllerOwner.HOME_HOLD
ExternalCameraControllerOwner.ACTIVE_REOBSERVE
ExternalCameraControllerOwner.FAILSAFE_RETURN
```

主动观察期间的所有 Tick 必须满足：

```text
arm_owner == SAFE_HOLD
gripper_owner == SAFE_HOLD_OPEN
external_camera_owner == ACTIVE_REOBSERVE | FAILSAFE_RETURN
```

camera lease 不能授予 arm/TCP 命令权。任一 owner overlap、arm command、gripper change 或 reference 未清空都
直接失败。

### 4.3 先用尽被动信息，再允许物理运动

触发解析顺序固定为：

```text
qualified current wrist measurement
  -> qualified current front-HOME measurement
  -> frozen Object Memory navigation fallback
  -> Active Front Reobserve request
```

用户目标中的“wrist 看不清且 Memory 不可靠”是必要条件，但不是充分条件；如果 HOME front 已经提供可靠状态，
额外移动没有信息价值，必须拒绝。

### 4.4 单次、确定性、默认关闭

- `enabled=false` 是所有旧配置和未知配置的默认值；
- 每 Episode 最多一次 attempt；
- runtime 只读取冻结 schedule，不计算连续最优位姿；
- 10 个低位姿保留为 qualification pool；
- 首轮 schedule 优先从 G0B 的四个 shortlist 中选定；
- 主对照前从通过 provider qualification 的候选中冻结一个 `PRIMARY_ALTERNATE`；
- config、pose、provider 或 schedule identity 不完整时 fail closed。

### 4.5 Candidate、Memory commit 与闭环恢复分离

首版采用两阶段提交语义：

1. alternate view 只产生不可变 `PendingActiveViewCandidate`；
2. 相机返回 HOME、arm hold 和 source invariants 全部验证后，才允许原子提交 Object Memory。

因此 camera return 或 HOME verify 失败不会留下一个已经生效但闭环不能恢复的 Memory 更新。即使提交成功，
该状态仍只可用于 navigation；进入 `FINAL_APPROACH` 前必须重新获得当前 direct object evidence。

## 5. 三阶段总览

| 阶段 | 核心结果 | 对应上位 Gate | 预计实现/Smoke 时间 |
|---|---|---|---:|
| 阶段一：控制闭环 | 动态 external sidecar、双不确定触发、phase/latch、camera state machine、SafeHold | G1、G3、G4、G6 的最小切片 | 3–5 小时 |
| 阶段二：信息与 Memory | provider qualification、candidate、信息增益、两阶段 Memory commit、resume | G2、G5、G7 的最小切片 | 2–3 小时 |
| 阶段三：安全与回归 | 真值表、失败注入、replay、Passive/Active development 对照 | G3–G8 回归与 promotion evidence | 3–4 小时 |

8–12 小时目标是得到最小可审计闭环，不包括完整 provider qualification 数据运行、扩大 seeds、修复 provider
或正式 test-once。若动态外参/provider 探针失败，关键路径将额外增加 1–3 天。

## 6. 阶段一：接通受限主动观察控制闭环

### 6.1 阶段目标

在不允许 live Memory write、不使用 test 数据的条件下，完成：

```text
detect -> request -> hold -> move -> settle -> collect -> return -> verify -> resume/fail
```

阶段一的成功只表示控制闭环和数据时序成立，不表示新视角信息可靠。

### 6.2 工作包 1A：动态 external-observation capability probe

先用同一 development seed 对 HOME 与一个 G0C 合格 alternate pose 做最小探针：

- 从 `sensor_param.base_camera.cam2world_gl` 读取每帧 actual pose；
- 原样保存 float32 raw sensor matrix；若因仿真舍入需投影到最近 SO(3)，必须升级 extractor identity、记录
  投影前后正交误差与修正量，并在修正量超限时 fail closed；
- 转换并记录 `base_from_external_camera_cv`；
- 验证 RGB timestamp 与 pose timestamp 分开保存；
- 验证移动帧、settle 帧和 collect 帧状态不同；
- 验证 HOME/alternate RGB digest 不同，且重复采集确定；
- 用离线 GT 只做诊断：静止 object 转换到 base frame 后应跨视角一致；
- runtime 输出不得包含 GT object pose、GT visibility 或 GT-selected viewpoint。

硬停止条件：

- 图像不随实际相机位姿变化；
- actual pose 无法逐帧获得；
- RGB/pose 不能建立时间对应；
- OpenGL/OpenCV 方向或矩阵方向无法唯一确定；
- raw rotation 到最近 SO(3) 的修正量超过冻结容差；
- 静止物体的跨视角 base-frame 几何明显不一致。

任何一项失败都先修 provider/schema，不进入 active controller actuation。

### 6.3 工作包 1B：版本化控制契约

建议新增 `src/robot_vla/precision/active_front_reobserve.py`，承载纯数据和纯状态机逻辑：

```text
ActiveFrontReobserveConfig
ActiveFrontReobserveRequest
ActiveFrontReobserveContext
ActiveFrontReobserveDecision
ActiveFrontReobserveReceipt
ActiveFrontReobserveFailure
ExternalCameraControllerOwner
ActiveFrontReobserveController
```

`ActiveFrontReobserveContext` 至少冻结：

```text
episode_id / request_id
source_phase / resume_phase
trigger_tick / trigger_timestamp_s
trigger_reasons / target_entities
arm_anchor_actual_q / arm_anchor_tcp_pose
home_pose_id / selected_primitive_id
attempt_index / camera_command_sequence_id
observation/provider/pose/config identities
action_chunk_generation_before
active_window_open_at_request
status
```

首版不向全局 `PhaseId`、`PlanCompilerConfig` 或 20-phase task graph 增加 active phase。新增
`ActiveFrontReobserveState` 作为 E018-only supervisor substate，读取 base Executive 的 source phase，但不改变
既有 plan identity。若后续正式并入 Executive，必须使用新的 versioned compiler/plan identity；不能只增加一个
默认 `false` 字段却声称旧 identity 未变。

### 6.4 工作包 1C：双不确定性触发器

首版 object-only 触发真值为：

```text
wrist_unusable = not current_wrist_object_measurement_usable
home_front_unusable = not current_front_home_object_measurement_usable
memory_unreliable = not object_memory_navigation_state_available

requestable =
    wrist_unusable
    and home_front_unusable
    and memory_unreliable
    and source_phase in {ACQUIRE_TRACK, STABILIZE_PREGRASP}
    and active_front_reobserve_window_open
    and attempt_budget_remaining
    and arm_hold_prerequisites_pass
    and camera_home_prerequisites_pass
    and failure_reason_is_viewpoint_resolvable
```

Stage 1 在 front provider qualification 前，只能用合成或预记录的 deployment-shaped evidence 验证这张真值表；
不得据此启动真实 provider-driven camera actuation。真实 `home_front_unusable` 和 trigger actuation 从 Stage 2
provider qualification 通过后才开放。

单帧失败不能触发。config 必须提供连续失败 Tick、hysteresis 和 cooldown。下列原因可触发：

```text
OUT_OF_FOV
OBJECT_OCCLUSION
LOW_VISUAL_CONFIDENCE
HIGH_LOCALIZATION_UNCERTAINTY
HIGH_GEOMETRIC_SENSITIVITY
```

下列原因只允许 Passive Reobserve/SafeHold：

```text
INVALID_SENSOR_OR_POSE
PROVIDER_IDENTITY_MISMATCH
UNSAFE_ARM_STATE
UNSAFE_CAMERA_STATE
UNKNOWN
```

### 6.5 工作包 1D：phase、latch 与状态机

新增 E018-only `ACTIVE_FRONT_REOBSERVE` supervisor state，但保留现有 `PhaseId.REOBSERVE` 的被动语义和全局
task graph。来源只允许：

```text
ALLOWED_SOURCE_PHASES = {ACQUIRE_TRACK, STABILIZE_PREGRASP}
```

Episode reset 时打开单调 latch；以下任一事件将其永久关闭至 Episode 结束：

- 第一次进入 `FINAL_APPROACH`；
- object contact；
- gripper-close command；
- `grasp_candidate`；
- `grasp_verified`；
- object 可能被推动的部署可得证据。

主动状态机：

```text
REQUESTED
  -> ACQUIRE_CAMERA_LEASE_AND_HOLD_ARM
  -> SELECT_FROZEN_PRIMITIVE
  -> MOVE_TO_VIEW
  -> SETTLE_AT_VIEW
  -> COLLECT
  -> STAGE_CANDIDATE
  -> RETURN_HOME
  -> VERIFY_HOME_AND_ARM_HOLD
  -> REBUILD_FOUR_HOME_FRAMES
  -> RECHECK_SOURCE_INVARIANTS
  -> COMMIT_AND_RESUME | FAIL
```

阶段一保留完整状态形状，但把 `STAGE_CANDIDATE` 固定为 shadow candidate、把 `COMMIT_AND_RESUME` 固定为
no-write resume；阶段二通过 provider 和信息门禁后才允许 live commit。

进入 active phase 时必须原子执行：

- 清空剩余 Action Chunk；
- 清空 temporal ensemble/RTC proposal history；
- 失效 previous commanded-target reference；
- 锚定 actual q/TCP；
- gripper 保持 open；
- 获取唯一 camera-motion lease；
- 把 reset generation 写入 ledger。

恢复时必须：

- 相机 HOME pose 验证通过；
- actual q/TCP hold 验证通过；
- motion/alternate capture 未进入 `ObservationV2History`；
- 四个全新、连续、全部来自 HOME 的 Observation V2 frame barrier 通过；
- source phase entry/invariant 重新检查；
- phase age 和 stability counter 清零；
- 使用当前实际状态构造新 observation；
- Action Chunk generation 严格递增。

### 6.6 阶段一预计修改范围

优先最小修改：

- 新增 `src/robot_vla/precision/active_front_reobserve.py`；
- 新增 E018 dynamic-external observation sidecar 模块；
- 首版不修改 `src/robot_vla/executive/contracts.py`、`planning.py` 或 `executive.py`；
- 复用 `src/robot_vla/precision/active_front_camera.py` 的轨迹与 write eligibility；
- 新增 `tests/test_e018_active_front_reobserve.py`；
- 新增 dynamic external observation contract 测试。

不得在阶段一修改模型权重、训练数据格式或 P0 Object Memory 的默认行为。

### 6.7 阶段一退出门禁

- capability probe 全部通过；
- 现有 20-phase plan identity、ledger 语义和 executive tests 保持不变；
- 允许阶段可以形成 request；禁止阶段 100% 拒绝；
- latch 关闭后不能通过 recovery/replan 重新打开；
- 全状态路径可以 replay，digest 稳定；
- active phase 全程 arm/TCP/gripper 无命令和无漂移；
- motion/unsettled frame 的 Memory-write eligibility 为 false；
- 成功路径返回 HOME、通过四帧 HOME barrier 后才生成新 Action Chunk；
- 失败路径保持 SafeHold/Abort，不恢复旧命令；
- test reads = 0，live Memory writes = 0。

## 7. 阶段二：信息增益、Provider 资格与 Object Memory 回写

### 7.1 阶段目标

让 alternate-view settled frames 形成一个来源明确、几何正确、跨帧一致的 object candidate；只有相对被动基线
确有信息改善，且相机已返回 HOME、机械臂仍保持时，才允许原子更新 Object Memory 并恢复来源阶段。

### 7.2 工作包 2A：front provider qualification

正式接受任何 active measurement 前，逐 viewpoint 评估冻结 provider：

- object observability precision/recall；
- base-frame XYZ error p50/p90/max；
- covariance calibration；
- unsafe acceptance；
- source camera、primitive、checkpoint 和 calibration identity；
- HOME 与每个 candidate 的分层结果；
- camera-pose OOD 检查。

10 个低位姿是运动合格池，不是 provider 合格池。某个位姿 provider qualification 失败，就从可运行 schedule
移除并继续评估其余候选；如果没有候选通过，则冻结 provider-qualification 负结果，建立独立上游 provider
adaptation/training identity，冻结新 parent 后重新进入阶段二。不得用 GT 替代 provider，也不得在查看
Active-vs-Passive 结果后微调 provider 并沿用同一实验身份。

首版 Object Memory active write 只接受冻结的精确 provenance pair：

```text
(source_camera, primitive_id, provider_identity, calibration_identity)
```

不能把 P0 的单一 source check 放宽为任意 front camera。若需支持 wrist 和多个 front primitive，应新增 P1
版本化 allowlist/policy，且 P0 默认 contract 和测试保持不变。

### 7.3 工作包 2B：信息增益契约

触发前保存同一 request 的被动基线：

```text
PassiveBaselineEvidence
  wrist_evidence
  front_home_evidence
  object_memory_resolution
  baseline_timestamp_s
  baseline_provider_identity
```

alternate view 使用已有 `ObjectWriteEvidence` 形成部署侧分数，并通过已有
`ObjectCandidateWindowVerifier` 检查连续帧。接受条件分为三层：

```text
structural_gate =
    motion_state == COLLECT
    and settled
    and actual_pose_valid
    and timestamp_skew_within_budget
    and provider/provenance qualified
    and object measurement structurally eligible

absolute_gate =
    candidate window verified
    and score >= frozen_active_write_threshold
    and covariance/std/innovation within frozen limits

relative_gain_gate =
    candidate improves the best computable passive baseline
    by at least frozen_min_information_gain
```

若被动基线结构上不可计算，不把它偷偷设为零；使用单独 reason
`BASELINE_UNAVAILABLE_CANDIDATE_ABSOLUTELY_VALID`，并要求更严格的绝对阈值。所有阈值只能在
train/development/fresh validation 上冻结，runtime 与正式 test 不读取 GT。

至少记录以下信息增益分量，而不是只保留一个最终布尔值：

- `ObjectWriteEvidence.score` 改善；
- visibility/projection/object-mask 分量；
- normalized entropy 下降；
- radial sigma 下降；
- covariance trace/max-std 下降；
- multi-frame position spread；
- 相对旧 Memory 的 innovation；
- candidate accepted/rejected reason。

### 7.4 工作包 2C：两阶段 Memory commit

阶段二使用如下顺序：

```text
COLLECT settled window
  -> verify PendingActiveViewCandidate
  -> freeze candidate digest
  -> RETURN_HOME
  -> verify HOME + arm hold + latch still open
  -> accumulate four fresh HOME-only Observation V2 frames
  -> re-evaluate candidate age and source invariants
  -> atomic Object Memory commit
  -> emit commit receipt
  -> resolve NAVIGATION state
  -> rebuild observation and resume source phase
```

任一检查失败：

- candidate 不提交；
- Object Memory 保持提交前版本或按既有安全规则失效；
- 不允许部分字段更新；
- 不允许恢复旧 Action Chunk；
- 进入 SafeHold/Abort，并记录唯一 failure reason。

`STABILIZE_PREGRASP` 的特殊限制：active candidate 最多恢复 navigation/pregrasp state。返回 HOME 后如果没有
新的 current direct object evidence，不能进入 `FINAL_APPROACH`；控制器应继续 SafeHold、调整 pregrasp 或终止，
不能把刚才 alternate-view 的 Memory 当作 contact-ready evidence。

### 7.5 工作包 2D：确定性 viewpoint schedule

development pool 保留全部 10 个 G0C 通过位姿：

```text
LEFT_LOW__CENTER
LEFT_LOW__YAW_LEFT
LEFT_LOW__YAW_RIGHT
LEFT_LOW__PITCH_UP
LEFT_LOW__PITCH_DOWN
RIGHT_LOW__CENTER
RIGHT_LOW__YAW_LEFT
RIGHT_LOW__YAW_RIGHT
RIGHT_LOW__PITCH_UP
RIGHT_LOW__PITCH_DOWN
```

首轮运行 schedule 只从已通过 provider qualification 的候选中产生，并满足：

- 相同 trigger evidence、attempt index 和 config identity 得到相同 primitive；
- 每 Episode 最多一次 attempt；
- 不读取 object/goal GT；
- 不做 alternate-to-alternate 路径；
- 优先评估 G0B 四个 shortlist；
- 主对照前冻结一个 `PRIMARY_ALTERNATE`；
- 其余合格位姿保留为开发消融，不在主对照中动态挑最好结果。

这样先回答“真实平移基线是否有净收益”，把多视角在线选择留给后续 E018-P2。

### 7.6 阶段二预计修改范围

- 扩展 `src/robot_vla/precision/active_front_reobserve.py` 的 candidate/receipt；
- 复用或小幅扩展 `src/robot_vla/precision/object_observability.py`；
- 以版本化 P1 policy 扩展 `src/robot_vla/precision/object_memory.py`，不改变 P0 默认值；
- 增加 front provider adapter/qualification runner；
- 新增 `configs/e018_p1_g1_active_closed_loop_development_v1.json`；
- 新增 provider、information-gain、atomic-commit 单元测试；
- 新增 qualification findings 和冻结 identity receipt。

### 7.7 阶段二退出门禁

- 至少一个 G0C 合格位姿通过 front provider qualification；
- runtime candidate 不依赖 GT；
- commanded/actual pose、frame convention、timestamp identity 全部可回放；
- motion/settle/return frame 不能生成 verified candidate；
- candidate 必须同时通过 structural、absolute 和适用的 relative gate；
- HOME/arm verify 前 Memory commit count = 0；
- HOME/arm verify 后只允许一次原子 commit；
- provider、pose、config 或 candidate digest 漂移均 fail closed；
- Memory-only resolution 的 `contact_authorized=false`；
- resume 后 Action Chunk generation 更新且 phase stability 从零累计；
- test reads = 0。

## 8. 阶段三：安全、回归与 development-only 对照

### 8.1 阶段目标

证明首版闭环不仅能跑成功路径，而且对禁止阶段、数据异常、控制异常、身份漂移和恢复失败都保持 fail closed；
随后用 paired development seeds 比较 Active 与匹配预算的 Passive HOME reobserve。

### 8.2 单元测试真值矩阵

触发器至少覆盖：

| Wrist | HOME front | Object Memory | 来源阶段 | Window | 预期 |
|---|---|---|---|---|---|
| usable | 任意 | 任意 | allowed | open | 不触发 |
| unusable | usable | 任意 | allowed | open | 不触发 |
| unusable | unusable | valid | allowed | open | 不触发 |
| unusable | unusable | invalid | `ACQUIRE_TRACK` | open | 可请求 |
| unusable | unusable | invalid | `STABILIZE_PREGRASP` | open | 可请求 |
| unusable | unusable | invalid | `FINAL_APPROACH` 及以后 | 任意 | 拒绝 |
| unusable | unusable | invalid | recovery 重入 allowed 名称 | closed | 拒绝 |
| sensor/pose invalid | 任意 | 任意 | allowed | open | Passive/SafeHold |

每个 `PhaseId` 都要显式参数化测试，不能只抽查一个禁止阶段。

### 8.3 状态机与故障注入矩阵

至少覆盖：

- camera lease 获取失败；
- move timeout；
- position/orientation tracking 超限；
- settle timeout；
- collect frame 不足；
- RGB/pose timestamp skew；
- actual pose 缺失或 commanded pose 冒充 actual；
- provider/checkpoint/calibration/primitive identity 漂移；
- candidate score 无改善；
- candidate covariance、spread 或 innovation 超限；
- return timeout；
- HOME verification 失败；
- active 期间 arm q/TCP 漂移；
- active 期间 contact 或 gripper state change；
- latch 在请求后、提交前关闭；
- Episode reset 发生在任意 active substate；
- runner crash 后 replay；
- duplicate request/command/commit receipt；
- attempt budget exhausted；
- stale Action Chunk、proposal history 或 command reference 泄漏。

每个失败路径必须断言：

```text
no illegal camera motion
no arm/TCP manipulation command
no gripper close
no live Memory partial commit
no stale Action Chunk resume
deterministic failure reason
SafeHold/Abort terminal behavior
replay digest stable
```

### 8.4 集成 smoke routes

使用 development seeds 运行以下最小路线：

1. `ACQUIRE_TRACK` 成功 active recovery；
2. `STABILIZE_PREGRASP` 状态恢复但 contact evidence 不足；
3. 无信息增益，正常返回 HOME 后失败关闭；
4. 运动中注入 arm drift；
5. collect 中注入 timestamp/provider mismatch；
6. return-home 失败；
7. `FINAL_APPROACH` 请求被拒绝且 camera command count 为零；
8. latch 关闭后的 recovery/replan 不能重新开启窗口；
9. Episode reset 清空 request、lease、candidate、attempt 和 Memory pending commit；
10. 同一输入 replay 得到完全相同的 decisions、receipts 和 digests。

G0C 的运动上限继续作为硬约束：

```text
max linear velocity       <= 0.31 m/s
max linear acceleration   <= 0.70 m/s^2
max angular velocity      <= 0.75 rad/s
max angular acceleration  <= 2.50 rad/s^2
```

并保持 arm joint drift、TCP drift、unexpected contact、illegal write frame 为零。

### 8.5 Paired development 对照

主比较：

```text
Passive B:
  frozen Dual Memory + same qualified front provider
  + HOME SafeHold
  + matched wall-clock and settled-frame budget

Active C:
  same Dual Memory + same provider
  + one frozen PRIMARY_ALTERNATE
  + same validation and Memory rules
```

两臂必须匹配：

- scene/environment seed；
- 初始机器人、object 和 goal 状态；
- trigger tick；
- provider/checkpoint/config；
- observation frame 数；
- wall-clock budget；
- 后续 Flow sampling seed；
- failure 与 success 定义；
- 执行顺序随机化或交错，并记录顺序。

`Always-move` 只作为诊断臂，不能代替主 Passive 对照。第一轮使用少量 development seeds 做 smoke；没有状态
机或 provider 结构性失败后再扩到预定 development 集。整个阶段不得读取 fresh test。

### 8.6 指标

主要指标：

- trigger-conditional reliable-state recovery；
- trigger-conditional closed-loop resume；
- Active 相对 Passive 的配对差；
- accepted information gain；
- false recovery；
- Memory commit acceptance/conflict；
- HOME return success；
- 每次恢复 latency、frames 和 path length。

次级任务指标：

- 困难场景任务成功率；
- 首次抓取成功率；
- pregrasp 重规划次数；
- abort/timeout；
- 每 Episode 新增时间。

零容忍指标：

- 禁止阶段 camera command；
- latch 关闭后 camera command；
- active 期间 arm/TCP manipulation command；
- unexpected contact/gripper change；
- motion/unsettled frame Memory write；
- HOME verify 前 Memory commit；
- memory-only `FINAL_APPROACH`/close authorization；
- hidden-GT runtime dependency；
- identity mismatch acceptance；
- Episode leakage、duplicate commit 或 retry-budget exceed。

任一零容忍项非零，都阻断 promotion，无论任务成功率是否提高。

### 8.7 阶段三退出门禁

- 新增单元/集成测试全部通过；
- 冻结 Goal Memory 和 P0 Object Memory 回归全部通过；
- Ruff 通过；
- 同输入 replay digest 一致；
- Active/Passive 无 test reads；
- 所有零容忍计数为零；
- Active 不增加 accepted-state catastrophic/unsafe error；
- Active 相对 matched Passive 的 recovery 改善达到 validation 前冻结的门槛；
- 默认配置仍为 active disabled；
- 形成可审计 development findings、config digest、source digest、provider/pose identity 和运行 receipt。

阶段三通过只允许进入 fresh-validation/freeze 流程；不自动授权 G9 test-once。

## 9. 配置冻结清单

首个 development config 至少包含以下组：

```text
experiment_identity
  experiment_id
  parent_commit
  parent_g0c_receipt
  development_only=true
  test_reads_allowed=false

feature
  active_front_reobserve_enabled
  object_only=true
  allowed_source_phases
  max_attempts_per_episode=1
  require_return_home=true

trigger
  consecutive_failure_ticks
  hysteresis
  cooldown_ticks
  viewpoint_resolvable_reasons

viewpoint
  library_identity
  qualified_candidate_ids
  primary_alternate_id
  deterministic_schedule_identity

motion
  path_identity
  velocity/acceleration limits
  move/settle/collect/return timeouts
  position/orientation/tracking tolerances

observation
  dynamic_external_schema_identity
  RGB/pose synchronization budget
  stable frame count
  frame convention
  maximum raw-to-SO3 correction

provider
  checkpoint/model identity
  qualification receipt
  per-viewpoint allowlist

information_gain
  absolute write threshold
  minimum relative gain
  baseline-unavailable absolute threshold
  covariance/spread/innovation limits

safety
  arm q/TCP hold tolerances
  contact threshold
  latch-closing events
  fail-safe return policy

evaluation
  Passive/Active pairing
  seed manifest
  matched frame/time budget
  promotion thresholds
```

未确定数值在 development config 中必须显式标记 provisional；不能使用代码隐藏默认值代替后续冻结。

## 10. Ledger 与交付物

每次 active request 至少输出：

- trigger evidence 和 resolver trace；
- source phase、latch、attempt budget；
- arm/gripper/camera owner timeline；
- camera commanded/actual pose ledger；
- RGB/pose timestamps 和 validity；
- settle/collect window；
- Passive baseline evidence；
- candidate components、information gain 和 rejection reasons；
- pending candidate digest；
- HOME/arm verification；
- Memory pre/post version 与 commit receipt；
- Action Chunk reset/new-generation identity；
- resume/failure result；
- config/source/provider/pose/schema identities；
- test-read counter。

预期代码/文档交付：

- E018 active-reobserve contract/controller；
- dynamic external-observation sidecar；
- front provider qualification runner/report；
- E018-only supervisor 与 mid-Episode Action-history invalidation receipt；
- deterministic viewpoint schedule；
- information-gain evaluator；
- atomic Memory commit receipt；
- Passive/Active paired development runner；
- 单元、集成、replay 和 failure-injection tests；
- development config、findings 和 artifact receipt。

## 11. 实施顺序与提交边界

建议保持三个可独立回滚的提交：

1. `feat(e018): add active front reobserve controller contract`
2. `feat(e018): gate active-view information and memory commit`
3. `test(e018): add active closed-loop safety and paired evaluation`

每个提交只在上一阶段退出门禁通过后开始。旧 E013-E017 和 E018-P0/G0 默认路径必须继续通过；若某阶段需要
修改核心 Observation/model shape、训练 checkpoint 或任务成功定义，应停止并新开实验 identity，而不是在本
计划内扩大范围。

### 11.1 回退、继续与三阶段终止语义

单条技术路线允许 no-go 和回退，但一次 gate 失败不得被解释为整个三阶段项目终止。决策 Agent 对每次失败
承担以下责任：

1. 冻结失败时的 config、source、provider、seed、ledger 和原始结果；
2. 区分实现错误、接口能力缺失、provider 不合格、视角无信息增益和研究假设负结果；
3. 回退到最近一个已通过且 identity 完整的 checkpoint；
4. 在不改变用户目标和安全边界的前提下，选择替代路线；
5. 若替代路线改变 schema、provider、候选 schedule 或核心假设，创建新的 config/experiment identity；
6. 为替代路线重新定义进入条件和退出门禁；
7. 向工程 Agent 下发下一项具体任务，并持续追踪到阶段重新完成；
8. 在 findings 中同时报告失败路线和最终路线，不能只保留表现较好的结果。

默认回退路径：

| 失败点 | 不允许的处理 | 必须继续的路径 |
|---|---|---|
| dynamic external sidecar/probe 失败 | 伪造固定外参或 commanded pose | 新建 schema/provider capability identity，补实际 pose/时间链后重跑阶段一 |
| Executive owner/latch/回放失败 | 删除负例或放宽禁止阶段 | active 默认关闭，修复最小 contract，完整重跑阶段一回归 |
| 某个 viewpoint provider qualification 失败 | 根据 Active 结果临时调模型 | 从其余运动合格候选继续逐项资格评估 |
| 全部 viewpoint provider qualification 失败 | 用 simulator GT 代替 provider | 建立独立上游 provider adaptation/training 实验，冻结后以新 parent 重启阶段二 |
| candidate 无信息增益 | 降低 write/safety threshold | 在 development 中按冻结规则评估其他 qualified candidate，并使用新 schedule identity |
| Memory commit/resume 失败 | 绕过 HOME 或 current-evidence gate | 保持 pending/no-write，修复提交或恢复协议后重跑阶段二 |
| Active 不优于 Passive | 只报告成功 seed 或扩大 test 次数 | 完成预定 paired runs、失败分层和候选消融，形成可解释负结果 |
| GPU/远端暂时不可用 | 丢弃已经完成的 evidence | 保存 receipt，继续本地纯 contract/test；恢复后从最近 checkpoint 续跑 |

“推进到结束”不等于必须得到正结果。三阶段的合法结束有两种：

- **正结果完成**：安全/身份门禁全过，Active 相对 Passive 达到冻结的 improvement gate；
- **负结果完成**：安全实验和预定替代路线全部完成，Active 仍未达到 improvement gate，并给出按触发类型、
  viewpoint、provider、no-gain、state-recovered-but-not-resumed 等维度的证据化原因。

不得以单个实现失败、单个位姿失败或一次运行失败提前结束整个项目；也不得为了满足“继续”而越过安全、
身份、provider 或 test-once 门禁。只有需要新增用户权限、外部资源长期不可用或必须改变研究问题时，才把
问题升级给用户决定。

### 11.2 轻量审查预算

工程 Agent 对实现、常规代码自审、测试和结果完整性负责；决策 Agent 用于快速发现实验逻辑和方向性错误，
不做逐行代码审查，也不与工程 Agent 重复执行完整实验：

- 审查总投入目标为工程/实验时间的约 10%–15%；
- 每阶段只安排一次入口审查和一次退出审查；
- 决策 Agent 日常只读取工程自审摘要、关键 diff、针对性测试摘要、零容忍计数、identity 和 receipt；
- 完整日志只在指标异常、identity 不一致或零容忍项非零时按需展开；
- 普通实现问题由工程 Agent 直接修复并继续，不等待长篇评审；
- 决策 Agent 每阶段只抽样核验 allowed phase/latch、actual pose 来源、HOME 后 Memory commit、
  Active/Passive 匹配等高风险证据；
- 决策输出限制为关键判断、最多五个硬门禁、go/no-go/回退决定和下一任务单。

## 12. 时间线与决策点

### T+0 至 T+1 小时

- 运行 dynamic external capability probe；
- 决定 sidecar 的确切矩阵方向、时间语义和 schema identity；
- 若失败，停止 controller actuation 工作，优先修 provider/schema。

### T+1 至 T+5 小时

- 完成阶段一 contract、trigger、phase/latch、owner 和纯状态机；
- 完成 shadow recommendation 与成功/失败 smoke；
- Memory write 保持关闭。

### T+5 至 T+8 小时

- 完成 provider qualification adapter 与小规模 capability/qualification smoke；
- 完成 information-gain/candidate/two-phase commit；
- 冻结 development schedule 和至少一个合格 alternate。

### T+8 至 T+12 小时

- 完成测试真值矩阵、故障注入、replay 和集成 smoke；
- 生成第一份 development-only closed-loop receipt。

### 后续 0.5–1.5 天

- 扩大 development seeds；
- 运行 matched Passive/Active；
- 分层分析 no-trigger、no-gain、state-recovered-but-not-resumed 和安全失败；
- 决定是否具备进入 fresh validation 的资格。

## 13. 最终允许结论

三阶段全部通过后，最多允许声明：

> 在 development 环境的 `ACQUIRE_TRACK` 与 `STABILIZE_PREGRASP` 无接触窗口中，当当前 wrist、HOME front
> 和冻结 Object Memory 都不能提供所需 object state 时，一个单次、确定性、预注册的 front alternate-view
> 原语能够在机械臂 SafeHold、动态外参有效、provider 合格、信息增益验收、相机归位和 source invariant
> 复核后更新 navigation-only Object Memory，并重新接回来源阶段。

在 G8 paired validation 和 G9 fresh test-once 完成前，不得声明主动视觉提高正式任务成功率，也不得声明已
实现通用 NBV、真实硬件安全或接触阶段主动观察。
