# E018-P1 受限 Front Active Reobserve 实验计划书

> 状态：`proposed / planning-only`
> 前置依赖：[`E018-P0 抓取前 Dual Memory`](e018_p0_dual_memory_plan.md) 通过全部门禁并冻结 parent
> identity。本文件不是正式预注册；相机机构、视角库、动作限制、seed、阈值和 test-once 规则仍需在
> 执行前冻结到版本化 config。

> 2026-09-05 development evidence：E018-P0 recorded validation 中有 8/20 episodes 从未初始化 Object
> Memory，41/41 个 GT-unobservable pregrasp 帧均为冷启动且 Memory 不可用。这支持主动观察的触发需求，
> 但不解除 P0 的正式前置条件；当前只能进入 P1 的 no-actuation camera/provider feasibility，不能开始正式
> Active-vs-Passive 评价。详见
> [`E018-P0 development findings`](e018_p0_development_findings_20260905.md)。

> 2026-09-05 G0 development evidence：ManiSkill 3.0.1 的独立 `base_camera` 已完成 4 个 development seeds ×
> 4 个备用锚点共 16 次 `HOME -> 单个 alternate -> HOME` 的非零时延运动，16/16 路线通过相机跟踪、
> HOME 返回、机械臂/TCP 静止、open-gripper、无接触和运动帧失效门禁。该结果只通过
> **G0 Simulator/API feasibility**，视角仍是 provisional；右侧两个锚点在少量 seed 上存在 object/goal
> 完全不可见，不能直接冻结为正式 viewpoint library。详见
> [`E018-P1 G0 development findings`](e018_p1_g0_development_findings_20260905.md)。

> 2026-09-05 G0B development evidence：已在 50 个 development seeds 上静态筛选
> `5 个平移位置 × 5 个离散朝向 = 25 个完整位姿`。3750/3750 帧和 1250/1250 个重复采集单元通过
> 完整性门禁；`LEFT_LOW`、`RIGHT_LOW` 的全部 10 个朝向达到当前静态阈值，高位锚点的全部朝向均未
> 达标。自动 shortlist 因每锚点最多两个的约束输出四个优先候选，但正式库尚未冻结。详见
> [`E018-P1 G0B viewpoint findings`](e018_p1_g0b_viewpoint_screen_findings_20260905.md)。

> 2026-09-05 G0C development evidence：G0B 的 10 个低位静态合格位姿已完成 4 个新 development seeds ×
> 10 poses 共 40 条有时延 `HOME -> full pose -> HOME` 路线，40/40 通过。平移与局部 yaw/pitch offset
> 共用 2.0 秒五次平滑进度；最坏角速度/角加速度为 `0.5391 rad/s / 0.8252 rad/s²`，arm joint drift、
> TCP position drift、contact、非法 write eligibility 和 Memory write 均为零。该门禁不包含真实相机
> 动力学或 provider 资格。详见
> [`E018-P1 G0C rotated-motion findings`](e018_p1_g0c_rotated_motion_findings_20260905.md)。
> 首轮结果因 loader 未逐项拒绝 pose 数值漂移而保留为审计；相同轨迹在 fail-closed 加固后的 v2
> 逐帧复现并通过，后续以 v2 为 parent。

> 三阶段实施细化见
> [`E018-P1 三阶段主动视觉闭环实施计划书`](e018_p1_three_stage_active_vision_closed_loop_plan.md)。该文件把
> 动态 external observation、受限控制闭环、信息增益/Object Memory 两阶段提交以及安全回归拆成可执行
> 工作包，但不替代本文 G0-G9 的顺序和 promotion gate。

> 2026-09-05 provider qualification 更新：G2A/G2B 已排除“直接复用 wrist provider + 只做 covariance
> 校准”作为可接受路线；当前按 D036 执行新的 seed-disjoint、development-only G2C front provider
> adaptation。冻结协议见
> [`E018-P1-G2C Front Provider Adaptation`](e018_p1_g2c_front_provider_adaptation_plan.md)。该上游 Gate
> 通过前不开放 live Object Memory write、Active-vs-Passive 评价、canonical runtime 或 actuator 权限。

## 1. 研究问题

当抓取前当前 wrist/front-home measurement 无法安全提供 phase-required object/goal state，而且冻结的
Object/Goal Memory 也不能提供该状态时，一个与机械臂解耦、具有离散二维平移和受限旋转能力的
**front active reobserve 原语**，是否比匹配时间和帧数的原地 Passive Reobserve 更容易恢复可靠状态，
并在不复用旧 Action Chunk、不移动机械臂、不进入接触阶段主动采集的前提下安全接回原闭环？

E018-P1 的主要 estimand 是 trigger-conditional reliable-state recovery，不是完整 Pick-and-Place 成功率。

## 2. 相机机构与首版解释

### 2.1 物理形态

本文暂按以下可实现机构规划：

- front camera 与 Franka arm/TCP 独立；
- 水平方向有两个经过标定的离散位置：`LEFT`、`RIGHT`；
- 竖直方向有两个经过标定的离散位置：`LOW`、`HIGH`；
- 平移锚点默认是二者笛卡尔积，最多形成
  `LEFT_LOW / LEFT_HIGH / RIGHT_LOW / RIGHT_HIGH` 四个备用观察位置；
- 另有正常策略使用的 `HOME` 位姿；
- 每个平移锚点允许预注册的 yaw/pitch，roll 首版固定；
- 正式运行只允许 `HOME <-> 某个备用锚点`，不允许备用锚点之间自由移动。

如果“横向和竖向两个离散位置”最终指总共两个备用位置，而不是每个轴各两个位置，状态机和数据契约
不变，只缩小 `ViewpointPrimitive` 库；正式数量由 config identity 冻结。

### 2.2 为什么同时需要平移和旋转

- 平移改变观察基线，主要解决遮挡、视差和几何条件差；
- yaw/pitch 负责让固定 workspace ROI 在每个锚点都位于有效视场；
- 只有旋转而没有平移，通常不能解除同一视线上的遮挡；
- 首版旋转不是模型自由动作，也不根据 simulator hidden GT 做 target look-at。

每个 `ViewpointPrimitive` 都包含完整、固定的平移锚点和旋转姿态。运行时选择的是有限 primitive，
不是分别连续优化 XYZ 和旋转。

## 3. Parent、前置资格与单一 Treatment

### 3.1 Parent 必须冻结

- E016-P1 selected epoch 12 checkpoint、模型输出语义与公开结果；
- E016 Goal Memory/write rules；
- 通过 E018-P0 门禁的 position-only、pregrasp-only Object Memory；
- Executive task graph、manipulation controller 和成功定义；
- Qwen/Action Expert 权重、Prompt、Flow sampler；
- phase allowlist、active-view window、失败语义和 paired-evaluation protocol；
- 正式 P1 使用的 front measurement provider、视角库和相机标定 identity。

### 3.2 Front measurement provider 是硬前置，不得隐藏

当前 E016 Precision 数据和 checkpoint 明确绑定 `hand_camera`。不能把同一模型直接用于动态 front 图像后，
默认它仍然可信。正式主动运动前必须完成一个 no-claim 的资格门禁：

1. 在 train/development 与 fresh validation 上收集所有候选 front 位姿的 RGB、动态外参和离线 GT 标签；
2. 先评估冻结的 E016 checkpoint 是否能跨 camera role 和 viewpoint 使用；
3. 按每个 viewpoint 单独报告 observability、localization、covariance 和 unsafe acceptance；
4. 只有全部将用于正式运行的 viewpoint 都通过冻结门槛，才可把适配器冻结为
   `PrecisionExternalProvider`；
5. 若不通过，E018-P1 在此处停止。训练/微调 front provider 必须使用新的上游实验/config identity，
   冻结后再重新预注册 P1，不能在本实验中边看 Active 结果边修改 perception。

离线 GT 只用于标签、资格评估和正式指标；runtime viewpoint selection、measurement、Memory write 和控制
都不得读取 GT。

### 3.3 主比较中的唯一 Treatment

Control 与 Treatment 都使用同一个冻结的 `PrecisionExternalProvider`。唯一差别是 camera pose 是否变化：

```text
phase-required state unavailable after current wrist/front-home + Dual Memory
  -> Control: arm/camera SafeHold + matched-time/frame Passive Reobserve at HOME
  -> Treatment: arm SafeHold + one frozen front viewpoint primitive
  -> same candidate validation + Memory rules
  -> camera HOME verification
  -> source phase invariant re-check + newly generated Action Chunk
```

不得在 Active 与 Passive 两臂之间同时改变 provider、checkpoint、Memory threshold、采集帧数、等待预算或
manipulation policy。

## 4. 明确范围

### 4.1 本实验包含

- front camera 独立于机械臂的离散二维平移；
- 每个平移锚点的预注册 yaw/pitch；
- wrist camera 被动采集，Franka arm/TCP 保持不动；
- 有限、预注册、可审计的 `ViewpointPrimitive` 库；
- `hold -> move -> settle -> collect -> validate -> return-home -> verify -> replan`；
- 来源 phase allowlist 和 Episode 级 active-view window latch；
- 独立 camera actuator owner、request/receipt 和 ledger；
- per-frame dynamic external-camera pose、时间与 tracking 契约；
- 与 matched-time Passive Reobserve 的 paired comparison；
- 仿真中具有非零运动时间、速度/加速度和路径约束的相机运动。

### 4.2 本实验不包含

- 移动 wrist camera、Franka arm/TCP 或 gripper 来观察；
- 连续自由相机轨迹、RL/learned Next-Best-View 或 VLA 自由生成相机动作；
- runtime 使用 simulator hidden GT 选择视角或判断恢复成功；
- 接触后、夹取、transport、deposit 或 release 阶段的相机移动；
- Object orientation、held-relative memory、slip/release memory；
- 未经资格验证的 wrist checkpoint 直接冒充 front provider；
- 相机瞬移且运动时间记为零的正式结论；
- 将动态视角图像直接喂给只在固定 HOME 视角下冻结的 manipulation policy；
- 真实机器人 actuator promotion。

## 5. 来源阶段与不可绕过的时间窗口

### 5.1 唯一允许来源阶段

```text
ALLOWED_SOURCE_PHASES = {
  ACQUIRE_TRACK,
  STABILIZE_PREGRASP,
}
```

- `ACQUIRE_TRACK` 对应信息收集与粗定位；
- `STABILIZE_PREGRASP` 是进入接触窗口前最后一个允许相机运动的来源阶段；
- 相机运动在独立 `ACTIVE_FRONT_REOBSERVE` phase 中执行；
- 返回后重新进入来源 phase，phase age 与 stability evidence 从零累计。

### 5.2 Episode 级 active-view window

只检查当前 phase 名称不足以防止 recovery/replan 绕过限制。新增单调 latch：

```text
active_front_reobserve_window_open = true at Episode reset
```

以下任一事件后永久变为 `false`，直到下一个 Episode：

- 第一次进入 `FINAL_APPROACH`；
- 检测到 object contact；
- 发送 gripper-close command；
- `grasp_candidate` 或 `grasp_verified`；
- 任何部署可得证据表明 object 可能已被推动。

即使 recovery/replan 后回到名为 `ACQUIRE_TRACK` 的 phase，也不得重新打开 latch。front camera 在关闭时
只能保持或按故障安全流程返回 HOME，不能发起新的信息收集动作。

## 6. Phase-specific 触发

所有触发都要求：

```text
source_phase in ALLOWED_SOURCE_PHASES
AND active_front_reobserve_window_open
AND active attempt budget remaining
AND arm/camera controller and safety prerequisites pass
AND failure is viewpoint-resolvable
```

### 6.1 先用完当前被动信息

主动移动不能只看 wrist 与 Memory，而忽略 HOME front 当前就能提供的信息。统一状态解析顺序为：

```text
qualified current wrist measurement
  -> qualified current front-home measurement
  -> frozen Goal/Object Memory fallback
  -> Active Front Reobserve request
```

若 HOME front 已有可靠 measurement，应直接更新/使用状态，不得为了“主动视觉”而额外移动。

### 6.2 `AcquireTrack`

```text
goal_missing =
    not current_goal_measurement_usable_from_any_home_camera
    and not goal_memory.valid

object_missing =
    not current_object_measurement_usable_from_any_home_camera
    and not object_memory.valid

trigger = goal_missing or object_missing
```

这里的 Memory 只提供冻结规则允许的 navigation/state-availability fallback。

### 6.3 `StabilizePregrasp`

Goal 可以由可靠 Memory 提供；object 在进入接触前仍必须重新获得当前直接证据：

```text
goal_missing =
    not current_goal_measurement_usable_from_any_home_camera
    and not goal_memory.valid

object_not_contact_ready =
    not current_object_measurement_fresh_and_stable

trigger = goal_missing or object_not_contact_ready
```

Active front measurement 可以形成 candidate 并更新 Object Memory，但不能因为 `object_memory.valid=true` 就
单独授权 `FinalApproach`。camera 返回 HOME 后仍要重新检查当前 direct object evidence；如果 wrist/HOME
front 仍不可见，则本次主动观察最多算“状态恢复”，不能算“接触授权恢复”，必须继续 SafeHold/Abort。

### 6.4 稳定触发与 reason code

单帧低 confidence 不触发物理运动。正式 config 冻结连续失败 Tick、hysteresis 和 cooldown。原因至少区分：

```text
OUT_OF_FOV
OBJECT_OCCLUSION
GOAL_OCCLUSION
LOW_VISUAL_CONFIDENCE
HIGH_LOCALIZATION_UNCERTAINTY
HIGH_GEOMETRIC_SENSITIVITY
INVALID_SENSOR_OR_POSE
UNSAFE_ARM_STATE
UNSAFE_CAMERA_STATE
UNKNOWN
```

只有前六类可请求 Active Front Reobserve。传感器/pose 无效、controller unsafe 或 unknown 必须
Passive Reobserve/SafeHold，不得用移动掩盖数据链或控制链故障。

## 7. 多执行器控制语义

### 7.1 Phase 与分域 owner

现有 `REOBSERVE` 保持 passive。E018-P1 新增：

```text
PhaseId.ACTIVE_FRONT_REOBSERVE
ExternalCameraControllerOwner.ACTIVE_REOBSERVE
```

不应把 front camera owner 塞进现有 arm `ControllerOwner` 后假装它拥有 TCP。主动观察期间按执行器域记录：

```text
arm_owner = SAFE_HOLD
gripper_owner = SAFE_HOLD_OPEN
external_camera_owner = ACTIVE_REOBSERVE
```

进入 phase 时必须原子完成：

- 失效并清空剩余 Action Chunk；
- 清空 temporal ensemble/RTC proposal history；
- 失效旧 commanded-target reference；
- 保存 versioned `ActiveFrontReobserveContext`；
- arm 锁定当前实际 q/TCP，gripper 保持 open；
- 相机 owner 获取唯一 camera-motion lease；
- arm、contact、camera-path 与 tracking monitor 每个 control tick 持续运行。

任何 Action Chunk、Precision TCP residual 或 recovery controller 都不得在该 phase 输出 arm/TCP motion。

### 7.2 持久化上下文

`ActiveFrontReobserveContext` 至少包含：

```text
episode_id
request_id
source_phase
resume_phase
target_entities
trigger_reasons
trigger_tick/timestamp
arm_anchor_actual_q
arm_anchor_tcp_pose
front_home_pose_id
viewpoint_primitive_id
viewpoint_schedule_index
attempt_index
camera_command_sequence_id
provider/checkpoint/config identity
status
```

context、每步 camera command、actual-pose sample、receipt、Memory commit 和恢复结果必须进入可回放 ledger，
并在进程异常或 identity 不完整时 fail closed。

## 8. Viewpoint primitive 与选择策略

### 8.1 Primitive 契约

每个预注册 primitive 至少包含：

```text
primitive_id
home_pose_id
translation_anchor = {LEFT|RIGHT} x {LOW|HIGH}
target_position_base/world_m[3]
target_orientation_cv_quaternion[4]
pan_rad / tilt_rad / fixed_roll_rad
path_id
workspace_and_collision_envelope_id
max_velocity / max_acceleration
move_timeout / settle_timeout / return_timeout
expected_duration_s
calibration_and_actuator_identity
```

姿态在 development/validation 阶段针对固定 task-workspace ROI 预注册，不以正式 Episode 的 GT object/goal
pose 为输入。

### 8.2 首版不做在线 NBV

首版允许在四个备用锚点上做离线 feasibility，但正式 config 必须冻结一个有限、确定性的
`viewpoint_schedule`：

- runtime 不计算连续最优位姿；
- 不读取 GT visibility 或 GT error 给候选打分；
- 相同 trigger evidence、attempt index 和 config 必须得到同一个 primitive；
- MVP 建议每个允许来源 phase entry 最多一次尝试；
- 若正式允许第二次尝试，其顺序、总预算和失败语义必须在 test 前冻结。

development 可以保留较大的离线候选池，但静态几何合格不等于可以进入正式运行。当前 G0B 将 25 个开发
位姿收敛为 10 个低位静态合格候选，并给出四个优先动态验证候选；二者都不是正式冻结库。所有可能进入
正式运行的位姿仍须分别通过有时间的运动门禁和 front provider 资格验证。

最简单的正式版本是在 fresh validation 前从通过全部资格门禁的候选中选定一个 workspace-wide
`PRIMARY_ALTERNATE`，所有 Treatment trigger 都使用它。多视角自适应选择留给后续 P2，这能先回答“产生
真实观察基线是否有净收益”，同时避免把候选数量扩大误当作实际信息增益。

## 9. 动态 external-camera 观测契约

### 9.1 必须升级固定外参假设

当前 `CameraCalibration` 把 `world_from_external` 定义为每个 Episode 固定值。front camera 运动后不能继续
复用这个语义。P1 应显式升级 Observation schema/version，而不是静默改变 V2：

```text
episode-level:
  intrinsic_external[3,3]
  distortion_model/coefficients
  external_camera_actuator_identity
  viewpoint_library_identity
  static_mount/calibration_version

per-frame:
  external_rgb_timestamp_s
  external_pose_timestamp_s
  base_from_external_camera_cv[4,4]
  commanded_external_camera_pose[4,4]
  actual_external_camera_pose[4,4]
  external_pose_valid
  external_tracking_error
  external_camera_motion_state
  viewpoint_primitive_id
  settled
```

下游几何只使用与 RGB 时间对齐、有效、实际测得的 `base_from_external_camera_cv`，不得用 commanded pose
替代 actual pose。OpenGL/OpenCV frame convention、单位和矩阵方向必须进入 schema identity。

### 9.2 时间与运动帧

- camera 正在移动、tracking 超限或 pose/RGB 超出同步预算的帧不得写 Object/Goal Memory；
- `SETTLE` 必须同时满足位置、旋转、速度、tracking 与连续稳定 Tick 条件；
- `COLLECT` 只接受 settled window 内固定数量的连续帧；
- wrist、front、arm proprio 和 control tick 各自保留 timestamp/validity，不能伪造为同一时间；
- Episode reset 必须同时重置 camera pose ledger、attempt budget 和 HOME verification。

## 10. 主动观察状态机

```text
REQUESTED
  -> ACQUIRE_CAMERA_LEASE_AND_HOLD_ARM
  -> SELECT_FROZEN_PRIMITIVE
  -> MOVE_TO_VIEW
  -> SETTLE_AT_VIEW
  -> COLLECT
  -> VALIDATE_CANDIDATE
  -> RETURN_HOME
  -> VERIFY_HOME_AND_ARM_HOLD
  -> RECHECK_SOURCE_INVARIANTS
  -> RESUME | FAIL
```

- `MOVE_TO_VIEW` / `RETURN_HOME` 中的 front frame 不得授权 Memory write；
- `SETTLE_AT_VIEW` 要求 camera actual pose、速度、tracking 与 timestamp 稳定；
- `COLLECT` 使用固定数量的稳定 front frame，wrist frame只作同步审计或按冻结 resolver 使用；
- `VALIDATE_CANDIDATE` 在 base frame 检查 measurement、covariance、multi-frame consistency 和冻结 write rule；
- `VERIFY_HOME_AND_ARM_HOLD` 同时检查 camera HOME error 与 q/TCP 漂移；
- `RESUME` 永远生成新的 Action Chunk，不继续主动观察前的 chunk。

首版要求返回 HOME，是为了让正常 manipulation policy 继续看到其冻结训练/验证分布内的 front 视角。未来若
策略显式接收 camera pose 并在多视角上重新验证，可以另立实验研究“相机停在备用视角继续操作”。

## 11. Candidate、Memory 与闭环恢复

### 11.1 Candidate measurement

主动观察后的第一帧不得直接写入 live Memory。稳定窗口先产生独立 candidate：

- object/goal 每帧 measurement identity 完整；
- source 明确为 `external_camera` 和对应 `primitive_id`；
- 使用同 Tick actual dynamic extrinsic 转换到 base frame；
- base-frame estimates 在冻结 covariance/innovation/multi-frame consistency 范围内；
- 对应 E016/E018-P0 write gate 通过；
- 无 motion-frame、invalid geometry、time skew 或 provider mismatch；
- 只提交 request 指定且正式 config 允许的实体。

Object/Goal Memory update 必须是原子、版本化结果。若其中一个实体失败，不得把未请求或未通过的实体偷偷
标为恢复；部分成功必须按 request target 独立记录。

### 11.2 两级成功语义

必须区分“信息恢复”和“闭环恢复”：

```text
state_recovered =
    requested phase-required state available
    and requested candidate/Memory result valid
    and no safety/anomaly event

closed_loop_resumed =
    state_recovered
    and camera HOME verified
    and arm hold verified
    and source phase entry/invariant predicates pass again
    and current evidence needed for contact authorization exists
```

这样可以避免在 alternate view 看见 object 后，仅凭返回 HOME 前的历史测量绕过
`StabilizePregrasp -> FinalApproach` 的 current-evidence 门禁。

成功恢复后：

1. camera owner 释放 lease；
2. arm owner 仍保持 SafeHold，直到 source phase 原子恢复完成；
3. phase age/stability counter 清零；
4. 使用当前真实 q/TCP、HOME camera pose 与更新后的 Memory 构造新 observation/state；
5. 重新生成 Action Chunk 和 sampling identity；
6. ledger 分别记录 `STATE_RECOVERED` 与 `CLOSED_LOOP_RESUMED`。

### 11.3 失败条件

以下任一情况进入 `FAIL -> SafeHold/Abort`：

- camera lease、move、settle、capture、validate、return 或 HOME verify 超时；
- camera path/collision/workspace 或 actuator tracking 门禁失败；
- arm q/TCP 超出 hold tolerance；
- unexpected contact、gripper change或 active window 关闭；
- 稳定窗口仍不能恢复 requested state；
- candidate measurement 冲突、unsafe 或 provider identity 不一致；
- HOME return 未验证；
- attempt budget 耗尽；
- timestamp、dynamic extrinsic 或 ledger 不完整。

失败后不得继续旧任务命令，也不得临时选择未预注册的新视角。

## 12. Safety prerequisites 与预算

执行 Active Front Reobserve 前至少要求：

- gripper open；
- 无 finger/object contact；
- 未进入或跨越 `FINAL_APPROACH`；
- actual arm/controller state 有效且 tracking 正常；
- camera HOME actual pose 有效；
- external RGB 与 actual camera pose 时间链有效；
- `HOME -> alternate -> HOME` 路径通过相机 workspace、速度、加速度与 collision smoke；
- 相机机构不会进入机械臂当前或允许工作包络；
- episode、phase-entry 与 wall-clock budget 未耗尽。

正式预注册需冻结：

- 每 phase entry/每 Episode 最大尝试数；
- 每个锚点的 position、yaw/pitch 与 pose tolerance；
- camera velocity、acceleration、tracking-error threshold；
- move、settle、collect、return timeout；
- stable observation frame 数；
- arm hold 与 HOME return tolerance；
- unexpected-contact、camera collision 和 emergency-stop 语义。

## 13. 实验设计与归因

### 13.1 主比较

```text
Control B: frozen Dual Memory + qualified front provider
           + matched-time/frame Passive Reobserve at HOME

Treatment C: same Dual Memory + same qualified front provider
             + frozen Front Active Reobserve primitive
```

Control 必须在原地保持 arm 与 camera，等待与 Treatment 匹配的完整 wall-clock budget，读取相同数量的
settled frame，并执行相同 estimator/Memory gate。两臂保持：

- 相同 scene、environment seed、初始状态和触发 Tick；
- 相同 checkpoint、provider、Memory config、threshold 和 observation count；
- 相同后续 Flow sampling seed；
- 相同 timeout、failure 和 task-success 定义；
- 交错或随机化执行顺序并记录顺序。

这样 `C-B` 才表示 front viewpoint change 的净作用，而不是额外时间、额外帧或新增 provider 的作用。

### 13.2 与 E018-P0 的关系

可同时报告：

```text
A: Goal Memory only + frozen wrist stream
P0: Dual Memory + same frozen wrist stream
B: Dual Memory + qualified front provider + Passive HOME reobserve
C: Dual Memory + same front provider + Active alternate-view reobserve
```

- `P0-A`：Object Memory 的净作用；
- `C-B`：主动 front 视角的主要净作用；
- `B-P0` 含 front provider/观测链差异，只作诊断，不能解释为 Memory 或 motion 单一效果；
- 不应直接把 `C-A` 全部归因于主动视觉。

## 14. 渐进实现与实验 Gate

1. **G0A Simulator/API feasibility**：证明 `base_camera` 能通过可审计 actuator/path 动态改变实际 pose，
   不是零时延 teleport；若现有 ManiSkill sensor 不支持，先实现最小 movable-camera mount。已在四个 nominal
   平移锚点上通过 development gate。
2. **G0B Static viewpoint screen**：在不推进环境时间和机器人状态的条件下筛选
   `translation anchor × discrete orientation`，审计构图、目标像素、边界和 robot 遮挡。已通过 development
   gate，得到 10 个低位静态合格候选。
3. **G0C Rotated-pose motion feasibility**：对全部静态合格候选执行有时间的
   `HOME -> full pose -> HOME`，验证新旋转路径、settle、tracking、arm/TCP invariance 与无接触；不得把 G0B
   的静态 teleport 当作运动证据。已在 10 个低位候选上通过 development gate。
4. **G1 Observation schema**：动态外参、RGB/pose 时间、motion-frame invalidation 和 replay round-trip。
5. **G2 Provider qualification**：冻结模型逐 viewpoint 评估；失败则停止 P1，不在同一实验内训练修补。
6. **G3 Controller contract**：分域 owner、phase allowlist、latch、reset、hold、budget 和非法转换。
7. **G4 Integrated motion smoke**：在 G3 owner/state-machine 下复验正式候选的路径、settle、tracking、
   arm invariance、collision 和 fail-closed 行为。
8. **G5 Viewpoint feasibility**：用 train/development/fresh validation 冻结正式 primitive/schedule，不看 test。
9. **G6 Shadow recommendation**：只记录 trigger 与 primitive，不发送 camera command。
10. **G7 Simulation active motion**：只在允许窗口移动 camera，arm 与 manipulation progression 冻结。
11. **G8 Paired closed-loop**：冻结全部规则后，以 fresh seeds 比较 Passive 与 Active。
12. **G9 Fresh test-once**：validation 完成、规则冻结并创建 claim 后只执行一次正式 test。

前一 Gate 未过不得跳到下一 Gate。

## 15. 指标与 Promotion Gate

### 15.1 Trigger 与恢复

- allowed-source trigger count；
- forbidden-source rejection count；
- trigger reason/type 分布；
- object-only、goal-only、both-missing count；
- trigger-conditional `state_recovered` rate；
- trigger-conditional `closed_loop_resumed` rate；
- Goal/Object Memory initialization/recovery rate；
- attempts、latency、camera path length 和 frames per recovery；
- Passive HOME 自然恢复率；
- alternate-view 可见但 HOME-return 后 contact evidence 不足的 count。

### 15.2 感知与信息收益

- 每个 viewpoint 的 object/goal observability；
- Active/Passive 前后 world XYZ error p50、p90、max；
- covariance reduction 与 accepted information gain；
- candidate consistency、innovation 和 write acceptance；
- unsafe Memory update；
- false recovery：状态被声明 valid 但 oracle safety 评估失败；
- viewpoint 相对 provider train/validation camera-pose support 的 OOD 审计。

### 15.3 相机与机械臂安全

- camera position/orientation tracking error；
- settle time、return-home error 和 timeout；
- camera path/workspace/collision violation；
- Active 期间 arm q/TCP 最大漂移；
- unexpected contact 或 gripper state change；
- camera/arm owner overlap 或 command-domain mismatch。

### 15.4 零容忍项

- 禁止 phase 的 active camera motion；
- latch 关闭后的 active camera motion；
- camera motion期间 arm/TCP manipulation command；
- stale Action Chunk/proposal/command reference；
- motion/unsettled frame 写入 Memory；
- commanded pose 冒充 actual pose；
- dynamic extrinsic、timestamp 或 frame-convention mismatch；
- camera/arm collision、workspace violation；
- HOME 未验证却恢复固定视角 policy；
- memory-only `FinalApproach`/close authorization；
- recovery loop 或 attempt-budget exceed；
- Episode reset leakage；
- hidden-GT runtime dependency；
- Active 结果反向修改 Parent perception/Memory rules。

以上任一非零均阻断 promotion，无论 recovery rate 提升多少。

### 15.5 主要 success gate

正式数值在 validation 后按预注册协议冻结，但门禁顺序固定：

1. G0A-G7 的 contract、provider 与运动资格全部通过；
2. 全部 system/safety/identity 零容忍项为零；
3. Active 不得恶化 accepted-state safe/catastrophic error；
4. `C` 的 trigger-conditional reliable-state recovery 相对 matched Passive `B` 达到预注册改善；
5. camera HOME、arm hold、resume 与 newly-generated Action parity 全部通过；
6. 完整任务成功率只作次级指标，不得掩盖主动观察本身失败。

若未通过，冻结负结果并保持 Active Front Reobserve 默认关闭。

## 16. 交付物

- dynamic-external-camera Observation schema/version 与 migration/reader；
- `ACTIVE_FRONT_REOBSERVE` phase、分域 camera owner 和 versioned context/receipt；
- active-window latch 与 phase-source allowlist；
- HOME 加二维离散平移/旋转 `ViewpointPrimitive` 库；
- front provider qualification report 与 frozen identity；
- motion-frame invalidation、settle/collect/return/verify 状态机；
- arm SafeHold 与 Action Chunk/proposal/command-reference 原子 reset；
- Passive/Active paired runner 与相同时间/帧预算；
- ledger replay、异常恢复和负例测试；
- formal JSON config、source/data/pose/provider/Memory identity verifier；
- fresh validation、test-once claim、private result与 GitHub 脱敏 summary；
- 失败边界、默认关闭状态和后续 pose-conditioned policy/P2 建议。

## 17. 待正式预注册冻结的参数

- HOME 与 `LEFT/RIGHT x LOW/HIGH` 的准确坐标；
- 每个锚点的 yaw/pitch、固定 roll 和 pose identity；
- 正式 `PRIMARY_ALTERNATE` 或确定性 viewpoint schedule；
- camera path、速度、加速度和 workspace/collision envelope；
- trigger hysteresis、cooldown 与 viewpoint-resolvable reason mapping；
- stable-frame、candidate consistency、settle/HOME/arm-hold tolerance；
- provider 的逐视角 qualification threshold；
- Active/Passive matched time、frame 和 retry budget；
- fresh seed、Episode 数、trigger strata 与执行顺序；
- recovery improvement、safe/catastrophic error 和零容忍 Gate 的精确数值。

不得查看 fresh test 后修改这些值。任何改变研究问题、允许 phase、active window、provider 或 viewpoint
schedule 的修改都要求新的 config identity，必要时使用新的实验 ID。

## 18. 允许的结论

通过后最多可以声明：

> 在 `AcquireTrack` 与 pregrasp 的无接触窗口内，当冻结的当前 wrist/front-home measurement 与 Dual
> Memory 无法提供 phase-required state 时，一个与机械臂解耦、具有真实平移基线和预注册旋转的 front
> 主动观察原语，相对匹配时间的 HOME 原地重新采样提高了可靠状态恢复率，并能在相机归位、机械臂保持、
> 旧命令清空和 source invariant 复核后重新接回原闭环。

不得声明：

- 已实现通用或学习式 Next-Best-View；
- 四个视角可根据任意场景在线最优选择；
- 接触后、抓持中、transport 或 release 阶段可以移动相机；
- Object Memory 可以单独授权 FinalApproach、close、lift 或 release；
- wrist-only checkpoint 未经资格验证即可用于任意 front pose；
- 仿真中的虚拟相机运动已经证明真实硬件安全；
- 已解决 E016/E017 的所有 unsafe write 或全部 manipulation failure。
