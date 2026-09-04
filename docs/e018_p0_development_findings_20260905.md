# E018-P0 Recorded Validation 开发结果（2026-09-05）

> 状态：`development-only / not formal held-out / not promotion-eligible`
>
> 本轮只读取 E016 fresh `val`，未读取或 forward `test`；全程 no-actuation、
> no-camera-motion。这里的参数搜索结果只能用于发现问题和设计下一轮数据，不能冻结为 E018 正式规则。

## 1. 运行身份

- GPU：NVIDIA RTX 6000 Ada Generation（48 GB）
- Parent checkpoint：
  `c0be8769f75e4b991daa3a71878df72d2e2aaad70b5139abb4190512d6110552`
- 冻结 Goal Memory rules：
  `29233975680808296201bafb500a8af2ab4abb2520046f6ecba735d6dc849010`
- E018 development config：
  `67247e1eea057fb48a07d83353ae7b1145ae14f57c75d7b923df0acffe95ab40`
- Validation data identity：
  `f2c5d47db7f3f7d87500fd7572c695299b4cce812a70f859981d9b50fcdc6af5`
- 数据规模：20 episodes、4,154 frames，其中 offline-aligned pregrasp/reach 为 1,135 frames
- GPU 私有工件：`/root/robot-vla-runs/e018-dual-memory/development-v1`
- Drive 直传备份：
  `gdrive:VLA/experiments/e018/e018-p0-recorded-validation-development-v1-20260905`
- Drive `rclone check`：7 files matching，0 differences

## 2. 结果摘要

| 指标 | 结果 | 解释 |
|---|---:|---|
| 当前单帧 write gate | 151 / 1,135（13.30%） | 零 unsafe 阈值非常保守 |
| 当前稳定 candidate | 130 / 1,135（11.45%） | 还要通过连续两帧 candidate gate |
| Object Memory navigation 可用 | 504 / 1,135（44.41%） | 相比当前 candidate 增加 374 帧、+32.95 个百分点 |
| Memory 覆盖 current-unavailable | 374 / 1,005（37.21%） | 主要桥接模型低分/门禁拒绝帧 |
| GT 真正不可观察 | 41 / 1,135 | 全部属于冷启动前段 |
| Memory 覆盖 GT-unobservable | 0 / 41（0%） | 没有发生“先可靠观察，再真实遮挡” |
| Memory XYZ error p50 / p90 / max | 0.547 / 1.072 / 2.263 mm | 仅统计 valid Memory |
| Unsafe accepted update | 0 | development safety gate 通过 |
| Catastrophic Memory（>20 mm） | 0 | development safety gate 通过 |
| Episode reset leakage | 0 | development safety gate 通过 |
| pregrasp 后仍 valid | 0 | phase/contact 单调失效生效 |
| Memory-only contact authorization | 0 | navigation fallback 未越权 |

探索性网格按当前 objective 选中了：连续 2 帧、candidate 最大 2 mm spread、最大 age 2 s。
这三个数字**不得冻结**，原因见下文。

## 3. 核心发现

### 3.1 Memory 的安全机制成立，但核心物理遮挡效应尚未被数据验证

Object Memory 将 navigation availability 从 11.45% 提高到 44.41%，说明它能保持已经通过严格门禁的
object position，并跨越后续低置信度或 write-gate rejection。所有安全零容忍项均为零，证明当前
position-only、pregrasp-only、contact 后单调失效的机制在这组 replay 上按设计工作。

但是，41 个 GT-unobservable 帧全部发生在两个 episode 的开头：

- `pick-place-seed-134003`：timestep 0–22；
- `pick-place-seed-134091`：timestep 0–17。

它们在不可观察之前没有任何可靠 object candidate，因此 Memory 仍是 `UNINITIALIZED`。这不是实现 bug，
而是记忆的不可消除边界：**从未可靠看见过的状态不能由 Memory 安全恢复**。这两个 episode 后续也没有
越过冻结式 write threshold，故整段 episode 都没有初始化 Object Memory。

这类场景正是 E018-P1 主动 front reobserve 的有效触发目标：current wrist 不可靠且 Object Memory 不可用。

### 3.2 当前 write threshold 对少量 false positive 极敏感

- 新 object observability 下，visibility precision 为 96.39%，recall 为 100%；
- 41 个 GT-unobservable 帧被 visibility head 全部预测为 visible；
- unobservable write score 最大值为 `0.619327`；
- 零 unsafe calibration 选出的 threshold 为 `0.619403`，间隔只有约 `7.6e-5`；
- 最终 safe coverage 只有 13.80%。

因此当前 threshold 虽然在这组 validation 上实现零 unsafe，但 margin 太窄，不能直接主张跨 seed 稳健。
candidate window 不能弥补“连续多帧都自信地误判可观察”的情况；需要新的困难场景验证或更有区分度的
object observability evidence。

### 3.3 最大安全 age 在当前数据上不可识别

pregrasp/reach 期间 object GT 基本完全静止，最大帧间位移小于约 1 微米；网格内所有 age 都没有
catastrophic error，coverage 随 age 单调增加，于是选择器机械地选到上界 2 s。三个 candidate spread
候选也产生相同指标。

这只能说明“在静止专家轨迹上持有更久覆盖更多”，不能证明 2 s 是真实部署中的安全 age。正式冻结前必须
加入 object 被轻推、意外接触、pose/tracking 短暂异常等可部署失效证据；不能为了 coverage 把本轮上界
当作已识别参数。

### 3.4 Episode 级覆盖仍不足

20 个 validation episodes 中只有 12 个至少形成过一次可靠 Object Memory，8 个从未初始化。除了上述两个
冷启动不可观察 episode，还包括整体 write score 低或 contact safety latch 触发的 episode。这进一步说明
“Memory 升级”和“主动获取新信息”不是互相替代的方向：Memory 只能复用曾经可靠获得的信息，主动观察负责
冷启动或长期无法形成可靠 candidate 的情况。

## 4. 对实验计划的修正

在正式 E018-P0 预注册前，增加一个独立 development challenge set，至少覆盖：

1. **先看清、后遮挡、物体静止**：直接测量已初始化后，遮挡 0.1/0.25/0.5/1/2 s，用于识别 Memory
   的真实 bridge 能力；
2. **冷启动不可观察**：验证 Memory 保持 `UNINITIALIZED`，并生成 P1 active-reobserve trigger；
3. **低分但 GT 可观察**：测量 epistemic rejection 与 Memory fallback，不与物理遮挡混为一类；
4. **写入后物体被推动或发生接触**：使用可部署 finger-force、measurement innovation、phase latch 验证
   Memory 及时失效；
5. **时间、pose、controller 与 source identity 故障**：逐项验证 fail closed。

指标必须拆成两类：

- `memory_valid_while_gt_physically_unobservable`：回答真正遮挡；
- `memory_valid_while_current_measurement_unusable`：回答模型不确定、几何或 write gate 拒绝。

下一轮只有在困难场景中观察到非零物理遮挡 bridge、零 unsafe、并且 age 不再由搜索上界决定后，才进入
fresh validation 参数冻结。当前 E016 validation 继续只作为 development evidence，不创建 E018 test-once
claim。

## 5. 当前结论

本轮支持以下结论：

- Object Memory 的契约、candidate verifier、phase-specific resolver 和失效规则值得保留；
- 它能显著桥接严格 write gate 下的短时 measurement-unusable 帧；
- 它不能解决冷启动从未看清的 episode；
- 当前数据不能验证真实遮挡 bridge，也不能识别最大安全 age；
- 因此 E018-P0 目前是“安全机制开发通过、核心效应证据不足”，不得 promotion，也不得启动正式 test-once。
