# E015 — Explicit geometric goal state memory

E015-A 与 E015-B 使用 frozen E013 checkpoint 和全新 seeds 完成；没有训练、没有修改
checkpoint、没有 Action 输出，也没有 actuator。fresh validation 只用于冻结 write threshold 与
memory age，fresh test 只评估一次。

术语边界：数据生成后、calibration 前会对所有 split 做 schema、文件 identity 和 oracle
round-trip 完整性 audit；该 audit 不产生 test prediction，也不参与 threshold/age 选择。test-once
claim 的精确范围是 **U-Net model forward 与 shadow replay**，这部分只执行一次。

## V1.0 瓶颈判断

**显式 base-frame memory 的状态保持机制成立，但 E015 工程 gate 未通过。** memory 能在当前
RGB 不可观察时安全保留历史 goal，未产生 catastrophic state；真正限制 V1.0 的是可靠
measurement 的准入和 Episode 初始化，而不是 memory 的 hold 逻辑，也不是主体分布的毫米级定位。

- 100 个 test Episode 中只有 `35` 个得到过可靠 write，
  `65` 个始终没有初始化 memory。
- 当前帧 measurement coverage 为 `1.6983%`，memory coverage
  提升到 `13.0019%`。
- GT 不可观察时，当前 measurement 有效 `2` 帧，
  memory 有效 `663` 帧；在已初始化 Episode 内的
  遮挡覆盖为 `19.7851%`。
- memory world-XY p90/max 为 `1.078` /
  `1.538 mm`，catastrophic=`0`，
  Episode reset leakage=`0`。

## E015-A：observability contract

- test frames：`20197`；goal exists/projected：
  `20197` / `20197`；真正 observable：
  `10281`。
- legacy visible 为 `14970`，其中 `4689`
  帧（`31.32%`）满足“mask 还有像素”，但 projected center 已无直接视觉证据。
- 不可观察主要由 out-of-frame `6644`、
  object occlusion `2730` 和其他遮挡/背景
  `542` 构成。

这证实 `exists / projected / observable` 不能继续混用；旧 keypoint-visible 标签会把一部分遮挡帧
当作普通 goal-center supervision。

## E015-B：write gate 与 memory

validation 上冻结的 threshold 为 `0.618484616`，接受
`42` 帧、unsafe=`0`，安全
measurement coverage 仅 `2.0792%`。test 上接受 `343`
帧，其中 unsafe=`2`、catastrophic=`0`。

两次 unsafe write 都是 strict center-ray contract 下的 `other_occlusion_or_background`：定位误差仍小，
但当前 RGB 没有足够的 goal-center 视觉证据。它们的 score 只比冻结阈值高
`0.0004891–0.0006792`，
world-XY 最大误差 `1.423 mm`。因此失败点不是 20 cm hallucination，
而是单帧 scalar gate 在阈值边界无法保证跨 split 的零 false acceptance。

## 聚合修正

原始 generator 的 `stale_or_uninitialized_occluded_count` 错误地排除了 `memory_age_s=null` 的
未初始化帧，因此从 `0` 确定性修正为
`9253`。该修正只重算既有 JSONL，未执行模型 forward、
未改规则、未重新读取 test 模型输出；canonical private receipt 与 test-once claim 均保留。

## 下一步边界

1. 先修训练/评估 label contract：不可观察 goal 不再作为普通 keypoint supervision，并单独训练、校准
   observability；必须使用新的 validation/test seeds 验证。
2. 提高安全初始化覆盖：优先加入部署可得的多帧一致性与明确初始化阶段；若 wrist 在任务早期仍看不到
   goal，再通过统一 base-frame 接口接入 external-camera measurement。
3. write authorization 继续 fail closed：使用时间一致性、innovation、workspace 和 mask-support margin 的
   联合门禁，而不是只依赖一个单帧 scalar score；新规则仍需预注册并在 fresh test 上验证。
4. 在 unsafe write=0、catastrophic=0、初始化覆盖达到新门槛之前，不接 controller/actuator；本次结果
   `safe_for_actuator_promotion=false`。

公开目录只含脱敏聚合与 SHA-256。逐帧身份、位置、mask/RGB 和私有路径未发布。
