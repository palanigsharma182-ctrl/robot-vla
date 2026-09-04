# E016-P0 — Corrected observability pretraining contract

E016-P0 已在云端 RTX 4090 上从 clean commit `d1481ba` 完成。它修复的是 E013
训练监督合同，而不是给 Memory、Qwen 或 VLA 增加 learned memory：当前 RGB 只产生
measurement 与 observability，显式 base-frame Memory 仍保持确定性。

## 结论

**E016-P0 通过，可以进入 E016-P1 正式训练配置冻结阶段；当前产物不能部署，也不能接
actuator。** P0 没有保存 checkpoint、没有读取 test label arrays，也没有使用 E015 test
选择标签、loss、训练步数或阈值。

## Corrected label audit

P0 只从 E013 的 24 条 train 和 6 条 validation trajectory 派生 delta-sidecar，共
`5943` 帧：

- goal exists：`5943`；
- goal projection valid：`4223`；
- goal observable：`3210`；
- goal unobservable：`2733`；
- legacy goal visible：`4635`；
- legacy visible 但 corrected unobservable：`1425`。

这 `1425` 帧是旧合同会继续监督 goal center、而新合同会把 localization target 清零并
mask 掉的样本。不可观察原因包括 projection invalid `1720`、object occlusion `750`、
other occlusion/background `263`。

## Loss contract

- `goal_observable=false` 时 heatmap、coordinate、keypoint uncertainty localization
  loss 均为 `0`；
- all-negative batch 的 loss 有限，heatmap logits、subpixel offsets、keypoint log-variance
  的 localization gradient 最大绝对值均为 `0`；
- visibility 使用独立的 `keypoint_observable` target；projection target 保持独立；
- mask 继续学习真实可见 instance pixels，不再用 `mask.any()` 代理 center observability；
- E016 对稀疏 mask 显式启用 BCE + soft Dice，E013 默认 loss 行为保持不变。

## 128-frame stratified overfit

固定 train-only 子集由 64 个 observable、24 个 object occlusion、24 个 other
occlusion/background 和 16 个 projection-invalid/out-of-frame 样本组成。600 optimizer
steps 后全部预注册门禁通过：

| 指标 | 初始 | 最终 | 门禁 |
|---|---:|---:|---:|
| observable goal normalized-UV MAE | 0.156435 | 0.001061 | ≤ 0.01 |
| goal mask IoU | 0.011782 | 0.878079 | ≥ 0.75 |
| goal visibility precision | 0.000000 | 1.000000 | ≥ 0.95 |
| goal visibility recall | 0.000000 | 1.000000 | ≥ 0.95 |
| unobservable false-positive rate | 0.000000 | 0.000000 | ≤ 0.05 |
| projection accuracy | 0.875000 | 1.000000 | 诊断项 |

Motion Head 训练前后 SHA-256 一致。P0 权重只在内存中用于训练链路验证，未持久化。

## 3-epoch full train/val preflight

Preflight 使用另一个随机种子重新初始化模型，不继承 overfit 权重。它完成 `450` 个
optimizer steps，验证集最终结果为：

- observable goal normalized-UV MAE：`0.004206`；
- goal mask IoU：`0.477174`；
- visibility precision / recall：`0.992806 / 0.906404`；
- unobservable false-positive rate：`0.007168`；
- projection accuracy：`0.984576`；
- Motion Head hash unchanged，checkpoint persisted=`false`。

这只是 3 epoch 的链路与学习趋势检查。尤其是 mask IoU 和 visibility recall 仍不能作为正式
模型结论；需要 E016-P1 完整训练、validation-only checkpoint/threshold selection，以及冻结后
一次性的新 held-out evaluation。

## 下一步边界

1. 冻结 E016-P1 的正式 train/val/test manifest、20-epoch 配置和 validation safety-first
   checkpoint selection；canonical 模型继续从随机初始化开始。
2. 正式训练后只在 validation 选择 observability/write threshold；不得使用已经消费的 E015
   test 调参，也不得把它重新声明为 unseen test。
3. 冻结所有规则后执行一次 fresh no-actuation held-out + memory replay，重新检查 unsafe write、
   initialized Episode rate、unobservable memory coverage、catastrophic state 与 reset leakage。
4. 在这些 gate 通过前，`safe_for_actuator_promotion=false`。

公开目录只包含脱敏聚合与 SHA-256；逐帧身份、corrected NPZ、原始 RGB/mask 和私有路径未发布。
