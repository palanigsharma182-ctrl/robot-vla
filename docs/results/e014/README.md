# E014 — E013 Precision long-tail failure decomposition

本实验只诊断冻结的 E013 checkpoint、val/test split、temperature、confidence threshold、soft-argmax 与 GT z-plane；没有重新训练或按 test 调参。原始 RGB、heatmap、逐样本身份和完整几何证据只保存在私有本地结果中。

## 结论

- 重新计算得到 world-XY p50 `0.370 mm`、p90 `1.517 mm`、max `208.538 mm`，与 E013 canonical 指标在预注册容差内一致。
- 最大异常是 `goal_center`，匿名指纹 `25baeec97247a4cff83c`，分类为 `semantic_swap_failure`；像素误差 `84.959 px`，world-XY 误差 `208.538 mm`。
- 最大异常的 heatmap 判定为 `no-frozen-multimodal-signature`，confidence accepted=`True`，visibility probability=`0.055823`。
- 当前主要瓶颈是：**毫米级主体分布之外的离散 catastrophic perception failure**。
- confidence gate 对超过 20 mm 的错误不能可靠 fail-closed；被接受的 >20 / >50 / >100 mm 数量分别为 `15` / `9` / `9`。
- E013 的旧 `confidence_precision` 实际是 accepted validity precision，不是定位准确率；E014 单独报告 5/10/20 mm accepted accuracy。

## Top-20 五类 failure family（实验 Q2 口径）

- `correspondence_failure`: 4
- `multimodal_softargmax_failure`: 0
- `visibility_or_ood_failure`: 0
- `geometry_conditioning_failure`: 0
- `unclear_or_mixed`: 16

## Top-20 细粒度固定 taxonomy

- `label_or_channel_contract_failure`: 14
- `temporal_alignment_failure`: 0
- `semantic_swap_failure`: 2
- `geometry_conditioning_failure`: 0
- `multimodal_softargmax_failure`: 0
- `visibility_or_ood_failure`: 0
- `generic_correspondence_failure`: 2
- `unclear_or_mixed`: 2

## 最大异常的可证伪检查

- soft-argmax 到 top-1 距离：`24.206 px`
- top-1 自身误差：`102.294 px`
- top-2 / top-1 probability ratio：`0.179290`
- `|n·unit_ray|`：`0.789901`
- 局部几何 Jacobian 最大奇异值：`2.468 mm/px`
- object/goal 物理间距：`182.238 mm`

这些量分别用于区分语义换位、相邻帧错位、多峰 soft-argmax、不可见/OOD 未拒绝和 ray-plane 几何放大。分类规则及所有数值阈值在读取 test 前由 frozen validation 封存。

## 安全含义

如果未来把该 perception 结果直接接入 actuator，真正可能被执行的是 `confidence_accepted=true` 的 catastrophic rows；其数量和最大误差见脱敏汇总。后续任何修复必须使用新的 validation/test seeds；本次 E013 test split 已标记为 `consumed-for-diagnostic-postmortem`，不能再声称是最终未见 test。
