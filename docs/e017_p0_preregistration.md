# E017-P0 保守可观察性门控微调预注册

E017-P0 在不重新采集数据的条件下，验证 E016-P1 的 `unsafe write` 上游瓶颈能否通过一个最小、
可审计的分类头微调得到改善。E016-P1 canonical checkpoint、fresh test-once 与公开结论保持只读。

## 单一改动

- 从 E016-P1 selected epoch 12 的正式 checkpoint warm-start。
- 冻结 encoder、decoder、localization/mask、state encoder、Motion Head，以及 uncertainty head 的其余输出。
- 只允许最后线性层的 `goal_visibility` 与 `projection_validity` 两行更新。
- 训练损失只包含 goal visibility 和 projection validity BCE；goal 不可观察负样本权重固定为 `4.0`。
- AdamW weight decay 固定为 `0`，每步还原所有非目标行，防止零梯度参数被解耦 weight decay 漂移。

这个改动不让网络学习被遮挡 goal 的位置，也不改变显式 base-frame memory。RGB 仍然只提供当前
measurement，memory 仍然负责维护 world state。

## 数据与训练

- 只读取 E013 的 24 条 train 与 6 条 validation trajectory，以及 E016-P0 corrected observability
  sidecar；所有 identity 必须与 E016-P1 receipt 一致。
- 禁止打开 E013/E015/E016 的任何 test trajectory、test label、fresh test prediction 或 test-once 输出。
- seed `17017`，8 epochs，batch 32，BF16，AdamW，learning rate `1e-4`，cosine annealing。
- canonical hardware 为 NVIDIA GeForce RTX 5080。
- 输出目录必须不存在；只持久化 validation 选择的一个 weights-only checkpoint。

## Validation-only 选择

父 checkpoint 先在同一 validation 上复算为 epoch 0 baseline。训练后 checkpoint 必须同时满足：

- goal unobservable FPR `<= 0.01`；
- goal visibility precision `>= 0.99`；
- goal visibility recall `>= 0.90`；
- projection accuracy `>= 0.98`；
- observable-goal UV MAE 与 goal mask IoU 相对父 checkpoint 的绝对漂移均 `<= 1e-8`；
- safety 排序相对父 checkpoint 严格改善：FPR 更低，或 FPR 相同且 recall 更高。

合格 epoch 先最小化 FPR，再最大化 recall，最后选更早 epoch。若没有 epoch 严格改善，保留失败
receipt，不写新 checkpoint；不得为了产生 checkpoint 放宽门槛。

## 结论边界

E017-P0 是 train/validation-only 实验，不产生新的 held-out 或 actuator 证据。即使通过，仍保持
`safe_for_actuator_promotion=false`；后续必须使用新的预注册 held-out 数据验证 unsafe write 和 memory，
不能复用 E016-P1 已消费的 test-once。
