# E016-P1 正式训练与 fresh held-out 预注册

E016-P1 的问题是：在 E016-P0 已证明 corrected observability supervision 可学习之后，完整
20-epoch 训练能否产生一个同时满足定位质量和 fail-closed 可观察性约束的正式 Precision U-Net，
并在全新数据上安全驱动显式 base-frame goal memory。

本文件和 `configs/e016_p1_precision_observability_v1.json` 必须先于正式训练、fresh validation
校准和 fresh test 首次读取提交。E013 canonical 数据、checkpoint 和结论保持只读。

## 正式训练

- 数据只允许使用 E013 的 24 条 train 与 6 条 validation trajectory，以及 E016-P0 生成且由
  SHA-256 绑定的 corrected delta-sidecar；训练入口没有 test 参数。
- canonical checkpoint 使用 seed `16018` 从随机初始化开始，不继承 P0 debug/preflight 权重，
  也不 warm-start E013 checkpoint。
- 训练固定为 20 epochs、batch 32、AdamW、learning rate `3e-4`、weight decay `1e-4`、
  clip norm `1.0`、BF16 和 cosine annealing；Motion Head 保持 frozen-zero-shadow-only。
- 使用 P0 已验证的 observable-only goal localization、独立 visibility/projection target 和
  BCE + soft Dice mask loss。
- canonical hardware 是云端 NVIDIA GeForce RTX 4090。输出目录拒绝覆盖；只持久化最终选中的
  checkpoint，并执行 weights-only strict reload 与 parameter/provenance hash 校验。

## Safety-first checkpoint selection

每个 epoch 在原 E013 validation 上以固定 `0.5` visibility/projection threshold 评估。checkpoint
必须先同时满足：

- goal unobservable false-positive rate `<= 0.01`；
- goal visibility precision `>= 0.99`；
- goal visibility recall `>= 0.90`；
- projection accuracy `>= 0.98`；
- goal mask IoU `>= 0.50`。

只有通过全部门禁的 epoch 才进入候选集；随后选择 observable goal normalized-UV MAE 最低者，
数值同分时选更早 epoch。若没有 epoch 全部通过，则本实验保留失败 receipt，但不写正式 checkpoint。
这些门槛来自 P0 的 train/validation preflight，不得根据 fresh test 结果修改。

## Fresh held-out 与 memory replay

- candidate seed 冻结为 `134000..134999`，与 E013、E014、E015 已使用范围不重叠；采集目标是
  collector-contract-only train 1 条、calibration validation 20 条、test 100 条。
- collector 的 1 条 train 不进入 P1 训练。fresh validation 只选择 goal write threshold 和 memory
  max-unobserved-age；不重新选择 checkpoint、loss 或训练轮数。
- checkpoint、write-score、threshold policy、memory update/age rule 全部冻结后，必须在读取任何
  fresh test privileged label 或执行 test model forward 之前原子创建 test-once claim。已存在 claim
  时禁止通过更换输出目录重复评估。
- write threshold 按 validation 上“零 unsafe write 下最大 coverage”选择；memory age 按“零
  catastrophic state 下最大 occluded coverage”选择。goal state 始终保存为 robot-base frame。
- test 要求 unsafe write、memory catastrophic state 和 Episode reset leakage 都为零，并要求 memory
  提高不可观察帧 coverage。无论结果如何，本阶段均为 no-actuation，
  `safe_for_actuator_promotion=false`。

完整数值、身份 hash 和拒绝条件以冻结 JSON config 为准；本文件只提供其工程解释。
