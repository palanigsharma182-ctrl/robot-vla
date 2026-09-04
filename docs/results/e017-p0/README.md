# E017-P0 — Conservative observability row fine-tuning

E017-P0 在云端 NVIDIA GeForce RTX 5080 上完成 8 epochs / 1,200 optimizer steps 的
train/validation-only 微调。它从 E016-P1 selected epoch 12 warm-start，只允许 uncertainty final
linear 的 `goal_visibility` 与 `projection_validity` 两行更新；所有定位、mask、其他 uncertainty 输出和
Motion Head 保持冻结。

## 结果

**E017-P0 未通过，不产生 checkpoint。**

- 父模型 validation goal visibility precision / recall：`0.994662 / 0.917898`。
- 父模型 validation unobservable FPR：`3/558 = 0.005376`。
- 微调 epoch 1 已把 FPR 提高到 `14/558 = 0.025090`；epoch 3–8 为
  `16/558 = 0.028674`。
- recall 最高提高到 `0.967159`，但 precision 最低降到约 `0.973510`。
- observable-goal UV MAE、goal mask IoU 和全部冻结参数 identity 均无训练漂移。
- 8 个 epoch 均未通过 validation safety guardrail；`formal_checkpoint_written=false`。
- 训练与 input audit 均只包含 train/val，`test_split_read=false`；始终 no-actuation。

训练 loss 从 epoch 1 的 `0.074603` 降到 epoch 8 的 `0.066845`，但安全指标同步恶化。这说明 loss
下降不能作为 goal memory write gate 的替代证据。当前 train distribution 上的最后两行微调把分类器推向
更高 recall、更低 precision，并没有解决 unsafe write。

## Post-result validation diagnostic

失败结果冻结后，对未修改的 E016-P1 父 checkpoint 做了一个只读 validation threshold sweep；它不是
预注册训练结果，也不改变 E017-P0 的 failed 状态。

- threshold `0.500`：precision `0.994662`、recall `0.917898`、FPR `3/558`；
- threshold `0.525`：precision `0.996422`、recall `0.914614`、FPR `2/558`；
- threshold `0.600`：precision `0.996370`、recall `0.901478`、FPR `2/558`；
- threshold `0.625`：precision `0.998175`、recall `0.898194`、FPR `1/558`，已跌破 recall `0.90`；
- threshold `0.950`：FPR 为零，但 recall 只有 `0.783251`。

因此 `0.525` 是当前 validation 上最小且明确改善 precision/FPR、同时保留 recall 余量的探索性
operating point。它不能用 E016 已消费的 test-once 追认；必须在新的 held-out 数据上预注册后验证。

## 结论与边界

当前证据不支持继续对同一两行追加 epoch、事后改变负类权重，或保存任一失败 epoch。更合理的下一步是：

1. 保留 E016-P1 checkpoint 不变；
2. 把 `0.525` 作为新实验的 validation-only 候选，不回写 E016；
3. 等支持 NVIDIA Vulkan 的采集容器可用后，采集全新 held-out，再验证 unsafe write 与 memory；
4. 在新 held-out 前继续保持 `safe_for_actuator_promotion=false`。

本目录不包含 Dataset、RGB、NPZ、模型权重或私有逐样本 test 输出。
