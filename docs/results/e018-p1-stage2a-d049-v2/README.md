# E018-P1 Stage 2A / D049 v2 — selected-gain development evaluation

D049 v2 已完成，结果分类为
`effect-negative-persist-publish-pause-for-reusability-refactor`。这是
**development offline / no-test / no-actuation** 证据，不是正式闭环或部署证据。

## 聚合结果

| 阶段 | gain | support | recovered | recovery rate |
| --- | ---: | ---: | ---: | ---: |
| 选择阶段 | `0.10` | 24 | 5 | 20.83% |
| 正式 development evaluation | `0.10` | 25 | 7 | 28.00% |

选择阶段的三个候选 gain 都是 `5/24`；`0.10` 仅由预注册的“大 gain 优先”平局规则选中，
不代表它优于其他 gain。正式 development evaluation 达到最小 support `10`，但 `7/25 = 28%`
远低于要求的 `70%`，因此 effect gate 失败。unsafe、catastrophic、false recovery 和 protocol
violation 的聚合计数均为零，但这不能推出 actuator safety。

选择阶段的 `5/24` 与新正式身份上的 `7/25` 存在样本身份间波动。后者的观察比例虽高
`7.17` 个百分点，但两者都明显低于效果门槛，不能证明稳定泛化，更不能支持 gain 优越性。

## 证据边界与后续状态

- fresh test、runtime ground truth、goal ground truth 的读取计数均为零；
- 相机、机械臂/TCP、夹爪执行计数均为零，canonical runtime 未修改；
- 已消费的实验身份不得复用或重新声明为未见数据，也不得事后调阈值追认本结果；
- `stage2b_continuation_required=false`，当前状态进入 `PAUSE_FOR_REUSABILITY_REFACTOR`；
- 本结果不支持 gain superiority、canonical promotion、actuator safety、物理闭环成功或 deployment readiness。

本目录只包含脱敏聚合、证据 identity 和标准库验证器，不包含逐样本记录、图像、标签或模型内容。
可从任意工作目录执行：

```bash
python3 /absolute/path/to/docs/results/e018-p1-stage2a-d049-v2/verify_summary.py
```
