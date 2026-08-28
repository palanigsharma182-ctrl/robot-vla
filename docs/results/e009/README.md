# E009 — Layer 12 periodic checkpoint 的 Reach/Transport sweep

## 技术摘要

- **没有 checkpoint 通过预注册 promotion 门槛。** epoch 100 把 Reach 从 epoch 98 anchor 的
  `0/10` 提高到 `3/10`，并将平均 Predicate residual 从 `6.93 cm` 降至 `3.06 cm`；但 Transport
  同时从 `7/10` 降到 `2/10`，因此不能替换 anchor，也不触发后续 guardrail。
- **证据支持“技能/checkpoint 冲突”，不支持“只是选错 best.pt”。** Reach 的最好确认结果出现在
  epoch 100，Transport 的最好确认结果仍在 epoch 90/98；没有单个候选同时保住两项技能。
- **聚合 validation loss 不能代表各技能闭环行为。** epoch 98 与 epoch 100 的 validation total loss
  只差 `0.000029`，但二者在相同 10 个 seed 上表现为 `0/10 Reach + 7/10 Transport` 与
  `3/10 Reach + 2/10 Transport` 两种明显不同的行为折中。
- **结论仍受 10-seed 小样本限制。** Wilson 区间较宽，配对 exact McNemar 检查也没有达到传统
  `p<0.05`；本实验足以执行预注册的“不晋级”决策，但不能单独证明梯度干扰的具体机制。

## epoch 100 改善 Reach，但以 Transport 明显退化为代价

Stage B 使用独立 seeds `10023–10032`。成功率括号内为 Wilson 95% 区间；residual 是相对 4 cm
Outcome Predicate 阈值的平均超额距离，成功 Episode 的 residual 为 0。

| Candidate | Reach | Reach residual | Transport | Transport residual | 行为告警 |
| --- | ---: | ---: | ---: | ---: | --- |
| e090 | 0/10，0% [0.0%, 27.8%] | 6.54 cm | 7/10，70% [39.7%, 89.2%] | 2.60 cm | 无 |
| e098-best（anchor） | 0/10，0% [0.0%, 27.8%] | 6.93 cm | 7/10，70% [39.7%, 89.2%] | **0.86 cm** | 无 |
| e100 | **3/10，30% [10.8%, 60.3%]** | **3.06 cm** | 2/10，20% [5.7%, 51.0%] | 5.29 cm | 无 |

相对 anchor 的逐 seed 配对结果如下。`win/loss` 表示候选成功而 anchor 失败/候选失败而 anchor 成功，
不是按 residual 大小定义。

| Candidate | Skill | 成功数变化 | 配对 win/loss | residual 变化 | residual 比率 |
| --- | --- | ---: | ---: | ---: | ---: |
| e090 | Reach | 0 | 0/0 | -0.40 cm | 0.943× |
| e090 | Transport | 0 | 1/1 | +1.75 cm | 3.038× |
| e100 | Reach | **+3** | **3/0** | **-3.87 cm** | **0.442×** |
| e100 | Transport | **-5** | **1/6** | **+4.43 cm** | **6.172×** |

e100 的 Reach 同时满足“多成功至少 2 条”和“residual 至少降低 20%”，但 Transport 成功数回退 5 条，
违反“两个技能均不低于 anchor”的硬门槛。e090 的成功数不变，且没有技能达到 20% residual 改善。
所以 `promotion_candidate_labels=[]`，预注册的 Grasp/Lift/Place guardrail 不运行。

作为未纳入 promotion 规则的敏感性检查，e100 对 anchor 的配对 exact McNemar 双侧检验在 Reach 上为
`p=0.25`（3 个 discordant seed 全部为 win），Transport 上为 `p=0.125`（1 win、6 loss）。方向与
预注册指标一致，但样本不足以把差异表述为高置信总体效应。

## 三 seed 筛选正确缩小了范围，但不能承担正式结论

Stage A 使用 seeds `10020–10022`，对 11 个 checkpoint 共运行 66 个 Episode。下表的 residual 仍是
4 cm 阈值以上的平均超额距离。这里保留精确表格而不画连续训练曲线：Stage A 每点只有 3 个 seed，
且 checkpoint 是离散候选；连线图容易把筛选噪声误读成平滑训练趋势。

| Candidate | Reach | Reach residual | Transport | Transport residual | 行为告警 |
| --- | ---: | ---: | ---: | ---: | --- |
| e010 | 0/3 | 8.17 cm | 0/3 | 16.23 cm | 无 |
| e020 | 0/3 | 11.62 cm | 0/3 | 14.38 cm | 无 |
| e030 | 0/3 | 18.51 cm | 0/3 | 15.31 cm | Transport anomaly/saturation 1 |
| e040 | 0/3 | 13.08 cm | 0/3 | 7.52 cm | Reach anomaly/saturation 4 |
| e050 | 0/3 | 9.31 cm | 1/3 | 6.95 cm | 无 |
| e060 | 0/3 | 10.97 cm | 0/3 | 9.13 cm | 无 |
| e070 | 0/3 | 7.17 cm | 1/3 | 6.49 cm | 无 |
| e080 | 0/3 | 8.15 cm | 1/3 | 1.89 cm | 无 |
| e090 | 0/3 | 9.68 cm | **2/3** | **0.96 cm** | 无 |
| e098-best | 0/3 | 5.44 cm | **2/3** | 1.98 cm | 无 |
| e100 | **2/3** | **1.08 cm** | **2/3** | 1.16 cm | 无 |

Reach Top-2 为 `e100, e098-best`，Transport Top-2 为 `e090, e100`；加上强制 anchor 后，Stage B
并集为 `e090, e098-best, e100`。筛选中 e100 Transport 为 `2/3`，确认阶段却只有 `2/10`，说明
3-seed screening 只能用于控制成本，不能替代独立确认。

## 范围、数据与指标定义

- 模型：E008 Layer 12 五技能联合训练，Qwen hidden state 固定为第 12 层。
- 唯一实验变量：训练 checkpoint；模型、数据、评估器、控制器与 sampling 配置全部固定。
- 数据：`trusted-v0.4-event-recovery-220`，dataset SHA256
  `bc024b6b9c566ca9500945fb6ac262bf657bee713d8a5816229bdc8478139407`。
- Stage A：11 个候选 × Reach/Transport × 3 seeds，共 66 Episodes。
- Stage B：3 个入围候选 × Reach/Transport × 10 独立 seeds，共 60 Episodes。
- 策略预算：最多 100 policy environment steps；10-step Flow；每执行 4 步用新观测重新规划。
- 控制：temporal ensemble 开启，`recency_decay=0.5`，最多 3 次 anomaly replan。
- 采样：base seed `42424`；同一 skill/环境 seed 在所有 checkpoint 间严格配对。
- Reach residual：`max(final_tcp_to_object_distance_m - 0.04, 0)`。
- Transport residual：`max(final_object_to_goal_xy_distance_m - 0.04, 0)`。

## 实验方法与 promotion 规则

Orchestrator 从训练 `metrics.jsonl` 自动发现真实存在的 periodic checkpoint，并把 validation 最优的
epoch 98 `best.pt` 作为 anchor。Stage A 对每个技能先排除带系统错误、tracking saturation 或 anomaly
的 residual tie-break，再依次按成功数、residual、平均 policy steps 和 epoch 排名。每技能 Top-2 与
anchor 的并集进入 Stage B。

候选只有同时满足以下条件才可进入 guardrail，而不能直接替换默认 checkpoint：

1. Reach 和 Transport 成功数均不低于 anchor；
2. 至少一个技能多成功 `>=2/10`；
3. 被改善技能的 residual 至少降低 20%；
4. 两项技能均无系统错误、tracking saturation 或 anomaly replan。

本次没有候选满足第 1 条，因此不运行条件 guardrail，也不使用完整任务 test seed 继续调权重。

## 结果验证与可审计性

完成后从原始 JSONL 独立重算，未调用 sweep 聚合函数，得到：

- Stage A 66 行、11 个候选，Stage B 60 行、3 个候选；所有预期 skill/seed 身份恰好出现一次；
- 原始成功数、4 cm residual 与两个总汇总文件逐项一致；
- Stage B 的三个候选均为 0 个 system issue、0 次 tracking saturation、0 次 anomaly replan；
- 逐 seed win/loss 与 `sweep-summary.json` 一致；
- `sweep_code_revision` 为
  `source-tree-sha256:aeb1c1647de4eadf838838c0abf4e5c1c517d0d871e2f33c75d2b8b288a351f1`。

正式聚合文件 SHA256：

| File | SHA256 |
| --- | --- |
| `sweep-manifest.json` | `99788de3ad9e2654af8e1b8f7dec20bc2a0b17a20345391251c2dba281e00c14` |
| `screen-summary.json` | `ba5fdc9155e6cffcf3226123c8986affa591896e4481afa174dd78781a0ccf09` |
| `sweep-summary.json` | `3f74f6b9a3e1c63c86b5b62c750e530a0699b200cc4c5dacd33db52d82b287fb` |

`sweep-manifest.json` 保存候选、seed、validation 指标、控制参数和源码 revision；每个候选的
`experiment.json` 另外保存 checkpoint SHA256、完整模型契约和 dataset SHA256。仓库不包含模型权重。

## 结论能说明什么，不能说明什么

**可以说明：** 在这组严格配对的确认 seed 上，epoch 100 恢复了一部分 Reach 能力，但没有保住
Transport；epoch 98/90 保住 Transport，却没有 Reach 成功。由此排除“存在一个明显更好的单一
periodic checkpoint，只是 validation best 选错了”这一解释，并把主要问题收敛到多技能行为目标的
checkpoint 冲突。

**不能说明：** 本实验没有直接观测梯度、注意力寻址或 Reach→Grasp 的状态分布，所以不能把冲突机制
唯一归因为梯度干扰、Layer 12 语义退化或某个具体模块。10 个 seed 也不足以给出窄置信区间；这里的
结论是预注册候选决策和问题归因，不是总体成功率的精确估计。

## 下一步

1. 按 D024 先做 Reach→Grasp handoff probe，比较策略自产 Reach 终态与专家准备态的 TCP/物体相对
   位姿、速度、夹爪开度和双相机观测分布；这是当前成本最低、归因最清楚的下一步。
2. 训练和 checkpoint 诊断中保留 per-skill 闭环指标，不能继续只用聚合 validation loss 选行为模型；
   但不部署按技能手写切换 e100/e098 的 checkpoint router。
3. 若 handoff probe 继续支持“Layer 12 几何有效但语义寻址/交接退化”，再受控比较 Layer 24
   semantic Key + 同 token Layer 12 geometry Value；重新训练并重新走独立原子/完整闭环门槛。
4. 不继续盲目增加相同配置的训练 epoch；E009 已表明最后两个 epoch 的主要变化是技能折中移动，
   而不是两个技能共同改善。

## 尚待回答的问题

- 策略自产 Reach 终态与独立 Grasp 的专家准备态，主要差在 TCP/物体相对位姿、速度、夹爪开度，
  还是双相机中的目标可见性？
- epoch 98→100 的技能折中来自样本/事件 loss 权重、共享 Action Expert 的梯度竞争，还是 Layer 12
  Context 的语义目标寻址退化？E009 只能定位行为现象，不能区分这些机制。
- e100 的 Reach `3/10` 是否能在新的确认 seed 上稳定复现？在 handoff 机制尚未明确前，不应为回答
  这个问题直接消耗完整任务 test seed。

## 代码与结果文件

- Sweep 聚合与 promotion 逻辑：
  [`src/robot_vla/evaluation/checkpoint_sweep.py`](../../../src/robot_vla/evaluation/checkpoint_sweep.py)
- 可恢复的顺序运行 CLI：
  [`src/robot_vla/cli/evaluate_checkpoint_sweep.py`](../../../src/robot_vla/cli/evaluate_checkpoint_sweep.py)
- 纯函数与边界测试：[`tests/test_checkpoint_sweep.py`](../../../tests/test_checkpoint_sweep.py)
- 实验身份与冻结配置：[`sweep-manifest.json`](sweep-manifest.json)
- Stage A 汇总与排名：[`screen-summary.json`](screen-summary.json)
- Stage B 汇总、配对比较和 promotion 结果：[`sweep-summary.json`](sweep-summary.json)
- Stage A 原始结果：[`screen/`](screen/)
- Stage B 原始结果：[`confirm/`](confirm/)

每个候选目录均包含 `experiment.json`、逐 Episode 的 `episodes.jsonl` 和 `summary.json`。运行日志保留在
本地 ignored artifacts 中，没有作为分析证据或 GitHub 正式结果上传。
