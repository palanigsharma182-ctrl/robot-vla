# 实验记录

本文用于记录机器学习训练、Benchmark、消融、仿真评估和闭环实验，使结果能够复现、比较并指导下一步工作。

## 当前状态

项目已经完成 30 条可信轨迹的首轮 Stage 1、扩充到 100/120/220 条后的独立重训和受控消融、
Action 安全拒绝诊断、事件损失、temporal ensemble，以及固定低事件权重的 100-epoch 正式训练和
最终闭环评估。随后完成 Qwen Layer 12/24 空间 probe、Oracle/Layer 12 Reach 诊断，以及 Layer 12
五技能联合训练和统一闭环评估；Layer 12 改善了部分 Reach 表现，但没有提高完整任务成功率。
单元测试和模型 smoke test 只证明接口与计算链路可运行，不记作任务效果实验；
下列任务效果结论只来自完整 test/unseen seed Rollout。

## 记录原则

- 每个实验只回答一个明确的主要问题。
- 保留简单且固定的 Baseline，并说明相对 Baseline 唯一或主要变化。
- Config 必须足以复现，至少包含代码版本、数据、任务/环境、模型、训练或推理配置、硬件和随机种子。
- Result 区分事实与解释；失败、无提升和中止的实验同样保留。
- 优先报告任务成功率，同时记录能够解释变化的离线指标、失败类型和资源消耗。
- 指标必须注明聚合方式、样本数或 Episode 数；多个随机种子应报告均值和离散程度。
- 原始日志、Checkpoint 和大体积数据不写入本文，只记录其稳定路径或外部引用。
- 如果实验改变了 Observation / Action、Dataset、Loss、Planner 或 Evaluation 等长期契约，应先在 [decisions.md](decisions.md) 记录或更新相关决策。

## 实验索引

| ID | 日期 | 简称 | 状态 | 主要结论 |
| --- | --- | --- | --- | --- |
| E001 | 2026-08-25 | 30 条数据 Stage 1 与首轮闭环 | completed | 离线 loss 下降，但 23 个闭环 Episode 均在 reach 阶段失败 |
| E002 | 2026-08-25 | 扩充到 100 条后的独立重训 | completed | 未见场景的原子技能明显改善，但完整成功仍为 0%，并出现 4 次动作安全拒绝 |
| E003 | 2026-08-25 | 安全拒绝根因与控制饱和复现 | completed | 4 次拒绝来自执行器跟踪滞后；显式饱和后 4/4 不再误报模型越界 |
| E004 | 2026-08-25 | E002 独立原子技能基线 | completed | grasp/lift 稳定，独立瓶颈为 reach、transport 和 place |
| E005 | 2026-08-25 | 20 条恢复数据的 30-epoch A/B | completed | 完整成功仍为 0，但恢复数据显著推进 unseen 闭环阶段，暂不换架构 |
| E006 | 2026-08-26 | v0.4 事件数据、事件损失与 warmup 消融 | completed | 高事件权重造成技能竞争；固定 `lambda=0.25` 是唯一完整保住 grasp/lift 的低回归方案，进入独立 100-epoch 正式训练 |
| E007 | 2026-08-26 | 固定 `lambda=0.25` 的正式训练与控制消融 | completed | 原子提升到 16/25，ensemble 明显改善阶段深度，但 20 unseen 完整成功仍为 0/20 |
| E008 | 2026-08-28 | Qwen Layer 12 空间表示、Reach 与五技能组合诊断 | completed | Layer 12 的位置可解码性和完整 Reach 通过数优于 Layer 24，但原子仍为 16/25、完整仍为 0/20；主要问题收敛到多技能目标冲突和 Reach→Grasp/Lift→Transport 交接 |
| E009 | 2026-08-29 | Layer 12 periodic checkpoint 的 Reach/Transport sweep | completed | epoch 100 将 Reach 从 0/10 提到 3/10，却使 Transport 从 7/10 降到 2/10；无单一 promotion 候选，确认技能/checkpoint 冲突而非只选错 best.pt |
| E010 | 2026-08-29 | Layer 12 五技能梯度冲突与 base/event 归因 probe | planned | 用严格配对的 per-skill 梯度 Gram/cosine 定位 epoch 98→100 的 Reach/Transport 行为交换发生在输出头、Expert 层、Adapter，还是 base/event 目标内部 |

## E001 — 30 条数据 Stage 1 与首轮闭环

**Date:** 2026-08-25

**Status:** completed

**Experiment:**

使用首批 30 条成功轨迹从头训练 Frozen Qwen + Adapter/Action Expert，并在 3 个 test scene 和
20 个不在 Dataset manifest 中的新 seed 上运行 20 Hz、每 4 步重规划的真实 ManiSkill 闭环。

**Goal:**

验证第一批小数据能否让策略在严格的必须释放放置环境中完成任务；离线 Flow loss 仅用于选择
候选权重，完整任务和五个原子技能成功率才是效果标准。

**Config:**

- Code version: `source-tree-sha256:07e78a21caaff741f4900d7b8a13cd99ac2c0ec674142f1abccded3b35b1fa56`
- Dataset: `trusted-v0.1-small`，30 trajectories / 5996 steps，SHA256 `10848c88905a6ccf9f759e6fb7d5e7833a2087fc23198b5225df2ce243bb141b`
- Task / environment: `RobotVLAPickCubeToRegion-v1`
- Model: Frozen `Qwen/Qwen3.5-2B` revision `15852e8c16360a2fea060d615a32b45270f8a8fc` + Adapter + 16-layer Action Expert
- Train: 100 epochs，4096 samples/epoch，batch 64，AdamW `1e-4`，warmup 1000，cosine 30000
- Evaluation: 10-step Flow Euler，执行 Chunk 前 4 步；3 test + 20 unseen，sampling seed 42424
- Hardware: NVIDIA GeForce RTX 4090 24GB
- Artifacts: `/home/ubuntu/robot-vla-runs/stage1-v0.1`、`/home/ubuntu/robot-vla-runs/stage1-v0.1-rollout`

**Result:**

- 训练完整完成 100 epochs / 6400 optimizer steps / 409600 examples；日志无 traceback/OOM。
- 全部 epoch 最低 val loss 为 `0.0147566`（epoch 89），但旧周期保存逻辑没有留下该非周期权重。
- 实际 periodic 候选中最低 val loss 为 `0.0149788`（epoch 100，`step-00006400.pt`）；正式闭环使用该权重。
- 闭环完整任务成功率：`0/23 = 0%`，95% Wilson 区间 `[0, 0.1431]`。
- test：`0/3`；unseen：`0/20`，unseen 95% Wilson 上界 `0.1611`。
- reach/grasp/lift/transport/place 通过率均为 `0%`；23 条失败全部为 `reach_failed`。
- 所有 Episode 均正常运行 300 步 / 75 次 Replan 后超时；推理、控制和动作安全错误均为 0。
- 最终 TCP 到方块距离平均 `0.3039 m`，范围 `0.2229–0.4195 m`。

**Conclusion:**

五样本过拟合和低离线 loss 只证明训练链路可学习，不能推出闭环操作能力。30 条成功轨迹没有提供
足够的场景覆盖让当前 Frozen-Qwen 策略学会最早的 reach；由于系统错误计数为 0，失败主要归因于
策略/数据，而不是评估或控制链路崩溃。

**Next step:**

保持模型架构、采样权重和训练计算不变，将可信场景扩充到 100 条后独立重训，并在相同 20 个
unseen seed 上复评，以隔离数据覆盖增加的影响。

## E002 — 扩充到 100 条后的独立重训

**Date:** 2026-08-25

**Status:** completed

**Experiment:**

在保持 E001 模型与优化配置不变的条件下，把可信成功轨迹扩充为 80/10/10 共 100 条，并从头
初始化 Adapter/Expert 完成一轮训练；Checkpoint 只改变存储频率与 best 正确性，不改变优化。

**Goal:**

判断场景覆盖由 30 条扩大到 100 条后，未见 seed 的 reach 和完整任务成功率是否出现真实改善。

**Config:**

- Code version: `source-tree-sha256:cc0714e4eee79239582ca1232ab8e91dedaf629addb0df3d1d260e80c91a6404`
- Dataset: `trusted-v0.2-100`，100 trajectories / 20018 steps，SHA256 `b9ea0f47f8140ed91ddc3f0ded1eb34d326f9917ccf94ef293a93ea02ad305a1`
- Manifest SHA256: `7802f13a3d14b2eedee088fee02e8a14547e6fca768e3ed361fed2ed17141e32`
- Train / model / hardware: 与 E001 相同；periodic 每 10 epochs / 640 steps，任何 val 改善都更新 best
- Evaluation: 10-step Flow Euler，执行 Chunk 前 4 步；10 test + 与 E001 相同的 20 unseen，sampling seed 42424
- Artifacts: `/home/ubuntu/robot-vla-runs/stage1-v0.2-data100`、`/home/ubuntu/robot-vla-runs/stage1-v0.2-data100-rollout`

**Result:**

- 训练完整完成 100 epochs / 6400 optimizer steps / 409600 examples；日志无 traceback/OOM/RuntimeError。
- 全部 epoch 最低 val loss 为 `0.0124065`（epoch 90）；`best.pt` 的 trainer state 与该最低点一致，
  Checkpoint SHA256 为 `4f9b8a0d9b0966674e9232e0ddc08ad85aff7d0db2a72f3cb5d3ec03f2a810ae`。
- 闭环完整任务成功率：`0/30 = 0%`，95% Wilson 区间 `[0, 0.1135]`；test `0/10`，
  unseen `0/20`，unseen 95% Wilson 上界 `0.1611`。
- test 原子技能通过数：reach `7/10`、grasp `3/10`、lift `2/10`、transport `0/10`、place `0/10`。
- unseen 原子技能通过数：reach `9/20`、grasp `5/20`、lift `3/20`、transport `1/20`、place `0/20`。
- overall 失败分类：reach 10、grasp 8、lift 3、transport 4、release 1、Action 安全拒绝 4；
  除安全契约拒绝外没有推理、控制或环境运行时错误。
- 与 E001 相同 20 个 unseen seed 的平均最终 TCP-to-object 距离从 `0.3109 m` 降至 `0.0808 m`；
  reach 从 `0/20` 提升至 `9/20`，但完整成功仍同为 `0/20`。

**Conclusion:**

把可信轨迹从 30 条扩充到 100 条，在未见场景中真实改善了 reach，并让少数 Episode 进入 grasp、lift、
transport 甚至 release 阶段，因此提升不是仅由离线 loss 得出的推测。但当前策略仍没有任何完整成功，place
为 0，且 `4/20` unseen Episode 触发动作安全拒绝；所以本轮证明了数据覆盖方向有效，却仍不足以让系统可用。

**Next step:**

优先复现并定位 4 个安全拒绝 seed 的具体越界维度，同时针对 grasp、transport/release/place 失败增加
分层覆盖和失败恢复数据；下一轮仍使用相同 unseen seed 作为不可训练的固定对照集。

## E003 — 安全拒绝根因与控制饱和复现

**Date:** 2026-08-25

**Status:** completed

**Experiment:**

对 E002 unseen seed `10006、10008、10009、10016` 的 Action 安全拒绝增加结构化诊断，固定
Checkpoint、采样 seed 和环境 seed 逐步复现；随后仅对执行器内部 tracking correction 应用 D020
的显式饱和并重跑相同 4 seed。

**Goal:**

区分模型 Action 越界和控制器跟踪误差，消除错误失败归因，同时保持模型 Action 契约不变。

**Config:**

- Dataset / Checkpoint: E002 `trusted-v0.2-100` / epoch 90 `best.pt`
- Evaluation seeds: `10006、10008、10009、10016`
- Action limit: 每关节有效上限不超过 `0.05 rad/control-step`
- Artifacts: `/home/ubuntu/robot-vla-runs/stage1-v0.2-data100-safety-diagnostics`、
  `/home/ubuntu/robot-vla-runs/stage1-v0.2-data100-saturated-control`

**Result:**

- 4 次拒绝都发生在 Chunk 执行前缀索引 3、Franka 关节索引 3。
- 请求 tracking correction 分别为 `+0.05188167、+0.05381572、-0.05060697、+0.05012619 rad`。
- 模型原始 Action Chunk 和 gripper 均未越界，根因是前三步目标的仿真跟踪滞后。
- 饱和修复后 4/4 不再触发安全拒绝；实际修正最大值始终不超过 `0.05 rad`。
- 修复后仍无完整成功：2 个 reach_failed、2 个 grasp_failed；没有伪造能力提升。

**Conclusion:**

E002 的 4 次安全拒绝不代表模型直接产生危险动作。D020 修复了控制层错误归因，但没有提升任务
能力，因此后续仍需针对策略和数据瓶颈训练。

**Next step:**

用专家准备的精确前置状态独立评估五个原子技能，隔离前序误差传递。

## E004 — E002 独立原子技能基线

**Date:** 2026-08-25

**Status:** completed

**Experiment:**

专家只完成目标原子技能之前的 Predicate 阶段，策略在 seeds `10000–10004` 上分别执行
reach、grasp、lift、transport、place，每个技能最多 100 个策略环境步。

**Goal:**

判断完整闭环失败来自原子能力本身，还是前序技能误差向后传播。

**Config:**

- Dataset / Checkpoint: E002 `trusted-v0.2-100` / epoch 90 `best.pt`
- Preparation: `trusted-mplib-prerequisites/v1`，由 `PickPlaceTaskTracker` 精确验证前置阶段
- Evaluation: 5 seeds × 5 skills = 25 Episodes，10-step Flow，sampling seed 42424
- Artifacts: `/home/ubuntu/robot-vla-runs/stage1-v0.2-data100-atomic-seeds5-v3`

**Result:**

- reach `0/5`，平均最终 TCP→方块距离 `0.0790 m`。
- grasp `5/5`，平均 4 个策略步。
- lift `5/5`，平均 28 个策略步。
- transport `0/5`，平均最终目标 XY 距离 `0.09697 m`，方块一直保持抓取。
- place `1/5`；其余 4 个失败均跑满 100 步，4/5 最终仍保持抓取。
- 合计 `11/25`；正式日志无 Traceback、OOM 或 RuntimeError。

**Conclusion:**

grasp/lift 在可信前置状态下已经稳定，完整 Rollout 中的部分 grasp/lift 失败来自前序误差传递；
独立能力瓶颈是 reach、transport 和 release/place。下一批数据和训练预算不应继续平均对待五个阶段。

**Next step:**

保持 trajectory/v2 完整成功契约，在完整专家轨迹内增加五类可恢复扰动，并用同架构小规模 A/B
隔离数据变化的贡献。

## E005 — 20 条恢复数据的 30-epoch A/B

**Date:** 2026-08-25

**Status:** completed

**Experiment:**

在不修改模型架构、Flow loss、优化器和控制协议的条件下，把 20 条五类恢复轨迹加入 E002 数据，
并从相同 seed 独立初始化 30-epoch Control/Treatment；随后运行相同原子与 unseen 闭环评估。

**Goal:**

判断完整成功轨迹内的失败恢复数据是否能在相同小规模训练预算下推进闭环技能阶段，并据此决定
继续数据/训练目标方向还是更换模型架构。

**Config:**

- Code version: `source-tree-sha256:aee70a16323011beda77d6b2ac95fcc89fc0cab9c5d8d292bfef665a56d1d97f`
- Control data: `trusted-v0.2-100`，100 trajectories / 20018 steps，SHA256
  `b9ea0f47f8140ed91ddc3f0ded1eb34d326f9917ccf94ef293a93ea02ad305a1`
- Treatment data: `trusted-v0.3-recovery-120`，120 trajectories / 24841 steps，96/12/12，SHA256
  `a9928519b9581ba3dae911aa547f2ba36a7e746e4ffe5cba8cf8e920afd4f28a`
- Recovery: reach/grasp/lift/transport/place 各 4 条；新增 split 16/2/2；全部完整成功；Action 饱和率 0
- Train: 两组各 30 epochs、4096 samples/epoch、batch 64、seed 42；其余与 E002 相同
- Evaluation: seeds `10000–10004` 的 25 原子 Episodes，以及 seeds `10000–10019` 的 20 unseen
  完整闭环；10-step Flow，sampling seed 42424
- Control artifacts: `/home/ubuntu/robot-vla-runs/ablation-v0.3-control-data100-e30*`
- Treatment artifacts: `/home/ubuntu/robot-vla-runs/ablation-v0.3-recovery-data120-e30*`

**Result:**

- 父数据复审 SHA256 与 E002 完全一致；20 条恢复轨迹采集另拒绝 3 个不可信候选，未写入 Dataset。
- 两组均完整训练 30 epochs / 1920 optimizer steps / 122880 examples，无 Traceback/OOM/NaN。
- Control best: epoch 29，val loss `0.0243695`，Checkpoint SHA256
  `90bf893c62ca93d632b3b005354d23d411b52127beb098be1ca0b5b9babc8e15`。
- Treatment best: epoch 30，val loss `0.0257156`，Checkpoint SHA256
  `b9c473d6f0be68d5ceaf661f261e07a61506c99881bbe99368a3afb161d66e79`。
- 原子 Control：reach/grasp/lift/transport/place = `0/5、5/5、4/5、1/5、0/5`，合计 `10/25`。
- 原子 Treatment：`0/5、5/5、5/5、0/5、0/5`，合计同为 `10/25`；reach 最终 TCP 距离均值
  从 `0.2130 m` 降到 `0.1188 m`，transport 目标 XY 距离均值从 `0.1832 m` 降到 `0.1367 m`，
  但都没有转化为 Predicate 成功；place 仍未改善。
- unseen 完整成功两组均为 `0/20`。
- unseen 阶段通过数 Control 为 reach/grasp/lift/transport/place = `2/0/0/0/0`；Treatment 为
  `10/8/4/1/0`。20 个配对 seed 中 10 个阶段更深、10 个相同、0 个回退；平均完成技能数从
  `0.10` 提高到 `1.15`，平均最终 TCP→方块距离从 `0.4044 m` 降到 `0.1233 m`。
- Treatment 失败分布为 reach 10、grasp 2、lift 4、transport 3、release 1；tracking correction
  饱和 Episode 从 Control `10/20` 降到 `3/20`，没有 Action 安全拒绝或运行时错误。

**Conclusion:**

恢复数据在相同小预算下显著推进了完整闭环的前半和中段技能，因此现有 Frozen Qwen + Expert
架构具备利用这类数据的能力，当前没有更换核心架构的证据。但完整成功和 place 仍为 0，独立
reach/transport/place 也没有形成成功提升，所以 20 条均匀恢复数据还不够。下一项受控变量应是
D021 的瓶颈阶段采样，而不是同时修改模型或 Flow loss。

**Next step:**

保持 v0.3 数据、架构、优化器、seed 和计算预算不变，显式使用
`--skill-weights 1.5 1 1 1.5 2` 与旧权重做受控训练对照；只有该方向仍不能改善独立
reach/transport/place 时，才测试关键阶段 loss 或 Qwen 表示层。

## E006 — v0.4 事件数据、事件损失与 warmup 消融

**Date:** 2026-08-26

**Status:** completed

**Experiment:**

把可信数据扩充到 220 条事件密集型完整成功/恢复轨迹，在保持 Frozen Qwen、Action Expert、
joint-space Action、sampler、优化器、seed 和控制协议不变的条件下，对数据量、事件损失权重和事件
权重 warmup 做 A–F 受控消融。每组先训练 30 epochs，再对通过离线完整性检查的 best Checkpoint
运行相同的 25 个原子技能 Episode；对固定 `lambda=0.25` 候选另补 20 个 unseen 完整闭环。

**Goal:**

验证下面的目标能否在不破坏原始 16-step BC/Flow 学习的情况下，让实际执行前 4 步中的关键事件
获得额外监督：

```text
L = L_base + lambda_event * L_event
critical_mask = event_mask & action_mask & [1, 1, 1, 1, 0, ..., 0]
```

正式方案必须优先保住已稳定的 grasp/lift；reach 不应明显退化，并检查 release/place 是否相对
base-only 或固定低权重出现改善。单独降低离线 loss 不能通过该门槛。

**Config:**

- Dataset A: `trusted-v0.3-recovery-120`，120 trajectories，SHA256
  `a9928519b9581ba3dae911aa547f2ba36a7e746e4ffe5cba8cf8e920afd4f28a`
- Dataset B–F: `trusted-v0.4-event-recovery-220`，220 trajectories / 48922 steps，176/22/22，
  success rate 100%，SHA256
  `bc024b6b9c566ca9500945fb6ac262bf657bee713d8a5816229bdc8478139407`
- Events: grasp/release command、contact、linear/angular velocity jump、pickup、place
- Train: 30 epochs，4096 samples/epoch，batch 64，seed 42，AdamW `1e-4`，LR warmup 1000，
  cosine 30000，skill weights `1.5/1/1/1.5/2`
- A/B: `lambda=0`；C: 固定 `lambda=2`；D: 固定 `lambda=0.25`；E: 目标 `lambda=1` +
  1000-step linear warmup；F: 目标 `lambda=0.5` + 1000-step linear warmup
- Train code revisions: A–C `174c7ce7...1b3376a`；D `64fd5c9b...54c6`；E–F
  `ce39eb8c...d47b187`。后两个 revision 只新增向后兼容的事件权重 warmup 与指标字段；
  validation 始终使用固定目标权重
- Atomic evaluation: seeds `10000–10004`，5 skills × 5 seeds，100 policy steps，sampling seed
  42424，10-step Flow，temporal ensemble `rho=0.5`，max anomaly replans 3
- Full evaluation: test episodes 0，unseen seeds `10000–10019`，其余协议与原子评估相同
- A–C artifacts: `/home/ubuntu/robot-vla-runs/ablation-v0.4-*` 与
  `/home/ubuntu/robot-vla-runs/e006-*`
- D–F RAM artifacts: `/dev/shm/robot-vla-runs/ablation-v0.4-*` 与
  `/dev/shm/robot-vla-runs/e006-*`；均在每阶段结束后用 `rsync -aH` 持久化到
  `<local-artifact-root>/server-runs/` 并校验 SHA256

训练结果如下；不同 `lambda` 的 total validation loss 标尺不同，跨配置只能结合 base/event 分项与
闭环行为解释：

| 组 | 数据 / 事件目标 | best epoch | val total | val base | val event | best Checkpoint SHA256 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| A | v0.3 / `lambda=0` | 30 | 0.02728 | 0.02728 | — | `730c3fcc...232b3f` |
| B | v0.4 / `lambda=0` | 29 | 0.02830 | 0.02830 | — | `9becf8fb...c03fb4` |
| C | v0.4 / 固定 `lambda=2` | 26 | 0.30535 | 0.12464 | 0.09063 | `50b35d81...e17e99` |
| D | v0.4 / 固定 `lambda=0.25` | 30 | 0.07290 | 0.05443 | 0.06833 | `6b368fe3...6ec2` |
| E | v0.4 / `lambda=1`，warmup 1000 | 23 | 0.17540 | 0.09420 | 0.08237 | `5be54933...31e0a9` |
| F | v0.4 / `lambda=0.5`，warmup 1000 | 25 | 0.11342 | 0.07126 | 0.08047 | `d1713f98...1dc8eb` |

**Result:**

采样器在 30 epochs 的 122880 个窗口中产生 7502 个含事件窗口、9923 个合并关键 step。事件按
阶段极不均衡：reach 6、grasp 2746、lift 1402、transport 0、place 5769；其中 place 包含
3268 次 release exposure 和 3608 次 angular jump。事件损失因此不能直接改善几乎没有事件的
reach/transport，并会在权重过高时让共享网络偏向 place。

固定 `lambda=2` 的 C 在前两轮平均梯度约 81/53，直到约 epoch 23 才降到 10 以下，长期被
`max_grad_norm=10` 主导。E 的 warmup 把完整 `lambda=1` 的 14 个 epoch 平均/最大梯度降到
`5.44/7.01`；F 的完整 `lambda=0.5` 进一步降到 `3.02/3.78`。warmup 解决了优化稳定性，
但没有自动解决不同技能在共享参数上的行为竞争。

统一原子结果：

| 组 / Checkpoint | Reach | Grasp | Lift | Transport | Place | 总成功 | saturation / anomaly replan |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A best | 2/5 | 5/5 | 5/5 | 0/5 | 0/5 | 12/25 | 0 / 0 |
| B best | 1/5 | 5/5 | 5/5 | 0/5 | 0/5 | 11/25 | 4 / 4 |
| C best | 0/5 | 5/5 | 0/5 | 0/5 | 2/5 | 7/25 | 0 / 0 |
| D best | 0/5 | 5/5 | 5/5 | 0/5 | 0/5 | 10/25 | 0 / 0 |
| E best (epoch 23) | 1/5 | 5/5 | 0/5 | 1/5 | 1/5 | 8/25 | 0 / 0 |
| E latest (epoch 30) | 0/5 | 5/5 | 3/5 | 0/5 | 0/5 | 8/25 | 0 / 0 |
| F best | 0/5 | 5/5 | 3/5 | 0/5 | 0/5 | 8/25 | 1 / 1 |

E best 把 place 最终仍抓取率从 D 的 100% 降到 60%，证明模型确实学到部分 release；但 lift
完全丢失。E latest 把 lift 恢复到 3/5，却把 place 仍抓取率拉回 100%，说明能力随训练时点漂移。
F 的离线 base/event 同时优于 E，但闭环仍只有 lift 3/5 且没有 release/place，进一步证明离线
平衡不能代替行为评估。

A–D 的统一 20 unseen 完整闭环结果：

| 组 | 完整成功 | Reach | Grasp | Lift | Transport | Place | 平均完成技能数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 0/20 | 8/20 | 4/20 | 2/20 | 0/20 | 0/20 | 0.70 |
| B | 0/20 | 6/20 | 2/20 | 0/20 | 0/20 | 0/20 | 0.40 |
| C | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 | 0.00 |
| D | 0/20 | 3/20 | 0/20 | 0/20 | 0/20 | 0/20 | 0.15 |

D 的 20 unseen 失败为 reach 17、grasp 3；无 saturation 或 anomaly replan。30-epoch D 尚未
改善完整闭环，但在所有带事件目标的配置中，只有它完整保住原子 grasp/lift `5/5、5/5`。

**Conclusion:**

固定高事件权重会把 place/release 信号换成 reach/lift 退化；warmup 能解决早期梯度冲突，但
`lambda=0.5–1` 仍没有得到稳定技能组合。继续盲试相邻常数的证据收益已经很低，因此第一版选择
固定 `lambda=0.25` 作为最低回归风险方案：接受 30-epoch 时 release 尚未改善，通过独立的
100-epoch 正式训练让较弱事件监督在更长预算中累积。不能把 D 的 0/20 完整成功描述为通过或提升；
正式结论只来自新 run 的最终闭环评估。

**Next step:**

使用 v0.4 数据、固定 `lambda=0.25`、相同 sampler/optimizer/seed 从随机初始化训练 100 epochs，
保留每 10 epochs 的 periodic Checkpoint；随后运行 25 个原子、20 个 unseen 完整闭环，以及
newest-only / ensemble-only / ensemble+replan 三组控制消融。

## E007 — 固定 `lambda=0.25` 的正式训练与控制消融

**Date:** 2026-08-26

**Status:** completed

**Experiment:**

使用 E006 选出的固定 `lambda_event=0.25`，在 v0.4 的 220 条可信轨迹上从随机初始化独立训练
100 epochs；选择验证总损失最低的 checkpoint，完成正式原子评估和三组 20 unseen 控制消融。

**Goal:**

验证较弱事件监督在更长预算中能否同时保留 grasp/lift 并积累 transport/release/place 能力；同时
隔离 temporal ensemble 和异常重规划预算对完整闭环的实际贡献。完整任务效果只以统一 unseen
闭环为准，原子成功率和离线 loss 不能代替完整成功率。

**Config:**

- Code revision: `source-tree-sha256:ce39eb8c7548fd433e84b75e50415e2e16313b17bb1a19563cc256994d47b187`
- Dataset: `trusted-v0.4-event-recovery-220`，220 trajectories / 48922 steps，176/22/22，
  dataset SHA256 `bc024b6b...39407`，manifest SHA256 `43f131cc...f477f`
- Train: 100 epochs，4096 samples/epoch，batch 64，6400 optimizer steps，seed 42，AdamW `1e-4`，
  LR warmup 1000 / cosine 30000，固定 `lambda_event=0.25`，无事件权重 warmup，技能采样权重
  `1.5/1/1/1.5/2`；每 10 epochs 保存 periodic checkpoint
- Atomic: seeds `10000–10004`，5 skills × 5 seeds，100 policy steps，10-step Flow，sampling seed
  42424，temporal ensemble `rho=0.5`，max anomaly replans 3
- Full/control: 同一 best、unseen seeds `10000–10019`、test episodes 0；newest-only、
  ensemble-only、ensemble+replan 只改变 D023 规定的两个控制变量
- Hardware: 单张 RTX 4090 24GB；训练输出写入 `/dev/shm`，每阶段结束后立即用 `rsync -aH`
  持久化到 `<local-artifact-root>/server-runs/`
- Formal run: `stage1-v0.4-data220-event025-e100`；评估产物：`e007-formal-*`

**Result:**

正式训练完整结束，metrics 恰好 100 行且 epoch 连续，无 NaN/Inf、Traceback、OOM 或
RuntimeError；10 个 periodic checkpoint、best/latest 齐全。峰值 CUDA allocated/reserved 约
10.42/11.29 GB。训练目录保留 latest 与 step-6400 的硬链接，并已完成远端/本地全文件 SHA256
一致性校验。

| Checkpoint | Epoch | Val total | Val base | Val event | SHA256 |
| --- | ---: | ---: | ---: | ---: | --- |
| best | 98 | 0.030852 | 0.023178 | 0.032872 | `636d4374...e168` |
| latest | 100 | 0.037377 | 0.023225 | 0.055372 | `44932bfb...4875` |

正式原子结果：

| Reach | Grasp | Lift | Transport | Place | 总成功 | saturation / anomaly replan |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0/5 | 5/5 | 5/5 | 2/5 | 4/5 | 16/25 | 0 / 0 |

相对 E006 的 30-epoch D，grasp/lift 保持 `5/5、5/5`，transport 从 `0/5` 到 `2/5`，place 从
`0/5` 到 `4/5`，总计从 `10/25` 到 `16/25`；reach 仍为 `0/5`。这说明长期低权重事件监督确实
积累了后期技能，但没有直接修复几乎无事件曝光的 reach。

三组 20 unseen 完整闭环：

| 控制组 | 完整成功 | Reach/Grasp/Lift/Transport/Place | 平均完成技能数 | 失败分布 | max spread / 最低最新权重 | saturation / anomaly | wall time |
| --- | ---: | --- | ---: | --- | --- | ---: | ---: |
| newest-only | 0/20 | 1/0/0/0/0 | 0.05 | reach 18、grasp 1、anomaly exhausted 1 | 0 / — | 1 / 0 | 459.3 s |
| ensemble-only | 0/20 | 5/4/3/1/0 | 0.65 | reach 15、grasp 1、lift 1、transport 2、release 1 | 1.8667 / 0.5333 | 0 / 0 | 453.6 s |
| ensemble+replan | 0/20 | 5/4/3/1/0 | 0.65 | reach 15、grasp 1、lift 1、transport 2、release 1 | 1.8667 / 0.5333 | 0 / 0 | 488.1 s |

ensemble 两组的平均最终 TCP→物体距离为 0.0990 m，newest-only 为 0.1561 m；平均目标 XY
距离分别为 0.1547 m 和 0.1667 m。ensemble 只有 1 条进入 release/place，最终仍抓取且归类为
`release_failed`，因此 place 尝试的仍抓取率是 1/1；完整 place 仍为 0/20。

ensemble 两组均未触发异常。去掉 wall time 后，两组 20 条 Episode 记录及 summary 逐字段相等，
所以这批 seed 只能证明“重规划预算没有改变无异常轨迹”，不能证明其恢复收益。newest-only 的
seed 10017 出现一次 tracking-correction saturation；由于重规划预算为 0，按协议记录为
`replan_anomaly_exhausted`，不是系统崩溃。四个正式评估共无 OOM、Vulkan、Traceback 或进程错误，
产物和日志均已持久化并校验 SHA256。

**Conclusion:**

100-epoch 固定低事件权重保住 grasp/lift，并把独立 transport/place 提升到可见水平，验证了 E006
选择低回归方案的合理性。temporal ensemble 对正式完整闭环有明确净收益：reach 从 1/20 提高到
5/20，平均完成技能数从 0.05 提高到 0.65，并消除了 newest-only 中的跟踪饱和失败，因此继续作为
默认控制方式。`ensemble+replan` 在无异常时与 ensemble-only 行为相同，仍保留为默认安全能力。

第一版预定训练和消融已经完成，但行为成功门槛没有通过：完整成功仍为 0/20，主要瓶颈仍是 reach
泛化；少量进入后期的轨迹还暴露 transport 和 release 失败。下一轮应优先增加 reach/transport 的
有效数据和监督覆盖，再独立验证 release；不通过手写 stable-grasp、release-hold 或 settle 状态机
掩盖学习问题，也不据此修改 Qwen、Action Expert 或 joint-space Action 契约。

## E008 — Qwen Layer 12 空间表示、Reach 与五技能组合诊断

**Date:** 2026-08-28

**Status:** completed

**Experiment:**

按“先做便宜且归因清楚的 probe，再进入闭环”的顺序，先用冻结 Qwen Layer 12/24 visual token
训练相同线性位置 probe；再用严格相同 seed、训练预算和控制协议比较 Layer 24 Control、Layer 12
和带 GT 相对几何 token 的 Oracle Reach；最后保持 E007 数据、损失、优化器和闭环协议不变，只把
Qwen Context 从 Layer 24 改为 Layer 12，完成 100-epoch 五技能联合训练、25 个独立原子 Episode
和 20 个 unseen 完整 Episode。

**Goal:**

判断 E007 的 Reach 瓶颈是否来自 Qwen 最终层丢失精细位置，以及 Layer 12 的几何优势能否在不使用
GT token/GT 几何的情况下转化为原子技能组合和完整任务收益。离线 probe 和 validation loss 只用于
归因；最终标准仍为统一闭环成功率和阶段深度。

**Config:**

- Code revision: `source-tree-sha256:3ee22910df7912f99143c4bea02ba14fd92316443fd79a8e17b89841e4768bbd`
- Dataset: `trusted-v0.4-event-recovery-220`，220 trajectories / 48922 steps，dataset SHA256
  `bc024b6b...39407`，manifest SHA256 `43f131cc...f477f`
- Spatial probe: 同一次冻结 Qwen 前向读取 Layer 12/24；相同初始化的线性 probe；test 2048 samples /
  136 unique windows；GT 只用于选择包含方块的 external-camera 粗 visual token 和评价坐标，不作为
  VLA 输入
- Reach diagnosis: 三组均为 30 epochs、4096 samples/epoch、batch 64、1920 optimizer steps；
  Control 使用 Layer 24，Treatment 使用 Layer 12，Oracle 额外提供 TCP→物体相对几何；闭环 seeds
  `10000–10004`，每条最多 100 policy steps
- Combination train: 与 E007 相同的 100 epochs、4096 samples/epoch、batch 64、6400 optimizer
  steps、seed 42、AdamW `1e-4`、warmup 1000 / cosine 30000、固定 `lambda_event=0.25`、技能采样
  权重 `1.5/1/1/1.5/2`；唯一模型变量为 `qwen_context_layer=12`，无 GT token 或 GT 几何
- Combination evaluation: seeds `10000–10004` 的 25 个原子 Episode；unseen seeds
  `10000–10019` 的 20 个完整 Episode；10-step Flow、每 4 步重规划、temporal ensemble
  `rho=0.5`、max anomaly replans 3
- Hardware: 单张 RTX 4090 24GB
- Artifacts: `artifacts/qwen-spatial-probe-20260828/`、`artifacts/oracle-reach-20260828/`、
  `artifacts/layer12-combination-20260828/`；组合 best SHA256
  `a542076f...41ad6`

**Result:**

空间 probe 的 test 结果：

| Context | median visual-token error | median world XY | p90 world XY | within 1 token |
| --- | ---: | ---: | ---: | ---: |
| Layer 12 | **0.1587** | **0.0253 m** | **0.0388 m** | **100.0%** |
| Layer 24 | 0.8369 | 0.1245 m | 0.2439 m | 60.7% |
| 最近粗 token 中心 | 0.2015 | 0.0302 m | 0.0658 m | 100.0% |

Layer 12 的 median world-XY error 只有 Layer 24 的 `20.4%`，明确证明中层位置更容易线性解码；
但 `0.0253 m` 仍高于预设 `0.02 m` Reach 门槛，而且相对最近粗 token 中心的 ratio 为 `0.839`，
没有通过预设 `<=0.8` 的 sub-token 增益门槛。因此该 probe 支持“Layer 12 明显优于 Layer 24”，
但不单独宣称“已具备精确 Reach”。

Reach-only 正式闭环：

| 模式 | best epoch / val loss | Reach 成功 | 平均最终 TCP→物体距离 | 最小距离 |
| --- | --- | ---: | ---: | ---: |
| Layer 24 Control | 30 / 0.018651 | 1/5 | 0.098048 m | 0.037208 m |
| Layer 12 | 30 / **0.017167** | **2/5** | **0.062761 m** | 0.037239 m |
| Oracle Geometry | 28 / 0.019052 | **4/5** | **0.039665 m** | 0.033607 m |

Layer 12 相对 Layer 24 把平均最终距离降低约 36%，并多成功 1 条；Oracle 达到 4/5，说明显式几何
仍提供明显上界。Layer 12 有真实闭环收益，但没有达到 Oracle，剩余问题包括目标 token 寻址、双相机
对齐、视觉到关节动作映射和闭环误差修正，而不只是“有没有位置”。

五技能联合训练完整结束，100 行 epoch 指标连续，无 NaN/Inf、OOM 或训练错误：

| Checkpoint | Context | Epoch | Val total | Val base | Val event |
| --- | --- | ---: | ---: | ---: | ---: |
| E007 best | Layer 24 | 98 | 0.030852 | **0.023178** | 0.032872 |
| E008 best | Layer 12 | 98 | **0.027532** | 0.023730 | **0.016258** |
| E008 latest | Layer 12 | 100 | 0.027561 | 0.021881 | 0.023105 |

Layer 12 best total loss 比 E007 低约 10.8%，event loss 低约 50.5%，但 base loss 略高约 2.4%。
这说明优化收益主要集中在 contact、grasp/pickup、release/place 等关键事件，不代表 Reach/Transport
连续空间运动同步改善。

独立原子结果：

| Context | Reach | Grasp | Lift | Transport | Place | 总成功 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Layer 24 E007 | 0/5 | 5/5 | 5/5 | **2/5** | 4/5 | 16/25 |
| Layer 12 E008 | 0/5 | 5/5 | 5/5 | 1/5 | **5/5** | 16/25 |

总数完全不变；Layer 12 只是把 1 个成功从 Transport 移到 Place。Layer 12 原子 Reach 5 条均跑满
100 步，平均最终 TCP→物体距离 `0.1015 m`。这与 Reach-only 的 2/5 形成受控反差，说明联合训练、
聚合 checkpoint 选择或技能梯度竞争会抵消中层几何收益。

20 unseen 完整闭环：

| Context | 完整成功 | Reach/Grasp/Lift/Transport/Place | 平均完成技能数 | 平均最终 TCP 距离 | saturation / anomaly |
| --- | ---: | --- | ---: | ---: | ---: |
| Layer 24 E007 | 0/20 | 5/4/3/1/0 | 0.65 | 0.0990 m | 0 / 0 |
| Layer 12 E008 | 0/20 | **9/3/2/0/0** | **0.70** | 0.0989 m | 0 / 0 |

Layer 12 失败分布为 reach 11、grasp 6、lift 1、transport 2。9 条通过 Reach 后只有 3 条完成
Grasp，条件成功率为 `3/9=33%`；Layer 24 对应为 `4/5=80%`。Layer 12 的 Reach `9/20` 对
Layer 24 的 `5/20` 只有方向性改善，双侧两比例近似检验 `p≈0.185`，在当前样本数下不能宣称稳健
显著。20 条完整成功仍为 0，Wilson 95% 上界为 `0.1611`。

独立 Grasp/Lift/Place 分别为 `5/5、5/5、5/5`，但完整链中只有 `3/20、2/20、0/20`，说明
“专家准备的干净前置状态”与“策略自己完成前序技能后的状态”存在明显 handoff distribution
mismatch：Reach Predicate 可以刚过阈值，但 TCP 姿态、速度、夹爪开度或视觉相对位置未必形成
高质量 Grasp 输入。两条完成 Lift 的轨迹又全部在 Transport 失败；Place 虽独立 5/5，却从未被
完整链触达。

所有 25+20 Episode 都正常完成，tracking saturation、anomaly replan 和系统错误均为 0；因此
差异归因于策略表示、训练目标和跨技能状态分布，而不是控制器或仿真故障。原子与完整 summary 均为
`complete=true`，使用同一 checkpoint SHA256 和 dataset SHA256。

**Conclusion:**

“Layer 24 丢失精细位置”得到部分支持：Layer 12 的线性位置误差显著更低，Reach-only 和完整任务
前段也呈一致的正向变化。但“直接把完整 Context 换成 Layer 12 就能解决组合”被否定：原子总数
仍为 16/25、完整仍为 0/20，完整链只是把一部分瓶颈从 Reach 移到 Grasp，Transport 进一步退化。

现有结果不支持把纯 Layer 12 升级为默认架构，也不支持继续盲目增加相同训练 epoch。第一版原子
技能组合目标尚未达到；当前最清楚的问题是聚合 loss/checkpoint 对事件技能的偏置，以及
Reach→Grasp、Lift→Transport 的交接状态质量。

**Next step:**

1. 不重训，先对已有 epoch 10/20/.../100 periodic checkpoint 只跑 Reach/Transport sweep，判断
   最优能力是否出现在不同 epoch，从而区分 checkpoint 选择问题与持续梯度干扰。
2. 新增 Reach→Grasp handoff probe：在首次满足 Reach Predicate 时记录 TCP 相对位姿/速度、夹爪
   开度、物体运动和双相机位置，并与专家准备的 Grasp 初态比较。
3. 只有前两项确认“Layer 12 几何有益但语义寻址或技能切换退化”后，再正式比较 Layer 24 semantic
   Key + 同 token Layer 12 geometry Value；不把 Oracle GT 几何带入生产模型，也不增加手写任务
   语义状态机掩盖学习问题。

## E009 — Layer 12 periodic checkpoint 的 Reach/Transport sweep

**Date:** 2026-08-29

**Status:** completed

**Experiment:**

不重新训练，固定 E008 的 Layer 12 架构、数据、评估器、控制协议和 sampling seed，只改变五技能联合
训练 checkpoint。对 epoch 10/20/.../100 的 10 个 periodic 权重和 epoch 98 best 共 11 个候选，
先用少量全新 seed 筛选，再在相互独立的全新 seed 上确认 Reach/Transport 候选，判断聚合 validation
loss 是否选错行为 checkpoint，或两项空间技能是否在训练过程中出现不同最佳 epoch。

**Goal:**

区分三个互斥的主要解释：

1. **Checkpoint selection 问题：**至少一个较早 checkpoint 同时不劣于 epoch 98，并明确改善 Reach
   或 Transport；聚合 total loss 没有选到闭环最优权重。
2. **技能目标冲突：**Reach 与 Transport 的最佳 checkpoint 分属不同 epoch，且不存在单个候选同时
   保住两者；共享动作头在训练过程中沿 Pareto frontier 移动。
3. **持续训练/表示问题：**全部 periodic checkpoint 都与 epoch 98 同样失败；问题不是选择时点，
   应继续做 handoff probe、数据分布或 Layer 24 Key / Layer 12 Value 架构诊断。

本实验只做 checkpoint 归因，不用完整任务成功率反向挑权重，也不把 screening seed 的结果当作正式
提升证据。

**Frozen config:**

- Dataset: `trusted-v0.4-event-recovery-220`，dataset SHA256 `bc024b6b...39407`
- Model: E008 Layer 12 五技能联合训练，`qwen_context_layer=12`，无 GT token/GT 几何
- Evaluation: `RobotVLAPickCubeToRegion-v1` 独立原子评估，技能仅 `reach transport`
- Policy budget: 每技能每 seed 最多 100 policy steps；10-step Flow；每 4 步重规划
- Control: temporal ensemble on，`recency_decay=0.5`，max anomaly replans 3
- Sampling: base seed `42424`；原子 sampling seed 只由 base seed、环境 seed 和 skill 派生，因此同一
  skill/seed 在所有 checkpoint 间形成严格配对比较
- Hardware: 单张 RTX 4090 24GB；所有 checkpoint 顺序运行，不并行共享 GPU/SAPIEN

**Candidates:**

| Candidate | Weight | Epoch val total | Val base | Val event |
| --- | --- | ---: | ---: | ---: |
| e010 | `step-00000640.pt` | 0.135433 | 0.108168 | 0.106300 |
| e020 | `step-00001280.pt` | 0.089931 | 0.071453 | 0.078659 |
| e030 | `step-00001920.pt` | 0.078140 | 0.059799 | 0.068425 |
| e040 | `step-00002560.pt` | 0.072662 | 0.057543 | 0.061529 |
| e050 | `step-00003200.pt` | 0.050218 | 0.041362 | 0.034399 |
| e060 | `step-00003840.pt` | 0.041849 | 0.032991 | 0.033336 |
| e070 | `step-00004480.pt` | 0.039321 | 0.031331 | 0.033295 |
| e080 | `step-00005120.pt` | 0.037129 | 0.030900 | 0.026218 |
| e090 | `step-00005760.pt` | 0.035522 | 0.028538 | 0.029539 |
| e098-best | `best.pt` | **0.027532** | 0.023730 | **0.016258** |
| e100 | `step-00006400.pt` | 0.027561 | **0.021881** | 0.023105 |

`latest.pt` 与 step 6400 表示同一 epoch 100 状态，不作为第 12 个重复候选。validation 指标只用于
解释训练轨迹，不能参与闭环候选排名。

**Stage A — screening:**

- Seeds: `10020–10022`，均不在 Dataset manifest，也没有用于 E008 原子或完整评估
- Candidates: 全部 11 个
- Skills: Reach + Transport
- Episodes: `11 × 2 × 3 = 66`
- 每个 checkpoint 只加载一次，顺序执行 6 个 Episode

对每个技能分别按以下稳定顺序排名：

1. 成功数降序；
2. Predicate 超额残差均值升序；
3. 平均 policy steps 升序；
4. epoch 升序，仅作为完全相同时的确定性 tie-break。

Predicate 超额残差与正式 Outcome 阈值严格一致：

```text
Reach residual     = max(final_tcp_to_object_distance_m - 0.04, 0)
Transport residual = max(final_object_to_goal_xy_distance_m - 0.04, 0)
```

任何出现 inference/controller/action-safety error、tracking saturation 或 anomaly replan 的候选不直接
删除，但单独标记为非行为可比，不能依靠距离 tie-break 晋级。

**Stage B — independent confirmation:**

- Seeds: `10023–10032`，10 个全新、非 Dataset seed，与 Stage A 和 E008 都不重叠
- Candidates: Stage A Reach Top-2、Transport Top-2 与 epoch 98 best 的并集，最多 5 个
- 所有入围候选都同时跑 Reach 和 Transport，不能只跑它在 screening 中表现好的技能
- Episodes upper bound: `5 × 2 × 10 = 100`
- 正式报告每技能成功数/Wilson 95% 区间、Predicate residual、平均 steps、逐 seed 配对胜负，以及
  saturation/anomaly/system failure

只有同时满足下列条件的单一 checkpoint 才能标记为“值得进入下一级完整评估的候选”，不能直接替换
默认 best：

1. confirmation 中 Reach 和 Transport 成功数都不低于 epoch 98；
2. 至少一个技能比 epoch 98 多成功 `>=2/10`；
3. 被改善技能的平均 Predicate residual 至少降低 20%；
4. 两个技能均无系统错误、tracking saturation 或 anomaly replan。

如果 Reach/Transport 分别由不同 epoch 获胜，而没有单个候选满足上述条件，则结论为技能/checkpoint
冲突，不把两个 checkpoint 组合成运行时手写路由。若没有候选超过 epoch 98，则排除“只选错 best”
作为主要解释，直接进入 handoff probe。

**Promotion guardrail:**

若 Stage B 产生单一候选，再额外用相同协议对候选与 epoch 98 在 seeds `10023–10027` 上配对评估
Grasp/Lift/Place，共 `2 × 3 × 5 = 30` Episode。候选最多允许每个 guardrail 技能相对 epoch 98
回退 1/5；通过后才能另立实验运行 20 unseen 完整闭环。本 E009 不用完整任务结果继续调候选，避免
把正式 test seed 变成超参数选择集。

**Execution and artifacts:**

- Remote root: `/home/ubuntu/robot-vla-runs/e009-layer12-checkpoint-sweep/`
- Layout: `screen/<candidate>/`、`confirm/<candidate>/`、可选 `guardrail/<candidate-or-best>/`
- 每个输出目录沿用 `evaluate_atomic_maniskill` 的 `experiment.json`、增量 `episodes.jsonl`、
  `summary.json` 和 `--resume` 身份校验
- 运行前生成只读 sweep manifest，记录 candidate label、epoch、绝对 checkpoint 路径、validation
  指标、seed 集合、源码树 revision 和统一控制配置；每个候选的 `experiment.json` 另存 checkpoint
  SHA256、dataset SHA256 和模型契约；聚合器只读取各目录的 experiment/episodes/summary，不重新推理
- Python orchestrator 让单 checkpoint 子进程失败时停止；恢复时只对未完成目录使用 `--resume`
- 结果完成后同步 manifest、JSONL、summary 和日志到本机 artifacts；不重复同步 11 个已有 checkpoint

**Budget:**

- 必跑：Stage A 66 Episode
- 最大 confirmation：100 Episode
- 条件 guardrail：30 Episode
- 基于 E008 Reach/Transport 每条约 7.5 秒，Stage A + 最大 Stage B 预计约 20–30 分钟；guardrail
  通常更快。相比新的 100-epoch 训练，成本和归因复杂度都显著更低。

**Results:**

Stage A 完成 66/66 Episodes。Reach Top-2 为 `e100, e098-best`，Transport Top-2 为
`e090, e100`；加上强制 anchor 后，Stage B 候选并集为 `e090, e098-best, e100`。Stage A 的
3-seed 结果只用于筛选；例如 e100 Transport 在 screening 为 `2/3`，在独立 confirmation 中只有
`2/10`，验证了不能把 screening 当正式结论。

Stage B 完成 60/60 Episodes：

| Candidate | Reach | Reach residual | Transport | Transport residual | System/saturation/anomaly |
| --- | ---: | ---: | ---: | ---: | ---: |
| e090 | 0/10 | 0.06536 m | 7/10 | 0.02603 m | 0 / 0 / 0 |
| e098-best | 0/10 | 0.06932 m | **7/10** | **0.00857 m** | 0 / 0 / 0 |
| e100 | **3/10** | **0.03062 m** | 2/10 | 0.05288 m | 0 / 0 / 0 |

相对 epoch 98，e100 Reach 多成功 3 条，逐 seed 为 3 win/0 loss，residual 降低 55.8%；但
Transport 少成功 5 条，逐 seed 为 1 win/6 loss，residual 增至 6.17 倍。e090 与 anchor 的成功数
完全相同，Reach/Transport residual 比率分别为 0.943 和 3.038，没有达到 20% 改善门槛。

因此 `promotion_candidate_labels=[]`，没有候选满足“Reach/Transport 均不回退”的首要条件，条件
Grasp/Lift/Place guardrail 不运行。原始 126 行 JSONL 经独立重算，skill/seed 身份、成功数、residual
和配对 win/loss 与聚合文件一致；Stage B 无系统错误、tracking saturation 或 anomaly replan。

**Analysis:**

E009 支持预注册解释 2：不同技能的最佳 checkpoint 分离。epoch 100 恢复部分 Reach，却破坏
Transport；epoch 98/90 保住 Transport，却没有 Reach 成功。结果排除“存在一个明显更好的单一
periodic checkpoint，只是 total validation loss 选错了”的简单解释，也不支持部署按技能手写切换
checkpoint。validation total loss 在 epoch 98/100 之间只差约 `0.000029`，但行为折中明显变化，
后续 checkpoint 诊断必须保留 per-skill 闭环指标。

本实验没有直接测量梯度或 handoff 状态，10-seed Wilson 区间也较宽，因此不能单独把机制唯一归因于
共享动作头梯度干扰。下一步按 D024 做 Reach→Grasp handoff probe；只有继续确认 Layer 12 几何有效
但语义寻址/交接退化后，才比较 Layer 24 semantic Key + Layer 12 geometry Value。

完整技术报告、冻结 manifest、聚合 JSON 和逐 Episode 结果见
[`docs/results/e009/`](results/e009/README.md)。

**Implementation and verification:**

现有原子评估 CLI 已支持所需 checkpoint、skills、seed、Layer 12 契约、增量 summary 和 resume，不
修改模型、训练器或 ManiSkill evaluator。只新增一个薄的 sweep orchestrator/aggregator，用于生成
manifest、顺序启动现有 CLI、计算上述 residual/ranking 和选择 Stage B 候选；排名逻辑需要纯函数
单元测试，禁止把 validation loss 混入行为排序。目标测试共 10 项通过，新增文件通过 Ruff；实验
源码树 revision 为 `source-tree-sha256:aeb1c1647de4eadf838838c0abf4e5c1c517d0d871e2f33c75d2b8b288a351f1`。

## E010 — Layer 12 五技能梯度冲突与 base/event 归因 probe

**Date:** 2026-08-29

**Status:** planned

**Experiment:**

不更新参数、不运行闭环，只在 E009 的 epoch 90、98、100 checkpoint 上恢复相同 Layer 12
Adapter/Action Expert 权重，对严格配对的 per-skill Action Chunk batch 计算训练目标梯度。保存逐模块
精确 Gram matrix，再由 Gram 独立重算 cosine、范数和负冲突比例。Discovery 使用 train split 的五技能
全矩阵；Confirmation 使用独立 val split，只对 Reach/Transport 分解 base gradient、加权 event
gradient 和二者之和。

本 probe 只回答“训练目标的局部一阶更新方向是否冲突，以及冲突位于哪里”，不把离线 loss 或梯度
cosine 当作闭环成功率，也不在本实验中直接实现多动作头。

**Goal:**

区分以下架构分支：

1. **仅输出头冲突：**共享 Expert 大部分层的 Reach/Transport 梯度相容，但 `velocity_head` 稳定负
   冲突；下一候选才是共享 trunk + 轻量多输出头。
2. **后层 Expert 冲突：**冲突集中在最后 4 个 Expert block；下一候选是共享前层 + skill adapter/
   后层分支，而不是只拆最终 Linear。
3. **广泛共享表示冲突：**至少一半 Expert block 稳定负冲突；先比较 PCGrad/CAGrad/GradNorm，再考虑
   更重的 MoE，不能假设多个输出头足够。
4. **Adapter/Context 投影冲突：**Adapter 本身稳定负冲突；优先继续 Layer 24 semantic Key + Layer 12
   geometry Value，而不是只修改动作头。
5. **base/event 目标内部冲突：**同一技能的 base gradient 与 `0.25 × event gradient` 稳定相反；
   优先处理 loss 权重、事件采样或分阶段优化。
6. **未确认梯度冲突：**独立 val split 不复现负方向；不据此增加多头，继续 Reach→Grasp handoff
   状态分布 probe。

**Frozen model and loss:**

- Dataset: `trusted-v0.4-event-recovery-220`；只读取既有 train/val split
- Qwen: `Qwen/Qwen3.5-2B` fixed revision，完全冻结，Context hidden state 固定 Layer 12
- Checkpoints: `e090=step-00005760.pt`、`e098-best=best.pt`、`e100=step-00006400.pt`
- Trainable boundary: 只测 QwenVLAAdapter + StandaloneActionExpert；Qwen 参数不得出现 gradient
- Objective: `base_loss + 0.25 × event_loss`，event 只使用前 4 个实际执行步的 critical mask
- Precision: 与正式训练一致的 BF16 autocast；梯度 Gram 在 FP32 中累计
- Flow target: 每个 stage/repeat 使用固定 seed；相同 repeat 的各技能和各 checkpoint 共享相同 Flow
  time/noise，减少采样噪声
- Optimizer: 不构造 step，不裁剪梯度，不写回参数；probe 前后 checkpoint 参数 SHA256 必须不变

**Module groups:**

1. `adapter`
2. `state_encoder`
3. `action_encoder`
4. `block_00` … `block_15`
5. `final_norm`
6. `velocity_head`
7. `all_trainable`，由以上互斥分组的 Gram 求和

分组必须完整覆盖所有且仅覆盖可训练参数；任一参数缺失、重复或产生 `None/NaN/Inf` gradient 时停止。

**Stage A — train discovery:**

- Checkpoints: e090、e098-best、e100
- Split: train
- Skills: reach、grasp、lift、transport、place
- Repeats: 8
- Batch: 每技能每 repeat 8 个不同 trajectory，共每技能 64 个不同 train trajectory
- Sampling: 同一 repeat 五技能使用相同 8 个 trajectory，只在各自 skill 段内随机选一个 timestep；
  checkpoint 间完全复用 sample identity
- Work: `3 checkpoints × 8 repeats × 5 skill gradients = 120` 次 gradient pass
- Output: 每 repeat 保存五技能逐模块 `5×5` Gram、loss/base/event、有效/critical steps、gradient norm、
  trajectory/timestep 和 Flow seed

**Stage B — independent confirmation and decomposition:**

- Checkpoints: e098-best、e100
- Split: val，与 Stage A trajectory 不重叠
- Skills: Reach、Transport
- Repeats: 5
- Batch: 每技能每 repeat 4 个不同 trajectory，共覆盖 20 个不同 val trajectory
- Sampling: 与 Stage A 相同的同场景/同 Flow seed 配对原则
- 对每个 skill 从同一次 forward 分别求：`base gradient`、`0.25 × event gradient`，并用向量和得到
  `total gradient`；零 event norm 记录为不可计算，不把 cosine 人工填成 0
- Work: `2 checkpoints × 5 repeats × 2 skills = 20` 次 forward、40 次 component backward
- Output: 每 repeat 保存 `reach_base/reach_event/reach_total/transport_base/transport_event/
  transport_total` 的逐模块 `6×6` Gram

**Pre-registered interpretation thresholds:**

同一 checkpoint 的 Reach/Transport `all_trainable` total gradient 只有同时满足以下条件，才标记为
“确认训练梯度冲突”：

1. Stage A median cosine `<= -0.10` 且至少 `6/8` repeats 为负；
2. Stage B median cosine `<= -0.10` 且至少 `4/5` repeats 为负。

定位规则在确认冲突后应用：

- **Output-head localized:** `velocity_head` 在 Stage B 满足 `<=-0.10`、至少 4/5 为负，同时 Adapter
  不满足，且满足同一门槛的 Expert blocks 少于 4 个。
- **Late-Expert localized:** blocks 12–15 至少 3 个满足门槛，blocks 0–11 满足者少于 4 个。
- **Broad Expert:** 16 个 blocks 至少 8 个满足门槛。
- **Adapter conflict:** Adapter 满足 Stage B 门槛。
- **Within-skill objective conflict:** Reach 或 Transport 的 base/event cosine 在 Stage B median
  `<=-0.10` 且至少 4/5 为负。

若 discovery 达标但 confirmation 不达标，结论为“训练样本上的不稳定冲突信号”，不能据此修改架构。
所有 median 同时报告 IQR；负比例报告 Wilson 95% 区间。模块定位是诊断标签，不是统计显著性声明。

**Artifacts and recovery:**

- Remote root: `/home/ubuntu/robot-vla-runs/e010-skill-gradient-conflict/`
- `probe-manifest.json`: checkpoint SHA256、dataset identity、源码 revision、样本计划、Flow seed 和冻结配置
- `measurements.jsonl`: 每完成一个 checkpoint/stage/repeat 原子追加一行；身份重复或缺失时拒绝汇总
- `probe-summary.json`: 由 raw Gram 聚合的 per-pair/per-group median、IQR、负比例和 Wilson 区间
- 重跑时 manifest 必须逐字段一致，只跳过已完成 measurement identity
- GitHub 发布 manifest、raw JSONL、summary 和分析；不上传 Qwen、dataset 或 checkpoint 权重

**Promotion boundary:**

E010 不产生可部署 checkpoint。即使确认梯度冲突，也只能决定下一项对照实验是多头、skill adapter、
PCGrad/CAGrad 或 Context Key/Value；任何架构晋升仍需重新训练，并通过独立原子和完整闭环门槛。

**Planned implementation:**

新增一个不写回参数的诊断模块和薄 CLI，复用现有 Dataset、Collator、Layer 12 policy factory 与严格
checkpoint loader。纯函数测试覆盖参数分组、Gram/cosine、零范数、重复恢复身份、Wilson 区间、
阈值判定和 sample plan；GPU runner 只负责固定样本 forward/backward 与原子落盘。

## 实验模板

复制下面的模板创建新条目，并使用递增编号 `E001`、`E002`……。完成后同步更新上方索引。

```markdown
## EXXX — 实验简称

**Date:** YYYY-MM-DD

**Status:** planned | running | completed | failed | stopped

**Experiment:**

一句话描述实验变量，以及相对 Baseline 改变了什么。

**Goal:**

要验证的单一问题或假设，以及预先定义的判断标准。

**Config:**

- Code version:
- Dataset / data version:
- Task / environment:
- Baseline:
- Model:
- Train / inference config:
- Evaluation protocol:
- Hardware:
- Seeds:
- Artifacts:

**Result:**

记录指标、样本数或 Episode 数、均值与离散程度、失败类型和资源消耗。没有结果时明确说明原因。

**Conclusion:**

说明假设是否得到支持、相对 Baseline 的变化是否可信，以及结论的适用范围和限制。

**Next step:**

只记录由本次证据直接支持的下一步验证或工程动作。
```
