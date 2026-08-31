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
| E010 | 2026-08-29 | Layer 12 五技能梯度冲突与 base/event 归因 probe | completed | e098-best/e100 均未通过两阶段负冲突门槛；五技能 train median 全为正，Reach/Transport event gradient 为零不可识别，不支持直接多头或 PCGrad |
| E011 | 2026-08-29 | RTC Action Chunk Transition 受控评估 | completed | RTC 在共同 Reach seed 上推进到 Transport，但把完整闭环 Reach 从 temporal 的 6/10 降到 2/10，未通过 promotion；默认继续 temporal ensemble，不进入 Stage B |
| E012 | 2026-08-29 | Local DAgger Boundary Recovery | failed | E012a 正式 RG pool 通过 20 条 gate；GL 为 10/100 eligible，低于固定 gate 20；按预注册停止，未创建 D1、未启动 E012b |

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
- Artifacts: `run://stage1-v0.1`、`run://stage1-v0.1-rollout`

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
- Artifacts: `run://stage1-v0.2-data100`、`run://stage1-v0.2-data100-rollout`

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
- Artifacts: `run://stage1-v0.2-data100-safety-diagnostics`、
  `run://stage1-v0.2-data100-saturated-control`

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
- Artifacts: `run://stage1-v0.2-data100-atomic-seeds5-v3`

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
- Control artifacts: `run://ablation-v0.3-control-data100-e30*`
- Treatment artifacts: `run://ablation-v0.3-recovery-data120-e30*`

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
- A–C artifacts: `run://ablation-v0.4-*` 与
  `run://e006-*`
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

- Remote root: `run://e009-layer12-checkpoint-sweep/`
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

**Status:** completed

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

- Remote root: `run://e010-skill-gradient-conflict/`
- `probe-manifest.json`: checkpoint SHA256、dataset identity、源码 revision、样本计划、Flow seed 和冻结配置
- `measurements.jsonl`: 每完成一个 checkpoint/stage/repeat 原子追加一行；身份重复或缺失时拒绝汇总
- `probe-summary.json`: 由 raw Gram 聚合的 per-pair/per-group median、IQR、负比例和 Wilson 区间
- 重跑时 manifest 必须逐字段一致，只跳过已完成 measurement identity
- GitHub 发布 manifest、raw JSONL、summary 和分析；不上传 Qwen、dataset 或 checkpoint 权重

**Promotion boundary:**

E010 不产生可部署 checkpoint。即使确认梯度冲突，也只能决定下一项对照实验是多头、skill adapter、
PCGrad/CAGrad 或 Context Key/Value；任何架构晋升仍需重新训练，并通过独立原子和完整闭环门槛。

**Implementation:**

新增一个不写回参数的诊断模块和薄 CLI，复用现有 Dataset、Collator、Layer 12 policy factory 与严格
checkpoint loader。纯函数测试覆盖参数分组、Gram/cosine、零范数、重复恢复身份、Wilson 区间、
阈值判定和 sample plan；GPU runner 只负责固定样本 forward/backward 与原子落盘。

**Result:**

- 34/34 measurement unit 完成：3 checkpoints × 8 train discovery + 2 checkpoints × 5 val
  confirmation；覆盖 64 个不同 train trajectory 和 20 个不同 val trajectory，跨 split 零重叠。
- Reach/Transport `all_trainable` train median cosine 为 e090 `+0.421`、e098-best `+0.164`、
  e100 `+0.173`；e098-best/e100 的独立 val 为 `-0.094（3/5 负）` 与 `+0.441（0/5 负）`。
  两个 confirmation checkpoint 都没有同时通过 `<=-0.10` 与负 repeat 计数门槛。
- 三个 checkpoint 的五技能 10 个 pair median 全部为正。e098-best 只有 Block 15 在 val 上单独达到
  module 负冲突门槛；Velocity head 为 `-0.120、3/5`，未达到计数门槛；e100 Velocity head 为
  `+0.480、0/5`。由于 overall 未确认，不激活 output-head/late/broad/adapter 标签。
- Reach/Transport 在 Stage A/B 的前 4 个执行步均没有 critical event；其加权 event gradient 范数为
  0，base/event cosine 按协议为 `null`。Grasp/Lift/Place 的 train critical steps 分别为 `29/10/5`，
  因此本次没有识别 Grasp/Place event 对 Reach/Transport 的间接共享参数影响。
- e098-best/e100 的 Grasp 中位梯度范数分别为 Transport 的 `2.67×/2.71×`，提示实际更新贡献或
  采样/尺度失衡是后续候选，但本结果没有证明其导致闭环行为交换。
- 三个 checkpoint 的 Adapter/Expert 参数 SHA256 前后完全一致。独立标准库脚本从 raw Gram 复算
  1320 个 summary row，`all_trainable` 求和最大差为 `0.0`、统计最大差为 `1.11e-16`。

完整技术报告、manifest、34 行 raw Gram、summary 和独立验证见
[E010 results](results/e010/README.md)。

**Conclusion:**

预注册的“稳定训练梯度冲突”假设未得到支持。E009 的 Reach/Transport checkpoint 行为交换不能由
当前 checkpoint 上稳定的 per-batch 负 cosine 直接解释；现有证据不支持立即实现多动作头、后层
分支或 PCGrad/CAGrad。零 event norm 只表示 Reach/Transport 的 within-skill base/event 归因不可
识别，不能推广为 event loss 全局无影响。

**Next step:**

下一项便宜 probe 直接计算 e098-best→e100 的真实参数位移 `Δθ`，并按模块评估
`g_reach·Δθ/g_transport·Δθ`；同时构造 guaranteed-critical 的 Grasp/Place event batch 和技能边界
batch。只有实际位移稳定呈现帮助一个技能、伤害另一个技能并定位到 Head/后层时，才进入多头或后层
分支 A/B；否则继续 handoff 状态分布归因。

## E011 — RTC Action Chunk Transition 受控评估

**Date:** 2026-08-29

**Status:** completed

**Experiment:**

冻结同一模型、Checkpoint、数据、环境与 paired environment/sampling seeds，对比 `newest-only`、
`temporal-ensemble` 与 `rtc`。RTC 只改变 inference-time Flow sampling，不修改 Qwen、Adapter、Action
Expert、训练目标、Action/Observation、Controller、Predicate 或 anomaly-replan 契约。

**Goal:**

验证当前 temporal ensemble 的历史 proposal 是否在 Reach→Grasp、Lift→Transport 边界降低闭环
reactivity，以及 RTC 是否能在保持 Chunk continuity 的同时提高条件交接率，而不是只让运动变慢或变平滑。

**Config:**

- Action shape: normalized model action `[B,16,8]`
- `execute_steps=4`，20 Hz 控制，5 Hz Replan
- Flow: 与冻结模型相同的 10-step Euler，最终才 clamp 到 `[-1,1]`
- RTC target: 上一次实际生成并用于执行的 guided clean Chunk，执行 4 步后以 `prev[4:16]` 对齐
  `new[0:12]`；reference 始终 detach
- RTC clean endpoint: 本项目 `x_t=t*noise+(1-t)*action`、`v=noise-action`，因此
  `A_clean_hat=x_t-t*v`
- RTC velocity guidance: 使用论文 Eq.(2) 的 VJP；由于本项目积分方向与论文相反，在项目 velocity 上
  使用相反 guidance 符号，仍由原 Euler `dt=-1/n` 生效
- Eq.(5): 本实验不模拟异步推理延迟，因此 `d=0`；执行 horizon `s=4`；slots `0..11` 按公式
  由强到弱衰减，slots `12..15` 权重为 0；同一 slot 权重 broadcast 到全部 8 个 action dims
- `rtc_max_guidance_weight=10.0`；每步只 clip guidance coefficient，不额外 clamp 中间 action state
- 首次 Replan、显式 reset、inference/safety/tracking anomaly 后没有 previous reference，退化为普通 Flow
- RTC 诊断使用同一 Context、Proprio 和 initial Flow noise 生成 paired raw/RTC Chunk；只有 RTC Chunk 执行
- CLI: `--inference-strategy newest-only|temporal-ensemble|rtc`，旧 temporal bool 只保留兼容
- Stage A: 10 个全新 paired seeds，三组共 30 个完整 Episode
- Stage B: Stage A 无明显回归后，至少 20 个新的 paired seeds 比较 temporal 与 RTC
- E011 按受控协议显式使用 `--qwen-context-layer 12` 和对应的同一固定 Checkpoint；全局 CLI 默认仍保持
  Layer 24，避免本实验在未显式选择时改变其他评测；`experiment.json` 固定实际 Context Layer、
  Checkpoint/Dataset SHA256 和 source revision

**Pre-registered run identity:**

- Layer 12 Checkpoint: `a542076f291e29b68e3d28930b15c40396d511a44eb358c2eaeb4e113c041ad6`
- Dataset: `bc024b6b9c566ca9500945fb6ac262bf657bee713d8a5816229bdc8478139407`
- Flow sampling seed base: `42424`
- 非正式 smoke environment seed: `19999`；不进入 Stage A 统计或调参证据
- Stage A environment seeds: `20000..20009`；在运行结果前冻结，且与 dataset seeds `0..239`、既有
  evaluation seeds `0..10032` 无重合
- Stage A groups: `newest-only`、`temporal-ensemble`、`rtc`；除 inference strategy 外配置相同
- Atomic guardrail seeds: `20010..20014`；每组对 Reach/Grasp/Lift/Transport/Place 各运行 5 个 Episode，
  只用于检查 RTC 是否破坏既有原子能力，不替代 30 个完整 Stage A Episode 的 handoff 判断
- RTC: `execution_horizon=4`、`max_guidance_weight=10.0`、`rtc-eq5-soft-mask`
- Remote output root: `run://e011-rtc/`

**Diagnostics:**

每个 Replan 记录策略、sampling seed、RTC 配置、previous availability、12-step overlap、Eq.(5) slot
weights、逐 denoising step coefficient、paired raw/previous disagreement、RTC/previous prefix disagreement、
RTC correction、future correction、TCP 距离/速度、joint velocity、gripper target、阶段前后状态和 temporal
proposal spread。Episode 同时记录五技能完成 control step，聚合 `P(Grasp|Reach)`、
`P(Transport|Lift)`、平均完成技能数及 Reach/Grasp/Lift/Transport 的阶段耗时。

**Result:**

工程实现、静态检查和 GPU 测试完成：RTX 4090 / PyTorch 2.11.0+cu128 环境实际覆盖 Flow VJP、
Runtime history/reset/anomaly、无历史等价普通 Flow、CUDA BF16 autocast 与 ManiSkill runtime；完整测试
套件 `208 passed`（另有 12 条 ManiSkill 依赖弃用警告）。随后完成非正式 smoke、30 个 Stage A 完整
Episode 和 75 个 atomic guardrail Episode；正式 JSONL 没有 NaN/Inf。

- Stage A newest-only：Reach/Grasp/Lift/Transport/Place 为 `2/1/1/1/0`（各自分母 10），平均完成
  技能数 `0.5`，完整成功 `0/10`。
- Stage A temporal-ensemble：`6/4/0/0/0`，`P(Grasp|Reach)=4/6=66.7%`，平均完成技能数
  `1.0`，完整成功 `0/10`。
- Stage A RTC：`2/2/2/2/0`，`P(Grasp|Reach)=2/2`、`P(Transport|Lift)=2/2`，平均完成技能数
  `0.8`，完整成功 `0/10`。
- temporal 与 RTC 共同 Reach 的只有 seeds `20005、20009`；共同支持集上两者 Grasp 均为 `2/2`。
  RTC 在这两条随后均完成 Lift/Transport，但另有 4 条 temporal-only Reach、0 条 RTC-only Reach；
  Reach 的 paired exact McNemar 双侧 `p=0.125`。
- Atomic newest/temporal/RTC 总成功分别为 `17/25、17/25、18/25`；三组 Grasp/Lift/Place 均为
  `5/5`，Transport 为 `2/5、2/5、3/5`，Reach 均为 `0/5`。RTC 没有破坏预注册的强原子能力。
- RTC Stage A 共 750 个 Replan，其中 740 个有 previous overlap；prefix/future mean correction 为
  `0.00684/0.00430`，p95 为 `0.01190/0.00607`。没有全局数值锁死，但少量技能边界出现最大 2.0
  的修正；当前 future 聚合未单独隔离零权重 slots `12..15`。
- Stage A system error 和 Action safety rejection 三组均为 0；temporal/RTC 的 anomaly 与 tracking
  saturation 均为 0。RTC 平均每完整 Episode `53.19 s`，temporal 为 `30.99 s`；当前 paired
  raw/RTC 诊断实现约慢 `1.72x`。

完整配对表、阶段耗时、前 80 步普通运动、边界 disagreement/correction、atomic 和正式文件 SHA256
见 [E011 results](results/e011/README.md)。正式原始资产保留在
`run://e011-rtc/`；evaluation source revision 为
`source-tree-sha256:adfce370c438d460eb4178be9af38ee5741554741a3c99f6acd8485847244dec`。

**Conclusion:**

**事实：** RTC 在两个共同 Reach seed 上保住 Grasp 并比 temporal 多完成 Lift/Transport，atomic
Grasp/Lift/Place 没有回归；但 RTC 把 temporal 的 Reach `6/10` 降到 `2/10`，平均完成技能数也从
`1.0` 降到 `0.8`。因此 RTC 不满足“保持 reactivity 且改善 handoff”的 promotion 标准。

**解释：** temporal 历史 proposal 并非只有旧计划污染，在初始 Reach 阶段还提供了有用的轨迹稳定性；
RTC continuity 可能帮助少数已接近/抓住物体的后段转换，但当前配置的代价是 Reach 覆盖下降。最强
边界冲突信号出现在 Grasp→Lift，而不是稳定出现在 Reach→Grasp 或 Lift→Transport。

**尚不能得出的结论：** `2/2` 的 RTC 条件率存在 Reach 选择效应，不能写成 RTC 已改善
Reach→Grasp；两条后段成功也不足以证明 RTC 修复 Lift→Transport。10 个 seed 不能唯一归因 guidance
weight、Flow、Layer 或控制速度，也不能排除所有其它 RTC 配置。

**Next step:**

不 promotion RTC，不运行 Stage B；默认继续 temporal ensemble。不得使用 seeds `20000..20009` 调
guidance 后复用为确认集。按预案把主要精力转向 Reach→Grasp、Grasp→Lift 失败附近的 handoff 状态分布
与 Local DAgger recovery；RTC 保留为显式实验/诊断策略。如未来重开 RTC，先在新 smoke seed 上确定数值
范围，再使用全新的 paired seeds，并单独记录 free-tail slots `12..15` 和同 slot raw/post disagreement。

## E012 — Local DAgger Boundary Recovery

**Date:** 2026-08-29

**Status:** stopped

**Result:**

E012a 已在冻结的 source revision、E011 Layer 12 checkpoint、D0 与正式 temporal-ensemble 配置下完整扫描
两个预注册 100-seed pool。Reach→Grasp 得到 `31 accepted / 69 rejected / 0 error`，并按固定 mid-rank
risk rule 选出 `14 high + 6 low`；Grasp→Lift 只有 `10 accepted / 90 rejected / 0 error`，低于每个
boundary 固定要求的 20 条 eligible trajectory。GL 的 90 条拒绝中，71 条在目标 boundary 前终止或
截断、16 条达到时间上限，其余 3 条分别来自 Expert 未完成完整任务、MPlib 路径失败和 snapshot gate。

独立核验确认 200 个 candidate record 的 seed/source/checkpoint/status 一致；所有 accepted trajectory 均
通过 paired clean Expert、snapshot、完整五技能 trajectory audit 与 Expert-only supervision 检查，每条
具有 49 个完整 16-step anchors。由于 GL 容量 gate 失败，未创建 D1、未运行 D0+D1 union audit，也未
启动 replay/DAgger 训练。该结果只证明采集链路可运行且当前 GL 候选容量不足，不构成行为收益或 Chunk
uncertainty 机制证据。完整技术报告见
[E012a portable report](results/e012/report.html)，机器可读证据见
[collection summary](results/e012/collection_summary.json)、
[boundary distribution](results/e012/boundary_distribution.json) 与
[independent validation](results/e012/independent_validation.json)。

### E012a Grasp→Lift failure postmortem

在不改写上述 formal 结果的前提下，后续对 71 条 boundary-before rejection 和 16 条 takeover 后
time-limit rejection 做了同 environment seed、checkpoint、D0、sampling seed 与 temporal 配置的无干预
diagnostic replay。`87/87` 条重放均与 formal status/reason 完全一致，未出现 outcome mismatch 或
engineering error。由此需要修正一个重要解释口径：`10/100` 是最终 eligible trajectory 比例，不是 Policy
到达 stable-Grasp boundary 的比例；formal 数据能够证明至少 `29/100` 条到达了该 boundary，其中包括
10 条 accepted、16 条 takeover 后 time-limit 和 3 条只能在 boundary 后发生的其他 rejection。

71 条未形成 stable-Grasp boundary 的 Policy rejection 可以互斥分解为：57 条从未完成 Reach、12 条完成
Reach 但从未出现 raw grasp、2 条出现 transient raw grasp 后在形成两步稳定 Grasp 前丢失；三类全部在
Policy 的 300-step time limit 结束。16 条 takeover 后 time-limit 按失败 Action 当时的 Expert commanded
phase 分为 Lift motion `3`、Transport motion `8`、Lower motion `3`、Release/settle `2`。commanded phase
只表示 collector 正在调用的控制阶段，不证明对应 Predicate 已完成。

该分解说明候选容量不足主要表现为 Policy roll-in 在完成 Reach 前耗尽 300-step 预算，同时还有一个可
单独检验的 Expert recovery budget 截断现象；它本身不识别 Action Chunk 冲突或任何控制参数的
因果效应。10 条 accepted
survivor 的 takeover→完整成功为 `119 / 134 / 151` steps（min/median/max），只能用于设计有限预算的
探索性 counterfactual，不能外推到 90 条 rejected episode。完整报告见
[GL failure diagnostic report](results/e012/gl-failure-report/report.html)，canonical 机器可读证据见
[GL failure decomposition](results/e012/gl_failure_decomposition.json)。在新的 fixed-budget counterfactual
通过前，不重开 formal GL pool、不创建 D1，也不启动 E012b。

### E012a segmented-budget counterfactual

随后在不回写旧 formal record、且禁止任何新 trajectory 进入训练的前提下，冻结并验证了
`segmented-300-180-480`：Policy roll-in 最多执行 300 个真实 environment actions，Expert takeover 后最多
执行 180 个 actions，episode 的 environment hard limit 为 480；完整成功必须严格早于 environment
truncation。paired clean Expert 继续使用历史 `legacy-300`，trajectory 仍来自同一个
`CollectionSession`，正式五技能 success/audit 契约不变。Counterfactual source identity 为
`source-tree-sha256:e73675abed4b0d14117c98df0c790f358a13b2eb8429db034826a9e2fe3ca5d9`，checkpoint 与
D0 继续使用上述冻结 SHA。

固定三条 diagnostic smoke（历史 seed `30111 / 30193 / 30171`，禁止用于训练）先验证 accepted
control、early timeout 与 late timeout，结果 `3/3` prefix metadata aligned、`0` engineering error；
accepted control seed `30111` 精确复现 legacy 的 takeover step 154、
39 次 Policy replan 和总计 294 actions。之后对 16 条 canonical legacy post-takeover TimeLimit seed 做
受控重放，`16/16 candidate executions finalized`、`16/16` prefix metadata aligned、`0` prefix
mismatch、`0` engineering error、`0` 条命中 480-action hard deadline。互斥结果为：

```text
recovered full eligible:                              5
Expert completed but snapshot/paired gate failed:     1
Expert recovery budget exhausted at 180 actions:      4
other behavioral rejection:                           6
```

5 条完整 eligible 的 Expert action 使用量为 `117 / 121 / 122 / 136 / 161`，median 122、mean 131.4。
Seed `30181` 完成五技能、environment success 与 terminated，但 snapshot immediate RGB mean absolute
error 为 `0.0051167806`，高于 `0.002` 门槛，因而不能计入 eligible 或 D1。其余 10 条行为未完成轨迹
均只达到 `max_completed_skill_count=2`，从未形成 Lift/Transport predicate；其中 4 条精确命中 Expert
180-action cap，另 6 条在 157–178 Expert actions 内已结束 nominal recovery sequence。所有 rejected 的
`phase_at_failure=expert_release_settle` 只表示 commanded callsite，不能解释为 Place 已完成。

该 counterfactual 证明分段预算能恢复部分旧 TimeLimit，并说明 480 hard limit 已不再是当前直接瓶颈；
它不证明 Local DAgger 训练收益，也不识别 Action Chunk 冲突的因果作用。`5/16` 只是旧 TimeLimit 子群的
条件恢复率。正式 pool 的容量规划点估计必须使用原 100-seed 分母：`(10 legacy eligible + 5 recovered) /
100 = 15%`。在该 evidence 下，达到固定 gate `eligible >= 20` 的规划概率为：

| Fixed pool | Fixed-p Binomial | Jeffreys Beta-binomial predictive |
|---:|---:|---:|
| 120 | 34.1% | 40.1% |
| 160 | 84.1% | 74.4% |
| 180 | 94.6% | 84.5% |
| 200 | 98.5% | 90.9% |
| 220 | 99.7% | 94.7% |
| 240 | 99.9% | 96.9% |

Jeffreys posterior-predictive 达到 95% 的最小 pool size 为 223；因此原计划 120 条不再满足低风险容量
目标，200 是明确接受约 9.1% predictive gate-failure risk 的折中，240 是按 20 条批量向上越过 223 的
严格低风险候选。该概率仍依赖新旧 seed 可交换、旧 10 条 accepted 在 amended protocol 下保持 eligible
等假设，不是成功保证；正式 pool size 必须在 rollout 前一次性冻结，禁止根据中途 eligible 数量自适应
续采。staged publish、canonical record manifest、`30200..<31000` seed registry guard、Qwen/runtime identity
与 exact-prefix resume 已实现并通过当时针对性测试。以上“等待 owner 冻结 pool / 尚未创建 D1”是
2026-08-29 capacity-planning artifact 的 point-in-time 状态；后续 owner-frozen amended formal pool、D1 与
repeat-1 训练的结果见下方独立小节，不能反向改写本段 counterfactual 的条件恢复率或用途边界。

### E012a frozen D0 compatibility preflight

在恢复新源码 smoke 前，当前 audit code 对同一 frozen D0 重算出 `bb066…`，与历史正式 identity
`bc024…` 不一致，因此 smoke 在任何 rollout/output 写入前受控停止。逐文件取证排除了数据漂移：raw
manifest 仍为 `43f131…`，220 个 NPZ 全部存在且逐字节 receipt 匹配，audit report 为 `b7ab50…`，
proprio stats raw/semantic hash 分别为 `fdad911… / e0638a…`，checkpoint 内嵌 stats 与其解析语义 exact
equal。两个 dataset hash 的 payload 大小只差 `4,400 = 220 × 20` bytes，恰好对应每行新增的
`,"local_dagger":null`。

因此 amended runner 增加了 E012-specific、只读、版本化的双 projection verifier，而没有修改 D0 或全局
audit contract。真实 3.24GB D0 的实现验证结果为：历史 projection 精确恢复 `bc024…`，当前 projection
精确得到 `bb066…`，220 个 NPZ 的 sorted file/SHA aggregate 为 `b6ea7f…`，trajectory/step/split/stats
全部匹配。verifier 同时拒绝 raw `local_dagger`、NPZ byte drift、missing/extra/duplicate、path escape、
symlink/hardlink、manifest/count/step/stats 漂移，并把 receipt 纳入 exact-resume experiment identity；失败
发生在 CUDA 初始化和 formal output 创建前。Compact 机器可读证据见
[D0 compatibility audit](results/e012/d0_compatibility_audit.json)。该 preflight 只解决 D0 身份解释与内容
完整性，不构成 amended GL 效果证据，也不授权把任何 smoke/counterfactual trajectory 用于训练。

完整技术报告见
[segmented-budget report](results/e012/segmented-budget-report/report.html)，可执行 notebook 见
[reproduce.ipynb](results/e012/segmented-budget-report/reproduce.ipynb)，canonical compact analysis 见
[analysis.json](results/e012/segmented-budget-counterfactual/analysis.json)。Smoke/counterfactual wrapper 均冻结
`trajectory_usage=forbidden as training data` 与 `successful_npz_may_enter_d1=false`；后续 D1 builder 只允许
消费正式 canonical `accepted + selected` record，禁止扫描这些诊断目录中的 NPZ。发布包的时间线、证据
范围与历史 artifact 字段说明见 [E012 results index](results/e012/README.md)。

### E012 repeat-1 formal training 与 checkpoint selection

后续 amended collection、D1 build、D0+D1 union audit 和训练前身份门禁均通过；这些新 formal 产物没有
回写 legacy `10/100` GL 结论，也没有消费 smoke / counterfactual trajectory。`pi_replay[1]` 与
`pi_dagger[1]` 都从同一 pi0 权重独立初始化，各完成 30 epochs、122,880 examples、1,920 optimizer steps。
正式 paired verifier 通过，SHA-256 为
`1fa9b11c184e06618bf573a984572276d0d72d83cd910e88a5f022fc47f589ff`。Replay exposure 仅为
`base_d0=122880`；Dagger exposure 精确为 `base_d0=98310`、`dagger_reach_grasp=12290`、
`dagger_grasp_lift=12280`。D1 只来自 Expert-only boundary-local anchors，D0 offset 为 null，D1 offset
为整数 `0..48`。

Checkpoint validation 在单 GPU 上严格顺序评估 pi0、replay epoch 10/20/30 和 Dagger epoch 10/20/30；
每个模型使用 full-chain seeds `31000..31019` 的 20 条 Episode，以及 atomic seeds `31020..31024 ×`
Reach/Grasp/Lift/Transport/Place 的 25 条 Episode，共 14 组输出、315 Episodes。Checkpoint、D0、evaluation
code、执行配置、environment/Flow seed pairing 和 seed registry 均通过独立 identity audit；错误扫描、
system error 与 Action safety rejection 都为 0。

预注册 checkpoint-selection 结果如下；`net` 是 candidate wins 减 pi0 wins：

| Arm / epoch | Full Reach / Grasp / Lift net | Atomic Place net | Mean completed skills Δ | 排除原因 |
|---|---:|---:|---:|---|
| replay e10 | `-2 / -2 / -2` | `-2` | `-0.30` | Reach、atomic Place |
| replay e20 | `+8 / +9 / +6` | `-3` | `+1.25` | atomic Place |
| replay e30 | `-6 / -2 / -2` | `0` | `-0.50` | Reach |
| Dagger e10 | `-7 / -3 / -2` | `0` | `-0.60` | Reach；新增 3 anomaly episodes / 7 replans |
| Dagger e20 | `+1 / +1 / 0` | `-4` | `+0.10` | atomic Place |
| Dagger e30 | `+2 / +3 / +4` | `-2` | `+0.45` | atomic Place；新增 anomaly 与 tracking saturation |

两臂的正式 `select_e012_checkpoint` 均返回 `selection_gate_passed=false`、`selected=null`、
`eligible_ranking=[]`；selection receipt SHA-256 分别为 replay
`0fdc195552e742b017d71da57974a98ff626c018d289a1f5ffa891f74e1ee838`、Dagger
`84ba2e7435438d65ebd1fb926cda21fbf69f92093021cf68cc4f0abc579586f6`。因此不存在合法 Stage A pair：不运行
Stage A，不消费 `32000..` seeds，不训练 repeat 2，不运行 Stage B 或消费 `32100..` seeds；需要 selected
pair 的 matched-state diagnostics 同样记为 not run。不得用 replay e20 / Dagger e30 的正向 validation 信号
绕过 guardrail，也不得临时加入 epoch 24、Dagger epoch 9 或 `best.pt`。六个候选的 full success 都为
`0/20`，当前证据不支持 Local DAgger 改善或 Chunk uncertainty 下降。

Replay 曾从不可变 epoch-24 full-state checkpoint 严格恢复。恢复后的 trainer 结构、examples、optimizer
steps、source exposure、boundary offsets 与冻结 identity 一致，但 CUDA loss/gradient 数值轨迹不再 bitwise
reproducible；最终可重复性声明必须保留该限制。脱敏 compact evidence 与独立 verifier 见
[checkpoint-validation summary](results/e012/checkpoint-validation/summary.json)，人类可读结果见
[repeat-1 technical report](results/e012/training-repeat-1-report/report.html)。

**Experiment:**

冻结 E011 的 Layer 12 模型、Observation/Action、Flow/Event loss、Controller、Predicate 与正式
`temporal-ensemble` 执行协议，只改变训练状态分布：从 frozen policy 自己到达的 Reach→Grasp、
Grasp→Lift 边界开始，由同一 CollectionSession 内的可信 Expert 接管并完成整条 Pick-and-Place；训练只
监督 takeover 后的 boundary-local Expert Action Chunk。主因果比较不是历史模型与新模型，而是从同一
checkpoint 初始化、使用相同额外 optimizer steps 的 `pi_dagger` 与 `pi_replay`。

**Goal:**

回答两个分开的主问题：

1. policy-induced boundary state 与相同 environment seed 的 clean Expert boundary state 是否存在
   可测量的分布差异，Local DAgger 是否提高无条件 Grasp/Lift 完成数和平均完成技能数；
2. 若闭环 handoff 改善，同一 held-out boundary state 上的 Action Chunk disagreement / Flow-seed
   prediction variance 是否同时下降。

成功率提高而 matched-state disagreement 不下降时，只能结论为 recovery supervision 有效，不能写成
“Local DAgger 消除了 Chunk mode uncertainty”。本实验也不试图证明完整的
`representation error -> covariate shift -> uncertainty -> disagreement -> temporal pathology` 因果链。

### 冻结资产与非变量

- Collection policy / historical reference `pi_0`：E011 正式 Layer 12 checkpoint，SHA256
  `a542076f291e29b68e3d28930b15c40396d511a44eb358c2eaeb4e113c041ad6`
- Base Dataset `D0`：SHA256
  `bc024b6b9c566ca9500945fb6ac262bf657bee713d8a5816229bdc8478139407`
- Qwen：`Qwen/Qwen3.5-2B` revision
  `15852e8c16360a2fea060d615a32b45270f8a8fc`
- Context hidden state：Layer 12；Qwen 继续完全冻结
- Action：`[16,8] = delta_q[7] + gripper_target[1]`
- 20 Hz control、每 4 步 Replan、10-step Flow Euler
- 正式执行：`temporal-ensemble`、`recency_decay=0.5`；`newest-only` 只作诊断
- 保持 Adapter、Action Expert、Flow Matching objective、`lambda_event=0.25`、skill weights、
  Action normalization、Controller、Safety、Predicate 和 language instruction 不变
- 不加入 skill/recovery/boundary token，不把 simulator oracle state 输入 VLA，不使用 runtime Predicate
  切 checkpoint，不加入 RTC、Critic、RL 或任务语义状态机
- 训练和推理固定使用 `D0/proprio_stats.json`；`D1` 上重算的 stats 只作为 distribution diagnostic，
  不改变输入归一化或 checkpoint metadata
- E012 实现完成时，必须在第一次 smoke 前把 source revision、ManiSkill/MPlib/CUDA 版本和远端输出根目录
  写入 run manifest；此后不可在同一实验身份下静默改变

### 两阶段结构

E012 拆成两个有独立停止条件的阶段：

```text
E012a: boundary diagnosis + live takeover collection feasibility
E012b: pi_replay vs pi_dagger controlled training/evaluation
```

E012a 没有通过数据 provenance、连续性和 audit gate 时，不启动 E012b。工程测试、snapshot round-trip
或 smoke 通过只证明采集链路可运行，不是行为收益。

### Legacy formal E012a 预注册 seed 分区

Environment seeds 固定为：

```text
legacy E012a smoke only:        29990..29999
E012a Reach->Grasp collection: 30000..30099
E012a Grasp->Lift collection: 30100..30199
checkpoint closed-loop val:   31000..31019
checkpoint atomic val:        31020..31024
Stage A paired full-chain:     32000..32019
Stage A atomic guardrail:      32020..32024
Stage B paired full-chain:     32100..32129
```

两个 boundary 使用 disjoint roll-in seed pool；不能从一次 Reach takeover 后的 Expert trajectory 再抽取
“policy-induced Grasp boundary”。若任一 collection pool 不能产生足够数据，E012a 记录失败并在新增 seed
前修订 manifest；不能临时挑 seed、重复 reset 到满意状态或挪用 validation/test seeds。

后续 amendment diagnostic smoke 复用了历史 GL records 的 `30111 / 30193 / 30171` 做无训练用途的确定性
重放；这三条不属于新增 formal collection、validation 或 training exposure。amended formal pool 必须另用
一次性冻结、与本表及 D0 全部零重叠的新连续 seed range。

训练 repeat 使用 paired seeds：

```text
repeat 1: pi_replay=12012, pi_dagger=12012
repeat 2: pi_replay=22012, pi_dagger=22012
```

同一 repeat 两组使用相同初始化权重、optimizer/scheduler 配置、batch 数和随机 seed。Rollout 的 Flow
sampling seed 在同一 environment seed、同一策略比较中严格配对；具体 derivation 写入 manifest，并与
environment seed 分开记录。Collection、训练、checkpoint validation、Stage A 和 Stage B 身份不得重叠。

### E012a：paired boundary diagnosis

Collection policy 固定为 `pi_0 + temporal-ensemble`，采集期间不更新。对每个出现目标 boundary 的
collection seed，运行相同随机化的 clean Expert rollout，按 environment seed 配对比较；不能把既有
Dataset seeds 的 Expert boundary 与 `30000+` policy boundary 直接作为主比较。

每个 boundary 的预注册 oracle diagnostics 为：

- TCP-object relative XYZ、XY error、Z error、TCP linear speed
- 7D joint velocity 的 RMS/max
- gripper opening、command target
- object linear/angular velocity
- object-TCP relative pose、contact/grasp stability
- arm 与 gripper 分开的 newest-vs-history / pairwise proposal disagreement

这些 feature 只用于采集分层和离线分析，不进入 VLA Observation。统计单位是
`episode/environment-seed/boundary`：先在单个 boundary window 内聚合，再跨 Episode 做 paired effect
size/bootstrap；不得把每个 control step 或 Replan 当作独立样本伪增大样本量。

### E012a：候选池和确定性选样

两个 seed pool 都必须完整扫描后再选择训练轨迹，不能达到 20 条后提前停止。每个发生指定 boundary 的
rollout 都记录 candidate、takeover attempted、Expert plan、full recovery、正式 audit 和拒绝原因。

Near-failure 不再按“最终失败/成功”人工分类，而使用以下确定性 boundary risk score。对每个 policy
boundary 与相同 seed 的 clean Expert boundary，先计算 state deviation；速度/不稳定性使用越大越危险的
标量，Chunk disagreement 使用 policy boundary 原值：

```text
Reach->Grasp components:
  TCP-object XY error
  abs(relative-Z_policy - relative-Z_expert)
  TCP speed
  joint velocity RMS
  gripper-opening deviation from paired Expert
  arm mean-pairwise disagreement
  gripper mean-pairwise disagreement

Grasp->Lift components:
  object-TCP relative-position deviation from paired Expert
  object linear speed
  object angular speed
  joint velocity RMS
  gripper-opening deviation from paired Expert
  contact/grasp instability (stable=0, unstable=1)
  arm mean-pairwise disagreement
  gripper mean-pairwise disagreement
```

每个 component 在同 boundary、完整恢复且 audit 通过的 candidate pool 内转成 mid-rank empirical
percentile `[0,1]`，risk score 是所有必需 component percentile 的算术平均。缺失、NaN/Inf 或无法与
clean Expert 配对的 candidate 不参与选择并记录拒绝原因；不做运行后插值或人工补分。同分按
`environment_seed` 升序。该公式和 feature 单位还必须逐字段写入正式 manifest 和单元测试。完整候选池
结束后，每个 boundary 确定性选择：

```text
14 条最高 risk 且完整恢复/audit 通过的 trajectory
 6 条最低 risk 且完整恢复/audit 通过的 trajectory
```

得到 20 Reach→Grasp + 20 Grasp→Lift。Expert 无法恢复或 full success audit 失败的轨迹不进入训练，
但必须保留诊断记录；正式结论限于 Expert 可恢复状态。Risk score 只决定 exposure 分层，不作为 VLA
输入或训练权重。

### E012a：同一 CollectionSession 的 live takeover 契约

正式 Local DAgger trajectory 必须从 episode reset 起完整记录 Policy roll-in。第一次检测到指定
Predicate `False -> True` 后，不 reset、不恢复到标准原子技能起点、不重建 env，也不拼接第二条轨迹；
只把下一条 Action 的 producer 从 Policy 切为 Expert。Expert 随后一直执行到完整五技能
Pick-and-Place 成功。

时间索引冻结为：若 `action[t-1]` 执行后得到 `observation[t]`，并在该 observation 检测到 boundary，
则：

```text
boundary_detection_step = t
expert_takeover_step     = t
action[t]                = first Expert action
```

因此 `action[0:t]` 是 Policy roll-in，`action[t:T]` 是 Expert；前者保留为状态生成和诊断 provenance，
但永远不能成为 BC/Flow target。当前 `ActionChunkDataset` 是单时刻双图+proprio 输入，并不读取历史帧；
真正进入 takeover 首个训练样本的是 policy roll-in 产生的 `observation[t]`，更早的 RGB/proprio 只保留
在完整轨迹中供审计和离线诊断。

takeover 只能切换 Action producer，必须保持 episode id、elapsed step、q/dq、object pose/velocity、
controller state/target、当前 commanded q、camera、RNG 和 tracker 连续。禁止像 reset-only collector 那样
把 `previous_command_q` 重置为 actual q，否则第一个 Expert label 会包含人工 controller correction。

Collector 同时维护 Replan snapshot ring，至少保存 crossing 前一个 Replan 和 crossing Replan。Snapshot
bundle 包含 actors/articulations、controller state/target、主/episode RNG、wrapper elapsed step、q/dq、
object pose/velocity、progress tracker、policy control step、current commanded q、observation hashes、camera
calibration 和 collection policy identity。它用于 round-trip 验证与诊断；第一版 accepted trajectory 仍以
同一 live session 为准，不从 restored branch 拼接数据。

ManiSkill 3.0.1 的 `get_state_dict()` 只保存 actor/articulation state，当前
`pd_joint_delta_pos` controller 的公开 `get_state()` 还是空字典。因此 E012a snapshot 额外冻结每个子
controller 的 `_step/_start_qpos/_target_qpos`；round-trip 在完整 live trajectory 已封存后才在同一个 env
执行，不让 restored branch 产生任何 accepted training transition。PhysX contact impulse cache 不在该
state dict 中，刚 restore 后读取的 raw contact force 以及依赖该 cache 的 `is_grasped` 只作诊断、不作为
瞬时 pass 条件；contact gate 固定为两次 restore 后执行同一个下一步 hold Action，所得 pairwise contact
force 最大绝对差不超过 `1e-3 N`，且两次 replay 的 `is_grasped` 都必须重新等于 source boundary 的值。
source boundary contact 仍原样保存在 snapshot evidence。其余 round-trip 阈值在正式 seed 前冻结为：

```text
external/wrist segmentation: exact
external/wrist RGB:          max uint8 error <= 1, all-pixel mean abs error <= 0.002
physical proprio:            exact
Predicate numeric fields:    restore max abs error <= 1e-7
is_grasped:                  same-next-hold 后与 source bool exact
camera calibration numeric:  max abs error <= 1e-6; version exact
controller target:           max abs error <= 1e-7
same-next-Action sim state:   max abs error <= 2e-4
same-next-Action contact:     max abs error <= 1e-3 N
main/episode RNG probe:       exact
```

snapshot summary 仍报告 source-vs-restored raw contact 差和完整 observation hash 是否逐字节相等，禁止因其
不参与上述 PhysX-aware pass rule 而隐藏。

### Local DAgger trajectory 与监督契约

第一版不修改正式 audit 对“最终 Transition 成功、五技能完整且连续覆盖”的要求。新 trajectory 增加：

```text
source = dagger_reach_grasp | dagger_grasp_lift
rollin_seed
rollin_policy_checkpoint_sha256
boundary_type
boundary_detection_step
expert_takeover_step
training_window_start
training_window_end
expert_supervision_mask[T]
action_source[T] = policy | expert
expert_recovery_success
```

`expert_takeover_step` 是第一条由 Expert 生成并执行的 Action 索引；`training_window_end` 为 exclusive。
第一版冻结：

```text
training_window_start = expert_takeover_step
training_window_end   = min(expert_takeover_step + 64, num_steps)
```

64 个 control steps 等于 3.2 秒、16 个 Replan、4 个 Action Horizon。Expert 仍继续到完整任务成功，但
window 后的普通 Expert Transport/Place 不进入 Local DAgger exposure。若窗口内不足一个完整 16-step
Expert Chunk，该 candidate 拒绝进入训练并记录原因。

一致性规则为：

```text
first_true(expert_supervision_mask) == expert_takeover_step
expert_supervision_mask[t] == (action_source[t] == expert)
expert_supervision_mask[expert_takeover_step:T].all()
```

Local DAgger 缺少任一监督/provenance 字段时 fail closed；不得默认为整条轨迹可监督。Clean Expert
trajectory 依据显式 `source=clean_expert` 保持全部有效 Action 可监督。

当前 `sample_flow_training_target` 会把 `action_mask=False` 的 noisy Action/target 置零，理论上可以支持
slot-level supervision；但第一版不依赖这一实现细节，仍禁止 mixed-source 或 partial-window Chunk，以
保持采样契约简单、可审计并给 Policy target 留下零进入路径。对长度 `H=16` 的 Local DAgger sample
anchor `u`，必须同时满足：

```text
training_window_start <= u
u + H <= training_window_end
expert_supervision_mask[u:u+H].all()
```

takeover 时刻的 policy-induced RGB/proprio 可以成为样本 Observation；更早 roll-in 帧不属于当前模型
输入。Policy action 不得直接参与 loss，也不作为同一 noisy Chunk 中仅被 mask 掉的 slot。Dataset/Collator
仍显式传递 supervision mask，训练 loss 使用
`valid_action_mask & expert_supervision_mask & training_window_mask`，并按实际监督元素归一化。Sampler 和
loss 两层都必须断言 Local DAgger target 是 Expert-only。

### E012a 进入训练的 gate

只有以下条件全部满足才创建 `D1` 并进入 E012b：

- 两个 boundary 各有 20 条按固定 risk rule 选择的完整成功轨迹
- 正式 audit 在不放宽 success/五技能契约的前提下通过
- takeover 前后 controller/state/timestamp 连续性测试通过
- Local DAgger provenance validator 和 fail-closed 测试通过
- 所有可采样 Action Chunk 均完整位于 Expert mask 和 `[training_window_start, training_window_end)`
- 改写 takeover 前 Policy action 后，采样 target 与训练 loss 完全不变
- snapshot smoke 的 RGB、proprio、Predicate/contact、controller target、next-step dynamics 和 RNG
  round-trip 通过；snapshot 不用于掩盖 live takeover 失败
- collection summary 完整报告 candidate/attempt/recovery/audit/accepted 数量和所有拒绝原因
- collection、D0、validation、Stage A/B seeds 零重叠

没有检测到大的 paired boundary effect 不自动使数据无效，但必须在 E012b 前报告；此时 E012b 是对弱
covariate-shift 假设的否证性测试，不能预写正向机制结论。

### E012b：replay-controlled 训练

三类模型定义为：

```text
pi_0:
  冻结历史 checkpoint，不训练，只作历史参考

pi_replay[r]:
  从 pi_0 只初始化 Adapter/Expert 权重
  optimizer/scheduler/scaler/RNG 全部重置
  在 D0 上继续固定 K steps

pi_dagger[r]:
  从同一 pi_0 只初始化 Adapter/Expert 权重
  optimizer/scheduler/scaler/RNG 全部重置
  在 D0 + Dagger-RG + Dagger-GL 上继续相同 K steps
```

主因果比较是 `pi_dagger[r] vs pi_replay[r]`；`pi_dagger vs pi_0` 只能作为历史效果描述，不能隔离额外
训练步数。训练入口必须新增独立 `--init-checkpoint` 语义；现有 `--resume` 会恢复 optimizer、scheduler、
trainer step 和 RNG，只用于中断续训，禁止用作 E012 warm start。

第一轮固定：

```text
K = 1920 optimizer steps
effective batch size = 64
4096 samples/epoch-equivalent
30 epoch-equivalents
source exposure for pi_dagger = base-D0 0.80 / RG 0.10 / GL 0.10
source exposure for pi_replay = base-D0 1.00
```

两组保持相同 learning rate、1000-step warmup、cosine schedule、event loss、skill weights、gradient
accumulation 和 checkpoint cadence。`D1` 是逻辑训练混合，不重新拟合 ProprioStats；D0 validation split
保持不变，40 条 Local DAgger 只进入训练 source。Sampler 先选 source，再选 task/episode/timestep；实际
每 epoch 输出 `source x skill x boundary-offset` exposure，不能只记录配置概率。RG/GL 使用 disjoint
roll-in seeds，防止相关 trajectory 被双重加权。

### Checkpoint 预选择

每组固定保存 10/20/30 epoch-equivalent checkpoint。`31000..31019` full-chain validation 和
`31020..31024` atomic validation 只用于 checkpoint 选择，永远不进入 Stage A/B 或 DAgger collection。
同一模型按以下 lexicographic rule 选一个 checkpoint：

1. 排除新增 system error、Action safety rejection 或 tracking/anomaly regression 的候选；
2. 排除 atomic Grasp/Lift/Place 相对同 seed `pi_0` 出现净成功数回归的候选；
3. 排除 full-chain Reach 相对 `pi_0` 下降超过 1/20 的候选；
4. 最大化无条件 `Lift/20`；
5. 最大化无条件 `Grasp/20`；
6. 最大化 mean completed skills；
7. 再以固定 D0 validation total loss、较早 checkpoint 依次 tie-break。

不得用 Stage A/B seeds 选 checkpoint。Checkpoint validation 是模型选择数据，不能合并进最终效果样本。

### 正式评估与统计

所有模型在同一 environment/Flow seeds 上 paired 运行，主协议为 temporal ensemble。每个模型/repeat
同时报告：

- `Reach/N、Grasp/N、Lift/N、Transport/N、Place/N、Full/N`
- `P(Grasp|Reach)、P(Lift|Grasp)、P(Transport|Lift)` 与 Wilson interval
- mean completed skills 和阶段耗时
- 共同 predecessor seeds 上的 handoff、paired wins/losses 和 exact paired test
- system error、Action safety rejection、tracking saturation、anomaly replan
- atomic Reach/Grasp/Lift/Transport/Place guardrail

条件率不能单独作为 promotion 证据。E011 的 RTC `2/2` 已证明 predecessor 选择效应会制造虚高条件率；
因此必须并列给出无条件分子、共同支持集和 predecessor 本身的 paired 变化。两个 training repeat 分开
报告；跨 repeat 汇总只作分层 bootstrap/辅助描述，不把同一 environment seed 的两次训练结果当完全独立。

Stage 流程为：

1. 训练 repeat 1 的 replay/DAgger pair，按 validation rule 选择 checkpoint；
2. 在 Stage A 20 个 full-chain paired seeds 和 5 个 atomic seeds 上评估；
3. 仅当 Stage A gate 通过，训练 repeat 2，并在同一 Stage A seeds 检查训练方向一致性；
4. 两个 repeat 均满足方向 gate 后，在全新 Stage B 30 seeds 上同时确认；
5. 任一 gate 失败即保留结果并停止 promotion，不调参后复用同一 seeds。

Stage A 方向 gate：

- `pi_dagger` 相对 paired `pi_replay` 无新增 system/safety/tracking failure；
- atomic Grasp/Lift/Place 各自净成功数不下降；
- full-chain Reach 下降不超过 1/20；
- 无条件 Grasp、Lift 均不下降，且至少一个提高不少于 2/20；
- mean completed skills 提高。

最终 promotion 要求两个 training repeat 在各自 Stage A+Stage B 的 50 个 seeds 上方向一致：Reach 下降
不超过 2/50，Grasp/Lift 均不下降且至少一个存在正 paired net wins，mean completed skills 均提高，
atomic Grasp/Lift/Place 不回归，并且没有新增 system/safety/tracking failure。只有一个 repeat 改善时可写
“存在候选信号”，不能 promotion。Full success 提高是强证据，但不作为首次 E012 的必要条件。

### Boundary Chunk 与 matched-state diagnostics

在 Reach 和 Grasp crossing 周围固定记录：

```text
前一个 Replan / crossing Replan / 后第 1 个 Replan / 后第 2 个 Replan
```

每个位置分别报告 proposal count、per-slot mean pairwise disagreement、max disagreement、
newest-vs-oldest、newest-vs-weighted-history；arm 7D 与 gripper 1D 必须分开，不能让 gripper `0<->1`
切换支配一个 8D max spread。先在 Episode/boundary 内聚合，再跨 Episode bootstrap。

Newest-only 只作为 behavioral diagnostic。即使 `pi_dagger + newest-only` 提高并缩小与 temporal 的差距，
也不能单独写成 policy prediction variance 下降。真正的 variance test 使用 checkpoint validation 的
held-out `pi_0` boundary observation sequence：在相同逐帧 RGB/proprio 上，`pi_replay` 与 `pi_dagger`
使用相同的一组 per-frame Flow seeds，分别比较 across-seed Action variance 和 absolute-time-aligned Chunk
disagreement。当前模型没有 observation-history 输入；这里固定的是同一录制 observation sequence，各模型
的 proposal history 只由它在该序列上的预测确定。这样区分：

```text
新 policy 到达了更容易的 state
vs
同一个 state 上新 policy 本身更稳定
```

这些 disagreement 指标是机制证据和解释变量，不替代无条件闭环成功数与 atomic guardrail。

### 允许的结论边界

若 `pi_dagger` 在两个 training repeat 上均优于 `pi_replay`，且 matched-state disagreement 同时下降，
允许结论为：

> 在当前模型、Expert、Dataset mixer 和 temporal execution 配置下，补充 policy-induced boundary state
> 上的 Expert supervision 提高了后续技能闭环成功率，并支持 boundary coverage 不足是部分预测不稳定的
> 原因。

若 handoff 提高但 matched-state disagreement 不变，只能结论为 Local DAgger 提高 recovery；若二者都
不变，则当前 40 条、固定窗口和 sampler 下没有支持 boundary coverage 是主要瓶颈。由于监督从 Predicate
crossing 后才开始，第一版不会直接纠正 crossing 前几个 Replan 产生的旧 Chunk；任何关于“修复了
pre-boundary Chunk generation”的表述都被禁止。

### 计划实现顺序与现有接口映射

按以下顺序渐进实现，每一步先做针对性测试，不在 schema/collection 未稳定时同时修改训练器：

1. **Versioned Local DAgger provenance：**在保持 clean `trajectory/v2` 可读的前提下增加版本化、
   all-or-none 的 Local DAgger metadata 与逐 Action `action_source/expert_supervision_mask`；Writer、Reader、
   `validate_trajectory` 和专用 validator 先完成 round-trip/fail-closed 测试。第一版采用 additive optional
   contract，不因新增字段直接让固定 `pi_0` checkpoint 的 `dataset_schema` 身份失效。
2. **Live takeover collector：**复用现有双相机/Predicate/Outcome evidence 和 MPLib Expert，只新增
   frozen policy roll-in、指定 boundary 切换 Action producer、controller target 连续性与完整成功写入；
   不重构普通 clean Expert collector。
3. **Snapshot diagnostics：**单独封装 snapshot bundle 和 ring buffer，先验证 save/restore round-trip；
   accepted trajectory 继续来自 live session，避免把 snapshot 功能与数据拼接耦合。
4. **Dataset 与 audit：**`ActionChunkDataset` 只为 Local DAgger entry 构造完整 Expert/window anchors；
   clean entry 保持原索引。正式 success audit 规则不改，另加 provenance audit。D1 observed stats 单独落盘，
   训练 CLI 显式选择冻结的 D0 ProprioStats。
5. **Source sampler 与 loss 防线：**在现有 Task→Episode→timestep 逻辑外增加 source-first quota 和 exposure
   ledger；Collator/Trainer 显式验证 supervision mask，最终 effective mask 才传给 Flow/Event loss。
6. **独立 warm start：**新增与 `--resume` 互斥的 `--init-checkpoint`，复用严格 checkpoint metadata/权重
   loader，但不恢复 optimizer/scheduler/scaler/trainer/RNG；测试两组初始参数 SHA 相同、训练状态为零。
7. **Evaluation metrics：**扩展 rollout 聚合器，正式增加 `P(Lift|Grasp)`、无条件阶段分子、共同
   predecessor paired table，以及 arm/gripper 分开的固定 boundary-window disagreement。
8. **E012a smoke：**只用非正式 smoke seeds 验证一次 RG 和一次 GL live takeover、完整 audit、无 Policy
   target leakage、snapshot 和产物恢复；smoke 不进入 40 条 Dataset、risk percentile 或效果统计。
9. **正式执行：**E012a gate 通过后才完整扫描 `30000..30199`；40 条和 D1 身份冻结后才训练 repeat 1。

上述实现不得顺手改变普通 clean Dataset、默认 temporal ensemble 或历史 `--resume` 语义。任何不得不修改
核心 trajectory schema version、Predicate 或 Controller 的发现都视为 protocol amendment，在正式
collection 前单独记录并重新检查 checkpoint/Dataset 兼容性。

### 计划产物

E012a/E012b 运行后在 `docs/results/e012/` 保留人类可读报告及以下小型、可审计文件；权重、RGB 和大
trajectory 不提交 Git：

```text
preregistration.json
collection_manifest.json
collection_candidates.jsonl
collection_summary.json
boundary_distribution.json
train_repeat_1.json
train_repeat_2.json
sampler_exposure.jsonl
checkpoint_selection.json
stage_a_episodes.jsonl
stage_a_summary.json
stage_b_episodes.jsonl
stage_b_summary.json
matched_state_diagnostics.json
```

每个文件记录 checkpoint/Dataset/source SHA256、seed、配置、环境版本和上游文件哈希。正式结果发布前
独立从 raw JSONL 复算汇总；GitHub 只发布源码、manifest、聚合与必要 raw measurement，不上传模型、
Dataset 或含图像的 trajectory。

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
