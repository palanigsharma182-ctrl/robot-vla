# E011 — RTC Action Chunk Transition 受控评估

## 技术摘要

- **RTC 不通过 promotion，也不进入 Stage B。** 在严格配对的 10 个完整 Episode 中，当前正式
  `temporal-ensemble` 的 Reach 为 `6/10`，RTC 只有 `2/10`；平均完成技能数从 `1.0` 降到
  `0.8`。这是预注册门槛中的明显行为回归，因此默认执行协议继续使用 temporal ensemble。
- **RTC 的条件率不能直接解释为 Reach→Grasp 改善。** 聚合值是 temporal `4/6`、RTC `2/2`，
  但两组共同通过 Reach 的只有 seeds `20005、20009`，在这个共同支持集上两者都是 Grasp `2/2`。
  RTC 的 `100%` 主要受 Reach 选择效应影响，不是可识别的 handoff 增益。
- **存在一个小样本的后段正向信号。** 对共同通过 Grasp 的 `20005、20009`，temporal 都停在 Lift
  前，RTC 都继续完成 Lift 和 Transport；但只有 2 个配对 seed，且 RTC 另外丢失了 4 个 temporal
  Reach 成功，所以不足以抵消总体回归或触发独立确认。
- **RTC 没有破坏独立 Grasp/Lift/Place。** 75 个 atomic guardrail Episode 中，三个策略的
  Grasp/Lift/Place 均为 `5/5`；Transport 为 newest `2/5`、temporal `2/5`、RTC `3/5`，Reach
  三组均为 `0/5`。RTC atomic 总计 `18/25`，两个对照均为 `17/25`。
- **工程与安全链路稳定，但当前诊断实现更慢。** Stage A 三组均无 system error 或 Action safety
  rejection；temporal/RTC 均无 tracking saturation。由于每个有历史的 RTC Replan 同时生成 paired
  raw/RTC Chunk，RTC 完整 Episode 平均耗时 `53.19 s`，是 temporal `30.99 s` 的 `1.72x`。

## Stage A：30 个完整闭环 Episode

三个策略严格共享环境 seeds `20000..20009`、Flow sampling seed base `42424`、Checkpoint、Dataset、
10-step Flow、Layer 12 Context、Controller、安全契约和 anomaly-replan 配置。

| Strategy | Reach | Grasp | Lift | Transport | Place | P(Grasp\|Reach) | P(Transport\|Lift) | 平均技能数 | Full success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| newest-only | 2/10 | 1/10 | 1/10 | 1/10 | 0/10 | 1/2 = 50.0% | 1/1 = 100% | 0.5 | 0/10 |
| temporal-ensemble | **6/10** | **4/10** | 0/10 | 0/10 | 0/10 | 4/6 = 66.7% | n/a | **1.0** | 0/10 |
| rtc | 2/10 | 2/10 | **2/10** | **2/10** | 0/10 | 2/2 = 100% | 2/2 = 100% | 0.8 | 0/10 |

条件率的 Wilson 95% 区间很宽：temporal 的 `P(Grasp|Reach)` 为 `[30.0%, 90.3%]`，RTC 为
`[34.2%, 100%]`；RTC 的 `P(Transport|Lift)` 同样只有 2 个分母。它们用于描述本组 Episode，
不能视为精确总体效应。

逐 seed 完成技能数如下；技能顺序固定为 Reach、Grasp、Lift、Transport、Place。

| Seed | newest-only | temporal-ensemble | rtc | 配对观察 |
| ---: | ---: | ---: | ---: | --- |
| 20000 | 0 | 2 | 0 | RTC 丢失 temporal Reach/Grasp |
| 20001 | 0 | 0 | 0 | 三组均未 Reach |
| 20002 | 0 | 0 | 0 | 三组均未 Reach |
| 20003 | 0 | 1 | 0 | RTC 丢失 temporal Reach |
| 20004 | 0 | 1 | 0 | RTC 丢失 temporal Reach |
| 20005 | 4 | 2 | 4 | RTC 与 newest 到 Transport，temporal 到 Grasp |
| 20006 | 0 | 2 | 0 | RTC 丢失 temporal Reach/Grasp |
| 20007 | 0 | 0 | 0 | 三组均未 Reach |
| 20008 | 0 | 0 | 0 | 三组均未 Reach |
| 20009 | 1 | 2 | 4 | RTC 到 Transport，temporal 到 Grasp |

temporal 与 RTC 的 Reach 配对表为：两者成功 2、两者失败 4、temporal-only 4、RTC-only 0；exact
McNemar 双侧 `p=0.125`。方向和效应量足以执行预注册的“不晋级”决策，但 10 个 seed 还不足以把
Reach 下降写成高置信总体效应。

## 条件 handoff 与阶段耗时

| Strategy | mean steps to Reach | Reach→Grasp | Lift→Transport | mean episode steps |
| --- | ---: | ---: | ---: | ---: |
| newest-only | 98.5 | 21.0 | 160.0 | 300.0 |
| temporal-ensemble | 124.17 | 41.25 | n/a | 300.0 |
| rtc | 98.0 | 23.5 | 135.5 | 300.0 |

这些阶段均值只在成功进入对应阶段的 Episode 上聚合，分母不同，不能据此得出 RTC 整体更快。
在共同 Reach seeds `20005、20009` 上，temporal 的 Reach steps 是 `81、95`，RTC 是 `92、104`；
RTC 分别慢 11 和 9 步。对应 Reach→Grasp steps，temporal 是 `21、37`，RTC 是 `13、34`，RTC
分别快 8 和 3 步。共同支持集说明 RTC 在这两个 seed 的 Reach 后推进不慢，但进入 Reach 更慢。

为避免成功者选择偏差，另对所有 10 个 Episode 固定比较前 80 个环境步：

| Strategy | mean TCP speed | mean max joint velocity | step 80 前的平均 TCP-to-object distance |
| --- | ---: | ---: | ---: |
| newest-only | 0.05277 m/s | 0.14037 rad/s | 0.11953 m |
| temporal-ensemble | **0.05510 m/s** | 0.13624 rad/s | **0.09369 m** |
| rtc | 0.05266 m/s | 0.14088 rad/s | 0.11711 m |

RTC 不是简单把所有关节速度压低：它的早期 joint velocity 与 newest/temporal 接近；但前 80 步后
TCP 到物体的距离更接近 newest-only，明显大于 temporal。当前证据更像 temporal averaging 在早期
Reach 轨迹上提供了有用的方向稳定性，而不是 RTC 仅仅“整体减速换成功率”。

## Chunk disagreement 与边界诊断

Stage A 的 RTC 共记录 750 个 Replan，其中每个 Episode 首次 Replan 没有 reference，另 740 个均有
正确的 12-step previous overlap。没有 NaN/Inf，reset/anomaly 后无历史的退化路径也在 smoke/atomic
中实际触发。

740 个有历史 Replan 的 normalized action 诊断：

| Metric | mean | p50 | p95 | p99 | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw/previous overlap mean disagreement | 0.03212 | 0.02897 | 0.04865 | 0.13262 | 0.24865 |
| RTC/previous prefix mean disagreement | 0.02124 | 0.01906 | 0.03205 | 0.08162 | 0.19744 |
| RTC prefix mean correction | 0.00684 | 0.00423 | 0.01190 | 0.03171 | prefix max 2.0 |
| RTC future mean correction | 0.00430 | 0.00272 | 0.00607 | 0.05338 | future max 2.0 |

`prefix max correction >= 1` 只出现在 `5/740` 个 Replan，`future max correction >= 1` 出现在
`8/740` 个 Replan；两者各只有 1 次达到 2.0，集中在能推进到后段的 seeds `20005、20009`。
因此没有证据表明所有 Replan 的整个 horizon 都被旧计划数值锁死；但少量边界处存在大修正，且当前
`future` 聚合覆盖 slots `4..15`，没有把零权重的 slots `12..15` 单独报告，所以不能仅凭该聚合量
完全排除 free-tail 的跨 slot 网络耦合。

raw disagreement 在 12 个 overlap slots 上聚合，RTC post disagreement 按受控协议只报告执行前缀
4 slots；二者不是严格相同 slot 集，因此均值下降只能作为方向性诊断，不能当作无偏的 before/after
效应量。

技能边界给出更具体的证据：

- temporal 的 6 个 Reach 边界 proposal spread 范围为 `0.081..1.642`。两个随后 Grasp 失败的
  seed 中，一个是全组最高值 `1.642`，另一个却是最低值 `0.081`，所以 Reach→Grasp 失败并不由
  大 spread 单调解释。
- temporal 的 4 个 Grasp 边界 proposal spread 均较大，为 `1.093..1.658`；四条随后全部未完成
  Lift。这支持在 Grasp→Lift 边界存在显著旧/新 proposal 冲突，但仍是相关性。
- RTC 在 seeds `20005、20009` 的 Grasp 边界 raw max disagreement 为 `1.813、1.985`，两条均继续
  完成 Lift 和 Transport。其 Lift 边界 raw max disagreement 只有 `0.045、0.069`，修正也很小；
  因而这两条成功不能证明 Lift→Transport 原本存在强冲突并被 RTC 修复。

## Atomic guardrail：75 个 Episode

每个策略使用同一 seeds `20010..20014`，五技能各 5 个 Episode，最大 100 policy steps。前置状态由
同一可信 MPLib preparation 产生，只隔离目标原子技能。

| Strategy | Reach | Grasp | Lift | Transport | Place | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| newest-only | 0/5 | 5/5 | 5/5 | 2/5 | 5/5 | 17/25 |
| temporal-ensemble | 0/5 | 5/5 | 5/5 | 2/5 | 5/5 | 17/25 |
| rtc | 0/5 | 5/5 | 5/5 | **3/5** | 5/5 | **18/25** |

| Strategy | Reach steps | Grasp steps | Lift steps | Transport steps | Place steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| newest-only | 100.6 | 4.0 | 31.2 | 85.6 | 6.4 |
| temporal-ensemble | 100.0 | 4.0 | 25.6 | 87.2 | 4.2 |
| rtc | 100.0 | 4.0 | 25.6 | 83.2 | 5.0 |

atomic 结果满足“独立 Grasp/Lift/Place 不发生明显回归”的必要条件，但不能挽回完整闭环 Reach 的
回归，也不能替代 Stage A 的 handoff 结论。

## 系统、安全与开销

| Protocol | Strategy | System error | Action safety rejection | Anomaly replans | Tracking saturation |
| --- | --- | ---: | ---: | ---: | ---: |
| Stage A | newest-only | 0 | 0 | 2 | 2 |
| Stage A | temporal-ensemble | 0 | 0 | 0 | 0 |
| Stage A | rtc | 0 | 0 | 0 | 0 |
| Atomic | newest-only | 0 | 0 | 1 | 1 |
| Atomic | temporal-ensemble | 0 | 0 | 0 | 0 |
| Atomic | rtc | 0 | 0 | 1 | 1 |

Stage A 总 wall time：newest `316.32 s`、temporal `309.90 s`、RTC `531.94 s`。Atomic 每 policy
step 平均约 newest `0.103 s`、temporal `0.107 s`、RTC `0.171 s`。当前 RTC 的额外开销主要来自
为了记录同噪声 paired raw/RTC 诊断而运行两条 Expert Flow；这是本实验实现的开销，不自动等同于
去掉 paired 诊断后的最小 RTC 部署开销。

## 实验身份与可审计性

- Base Git commit：`aec3d7ccd623c20073f101726a0aaddd6d70ca16`；RTC 在未提交 dirty
  worktree 上运行，正式 `experiment.json` 另固定源码 revision。
- Evaluation source revision：
  `source-tree-sha256:adfce370c438d460eb4178be9af38ee5741554741a3c99f6acd8485847244dec`。
- Layer 12 Checkpoint SHA256：
  `a542076f291e29b68e3d28930b15c40396d511a44eb358c2eaeb4e113c041ad6`。
- Dataset SHA256：
  `bc024b6b9c566ca9500945fb6ac262bf657bee713d8a5816229bdc8478139407`；220 trajectories / 48922 steps。
- Qwen revision：`15852e8c16360a2fea060d615a32b45270f8a8fc`。
- GPU：NVIDIA GeForce RTX 4090 24 GB；PyTorch 2.11.0+cu128；BF16。
- 非正式 smoke：seed `19999`；三组均无 system/safety/NaN/tracking 问题，不进入效果统计。
- 正式原始结果根目录：`runs/e011-rtc/`。

正式文件 SHA256：

| Protocol / strategy | `experiment.json` | `episodes.jsonl` | `summary.json` |
| --- | --- | --- | --- |
| Stage A / newest | `f82a7a2e…aaea0` | `eb145370…46f08` | `8bf79cf3…41ec` |
| Stage A / temporal | `dcea980c…0c28` | `b5adb22f…f501` | `76a0f170…61fc` |
| Stage A / rtc | `c88dd074…59fc` | `41a106a7…5576` | `935aa472…03a` |
| Atomic / newest | `a33fdb58…259c` | `78845a65…5508` | `6f20e496…44b2` |
| Atomic / temporal | `26fe0b16…0dc` | `fe6eb79a…80b2` | `4e1fe148…61fc` |
| Atomic / rtc | `87b2159f…307e` | `81f38ed6…5679` | `71680d3b…7c98` |

所有正式目录均为全新输出、未使用 `--resume`；Stage A 恰有 10 行/组，atomic 恰有 25 行/组；
Checkpoint、Dataset、source revision、seed 和除 strategy 外的配置逐组一致。原始 JSONL 全量扫描没有
`NaN`/`Infinity`。

## 事实、解释与尚不能得出的结论

**事实：** RTC 在 Stage A 丢失了 4 个 temporal-only Reach seed，没有获得 RTC-only Reach seed；
在两个共同 Reach seed 上，两者 Grasp 均为 `2/2`，RTC 随后多完成了 Lift 和 Transport；RTC 的
atomic Grasp/Lift/Place 没有回归；没有新增 Stage A system/safety/tracking 问题。

**当前最简解释：** temporal ensemble 的历史 proposal 并非只产生“旧计划污染”；在初始 Reach 阶段，
它还提供了有用的轨迹稳定性。RTC 的 prefix continuity 在少数已接近/抓住物体的 seed 上可能帮助了
后段转换，但当前配置以 Reach reactivity/覆盖率为代价。边界诊断最强的冲突信号出现在 Grasp→Lift，
而不是稳定地出现在 Reach→Grasp 或 Lift→Transport。

**尚不能得出：** 不能声称 RTC 修复了 Reach→Grasp 或 Lift→Transport；不能把 `2/2` 条件率推广到
总体；不能唯一归因于 guidance weight=10、Flow VJP 符号、Layer 12、速度变慢或某一个控制维度；也
不能由 10 个 seed 排除所有更弱 RTC 配置。尤其不得用 Stage A seeds `20000..20009` 调权后再把它们
当作确认集。

## 决策与下一步

1. 不 promotion RTC，不运行 Stage B；默认继续 `temporal-ensemble`，全局默认和正式执行协议均不变。
2. 不用本轮 Stage A seeds 调 `rtc_max_guidance_weight` 或 schedule。若未来重新设计 RTC，必须使用新的
   smoke seed 选数值范围，再使用全新的独立 paired seeds。
3. 按受控实验预案，把主要精力转向 handoff 状态分布与 Local DAgger：优先收集 temporal 在
   Reach→Grasp、Grasp→Lift 失败附近的 observation/action recovery，而不是继续盲调 RTC。
4. 保留当前 RTC 实现为显式实验策略和诊断工具，但不替换 temporal ensemble；如需进一步归因，可在
   新实验中单独记录 slots `12..15` free-tail correction，并增加同 slot 的 raw/post disagreement。
