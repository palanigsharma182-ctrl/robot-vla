# E010 — Layer 12 五技能梯度冲突与 base/event 归因 probe

## 技术摘要

- **没有 checkpoint 确认 Reach/Transport 训练梯度冲突。** e098-best 与 e100 都未同时通过预注册的
  train discovery 和独立 val confirmation 门槛；`confirmed_checkpoint_labels=[]`。e098-best 的
  val median cosine 为 `-0.094`、`3/5` 为负，接近但仍同时错过 `<=-0.10` 和 `4/5` 两项门槛；
  e100 为明显正向的 `+0.441`、`0/5` 为负。
- **当前证据不支持直接开始多动作头，也不支持先上 PCGrad/CAGrad。** e098-best 的
  `velocity_head` 为 `-0.120`、`3/5` 为负，负方向不够稳定；e100 的同一输出头为 `+0.480`、
  `0/5` 为负。三个 discovery checkpoint 的五技能所有 pair median 均为正，梯度手术在大多数
  已测 batch 上不会触发。
- **Reach/Transport 的 base/event 内部归因在本次样本上不可识别。** 两个技能在全部 10 个 val
  confirmation unit 中都没有前 4 个执行步 critical event，故 `0.25 × event gradient` 范数为 0，
  base/event cosine 按预注册规则为 `null`。这不能解释为“base/event 相容”；它只说明当前分段与
  event mask 没有给 Reach/Transport 直接 event 信号。
- **更值得继续验证的是实际更新位移和梯度幅度，而非静态 pairwise 反向。** e098-best/e100 的
  Grasp 中位 `all_trainable` 梯度范数分别是 Transport 的 `2.67×/2.71×`。这是更新幅度或采样暴露
  失衡的候选信号，但本 probe 没有证明它导致 E009 的闭环行为交换。
- **结论是局部一阶诊断，不是行为成功率。** E009 的 epoch 98→100 Reach/Transport 闭环交换仍然
  存在；E010 否定的是“该交换可由稳定、同时的 per-batch 负梯度直接解释”，不是“多技能训练没有
  任何竞争”。下一项低成本归因应直接投影实际 checkpoint 位移，再继续 handoff 状态 probe。

## Reach/Transport 没有通过两阶段确认门槛

`median [Q25, Q75]` 来自严格配对的 repeat；负比例括号内为 Wilson 95% 区间。Stage A 每个
checkpoint 使用 8 个 train repeat，Stage B 每个 confirmation checkpoint 使用 5 个独立 val repeat。

| Checkpoint | Split/stage | Reach–Transport median cosine [IQR] | 负 repeat | 负比例 Wilson 95% | 预注册门槛 |
| --- | --- | ---: | ---: | ---: | --- |
| e090 | train discovery | `+0.421` [`+0.364`, `+0.504`] | 0/8 | 0.0%–32.4% | 不进入 confirmation |
| e098-best | train discovery | `+0.164` [`+0.015`, `+0.355`] | 2/8 | 7.1%–59.1% | 未通过 `<=-0.10, >=6/8` |
| e098-best | val confirmation | `-0.094` [`-0.146`, `+0.077`] | 3/5 | 23.1%–88.2% | 未通过 `<=-0.10, >=4/5` |
| e100 | train discovery | `+0.173` [`+0.015`, `+0.313`] | 2/8 | 7.1%–59.1% | 未通过 `<=-0.10, >=6/8` |
| e100 | val confirmation | `+0.441` [`+0.348`, `+0.546`] | 0/5 | 0.0%–43.4% | 未通过 |

逐 repeat 也显示 e098-best 只是混合方向，不是稳定负冲突：train 为
`[+0.342, -0.162, -0.355, +0.395, +0.559, +0.101, +0.074, +0.227]`，val 为
`[-0.146, +0.077, -0.094, +0.414, -0.240]`。e100 的五个 val repeat 全部为正：
`[+0.546, +0.330, +0.348, +0.613, +0.441]`。

因此 E009 的行为交换与当前 checkpoint 上的 pairwise 梯度 cosine 没有单调关系：e090 的
Reach/Transport 梯度最相容，但闭环行为接近 e098-best；e100 的 val 梯度最相容，却是 E009 中
Transport 退化最明显的 checkpoint。静态 cosine 不是该行为现象的充分解释变量。

## 五技能 train 梯度矩阵全部以正 median 为主

下表是 `all_trainable` 的 8-repeat median cosine。保留精确矩阵而不画 checkpoint 连线图：这里只有
3 个离散 checkpoint、每点 8 个 repeat，且决策依赖正负号和预注册计数；矩阵比连续趋势图更可审计。

### e090

|  | Reach | Grasp | Lift | Transport | Place |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reach | 1.000 | 0.305 | 0.537 | 0.421 | 0.487 |
| Grasp | 0.305 | 1.000 | 0.479 | 0.402 | 0.559 |
| Lift | 0.537 | 0.479 | 1.000 | 0.663 | 0.566 |
| Transport | 0.421 | 0.402 | 0.663 | 1.000 | 0.434 |
| Place | 0.487 | 0.559 | 0.566 | 0.434 | 1.000 |

### e098-best

|  | Reach | Grasp | Lift | Transport | Place |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reach | 1.000 | 0.506 | 0.393 | 0.164 | 0.187 |
| Grasp | 0.506 | 1.000 | 0.432 | 0.357 | 0.364 |
| Lift | 0.393 | 0.432 | 1.000 | 0.809 | 0.426 |
| Transport | 0.164 | 0.357 | 0.809 | 1.000 | 0.425 |
| Place | 0.187 | 0.364 | 0.426 | 0.425 | 1.000 |

### e100

|  | Reach | Grasp | Lift | Transport | Place |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reach | 1.000 | 0.476 | 0.339 | 0.173 | 0.458 |
| Grasp | 0.476 | 1.000 | 0.265 | 0.315 | 0.379 |
| Lift | 0.339 | 0.265 | 1.000 | 0.601 | 0.484 |
| Transport | 0.173 | 0.315 | 0.601 | 1.000 | 0.510 |
| Place | 0.458 | 0.379 | 0.484 | 0.510 | 1.000 |

30 个 checkpoint/pair 组合的 median 全部为正。e098-best 和 e100 的 Reach/Transport 都各有
`2/8` 负 repeat，是所有 pair 中最弱的相容方向，但远未达到 `6/8` 的确认要求。该结果反对把
“广泛共享表示负冲突”作为当前主解释；它不排除不同状态子分布、训练历史或梯度幅度造成竞争。

## e098-best 有后层局部负信号，但没有稳定输出头定位

下表是独立 val 上 Reach total 与 Transport total 的 module median cosine 和负 repeat 计数。
模块门槛与 overall 相同：median `<=-0.10` 且至少 `4/5` 为负；但只有 overall 两阶段先确认后，模块
标签才允许激活。

| Module | e098-best median（负/5） | e100 median（负/5） |
| --- | ---: | ---: |
| Adapter | +0.066（2） | +0.118（1） |
| State encoder | +0.015（1） | +0.027（1） |
| Action encoder | +0.071（1） | +0.227（0） |
| Block 00 | -0.008（4） | +0.025（0） |
| Block 01 | -0.004（3） | +0.237（0） |
| Block 02 | -0.029（3） | +0.195（0） |
| Block 03 | -0.047（3） | +0.275（0） |
| Block 04 | +0.042（2） | +0.377（0） |
| Block 05 | +0.030（2） | +0.099（0） |
| Block 06 | +0.151（1） | +0.349（1） |
| Block 07 | -0.029（4） | +0.076（0） |
| Block 08 | +0.088（1） | +0.367（1） |
| Block 09 | -0.048（4） | +0.312（0） |
| Block 10 | +0.103（1） | +0.434（1） |
| Block 11 | +0.063（1） | -0.012（5） |
| Block 12 | -0.091（4） | +0.402（0） |
| Block 13 | +0.026（2） | -0.095（5） |
| Block 14 | -0.103（3） | +0.431（0） |
| Block 15 | **-0.152（4）** | +0.238（1） |
| Final norm | +0.100（0） | +0.329（0） |
| Velocity head | -0.120（3） | +0.480（0） |
| All trainable | -0.094（3） | +0.441（0） |

e098-best 只有 Block 15 单独同时满足幅度和计数门槛；Velocity head 虽达到负幅度，却只有 `3/5`
为负，Block 14 也只有 `3/5`。e100 的 Block 11/13 虽为 `5/5` 负，但 median 只有 `-0.012/-0.095`，
没有达到幅度门槛，且 total/head 明显为正。这些都是可报告的局部信号，但不能事后把门槛放宽并称为
“后层冲突”或“输出头冲突”。

架构含义很直接：

- **多动作头：暂不做。** 输出头负方向没有跨 checkpoint 和 repeat 稳定复现。
- **后四层分支/skill adapter：暂不做。** 只有 e098-best 的单个 Block 15 达标，未形成 3/4 后层模式。
- **PCGrad/CAGrad：暂不做。** 五技能 median 全部为正，e100 confirmation 全部为正；基于负 dot 的
  梯度手术在已测样本上大多不会改变更新。
- **Layer 24 Key + Layer 12 Value：E010 不新增支持也不否定。** Adapter 没有确认负冲突；多层
  Key/Value 的依据仍来自 E008 的几何/语义表示与后续 handoff 证据，而不是本次梯度结果。

## event 归因为空，但梯度幅度出现跨技能不平衡

Stage B 中 Reach 与 Transport 的 `event gradient` 在两个 checkpoint、每个 `5/5` repeat 上范数均为
0；因此 `reach_base↔reach_event` 与 `transport_base↔transport_event` 的 available repeats 都是
`0/5`，median/IQR 为 `null`。这是预注册的零范数处理，不是数据错误。

同一份 train sample plan 的 critical action step 分布如下；三个 checkpoint 的样本身份完全相同，
所以计数也相同。

| Skill | 8 个 batch 的 critical action steps | 含 event 的 batch | 解释 |
| --- | ---: | ---: | --- |
| Reach | 0 | 0/8 | 没有直接 event 梯度 |
| Grasp | 29 | 7/8 | event 信号最密集 |
| Lift | 10 | 5/8 | 少量跨阶段/状态事件 |
| Transport | 0 | 0/8 | 没有直接 event 梯度 |
| Place | 5 | 3/8 | 稀疏 release/place event |

所以 E010 只说明本次 sampled windows 中没有出现“Reach 自己的 base 与 Reach event 抵消”或
“Transport 自己的 base 与 Transport event 抵消”，不能排除 **Grasp/Place 的 event 更新通过共享
Expert 间接改变 Reach/Transport**。要回答后者，必须显式构造含 critical event 的 Grasp/Place
batch，再与 Reach/Transport base gradient 配对。

`all_trainable` 梯度范数还显示 checkpoint 后期出现明显幅度差：

| Checkpoint | Reach | Grasp | Lift | Transport | Place | 最大/最小中位范数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| e090 | 1.099 | 1.047 | 1.037 | 0.941 | 1.040 | 1.17× |
| e098-best | 0.754 | **1.262** | 0.737 | **0.474** | 0.788 | 2.67× |
| e100 | 0.727 | **1.272** | 0.716 | **0.470** | 0.840 | 2.71× |

正式训练的 skill sampling weights 为 Reach/Grasp/Lift/Transport/Place = `1.5/1/1/1.5/2`；probe
这里固定每技能相同 batch size，所以表中是单独的梯度幅度，不含采样频率。Grasp 在更低采样权重下
仍具有最大单-batch 梯度范数，提示下一步应检查实际更新贡献或 GradNorm 类尺度控制。但正 cosine
加上范数不平衡仍不足以证明 Grasp “伤害” Transport，不能直接修改 loss 权重。

## 范围、数据与指标定义

- 模型：E008 Layer 12 五技能联合训练，Qwen `Qwen/Qwen3.5-2B` 固定 revision 并完全冻结。
- Checkpoint：e090 `step-00005760.pt`、e098-best `best.pt`、e100 `step-00006400.pt`。
- Dataset：`trusted-v0.4-event-recovery-220`，220 trajectories、48,922 steps；dataset SHA256
  `bc024b6b9c566ca9500945fb6ac262bf657bee713d8a5816229bdc8478139407`。
- Discovery：train split，8 repeats × 每技能 8 个不同 trajectory；共覆盖 64 个不同 train
  trajectory，三个 checkpoint 严格复用 sample identity。
- Confirmation：val split，5 repeats × 每技能 4 个不同 trajectory；共覆盖 20 个不同 val
  trajectory，与 train coverage 零重叠。
- Flow：同一 stage/repeat 的所有技能和 checkpoint 使用相同 flow time/noise seed。
- Objective：`L_total = L_base + 0.25 × L_event`；event 只看 action chunk 前 4 个实际执行步。
- 精度：BF16 autocast；每个 parameter group 的 Gram 用 FP32 dot/累加后保存。
- 参数组：Adapter、State/Action encoder、16 个 Expert block、Final norm、Velocity head，共 21 个
  互斥 primitive group；`all_trainable` 是这些 Gram 的逐元素和。

对 checkpoint `c`、repeat `r`、技能或 component `i`，设模块 `m` 的梯度为
`g(c,r,i,m)`。原始记录保存：

```text
G(c,r,m)[i,j] = <g(c,r,i,m), g(c,r,j,m)>
cos(i,j,m) = G[i,j] / sqrt(G[i,i] * G[j,j])
```

零范数时 cosine 为 `null`。Stage B 只保存四个 primitive 向量
`reach_base/reach_event/transport_base/transport_event`，再用严格线性 Gram 变换构造两个 total；没有
额外 forward，也没有把 `null` 填成 0。probe 不构造 optimizer、不裁剪、不调用 `step()`。

## 独立复算与可审计性

独立验证脚本 [validate_e010_gradient_probe.py](../../../scripts/validate_e010_gradient_probe.py) 只使用
Python 标准库，不导入实验聚合模块。它从 raw JSONL 重新完成以下检查：

- 34/34 measurement identity 完整且唯一；24 个 discovery、10 个 confirmation；
- 22 个保存 group 的每个 Gram 均有限、对称、对角非负；
- `all_trainable` 与 21 个 primitive group 逐元素求和完全一致，最大绝对差为 `0.0`；
- 1320 个 summary row 的 cosine、median、IQR、min/max、负比例和 Wilson 95% 全部独立复现，
  与正式 summary 的最大绝对差为 `1.11e-16`；
- sample identity、Flow seed、train/val trajectory coverage 与 manifest 一致，跨 split 重叠为 0；
- 三个 checkpoint 的 Adapter/Expert 参数 SHA256 前后相同，parameter version 未改变；
- 预注册 assessment 独立复现为两个 checkpoint 均 `no_confirmed_gradient_conflict`。

代码验证：新增梯度诊断/CLI 的 34 个远端针对性测试全部通过，三个新增文件通过 Ruff；实验代码
commit 为 `b11c9af`，source-tree revision 为
`source-tree-sha256:71c9284e6873fe5b7cb888e041c781911b7dec9b5bf9ceee624713506241d992`。

正式 artifact SHA256：

| File | SHA256 |
| --- | --- |
| `probe-manifest.json` | `2a286f2accb683f2b14908b57e412ca8c7e39f18b20e08aaf315068b52ab551c` |
| `measurements.jsonl` | `7b14dbdf15abb2df9f803015b9ee0775aa0bca9193e1e957846c2131bf5fab5d` |
| `probe-summary.json` | `7f0087c47bd1a6ea1188ad499c954a85e46fc7cd385df4bbed55e9401e092705` |

仓库同时保存 [probe-manifest.json](probe-manifest.json)、[measurements.jsonl](measurements.jsonl)、
[probe-summary.json](probe-summary.json) 与
[independent-validation.json](independent-validation.json)。模型权重、Qwen cache 和 dataset 不上传。

## 限制、稳健性与不能得出的结论

1. **这是 checkpoint 局部一阶诊断。** 它测当前参数处的瞬时梯度，不重放 epoch 98→100 之间真实
   optimizer trajectory，也不包含 Adam moment、学习率或历史 batch 顺序。
2. **静态正 cosine 不等于不存在多任务竞争。** 方向相容时仍可能有幅度、采样频率、曲率、容量或
   状态条件竞争；模型行为也可能对小参数变化高度非线性。
3. **event 分解对目标技能为空。** Reach/Transport 没有 critical event，无法检验 Grasp/Place event
   更新对它们的间接影响。报告明确保留 `null`，不把不可识别结果写成负结论。
4. **sample 数仍有限。** Discovery 8 repeats、confirmation 5 repeats；Wilson 区间较宽，尤其
   e098-best 的 val 负比例 `3/5` 区间为 23.1%–88.2%。预注册门槛适合做本实验的架构分流，不是总体
   统计显著性声明。
5. **离线 loss/gradient 不是闭环成功率。** E009 的行为交换仍以闭环结果为准；E010 只约束机制解释。
6. **随机技能段可能稀释边界冲突。** 如果问题集中在 Reach→Grasp 或 Lift→Transport 的最后/最初几
   帧，随机段内 timestep 的平均 Gram 可能看不到它；需要 boundary-conditioned probe。

按 `validate-data` 的发布标准，本分析为 **可发布但必须带上述限制**。不存在阻止发布的 calculation
或数据完整性问题；主要不确定性来自实验可识别范围，而不是结果文件质量。

## 下一步：先投影真实 checkpoint 位移，再决定训练或架构改动

E010 直接支持的下一项便宜归因是 **e098-best→e100 实际更新位移 probe**，而不是立刻重训：

1. 固定 E010 的 train/val sample identity，计算每个模块的 `Δθ = θ_e100 - θ_e098`；
2. 在 e098-best 处计算 `g_skill · Δθ`。一阶近似下，负值表示实际位移降低该技能 loss，正值表示提高；
3. 单独构造 guaranteed-critical 的 Grasp/Place event batch，比较
   `reach_base/transport_base ↔ grasp_event/place_event`，补上 E010 的 event 可识别缺口；
4. 增加 Reach→Grasp、Lift→Transport boundary-conditioned batch，并与随机段内结果并排；
5. 只有 `Δθ` 明确呈现“帮助 Reach、伤害 Transport”并稳定定位到 Head/后层时，才进入共享 trunk +
   多头或后层分支 A/B；若主要是范数/频率失衡，再比较 GradNorm、loss normalization 或 sampling；
   若离线一阶投影仍解释不了闭环交换，则优先完成 handoff 状态分布 probe。

这条顺序保持变量单一，也直接回答 E010 尚未回答的问题：不是两个技能的**当前梯度是否相反**，而是
训练最后两轮的**实际参数位移究竟更接近哪个技能、在哪个模块发生**。

## 仍未回答的问题

- e098-best→e100 的真实参数位移对 Reach/Transport loss 的一阶预测变化分别是什么？
- Grasp/Place 的 event gradient 是否通过共享 Expert 与 Reach/Transport base gradient 冲突？
- 负方向是否只集中在技能边界，而被随机段内 timestep 稀释？
- E009 的闭环行为交换主要来自参数更新、Layer 12 语义寻址，还是策略自产 handoff 状态分布？
- 梯度范数差在真实 sampler 暴露和 Adam moment 加权后是否仍然存在？
