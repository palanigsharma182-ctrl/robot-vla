# World Model / Critic V0 候选

## 身份与边界

- Candidate ID：`wmc-v0-main-20260906`
- Status：`development-only`
- Base：`origin/main@1508f2c0f8d07373cbdeac22167dd5ff38574967`
- E018 inheritance：`false`
- Actuation：`forbidden`
- Fresh test consumed：`false`
- Training/checkpoint：未运行、未生成

这个分支只发布可审阅的最小实现与合同，不把候选晋升为 canonical World Model、Evaluator 或
Planner。它不能拒绝、替换、重排或修改 VLA 动作，也不提供安全性和闭环收益结论。

## 为什么从更小的候选开始

当前公开 `main` 的稳定在线结构化状态只有：

```text
normalized proprio = q[7] + dq[7] + gripper_opening[1]  -> [15]
normalized action  = delta_q[7] + absolute gripper target[1] -> [16,8]
control             = 20 Hz，执行前 4 步后 5 Hz replan
```

因此本候选解决的是两个可反证的基础问题：

1. 一个极小的、显式因果的模型能否预测实际执行的未来四步 robot state；
2. 相邻 raw policy Chunk 的重叠预测是否一致，当前执行前缀是否出现异常幅值。

它没有物体、目标或接触的 deployable 输入，所以准确名称是 **robot structured dynamics
baseline**，不能宣称已经学会完整 manipulation world state。后续如果引入物体状态，必须先建立新的
deployable observation schema；不能把 simulator object GT 静默填入在线输入。

公开 `main` 还有一个与候选无关但会影响完整测试的基线限制：这个 commit 没有跟踪
`src/robot_vla/data/`，而部分既有代码和测试仍导入 `robot_vla.data.*`。本分支不复制、重建或伪造
canonical trajectory 包；当前只运行与候选直接相关、且不依赖该包的测试。

## 开源参考与实际采用范围

| 参考 | 采用 | 明确不采用 |
| --- | --- | --- |
| [TD-MPC2](https://github.com/nicklashansen/tdmpc2) | action-conditioned residual latent/state transition、短 horizon | actor、MPPI、在线 RL、Q ensemble |
| [Robotic World Model Lite](https://github.com/leggedrobotics/robotic_world_model_lite) | 显式短时动力学与未来 uncertainty 扩展方向 | locomotion 状态合同、完整依赖栈 |
| [ActProbe](https://github.com/air-embodied-brain/actprobe) | raw Action Chunk 的 TCE/ACM 分解 | task embedding、learned LSTM、阈值与安全 gate |
| [Sentinel/STAC](https://github.com/agiachris/sentinel) | temporal action consistency 作为零训练 baseline | VLM monitor、大量策略重采样 |

当前代码是项目内独立实现，没有 vendoring 上述仓库源码或权重。
候选时核对的仓库许可证分别为 TD-MPC2/Sentinel 的 MIT 和 RWM Lite/ActProbe 的
Apache-2.0；引用仅说明思路来源，本分支不引入它们的运行时依赖。

## V0 数据流

```text
VLA raw normalized chunk [16,8]
  |                                  +------------------------------+
  +-> ActionConsistencyCritic V0 --->| TCE / ACM / warmup / abstain |--+
  |                                  +------------------------------+  |
  |                                                                      v
  +-> existing temporal ensemble -> effective normalized chunk [16,8] -> shadow receipt
                                       |
normalized proprio [15] ---------------+
normalized commanded-q prefix [4,7] ---+
                                       v
                         TinyStructuredWorldModel
                         rollout effective chunk[:4]
                                       |
                                       v
                       predicted normalized proprio [4,15]

existing Action Adapter / Executor path is unchanged
```

Critic 必须看 temporal ensemble 之前的 raw proposal；否则被平滑后的动作会人为降低不一致度。
World Model 必须看实际准备执行的 effective prefix；完整 Chunk 的后 12 步只是 policy intent，没有对
这次 5 Hz macro-transition 产生因果作用。此外，执行器在后续步使用“replan 起点 controller q +
累积 delta-q”形成 commanded target，所以模型还必须显式接收这个 controller reference；仅有
actual proprio 与当前 delta-q 不足以唯一决定下一状态。

## TinyStructuredWorldModel

架构身份为 `tiny-structured-world-model/v0`：

```text
[state 15, action 8, commanded-q 7] -> Linear(30,128) -> SiLU
                                      -> Linear(128,128) -> SiLU -> FP32 RMSNorm
                                      -> Linear(128,15) -> predicted state residual
```

同一个 transition MLP 按时间因果地递推四次：

```text
z_(k+1) = z_k + f(z_k, action_k, commanded_q_k),  k = 0..3
```

默认参数总数固定为 `22,543`。最后一个 delta head 零初始化，因此未训练模型是显式的
state-hold baseline，而不是随机输出未来状态。
默认数值配置的 SHA256 为
`dd15873a6cad15282da903c2e53fc46d530f906f8a78b9344ff0c458a60b0bf2`；receipt 必须同时绑定
architecture、config digest 和 checkpoint identity，避免不同 hidden size 或数值限制共用同一身份。

输入：

```text
initial_state:   float [B,15]
action_prefix:   float [B,4,8]
command_target_prefix: float [B,4,7]
transition_mask: bool  [B,4]
```

`command_target_prefix` 是 absolute joint target，必须从执行路径使用的同一 replan 起点
controller q 和同一 Action Adapter 产生，再使用与 `normalized_proprio[:7]` 相同的
`ProprioStats` q mean/std 归一化。observer 不得使用可能已滞后的 `OnlineObservation.q`
独立重算这个 reference。

`transition_mask` 只能是连续 True-prefix。Episode 尾部缺少 next state 时使用 False，不复制最后一帧，
也不允许跨 Episode 取 target。模型输出 FP32 `[B,4,15]` prediction/delta，invalid tail 固定为零。

训练目标只有等权的四步 normalized state MSE：

```text
loss = sum(mask * (prediction - target)^2)
       / (valid_transition_count * 15)
```

V0 不加入 RGB reconstruction、reward、success、critic、NLL、KL、contrastive loss 或 teacher
forcing。这样四步误差变化可以明确归因于 dynamics，而不是多个同时变化的目标。

## ActionConsistencyCritic V0

设上一条 raw Chunk 为 `P`、当前 raw Chunk 为 `C`，两次 origin control step 的差为 `d`，
`H=16`。仅当同一 Episode 且 `1 <= d < H` 时存在重叠：

```text
L = H - d
arm_TCE     = mean((P[d:H, :7] - C[:L, :7])^2)
gripper_TCE = mean((P[d:H,  7] - C[:L,  7])^2)
```

ACM 只覆盖即将执行的前四步：

```text
arm_ACM = sqrt(mean(C[:4, :7]^2))

g0 = 2 * observed_gripper_opening - 1
gripper_transition_RMS_opening_ratio =
  sqrt(mean(diff(concat(g0, C[:4, 7]))^2)) / 2
```

arm ACM 是 normalized joint-delta 幅值；gripper transition RMS 在除以 2 后使用 `[0,1]`
opening-ratio 单位，而 gripper TCE 仍在 `[-1,1]` normalized target 空间。三者保持为不同字段，
不组成一个未经校准的总分。首次 replan/reset 后状态为 `warmup`：ACM 有值，TCE 为 `null`。跨
Episode、无重叠或显式 unavailable 返回 `abstain`；非法输入在更新历史前抛错，外层
shadow observer 必须将它记录为 `error/abstain`，不能把缺失值写成 0。

这个 critic 有 0 个参数，也没有 threshold。它只建立时间对齐、reset、序列化和后续校准接口。

## Sim-to-real shadow 合同

`CandidateEvaluationRequest` 只允许包含：

- normalized 15D proprio；
- normalized absolute commanded-q prefix `[4,7]`；
- deployable physical `observed_gripper_opening_ratio`，供 absolute gripper target 计算首个 transition；
- raw/effective normalized Action Chunk；
- control step 与 observation timestamp；
- Episode、Task、Candidate、Policy Checkpoint 和 ProprioStats identity；
- `source_domain = sim | real | offline`。

请求类型没有 simulator hidden state、object GT、contact GT 或 success/failure label 字段。训练标签必须
留在单独的 offline data contract 中，不能流入同一 runtime request。

固定 Observation schema 为 `franka-normalized-proprio-command/v0`，Action schema 为
`franka-normalized-delta-q-absolute-gripper/v1`；其他 schema 不能只因 shape 相同就通过。
所有数组进入请求时都会 defensive copy 并设为只读；proprio、commanded target 与
raw/effective Chunk 都进入 request SHA256。receipt：

- 不包含 Action、controller 或 executor；
- 绑定完整 request digest，并通过 `evaluation_payload_digest` 引用单独持久化的预测/critic payload；
- `actuation_allowed` 永远为 `false`；
- 显式记录 `action_parity_equal`；若为 `false`，必须带 `action_parity_mismatch` 原因码并判定
  promotion gate 失败；
- 记录 model architecture/config digest/checkpoint、critic/calibration identity、状态、原因码、digest 与
  latency；
- 可用 `validate_against(request)` 拒绝把相同 Action 但不同 control step/timestamp 的 receipt 拼接。

当前分支有意只发布合同，尚未把 observer 注入 `QwenVLAReplanLoop`。在用户批准接入前，现有
Action Adapter、Executor、hold 和 temporal ensemble 行为完全不变。

## 当前验收

只运行候选定向测试：

```bash
PYTHONPATH=src TMPDIR=/tmp python -m pytest --noconftest \
  tests/test_candidate_evaluation.py \
  tests/test_action_consistency.py \
  tests/test_structured_world_model.py -q
```

再运行不依赖缺失 data package 的相关回归：

```bash
PYTHONPATH=src TMPDIR=/tmp python -m pytest --noconftest \
  tests/test_contracts.py \
  tests/test_adapters.py \
  tests/test_flow_matching.py \
  tests/test_temporal_ensemble.py \
  tests/test_runtime.py -q
```

这些测试只能支持“实现、shape、mask、因果、reset、合同与回归通过”，不能支持：

- trained world-model accuracy；
- critic failure detection accuracy；
- object/task outcome prediction；
- sim-to-real transfer；
- simulator 或真实机器人闭环提升；
- actuator safety。

## 后续 Gate

1. 先恢复/选择正式的 canonical trajectory package，并建立独立 train/validation data identity；
2. 训练 23K 以内的模型，验证 `mse@1..4` 是否优于 state-hold；
3. 按 Episode/scene/domain 切分，比较 sim 与 real no-actuation shadow 误差和 calibration；
4. 有可信 failure onset label 后，才把零参数特征接到 tiny learned critic；
5. 有 same-state、不同 action 的 counterfactual rollout 后，才讨论 Q/value critic；
6. critic 拒绝、rerank、hold、reobserve、进入 Planner 或取得任何 actuator 权限前，建立独立
   Decision Gate。

任何一步失败都保留现有 baseline，候选可以通过删除本分支完整回滚。
