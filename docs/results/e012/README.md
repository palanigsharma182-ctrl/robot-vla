# E012 results index

本目录按时间保留 E012 的历史 legacy formal 结果、failure diagnostic / segmented-budget
counterfactual、amended collection 与 repeat-1 训练/验证证据。不同时间点的 artifact 用途不同：后续结果不能
回写旧 formal status，diagnostic / counterfactual trajectory 不能进入 D1，checkpoint validation 也不能冒充
未运行的 Stage A/B 效果样本。

## 当前结论

- 后续 amended protocol 已通过 collection、D1 与训练前门禁。`pi_replay[1]` 和 `pi_dagger[1]` 均正式完成
  30 epochs、122,880 examples、1,920 optimizer steps，并通过 paired training verifier。
- replay exposure 仅为 `base_d0=122880`；Dagger exposure 为 `base_d0=98310`、
  `dagger_reach_grasp=12290`、`dagger_grasp_lift=12280`。D1 只含 Expert-only boundary-local anchors；D0
  `boundary_offset=null`，D1 offset 为整数 `0..48`。
- Checkpoint validation 对 pi0、两臂 epoch 10/20/30 共完成 14 组输出、315 Episodes：每个模型 20 条
  full-chain（seeds `31000..31019`）和 25 条 atomic（`31020..31024 × 5 skills`）。身份、配置、seed、Flow
  pairing 与错误扫描均通过。
- 两臂的正式 `select_e012_checkpoint` 都得到 `selection_gate_passed=false`、`selected=null`、
  `eligible_ranking=[]`。六个预注册候选均违反至少一个 guardrail，因此 promotion 在 checkpoint selection
  停止；Stage A、repeat 2、Stage B 与需要 selected pair 的 matched-state diagnostics 均未运行。
- replay e20 的 full Reach/Grasp/Lift paired net wins 为 `+8/+9/+6`，但 atomic Place 为 `-3`；Dagger e30
  为 `+2/+3/+4`，但 atomic Place 为 `-2`，并新增 anomaly / tracking saturation。它们是 model-selection
  信号，不是 Stage A 效果，不能支持“Local DAgger 改善”或“matched-state uncertainty 降低”。所有候选
  full success 都是 `0/20`。
- Replay 中断恢复的结构状态、examples/steps、exposure、offsets 与冻结身份一致；但从不可变 epoch-24
  full-state checkpoint 严格恢复后，CUDA loss/gradient 数值轨迹不再 bitwise reproducible。该限制必须随
  最终结果保留。

## 证据地图

| 证据 | 角色 | 训练/结论用途 |
|---|---|---|
| `checkpoint-validation/pi-replay-selection.json` | replay epoch 10/20/30 正式 selection receipt | 冻结 `selected=null` 与逐候选 guardrail |
| `checkpoint-validation/pi-dagger-selection.json` | Dagger epoch 10/20/30 正式 selection receipt | 冻结 `selected=null` 与逐候选 guardrail |
| `checkpoint-validation/summary.json` | 训练、315 Episodes、候选指标与 stop boundary 的脱敏 compact projection | 当前 canonical GitHub summary |
| `training-repeat-1-report/report.html` | repeat-1 训练与 checkpoint-selection 技术报告 | 当前人类可读结论 |
| `training-repeat-1-report/artifact.json` | portable report 的 canonical manifest / snapshot / sources | 可执行报告输入 |
| `collection_manifest.json`、`collection_summary.json`、`independent_validation.json` | 历史 legacy E012a formal compact evidence | 只冻结旧 `10/100` GL 容量失败 |
| `gl_failure_decomposition.json`、`gl-failure-report/` | 87-seed diagnostic replay 与 point-in-time postmortem | 禁止进入 D1 |
| `segmented-budget-smoke/`、`segmented-budget-counterfactual/` | 分段预算 smoke 与 16-seed exploratory counterfactual | `trajectory_usage=forbidden as training data` |
| `segmented-budget-report/` | 容量规划技术报告、notebook 与 portable artifact | 只读分析产物 |
| `d0_compatibility_audit.json` | frozen D0 双 projection 与 leaf content preflight | 只读身份/完整性证据 |

早期报告的 “Recommended next steps / Further questions” 是生成当时的时间点快照。后续 segmented-budget、
amended collection、repeat-1 训练和 checkpoint validation 已推进到新的 stop boundary；保留旧原文是为了
不改写历史证据，不表示旧待办仍有效。

`segmented-budget-smoke/experiment.json` 中的
`full_dataset_audit = "pending D0 union"` 是通用 accepted-trajectory contract 在当时 receipt 中的遗留状态；
对 smoke/counterfactual 的实际用途应以同一 receipt 和 summary 中更严格的
`trajectory_usage=forbidden as training data`、`successful_npz_may_enter_d1=false` 为准。它们的 D0 union
是 not applicable，不是等待执行的训练步骤。

## 身份、复算与交付 QA

正式 paired training verifier SHA-256 为：

```text
1fa9b11c184e06618bf573a984572276d0d72d83cd910e88a5f022fc47f589ff
```

两份正式 selection receipt SHA-256 分别为：

```text
pi_replay  0fdc195552e742b017d71da57974a98ff626c018d289a1f5ffa891f74e1ee838
pi_dagger  84ba2e7435438d65ebd1fb926cda21fbf69f92093021cf68cc4f0abc579586f6
```

从仓库根目录独立核对 selection receipts 与 compact summary：

```bash
python3 docs/results/e012/checkpoint-validation/verify_summary.py
```

核对 report snapshot、三个 SQLite projection 与 HTML 内嵌 payload：

```bash
python3 docs/results/e012/training-repeat-1-report/verify_report.py
```

Portable builder 的 artifact validation、packaging 与 structural verification 已通过；当前环境没有兼容的
Chromium headless-shell，因此 enhanced reader 的 desktop/narrow browser smoke 和 source-dialog interaction
没有执行。自包含 HTML 仍保留同一 canonical payload 与可读 semantic fallback；该限制属于交付 QA，不是
实验统计门禁。

历史 counterfactual 的 compact 复算仍可运行：

```bash
python3 scripts/analyze_e012_budget_counterfactual.py \
  --collection-summary docs/results/e012/collection_summary.json \
  --smoke-root docs/results/e012/segmented-budget-smoke \
  --counterfactual-root docs/results/e012/segmented-budget-counterfactual \
  --output /tmp/e012-segmented-analysis.json
```

仓库刻意不上传原始 candidate record、NPZ、图像、视频、模型权重、stdout/stderr 或 replay interruption 原始
证据；GitHub package 只保留源码、脱敏 measurement、聚合与 portable report。完整实验叙述见
[`../../experiments.md`](../../experiments.md)，协议决策见 [`../../decisions.md`](../../decisions.md)。
