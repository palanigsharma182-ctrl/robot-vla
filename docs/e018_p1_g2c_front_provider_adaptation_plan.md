# E018-P1-G2C Front Provider Adaptation 实验计划书

> 状态：`protocol-frozen / implementation-in-progress / development-only`
> 日期：2026-09-05
> Experiment ID：`E018-P1-G2C-FRONT-PROVIDER-ADAPTATION-DEVELOPMENT/v1`
> Data identity：`E018-P1-G2C-DATA/v1`
> Train identity：`E018-P1-G2C-TRAIN/v1`
> Decision Gate：[`D036`、`D037`、`D038`](decisions.md)
> 上位计划：[`E018-P1 三阶段主动视觉闭环`](e018_p1_three_stage_active_vision_closed_loop_plan.md)

> 2026-09-05 DATA Gate：`E018-P1-G2C-DATA/v1` 已通过独立 verifier 与 R2，接受为 canonical
> development-only 数据 parent。固定计数为 550 seeds、6050 eligible rows、raw/post reset diagnostics
> 550/550、2750 SafeHold-open steps 和 8800 simulator pose-set；test/Memory/runtime/actuator 均为零。DATA
> identity 为 `07919f413224fba797d4c12df25e2d5aec8ded8213e3283a07feed282701cfa3`，receipt raw/internal
> 为 `0bd4c2c6dd008889f9c02bb09e050d65b98d97620acbc8bfa5d225f1ed16e99d` /
> `0b52c3f1463087ad04275237c4567e656e698ab1043991b11d6c41d6711aa383`。当前允许 Drive 持久化和 TRAIN
> runner 实现；正式训练仍需新 source R2 GO。

本实验是 E018 Stage 2 的上游 provider 资格实验。它只回答“受限动态 front 视角是否能产生可部署语义的
object measurement”，不评价 Active 相对 Passive 的任务收益，也不授予 canonical runtime、机械臂、夹爪、
Memory 或真实相机控制权。

## 1. 背景、问题与假设

G2A 已通过动态 front capture、两阶段 prediction/GT scoring 和 identity 审计，但冻结的 E016 wrist provider
不具备 front camera 资格：

- native wrist object mean geometry 仍较好，但 covariance-95 coverage 只有 0.64，低于 0.90；
- 最佳诊断性 front viewpoint 的 object XYZ p90 约为 141.474 mm，远高于 5 mm 门槛；
- covariance calibration 只能调整 uncertainty，不能修正上述 front-domain mean error。

G2B-CAL-v2 在完整 1135-frame skill-0 cohort 中遇到一个 reset-first-frame contact cache transient。该运行按
预冻结规则已整体收口为 protocol-invalid；不得事后删行或在已读取的 E016 validation 上更改 cohort。

本实验假设为：

> 使用全新、seed-disjoint、free-static/pregrasp 数据对同一 Precision 模型家族做 front-domain supervised
> adaptation 后，至少一个已经通过 G0C 动态运动门禁的 non-HOME alternate 能达到 object world-XYZ p90
> 不高于 5 mm，并在冻结 covariance/write gate 后保持零 unsafe/catastrophic accepted。

## 2. Control、Treatment 与冻结变量

### 2.1 Control

Control 是冻结的 E016-P1 epoch-12 wrist checkpoint 经原 role-substitution adapter 在 G2C 新数据上的
baseline。它只用于量化 domain gap，永不参与 checkpoint selection，也不能获得新 provider identity。

### 2.2 Treatment

读取 model-validation label 前固定两个候选：

- `W-KV0`：`PrecisionThreeHeadUNet`，从 E016 epoch-12 warm start 后，只将 final uncertainty Linear 的
  全部 keypoint-logvariance output rows 的 weight/bias 确定性置零；
- `S`：相同架构，random initialization。

两个候选只改变 initialization。两者共享数据、tensor/loss 语义、optimizer、epoch 预算和选择规则。沿用
E016 corrected-observability contract；Motion Head frozen-zero/shadow-only。Goal Head 最多保留为既有辅助
输出，本实验不允许它写 Goal Memory、参与 qualification 或接入 consumer。只有 object measurement 可以
获得候选 provider identity。

### 2.3 不变项

- Qwen、Action Expert、VLA Action、Observation V2 不变；
- canonical Dataset schema、Label、正式 success/failure 和 test-once 不变；
- G0C-v2 的相机位姿、轨迹、速度、加速度、tracking 和 HOME-return 门槛不变；
- 不读取 E016 val/test、G2A output 或后续 Active-vs-Passive 结果进行训练或调参；
- runtime viewpoint selection、provider input 和控制逻辑不得读取 simulator hidden GT。

## 3. Viewpoint 与数据 split

11 个 front pose 固定为：

```text
HOME
LEFT_LOW__CENTER
LEFT_LOW__YAW_LEFT
LEFT_LOW__YAW_RIGHT
LEFT_LOW__PITCH_UP
LEFT_LOW__PITCH_DOWN
RIGHT_LOW__CENTER
RIGHT_LOW__YAW_LEFT
RIGHT_LOW__YAW_RIGHT
RIGHT_LOW__PITCH_UP
RIGHT_LOW__PITCH_DOWN
```

split、seed 和用途在采集前冻结：

| Split | Seeds | Pose/seed | Eligible/scored rows | 唯一用途 |
|---|---:|---:|---:|---|
| train | 76001–76400 | 11 | 4400 | 拟合 W-KV0/S |
| model validation | 76501–76600 | 11 | 1100 | eligibility 与 checkpoint ranking |
| per-view calibration | 76601–76650 | 11 | 550 | covariance 与 write threshold |
| one-shot qualification | 76701–76750 | 11 | 550 | 最终 development-only qualification |
| test | none | — | 0 | 禁止读取 |

必须把以上 seed 与 canonical manifest、E016、G0/G0B/G0C/G1A、G2A `75001..75050` 和 artifact inventory 中
全部已登记 development/test seed 做机械化 disjoint audit。任何 overlap 都使对应 split protocol-invalid。

`G2C-DATA/v1` 先冻结 schema、seed、视角、lifecycle 和 label 隔离规则。DATA receipt 通过后，`G2C-TRAIN/v1`
的 config 必须机械绑定 dataset manifest 和每个输入文件 SHA；不得在训练后补写 identity。

## 4. Reset、采集与 label 隔离

### 4.1 固定 reset lifecycle

每个新 Episode：

1. 执行 environment reset；
2. 把 raw reset-return observation 写入 `reset_diagnostic`；
3. 执行 5 个 20 Hz SafeHold-open warmup steps；
4. 把 post-warmup observation 写入另一条 diagnostic；
5. 只对 warmup 后、通过 settle rule 的 capture 标记 `eligible=true`。

raw reset observation 永不进入 train、selection、calibration 或 qualification。这个规则用于规避已经定位的
ManiSkill reset-first-frame contact cache transient，必须由 config 和 verifier 在采集前强制，不是看到结果后
删除异常行。

每 seed 必须恰有一条 raw-reset 和一条 post-warmup diagnostic，两条都必须 `eligible=false` 且所有用途标志
为 false。smoke 计数固定 raw/post/total=`4/4/8`，full DATA 固定为 `550/550/1100`；缺失、重复或计数
不一致均 fail closed。

### 4.2 Static splits

train、model validation 和 calibration 可以使用 static-render-only pose configuration。每个 pose 仍必须：

- 从产生 RGB 的同一 observation 读取 actual external pose；
- 保存 intrinsic、RGB timestamp 和 pose timestamp；
- 显式完成 OpenGL camera 到 OpenCV optical frame，再转换到 base frame；
- 在冻结 settle rule 后只生成一个 eligible row；
- 证明 arm/TCP hold、open gripper 和 no contact。

5 个 20 Hz warmup 后真实 simulation-control-time 为 0.25 s。static-render viewpoint 之间不执行
environment step，因此 11 个 view 可以保留同一个真实 timestamp；顺序由 `sample_index` /
`capture_sequence` 表达。禁止用 sample index 增加微秒、回填时间或制造伪单调 timestamp。

### 4.3 Split fail-whole invariant

每个 eligible capture 必须满足：

```text
object position finite
object z = 0.02 ± 1e-5 m
is_grasped = false
finger-force-valid = true
F_L <= 0.01 N and F_R <= 0.01 N
raw gripper opening >= 0.95
arm/TCP hold
no object contact
RGB/pose skew within frozen limit
geometry/provider identity complete
```

任一 eligible row 失败，整个 split protocol-invalid；不删除 row、不补 seed、不替换 route。

### 4.4 Privileged label sidecar

RGB/deployable inputs 与 privileged mask、position、observability label 分开写入并分别 hash。model
validation、calibration 和 qualification 都必须：

```text
run deployable inference
  -> fsync prediction ledger file
  -> fsync parent directory
  -> freeze ledger SHA and destroy inference/data context
  -> only then open privileged label arrays
  -> write a separate scoring ledger
```

prediction-before-label 顺序一旦破坏，整个阶段 protocol-invalid。

## 5. 训练预算与 checkpoint selection

训练参数固定为：

| 参数 | 值 |
|---|---|
| optimizer | AdamW |
| precision | BF16 |
| batch size | 32 |
| epochs | 20 / candidate |
| learning rate | 3e-4 |
| weight decay | 1e-4 |
| gradient clip | 1.0 |
| scheduler | cosine annealing，eta-min 5% |
| num workers | 0 |
| spatial augmentation | none |
| W-KV0/S initialization/run seed | 18021 / 18022 |
| shared sampler seed | 18020 |
| candidate epochs | 5, 10, 15, 20 |

两条 treatment 最多共 40 model-epochs；RTX 6000 Ada GPU execution 上限 10 小时；data + artifact 上限
20 GB。正式数据前允许 4-seed、无持久 checkpoint engineering smoke。

checkpoint eligible 要求至少一个 G0C motion-qualified non-HOME view 同时满足：

```text
object visibility precision >= 0.95
object visibility recall >= 0.90
observable-positive support N >= 30
observable world-XYZ p90 <= 0.005 m
observable world-XYZ max <= 0.020 m
all geometry finite and valid
```

W-KV0/S 必须使用 `sampler_seed=18020` 产生完全相同的逐 epoch shuffle 顺序；`18021/18022` 只驱动各自的
initialization/run RNG，不能驱动 sampler。某 viewpoint 的 observable-positive support 小于 30 时仅该
viewpoint 因 `insufficient_observable_positive_support` 不合格；如果因此没有任何 non-HOME eligible
viewpoint，则 `selected=null` 并收口为 protocol-valid model-selection negative。只有 support 计数与冻结
label/ledger 不一致或指标实现异常才使 selection protocol-invalid。

只在 model-validation split 上按下列固定顺序排名：

1. eligible alternate 数量降序；
2. 最佳 alternate p90 升序；
3. 对应 XYZ max 升序；
4. validation loss 升序；
5. 较早 epoch；
6. 完全相同时 W-KV0 优先于 S。

全部 checkpoint 和 SHA 必须在读取 validation label 前冻结。无 eligible checkpoint 时 selected checkpoint
必须为 null；不得选“最不坏”候选、临时增加 epoch 或查看 qualification 后重选。

## 6. Per-view covariance 与 write threshold

只有 selected checkpoint SHA 冻结后，才允许打开 calibration split。每个 viewpoint 独立拟合 scalar XY
Mahalanobis split-conformal covariance：

```text
alpha = 0.05
target coverage = 0.95
chi-square = 5.991
k = ceil((N + 1) * 0.95)
support N >= 30
scale = max(1, q / 5.991)
maximum calibrated position std <= 0.020 m
```

同一 calibration split 按冻结的 E018 object write-score semantics，逐 viewpoint 选择“accepted-safe coverage
最大且 unsafe accepted=0”的 threshold；coverage 必须至少 0.10，tie 选择更高、更保守的 threshold。

confidence、write acceptance 或 prediction error 不参与 checkpoint selection。qualification 后不得修改
covariance scale、visibility/projected threshold、write threshold 或 viewpoint pool。

## 7. 动态 one-shot qualification

### 7.1 路线与统计样本

qualification 使用 50 个新 seed。每个 seed 对 10 个 non-HOME alternate 分别执行一条独立 reset 的
`HOME -> alternate -> HOME` route，共 500 routes。每 route/Episode 只有一次 active attempt；不重试、不补
seed、不替换 route、不做 alternate-to-alternate。

每条 route 固定为：

```text
initial camera pose set: 1
HOME warmup: 5 ticks
HOME anchor: 1 frame
outbound: 40 ticks / 2.0 s
alternate settle: 4 ticks
collect: 3 ticks
return HOME: 40 ticks / 2.0 s
HOME verify: 4 ticks
```

固定审计计数：

| 事件 | 每 route | 500 routes |
|---|---:|---:|
| camera pose-set | 97 | 48,500 |
| moving interpolation command | 80 | 40,000 |
| simulator SafeHold-open step | 96 | 48,000 |
| ledger frame | 92 | 46,000 |

provider 评分只取：

- 每个 seed 固定首条 `LEFT_LOW__CENTER` route 的 HOME anchor，共 50；
- 每条 route 第三个、最后一个 settled COLLECT frame，共 500。

总 scored frames=550。其余 45,450 帧只用于时序、运动和安全审计，不扩大 provider 统计 N。

### 7.2 Motion parent 与门槛

qualification 允许且只允许 isolated ManiSkill RenderCamera 复用 G0C-v2 路线。父 identity 固定绑定：

- G0C config SHA：`c93bbfd48b6d9bc2fc75b5b87e4ded7161efebd7eda50cd81cc2ded47810e965`；
- G0C receipt SHA：`bf8232b620cd5ff8de8c0007391252b8829c3ebbac320a7d5a60507beaca258e`。

运动和保持门槛：

| 指标 | 上限 |
|---|---:|
| linear velocity | 0.31 m/s |
| linear acceleration | 0.70 m/s² |
| angular velocity | 0.75 rad/s |
| angular acceleration | 2.50 rad/s² |
| camera tracking | 1e-5 m / 1e-4 rad |
| settled velocity，连续至少 3 ticks | 0.002 m/s / 0.01 rad/s |
| HOME error | 1e-5 m / 1e-4 rad |
| arm q drift | 1e-5 rad |
| TCP drift | 1e-5 m / 1e-4 rad |
| finger contact | 0.01 N |

任一 route 出现 safety、identity、lifecycle、termination 或 truncation 失败，整个 qualification
protocol-invalid；不得删除或重跑该 route。

### 7.3 权限计数

仿真相机运动必须如实记录 48,500 pose-set 和 40,000 moving commands；不能把它们写成零。与此同时，下列
权限必须保持为零：

```text
runtime/canonical camera-controller actuation
physical camera actuation
nonzero arm-motion command
gripper-close command
manipulation progression
object-contact event
Object/Goal Memory read or write
test-array read
```

SafeHold-open simulator step 本身按 48,000 如实记录。

## 8. Qualification Gate 与 PRIMARY

每个 viewpoint 必须同时满足：

| 指标 | Gate |
|---|---:|
| visibility precision | >= 0.95 |
| visibility recall | >= 0.90 |
| observable XYZ p90 | <= 0.005 m |
| observable XYZ max | <= 0.020 m |
| unsafe accepted | 0 |
| catastrophic accepted | 0 |
| accepted-safe coverage | >= 0.10 |
| covariance-95 coverage | >= 0.90 |
| covariance support | >= 30 |
| maximum calibrated std | <= 0.020 m |

至少一个 non-HOME alternate 通过才允许产生 PRIMARY。PRIMARY 只在 qualified 集中按 frozen shortlist tier、
coverage 降序、p90/max 升序、`|cov95-0.95|` 升序、recall 降序和 frozen pose order 选择。G2A 的
native-wrist parent-health 不阻断这个 front-trained provider；Control 结果仍只作诊断。

qualification 只运行一次。无 eligible checkpoint、无可校准 viewpoint、零 qualified alternate、预算超限或
任何 protocol violation 都必须冻结 negative/protocol-invalid receipt；不得现场改阈值、加 epoch、补 seed
或切模型。

## 9. 实施顺序与审查点

工程顺序固定为：

1. 实现 DATA config、collector、writer、verifier 和 label sidecar；
2. 实现 seed/split/lifecycle/pose/timestamp/identity/disjoint audits；
3. 实现 W-KV0/S train、model-validation、per-view calibration 和 dynamic qualification 接口；
4. 运行 targeted unit tests 和工程自审；
5. 运行 4-seed、无持久 checkpoint smoke；
6. 在 full-data 前停止，提交 DATA/sampling/label/identity evidence 给决策 Agent 做一次 R2 抽样；
7. R2 通过后收集 full DATA 并冻结 receipt；
8. 由 receipt 机械冻结 train config，训练 W-KV0/S；
9. 冻结 checkpoint selection，再做 calibration；
10. 冻结 calibration/threshold 后执行一次 qualification；
11. verifier、Drive/local 持久化和 evidence packet 完成后才关闭本 Gate。

工程 Agent 完成常规代码审查与完整 targeted tests。决策 Agent 只抽查 Dataset/label/sampling、坐标/时间、
checkpoint/result identity、安全计数和实验归因，审查预算控制在总工作量约 10%–15%。

## 10. 允许结论、失败回退与完成边界

PASS 只允许声明：

> 至少一个冻结离散 alternate 获得 simulation development-only front object-provider 资格，可以作为新 parent
> 进入 E018 Stage 2 information-gain、pending candidate 和 Object Memory no-test 实现与验证。

不得声明主动视觉优于 Passive、任务成功率提升、正式闭环、canonical promotion、actuator 安全、真实相机
动力学或真实机器人安全。

若 W-KV0/S 均失败：

1. 冻结 control/candidate/data/ledger/checkpoint/receipt 和所有负结果；
2. 不复用 qualification 调整当前模型；
3. 由决策 Agent 建立新的独立 B Gate；
4. 优先评估 deployable object-mask centroid decoder 或更小 object-only provider；
5. provider parent 的正式训练路线总数限制为 2–3 次，超过后以证据化 provider no-go 收口。

单条 provider 路线失败不终止 E018 的其余独立安全、回放和失败归因工作。任何 canonical 晋升、稳定
schema/label 改变、fresh formal test 或 actuator 权限仍属 A 级，必须返回用户决定。

关键 dataset、selected checkpoint 和不可复现实验证据必须按项目 storage policy 达到 Drive 与本机双验证
副本；未达到对应 release gate 前不释放唯一源。公开文档只包含脱敏 config、聚合指标和 hash，不包含原始
RGB/NPZ、checkpoint、私有路径、凭据或存储拓扑。
