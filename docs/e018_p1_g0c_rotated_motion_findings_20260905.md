# E018-P1 G0C 离散旋转位姿动态运动结果（2026-09-05）

## 结论

E018-P1 的 **G0C rotated-pose motion feasibility 已通过**。以下结论以审计加固后的 v2 结果为准。
G0B 静态筛选出的 10 个低位完整位姿均完成
4 个新 development seeds 上的有时延 `HOME -> full pose -> HOME` 路线，共 `40/40` 条路线、
`3680` 个 ledger frames 通过全部门禁。

平移和局部 yaw/pitch offset 共用同一个五次 smootherstep 进度。新增旋转没有在首帧瞬时施加，也没有在
返回 HOME 时突然清零；逐帧 actual pose、速度、加速度、settle、机械臂/TCP hold、open gripper、接触和
写入资格均被记录。

```text
gate:                              G0C_ROTATED_POSE_MOTION_FEASIBILITY
status:                            complete-development-only
routes:                            40 / 40 passed
frames:                            3680
translation anchors:               2
orientation modes:                 5
full alternate poses:              10
test episodes / test reads:        0 / 0
provider forwards:                 0
Memory reads / writes:             0 / 0
physical robot actuation:          false
formal claim allowed:              false
```

该结果只证明当前仿真 camera API 下的时序运动与 SafeHold 契约可行。RenderCamera 没有真实质量、碰撞体、
驱动器回差或标定噪声，因此不代表真实相机滑台/云台安全，也不代表 front provider 已经能从这些图像可靠
定位 object/goal。

## Parent 与运行范围

配置：
[`configs/e018_p1_g0c_rotated_motion_development_v1.json`](../configs/e018_p1_g0c_rotated_motion_development_v1.json)

G0C config 直接绑定已通过的 G0B：

```text
G0B config SHA-256:
b413487cc35a8ffb8bbaeba8ec6401bbe76eb885b6a3774d1e53b16e18b058a3

G0B receipt SHA-256:
faf4609954cb1be33e16198ee7b305c7e8f5fabd58a8a48710fd30035f5309ea
```

只有 G0B 的 10 个 `STATIC_ELIGIBLE_POOL` 位姿进入本轮：

```text
LEFT_LOW  × {CENTER, YAW_LEFT, YAW_RIGHT, PITCH_UP, PITCH_DOWN}
RIGHT_LOW × {CENTER, YAW_LEFT, YAW_RIGHT, PITCH_UP, PITCH_DOWN}
```

高位锚点没有进入 G0C，因为其五种朝向均未通过 G0B 静态构图阈值。G0C 使用新的 development seeds
`73001 / 73013 / 73027 / 73039`，不读取任何 train/validation/test 数据集，不加载 E016 checkpoint，
不运行 provider，也不读取或写入 Goal/Object Memory。

## 路线与插值

每条路线使用一个独立 environment reset：

```text
5 Tick HOME warmup（不进入 ledger）
-> 1 HOME anchor frame
-> 40 Tick / 2.0 s outbound translation + local rotation
-> 4 Tick settle
-> 3 Tick collect
-> 40 Tick / 2.0 s return translation + local derotation
-> 4 Tick HOME and arm-hold verify
```

共记录 92 frames，最后一个 ledger timestamp 为 `4.55 s`。相机位置使用五次 smootherstep；目标局部
yaw/pitch offset 使用完全相同的进度：

```text
outbound: offset(t) = target_offset × smootherstep(t)
return:   offset(t) = target_offset × [1 - smootherstep(t)]
```

每个 Tick 的 nominal orientation 仍指向冻结 workspace ROI，离散 offset 再在 SAPIEN camera-local
`+X forward / +Y left / +Z up` 坐标中合成。由于首轮每个 mode 最多只有一个非零 yaw 或 pitch，
“缩放角度”正好是该局部单轴旋转的最短插值，不包含 roll 或 diagonal rotation。

路径审计结果：

```text
40/40 outbound progress strictly nondecreasing
40/40 return progress strictly nonincreasing
40/40 outbound endpoint progress = 1
40/40 return endpoint progress = 0
max first-outbound-tick angular speed = 0.0017075730 rad/s
max first-HOME-verify angular speed     = 0.0 rad/s
path audit failures                    = 0
```

因此没有把静态 pose offset 在运动开始或结束边界瞬间跳变。

## 动力学、跟踪与 SafeHold 门禁

| 指标 | 最坏值 | 门槛 | 结果 |
|---|---:|---:|---|
| camera position tracking error | `3.0901e-8 m` | `<= 1e-5 m` | 通过 |
| camera orientation tracking error | `5.1619e-8 rad` | `<= 1e-4 rad` | 通过 |
| target orientation endpoint error | `4.2147e-8 rad` | `<= 1e-4 rad` | 通过 |
| linear velocity | `0.18719 m/s` | `<= 0.31 m/s` | 通过 |
| linear acceleration | `0.28744 m/s²` | `<= 0.70 m/s²` | 通过 |
| angular velocity | `0.53907 rad/s` | `<= 0.75 rad/s` | 通过 |
| angular acceleration | `0.82521 rad/s²` | `<= 2.50 rad/s²` | 通过 |
| Franka arm joint drift | `0.0 rad` | `<= 1e-5 rad` | 通过 |
| TCP position drift | `0.0 m` | `<= 1e-5 m` | 通过 |
| TCP orientation drift | `2.1073e-8 rad` | `<= 1e-4 rad` | 通过 |
| finger-object contact force | `0.0 N` | `<= 0.01 N` | 通过 |
| return-HOME RGB difference | `0.0` | `<= 0.1` | 通过 |
| non-collect write-eligible frames | `0` | `0` | 通过 |
| actual Memory writes | `0` | `0` | 通过 |
| terminated/truncated frames | `0` | `0` | 通过 |

2.0 秒单程是预运行设计选择。若沿用原 G0 的 1.5 秒，额外 12° offset 可能把 nominal look-at 转动与局部
旋转叠加到原 `0.75 rad/s` 上限附近；本轮选择延长时长，而不是为了通过结果而放宽速度/加速度门槛。

## 逐完整位姿动态结果

下表取四条 seed 路线的最坏值。object/goal pixel 仅是 collect 后的 oracle segmentation 诊断，不参与
运动、安全、settle 或候选选择。

| 完整位姿 | 最大角速度 rad/s | 最大角加速度 rad/s² | endpoint error rad | object min px | goal min px |
|---|---:|---:|---:|---:|---:|
| `LEFT_LOW__CENTER` | `0.39536` | `0.60680` | `0` | 24 | 12 |
| `LEFT_LOW__YAW_LEFT` | `0.31443` | `0.48613` | `0` | 29 | 13 |
| `LEFT_LOW__YAW_RIGHT` | `0.53907` | `0.82521` | `0` | 24 | 12 |
| `LEFT_LOW__PITCH_UP` | `0.46521` | `0.72141` | `0` | 27 | 15 |
| `LEFT_LOW__PITCH_DOWN` | `0.36248` | `0.57322` | `4.2147e-8` | 24 | 12 |
| `RIGHT_LOW__CENTER` | `0.39536` | `0.60680` | `0` | 31 | 21 |
| `RIGHT_LOW__YAW_LEFT` | `0.53907` | `0.82521` | `0` | 34 | 21 |
| `RIGHT_LOW__YAW_RIGHT` | `0.31443` | `0.48613` | `0` | 31 | 22 |
| `RIGHT_LOW__PITCH_UP` | `0.46521` | `0.72141` | `0` | 35 | 21 |
| `RIGHT_LOW__PITCH_DOWN` | `0.36248` | `0.57322` | `4.2147e-8` | 30 | 20 |

左右两侧出现对称关系：与 nominal path 转动方向相反的 yaw 会降低峰值角速度，同向叠加的 yaw 会形成
全局最坏值 `0.53907 rad/s`，但仍保留约 28% 的门槛余量。这个结果证明路线预算覆盖了旋转方向差异，
不能用于判断哪个图像视角对 provider 最好。

## 工程修改

G0C 没有复制一套独立运动控制逻辑，而是在已有 G0 route engine 上增加以下参数化能力：

- nominal look-at 后叠加可选的 camera-local `FrontCameraOrientationMode`；
- outbound/return 共享五次 orientation progress；
- 每帧记录 target orientation id、progress 和实际 yaw/pitch/roll offset；
- route gate 显式验证 alternate endpoint 与注册完整姿态一致；
- result version、episode prefix、source phase 和 camera owner 可由具体 Gate 注入。

旧 G0 config 仍默认使用 `CENTER`、零 offset；现有严格 2×2 config validator 与回归测试保持不变。G0C
另有独立 config loader、CLI、parent identity 和输出版本，避免静默改变历史 G0 结果语义。

## v1 配置审计缺口与 v2 加固

首轮 v1 动态结果本身使用了正确的 G0B 坐标与 ±12°/±8° offset，config snapshot 和 SHA-256 也完整；
但运行后复核发现，严格 loader 只绑定了 primitive ID，没有拒绝“ID 保持不变、数值坐标或角度被修改”的
假配置。这个缺口不改变 v1 已执行轨迹，却不足以作为后续 Gate 的 fail-closed parent。

修正后 loader 逐项绑定 HOME、两个低位平移锚点、五个 orientation mode 的精确数值与顺序，并新增
anchor numeric drift 和 orientation numeric drift 负例测试。相同 config、seed 和运动参数在新目录执行
v2：

```text
v1 vs v2 camera_pose_ledger.jsonl:   byte-identical
v1 vs v2 route_summaries.jsonl:      byte-identical
v1 vs v2 viewpoint_contact_sheet:    byte-identical
```

因此 v2 只加固了输入拒绝语义，没有改变轨迹或运动结论。v1 保留为审计记录，后续 parent 应绑定 v2。

远端定向回归：

```text
Ruff: all checks passed
65 passed
```

覆盖 G0/G0B/G0C active camera、P0 Object Memory、Object observability/evaluation 与冻结 Goal Memory。

## 工件、身份与备份

GPU 结果目录：

```text
/root/robot-vla-runs/e018-active-front-reobserve/
g0c-rotated-motion-development-v2
```

Drive 目录：

```text
gdrive:VLA/experiments/e018/
e018-p1-g0c-rotated-motion-development-v2-20260905
```

关键身份：

```text
config SHA-256:
c93bbfd48b6d9bc2fc75b5b87e4ded7161efebd7eda50cd81cc2ded47810e965

receipt SHA-256:
bf8232b620cd5ff8de8c0007391252b8829c3ebbac320a7d5a60507beaca258e

source identity SHA-256:
0c8cba2e2e36fac1accadc81b153c7a0ca3ffdf0798ac8a437e66d9cdff0fa7a

Git parent:
cef61f0a6216b8504215a6745ab5a3925bfb0cc0
```

`receipt.json` 绑定 125 个文件，GPU 本地逐文件 SHA-256 复核为零差异。Drive 下载式校验为
`127 matching files / 0 differences`，另两个文件是 receipt 与 fail-closed `run_state.json`。

被取代的 v1 审计目录仍保留在 GPU 与 Drive：

```text
/root/robot-vla-runs/e018-active-front-reobserve/g0c-rotated-motion-development-v1
gdrive:VLA/experiments/e018/e018-p1-g0c-rotated-motion-development-v1-20260905
v1 receipt SHA-256: 636f3925227a6a1737de3d106d8da793b8c60633d381d1f9ecf637584e398e35
```

## Promotion 边界与下一步

G0C 通过后可以进入 G1 Observation schema 工程，但仍不能进入正式 Active-vs-Passive 比较。下一步顺序：

1. G1 冻结动态 external-camera observation schema：actual extrinsic、RGB/pose timestamps、motion state、
   settled/write eligibility 与 replay round-trip。
2. G2 用冻结模型逐视角验证 front provider。必须报告 object/goal observability、base-frame XYZ error、
   covariance、false recovery 和 unsafe acceptance；几何可见不能替代这一门禁。
3. 只有 provider 合格的位姿才能进入 G3 controller/state-machine 和之后的 fresh validation schedule。
4. P0 尚未通过正式前置门禁，所以 P1 formal preregistration、paired comparison 和 test-once 继续禁止。

G0C 不能支持以下结论：

- 10 个候选都应该保留到正式运行，或当前四个优先候选已经最优；
- simulator RenderCamera 路径已证明真实滑台/云台的碰撞、回差和标定安全；
- front provider 在约 13–20 个 p10 entity pixels 上仍能可靠定位；
- runtime 可以基于 hidden GT 选择视角；
- 主动移动相对 matched Passive 已提高信息恢复率；
- `FINAL_APPROACH`、contact、close 或 grasp 后可以继续主动观察。
