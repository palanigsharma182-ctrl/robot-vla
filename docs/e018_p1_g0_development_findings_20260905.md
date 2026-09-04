# E018-P1 G0 Front Camera Feasibility 开发结果（2026-09-05）

## 结论

E018-P1 的 **G0 Simulator/API feasibility 已通过**。在美国 RTX 6000 Ada 上，ManiSkill 3.0.1 默认的
独立 `base_camera` 可以在不移动 Franka arm/TCP 的条件下按 20 Hz 时序更新实际位姿；相机外参和 RGB
会随逐 Tick 命令同步变化，且能精确返回 HOME。

本结果不是 P1 正式实验，也不代表 front provider 已经合格。四个视角仍是 provisional development
geometry；`RIGHT_LOW` 和 `RIGHT_HIGH` 在部分 seed 上出现 object 或 goal 的零可见像素，不能直接冻结为
正式 viewpoint library。

```text
G0 gate:                         passed
routes:                          16 / 16 passed
frames:                          1152
test episodes / test reads:      0 / 0
provider forward count:          0
Memory write count:              0
physical robot actuation:        false
manipulation progression:        false
formal claim allowed:            false
ready for G1 engineering:        true
ready for formal preregistration:false
```

## 实验范围与方法

配置：[`configs/e018_p1_g0_camera_feasibility_v1.json`](../configs/e018_p1_g0_camera_feasibility_v1.json)

实验只验证相机 API、运动时序和 hold 不变量，不加载 E016 checkpoint、不运行 provider、不读取任何数据集，
也不接入 Goal/Object Memory 或 Executive。每条路线使用一个全新 reset：

```text
5 Tick HOME warmup（不计入 ledger）
-> HOME anchor frame
-> 30 Tick / 1.5 s quintic move
-> 4 Tick settle
-> 3 Tick collect
-> 30 Tick / 1.5 s return
-> 4 Tick HOME + arm-hold verify
```

每条路线共记录 72 帧、3.55 秒相对仿真时间；4 个 development seeds 与 4 个备用锚点构成 16 条路线。
同一 episode 只允许 `HOME -> 一个备用锚点 -> HOME`，没有在备用锚点之间串行移动。

相机使用固定 workspace target `[-0.1, 0.0, 0.1] m`。平移按五次 smootherstep 展开，yaw/pitch 在每个
位置由预注册 target 确定，roll 固定为 0。每帧分别记录 commanded pose 与来自
`sensor_param.base_camera.cam2world_gl` 的 actual pose，并转换出 OpenCV optical frame 的动态
`base_from_external_camera_cv`。

## 2×2 provisional 视角

| 视角 | world position (m) | yaw / pitch (rad) | alternate RGB mean-abs diff 范围 | collect object pixels 最小值 | collect goal pixels 最小值 |
|---|---:|---:|---:|---:|---:|
| `LEFT_LOW` | `[0.30, -0.16, 0.48]` | `2.761 / -0.723` | `28.262–28.583` | 31 | 20 |
| `LEFT_HIGH` | `[0.30, -0.16, 0.72]` | `2.761 / -0.964` | `21.847–22.115` | 16 | 6 |
| `RIGHT_LOW` | `[0.30, 0.16, 0.48]` | `-2.761 / -0.723` | `29.382–29.857` | 18 | 0 |
| `RIGHT_HIGH` | `[0.30, 0.16, 0.72]` | `-2.761 / -0.964` | `21.642–22.028` | 0 | 0 |

RGB 差值远高于预设 `5.0` 门槛，证明渲染画面确实随视角变化，而不是只更新了 pose metadata。返回
HOME 后与出发前 RGB 的 mean absolute difference 在全部 16 条路线中均为 `0.0`。

object/goal pixel 只在命令和图像已经产生后由 segmentation 计算，用于离线诊断；轨迹选择、相机命令、
settle 和安全门禁均不读取这些 GT 信息。零像素结果说明右侧路线容易被 Franka 本体遮挡或把任务实体推到
视野边缘。这是 G5 视角可行性的重要负证据：后续应在 development 中检查更外移的 x/y 基线或重新设定
固定 workspace ROI，然后在 fresh validation 前冻结，而不是因为 G0 通过就沿用当前数值。

## 运动、跟踪与 hold 结果

| 指标 | 最坏值 | 门槛 | 结果 |
|---|---:|---:|---|
| camera position tracking error | `3.1200e-8 m` | `<= 1e-5 m` | 通过 |
| camera orientation tracking error | `5.1619e-8 rad` | `<= 1e-4 rad` | 通过 |
| linear velocity | `0.2493 m/s` | `<= 0.31 m/s` | 通过 |
| linear acceleration | `0.5102 m/s²` | `<= 0.70 m/s²` | 通过 |
| angular velocity | `0.5265 rad/s` | `<= 0.75 rad/s` | 通过 |
| angular acceleration | `1.0745 rad/s²` | `<= 2.50 rad/s²` | 通过 |
| Franka arm joint drift | `0.0 rad` | `<= 1e-5 rad` | 通过 |
| TCP position drift | `0.0 m` | `<= 1e-5 m` | 通过 |
| TCP orientation drift | `2.9802e-8 rad` | `<= 1e-4 rad` | 通过 |
| finger-object contact force | `0.0 N` | `<= 0.01 N` | 通过 |
| non-collect frame write eligibility | `0` | `0` | 通过 |
| Memory writes | `0` | `0` | 通过 |
| terminated/truncated frames | `0` | `0` | 通过 |

只有连续 3 Tick 同时满足 pose tracking、线速度和角速度门槛后，帧才被标记为 settled；只有 settled
`COLLECT` 帧在数据契约中可成为 measurement candidate。G0 本身继续把实际 Memory write 全部禁用。

## v1 数值审计失败与修正

首轮完整目录 `g0-camera-feasibility-development-v1` 被门禁正确保留为失败：12/16 路线报告 TCP
orientation drift `2.99e-4–7.54e-4 rad`，但对应帧从 frame 0 开始就出现该值，且 7 个 arm joint 的逐位
漂移始终为 0。

根因是 SAPIEN float32 pose matrix 在 `validate_se3` 容差内存在轻微非正交，直接使用
`acos((trace(R1ᵀR2)-1)/2)` 会把同一矩阵的舍入误差放大成非零旋转。修正是先通过 SVD 极分解把输入投影到
最近的 SO(3)，再计算相对转角，并增加“同一轻微非正交矩阵距离为 0”的回归测试。修正后的 v2 其余配置、
seed 和相机轨迹完全不变，TCP orientation 最坏值降为 `2.98e-8 rad`。

## 工件与身份

GPU 结果目录：

```text
/root/robot-vla-runs/e018-active-front-reobserve/g0-camera-feasibility-development-v2
```

关键身份：

```text
config SHA-256:
3976a668566ed2950fb1e8b34e15842eea4c96b542264cd4cf1788f660cb9629

receipt SHA-256:
4e3e6639696b6c76e427c613edeb92cf2311068e1ee2115b34c14a91a1941ae8

Git parent:
cef61f0a6216b8504215a6745ab5a3925bfb0cc0
```

`receipt.json` 绑定 53 个结果文件；复核结果为 53/53 SHA-256 匹配。核心工件包括：

- `camera_pose_ledger.jsonl`：1152 帧 commanded/actual pose、动态 CV 外参、时间、运动与 hold 审计；
- `route_summaries.jsonl`：16 条路线的逐项 gate；
- `viewpoint_contact_sheet.png`：4 seed × `HOME + 2×2` 的 settled RGB；
- `images/`：每条路线的 HOME-before、alternate、HOME-after RGB；
- `summary.json`、`config_snapshot.json`、`receipt.json` 与 fail-closed `run_state.json`。

## 能证明与不能证明的内容

G0 已证明：

- 当前 ManiSkill/SAPIEN 组合允许独立 external camera 逐 Tick 动态变更 actual pose；
- 相机可以用非零时延路径完成 `HOME -> alternate -> HOME`；
- 每帧能记录动态 actual extrinsic，而不是复用 Episode 固定外参；
- 该运动探针下 arm/TCP/open gripper 可保持不动；
- motion、unsettled、return 和 HOME-verify 帧不会成为 Memory 写入候选。

G0 尚未证明：

- 真实相机滑台/云台的动力学、回差、标定或故障语义；
- 相机实体与机械臂的碰撞包络安全；
- E016 wrist checkpoint 或任何 front provider 在这些视角上的定位可靠性；
- Active 相比 matched Passive 的信息恢复增益；
- Executive 阶段 allowlist、episode latch、Action Chunk 清空与 closed-loop resume；
- 完整抓取成功率或任何正式 test claim。

因此，代码层面可进入 G1 dynamic-external-camera Observation schema；但 P1 的 provider、active motion 与
paired closed-loop 仍受 P0 正式前置条件和后续 Gate 约束，当前不得运行 P1 formal/test-once。
