# E018-P1 G0B 离散平移与旋转视角筛选结果（2026-09-05）

## 结论

E018-P1 的 **G0B static viewpoint geometry screen 已通过完整性门禁**。本轮不是只测试四个相机位姿，
而是在 `HOME + LEFT/RIGHT × LOW/HIGH` 五个平移位置上分别组合
`CENTER / YAW_LEFT / YAW_RIGHT / PITCH_UP / PITCH_DOWN` 五种离散朝向，共筛选 25 个完整相机位姿。

50 个 development seeds 的结果表明：

- 25 个位姿均能按命令设置并精确读取实际外参，重复渲染完全确定；
- `LEFT_LOW` 与 `RIGHT_LOW` 的全部 10 个朝向通过当前静态几何阈值；
- `LEFT_HIGH` 与 `RIGHT_HIGH` 的全部 10 个朝向均未通过，离散旋转不足以补偿目标过小和机械臂遮挡；
- 自动输出的四个候选只是受“每个平移锚点最多两个”约束的优先集合，不代表只有四个位姿可用；
- 现阶段应保留 10 个低位候选进入动态运动与 provider 资格验证，再根据真实 localization、uncertainty 和
  unsafe acceptance 冻结更小的正式运行库。

本结果仍是 development-only。它不运行 front provider、不读取 Memory、不执行 `env.step()`、不推进操作
阶段，也不读取正式 test split。因此它不能证明旋转路径的时序安全、provider 可靠性、主动视角选择或
闭环恢复收益。

> 后续状态：G0C 已对本文件定义的 10 个低位静态合格候选完成有时延运动验证，40/40 路线通过。详见
> [`E018-P1 G0C rotated-motion findings`](e018_p1_g0c_rotated_motion_findings_20260905.md)。该后续结果不
> 改变 G0B 的静态筛选边界，也不构成 provider 或正式闭环资格。

```text
gate:                              G0B_STATIC_VIEWPOINT_GEOMETRY_SCREEN
status:                            complete-development-only
screen integrity:                  passed
translation positions:             5
orientation modes per position:    5
full camera poses:                 25
development scenes:                50
captures per scene-pose:           3
frame rows:                        3750 / 3750
repeatability audits:              1250 / 1250 passed
static-eligible alternate poses:   10
priority motion candidates:        4
test reads / provider forwards:    0 / 0
Memory accesses / env steps:       0 / 0
formal claim allowed:              false
```

## 实验设计

配置：
[`configs/e018_p1_g0b_viewpoint_screen_development_v1.json`](../configs/e018_p1_g0b_viewpoint_screen_development_v1.json)

### 5 × 5 候选格

平移位置：

```text
HOME        [0.30,  0.00, 0.60] m
LEFT_LOW    [0.30, -0.16, 0.48] m
LEFT_HIGH   [0.30, -0.16, 0.72] m
RIGHT_LOW   [0.30,  0.16, 0.48] m
RIGHT_HIGH  [0.30,  0.16, 0.72] m
```

每个位置先使用面向固定 workspace ROI `[-0.1, 0.0, 0.1] m` 的 nominal orientation，再叠加
camera-local 离散旋转：

```text
CENTER       yaw   0°, pitch  0°
YAW_LEFT     yaw +12°, pitch  0°
YAW_RIGHT    yaw -12°, pitch  0°
PITCH_UP     yaw   0°, pitch +8°
PITCH_DOWN   yaw   0°, pitch -8°
roll         始终为 0°
```

SAPIEN camera-local 坐标是 `+X forward / +Y left / +Z up`。远端方向探针确认：正 yaw 会使固定物体在
图像中向右移动，正 pitch 会使固定物体在图像中向下移动，符号与预期一致。首轮刻意不加入 yaw+pitch
对角组合，也不让模型生成连续角度。

### 静态采集语义

每个 seed 只 reset 一次，机器人、object 和 goal 状态随后保持不变。runner 依次设置 25 个相机位姿，
刷新 renderer，并在每个位姿连续捕获三次；整个过程不调用 `env.step()`，也不伪造仿真时间戳。三次 RGB、
segmentation 和 actual pose hash 必须完全一致，之后只保留一条代表帧做几何汇总。

GT segmentation、GT center projection 和 robot center-ray 仅用于离线筛选，不参与运行时控制、相机命令、
候选选择或 Memory 写入。

## 完整性门禁

| 门禁 | 结果 | 阈值 |
|---|---:|---:|
| frame lattice | `3750 / 3750` | 必须完整 |
| seed-pose repeat audit | `1250 / 1250` | 零失败 |
| camera position tracking error | `3.1200e-8 m` | `<= 1e-5 m` |
| camera orientation tracking error | `4.2147e-8 rad` | `<= 1e-4 rad` |
| Franka arm joint drift | `0.0 rad` | `<= 1e-5 rad` |
| TCP position drift | `0.0 m` | `<= 1e-5 m` |
| TCP orientation drift | `5.5755e-8 rad` | `<= 1e-4 rad` |
| HOME return position error | `2.6656e-8 m` | 必须通过 |
| HOME return orientation error | `0.0 rad` | 必须通过 |
| HOME return RGB mean absolute difference | `0.0` | 必须为 0 |
| env step / manipulation progression | `0 / 0` | 必须为 0 |
| provider / Memory / test access | `0 / 0 / 0` | 必须为 0 |

因此 25 个静态外参和对应图像指标可用于开发筛选，但这里没有任何 time-resolved motion 证据。

## 逐平移锚点结果

“双实体可见”要求 object 与 goal 都至少有一个 GT mask pixel；“双实体构图可用”进一步要求每个实体至少
8 pixels 且离图像边缘至少 2 pixels。以下范围覆盖该平移位置的五种离散朝向：

| 平移位置 | 双实体几何可见率 | 双实体构图可用率 | 较小实体 pixel p10 | 较小边界 margin p10 | 静态合格朝向 |
|---|---:|---:|---:|---:|---:|
| `HOME` | `0.74–0.78` | `0.48–0.54` | `0.0` | `-1 px` | `0 / 5` |
| `LEFT_LOW` | `0.98–1.00` | `0.96` | `15.9–19.9` | `22–44 px` | `5 / 5` |
| `LEFT_HIGH` | `0.90` | `0.72–0.74` | `2.9–3.0` | `27.8–44.9 px` | `0 / 5` |
| `RIGHT_LOW` | `0.98–1.00` | `0.94–0.96` | `12.9–14.5` | `23.9–45 px` | `5 / 5` |
| `RIGHT_HIGH` | `0.84–0.88` | `0.52–0.56` | `0.9–1.8` | `-1 px` | `0 / 5` |

HOME 基线有 14 个 object framing failure 和 12 个 goal framing failure。10 个低位候选中的每一个都恢复了
全部 14 个 object failure 和全部 12 个 goal failure；同时，低位候选自身少数构图失败的 seed 均可由
HOME 覆盖，所以 `HOME + 任一低位候选` 在这 50 个静态场景上的离线集合并集已达到 100%。这只是事后
GT coverage 诊断，不能解释为运行时已经知道该选哪个视角。

### 自动优先集合

当前 v1 配置在 10 个合格候选中先按双实体构图、几何可见、最差实体 pixel、边界余量和 robot occupancy
排序，再限制每个平移锚点最多两个，因此输出以下四个优先候选：

| 排名 | 完整位姿 | 双实体可见 | 构图可用 | 较小实体 pixel p10 | margin p10 | object/goal center-ray 被 robot 占据 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `LEFT_LOW__YAW_LEFT` | `1.00` | `0.96` | `19.9` | `30.0 px` | `0 / 0` |
| 2 | `LEFT_LOW__PITCH_UP` | `1.00` | `0.96` | `18.9` | `22.0 px` | `0 / 1` |
| 3 | `RIGHT_LOW__YAW_RIGHT` | `0.98` | `0.96` | `14.0` | `28.9 px` | `0 / 1` |
| 4 | `RIGHT_LOW__YAW_LEFT` | `1.00` | `0.94` | `14.5` | `35.0 px` | `0 / 0` |

`requested_shortlist_size=8` 却只产生四个候选并非数据或代码漏算：合格候选只分布在两个低位锚点，
而 `shortlist_max_per_anchor=2` 最多允许选出四个。这个约束适合形成小型优先集合，但不适合在 provider
资格验证之前宣称其余六个低位旋转已经无价值。

## 离散旋转带来的实际增益

离散旋转有可测影响，但本轮证据支持的是“微调构图”，不是“旋转可替代平移”：

- `LEFT_LOW__YAW_LEFT` 相比 `LEFT_LOW__CENTER` 将较小实体 pixel p10 从 `16.0` 提高到 `19.9`，并把
  goal center-ray 的 robot 占据从 `1/50` 降到 `0/50`；代价是 robot 平均画面占比从 `11.89%` 增至
  `12.77%`。
- `RIGHT_LOW__YAW_RIGHT` 将双实体构图可用率从 `0.94` 提高到 `0.96`，但双实体几何可见率从 `1.00`
  降到 `0.98`，说明某一个方向的旋转会帮助一侧边界，同时可能伤害另一侧极端场景。
- `RIGHT_LOW__YAW_LEFT` 保持 `1.00` 双实体几何可见率，并把较小实体 pixel p10 从 `12.9` 提高到
  `14.5`，但没有提高整体构图可用率。
- 两个 `PITCH_DOWN` 都把边界 p10 提高约 `10 px`，却没有提高构图成功率；它们可作为 provider OOD、
  俯仰方向和运动控制的诊断视角，不应只凭几何排序直接成为 primary view。
- 高位锚点无论如何旋转，goal p10 仍只有约 `1–3 pixels`，且部分场景存在零像素或 robot center-ray
  遮挡。问题主要来自高度、距离和遮挡关系，而不是简单的 optical-axis 偏转。

## 后续决策

本轮采用以下分层，不在 G0B 提前冻结正式动作库：

```text
DEVELOPMENT_LATTICE
  25 poses = 5 translations × 5 orientations

STATIC_ELIGIBLE_POOL
  10 poses = LEFT_LOW/RIGHT_LOW × all 5 orientations

MOTION_PRIORITY_SET
  LEFT_LOW__YAW_LEFT
  LEFT_LOW__PITCH_UP
  RIGHT_LOW__YAW_RIGHT
  RIGHT_LOW__YAW_LEFT

FORMAL_RUNTIME_LIBRARY
  not selected; requires dynamic-motion and provider qualification
```

下一步应按顺序执行：

1. G0C 对全部 10 个静态合格低位候选执行有时间的 `HOME -> pose -> HOME` 运动门禁；优先集合用于首轮
   故障定位，但不能替代完整候选池的动态审计。该项已完成，40/40 路线通过。
2. G1 明确 dynamic external-camera per-frame actual extrinsic、RGB/pose timestamp、motion-frame invalidation
   与 replay contract。
3. G2 使用冻结的 front provider 逐位姿报告 observability、XYZ error、covariance、false/unsafe acceptance；
   只有 provider 指标通过的位姿才有资格进入正式库。
4. 在 fresh validation 前冻结一个小型、确定性的 primitive schedule。运行时不得用 simulator GT 在 10 个
   候选间挑选，也不得自由输出 yaw/pitch。

暂不增加更大旋转角度、roll 或 yaw+pitch 对角组合。现有 ±12° yaw 与 ±8° pitch 已产生约 10–14 pixels
的可见位移；继续扩张动作空间应由 provider 误差或边缘失败证据驱动，而不是只因为候选数量越多越好。

## 工件、身份与备份

GPU 结果目录：

```text
/root/robot-vla-runs/e018-active-front-reobserve/
g0b-static-viewpoint-screen-development-v1
```

Drive 目录：

```text
gdrive:VLA/experiments/e018/
e018-p1-g0b-static-viewpoint-screen-development-v1-20260905
```

关键身份：

```text
config SHA-256:
b413487cc35a8ffb8bbaeba8ec6401bbe76eb885b6a3774d1e53b16e18b058a3

receipt SHA-256:
faf4609954cb1be33e16198ee7b305c7e8f5fabd58a8a48710fd30035f5309ea

Git parent:
cef61f0a6216b8504215a6745ab5a3925bfb0cc0
```

`receipt.json` 绑定 11 个内容文件；Drive 下载校验为 `13 matching files / 0 differences`（另包括 receipt
与 fail-closed `run_state.json`）。四张 5×5 contact sheet 对应 seeds 72001–72004。定向回归覆盖 E018 active camera、P0 Object
Memory、Object observability/evaluation 与冻结 Goal Memory，共 `57 passed`。

## 能证明与不能证明的内容

G0B 已证明：

- 25 个固定平移与旋转组合可获得精确 actual extrinsic 和确定性静态图像；
- 当前 development 分布中，低位锚点的五种朝向均明显优于 HOME 与高位锚点；
- 离散 yaw/pitch 会改变 entity pixel、边界余量和 robot center-ray 遮挡；
- 10 个低位候选值得进入下一阶段，而不是把动作空间误解为仍只有四个位姿。

G0B 尚未证明：

- 新旋转的逐 Tick 路径、角速度、角加速度、settle、collision 或真实硬件安全；
- 约 `13–20` 个 p10 visible pixels 足以支持 front provider 安全定位；
- 10 个候选都应进入正式 runtime library，或四个优先候选已经是最优集合；
- runtime viewpoint selection、Memory candidate validation 或 closed-loop resume；
- Active 相对 matched Passive 的可靠状态恢复增益；
- 任何 formal、test-once 或完整 manipulation success claim。
