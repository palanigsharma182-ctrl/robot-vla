# E013 — 厘米级闭环精调执行层

## 状态与边界

**Status:** engineering scaffold；尚未运行正式数据采集、训练、GPU smoke 或闭环效果实验。

E013 在任何正式训练开始前由“八图 Frozen Qwen Layer 12 状态归因”修订为两时间尺度架构：低频
VLA 只负责语义、技能和粗 ROI；闭环精调层直接读取腕部原始高分辨率 ROI，通过三头 U-Net、显式
相机几何和视觉伺服生成小幅 TCP commanded-target delta。旧 E013 online geometry smoke 保留为历史
Layer-12 粗定位诊断；它测量单帧表示，不是最终物体放置闭环门槛。

本次修改没有运行或改写 E012 Dataset、checkpoint、evaluation 或其他冻结产物。新增代码只是可测试的
模型/几何/控制边界，不能写成已经达到厘米级闭环效果。

端到端验收按最终 object→goal 平面位置误差分档。推荐求职档是正式目标，工程可用档是可接受底线，
个位数毫米只作可选挑战，不影响 E013 是否完成：

```text
E008 Layer-12 spatial-probe reference（不是闭环 placement baseline）
  p50 decoded world-XY localization error = 0.0253 m
  p90 decoded world-XY localization error = 0.0388 m

engineering floor（可接受底线）
  p50 final placement XY error <= 0.015 m
  p90 final placement XY error <= 0.025 m

recommended portfolio target（正式目标）
  p50 final placement XY error <= 0.012 m
  p90 final placement XY error <= 0.020 m
  P(final placement XY error <= 0.020 m) >= 0.90

optional stretch（不作为项目成败门槛）
  p50 final placement XY error <= 0.010 m
  p90 final placement XY error <= 0.015 m
```

正式评估至少包含 `100` 个预注册 unseen Episode，control/treatment 使用相同环境和采样 seed。有效任务
失败只要存在可测最终位置就必须进入误差统计；invalid projection、system/safety/tracking failure、控制器
重叠和 stale-observation command 均单独计数，任一非零都阻断 promotion，不能删除失败样本后复算。
正式报告同时给出 bootstrap 95% 区间；区间用于披露不确定性，不在看到结果后改变上述点估计门槛。
正式相对改善必须用这 100 个 paired seed 上实际运行的 coarse-only control 复算；不得把 E008 线性 probe
的定位误差当成 final-placement control。Probe reference 只用于说明 Qwen 表示的空间上限和量级；当前
`PrecisionEvaluationAssessment` 因此不输出“相对 probe 改善百分比”，相对效果留给正式 paired analyzer。

## 两时间尺度架构

```text
2–5 Hz VLA
  instruction + semantic view
  -> object / goal / skill / coarse ROI / precision-mode request

20 Hz required precision loop; 30 Hz optional stretch
  current high-resolution wrist ROI
  -> U-Net Localization Head
  -> explicit camera geometry
  -> geometry TCP delta
  -> shadow Metric Residual Head
  -> calibrated uncertainty gates
  -> bounded commanded TCP target delta

200–1000 Hz robot inner controller
  Cartesian target / IK / impedance tracking
  -> joint or torque command
```

VLA 与 precision layer 不能同时争夺位置控制权。`SEARCH/COARSE_APPROACH` 由 VLA 控制；进入
`FINE_ALIGN` 后，VLA 只保留监督和 replan 权限；`CONTACT` 再加入 `F_L/F_R`。关键点丢失、投影无效、
时间跳变或 uncertainty gate 失败时，precision layer 输出零命令并交还控制权，不能盲目继续。

## 三头 U-Net

模型身份固定为：

```text
precision_unet_three_head_v1
```

当前工程默认使用轻量 U-Net encoder channels `32/64/128/256`、GroupNorm 和 bilinear decoder。
GroupNorm 避免 batch 1 高频推理时 BatchNorm 统计漂移。模型只处理当前腕部 RGB ROI；四帧历史在第一版
由模型外的关键点/位姿状态估计器融合，避免让网络同时学习相机自运动。若该方案被证明确受遮挡限制，
才预注册 pose-aware temporal feature fusion。

### Head 1：Localization

定位分支只读取图像 decoder feature，不读取 TCP、camera pose、force 或 controller state，避免通过机器人
状态猜测目标位置：

```text
heatmap_logits       [B,K,H,W]
mask_logits          [B,M,H,W]
subpixel_offsets     [B,K,2,H,W], bounded to [-0.5,0.5] px
```

soft-argmax 同时输出 pixel UV、normalized UV、peak probability 与 normalized entropy。项目继续使用：

```text
normalized_u = (pixel_u + 0.5) / image_width
normalized_v = (pixel_v + 0.5) / image_height
```

这与现有空间 probe 的像素中心约定一致。第一版 keypoint 为 object/goal center；角点可通过配置增加，
但不能在正式结果后临时改变输出定义。

### Head 2：MetricResidual

Motion Head 不输出旧 VLA 的 7-DoF joint Action。它读取 U-Net global feature、当前结构化 V2 frame state
和显式几何结果，输出四维 bounded residual：

```text
[delta_x_base_m, delta_y_base_m, delta_z_base_m, delta_yaw_base_rad]
```

动作语义固定为：

```text
commanded-tcp-target-delta/base-frame/m-rad/v1
```

它与 `command_target_delta_q[7] + gripper_target[1]` 的 VLA Action 契约显式隔离。从 Cartesian delta 到
IK、关节 target 和底层 controller 的适配尚未接入现有 Runtime，不能通过复用 joint `ActionAdapter`
隐式转换。

当前 step/residual 安全上限只是工程默认值，不是效果阈值：

| 分量 | 单 Tick geometry step limit | learned residual limit |
|---|---:|---:|
| base X | 1.00 mm | 0.25 mm |
| base Y | 1.00 mm | 0.25 mm |
| base Z | 0.50 mm | 0.20 mm |
| base yaw | 0.20° | 0.05° |

Residual Head 最后一层零初始化。第一阶段固定 `mode=shadow`：正式候选命令严格等于 clipped geometry
命令；Motion Head 预测及其 motion uncertainty 只产生诊断 warning，不能阻断或改变 geometry 命令。
只有 held-out shadow 证据通过新的预注册门槛后，才能以新实验身份启用 `bounded_residual`，并把 motion
uncertainty 升级为执行门禁。

### Head 3：Uncertainty

不使用单一 confidence 标量。模型输出：

```text
keypoint_log_variance      [B,K,2]
motion_log_variance        [B,4]
visibility_logits          [B,K]
projection_validity_logit  [B]
```

训练使用异方差 NLL；部署阈值必须在独立 calibration split 上换算为 pixel/metric uncertainty。最终
控制门禁还必须组合 heatmap entropy、几何重投影残差、四帧 innovation、工作空间和 force 状态，不能
只相信网络自报 confidence。

## 显式几何

当前实现使用 Observation V2 已冻结的 OpenCV optical `base_from_camera_cv`。像素射线为：

```text
r_camera = inverse(K) @ [u,v,1]
r_base   = R_base_camera @ r_camera
c_base   = t_base_camera
```

与平面 `n·X+d=0` 的交点为：

```text
lambda = -(n·c_base + d) / (n·r_base)
X_base = c_base + lambda * r_base
```

`lambda <= 0`、平行射线、非 SE(3)、非有限输入和无效 intrinsic 均 fail closed。第一版提供已知
`base z` 平面的 object/goal center 反投影，以及从目标点和当前 `base_from_tcp` 生成显式四维 TCP delta。
输入图像必须与内参使用相同畸变校正；畸变处理不能被静默忽略。

## 四帧与双相机

第一版不把四帧堆成 12 通道，也不把双相机八图送进 precision U-Net：

```text
external current frame -> 全局 object/goal + coarse ROI
wrist current frame    -> U-Net 精定位
last four wrist detections + matching camera poses/timestamps
                       -> world/TCP-frame temporal filter
```

每个历史检测必须绑定采集时刻的 camera pose。旧图不能使用当前 TCP/camera pose 解释。后续只有在单帧
遮挡成为实测瓶颈后，才比较共享 encoder + pose-aware warp/ConvGRU；不能默认把更多图像当成毫米精度来源。

## 监督与数据来源

U-Net 不从 220 条 Expert Action 轨迹学习“怎么移动”。训练标签优先由随机 scene/view 采样生成：

- actor segmentation 或 GT pose 只作为 oracle/训练标签；正式 RGB-only forward 不读取它们；
- object/goal center、角点、visibility 和 mask 由仿真状态自动生成；
- geometric motion target 来自冻结相机公式和确定性 controller；
- residual target 只能来自 command→achieved 的可重复系统偏差，不使用语义不清的 historical action；
- lighting、texture、occlusion、camera pose 和 calibration perturbation 在结果前冻结。

损失包括 heatmap、mask、normalized-UV、motion residual、visibility/projection 和 heteroscedastic uncertainty。
正式评价必须另报 world-XY error；低 pixel loss 不能替代毫米指标。

## 渐进门禁

1. **Action/coordinate gate：**Cartesian action frame/unit/semantics、图像时间戳、camera pose 和 TCP pose
   round-trip 全部通过。
2. **Oracle geometry gate：**privileged mask/GT pixel + 显式几何在固定 unseen states 上达到
   `p90 <= 5 mm`，invalid=0。它是公式/标定 lower bound，不是最终效果声明。
3. **RGB-only perception gate：**先比较 HSV/轮廓，再比较 U-Net；held-out world-XY `p90 <= 15 mm`
   后才接控制器，不运行长 VLA 训练。
4. **GT-pose controller gate：**使用 GT target 检查 Cartesian/IK/底层跟踪和接触达到
   `p90 <= 10 mm`，为 RGB 感知误差保留预算。
5. **Shadow gate：**Motion Head 不控制，只比较 geometry、prediction 和 achieved delta，并校准
   uncertainty。
6. **Bounded residual gate：**前五项通过后才以新身份允许 learned residual 进入命令。
7. **完整 RGB-only closed loop：**最后按 engineering/recommended/stretch 三档评价端到端放置误差、
   `within 20 mm` 成功率和全部 guardrail。

所有 gate 使用固定 unseen seeds，不能删除失败帧或在结果后调整阈值。20–30 个 smoke state 只能发现
工程错误；正式分档至少需要 100 个独立 paired Episode 和置信区间。闭环运行门槛固定为有效控制频率
`>=20 Hz` 且端到端延迟 `p95 <= 50 ms`；30 Hz 只作可选性能结果，不要求 60 Hz。

## 当前实现

已实现：

- `PrecisionMotionSpec` 的 frame/unit/step/residual 契约；
- OpenCV pixel→base plane 和 base point→pixel round-trip；
- 三头轻量 U-Net、亚像素 soft-argmax 和零初始化 residual；
- heatmap/mask/coordinate/residual/heteroscedastic loss；
- shadow/bounded-residual 仲裁与 visibility/entropy/uncertainty fail-closed gate；
- Precision decoded keypoint 到部署 wrist detection 的版本化适配，保留 visibility、projection validity、
  peak、entropy 与 pixel sigma；
- Window 原始 float64 frame/modality timestamp、四时刻 base-frame track/velocity/innovation 融合，以及
  默认关闭的 replan-boundary Shadow Executive hook；
- 合成几何和控制单元测试。

尚未实现或验证：

- ManiSkill 相机真实分辨率、畸变和 tabletop mm/pixel receipt；
- oracle/HSV/RGB-only Dataset 与训练 CLI；
- 真实 RGB keypoint model provider、track/outcome confidence 标定与 20 Hz shadow measurement；
- Cartesian IK、机器人底层接口和 20 Hz / p95 50 ms latency；
- force-contact controller；
- PyTorch 环境中的 forward/backward 实测；
- 任意厘米级闭环效果证据。

本机当前 Python 环境没有 PyTorch，模型测试会明确 skip；轻依赖几何/控制测试可以执行。下一步应先在
正式 ManiSkill/PyTorch 环境读取相机能力并完成 oracle geometry lower-bound，而不是启动旧 E013 的四轮
Qwen 训练。
