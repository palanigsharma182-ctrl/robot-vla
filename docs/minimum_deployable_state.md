# 最小可部署状态与 Action 语义修正

## 范围与结论

本次 amendment 解决两个彼此独立、但都会破坏闭环归因的问题：

1. Expert 数据中的 Action 是相邻 **commanded joint target** 的增量，而旧 Runtime 会在每次
   Replan 时重新以滞后的 actual joint position 为基准，导致同一个标签在训练和执行时表示不同目标；
2. Observation V1 只有当前双图与 `q/dq/gripper`，缺少 TCP 位姿、动态腕部相机位姿、短时视觉历史、
   双指接触力以及 controller reference，无法区分几何误差、目标运动、接触变化和跟踪滞后。

修正后的 Observation V2 是一个与 V1 checkpoint 显式不兼容的新模型身份。旧 V1 数据和 checkpoint
仍可读取，但不得把旧 D0 中不存在的左右指力、位姿或图像历史伪造为 V2 数据。

## Action 标签与执行语义

可信 Expert 在控制步 `t` 保存前一 commanded target `r_(t-1)`、新的 target `r_t` 和 actual joint
position `q_t`。训练标签固定为：

```text
a_t = r_t - r_(t-1)
```

跨 Chunk、跨 Replan 的目标必须继续从最后一次成功发送的 command reference 积分：

```text
r_(t+k) = r_(t-1) + sum(a_t ... a_(t+k))
```

控制器看到的单步 correction 则始终由最新 actual state 计算：

```text
correction_t = r_t - q_t
```

因此，`a_t` 是策略/数据契约，`correction_t` 是执行器内部的跟踪补偿，两者不能共用同一个
`delta_q` 名称或相互替代。Episode reset、hold、发送失败、安全拒绝、tracking saturation 和 anomaly
都会清空 command reference；新 Episode 的第一次 Chunk 才允许从 actual `q` 初始化。

轨迹 V2 额外保存 `previous_command_q_rad`、`commanded_joint_target_rad` 和
`applied_joint_correction_rad`。validator 对每个控制步验证：

```text
commanded_joint_target_rad == previous_command_q_rad + action[:7]
```

这使 Action label—execution parity 成为可审计事实，而不是仅靠代码注释约定。

## Observation V2

### 时间与 Padding

- 固定最近 4 个连续控制步，stride 为 1；
- 顺序固定为 `t-3, t-2, t-1, t`；
- Episode 开头只允许前缀零 padding，并令 `history_valid=False`；
- 不复制首帧，不跨 Episode，不使用未来帧；
- 当前时刻六个模态必须完整有效；历史时刻允许局部缺失，但无效值必须为零并保留 modality validity；
- 每个模态保存采样时间，验证有效时间戳不晚于所属控制 Tick；模型状态保留 frame age 和 validity。

### 模态

| 模态 | 模型表示 | 审计表示 | 语义 |
|---|---|---|---|
| external/front RGB | 最近 4 帧 | 原始 uint8 | 固定相机角色，不得与 wrist 交换 |
| wrist RGB | 最近 4 帧 | 原始 uint8 | 与 wrist pose 同一 Simulator Tick |
| proprio | 每帧 15 维 | `q[7]+dq[7]+gripper[1]` | 继续使用 train-only ProprioStats |
| TCP pose | position + Rotation-6D | 完整 `base_from_tcp` SE(3) | 机器人 base frame |
| wrist camera pose | position + Rotation-6D | 完整 `base_from_wrist_camera_cv` SE(3) | OpenCV optical frame，机器人 base frame |
| `F_L/F_R` | train-only 稳健归一化后 2 维 | 牛顿值与 sensor version | 左/右 finger link 对 target cube 的 pairwise contact-force magnitude |
| controller state | command、tracking error、previous action、valid bits | 同值 | 显式区分策略动作与跟踪补偿 |

当前 ManiSkill 的 `F_L/F_R` 是仿真接触力近似，不是真实夹爪应变片读数。旧 aggregate 字段保持为：

```text
robot_object_contact_force_n = max(F_L, F_R)
```

旧 D0 只有 aggregate，无法唯一恢复左右值；把 aggregate 复制到两侧会伪造对称接触，V2 validator 会拒绝。

### 坐标公式

项目使用 `A_from_B` 表示把 B-frame 坐标变换到 A-frame。ManiSkill 的相机矩阵是 OpenGL camera
frame；模型与审计统一使用 OpenCV optical frame（x right、y down、z forward）：

```text
camera_gl_from_camera_cv = diag(1, -1, -1, 1)
world_from_camera_cv = world_from_camera_gl @ camera_gl_from_camera_cv
base_from_camera_cv = inverse(world_from_base) @ world_from_camera_cv
base_from_tcp = inverse(world_from_base) @ world_from_tcp
```

Rotation-6D 使用旋转矩阵前两列的列优先顺序：

```text
[r00, r10, r20, r01, r11, r21]
```

SE(3) validator 会拒绝非齐次矩阵、缩放、反射和非正交 rotation。FK 诊断同时报告 TCP position error
与 SO(3) geodesic orientation error，不能再只用位置正确替代完整 pose 正确。

## `F_L/F_R` 归一化与身份

只使用 train split 中 validity 为真的左右指力拟合 `finger_force_stats.json`：

1. 保留零接触的物理零点，不做去中心化；
2. 对牛顿值应用 `log1p`；
3. 每个 finger 用正接触样本的 P95 作为独立尺度；
4. 默认裁剪到 `[0, 2]`；
5. 任一 finger 在 train split 没有正接触样本时 fail closed。

统计文件冻结 observation schema、sensor approximation version、embodiment、样本数、正样本数、
quantile 与 clip。Dataset、Runtime、checkpoint save/load/resume/init 和 evaluation 必须使用完全相同的
FingerForceStats；V1 checkpoint 禁止混入该字段。

## 数据、训练与评估入口

新采集器写入 Observation V2 和 Action parity 数组，采集结束时的正式 audit 会只从 train split 生成
`proprio_stats.json` 与 `finger_force_stats.json`：

```bash
python -m robot_vla.cli.collect_maniskill \
  --output /path/to/new-v2-dataset \
  --train 176 --val 22 --test 22 \
  --start-seed 0 --max-candidates 2200
```

V2 训练必须显式启用开关；没有完整 V2 arrays 或 force stats 的数据会在加载阶段被拒绝：

```bash
python -m robot_vla.cli.train_stage1 \
  --observation-v2 \
  --data /path/to/new-v2-dataset \
  --model-cache /path/to/model-cache \
  --output /path/to/v2-run \
  --qwen-context-layer 12
```

Full-chain 和 atomic 闭环使用相同开关，checkpoint metadata 会验证 V2 schema 与 force stats identity：

```bash
python -m robot_vla.cli.evaluate_maniskill \
  --observation-v2 \
  --data /path/to/new-v2-dataset \
  --model-cache /path/to/model-cache \
  --checkpoint /path/to/v2-checkpoint.pt \
  --output /path/to/v2-full-eval

python -m robot_vla.cli.evaluate_atomic_maniskill \
  --observation-v2 \
  --data /path/to/new-v2-dataset \
  --model-cache /path/to/model-cache \
  --checkpoint /path/to/v2-checkpoint.pt \
  --output /path/to/v2-atomic-eval
```

## 归因方案

Action 语义修正属于 correctness gate，不作为可选择的 treatment；后续所有 arm 都必须使用修正后的
executor。Observation 的第一项正式 estimand 是“完整最小可部署状态包”相对 V1 的净作用：

- 从同一批重新采集、通过 V2 audit 的轨迹构造 V1 projection control 和 V2 treatment；
- 两臂使用相同轨迹、Action target、Qwen revision/layer、训练 seed、sample exposure、optimizer steps、
  Flow seed、checkpoint 候选和新 held-out evaluation seeds；
- control 只读取当前双图/proprio，treatment 读取完整 V2；
- 先比较 full success、五技能无条件完成数、mean completed skills 与 system/safety/tracking guardrail，
  再解释条件交接率；
- 若 bundled treatment 有正向证据，再以预注册的 modality masking 做 TCP、camera pose、history、force
  component ablation；在看到结果后临时删模态不属于有效归因。

旧 E012 D0/D1、checkpoint validation seeds 和 selection 结果保持冻结，不作为新 V2 数据，也不得通过
复制 aggregate force 或重复首帧升级成 V2。

## 当前验证边界

当前无重依赖回归为 `313 passed, 18 skipped`。18 个 skip 都来自本地缺少 PyTorch、Transformers、
ManiSkill/Gymnasium 等正式依赖。发布代码已经覆盖 schema、Action parity、历史、坐标、归一化、
checkpoint/runtime fail-closed 契约，但在正式训练前仍必须在原项目 GPU/仿真环境完成：

- Temporal Expert forward/backward 与 V2 checkpoint round-trip；
- 真实 Qwen 八图 Processor、padding attention mask、显存峰值和 p95 latency；
- ManiSkill TCP/camera SE(3)、左右 pairwise force shape/dtype；
- full-chain/atomic 的 plain Flow、temporal ensemble 和 RTC 单次 smoke；
- 新 D0-V2 的完整成功、五技能、Action parity、modality coverage 与 train-only stats audit。

在这些门禁通过前，不启动 Observation V2 正式训练，也不把静态/单元测试写成闭环效果。
