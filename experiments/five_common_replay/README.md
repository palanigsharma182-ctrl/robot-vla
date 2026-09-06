# 五个通用候选接入 main

2026-09-06 工程接入，来源提交 `37851ad648c42713511f869eb782eb897aafecc5`，
接入前 main 为 `79160c4`。五个模块均位于 `src/robot_vla/precision/`；实验消费者留在本目录。
本轮没有改变 canonical VLA、Goal Memory、Action 或默认控制链，也没有晋级研究效果。

## 模块与数据流

| 模块 | 当前消费者及用途 |
|---|---|
| `active_front_camera.py` | 合成相机路径、运动阶段的写入资格；不负责实际相机执行 |
| `active_external_observation.py` | 从同次传感器返回值提取 RGB、内参和实际位姿，保留真实 camera UID 与审计身份 |
| `calibrated_front_provider.py` | 校准预测协方差与像素 sigma，构造稳定相机/校准身份 |
| `object_observability.py` | 由预测证据得到可观察性和写入评分，离线 GT 标签接口不参与写入 |
| `object_memory.py` | 验证连续新图像、来源、时间和接触条件，保存位置并按用途读取 |

`run.py` 将实际位姿格式的合成观测与人工预测串接：
相机阶段/采集时间 → 实际 camera-to-base 变换 → 校准 covariance 与 evidence →
阶段绑定、写入门槛 → candidate window → Object Memory → NAVIGATION。
相机位置 `[3]` 单位 m，协方差 `[3,3]` 单位 m²，变换为 robot-base-from-OpenCV-camera。
相机位姿用于旋转协方差和变换预测位置；commanded pose 不参与测量几何。

本入口仅是合成消费者：预测坐标、mask 分数、像素 sigma、校准元数据和阶段切换均是人工 fixture，
没有训练出的 provider 或真实置信度估计。其来源使用 `synthetic-prediction/` 命名空间，
没有将原 G2B `qualification-only/no-memory/no-actuation` 身份重新解释为写入授权。
`sidecar.ledger_record()` 只描述提取层；本入口单独记录实际 Memory 更新。

## 接入时修复的问题

- 图像时间必须递增，拒绝重复/倒退帧后不降低已见时间水位；拒绝会清空候选窗口，需连续新帧重建。
  同时检查控制 tick 与 RGB 的相邻间隔，逐帧拒绝输入时已经过期的观测。
- `last_observed_timestamp_s` 使用 RGB 采集时间，`state_timestamp_s` 使用状态更新时刻。
  普通写入与延迟提交的 age、协方差增长均包含采集延迟；pending 超时仍从候选接收 tick 起算。
- 拒绝延迟提交也会生成老化至 commit 时刻的旧状态；preview 无副作用，apply 再更新。
  防止同一图像重复写入，拒绝早于 Memory reset 时刻的候选。
- 相机 UID 进入 sidecar 和审计摘要，actual pose source 绑定相应 sensor key。
  默认 `base_camera` 的来源字符串保持兼容；新增 UID 字段意味着审计摘要随新代码重新计算。
- covariance 在缩放前对称化、去除容差内的负特征值；显著非 PSD 输入仍拒绝。
  校准参数必须满足其声明的 order statistic 与 scale 公式。
- 姿态工具拒绝零范数四元数、明显非旋转矩阵及非布尔 settled；只允许小量 SO(3) 舍入修正。

调用约定：每个新 Episode 使用新的 ID，同时 reset Memory 和 verifier；同 Episode 内重置须提供
正确的单调 reset 时间。不通过复用旧 Episode ID 并把时钟归零来隔离延迟队列。
采集阶段开始时间由调用者提供，旧阶段图像不能因当前已经 COLLECT 而获得写入资格。
TCP 时间须显式提供，不能从相机时间推造。sidecar 的 `memory_write_eligible` 仅表达阶段资格，
完整消费者还必须联合验证采集阶段、时间偏差、观测新鲜度、来源和预测 evidence。

## 运行与验收

优先在项目已有远端环境运行；这些入口不自动安装依赖或下载权重。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python experiments/five_common_replay/run.py

PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m pytest -q -p no:cacheprovider \
  tests/test_e018_object_memory.py tests/test_e018_object_observability.py \
  tests/test_e018_active_external_observation.py tests/test_e018_active_front_camera.py \
  tests/test_e018_calibrated_front_provider.py tests/test_five_common_replay.py \
  tests/test_object_memory_replay.py tests/test_precision_state_memory.py \
  tests/test_maniskill_observation_boundary.py tests/test_chunk_executor.py \
  tests/test_observation_v2.py tests/test_runtime.py tests/test_adapters.py \
  tests/test_temporal_ensemble.py

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python experiments/five_common_replay/maniskill_smoke.py
```

本轮远端 Python 3.10.12 / NumPy 1.26.4 / PyTorch 2.11.0+cu128 / pytest 9.1.1：

- 定向回归 **138 passed**，退出码 0，包含完整 Goal Memory 14 项，未使用提取子集或 skip 替代。
- 最初新增的 7 个时间反例在旧实现全部失败；修复及审查补充回归后通过。
- 五模块合成 CLI 回放 **4 Episode / 23 Tick**，退出码 0；覆盖运动、缺测、低分、重复/乱序帧、
  过期、校准来源变化、接触和 reset。原 Object Memory 19 Tick 回放继续通过。
- ManiSkill **3.0.1** / SAPIEN **3.0.3** 实际 RGB 与位姿 smoke：**2 Episode / 8 env.step / 10 capture**，
  坐标往返最大误差 **0 m**，退出码 0。SAPIEN 自带 Vulkan 回退警告未阻碍渲染，没有改驱动。
  此 smoke 使用静止相机、零关节增量和开夹爪目标，仅验证接口，没有执行主动重观察路线。

两次独立只读审查分别覆盖 Memory 时间/延迟提交，以及相机/校准与串联消费者。
第一遍发现的问题均完成修复，第二遍在与远端清单一致的源码上通过复核。
Memory 审查独立运行 28 项测试及 12 组合成检查；相机/校准审查另验了非单位旋转、
各向异性 covariance、错误相机和采集阶段输入。独立结论不包含实际 ManiSkill smoke，
该项由主 Agent 核验原始输出及执行源码 SHA-256。

原始输出、失败记录、传输代码的 SHA-256 清单和执行耗时保留于忽略目录
`artifacts/reviews/five-common-integration-20260906/`；仅传输本任务所需源码、配置和测试。
远端每个测试阶段使用独立目录并先核对清单。没有下载依赖、恢复权重/数据或执行训练。

## 历史证据与后续边界

以上是单测、合成串联和仿真接口证据，不证明真实 provider 精度、主动重观察收益或任务成功率。
时间修复影响 E018 P0/P1 中调用这些 Memory 接口且存在图像延迟、重复帧或延迟拒绝的路径；
本轮没有重读真实逐帧数据，不能宣称已确定历史影响数量，也不能直接改写原结果。
校准与相机身份问题同理：发现构造边界不足不等于历史 artifact 已有错误。
旧结果及源码覆盖 inventory 保持冻结；新运行绑定本轮代码/配置身份。

五模块工程接入完成后，下一项是冻结 G2C provider 的真实测量消费，再按实际需要接 D049 编排。
`active_front_provider`、`active_front_memory*`、`active_front_reobserve` 及其研究效果不计入本轮五模块完成范围。
