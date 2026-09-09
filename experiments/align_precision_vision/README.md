# Align Precision Vision：独立精确视觉与 Action Expert 条件接口

2026-09-10。用户批准的隔离实现：Qwen 负责语义与目标关联，Precision Vision 负责几何测量，当前 Align Expert 负责动作。稳定 `src/`、七技能原模型和历史结果保持原样。本目录没有机器人动作发送接口。

当前提供可训练的定位网络、已知模板 PnP 位姿计算、可选内部深度细化、Align 条件接口和 Expert 的 BC loss/Flow 推理。**已完成976帧仿真图像采集、1000步pilot及两阶段共40000步追加训练；已有development定位信号，尚未通过正式定位验收或运行新的学生闭环。** 角点深度问题与三组对照见 [PnP 结果](pnp_results.md)；先前数据结果保存在 [数据接入与结果](data_results.md)。

并行局部几何与移动RGB-D实验及续训记录见 [two_route_results.md](two_route_results.md)，执行上限见 [two_route_plan.md](two_route_plan.md)。

最新development选择仍为全局6000步checkpoint：PnP有效93/236、准确90/236；局部三点加内部深度有效115/236、准确97/236；多视角有效95/236、准确92/236。准确指位置≤8mm且姿态≤5°，不代表技能成功。两阶段均达到更新数上限，没有触发稳定平台期。全部有效结果来自腕部相机；两相机各118帧，因此PnP腕部有效率为93/118（78.8%），按全部相机图像计为39.4%。该coverage不是控制时刻成功率。多视角RGB-D与RGB结果相同，未显示内部深度检查的额外收益。

官方演示导入前的元数据检查工具见 [ManiSkill demos检查](../maniskill_data/README.md)；尚未完成官方演示转换或据此训练学生。

## 分工与范围

```text
Qwen context ───────────────────────────────────────────┐
proprio[15] ─────────────────────────────────────────────┤
原始 RGB → 8个物体角点 + 可见性                         │
                  + 4cm模板/内参/OpenCV外参              │
                  → PnP/RANSAC → base_from_object        │
                  ↳ 可选：腐蚀前景内部深度检查/细化       │
                  + 固定 object_from_pregrasp + 当前FK   │
                  → TCP坐标系相对位姿[6] + valid + age ──┤
                                                       ↓
                                           Align Action Expert
                                                       ↓
                                      原有16×7 TCP chunk / IK /执行器
```

第一版明确针对**一个已关联的、已知4cm直立方块**，只处理 Align。上层提供 episode/target 身份；本模块不实现多物体语言指代。没有从 GT token、分割、模拟器物体 pose 或未来动作构造 measured 输入。Oracle 有独立构造函数和来源，实测模式拒绝 Oracle 来源。

先保留原始图像分辨率，不做 ROI/resize。未来裁剪必须同步修改内参和像素映射，不能直接复用原图 K。训练实际画面应是原生高分辨率采集；把旧128×128图像放大不能补回细节。当前帧表示已对齐的 RGB、深度与同帧相机外参；`calibration_timestamp_s` 指外参对应的时刻，不是内参标定文件的创建日期。

## 实现

- `vision.py`：复用旧 Precision U-Net 的 encoder/decoder、热图和格内偏移结构，只执行图像分支。另有图像可见性头。没有 Motion Head，也没有伪造 V2 state。八角点的整体四重旋转对称 loss 包含热图、坐标和可见性监督；不可见坐标不回归，可为 NaN。旧两关键点 checkpoint 不能直接当作这个八角点模型；架构复用不表示已有合格权重。
- `pose_recovery.py`：默认 `pnp-only`，至少四个通过 visibility filtering 的2D角点对应已知4cm模板。平面点用 IPPE、非平面点用 SQPnP，并加入 RANSAC/LM 候选，按正深度、内点数与重投影残差选择。`pnp-depth-refine` 只从 RGB 红色前景与 PnP 投影轮廓交集经5×5腐蚀后的内部取深度，做已知cube射线相交一致性检查及有界6D细化；没有 GT mask。当前红色阈值只适用于这个已关联红色方块，未实现通用分割。内部点不足时显式返回未细化 PnP，深度不一致且无法细化则拒绝。
- `geometry.py`：保留模板、坐标变换、四重 yaw 对称和 Align 条件构造。旧角点深度/SVD 仅由显式 `old-corner-depth` 对照调用，新路径失败不会自动切回旧方法。目标仍是 `base_from_object @ object_from_pregrasp`。
- `pipeline.py`：默认 PnP-only，绑定 RGB/外参时间、episode 和 target；仅深度分支检查深度时间同步。现有 frame 容器保留浮点深度字段，PnP-only 可传同尺寸 NaN 数组，不消费深度值。当前 TCP 位姿必须来自该 proprio 对应的 FK，由调用方提供对应时间。在线 `predict` 的 `now_s` 必须为同一传感器时基的时钟函数，测量后和动作生成后重新计龄；推理耗时导致过期则抛出 TimeoutError，不能继续执行该 chunk。四重对称采用当前 Align 合同中绕目标接近轴的旋转等价；抓取变换须符合这一中央对称抓取定义，不可任意用于偏置抓取或非对称物体。
- `expert.py`：复制已有15维proprio、7维TCP动作的独立 Expert；保留旧状态投影，另加8维几何投影。新增权重零初始化，两路结果在原状态第一层相加。输入通过函数参数传入，无可变全局条件/forward hook，避免跨调用残留。提供 `flow_loss`、`predict`、两组显式学习率和同噪声 `geometry_response`。
- `train.py`：独立定位器训练入口，读取带 hash 与场景 split 的 NPZ manifest；只打开 train 样本。保存配置、精确采样序列、模型/优化器和完成结果，固定最终步并严格重载。不提供从该文件精确续训的承诺。

网络输出 `[B,8,2]` 原图像素中心坐标与 `[B,8]` 可见性概率。结构化输入 `[B,8]` 为：

```text
[dx/0.05, dy/0.05, dz/0.05,
 rx/0.5,  ry/0.5,  rz/0.5,
 valid, age/max_age]
```

平移单位米，旋转为当前TCP轴向的旋转向量（弧度）。缺失条件带 `valid=0`，送进 Expert 时整个额外分支为零，得到同一权重的无几何预测；这不意味着已对齐，也不保证回退动作成功。`condition.reason` 和原测量必须进入实际评估记录。有效性/年龄不是夹爪或技能退出许可。

候选工程默认：可见性≥0.8，PnP至少4点、内点比例≥0.6、RANSAC阈值/内点重投影RMS≤3px；可选内部深度0.05–2m、一致性median≤5mm/p90≤10mm。旧对照仍用刚体拟合RMS≤2mm、点间距离误差≤4mm。最大年龄50ms、传感器错配25ms保持不变。这些是待验证配置，**不是已校准置信度、已验证定位精度或新的技能验收标准**。必须联合报告误差与覆盖率；低重投影误差本身不能消除平面解歧义或保证目标身份正确。

三组离线 A/B 共用每帧的一次预测与同一0.8阈值，报告完整分母的 coverage，以及有效样本的 reprojection/position/orientation median、p90、mean，另报共同有效帧比较：

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=.:src python \
  -m experiments.align_precision_vision.compare_pose \
  --manifest /path/to/manifest.json --checkpoint /path/to/final.pt \
  --output /path/to/new-comparison.json --split development \
  --keypoints predicted --wall-seconds 600
```

`--keypoints gt-diagnostic` 仅替换角点与可见性作为几何求解诊断，必须独立报告。`evaluate.py` 单方法评估也默认 `pnp-only`，可用 `--method` 显式选择三种方法。新来源标记为 `precision-rgb-keypoints-pnp/v1` 和 `precision-rgbd-keypoints-pnp-refine/v1`；旧对照保留原来源。

## 训练数据与运行

新建 manifest，不把旧无高分辨率/角点字段的轨迹伪装成本方案数据：

```json
{
  "schema": "align-precision-keypoints/v1",
  "object_model": "upright-cube-4cm",
  "pixel_convention": "zero-based-pixel-centers",
  "records": [
    {"id": "scene1-t0", "scene": "scene1", "split": "train",
     "file": "scene1-t0.npz", "sha256": "该文件的SHA256"}
  ]
}
```

每个 NPZ：`rgb: uint8[H,W,3]`、`pixel_uv: float[8,2]`、`visible: bool[8]`；角点顺序是 `geometry.CUBE_POINTS`。遮挡标签必须来自实际可见性审计，不能把投影在画面内当作可见。深度用于测量阶段，本定位网络训练不读取深度/GT pose。多个帧按整个场景划分 train/development；未知 split、重复文件/样本、越界路径、训练 hash 错误均拒绝。

准备真实数据后，在云端明确给定预算运行，例如：

```bash
PYTHONPATH=.:src python -m experiments.align_precision_vision.train \
  --manifest /path/to/manifest.json --output /path/to/new-run \
  --steps 400 --wall-seconds 600 --device cuda
```

以上是调用示例。后续数据接入轮已建立 `torch 2.11.0+cu128` / ManiSkill 3.0.1 环境，并在仿真渲染数据上执行固定1000步定位pilot；未采集真实相机数据。首次实现轮的CPU测试环境和证据仍单独保留。

Expert 初始化示例：

```python
candidate = PrecisionActionExpert(existing_align_expert)
optimizer = candidate.optimizer(expert_lr=1e-5, geometry_lr=1e-3)
loss = candidate.flow_loss(context, proprio, action, mask,
                           conditions=conditions, mode="measured")
```

学习率只是显式配置示例，尚未做效果实验；监督仍为原 commanded-target TCP action，不用相对误差直接替换标签。Qwen 编码及已有 memory 的屏蔽/条件化由原调用方负责。三个实验模式为 baseline、oracle、measured，不混合报告。Oracle 必须用于明确标记的隔离对照。

## 分层验收

1. 测量：完整分母上的有效率、位置/姿态误差、尾部误差与时间偏差；错误与拒绝均保留。先独立训练视觉，再冻结测量。
2. 消费：固定context/proprio/噪声，测量置零或相对位置/姿态发生有意义变化时，预测是否产生足够明显、方向合理的响应。`geometry_response` 只给动作差；非零梯度或响应不等于使用正确。
3. 任务：相同数据和预算比较 baseline / Oracle / measured 独立 Align，维持8mm/5°出口和原夹爪要求；测学生轨迹和真实入口，最后再考虑其他技能。

数据采集与定位pilot按后续用户授权执行；Expert效果实验、闭环和自动训练尚未启动。

## 验证

```bash
OMP_NUM_THREADS=2 PYTHONPATH=.:src python -m pytest \
  experiments/align_precision_vision -q -p no:cacheprovider
```

测试使用合成几何、随机小网络及两步合成训练验证来源隔离和保存重载。流水线测试用固定关键点替身隔离网络准确率；Expert 是真实实现的小尺寸配置。测试在新云机执行，WSL 只编辑、传输和校验。最终次数和日志身份见同目录 `validation.md`。

以上 `validation.md` 是首次实现轮的历史验证。数据接入轮见 [data_results.md](data_results.md) 与 [data_plan.md](data_plan.md)。PnP改造轮云端49项联合测试通过，三组对照与限制见 [pnp_results.md](pnp_results.md)。

2026-09-10发布前，对本目录与 `experiments/maniskill_data` 的待发布快照逐文件核验SHA-256，并在云端隔离源码目录联合回归：**81 passed，5.18s，退出码0**。本次未重跑数据采集、development评估或正式训练；实验数值来自上面链接的既有运行记录。
