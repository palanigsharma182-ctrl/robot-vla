# Align Precision Vision：独立精确视觉与 Action Expert 条件接口

2026-09-10。用户批准的隔离实现：Qwen 负责语义与目标关联，Precision Vision 负责几何测量，当前 Align Expert 负责动作。稳定 `src/`、七技能原模型和历史结果保持原样。本目录没有机器人动作发送接口。

当前提供可训练的定位网络、RGB-D 位姿计算、Align 条件接口、Expert 的 BC loss/Flow 推理和针对性测试。**尚未采集本方案的真实关键点数据、训练合格定位权重或运行新的学生闭环。合成测试通过不代表视觉精度达标。**

## 分工与范围

```text
Qwen context ───────────────────────────────────────────┐
proprio[15] ─────────────────────────────────────────────┤
原始 RGB → 8个物体角点 + 可见性                         │
                  + 同帧深度(m)/内参/OpenCV外参          │
                  → base_from_object                    │
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
- `geometry.py`：可见关键点和 OpenCV 光轴 Z 深度反投影；已知模板对应关系通过刚体 SVD 得到物体位姿。至少三个非共线有效点，检查点间距离和拟合残差。缺深度、错误单位、退化、刚体不一致保留拒绝原因，不用背景深度或点云质心补造中心。目标是 `base_from_object @ object_from_pregrasp`，不是物体中心本身。
- `pipeline.py`：绑定 RGB-D/外参时间、episode 和 target，计算当前条件并调用 Expert。当前 TCP 位姿必须来自该 proprio 对应的 FK，由调用方提供对应时间。在线 `predict` 的 `now_s` 必须为同一传感器时基的时钟函数，测量后和动作生成后重新计龄；推理耗时导致过期则抛出 TimeoutError，不能继续执行该 chunk。四重对称采用当前 Align 合同中绕目标接近轴的旋转等价；抓取变换须符合这一中央对称抓取定义，不可任意用于偏置抓取或非对称物体。
- `expert.py`：复制已有15维proprio、7维TCP动作的独立 Expert；保留旧状态投影，另加8维几何投影。新增权重零初始化，两路结果在原状态第一层相加。输入通过函数参数传入，无可变全局条件/forward hook，避免跨调用残留。提供 `flow_loss`、`predict`、两组显式学习率和同噪声 `geometry_response`。
- `train.py`：独立定位器训练入口，读取带 hash 与场景 split 的 NPZ manifest；只打开 train 样本。保存配置、精确采样序列、模型/优化器和完成结果，固定最终步并严格重载。不提供从该文件精确续训的承诺。

网络输出 `[B,8,2]` 原图像素中心坐标与 `[B,8]` 可见性概率。结构化输入 `[B,8]` 为：

```text
[dx/0.05, dy/0.05, dz/0.05,
 rx/0.5,  ry/0.5,  rz/0.5,
 valid, age/max_age]
```

平移单位米，旋转为当前TCP轴向的旋转向量（弧度）。缺失条件带 `valid=0`，送进 Expert 时整个额外分支为零，得到同一权重的无几何预测；这不意味着已对齐，也不保证回退动作成功。`condition.reason` 和原测量必须进入实际评估记录。有效性/年龄不是夹爪或技能退出许可。

候选工程默认：可见性≥0.8，光轴深度0.05–2m，拟合RMS≤2mm、点间距离误差≤4mm，最大年龄50ms、传感器错配25ms。这些是待验证配置，**不是已校准置信度、已验证定位精度或新的技能验收标准**。误差大的点可能导致拒绝，须同时统计覆盖率。三点拟合精确也不证明对应点身份或目标位姿正确。

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

以上是调用示例，未执行该真实训练。此次新云端建立的是 `torch 2.11.0+cpu` 测试环境，实际 CUDA 训练需先准备 GPU 依赖；不会自动把 CUDA 请求降级为 CPU。

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

不会因本实现完成而启动这些效果实验或恢复自动训练。

## 验证

```bash
OMP_NUM_THREADS=2 PYTHONPATH=.:src python -m pytest \
  experiments/align_precision_vision -q -p no:cacheprovider
```

测试使用合成几何、随机小网络及两步合成训练验证来源隔离和保存重载。流水线测试用固定关键点替身隔离网络准确率；Expert 是真实实现的小尺寸配置。测试在新云机执行，WSL 只编辑、传输和校验。最终次数和日志身份见同目录 `validation.md`。
