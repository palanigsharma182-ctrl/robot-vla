# 已知4cm模板 PnP 与内部深度细化：三组离线对照

2026-09-10，隔离 Precision Vision 实现与 development 诊断。**默认已切换 PnP-only；几何求解有改善证据，现有预测网络的位姿覆盖率仍为0。** 未重新训练，未运行学生闭环，未改变稳定 `src/` 或历史结果。

## 方法与冻结条件

假设：RGB轮廓角点对应的最近深度像素容易落在背景，已知4cm模板与2D对应点的 PnP 可消除这一依赖。三组共用同一帧的一次网络预测、原始320×240 RGB、内外参、角点顺序及 visibility≥0.8；姿态误差保留物体整体四重 yaw 等价。

| 方法 | 位姿计算 | 深度使用 |
|---|---|---|
| PnP-only（默认） | ≥4点；IPPE/SQPnP候选及RANSAC/LM，正深度与内点重投影选解 | 不读取深度 |
| PnP+Depth-refine | 同一PnP初始化，有界6D鲁棒最小二乘 | 仅腐蚀后的前景内部，用于一致性检查/细化 |
| old-corner-depth | 原角点同像素深度反投影+SVD，至少3点，原拒绝规则 | 仅历史对照允许角点深度 |

PnP：RANSAC 100次、seed17、阈值3px、confidence0.999；至少4个内点且比例≥0.6，内点RMS≤3px，模板全部位于相机前方。平面候选按观测选解，不读取GT；低重投影不能保证排除所有平面歧义。

当前没有分割头，内部前景明确采用 `R≥60, R>1.5G, R>1.5B` 与PnP投影轮廓交集，再以5×5核腐蚀。**只适用于已关联的单个红色方块，不是通用物体分割。** 不加载GT mask。内部保留32–256个有效深度点，单位米、范围0.05–2；按cube表面射线相交计算深度残差。细化平移各轴限制±10mm、旋转向量各分量±0.15rad、最多40次函数评估，soft-L1。接受需满足深度median≤5mm/p90≤10mm、重投影RMS≤3px且增幅≤0.5px、残差范数不增及优化成功。内部不足显式回退原PnP；初始深度不一致且细化失败则拒绝，不回退角点深度。

数据为此前采集的976帧中的236帧development，4个场景，external/wrist各118帧。冻结1000步定位器权重；不调阈值、不追加数据或训练。两个角点条件分别运行：`predicted` 为真实模型输出；`gt-diagnostic` 为GT角点/可见性，仅诊断几何求解上限。GT pose仅在求解后计算误差。

- Manifest SHA-256：`f387e329f589ef2705268fa024608448eb0a8524a879b6562f4b4e6df3879c8c`
- Checkpoint SHA-256：`7ab244bb5065328780fd3388119fe278b4cb8677c167bae7587701308f718135`
- 执行快照：`source-v2.json/tar.gz`，153个文件；基于 `b89d6970c12f8257e213666c0c1a320957334d4d` 加当时工作区修改，精确内容以快照为准。协议另存求解/比较/几何/网络源码hash。
- 每次比较上限600秒；v1首次比较后，仅补充优化异常处理与单方法评估默认PnP及测试，再用v2复测。四次比较全部退出0，未按结果调参。最终v2 predicted/GT循环分别约2.23/2.15秒（不包含进程启动及模型加载）。v1/v2全部分组聚合指标一致，不作为独立重复证据。

## 指标定义

Coverage = 有效位姿帧数 / 全部236帧。所有失败仍计入分母。误差在有效帧上报告median/p90/mean；无有效帧时为null，不能解释为零误差。

- Reprojection：所有通过visibility filtering的输入对应点的像素距离RMS，**包含RANSAC剔除点**；不同于求解器内部的内点RMS。
- Position：物体中心在base frame的欧氏距离，mm。
- Orientation：四种整体yaw等价中最小SO(3)角度，degree。
- 附加 accuracy coverage：位置≤8mm且姿态≤5°的帧数 / 全部帧，仅为诊断，不构成技能验收。
- 同时报camera/phase/scene分组、拒绝原因、深度回退状态，以及两方法共同有效帧，防止有效样本选择差异误导比较。

## 真实预测角点：主结果

| 方法 | Coverage | Reprojection median / p90 (px) | Position median / p90 (mm) | Orientation median / p90 (°) |
|---|---:|---:|---:|---:|
| PnP-only | 0/236，0% | null / null | null / null | null / null |
| PnP+Depth-refine | 0/236，0% | null / null | null / null | null / null |
| old-corner-depth | 0/236，0% | null / null | null / null | null / null |

0.8阈值与图内过滤后：118帧0点、61帧2点、57帧3点，**没有≥4点帧**。因此两种PnP均236帧 `insufficient_visible_pnp`，深度细化尚未进入。旧方法179帧点数不足、57帧刚体拟合拒绝。当前不能用这批预测证明PnP或深度细化提高了实际定位/控制效果。

## GT角点：独立几何诊断

| 方法 | Coverage | Reprojection median / p90 (px) | Position median / p90 (mm) | Orientation median / p90 (°) |
|---|---:|---:|---:|---:|
| PnP-only | 88/236，37.29% | 3.75e-6 / 7.05e-6 | 1.13e-5 / 3.42e-5 | 1.06e-5 / 4.26e-5 |
| PnP+Depth-refine | 88/236，37.29% | 0.2753 / 0.3971 | 0.7194 / 1.0976 | 0.3121 / 0.7637 |
| old-corner-depth | 1/236，0.42% | 0.3249 / 0.3249 | 7.2535 / 7.2535 | 14.8412 / 14.8412 |

PnP-only近零误差来自同一仿真几何的精确投影一致性，**不代表真实相机的纳米级精度**。旧方法仅1个有效样本，p90等于该样本，不能描述尾部风险。完整JSON另含mean。

88帧具有≥4个GT可见角点，两种PnP全部解出；余148帧仍因点不足拒绝。88个有效帧全部来自wrist（88/118），external为0/118。可见性标签/视角仍限制覆盖率，不能外推所有场景。

两种PnP共同有效88帧，全部在8mm/5°以内；深度分支86帧接受细化、2帧保持深度一致的原PnP。加入深度相对PnP使位置误差增加median 0.7194mm、p90 1.0976mm；在理想2D输入下未展示收益，因此保留PnP-only默认。不能由此推断有噪声的预测角点上深度一定有害。

旧方法唯一有效帧只有3个可见点，与PnP有效集合**共同支持为0帧**，且姿态不满足5°。因此旧方法与PnP的有效样本误差不是配对改善；完整分母下的coverage与accuracy coverage可比较：旧方法分别0.42%/0%，PnP两种方法均37.29%/37.29%。

## 验证与后续边界

云端 Python3.10.12 / torch2.11.0+cu128 / NumPy1.26.4 / OpenCV4.11 / SciPy1.15.3，RTX4090。命令：

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=.:src python -m pytest \
  experiments/align_precision_vision experiments/maniskill_data -q -p no:cacheprovider
```

最终 **49 passed in 3.21s，exit0**。覆盖PnP无深度依赖、外参、平面解、四重对称、RANSAC离群点、退化拒绝、内部腐蚀/轮廓深度污染不影响细化、合成range偏差改善、深度不一致拒绝、优化异常显式记录与回退、旧方法完全等价、默认pipeline/evaluator切换和失败分母保留。

可重复命令见README，原始证据为 `compare-{predicted,gt-diagnostic}-v2.json`、对应 `.protocol.json`、日志与退出码、`tests-v2.log/exitcode`，保存在私有run目录，不把逐帧数据提交GitHub。

已完成implementation、unit/synthetic tests和development offline comparison；未验证真实相机、噪声预测角点上的有效PnP精度、学生闭环与任务成功率。下一步应先区分GT实际可见点不足与网络可见性漏检，再验证有≥4个可靠预测点的定位；本轮没有擅自放宽visibility或启动训练。
