# E013 Layer-12 在线相对几何 smoke

> **Status: superseded before execution.** 本 smoke 只测 Frozen Qwen Layer-12 单帧表示，不能替代
> 最终物体放置闭环；E013 后续把正式目标调整为厘米级也不会自动恢复本实验。其 10–15 mm
> deployable-candidate 阈值只保留为历史表示诊断，不得作为新 E013 的 promotion gate。当前文件保留
> 旧设计和命令供审计；没有由此生成正式结果。新的 canonical 方案见
> [E013 — 厘米级闭环精调执行层](e013_precision_execution.md)。

本 smoke 在 E013 正式采集和四轮训练前回答一个更小的问题：冻结 Qwen Layer 12 在四步双视角
Observation V2 中，能否在 **测试阶段不读取 GT token index** 的情况下自行定位当前 external 图像中的
方块和目标区域，并达到值得继续正式实验的 world-XY 精度。

它是资源与表示门禁，不是 Action 策略训练，也不是闭环抓取/放置效果实验。通过不能写成“机器人已经
可部署”；失败也不能证明完整 Qwen 表征绝对没有空间信息。

## 隔离数据

- 只使用 seeds `39000..39099`，与 E013 正式 collection `40000..42199` 和 evaluation
  `43000..43139` 完全隔离；
- 固定采集 30 条完整成功 Expert trajectories：train/validation/test=`20/5/5`；
- 这些 trajectory、probe checkpoint 和 raw RGB 禁止进入 E013 正式 Dataset、训练或效果统计；
- Observation V2、Action semantic、camera transform、时间同步和 `F_L/F_R` 必须通过现有 audit；
- 任一 split 数量、seed 范围或 identity 不一致时，probe fail closed。

建议采集命令：

```bash
python -m robot_vla.cli.collect_maniskill \
  --output /work/e013-v2-online-geometry-smoke-data \
  --train 20 \
  --val 5 \
  --test 5 \
  --start-seed 39000 \
  --max-candidates 100
```

若 100 个 candidate 内不能得到固定的 `20/5/5`，保留失败记录并停止，不减少轨迹数或改变 seed 范围。

## Probe 边界

- 输入：`t-3..t` 四个连续且 Observation V2 全模态有效的控制步，每步 external→wrist，共八图；不足
  四步或任一历史步无效的 window 必须排除并计数，不能用 padding window 充当四帧测试；
- 表征：固定 revision 的 Frozen Qwen Layer 12；
- 候选 token：当前 `t` 的 external visual tokens；这些 token 已由完整八图 prompt 上下文化；
- 输出：分别面向 red cube 和 green goal 的可学习 token selector，以及连续 normalized-UV decoder；
- 训练：GT object/goal UV 和最近粗 token 可以作为小 probe 的训练监督；
- validation/test：selector forward 不接受 GT UV、GT token index、object state 或 goal state；GT 只在 forward
  之后计算误差；
- world metric：仅在 Reach window 中将预测像素射线与已知桌面高度平面求交；
- 不测：Z、TCP orientation、Action 解码、接触动力学和完整闭环成功。

建议运行命令：

```bash
python -m robot_vla.cli.probe_v2_online_geometry \
  --data /work/e013-v2-online-geometry-smoke-data \
  --model-cache /work/model-cache \
  --output /work/e013-v2-online-geometry-smoke-result \
  --device cuda \
  --qwen-batch-size 1 \
  --probe-batch-size 64 \
  --train-samples 512 \
  --validation-samples 256 \
  --test-samples 512 \
  --epochs 30 \
  --seed 913013
```

输出目录必须为空。正式读取 `experiment.json`、`metrics.jsonl` 和 `summary.json`；`probe.pt` 只在私有
归档中保存，不上传 GitHub。

## 预先冻结的解释

`summary.json` 同时报两个层次：

1. coarse Reach screening：
   - object median world-XY `<=0.02 m`；
   - object p90 world-XY `<=0.04 m`；
   - 不允许无效 world unprojection。
2. deployable-precision candidate：
   - object p90 world-XY `<=0.01 m`；
   - goal p90 world-XY `<=0.015 m`；
   - object→goal relative-XY p90 `<=0.015 m`；
   - 不允许无效 world unprojection。

解释规则：

- coarse Reach 失败：不启动 E013 四轮正式训练；先处理在线目标寻址或视觉空间表示；
- coarse Reach 通过但 deployable candidate 失败：只能说明值得做粗 Reach attribution，不能把当前架构称为
  最小可部署；是否继续完整 E013 必须明确保留这一限制；
- 两者都通过：只允许继续 E013 其余 GPU/ManiSkill 和闭环门禁，仍不能替代完整成功率；
- 任何结果后都不得调整阈值、改 seed、删除失败窗口或将 GT token selection 加回 test。

## 租用与时间

本机是 RTX 4060 Laptop 8GB，且本地 Python 没有正式 PyTorch/Transformers/ManiSkill 环境，不能运行
本 smoke 的真实八图 Qwen 前向。建议先短租一张 RTX 5090 32GB：

- 16+ CPU cores、64GB+ RAM、至少 500GB 本地 NVMe；
- 驱动、CUDA 和 PyTorch 必须支持 Blackwell `sm_120` 与 BF16；
- 先验证八图 batch 1 峰值显存，禁止为通过 smoke 缩短 history、减少图像或降低冻结分辨率；
- 预估环境与模型缓存 30–60 分钟、30 条采集 30–90 分钟、Qwen 特征抽取与 probe 20–60 分钟，建议
  预留 2–4 小时；
- 若 batch 1 OOM 或峰值显存超过约 30GB，停止并换 H100 80GB，不静默改变实验配置。

完成后把 smoke Dataset、result、环境 receipt 和 SHA-256 保存到私有本地归档；GitHub 只发布脱敏的
聚合 `summary.json` 与分析，不上传 NPZ、RGB、权重、stdout/stderr、物理 GPU UUID 或私有绝对路径。
