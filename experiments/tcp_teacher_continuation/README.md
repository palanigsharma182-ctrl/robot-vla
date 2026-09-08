# TCP 学生继续学习教师策略

2026-09-08：用户明确授权从现有学生继续训练，不预设步数或时长预算，达到收敛后通知。范围仅为已有 TCP 学生与原教师数据的离线续训；不增加数据、改架构、改动作语义或运行仿真。不将本次 checkpoint 自动晋升为 canonical 或闭环可用。

## 冻结项与唯一主要变量

主要变量为继续优化原学生。复用 16 条训练轨迹的 176 个窗口；8 条 development 轨迹的 88 个窗口只用于固定评估与候选 checkpoint 选择。全部窗口、标签、Qwen/Adapter 上游、TCP Expert/Memory 结构、归一化、BF16、AdamW 学习率 `1e-5`、累计两样本、Memory dropout 0.25 和采样 10 步 Euler 均沿用原实现。

旧学生 checkpoint 的 SHA-256 为 `eaf50c82b8a897261ecca1a25154e750bc23b477d0382c54d9c53023dd449a27`。加载所有已训练 Expert 和 Memory 权重，**不重新初始化动作投影**。旧文件没有 optimizer 状态，因此这次 AdamW 从空动量开始；这是续训的已知差异，不声称逐位衔接原优化轨迹。新抽样使用独立固定 seed `20260908`，保留原有放回抽样分布；每个窗口曝光次数被记录。以后本轮 `--resume` 恢复权重、optimizer、步数和随机状态。

## 固定评估与停止

开始前在全部 264 个教师窗口复现旧结果，原 development MAE 最大绝对差需不超过 `1e-5`。每新增 512 次更新再次评估全部窗口，固定原 `sampling_seed(scene, anchor)`。统计 train/development 的 Flow MSE、前四步每步平移和旋转误差、四步累计目标平移和旋转误差，同时记录完整窗口预测与 IK 0.05/0.1 rad 通过数。所有轨迹均有 11 个窗口，窗口均值等于先按轨迹求均值再平均。

操作性收敛标准在运行前固定：对两种 split 的五项连续指标，最近八次评估中前四次和后四次均值的相对差都不超过 2%，且八次最大值与最小值之差不超过均值尺度的 10%；连续三次检查成立才标记 `offline_converged`。这是当前固定数据、噪声、学习率下的离线平台，不能证明全局最优或闭环成功。

开发集候选选择分数为四步累计平移和旋转误差各自相对起点的比例均值，越小越好。若 train 已平台、development 分数连续四次超过历史最佳的 1.2 倍，停止并标记 `development_regression`，不称为收敛。非有限 loss/梯度、输入身份不符或环境异常标记失败并通知；用户可随时停止。不因步数或耗时达到某值停止，也不自动改学习率来制造收敛。

## 产物与恢复

在本轮独立输出目录中保存 `config.json`、输入核验、每次不可变评估、`history.json`、`status.json` 和完整 `latest.pt` / `best.pt`。后两个是本轮明确可更新的恢复点，原 checkpoint 和原实验结果只读。退出信号在当前更新/评估结束后保存恢复点。失败不覆盖已有健康恢复点。

训练仅在云端 GPU 执行；本机承担控制、文本记录和产物持久化。通知检查只读取状态，在训练正常推进时保持安静；收敛、退化、失败或需要用户处理时通知。完整恢复数据与渲染环境不是本实验依赖。

```bash
PYTHONPATH=.:src python -m experiments.tcp_teacher_continuation.run \
  --data DATA_ROOT --training ORIGINAL_TRAINING_ROOT \
  --checkpoint UPSTREAM_CHECKPOINT --model-cache MODEL_CACHE \
  --source-manifest ORIGINAL_SOURCE_MANIFEST --output NEW_RUN_DIRECTORY
```

同一输出目录的显式恢复加 `--resume`；不能使用旧冻结训练器加载本轮新格式 checkpoint。关闭终端不应结束云端作业，启动器应使用持久会话并记录 PID 和退出码。

启动记录：第一版在缓存首个图像 context 时缺少 BF16 autocast，Adapter 报输入/权重 dtype 不匹配，尚未进行训练更新。修复为与原诊断一致的 autocast 后使用新的运行目录，保留第一版源码与失败日志；是否启动更新仍以旧 checkpoint 固定评估复现为前置条件。
