# TCP执行上限调整到0.1 rad/步

2026-09-07，用户明确批准适度放宽到0.1 rad。本次是执行配置修改及CPU验证；未重训，也未启动新的GPU闭环评估。

## 行为

- 当前 `run smoke/evaluate` 使用 `TCPExecutionCandidate`：相邻IK关节目标增量和相对actual的跟踪修正共用0.1 rad/步上限。
- 20 Hz、执行前4步、关节位置/速度限制及ManiSkill ±0.1 rad映射不变；超过上限仍拒绝，跟踪修正饱和仍显式记录并停止。
- joint-world基线仍为0.05 rad/步。TCP模型的平移/旋转归一化、标签、Memory、权重及原 `TCPChunkExecutor` 不变。
- 新执行身份为 `tcp-memory-execution-limit/v2`，结果分别记录运行源码、原训练源码及训练身份。既有0.05结果和原始产物保持原身份。

## 消费已有权重

原权重仍严格核对checkpoint hash、训练identity、动作合同及参数。当前源码manifest必须包含新增execution模块并通过全部hash校验。

使用旧训练权重时，在原evaluate命令上增加 `--training-source-manifest <原训练source-manifest-v4.json>`，同时 `--source-manifest` 指向当前运行源码manifest，`--output` 指向新的空目录。兼容检查核对原manifest与训练identity中的SHA，仅允许旧文件中的evaluate.py/run.py发生变化；FK/IK、数据、模型、训练、几何及稳定src必须与训练时一致。它是这次执行修改的限定兼容路径，不是任意改代码后复用权重的许可。

复现原0.05评估使用原冻结源码及其manifest；不能拿当前0.1入口覆盖原结果。

## 已完成验证

- 远端现有Python环境，关闭CUDA：`test_execution.py` 与 `test_executor.py` 共13项通过，1.49秒，退出码0。
- 0.08和0.1 rad动作均执行4步、零修正截断；0.101 rad在发送前被拒绝。
- 旧0.05执行器仍拒绝0.08；两种模型的输出尺度未改变。
- 原训练manifest缺失、hash错误、几何等非执行入口变化、新execution模块缺失、额外未知文件及旧文件删除均被拒绝。
- 真实184文件运行manifest校验和原训练manifest兼容检查通过，退出码0。只验证源码兼容；本次未加载真实权重再次推理。
- 本机最初无法收集测试，原因是缺少PyTorch；实际测试转到现有远端CPU环境，无依赖安装。CPU导入SAPIEN出现Vulkan警告，本次不使用渲染。

Sol/high独立审查发现并推动修复了current-only未知文件未被拒绝的问题；修复后源码限定通过，父agent核验13项测试全部通过。

放宽后的完整仿真跟踪、实际关节速度及Reach仍未验证；此前四个拒绝计划低于0.1不意味着后续新计划或任务必然成功。
