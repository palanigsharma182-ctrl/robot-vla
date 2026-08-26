# AGENTS.md

## 项目目标

这是一个用于学习 VLA、世界模型和机器人仿真并且面向求职的的端到端项目。

整体方向：

```text
VLA
→ World Model
→ Evaluator
→ Planner
→ Simulator
→ Data / Training Loop
```

目标是掌握真实机器人学习系统的工程方法，而不是从零重写所有基础模型。

## 核心原则

### 1. 开源优先

成熟基础能力直接复用：

- PyTorch / Transformers / LoRA
- ManiSkill / Isaac Lab
- 开源 VLA / VLM
- 预训练视觉、视频 Encoder
- 物理引擎、IK、标准训练工具

不要重复实现 Transformer、物理引擎、CUDA 等成熟基础设施。

### 2. 核心系统自己实现

重点自己实现和理解：

- Observation / Action Adapter
- 数据与时间对齐
- Action Chunk 执行
- Task / Success / Failure / Progress
- World Model Dynamics
- Evaluator
- Planner / Retry / Replan
- Dataset Pipeline
- Evaluation / Failure Analysis
- Closed-loop Training

开源模型不能只当黑盒 API 使用。

### 3. 重要决策不要擅自修改

不要静默改变：

- 模型架构
- Observation / Action 定义
- Action Horizon
- Task 定义
- Dataset 格式
- Loss
- 训练目标
- Planner 逻辑
- Evaluation 标准

需要修改时，先说明：

```text
问题
→ 可选方案
→ trade-off
→ 推荐方案
```

由用户决定重要方向。

### 4. 4090 优先

默认硬件：

```text
RTX 4090 24GB
```

优先使用：

- BF16
- Frozen Backbone
- LoRA
- Gradient Accumulation
- 小型可训练模块
- 分阶段训练
- 按需加载模型

不要默认依赖多卡或 80GB GPU。

### 5. 工程方式

修改代码前：

1. 先读现有代码；
2. 理解数据流和 Tensor Shape；
3. 给出简短计划；
4. 做最小必要修改；
5. 运行最小测试；
6. 汇报结果和问题。

避免无关重构和过度工程。

### 6. 调试优先级

出现问题时优先检查：

```text
数据
→ Shape
→ 坐标系
→ Normalization
→ Action 语义
→ 时间对齐
→ Controller
→ Label
→ Model
→ Loss
```

不要首先通过扩大模型解决问题。

### 7. 实验原则

每个实验必须回答明确问题，并保留简单 Baseline。

最终关注：

> 新模块是否真实提高机器人任务成功率，以及为什么。