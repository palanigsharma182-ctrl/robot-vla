# 七个独立技能策略：BC、接管式 DAgger 与采样覆盖对照

2026-09-09。分别训练 Approach、Align、Grasp、Lift、Transport、Lower、Release 七份独立 Expert，保留各自优化器、checkpoint 与纠偏桶；共享冻结编码器及只读特征缓存。架构见 [七策略与上层调度](../skill_dagger/hierarchical_policy_plan.md)，初始方案见 [训练方案](../skill_dagger/independent_training_plan.md)。

**本阶段实验已完成，尚未通过七技能完整验收。** [最新结果](results.md) 汇总全部成功与失败，[机器可读摘要](results.json) 绑定各训练、父权重、源码及评估结果身份。原始数据、权重和逐帧日志不包含在仓库中。

| 技能 | 初始 BC | 首轮继续 BC | 首轮 DAgger | 扩大原示范覆盖 BC |
|---|---:|---:|---:|---:|
| Approach | 1/8 | 1/8 | 8/8 | — |
| Align | 5/8 | 2/8 | 0/8 | — |
| Grasp | 8/8 | — | — | — |
| Lift | 8/8 | — | — | — |
| Transport | 6/8 | 7/8 | 5/8 | 7/8 |
| Lower | 5/8 | 3/8 | 4/8 | 6/8 |
| Release | 8/8 | — | — | — |

各列都是同一组八个标准入口 development 场景，不是最终 test 或扰动/交接验收；破折号表示未运行。Align 第二轮继续 BC 为3/8、DAgger为0/8，父权重与首轮不同，单列在结果页。Lower 6/8伴随一次提前松开导致的抓持丢失，保留为待诊断候选，不直接替换基线。

## 实现

- `data.py`：单技能监督在第一次技能出口截断，包含完成出口的动作，尾部允许只有一步；纠偏数据核验来源、文件hash、技能与teacher-only标签；支持跨轮回放。
- `train.py`：只更新指定技能Expert，冻结Qwen、adapter、memory encoder，Memory masked。每128次保存，结束严格重载；缺共享缓存即报错。纯BC每更新4个原窗口，`--corrective`对应3原+1纠偏，`--replay-manifest`绑定历史纠偏集合。
- `runtime.py`：教师从真实学生状态恢复，持物时使用实际TCP–object偏置；20Hz，每次最多执行chunk前4步再观察。仅采集Approach/Align时可在错误闭爪命令发送前接管，普通评估不应用该guard。
- `run.py`：`teacher/evaluate/collect`，学生必须指定技能与checkpoint SHA；标签只来自教师接管后的真实动作，保存失败分母。
- `aggregate.py`：合并互斥train分片，校验来源并拒绝重复、跨技能或开发集数据。
- `diagnose_mask.py`：对既有真实train入口进行同噪声离线mask比较，不执行环境动作；其单步结果不能替代闭环验证。
- `test_data.py`、`test_runtime.py`、`test_aggregate.py`：标签边界、来源拒绝、回放索引、采样续接、采集guard与合并约束。

纯BC的`--nominal-offset`默认0，表示从同一确定性采样序列跳过多少个窗口；跨桶末尾继续下一遍，训练噪声seed保持不变。非零offset暂不支持与纠偏混用，避免混淆四窗口序列与实际三个原窗口的消费。本次覆盖对照使用offset2048。

## 运行与验证

这是依赖既有模型和已审计数据的隔离研究实现，不是下载仓库即可重建结果的独立样例。路径由命令参数传入；需要匹配的上游checkpoint、离线模型缓存、教师collection、训练配置与冻结特征缓存。源码manifest绑定实际依赖文件，输出必须是新目录。私有调度与存储控制脚本未包含在本次发布中。

在项目根目录、已配置的云端CUDA/ManiSkill/SAPIEN及规划环境中运行：

```bash
PYTHONPATH=.:src python -m experiments.seven_skill_dagger.train --help
PYTHONPATH=.:src python -m experiments.seven_skill_dagger.run --help
PYTHONPATH=.:src python -m pytest experiments/seven_skill_dagger/test_data.py experiments/seven_skill_dagger/test_runtime.py experiments/seven_skill_dagger/test_aggregate.py -q -p no:cacheprovider
```

设置`CUBLAS_WORKSPACE_CONFIG=:4096:8`；使用既有BF16/math SDPA配置。最新冻结代码的21项针对性测试通过，原始执行记录为3.35秒。训练、测试与仿真均在云端执行。

下一阶段仍需解决Align对齐与Lower抓持稳定性，再验证合法扰动和六个相邻技能的真实交接。七项未通过前不运行完整Pick→Carry→Place；GT触发的上层切换仅可报告为oracle调度诊断。
