# Memory 与一次观察的开发结果

以下为固定预算内的开发实验，不是历史 final test 或实机验证。训练使用预定1024次更新，不代表全量数据收敛。

| 训练条件 | 每场景先平均的 Flow MSE | 更新次数 |
|---|---:|---:|
| visual | 0.012152241 | 1024 |
| memory | 0.012180057 | 1024 |

Memory 相对视觉的开发 loss 变化为 +0.229%（越低越好）。有效 Memory 实际占训练曝光的 4.44%。
开发 loss 先在各场景内平均，再对8个场景等权平均；有合法 Memory 的仅3个场景，不能把多个锚点或噪声抽样当作独立场景。

| 任务 | visual 成功/完成 | fixed 成功/完成 | evidence 成功/完成 |
|---|---:|---:|---:|
| reach | 1/4 | 1/4 | 1/4 |
| grasp | 4/4 | 4/4 | 4/4 |
| lift | 4/4 | 4/4 | 4/4 |
| transport | 2/4 | 2/4 | 2/4 |
| place | 4/4 | 4/4 | 4/4 |
| reach+grasp | 1/4 | 1/4 | 1/4 |
| grasp+lift | 4/4 | 4/4 | 4/4 |
| lift+transport | 3/4 | 3/4 | 3/4 |
| transport+place | 1/4 | 0/4 | 0/4 |
| full-task | 0/4 | 0/4 | 0/4 |

每格计划4个场景，所有任务和条件复用这4个新 development seeds，不能把120个单元视为120个独立场景。完整执行不等于任务成功。原子技能由专家准备起点，组合只在起点准备，交接不重置或补教师。
fixed/evidence 共用同一 Memory checkpoint。evidence 仅在起始三个真实 HOME tick 使用未校准的 score 请求候选；HOME 分数不被冒充合格三维测量。

| 条件 | 完成单元 | 请求 | 合格 Memory 提交 | Memory 推理 | 纯视觉推理 |
|---|---:|---:|---:|---:|---:|
| visual | 40 | 0 | 0 | 0 | 1279 |
| fixed | 40 | 12 | 0 | 0 | 1304 |
| evidence | 40 | 0 | 0 | 0 | 1304 |

**本次闭环没有实际消费有效 Memory，Memory 动作收益不可估计。** 这能诊断当前观察/提交链的覆盖问题，不能作为 Memory 本身无效的结论。

| 相邻组合 | visual 成功/已完成前一技能 | fixed | evidence |
|---|---:|---:|---:|
| reach+grasp | 1/2 | 1/2 | 1/2 |
| grasp+lift | 4/4 | 4/4 | 4/4 |
| lift+transport | 3/4 | 3/4 | 3/4 |
| transport+place | 1/4 | 0/4 | 0/4 |

抓后原子技能没有伪造的 Memory；抓前 Memory 失效后清理旧动作并用实时视觉继续。未运行、实现错误、任务失败和完整分母保留在配套 JSON 中。

训练、数据和评估协议见 [README.md](README.md)；配套聚合结果见 [summary.json](summary.json)。
