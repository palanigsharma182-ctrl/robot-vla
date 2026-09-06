# ManiSkill 基础接口 smoke

目的：确认项目的 commanded-target 增量经当前执行器转换后，能正确对接 ManiSkill 官方
`pd_joint_delta_pos`；比较官方 `pd_joint_target_delta_pos` 的可替代范围。
这是工程接口验证，不是任务成功率实验，也不更改默认控制模式。

依赖：Python 3.10、项目基础测试依赖及 `mani-skill==3.0.1`。无需 Qwen 权重、数据集、渲染或 GPU。

```bash
PYTHONPATH=src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python experiments/maniskill-interface-smoke/run.py
python -m pytest -q tests/test_maniskill_observation_boundary.py tests/test_chunk_executor.py tests/test_adapters.py
```

脚本在官方 `PickCube-v1` 中使用 CPU 物理、state 观测和固定合成动作。每种模式执行八步
`0.005 rad` 的第一关节目标增量、一条零增量，以及 reset 后一条零增量，共 20 个控制步。
seed 0/1 仅用于接口 smoke，不读取项目冻结的评估配置或数据。
记录官方下发给 PD 的目标并继续执行原控制方法，不替换物理仿真；不计算任务指标。

2026-09-06 验证：ManiSkill 3.0.1 / SAPIEN 3.0.3 / Gymnasium 1.3.0 / PyTorch 2.11.0+cu128 /
NumPy 1.26.4，20 步全部完成。

- 当前 adapter 的关节顺序、关节范围和控制器增量尺度与官方 Panda 匹配。
- 两次 replan 的目标连续；两模式八步目标和实际轨迹在 `1e-5 rad` 绝对容差内一致。
- 实际位置增量模式的零动作将目标设为当前位置；目标增量模式的零动作保留旧目标，
  本次与实际位置的最大差为约 `0.007612 rad`。
- reset 后两种模式的第一条零动作均以新 Episode 的状态为参考。

结论：官方目标增量模式可作为后续简化候选，但本次未覆盖限幅情况下的等价性，不能直接替换
当前执行器的 hold、异常重规划与 reference 管理。无完整 VLA、RGB 渲染或任务成功率结论。

观测修复的回归另外验证 V1、V2 起点 padding 和 V2 完整四帧：旧代码为 1 passed / 2 failed
（V2 空切片），修复后 3 passed；相关 adapters、executor、observation、contracts、runtime
共 47 passed。测试使用真实评估循环及合成观测，保留完整历史和当前物理速度的区别。
