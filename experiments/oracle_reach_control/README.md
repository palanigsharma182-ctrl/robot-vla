# GT 几何下的 Reach 控制对照

这是隔离的 M0 仿真诊断：比较历史 Oracle 几何编码器 + Expert/Flow 与已知运动学的显式位置伺服，回答“是否有必要让网络学习这一段几何到关节动作的映射”。不修改 canonical 默认控制路径，不重新训练，不消费受保护 final test。

## 冻结方案

- 10 个 development seed：1200000–1200009。运行前检查与训练 manifest 无交集；每个 seed 分别 reset 两组，核验相同初始 q、TCP 和物体位置。
- 两组都读当前 GT `object_position - TCP_position` 与当前 proprio；夹爪共同固定张开，不增加姿态目标。目标仍是物体中心，成功阈值仍为 TCP→物体距离 4cm。
- 共同使用现有 ActionAdapter、安全限幅、关节控制器、16步动作块/执行前4步、temporal ensemble decay 0.5，最多100个实际控制步。两组按seed奇偶交替执行顺序。
- 学习组：已恢复 Oracle checkpoint `294b555fd16e2c6ceb2528b167289cdb94edb93d36844d16715cae58f89c9f05`，10步Flow。夹爪固定处理意味着它不等同上一轮未经此处理的历史复现条件。
- 显式组：复用 Franka URDF/FK，中央差分计算位置雅可比，阻尼最小二乘解关节变化；damping=0.02、gain=0.25、每步笛卡尔位移上限0.005m、差分步长0.0001rad。按已有joint limits处理；动作标签明确相对于此前 commanded target。
- FK使用真实robot root平移并验证root旋转为单位阵；每回合检查FK与仿真TCP差小于1e-5m。不读取物体姿态、OBB或专家抓取路径。
- 主指标：每组成功数/10。辅助记录最终距离、实际步数、错误、限幅、异常重规划及最终物体位移；执行耗时不作正式性能benchmark。
- 沿用前轮累计900秒GPU进程、2GB新增磁盘上限；本轮进程最多800秒。10对完成即停止，不按结果调参或增加seed，保留所有失败。

## 解释边界

这比较的是两种完整控制方法，不是仅更换空间表示的严格消融。显式组使用已知运动学，学习组使用历史数据训练的模型；它们没有同等训练成本或架构。正结果只支持在这个简单位置接近问题上优先考虑显式控制，不证明复杂接触、姿态对齐、抓取、Memory或视觉误差问题已经解决。

## 运行

依赖当前项目已有仿真环境。权重和metadata在私有存储中，只传入路径，不纳入Git。

```bash
PYTHONPATH=.:src python experiments/oracle_reach_control/runner.py \
  --checkpoint "$ORACLE_CHECKPOINT" --data "$DATA_METADATA" --output "$NEW_OUTPUT"
PYTHONPATH=.:src python -m pytest experiments/oracle_reach_control/test_runner.py -q
```

结果与实际资源记录见运行完成后生成的 `results.md`。
