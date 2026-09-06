# 显式TCP几何与TCP动作的联合实验

当前执行配置已按用户批准更新为TCP关节上限 **0.1 rad/步**；训练身份和下文原0.05实验结果保留不变。修改范围、兼容方式与测试见[执行上限更新](execution-limit-01.md)。
用户于2026-09-07批准把两块耦合修改作为一个实验变量：base-frame Memory转换为当前TCP相对目标；Action Expert由joint增量改为TCP位姿增量。代码直接放main的实验目录，旧实验和canonical合同保持原身份。资源上限为本轮累计1800秒GPU进程墙钟、4GB新增磁盘（包含全部副本），单卡排队、不训练Qwen、不消费历史final test。

## 两个模块

- 世界Memory仍保存base位置、归一化协方差、时间和有效性。`geometry.relative_features`只读地计算目标相对当前TCP的位置，并旋转协方差；base中的“上方8cm”先组成目标再转换。缺失状态仍是全零+mask，转换不刷新观察时间。目标姿态尚未加入Memory，当前旋转标签来自已有教师的实际运动，不声称学会物体姿态相关抓取。
- `TCPActionSpec`定义`[Δx,Δy,Δz,rotvec_x,rotvec_y,rotvec_z,gripper_target]`，模型输出`[B,16,7]`；平移每分量10mm、旋转每分量0.1rad作为归一化边界，夹爪目标为[0,1]。整个chunk固定使用规划开始时的实际TCP轴向。旋转使用在此轴向表达的左乘增量，不能按Euler角直接相加。`TCPChunkExecutor`通过SAPIEN Pinocchio对同一个`panda_hand_tcp` link求IK、FK回代，再交给原joint执行器执行前4步。

本轮两arm共同每次重规划从actual状态重建command reference，第一步标签为actual→下一command，后续为相邻command差分。这样模型不需要猜不可观测的历史command reference。该共同规则是新实验身份的一部分，不改写旧joint实验的跨chunk参考语义。

## 对照与公平性

| 条件 | Memory特征 | 动作 |
|---|---|---|
| joint-world | 原base位置和协方差 | 7关节增量+夹爪，8维 |
| tcp-relative | TCP相对目标与旋转后的协方差 | 6维TCP位姿增量+夹爪，7维 |

两项同时变动，仅允许归因于联合方案，不声称已经分离输入表示和动作空间各自贡献。两组使用同一24条已核验教师轨迹、16train/8development、相同176/88窗口、同一采样/dropout与256次更新（每更新2样本）。新标签由actual/commanded关节通过已知FK导出，不把物体GT加入学生输入。所有标签必须落入预定范围，禁止截断标签以通过检查。

双方共享原Qwen、Adapter和Expert主体初值；由于动作维数和语义不同，双方的action input projection与velocity head都按相同规则重新初始化，Memory encoder也采用相同初值。不能直接拿之前已训练的joint head与新TCP head比较。这里只冻结Qwen/Adapter；两组更新Expert和Memory encoder。

主结果是新场景的每个实际控制步Reach（2cm）和配对最终距离；不同动作空间的FlowMSE不用于跨组排名。真实执行记录保存每步TCP、joint命令、模型chunk、IK目标及每步距离，以便区分模型方向错误与底层控制偏差。场景种子固定，Flow噪声按(scene,index)哈希派生，跨arm同场景共享规则。

执行中继续使用真实RGB-D→Memory更新，缺失时屏蔽并用实时图像规划；原有Memory过期中断、接触/终止/跟踪判定保留。IK失败、关节限位/增量越界明确保留为停止，不把拒绝样本移出分母。只在即将执行的完整四步前缀均可求解时发送动作。

## 验证与运行

先进行坐标/旋转/标签/模型梯度和身份测试，再执行六轴TCP控制smoke，然后顺序运行两组训练及新场景对照。复用当前无碰撞静止立方体夹具，不执行抓取或物理操作。必要的CPU测试不加载真实Qwen。

入口是`python -m experiments.tcp_memory_control.run {smoke,train,evaluate}`；各阶段必须提供`--output`和`--source-manifest`，训练另外提供`--data --checkpoint --model-cache`，评估提供`--training --checkpoint --model-cache`。外层启动器累计约束本轮全部进程时间和产物大小。源码manifest包含实际运行的依赖源码；数据、URDF、上游和新checkpoint身份单独记录。

状态：已完成实现、19项测试、六轴仿真smoke、两组训练与4场景/组的联合对照。TCP候选在4个场景中均因项目关节步长限制提前停止；未证明完整任务收益。事后只读IK诊断定位到旋转分量。详见[结果报告](results.md)和[脱敏数值](summary.json)。

最新联合方案对照已完成：原表示Baseline与TCP 0.1方案在新16个Reach场景均0/16成功，TCP提前停止。见[完整对照结果](baseline-comparison-results.md)。

该对照已补入教师TCP动作原样重放诊断：24/24完整执行88步，22/24 Reach，与原教师同长度前缀一致；教师所需最大关节修正约0.012 rad。两批场景分开统计，联合结论及后续排查见[完整对照结果](baseline-comparison-results.md#后续诊断教师tcp-action原样重放)。
