# 原表示Baseline与TCP 0.1联合方案：预先固定的开发实验

用户于2026-09-07明确要求比较原base Memory＋joint chunk与新TCP相对Memory＋TCP chunk（关节执行上限0.1 rad），不以TCP 0.05/0.1为主对照。

假设：显式几何与TCP动作组成的联合方案，在适度放宽执行步长后能改善Reach原子技能。它不能分离三项修改各自贡献。

## 固定比较

- baseline：上一轮joint-world最后256更新权重，base Memory，8维joint chunk，0.05 rad执行上限。
- candidate：同轮tcp-relative最后256更新权重，TCP相对Memory，7维TCP chunk，0.1 rad执行上限。
- 两权重来自同24条轨迹、同256更新与共同预训练主体；双方投影头重置、actual-reference规则、Memory初始化/采样在训练时配对。不是直接拿更早一次历史joint权重混比。
- 本轮不重训；冻结Qwen、模型、数据、几何、action scale、20 Hz、每次执行4步、每场景88步及原Memory更新/遮挡/实时图像fallback。
- 跟踪误差0.05 rad判定在两组保持一致；这是actual与目标的残差阈值，不是候选0.1 rad的命令步长上限。位置/速度和其他终止判断不改。
- 两组逐场景验证初态一致。相同scene/plan index使用同采样seed；8/7维噪声不声称逐通道相同，状态分叉后也不声称逐动作配对。

## 场景和指标

旧场景1600100–1600103仅用于复查。新development场景1600200–1600215用于主比较；各组各16个新场景，加复查合计40次仿真。无历史confirmation/final test，无物理动作。

主指标：逐实际控制步距离≤2cm的Reach成功率，完整16场景分母保留停止/失败。初始点也计入，与既有协议一致。预先定义明显正信号为新场景成功率增加至少25个百分点且配对exact双侧检验p≤0.05；这是开发性证据，不是独立最终确认。先看任务成功，不能用拒绝减少替代成功率。

次指标：完成88步比例、逐场景最小距离、共同实际前缀末距离、Memory条件动作、实际关节速度、目标跟踪误差、拒绝/修正截断与其他停止原因。不同执行长度的终止距离不作为等时性能排名。新旧场景分别汇总，失败不删除。完整world/base/TCP及目标坐标只写入指标日志，不进入模型决策。

本轮是静止无碰撞立方体的抓前接近，夹爪保持打开。不能外推抓取、抬升、运输、放置或完整五技能。

## 运行边界

先CPU验证和源码审查，再六轴smoke与配对评估，现有单卡串行。用户已批准累计30分钟GPU进程时间、4GB新增磁盘（含全部副本），私有启动器累计执行并到限停止。只允许一次必要环境/实现修复；超预算保留not_run，不追加场景或调参追求显著性，不覆盖旧产物。

入口 `python -m experiments.tcp_memory_control.run compare-baseline`；沿用evaluate必要输入，原训练manifest与新运行manifest分别传入。每份结果保存evaluation_protocol、训练身份和运行源码身份。

状态：实现、14项CPU测试及Sol/high独立审查通过；两份真实权重SHA和184文件源码manifest核验通过。已完成40次仿真与CPU汇总：新16场景两组均0/16 Reach，Baseline全部完成88步，TCP全部在3–8步停止。详见[结果报告](baseline-comparison-results.md)。
