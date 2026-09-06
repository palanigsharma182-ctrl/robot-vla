# RGB-D Memory进入训练和实际执行：第二阶段

状态：已完成新数据采集、两臂训练、三条件仿真执行及事后只读XYZ诊断。真实Memory消费成立，三条件Reach均为0/4，未得到任务性能正信号；详见[结果报告](results.md)和[脱敏数值](summary.json)。用户批准本轮累计30分钟GPU进程时间、4GB新增产物（所有副本计入）。只使用现有单卡、现有依赖和已有Qwen/Expert权重。

## 目标和边界

验证第一阶段测量生成的Memory是否真正进入训练样本、参与Expert梯度更新，并在新场景逐步执行中被消费。复用第一阶段三面/已知4cm立方体估计器，不改变其已发布结果或把它推广成任意物体provider。本轮隔离的抓前接近实验，不运行完整五技能，不接触、抓取或训练Qwen；静止运动学目标无碰撞，不声称物理任务成功。

沿用V1双图/15D proprio、冻结Qwen第12层上下文与Adapter、16×8 commanded-target增量Chunk、20Hz执行前4步。高分辨率front640×480 RGB-D用于几何；Qwen前端使用同帧front缩放到128×128及原wrist128×128，两臂完全相同。独立格式`rgbd-pregrasp-memory-policy/v1`，不复用旧五技能checkpoint身份。

## 预定协议

- 24个新采集seed：16 train、8 development；4个另行预定rollout seed。全部位置和姿态由seed生成，没有按Memory是否成功替换场景。高度5/12/19/26cm，X/Y小范围变化；具体常数见`protocol.py`。
- 每场景先按HOME→第二视角→HOME观察，各3个真实hold步；每次视角切换只重置候选窗口。随后教师或策略执行时相机固定HOME，逐步消费新RGB-D。相机直接设置是仿真实验操作，不是学到的观察策略。
- 遮挡开关按发送下一动作前的已完成步数判断：计数16–75时遮挡，对应动作后观测step 17–76；随后恢复视野，遮挡3秒超过2.5秒TTL。夹具位置只用于场景生成，不传入学生测量/策略。wrist可能仍看到物体，因此不能把该条件称为双相机完全失明。
- 教师使用现有阻尼最小二乘位置伺服，以GT物体中心上方8cm为目标，实际执行96步、夹爪始终打开。GT用于教师和评估，不进入学生输入。只对实际执行命令生成标签，按上一command target作差，不把actual-q correction当标签。
- 每8步取一个anchor，标签为其后实际执行的16步；输入Memory只取anchor时刻快照。没有Memory的样本仍保留并屏蔽token，不丢掉失败场景以提高曝光率。原始RGB-D/标定、逐步测量/快照和动作参考保留私有。
- 两臂从同一上游Expert权重开始；Visual训练Expert，Memory训练Expert+现有12D Memory encoder。固定256更新/臂、累计2样本/更新、lr1e-5、相同样本顺序和Flow噪声；Memory对可用快照做25%预定dropout。Qwen/Adapter无梯度，视觉context缓存不含Memory。
- Memory需要至少4个train场景、1个development场景可用，否则停止训练并报告数据不足；不放宽测量门槛补分母。development只作诊断，不选checkpoint；保存预定最后一步。推理checkpoint不带Adam状态，不宣称无损恢复训练。
- 三个执行条件：Visual权重且屏蔽Memory；Memory权重且启用Memory；同一Memory权重但屏蔽Memory。每条件相同4个新场景，最多88步。使用现有Chunk executor，保留跨规划command reference；不使用ensemble/RTC。Memory失效中断余下动作并清空reference，以新帧重新规划。
- 每次规划绑定当前快照与实际双图/proprio/指令，消费一次后清除。每个动作后再更新Memory，旧快照不可用时触发中断。所有条件夹爪固定打开；不让随机夹爪输出混入接近对照。

## 验收与停止

分别报告：数据中可用/缺失/过期Memory比例；真实训练token曝光、encoder梯度与参数变化；保存重载一致性；新场景Memory规划数/动作数、遮挡期消费、失效回退与恢复；相同权重/同帧/同噪声屏蔽Memory的关节动作差异；完整分母下接近2cm比例、最终距离与失败。

工程消费成立要求数据及训练有非零真实Memory曝光、encoder有有限非零梯度和参数更新、新场景存在Memory-conditioned实际动作，且同权重屏蔽能改变关节动作。收益必须另看三个执行条件，不能用梯度、loss下降或动作差异冒充任务成功。24个采集场景及12个执行单元的失败、未运行全部保留。允许一次有界工程修复，不追加seed、不按结果改阈值。

外层runner累计记录GPU进程墙钟和全部产物，超时终止并保留部分记录；达到预算或共同数据协议失效时停止后续阶段。源码在main实验目录，稳定src及第一阶段冻结源码不变。单写者集成，独立审查覆盖数据因果性、坐标、checkpoint身份及执行中断。
