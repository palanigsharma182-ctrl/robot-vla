# 教师TCP action原样重放

本目录是[TCP联合对照实验](../tcp_memory_control/baseline-comparison-results.md#后续诊断教师tcp-action原样重放)的后续执行链诊断；保留独立运行入口与来源身份，结果在父实验报告中统一解读。

2026-09-07。用户要求把教师TCP action通过当前0.1 rad执行链重放，检查教师能否完成Reach。本实验不调用Qwen/Action Expert、不训练，也不依据在线误差或GT目标重新生成动作。

## 动作来源与原样消费

源数据是已核验的24条教师轨迹，每条96步。实际存储的是actual proprio、commanded joint target及previous command；TCP训练标签一直由FK确定性转换得到，并不是另有一套手工TCP教师。

本入口使用与`tcp_memory_control.data.prepare_examples`相同的标签规则：chunk第一步为记录的actual→next-command，其后为相邻command目标差分，全部表达在该chunk记录的TCP轴向；normalized float32标签再按模型相同方式解码为物理动作。针对性测试要求这些标签逐值相等。

在仿真前冻结source index 0–87的全部chunk，以真实执行步数索引；禁止缩放、旋转置零、截断教师标签、在线修正和重新调用教师。若Memory中断四步前缀，则按已执行步数取下一份预先冻结的chunk。每次执行前后核对动作hash，原数组只读。

每次仍通过原执行器要求的live TCP anchor执行，原标签数值不改。因此实际状态与源轨迹分叉后，标签的世界目标也可能偏离；记录teacher/live anchor的平移和旋转偏差。不能把此实验称为逐步恢复教师状态、逐点世界目标伺服或在线教师重规划。

重放前88步，与之前模型协议时长一致；原数据有96步。最后一些16步chunk只有9–15个源动作，只有不会执行的尾部使用中性padding；每个实际执行的前缀均来自原数据。88步终止仍由既有controller边界执行。

## 不变条件

- 同源场景seed与初始化；warmup后必须与原教师proprio和视觉输入digest匹配。
- 20 Hz、最多4步/次、0.1 rad命令/修正上限、0.05 rad实际跟踪残差、位置/速度及其他停止判断。
- 相同逐步RGB-D/Memory更新和执行中断。Memory只参与既有执行边界；教师动作不由Memory生成，继承trace的`action_used_memory`仅表示Memory可用时绑定执行，不能作为教师预测消费Memory的证据。
- 源NPZ/metadata/observations hash匹配之前训练身份，URDF和全部运行源码固定。
- GT来源的教师命令仅作为明确标记的诊断输入；目标位置只用于结果计算。

## 验收与解释

完整24条分母，保留原16 train/8 development标记；这不是新的泛化测试。记录88步完成数、逐步≤2cm Reach、源教师相同前88步的距离、停止原因、跟踪误差和anchor漂移。源教师reference从已存actual q通过FK及同一world_from_base重建，不是另一次闭环运行。

教师成功可以支持执行链能处理教师分布的动作，不能证明任意TCP输出、物理控制或所有原子技能可靠。教师失败则须区分原样动作重放中的状态漂移和真正的IK/跟踪失败，不能直接归罪于标签或模型。

资源累计承接上一轮已批准30分钟GPU/4GB上限，扣除前一轮239.332秒和已产生副本；不重置预算。先CPU标签/来源核验和独立审查，再运行一轮仿真。必要修复最多一次，超限停止并保留证据。

入口：`python -m experiments.teacher_tcp_replay.runner {preflight,replay}`，显式提供`--data --training-identity --source-manifest --output`。状态：21项CPU测试、真实24条数据preflight及独立审查通过；2,112个冻结chunk在来源检查修复前后hash完全相同，GPU重放已完成：24/24完成88步，22/24 Reach，与原教师前88步一致。详见[结果报告](results.md)。
