# 抓取前 Object Memory 条件输入实验

状态：隔离 M0 候选。真实采集、两个条件的最小训练及 state-dict 重载完成；
本轮未见 Memory 额外收益，任务效果仍为 inconclusive。
研究问题：历史合格位置是否帮助 VLA 在当前图像信息不足时接近物体？

## 范围与观察权限

用户于本轮同意单物体显式 Memory 条件方案并要求开始实验。第一版维持冻结 Qwen、
V1 双图/15 维 proprio、16×8 commanded-target Action、原 Flow loss 和执行器。
新增训练变量为 Memory Encoder 与 Action Expert；Adapter 在候选中冻结。
这是隔离 M0 候选，不代表 canonical 策略晋升。实现放在 main 的实验目录。

观察行为分工：用户规定可用阶段/路线/资源范围；模型只提出请求；
已有 `ActiveFrontReobserveController.consider_trigger()` 按当前事实决定是否可请求；
实际 camera runner 在清除动作历史并获得控制后执行。Memory 和 Encoder 不持有 actuator。
当前已有 Supervisor 仅允许 ACQUIRE_TRACK / STABILIZE_PREGRASP，检查双路当前测量与
Memory 是否不可用、机械臂保持/相机 HOME 条件、接触与夹爪事件、冷却和每 Episode 一次预算。
默认禁止把缺少合格 provider 当作遮挡。真实调用者仍需提供连续、新鲜的测量证据，
不能把重复调用或每次 replan 冒充每控制 tick 观测。

本轮测试调用已有 Supervisor 验证权限边界，没有把它接成动态相机控制器。
接触、夹爪关闭、物体可能移动或 pregrasp 窗口结束后，静态 Object Memory 不再可用。
Memory 可用不等于 close/lift/release 授权。

## 输入与最小接线

`conditioning.py`：`snapshot_memory` 在当前规划时刻校验 Episode、年龄、来源、协方差增长及
抓取前使用条件，产生不可变 12 维特征与独立可用掩码；不修改 live Memory，不重写观测时间。
坐标为 robot-base 米，协方差为平方米。固定尺度为 1 m、配置最大标准差平方、配置最大年龄；
不从验证或 test 拟合统计。特征顺序见类 docstring，缺失全部清零且由 mask 屏蔽。

`MemoryConditionedPolicy` 是实验 V1 子类，将显式 `MemoryBatch` 放在本次 `model_inputs` 的
`experiment_object_memory` 项。`encode_context()` 在 Qwen Adapter 后添加一个 720 维 token；
训练 loss、普通 Flow 和 RTC 都复用原 Policy 实现。Qwen 不接收这一实验字典项。
没有跨调用缓存 Memory；省略输入或全 batch 不可用走原始上下文路径。
混合 batch 中，不可用样本的 token 使用 attention mask。Flow 迭代共用本次上下文/KV。
运行中的自动快照获取、候选 Runtime 包装及正式 checkpoint 消费尚未接入。

新增参数不能当成原 V1 checkpoint 已训练能力。本轮候选保存包含 Encoder/Expert 参数、
特征顺序与尺度、Memory/provider 配置、上游权重身份，并严格回载到同构离线模块。

## 已执行的接口验证

```bash
PYTHONPATH=src OMP_NUM_THREADS=2 python -m pytest \
  experiments/memory_conditioning/test_conditioning.py tests/test_policy.py \
  tests/test_e018_active_front_reobserve.py tests/test_e018_object_memory.py -q
PYTHONPATH=src OMP_NUM_THREADS=2 python experiments/memory_conditioning/probe.py \
  --output /path/to/new/probe.json
```

2026-09-06：现有远端环境 CPU 执行初版定向检查 136 passed；补充混合 batch 基线对照、
屏蔽输入梯度和 RTC/Flow 一致性后 138 passed，无 skip，退出码 0。
独立探针使用合成上游 context 与真实小型 Action Expert：缺失/全屏蔽与原路径一致、
条件动作有限、Memory 梯度有限且非零、冻结上游无梯度。没有 optimizer step，没有 actuator。
动作发生变化只能证明输入可影响计算，不能当作位置利用或任务改善证据。
最终补齐训练缓存插入点、采集权限与冻结协议反例后，远端定向回归 **148 passed**，
退出码 0。Sol/high 独立 R2 审查和 Terra/max 真实小数据复核完成；实际模型元数据已核验。
本机原始记录位于忽略目录 `artifacts/reviews/memory-conditioning-20260906/`。

## 真实数据盘点

父 Agent 已核验历史 D0/D1 的 manifest 校验和，并检查全部 train NPZ 的中央目录：
D0 为 176 条，D1 为 40 条，文件均存在。只读取字段名及少量数组头，不读取 RGB、GT payload
或 protected split。所有 216 条都没有显式逐帧 camera pose/time 数组；D1 全部有
`expert_supervision_mask`、`action_source` 和 `commanded_joint_target_rad`，D0 有 80 条
含 commanded target。数据已经在本机展开，不需要因“archive”命名而重新恢复。

这不证明旧数据不可用于任何训练，也不要求本候选必须变成 Observation V2。
它说明当前不能直接重放 D049 动态 PRIMARY/front 观测历史；若利用 q 与冻结机器人标定
重建历史 wrist 几何，还需验证坐标/时间/provider 适用性和动作标签，不能把重建值说成
已记录的实际传感器外参。无法重建的动态视角需要有界新开发采集。仿真 GT 位置不能
代替 deployable Memory 输入。本轮通过新开发场景补齐所需快照，没有重构旧轨迹或消费旧 test。

## 2026-09-06 有界真实小试

用户授权累计 30 分钟 GPU 采集/训练、4GB 新增磁盘，优先数据与最小训练。
`collect.py` 固定新 seed 1000100–1000111，前 8 个 train、后 4 个 development，不替换 seed。
复用正式 G2C/校准包及 D049 route；只有 HOME commit 且快照合格后才执行教师采集。
GT 只用于 MPlib 教师规划。标签来自实际发送的前 16 步 target，首步使用 HOME 历史重置后的
显式 initial command reference，后续相对上一发送 target。控制 correction 始终另用 actual q。
记录初始真实双图/15D proprio、实际 pose/time、Memory/provider 配置与源身份、命令及标签。
它是 V1 HOME snapshot 样本，不是完整 V2 时序轨迹。失败部分动作另存，不进入完整样本。

采集 12/12 场景，5 个合格样本：train 4/8（1000100、1000105–1000107），
development 1/4（1000110）。其余 7 个保留拒绝原因，没有换门槛、换 seed 或使用 GT 补 Memory。
采集进程耗时 38.89 秒，退出码 0；数据和审计文件约 1.58MB。
合格快照时刻 4.8s、末次测量 2.6s，年龄 2.2s；2.5s 上限下仅剩约 0.3s。
快照只代表规划开始时输入，不表示整段 16 步持续有效。

对照为同一上游初始化、相同 Expert 可训练参数/数据/优化器步数的无 Memory 策略；
treatment 仅增加合格 Memory token。`train.py` 缓存真实冻结 Qwen Layer12+Adapter context，
两个条件从同一已有 Expert 初始化，各 32 步、batch 1、AdamW lr=1e-5、Flow seed=42。
开发指标为每 scene 四组固定 noise/time 的 normalized action Flow MSE，取最后一步，无选择。
训练后另移除 Memory 作诊断。两个条件保存独立候选 state dict，记录 schema/config/数据 hash，
严格回载并复算相同开发输出；这不等于 fresh Runtime 候选消费已验收。
训练入口检查固定 seed/split/参数，并绑定采集记录。所有 12 个场景保留在资格分母中；
Flow MSE 仅在有合格标签的子集有定义，不给拒绝场景补零或按成功计入。
只有一个合格开发场景，结果限定为条件子集诊断，完整任务效果为 inconclusive。
关闭功能的原权重 parity 与训练后候选屏蔽 token 是不同实验，不混淆。

实际运行（development seed 1000110，每项为四组相同 Flow noise/time 的平均 MSE）：

| 条件 | 训练前 | 32 步后 |
|---|---:|---:|
| 无 Memory | 0.01399851 | 0.00362964 |
| 有 Memory | 0.01401346 | 0.00363381 |
| 训练后有 Memory 的模型移除 token | — | 0.00362276 |

有 Memory 比无 Memory 误差约高 0.115%；移除 token 也没有使误差变差。
当前样本没有额外构造遮挡，只有单一合格开发场景，不能据此判定 Memory 无效或有效。
它证明了真实 G2C Memory→冻结 Qwen context→Expert optimizer→候选保存/回载链路可运行。
两份 checkpoint 分别为 408,323,979 和 408,696,461 bytes，均 strict load 成功，
重算开发输出与保存前完全一致；没有 optimizer 状态，不宣称完整训练恢复 checkpoint。

训练前两次启动分别因设备参数类型和遗漏已有 Runtime BF16 AMP 上下文失败，均为 0 optimizer
steps，原始失败日志保留。修复使用原 Runtime/Stage1 的 BF16 autocast，两个条件一致，loss
保持 FP32 累加。父 Agent 明确告知用户调整自己原先的一次重试执行计划，没有增加用户预算，
没有改变实验 seed、门槛、32 步训练预算或指标。最后训练进程 21.07 秒，退出码 0；
包括采集 38.89 秒及两个失败进程 11.11/11.26 秒，累计 GPU 作业进程 82.33 秒。
两份权重及数据/日志共 81 个文件已从远端落到本机，逐文件 SHA-256 全部匹配；
本机与远端本轮运行产物合计约 1.65GB。权重为 exploratory candidate，不是 canonical 发布。

外层采集 timeout 600s、训练 timeout 1100s，总上限 1700s。所有原始记录保存在本机
`artifacts/reviews/memory-conditioning-20260906/real/`；候选关闭不影响 canonical 默认路径。
数据或协议不足时停止依赖它的训练，不以合成探针宣布 positive-signal。
正式/受保护 split、本体控制、完整任务效果及后续阶段均未由本次接口结果验收。
