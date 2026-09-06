# Memory、一次观察与视觉回退的五技能开发实验

本批按用户明确批准在 main 实现，现有单卡累计上限为 7200 GPU 作业秒、20GB 新增磁盘。
包括两次数据采集、两臂训练、开发评估及工程修复；不使用历史 validation/final test。
仍为隔离实验，不更换 canonical 默认入口，也没有物理机器人实验。

本批已完成采集、两臂各1024次更新、120/120开发评估和新权重回退探针。
完整五技能任务三组均为0/4；新评估场景没有实际消费有效 Memory，收益结论为
**inconclusive**，不能据此认定 Memory 无效。数值与完整分母见 [results.md](results.md)。

## 实际数据流

真实双图/15D proprio → 冻结 Qwen 与 Adapter → Action Expert；合法的抓前 Object Memory
由 12D Encoder 生成一个附加 context token。每次重规划重新绑定同帧、同 Episode 的快照；
一次 Flow 求解期间快照不变，逐动作发送前后仍重验 Memory 的年龄、接触和抓前范围。

Memory 不可用时，12D 特征清零并屏蔽 token。用户已选择：撤销旧动作、ensemble、RTC 和
command reference，保留真实控制时钟，再用实时双图重新规划继续。零步中断需要新帧时
实际 hold 一步；终止或控制故障停止，不被解释成可恢复的 Memory 失效。

重观察请求由规则与 Supervisor 产生；Qwen 没有训练观察决策头。相机按已验证的 PRIMARY
路线采集，返回 HOME、建立四个新帧后才允许 D049 提交。HOME raw score 仅作候选请求线索，
不能被称为合格三维测量；测量和 Memory 写入资格保持原规则。每 Episode 最多一次观察。

## 文件与复用边界

- `runtime.py`：候选权重、特征配置、归一化、训练身份和 arm 的严格验证；一次消费快照。
- `session.py`、`controller.py`：逐控制步检查、暂停、失效清理和已批准的视觉回退。
- `protocol.py`、`collect.py`：固定新场景及真实抓前教师序列；失败与已发送动作保留。
- `data.py`、`train.py`：train-only D0 回放、新序列身份/标签/时间检查、共享条件拼接和两臂训练。
- `evaluate.py`：复用可信专家的起点准备与项目成功谓词；评估五原子技能、四相邻组合和完整任务。
- `summarize.py`：只读聚合训练曝光、场景等权 loss、完整评估分母、观察拒绝及技能交接条件分母。
- `probe_runtime.py`、`verify.py`、`probe_return_time.py`、`probe_fallback.py`：工程诊断入口。

稳定源码只增加必要的 Runtime 编码 hook、同步动作历史清理及控制器发送前中断 hook。
数据 schema、Flow loss、动作单位/归一化、默认执行前4步与现有评估成功谓词保持不变。

## 采集与训练协议

D0 使用已核验的176条 train轨迹；保留原 action mask 与跨技能标签。从中固定选取每技能
128个窗口，共640个。缺失历史相机位姿/时间的 D0 不重建 Memory，全部屏蔽 Memory token。

新序列固定32个场景：前24 train、后8 development。每4个真实帧形成一个锚点，只有实际
执行的未来16步构成监督。物体 GT 只用于教师路径和评估，不作为模型或 Memory 输入。
不把采集的 pregrasp fragment 伪装成完整五技能成功轨迹。

采集 v1 得到141个窗口，但没有合法锚点覆盖 Memory 从有效变成到期。保留 v1 原始证据；
v2 对偶数 seed 增加28步真实 HOME hold，奇数保持零等待。原 last-observed 时间与2.5秒
TTL 不变，教师从等待后的真实状态规划。该修正是 development 数据覆盖修正，不能将相同
seed 称为新的独立评估。v2 同样32/32执行、0实现错误，得到110个 train、31个 development窗口：

| 数据范围 | 有效 Memory 锚点 | 屏蔽锚点 | 有效→到期的场景数 |
|---|---:|---:|---:|
| train | 27 | 83 | 3 |
| development | 8 | 23 | 2 |

两臂均从同一已恢复 Layer12 Expert 初始化，冻结 Qwen/Adapter，分别训练 Visual Expert
和 Memory Expert+Encoder。每臂1024次更新、每次累计4个单样本，学习率1e-5、共同 Flow
噪声与数据顺序。每次3个 D0 加1个新序列，Memory 候选对合法 Memory 做25%人工 dropout。
这使抓前窗口约占40%、其余技能各约15%；**并非五技能均匀曝光**。两臂曝光完全相同。

缓存最多600秒、每臂训练最多1000秒，评估与保存预留400秒；外层执行使用3000秒硬超时。
只使用最后一次预定更新，不按 development loss 挑 checkpoint。保存实际更新步数、优化器、
随机状态、全部失败分母、采样顺序、代码与数据身份；未完成的 checkpoint 不用于成对效果比较。

## 开发评估

120个固定单元 = 4个新 development seeds ×（5原子技能 + 4相邻组合 + 完整任务）×3条件。
运行前写全分母，状态区分未运行、进行中、完成和实现错误。每单元160/240/400个实际策略
控制步，观察前缀另报；共同墙钟由内部deadline与外层2600秒超时限制。

| 条件 | 权重 | 起始观察 |
|---|---|---|
| visual | Visual checkpoint | 不主动移动相机，Memory屏蔽 |
| fixed | Memory checkpoint | 三个真实 HOME tick 后提出一次请求 |
| evidence | 同一个 Memory checkpoint | 三个连续 HOME score 低于固定开发候选值时提出请求 |

请求候选值为0.6123920381069183，借用已有分值作开发候选，**不是校准过的请求阈值**。
HOME 仍只输出 score-only evidence；请求同时依赖真实 hold、相机几何、pose/time 和现有
qualified wrist capability 缺失。这里只检验起始三个真实 tick 的规则，不声称学会运行中
任意时刻的观察策略或支持无限刷新。

从 grasp 及以后开始的原子/组合没有可重建的抓前 Memory，直接屏蔽。组合只在起点做
专家准备，交接不 reset、不补教师动作。报告初始/最终技能进度、条件及无条件成功分母、
未到达技能、Memory/视觉推理数、观察次数、拒绝和失效。同步仿真时间与 GPU 墙钟分别记录；
不能由这些结果外推物理相机速度或实机任务收益。

## 已有工程证据与当前状态

- 初始四场景验证保留全部结果：2次信息增益不足拒绝、1次成功提交并消费 Memory、
  1次因策略提前关闭夹爪而不能请求。早期 PRIMARY 常量接线错误已修复并保留原失败。
- 在同一个已见开发场景，仿真 RETURN 从40步改为10步，消费开始年龄由2.2s变为0.7s，
  有效 Memory 参与动作由7步变为37步；不是任务成功率结论。
- 真实旧M0权重回退探针通过：10次有效 Memory 推理后，2次新的 masked 视觉推理继续发送
  动作，历史清空、时间新鲜、无第二次观察。旧32步M0权重只作工程验证，不作新训练初始化。
- 最终相关 CPU 回归210通过、1个 CUDA 用例未执行，退出码0；独立 Sol/high 审查覆盖控制、
  Memory与评估，Terra/max审查覆盖数据和训练，均对所审范围给出限定 PASS。实际模型由会话
  turn_context核验；静态审查不替代运行结果。
- 两臂均完成1024次更新，训练进程624.38秒，退出码0；120/120开发单元全部完成，评估进程
  2091.47秒、退出码0。逐项检查没有实现异常或进程预算中断；任务失败均保留。
- 在同一个已见工程场景，以本次1024步 Memory checkpoint 重跑回退探针：10次有效 Memory
  推理后因 `memory_stale` 清理动作历史，再有2次新鲜 masked 视觉推理并实际继续动作；
  六项检查全部通过，进程25.69秒、退出码0。source-v12只为原探针增加显式训练结果入口，
  严格绑定 checkpoint SHA、训练身份和 memory arm；未改变场景、TTL、动作或观察次数。
  该记录证明新权重的工程链路，不增加独立任务效果证据。

## 瓶颈与允许结论

1. **合格信息覆盖先于动作收益。** 固定观察在四个新场景的三类抓前起点共请求12次，均因
   `information_gain_below_threshold` 拒绝提交。起始视角分数增量约0.0014–0.0119，低于现有
   0.10门槛；这12次并非12个独立场景。证据候选没有触发请求。当前结果检验了观察路线与
   提交覆盖，尚不能检验有效 Memory 对动作的帮助；不能为制造正结果事后降低门槛。
2. **训练对有效 Memory 的曝光不足。** 每臂4096次样本曝光中只有182次 token-on，即4.44%。
   Memory开发 loss 比视觉高0.229%，同一 Memory权重屏蔽 token后的误差几乎相同；没有额外
   收益信号。少量场景、1024次更新和抓前片段均限制结论，不能称作全量训练收敛。
3. **动作与技能交接仍是独立瓶颈。** reach原子1/4、transport原子2/4；grasp/lift/place在
   专家准备起点各4/4。transport+place实际到达transport完成的分母为4，之后visual仅1/4、
   两个Memory权重条件0/4；完整任务均0/4。组合控制预算较长，不能把其第一技能进度直接
   当作与原子预算相同的比较。结果支持检查接近、持续抓持和交接状态分布，尚不证明单一原因。

下一批优先独立检查真实几何误差与视角/提交覆盖，再补能反映策略实际交接状态的数据；
随后才有依据决定是否增加 Memory训练曝光或训练量。当前 Object Memory仅覆盖抓前物体，
抓持状态和目标区域 Memory 尚未接入，不能把它描述为五技能全过程 Memory。

累计GPU进程墙钟3055.10秒（50.92分钟），包括所有已保存失败尝试与两次采集；
本批远端与本地新增目录逻辑大小合计约7.90GB，含两地副本，低于20GB上限。
新数据、两组完整checkpoint、评估和探针均有远端源与本地 SHA-256一致副本。
Drive副本仍未验证，保留远端产物，未执行释放操作。

## 重放入口与身份

在项目根目录设置 `PYTHONPATH=.:src:experiments/memory_conditioning`。采集入口为
`python -m experiments.memory_reobserve.collect --bundle <G2C包> --output <新目录>`；
训练入口为 `python -m experiments.memory_reobserve.train`，必需参数为 `--data`、`--d0`、
`--checkpoint`、`--model-cache`、`--output`、`--source-manifest`。训练外层使用
`timeout 3000s`。评估入口为 `python -m experiments.memory_reobserve.evaluate`，必需参数为
`--training`、`--upstream`、`--model-cache`、`--bundle`、`--output`，外层使用 `timeout 2600s`。
输出目录必须是新的；这些命令会消费 GPU，不能当作无成本检查。

本次采集 v2、训练与评估分别保存 source-v9/v10/v11 的逐文件 SHA-256 清单及源码归档，
包括当时未提交的代码；不能只凭共同 HEAD 重现。训练使用 source-v10 的共享训练接口，
评估 source-v11 在加载端增加 expected_arm 校验，权重格式相容。早期失败和采集 v1 保留。
公开聚合保留训练身份、两臂 checkpoint SHA-256、原结果 SHA-256和完整协议；私有路径、
原始轨迹、权重、详细日志与审查会话留在本地控制目录，不进入 Git。
