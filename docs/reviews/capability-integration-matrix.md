# 33 项能力：模块归属、实际接线与验收

2026-09-06；接续 [原始接入清单](e018-main-integration-checklist.md)，工程基线 `b244c3a`。
按用户确认补实际接线和全部 33 项验收对应关系，直接维护 main。训练、数据、执行仍保留原归属。
下方追加正式 G2C / D049 的真实工程消费验收；前一阶段的 synthetic-debug 结果保持原边界。

## 前一阶段接线（合成验收）

```text
现有 checkpoint 加载 / 冻结 Precision U-Net
                       ↑
实际相机观测 EO → front qualification 输入适配（42 维状态）
                       ↓
同一次 forward：关键点 + visibility + projection + sigma + 可选 mask
                       ↓
按预测 object UV 采样 object/goal mask → CP 校准 sigma → OO 写入证据
```

`active_front_provider.py` 从 E018 `37851ad` 复用，负责显式相机角色适配，是五模块的消费者。
新增模型输出到校准工具的连接复用现有关键点校验和双线性 mask 采样，不另建推理平台。
默认 `include_mask_probability=False` 保留原预测数值和训练路径；开启时只增加同次 forward 的 mask 解码。
单帧 helper 要求 keypoints `[1,K,2]` 与 masks `[1,M,H,W]` 对应，拒绝缺失或非法 mask。

新的 [模型接口回放](../../experiments/precision_module_integration/README.md) 使用实际小型 U-Net 和
临时 synthetic-debug checkpoint，校准、图像、状态均为合成 fixture；没有恢复 G2C 正式权重。
front 输入仍 qualification-only，未具备合格三维测量时 `geometry_valid=False`，不写 Memory。
原 [五模块回放](../../experiments/five_common_replay/README.md) 继续验证合成测量进入 OM 的状态规则。
这两层验收不能拼接成“正式 provider 已写 Memory”的结论。

为如实保存未训练模型，synthetic-debug provenance 允许零样本、零 optimizer step；formal-training
仍要求真实训练计数，按 formal 角色加载 debug checkpoint 必须拒绝。没有修改 checkpoint 参数结构或默认模型。

## 对应关系

五模块简称：**OM**=`object_memory`，**OO**=`object_observability`，**EO**=`active_external_observation`，
**FC**=`active_front_camera`，**CP**=`calibrated_front_provider`；均位于 `src/robot_vla/precision/`。
“无直接依赖”表示保留原模块，不创建转发层。表内源码相对 `src/robot_vla/`，测试简称 `x` 指
`tests/test_x.py`。每行测试仅支持该行所列工程边界，不代表整项实验复现；共享测试不重复计数。

### A：基础主线与 E001–E013（14 项）

| ID / 能力 | 原归属与消费者 | 与五模块关系 | 定向测试 | 本轮验收边界 / 剩余项 |
|---|---|---|---|---|
| A01 机器人合同 / 映射 | `contracts.py`、`adapters.py` → data/runtime/controller | front 输入复用现有 normalizer；不把关节映射放进 FC | `contracts`、`adapters` | 单位、shape、归一化规则；真实硬件驱动未验 |
| A02 trajectory/v2 / 审计 | `data/` → Dataset/training | 无直接依赖；观测工具不改 dataset label | `trajectory_v2`、`dataset_v2`、`writer`、`data_audit` | 合成文件 schema/mask/统计；真实数据 manifest/stats 未恢复 |
| A03 Qwen / Expert / Flow | `model/`、`training/flow_matching.py` → VLA | 无直接依赖；五模块不接管模型架构 | `policy`、`expert`、`qwen_context`、`flow_matching` | Fake Qwen、小张量 forward/loss；真实 Qwen/Processor 未验 |
| A04 Stage1 / checkpoint | `training/`、`evaluation/checkpoint_selection.py` | 无直接依赖；Precision 继续用原 checkpoint loader | `stage1_training`、`checkpoint`、`checkpoint_selection` | ToyPolicy 保存/恢复与选择规则；正式权重/CUDA resume 未验 |
| A05 action 执行语义 | `execution/chunk_executor.py` → controller | 无直接依赖；OM validity 不授予 actuator 权限 | `chunk_executor`、`adapters` | commanded reference、correction、reset；不切换官方控制模式 |
| A06 任务 / 采集 / rollout | `tasks/`、`sim/`、`evaluation/` | EO 是观测 sidecar；不代替任务/evaluator | `pick_place_tasks`、`atomic_rollout_evaluation`、`rollout_evaluation`、`maniskill_observation_boundary` | 合成任务/失败规则与观测边界；完整模型 rollout 未验 |
| A07 event / sampling / loss | `data/events.py`、`data/sampler.py`、Flow loss | 无直接依赖 | `events`、`sampler`、`flow_matching` | 标签/采样/权重运算；不重跑 E006/E007 效果 |
| A08 temporal / anomaly | `execution/temporal_ensemble.py`、`runtime/` | 无直接依赖；保留原消费者 | `temporal_ensemble`、`runtime` | 组合/replan 回归；不继承主动视觉收益 |
| A09 probe / Oracle 诊断 | `diagnostics/`、`evaluation/checkpoint_sweep.py` | 无直接依赖；Oracle 不进入五模块部署输入 | `oracle_reach`、`qwen_layer_reach`、`qwen_spatial_probe`、`gradient_conflict`、`checkpoint_sweep` | 合成诊断/冻结规则；无真实 Qwen probe 复现 |
| A10 RTC | `execution/rtc.py` → runtime 可选路径 | 无直接依赖，保留默认关闭 | `rtc`、`runtime` | CPU VJP/关闭 parity；排除 CUDA 节点；E011 不晋级结论不变 |
| A11 Local DAgger | `local_dagger_protocol.py`、trajectory/dataset | 无直接依赖，保留来源兼容 | `local_dagger_action_budget`、`trajectory_v2`、`dataset_v2` | 合同/预算规则；六候选不 eligible 的历史不变 |
| A12 V2 observation | `observation.py` → Dataset/Processor/runtime | EO 复用位姿转换；front 适配复用单帧 42 维状态布局 | `observation_v2`、`dataset_v2`、`qwen_processor_contract`、`runtime`、`checkpoint`、`e018_active_front_provider` | 单帧适配不代替 V2 四帧时序；真实 Qwen V2 forward 未验 |
| A13 Precision 感知 | `precision/{model,provider,geometry,checkpoint,data,training,held_out}.py` | **新增真实模型输出 → CP/OO；front 输入消费 EO/FC** | `precision_unet`、`precision_detection_provider`、`precision_geometry_control`、`precision_checkpoint`、`precision_module_integration`、`precision_detection_adapter`、`precision_data`、`precision_training`、`precision_held_out` | 模型/重载、标签隔离、训练配置和评估规则；正式 G2C / D048 消费见 C08；三维仅限固定高度 FREE_STATIC 平面，不代表任意物体高度估计 |
| A14 shadow / Executive | `precision/{control,shadow}.py`、`executive/` | 不把 OO 分数或 OM 有效性直接接 actuator | `precision_geometry_control`、`hierarchical_executive`、`deployable_state_estimator`、`shadow_executive_observer`、`runtime` | observer/控制仲裁边界；完整 shadow rollout 和物理控制未验 |

### B：E014–E017（7 项）

| ID / 能力 | 原归属与消费者 | 与五模块关系 | 定向测试 | 本轮验收边界 / 剩余项 |
|---|---|---|---|---|
| B01 长尾分类 | `precision/outliers.py` → 离线诊断 | 几何诊断保留，旧 confidence 不等同 OO 分数 | `precision_outliers` | 分类/几何计算；不把事后 taxonomy 当已修复 bug |
| B02 observability 合同 | `precision/observability.py` → label/Memory | OO 复用像素索引，保留独立 Object evidence/score；CP 采样预测 mask | `precision_state_memory`、`e018_object_observability`、`precision_module_integration` | exists/projected/observable 与 mask 规则；GT 派生标签不冒充预测 |
| B03 Goal Memory | `precision/state_memory.py`、`memory_evaluation.py` | OM 复用 GoalState/双状态快照，Goal 更新语义不变 | `precision_state_memory`、`e018_object_memory` | reset/time/age/双状态；历史 unsafe write=2 不改为通过 |
| B04 corrected 监督 | `precision/e016_pretraining.py`、`losses.py` | 训练归属保留，不进入五个推理工具 | `e016_precision_pretraining` | 合成 batch 梯度/全负定位监督；不重跑 128 帧 overfit |
| B05 E016-P1 训练/评估 | `precision/e016_training.py`、`e016_evaluation.py` | 保留原训练/评估；不能自动给 OM qualified provider | `e016_precision_formal_training`、`e016_precision_held_out` | 合成配置/metrics/临时 claim 校验，**未消费真实 held-out**；epoch12 与 unsafe=4 结论不变 |
| B06 E017 两行微调 | `precision/e017_training.py` | 无直接依赖，失败路线保留历史 | `e017_precision_training` | 小张量梯度/配置规则；不恢复微调或宣称效果改善 |
| B07 threshold=0.525 | E017 事后诊断记录 | 不写入 CP/OO 默认阈值 | 无专属独立验收 | **仅历史诊断**；没有新 held-out 确认，不标测试通过 |

### C：E018（12 项）

| ID / 能力 | 原归属与消费者 | 与五模块关系 | 定向测试 | 本轮验收边界 / 剩余项 |
|---|---|---|---|---|
| C01 Object Memory | `precision/object_memory.py` | **OM/OO 直接实现**，五模块回放消费 | `e018_object_memory`、`e018_object_observability`、`object_memory_replay`、`five_common_replay` | 正式 G2C → 条件平面三维测量 → OM 已接通；真实工程 smoke 验证一次提交与一次拒写，详见追加验收 |
| C02 相机路线 | 历史 `e018_p1_g0.py::_run_route` | **FC/EO 工具已接入**；原 route runner 保留实验 | `e018_active_front_camera`、`e018_active_external_observation`、`five_common_replay` | 几何、来源、运动阶段规则；本轮没有重跑真实路线 |
| C03 视角筛选/动态路线 | 历史 G0B/G0C runner | FC/EO 可复用；筛选协议不并入工具 | 无专属当前 runner 验收 | **历史实验保留**；工具测试不证明视角合格，不在线用 GT 选视角 |
| C04 动态观测/supervisor | EO；历史 `active_front_reobserve.py` | **EO 与 D049 supervisor 源码已接入**；实验入口显式请求固定 PRIMARY | `e018_active_external_observation`、`five_common_replay`、`precision_module_integration` | HOME-only 四帧已真实验收；实际为空的历史组件已清除；自动触发、非空 VLA 缓存和完整 G1 编排仍未验 |
| C05 wrist→front 适配 | **新增** `precision/active_front_provider.py` → 模型消费者 | 消费 EO/FC，输出资格验证输入 | `e018_active_front_provider`、`precision_module_integration` | 相机角色/42维/时间/skew/来源；原 G2A inconclusive 不变 |
| C06 covariance 校准 | CP 与历史 G2B runner | **CP/OO 已接预测 mask、sigma** | `e018_calibrated_front_provider`、`precision_module_integration` | 数值/通道/batch 对应；CAL-v2 protocol-invalid，不宣称校准机制失败或通过 |
| C07 G2C 训练/选择 | 原 G2C data/training/model_val 实验 | 基础 Precision loader 已复用；不搬训练到 CP | 无专属 G2C 验收；基础接口见 A13 | 正式 epoch15 已恢复并严格加载；**训练/选择 runner 未恢复或重跑**，不改变历史 checkpoint 选择 |
| C08 D048 qualified provider | `qualified_front_provider.py` → D049 frame | **正式权重、统计、D046/D048 包已恢复并核验**；EO/FC/OO/CP 共同消费 | `qualified_front_provider`；远端 CUDA BF16 严格加载及真实相机 forward | D049 score 保持 raw sigma，scale 仅乘 covariance；绑定实际 HOME/PRIMARY K 与 pose，Memory-write 限 PRIMARY，资格仍限 simulation development |
| C09 Stage2A/D049 编排 | `active_front_memory*`、`active_front_reobserve`；实验局部 consumer | **三帧 → HOME 四帧 → source recheck → commit/no-commit 已接入** | `e018_active_front_memory`、`e018_active_front_reobserve`；两条真实工程 route | seed 1000001 拒写；已知历史开发样例 76903 提交一次。非独立效果证据；7/25=28% 的负结果不变；未接 VLA 重规划/操纵 |
| C10 wrist 触发语义 | D050 capability/observability 区分 | OO predicate 不等同 qualified wrist capability | 无专属 qualifier 当前验收 | **语义保留，资格证据不足**；无效输入不自动触发相机运动 |
| C11 Stage2B 七视角 | 历史配置 + 用户未跟踪草稿 | 未来按问题消费 EO/FC/CP；本轮不接 | 未运行、未修改用户草稿 | **候选/未验收**；不为编号完整而启动 |
| C12 Stage3A/B | D051 冻结方案 | 未来 fault/comparison 消费工具，保持实验归属 | 无完整矩阵/效果对照验收 | **计划/未实施**；局部 fault 单测不替代 Stage3A/B |

## 追加验收：正式 G2C → D049 Memory（2026-09-06）

[实验入口与运行说明](../../experiments/g2c_memory_integration/README.md) 复用冻结 E018 `37851ad`
的 D049 三文件，仅新增薄 provider/实验 consumer，没有引入整套 G0/G2C/Stage2A runner。
原始 sigma score 与校准 covariance 分开，固定 FREE_STATIC 方块中心高度 0.02 m；
严格绑定权重、normalizer、D046/D048 receipt 和同源实际相机几何。

修复了 main OM 与旧事务消费者的两处兼容问题：提交后保留 RGB 采集时间；普通 commit 拒绝不应用
只读 preview 产生的年龄状态。安全失效规则保持。main `object_memory.py` 和 V2 evaluator 未变。
独立 R2 检查覆盖 provider 身份替换、低分字段一致性、相机资格、可变数组、校准数学，以及实验时序/拒绝落盘。

| 实际执行 | 结果与边界 |
|---|---|
| 冻结包恢复 | epoch15 文件 SHA-256 `97e3b7289911bc73f67755a8d9c3598c50b6c80ef01e1af13cec698ec59d3d77`；两份统计和全部资格/校准组件匹配；远端 CUDA/BF16 严格加载通过 |
| 最终定向测试 | 212 passed，2.62 s，退出码 0；无 skip；与前一阶段 435 项不直接相加 |
| 新开发场景 1000001 | 96 simulator ticks、4 次正式 G2C forward、PRIMARY 三帧合格、HOME 四帧；信息增益不足，0 次写入，正常拒绝终态，退出码 0 |
| 已知历史开发样例 76903 | 96 ticks、4 次 forward、HOME 四帧；minimum score 0.6212714911、gain 0.6200649013；4.8 s 时提交 1 次，退出码 0 |
| Action / 控制边界 | 无 VLA 推理；实际缓存最初为空并已失效；Memory 提交后 observable_now=false、contact_authorized=false |

拒写 route 使用增加固定 case 和日志字段之前的 consumer 快照；其 provider/Memory 与交付一致。
成功 route 的 manifest 与当前 consumer/provider/Memory 完全匹配。
所有真实 route 的 arm/TCP 平移漂移为 0。76903 按已有正样例选定，属于工程回放，不能计为新 unseen
效果或成功率。未消费历史 selection/final-test，不恢复 Stage2B/3 或训练。
两次前置工程失败均保留：ManiSkill batched quaternion 形状适配、PRIMARY pitch 错用 yaw 角度；
分别按实际接口和冻结 8° pitch 修复，未放宽资格阈值。最初快照缺配置的测试失败也保持原记录。
私有恢复包、逐帧证据、运行命令/退出码及 SHA manifest 位于本轮本机审查产物目录，不进入 Git。

## 前一阶段执行证据与后续入口

远端已有 Python 3.10.12 / PyTorch 2.11.0+cu128 / NumPy 1.26.4 / pytest 9.1.1 环境，
本轮使用 CPU，限制 OMP/OpenBLAS 为 2 线程，禁用额外 pytest plugin 与 HF 网络下载：

| 阶段 | 实际结果 | 范围 |
|---|---|---|
| 初次接线回归 | 37 passed / 1.98s | 新 front 适配、model/provider/checkpoint/校准 |
| 能力依赖回归 | 421 passed，1 deselected / 12.65s | 表内主体与五模块相关规则；排除 CUDA BF16 RTC 节点 |
| 最终接线回归 | 38 passed / 1.93s | 追加非恒定 mask/UV、非法概率测试；forward hook 实测调用次数 |
| Precision 补充回归 | 13 passed / 1.45s | detection adapter、Precision data/training/held_out 合成规则 |
| 独立 CLI 回放 | 退出码 0 | 1 Episode / 3 Tick，实际 forward 次数 `[0,1,1]`；训练步/Memory write/actuation 均为 0 |

四阶段按 JUnit testcase identity 去重为 **435 个通过的测试节点，57 个测试文件**，没有 skip 或失败。
重复回归没有累计成更多独立覆盖。阶段间生产代码未变；最终新增测试和 runner 计数由最后定向回归覆盖。
每阶段在独立远端目录执行，先核对源码 SHA-256 manifest；原始命令、JUnit、输出、快照和摘要在忽略目录
`artifacts/reviews/capability-integration-20260906/`。CUDA 节点未执行，不把它计入通过。

测试集合按上述依赖挑选，不运行下载型 Processor/Collator、完整仿真、
真实模型恢复、正式训练、confirmation/final test；远端已有环境中使用 CPU 小张量与临时合成文件。
测试名含 formal/held_out 的用例只校验合成规则，不消费受保护数据。

独立审查首先发现单帧 mask/多帧 keypoint 对应校验缺口，以及新 checkpoint 测试期待了错误异常类型；
两项已修复并纳入回归。验收矩阵审查补回 A13 的 data/training/held_out 覆盖，并明确 Object/Goal
score 互不继承。最终独立 R2 复核确认模型接线 PASS，核对最终 139 文件快照、原始测试输出和 CLI
source hash；另一只读审查确认 33 项/57 文件映射完整。审查 PASS 限定上述工程边界。
本轮不修改原始 33 项静态 inventory、用户未跟踪 Stage2B 或冻结结果。

下一处实际缺口是 **C08 的冻结 G2C 权重、normalizer、校准包和视角身份消费**。
需先核验 artifact 和评分语义，再做只读预测回放；不能将本轮合成 calibration 或分数阈值直接用于 D048。
C09 接在合格测量之后；Stage2B/3 继续按研究问题决定，不属于本轮补工程对应关系的必做项。
