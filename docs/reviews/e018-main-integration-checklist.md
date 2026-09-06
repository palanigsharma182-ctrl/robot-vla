# 截至 E018 的 main 能力接入清单

日期：2026-09-06。用途：确定主线保留、下一批接入、实验维护和历史保留的范围。
本清单是静态接入审查，不修改研究结论，不代表全部源码已逐行审查或当前环境已通过全部验收。

## 结论与范围

1. E014–E017 的主要代码和结果已随 E013 分支进入 `main`，无需再次合并。是否有效必须按能力与证据分别判断。
2. E018 中值得优先复用的是动态相机观测、受限相机运动、合格 front provider 和显式 Object Memory 的接口。
   它们是接入候选；现有证据不支持整套主动视觉已经有效，更不支持直接取得 manipulation 控制权。
3. RTC、Local DAgger、E017 两行微调、E018 Stage 2A 均有明确未晋级/负结果。保留复现入口和仍被消费的合同，
   不把这些方案当作已经提高任务成功率的默认能力。
4. main 的 V2 速度诊断修复必须保留；E018 分支中的旧 evaluator 不能覆盖它。
5. 后续主体与实验统一维护在 `main`；新实验入口放在 `experiments/<id>/`。历史实现不因目录约定立即搬迁。

审查输入固定为 main `e8eae165816e8d3a63846fd97e1b97379cf5a9a7`、E018
`37851ad648c42713511f869eb782eb897aafecc5`、WMC `8f525c76a77b6096bf7d5d919854c985c8eb02b0`，
以及当前可见的其余本地/origin refs，完整身份见 [源码覆盖清单](e018-main-source-inventory.json)。
本轮没有再次 fetch，不对审查之后的远端更新作结论。

源码覆盖：12 个本地/远端分支引用，共 167 个不同 Python 路径。main 与 E018 相同 114 个、不同 2 个；
E018 缺入 main 47 个；WMC 独立候选 4 个。47 个缺失文件由 9 个核心候选、1 个 Memory 评估模块和
37 个实验 runner/协议/CLI 构成。文件覆盖只保证已分类，不等于每个文件都完成了语义验证。
配置、测试和结果按能力定点读取；没有完整重审所有配置、原始数据、权重或历史提交。

## 如何读“有效”和“已复核”

| 标记 | 本清单中的准确含义 |
|---|---|
| 历史结果 | 已核对仓库结果记录/小型聚合及相关源码；未重新执行历史实验，未逐字节重验私有原始产物 |
| 静态实现 | 已核对接口或关键代码/测试，能支持接入判断；不能据此报告运行通过 |
| 当前定向验证 | 上一轮 e8eae16 的真实执行证据：观测回归 3 passed、相关回归 47 passed、ManiSkill 20 步 smoke；本轮未重跑 |
| 主体保留 | main 已有且当前消费者需要；不表示其所有研究效果都成立 |
| 按需接入 | 历史证据支持复用，但还需对应 main 回归/输入身份验证；不是直接合入全部依赖的批准清单 |
| 实验保留 | 候选、诊断或负结果复现；默认主链不依赖其效果承诺 |
| 历史保留 | 已结束/被替代的身份、配置、结果保持可追溯；不等于本轮删除文件或分支 |
| 待核实 | 证据不足或尚未完成当前环境验证；在待办中具体说明缺口 |

“存在代码”“工程能跑”“局部效果正向”“完整任务有效”是四种不同状态。
本轮没有训练、仿真、新测试、fresh test 消费或分支合并；文档与 Git/JSON 静态检查单独记录。

## A. 基础主线及 E001–E013

表内源码路径均相对仓库根目录；完整分支/blob 映射见配套 JSON。

| ID / 能力 | 证据与结论边界 | main 状态 / 处理 | 接入或继续使用前的具体核验 |
|---|---|---|---|
| A01 RobotSpec、单位、关节/夹爪映射 | `contracts.py`、`adapters.py`；当前定向验证及真实 Panda 关节顺序/范围/尺度 smoke 通过 | 已有，**主体保留** | 新设备必须单独验证单位、关节顺序、夹爪范围；目前不能称真实机器人驱动完成 |
| A02 trajectory/v2、writer/audit、split、统计、mask | `data/`；[E013][e013] 记录 7,987/7,987 action semantic parity、split audit 通过 | 已有，**主体保留** | 真实数据重新装载时验证 manifest/schema/stats；不能用旧 aggregate force 伪造 V2 左右力 |
| A03 Qwen + Adapter + Expert + Flow | `model/`、`training/flow_matching.py`；[E001–E007][experiments] 证明训练/推理链路运行；完整任务仍为 0 | 已有，**主体保留**为基线 | 当前新环境尚未恢复真实 Qwen/Processor/checkpoint，不称完整 VLA 已验收；V1/V2 权重身份分开 |
| A04 Stage 1、checkpoint、resume、selection | `training/stage1.py`、`training/checkpoint.py`、`evaluation/checkpoint_selection.py`；历史正式训练及恢复记录 | 已有，**主体保留** | strict reload、契约/RNG/optimizer identity；[E012][e012] 已说明恢复后 CUDA 数值轨迹并非 bitwise 相同 |
| A05 commanded target / correction / reset | `execution/chunk_executor.py`、`execution/maniskill_controller.py`；E003 饱和修正、D028 语义修正；当前 laggy replan/限幅后 reset 测试通过 | 已有，**主体保留** | 继续使用 D028；不能回退为 D017 每次 replan 从 actual q 重新解释标签；不能直接切换官方控制模式 |
| A06 Outcome Predicate、采集、atomic/full rollout | `tasks/pick_place.py`、`sim/`、`evaluation/atomic.py`、`evaluation/maniskill.py`；历史 ManiSkill rollout 有完整失败记录 | 已有，**主体保留** | 当前 20 步只验证官方 state 接口；RGB、真实模型、完整自定义任务需另验。privileged preparation/label 不进入 deployable 输入 |
| A07 事件标签、采样、低权重 event loss | `data/events.py`、`data/sampler.py`、`training/flow_matching.py`；E006 高权重有技能回归；E007 λ=0.25 原子 16/25、完整 0/20 | 已有，**保留可配置能力** | 历史局部改善不代表 V2 上同样有效；不默认恢复所有 A–F 训练条件或改变权重 |
| A08 Temporal ensemble / anomaly replan | `execution/temporal_ensemble.py`、`runtime/control_loop.py`；E007 同组 unseen Reach 5/20 对 newest 1/20；本轮 runtime 局部回归通过 | 已有，**主体保留现有默认** | 原组未触发 anomaly，不能把其收益归于重规划预算；完整任务仍 0/20；消费者区分 raw proposal 和组合后 chunk |
| A09 Oracle、Layer 12/24、checkpoint/gradient probe | `diagnostics/`、`evaluation/checkpoint_sweep.py`；E008–E010 有空间可解码性与技能权衡证据，无默认层/多头/PCGrad 晋级 | 已有，**实验/历史保留** | Oracle 输入仅作标记诊断；Layer 12 不替换默认 Qwen Context；扫 checkpoint 不等于获得正式 eligible 候选 |
| A10 RTC | `execution/rtc.py` 及 runtime RTC 路径；[E011][e011] Stage A 不晋级：Reach temporal 6/10、RTC 2/10，完整均 0 | 已有，**实验保留** | 保留 first-replan parity、overlap、reset 和默认关闭回归；runtime 仍导入它，不能直接删模块 |
| A11 Local DAgger provenance 与局部监督 | `local_dagger_protocol.py`、`data/trajectory.py`、`data/dataset.py`；[E012][e012] 六候选均不 eligible | 合同已有，**主体保留数据兼容**；采集/训练 runner **历史保留** | Dataset 仍消费来源、takeover mask/offset；负结果不授权删除校验；不默认恢复 repeat-2/Stage A/B |
| A12 Observation V2 / force / runtime identity | `observation.py`、V2 Dataset/Processor/Expert/runtime/checkpoint；E013 数据/接口记录和当前观测回归 | 已有，**主体保留** | 当前有效帧、前缀零 padding、图像/位姿时间对齐、force stats、schema 拒绝；真实 Qwen V2 forward 待新环境验证 |
| A13 RGB Precision U-Net / 几何 / 四帧 Provider | `precision/{model,data,geometry,provider,training,held_out}.py`；[E013][e013] 步骤 1–8 通过，四帧 p95 18.84ms | 已有，**保留感知与训练能力**，按相机身份复用 | 1.52mm p90 是离线 GT z 平面条件下的 XY 感知误差；最大误差 208.5mm；不等于 deployable 3D 或 placement 精度 |
| A14 Precision residual / observer 与 Executive | `precision/{control,shadow}.py`、`executive/`；E013 100-seed shadow 只有95干净pair，7次 deadline miss；D031–D033 为 shadow 约束 | 已有，**shadow/实验保留** | Motion Head 零初始化冻结、无 Precision actuator 晋级；Executive replan 采样不冒充逐20Hz控制。事件时序与关闭 parity 要保留 |

证据选择：[E009][e009] 没有单一 promotion checkpoint；[E010][e010] 没有确认预注册梯度冲突。
这些结果支持“停止对应晋级”，不支持删除有复现价值的 probe，也不支持以新复杂架构替代证据。
E001–E012的效果数字属于各自冻结的旧模型/执行语义；D028之后的main不能直接继承这些效果量。

实验编号覆盖索引：E001/E002→A02–A04；E003→A05；E004→A06；E005→A02/A07；
E006/E007→A07/A08；E008/E009/E010→A09；E011→A10；E012→A11；E013→A12–A14；
E014→B01；E015→B02/B03；E016→B04/B05；E017→B06/B07；E018→C01–C12。

## B. E014–E017 的能力复核

独立审查逐项比较了结果目录、相关配置、核心实现、入口与测试；这些主体在 main 与 E018 相同。
唯一共享源码差异是后文单列的 `precision/losses.py` 冻结输出评分扩展。

| ID / 能力 | 已核对的结果与限制 | main 状态 / 处理 | 对应源码、测试及剩余核验 |
|---|---|---|---|
| B01 E014 长尾分类与 confidence 口径 | [结果][e014] 复算 p50/p90/max=0.370/1.517/208.538mm；>20mm的17条异常中15条被接受。Top20中14条为 label/channel-contract taxonomy，不能改写为14处已证实实现bug | 已有，**实验保留** | `precision/outliers.py`、`tests/test_precision_outliers.py`；事后分类没有证明某项修复有效 |
| B02 E015 exists/projected/observable 合同 | [结果][e015] 20,197帧中observable=10,281；旧visible=14,970，其中4,689帧不满足中心射线观测合同 | 已有，**主体保留** | `precision/observability.py`、`test_precision_state_memory.py`；标签派生使用privileged segmentation，不等于运行时分类器合格 |
| B03 E015 显式 Goal Memory | 同一replay current→memory coverage 1.70%→13.00%；100个Episode仅35个初始化；unsafe write=2，工程gate失败 | 已有，**保留状态机制，冻结gate留实验** | `precision/state_memory.py`、`memory_evaluation.py`；测试覆盖reset/frame/time/age。GT z平面条件下的回放不等于纯deployable三维状态链路 |
| B04 E016-P0 corrected observability 监督 | [结果][e016p0] 5,943帧修正1,425个旧visible样本；全负batch定位loss/输出梯度为0；128帧overfit通过；无正式checkpoint | 已有，**主体保留监督修复** | `precision/e016_pretraining.py`、`losses.py`、`test_e016_precision_pretraining.py`；train-only可学习性不是泛化结果，P0 runner留实验 |
| B05 E016-P1 正式训练与Memory评估 | [结果][e016p1] 20epoch、选epoch12；100条fresh test trajectory的no-actuation perception＋Memory replay；current→memory coverage 25.88%→95.11%；接受5,225帧、unsafe=4，工程gate失败；Memory catastrophic/reset leakage=0，原始observable-goal XY最大误差仍为203.21mm | 已有，**保留训练/评估工程，checkpoint按需作冻结候选** | `e016_training.py`、`e016_evaluation.py`，formal-training/held-out测试；依然使用GT goal z几何。不能晋升为合格wrist state/write provider |
| B06 E017-P0 两行微调 | [结果][e017] 8epoch/1,200step均未过门槛；FPR由3/558恶化到14–16/558；没有保存checkpoint | 已有，**历史保留失败路线** | `precision/e017_training.py`、`test_e017_precision_training.py`；不自动进入默认训练策略，也不推论所有小范围微调都无效 |
| B07 E017事后threshold=0.525 | validation FP由3降至2、recall=0.9146；仅对未改父模型的事后诊断 | 已有记录，**待核实/实验候选** | 没有独立新held-out确认；不回写E016默认阈值、不重用已消费test追认 |

补充辨别：E015的2次unsafe是中心不可观察的接受错误；E016的4次unsafe中GT-unobservable=0，
违反的是5mm定位门槛，最大accepted error约6.362mm，不能统一归因为observability误报。
E015→E016同时改变checkpoint、seed、Memory age，coverage差异不是单因素因果估计。
E015原“未初始化/过期遮挡帧=0”已更正为9,253；清单不采用旧值。
E014/E015/E016已消费的数据用途保持原记录。D036的旧validation lifecycle mismatch只限制新用途，
不能把整个E016历史实验追溯改判无效。

## C. E018 的能力复核

接入依赖分为三个层次：测量/几何 → Object Memory/provider → 受限相机与恢复编排。
历史 provider qualification 的输入身份必须与消费者匹配，不能只复制 Python 文件就认为已恢复。

| ID / 能力 | 已复核证据与结论 | main 状态 / 建议 | 合入前必须核验 |
|---|---|---|---|
| C01 P0 Object Memory / object observability | [P0结果][e018p0] navigation availability 130/1135→504/1135；但41个真正不可观察帧都在冷启动，Memory覆盖0/41。支持跨低分/拒绝保持，未证明先见后遮挡恢复 | 缺失，**按需接入状态机制**；探索阈值留实验 | `object_memory.py`、`object_observability.py`，对应两份test及既有Goal Memory回归；reset/time/source/contact、navigation与contact分离；2秒age未获可靠上限证明 |
| C02 G0 相机位姿/时延/回HOME | [G0结果][g0] 16/16路线通过；没有provider forward或Memory write | 几何工具缺失，**按需接入**；路线runner **实验保留** | `active_front_camera.py`、`test_e018_active_front_camera.py`；actual/commanded pose、SO(3)、settle与运动帧禁写。真实route engine仍在`e018_p1_g0.py::_run_route`，按消费者需要局部提取 |
| C03 G0B视角筛选 / G0C动态路线 | [G0B][g0b] 50场景×25pose，10个低位alternate静态合格，env.step=0；[G0C v2][g0c] 40/40路线、3,680帧通过 | 缺失，**实验/历史保留** | `test_e018_p1_viewpoint_screen.py`、`test_e018_p1_g0c.py`；静态GT筛选不能用于在线选视角，RenderCamera无真实质量/碰撞/回差 |
| C04 G1A动态观测 / G1 supervisor replay | [三阶段计划][stages]记录Stage1 development通过；G1测试验证两条成功路径和六类失败，实际是合成证据replay，无provider/Memory/actuation | `active_external_observation.py` **按需接入**；`active_front_reobserve.py` **实验保留** | 对应active_external_observation、active_front_reobserve、G1/G1A测试；同步actual pose、HOME-only四帧、旧Chunk清空、reset/latch。尚无canonical VLA真实接通证据 |
| C05 G2A wrist→front直接迁移 | [provider计划][provider-plan] native wrist covariance-95=0.64<0.90；front正式资格因parent健康失败为inconclusive；附加诊断front XYZ p90约141.474mm | 资格runner **历史保留**；`active_front_provider.py` **按需实验适配** | camera角色、来源和parent failure→inconclusive；不能只改camera名称获得front资格，也不能把诊断均值误差说成正常资格协议结果 |
| C06 G2B covariance校准 | D036记录CAL-v2的1,135帧cohort有reset-first-frame contact-cache异常，整体protocol-invalid；fit/selection未执行，校准gate未评估。CAL-v1完整归因本轮不足 | `calibrated_front_provider.py` **按需复用数值原语**，旧runner/cohort **历史保留** | 对应calibrated-provider/G2B测试；finite/sample support/奇异covariance、一致scale、全cohort判定；“未评估”不能写成“校准机制无效” |
| C07 G2C数据/训练/模型选择 | [provider计划][provider-plan] W-KV0/S对照选W-KV0 epoch15，model-validation中10/10 alternate eligible | 新数据与训练runner缺失，**实验保留**；基础U-Net已有 | G2C data/training/model_val及test；初始化/公平采样、split/label隔离、prediction freeze；model-val通过不等于动态资格 |
| C08 G2C静态校准 / D048动态provider | 静态10/10 non-HOME，accepted/oracle-safe=497/499；[D048][d048]动态500路线、7/10 alternate合格；PRIMARY `LEFT_LOW__PITCH_UP`为47/47，XYZ p90约1.753mm；失败尾部见下文 | **按冻结身份消费合格object provider**；校准/资格runner保留实验 | qualification/test中raw/canonical pose边界与逐视角分母；HOME及3个淘汰视角禁止active Memory write；其余6个合格视角仅shadow；Goal辅助输出不获得Goal Memory资格 |
| C09 Stage2A候选/Memory提交/恢复编排 | [最新结果][s2a] selection三gain都是5/24，0.10按平局规则选择；evaluation为7/25=28%<70%，support通过、effect失败，offline/no-actuation | `active_front_memory*.py`及stage2a runner缺失，**实验保留/按消费者适配**；负结果历史保留 | 三帧稳定性、第三帧原值、HOME后延迟提交、2.5秒pending、重复/部分提交、source recheck；D049身份硬绑定不能直接称通用层。10-seed真实integration完成证据本轮待核实 |
| C10 D050 wrist触发语义 | [D050][d050]区分无合格wrist capability和observability-only predicate；没有把E016 epoch12晋升为qualified wrist | **保留语义区分**，qualifier完成状态 **待核实** | absent-capability baseline不能声称“wrist模型判断看不清”；无效输入不能直接转成camera motion请求 |
| C11 Stage2B七视角shadow | tracked E018有配置，未找到tracked专属实现/完成结果；用户本地有未跟踪纯评分草稿 | **待核实/实验候选**；不是已完成能力 | 三帧、完整视角分母、oracle隔离、禁止在线fallback选择；最新2A明确2B非必做，不能为补齐编号自动执行 |
| C12 Stage3A fault/replay、3B matched comparison | [D051][d051]仅有冻结方案及前置条件；本轮未找到专属runner或完成结果 | **计划保留，待实施** | G1或2A局部故障测试不能代替S3A整矩阵；S3A不是S3B效果对照；Stage2A负结果不支持正向恢复收益晋级 |

### 九个核心候选的准确接入位置

| 模块 | 建议位置/身份 | 当前判定 |
|---|---|---|
| `object_memory.py`、`object_observability.py` | 主体中可检查的状态/观测工具 | 按需接入，保留明确的phase/contact和来源合同；不携带探索参数的有效性承诺 |
| `active_external_observation.py`、`active_front_camera.py` | 主体几何/观测工具，消费者显式调用 | 按需接入；相机实际移动runner保持实验身份 |
| `calibrated_front_provider.py` | 可复用校准/证据数值工具 | 按需接入；历史已校准权重/视角/输入身份另行核验 |
| `active_front_provider.py` | 显式角色替换/资格适配 | 只在已明确provider身份的实验消费者中接入，不能直接变成默认wrist/front互换 |
| `active_front_memory.py`、`active_front_memory_provider.py`、`active_front_reobserve.py` | D049实验编排与supervisor | 实验保留，按真实消费者做局部适配；不整套抽成通用框架、不挂默认Executive |

源码入口（均固定E018提交，当前main尚无这些文件）：
[Object Memory][code-object-memory]、[object observability][code-object-observability]、
[dynamic observation][code-external]、[camera geometry][code-camera]、[calibration][code-calibration]、
[front adapter][code-front]、[Memory transaction][code-memory]、[Memory provider][code-memory-provider]、
[supervisor][code-supervisor]。

**已证实的接入陷阱：** `active_front_reobserve.py:142` 默认
`selected_primitive_id="LEFT_LOW__YAW_LEFT"`；D048已淘汰该视角。
旧G1无provider replay不因此自动失效，但新的Memory-write消费者必须显式选择合格PRIMARY
`LEFT_LOW__PITCH_UP`并验证allowlist，不能沿用旧默认，也不能改写冻结G1配置来伪装历史一致。

注意：D048是final-frame provider资格，Stage2A增加三帧稳定性、HOME信息增益和返回后提交；
47/47和7/25不是同一指标、同一cohort或同一链路，不能直接相减或断言已有provider退化。
失败尾部同时保留：静态校准有10条raw catastrophic，均被拒绝；D048全局4次unsafe accepted分别来自
HOME和三个淘汰alternate，另5条raw catastrophic全部来自HOME且被拒绝；七个qualified视角的unsafe accepted为零。

## D. 两处已有文件差异与独立候选

| 项目 | 本轮核对 | 处置 |
|---|---|---|
| `evaluation/maniskill.py` | E018 仍含 V2 `[7:14]` 空切片；main e8eae16 已用当前帧修复并完成先失败后通过回归 | **保留 main**；禁止用旧整树 checkout 覆盖 |
| `precision/losses.py` | E018 增加 `frozen_decoded_normalized_uv`，限定 finite float32 `[B,K,2]`、[0,1]、`no_grad`；默认路径仍从模型解码 | **按冻结输出评分消费者需要接入**；先验默认 loss/gradient parity、override 在 grad-enabled 时拒绝；本轮未运行该回归 |
| WMC 四模块 | `structured_world_model`、`candidate_evaluation`、`action_consistency`、训练 loss；在独立 8f525c7 分支，历史本会话已有84项候选测试与合成GPU前后向 | **单独实验候选**；不是 E018 有效能力或训练收益证据；E013/main command reference 与原分支不同，必须重接数据边界 |
| WMC Critic 输入 | 当前 consistency 消费连续 raw chunks，main replan 返回组合后 chunk；它不消费 World Model 预测 | 后续接入必须在组合前捕获 raw；不能把它表述为 WM→value/success Critic 或动作选择器 |
| 用户未跟踪 Stage 2B | `precision/e018_p1_stage2b.py` 及同名测试，独立于 E018 tracked tip；源码与测试约46KB | **只读保留、未验收**；缺失 provider 依赖；本轮不覆盖、不提交、不运行、不视作已经完成的2B |

## E. 文档冲突与未验证项

| 问题 | 处理依据 / 下一步 |
|---|---|
| `docs/experiments.md` 索引仍写 E013 engineering、正式结果未开始 | `docs/results/e013/README.md` 与 summary 已记录步骤1–8及step9失败；本清单按具体结果收窄结论。后续维护总索引，不改冻结结果 |
| README 仍提 D017，而 D028 改为跨 replan command reference | 当前源码和本轮测试符合 D028；旧D017是历史语义，接入前维护当前入口说明的链接/状态 |
| E018 三阶段计划仍写 stage2 runner进行中 | 最新 Stage2A v2 结果明确28%负结果、pause-for-reusability-refactor、2B非必做；不按旧待办自动续跑 |
| 已有代码不等于新 GPU 环境通过 | 新环境已有 PyTorch、ManiSkill state/no-render smoke；真实 Qwen、RGB/Vulkan、已选 Precision checkpoint 的完整消费尚未验收 |
| 历史配对样本/指标可用性 | 当前V2 evaluator bug不能自动污染所有历史实验：E013/E018不少路径独立；需按实际调用栈核对，不能声称已证明所有历史结果无影响 |
| 公共摘要与私有证据 | 本轮没有访问原始RGB/label/checkpoint/逐样本记录，也未重验全部 artifact。Stage2A失败归因仍需已有私有记录；不能凭聚合7/25猜原因 |
| 测试覆盖范围 | 本轮只定位测试、审查关键逻辑；历史测试输出不冒充本次执行。main上的50项接口测试不覆盖全部116个已有模块 |

## F. 下一批接入次序与验收

1. **首批复用工具与状态**：以最小回放消费者为依据，接入 Object Memory/object observability、
   dynamic external observation/相机几何及必要校准原语，即上表前五个通用候选。
   验reset/time/source/contact、actual pose、坐标、finite/covariance，不启用默认相机或manipulation控制。
2. **再接冻结G2C provider的实验消费**：`active_front_provider.py`只承担资格输入/角色适配，
   不包含完整G2C推理消费链。推理消费者须绑定已确认的checkpoint、calibration、视角和输入身份，
   显式配置D048 PRIMARY；验camera role、frame/time、prediction先于label。该步不写Memory，
   D049写入适配留在第3步；拒绝时不以GT补值。
3. **最后接D049实验编排**：按需适配 `active_front_memory_provider`、`active_front_memory`、
   `active_front_reobserve`。先回放 HOME→alternate→HOME、动作引用清空、HOME-only四帧、延迟提交与来源阶段恢复；
   历史相机路线通过不替代本次接线验收，Memory navigation-valid不放行接触/闭爪。
4. 新的最小串联入口放 `experiments/`，实际依赖到旧 runner helper 时才局部提取。
   不把上述九个文件全部合入作为起跑前提，不批量带入37个历史入口/协议模块。
5. 接入后再基于原始既有记录分析 Stage2A未恢复样本；区分实现错误、无测量、误差、跨帧稳定性和提交条件。
   Stage2B/Stage3的执行要由明确问题与数据用途决定，不能因代码已经存在就启动。

以上是建议执行次序，**本清单不声称这些接入已经完成**。每批完成后只更新对应行的来源、回归结果与剩余缺口。
归档指维护分类；本轮不删除历史文件、分支、产物或 checkpoint。

## 审查记录与复核限制

主审查负责 E001–E013、所有当前 refs/source 路径覆盖、两处已有源码差异与最终清单一致性；
两个独立只读审查分别负责 E014–E017、E018，主审查复查关键结论与原始聚合。
源码覆盖 JSON 用 Git blob 比对和路径分类生成，未执行项目代码。

独立复核后的三项修正已纳入：B05区分Memory与原始感知的灾难性误差统计；C08保留全部视角的失败尾部；
F区分qualification输入适配器与完整provider消费链。没有把独立审查表述为再次执行实验。

文档/Git静态校验：33个能力ID唯一，167个源码路径无遗漏/重复，12个refs及对应blob一致；31个证据链接的
本地路径或固定Git对象存在；两份用户Stage2B文件SHA-256未变；`git diff --check`通过。
这是文件/引用/快照校验，不是运行项目测试或历史verifier。
本轮仅新增本清单、源码inventory和README入口，审查输入业务源码与历史结果保持不变。
接入清单完成与源码合入、研究晋级、当前全套测试通过是不同事件。

[experiments]: ../experiments.md
[e009]: ../results/e009/README.md
[e010]: ../results/e010/README.md
[e011]: ../results/e011/README.md
[e012]: ../results/e012/README.md
[e013]: ../results/e013/README.md
[e014]: ../results/e014/README.md
[e015]: ../results/e015/README.md
[e016p0]: ../results/e016-p0/README.md
[e016p1]: ../results/e016-p1/README.md
[e017]: ../results/e017-p0/README.md
[e018p0]: https://github.com/palanigsharma182-ctrl/robot-vla/blob/37851ad648c42713511f869eb782eb897aafecc5/docs/e018_p0_development_findings_20260905.md
[g0]: https://github.com/palanigsharma182-ctrl/robot-vla/blob/37851ad648c42713511f869eb782eb897aafecc5/docs/e018_p1_g0_development_findings_20260905.md
[g0b]: https://github.com/palanigsharma182-ctrl/robot-vla/blob/37851ad648c42713511f869eb782eb897aafecc5/docs/e018_p1_g0b_viewpoint_screen_findings_20260905.md
[g0c]: https://github.com/palanigsharma182-ctrl/robot-vla/blob/37851ad648c42713511f869eb782eb897aafecc5/docs/e018_p1_g0c_rotated_motion_findings_20260905.md
[stages]: https://github.com/palanigsharma182-ctrl/robot-vla/blob/37851ad648c42713511f869eb782eb897aafecc5/docs/e018_p1_three_stage_active_vision_closed_loop_plan.md
[provider-plan]: https://github.com/palanigsharma182-ctrl/robot-vla/blob/37851ad648c42713511f869eb782eb897aafecc5/docs/e018_p1_g2c_front_provider_adaptation_plan.md
[d048]: https://github.com/palanigsharma182-ctrl/robot-vla/blob/37851ad648c42713511f869eb782eb897aafecc5/docs/decisions.md#L3487
[d050]: https://github.com/palanigsharma182-ctrl/robot-vla/blob/37851ad648c42713511f869eb782eb897aafecc5/docs/decisions.md#L3770
[d051]: https://github.com/palanigsharma182-ctrl/robot-vla/blob/37851ad648c42713511f869eb782eb897aafecc5/docs/decisions.md#L3960
[s2a]: https://github.com/palanigsharma182-ctrl/robot-vla/blob/37851ad648c42713511f869eb782eb897aafecc5/docs/results/e018-p1-stage2a-d049-v2/README.md
[code-object-memory]: https://github.com/palanigsharma182-ctrl/robot-vla/blob/37851ad648c42713511f869eb782eb897aafecc5/src/robot_vla/precision/object_memory.py
[code-object-observability]: https://github.com/palanigsharma182-ctrl/robot-vla/blob/37851ad648c42713511f869eb782eb897aafecc5/src/robot_vla/precision/object_observability.py
[code-external]: https://github.com/palanigsharma182-ctrl/robot-vla/blob/37851ad648c42713511f869eb782eb897aafecc5/src/robot_vla/precision/active_external_observation.py
[code-camera]: https://github.com/palanigsharma182-ctrl/robot-vla/blob/37851ad648c42713511f869eb782eb897aafecc5/src/robot_vla/precision/active_front_camera.py
[code-calibration]: https://github.com/palanigsharma182-ctrl/robot-vla/blob/37851ad648c42713511f869eb782eb897aafecc5/src/robot_vla/precision/calibrated_front_provider.py
[code-front]: https://github.com/palanigsharma182-ctrl/robot-vla/blob/37851ad648c42713511f869eb782eb897aafecc5/src/robot_vla/precision/active_front_provider.py
[code-memory]: https://github.com/palanigsharma182-ctrl/robot-vla/blob/37851ad648c42713511f869eb782eb897aafecc5/src/robot_vla/precision/active_front_memory.py
[code-memory-provider]: https://github.com/palanigsharma182-ctrl/robot-vla/blob/37851ad648c42713511f869eb782eb897aafecc5/src/robot_vla/precision/active_front_memory_provider.py
[code-supervisor]: https://github.com/palanigsharma182-ctrl/robot-vla/blob/37851ad648c42713511f869eb782eb897aafecc5/src/robot_vla/precision/active_front_reobserve.py#L142
