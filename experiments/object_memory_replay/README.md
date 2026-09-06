# Object Memory 最小合成回放

后续状态：五模块完整接入已修复本文末尾记录的时间问题，并完成远端完整 Goal Memory 回归及
delayed preview/commit 定向测试，见 [五模块接入说明](../five_common_replay/README.md)。
本文的原样复用与首轮测试数字描述 `79160c4` 时的历史状态。

目的：在 `main` 验证观测预测、候选窗口、Object Memory 与既有 Goal Memory 的接口连接。
这是工程接入验证，不是新的研究实验；全部输入均为显式标记的合成测量。

数据流：`ObjectWriteEvidence` → `ObjectMeasurement` → `ObjectCandidateWindowVerifier`
→ `ExplicitObjectStateMemory` → `resolve_object_state(NAVIGATION)`。
位置使用 robot-base 米制三维向量，协方差为 `[3,3]`、单位 m²；RGB/相机位姿/TCP 时间在此回放中相等。
离线 `derive_object_observability` 的 privileged label 不参与回放的写入判断。

首批 `src/robot_vla/precision/object_memory.py`、`object_observability.py` 及其两份
`test_e018_object_*` 测试逐字节复用 E018 提交 `37851ad648c42713511f869eb782eb897aafecc5`。
复用完整模块保留原 API；其中 delayed preview/commit 路径本批没有消费者，尚未在当前 main 验证。
Goal Memory 源码及默认运行链未修改。历史测试中的模型身份仅是合成 fixture，不能作为真实模型资格证据。

在仓库根目录运行，依赖 Python、NumPy、pytest：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 experiments/object_memory_replay/run.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider \
  tests/test_e018_object_memory.py tests/test_e018_object_observability.py \
  tests/test_object_memory_replay.py
```

回放输出 JSON，共 4 个 Episode、19 个 Tick：

| 场景 | 检查行为 |
|---|---|
| 初始化、保持、过期 | 低分/坏几何不能初始化；连续两帧合格测量初始化；异位低分预测不能覆盖位置；缺测时短暂保持，超时后拒绝导航 |
| 接触 | 已初始化状态遇到接触失效，之后高分帧不能在同一 Episode 恢复 |
| 来源变化 | 模型身份变化使状态失效，切回原身份也不能在同一 Episode 恢复 |
| Episode reset | 同一实例显式重置后清空旧位置、候选窗口和失效状态；新位置仍需要两帧验证 |

每个 Tick 独立更新合成 Goal 测量，再检查 Object 更新没有修改 Goal 状态。
回放只请求 NAVIGATION，所有 `contact_authorized` 输出均为 false。
原模块的 CONTACT_READY 是历史数据接口分支；本次接入不连接执行器，也不授予真实接触/闭爪权限。
阈值 0.7、最大 age 0.5s 仅为构造测试边界，不能替代真实测量的标定与效果验证。

2026-09-06 本机验证：Python 3.12.3、NumPy 1.26.4、pytest 7.4.4；23 个历史用例与
5 个新增集成用例全部通过（退出码 0），独立 CLI 回放退出码 0。
原 `test_precision_state_memory.py` 因实验 CLI 顶层依赖本机缺少的 torch，整体收集失败（退出码 2）。
在临时目录仅移除该 CLI import 与对应 claim 测试，保留其余 13 个测试体，经 AST 比对不变后运行，
13 passed（退出码 0）。这属于隔离的 Goal Memory 回归子集，不是整份历史测试通过；claim 测试未验证。
新增回放第一次测试发现 `accepted` 要求 keyword-only 阈值，改为 `threshold=` 后通过。

完整输出、CPU 子集副本与输入 SHA-256 留在忽略目录
`artifacts/reviews/object-memory-integration-20260906/`；不提交生成物或私有规则。
本批不加载真实 provider/Qwen、不下载依赖、不运行 GPU、仿真或训练，不消费真实评估数据。
通过仅证明合成接口与状态规则，不支持真实遮挡恢复、任务成功率或部署结论。

独立只读审查在隔离快照中另行运行上述 28 项测试通过，并复现两项历史时间问题：

- 候选窗口只检查控制 tick 递增，未保证 RGB 是新帧。tick 为 0、0.005s 而 RGB/位姿时间均为 0，
  在允许 10ms skew 时仍会被计为两帧并初始化。RGB 时间倒退但仍在 skew 内也存在同类问题。
- `last_observed_timestamp_s` 使用接受时的 tick，age 少计图像延迟。例如 tick 0.05s 接受 RGB 0.041s，
  到 tick 0.55s 报告 age 0.5s，但实际图像年龄 0.509s，刚超过本例上限仍允许导航。

这两项不影响本回放的同步、新帧假设，但必须在接入真实/异步 provider 前修复并增加边界回归。
本次保留来源实现，未将同步合成测试通过表述为完整时间合同通过。

下一批接入动态相机观测、几何与校准原语，再接冻结 G2C provider 的测量消费。
真实 provider 前先修复上述时间边界；延迟写入及 D049 编排需各自完成接线验证。
