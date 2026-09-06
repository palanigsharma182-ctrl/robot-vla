# Oracle Reach 历史权重推理复现 — 2026-09-07

本轮在当前环境恢复历史 GT geometry-only Oracle 权重并复跑推理，结果 **4/5 成功**，平均最终 TCP→物体距离 **3.729cm**。这是历史场景复现，不是重新训练、独立 held-out 验证或可部署视觉策略验收。

## 问题与实际数据流

检验几何已知时 Action Expert 能否完成 Reach，辅助区分几何输入与动作执行瓶颈。

`当前仿真 GT (object_position - TCP_position, distance) [B,4] → MLP → 单个 [B,1,720] context token → Action Expert（另输入当前归一化 proprio [B,15]）→ 10 步 Flow → action chunk [B,16,8] → 原动作适配/安全/temporal ensemble → 每次最多执行前4步。`

Oracle 替代 Qwen context，不是给 Qwen 追加 GT。不使用 Memory、重新观察、语义决策或专家代做 Reach。GT 仅保留在已有隔离诊断入口。权重来自历史 Reach-only 训练：30 epochs、每 epoch 4096 samples、batch 64、1920 optimizer steps，按 validation loss 选择 epoch 28；本轮没有训练或重选 checkpoint。

## 冻结条件与身份

- seed 10000–10004；sampling seed 42424；Reach 距离阈值 0.04m；每回合最多100个真实控制步；20Hz；Flow 10步；ensemble decay 0.5；max anomaly replans 3。
- 已消费的历史场景仅用于复现/事后诊断，不称 unseen。读取原始 normalization、manifest 与 audit metadata，没有读取训练图像或重新拟合。
- 恢复 checkpoint SHA256：`294b555fd16e2c6ceb2528b167289cdb94edb93d36844d16715cae58f89c9f05`。文件哈希与其配套 train/eval 原始记录一致；strict state_dict 加载通过。
- 仓库历史摘要记录另一个 SHA：`4ad363eef44aebe7ff9a34ac65cf9466f2f728eba0d68195365c6fd35883fa6a`。未取得该文件，**不声明两个 checkpoint 等价**。本结果仅对应已恢复的 `294b…` 版本。
- Dataset identity：`bc024b6b9c566ca9500945fb6ac262bf657bee713d8a5816229bdc8478139407`；manifest：`43f131cc1b79b93cf6e38f3f5e476d7a03fe29410daa720a672989c95afc477f`。本轮仅校验消费的 metadata 及其 identity，不宣称重新审计全数据。
- 源码 base：`f60a443acda32f67639b672aeb85cda056127b7b`，加本次局部修复；最终实际源码：`source-tree-sha256:89bd189c6ceddfb9146e6070a310fec90cad2615383bd0761fc93cb0786937b2`。远端核验131个快照文件；未包含用户无关未提交文件。
- 单卡 RTX4090；Python 3.10.12、PyTorch 2.11.0+cu128、Transformers 5.15.1、ManiSkill 3.0.1、SAPIEN 3.0.3、NumPy 1.26.4。历史完整软件环境未恢复，因此不是逐位数值复现。

## 结果

下表历史列取自恢复权重自身配套记录；本轮只运行 Oracle，未复跑 Qwen control 或 Layer12，不能混合不同来源的对照数字构造本轮 A/B 收益。

| seed | 本次结果 | 本次控制步 | 历史最终距离 cm | 本次最终距离 cm |
|---|---|---:|---:|---:|
| 10000 | 失败 | 100 | 4.886 | 4.671 |
| 10001 | 成功 | 64 | 3.361 | 3.301 |
| 10002 | 成功 | 60 | 3.970 | 3.696 |
| 10003 | 成功 | 64 | 3.895 | 3.807 |
| 10004 | 成功 | 64 | 3.721 | 3.171 |

历史与本轮均4/5成功，失败均为10000；5个初始 TCP→物体距离逐值相同。历史平均最终距离3.966cm，本轮3.729cm；代码/软件环境不同，该差值不能作为方法改进证据。

失败场景10000：0/40/60/80/100步距离分别18.821/6.895/5.191/4.869/4.671cm。大幅接近后收敛变慢，仍比4cm阈值多6.71mm；出现一次 tracking-correction saturation / anomaly replan，未触发错误退出。仅凭这条轨迹无法区分数据覆盖、局部控制/动作平滑与训练损失造成的精度瓶颈，也不能证明额外步数一定能解决。

## 发现并修复的工程问题

1. Oracle 旧入口未传当前 `_TrackingManiSkillController` 必需的 `observation_adapter`，动作前 TypeError。提前构造并传入同一适配器，保留V1语义。
2. 步数上限只在 chunk 之间检查。异常重规划破坏四步对齐，实际出现103步。Oracle局部controller新增逐步上限及已有 `chunk_stop_requested` 接口，使实际动作、executor计数、loop时钟、command reference同步截止；没有修改通用执行器或正式阈值。

保留全部尝试：首次启动路径错误（动作前）、适配器缺参（动作前）、103步 `protocol-deviation`、只有controller截止但executor计数未同步的中间版本、最终同步截止版本。前四项不纳入最终结果。本轮两个独立的实际接口缺陷及审查修正均在原资源上限内处理，未改变训练/候选范围。

## 验证与资源

- 原6个Oracle测试；新增真实评估入口测试，以及limit=1/3/100的真实 `QwenVLAReplanLoop + RecedingHorizonChunkExecutor` 跨边界测试；另核验3个现有观测边界测试：**13 passed，exit 0**。
- 首次测试快照遗漏原 `conftest.py` 导致4 passed/2 errors，恢复原fixture后通过；没有修改fixture绕过失败。
- 独立 Sol/high 源码审查指出仅controller no-op无法保证executor计数正确；补齐现有停止信号与对应回归后限定PASS。实际模型/强度已从会话 turn_context核验。审查者未自行运行测试；测试结果来自父进程原始输出。
- 最终GPU推理进程21.363秒、exit0；全部5次启动累计 **79.272秒**，低于本轮900秒上限。训练步数0。
- 远端新增约416.4MB，本机本轮快照与证据约2.5MB，合计约0.419GB，低于2GB上限；已有本机checkpoint没有重复复制到本机。
- 47个远端证据文件已回传并SHA256逐文件核验。原始逐步结果保存在私有run目录，不纳入GitHub；旧历史文件未覆盖，云端副本保留。

## 允许的结论与后续诊断

- 几何明确时，联合训练的Oracle专用 geometry encoder + Expert/Flow策略可以在这5个历史场景的大多数场景完成Reach；这不是对原Qwen策略保持同一Expert、仅替换context的严格消融。
- 与视觉/Memory路线的差距值得优先检查可用几何是否真正进入Expert并被使用，但本轮没有做同环境、同训练预算的视觉对照，不能量化Qwen或Memory的因果贡献。
- 对“原子任务拆分不够细/准”的观点，只得到局部支持：距离达标不保证抓取对齐和技能交接，且末端接近仍有精度问题。本轮只测Reach，没有验证抓取、组合或完整任务。
- 下一项最有辨识力的工作是记录/比较真实Reach结束状态与Grasp训练起点的TCP姿态、夹爪和物体相对位姿，再做单因素对照；这属于后续实验建议，本轮未执行。

## 复跑命令

在已具备上述依赖的环境，提供同SHA的原权重和配套metadata，使用新的非空检查通过的输出目录：

```bash
PYTHONPATH=src python -m robot_vla.cli.diagnose_oracle_reach evaluate \
  --mode oracle --checkpoint "$ORACLE_CHECKPOINT" --data "$DATA_METADATA" \
  --output "$NEW_OUTPUT" --seed-start 10000 --episodes 5 \
  --max-policy-steps 100 --sampling-seed 42424 --num-flow-steps 10 \
  --recency-decay 0.5 --max-anomaly-replans 3
```
