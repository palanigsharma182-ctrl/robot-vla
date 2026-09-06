# 正式 G2C 到 D049 Memory 的工程消费入口

这个入口显式消费冻结的 W-KV0 epoch-15、D046 校准和 D048 PRIMARY 资格包，复用 EO/FC、OO/CP 和 OM。
它运行单环境开发 smoke，默认新场景 seed `1000001`，终点是返回 HOME 后提交或明确拒绝。
`--case historical-positive-76903` 只回放已知成功的历史开发样例，用于验成功分支，不构成独立效果证据。
不加载 Qwen/VLA，不验证操纵任务收益，也不重开历史 selection/final test。

```bash
PYTHONPATH=src python experiments/g2c_memory_integration/run.py --bundle /private/bundle --output /new/run
```

运行环境需要项目既有的 CUDA PyTorch、ManiSkill 3.0.1、SAPIEN 3.0.3。输出目录必须不存在。
正式包由本机私有存储恢复，内容不进入 Git；必需文件为 `weights.pt`、`proprio_stats.json`、
`finger_force_stats.json`、`qualification_config.json`、`qualification_receipt.json`、
`qualification_verification.json`、`qualification_execution.json`、`qualification_predictions.jsonl`、
`calibration_config.json`、`calibration_receipt.json`、`viewpoint_calibrations.json`。
所有来源与资格组件先按冻结 hash 核验；prediction ledger 仅用于绑定已有 HOME/PRIMARY 相机资格范围。

流程为 HOME baseline → 显式 PRIMARY 请求 → 清空实际为空的 Action 历史组件 → 三个 COLLECT frame →
40 tick 返回 HOME → 四个全新、连续、完整的 HOME V2 frame → source recheck → delayed commit/no-commit。
请求是本 smoke 的显式固定路线，不冒充自动触发或运行中的 VLA 暂停恢复；Action 缓存最初为空的事实写入结果。
使用原始 sigma 计算 D049 score，校准 scale 只乘几何传播 covariance；信息增益门槛固定为 0.10。
三维位置条件于本任务静止方块中心高度 0.02 m，零 Z 方差不表示测得真实高度。

`result.json` 保存终态、拒绝原因及 commit/no-commit 凭据；`camera.jsonl` 保存时钟、实际外参和 hold 证据；
`provider.jsonl` 保存实际四次 G2C forward 的身份、分数及 adapter 拒绝原因。
工程异常写入记录后仍以失败退出。相机返回不成功、接触、机械臂/夹爪漂移等不会转成成功提交。

定向单测见 `test_qualified_front_provider.py`、`test_e018_active_front_memory.py` 和
`test_e018_active_front_reobserve.py`。真实运行结果与未验证边界见项目能力接入矩阵；
单个开发 smoke 不能改变 E018 Stage2A 的既有负结果或授予 manipulation 控制权。

## 真实 VLA 基线与运行中暂停恢复

`vla.py` 显式恢复 E012 冻结初始化 V1 基线（E011 Layer-12），固定 checkpoint SHA-256，
从该 checkpoint 恢复统计、Adapter/Expert，并严格检查原模型、Processor 与 Qwen revision。
不把 V1 权重加载为 V2，不训练、不消费历史 selection/final test。

```bash
PYTHONPATH=src python experiments/g2c_memory_integration/vla.py \
  --checkpoint /private/e011-layer12-best.pt --model-cache /private/model-cache \
  --mode baseline --output /new/baseline
PYTHONPATH=src python experiments/g2c_memory_integration/vla.py \
  --checkpoint /private/e011-layer12-best.pt --model-cache /private/model-cache \
  --mode reobserve --bundle /private/bundle --case historical-positive-76903 \
  --output /new/reobserve
```

基线入口在一个固定新开发场景执行两次真实 VLA replan；重观察入口在真实策略执行两步后，
利用 executor 已有的控制步中断边界停止旧 chunk。Runtime 暂停后清除 temporal/RTC/reference，
保留 Episode 控制时间和随机采样序列；调用方负责 hold 和相机路线。只有 HOME 实际几何、
四个全新连续同步帧、source recheck 与 D049 commit 均通过，才调用 fresh VLA replan 并继续执行。
失败或拒写保持暂停，不自动改用不合格测量。Memory 内容不注入 VLA，也不输出操作命令。

四帧 HOME 是恢复屏障，冻结 V1 策略仍只使用最新一帧双图与 15 维 proprio。
此入口暂只恢复 V1；通用 Runtime 的 V2 分支另有合成接口测试，不构成已训练 V2 验收。
重观察的固定请求不代表自主触发；历史正样例不代表独立效果证据。
当前为单线程同步仿真，不能跨线程抢占正在执行的 chunk，也不证明真实硬件时延。
实际执行结果及限制以本次能力矩阵追加记录为准。

来源复核的当前范围是单个隔离开发 runner：phase 固定为 ACQUIRE_TRACK，未构造外部
Executive 或 qualified wrist owner，也没有其他控制写入者。它不证明运行中真实 Executive
的动态 phase/owner 重读取。暂停后的 20 个稳定步尚未建立 hold 参考，日志以 null 漂移及
`hold_reference_available=false` 标记；只有之后的相机路线才按固定参考验证 hold。
