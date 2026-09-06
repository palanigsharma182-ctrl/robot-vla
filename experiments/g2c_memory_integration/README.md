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
