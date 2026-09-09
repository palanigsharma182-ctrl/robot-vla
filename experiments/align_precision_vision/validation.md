# Align Precision Vision 实现验证

日期：2026-09-10。状态：**实现与合成测试通过；真实定位精度、策略收益及闭环尚未验证。**

## 执行环境与结果

全部 pytest 在新云机执行，WSL 仅用于编辑、打包、传输和 SHA256 校验。环境为 Ubuntu 22.04、Python 3.10.12、PyTorch 2.11.0+cpu、NumPy 1.26.4、pytest 8.4.2；本次未使用 GPU 计算。

| 代码快照 | 结果 | 耗时 | 退出码 |
| --- | --- | --- | --- |
| code-v1 | 20 passed | 2.69 s | 0 |
| code-v2 | 27 passed | 3.00 s | 0 |
| code-v3（最终） | 28 passed | 2.82 s | 0 |

每次在独立目录解包运行，保留前次日志。最终快照含 137 个代码/配置文件；本地相关源码逐文件匹配测试快照，源码包、manifest、最终日志和退出码文件的本地/远端 SHA256 一致。

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
PYTHONPATH=.:src python -m pytest \
  experiments/align_precision_vision -q -p no:cacheprovider
```

最终 source-v3.json SHA256：
`41116d8ff05c05d57de9f61279524eef14821be4acfbf11f63844c6549c94d91`

最终 tests-v3.log SHA256：
`2a9e5afb172d7e0fcee5f8d192930b354ad34500375bb47e31b0f4e8c10debaa`

完整源快照、日志和传输校验记录保存在私有运行目录，不随代码发布。

## 已验证内容

- 合成 RGB-D 关键点的反投影、刚体拟合、TCP 相对位姿和四重旋转对称。
- 缺失/退化/不一致测量、时间错配、身份错配及 Oracle/实测来源隔离。
- 图像定位 loss、梯度、整体对称监督和不可见坐标的 NaN mask。
- 新几何分支零初始化时与原 Expert 的输出一致；父模型不被修改；同噪声几何响应及诊断后训练模式恢复。
- 测量到真实 Expert 小尺寸配置的 chunk 接口；流水线中的定位网络使用固定关键点替身，因此不验证视觉准确率。
- 动作推理结束后重新检查观测年龄，过期 chunk 被拒绝。
- 数据 manifest 的来源与场景拆分检查；两步合成定位训练、checkpoint 保存和严格重载。

## 证据限制与后续

没有消费真实新图像或正式评估集，没有加载真实 Qwen/Align checkpoint，没有训练合格视觉权重，没有新 rollout，也没有 GPU 延迟或闭环成功率结果。两步合成训练只验证训练代码能运行和重载，不能说明收敛。

当前目标限已关联的单个 4 cm 直立方块、沿物体 Z 轴的中央抓取。2 mm 拟合残差、50 ms 年龄等只是候选工程阈值；须在真实图像分布上同时检查误差和有效率。角点位于深度边缘，采样误差及遮挡可能降低覆盖率，合成几何测试无法排除这一问题。

后续依次验证关键点/可见性与位姿测量、Expert 对几何输入的正确响应、相同预算下 baseline / Oracle / measured 的独立 Align 闭环。当前尚未开展这些效果实验。
