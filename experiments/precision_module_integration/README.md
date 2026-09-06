# Precision 模型到通用观测/校准工具的接口回放

目的：验证现有 checkpoint/predictor 可以消费 front 资格输入，并将同次模型 forward 的 mask 和
关键点证据交给校准工具。它补足原五模块回放使用人工预测、没有实际模型调用的覆盖缺口。
它是工程验收，不是新的训练实验或 G2C provider 资格验证。

`run.py` 建立 seed=0 的小型 U-Net，保存零训练步 synthetic-debug checkpoint，在冻结 predictor 中
严格重载。移动阶段不推理，两帧 COLLECT 各执行一次 forward；输入是合成 RGB `[32,32,3]`、
单帧状态 `[42]`、几何 motion `[4]`。输入适配使用同帧实际 camera-to-base 位姿和各自时间戳。
关键点 `[1,K,2]` 与 mask `[1,M,32,32]` 按通道名对应，按 object 预测 UV 双线性采样，
sigma `[2]` 乘 `sqrt(scale)` 后进入 object evidence。校准 scale=4 是 fixture，不是拟合结果。

源码、checkpoint/provenance、calibration 和输入摘要进入回放结果。临时 checkpoint 结束后清理。
没有合格 3D geometry，因此 evidence `geometry_valid=False`；输入 qualification-only、Memory write=0、
actuation=0。模型是随机初始化，输出分数只能验证数值流，不能评价精度。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python experiments/precision_module_integration/run.py

PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m pytest -q -p no:cacheprovider \
  tests/test_precision_module_integration.py tests/test_e018_active_front_provider.py \
  tests/test_precision_checkpoint.py tests/test_precision_unet.py \
  tests/test_precision_detection_provider.py tests/test_e018_calibrated_front_provider.py
```

工程回归、全部 33 项归属及证据限制统一记录在
[验收对应表](../../docs/reviews/capability-integration-matrix.md)，避免把通过的基础测试计成正式研究能力验收。
带真实 G2C artifact 的消费和 D049 返回 HOME 后提交仍是后续接线点。
