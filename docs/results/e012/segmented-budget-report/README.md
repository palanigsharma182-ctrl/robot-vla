# E012 segmented-budget report

本目录是 E012 `segmented-300-180-480` counterfactual 的便携技术报告与最小可复现伴侣。

权威输入：

- `../segmented-budget-smoke/`：固定三条 smoke 的 compact 结果；
- `../segmented-budget-counterfactual/`：16 条旧 TimeLimit seed 的 compact 结果与 canonical `analysis.json`；
- `../collection_summary.json`：旧正式 100-seed GL 分母和 10 条 eligible；
- `../../../../scripts/analyze_e012_budget_counterfactual.py`：严格校验与统计计算。

本报告没有归档或读取 NPZ、RGB、原始 trajectory、stdout/stderr、模型权重或 GPU candidate dataset。Counterfactual trajectory 明确禁止进入 D1。

从仓库根目录复现 compact analysis：

```bash
python3 scripts/analyze_e012_budget_counterfactual.py \
  --collection-summary docs/results/e012/collection_summary.json \
  --smoke-root docs/results/e012/segmented-budget-smoke \
  --counterfactual-root docs/results/e012/segmented-budget-counterfactual \
  --output /tmp/e012-segmented-analysis.json
```

不安装 Jupyter 也可以执行全部 notebook code cells，并核对 `artifact.json` 中的图表数据：

```bash
python3 docs/results/e012/segmented-budget-report/verify_notebook.py
```

如果本机安装了 Jupyter，也可直接打开 `reproduce.ipynb`。`report.html` 由 Data Analytics portable artifact builder 从 `artifact.json` 单向生成；不要手工维护第二套 HTML 数字。

当前打包结果为 `validation=passed`、`package=passed`、`verification=structural_only`。生成环境没有可用的
Chromium headless-shell，因此尚未做真实桌面/窄屏浏览器 smoke；内嵌 manifest、snapshot、source、四个
SQL projection 和全部 notebook code cells 已独立核对。该限制只影响视觉 smoke，不影响机器可读统计。
