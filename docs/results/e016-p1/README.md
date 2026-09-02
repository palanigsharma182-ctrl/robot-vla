# E016-P1 — Corrected-observability formal Precision + goal memory

E016-P1 从随机初始化完成 20-epoch corrected-observability Precision U-Net 训练，并在规则冻结后对
100 条 fresh test trajectory 执行一次 no-actuation perception 与显式 base-frame goal-memory replay。

## Formal training

- selected epoch：`12`
- validation observable-goal normalized-UV MAE：`0.014128`
- validation visibility precision / recall：`0.994662` /
  `0.917898`
- validation unobservable FPR：`0.005376`
- Motion Head unchanged：`True`

## Fresh test-once

- observable goal pixel p50 / p90 / max：`0.268` /
  `0.691` / `85.281` px
- visibility precision / recall：`0.991510` /
  `0.924077`
- write accepted / unsafe：`5225` / `4`
- current / memory coverage：`0.258779` /
  `0.951117`
- memory valid while GT unobservable：`8827`
- memory catastrophic / reset leakage：`0` /
  `0`

Engineering gate passed=`False`。本实验始终 no-actuation；即使门禁通过，
`safe_for_actuator_promotion` 仍为 `false`，后续还需要独立 controller/shadow safety 验证。
