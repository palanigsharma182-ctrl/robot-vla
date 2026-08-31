-- 从 docs/results/e012/collection_summary.json 复现报告中的 gate、system 与 rejection 行。
WITH
gate_comparison(comparison, boundary, measure, count, required, scanned, accepted, rejected, acceptance_rate, gate) AS (
  VALUES
    ('RG eligible', 'Reach→Grasp', 'Eligible', 31, 20, 100, 31, 69, 0.31, 'PASS'),
    ('RG required', 'Reach→Grasp', 'Fixed requirement', 20, 20, 100, 31, 69, 0.31, 'PASS'),
    ('GL eligible', 'Grasp→Lift', 'Eligible', 10, 20, 100, 10, 90, 0.10, 'FAIL'),
    ('GL required', 'Grasp→Lift', 'Fixed requirement', 20, 20, 100, 10, 90, 0.10, 'FAIL')
),
system_summary(scope, errors, scanned, accepted, rejected) AS (
  VALUES ('RG+GL formal', 0, 200, 41, 159)
),
failure_reasons(boundary, reason, count) AS (
  VALUES
    ('Reach→Grasp', 'Policy 在目标 boundary 前终止或截断', 51),
    ('Reach→Grasp', '可信成功前达到时间上限', 14),
    ('Reach→Grasp', 'Expert takeover 后未形成稳定 Grasp', 3),
    ('Reach→Grasp', 'Snapshot round-trip 未通过', 1),
    ('Grasp→Lift', 'Policy 在目标 boundary 前终止或截断', 71),
    ('Grasp→Lift', '可信成功前达到时间上限', 16),
    ('Grasp→Lift', 'Expert 未完成完整 Pick-and-Place', 1),
    ('Grasp→Lift', 'MPlib 无可信 screw path', 1),
    ('Grasp→Lift', 'Snapshot round-trip 未通过', 1)
)
SELECT
  'gate_comparison' AS dataset,
  comparison AS row_key,
  boundary,
  measure,
  count AS value,
  required,
  scanned,
  accepted,
  rejected,
  acceptance_rate,
  gate
FROM gate_comparison
UNION ALL
SELECT
  'system_summary',
  scope,
  NULL,
  'Runner errors',
  errors,
  NULL,
  scanned,
  accepted,
  rejected,
  NULL,
  CASE WHEN errors = 0 THEN 'PASS' ELSE 'FAIL' END
FROM system_summary
UNION ALL
SELECT
  'failure_reasons',
  boundary || ':' || reason,
  boundary,
  reason,
  count,
  NULL,
  100,
  NULL,
  NULL,
  NULL,
  'REJECTED'
FROM failure_reasons;
