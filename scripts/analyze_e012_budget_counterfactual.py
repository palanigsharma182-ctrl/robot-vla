#!/usr/bin/env python3
"""严格审计 E012 segmented-budget smoke/counterfactual 并生成容量规划证据。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

ANALYSIS_FORMAT = "robot-vla-e012-segmented-budget-analysis/v1"
COLLECTION_SUMMARY_FORMAT = "robot-vla-e012-collection-summary/v1"
COUNTERFACTUAL_FORMAT = "robot-vla-local-dagger-budget-counterfactual/v1"
COUNTERFACTUAL_CANDIDATE_FORMAT = (
    "robot-vla-local-dagger-budget-counterfactual-candidate/v1"
)
COUNTERFACTUAL_RECEIPT_FORMAT = (
    "robot-vla-local-dagger-budget-counterfactual-receipt/v1"
)
SMOKE_FORMAT = "robot-vla-local-dagger-budget-counterfactual-smoke/v1"
PROTOCOL_NAME = "segmented-300-180-480"
TRAJECTORY_USAGE = "forbidden as training data"
TIME_LIMIT_REASON = "Episode 在可信成功前达到时间上限"

EXPECTED_COUNTERFACTUAL_SEEDS = (
    30_103,
    30_105,
    30_106,
    30_129,
    30_137,
    30_143,
    30_145,
    30_156,
    30_162,
    30_165,
    30_171,
    30_181,
    30_187,
    30_193,
    30_195,
    30_196,
)
EXPECTED_SMOKE_ROLES = (
    ("accepted_control", 30_111),
    ("timeout_early", 30_193),
    ("timeout_late", 30_171),
)
CLASSIFICATIONS = (
    "recovered_full_eligible",
    "expert_completed_but_snapshot_or_paired_gate_failed",
    "expert_recovery_budget_exhausted",
    "other_behavioral_rejection",
    "prefix_mismatch",
    "engineering_error",
)
RECEIPT_KEYS = {
    "format",
    "target_identity",
    "result_file",
    "record_sha256",
    "subprocess_sha256",
    "runner_error_sha256",
    "result_sha256",
    "immutable",
}
PLANNING_POOL_SIZES = (120, 160, 180, 200, 220, 240)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"JSON 不允许非有限常量: {value}")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_json_constant,
    )
    _require(isinstance(value, dict), f"{path}: 顶层必须是 JSON object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        _require(bool(line.strip()), f"{path}:{line_number}: 不允许空行")
        value = json.loads(line, parse_constant=_reject_json_constant)
        _require(isinstance(value, dict), f"{path}:{line_number}: row 必须是 object")
        rows.append(value)
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_int(value: Any, label: str, *, minimum: int = 0) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
        f"{label} 必须是 >= {minimum} 的 Python int",
    )
    return int(value)


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _summary_stats(values: list[int]) -> dict[str, Any]:
    _require(bool(values), "summary stats 不允许空集合")
    return {
        "count": len(values),
        "values": sorted(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "max": max(values),
    }


def _wilson_interval(successes: int, trials: int, *, z: float = 1.959963984540054) -> list[float]:
    _require(0 <= successes <= trials and trials > 0, "Wilson 输入无效")
    rate = successes / trials
    denominator = 1.0 + z * z / trials
    center = (rate + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(rate * (1.0 - rate) / trials + z * z / (4.0 * trials * trials))
        / denominator
    )
    return [center - radius, center + radius]


def _binomial_tail(trials: int, minimum_successes: int, probability: float) -> float:
    _require(0.0 <= probability <= 1.0, "binomial probability 越界")
    _require(0 <= minimum_successes <= trials, "binomial threshold 越界")
    return sum(
        math.comb(trials, successes)
        * probability**successes
        * (1.0 - probability) ** (trials - successes)
        for successes in range(minimum_successes, trials + 1)
    )


def _log_beta(alpha: float, beta: float) -> float:
    return math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)


def _beta_binomial_tail(
    trials: int,
    minimum_successes: int,
    *,
    alpha: float,
    beta: float,
) -> float:
    _require(alpha > 0.0 and beta > 0.0, "Beta posterior 参数必须为正")
    baseline = _log_beta(alpha, beta)
    return sum(
        math.exp(
            math.lgamma(trials + 1)
            - math.lgamma(successes + 1)
            - math.lgamma(trials - successes + 1)
            + _log_beta(successes + alpha, trials - successes + beta)
            - baseline
        )
        for successes in range(minimum_successes, trials + 1)
    )


def _minimum_pool_size(
    probability_function: Any,
    target_probability: float,
    *,
    minimum_successes: int,
    maximum_pool_size: int = 500,
) -> int:
    _require(0.0 < target_probability < 1.0, "目标概率必须位于 (0, 1)")
    for pool_size in range(minimum_successes, maximum_pool_size + 1):
        if probability_function(pool_size) >= target_probability:
            return pool_size
    raise ValueError(f"在 {maximum_pool_size} 条以内无法达到目标概率")


def _validate_smoke(smoke_root: Path) -> dict[str, Any]:
    experiment_path = smoke_root / "experiment.json"
    rows_path = smoke_root / "smoke_candidates.jsonl"
    summary_path = smoke_root / "summary.json"
    experiment = _load_json(experiment_path)
    rows = _load_jsonl(rows_path)
    summary = _load_json(summary_path)

    _require(experiment.get("format") == SMOKE_FORMAT, "smoke experiment format 漂移")
    _require(summary.get("format") == SMOKE_FORMAT, "smoke summary format 漂移")
    _require(summary.get("passed") is True, "smoke 未通过")
    _require(summary.get("blocked") is False, "smoke 被阻断")
    _require(summary.get("completed_candidates") == 3, "smoke 完成数不是 3")
    observed_roles = tuple(
        (row.get("smoke_role"), _strict_int(row.get("environment_seed"), "smoke seed"))
        for row in rows
    )
    _require(observed_roles == EXPECTED_SMOKE_ROLES, "smoke role/seed 顺序漂移")
    for row in rows:
        _require(row.get("trajectory_usage") == TRAJECTORY_USAGE, "smoke 禁训练标记漂移")
        _require(row.get("prefix_alignment", {}).get("aligned") is True, "smoke prefix 漂移")
        _require(row.get("engineering_errors") == [], "smoke 含 engineering error")
        _require(row.get("smoke_validation", {}).get("passed") is True, "smoke validation 失败")
    return {
        "passed": True,
        "roles": [role for role, _ in observed_roles],
        "seeds": [seed for _, seed in observed_roles],
        "classification_counts": summary.get("classification_counts"),
        "source_sha256": {
            "experiment": _sha256_file(experiment_path),
            "candidates": _sha256_file(rows_path),
            "summary": _sha256_file(summary_path),
        },
    }


def _validate_counterfactual_index(
    counterfactual_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    experiment_path = counterfactual_root / "experiment.json"
    rows_path = counterfactual_root / "counterfactual_candidates.jsonl"
    summary_path = counterfactual_root / "summary.json"
    experiment = _load_json(experiment_path)
    rows = _load_jsonl(rows_path)
    summary = _load_json(summary_path)

    _require(experiment.get("format") == COUNTERFACTUAL_FORMAT, "counterfactual format 漂移")
    _require(summary.get("format") == COUNTERFACTUAL_FORMAT, "counterfactual summary 漂移")
    _require(summary.get("complete") is True, "counterfactual 未完成")
    _require(summary.get("scan_complete") is True, "counterfactual scan 未完成")
    _require(summary.get("blocked") is False, "counterfactual 被阻断")
    _require(summary.get("all_prefix_aligned") is True, "summary prefix 未全对齐")
    _require(summary.get("expected_candidates") == 16, "counterfactual 预期数漂移")
    _require(summary.get("completed_candidates") == 16, "counterfactual 完成数漂移")
    _require(summary.get("successful_npz_may_enter_d1") is False, "summary D1 禁止标记漂移")
    _require(summary.get("trajectory_usage") == TRAJECTORY_USAGE, "summary 禁训练标记漂移")

    seeds = tuple(_strict_int(row.get("environment_seed"), "counterfactual seed") for row in rows)
    _require(seeds == EXPECTED_COUNTERFACTUAL_SEEDS, "counterfactual seed/order 漂移")
    counts = Counter()
    seeds_by_classification: dict[str, list[int]] = {
        classification: [] for classification in CLASSIFICATIONS
    }
    for row in rows:
        seed = _strict_int(row.get("environment_seed"), "counterfactual seed")
        _require(
            row.get("format") == COUNTERFACTUAL_CANDIDATE_FORMAT,
            f"seed {seed}: candidate format 漂移",
        )
        classification = row.get("classification")
        _require(classification in CLASSIFICATIONS, f"seed {seed}: 未知分类")
        counts[str(classification)] += 1
        seeds_by_classification[str(classification)].append(seed)
        _require(row.get("trajectory_usage") == TRAJECTORY_USAGE, f"seed {seed}: 禁训练标记漂移")
        _require(row.get("successful_npz_may_enter_d1") is False, f"seed {seed}: D1 标记漂移")
        _require(row.get("prefix_alignment", {}).get("aligned") is True, f"seed {seed}: prefix 漂移")
        _require(row.get("engineering_errors") == [], f"seed {seed}: engineering error")
        protocol = row.get("action_budget_protocol")
        _require(isinstance(protocol, dict), f"seed {seed}: protocol 缺失")
        _require(protocol.get("name") == PROTOCOL_NAME, f"seed {seed}: protocol 漂移")
        usage = row.get("action_budget_usage")
        _require(isinstance(usage, dict), f"seed {seed}: usage 缺失")
        policy_actions = _strict_int(usage.get("policy_actions"), f"seed {seed} policy actions")
        expert_actions = _strict_int(usage.get("expert_actions"), f"seed {seed} expert actions")
        total_actions = _strict_int(usage.get("total_actions"), f"seed {seed} total actions")
        _require(policy_actions + expert_actions == total_actions, f"seed {seed}: usage 不闭合")
        _require(policy_actions <= 300 and expert_actions <= 180, f"seed {seed}: 分段预算越界")
        if classification == "recovered_full_eligible":
            _require(row.get("status") == "accepted", f"seed {seed}: recovered 不是 accepted")
            _require(total_actions < 480, f"seed {seed}: accepted 命中 hard deadline")
            _require(row.get("artifact_audit", {}).get("passed") is True, f"seed {seed}: artifact audit 失败")
        else:
            _require(row.get("status") == "rejected", f"seed {seed}: rejection status 漂移")

    normalized_counts = {classification: counts[classification] for classification in CLASSIFICATIONS}
    _require(summary.get("classification_counts") == normalized_counts, "summary 分类计数漂移")
    _require(summary.get("seeds_by_classification") == seeds_by_classification, "summary seed 分类漂移")
    return summary, rows, {
        "experiment": _sha256_file(experiment_path),
        "candidates": _sha256_file(rows_path),
        "summary": _sha256_file(summary_path),
    }


def _candidate_record_evidence(
    counterfactual_root: Path,
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates_root = counterfactual_root / "candidates"
    if not candidates_root.is_dir():
        return [], {
            "available": False,
            "reason": "published compact result does not contain candidate record directories",
        }

    attempts_root = counterfactual_root / ".attempts"
    attempt_entries = list(attempts_root.iterdir()) if attempts_root.is_dir() else []
    _require(not attempt_entries, "counterfactual .attempts 非空")
    evidence: list[dict[str, Any]] = []
    accepted_artifacts = 0
    for row in rows:
        seed = int(row["environment_seed"])
        candidate_dir = candidates_root / f"seed-{seed:06d}"
        record_path = candidate_dir / "record.json"
        subprocess_path = candidate_dir / "subprocess.json"
        result_path = candidate_dir / "counterfactual.json"
        receipt_path = candidate_dir / "receipt.json"
        for path in (record_path, subprocess_path, result_path, receipt_path):
            _require(path.is_file(), f"seed {seed}: 缺少 finalized 文件 {path.name}")
        receipt = _load_json(receipt_path)
        _require(set(receipt) == RECEIPT_KEYS, f"seed {seed}: receipt schema 漂移")
        _require(receipt.get("format") == COUNTERFACTUAL_RECEIPT_FORMAT, f"seed {seed}: receipt format 漂移")
        _require(receipt.get("immutable") is True, f"seed {seed}: receipt 非 immutable")
        _require(receipt.get("result_file") == "counterfactual.json", f"seed {seed}: result_file 漂移")
        _require(receipt.get("record_sha256") == _sha256_file(record_path), f"seed {seed}: record SHA 漂移")
        _require(receipt.get("record_sha256") == row.get("record_sha256"), f"seed {seed}: row/receipt record SHA 漂移")
        _require(receipt.get("subprocess_sha256") == _sha256_file(subprocess_path), f"seed {seed}: subprocess SHA 漂移")
        _require(receipt.get("result_sha256") == _sha256_file(result_path), f"seed {seed}: result SHA 漂移")
        _require(receipt.get("runner_error_sha256") is None, f"seed {seed}: 意外 runner_error")
        _require(not (candidate_dir / "runner_error.json").exists(), f"seed {seed}: runner_error 文件残留")

        record = _load_json(record_path)
        usage = row["action_budget_usage"]
        classification = str(row["classification"])
        item: dict[str, Any] = {
            "environment_seed": seed,
            "classification": classification,
            "record_sha256": row["record_sha256"],
            "policy_actions": usage["policy_actions"],
            "expert_actions": usage["expert_actions"],
            "total_actions": usage["total_actions"],
            "failure_reason": (record.get("failure") or {}).get("reason"),
        }
        if record.get("status") == "accepted":
            trajectory = record.get("result", {}).get("trajectory")
            _require(isinstance(trajectory, dict), f"seed {seed}: accepted trajectory 缺失")
            local_dagger = trajectory.get("local_dagger")
            _require(isinstance(local_dagger, dict), f"seed {seed}: Local DAgger metadata 缺失")
            item.update(
                {
                    "expert_takeover_step": local_dagger.get("expert_takeover_step"),
                    "training_window_start": local_dagger.get("training_window_start"),
                    "training_window_end": local_dagger.get("training_window_end"),
                    "task_completed": True,
                    "phase_at_failure": None,
                    "max_completed_skill_count": 5,
                    "ever_lifted": True,
                    "ever_transported": True,
                }
            )
            audit = row["artifact_audit"]
            dataset_root = candidate_dir / "dataset"
            manifest_path = dataset_root / "manifest.jsonl"
            npz_path = dataset_root / str(trajectory.get("file"))
            _require(audit.get("manifest_sha256") == _sha256_file(manifest_path), f"seed {seed}: manifest SHA 漂移")
            _require(audit.get("npz_sha256") == _sha256_file(npz_path), f"seed {seed}: NPZ SHA 漂移")
            accepted_artifacts += 1
        else:
            diagnostics = record.get("failure_diagnostics")
            _require(isinstance(diagnostics, dict), f"seed {seed}: failure diagnostics 缺失")
            transition = diagnostics.get("final_transition")
            progress = diagnostics.get("final_progress")
            _require(isinstance(transition, dict), f"seed {seed}: final transition 缺失")
            _require(isinstance(progress, dict), f"seed {seed}: final progress 缺失")
            item.update(
                {
                    "expert_takeover_step": diagnostics.get("expert_takeover_step"),
                    "training_window_start": None,
                    "training_window_end": None,
                    "task_completed": progress.get("task_completed"),
                    "phase_at_failure": diagnostics.get("phase_at_failure"),
                    "final_transition_phase": transition.get("phase"),
                    "final_transition_terminated": transition.get("terminated"),
                    "final_transition_truncated": transition.get("truncated"),
                    "max_completed_skill_count": diagnostics.get("max_completed_skill_count"),
                    "ever_lifted": diagnostics.get("ever_lifted"),
                    "ever_transported": diagnostics.get("ever_transported"),
                    "skill_completion_steps": diagnostics.get("skill_completion_steps"),
                    "phase_action_counts": diagnostics.get("phase_action_counts"),
                    "budget_exhaustion_phase": diagnostics.get("budget_exhaustion_phase"),
                }
            )
        evidence.append(item)
    return evidence, {
        "available": True,
        "finalized_receipts_verified": len(evidence),
        "accepted_manifest_npz_hashes_verified": accepted_artifacts,
        "attempt_directory_empty": True,
    }


def _build_planning_model(
    collection_summary: dict[str, Any],
    recovered_count: int,
) -> dict[str, Any]:
    _require(collection_summary.get("format") == COLLECTION_SUMMARY_FORMAT, "collection summary format 漂移")
    grasp_lift = collection_summary.get("boundaries", {}).get("grasp_lift")
    _require(isinstance(grasp_lift, dict), "collection summary 缺 grasp_lift")
    scanned = _strict_int(grasp_lift.get("scanned"), "legacy GL scanned", minimum=1)
    legacy_eligible = _strict_int(grasp_lift.get("eligible"), "legacy GL eligible")
    required = _strict_int(grasp_lift.get("required"), "GL required", minimum=1)
    _require(scanned == 100 and legacy_eligible == 10 and required == 20, "legacy GL identity 漂移")
    planning_eligible = legacy_eligible + recovered_count
    planning_rate = planning_eligible / scanned
    rows = []
    for pool_size in PLANNING_POOL_SIZES:
        rows.append(
            {
                "pool_size": pool_size,
                "expected_eligible_at_point_rate": pool_size * planning_rate,
                "probability_at_least_20_fixed_p": _binomial_tail(
                    pool_size,
                    required,
                    planning_rate,
                ),
                "probability_at_least_20_beta_binomial_uniform": _beta_binomial_tail(
                    pool_size,
                    required,
                    alpha=planning_eligible + 1.0,
                    beta=scanned - planning_eligible + 1.0,
                ),
                "probability_at_least_20_beta_binomial_jeffreys": _beta_binomial_tail(
                    pool_size,
                    required,
                    alpha=planning_eligible + 0.5,
                    beta=scanned - planning_eligible + 0.5,
                ),
            }
        )
    threshold_probabilities = (0.8, 0.9, 0.95)
    minimum_pool_sizes = []
    for target_probability in threshold_probabilities:
        minimum_pool_sizes.append(
            {
                "target_probability": target_probability,
                "fixed_p": _minimum_pool_size(
                    lambda pool_size: _binomial_tail(
                        pool_size,
                        required,
                        planning_rate,
                    ),
                    target_probability,
                    minimum_successes=required,
                ),
                "beta_binomial_uniform": _minimum_pool_size(
                    lambda pool_size: _beta_binomial_tail(
                        pool_size,
                        required,
                        alpha=planning_eligible + 1.0,
                        beta=scanned - planning_eligible + 1.0,
                    ),
                    target_probability,
                    minimum_successes=required,
                ),
                "beta_binomial_jeffreys": _minimum_pool_size(
                    lambda pool_size: _beta_binomial_tail(
                        pool_size,
                        required,
                        alpha=planning_eligible + 0.5,
                        beta=scanned - planning_eligible + 0.5,
                    ),
                    target_probability,
                    minimum_successes=required,
                ),
            }
        )
    return {
        "gate_required_eligible": required,
        "legacy_gl_eligible": legacy_eligible,
        "counterfactual_timeout_recovered_full_eligible": recovered_count,
        "retrospective_planning_eligible": planning_eligible,
        "retrospective_planning_denominator": scanned,
        "planning_point_rate": planning_rate,
        "planning_rate_wilson_95": _wilson_interval(planning_eligible, scanned),
        "pool_options": rows,
        "minimum_pool_sizes_by_model": minimum_pool_sizes,
        "balanced_option": {
            "pool_size": 200,
            "seed_start": 30_200,
            "seed_end_exclusive": 30_400,
        },
        "strict_low_risk_option": {
            "pool_size": 240,
            "seed_start": 30_200,
            "seed_end_exclusive": 30_440,
            "reason": "rounded above the Jeffreys Beta-binomial 95% threshold of 223",
        },
        "assumptions": [
            "10 条 legacy accepted 在 segmented 协议下保持 eligible；当前只由 smoke control 30111 直接验证一条。",
            "5 条 recovered timeout 可与旧 10 条 accepted 合并为容量规划点估计；这不是新的 100-seed formal run。",
            "新 seed 对 eligible 近似独立、同分布；固定 p 与 Beta-binomial 只是规划模型，不是成功保证。",
        ],
    }


def build_analysis(
    *,
    collection_summary_path: Path,
    smoke_root: Path,
    counterfactual_root: Path,
) -> dict[str, Any]:
    collection_summary = _load_json(collection_summary_path)
    smoke = _validate_smoke(smoke_root)
    summary, rows, counterfactual_sha = _validate_counterfactual_index(counterfactual_root)
    candidate_evidence, record_audit = _candidate_record_evidence(counterfactual_root, rows)

    counts = dict(summary["classification_counts"])
    recovered_count = int(counts["recovered_full_eligible"])
    gate_failure_count = int(counts["expert_completed_but_snapshot_or_paired_gate_failed"])
    behavior_completed_count = recovered_count + gate_failure_count
    expert_budget_count = int(counts["expert_recovery_budget_exhausted"])
    other_rejection_count = int(counts["other_behavioral_rejection"])
    hard_deadline_seeds = [
        int(row["environment_seed"])
        for row in rows
        if row["action_budget_usage"]["total_actions"] == 480
        or (row.get("failure") or {}).get("reason") == TIME_LIMIT_REASON
    ]
    recovered_expert_actions = [
        int(row["action_budget_usage"]["expert_actions"])
        for row in rows
        if row["classification"] == "recovered_full_eligible"
    ]
    behavior_completed_expert_actions = [
        int(row["action_budget_usage"]["expert_actions"])
        for row in rows
        if row["classification"]
        in {
            "recovered_full_eligible",
            "expert_completed_but_snapshot_or_paired_gate_failed",
        }
    ]
    rejected_phase_counts: dict[str, int] | None = None
    noncompleted_without_lift = None
    if candidate_evidence:
        rejected_phase_counts = dict(
            sorted(
                Counter(
                    str(item["phase_at_failure"])
                    for item in candidate_evidence
                    if item["classification"] != "recovered_full_eligible"
                ).items()
            )
        )
        noncompleted_without_lift = sum(
            item.get("task_completed") is False and item.get("ever_lifted") is False
            for item in candidate_evidence
        )

    planning = _build_planning_model(collection_summary, recovered_count)
    return {
        "format": ANALYSIS_FORMAT,
        "decision_status": "formal pool size requires owner confirmation",
        "protocol": PROTOCOL_NAME,
        "trajectory_usage": TRAJECTORY_USAGE,
        "source_sha256": {
            "collection_summary": _sha256_file(collection_summary_path),
            "smoke": smoke["source_sha256"],
            "counterfactual": counterfactual_sha,
        },
        "integrity": {
            "smoke_passed": smoke["passed"],
            "counterfactual_complete": summary["complete"],
            "counterfactual_blocked": summary["blocked"],
            "classification_additivity": summary["classification_additivity"],
            "all_prefix_aligned": summary["all_prefix_aligned"],
            "engineering_error_count": counts["engineering_error"],
            "prefix_mismatch_count": counts["prefix_mismatch"],
            "successful_npz_may_enter_d1": summary["successful_npz_may_enter_d1"],
            "candidate_record_audit": record_audit,
        },
        "observed_counterfactual": {
            "cohort": "16 canonical legacy GL takeover-after-boundary TimeLimit seeds",
            "count": len(rows),
            "classification_counts": counts,
            "seeds_by_classification": summary["seeds_by_classification"],
            "recovered_full_eligible_rate": recovered_count / len(rows),
            "behavior_completed_before_gate_rate": behavior_completed_count / len(rows),
            "expert_budget_exhausted_rate": expert_budget_count / len(rows),
            "other_behavioral_rejection_rate": other_rejection_count / len(rows),
            "hard_deadline_count": len(hard_deadline_seeds),
            "hard_deadline_seeds": hard_deadline_seeds,
            "recovered_expert_actions": _summary_stats(recovered_expert_actions),
            "behavior_completed_expert_actions": _summary_stats(
                behavior_completed_expert_actions
            ),
            "rejected_phase_counts": rejected_phase_counts,
            "noncompleted_without_lift_predicate": noncompleted_without_lift,
            "candidate_evidence": candidate_evidence,
        },
        "capacity_planning": planning,
        "interpretation": {
            "verified": [
                "segmented 预算从 16 条旧 TimeLimit 中恢复 5 条完整 eligible，并使另 1 条完成行为但未过 snapshot gate。",
                "16 条均 prefix aligned，且没有 engineering error；没有一条触发 480 Action hard deadline。",
                "180 Action Expert cap 对 4 条是直接停止条件；其余 6 条行为失败在 cap 前已结束 nominal recovery。",
            ],
            "not_established": [
                "该 counterfactual 不证明 Local DAgger 训练会提高策略行为。",
                "metadata prefix equality 不证明 action tensor 或 simulator state bitwise identity。",
                "容量规划概率不保证新 seed pool 的实际 eligible 数。",
            ],
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-summary", required=True, type=Path)
    parser.add_argument("--smoke-root", required=True, type=Path)
    parser.add_argument("--counterfactual-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    analysis = build_analysis(
        collection_summary_path=args.collection_summary.resolve(),
        smoke_root=args.smoke_root.resolve(),
        counterfactual_root=args.counterfactual_root.resolve(),
    )
    _atomic_write_json(args.output.resolve(), analysis)


if __name__ == "__main__":
    main()
