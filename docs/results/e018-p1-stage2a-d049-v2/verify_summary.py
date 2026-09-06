"""独立验证 D049 v2 脱敏聚合结果及其证据边界。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
VERSION = "e018-p1-stage2a-d049-selected-gain-public/v2"

EXPECTED_IDENTITY = {
    "source_commit": "d8a721612858995c40602f0e755a5cbd071f3d12",
    "source_identity_sha256": (
        "9a13827a5e22973a8f3e9bb2e935c4f70f9315806e6a93d3d4260e590ee63140"
    ),
    "config_raw_sha256": (
        "9105ea421709951636c7778f94f83bd07144ed14f52033e5dabaf8930a24e08a"
    ),
    "config_canonical_sha256": (
        "ce1b78a80745fa86c8f630f42ed16ee28600ca7f0369f46acd036b89c6bfc1ee"
    ),
    "transaction_identity_sha256": (
        "3a1276542416ae7ad4f07dc316d5100d98a9d2fbe49b00b122bf320b81f86d2f"
    ),
    "public_verification_sha256": (
        "8a3209f4794e4912a20c4d9b52084c2b38e532225d450593de449635c7391395"
    ),
    "result_verification_sha256": (
        "3bee7e9c7735f5034e69a9e5f4066eea04a2ab3d21dbca716517701ac69d6fa9"
    ),
    "result_complete_internal_sha256": (
        "09ace7cb1236370c879319ac3067ca5e1fede22de4d1a29f7d330cd5969261c8"
    ),
}

EXPECTED_TOP_LEVEL_KEYS = {
    "version",
    "decision_id",
    "experiment_id",
    "status",
    "classification",
    "evidence_scope",
    "identity",
    "selection",
    "formal_development_aggregate",
    "boundary_counters",
    "interpretation",
    "claim_boundaries",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _require_exact_keys(value: dict[str, Any], expected: set[str], name: str) -> None:
    _require(set(value) == expected, f"{name} 字段集合漂移")


def main() -> None:
    summary = json.loads((ROOT / "summary.json").read_text(encoding="utf-8"))
    _require(isinstance(summary, dict), "summary 顶层必须是对象")
    _require_exact_keys(summary, EXPECTED_TOP_LEVEL_KEYS, "summary")

    _require(summary["version"] == VERSION, "公开结果版本漂移")
    _require(summary["decision_id"] == "D049", "Decision identity 漂移")
    _require(
        summary["experiment_id"]
        == "E018-P1-S2A-SELECTED-GAIN-DEVELOPMENT-EVALUATION/v2",
        "Experiment identity 漂移",
    )
    _require(summary["status"] == "completed", "D049 v2 完成状态漂移")
    _require(
        summary["classification"]
        == "effect-negative-persist-publish-pause-for-reusability-refactor",
        "结果分类漂移",
    )
    _require(summary["identity"] == EXPECTED_IDENTITY, "正式身份或验证哈希漂移")

    scope = summary["evidence_scope"]
    _require_exact_keys(
        scope,
        {
            "data_scope",
            "evaluation_mode",
            "test_split_read",
            "actuation_mode",
            "canonical_runtime_mutated",
        },
        "evidence_scope",
    )
    _require(scope["data_scope"] == "development", "证据不得扩写为 test")
    _require(scope["evaluation_mode"] == "offline", "证据不得扩写为闭环")
    _require(scope["test_split_read"] is False, "正式结果必须保持 no-test")
    _require(scope["actuation_mode"] == "none", "正式结果必须保持 no-actuation")
    _require(scope["canonical_runtime_mutated"] is False, "不得修改 canonical runtime")

    selection = summary["selection"]
    _require_exact_keys(
        selection,
        {
            "selected_gain",
            "selection_rule",
            "support",
            "recovered",
            "recovery_rate",
            "gain_superiority_established",
        },
        "selection",
    )
    _require(math.isclose(selection["selected_gain"], 0.1), "selected gain 漂移")
    _require(
        selection["selection_rule"] == "largest-preregistered-gain-on-exact-tie",
        "平局选择规则漂移",
    )
    _require(selection["support"] == 24, "selection support 漂移")
    _require(selection["recovered"] == 5, "selection recovered 漂移")
    _require(
        math.isclose(
            selection["recovery_rate"],
            selection["recovered"] / selection["support"],
        ),
        "selection recovery rate 算术不一致",
    )
    _require(
        selection["gain_superiority_established"] is False,
        "平局不得声称 gain superiority",
    )

    aggregate = summary["formal_development_aggregate"]
    _require_exact_keys(
        aggregate,
        {
            "route_count",
            "oracle_recoverable_support",
            "recovered_count",
            "recovery_rate",
            "required_recovery_rate",
            "minimum_support",
            "recovery_distance_max_m",
            "catastrophic_distance_strictly_greater_than_m",
            "unsafe_count",
            "catastrophic_count",
            "false_recovery_count",
            "protocol_violation_count",
            "support_gate_passed",
            "effect_gate_passed",
        },
        "formal_development_aggregate",
    )
    _require(aggregate["route_count"] == 25, "route count 漂移")
    _require(aggregate["oracle_recoverable_support"] == 25, "support 漂移")
    _require(aggregate["recovered_count"] == 7, "recovered count 漂移")
    _require(
        math.isclose(
            aggregate["recovery_rate"],
            aggregate["recovered_count"] / aggregate["oracle_recoverable_support"],
        ),
        "formal recovery rate 算术不一致",
    )
    _require(math.isclose(aggregate["required_recovery_rate"], 0.7), "效果门槛漂移")
    _require(aggregate["minimum_support"] == 10, "最小 support 门槛漂移")
    _require(math.isclose(aggregate["recovery_distance_max_m"], 0.005), "恢复阈值漂移")
    _require(
        math.isclose(aggregate["catastrophic_distance_strictly_greater_than_m"], 0.02),
        "catastrophic 阈值漂移",
    )
    _require(
        aggregate["oracle_recoverable_support"] >= aggregate["minimum_support"],
        "support gate 算术不一致",
    )
    _require(aggregate["support_gate_passed"] is True, "support gate 状态漂移")
    effect_gate = (
        10 * aggregate["recovered_count"]
        >= 7 * aggregate["oracle_recoverable_support"]
    )
    _require(effect_gate is False, "正式效果门槛应失败")
    _require(aggregate["effect_gate_passed"] is False, "不得把负向结果标为通过")
    for field in (
        "unsafe_count",
        "catastrophic_count",
        "false_recovery_count",
        "protocol_violation_count",
    ):
        _require(aggregate[field] == 0, f"{field} 漂移")

    counters = summary["boundary_counters"]
    _require_exact_keys(
        counters,
        {
            "fresh_test_reads",
            "runtime_ground_truth_reads",
            "goal_ground_truth_reads",
            "physical_camera_actuation",
            "arm_tcp_actuation",
            "gripper_close",
            "canonical_runtime_mutations",
        },
        "boundary_counters",
    )
    _require(all(value == 0 for value in counters.values()), "证据或执行边界不得扩张")

    interpretation = summary["interpretation"]
    _require_exact_keys(
        interpretation,
        {
            "selection_to_formal_rate_delta",
            "generalization_sufficient",
            "threshold_retuning_allowed",
            "consumed_identity_reuse_allowed",
            "stage2b_continuation_required",
            "next_state",
        },
        "interpretation",
    )
    expected_delta = aggregate["recovery_rate"] - selection["recovery_rate"]
    _require(
        math.isclose(interpretation["selection_to_formal_rate_delta"], expected_delta),
        "selection/formal rate delta 算术不一致",
    )
    for field in (
        "generalization_sufficient",
        "threshold_retuning_allowed",
        "consumed_identity_reuse_allowed",
        "stage2b_continuation_required",
    ):
        _require(interpretation[field] is False, f"{field} 必须保持 false")
    _require(
        interpretation["next_state"] == "PAUSE_FOR_REUSABILITY_REFACTOR",
        "D049 收尾后的暂停状态漂移",
    )

    claims = summary["claim_boundaries"]
    _require_exact_keys(
        claims,
        {
            "gain_superiority",
            "actuator_safety",
            "canonical_promotion",
            "physical_closed_loop_success",
            "deployment_readiness",
        },
        "claim_boundaries",
    )
    _require(all(value is False for value in claims.values()), "公开结果不得产生越界 claim")

    print(json.dumps({"status": "passed", "version": VERSION}, sort_keys=True))


if __name__ == "__main__":
    main()
