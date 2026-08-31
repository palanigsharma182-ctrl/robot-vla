"""E012 GL segmented-budget 的独立、禁止训练使用的反事实重放。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from robot_vla.cli.replay_local_dagger_failures import (
    COLLECTION_FORMAT,
    POOL_FORMAT,
    _atomic_write_json,
    _atomic_write_jsonl,
    _nonnegative_int,
    _read_json,
    _read_jsonl,
    _sha256_file,
    _validate_runtime_inputs,
    select_replay_rows,
    validate_original_record,
)
from robot_vla.cli.replay_local_dagger_failures import (
    REPLAY_FORMAT as DIAGNOSTIC_REPLAY_FORMAT,
)
from robot_vla.cli.replay_local_dagger_failures import (
    candidate_command as legacy_candidate_command,
)
from robot_vla.local_dagger_protocol import (
    EXPERT_ACTION_BUDGET_EXHAUSTED_REASON,
    LOCAL_DAGGER_ACTION_BUDGET_PROTOCOL_FIELD,
    LOCAL_DAGGER_ACTION_BUDGET_USAGE_FIELD,
    resolve_local_dagger_action_budget,
)
from robot_vla.sim.local_dagger_diagnostics import (
    EPISODE_TIME_LIMIT_REASON,
    LOCAL_DAGGER_DIAGNOSTIC_FORMAT,
)

COUNTERFACTUAL_FORMAT = "robot-vla-local-dagger-budget-counterfactual/v1"
COUNTERFACTUAL_CANDIDATE_FORMAT = (
    "robot-vla-local-dagger-budget-counterfactual-candidate/v1"
)
COUNTERFACTUAL_SUBPROCESS_FORMAT = (
    "robot-vla-local-dagger-budget-counterfactual-subprocess/v1"
)
COUNTERFACTUAL_RECEIPT_FORMAT = (
    "robot-vla-local-dagger-budget-counterfactual-receipt/v1"
)
COUNTERFACTUAL_RECEIPT_KEYS = frozenset(
    {
        "format",
        "target_identity",
        "result_file",
        "record_sha256",
        "subprocess_sha256",
        "runner_error_sha256",
        "result_sha256",
        "immutable",
    }
)
COUNTERFACTUAL_PROTOCOL = "segmented-300-180-480"
COUNTERFACTUAL_PURPOSE = "exploratory counterfactual"
TRAJECTORY_USAGE = "forbidden as training data"
TARGET_BOUNDARY_TYPE = "grasp_lift"
EXPECTED_DIAGNOSTIC_REPLAY_COUNT = 87
EXPECTED_COUNTERFACTUAL_COUNT = 16

E012_FORMAL_SEEDS = tuple(range(30_100, 30_200))
E012_COUNTERFACTUAL_TARGET_SEEDS = (
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
E012_FORMAL_SOURCE_REVISION = (
    "source-tree-sha256:a847e9f90fb255405714351379b1530691f4224f6a29a8e21daad76a5ef8ee00"
)
E012_CHECKPOINT_SHA256 = (
    "a542076f291e29b68e3d28930b15c40396d511a44eb358c2eaeb4e113c041ad6"
)
E012_D0_DATASET_SHA256 = (
    "bc024b6b9c566ca9500945fb6ac262bf657bee713d8a5816229bdc8478139407"
)
E012_FORMAL_EXPERIMENT_SHA256 = (
    "a7e8952a24932cbe8de11f7bf316b7a806d3832413ce697ebb3d51b6eb523a2e"
)
E012_FORMAL_CANDIDATES_SHA256 = (
    "bec0b41f0adddd36e85fac1694d0e59c874a19eda9e2dcbbdefeabb0c164bcc9"
)
E012_DIAGNOSTIC_EXPERIMENT_SHA256 = (
    "3285b9fa0fb75dbd73ed79d3654d7af1141ab53031d7376a756915fb9a120356"
)
E012_DIAGNOSTIC_SUMMARY_SHA256 = (
    "8af443f5f5dbb53c9dd8e1f3f39d30596b02adbc584d8ff456e825d0424c43a7"
)
E012_DIAGNOSTIC_CANDIDATES_SHA256 = (
    "045aae3e9b4b47b53d7a1a87778642e727e952443b2f746b2f20b25d7c22f21d"
)

CLASSIFICATIONS = (
    "recovered_full_eligible",
    "expert_completed_but_snapshot_or_paired_gate_failed",
    "expert_recovery_budget_exhausted",
    "other_behavioral_rejection",
    "prefix_mismatch",
    "engineering_error",
)

PREFIX_FIELDS = (
    "expert_takeover_step",
    "boundary_detection_step",
    "grasp_completion_step",
    "policy_replan_count",
    "policy_replan_traces",
    "policy_sampling_seeds",
)
PREFIX_EVIDENCE_SCOPE = (
    "deterministic metadata alignment through the stable-Grasp boundary and "
    "Expert takeover"
)
PREFIX_NON_PROOF = (
    "Metadata equality is high-confidence prefix evidence; it does not prove "
    "bitwise equality of policy action tensors or simulator state."
)


class CounterfactualMaterializationError(RuntimeError):
    """Subprocess attempt 已原子保留，但没有可审计 collection record。"""

    def __init__(self, seed: int, candidate_dir: Path, reason: str) -> None:
        super().__init__(f"seed {seed}: {reason}")
        self.seed = seed
        self.candidate_dir = candidate_dir
        self.reason = reason


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formal",
        type=Path,
        required=True,
        help="原 E012a Grasp→Lift formal pool 根目录",
    )
    parser.add_argument(
        "--diagnostic-replay",
        type=Path,
        required=True,
        help="已完成且 87/87 matched 的 legacy failure replay 根目录",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="独立 counterfactual 输出根目录；其中 trajectory 禁止进入 D1",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _require_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} 必须是 JSON object")
    return dict(value)


def _resolved_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} 必须是非空路径字符串")
    return Path(value).resolve()


def _unique_seeds(values: Any, *, label: str, expected_count: int) -> list[int]:
    if not isinstance(values, list):
        raise TypeError(f"{label} 必须是 seed list")
    seeds = [_nonnegative_int(value, label=label) for value in values]
    if len(seeds) != expected_count or len(set(seeds)) != expected_count:
        raise ValueError(
            f"{label} 必须包含恰好 {expected_count} 个不重复 seed"
        )
    return seeds


def select_counterfactual_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_seeds: Sequence[int],
) -> list[dict[str, Any]]:
    """只选 formal 中精确的 16 条 post-takeover legacy timeout。"""

    normalized_expected = tuple(
        _nonnegative_int(value, label="formal expected environment_seed")
        for value in expected_seeds
    )
    if normalized_expected != E012_FORMAL_SEEDS:
        raise ValueError("formal environment seeds 不是冻结的 30100..30199")
    replayable = select_replay_rows(rows, expected_seeds=normalized_expected)
    selected = [
        row
        for row in replayable
        if row["failure"]["reason"] == EPISODE_TIME_LIMIT_REASON
    ]
    if len(selected) != EXPECTED_COUNTERFACTUAL_COUNT:
        raise ValueError(
            "counterfactual target 必须是 formal 中精确的 16 条 legacy timeout；"
            f"实际为 {len(selected)}"
        )
    selected_seeds = tuple(int(row["environment_seed"]) for row in selected)
    if selected_seeds != E012_COUNTERFACTUAL_TARGET_SEEDS:
        raise ValueError(
            "counterfactual target tuple 与冻结 E012 16-seed tuple 不一致"
        )
    return selected


def _validate_diagnostic_summary(summary: Mapping[str, Any]) -> list[int]:
    if summary.get("format") != DIAGNOSTIC_REPLAY_FORMAT:
        raise ValueError("diagnostic replay summary format 不兼容")
    required_flags = {
        "scan_complete": True,
        "all_reconciled": True,
        "blocked": False,
    }
    for key, expected in required_flags.items():
        if summary.get(key) is not expected:
            raise RuntimeError(
                f"counterfactual 前置未满足：diagnostic summary {key} "
                f"必须为 {expected!r}"
            )
    for key in ("expected_candidates", "completed_candidates"):
        if summary.get(key) != EXPECTED_DIAGNOSTIC_REPLAY_COUNT:
            raise RuntimeError(
                f"counterfactual 前置未满足：diagnostic summary {key} "
                f"必须为 {EXPECTED_DIAGNOSTIC_REPLAY_COUNT}"
            )
    if summary.get("reconciliation_counts") != {
        "matched": EXPECTED_DIAGNOSTIC_REPLAY_COUNT
    }:
        raise RuntimeError("counterfactual 前置未满足：必须精确 87/87 matched")
    if summary.get("mismatched_seeds") != []:
        raise RuntimeError("diagnostic summary 含 mismatch seed")
    if summary.get("engineering_error_seeds") != []:
        raise RuntimeError("diagnostic summary 含 engineering error seed")
    return _unique_seeds(
        summary.get("matched_seeds"),
        label="diagnostic matched_seeds",
        expected_count=EXPECTED_DIAGNOSTIC_REPLAY_COUNT,
    )


def _diagnostic_prefix(diagnostics: Mapping[str, Any], *, seed: int) -> dict[str, Any]:
    if diagnostics.get("format") != LOCAL_DAGGER_DIAGNOSTIC_FORMAT:
        raise ValueError(f"seed {seed}: diagnostic format 不兼容")
    if diagnostics.get("environment_seed") != seed:
        raise ValueError(f"seed {seed}: diagnostics environment_seed 漂移")
    if diagnostics.get("boundary_type") != TARGET_BOUNDARY_TYPE:
        raise ValueError(f"seed {seed}: diagnostics boundary_type 漂移")
    takeover = _nonnegative_int(
        diagnostics.get("expert_takeover_step"),
        label=f"seed {seed} diagnostic expert_takeover_step",
    )
    boundary = _nonnegative_int(
        diagnostics.get("boundary_detection_step"),
        label=f"seed {seed} diagnostic boundary_detection_step",
    )
    skill_steps = _require_mapping(
        diagnostics.get("skill_completion_steps"),
        label=f"seed {seed} diagnostic skill_completion_steps",
    )
    grasp = _nonnegative_int(
        skill_steps.get("grasp"),
        label=f"seed {seed} diagnostic grasp completion",
    )
    replan_count = _nonnegative_int(
        diagnostics.get("policy_replan_count"),
        label=f"seed {seed} diagnostic policy_replan_count",
    )
    traces_value = diagnostics.get("policy_replan_traces")
    if not isinstance(traces_value, list) or not all(
        isinstance(trace, dict) for trace in traces_value
    ):
        raise ValueError(f"seed {seed}: diagnostic policy_replan_traces 无效")
    traces = [dict(trace) for trace in traces_value]
    if len(traces) != replan_count:
        raise ValueError(f"seed {seed}: diagnostic replan count/traces 不一致")
    sampling_seeds = [trace.get("sampling_seed") for trace in traces]
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in sampling_seeds
    ):
        raise ValueError(f"seed {seed}: diagnostic sampling seeds 无效")
    if not bool(diagnostics.get("boundary_reached")):
        raise ValueError(f"seed {seed}: legacy timeout 未记录 boundary reached")
    if not takeover == boundary == grasp:
        raise ValueError(f"seed {seed}: legacy Grasp/boundary/takeover 不同步")
    return {
        "expert_takeover_step": takeover,
        "boundary_detection_step": boundary,
        "grasp_completion_step": grasp,
        "policy_replan_count": replan_count,
        "policy_replan_traces": traces,
        "policy_sampling_seeds": sampling_seeds,
    }


def _validate_diagnostic_candidate(
    *,
    seed: int,
    row: Mapping[str, Any],
    replay_record: Mapping[str, Any],
    replay_record_path: Path,
    original_record: Mapping[str, Any],
    original_record_sha256: str,
    diagnostic_source_revision: Any,
) -> dict[str, Any] | None:
    if row.get("environment_seed") != seed:
        raise ValueError(f"seed {seed}: diagnostic row seed 漂移")
    if _resolved_path(row.get("record"), label=f"seed {seed} diagnostic record") != (
        replay_record_path.resolve()
    ):
        raise ValueError(f"seed {seed}: diagnostic row record path 漂移")
    reconciliation = _require_mapping(
        row.get("reconciliation"),
        label=f"seed {seed} diagnostic reconciliation",
    )
    if (
        reconciliation.get("classification") != "matched"
        or reconciliation.get("reconciled") is not True
        or reconciliation.get("exact_match") is not True
        or row.get("reconciled") is not True
    ):
        raise RuntimeError(f"seed {seed}: diagnostic candidate 不是 exact matched")
    original = _require_mapping(
        row.get("original"),
        label=f"seed {seed} diagnostic original",
    )
    if original.get("record_sha256") != original_record_sha256:
        raise ValueError(f"seed {seed}: diagnostic row 的 formal record SHA 漂移")
    if original.get("failure") != original_record.get("failure"):
        raise ValueError(f"seed {seed}: diagnostic row 的 formal failure 漂移")

    if replay_record.get("format") != COLLECTION_FORMAT:
        raise ValueError(f"seed {seed}: diagnostic replay record format 不兼容")
    if replay_record.get("status") != "rejected":
        raise ValueError(f"seed {seed}: diagnostic replay status 不是 rejected")
    if replay_record.get("failure") != original_record.get("failure"):
        raise ValueError(f"seed {seed}: diagnostic replay failure 未与 formal 匹配")
    if replay_record.get("config") != original_record.get("config"):
        raise ValueError(f"seed {seed}: diagnostic replay config 漂移")
    if replay_record.get("checkpoint") != original_record.get("checkpoint"):
        raise ValueError(f"seed {seed}: diagnostic replay checkpoint 漂移")
    if replay_record.get("base_dataset") != original_record.get("base_dataset"):
        raise ValueError(f"seed {seed}: diagnostic replay base dataset 漂移")
    if replay_record.get("source_revision") != diagnostic_source_revision:
        raise ValueError(f"seed {seed}: diagnostic replay source revision 漂移")
    diagnostics = _require_mapping(
        replay_record.get("failure_diagnostics"),
        label=f"seed {seed} diagnostic failure_diagnostics",
    )
    if row.get("failure_diagnostics") != diagnostics:
        raise ValueError(f"seed {seed}: diagnostic row/record diagnostics 漂移")
    if diagnostics.get("format") != LOCAL_DAGGER_DIAGNOSTIC_FORMAT:
        raise ValueError(f"seed {seed}: diagnostic format 不兼容")
    if diagnostics.get("environment_seed") != seed:
        raise ValueError(f"seed {seed}: diagnostic environment_seed 漂移")
    if diagnostics.get("boundary_type") != TARGET_BOUNDARY_TYPE:
        raise ValueError(f"seed {seed}: diagnostic boundary_type 漂移")
    replay_failure = replay_record["failure"]
    if diagnostics.get("failure_reason") != replay_failure.get("reason"):
        raise ValueError(f"seed {seed}: diagnostic failure reason 漂移")
    if replay_failure.get("reason") == EPISODE_TIME_LIMIT_REASON:
        return _diagnostic_prefix(diagnostics, seed=seed)
    return None


def build_counterfactual_experiment(
    formal_root: Path,
    diagnostic_replay_root: Path,
    *,
    counterfactual_source_revision: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """冻结 formal、87-seed replay 及 16 个 target 的全部输入身份。"""

    formal_root = formal_root.resolve()
    diagnostic_replay_root = diagnostic_replay_root.resolve()
    if formal_root == diagnostic_replay_root:
        raise ValueError("formal 与 diagnostic replay 必须是独立目录")

    formal_experiment_path = formal_root / "experiment.json"
    formal_candidates_path = formal_root / "collection_candidates.jsonl"
    if _sha256_file(formal_experiment_path) != E012_FORMAL_EXPERIMENT_SHA256:
        raise ValueError("formal experiment.json 不是已发布 E012 content SHA")
    if _sha256_file(formal_candidates_path) != E012_FORMAL_CANDIDATES_SHA256:
        raise ValueError("formal collection_candidates 不是已发布 E012 content SHA")
    formal_experiment = _read_json(
        formal_experiment_path,
        label="formal experiment.json",
    )
    if formal_experiment.get("format") != POOL_FORMAT:
        raise ValueError("formal experiment format 不兼容")
    if formal_experiment.get("boundary_type") != TARGET_BOUNDARY_TYPE:
        raise ValueError("counterfactual 首版只接受 grasp_lift formal pool")
    if formal_experiment.get("source_revision") != E012_FORMAL_SOURCE_REVISION:
        raise ValueError("formal source revision 不是冻结 E012 revision")
    formal_checkpoint = formal_experiment.get("checkpoint")
    if not isinstance(formal_checkpoint, dict) or formal_checkpoint.get(
        "sha256"
    ) != E012_CHECKPOINT_SHA256:
        raise ValueError("formal checkpoint 不是冻结 E012 checkpoint")
    formal_dataset = formal_experiment.get("base_dataset")
    formal_dataset_audit = (
        formal_dataset.get("audit") if isinstance(formal_dataset, dict) else None
    )
    if not isinstance(formal_dataset_audit, dict) or formal_dataset_audit.get(
        "dataset_sha256"
    ) != E012_D0_DATASET_SHA256:
        raise ValueError("formal base dataset 不是冻结 E012 D0")
    formal_seeds_value = formal_experiment.get("environment_seeds")
    if not isinstance(formal_seeds_value, list):
        raise TypeError("formal experiment 缺少 environment_seeds")
    formal_seeds = [
        _nonnegative_int(value, label="formal environment_seed")
        for value in formal_seeds_value
    ]
    formal_rows = _read_jsonl(
        formal_candidates_path,
        label="formal collection_candidates.jsonl",
    )
    selected_formal_rows = select_counterfactual_rows(
        formal_rows,
        expected_seeds=formal_seeds,
    )
    formal_rows_by_seed = {
        int(row["environment_seed"]): row for row in formal_rows
    }

    diagnostic_experiment_path = diagnostic_replay_root / "experiment.json"
    diagnostic_summary_path = diagnostic_replay_root / "summary.json"
    diagnostic_candidates_path = diagnostic_replay_root / "replay_candidates.jsonl"
    if _sha256_file(
        diagnostic_experiment_path
    ) != E012_DIAGNOSTIC_EXPERIMENT_SHA256:
        raise ValueError("diagnostic experiment 不是已发布 E012 content SHA")
    if _sha256_file(diagnostic_summary_path) != E012_DIAGNOSTIC_SUMMARY_SHA256:
        raise ValueError("diagnostic summary 不是已发布 E012 content SHA")
    if _sha256_file(
        diagnostic_candidates_path
    ) != E012_DIAGNOSTIC_CANDIDATES_SHA256:
        raise ValueError("diagnostic candidates 不是已发布 E012 content SHA")
    diagnostic_experiment = _read_json(
        diagnostic_experiment_path,
        label="diagnostic replay experiment.json",
    )
    diagnostic_summary = _read_json(
        diagnostic_summary_path,
        label="diagnostic replay summary.json",
    )
    matched_seeds = _validate_diagnostic_summary(diagnostic_summary)
    if diagnostic_experiment.get("format") != DIAGNOSTIC_REPLAY_FORMAT:
        raise ValueError("diagnostic replay experiment format 不兼容")
    diagnostic_seeds = _unique_seeds(
        diagnostic_experiment.get("environment_seeds"),
        label="diagnostic experiment environment_seeds",
        expected_count=EXPECTED_DIAGNOSTIC_REPLAY_COUNT,
    )
    if diagnostic_seeds != matched_seeds:
        raise ValueError("diagnostic experiment 与 summary seed 顺序/集合漂移")
    if diagnostic_experiment.get("selected_count") != EXPECTED_DIAGNOSTIC_REPLAY_COUNT:
        raise ValueError("diagnostic experiment selected_count 不是 87")

    embedded_formal = _require_mapping(
        diagnostic_experiment.get("formal_pool"),
        label="diagnostic experiment formal_pool",
    )
    if _resolved_path(embedded_formal.get("path"), label="embedded formal path") != formal_root:
        raise ValueError("diagnostic replay 指向的 formal pool 与 --formal 不一致")
    if embedded_formal.get("experiment") != formal_experiment:
        raise ValueError("diagnostic replay 内嵌 formal experiment 已漂移")
    if embedded_formal.get("experiment_sha256") != _sha256_file(
        formal_experiment_path
    ):
        raise ValueError("diagnostic replay 冻结的 formal experiment SHA 已漂移")
    if _resolved_path(
        embedded_formal.get("collection_candidates"),
        label="embedded formal collection_candidates",
    ) != formal_candidates_path.resolve():
        raise ValueError("diagnostic replay 冻结的 formal candidates path 已漂移")
    if embedded_formal.get("collection_candidates_sha256") != _sha256_file(
        formal_candidates_path
    ):
        raise ValueError("diagnostic replay 冻结的 formal candidates SHA 已漂移")

    selected_candidates_value = diagnostic_experiment.get("selected_candidates")
    if not isinstance(selected_candidates_value, list):
        raise TypeError("diagnostic experiment 缺少 selected_candidates")
    diagnostic_selected_by_seed: dict[int, dict[str, Any]] = {}
    for index, value in enumerate(selected_candidates_value):
        candidate = _require_mapping(
            value,
            label=f"diagnostic selected_candidates[{index}]",
        )
        seed = _nonnegative_int(
            candidate.get("environment_seed"),
            label="diagnostic selected environment_seed",
        )
        if seed in diagnostic_selected_by_seed:
            raise ValueError(f"diagnostic selected_candidates 重复 seed {seed}")
        diagnostic_selected_by_seed[seed] = candidate
    if set(diagnostic_selected_by_seed) != set(diagnostic_seeds):
        raise ValueError("diagnostic selected_candidates seed 集合漂移")

    diagnostic_rows = _read_jsonl(
        diagnostic_candidates_path,
        label="diagnostic replay_candidates.jsonl",
    )
    if len(diagnostic_rows) != EXPECTED_DIAGNOSTIC_REPLAY_COUNT:
        raise ValueError("diagnostic replay_candidates 必须精确包含 87 条")
    diagnostic_rows_by_seed: dict[int, dict[str, Any]] = {}
    for index, row in enumerate(diagnostic_rows):
        seed = _nonnegative_int(
            row.get("environment_seed"),
            label=f"diagnostic row {index} environment_seed",
        )
        if seed in diagnostic_rows_by_seed:
            raise ValueError(f"diagnostic replay_candidates 重复 seed {seed}")
        diagnostic_rows_by_seed[seed] = row
    if list(diagnostic_rows_by_seed) != diagnostic_seeds:
        raise ValueError("diagnostic replay_candidates seed 顺序/集合漂移")

    # summary 可能过期，所以仍逐 seed 验证 87 条 manifest/record reconciliation。
    validated_inputs: dict[int, dict[str, Any]] = {}
    diagnostic_source_revision = diagnostic_experiment.get(
        "replay_source_revision"
    )
    for seed in diagnostic_seeds:
        formal_row = formal_rows_by_seed.get(seed)
        if formal_row is None:
            raise ValueError(f"seed {seed}: diagnostic seed 不在 formal manifest")
        formal_record_path = (
            formal_root / "candidates" / f"seed-{seed:06d}" / "record.json"
        )
        formal_record = _read_json(
            formal_record_path,
            label=f"seed {seed} formal record",
        )
        validate_original_record(
            formal_row,
            formal_record,
            formal_experiment=formal_experiment,
            expected_record_path=formal_record_path,
        )
        formal_record_sha256 = _sha256_file(formal_record_path)
        selected_identity = diagnostic_selected_by_seed[seed]
        if _resolved_path(
            selected_identity.get("record"),
            label=f"seed {seed} diagnostic frozen formal record",
        ) != formal_record_path.resolve():
            raise ValueError(f"seed {seed}: diagnostic frozen formal path 漂移")
        if selected_identity.get("record_sha256") != formal_record_sha256:
            raise ValueError(f"seed {seed}: diagnostic frozen formal SHA 漂移")
        if selected_identity.get("failure") != formal_record.get("failure"):
            raise ValueError(f"seed {seed}: diagnostic frozen formal failure 漂移")

        replay_record_path = (
            diagnostic_replay_root
            / "candidates"
            / f"seed-{seed:06d}"
            / "record.json"
        )
        replay_record = _read_json(
            replay_record_path,
            label=f"seed {seed} diagnostic replay record",
        )
        replay_record_sha256 = _sha256_file(replay_record_path)
        prefix = _validate_diagnostic_candidate(
            seed=seed,
            row=diagnostic_rows_by_seed[seed],
            replay_record=replay_record,
            replay_record_path=replay_record_path,
            original_record=formal_record,
            original_record_sha256=formal_record_sha256,
            diagnostic_source_revision=diagnostic_source_revision,
        )
        validated_inputs[seed] = {
            "formal_record": formal_record,
            "formal_record_path": formal_record_path,
            "formal_record_sha256": formal_record_sha256,
            "diagnostic_record": replay_record,
            "diagnostic_record_path": replay_record_path,
            "diagnostic_record_sha256": replay_record_sha256,
            "reference_prefix": prefix,
        }

    model_cache = formal_experiment.get("model_cache")
    if not isinstance(model_cache, str) or not model_cache:
        raise ValueError("formal experiment 缺少 model_cache")
    action_budget = resolve_local_dagger_action_budget(COUNTERFACTUAL_PROTOCOL)
    protocol_metadata = action_budget.planned_metadata()
    if protocol_metadata is None:
        raise RuntimeError("counterfactual 必须使用 amended action-budget protocol")

    targets: list[dict[str, Any]] = []
    frozen_targets: list[dict[str, Any]] = []
    for formal_row in selected_formal_rows:
        seed = int(formal_row["environment_seed"])
        validated = validated_inputs[seed]
        formal_record = validated["formal_record"]
        if formal_record["failure"]["reason"] != EPISODE_TIME_LIMIT_REASON:
            raise ValueError(f"seed {seed}: counterfactual target 不是精确 timeout")
        if not isinstance(validated["reference_prefix"], dict):
            raise TypeError(f"seed {seed}: timeout target 缺少可信 prefix")
        target = {
            "environment_seed": seed,
            "boundary_type": TARGET_BOUNDARY_TYPE,
            "model_cache": model_cache,
            "replay_source_revision": counterfactual_source_revision,
            "original_record": formal_record,
            "original_record_path": str(
                validated["formal_record_path"].resolve()
            ),
            "original_record_sha256": validated["formal_record_sha256"],
            "diagnostic_record": validated["diagnostic_record"],
            "diagnostic_record_path": str(
                validated["diagnostic_record_path"].resolve()
            ),
            "diagnostic_record_sha256": validated[
                "diagnostic_record_sha256"
            ],
            "reference_prefix": validated["reference_prefix"],
        }
        targets.append(target)
        frozen_targets.append(
            {
                "environment_seed": seed,
                "boundary_type": TARGET_BOUNDARY_TYPE,
                "original_failure": formal_record["failure"],
                "formal_record": target["original_record_path"],
                "formal_record_sha256": target["original_record_sha256"],
                "diagnostic_record": target["diagnostic_record_path"],
                "diagnostic_record_sha256": target[
                    "diagnostic_record_sha256"
                ],
                "reference_prefix": target["reference_prefix"],
            }
        )

    experiment = {
        "format": COUNTERFACTUAL_FORMAT,
        "purpose": COUNTERFACTUAL_PURPOSE,
        "trajectory_usage": TRAJECTORY_USAGE,
        "source_revision": counterfactual_source_revision,
        "checkpoint": formal_experiment["checkpoint"],
        "base_dataset": formal_experiment["base_dataset"],
        "model_cache": model_cache,
        "legacy_config": formal_experiment["config"],
        "action_budget_protocol": protocol_metadata,
        "environment_seeds": [item["environment_seed"] for item in targets],
        "selected_count": len(targets),
        "target_boundary_type": TARGET_BOUNDARY_TYPE,
        "target_original_failure_reason": EPISODE_TIME_LIMIT_REASON,
        "formal_input": {
            "path": str(formal_root),
            "experiment": str(formal_experiment_path.resolve()),
            "experiment_sha256": _sha256_file(formal_experiment_path),
            "collection_candidates": str(formal_candidates_path.resolve()),
            "collection_candidates_sha256": _sha256_file(
                formal_candidates_path
            ),
        },
        "diagnostic_replay_input": {
            "path": str(diagnostic_replay_root),
            "experiment": str(diagnostic_experiment_path.resolve()),
            "experiment_sha256": _sha256_file(diagnostic_experiment_path),
            "summary": str(diagnostic_summary_path.resolve()),
            "summary_sha256": _sha256_file(diagnostic_summary_path),
            "replay_candidates": str(diagnostic_candidates_path.resolve()),
            "replay_candidates_sha256": _sha256_file(
                diagnostic_candidates_path
            ),
            "required_precondition": {
                "scan_complete": True,
                "all_reconciled": True,
                "blocked": False,
                "matched": EXPECTED_DIAGNOSTIC_REPLAY_COUNT,
            },
            "formal_record_sha256_by_seed": {
                str(seed): validated_inputs[seed]["formal_record_sha256"]
                for seed in diagnostic_seeds
            },
            "diagnostic_record_sha256_by_seed": {
                str(seed): validated_inputs[seed]["diagnostic_record_sha256"]
                for seed in diagnostic_seeds
            },
        },
        "selected_candidates": frozen_targets,
        "prefix_alignment": {
            "required_fields": list(PREFIX_FIELDS),
            "evidence_scope": PREFIX_EVIDENCE_SCOPE,
            "bitwise_action_identity_proven": False,
            "non_proof": PREFIX_NON_PROOF,
        },
        "execution": {
            "entrypoint": "robot_vla.cli.collect_local_dagger",
            "only_command_delta": [
                "--action-budget-protocol",
                COUNTERFACTUAL_PROTOCOL,
            ],
            "offline_environment": True,
            "resume_contract": "completed candidate directories are immutable",
            "trajectory_usage": TRAJECTORY_USAGE,
            "successful_npz_may_enter_d1": False,
        },
    }
    normalized = json.loads(json.dumps(experiment, sort_keys=True, allow_nan=False))
    return normalized, targets


def candidate_command(
    target: Mapping[str, Any],
    *,
    candidate_dir: Path,
    python_executable: str = sys.executable,
) -> list[str]:
    """与 legacy replay 命令相比只追加一个冻结 protocol 参数。"""

    command = legacy_candidate_command(
        target,
        candidate_dir=candidate_dir,
        python_executable=python_executable,
    )
    if "--action-budget-protocol" in command:
        raise RuntimeError("legacy replay command 意外已包含 action-budget protocol")
    return [
        *command,
        "--action-budget-protocol",
        COUNTERFACTUAL_PROTOCOL,
    ]


def _expected_config(target: Mapping[str, Any]) -> dict[str, Any]:
    original = _require_mapping(
        target["original_record"].get("config"),
        label="original config",
    )
    if LOCAL_DAGGER_ACTION_BUDGET_PROTOCOL_FIELD in original:
        raise ValueError("counterfactual original config 必须是 legacy protocol")
    planned = resolve_local_dagger_action_budget(
        COUNTERFACTUAL_PROTOCOL
    ).planned_metadata()
    if planned is None:
        raise RuntimeError("counterfactual protocol 没有 planned metadata")
    return {
        **original,
        LOCAL_DAGGER_ACTION_BUDGET_PROTOCOL_FIELD: planned,
    }


def _result_prefix(record: Mapping[str, Any], *, seed: int) -> dict[str, Any]:
    result = _require_mapping(record.get("result"), label=f"seed {seed} result")
    trajectory = _require_mapping(
        result.get("trajectory"),
        label=f"seed {seed} result trajectory",
    )
    provenance = _require_mapping(
        trajectory.get("local_dagger"),
        label=f"seed {seed} result local_dagger",
    )
    takeover = _nonnegative_int(
        provenance.get("expert_takeover_step"),
        label=f"seed {seed} result expert_takeover_step",
    )
    boundary = _nonnegative_int(
        provenance.get("boundary_detection_step"),
        label=f"seed {seed} result boundary_detection_step",
    )
    replan_count = _nonnegative_int(
        result.get("policy_replans"),
        label=f"seed {seed} result policy_replans",
    )
    traces_value = result.get("policy_replan_traces")
    sampling_value = result.get("policy_sampling_seeds")
    if not isinstance(traces_value, list) or not all(
        isinstance(trace, dict) for trace in traces_value
    ):
        raise ValueError(f"seed {seed}: result policy_replan_traces 无效")
    if not isinstance(sampling_value, list) or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in sampling_value
    ):
        raise ValueError(f"seed {seed}: result policy_sampling_seeds 无效")
    traces = [dict(trace) for trace in traces_value]
    sampling = list(sampling_value)
    if len(traces) != replan_count or len(sampling) != replan_count:
        raise ValueError(f"seed {seed}: result replan count/traces/seeds 不一致")
    if [trace.get("sampling_seed") for trace in traces] != sampling:
        raise ValueError(f"seed {seed}: result trace/sampling seed 不一致")
    if not takeover == boundary:
        raise ValueError(f"seed {seed}: result boundary/takeover 不同步")
    return {
        "expert_takeover_step": takeover,
        "boundary_detection_step": boundary,
        # Grasp→Lift 的 boundary 即 stable-Grasp completion。
        "grasp_completion_step": boundary,
        "policy_replan_count": replan_count,
        "policy_replan_traces": traces,
        "policy_sampling_seeds": sampling,
    }


def extract_counterfactual_prefix(
    record: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    if record.get("status") == "accepted":
        return _result_prefix(record, seed=seed)
    diagnostics = _require_mapping(
        record.get("failure_diagnostics"),
        label=f"seed {seed} failure_diagnostics",
    )
    return _diagnostic_prefix(diagnostics, seed=seed)


def compare_prefix(
    reference: Mapping[str, Any],
    observed: Mapping[str, Any],
) -> dict[str, Any]:
    field_matches = {
        field: observed.get(field) == reference.get(field)
        for field in PREFIX_FIELDS
    }
    return {
        "aligned": all(field_matches.values()),
        "field_matches": field_matches,
        "reference": dict(reference),
        "observed": dict(observed),
        "evidence_scope": PREFIX_EVIDENCE_SCOPE,
        "bitwise_action_identity_proven": False,
        "non_proof": PREFIX_NON_PROOF,
    }


def _validate_usage(
    record: Mapping[str, Any],
    *,
    observed_prefix: Mapping[str, Any],
    seed: int,
) -> list[str]:
    errors: list[str] = []
    usage = record.get(LOCAL_DAGGER_ACTION_BUDGET_USAGE_FIELD)
    if not isinstance(usage, dict):
        return ["missing top-level action_budget_usage"]
    if set(usage) != {"policy_actions", "expert_actions", "total_actions"}:
        return ["action_budget_usage keys drifted"]
    values: dict[str, int] = {}
    for key in ("policy_actions", "expert_actions", "total_actions"):
        value = usage.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            errors.append(f"action_budget_usage.{key} invalid")
        else:
            values[key] = value
    if errors:
        return errors
    if values["policy_actions"] + values["expert_actions"] != values["total_actions"]:
        errors.append("action_budget_usage is not additive")
    if values["policy_actions"] > 300:
        errors.append("policy action usage exceeds 300")
    if values["expert_actions"] > 180:
        errors.append("expert action usage exceeds 180")
    if record.get("status") == "accepted":
        # 第 480 个 Action 已触发环境 hard deadline；可信 success 必须先于它。
        if values["total_actions"] >= 480:
            errors.append("accepted total action usage must be strictly below 480")
    elif values["total_actions"] > 480:
        errors.append("rejected total action usage exceeds 480")
    takeover = observed_prefix.get("expert_takeover_step")
    if isinstance(takeover, int) and values["policy_actions"] != takeover:
        errors.append(f"seed {seed}: policy usage/takeover mismatch")

    if record.get("status") == "accepted":
        result = record.get("result")
        if not isinstance(result, dict) or result.get(
            LOCAL_DAGGER_ACTION_BUDGET_USAGE_FIELD
        ) != usage:
            errors.append("accepted result/top-level action budget usage mismatch")
        else:
            trajectory = result.get("trajectory")
            if not isinstance(trajectory, dict) or trajectory.get(
                "num_steps"
            ) != values["total_actions"]:
                errors.append(
                    "accepted trajectory.num_steps/action budget total mismatch"
                )
            randomization = (
                trajectory.get("randomization")
                if isinstance(trajectory, dict)
                else None
            )
            if not isinstance(randomization, dict) or randomization.get(
                LOCAL_DAGGER_ACTION_BUDGET_USAGE_FIELD
            ) != usage:
                errors.append("trajectory/top-level action budget usage mismatch")
    else:
        diagnostics = record.get("failure_diagnostics")
        if not isinstance(diagnostics, dict) or diagnostics.get(
            LOCAL_DAGGER_ACTION_BUDGET_USAGE_FIELD
        ) != usage:
            errors.append("failure diagnostics/top-level action budget usage mismatch")
        elif diagnostics.get("action_count") != values["total_actions"]:
            errors.append("rejected diagnostics.action_count/usage total mismatch")
        if isinstance(diagnostics, dict):
            final_transition = diagnostics.get("final_transition")
            if isinstance(final_transition, dict) and final_transition.get(
                "action_step"
            ) != values["total_actions"]:
                errors.append("rejected final transition/action usage mismatch")
            failure = record.get("failure")
            reason = failure.get("reason") if isinstance(failure, dict) else None
            is_time_limit_reason = reason == EPISODE_TIME_LIMIT_REASON
            is_hard_limit_usage = values["total_actions"] == 480
            has_truncated_transition = (
                isinstance(final_transition, dict)
                and final_transition.get("truncated") is True
            )
            # 这三个 hard-deadline 信号必须双向闭合。否则损坏记录可能把
            # step-480 truncation 伪装成 snapshot/paired-gate rejection。
            if any(
                (
                    is_time_limit_reason,
                    is_hard_limit_usage,
                    has_truncated_transition,
                )
            ) and not all(
                (
                    is_time_limit_reason,
                    is_hard_limit_usage,
                    has_truncated_transition,
                )
            ):
                errors.append(
                    "hard-deadline reason/usage/truncation evidence is not closed"
                )
            if reason == EPISODE_TIME_LIMIT_REASON:
                if values["total_actions"] != 480:
                    errors.append("time-limit rejection must occur at total action 480")
                if not isinstance(final_transition, dict) or final_transition.get(
                    "truncated"
                ) is not True:
                    errors.append("time-limit rejection must carry truncated transition")
            if reason == EXPERT_ACTION_BUDGET_EXHAUSTED_REASON:
                if values["expert_actions"] != 180:
                    errors.append("expert-budget rejection must consume 180 Expert actions")
                if values["total_actions"] >= 480:
                    errors.append("expert-budget rejection must precede hard deadline")
                if not isinstance(final_transition, dict) or final_transition.get(
                    "truncated"
                ) is not False:
                    errors.append("expert-budget rejection must be nontruncated")
                if diagnostics.get("budget_exhaustion_phase") != "expert":
                    errors.append("expert-budget rejection lacks Expert exhaustion phase")
                budget_step = diagnostics.get(
                    "budget_exhaustion_step",
                    diagnostics.get("failure_control_step"),
                )
                if budget_step != values["total_actions"]:
                    errors.append("expert budget exhaustion step/usage total mismatch")
    return errors


def _accepted_artifact_contract_errors(
    record: Mapping[str, Any],
    *,
    record_path: Path,
) -> list[str]:
    """验证 accepted record、单条 manifest 与 NPZ 的最小闭环身份。"""

    result = record.get("result")
    trajectory = result.get("trajectory") if isinstance(result, dict) else None
    if not isinstance(trajectory, dict):
        return ["accepted artifact lacks trajectory metadata"]
    file_value = trajectory.get("file")
    num_steps = trajectory.get("num_steps")
    if not isinstance(file_value, str) or not file_value:
        return ["accepted trajectory file is invalid"]
    if not isinstance(num_steps, int) or isinstance(num_steps, bool) or num_steps <= 0:
        return ["accepted trajectory num_steps is invalid"]
    dataset_root = record_path.parent / "dataset"
    manifest_path = dataset_root / "manifest.jsonl"
    errors: list[str] = []
    try:
        manifest_rows = _read_jsonl(
            manifest_path,
            label="accepted candidate manifest.jsonl",
        )
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        return [f"accepted candidate manifest invalid: {error}"]
    if len(manifest_rows) != 1 or manifest_rows[0] != trajectory:
        errors.append("accepted manifest is not the exact record trajectory")
    relative_file = Path(file_value)
    if relative_file.is_absolute():
        errors.append("accepted trajectory file must be relative to candidate dataset")
        return errors
    npz_path = (dataset_root / relative_file).resolve()
    if not npz_path.is_relative_to(dataset_root.resolve()):
        errors.append("accepted trajectory file escapes candidate dataset")
        return errors
    if not npz_path.is_file():
        errors.append("accepted trajectory NPZ is missing")
        return errors
    try:
        import numpy as np

        with np.load(npz_path, allow_pickle=False) as arrays:
            required = {
                "action",
                "action_source",
                "expert_supervision_mask",
                "success",
                "skill_id",
            }
            missing = sorted(required - set(arrays.files))
            if missing:
                errors.append(f"accepted trajectory NPZ lacks arrays: {missing}")
            for name in sorted(required & set(arrays.files)):
                value = arrays[name]
                if value.ndim < 1 or value.shape[0] != num_steps:
                    errors.append(
                        f"accepted trajectory NPZ {name} length/num_steps mismatch"
                    )
    except (OSError, TypeError, ValueError) as error:
        errors.append(f"accepted trajectory NPZ is unreadable: {error}")
    return errors


def audit_accepted_candidate_artifact(
    record: Mapping[str, Any],
    *,
    record_path: Path,
) -> dict[str, Any]:
    """为 GPU smoke 复用正式单轨迹 audit，而不写 Dataset 审计产物。"""

    structural_errors = _accepted_artifact_contract_errors(
        record,
        record_path=record_path,
    )
    if structural_errors:
        raise ValueError("; ".join(structural_errors))
    from robot_vla.contracts import RobotSpec
    from robot_vla.data.audit import audit_trajectory
    from robot_vla.data.trajectory import TrajectoryMeta, TrajectoryStore

    trajectory = record["result"]["trajectory"]
    dataset_root = record_path.parent / "dataset"
    meta = TrajectoryMeta.from_dict(trajectory)
    spec = RobotSpec()
    arrays = TrajectoryStore(dataset_root, spec, cache_size=0).get(meta)
    audit_trajectory(arrays, meta, spec)
    npz_path = (dataset_root / meta.file).resolve()
    return {
        "passed": True,
        "trajectory_id": meta.trajectory_id,
        "num_steps": arrays.num_steps,
        "manifest_sha256": _sha256_file(dataset_root / "manifest.jsonl"),
        "npz_sha256": _sha256_file(npz_path),
    }


def _accepted_contract_errors(record: Mapping[str, Any], *, seed: int) -> list[str]:
    errors: list[str] = []
    if record.get("failure") is not None:
        errors.append("accepted record unexpectedly contains failure")
    if record.get("eligible_for_risk_selection") is not True:
        errors.append("accepted record is not risk-selection eligible")
    audit = record.get("audit")
    if not isinstance(audit, dict) or audit.get("trajectory_contract") != "passed":
        errors.append("accepted trajectory audit did not pass")
    result = record.get("result")
    trajectory = result.get("trajectory") if isinstance(result, dict) else None
    provenance = (
        trajectory.get("local_dagger") if isinstance(trajectory, dict) else None
    )
    outcome = (
        trajectory.get("outcome_evidence") if isinstance(trajectory, dict) else None
    )
    if not isinstance(provenance, dict):
        errors.append("accepted record lacks Local DAgger provenance")
    else:
        if provenance.get("rollin_seed") != seed:
            errors.append("accepted Local DAgger rollin seed drifted")
        if provenance.get("boundary_type") != TARGET_BOUNDARY_TYPE:
            errors.append("accepted Local DAgger boundary type drifted")
        if provenance.get("expert_recovery_success") is not True:
            errors.append("accepted Local DAgger expert recovery is not successful")
    if not isinstance(outcome, dict) or outcome.get("task_completed") is not True:
        errors.append("accepted trajectory lacks full task success evidence")
    paired = record.get("paired_clean_expert")
    if not isinstance(paired, dict) or paired.get("task_completed") is not True:
        errors.append("accepted record lacks successful paired clean Expert")
    if not isinstance(record.get("risk_components"), dict):
        errors.append("accepted record lacks paired risk components")
    snapshot = result.get("snapshot_round_trip") if isinstance(result, dict) else None
    if not isinstance(snapshot, dict) or snapshot.get("passed") is not True:
        errors.append("accepted snapshot round-trip did not pass")
    return errors


def _rejected_contract_errors(record: Mapping[str, Any]) -> list[str]:
    failure = record.get("failure")
    if not isinstance(failure, dict):
        return ["rejected record lacks failure"]
    if failure.get("type") != "EpisodeRejected":
        return ["rejected record failure is not behavioral EpisodeRejected"]
    reason = failure.get("reason")
    if not isinstance(reason, str) or not reason:
        return ["rejected record failure reason is invalid"]
    diagnostics = record.get("failure_diagnostics")
    if not isinstance(diagnostics, dict):
        return ["rejected record lacks failure_diagnostics"]
    errors: list[str] = []
    if diagnostics.get("format") != LOCAL_DAGGER_DIAGNOSTIC_FORMAT:
        errors.append("rejected diagnostics format drifted")
    if diagnostics.get("failure_reason") != reason:
        errors.append("rejected diagnostics/failure reason mismatch")
    return errors


def _expert_completed_before_gate(record: Mapping[str, Any]) -> bool:
    diagnostics = record.get("failure_diagnostics")
    if not isinstance(diagnostics, dict):
        return False
    progress = diagnostics.get("final_progress")
    if not isinstance(progress, dict):
        return False
    return bool(
        progress.get("task_completed") is True
        and diagnostics.get("max_completed_skill_count") == 5
    )


def _is_environment_hard_deadline(record: Mapping[str, Any]) -> bool:
    failure = record.get("failure")
    diagnostics = record.get("failure_diagnostics")
    usage = record.get(LOCAL_DAGGER_ACTION_BUDGET_USAGE_FIELD)
    if not isinstance(failure, dict) or not isinstance(diagnostics, dict):
        return False
    if not isinstance(usage, dict):
        return False
    final_transition = diagnostics.get("final_transition")
    return bool(
        failure.get("reason") == EPISODE_TIME_LIMIT_REASON
        and usage.get("total_actions") == 480
        and isinstance(final_transition, dict)
        and final_transition.get("action_step") == 480
        and final_transition.get("truncated") is True
    )


def reconcile_counterfactual_record(
    target: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    record_path: Path,
    subprocess_returncode: int,
    subprocess_command: Sequence[str] | None = None,
    expected_command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """验证工程契约、比较高置信 prefix，并给出唯一 outcome 分类。"""

    seed = int(target["environment_seed"])
    engineering_errors: list[str] = []
    status = record.get("status")
    if record.get("format") != COLLECTION_FORMAT:
        engineering_errors.append("collection format drifted")
    if status not in {"accepted", "rejected"}:
        engineering_errors.append(f"unsupported collection status {status!r}")
    if record.get("source_revision") != target.get("replay_source_revision"):
        engineering_errors.append("counterfactual source revision drifted")
    if record.get("base_dataset") != target["original_record"].get("base_dataset"):
        engineering_errors.append("base dataset identity drifted")
    if record.get("checkpoint") != target["original_record"].get("checkpoint"):
        engineering_errors.append("checkpoint identity drifted")
    try:
        expected_config = _expected_config(target)
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        expected_config = None
        engineering_errors.append(f"invalid frozen config: {error}")
    if expected_config is not None and record.get("config") != expected_config:
        engineering_errors.append("collection config is not legacy config plus fixed protocol")
    if not isinstance(subprocess_returncode, int) or isinstance(
        subprocess_returncode, bool
    ):
        engineering_errors.append("subprocess returncode is invalid")
    elif not (
        (status == "accepted" and subprocess_returncode == 0)
        or (status == "rejected" and subprocess_returncode != 0)
    ):
        engineering_errors.append("subprocess returncode/status contract mismatch")
    if expected_command is not None and list(subprocess_command or ()) != list(
        expected_command
    ):
        engineering_errors.append("subprocess command identity drifted")

    if status == "accepted":
        engineering_errors.extend(_accepted_contract_errors(record, seed=seed))
        engineering_errors.extend(
            _accepted_artifact_contract_errors(
                record,
                record_path=record_path,
            )
        )
    elif status == "rejected":
        engineering_errors.extend(_rejected_contract_errors(record))

    observed_prefix: dict[str, Any] | None = None
    prefix_error: str | None = None
    try:
        observed_prefix = extract_counterfactual_prefix(record, seed=seed)
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        prefix_error = str(error)
        engineering_errors.append(f"counterfactual prefix is malformed: {error}")

    if observed_prefix is not None:
        engineering_errors.extend(
            _validate_usage(record, observed_prefix=observed_prefix, seed=seed)
        )
    planned = resolve_local_dagger_action_budget(
        COUNTERFACTUAL_PROTOCOL
    ).planned_metadata()
    diagnostics_or_randomization: Mapping[str, Any] | None = None
    if status == "rejected" and isinstance(record.get("failure_diagnostics"), dict):
        diagnostics_or_randomization = record["failure_diagnostics"]
    elif status == "accepted" and isinstance(record.get("result"), dict):
        trajectory = record["result"].get("trajectory")
        if isinstance(trajectory, dict) and isinstance(
            trajectory.get("randomization"), dict
        ):
            diagnostics_or_randomization = trajectory["randomization"]
    if (
        diagnostics_or_randomization is None
        or diagnostics_or_randomization.get(
            LOCAL_DAGGER_ACTION_BUDGET_PROTOCOL_FIELD
        )
        != planned
    ):
        engineering_errors.append("segmented protocol metadata missing or drifted")

    if observed_prefix is None:
        prefix_alignment = {
            "aligned": False,
            "field_matches": {field: False for field in PREFIX_FIELDS},
            "reference": dict(target["reference_prefix"]),
            "observed": None,
            "evidence_scope": PREFIX_EVIDENCE_SCOPE,
            "bitwise_action_identity_proven": False,
            "non_proof": PREFIX_NON_PROOF,
            "extraction_error": prefix_error,
        }
    else:
        prefix_alignment = compare_prefix(
            target["reference_prefix"],
            observed_prefix,
        )

    if engineering_errors:
        classification = "engineering_error"
    elif not prefix_alignment["aligned"]:
        classification = "prefix_mismatch"
    elif status == "accepted":
        classification = "recovered_full_eligible"
    else:
        failure = record["failure"]
        diagnostics = record["failure_diagnostics"]
        if _is_environment_hard_deadline(record):
            # TimeLimit 的环境语义优先，不能因 final_progress 恰为 completed
            # 而误写成 snapshot/paired gate failure。
            classification = "other_behavioral_rejection"
        elif _expert_completed_before_gate(record):
            classification = (
                "expert_completed_but_snapshot_or_paired_gate_failed"
            )
        elif (
            failure.get("reason") == EXPERT_ACTION_BUDGET_EXHAUSTED_REASON
            and diagnostics.get("budget_exhaustion_phase") == "expert"
            and record[LOCAL_DAGGER_ACTION_BUDGET_USAGE_FIELD].get(
                "expert_actions"
            )
            == 180
        ):
            classification = "expert_recovery_budget_exhausted"
        else:
            classification = "other_behavioral_rejection"

    return {
        "format": COUNTERFACTUAL_CANDIDATE_FORMAT,
        "purpose": COUNTERFACTUAL_PURPOSE,
        "trajectory_usage": TRAJECTORY_USAGE,
        "environment_seed": seed,
        "boundary_type": TARGET_BOUNDARY_TYPE,
        "status": status,
        "failure": record.get("failure"),
        "record": str(record_path.resolve()),
        "record_sha256": _sha256_file(record_path),
        "classification": classification,
        "engineering_errors": engineering_errors,
        "prefix_alignment": prefix_alignment,
        "original": {
            "failure": target["original_record"].get("failure"),
            "formal_record": target["original_record_path"],
            "formal_record_sha256": target["original_record_sha256"],
            "diagnostic_record": target.get("diagnostic_record_path"),
            "diagnostic_record_sha256": target.get(
                "diagnostic_record_sha256"
            ),
        },
        "action_budget_protocol": planned,
        "action_budget_usage": record.get(
            LOCAL_DAGGER_ACTION_BUDGET_USAGE_FIELD
        ),
        "successful_npz_may_enter_d1": False,
        "subprocess": {
            "returncode": subprocess_returncode,
            "command_matches": (
                True
                if expected_command is None
                else list(subprocess_command or ()) == list(expected_command)
            ),
        },
    }


def summarize_counterfactual(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_candidates: int = EXPECTED_COUNTERFACTUAL_COUNT,
) -> dict[str, Any]:
    counts = Counter(str(row.get("classification")) for row in rows)
    unknown = sorted(set(counts) - set(CLASSIFICATIONS))
    if unknown:
        raise ValueError(f"counterfactual summary 含未知分类: {unknown}")
    complete_counts = {name: counts.get(name, 0) for name in CLASSIFICATIONS}
    classified = sum(complete_counts.values())
    scan_complete = len(rows) == expected_candidates
    additivity_holds = classified == len(rows)
    return {
        "format": COUNTERFACTUAL_FORMAT,
        "purpose": COUNTERFACTUAL_PURPOSE,
        "trajectory_usage": TRAJECTORY_USAGE,
        "complete": scan_complete,
        "scan_complete": scan_complete,
        "blocked": complete_counts["engineering_error"] > 0,
        "all_prefix_aligned": (
            scan_complete
            and complete_counts["prefix_mismatch"] == 0
            and complete_counts["engineering_error"] == 0
        ),
        "expected_candidates": expected_candidates,
        "completed_candidates": len(rows),
        "classification_counts": complete_counts,
        "classification_additivity": {
            "classified_candidates": classified,
            "completed_candidates": len(rows),
            "holds": additivity_holds,
        },
        "seeds_by_classification": {
            name: [
                int(row["environment_seed"])
                for row in rows
                if row.get("classification") == name
            ]
            for name in CLASSIFICATIONS
        },
        "successful_npz_may_enter_d1": False,
    }


def _write_progress(
    output: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_candidates: int,
) -> None:
    ordered = sorted(rows, key=lambda row: int(row["environment_seed"]))
    _atomic_write_jsonl(output / "counterfactual_candidates.jsonl", ordered)
    _atomic_write_json(
        output / "summary.json",
        summarize_counterfactual(
            ordered,
            expected_candidates=expected_candidates,
        ),
    )


def _prepare_output(
    output: Path,
    experiment: Mapping[str, Any],
    *,
    resume: bool,
) -> None:
    experiment_path = output / "experiment.json"
    if resume:
        existing = _read_json(
            experiment_path,
            label="counterfactual experiment.json",
        )
        if existing != experiment:
            raise ValueError("--resume counterfactual experiment identity 漂移")
        return
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("counterfactual output 目录非空；拒绝覆盖")
    output.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(experiment_path, experiment)


def _receipt_target_identity(target: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "environment_seed": int(target["environment_seed"]),
        "source_revision": target.get("replay_source_revision"),
        "formal_record_sha256": target.get("original_record_sha256"),
        "diagnostic_record_sha256": target.get("diagnostic_record_sha256"),
    }


def _load_finalized_candidate(
    candidate_dir: Path,
    target: Mapping[str, Any],
    *,
    result_filename: str,
    allow_missing_record: bool = False,
) -> dict[str, Any] | None:
    """验证 immutable receipt；未 finalize 的 candidate 返回 ``None``。"""

    result_path = candidate_dir / result_filename
    receipt_path = candidate_dir / "receipt.json"
    if not result_path.exists() and not receipt_path.exists():
        return None
    # result 先原子落盘、receipt 后落盘；这一种单向中断由重算等值恢复。
    if result_path.is_file() and not receipt_path.exists():
        return None
    if not result_path.is_file() or not receipt_path.is_file():
        raise RuntimeError(
            f"candidate {candidate_dir.name}: result/receipt 只存在一项，拒绝覆盖"
        )
    receipt = _read_json(receipt_path, label=f"{candidate_dir.name} receipt")
    if set(receipt) != COUNTERFACTUAL_RECEIPT_KEYS:
        raise RuntimeError(f"candidate {candidate_dir.name}: receipt schema 漂移")
    if receipt["format"] != COUNTERFACTUAL_RECEIPT_FORMAT:
        raise RuntimeError(f"candidate {candidate_dir.name}: receipt format 漂移")
    if receipt["immutable"] is not True:
        raise RuntimeError(f"candidate {candidate_dir.name}: receipt immutable 标记漂移")
    if receipt["result_file"] != result_filename:
        raise RuntimeError(f"candidate {candidate_dir.name}: receipt result_file 漂移")
    if receipt["target_identity"] != _receipt_target_identity(target):
        raise RuntimeError(f"candidate {candidate_dir.name}: receipt target identity 漂移")
    subprocess_path = candidate_dir / "subprocess.json"
    if not subprocess_path.is_file() or receipt[
        "subprocess_sha256"
    ] != _sha256_file(subprocess_path):
        raise RuntimeError(f"candidate {candidate_dir.name}: subprocess receipt SHA 漂移")
    record_path = candidate_dir / "record.json"
    expected_record_sha = receipt["record_sha256"]
    if record_path.is_file():
        if expected_record_sha != _sha256_file(record_path):
            raise RuntimeError(f"candidate {candidate_dir.name}: record receipt SHA 漂移")
    elif not allow_missing_record or expected_record_sha is not None:
        raise RuntimeError(f"candidate {candidate_dir.name}: receipt record 缺失")
    runner_error_path = candidate_dir / "runner_error.json"
    expected_runner_error_sha = receipt["runner_error_sha256"]
    if runner_error_path.is_file():
        if expected_runner_error_sha != _sha256_file(runner_error_path):
            raise RuntimeError(
                f"candidate {candidate_dir.name}: runner_error receipt SHA 漂移"
            )
    elif expected_runner_error_sha is not None:
        raise RuntimeError(f"candidate {candidate_dir.name}: runner_error 缺失")
    if receipt["result_sha256"] != _sha256_file(result_path):
        raise RuntimeError(f"candidate {candidate_dir.name}: result receipt SHA 漂移")
    result = _read_json(result_path, label=f"{candidate_dir.name} finalized result")
    if record_path.is_file() and result.get("status") == "accepted":
        recorded_artifact_audit = result.get("artifact_audit")
        if not isinstance(recorded_artifact_audit, dict):
            raise RuntimeError(
                f"candidate {candidate_dir.name}: accepted receipt 缺少 artifact audit"
            )
        record = _read_json(
            record_path,
            label=f"{candidate_dir.name} immutable accepted record",
        )
        try:
            current_artifact_audit = audit_accepted_candidate_artifact(
                record,
                record_path=record_path,
            )
        except (KeyError, OSError, TypeError, ValueError, RuntimeError) as error:
            raise RuntimeError(
                f"candidate {candidate_dir.name}: accepted artifact 不再可信: {error}"
            ) from error
        if current_artifact_audit != recorded_artifact_audit:
            raise RuntimeError(
                f"candidate {candidate_dir.name}: accepted artifact identity 漂移"
            )
    return result


def _finalize_candidate(
    candidate_dir: Path,
    target: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    result_filename: str,
    allow_missing_record: bool = False,
) -> dict[str, Any]:
    """只写一次 result+receipt；崩溃在两次原子写之间时仅允许等值恢复。"""

    result_path = candidate_dir / result_filename
    receipt_path = candidate_dir / "receipt.json"
    if receipt_path.exists():
        existing = _load_finalized_candidate(
            candidate_dir,
            target,
            result_filename=result_filename,
            allow_missing_record=allow_missing_record,
        )
        if existing != row:
            raise RuntimeError(
                f"candidate {candidate_dir.name}: finalized result 与重算结果漂移"
            )
        return dict(existing)
    if result_path.exists():
        existing = _read_json(
            result_path,
            label=f"{candidate_dir.name} interrupted finalized result",
        )
        if existing != row:
            raise RuntimeError(
                f"candidate {candidate_dir.name}: 无 receipt 的 result 与重算结果漂移"
            )
    else:
        _atomic_write_json(result_path, row)

    record_path = candidate_dir / "record.json"
    subprocess_path = candidate_dir / "subprocess.json"
    runner_error_path = candidate_dir / "runner_error.json"
    if not subprocess_path.is_file():
        raise RuntimeError(f"candidate {candidate_dir.name}: finalize 缺少 subprocess")
    if not record_path.is_file() and not allow_missing_record:
        raise RuntimeError(f"candidate {candidate_dir.name}: finalize 缺少 record")
    receipt = {
        "format": COUNTERFACTUAL_RECEIPT_FORMAT,
        "target_identity": _receipt_target_identity(target),
        "result_file": result_filename,
        "record_sha256": (
            _sha256_file(record_path) if record_path.is_file() else None
        ),
        "subprocess_sha256": _sha256_file(subprocess_path),
        "runner_error_sha256": (
            _sha256_file(runner_error_path) if runner_error_path.is_file() else None
        ),
        "result_sha256": _sha256_file(result_path),
        "immutable": True,
    }
    _atomic_write_json(receipt_path, receipt)
    return dict(row)


def _materialization_error_row(
    target: Mapping[str, Any],
    error: CounterfactualMaterializationError,
) -> dict[str, Any]:
    subprocess_path = error.candidate_dir / "subprocess.json"
    runner_error_path = error.candidate_dir / "runner_error.json"
    return {
        "format": COUNTERFACTUAL_CANDIDATE_FORMAT,
        "purpose": COUNTERFACTUAL_PURPOSE,
        "trajectory_usage": TRAJECTORY_USAGE,
        "environment_seed": int(target["environment_seed"]),
        "boundary_type": TARGET_BOUNDARY_TYPE,
        "status": "error",
        "failure": {
            "type": type(error).__name__,
            "reason": error.reason,
        },
        "record": None,
        "record_sha256": None,
        "classification": "engineering_error",
        "engineering_errors": [error.reason],
        "prefix_alignment": {
            "aligned": False,
            "field_matches": {field: False for field in PREFIX_FIELDS},
            "reference": dict(target["reference_prefix"]),
            "observed": None,
            "evidence_scope": PREFIX_EVIDENCE_SCOPE,
            "bitwise_action_identity_proven": False,
            "non_proof": PREFIX_NON_PROOF,
        },
        "original": {
            "failure": target["original_record"].get("failure"),
            "formal_record": target["original_record_path"],
            "formal_record_sha256": target["original_record_sha256"],
            "diagnostic_record": target.get("diagnostic_record_path"),
            "diagnostic_record_sha256": target.get(
                "diagnostic_record_sha256"
            ),
        },
        "action_budget_protocol": resolve_local_dagger_action_budget(
            COUNTERFACTUAL_PROTOCOL
        ).planned_metadata(),
        "action_budget_usage": None,
        "successful_npz_may_enter_d1": False,
        "subprocess": {
            "path": str(subprocess_path.resolve()),
            "sha256": (
                _sha256_file(subprocess_path) if subprocess_path.is_file() else None
            ),
        },
        "runner_error_sha256": (
            _sha256_file(runner_error_path) if runner_error_path.is_file() else None
        ),
    }


def _completed_attempt(output: Path, seed: int) -> Path | None:
    attempts_root = output / ".attempts"
    if not attempts_root.is_dir():
        return None
    completed = [
        path
        for path in sorted(attempts_root.glob(f"seed-{seed:06d}-*"))
        if (path / "record.json").is_file()
        and (path / "subprocess.json").is_file()
    ]
    if len(completed) > 1:
        raise RuntimeError(
            f"seed {seed}: 存在多个完整 counterfactual attempt，拒绝自动选择"
        )
    return completed[0] if completed else None


def _materialize_candidate(
    args: argparse.Namespace,
    target: Mapping[str, Any],
    *,
    project_root: Path,
) -> Path:
    seed = int(target["environment_seed"])
    candidate_dir = args.output / "candidates" / f"seed-{seed:06d}"
    if candidate_dir.exists():
        if not candidate_dir.is_dir():
            raise RuntimeError(f"seed {seed}: candidate path 不是目录")
        if not (candidate_dir / "record.json").is_file():
            if (candidate_dir / "subprocess.json").is_file() and (
                candidate_dir / "runner_error.json"
            ).is_file():
                runner_error = _read_json(
                    candidate_dir / "runner_error.json",
                    label=f"seed {seed} runner_error",
                )
                raise CounterfactualMaterializationError(
                    seed,
                    candidate_dir,
                    str(runner_error.get("reason", "subprocess missing record.json")),
                )
            raise RuntimeError(f"seed {seed}: incomplete candidate 缺少 record.json")
        if not (candidate_dir / "subprocess.json").is_file():
            raise RuntimeError(f"seed {seed}: completed candidate 缺少 subprocess.json")
        return candidate_dir

    recovered = _completed_attempt(args.output, seed)
    if recovered is not None:
        candidate_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(recovered, candidate_dir)
        return candidate_dir

    attempts_root = args.output / ".attempts"
    attempts_root.mkdir(parents=True, exist_ok=True)
    attempt_dir = Path(
        tempfile.mkdtemp(prefix=f"seed-{seed:06d}-", dir=attempts_root)
    )
    command = candidate_command(target, candidate_dir=attempt_dir)
    environment = os.environ.copy()
    environment.setdefault("HF_HUB_OFFLINE", "1")
    environment.setdefault("TRANSFORMERS_OFFLINE", "1")
    with (
        (attempt_dir / "stdout.log").open("w", encoding="utf-8") as stdout,
        (attempt_dir / "stderr.log").open("w", encoding="utf-8") as stderr,
    ):
        completed = subprocess.run(
            command,
            cwd=project_root,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    _atomic_write_json(
        attempt_dir / "subprocess.json",
        {
            "format": COUNTERFACTUAL_SUBPROCESS_FORMAT,
            "returncode": completed.returncode,
            "command": command,
            "purpose": COUNTERFACTUAL_PURPOSE,
            "trajectory_usage": TRAJECTORY_USAGE,
        },
    )
    if not (attempt_dir / "record.json").is_file():
        _atomic_write_json(
            attempt_dir / "runner_error.json",
            {
                "classification": "engineering_error",
                "reason": "subprocess completed without record.json",
                "returncode": completed.returncode,
            },
        )
        candidate_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(attempt_dir, candidate_dir)
        raise CounterfactualMaterializationError(
            seed,
            candidate_dir,
            "subprocess completed without record.json",
        )
    candidate_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(attempt_dir, candidate_dir)
    return candidate_dir


def run(args: argparse.Namespace) -> None:
    # 保持所有输入审计/分类函数可在无 Torch/ManiSkill 的环境导入。
    from robot_vla.cli.train_stage1 import compute_source_revision

    formal_root = args.formal.resolve()
    diagnostic_replay_root = args.diagnostic_replay.resolve()
    output = args.output.resolve()
    for source_root, label in (
        (formal_root, "formal"),
        (diagnostic_replay_root, "diagnostic replay"),
    ):
        if output == source_root or output.is_relative_to(source_root):
            raise ValueError(f"counterfactual output 必须独立于 {label}")

    project_root = Path(__file__).resolve().parents[3]
    experiment, targets = build_counterfactual_experiment(
        formal_root,
        diagnostic_replay_root,
        counterfactual_source_revision=compute_source_revision(project_root),
    )
    _validate_runtime_inputs(targets)
    _prepare_output(output, experiment, resume=args.resume)

    rows: list[dict[str, Any]] = []
    for target in targets:
        seed = int(target["environment_seed"])
        try:
            candidate_dir = _materialize_candidate(
                argparse.Namespace(output=output),
                target,
                project_root=project_root,
            )
        except CounterfactualMaterializationError as error:
            candidate_dir = error.candidate_dir
            row = _load_finalized_candidate(
                candidate_dir,
                target,
                result_filename="counterfactual.json",
                allow_missing_record=True,
            )
            if row is None:
                row = _finalize_candidate(
                    candidate_dir,
                    target,
                    _materialization_error_row(target, error),
                    result_filename="counterfactual.json",
                    allow_missing_record=True,
                )
            rows.append(row)
            _write_progress(output, rows, expected_candidates=len(targets))
            raise RuntimeError(
                f"seed {seed}: subprocess 未产生 record；"
                "已写 engineering row、blocked summary 与 immutable receipt"
            ) from error

        row = _load_finalized_candidate(
            candidate_dir,
            target,
            result_filename="counterfactual.json",
        )
        if row is not None:
            rows.append(row)
            _write_progress(output, rows, expected_candidates=len(targets))
            if row.get("classification") == "engineering_error":
                raise RuntimeError(
                    f"seed {seed}: immutable candidate 为 engineering error"
                )
            continue
        record_path = candidate_dir / "record.json"
        record = _read_json(
            record_path,
            label=f"seed {seed} counterfactual record",
        )
        subprocess_record = _read_json(
            candidate_dir / "subprocess.json",
            label=f"seed {seed} counterfactual subprocess",
        )
        expected_command = candidate_command(target, candidate_dir=candidate_dir)
        # attempt 被原子 rename 后，命令中的 --output/--record 仍带 attempt 路径；
        # 只将这两个输出路径标准化到最终 candidate，再验证其余命令完全冻结。
        actual_command = subprocess_record.get("command")
        normalized_command = list(actual_command) if isinstance(actual_command, list) else []
        if normalized_command:
            for flag, suffix in (("--output", "dataset"), ("--record", "record.json")):
                if flag in normalized_command:
                    normalized_command[normalized_command.index(flag) + 1] = str(
                        (candidate_dir / suffix).resolve()
                    )
        subprocess_format_matches = (
            subprocess_record.get("format") == COUNTERFACTUAL_SUBPROCESS_FORMAT
        )
        returncode = subprocess_record.get("returncode")
        if not isinstance(returncode, int) or isinstance(returncode, bool):
            returncode = -999_999
            subprocess_format_matches = False
        row = reconcile_counterfactual_record(
            target,
            record,
            record_path=record_path,
            subprocess_returncode=returncode,
            subprocess_command=normalized_command,
            expected_command=expected_command,
        )
        if not subprocess_format_matches:
            row["engineering_errors"].append("subprocess record format/returncode invalid")
            row["classification"] = "engineering_error"
        artifact_audit = None
        if record.get("status") == "accepted":
            try:
                artifact_audit = audit_accepted_candidate_artifact(
                    record,
                    record_path=record_path,
                )
            except (KeyError, OSError, TypeError, ValueError, RuntimeError) as error:
                row["engineering_errors"].append(
                    f"accepted full trajectory audit failed: {error}"
                )
                row["classification"] = "engineering_error"
        row["artifact_audit"] = artifact_audit
        row = _finalize_candidate(
            candidate_dir,
            target,
            row,
            result_filename="counterfactual.json",
        )
        rows.append(row)
        _write_progress(output, rows, expected_candidates=len(targets))
        print(
            json.dumps(
                {
                    "event": "budget_counterfactual_complete",
                    "seed": seed,
                    "classification": row["classification"],
                    "completed": len(rows),
                    "expected": len(targets),
                },
                sort_keys=True,
                allow_nan=False,
            ),
            flush=True,
        )
        if row["classification"] == "engineering_error":
            raise RuntimeError(
                f"seed {seed}: counterfactual engineering error，已写审计结果并阻断"
            )

    summary = summarize_counterfactual(rows, expected_candidates=len(targets))
    if not summary["complete"]:
        raise RuntimeError("counterfactual 未产生全部 16 条结果")


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
