"""固定三 seed 的 E012 segmented-budget GPU smoke 编排器。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from robot_vla.cli.replay_local_dagger_budget_counterfactual import (
    CLASSIFICATIONS,
    COUNTERFACTUAL_PROTOCOL,
    COUNTERFACTUAL_SUBPROCESS_FORMAT,
    TRAJECTORY_USAGE,
    CounterfactualMaterializationError,
    _finalize_candidate,
    _load_finalized_candidate,
    _materialization_error_row,
    _materialize_candidate,
    _prepare_output,
    _result_prefix,
    audit_accepted_candidate_artifact,
    build_counterfactual_experiment,
    candidate_command,
    reconcile_counterfactual_record,
)
from robot_vla.cli.replay_local_dagger_failures import (
    COLLECTION_FORMAT,
    POOL_FORMAT,
    _atomic_write_json,
    _atomic_write_jsonl,
    _read_json,
    _read_jsonl,
    _sha256_file,
    _validate_runtime_inputs,
)

SMOKE_FORMAT = "robot-vla-local-dagger-budget-counterfactual-smoke/v1"
SMOKE_PURPOSE = "segmented-budget counterfactual implementation smoke"
SMOKE_ORDER = (
    ("accepted_control", 30_111),
    ("timeout_early", 30_193),
    ("timeout_late", 30_171),
)

CONTROL_SEED = 30_111
CONTROL_FORMAL_RECORD_SHA256 = (
    "5022c7eebaa500f343ce48d8eba54d6bf57041fbf83990fe0c9185e23e963d6d"
)
CONTROL_TAKEOVER_STEP = 154
CONTROL_EXPERT_ACTIONS = 140
CONTROL_TOTAL_ACTIONS = 294
CONTROL_POLICY_REPLANS = 39

TIMEOUT_IDENTITIES = {
    30_193: {
        "role": "timeout_early",
        "formal_record_sha256": (
            "7da40aeea405e5a198221cb4ad2a8bd43cd8493497260983973000d8d4e8c472"
        ),
        "diagnostic_record_sha256": (
            "ab7c3f39de43f593b738fac1967891373458d3cc7e3f11b4228efae2ef010245"
        ),
        "expert_takeover_step": 123,
        "policy_replan_count": 31,
    },
    30_171: {
        "role": "timeout_late",
        "formal_record_sha256": (
            "be5f15dbc46f75f83e77a14f3cf999530210a35f4c1809eb7cee7e4dab289716"
        ),
        "diagnostic_record_sha256": (
            "a13b85d4e923e31667f1be35d78496e03a7c7a17b94ea020f6cf4405d4295988"
        ),
        "expert_takeover_step": 290,
        "policy_replan_count": 73,
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal", type=Path, required=True)
    parser.add_argument("--diagnostic-replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} 必须是 JSON object")
    return dict(value)


def _python_int_sampling_seeds(
    result: Mapping[str, Any],
    *,
    label: str,
) -> list[int]:
    values = result.get("policy_sampling_seeds")
    if not isinstance(values, list) or any(type(value) is not int for value in values):
        raise ValueError(f"{label} policy_sampling_seeds 必须全部是 Python int")
    traces = result.get("policy_replan_traces")
    if not isinstance(traces, list) or not all(isinstance(trace, dict) for trace in traces):
        raise ValueError(f"{label} policy_replan_traces 无效")
    trace_values = [trace.get("sampling_seed") for trace in traces]
    if any(type(value) is not int for value in trace_values):
        raise ValueError(f"{label} trace sampling_seed 必须全部是 Python int")
    if trace_values != values:
        raise ValueError(f"{label} sampling seeds 与 traces 不一致")
    return list(values)


def validate_control_reference(record: Mapping[str, Any]) -> dict[str, Any]:
    """验证历史 control 本身满足冻结 positive-control 契约。"""

    errors: list[str] = []
    if record.get("format") != COLLECTION_FORMAT:
        errors.append("formal control collection format drifted")
    if record.get("status") != "accepted":
        errors.append("formal control is not accepted")
    if record.get("eligible_for_risk_selection") is not True:
        errors.append("formal control is not risk-selection eligible")
    result = record.get("result")
    if not isinstance(result, dict):
        errors.append("formal control lacks result")
        result = {}
    trajectory = result.get("trajectory")
    if not isinstance(trajectory, dict):
        errors.append("formal control lacks trajectory")
        trajectory = {}
    provenance = trajectory.get("local_dagger")
    if not isinstance(provenance, dict):
        errors.append("formal control lacks Local DAgger provenance")
        provenance = {}
    takeover = provenance.get("expert_takeover_step")
    total_actions = trajectory.get("num_steps")
    if takeover != CONTROL_TAKEOVER_STEP:
        errors.append("formal control takeover step is not 154")
    if provenance.get("boundary_detection_step") != CONTROL_TAKEOVER_STEP:
        errors.append("formal control boundary detection step is not 154")
    if total_actions != CONTROL_TOTAL_ACTIONS:
        errors.append("formal control total actions is not 294")
    if (
        not isinstance(total_actions, int)
        or isinstance(total_actions, bool)
        or not isinstance(takeover, int)
        or isinstance(takeover, bool)
        or total_actions - takeover != CONTROL_EXPERT_ACTIONS
    ):
        errors.append("formal control Expert actions is not 140")
    if result.get("policy_replans") != CONTROL_POLICY_REPLANS:
        errors.append("formal control policy replans is not 39")

    try:
        _python_int_sampling_seeds(result, label="formal control")
        prefix = _result_prefix(record, seed=CONTROL_SEED)
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        prefix = None
        errors.append(f"formal control prefix invalid: {error}")
    if prefix is not None and prefix["policy_replan_count"] != CONTROL_POLICY_REPLANS:
        errors.append("formal control prefix replan count drifted")

    audit = record.get("audit")
    if not isinstance(audit, dict) or audit.get("trajectory_contract") != "passed":
        errors.append("formal control trajectory audit did not pass")
    paired = record.get("paired_clean_expert")
    if not isinstance(paired, dict) or paired.get("task_completed") is not True:
        errors.append("formal control paired clean Expert did not complete")
    snapshot = result.get("snapshot_round_trip")
    if not isinstance(snapshot, dict) or snapshot.get("passed") is not True:
        errors.append("formal control snapshot round-trip did not pass")
    outcome = trajectory.get("outcome_evidence")
    if not isinstance(outcome, dict) or outcome.get("task_completed") is not True:
        errors.append("formal control lacks full success evidence")
    if not isinstance(record.get("risk_components"), dict):
        errors.append("formal control lacks paired risk components")
    if errors:
        raise ValueError("; ".join(errors))

    return {
        "prefix": prefix,
        "timing": {
            "expert_takeover_step": CONTROL_TAKEOVER_STEP,
            "policy_actions": CONTROL_TAKEOVER_STEP,
            "expert_actions": CONTROL_EXPERT_ACTIONS,
            "total_actions": CONTROL_TOTAL_ACTIONS,
            "policy_replans": CONTROL_POLICY_REPLANS,
        },
        "audit": audit,
        "paired_clean_expert": paired,
        "risk_components": record["risk_components"],
        "snapshot_round_trip": snapshot,
        "boundary": result.get("boundary"),
        "local_dagger": provenance,
        "outcome_evidence": outcome,
    }


def _validate_control_formal_identity(
    formal_root: Path,
    formal_experiment: Mapping[str, Any],
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    candidates_path = formal_root / "collection_candidates.jsonl"
    rows = _read_jsonl(candidates_path, label="formal collection_candidates.jsonl")
    matches = [row for row in rows if row.get("environment_seed") == CONTROL_SEED]
    if len(matches) != 1:
        raise ValueError("formal manifest 必须精确包含一个 seed 30111 control")
    row = matches[0]
    record_path = (
        formal_root / "candidates" / f"seed-{CONTROL_SEED:06d}" / "record.json"
    )
    if row.get("boundary_type") != "grasp_lift" or row.get("status") != "accepted":
        raise ValueError("formal seed 30111 不是 accepted Grasp→Lift control")
    if row.get("failure") is not None:
        raise ValueError("formal accepted control 不得含 failure")
    if row.get("eligible_for_risk_selection") is not True:
        raise ValueError("formal control compact row 不是 eligible")
    row_record = row.get("record")
    if not isinstance(row_record, str) or Path(row_record).resolve() != record_path.resolve():
        raise ValueError("formal control record path 漂移")
    record = _read_json(record_path, label="formal seed 30111 record")
    actual_sha256 = _sha256_file(record_path)
    if actual_sha256 != CONTROL_FORMAL_RECORD_SHA256:
        raise ValueError("formal control record SHA256 与冻结值不一致")
    if record.get("source_revision") != formal_experiment.get("source_revision"):
        raise ValueError("formal control source revision 漂移")
    if record.get("base_dataset") != formal_experiment.get("base_dataset", {}).get(
        "path"
    ):
        raise ValueError("formal control base dataset 漂移")
    checkpoint = record.get("checkpoint")
    pool_checkpoint = formal_experiment.get("checkpoint")
    if not isinstance(checkpoint, dict) or not isinstance(pool_checkpoint, dict):
        raise TypeError("formal control checkpoint identity 缺失")
    for key in ("path", "sha256"):
        if checkpoint.get(key) != pool_checkpoint.get(key):
            raise ValueError(f"formal control checkpoint {key} 漂移")
    config = record.get("config")
    pool_config = formal_experiment.get("config")
    if not isinstance(config, dict) or not isinstance(pool_config, dict):
        raise TypeError("formal control config 缺失")
    if config.get("environment_seed") != CONTROL_SEED:
        raise ValueError("formal control environment seed 漂移")
    if row.get("episode_sampling_seed") != config.get("episode_sampling_seed"):
        raise ValueError("formal control episode sampling seed 漂移")
    for key in (
        "qwen_context_layer",
        "sampling_seed_base",
        "num_flow_steps",
        "recency_decay",
        "max_anomaly_replans",
        "snapshot_round_trip_required",
        "paired_clean_expert_required",
    ):
        if config.get(key) != pool_config.get(key):
            raise ValueError(f"formal control config {key} 与 pool 漂移")
    return record, record_path, validate_control_reference(record)


def build_smoke_experiment(
    formal_root: Path,
    diagnostic_replay_root: Path,
    *,
    smoke_source_revision: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """构建固定 control→early→late 身份；不改变正式 16-target builder。"""

    formal_root = formal_root.resolve()
    diagnostic_replay_root = diagnostic_replay_root.resolve()
    counterfactual_identity, formal_targets = build_counterfactual_experiment(
        formal_root,
        diagnostic_replay_root,
        counterfactual_source_revision=smoke_source_revision,
    )
    if counterfactual_identity.get("selected_count") != 16:
        raise RuntimeError("正式 counterfactual target contract 不再是 16 条")
    by_seed = {int(target["environment_seed"]): target for target in formal_targets}
    for seed, expected in TIMEOUT_IDENTITIES.items():
        target = by_seed.get(seed)
        if target is None:
            raise ValueError(f"fixed smoke timeout seed {seed} 不在正式 16 targets")
        if target.get("original_record_sha256") != expected[
            "formal_record_sha256"
        ]:
            raise ValueError(f"seed {seed}: formal record SHA256 漂移")
        if target.get("diagnostic_record_sha256") != expected[
            "diagnostic_record_sha256"
        ]:
            raise ValueError(f"seed {seed}: diagnostic record SHA256 漂移")
        prefix = target.get("reference_prefix")
        if not isinstance(prefix, dict):
            raise TypeError(f"seed {seed}: diagnostic reference prefix 缺失")
        if prefix.get("expert_takeover_step") != expected["expert_takeover_step"]:
            raise ValueError(f"seed {seed}: frozen takeover step 漂移")
        if prefix.get("policy_replan_count") != expected["policy_replan_count"]:
            raise ValueError(f"seed {seed}: frozen replan count 漂移")
        if any(type(value) is not int for value in prefix["policy_sampling_seeds"]):
            raise ValueError(f"seed {seed}: diagnostic sampling seeds 不是 Python int")

    formal_experiment = _read_json(
        formal_root / "experiment.json",
        label="formal experiment.json",
    )
    if formal_experiment.get("format") != POOL_FORMAT:
        raise ValueError("formal experiment format 不兼容")
    control_record, control_path, control_evidence = _validate_control_formal_identity(
        formal_root,
        formal_experiment,
    )
    control_target = {
        "environment_seed": CONTROL_SEED,
        "boundary_type": "grasp_lift",
        "model_cache": formal_experiment["model_cache"],
        "replay_source_revision": smoke_source_revision,
        "original_record": control_record,
        "original_record_path": str(control_path.resolve()),
        "original_record_sha256": CONTROL_FORMAL_RECORD_SHA256,
        "reference_prefix": control_evidence["prefix"],
        "control_reference": control_evidence,
    }
    targets_by_role = {
        "accepted_control": control_target,
        "timeout_early": by_seed[30_193],
        "timeout_late": by_seed[30_171],
    }
    targets: list[dict[str, Any]] = []
    frozen_targets: list[dict[str, Any]] = []
    for role, seed in SMOKE_ORDER:
        target = dict(targets_by_role[role])
        if target["environment_seed"] != seed:
            raise RuntimeError("smoke role/seed 固定顺序漂移")
        target["smoke_role"] = role
        targets.append(target)
        frozen_targets.append(
            {
                "role": role,
                "environment_seed": seed,
                "formal_record": target["original_record_path"],
                "formal_record_sha256": target["original_record_sha256"],
                "diagnostic_record": target.get("diagnostic_record_path"),
                "diagnostic_record_sha256": target.get(
                    "diagnostic_record_sha256"
                ),
                "reference_prefix": target["reference_prefix"],
            }
        )

    experiment = {
        "format": SMOKE_FORMAT,
        "purpose": SMOKE_PURPOSE,
        "trajectory_usage": TRAJECTORY_USAGE,
        "successful_npz_may_enter_d1": False,
        "source_revision": smoke_source_revision,
        "action_budget_protocol": counterfactual_identity[
            "action_budget_protocol"
        ],
        "formal_input": counterfactual_identity["formal_input"],
        "diagnostic_replay_input": counterfactual_identity[
            "diagnostic_replay_input"
        ],
        "formal_counterfactual_target_contract": {
            "selected_count": counterfactual_identity["selected_count"],
            "environment_seeds": counterfactual_identity["environment_seeds"],
            "unchanged_by_smoke": True,
        },
        "execution_order": [role for role, _ in SMOKE_ORDER],
        "environment_seeds": [seed for _, seed in SMOKE_ORDER],
        "selected_count": len(SMOKE_ORDER),
        "selected_candidates": frozen_targets,
        "control_contract": control_evidence,
        "execution": {
            "entrypoint": "robot_vla.cli.collect_local_dagger",
            "only_command_delta": [
                "--action-budget-protocol",
                COUNTERFACTUAL_PROTOCOL,
            ],
            "resume_contract": "completed candidate directories are immutable",
            "trajectory_usage": TRAJECTORY_USAGE,
            "successful_npz_may_enter_d1": False,
        },
    }
    normalized = json.loads(json.dumps(experiment, sort_keys=True, allow_nan=False))
    return normalized, targets


def validate_smoke_outcome(
    role: str,
    target: Mapping[str, Any],
    record: Mapping[str, Any],
    counterfactual_row: Mapping[str, Any],
) -> dict[str, Any]:
    """将通用 counterfactual 结果收紧为 smoke role-specific gate。"""

    violations: list[str] = []
    classification = counterfactual_row.get("classification")
    if classification == "engineering_error":
        violations.append("counterfactual engineering_error")
    if classification == "prefix_mismatch":
        violations.append("counterfactual prefix_mismatch")
    if role == "accepted_control":
        if classification != "recovered_full_eligible":
            violations.append("control did not recover as full eligible accepted")
        if record.get("status") != "accepted":
            violations.append("control record status is not accepted")
        usage = record.get("action_budget_usage")
        expected_usage = {
            "policy_actions": CONTROL_TAKEOVER_STEP,
            "expert_actions": CONTROL_EXPERT_ACTIONS,
            "total_actions": CONTROL_TOTAL_ACTIONS,
        }
        if usage != expected_usage:
            violations.append("control action-budget usage is not 154/140/294")
        result = record.get("result")
        if not isinstance(result, dict):
            violations.append("control result missing")
            result = {}
        trajectory = result.get("trajectory")
        if not isinstance(trajectory, dict):
            violations.append("control trajectory missing")
            trajectory = {}
        if result.get("policy_replans") != CONTROL_POLICY_REPLANS:
            violations.append("control policy replans is not 39")
        artifact_audit = counterfactual_row.get("artifact_audit")
        if not isinstance(artifact_audit, dict) or artifact_audit.get(
            "passed"
        ) is not True:
            violations.append("control full trajectory artifact audit did not pass")
        try:
            _python_int_sampling_seeds(result, label="counterfactual control")
        except ValueError as error:
            violations.append(str(error))
        reference = target.get("control_reference")
        if not isinstance(reference, dict):
            violations.append("control reference missing")
            reference = {}
        exact_fields = {
            "audit": record.get("audit"),
            "paired_clean_expert": record.get("paired_clean_expert"),
            "risk_components": record.get("risk_components"),
            "snapshot_round_trip": result.get("snapshot_round_trip"),
            "boundary": result.get("boundary"),
            "local_dagger": trajectory.get("local_dagger"),
            "outcome_evidence": trajectory.get("outcome_evidence"),
        }
        for field, observed in exact_fields.items():
            if observed != reference.get(field):
                violations.append(f"control {field} differs from frozen formal control")
        snapshot = result.get("snapshot_round_trip")
        if not isinstance(snapshot, dict) or snapshot.get("passed") is not True:
            violations.append("control snapshot round-trip did not pass")
        paired = record.get("paired_clean_expert")
        if not isinstance(paired, dict) or paired.get("task_completed") is not True:
            violations.append("control paired clean Expert did not complete")
        audit = record.get("audit")
        if not isinstance(audit, dict) or audit.get("trajectory_contract") != "passed":
            violations.append("control trajectory audit did not pass")
    elif role not in {"timeout_early", "timeout_late"}:
        violations.append(f"unknown smoke role {role!r}")
    elif classification not in set(CLASSIFICATIONS) - {
        "engineering_error",
        "prefix_mismatch",
    }:
        violations.append("timeout smoke result is not a recognized behavioral outcome")

    return {
        "role": role,
        "environment_seed": target["environment_seed"],
        "passed": not violations,
        "violations": violations,
        "counterfactual_classification": classification,
        "prefix_aligned": bool(
            (counterfactual_row.get("prefix_alignment") or {}).get("aligned")
        ),
        "trajectory_usage": TRAJECTORY_USAGE,
        "successful_npz_may_enter_d1": False,
    }


def summarize_smoke(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected_roles = [role for role, _ in SMOKE_ORDER]
    observed_roles = [str(row.get("smoke_role")) for row in rows]
    validations = [row.get("smoke_validation") for row in rows]
    valid_objects = all(isinstance(value, dict) for value in validations)
    all_passed = valid_objects and all(value.get("passed") is True for value in validations)
    scan_complete = len(rows) == len(SMOKE_ORDER) and observed_roles == expected_roles
    counts = Counter(str(row.get("classification")) for row in rows)
    return {
        "format": SMOKE_FORMAT,
        "purpose": SMOKE_PURPOSE,
        "trajectory_usage": TRAJECTORY_USAGE,
        "successful_npz_may_enter_d1": False,
        "complete": scan_complete,
        "scan_complete": scan_complete,
        "passed": scan_complete and all_passed,
        "blocked": any(
            value.get("passed") is not True
            for value in validations
            if isinstance(value, dict)
        )
        or not valid_objects,
        "expected_order": expected_roles,
        "observed_order": observed_roles,
        "expected_candidates": len(SMOKE_ORDER),
        "completed_candidates": len(rows),
        "classification_counts": dict(sorted(counts.items())),
        "classification_additivity": {
            "classified_candidates": sum(counts.values()),
            "completed_candidates": len(rows),
            "holds": sum(counts.values()) == len(rows),
        },
        "role_results": validations,
    }


def _write_progress(output: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _atomic_write_jsonl(output / "smoke_candidates.jsonl", rows)
    _atomic_write_json(output / "summary.json", summarize_smoke(rows))


def _normalized_subprocess_command(
    command: Any,
    *,
    candidate_dir: Path,
) -> list[str]:
    if not isinstance(command, list) or not all(isinstance(value, str) for value in command):
        return []
    normalized = list(command)
    for flag, suffix in (("--output", "dataset"), ("--record", "record.json")):
        if flag not in normalized or normalized.index(flag) + 1 >= len(normalized):
            return []
        normalized[normalized.index(flag) + 1] = str(
            (candidate_dir / suffix).resolve()
        )
    return normalized


def run(args: argparse.Namespace) -> None:
    from robot_vla.cli.train_stage1 import compute_source_revision

    formal_root = args.formal.resolve()
    diagnostic_replay_root = args.diagnostic_replay.resolve()
    output = args.output.resolve()
    for source_root, label in (
        (formal_root, "formal"),
        (diagnostic_replay_root, "diagnostic replay"),
    ):
        if output == source_root or output.is_relative_to(source_root):
            raise ValueError(f"smoke output 必须独立于 {label}")

    project_root = Path(__file__).resolve().parents[3]
    experiment, targets = build_smoke_experiment(
        formal_root,
        diagnostic_replay_root,
        smoke_source_revision=compute_source_revision(project_root),
    )
    _validate_runtime_inputs(targets)
    _prepare_output(output, experiment, resume=args.resume)

    rows: list[dict[str, Any]] = []
    for target in targets:
        role = str(target["smoke_role"])
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
                result_filename="smoke.json",
                allow_missing_record=True,
            )
            if row is None:
                row = _materialization_error_row(target, error)
                row["smoke_role"] = role
                row["smoke_validation"] = validate_smoke_outcome(
                    role,
                    target,
                    {},
                    row,
                )
                row = _finalize_candidate(
                    candidate_dir,
                    target,
                    row,
                    result_filename="smoke.json",
                    allow_missing_record=True,
                )
            rows.append(row)
            _write_progress(output, rows)
            raise RuntimeError(
                f"smoke {role}/seed {seed}: subprocess 未产生 record；"
                "已写 engineering row、blocked summary 与 immutable receipt"
            ) from error

        row = _load_finalized_candidate(
            candidate_dir,
            target,
            result_filename="smoke.json",
        )
        if row is not None:
            rows.append(row)
            _write_progress(output, rows)
            validation = row.get("smoke_validation")
            if not isinstance(validation, dict) or validation.get("passed") is not True:
                raise RuntimeError(f"immutable smoke {role}/seed {seed} 未通过")
            continue
        record_path = candidate_dir / "record.json"
        record = _read_json(record_path, label=f"smoke {role} record")
        subprocess_record = _read_json(
            candidate_dir / "subprocess.json",
            label=f"smoke {role} subprocess",
        )
        expected_command = candidate_command(target, candidate_dir=candidate_dir)
        normalized_command = _normalized_subprocess_command(
            subprocess_record.get("command"),
            candidate_dir=candidate_dir,
        )
        returncode = subprocess_record.get("returncode")
        valid_subprocess = (
            subprocess_record.get("format") == COUNTERFACTUAL_SUBPROCESS_FORMAT
            and isinstance(returncode, int)
            and not isinstance(returncode, bool)
        )
        if not valid_subprocess:
            returncode = -999_999
        row = reconcile_counterfactual_record(
            target,
            record,
            record_path=record_path,
            subprocess_returncode=returncode,
            subprocess_command=normalized_command,
            expected_command=expected_command,
        )
        if not valid_subprocess:
            row["engineering_errors"].append(
                "subprocess record format/returncode invalid"
            )
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
        validation = validate_smoke_outcome(role, target, record, row)
        row["smoke_role"] = role
        row["smoke_validation"] = validation
        row = _finalize_candidate(
            candidate_dir,
            target,
            row,
            result_filename="smoke.json",
        )
        rows.append(row)
        _write_progress(output, rows)
        print(
            json.dumps(
                {
                    "event": "budget_counterfactual_smoke_complete",
                    "role": role,
                    "seed": seed,
                    "classification": row["classification"],
                    "passed": validation["passed"],
                    "completed": len(rows),
                    "expected": len(SMOKE_ORDER),
                },
                sort_keys=True,
                allow_nan=False,
            ),
            flush=True,
        )
        if validation["passed"] is not True:
            raise RuntimeError(
                f"smoke {role}/seed {seed} 未通过：{validation['violations']}"
            )

    summary = summarize_smoke(rows)
    if summary["passed"] is not True:
        raise RuntimeError("segmented-budget smoke 未完整通过")


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
