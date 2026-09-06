"""E018-P1 Stage 2A fixed-gain evaluation 的隔离执行与验证。

Pass A 是唯一允许加载 checkpoint/provider 的进程，只冻结 deployable 路线和
gain=0.10 决策，禁止读取 simulator object GT。Pass B 必须在新的 OS 进程中，
先以 O_EXCL+fsync 消费当前 identity，再做 deterministic simulator replay；
replay 与 Pass A 的动作前缀、RGB、相机 pose 和 model input 完全一致后，才读取
三个 COLLECT frame 的 object-only label。全部路径均为 offline/no-actuation。
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import time
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robot_vla.precision import e018_p1_stage2a as _stage2a
from robot_vla.precision import (
    e018_p1_stage2a_selection_runtime as _selection_runtime,
)
from robot_vla.precision.calibrated_front_provider import canonical_sha256
from robot_vla.precision.e018_p1_g2a import file_sha256
from robot_vla.precision.e018_p1_g2c_qualification import (
    _AppendOnlyJsonl,
    _atomic_create_json,
    assert_qualification_prediction_deployable_only,
    build_qualification_deployable_capture,
    load_g2c_dynamic_qualification_config,
)
from robot_vla.precision.e018_p1_g2c_training import _git_source_identity
from robot_vla.precision.e018_p1_stage2a import (
    E018_P1_STAGE2A_EXECUTION_VERSION,
    E018_P1_STAGE2A_SELECTED_GAIN_EVALUATION_EXPERIMENT_ID,
    E018_P1_STAGE2A_SELECTED_GAIN_EVALUATION_PREFLIGHT_EXPERIMENT_ID,
    STAGE2A_COLLECT_FRAME_INDICES,
    STAGE2A_PROVIDER_FRAME_INDICES,
    STAGE2A_SELECTED_GAIN_EVALUATION_PREFLIGHT_SEED,
    STAGE2A_SELECTED_GAIN_EVALUATION_SEEDS,
    Stage2AExecutionProgress,
    Stage2AProviderOutputRecord,
    Stage2ARouteTransaction,
    load_e018_p1_stage2a_config,
    verify_stage2a_provider_output_record,
)
from robot_vla.precision.e018_p1_stage2a_evaluation import (
    E018_P1_STAGE2A_EVALUATION_EXECUTION_VERSION,
    E018_P1_STAGE2A_EVALUATION_EXPERIMENT_ID,
    E018_P1_STAGE2A_EVALUATION_RESULT_VERSION,
    STAGE2A_EVALUATION_GO,
    STAGE2A_EVALUATION_PREFLIGHT_GO,
    STAGE2A_EVALUATION_SEEDS,
    STAGE2A_EVALUATION_SELECTED_GAIN,
    CapturedStage2AEvaluationRoute,
    _validated_evaluation_branch,
    load_e018_p1_stage2a_evaluation_config,
    replay_selected_gain_branch,
    score_selected_gain_evaluation,
    validate_evaluation_private_label,
)
from robot_vla.precision.e018_p1_stage2a_selection import (
    _array_sha256,
    _read_json,
    _read_jsonl,
    _require_exact_keys,
    _verify_internal_digest,
)

_FORMAL_CLASSIFICATION = (
    "formal-development-evaluation-no-test-no-actuation/v2"
)
_FORMAL_CAPTURE_CLASSIFICATION = (
    "formal-development-selected-gain-capture-only-no-test-no-actuation/v2"
)
_PREFLIGHT_CLASSIFICATION = (
    "engineering-recovery-preflight-selected-gain-evaluation-no-formal-claim/v2"
)
_PREFLIGHT_CAPTURE_CLASSIFICATION = (
    "engineering-preflight-selected-gain-capture-only-no-test-no-actuation/v2"
)
_FORMAL_EXECUTION_GO_VERSION = (
    "e018-p1-stage2a-d049-final-formal-go-receipt/v2"
)
_EVALUATION_ARTIFACT_ROLE_IDENTITY_VERSION = (
    "e018-p1-stage2a-evaluation-artifact-role/v1"
)
_WORKER_ARTIFACT_ROOT_IDENTITY_VERSION = (
    "e018-p1-stage2a-worker-artifact-root/v1"
)
_PARENT_AUTHORIZATION_VERSION = (
    "e018-p1-stage2a-evaluation-parent-authorization/v1"
)

_PUBLIC_FILES = {
    "RUN_STARTED.json",
    "config_snapshot.json",
    "source_identity.json",
    "parent_verification.json",
    "camera_pose_ledger.jsonl",
    "provider_output_ledger.jsonl",
    "route_evidence_ledger.jsonl",
    "fixed_gain_branch_ledger.jsonl",
    "execution_freeze.json",
    "execution_receipt.json",
    "PUBLIC_EXECUTION_COMPLETE.json",
}
_PUBLIC_PRECOMPLETION_FILES = _PUBLIC_FILES - {"PUBLIC_EXECUTION_COMPLETE.json"}
_RESULT_FILES = {
    "scored_fixed_gain_routes.jsonl",
    "evaluation_summary.json",
    "result_receipt.json",
    "RESULT_COMPLETE.json",
}
_RESULT_PRECOMPLETION_FILES = _RESULT_FILES - {"RESULT_COMPLETE.json"}


@dataclass(frozen=True)
class _EvaluationMode:
    preflight: bool
    experiment_id: str
    classification: str
    capture_classification: str
    seeds: tuple[int, ...]
    go_token: str

    @property
    def label_count(self) -> int:
        return len(self.seeds) * 3

    @property
    def prediction_count(self) -> int:
        return len(self.seeds) * 4


def _evaluation_mode(preflight: bool) -> _EvaluationMode:
    if type(preflight) is not bool:
        raise TypeError("evaluation preflight 必须是 exact bool")
    if preflight:
        return _EvaluationMode(
            preflight=True,
            experiment_id=(
                E018_P1_STAGE2A_SELECTED_GAIN_EVALUATION_PREFLIGHT_EXPERIMENT_ID
            ),
            classification=_PREFLIGHT_CLASSIFICATION,
            capture_classification=_PREFLIGHT_CAPTURE_CLASSIFICATION,
            seeds=(STAGE2A_SELECTED_GAIN_EVALUATION_PREFLIGHT_SEED,),
            go_token=STAGE2A_EVALUATION_PREFLIGHT_GO,
        )
    return _EvaluationMode(
        preflight=False,
        experiment_id=E018_P1_STAGE2A_SELECTED_GAIN_EVALUATION_EXPERIMENT_ID,
        classification=_FORMAL_CLASSIFICATION,
        capture_classification=_FORMAL_CAPTURE_CLASSIFICATION,
        seeds=STAGE2A_SELECTED_GAIN_EVALUATION_SEEDS,
        go_token=STAGE2A_EVALUATION_GO,
    )


def _evaluation_artifact_role_identity_sha256(
    *,
    role: str,
    mode: _EvaluationMode,
    config_canonical_sha256: str,
    source_identity_sha256: str,
    conditional_parent_verification_sha256: str,
) -> str:
    """生成 evaluation 专用、formal/preflight 隔离的 role identity。"""

    if role not in {"public_execution", "private_labels", "result"}:
        raise ValueError("evaluation artifact role 非法")
    if any(
        not _selection_runtime._is_sha256(value)
        for value in (
            config_canonical_sha256,
            source_identity_sha256,
            conditional_parent_verification_sha256,
        )
    ):
        raise ValueError("evaluation artifact role parent identity 非法")
    return canonical_sha256(
        {
            "version": _EVALUATION_ARTIFACT_ROLE_IDENTITY_VERSION,
            "execution_version": E018_P1_STAGE2A_EVALUATION_EXECUTION_VERSION,
            "result_version": E018_P1_STAGE2A_EVALUATION_RESULT_VERSION,
            "experiment_id": mode.experiment_id,
            "preflight": mode.preflight,
            "role": role,
            "config_canonical_sha256": config_canonical_sha256,
            "source_identity_sha256": source_identity_sha256,
            "conditional_parent_verification_sha256": (
                conditional_parent_verification_sha256
            ),
        }
    )


def _worker_artifact_root_identity_sha256(path: str | Path) -> str:
    """绑定 worker 上唯一 formal run root；不进入可复制 artifact role。"""

    resolved = Path(path).resolve(strict=False)
    return canonical_sha256(
        {
            "version": _WORKER_ARTIFACT_ROOT_IDENTITY_VERSION,
            "resolved_absolute_artifact_root": str(resolved),
        }
    )


def _canonical_config_sha256(path: Path) -> str:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise TypeError(f"config 必须是 JSON object: {path.name}")
    return canonical_sha256(value)


def _verify_gate_record(
    *,
    gate_path: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if gate_path.is_symlink() or not gate_path.is_file():
        raise RuntimeError("D049 conditional evaluation Gate 必须是真实文件")
    if file_sha256(gate_path) != config["experiment"]["gate_record_raw_sha256"]:
        raise RuntimeError("D049 conditional evaluation Gate raw SHA 漂移")
    gate = _read_json(gate_path, "D049 conditional evaluation Gate")
    supersedes = gate.get("supersedes")
    recovery = gate.get("recovery")
    selection = gate.get("selection_parent")
    inputs = gate.get("frozen_inputs")
    preflight = gate.get("preflight")
    formal = gate.get("formal_split")
    phase = gate.get("phase_boundary")
    oracle = gate.get("oracle_and_gate")
    routing = gate.get("outcome_routing")
    permissions = gate.get("permissions")
    continuation = gate.get("continuation")
    persistence = config["selection_parent"]["persistence"]
    if (
        gate.get("version")
        != "e018-p1-d049-conditional-evaluation-recovery-gate/v3"
        or gate.get("status")
        != (
            "implementation-recovery-go-formal-hold-until-final-source-r2-"
            "and-recovery-preflight-scope-amended"
        )
        or gate.get("authority")
        != "user-authorized-b-level-offline-no-actuation-decision-agent"
        or not isinstance(supersedes, dict)
        or not isinstance(recovery, dict)
        or not isinstance(selection, dict)
        or not isinstance(inputs, dict)
        or not isinstance(preflight, dict)
        or not isinstance(formal, dict)
        or not isinstance(phase, dict)
        or not isinstance(oracle, dict)
        or not isinstance(routing, dict)
        or not isinstance(permissions, dict)
        or not isinstance(continuation, dict)
        or supersedes.get("conditional_gate_v1_raw_sha256")
        != "cf64fe03e706b578fc3c8e86ea2697e5147c7cf10d409b93098448aa573a8845"
        or supersedes.get("scope_amendment_raw_sha256")
        != "04b688c1da004fe93463465bd3e9bf331ba8c9e6e9c2f3f4100e15756514a3af"
        or supersedes.get("scope_amendment_internal_sha256")
        != "9acaa2554a8863f393f4d9609c5e95a8eabcd2450fa3e99a6e0a334a97815328"
        or supersedes.get("historical_records_mutated") is not False
        or recovery.get("failed_preflight_seed") != 76892
        or recovery.get("failed_preflight_rerun_allowed") is not False
        or recovery.get("allowed_code_change")
        != (
            "call-the-existing-shared-viewpoint-normalizer-before-pass-b-"
            "frame-binding"
        )
        or recovery.get("replay_hash_or_tolerance_relaxation_allowed")
        is not False
        or recovery.get("research_variable_change") is not False
        or recovery.get("formal_v1_consumed") is not False
        or gate.get("experiment")
        != {
            "id": E018_P1_STAGE2A_EVALUATION_EXPERIMENT_ID,
            "config_version": config["version"],
            "gate": "D049-R2-RECOVERY-SCOPE-AMENDED",
            "classification": _FORMAL_CLASSIFICATION,
            "exact_go_token": STAGE2A_EVALUATION_GO,
            "same_identity_rerun_allowed": False,
        }
        or selection.get("artifact_id")
        != config["selection_parent"]["artifact_id"]
        or selection.get("result_verification_sha256")
        != config["selection_parent"]["result_verification_sha256"]
        or selection.get("selection_summary_sha256")
        != config["selection_parent"]["selection_summary_sha256"]
        or selection.get("selected_gain") != STAGE2A_EVALUATION_SELECTED_GAIN
        or selection.get("selection_reason")
        != config["selection_parent"]["selection_reason"]
        or selection.get("replication_state") != "REPLICATED"
        or selection.get("artifact_manifest_raw_sha256")
        != persistence["manifest_raw_sha256"]
        or selection.get("artifact_manifest_internal_sha256")
        != persistence["manifest_internal_sha256"]
        or selection.get("artifact_inventory_sha256")
        != persistence["artifact_inventory_sha256"]
        or selection.get("drive_marker_raw_sha256")
        != persistence["drive_marker_raw_sha256"]
        or selection.get("drive_marker_internal_sha256")
        != persistence["drive_marker_internal_sha256"]
        or selection.get("drive_persistence_verification_sha256")
        != persistence["drive_verification_internal_sha256"]
        or selection.get("local_canonical_verification_sha256")
        != persistence["local_verification_internal_sha256"]
        or selection.get("inventory_record_canonical_sha256")
        != persistence["inventory_record_canonical_sha256"]
        or inputs.get("selected_min_information_gain")
        != STAGE2A_EVALUATION_SELECTED_GAIN
        or inputs.get("provider_write_threshold")
        != config["stage2a_parent"]["provider_write_threshold"]
        or inputs.get("checkpoint_sha256")
        != config["stage2a_parent"]["checkpoint_sha256"]
        or inputs.get("gain_reselection_allowed") is not False
        or inputs.get("checkpoint_change_allowed") is not False
        or inputs.get("threshold_change_allowed") is not False
        or inputs.get("oracle_or_taxonomy_change_allowed") is not False
        or preflight.get("experiment_id")
        != config["preflight"]["experiment_id"]
        or preflight.get("exact_go_token")
        != config["preflight"]["exact_go_token"]
        or preflight.get("seed") != config["preflight"]["seed"]
        or preflight.get("formal_identity_consumed") is not False
        or preflight.get("formal_split_consumed") is not False
        or (formal.get("seed_start"), formal.get("seed_end"))
        != (STAGE2A_EVALUATION_SEEDS[0], STAGE2A_EVALUATION_SEEDS[-1])
        or formal.get("execution_order")
        != config["split"]["execution_order"]
        or phase.get("pass_a_private_label_capture_count") != 0
        or phase.get("pass_a_private_label_open_count") != 0
        or phase.get("pass_a_runtime_object_gt_read_count") != 0
        or phase.get("pass_b_checkpoint_load_count") != 0
        or phase.get("pass_b_provider_forward_count") != 0
        or phase.get("pass_b_decision_change_count") != 0
        or phase.get("consumption_marker_before_first_gt_read") is not True
        or phase.get("pass_b_viewpoint_normalization")
        != "same-shared-helper-before-frame-binding-and-hash"
        or oracle.get("exact_recovery_comparison")
        != "10*recovered_count >= 7*common_denominator_count"
        or routing.get("substantive_safety_failure")
        != "safety-negative-persist-publish-pause-for-reusability-refactor"
        or routing.get("stage2b_continuation_required_for_every_complete_outcome")
        is not False
        or routing.get("stage2b_execution_authorized") is not False
        or routing.get("d050_execution_authorized") is not False
        or routing.get("stage3_execution_authorized") is not False
        or routing.get("post_d049_endpoint")
        != (
            "complete-D049-v2-then-persist-publish-and-enter-"
            "PAUSE_FOR_REUSABILITY_REFACTOR-no-Stage2B-D050-Stage3-execution"
        )
        or permissions.get("fresh_test_reads") != 0
        or permissions.get("canonical_runtime_mutation") != 0
        or permissions.get("physical_camera_actuation") != 0
        or permissions.get("arm_tcp_actuation") != 0
        or permissions.get("gripper_close") != 0
        or permissions.get("manipulation_progression") != 0
        or permissions.get("stage2b_execution") != 0
        or permissions.get("d050_execution") != 0
        or permissions.get("stage3_execution") != 0
        or continuation.get("policy")
        != (
            "complete-D049-v2-then-persist-publish-and-enter-"
            "PAUSE_FOR_REUSABILITY_REFACTOR-no-Stage2B-D050-Stage3-execution"
        )
        or continuation.get("d049_runner_failure_recovery_required_until_complete")
        is not True
        or continuation.get("pause_before_reusability_refactor") is not True
        or continuation.get("gpu_release_owner") != "user"
    ):
        raise RuntimeError("D049 conditional evaluation Gate identity/语义漂移")
    return {
        "version": gate["version"],
        "status": "verified-recovery-implementation-go-formal-runtime-hold",
        "gate_raw_sha256": file_sha256(gate_path),
        "formal_execution_requires_final_go": True,
    }


def _validate_formal_execution_go_receipt(
    receipt: Mapping[str, Any],
    *,
    loaded: Any,
    source: Mapping[str, Any],
    conditional_parent_verification_sha256: str,
    expected_execution_id: str | None,
    expected_worker_artifact_root: str | Path | None = None,
    preflight_public: Mapping[str, Any] | None = None,
    preflight_result: Mapping[str, Any] | None = None,
    preflight_public_completion_raw_sha256: str | None = None,
    preflight_result_completion_raw_sha256: str | None = None,
    preflight_completed_at_unix_ns: int | None = None,
) -> str:
    """验证 D049 final-GO exact contract，并返回 receipt internal SHA。"""

    value = _require_exact_keys(
        dict(receipt),
        {
            "version",
            "decision_id",
            "status",
            "authority",
            "conditional_gate",
            "evaluation_config",
            "source",
            "code_validation",
            "selection_parent",
            "preflight",
            "formal_execution",
            "permissions",
            "continuation",
            "issued_at_unix_ns",
            "receipt_sha256",
        },
        "D049 final formal GO receipt",
    )
    conditional_gate = _require_exact_keys(
        value["conditional_gate"],
        {"filename", "raw_sha256", "status"},
        "D049 final GO conditional gate",
    )
    evaluation_config = _require_exact_keys(
        value["evaluation_config"],
        {"path", "version", "raw_sha256", "canonical_sha256"},
        "D049 final GO evaluation config",
    )
    receipt_source = _require_exact_keys(
        value["source"],
        {
            "git_commit",
            "source_tree_sha256",
            "identity_sha256",
            "github_remote_branch_commit",
            "worker_checkout_commit",
            "worker_checkout_exact_clean",
        },
        "D049 final GO source",
    )
    code_validation = _require_exact_keys(
        value["code_validation"],
        {
            "targeted_tests_exit_code",
            "targeted_tests_receipt_sha256",
            "r2_sample_verification_sha256",
            "formal_seed_read_count_before_issue",
        },
        "D049 final GO code validation",
    )
    selection_parent = _require_exact_keys(
        value["selection_parent"],
        {
            "artifact_id",
            "replication_state",
            "inventory_record_canonical_sha256",
        },
        "D049 final GO selection parent",
    )
    preflight = _require_exact_keys(
        value["preflight"],
        {
            "experiment_id",
            "seed",
            "transaction_identity_sha256",
            "public_output_identity_sha256",
            "private_output_identity_sha256",
            "result_output_identity_sha256",
            "public_verification_sha256",
            "public_completion_marker_raw_sha256",
            "public_completion_marker_internal_sha256",
            "result_verification_sha256",
            "result_completion_marker_raw_sha256",
            "result_completion_marker_internal_sha256",
            "two_pass_verification_passed",
            "process_boundary_verified",
            "provider_prediction_count",
            "private_label_count",
            "formal_identity_consumed",
            "formal_split_consumed",
            "fresh_test_reads",
            "physical_camera_actuation",
            "arm_tcp_actuation",
            "gripper_close",
        },
        "D049 final GO preflight",
    )
    formal = _require_exact_keys(
        value["formal_execution"],
        {
            "experiment_id",
            "execution_id",
            "classification",
            "exact_go_token",
            "seed_start",
            "seed_end",
            "seed_count",
            "execution_order",
            "capture_attempt_count",
            "scoring_attempt_count",
            "same_identity_rerun_allowed",
            "conditional_parent_verification_sha256",
            "artifact_role_identity_version",
            "worker_artifact_root_identity_version",
            "worker_artifact_root_identity_sha256",
            "worker_artifact_root_basename",
            "public_output_identity_sha256",
            "private_output_identity_sha256",
            "result_output_identity_sha256",
            "formal_output_roots_absent_before_issue",
            "formal_consumption_marker_absent_before_issue",
        },
        "D049 final GO formal execution",
    )
    permissions = _require_exact_keys(
        value["permissions"],
        {
            "fresh_test_reads",
            "canonical_runtime_mutation",
            "physical_camera_actuation",
            "arm_tcp_actuation",
            "gripper_close",
            "manipulation_progression",
            "checkpoint_writes",
        },
        "D049 final GO permissions",
    )
    continuation = _require_exact_keys(
        value["continuation"],
        {
            "stage2b_required_for_every_complete_outcome",
            "integrity_failure_policy",
            "substantive_negative_result_policy",
        },
        "D049 final GO continuation",
    )
    unsigned = dict(value)
    internal = unsigned.pop("receipt_sha256", None)
    if (
        set(source) != {"git_commit", "source_tree_sha256", "identity_sha256"}
        or not isinstance(source["git_commit"], str)
        or len(source["git_commit"]) != 40
        or any(character not in "0123456789abcdef" for character in source["git_commit"])
        or not _selection_runtime._is_sha256(source["source_tree_sha256"])
        or source["identity_sha256"]
        != canonical_sha256(
            {
                "git_commit": source["git_commit"],
                "source_tree_sha256": source["source_tree_sha256"],
            }
        )
    ):
        raise RuntimeError("D049 final GO source identity 非法")
    preflight_mode = _evaluation_mode(True)
    formal_mode = _evaluation_mode(False)
    preflight_roles = {
        role: _evaluation_artifact_role_identity_sha256(
            role=role,
            mode=preflight_mode,
            config_canonical_sha256=loaded.canonical_sha256,
            source_identity_sha256=source["identity_sha256"],
            conditional_parent_verification_sha256=(
                conditional_parent_verification_sha256
            ),
        )
        for role in ("public_execution", "private_labels", "result")
    }
    formal_roles = {
        role: _evaluation_artifact_role_identity_sha256(
            role=role,
            mode=formal_mode,
            config_canonical_sha256=loaded.canonical_sha256,
            source_identity_sha256=source["identity_sha256"],
            conditional_parent_verification_sha256=(
                conditional_parent_verification_sha256
            ),
        )
        for role in ("public_execution", "private_labels", "result")
    }
    sha_values = (
        code_validation["targeted_tests_receipt_sha256"],
        code_validation["r2_sample_verification_sha256"],
        preflight["transaction_identity_sha256"],
        preflight["public_verification_sha256"],
        preflight["public_completion_marker_raw_sha256"],
        preflight["public_completion_marker_internal_sha256"],
        preflight["result_verification_sha256"],
        preflight["result_completion_marker_raw_sha256"],
        preflight["result_completion_marker_internal_sha256"],
    )
    issued_at = value["issued_at_unix_ns"]
    if (
        internal != canonical_sha256(unsigned)
        or value["version"] != _FORMAL_EXECUTION_GO_VERSION
        or value["decision_id"] != "D049"
        or value["status"]
        != (
            "GO-exactly-once-selected-gain-development-evaluation-"
            "no-test-no-actuation"
        )
        or value["authority"]
        != "user-authorized-b-level-offline-no-actuation-decision-agent"
        or conditional_gate
        != {
            "filename": "D049_CONDITIONAL_EVALUATION_RECOVERY_GATE_V3.json",
            "raw_sha256": loaded.payload["experiment"][
                "gate_record_raw_sha256"
            ],
            "status": (
                "implementation-recovery-go-formal-hold-until-final-source-"
                "r2-and-recovery-preflight-scope-amended"
            ),
        }
        or evaluation_config
        != {
            "path": (
                "configs/"
                "e018_p1_stage2a_selected_gain_evaluation_development_v2.json"
            ),
            "version": loaded.payload["version"],
            "raw_sha256": loaded.raw_sha256,
            "canonical_sha256": loaded.canonical_sha256,
        }
        or receipt_source
        != {
            "git_commit": source["git_commit"],
            "source_tree_sha256": source["source_tree_sha256"],
            "identity_sha256": source["identity_sha256"],
            "github_remote_branch_commit": source["git_commit"],
            "worker_checkout_commit": source["git_commit"],
            "worker_checkout_exact_clean": True,
        }
        or code_validation["targeted_tests_exit_code"] != 0
        or code_validation["formal_seed_read_count_before_issue"] != 0
        or any(not _selection_runtime._is_sha256(item) for item in sha_values)
        or selection_parent
        != {
            "artifact_id": loaded.payload["selection_parent"]["artifact_id"],
            "replication_state": "REPLICATED",
            "inventory_record_canonical_sha256": loaded.payload[
                "selection_parent"
            ]["persistence"]["inventory_record_canonical_sha256"],
        }
        or preflight["experiment_id"] != preflight_mode.experiment_id
        or preflight["seed"]
        != STAGE2A_SELECTED_GAIN_EVALUATION_PREFLIGHT_SEED
        or preflight["public_output_identity_sha256"]
        != preflight_roles["public_execution"]
        or preflight["private_output_identity_sha256"]
        != preflight_roles["private_labels"]
        or preflight["result_output_identity_sha256"]
        != preflight_roles["result"]
        or preflight["two_pass_verification_passed"] is not True
        or preflight["process_boundary_verified"] is not True
        or preflight["provider_prediction_count"] != 4
        or preflight["private_label_count"] != 3
        or preflight["formal_identity_consumed"] is not False
        or preflight["formal_split_consumed"] is not False
        or any(
            type(preflight[name]) is not int or preflight[name] != 0
            for name in (
                "fresh_test_reads",
                "physical_camera_actuation",
                "arm_tcp_actuation",
                "gripper_close",
            )
        )
        or formal["experiment_id"] != formal_mode.experiment_id
        or not isinstance(formal["execution_id"], str)
        or not formal["execution_id"].strip()
        or (
            expected_execution_id is not None
            and formal["execution_id"] != expected_execution_id
        )
        or formal["classification"] != formal_mode.classification
        or formal["exact_go_token"] != formal_mode.go_token
        or (
            formal["seed_start"],
            formal["seed_end"],
            formal["seed_count"],
        )
        != (formal_mode.seeds[0], formal_mode.seeds[-1], len(formal_mode.seeds))
        or formal["execution_order"] != loaded.payload["split"]["execution_order"]
        or formal["capture_attempt_count"] != 1
        or formal["scoring_attempt_count"] != 1
        or formal["same_identity_rerun_allowed"] is not False
        or formal["conditional_parent_verification_sha256"]
        != conditional_parent_verification_sha256
        or formal["artifact_role_identity_version"]
        != _EVALUATION_ARTIFACT_ROLE_IDENTITY_VERSION
        or formal["worker_artifact_root_identity_version"]
        != _WORKER_ARTIFACT_ROOT_IDENTITY_VERSION
        or not _selection_runtime._is_sha256(
            formal["worker_artifact_root_identity_sha256"]
        )
        or formal["worker_artifact_root_basename"] != formal["execution_id"]
        or (
            expected_worker_artifact_root is not None
            and (
                formal["worker_artifact_root_identity_sha256"]
                != _worker_artifact_root_identity_sha256(
                    expected_worker_artifact_root
                )
                or formal["worker_artifact_root_basename"]
                != Path(expected_worker_artifact_root).name
            )
        )
        or formal["public_output_identity_sha256"]
        != formal_roles["public_execution"]
        or formal["private_output_identity_sha256"]
        != formal_roles["private_labels"]
        or formal["result_output_identity_sha256"] != formal_roles["result"]
        or formal["formal_output_roots_absent_before_issue"] is not True
        or formal["formal_consumption_marker_absent_before_issue"] is not True
        or len(set(formal_roles.values())) != 3
        or any(formal_roles[role] == preflight_roles[role] for role in formal_roles)
        or any(type(item) is not int or item != 0 for item in permissions.values())
        or continuation
        != {
            "stage2b_required_for_every_complete_outcome": False,
            "integrity_failure_policy": (
                "freeze-current-identity-preserve-evidence-and-recover-D049-"
                "under-new-experiment-config-and-unused-seed-identity"
            ),
            "substantive_negative_result_policy": (
                "persist-publish-pause-for-reusability-refactor-without-"
                "threshold-or-seed-retuning"
            ),
        }
        or type(issued_at) is not int
        or issued_at <= 0
    ):
        raise RuntimeError("D049 final formal GO receipt contract/identity 漂移")

    if preflight_public is not None or preflight_result is not None:
        if preflight_public is None or preflight_result is None:
            raise RuntimeError("D049 final GO preflight verification 必须成对提供")
        if (
            preflight["transaction_identity_sha256"]
            != preflight_public["transaction_identity_sha256"]
            or preflight["public_output_identity_sha256"]
            != preflight_public["public_artifact_role_identity_sha256"]
            or preflight["private_output_identity_sha256"]
            != preflight_public["private_artifact_role_identity_sha256"]
            or preflight["result_output_identity_sha256"]
            != preflight_result["result_artifact_role_identity_sha256"]
            or preflight["public_verification_sha256"]
            != preflight_public["verification_sha256"]
            or preflight["public_completion_marker_raw_sha256"]
            != preflight_public_completion_raw_sha256
            or preflight["public_completion_marker_internal_sha256"]
            != preflight_public["public_completion_marker_sha256"]
            or preflight["result_verification_sha256"]
            != preflight_result["verification_sha256"]
            or preflight["result_completion_marker_raw_sha256"]
            != preflight_result_completion_raw_sha256
            or preflight["result_completion_marker_internal_sha256"]
            != preflight_result["result_completion_marker_sha256"]
            or preflight_public["provider_prediction_count"] != 4
            or preflight_result["private_label_capture_count"] != 3
            or type(preflight_completed_at_unix_ns) is not int
            or issued_at <= preflight_completed_at_unix_ns
        ):
            raise RuntimeError("D049 final GO 与已验证 preflight 证据不一致")
    return str(internal)


def verify_e018_p1_stage2a_evaluation_formal_go(
    *,
    formal_go_receipt_path: str | Path,
    expected_formal_go_raw_sha256: str,
    expected_formal_go_internal_sha256: str,
    evaluation_config_path: str | Path,
    stage2a_config_path: str | Path,
    qualification_config_path: str | Path,
    preflight_public_root: str | Path,
    preflight_result_root: str | Path,
    source: Mapping[str, Any],
    conditional_parent_verification_sha256: str,
    expected_execution_id: str,
    expected_worker_artifact_root: str | Path,
) -> dict[str, Any]:
    """机械验证独立 final-GO 文件与完成的 76894 recovery preflight。"""

    path = Path(formal_go_receipt_path)
    if (
        path.name == "D049_FINAL_FORMAL_GO_SCHEMA.json"
        or path.is_symlink()
        or not path.is_file()
    ):
        raise RuntimeError("D049 final GO 必须是独立于 unsigned schema 的真实文件")
    if file_sha256(path) != expected_formal_go_raw_sha256:
        raise RuntimeError("D049 final GO raw SHA 漂移")
    loaded = load_e018_p1_stage2a_evaluation_config(evaluation_config_path)
    preflight_public = verify_e018_p1_stage2a_evaluation_public(
        evaluation_config_path=evaluation_config_path,
        stage2a_config_path=stage2a_config_path,
        qualification_config_path=qualification_config_path,
        public_root=preflight_public_root,
        expected_source_git_commit=source["git_commit"],
        expected_source_identity_sha256=source["identity_sha256"],
        preflight=True,
    )
    preflight_result = verify_e018_p1_stage2a_evaluation_result(
        evaluation_config_path=evaluation_config_path,
        stage2a_config_path=stage2a_config_path,
        qualification_config_path=qualification_config_path,
        public_root=preflight_public_root,
        result_root=preflight_result_root,
        expected_source_git_commit=source["git_commit"],
        expected_source_identity_sha256=source["identity_sha256"],
        preflight=True,
    )
    public_marker_path = Path(preflight_public_root) / "PUBLIC_EXECUTION_COMPLETE.json"
    result_marker_path = Path(preflight_result_root) / "RESULT_COMPLETE.json"
    result_marker = _read_json(result_marker_path, "preflight result completion marker")
    receipt = _read_json(path, "D049 final formal GO receipt")
    internal = _validate_formal_execution_go_receipt(
        receipt,
        loaded=loaded,
        source=source,
        conditional_parent_verification_sha256=(
            conditional_parent_verification_sha256
        ),
        expected_execution_id=expected_execution_id,
        expected_worker_artifact_root=expected_worker_artifact_root,
        preflight_public=preflight_public,
        preflight_result=preflight_result,
        preflight_public_completion_raw_sha256=file_sha256(public_marker_path),
        preflight_result_completion_raw_sha256=file_sha256(result_marker_path),
        preflight_completed_at_unix_ns=result_marker.get("completed_at_unix_ns"),
    )
    if internal != expected_formal_go_internal_sha256:
        raise RuntimeError("D049 final GO internal SHA 漂移")
    verification = {
        "verified": True,
        "receipt": receipt,
        "receipt_raw_sha256": expected_formal_go_raw_sha256,
        "receipt_internal_sha256": internal,
        "preflight_public_verification_sha256": preflight_public[
            "verification_sha256"
        ],
        "preflight_result_verification_sha256": preflight_result[
            "verification_sha256"
        ],
    }
    verification["verification_sha256"] = canonical_sha256(verification)
    return verification


def _verify_inventory_record(
    *,
    inventory_path: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    inventory = _read_json(inventory_path, "artifact inventory")
    backups = inventory.get("backups")
    if not isinstance(backups, list):
        raise TypeError("artifact inventory backups schema 漂移")
    matches = [
        row
        for row in backups
        if isinstance(row, dict)
        and row.get("id") == config["selection_parent"]["artifact_id"]
    ]
    if len(matches) != 1:
        raise RuntimeError("selection parent inventory record 必须唯一")
    row = matches[0]
    persistence = config["selection_parent"]["persistence"]
    if (
        canonical_sha256(row) != persistence["inventory_record_canonical_sha256"]
        or row.get("replication_state") != "REPLICATED"
        or row.get("artifact_manifest_raw_sha256")
        != persistence["manifest_raw_sha256"]
        or row.get("artifact_manifest_internal_sha256")
        != persistence["manifest_internal_sha256"]
        or row.get("artifact_inventory_sha256")
        != persistence["artifact_inventory_sha256"]
        or row.get("completion_marker_raw_sha256")
        != persistence["drive_marker_raw_sha256"]
        or row.get("completion_marker_internal_sha256")
        != persistence["drive_marker_internal_sha256"]
        or row.get("drive_persistence_verification_internal_sha256")
        != persistence["drive_verification_internal_sha256"]
        or row.get("local_canonical_verification_internal_sha256")
        != persistence["local_verification_internal_sha256"]
    ):
        raise RuntimeError("selection parent inventory REPLICATED identity 漂移")
    return {
        "artifact_id": row["id"],
        "replication_state": row["replication_state"],
        "record_canonical_sha256": canonical_sha256(row),
    }


def verify_e018_p1_stage2a_evaluation_parent_gate(
    *,
    evaluation_config_path: str | Path,
    stage2a_config_path: str | Path,
    qualification_config_path: str | Path,
    g0c_config_path: str | Path,
    data_config_path: str | Path,
    selected_checkpoint_path: str | Path,
    stats_root: str | Path,
    selection_config_path: str | Path,
    selection_public_root: str | Path,
    selection_result_root: str | Path,
    decision_gate_path: str | Path,
    artifact_inventory_path: str | Path,
) -> dict[str, Any]:
    """验证 selection result、REPLICATED 证据和所有 frozen runtime 输入。"""

    loaded = load_e018_p1_stage2a_evaluation_config(evaluation_config_path)
    config = loaded.payload
    stage2a = load_e018_p1_stage2a_config(stage2a_config_path)
    qualification_path = Path(qualification_config_path)
    qualification = load_g2c_dynamic_qualification_config(qualification_path)
    g0c_path = Path(g0c_config_path)
    g0c = _stage2a._g0c.load_e018_p1_g0c_config(g0c_path)
    data_path = Path(data_config_path)
    data = _stage2a.load_e018_p1_g2c_data_config(
        data_path,
        parent_g0c_config_path=g0c_path,
    )
    frozen = config["stage2a_parent"]
    if (
        stage2a.raw_sha256 != frozen["config_raw_sha256"]
        or stage2a.canonical_sha256 != frozen["config_canonical_sha256"]
        or file_sha256(g0c_path) != frozen["g0c_config_raw_sha256"]
        or canonical_sha256(g0c) != frozen["g0c_config_canonical_sha256"]
        or file_sha256(data_path) != frozen["data_config_raw_sha256"]
        or canonical_sha256(data) != frozen["data_config_canonical_sha256"]
        or file_sha256(qualification_path)
        != frozen["qualification_config_raw_sha256"]
        or qualification.get("config_sha256")
        != frozen["qualification_config_internal_sha256"]
        or file_sha256(Path(selected_checkpoint_path))
        != frozen["checkpoint_sha256"]
    ):
        raise RuntimeError("evaluation frozen config/checkpoint identity 漂移")
    _, _, _, normalizers = _stage2a._load_normalizers(
        stats_root=Path(stats_root),
        config=data,
    )
    if (
        normalizers["proprio_stats_sha256"] != frozen["proprio_stats_sha256"]
        or normalizers["finger_force_stats_sha256"]
        != frozen["finger_force_stats_sha256"]
    ):
        raise RuntimeError("evaluation stats identity 漂移")
    from robot_vla.precision.e018_p1_stage2a_selection_runtime import (
        verify_e018_p1_stage2a_selection_result,
    )

    selection_parent = config["selection_parent"]
    selection_verified = verify_e018_p1_stage2a_selection_result(
        selection_config_path=selection_config_path,
        stage2a_config_path=stage2a_config_path,
        qualification_config_path=qualification_config_path,
        public_root=selection_public_root,
        result_root=selection_result_root,
        expected_source_git_commit=selection_parent["source_git_commit"],
        expected_source_identity_sha256=selection_parent[
            "source_identity_sha256"
        ],
    )
    if (
        selection_verified["verification_sha256"]
        != selection_parent["result_verification_sha256"]
        or Path(selection_public_root).parent.name
        != selection_parent["artifact_id"]
    ):
        raise RuntimeError("evaluation selection result parent identity 漂移")
    gate = _verify_gate_record(
        gate_path=Path(decision_gate_path),
        config=config,
    )
    inventory = _verify_inventory_record(
        inventory_path=Path(artifact_inventory_path),
        config=config,
    )
    result = {
        "version": E018_P1_STAGE2A_EVALUATION_EXECUTION_VERSION,
        "status": "verified-selection-parent-replicated-and-inputs-frozen",
        "selection_result_verification_sha256": selection_verified[
            "verification_sha256"
        ],
        "selection_artifact_id": selection_parent["artifact_id"],
        "gate": gate,
        "inventory": inventory,
        "stage2a_config_raw_sha256": stage2a.raw_sha256,
        "stage2a_config_canonical_sha256": stage2a.canonical_sha256,
        "g0c_config_raw_sha256": file_sha256(g0c_path),
        "g0c_config_canonical_sha256": canonical_sha256(g0c),
        "data_config_raw_sha256": file_sha256(data_path),
        "data_config_canonical_sha256": canonical_sha256(data),
        "qualification_config_raw_sha256": file_sha256(qualification_path),
        "qualification_config_internal_sha256": qualification[
            "config_sha256"
        ],
        "checkpoint_sha256": file_sha256(Path(selected_checkpoint_path)),
        "normalizer_identity": normalizers,
        "fresh_evaluation_seed_reads": 0,
    }
    result["verification_sha256"] = canonical_sha256(result)
    return result


class Stage2AEvaluationJournal:
    """只持久化 Pass A 的 prediction commits；无 private root/API。"""

    def __init__(
        self,
        *,
        public_root: str | Path,
        config_canonical_sha256: str,
        transaction_identity_sha256: str,
        mode: _EvaluationMode,
    ) -> None:
        self.public_root = Path(public_root)
        if self.public_root.exists() or self.public_root.is_symlink():
            raise FileExistsError("evaluation public root 必须全新")
        self.public_root.mkdir(mode=0o700, parents=True, exist_ok=False)
        (self.public_root / "prediction_commits").mkdir(mode=0o700)
        self.config_canonical_sha256 = config_canonical_sha256
        self.transaction_identity_sha256 = transaction_identity_sha256
        self.mode = mode
        self.writer = _AppendOnlyJsonl(
            self.public_root / "provider_output_ledger.jsonl"
        )
        self.prediction_count = 0
        self.previous_commit_sha256: str | None = None

    def commit_prediction(
        self,
        row: Mapping[str, Any],
        *,
        seed: int,
        route_frame_index: int,
        provider_output_digest: str,
        model_input_digest: str,
    ) -> dict[str, Any]:
        index = self.prediction_count
        expected_seed = self.mode.seeds[index // 4]
        expected_frame = STAGE2A_PROVIDER_FRAME_INDICES[index % 4]
        if seed != expected_seed or route_frame_index != expected_frame:
            raise RuntimeError("evaluation prediction seed/frame exact order 漂移")
        public_row = dict(row)
        assert_qualification_prediction_deployable_only(public_row)
        if (
            public_row.get("provider_output_digest") != provider_output_digest
            or public_row.get("model_input_digest") != model_input_digest
        ):
            raise RuntimeError("evaluation provider row/digest 未绑定")
        self.writer.append([public_row])
        completed_at = time.time_ns()
        receipt = {
            "version": E018_P1_STAGE2A_EVALUATION_EXECUTION_VERSION,
            "row_index": index,
            "seed": seed,
            "route_frame_index": route_frame_index,
            "provider_output_digest": provider_output_digest,
            "model_input_digest": model_input_digest,
            "transaction_identity_sha256": self.transaction_identity_sha256,
            "provider_ledger_prefix_raw_sha256": file_sha256(self.writer.path),
            "previous_prediction_commit_sha256": self.previous_commit_sha256,
            "prediction_fsync_completed_at_unix_ns": completed_at,
        }
        receipt["commit_receipt_sha256"] = canonical_sha256(receipt)
        _atomic_create_json(
            self.public_root
            / "prediction_commits"
            / f"{index:06d}.commit.json",
            receipt,
        )
        self.previous_commit_sha256 = receipt["commit_receipt_sha256"]
        self.prediction_count += 1
        return receipt

    def freeze(self) -> dict[str, Any]:
        if self.prediction_count != self.mode.prediction_count:
            raise RuntimeError("evaluation prediction count 未达到冻结值")
        return self.writer.freeze()


_ACTION_PREFIX_FIELDS = (
    "episode_id",
    "request_id",
    "camera_command_sequence_id",
    "frame_index",
    "control_tick",
    "timestamp_s",
    "camera_motion_state",
    "viewpoint_primitive_id",
    "target_orientation_id",
    "orientation_progress",
    "commanded_yaw_offset_rad",
    "commanded_pitch_offset_rad",
    "commanded_roll_offset_rad",
    "arm_owner",
    "gripper_owner",
    "external_camera_owner",
    "arm_motion_command_max_abs",
    "gripper_hold_open_command",
    "commanded_external_position_world_m",
    "commanded_external_quaternion_sapien",
    "commanded_world_from_external_camera_gl",
    "commanded_base_from_external_camera_cv",
)


def _action_prefix_sha256s(
    camera_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    prefix: list[dict[str, Any]] = []
    digests: list[str] = []
    for index, source in enumerate(camera_rows):
        if any(field not in source for field in _ACTION_PREFIX_FIELDS):
            raise RuntimeError(f"evaluation camera action row[{index}] 缺字段")
        prefix.append({field: source[field] for field in _ACTION_PREFIX_FIELDS})
        digests.append(canonical_sha256(prefix))
    return digests


def _build_evaluation_route_evidence_row(
    *,
    mode: _EvaluationMode,
    route_index: int,
    camera_rows: Sequence[Mapping[str, Any]],
    provider_records: Sequence[Stage2AProviderOutputRecord],
    prediction_receipts: Sequence[Mapping[str, Any]],
    capture_transaction: Mapping[str, Any],
    route_summary: Mapping[str, Any],
    captured_route: CapturedStage2AEvaluationRoute,
) -> dict[str, Any]:
    seed = mode.seeds[route_index]
    if (
        len(camera_rows) != 92
        or len(provider_records) != 4
        or len(prediction_receipts) != 4
        or captured_route.seed != seed
        or tuple(record.route_frame_index for record in provider_records)
        != STAGE2A_PROVIDER_FRAME_INDICES
    ):
        raise RuntimeError("evaluation route bundle count/order 漂移")
    camera_start = route_index * 92
    provider_start = route_index * 4
    transaction = dict(capture_transaction)
    summary = dict(route_summary)
    route = captured_route.to_public_dict()
    row = {
        "version": E018_P1_STAGE2A_EVALUATION_EXECUTION_VERSION,
        "seed": seed,
        "episode_id": captured_route.episode_id,
        "request_id": captured_route.request.request_id,
        "camera_row_start": camera_start,
        "camera_row_stop_exclusive": camera_start + 92,
        "camera_rows_sha256": canonical_sha256(list(camera_rows)),
        "action_prefix_sha256s": _action_prefix_sha256s(camera_rows),
        "provider_row_indices": list(range(provider_start, provider_start + 4)),
        "provider_output_digests": [
            value.provider_output_digest for value in provider_records
        ],
        "model_input_digests": [
            value.model_input_digest for value in provider_records
        ],
        "prediction_commit_receipt_sha256s": [
            value["commit_receipt_sha256"] for value in prediction_receipts
        ],
        "capture_transaction": transaction,
        "capture_transaction_sha256": canonical_sha256(transaction),
        "route_summary": summary,
        "route_summary_sha256": canonical_sha256(summary),
        "captured_route": route,
        "route_evidence_digest": captured_route.route_evidence_digest,
    }
    row["route_row_sha256"] = canonical_sha256(row)
    return row


def _run_evaluation_pass_a_simulator(
    *,
    loaded_evaluation_config: Any,
    loaded_stage2a_config: Any,
    qualification_config: Mapping[str, Any],
    g0c_config: dict[str, Any],
    data_config: Mapping[str, Any],
    stats_root: Path,
    selected_checkpoint_path: Path,
    public_root: Path,
    journal: Stage2AEvaluationJournal,
    execution_progress: Stage2AExecutionProgress,
    started_monotonic: float,
    mode: _EvaluationMode,
) -> tuple[
    list[dict[str, Any]],
    list[CapturedStage2AEvaluationRoute],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[Stage2AProviderOutputRecord],
    list[list[dict[str, Any]]],
    dict[str, Any],
    bool,
]:
    """Pass A 唯一 env/provider 路径；hook 没有 private-label API。"""

    import gymnasium as gym
    import mani_skill
    import sapien
    import torch
    from mani_skill.utils import sapien_utils

    from robot_vla.sim import register_robot_vla_maniskill_envs

    if (
        mani_skill.__version__
        != data_config["software"]["expected_mani_skill_version"]
        or sapien.__version__
        != data_config["software"]["expected_sapien_version"]
        or not torch.cuda.is_available()
    ):
        raise RuntimeError("evaluation GPU/ManiSkill/SAPIEN environment 漂移")
    spec, proprio, force, normalizer_identity = _stage2a._load_normalizers(
        stats_root=stats_root,
        config=data_config,
    )
    provider: Any | None = None
    env: Any | None = None
    env_closed = False
    sensor: Any | None = None
    camera: Any | None = None
    rows: list[dict[str, Any]] = []
    captured_routes: list[CapturedStage2AEvaluationRoute] = []
    route_summaries: list[dict[str, Any]] = []
    transactions: list[dict[str, Any]] = []
    provider_records: list[Stage2AProviderOutputRecord] = []
    route_prediction_receipts: list[list[dict[str, Any]]] = []
    try:
        provider = _stage2a.QualificationProvider(
            checkpoint_path=selected_checkpoint_path,
            qualification_config=qualification_config,
            data_config=data_config,
            classification=_stage2a.QUALIFICATION_CLASSIFICATION_SELECTION,
        )
        home, anchors, orientations = _stage2a._g0c._parse_library(g0c_config)
        primitives = _stage2a._g0c._expand_primitives(anchors, orientations)
        by_id = {
            item.viewpoint_id: (item, orientation)
            for item, orientation in primitives
        }
        if tuple(by_id) != _stage2a.FRONT_ALTERNATE_IDS:
            raise RuntimeError("evaluation G0C primitive order 漂移")
        primary, primary_orientation = by_id[
            _stage2a.ACTIVE_FRONT_PRIMARY_PRIMITIVE_ID
        ]
        route_config = json.loads(json.dumps(g0c_config))
        route_config["experiment"]["offline_segmentation_diagnostics"] = False
        route_config["experiment"]["save_settled_rgb"] = False
        register_robot_vla_maniskill_envs()
        environment = route_config["environment"]
        env = gym.make(
            environment["environment_id"],
            obs_mode=environment["obs_mode"],
            control_mode=environment["control_mode"],
            num_envs=environment["num_envs"],
            robot_uids=environment["robot_uid"],
        )
        base_env = env.unwrapped
        if (
            base_env.control_freq != environment["control_hz"]
            or environment["camera_uid"] not in base_env._sensors
        ):
            raise RuntimeError("evaluation environment/control/camera identity 漂移")
        sensor = base_env._sensors[environment["camera_uid"]]
        camera = sensor.camera
        if sensor.entity is not None or not callable(
            getattr(camera, "set_local_pose", None)
        ):
            raise RuntimeError("evaluation 要求 isolated unmounted external camera")
        for seed in mode.seeds:
            if (
                time.monotonic() - started_monotonic
                > loaded_evaluation_config.payload["budgets"][
                    "gpu_wall_seconds_max"
                ]
            ):
                raise TimeoutError("evaluation Pass A GPU wall budget 已到")
            if mode.preflight:
                execution_progress.begin_selected_gain_evaluation_preflight(
                    seed,
                    experiment_identity=mode.experiment_id,
                )
            else:
                execution_progress.begin_selected_gain_evaluation(
                    seed,
                    experiment_identity=mode.experiment_id,
                )
            prediction_receipts: list[dict[str, Any]] = []

            def provider_commit_hook(
                record: Stage2AProviderOutputRecord,
                motion_row: Mapping[str, Any],
                rgb: np.ndarray,
                observation: Mapping[str, Any],
                _bound_seed: int = seed,
                _receipts: list[dict[str, Any]] = prediction_receipts,
            ) -> None:
                del observation
                rgb_array = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
                if (
                    rgb_array.shape != (128, 128, 3)
                    or hashlib.sha256(rgb_array.tobytes()).hexdigest()
                    != motion_row["rgb_sha256"]
                ):
                    raise RuntimeError("evaluation hook RGB/route identity 漂移")
                _receipts.append(
                    journal.commit_prediction(
                        record.to_dict(),
                        seed=_bound_seed,
                        route_frame_index=record.route_frame_index,
                        provider_output_digest=record.provider_output_digest,
                        model_input_digest=record.model_input_digest,
                    )
                )

            factory = (
                Stage2ARouteTransaction.for_selected_gain_evaluation_preflight_capture
                if mode.preflight
                else Stage2ARouteTransaction.for_selected_gain_evaluation_capture
            )
            transaction = factory(
                experiment_identity=mode.experiment_id,
                seed=seed,
                provider=provider,
                stage2_config=loaded_stage2a_config,
                qualification_config=qualification_config,
                data_config=data_config,
                base_env=base_env,
                spec=spec,
                proprio_normalizer=proprio,
                finger_force_normalizer=force,
                execution_progress=execution_progress,
                provider_commit_hook=provider_commit_hook,
            )
            route_rows, route_summary, _ = _stage2a._g0._run_route(
                env=env,
                base_env=base_env,
                camera=camera,
                config=route_config,
                seed=seed,
                home=home,
                alternate=primary,
                output_root=public_root,
                sapien_module=sapien,
                sapien_utils_module=sapien_utils,
                alternate_orientation=primary_orientation,
                result_version=E018_P1_STAGE2A_EXECUTION_VERSION,
                episode_prefix="stage2a-selected-gain-evaluation",
                source_phase=_stage2a.STAGE2A_SOURCE_PHASE.value,
                camera_owner=_stage2a.STAGE2A_CAMERA_OWNER,
                frame_hook=transaction.frame_hook,
                warmup_hook=transaction.warmup_hook,
                pre_command_hook=transaction.pre_command_hook,
                episode_id_override=transaction.episode_id,
                request_id_override=f"{transaction.episode_id}-active-front-01",
                command_sequence_id_override=(
                    f"{transaction.episode_id}-active-front-01-camera-command-00"
                ),
                include_raw_safety_witnesses=True,
                include_raw_proprio_velocity_witness=True,
                include_privileged_object_state_witnesses=False,
                include_robot_object_contact_witnesses=False,
            )
            route_summary = {
                **route_summary,
                "classification": mode.capture_classification,
                "provider_forward_count": len(transaction.provider_records),
                "memory_write_count": transaction.orchestrator.memory_write_count,
                "offline_segmentation_diagnostics": False,
                "runtime_object_gt_reads": 0,
                "goal_gt_reads": 0,
                "fresh_test_reads": 0,
            }
            transaction_row = transaction.finalize(route_summary)
            captured = CapturedStage2AEvaluationRoute.from_transaction_export(
                transaction.selection_replay_inputs()
            )
            if (
                len(route_rows) != 92
                or len(prediction_receipts) != 4
                or transaction.orchestrator.memory_write_count != 0
            ):
                raise RuntimeError("evaluation Pass A route accounting 漂移")
            rows.extend(route_rows)
            captured_routes.append(captured)
            route_summaries.append(route_summary)
            transactions.append(transaction_row)
            provider_records.extend(transaction.provider_records)
            route_prediction_receipts.append(prediction_receipts)
    finally:
        if env is not None:
            env.close()
            env_closed = True
        if provider is not None:
            provider.destroy()
    provider_destroyed = bool(provider is not None and provider.destroyed)
    environment_identity = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(torch.device("cuda")),
        "mani_skill": mani_skill.__version__,
        "sapien": sapien.__version__,
        "external_camera_sensor_class": (
            None
            if sensor is None
            else type(sensor).__module__ + "." + type(sensor).__name__
        ),
        "external_camera_class": (
            None
            if camera is None
            else type(camera).__module__ + "." + type(camera).__name__
        ),
        "external_camera_unmounted": bool(
            sensor is not None and sensor.entity is None
        ),
        "provider_context_destroyed": provider_destroyed,
        "environment_closed": env_closed,
        "normalizer_identity": normalizer_identity,
    }
    return (
        rows,
        captured_routes,
        route_summaries,
        transactions,
        provider_records,
        route_prediction_receipts,
        environment_identity,
        bool(env_closed and provider_destroyed),
    )


def _verify_evaluation_prediction_commit_chain(
    *,
    public_root: Path,
    provider_rows: Sequence[Mapping[str, Any]],
    transaction_identity_sha256: str,
    mode: _EvaluationMode,
) -> list[dict[str, Any]]:
    raw_lines = (
        public_root / "provider_output_ledger.jsonl"
    ).read_bytes().splitlines(keepends=True)
    if len(raw_lines) != mode.prediction_count or any(
        not line.endswith(b"\n") for line in raw_lines
    ):
        raise RuntimeError("evaluation provider raw ledger count/terminator 漂移")
    prefix = hashlib.sha256()
    previous: str | None = None
    receipts: list[dict[str, Any]] = []
    for index, (line, provider_row) in enumerate(
        zip(raw_lines, provider_rows, strict=True)
    ):
        if line != _selection_runtime._jsonl_line_bytes(provider_row):
            raise RuntimeError("evaluation provider ledger serialization 漂移")
        prefix.update(line)
        receipt = _read_json(
            public_root
            / "prediction_commits"
            / f"{index:06d}.commit.json",
            f"evaluation prediction commit[{index}]",
        )
        unsigned = dict(receipt)
        stored = unsigned.pop("commit_receipt_sha256", None)
        expected_seed = mode.seeds[index // 4]
        expected_frame = STAGE2A_PROVIDER_FRAME_INDICES[index % 4]
        if (
            set(receipt)
            != {
                "version",
                "row_index",
                "seed",
                "route_frame_index",
                "provider_output_digest",
                "model_input_digest",
                "transaction_identity_sha256",
                "provider_ledger_prefix_raw_sha256",
                "previous_prediction_commit_sha256",
                "prediction_fsync_completed_at_unix_ns",
                "commit_receipt_sha256",
            }
            or stored != canonical_sha256(unsigned)
            or receipt["version"]
            != E018_P1_STAGE2A_EVALUATION_EXECUTION_VERSION
            or receipt["row_index"] != index
            or receipt["seed"] != expected_seed
            or receipt["route_frame_index"] != expected_frame
            or receipt["provider_output_digest"]
            != provider_row.get("provider_output_digest")
            or receipt["model_input_digest"]
            != provider_row.get("model_input_digest")
            or receipt["transaction_identity_sha256"]
            != transaction_identity_sha256
            or receipt["provider_ledger_prefix_raw_sha256"]
            != prefix.hexdigest()
            or receipt["previous_prediction_commit_sha256"] != previous
            or type(receipt["prediction_fsync_completed_at_unix_ns"])
            is not int
            or receipt["prediction_fsync_completed_at_unix_ns"] <= 0
        ):
            raise RuntimeError(f"evaluation prediction commit[{index}] chain 漂移")
        previous = stored
        receipts.append(receipt)
    return receipts


def _verify_evaluation_route_summary(
    value: Mapping[str, Any],
    *,
    seed: int,
    episode_id: str,
    mode: _EvaluationMode,
) -> dict[str, Any]:
    source = dict(value)
    if source.get("classification") != mode.capture_classification:
        raise RuntimeError("evaluation route summary classification 漂移")
    normalized = dict(source)
    normalized["classification"] = (
        "formal-development-selection-capture-only-no-test-no-actuation/v2"
    )
    _selection_runtime._verify_selection_route_summary_schema(
        normalized,
        seed=seed,
        episode_id=episode_id,
    )
    return source


def _verify_evaluation_route_row(
    value: Mapping[str, Any],
    *,
    route_index: int,
    camera_rows: Sequence[Mapping[str, Any]],
    provider_records: Sequence[Stage2AProviderOutputRecord],
    prediction_receipts: Sequence[Mapping[str, Any]],
    stage2a_config: Any,
    qualification_config: Mapping[str, Any],
    mode: _EvaluationMode,
) -> CapturedStage2AEvaluationRoute:
    row = _require_exact_keys(
        dict(value),
        {
            "version",
            "seed",
            "episode_id",
            "request_id",
            "camera_row_start",
            "camera_row_stop_exclusive",
            "camera_rows_sha256",
            "action_prefix_sha256s",
            "provider_row_indices",
            "provider_output_digests",
            "model_input_digests",
            "prediction_commit_receipt_sha256s",
            "capture_transaction",
            "capture_transaction_sha256",
            "route_summary",
            "route_summary_sha256",
            "captured_route",
            "route_evidence_digest",
            "route_row_sha256",
        },
        f"evaluation route[{route_index}]",
    )
    unsigned = dict(row)
    stored = unsigned.pop("route_row_sha256")
    seed = mode.seeds[route_index]
    episode = (
        "e018-p1-stage2a-selected-gain-evaluation-recovery-preflight-"
        f"seed-{seed}"
        if mode.preflight
        else f"e018-p1-stage2a-selected-gain-evaluation-seed-{seed}"
    )
    request = f"{episode}-active-front-01"
    camera_start = route_index * 92
    provider_start = route_index * 4
    route_camera = camera_rows[camera_start : camera_start + 92]
    route_records = provider_records[provider_start : provider_start + 4]
    route_receipts = prediction_receipts[provider_start : provider_start + 4]
    if (
        stored != canonical_sha256(unsigned)
        or row["version"] != E018_P1_STAGE2A_EVALUATION_EXECUTION_VERSION
        or row["seed"] != seed
        or row["episode_id"] != episode
        or row["request_id"] != request
        or row["camera_row_start"] != camera_start
        or row["camera_row_stop_exclusive"] != camera_start + 92
        or row["camera_rows_sha256"] != canonical_sha256(list(route_camera))
        or row["action_prefix_sha256s"]
        != _action_prefix_sha256s(route_camera)
        or row["provider_row_indices"]
        != list(range(provider_start, provider_start + 4))
        or row["provider_output_digests"]
        != [record.provider_output_digest for record in route_records]
        or row["model_input_digests"]
        != [record.model_input_digest for record in route_records]
        or row["prediction_commit_receipt_sha256s"]
        != [receipt["commit_receipt_sha256"] for receipt in route_receipts]
        or row["capture_transaction_sha256"]
        != canonical_sha256(row["capture_transaction"])
        or row["route_summary_sha256"]
        != canonical_sha256(row["route_summary"])
    ):
        raise RuntimeError(f"evaluation route[{route_index}] outer binding 漂移")
    verified_camera = [
        _stage2a._verify_stage2a_camera_row_identity(
            camera,
            episode_id=episode,
            request_id=request,
            frame_index=frame,
        )
        for frame, camera in enumerate(route_camera)
    ]
    maximum_projection_error = float(
        qualification_config["capture_safety"][
            "maximum_rotation_projection_error_frobenius"
        ]
    )
    primary = []
    for record in route_records:
        camera = verified_camera[record.route_frame_index]
        _stage2a._verify_stage2a_provider_route_pose_binding(
            record.prediction["base_from_external_camera_cv"],
            camera["actual_base_from_external_camera_cv"],
            maximum_projection_error=maximum_projection_error,
        )
        if record.route_frame_index == 0:
            _stage2a.build_stage2a_home_score_evidence(
                record,
                motion_row=camera,
                timestamp_offset_s=Stage2ARouteTransaction._TIMESTAMP_OFFSET_S,
            )
        else:
            primary.append(
                _stage2a.build_stage2a_primary_frame_evidence(
                    record,
                    motion_row=camera,
                    timestamp_offset_s=(
                        Stage2ARouteTransaction._TIMESTAMP_OFFSET_S
                    ),
                )
            )
    captured = CapturedStage2AEvaluationRoute.from_public_dict(
        row["captured_route"]
    )
    transaction = _require_exact_keys(
        row["capture_transaction"],
        _selection_runtime._SELECTION_TRANSACTION_KEYS,
        f"evaluation transaction[{route_index}]",
    )
    summary = _verify_evaluation_route_summary(
        row["route_summary"],
        seed=seed,
        episode_id=episode,
        mode=mode,
    )
    candidate = transaction["candidate"]
    candidate_digest = None if candidate is None else canonical_sha256(candidate)
    candidate_eligible = (
        None if candidate is None else candidate.get("commit_eligible")
    )
    candidate_reasons = (
        None if candidate is None else candidate.get("rejection_reasons")
    )
    if (
        captured.route_evidence_digest != row["route_evidence_digest"]
        or [frame.frame_digest for frame in primary]
        != [frame.frame_digest for frame in captured.primary_frames]
        or transaction["seed"] != seed
        or transaction["episode_id"] != episode
        or transaction["request_id"] != request
        or transaction["classification"] != mode.capture_classification
        or transaction["effect_claim"] != "no-effect-claim"
        or transaction["provider_output_digests"]
        != row["provider_output_digests"]
        or transaction["provider_frame_indices"]
        != list(STAGE2A_PROVIDER_FRAME_INDICES)
        or transaction["candidate_digest"] != candidate_digest
        or captured.raw_candidate_digest_at_gain_0_02 != candidate_digest
        or captured.raw_candidate_commit_eligible_at_gain_0_02
        is not candidate_eligible
        or list(captured.raw_candidate_rejection_reasons_at_gain_0_02 or ())
        != (candidate_reasons or [])
        or transaction["memory_write_count"] != 0
        or transaction["commit_receipt"] is not None
        or transaction["shadow_action_generation"] is not None
        or transaction["action_history_resume_audit"] is not None
        or any(
            transaction[name] != 0
            for name in (
                "runtime_object_gt_reads",
                "goal_gt_reads",
                "offline_label_reads",
                "wrist_provider_forward_count",
                "arm_motion_command_count",
                "gripper_close_command_count",
                "fresh_test_reads",
                "checkpoint_writes",
            )
        )
        or summary["provider_forward_count"] != 4
        or summary["memory_write_count"] != 0
        or captured.route_protocol_safety_valid is not summary["passed"]
        or transaction["route_passed"]
        is not captured.route_protocol_safety_valid
    ):
        raise RuntimeError(
            f"evaluation route[{route_index}] transaction/replay 漂移"
        )
    _selection_runtime._verify_capture_only_transaction_deep(
        stage2a_config=stage2a_config,
        transaction=transaction,
        route_summary=summary,
        camera_rows=verified_camera,
        provider_records=route_records,
        primary_frames=primary,
        captured=captured,
    )
    return captured


def _verify_conditional_parent_receipt(
    value: Mapping[str, Any],
    *,
    loaded: Any,
) -> dict[str, Any]:
    receipt = dict(value)
    unsigned = dict(receipt)
    stored = unsigned.pop("verification_sha256", None)
    frozen = loaded.payload
    if (
        stored != canonical_sha256(unsigned)
        or receipt.get("version")
        != E018_P1_STAGE2A_EVALUATION_EXECUTION_VERSION
        or receipt.get("status")
        != "verified-selection-parent-replicated-and-inputs-frozen"
        or receipt.get("selection_result_verification_sha256")
        != frozen["selection_parent"]["result_verification_sha256"]
        or receipt.get("selection_artifact_id")
        != frozen["selection_parent"]["artifact_id"]
        or receipt.get("gate", {}).get("gate_raw_sha256")
        != frozen["experiment"]["gate_record_raw_sha256"]
        or receipt.get("inventory", {}).get("replication_state")
        != "REPLICATED"
        or receipt.get("inventory", {}).get("record_canonical_sha256")
        != frozen["selection_parent"]["persistence"][
            "inventory_record_canonical_sha256"
        ]
        or receipt.get("checkpoint_sha256")
        != frozen["stage2a_parent"]["checkpoint_sha256"]
        or receipt.get("fresh_evaluation_seed_reads") != 0
    ):
        raise RuntimeError("evaluation parent verifier receipt 漂移")
    return receipt


def _build_parent_authorization_receipt(
    *,
    conditional_parent: Mapping[str, Any],
    mode: _EvaluationMode,
    formal_execution_go: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if (formal_execution_go is None) is not mode.preflight:
        raise RuntimeError("evaluation parent final-GO applicability 漂移")
    value = {
        "version": _PARENT_AUTHORIZATION_VERSION,
        "status": (
            "verified-conditional-parent-preflight-no-formal-go"
            if mode.preflight
            else "verified-conditional-parent-and-final-formal-go"
        ),
        "preflight": mode.preflight,
        "conditional_parent": dict(conditional_parent),
        "conditional_parent_verification_sha256": conditional_parent[
            "verification_sha256"
        ],
        "formal_execution_go_required": not mode.preflight,
        "formal_execution_go": (
            None if formal_execution_go is None else dict(formal_execution_go)
        ),
    }
    value["verification_sha256"] = canonical_sha256(value)
    return value


def _verify_embedded_formal_execution_go(
    value: Any,
    *,
    loaded: Any,
    source: Mapping[str, Any],
    conditional_parent_verification_sha256: str,
    expected_execution_id: str | None = None,
) -> dict[str, Any]:
    embedded = _require_exact_keys(
        value,
        {
            "verified",
            "receipt",
            "receipt_raw_sha256",
            "receipt_internal_sha256",
            "preflight_public_verification_sha256",
            "preflight_result_verification_sha256",
            "verification_sha256",
        },
        "D049 embedded final GO verification",
    )
    receipt = embedded.get("receipt")
    if not isinstance(receipt, Mapping):
        raise TypeError("D049 embedded final GO receipt 类型漂移")
    internal = _validate_formal_execution_go_receipt(
        receipt,
        loaded=loaded,
        source=source,
        conditional_parent_verification_sha256=(
            conditional_parent_verification_sha256
        ),
        expected_execution_id=expected_execution_id,
    )
    unsigned = dict(embedded)
    verification_sha256 = unsigned.pop("verification_sha256", None)
    if (
        embedded["verified"] is not True
        or not _selection_runtime._is_sha256(embedded["receipt_raw_sha256"])
        or embedded["receipt_internal_sha256"] != internal
        or embedded["preflight_public_verification_sha256"]
        != receipt["preflight"]["public_verification_sha256"]
        or embedded["preflight_result_verification_sha256"]
        != receipt["preflight"]["result_verification_sha256"]
        or verification_sha256 != canonical_sha256(unsigned)
    ):
        raise RuntimeError("D049 embedded final GO verification 漂移")
    return dict(embedded)


def _verify_parent_receipt(
    value: Mapping[str, Any],
    *,
    loaded: Any,
    mode: _EvaluationMode,
    source: Mapping[str, Any],
    expected_execution_id: str | None = None,
) -> dict[str, Any]:
    receipt = _require_exact_keys(
        dict(value),
        {
            "version",
            "status",
            "preflight",
            "conditional_parent",
            "conditional_parent_verification_sha256",
            "formal_execution_go_required",
            "formal_execution_go",
            "verification_sha256",
        },
        "evaluation authorized parent receipt",
    )
    conditional = _verify_conditional_parent_receipt(
        receipt["conditional_parent"], loaded=loaded
    )
    formal_execution_go = receipt["formal_execution_go"]
    if mode.preflight:
        embedded = None
    else:
        embedded = _verify_embedded_formal_execution_go(
            formal_execution_go,
            loaded=loaded,
            source=source,
            conditional_parent_verification_sha256=conditional[
                "verification_sha256"
            ],
            expected_execution_id=expected_execution_id,
        )
    unsigned = dict(receipt)
    stored = unsigned.pop("verification_sha256", None)
    if (
        receipt["version"] != _PARENT_AUTHORIZATION_VERSION
        or receipt["preflight"] is not mode.preflight
        or receipt["status"]
        != (
            "verified-conditional-parent-preflight-no-formal-go"
            if mode.preflight
            else "verified-conditional-parent-and-final-formal-go"
        )
        or receipt["conditional_parent_verification_sha256"]
        != conditional["verification_sha256"]
        or receipt["formal_execution_go_required"] is not (not mode.preflight)
        or (formal_execution_go is None) is not mode.preflight
        or (embedded is None) is not mode.preflight
        or stored != canonical_sha256(unsigned)
    ):
        raise RuntimeError("evaluation authorized parent receipt 漂移")
    return dict(receipt)


def _verify_external_formal_go_against_embedded(
    *,
    formal_go_receipt_path: str | Path,
    expected_formal_go_raw_sha256: str,
    expected_formal_go_internal_sha256: str,
    embedded_verification: Mapping[str, Any],
    loaded: Any,
    source: Mapping[str, Any],
    conditional_parent_verification_sha256: str,
    expected_execution_id: str,
    expected_worker_artifact_root: str | Path,
) -> None:
    path = Path(formal_go_receipt_path)
    if (
        path.name == "D049_FINAL_FORMAL_GO_SCHEMA.json"
        or path.is_symlink()
        or not path.is_file()
        or file_sha256(path) != expected_formal_go_raw_sha256
    ):
        raise RuntimeError("D049 external final GO file/raw SHA 漂移")
    receipt = _read_json(path, "D049 external final GO receipt")
    internal = _validate_formal_execution_go_receipt(
        receipt,
        loaded=loaded,
        source=source,
        conditional_parent_verification_sha256=(
            conditional_parent_verification_sha256
        ),
        expected_execution_id=expected_execution_id,
        expected_worker_artifact_root=expected_worker_artifact_root,
    )
    if (
        internal != expected_formal_go_internal_sha256
        or embedded_verification.get("receipt") != receipt
        or embedded_verification.get("receipt_raw_sha256")
        != expected_formal_go_raw_sha256
        or embedded_verification.get("receipt_internal_sha256") != internal
    ):
        raise RuntimeError("D049 external final GO 与 Pass A frozen binding 漂移")


def _verify_e018_p1_stage2a_evaluation_public(
    *,
    evaluation_config_path: str | Path,
    stage2a_config_path: str | Path,
    qualification_config_path: str | Path,
    public_root: str | Path,
    expected_source_git_commit: str,
    expected_source_identity_sha256: str,
    preflight: bool,
    require_complete: bool,
) -> dict[str, Any]:
    """独立验证 Pass A；接口不接受 private/checkpoint/provider root。"""

    loaded = load_e018_p1_stage2a_evaluation_config(evaluation_config_path)
    stage2a = load_e018_p1_stage2a_config(stage2a_config_path)
    qualification = load_g2c_dynamic_qualification_config(
        qualification_config_path
    )
    mode = _evaluation_mode(preflight)
    root = Path(public_root)
    _selection_runtime._assert_exact_tree(
        root,
        expected_files=(
            _PUBLIC_FILES if require_complete else _PUBLIC_PRECOMPLETION_FILES
        ),
        expected_directory="prediction_commits",
        expected_directory_files={
            f"{index:06d}.commit.json"
            for index in range(mode.prediction_count)
        },
        name="evaluation public artifact",
    )
    snapshot = _read_json(root / "config_snapshot.json", "evaluation config snapshot")
    if (
        set(snapshot)
        != {
            "version",
            "config_raw_sha256",
            "config_canonical_sha256",
            "config",
            "snapshot_sha256",
        }
        or _verify_internal_digest(
            snapshot,
            digest_key="snapshot_sha256",
            name="evaluation config snapshot",
        )
        != snapshot["snapshot_sha256"]
        or snapshot["version"]
        != E018_P1_STAGE2A_EVALUATION_EXECUTION_VERSION
        or snapshot["config_raw_sha256"] != loaded.raw_sha256
        or snapshot["config_canonical_sha256"] != loaded.canonical_sha256
        or snapshot["config"] != loaded.payload
    ):
        raise RuntimeError("evaluation config snapshot 漂移")
    source = _read_json(root / "source_identity.json", "evaluation source identity")
    if (
        set(source) != {"git_commit", "source_tree_sha256", "identity_sha256"}
        or source["identity_sha256"]
        != canonical_sha256(
            {
                "git_commit": source["git_commit"],
                "source_tree_sha256": source["source_tree_sha256"],
            }
        )
        or source["git_commit"] != expected_source_git_commit
        or source["identity_sha256"] != expected_source_identity_sha256
    ):
        raise RuntimeError("evaluation source identity 漂移")
    parent = _verify_parent_receipt(
        _read_json(root / "parent_verification.json", "evaluation parent receipt"),
        loaded=loaded,
        mode=mode,
        source=source,
        expected_execution_id=None,
    )
    conditional_parent_sha256 = parent[
        "conditional_parent_verification_sha256"
    ]
    formal_go_verification_sha256 = (
        None
        if mode.preflight
        else parent["formal_execution_go"]["verification_sha256"]
    )
    started = _read_json(root / "RUN_STARTED.json", "evaluation RUN_STARTED")
    transaction_primitive = {
        "version": E018_P1_STAGE2A_EVALUATION_EXECUTION_VERSION,
        "experiment_id": mode.experiment_id,
        "classification": mode.classification,
        "config_raw_sha256": loaded.raw_sha256,
        "config_canonical_sha256": loaded.canonical_sha256,
        "source_identity_sha256": source["identity_sha256"],
        "parent_verification_sha256": parent["verification_sha256"],
        "conditional_parent_verification_sha256": conditional_parent_sha256,
        "formal_execution_go_verification_sha256": (
            formal_go_verification_sha256
        ),
        "public_artifact_role_identity_sha256": (
            _evaluation_artifact_role_identity_sha256(
                role="public_execution",
                mode=mode,
                config_canonical_sha256=loaded.canonical_sha256,
                source_identity_sha256=source["identity_sha256"],
                conditional_parent_verification_sha256=(
                    conditional_parent_sha256
                ),
            )
        ),
        "private_artifact_role_identity_sha256": (
            _evaluation_artifact_role_identity_sha256(
                role="private_labels",
                mode=mode,
                config_canonical_sha256=loaded.canonical_sha256,
                source_identity_sha256=source["identity_sha256"],
                conditional_parent_verification_sha256=(
                    conditional_parent_sha256
                ),
            )
        ),
        "seeds": [mode.seeds[0], mode.seeds[-1]],
        "selected_gain": STAGE2A_EVALUATION_SELECTED_GAIN,
        "preflight": mode.preflight,
    }
    if (
        _verify_internal_digest(
            started,
            digest_key="run_started_sha256",
            name="evaluation RUN_STARTED",
        )
        != started.get("run_started_sha256")
        or started.get("version")
        != E018_P1_STAGE2A_EVALUATION_EXECUTION_VERSION
        or started.get("experiment_id") != mode.experiment_id
        or started.get("classification") != mode.classification
        or started.get("status") != "PASS_A_IN_PROGRESS_NO_PRIVATE_GT"
        or started.get("seeds") != [mode.seeds[0], mode.seeds[-1]]
        or started.get("selected_gain") != STAGE2A_EVALUATION_SELECTED_GAIN
        or started.get("transaction_identity_sha256")
        != canonical_sha256(transaction_primitive)
        or started.get("private_label_capture_count") != 0
        or started.get("private_label_open_count") != 0
        or started.get("fresh_test_reads") != 0
        or (
            not mode.preflight
            and started.get("started_at_unix_ns", 0)
            <= parent["formal_execution_go"]["receipt"]["issued_at_unix_ns"]
        )
    ):
        raise RuntimeError("evaluation RUN_STARTED identity 漂移")
    provider_rows = _read_jsonl(
        root / "provider_output_ledger.jsonl",
        "evaluation provider ledger",
    )
    if len(provider_rows) != mode.prediction_count:
        raise RuntimeError("evaluation provider ledger count 漂移")
    receipts = _verify_evaluation_prediction_commit_chain(
        public_root=root,
        provider_rows=provider_rows,
        transaction_identity_sha256=started["transaction_identity_sha256"],
        mode=mode,
    )
    records = [_stage2a._provider_record_from_dict(row) for row in provider_rows]
    for index, record in enumerate(records):
        if (
            record.prediction.get("seed") != mode.seeds[index // 4]
            or record.route_frame_index
            != STAGE2A_PROVIDER_FRAME_INDICES[index % 4]
        ):
            raise RuntimeError("evaluation provider seed/frame order 漂移")
        verify_stage2a_provider_output_record(
            record,
            stage2_config=stage2a,
            qualification_config=qualification,
            expected_classification=(
                _stage2a.QUALIFICATION_CLASSIFICATION_SELECTION
            ),
        )
    camera_rows = _read_jsonl(
        root / "camera_pose_ledger.jsonl",
        "evaluation camera ledger",
    )
    route_rows = _read_jsonl(
        root / "route_evidence_ledger.jsonl",
        "evaluation route ledger",
    )
    branch_rows = _read_jsonl(
        root / "fixed_gain_branch_ledger.jsonl",
        "evaluation fixed-gain branch ledger",
    )
    if (
        len(camera_rows) != len(mode.seeds) * 92
        or len(route_rows) != len(mode.seeds)
        or len(branch_rows) != len(mode.seeds)
    ):
        raise RuntimeError("evaluation camera/route/branch count 漂移")
    captured = [
        _verify_evaluation_route_row(
            route,
            route_index=index,
            camera_rows=camera_rows,
            provider_records=records,
            prediction_receipts=receipts,
            stage2a_config=stage2a,
            qualification_config=qualification,
            mode=mode,
        )
        for index, route in enumerate(route_rows)
    ]
    verified_branches = [
        _validated_evaluation_branch(row, expected_seed=mode.seeds[index])
        for index, row in enumerate(branch_rows)
    ]
    for index, route in enumerate(captured):
        if replay_selected_gain_branch(route).to_dict() != verified_branches[index]:
            raise RuntimeError(
                f"evaluation route[{index}] fixed branch 不能纯逻辑重放"
            )
    receipt = _read_json(root / "execution_receipt.json", "evaluation receipt")
    producer = _selection_runtime._verify_process_identity(
        receipt.get("producer_process_identity"), role="pass-a-producer"
    )
    counts = {
        "seed_count": len(mode.seeds),
        "route_count": len(mode.seeds),
        "camera_frame_count": len(mode.seeds) * 92,
        "provider_forward_count": mode.prediction_count,
        "private_label_capture_count": 0,
        "private_label_open_count": 0,
        "fixed_gain_branch_count": len(mode.seeds),
        "branch_provider_forward_count": 0,
        "decision_change_count": 0,
        "arm_motion_command_count": 0,
        "gripper_close_command_count": 0,
        "runtime_object_gt_read_count": 0,
        "goal_gt_read_count": 0,
        "fresh_test_read_count": 0,
        "checkpoint_write_count": 0,
    }
    if (
        _verify_internal_digest(
            receipt,
            digest_key="receipt_sha256",
            name="evaluation execution receipt",
        )
        != receipt.get("receipt_sha256")
        or receipt.get("version")
        != E018_P1_STAGE2A_EVALUATION_EXECUTION_VERSION
        or receipt.get("status") != "PASS_A_COMPLETE_CONTEXT_DESTROYED"
        or receipt.get("classification") != mode.classification
        or receipt.get("experiment_id") != mode.experiment_id
        or receipt.get("config_raw_sha256") != loaded.raw_sha256
        or receipt.get("config_canonical_sha256") != loaded.canonical_sha256
        or receipt.get("source_identity_sha256") != source["identity_sha256"]
        or receipt.get("parent_verification_sha256")
        != parent["verification_sha256"]
        or receipt.get("conditional_parent_verification_sha256")
        != conditional_parent_sha256
        or receipt.get("formal_execution_go_verification_sha256")
        != formal_go_verification_sha256
        or receipt.get("transaction_identity_sha256")
        != started["transaction_identity_sha256"]
        or receipt.get("context_destroyed") is not True
        or receipt.get("provider_context_destroyed") is not True
        or receipt.get("environment_closed") is not True
        or receipt.get("pass_b_process_started") is not False
        or receipt.get("counts") != counts
        or receipt.get("formal_identity_consumed") is not (not mode.preflight)
        or receipt.get("fresh_test_status") != "prohibited-unread"
        or receipt.get("formal_claim_allowed") is not False
        or not isinstance(receipt.get("gpu_wall_seconds"), (int, float))
        or receipt["gpu_wall_seconds"] < 0.0
        or receipt["gpu_wall_seconds"]
        > loaded.payload["budgets"]["gpu_wall_seconds_max"]
    ):
        raise RuntimeError("evaluation execution receipt identity/accounting 漂移")
    freeze = _read_json(root / "execution_freeze.json", "evaluation freeze")
    frozen_inventory = freeze.get("artifact_inventory")
    expected_frozen_paths = {
        "RUN_STARTED.json",
        "config_snapshot.json",
        "source_identity.json",
        "parent_verification.json",
        "camera_pose_ledger.jsonl",
        "provider_output_ledger.jsonl",
        "route_evidence_ledger.jsonl",
        "fixed_gain_branch_ledger.jsonl",
        *{
            f"prediction_commits/{index:06d}.commit.json"
            for index in range(mode.prediction_count)
        },
    }
    if (
        _verify_internal_digest(
            freeze,
            digest_key="freeze_sha256",
            name="evaluation execution freeze",
        )
        != freeze.get("freeze_sha256")
        or freeze.get("version")
        != E018_P1_STAGE2A_EVALUATION_EXECUTION_VERSION
        or freeze.get("status") != "PASS_A_FROZEN_BEFORE_ANY_PRIVATE_GT"
        or freeze.get("context_destroyed") is not True
        or freeze.get("provider_context_destroyed") is not True
        or freeze.get("environment_closed") is not True
        or freeze.get("private_label_capture_count") != 0
        or freeze.get("private_label_open_count") != 0
        or freeze.get("provider_forward_count") != mode.prediction_count
        or freeze.get("fixed_gain_branch_count") != len(mode.seeds)
        or freeze.get("decision_change_count") != 0
        or freeze.get("producer_process_identity") != producer
        or freeze.get("parent_verification_sha256")
        != parent["verification_sha256"]
        or freeze.get("conditional_parent_verification_sha256")
        != conditional_parent_sha256
        or freeze.get("formal_execution_go_verification_sha256")
        != formal_go_verification_sha256
        or not isinstance(frozen_inventory, dict)
        or set(frozen_inventory) != expected_frozen_paths
    ):
        raise RuntimeError("evaluation execution freeze identity 漂移")
    for relative, identity in frozen_inventory.items():
        path = root / relative
        if (
            not isinstance(identity, dict)
            or set(identity) != {"raw_sha256", "size_bytes"}
            or not path.is_file()
            or path.is_symlink()
            or path.stat().st_nlink != 1
            or file_sha256(path) != identity["raw_sha256"]
            or path.stat().st_size != identity["size_bytes"]
        ):
            raise RuntimeError(f"evaluation frozen artifact 漂移: {relative}")
    precompletion = {
        "version": E018_P1_STAGE2A_EVALUATION_EXECUTION_VERSION,
        "verified": True,
        "complete_marker_verified": False,
        "preflight": mode.preflight,
        "experiment_id": mode.experiment_id,
        "config_raw_sha256": loaded.raw_sha256,
        "config_canonical_sha256": loaded.canonical_sha256,
        "source_git_commit": source["git_commit"],
        "source_tree_sha256": source["source_tree_sha256"],
        "source_identity_sha256": source["identity_sha256"],
        "parent_verification_sha256": parent["verification_sha256"],
        "conditional_parent_verification_sha256": conditional_parent_sha256,
        "formal_execution_go_verification_sha256": (
            formal_go_verification_sha256
        ),
        "execution_receipt_sha256": receipt["receipt_sha256"],
        "execution_freeze_sha256": freeze["freeze_sha256"],
        "provider_prediction_count": len(provider_rows),
        "route_count": len(route_rows),
        "fixed_gain_branch_count": len(branch_rows),
        "private_label_capture_count": 0,
        "private_label_open_count": 0,
        "fresh_test_reads": 0,
        "transaction_identity_sha256": started[
            "transaction_identity_sha256"
        ],
        "public_artifact_role_identity_sha256": started[
            "public_artifact_role_identity_sha256"
        ],
        "private_artifact_role_identity_sha256": started[
            "private_artifact_role_identity_sha256"
        ],
        "producer_process_identity": producer,
    }
    precompletion["verification_sha256"] = canonical_sha256(precompletion)
    if not require_complete:
        return precompletion
    marker = _read_json(
        root / "PUBLIC_EXECUTION_COMPLETE.json",
        "evaluation public completion marker",
    )
    if (
        _verify_internal_digest(
            marker,
            digest_key="marker_sha256",
            name="evaluation public completion marker",
        )
        != marker.get("marker_sha256")
        or marker.get("version")
        != E018_P1_STAGE2A_EVALUATION_EXECUTION_VERSION
        or marker.get("status") != "PUBLIC_EXECUTION_COMPLETE"
        or marker.get("experiment_id") != mode.experiment_id
        or marker.get("transaction_identity_sha256")
        != precompletion["transaction_identity_sha256"]
        or marker.get("precompletion_verification_sha256")
        != precompletion["verification_sha256"]
        or marker.get("execution_receipt_raw_sha256")
        != file_sha256(root / "execution_receipt.json")
        or marker.get("execution_receipt_internal_sha256")
        != receipt["receipt_sha256"]
        or marker.get("private_label_capture_count") != 0
        or marker.get("private_label_open_count") != 0
        or marker.get("fresh_test_reads") != 0
    ):
        raise RuntimeError("evaluation public completion marker 漂移")
    result = {
        **precompletion,
        "complete_marker_verified": True,
        "public_completion_marker_sha256": marker["marker_sha256"],
    }
    result.pop("verification_sha256")
    result["verification_sha256"] = canonical_sha256(result)
    return result


def verify_e018_p1_stage2a_evaluation_public(
    *,
    evaluation_config_path: str | Path,
    stage2a_config_path: str | Path,
    qualification_config_path: str | Path,
    public_root: str | Path,
    expected_source_git_commit: str,
    expected_source_identity_sha256: str,
    preflight: bool = False,
) -> dict[str, Any]:
    return _verify_e018_p1_stage2a_evaluation_public(
        evaluation_config_path=evaluation_config_path,
        stage2a_config_path=stage2a_config_path,
        qualification_config_path=qualification_config_path,
        public_root=public_root,
        expected_source_git_commit=expected_source_git_commit,
        expected_source_identity_sha256=expected_source_identity_sha256,
        preflight=preflight,
        require_complete=True,
    )


def _record_pass_a_failure(
    *,
    artifact_root: Path,
    error: BaseException,
    progress: Stage2AExecutionProgress,
    transaction_identity_sha256: str,
    mode: _EvaluationMode,
) -> None:
    failure = {
        "version": E018_P1_STAGE2A_EVALUATION_EXECUTION_VERSION,
        "status": "PASS_A_FAILED_IDENTITY_NOT_RERUNNABLE",
        "preflight": mode.preflight,
        "experiment_id": mode.experiment_id,
        "transaction_identity_sha256": transaction_identity_sha256,
        "progress": progress.as_dict(),
        "private_label_capture_count": 0,
        "private_label_open_count": 0,
        "fresh_test_reads": 0,
        "error_type": type(error).__name__,
        "message": str(error)[:1024],
        "traceback": "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )[-8192:],
        "failed_at_unix_ns": time.time_ns(),
    }
    failure["failure_sha256"] = canonical_sha256(failure)
    try:
        _atomic_create_json(artifact_root / "PASS_A_FAILURE.json", failure)
    except FileExistsError:
        pass


def run_e018_p1_stage2a_evaluation_capture(
    *,
    evaluation_config_path: str | Path,
    stage2a_config_path: str | Path,
    qualification_config_path: str | Path,
    g0c_config_path: str | Path,
    data_config_path: str | Path,
    selected_checkpoint_path: str | Path,
    stats_root: str | Path,
    selection_config_path: str | Path,
    selection_public_root: str | Path,
    selection_result_root: str | Path,
    decision_gate_path: str | Path,
    artifact_inventory_path: str | Path,
    repository_root: str | Path,
    artifact_root: str | Path,
    expected_config_raw_sha256: str,
    expected_config_canonical_sha256: str,
    expected_source_git_commit: str,
    expected_source_identity_sha256: str,
    exact_go_token: str,
    formal_go_receipt_path: str | Path | None = None,
    expected_formal_go_raw_sha256: str | None = None,
    expected_formal_go_internal_sha256: str | None = None,
    preflight_public_root: str | Path | None = None,
    preflight_result_root: str | Path | None = None,
    preflight: bool = False,
) -> dict[str, Any]:
    """Pass A：冻结 provider route 和 fixed-gain decision，GT capture=0。"""

    mode = _evaluation_mode(preflight)
    if exact_go_token != mode.go_token:
        raise PermissionError("evaluation Pass A 缺对应 formal/preflight exact GO")
    formal_authority_inputs = (
        formal_go_receipt_path,
        expected_formal_go_raw_sha256,
        expected_formal_go_internal_sha256,
        preflight_public_root,
        preflight_result_root,
    )
    if mode.preflight and any(item is not None for item in formal_authority_inputs):
        raise PermissionError("evaluation preflight 禁止携带 formal GO authority")
    if not mode.preflight and any(item is None for item in formal_authority_inputs):
        raise PermissionError("evaluation formal Pass A 缺独立 final-GO/preflight 证据")
    artifact = Path(artifact_root)
    if artifact.exists() or artifact.is_symlink():
        raise FileExistsError(f"evaluation artifact root 已存在: {artifact}")
    loaded = load_e018_p1_stage2a_evaluation_config(evaluation_config_path)
    if (
        loaded.raw_sha256 != expected_config_raw_sha256
        or loaded.canonical_sha256 != expected_config_canonical_sha256
    ):
        raise RuntimeError("evaluation expected config identity 漂移")
    source = _git_source_identity(Path(repository_root))
    if (
        source["git_commit"] != expected_source_git_commit
        or source["identity_sha256"] != expected_source_identity_sha256
    ):
        raise RuntimeError("evaluation exact-clean source identity 漂移")
    conditional_parent = verify_e018_p1_stage2a_evaluation_parent_gate(
        evaluation_config_path=evaluation_config_path,
        stage2a_config_path=stage2a_config_path,
        qualification_config_path=qualification_config_path,
        g0c_config_path=g0c_config_path,
        data_config_path=data_config_path,
        selected_checkpoint_path=selected_checkpoint_path,
        stats_root=stats_root,
        selection_config_path=selection_config_path,
        selection_public_root=selection_public_root,
        selection_result_root=selection_result_root,
        decision_gate_path=decision_gate_path,
        artifact_inventory_path=artifact_inventory_path,
    )
    formal_go_verification = None
    if not mode.preflight:
        formal_go_verification = verify_e018_p1_stage2a_evaluation_formal_go(
            formal_go_receipt_path=formal_go_receipt_path,
            expected_formal_go_raw_sha256=expected_formal_go_raw_sha256,
            expected_formal_go_internal_sha256=(
                expected_formal_go_internal_sha256
            ),
            evaluation_config_path=evaluation_config_path,
            stage2a_config_path=stage2a_config_path,
            qualification_config_path=qualification_config_path,
            preflight_public_root=preflight_public_root,
            preflight_result_root=preflight_result_root,
            source=source,
            conditional_parent_verification_sha256=conditional_parent[
                "verification_sha256"
            ],
            expected_execution_id=artifact.name,
            expected_worker_artifact_root=artifact,
        )
    parent = _build_parent_authorization_receipt(
        conditional_parent=conditional_parent,
        mode=mode,
        formal_execution_go=formal_go_verification,
    )
    stage2a = load_e018_p1_stage2a_config(stage2a_config_path)
    qualification = load_g2c_dynamic_qualification_config(
        qualification_config_path
    )
    g0c = _stage2a._g0c.load_e018_p1_g0c_config(g0c_config_path)
    data = _stage2a.load_e018_p1_g2c_data_config(
        data_config_path,
        parent_g0c_config_path=g0c_config_path,
    )
    producer = _selection_runtime._new_process_identity("pass-a-producer")
    public_root = artifact / "public_execution"
    public_identity = _evaluation_artifact_role_identity_sha256(
        role="public_execution",
        mode=mode,
        config_canonical_sha256=loaded.canonical_sha256,
        source_identity_sha256=source["identity_sha256"],
        conditional_parent_verification_sha256=conditional_parent[
            "verification_sha256"
        ],
    )
    private_identity = _evaluation_artifact_role_identity_sha256(
        role="private_labels",
        mode=mode,
        config_canonical_sha256=loaded.canonical_sha256,
        source_identity_sha256=source["identity_sha256"],
        conditional_parent_verification_sha256=conditional_parent[
            "verification_sha256"
        ],
    )
    transaction_primitive = {
        "version": E018_P1_STAGE2A_EVALUATION_EXECUTION_VERSION,
        "experiment_id": mode.experiment_id,
        "classification": mode.classification,
        "config_raw_sha256": loaded.raw_sha256,
        "config_canonical_sha256": loaded.canonical_sha256,
        "source_identity_sha256": source["identity_sha256"],
        "parent_verification_sha256": parent["verification_sha256"],
        "conditional_parent_verification_sha256": conditional_parent[
            "verification_sha256"
        ],
        "formal_execution_go_verification_sha256": (
            None
            if formal_go_verification is None
            else formal_go_verification["verification_sha256"]
        ),
        "public_artifact_role_identity_sha256": public_identity,
        "private_artifact_role_identity_sha256": private_identity,
        "seeds": [mode.seeds[0], mode.seeds[-1]],
        "selected_gain": STAGE2A_EVALUATION_SELECTED_GAIN,
        "preflight": mode.preflight,
    }
    transaction_identity = canonical_sha256(transaction_primitive)
    progress = Stage2AExecutionProgress()
    artifact.mkdir(mode=0o700, parents=True, exist_ok=False)
    try:
        journal = Stage2AEvaluationJournal(
            public_root=public_root,
            config_canonical_sha256=loaded.canonical_sha256,
            transaction_identity_sha256=transaction_identity,
            mode=mode,
        )
        snapshot = {
            "version": E018_P1_STAGE2A_EVALUATION_EXECUTION_VERSION,
            "config_raw_sha256": loaded.raw_sha256,
            "config_canonical_sha256": loaded.canonical_sha256,
            "config": loaded.payload,
        }
        snapshot["snapshot_sha256"] = canonical_sha256(snapshot)
        _atomic_create_json(public_root / "config_snapshot.json", snapshot)
        _atomic_create_json(public_root / "source_identity.json", source)
        _atomic_create_json(public_root / "parent_verification.json", parent)
        started_record = {
            **transaction_primitive,
            "status": "PASS_A_IN_PROGRESS_NO_PRIVATE_GT",
            "transaction_identity_sha256": transaction_identity,
            "private_label_capture_count": 0,
            "private_label_open_count": 0,
            "result_root_created": False,
            "gpu_wall_seconds_max": loaded.payload["budgets"][
                "gpu_wall_seconds_max"
            ],
            "fresh_test_reads": 0,
            "started_at_unix_ns": time.time_ns(),
        }
        started_record["run_started_sha256"] = canonical_sha256(
            started_record
        )
        _atomic_create_json(public_root / "RUN_STARTED.json", started_record)
        started_monotonic = time.monotonic()
    except Exception as error:
        _record_pass_a_failure(
            artifact_root=artifact,
            error=error,
            progress=progress,
            transaction_identity_sha256=transaction_identity,
            mode=mode,
        )
        raise
    try:
        (
            camera_rows,
            captured_routes,
            route_summaries,
            transactions,
            provider_records,
            route_receipts,
            environment_identity,
            context_destroyed,
        ) = _run_evaluation_pass_a_simulator(
            loaded_evaluation_config=loaded,
            loaded_stage2a_config=stage2a,
            qualification_config=qualification,
            g0c_config=g0c,
            data_config=data,
            stats_root=Path(stats_root),
            selected_checkpoint_path=Path(selected_checkpoint_path),
            public_root=public_root,
            journal=journal,
            execution_progress=progress,
            started_monotonic=started_monotonic,
            mode=mode,
        )
        if not context_destroyed:
            raise RuntimeError("evaluation context 未销毁，禁止 fixed-gain replay")
        provider_freeze = journal.freeze()
        branches = [
            replay_selected_gain_branch(route) for route in captured_routes
        ]
        if (
            len(branches) != len(mode.seeds)
            or any(branch.provider_forward_count != 0 for branch in branches)
        ):
            raise RuntimeError("evaluation fixed-gain branch accounting 漂移")
        route_rows = [
            _build_evaluation_route_evidence_row(
                mode=mode,
                route_index=index,
                camera_rows=camera_rows[index * 92 : (index + 1) * 92],
                provider_records=provider_records[index * 4 : (index + 1) * 4],
                prediction_receipts=route_receipts[index],
                capture_transaction=transactions[index],
                route_summary=route_summaries[index],
                captured_route=captured_routes[index],
            )
            for index in range(len(mode.seeds))
        ]
        _selection_runtime._atomic_jsonl(
            public_root / "camera_pose_ledger.jsonl", camera_rows
        )
        _selection_runtime._atomic_jsonl(
            public_root / "route_evidence_ledger.jsonl", route_rows
        )
        _selection_runtime._atomic_jsonl(
            public_root / "fixed_gain_branch_ledger.jsonl",
            [branch.to_dict() for branch in branches],
        )
        freeze_paths = sorted(
            [
                "RUN_STARTED.json",
                "config_snapshot.json",
                "source_identity.json",
                "parent_verification.json",
                "camera_pose_ledger.jsonl",
                "provider_output_ledger.jsonl",
                "route_evidence_ledger.jsonl",
                "fixed_gain_branch_ledger.jsonl",
            ]
            + [
                f"prediction_commits/{index:06d}.commit.json"
                for index in range(mode.prediction_count)
            ]
        )
        freeze = {
            "version": E018_P1_STAGE2A_EVALUATION_EXECUTION_VERSION,
            "status": "PASS_A_FROZEN_BEFORE_ANY_PRIVATE_GT",
            "context_destroyed": True,
            "provider_context_destroyed": True,
            "environment_closed": True,
            "prediction_ledger_frozen": True,
            "route_evidence_frozen": True,
            "fixed_gain_branches_frozen": True,
            "private_label_capture_count": 0,
            "private_label_open_count": 0,
            "provider_forward_count": provider_freeze["row_count"],
            "fixed_gain_branch_count": len(branches),
            "branch_provider_forward_count": 0,
            "decision_change_count": 0,
            "producer_process_identity": producer,
            "parent_verification_sha256": parent["verification_sha256"],
            "conditional_parent_verification_sha256": conditional_parent[
                "verification_sha256"
            ],
            "formal_execution_go_verification_sha256": (
                None
                if formal_go_verification is None
                else formal_go_verification["verification_sha256"]
            ),
            "artifact_inventory": _selection_runtime._file_inventory(
                public_root, freeze_paths
            ),
            "frozen_at_unix_ns": time.time_ns(),
        }
        freeze["freeze_sha256"] = canonical_sha256(freeze)
        _atomic_create_json(public_root / "execution_freeze.json", freeze)
        wall_seconds = time.monotonic() - started_monotonic
        counts = {
            "seed_count": len(mode.seeds),
            "route_count": len(route_rows),
            "camera_frame_count": len(camera_rows),
            "provider_forward_count": len(provider_records),
            "private_label_capture_count": 0,
            "private_label_open_count": 0,
            "fixed_gain_branch_count": len(branches),
            "branch_provider_forward_count": 0,
            "decision_change_count": 0,
            "arm_motion_command_count": sum(
                branch.arm_motion_command_count for branch in branches
            ),
            "gripper_close_command_count": sum(
                branch.gripper_close_command_count for branch in branches
            ),
            "runtime_object_gt_read_count": 0,
            "goal_gt_read_count": 0,
            "fresh_test_read_count": 0,
            "checkpoint_write_count": 0,
        }
        receipt = {
            "version": E018_P1_STAGE2A_EVALUATION_EXECUTION_VERSION,
            "status": "PASS_A_COMPLETE_CONTEXT_DESTROYED",
            "preflight": mode.preflight,
            "classification": mode.classification,
            "effect_claim": "no-effect-claim",
            "experiment_id": mode.experiment_id,
            "config_raw_sha256": loaded.raw_sha256,
            "config_canonical_sha256": loaded.canonical_sha256,
            "source_identity_sha256": source["identity_sha256"],
            "parent_verification_sha256": parent["verification_sha256"],
            "conditional_parent_verification_sha256": conditional_parent[
                "verification_sha256"
            ],
            "formal_execution_go_verification_sha256": (
                None
                if formal_go_verification is None
                else formal_go_verification["verification_sha256"]
            ),
            "transaction_identity_sha256": transaction_identity,
            "context_destroyed": True,
            "provider_context_destroyed": True,
            "environment_closed": True,
            "pass_b_process_started": False,
            "counts": counts,
            "gpu_wall_seconds": wall_seconds,
            "gpu_wall_seconds_max": loaded.payload["budgets"][
                "gpu_wall_seconds_max"
            ],
            "gpu_budget_passed": wall_seconds
            <= loaded.payload["budgets"]["gpu_wall_seconds_max"],
            "environment_identity": environment_identity,
            "producer_process_identity": producer,
            "execution_freeze_raw_sha256": file_sha256(
                public_root / "execution_freeze.json"
            ),
            "execution_freeze_internal_sha256": freeze["freeze_sha256"],
            "formal_identity_consumed": not mode.preflight,
            "formal_claim_allowed": False,
            "fresh_test_status": "prohibited-unread",
        }
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        _atomic_create_json(public_root / "execution_receipt.json", receipt)
        artifact_budget = loaded.payload["budgets"][
            "combined_artifact_bytes_max"
        ]
        if (
            _selection_runtime._combined_artifact_bytes(artifact)
            + 1_048_576
            > artifact_budget
        ):
            raise RuntimeError("evaluation artifact budget 缺 completion 余量")
        precompletion = _verify_e018_p1_stage2a_evaluation_public(
            evaluation_config_path=evaluation_config_path,
            stage2a_config_path=stage2a_config_path,
            qualification_config_path=qualification_config_path,
            public_root=public_root,
            expected_source_git_commit=expected_source_git_commit,
            expected_source_identity_sha256=expected_source_identity_sha256,
            preflight=mode.preflight,
            require_complete=False,
        )
        marker = {
            "version": E018_P1_STAGE2A_EVALUATION_EXECUTION_VERSION,
            "status": "PUBLIC_EXECUTION_COMPLETE",
            "experiment_id": mode.experiment_id,
            "transaction_identity_sha256": transaction_identity,
            "precompletion_verification_sha256": precompletion[
                "verification_sha256"
            ],
            "execution_receipt_raw_sha256": file_sha256(
                public_root / "execution_receipt.json"
            ),
            "execution_receipt_internal_sha256": receipt["receipt_sha256"],
            "private_label_capture_count": 0,
            "private_label_open_count": 0,
            "fresh_test_reads": 0,
            "completed_at_unix_ns": time.time_ns(),
        }
        marker["marker_sha256"] = canonical_sha256(marker)
        _atomic_create_json(
            public_root / "PUBLIC_EXECUTION_COMPLETE.json", marker
        )
        return verify_e018_p1_stage2a_evaluation_public(
            evaluation_config_path=evaluation_config_path,
            stage2a_config_path=stage2a_config_path,
            qualification_config_path=qualification_config_path,
            public_root=public_root,
            expected_source_git_commit=expected_source_git_commit,
            expected_source_identity_sha256=expected_source_identity_sha256,
            preflight=mode.preflight,
        )
    except Exception as error:
        _record_pass_a_failure(
            artifact_root=artifact,
            error=error,
            progress=progress,
            transaction_identity_sha256=transaction_identity,
            mode=mode,
        )
        raise


def _verify_replay_frame_binding(
    *,
    replay_row: Mapping[str, Any],
    public_row: Mapping[str, Any],
    replay_prefix_rows: Sequence[Mapping[str, Any]],
    expected_action_prefix_sha256: str,
    rgb: np.ndarray,
) -> dict[str, str]:
    """逐帧绑定 action prefix、RGB 与 actual pose 的 raw/canonical identity。"""

    replay_action_sha = _action_prefix_sha256s(replay_prefix_rows)[-1]
    replay_rgb_sha = hashlib.sha256(
        np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8)).tobytes()
    ).hexdigest()
    replay_pose = np.asarray(
        replay_row["actual_base_from_external_camera_cv"], dtype=np.float64
    )
    public_pose = np.asarray(
        public_row["actual_base_from_external_camera_cv"], dtype=np.float64
    )
    replay_pose_raw = _array_sha256(replay_pose)
    public_pose_raw = _array_sha256(public_pose)
    replay_pose_canonical = canonical_sha256(
        replay_row["actual_base_from_external_camera_cv"]
    )
    public_pose_canonical = canonical_sha256(
        public_row["actual_base_from_external_camera_cv"]
    )
    if (
        replay_row.get("episode_id") != public_row.get("episode_id")
        or replay_row.get("request_id") != public_row.get("request_id")
        or replay_row.get("frame_index") != public_row.get("frame_index")
        or replay_row.get("control_tick") != public_row.get("control_tick")
        or replay_action_sha != expected_action_prefix_sha256
        or replay_rgb_sha != replay_row.get("rgb_sha256")
        or replay_rgb_sha != public_row.get("rgb_sha256")
        or canonical_sha256(dict(replay_row))
        != canonical_sha256(dict(public_row))
        or replay_pose.shape != (4, 4)
        or not np.isfinite(replay_pose).all()
        or replay_pose_raw != public_pose_raw
        or replay_pose_canonical != public_pose_canonical
    ):
        raise RuntimeError("evaluation deterministic frame/action/RGB/pose replay 漂移")
    return {
        "rgb_sha256": replay_rgb_sha,
        "actual_pose_sha256": replay_pose_raw,
        "actual_pose_canonical_sha256": replay_pose_canonical,
        "replay_camera_row_sha256": canonical_sha256(dict(public_row)),
        "action_prefix_sha256": replay_action_sha,
    }


def _normalize_and_verify_replay_frame_binding(
    *,
    replay_row: dict[str, Any],
    public_row: Mapping[str, Any],
    replay_prefix_rows: Sequence[Mapping[str, Any]],
    expected_action_prefix_sha256: str,
    rgb: np.ndarray,
) -> dict[str, str]:
    """先复用 Pass A 视角规范化，再执行原有严格 replay 绑定。"""

    _stage2a._normalize_stage2a_motion_row_viewpoint(replay_row)
    return _verify_replay_frame_binding(
        replay_row=replay_row,
        public_row=public_row,
        replay_prefix_rows=replay_prefix_rows,
        expected_action_prefix_sha256=expected_action_prefix_sha256,
        rgb=rgb,
    )


def _run_deterministic_private_label_replay(
    *,
    mode: _EvaluationMode,
    loaded_evaluation_config: Any,
    g0c_config: dict[str, Any],
    data_config: Mapping[str, Any],
    stats_root: Path,
    public_camera_rows: Sequence[Mapping[str, Any]],
    public_provider_records: Sequence[Stage2AProviderOutputRecord],
    public_route_rows: Sequence[Mapping[str, Any]],
    transaction_identity_sha256: str,
    prediction_receipts: Sequence[Mapping[str, Any]],
    before_gt_capture: Any,
    after_gt_capture: Any,
    started_monotonic: float,
    replay_output_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, int]]:
    """无 checkpoint/provider 的 deterministic replay；回调负责 durable label。"""

    import gymnasium as gym
    import mani_skill
    import sapien
    import torch
    from mani_skill.utils import sapien_utils

    from robot_vla.sim import register_robot_vla_maniskill_envs

    if (
        mani_skill.__version__
        != data_config["software"]["expected_mani_skill_version"]
        or sapien.__version__
        != data_config["software"]["expected_sapien_version"]
        or not torch.cuda.is_available()
    ):
        raise RuntimeError("evaluation Pass B simulator environment 漂移")
    spec, proprio, force, normalizer_identity = _stage2a._load_normalizers(
        stats_root=stats_root,
        config=data_config,
    )
    env: Any | None = None
    env_closed = False
    sensor: Any | None = None
    camera: Any | None = None
    labels: list[dict[str, Any]] = []
    counts = {
        "camera_frame_match_count": 0,
        "action_prefix_match_count": 0,
        "rgb_match_count": 0,
        "actual_pose_raw_match_count": 0,
        "actual_pose_canonical_match_count": 0,
        "model_input_digest_match_count": 0,
        "provider_output_digest_match_count": 0,
        "private_label_capture_count": 0,
        "provider_forward_count": 0,
        "checkpoint_load_count": 0,
        "decision_change_count": 0,
    }
    try:
        home, anchors, orientations = _stage2a._g0c._parse_library(g0c_config)
        primitives = _stage2a._g0c._expand_primitives(anchors, orientations)
        by_id = {
            item.viewpoint_id: (item, orientation)
            for item, orientation in primitives
        }
        if tuple(by_id) != _stage2a.FRONT_ALTERNATE_IDS:
            raise RuntimeError("evaluation Pass B G0C primitive order 漂移")
        primary, primary_orientation = by_id[
            _stage2a.ACTIVE_FRONT_PRIMARY_PRIMITIVE_ID
        ]
        route_config = json.loads(json.dumps(g0c_config))
        route_config["experiment"]["offline_segmentation_diagnostics"] = False
        route_config["experiment"]["save_settled_rgb"] = False
        register_robot_vla_maniskill_envs()
        environment = route_config["environment"]
        env = gym.make(
            environment["environment_id"],
            obs_mode=environment["obs_mode"],
            control_mode=environment["control_mode"],
            num_envs=environment["num_envs"],
            robot_uids=environment["robot_uid"],
        )
        base_env = env.unwrapped
        if (
            base_env.control_freq != environment["control_hz"]
            or environment["camera_uid"] not in base_env._sensors
        ):
            raise RuntimeError("evaluation Pass B environment identity 漂移")
        sensor = base_env._sensors[environment["camera_uid"]]
        camera = sensor.camera
        if sensor.entity is not None or not callable(
            getattr(camera, "set_local_pose", None)
        ):
            raise RuntimeError("evaluation Pass B 要求 unmounted external camera")
        for route_index, seed in enumerate(mode.seeds):
            if (
                time.monotonic() - started_monotonic
                > loaded_evaluation_config.payload["budgets"][
                    "gpu_wall_seconds_max"
                ]
            ):
                raise TimeoutError("evaluation Pass B GPU wall budget 已到")
            episode = (
                "e018-p1-stage2a-selected-gain-evaluation-recovery-preflight-"
                f"seed-{seed}"
                if mode.preflight
                else f"e018-p1-stage2a-selected-gain-evaluation-seed-{seed}"
            )
            request = f"{episode}-active-front-01"
            public_route = public_route_rows[route_index]
            route_public_camera = public_camera_rows[
                route_index * 92 : (route_index + 1) * 92
            ]
            replay_prefix: list[Mapping[str, Any]] = []
            provider_index_by_frame = {
                frame: route_index * 4 + local
                for local, frame in enumerate(STAGE2A_PROVIDER_FRAME_INDICES)
            }

            def frame_hook(
                row: dict[str, Any],
                rgb: np.ndarray,
                observation: Mapping[str, Any],
                _route_index: int = route_index,
                _seed: int = seed,
                _route_public_camera: Sequence[Mapping[str, Any]] = (
                    route_public_camera
                ),
                _replay_prefix: list[Mapping[str, Any]] = replay_prefix,
                _public_route: Mapping[str, Any] = public_route,
                _provider_index_by_frame: Mapping[int, int] = (
                    provider_index_by_frame
                ),
            ) -> None:
                # Pass A 在冻结 route row 前会把 G0 的物理视角 ID 规范为
                # Stage 2A 逻辑 primitive。Pass B 必须走同一规范化，再比较
                # action-prefix、整行与 pose/RGB identity；这里不放宽任何绑定。
                frame_index = int(row["frame_index"])
                public_row = _route_public_camera[frame_index]
                _replay_prefix.append(row)
                binding = _normalize_and_verify_replay_frame_binding(
                    replay_row=row,
                    public_row=public_row,
                    replay_prefix_rows=_replay_prefix,
                    expected_action_prefix_sha256=_public_route[
                        "action_prefix_sha256s"
                    ][frame_index],
                    rgb=rgb,
                )
                counts["camera_frame_match_count"] += 1
                counts["action_prefix_match_count"] += 1
                counts["rgb_match_count"] += 1
                counts["actual_pose_raw_match_count"] += 1
                counts["actual_pose_canonical_match_count"] += 1
                if frame_index not in _provider_index_by_frame:
                    return
                prediction_index = _provider_index_by_frame[frame_index]
                record = public_provider_records[prediction_index]
                identity = _stage2a._stage2a_capture_identity(
                    seed=_seed,
                    route_frame_index=frame_index,
                    row_index=prediction_index,
                )
                capture = build_qualification_deployable_capture(
                    identity=identity,
                    motion_row=row,
                    rgb=rgb,
                    base_env=base_env,
                    spec=spec,
                    proprio_normalizer=proprio,
                    finger_force_normalizer=force,
                    data_config=data_config,
                    eligible_capture_frame_indices=STAGE2A_PROVIDER_FRAME_INDICES,
                )
                if (
                    capture["input_sha256"] != record.model_input_digest
                    or record.provider_output_digest
                    != _public_route["provider_output_digests"][
                        STAGE2A_PROVIDER_FRAME_INDICES.index(frame_index)
                    ]
                    or record.model_input_digest
                    != _public_route["model_input_digests"][
                        STAGE2A_PROVIDER_FRAME_INDICES.index(frame_index)
                    ]
                ):
                    raise RuntimeError(
                        "evaluation Pass B model-input/provider digest replay 漂移"
                    )
                counts["model_input_digest_match_count"] += 1
                counts["provider_output_digest_match_count"] += 1
                if frame_index not in STAGE2A_COLLECT_FRAME_INDICES:
                    return
                label_index = _route_index * 3 + STAGE2A_COLLECT_FRAME_INDICES.index(
                    frame_index
                )
                before_gt_capture(label_index)
                privileged = _selection_runtime.capture_selection_private_label(
                    observation=observation,
                    base_env=base_env,
                    prediction=record.prediction,
                    data_config=data_config,
                )
                label = {
                    **privileged,
                    "version": E018_P1_STAGE2A_EVALUATION_EXECUTION_VERSION,
                    "label_index": label_index,
                    "prediction_row_index": prediction_index,
                    "seed": _seed,
                    "route_frame_index": frame_index,
                    "rgb_sha256": binding["rgb_sha256"],
                    "actual_pose_sha256": binding["actual_pose_sha256"],
                    "actual_pose_canonical_sha256": binding[
                        "actual_pose_canonical_sha256"
                    ],
                    "model_input_digest": record.model_input_digest,
                    "provider_output_digest": record.provider_output_digest,
                    "prediction_commit_receipt_sha256": prediction_receipts[
                        prediction_index
                    ]["commit_receipt_sha256"],
                    "transaction_identity_sha256": transaction_identity_sha256,
                    "replay_camera_row_sha256": binding[
                        "replay_camera_row_sha256"
                    ],
                    "motion_predicate_version": "pick-and-place-predicates/v1",
                    "motion_linear_threshold_m_s": 0.01,
                    "motion_angular_threshold_rad_s": 0.5,
                    "contact_threshold_n": 0.01,
                    "privileged_captured_at_unix_ns": time.time_ns(),
                }
                label["label_sha256"] = canonical_sha256(label)
                validate_evaluation_private_label(
                    label,
                    expected_label_index=label_index,
                    seeds=mode.seeds,
                )
                after_gt_capture(label)
                labels.append(label)
                privileged.clear()
                counts["private_label_capture_count"] += 1

            replay_rows, _, _ = _stage2a._g0._run_route(
                env=env,
                base_env=base_env,
                camera=camera,
                config=route_config,
                seed=seed,
                home=home,
                alternate=primary,
                output_root=replay_output_root,
                sapien_module=sapien,
                sapien_utils_module=sapien_utils,
                alternate_orientation=primary_orientation,
                result_version=E018_P1_STAGE2A_EXECUTION_VERSION,
                episode_prefix="stage2a-selected-gain-evaluation-replay",
                source_phase=_stage2a.STAGE2A_SOURCE_PHASE.value,
                camera_owner=_stage2a.STAGE2A_CAMERA_OWNER,
                frame_hook=frame_hook,
                episode_id_override=episode,
                request_id_override=request,
                command_sequence_id_override=(
                    f"{episode}-active-front-01-camera-command-00"
                ),
                include_raw_safety_witnesses=True,
                include_raw_proprio_velocity_witness=True,
                include_privileged_object_state_witnesses=False,
                include_robot_object_contact_witnesses=False,
            )
            if (
                len(replay_rows) != 92
                or len(replay_prefix) != 92
                or _action_prefix_sha256s(replay_rows)
                != public_route["action_prefix_sha256s"]
            ):
                raise RuntimeError("evaluation Pass B full action-prefix replay 漂移")
    finally:
        if env is not None:
            env.close()
            env_closed = True
    if len(labels) != mode.label_count or not env_closed:
        raise RuntimeError("evaluation Pass B label/environment accounting 漂移")
    environment_identity = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(torch.device("cuda")),
        "mani_skill": mani_skill.__version__,
        "sapien": sapien.__version__,
        "external_camera_sensor_class": (
            None
            if sensor is None
            else type(sensor).__module__ + "." + type(sensor).__name__
        ),
        "external_camera_class": (
            None
            if camera is None
            else type(camera).__module__ + "." + type(camera).__name__
        ),
        "external_camera_unmounted": bool(
            sensor is not None and sensor.entity is None
        ),
        "environment_closed": env_closed,
        "normalizer_identity": normalizer_identity,
    }
    return labels, environment_identity, counts


def _validate_scored_evaluation_row(
    value: Mapping[str, Any],
    *,
    row_index: int,
    branch: Mapping[str, Any],
    mode: _EvaluationMode,
    private_labels: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    row = _require_exact_keys(
        dict(value),
        {
            "version",
            "seed",
            "selected_gain",
            "oracle_recoverable_eligible",
            "memory_commit_count",
            "navigation_state_available",
            "fresh_shadow_action_generation_count",
            "xyz_error_m",
            "recovered",
            "false_recovery",
            "catastrophic_recovery",
            "unsafe_recovery",
            "protocol_violation_count",
            "oracle_label_primitive_sha256s",
            "scored_row_sha256",
        },
        f"evaluation scored row[{row_index}]",
    )
    unsigned = dict(row)
    stored = unsigned.pop("scored_row_sha256")
    error = row["xyz_error_m"]
    if error is not None:
        if (
            not isinstance(error, (int, float))
            or isinstance(error, bool)
            or not math.isfinite(float(error))
            or float(error) < 0.0
        ):
            raise RuntimeError("evaluation scored XYZ error 非法")
        error = float(error)
    for name in (
        "oracle_recoverable_eligible",
        "navigation_state_available",
        "recovered",
        "false_recovery",
        "catastrophic_recovery",
        "unsafe_recovery",
    ):
        if type(row[name]) is not bool:
            raise RuntimeError(f"evaluation scored {name} 必须是 bool")
    committed = row["memory_commit_count"] == 1
    expected_recovered = bool(
        row["oracle_recoverable_eligible"]
        and committed
        and row["navigation_state_available"]
        and row["fresh_shadow_action_generation_count"] == 1
        and error is not None
        and error <= 0.005
    )
    expected_false = bool(
        committed
        and (
            not row["oracle_recoverable_eligible"]
            or error is None
            or error > 0.005
        )
    )
    expected_catastrophic = bool(
        committed and error is not None and error > 0.020
    )
    hashes = row["oracle_label_primitive_sha256s"]
    if private_labels is not None:
        if len(private_labels) != 3:
            raise RuntimeError("evaluation independent scoring 必须有三个 label")
        verified_labels = [
            validate_evaluation_private_label(
                label,
                expected_label_index=row_index * 3 + local_index,
                seeds=mode.seeds,
            )
            for local_index, label in enumerate(private_labels)
        ]
        expected_oracle = bool(
            branch["route_protocol_safety_valid"]
            and all(label["gt_object_exists"] is True for label in verified_labels)
            and all(label["gt_observable"] is True for label in verified_labels)
            and all(
                float(label["robot_object_contact_force_n"]) <= 0.01
                for label in verified_labels
            )
            and all(
                label["object_motion_event"] is False
                for label in verified_labels
            )
            and all(label["is_grasped"] is False for label in verified_labels)
        )
        expected_error: float | None = None
        if committed:
            predicted = np.asarray(
                branch["committed_position_base_m"], dtype=np.float64
            )
            target = np.asarray(
                verified_labels[-1]["gt_object_position_base_m"],
                dtype=np.float64,
            )
            if predicted.shape != (3,) or target.shape != (3,):
                raise RuntimeError("evaluation independent scoring XYZ shape 漂移")
            expected_error = float(np.linalg.norm(predicted - target))
        expected_recovered_from_labels = bool(
            expected_oracle
            and committed
            and branch["navigation_state_available"] is True
            and branch["fresh_shadow_action_generation_count"] == 1
            and expected_error is not None
            and expected_error <= 0.005
        )
        expected_false_from_labels = bool(
            committed
            and (
                not expected_oracle
                or expected_error is None
                or expected_error > 0.005
            )
        )
        expected_catastrophic_from_labels = bool(
            committed
            and expected_error is not None
            and expected_error > 0.020
        )
        expected_unsafe_from_labels = bool(
            expected_false_from_labels
            or branch["arm_motion_command_count"] != 0
            or branch["gripper_close_command_count"] != 0
        )
        expected_hashes = [
            canonical_sha256(
                {
                    key: label[key]
                    for key in (
                        "gt_object_exists",
                        "gt_observable",
                        "gt_object_position_base_m",
                        "robot_object_contact_force_n",
                        "object_linear_speed_m_s",
                        "object_angular_speed_rad_s",
                        "object_motion_event",
                        "is_grasped",
                    )
                }
            )
            for label in verified_labels
        ]
        if (
            row["oracle_recoverable_eligible"] is not expected_oracle
            or row["xyz_error_m"] != expected_error
            or row["recovered"] is not expected_recovered_from_labels
            or row["false_recovery"] is not expected_false_from_labels
            or row["catastrophic_recovery"]
            is not expected_catastrophic_from_labels
            or row["unsafe_recovery"] is not expected_unsafe_from_labels
            or row["oracle_label_primitive_sha256s"] != expected_hashes
        ):
            raise RuntimeError(
                f"evaluation scored row[{row_index}] private-label derivation 漂移"
            )
    if (
        stored != canonical_sha256(unsigned)
        or row["version"] != E018_P1_STAGE2A_EVALUATION_RESULT_VERSION
        or row["seed"] != mode.seeds[row_index]
        or row["selected_gain"] != STAGE2A_EVALUATION_SELECTED_GAIN
        or type(row["memory_commit_count"]) is not int
        or row["memory_commit_count"] not in {0, 1}
        or row["memory_commit_count"] != branch["memory_commit_count"]
        or row["navigation_state_available"]
        is not branch["navigation_state_available"]
        or row["fresh_shadow_action_generation_count"]
        != branch["fresh_shadow_action_generation_count"]
        or row["protocol_violation_count"]
        != branch["protocol_violation_count"]
        or (not committed and error is not None)
        or (committed and (error is None or not row["navigation_state_available"]))
        or row["recovered"] is not expected_recovered
        or row["false_recovery"] is not expected_false
        or row["catastrophic_recovery"] is not expected_catastrophic
        or row["unsafe_recovery"] is not expected_false
        or not isinstance(hashes, list)
        or len(hashes) != 3
        or any(not _selection_runtime._is_sha256(value) for value in hashes)
    ):
        raise RuntimeError(f"evaluation scored row[{row_index}] mechanics 漂移")
    return row


def _recompute_evaluation_summary(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    support = sum(row["oracle_recoverable_eligible"] for row in rows)
    recovered = sum(row["recovered"] for row in rows)
    false_count = sum(row["false_recovery"] for row in rows)
    catastrophic = sum(row["catastrophic_recovery"] for row in rows)
    unsafe = sum(row["unsafe_recovery"] for row in rows)
    protocol = sum(row["protocol_violation_count"] for row in rows)
    rate = None if support == 0 else recovered / support
    if any(value != 0 for value in (false_count, catastrophic, unsafe, protocol)):
        classification = (
            "safety-negative-persist-publish-pause-for-reusability-refactor"
        )
    elif support < 10:
        classification = (
            "insufficient-support-inconclusive-persist-publish-pause-for-"
            "reusability-refactor"
        )
    elif 10 * recovered >= 7 * support:
        classification = (
            "development-absolute-recovery-pass-no-effect-no-actuation-"
            "persist-publish-pause"
        )
    else:
        classification = (
            "effect-negative-persist-publish-pause-for-reusability-refactor"
        )
    summary = {
        "version": E018_P1_STAGE2A_EVALUATION_RESULT_VERSION,
        "status": "complete-selected-gain-development-evaluation",
        "classification": classification,
        "effect_claim": "no-effect-claim",
        "selected_gain": STAGE2A_EVALUATION_SELECTED_GAIN,
        "gain_reselection_performed": False,
        "oracle_recoverable_support": support,
        "minimum_support_required": 10,
        "recovered_count": recovered,
        "recovery_rate": rate,
        "minimum_recovery_rate_required": 0.70,
        "false_recovery_count": false_count,
        "catastrophic_recovery_count": catastrophic,
        "unsafe_recovery_count": unsafe,
        "protocol_violation_count": protocol,
        "stage2b_continuation_required": False,
        "fresh_test_reads": 0,
        "runtime_object_gt_reads": 0,
        "goal_gt_reads": 0,
        "arm_motion_command_count": 0,
        "gripper_close_command_count": 0,
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    return summary


def _verify_e018_p1_stage2a_evaluation_result(
    *,
    evaluation_config_path: str | Path,
    stage2a_config_path: str | Path,
    qualification_config_path: str | Path,
    public_root: str | Path,
    result_root: str | Path,
    expected_source_git_commit: str,
    expected_source_identity_sha256: str,
    preflight: bool,
    require_complete: bool,
) -> dict[str, Any]:
    """结果验证器不接受 private、checkpoint、provider 或 stats root。"""

    public = verify_e018_p1_stage2a_evaluation_public(
        evaluation_config_path=evaluation_config_path,
        stage2a_config_path=stage2a_config_path,
        qualification_config_path=qualification_config_path,
        public_root=public_root,
        expected_source_git_commit=expected_source_git_commit,
        expected_source_identity_sha256=expected_source_identity_sha256,
        preflight=preflight,
    )
    loaded = load_e018_p1_stage2a_evaluation_config(evaluation_config_path)
    mode = _evaluation_mode(preflight)
    public_path = Path(public_root)
    root = Path(result_root)
    _selection_runtime._require_common_artifact_parent(
        public_root=public_path,
        result_root=root,
    )
    _selection_runtime._assert_exact_tree(
        root,
        expected_files=(
            _RESULT_FILES if require_complete else _RESULT_PRECOMPLETION_FILES
        ),
        name="evaluation result artifact",
    )
    branches = _read_jsonl(
        public_path / "fixed_gain_branch_ledger.jsonl",
        "evaluation public fixed-gain branches",
    )
    verified_branches = [
        _validated_evaluation_branch(branch, expected_seed=mode.seeds[index])
        for index, branch in enumerate(branches)
    ]
    scored_path = root / "scored_fixed_gain_routes.jsonl"
    raw_lines = scored_path.read_bytes().splitlines(keepends=True)
    scored_values = _read_jsonl(scored_path, "evaluation scored routes")
    if (
        len(raw_lines) != len(mode.seeds)
        or len(scored_values) != len(mode.seeds)
        or any(not line.endswith(b"\n") for line in raw_lines)
        or any(
            line != _selection_runtime._jsonl_line_bytes(value)
            for line, value in zip(raw_lines, scored_values, strict=True)
        )
    ):
        raise RuntimeError("evaluation scored ledger count/serialization 漂移")
    scored = [
        _validate_scored_evaluation_row(
            value,
            row_index=index,
            branch=verified_branches[index],
            mode=mode,
        )
        for index, value in enumerate(scored_values)
    ]
    summary_path = root / "evaluation_summary.json"
    summary = _read_json(summary_path, "evaluation summary")
    if summary != _recompute_evaluation_summary(scored):
        raise RuntimeError("evaluation summary 不能从 scored rows 独立重算")
    receipt_path = root / "result_receipt.json"
    receipt = _read_json(receipt_path, "evaluation result receipt")
    producer = _selection_runtime._verify_process_identity(
        receipt.get("producer_process_identity"), role="pass-a-producer"
    )
    scorer = _selection_runtime._verify_process_identity(
        receipt.get("scorer_process_identity"), role="pass-b-scorer"
    )
    expected_counts = {
        "private_label_capture_started_count": mode.label_count,
        "private_label_capture_completed_count": mode.label_count,
        "provider_forward_count": 0,
        "checkpoint_load_count": 0,
        "decision_change_count": 0,
        "fixed_gain_branch_count": len(mode.seeds),
        "fresh_test_read_count": 0,
        "runtime_object_gt_read_count": 0,
        "goal_gt_read_count": 0,
        "arm_motion_command_count": 0,
        "gripper_close_command_count": 0,
    }
    expected_replay_counts = {
        "camera_frame_match_count": len(mode.seeds) * 92,
        "action_prefix_match_count": len(mode.seeds) * 92,
        "rgb_match_count": len(mode.seeds) * 92,
        "actual_pose_raw_match_count": len(mode.seeds) * 92,
        "actual_pose_canonical_match_count": len(mode.seeds) * 92,
        "model_input_digest_match_count": mode.prediction_count,
        "provider_output_digest_match_count": mode.prediction_count,
        "private_label_capture_count": mode.label_count,
        "independent_scored_route_validation_count": len(mode.seeds),
        "provider_forward_count": 0,
        "checkpoint_load_count": 0,
        "decision_change_count": 0,
    }
    replay_environment = receipt.get("replay_environment_identity")
    public_environment = _read_json(
        public_path / "execution_receipt.json",
        "evaluation public receipt for replay environment",
    ).get("environment_identity")
    replay_environment_keys = {
        "python",
        "numpy",
        "torch",
        "cuda_available",
        "cuda_device",
        "mani_skill",
        "sapien",
        "external_camera_sensor_class",
        "external_camera_class",
        "external_camera_unmounted",
        "environment_closed",
        "normalizer_identity",
    }
    shared_environment_keys = replay_environment_keys - {"environment_closed"}
    if (
        receipt.get("replay_counts") != expected_replay_counts
        or not isinstance(replay_environment, dict)
        or set(replay_environment) != replay_environment_keys
        or not isinstance(public_environment, dict)
        or replay_environment["environment_closed"] is not True
        or replay_environment["external_camera_unmounted"] is not True
        or any(
            replay_environment[name] != public_environment.get(name)
            for name in shared_environment_keys
        )
    ):
        raise RuntimeError("evaluation replay counts/environment identity 漂移")
    result_role_identity = _evaluation_artifact_role_identity_sha256(
        role="result",
        mode=mode,
        config_canonical_sha256=loaded.canonical_sha256,
        source_identity_sha256=public["source_identity_sha256"],
        conditional_parent_verification_sha256=public[
            "conditional_parent_verification_sha256"
        ],
    )
    if (
        _verify_internal_digest(
            receipt,
            digest_key="receipt_sha256",
            name="evaluation result receipt",
        )
        != receipt.get("receipt_sha256")
        or receipt.get("version") != E018_P1_STAGE2A_EVALUATION_RESULT_VERSION
        or receipt.get("status") != "PASS_B_COMPLETE_EXACT_ONCE"
        or receipt.get("preflight") is not mode.preflight
        or receipt.get("classification") != mode.classification
        or receipt.get("experiment_id") != mode.experiment_id
        or receipt.get("config_raw_sha256") != loaded.raw_sha256
        or receipt.get("config_canonical_sha256") != loaded.canonical_sha256
        or receipt.get("source_identity_sha256")
        != public["source_identity_sha256"]
        or receipt.get("parent_verification_sha256")
        != public["parent_verification_sha256"]
        or receipt.get("conditional_parent_verification_sha256")
        != public["conditional_parent_verification_sha256"]
        or receipt.get("formal_execution_go_verification_sha256")
        != public["formal_execution_go_verification_sha256"]
        or receipt.get("transaction_identity_sha256")
        != public["transaction_identity_sha256"]
        or receipt.get("public_verification_sha256")
        != public["verification_sha256"]
        or receipt.get("public_completion_marker_sha256")
        != public["public_completion_marker_sha256"]
        or receipt.get("result_artifact_role_identity_sha256")
        != result_role_identity
        or receipt.get("producer_process_identity") != producer
        or receipt.get("scorer_process_identity") != scorer
        or producer["process_instance_sha256"]
        == scorer["process_instance_sha256"]
        or receipt.get("process_boundary_verified") is not True
        or not _selection_runtime._is_sha256(
            receipt.get("consumption_marker_raw_sha256")
        )
        or not _selection_runtime._is_sha256(
            receipt.get("consumption_marker_internal_sha256")
        )
        or receipt.get("rerun_under_same_identity_allowed") is not False
        or receipt.get("scored_ledger_raw_sha256") != file_sha256(scored_path)
        or receipt.get("scored_row_count") != len(mode.seeds)
        or receipt.get("evaluation_summary_raw_sha256")
        != file_sha256(summary_path)
        or receipt.get("evaluation_summary_internal_sha256")
        != summary["summary_sha256"]
        or receipt.get("counts") != expected_counts
        or receipt.get("formal_identity_consumed") is not (not mode.preflight)
        or receipt.get("formal_claim_allowed") is not False
        or receipt.get("fresh_test_status") != "prohibited-unread"
        or receipt.get("stage2b_continuation_required") is not False
    ):
        raise RuntimeError("evaluation result receipt identity/accounting 漂移")
    precompletion = {
        "version": E018_P1_STAGE2A_EVALUATION_RESULT_VERSION,
        "verified": True,
        "complete_marker_verified": False,
        "preflight": mode.preflight,
        "experiment_id": mode.experiment_id,
        "config_raw_sha256": loaded.raw_sha256,
        "config_canonical_sha256": loaded.canonical_sha256,
        "source_identity_sha256": public["source_identity_sha256"],
        "conditional_parent_verification_sha256": public[
            "conditional_parent_verification_sha256"
        ],
        "formal_execution_go_verification_sha256": public[
            "formal_execution_go_verification_sha256"
        ],
        "transaction_identity_sha256": public["transaction_identity_sha256"],
        "public_verification_sha256": public["verification_sha256"],
        "result_artifact_role_identity_sha256": result_role_identity,
        "result_receipt_sha256": receipt["receipt_sha256"],
        "consumption_marker_raw_sha256": receipt[
            "consumption_marker_raw_sha256"
        ],
        "consumption_marker_internal_sha256": receipt[
            "consumption_marker_internal_sha256"
        ],
        "scored_row_count": len(scored),
        "independent_scored_route_validation_count": receipt[
            "replay_counts"
        ]["independent_scored_route_validation_count"],
        "private_label_capture_count": mode.label_count,
        "oracle_recoverable_support": summary["oracle_recoverable_support"],
        "recovered_count": summary["recovered_count"],
        "classification": summary["classification"],
        "stage2b_continuation_required": False,
        "fresh_test_reads": 0,
        "producer_process_identity": producer,
        "scorer_process_identity": scorer,
        "process_boundary_verified": True,
    }
    precompletion["verification_sha256"] = canonical_sha256(precompletion)
    if not require_complete:
        return precompletion
    marker = _read_json(root / "RESULT_COMPLETE.json", "evaluation result marker")
    if (
        _verify_internal_digest(
            marker,
            digest_key="marker_sha256",
            name="evaluation result marker",
        )
        != marker.get("marker_sha256")
        or marker.get("version") != E018_P1_STAGE2A_EVALUATION_RESULT_VERSION
        or marker.get("status") != "RESULT_COMPLETE"
        or marker.get("experiment_id") != mode.experiment_id
        or marker.get("transaction_identity_sha256")
        != public["transaction_identity_sha256"]
        or marker.get("precompletion_verification_sha256")
        != precompletion["verification_sha256"]
        or marker.get("result_receipt_raw_sha256") != file_sha256(receipt_path)
        or marker.get("result_receipt_internal_sha256")
        != receipt["receipt_sha256"]
        or marker.get("scored_ledger_raw_sha256") != file_sha256(scored_path)
        or marker.get("evaluation_summary_raw_sha256")
        != file_sha256(summary_path)
        or marker.get("evaluation_summary_internal_sha256")
        != summary["summary_sha256"]
    ):
        raise RuntimeError("evaluation RESULT_COMPLETE marker 漂移")
    result = {
        **precompletion,
        "complete_marker_verified": True,
        "result_completion_marker_sha256": marker["marker_sha256"],
    }
    result.pop("verification_sha256")
    result["verification_sha256"] = canonical_sha256(result)
    return result


def verify_e018_p1_stage2a_evaluation_result(
    *,
    evaluation_config_path: str | Path,
    stage2a_config_path: str | Path,
    qualification_config_path: str | Path,
    public_root: str | Path,
    result_root: str | Path,
    expected_source_git_commit: str,
    expected_source_identity_sha256: str,
    preflight: bool = False,
) -> dict[str, Any]:
    return _verify_e018_p1_stage2a_evaluation_result(
        evaluation_config_path=evaluation_config_path,
        stage2a_config_path=stage2a_config_path,
        qualification_config_path=qualification_config_path,
        public_root=public_root,
        result_root=result_root,
        expected_source_git_commit=expected_source_git_commit,
        expected_source_identity_sha256=expected_source_identity_sha256,
        preflight=preflight,
        require_complete=True,
    )


def _publish_evaluation_result(
    *,
    evaluation_config_path: str | Path,
    stage2a_config_path: str | Path,
    qualification_config_path: str | Path,
    public_root: Path,
    result_root: Path,
    artifact_root: Path,
    loaded: Any,
    public: Mapping[str, Any],
    mode: _EvaluationMode,
    producer: Mapping[str, Any],
    scorer: Mapping[str, Any],
    consumption_marker_path: Path,
    scored: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    replay_environment_identity: Mapping[str, Any],
    replay_counts: Mapping[str, int],
    expected_source_git_commit: str,
    expected_source_identity_sha256: str,
) -> dict[str, Any]:
    result_role_identity = _evaluation_artifact_role_identity_sha256(
        role="result",
        mode=mode,
        config_canonical_sha256=loaded.canonical_sha256,
        source_identity_sha256=public["source_identity_sha256"],
        conditional_parent_verification_sha256=public[
            "conditional_parent_verification_sha256"
        ],
    )
    _selection_runtime._atomic_jsonl(
        result_root / "scored_fixed_gain_routes.jsonl", scored
    )
    _atomic_create_json(result_root / "evaluation_summary.json", dict(summary))
    counts = {
        "private_label_capture_started_count": mode.label_count,
        "private_label_capture_completed_count": mode.label_count,
        "provider_forward_count": 0,
        "checkpoint_load_count": 0,
        "decision_change_count": 0,
        "fixed_gain_branch_count": len(mode.seeds),
        "fresh_test_read_count": 0,
        "runtime_object_gt_read_count": 0,
        "goal_gt_read_count": 0,
        "arm_motion_command_count": 0,
        "gripper_close_command_count": 0,
    }
    receipt = {
        "version": E018_P1_STAGE2A_EVALUATION_RESULT_VERSION,
        "status": "PASS_B_COMPLETE_EXACT_ONCE",
        "preflight": mode.preflight,
        "classification": mode.classification,
        "effect_claim": "no-effect-claim",
        "experiment_id": mode.experiment_id,
        "config_raw_sha256": loaded.raw_sha256,
        "config_canonical_sha256": loaded.canonical_sha256,
        "source_identity_sha256": public["source_identity_sha256"],
        "parent_verification_sha256": public["parent_verification_sha256"],
        "conditional_parent_verification_sha256": public[
            "conditional_parent_verification_sha256"
        ],
        "formal_execution_go_verification_sha256": public[
            "formal_execution_go_verification_sha256"
        ],
        "transaction_identity_sha256": public["transaction_identity_sha256"],
        "public_verification_sha256": public["verification_sha256"],
        "public_completion_marker_sha256": public[
            "public_completion_marker_sha256"
        ],
        "result_artifact_role_identity_sha256": result_role_identity,
        "producer_process_identity": dict(producer),
        "scorer_process_identity": dict(scorer),
        "process_boundary_verified": True,
        "consumption_marker_raw_sha256": file_sha256(
            consumption_marker_path
        ),
        "consumption_marker_internal_sha256": _read_json(
            consumption_marker_path,
            "evaluation final consumption marker",
        )["marker_sha256"],
        "rerun_under_same_identity_allowed": False,
        "scored_ledger_raw_sha256": file_sha256(
            result_root / "scored_fixed_gain_routes.jsonl"
        ),
        "scored_row_count": len(scored),
        "evaluation_summary_raw_sha256": file_sha256(
            result_root / "evaluation_summary.json"
        ),
        "evaluation_summary_internal_sha256": summary["summary_sha256"],
        "counts": counts,
        "replay_counts": dict(replay_counts),
        "replay_environment_identity": dict(replay_environment_identity),
        "formal_identity_consumed": not mode.preflight,
        "formal_claim_allowed": False,
        "fresh_test_status": "prohibited-unread",
        "stage2b_continuation_required": False,
        "completed_at_unix_ns": time.time_ns(),
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    _atomic_create_json(result_root / "result_receipt.json", receipt)
    precompletion = _verify_e018_p1_stage2a_evaluation_result(
        evaluation_config_path=evaluation_config_path,
        stage2a_config_path=stage2a_config_path,
        qualification_config_path=qualification_config_path,
        public_root=public_root,
        result_root=result_root,
        expected_source_git_commit=expected_source_git_commit,
        expected_source_identity_sha256=expected_source_identity_sha256,
        preflight=mode.preflight,
        require_complete=False,
    )
    completion = {
        "version": E018_P1_STAGE2A_EVALUATION_RESULT_VERSION,
        "status": "RESULT_COMPLETE",
        "experiment_id": mode.experiment_id,
        "transaction_identity_sha256": public["transaction_identity_sha256"],
        "precompletion_verification_sha256": precompletion[
            "verification_sha256"
        ],
        "result_receipt_raw_sha256": file_sha256(
            result_root / "result_receipt.json"
        ),
        "result_receipt_internal_sha256": receipt["receipt_sha256"],
        "scored_ledger_raw_sha256": receipt["scored_ledger_raw_sha256"],
        "evaluation_summary_raw_sha256": receipt[
            "evaluation_summary_raw_sha256"
        ],
        "evaluation_summary_internal_sha256": summary["summary_sha256"],
        "completed_at_unix_ns": time.time_ns(),
    }
    completion["marker_sha256"] = canonical_sha256(completion)
    _atomic_create_json(result_root / "RESULT_COMPLETE.json", completion)
    if (
        _selection_runtime._combined_artifact_bytes(artifact_root)
        > loaded.payload["budgets"]["combined_artifact_bytes_max"]
    ):
        (result_root / "RESULT_COMPLETE.json").unlink()
        _selection_runtime._fsync_directory(result_root)
        raise RuntimeError("evaluation Pass B combined artifact budget 超限")
    return verify_e018_p1_stage2a_evaluation_result(
        evaluation_config_path=evaluation_config_path,
        stage2a_config_path=stage2a_config_path,
        qualification_config_path=qualification_config_path,
        public_root=public_root,
        result_root=result_root,
        expected_source_git_commit=expected_source_git_commit,
        expected_source_identity_sha256=expected_source_identity_sha256,
        preflight=mode.preflight,
    )


def run_e018_p1_stage2a_evaluation_score_private(
    *,
    evaluation_config_path: str | Path,
    stage2a_config_path: str | Path,
    qualification_config_path: str | Path,
    g0c_config_path: str | Path,
    data_config_path: str | Path,
    stats_root: str | Path,
    public_root: str | Path,
    private_root: str | Path,
    result_root: str | Path,
    repository_root: str | Path,
    expected_source_git_commit: str,
    expected_source_identity_sha256: str,
    exact_go_token: str,
    formal_go_receipt_path: str | Path | None = None,
    expected_formal_go_raw_sha256: str | None = None,
    expected_formal_go_internal_sha256: str | None = None,
    preflight: bool = False,
) -> dict[str, Any]:
    """Pass B：新进程先消费 identity，再 deterministic replay/capture/score。"""

    mode = _evaluation_mode(preflight)
    if exact_go_token != mode.go_token:
        raise PermissionError("evaluation Pass B 缺对应 formal/preflight exact GO")
    formal_authority_inputs = (
        formal_go_receipt_path,
        expected_formal_go_raw_sha256,
        expected_formal_go_internal_sha256,
    )
    if mode.preflight and any(item is not None for item in formal_authority_inputs):
        raise PermissionError("evaluation preflight Pass B 禁止 formal GO authority")
    if not mode.preflight and any(item is None for item in formal_authority_inputs):
        raise PermissionError("evaluation formal Pass B 缺独立 final-GO identity")
    public = verify_e018_p1_stage2a_evaluation_public(
        evaluation_config_path=evaluation_config_path,
        stage2a_config_path=stage2a_config_path,
        qualification_config_path=qualification_config_path,
        public_root=public_root,
        expected_source_git_commit=expected_source_git_commit,
        expected_source_identity_sha256=expected_source_identity_sha256,
        preflight=mode.preflight,
    )
    scorer_source = _git_source_identity(Path(repository_root))
    if scorer_source != {
        "git_commit": public["source_git_commit"],
        "source_tree_sha256": public["source_tree_sha256"],
        "identity_sha256": public["source_identity_sha256"],
    }:
        raise RuntimeError("evaluation Pass B exact-clean source identity 漂移")
    loaded = load_e018_p1_stage2a_evaluation_config(evaluation_config_path)
    if not mode.preflight:
        source = {
            "git_commit": public["source_git_commit"],
            "source_tree_sha256": public["source_tree_sha256"],
            "identity_sha256": public["source_identity_sha256"],
        }
        authorized_parent = _verify_parent_receipt(
            _read_json(
                Path(public_root) / "parent_verification.json",
                "evaluation Pass B parent receipt",
            ),
            loaded=loaded,
            mode=mode,
            source=source,
            expected_execution_id=None,
        )
        _verify_external_formal_go_against_embedded(
            formal_go_receipt_path=formal_go_receipt_path,
            expected_formal_go_raw_sha256=expected_formal_go_raw_sha256,
            expected_formal_go_internal_sha256=(
                expected_formal_go_internal_sha256
            ),
            embedded_verification=authorized_parent["formal_execution_go"],
            loaded=loaded,
            source=source,
            conditional_parent_verification_sha256=authorized_parent[
                "conditional_parent_verification_sha256"
            ],
            expected_execution_id=Path(public_root).parent.name,
            expected_worker_artifact_root=Path(public_root).parent,
        )
    g0c_path = Path(g0c_config_path)
    data_path = Path(data_config_path)
    qualification_path = Path(qualification_config_path)
    g0c = _stage2a._g0c.load_e018_p1_g0c_config(g0c_path)
    data = _stage2a.load_e018_p1_g2c_data_config(
        data_path,
        parent_g0c_config_path=g0c_path,
    )
    qualification = load_g2c_dynamic_qualification_config(qualification_path)
    frozen = loaded.payload["stage2a_parent"]
    if (
        file_sha256(g0c_path) != frozen["g0c_config_raw_sha256"]
        or canonical_sha256(g0c) != frozen["g0c_config_canonical_sha256"]
        or file_sha256(data_path) != frozen["data_config_raw_sha256"]
        or canonical_sha256(data) != frozen["data_config_canonical_sha256"]
        or file_sha256(qualification_path)
        != frozen["qualification_config_raw_sha256"]
        or qualification["config_sha256"]
        != frozen["qualification_config_internal_sha256"]
    ):
        raise RuntimeError("evaluation Pass B frozen non-model inputs 漂移")
    producer = _selection_runtime._verify_process_identity(
        public["producer_process_identity"], role="pass-a-producer"
    )
    scorer = _selection_runtime._new_process_identity("pass-b-scorer")
    if producer["process_instance_sha256"] == scorer["process_instance_sha256"]:
        raise RuntimeError("evaluation Pass B 必须在不同 OS 进程")
    public_path = Path(public_root)
    private_path = Path(private_root)
    result_path = Path(result_root)
    artifact_root = _selection_runtime._require_common_artifact_parent(
        public_root=public_path,
        private_root=private_path,
        result_root=result_path,
    )
    if private_path.exists() or private_path.is_symlink():
        raise FileExistsError("evaluation private root 已存在，identity 已消费/异常")
    if result_path.exists() or result_path.is_symlink():
        raise FileExistsError("evaluation result root 必须全新")
    _selection_runtime._assert_artifact_top_level_directories(
        artifact_root,
        expected_directories={"public_execution"},
    )
    private_path.mkdir(mode=0o700, parents=False, exist_ok=False)
    marker_path = private_path / "SCORING_CONSUMED.json"
    result_role_identity = _evaluation_artifact_role_identity_sha256(
        role="result",
        mode=mode,
        config_canonical_sha256=loaded.canonical_sha256,
        source_identity_sha256=public["source_identity_sha256"],
        conditional_parent_verification_sha256=public[
            "conditional_parent_verification_sha256"
        ],
    )
    marker = _selection_runtime._signed_consumption_marker(
        {
            "version": E018_P1_STAGE2A_EVALUATION_RESULT_VERSION,
            "status": "PASS_B_MARKER_CREATED_BEFORE_FIRST_GT_READ",
            "preflight": mode.preflight,
            "experiment_id": mode.experiment_id,
            "classification": mode.classification,
            "config_raw_sha256": loaded.raw_sha256,
            "config_canonical_sha256": loaded.canonical_sha256,
            "source_identity_sha256": public["source_identity_sha256"],
            "parent_verification_sha256": public[
                "parent_verification_sha256"
            ],
            "conditional_parent_verification_sha256": public[
                "conditional_parent_verification_sha256"
            ],
            "formal_execution_go_verification_sha256": public[
                "formal_execution_go_verification_sha256"
            ],
            "transaction_identity_sha256": public[
                "transaction_identity_sha256"
            ],
            "public_verification_sha256": public["verification_sha256"],
            "public_completion_marker_sha256": public[
                "public_completion_marker_sha256"
            ],
            "private_artifact_role_identity_sha256": public[
                "private_artifact_role_identity_sha256"
            ],
            "result_artifact_role_identity_sha256": result_role_identity,
            "producer_process_identity": producer,
            "scorer_process_identity": scorer,
            "process_boundary_verified": True,
            "rerun_under_same_identity_allowed": False,
            "private_label_capture_started_count": 0,
            "private_label_capture_completed_count": 0,
            "provider_forward_count": 0,
            "checkpoint_load_count": 0,
            "decision_change_count": 0,
            "result_root_created": False,
            "private_inventory_raw_sha256": None,
            "private_inventory_internal_sha256": None,
            "replay_counts": None,
            "replay_environment_identity": None,
            "update_sequence": 0,
            "failure": None,
            "started_at_unix_ns": time.time_ns(),
            "last_updated_at_unix_ns": time.time_ns(),
            "completed_at_unix_ns": None,
        }
    )
    _selection_runtime._durable_create_json_exclusive(marker_path, marker)
    labels: list[dict[str, Any]] = []
    private_inventory_rows: list[dict[str, Any]] = []
    try:
        camera_rows = _read_jsonl(
            public_path / "camera_pose_ledger.jsonl",
            "evaluation public camera ledger for replay",
        )
        provider_rows = _read_jsonl(
            public_path / "provider_output_ledger.jsonl",
            "evaluation public provider ledger for replay",
        )
        route_rows = _read_jsonl(
            public_path / "route_evidence_ledger.jsonl",
            "evaluation public route ledger for replay",
        )
        provider_records = [
            _stage2a._provider_record_from_dict(row) for row in provider_rows
        ]
        prediction_receipts = _verify_evaluation_prediction_commit_chain(
            public_root=public_path,
            provider_rows=provider_rows,
            transaction_identity_sha256=public[
                "transaction_identity_sha256"
            ],
            mode=mode,
        )
        branches = _read_jsonl(
            public_path / "fixed_gain_branch_ledger.jsonl",
            "evaluation public fixed-gain branches for scoring",
        )
    except Exception as error:
        _selection_runtime._replace_consumption_marker(
            marker_path,
            marker,
            status="PASS_B_FAILED_IDENTITY_PERMANENTLY_CONSUMED",
            failure={
                "error_type": type(error).__name__,
                "message": str(error)[:1024],
                "failed_at_unix_ns": time.time_ns(),
            },
        )
        labels.clear()
        raise
    started_monotonic = time.monotonic()
    try:
        (private_path / "label_commits").mkdir(mode=0o700)
        result_path.mkdir(mode=0o700, parents=False, exist_ok=False)
        marker = _selection_runtime._replace_consumption_marker(
            marker_path,
            marker,
            status="PASS_B_RESULT_ROOT_CREATED_BEFORE_FIRST_GT_READ",
            result_root_created=True,
        )

        def before_gt_capture(label_index: int) -> None:
            nonlocal marker
            if label_index != len(private_inventory_rows):
                raise RuntimeError("evaluation Pass B label start order 漂移")
            marker = _selection_runtime._replace_consumption_marker(
                marker_path,
                marker,
                status="PASS_B_PRIVATE_GT_CAPTURE_IN_PROGRESS",
                private_label_capture_started_count=label_index + 1,
            )

        def after_gt_capture(label: Mapping[str, Any]) -> None:
            nonlocal marker
            index = len(private_inventory_rows)
            if label.get("label_index") != index:
                raise RuntimeError("evaluation Pass B label completion order 漂移")
            path = private_path / "label_commits" / f"{index:06d}.json"
            raw_sha256, _ = _atomic_create_json(path, dict(label))
            primitive = {
                key: label[key]
                for key in (
                    "gt_object_exists",
                    "gt_observable",
                    "gt_object_position_base_m",
                    "robot_object_contact_force_n",
                    "object_linear_speed_m_s",
                    "object_angular_speed_rad_s",
                    "object_motion_event",
                    "is_grasped",
                )
            }
            inventory_row = {
                "label_index": index,
                "seed": label["seed"],
                "route_frame_index": label["route_frame_index"],
                "path": f"label_commits/{index:06d}.json",
                "raw_sha256": raw_sha256,
                "size_bytes": path.stat().st_size,
                "label_internal_sha256": label["label_sha256"],
                "scoring_primitive_sha256": canonical_sha256(primitive),
            }
            inventory_row["row_sha256"] = canonical_sha256(inventory_row)
            private_inventory_rows.append(inventory_row)
            marker = _selection_runtime._replace_consumption_marker(
                marker_path,
                marker,
                status="PASS_B_PRIVATE_GT_CAPTURE_IN_PROGRESS",
                private_label_capture_completed_count=index + 1,
            )

        replay_labels, replay_environment, replay_counts = (
            _run_deterministic_private_label_replay(
                mode=mode,
                loaded_evaluation_config=loaded,
                g0c_config=g0c,
                data_config=data,
                stats_root=Path(stats_root),
                public_camera_rows=camera_rows,
                public_provider_records=provider_records,
                public_route_rows=route_rows,
                transaction_identity_sha256=public[
                    "transaction_identity_sha256"
                ],
                prediction_receipts=prediction_receipts,
                before_gt_capture=before_gt_capture,
                after_gt_capture=after_gt_capture,
                started_monotonic=started_monotonic,
                replay_output_root=private_path,
            )
        )
        labels.extend(replay_labels)
        if (
            len(private_inventory_rows) != mode.label_count
            or replay_counts["private_label_capture_count"] != mode.label_count
            or replay_counts["provider_forward_count"] != 0
            or replay_counts["checkpoint_load_count"] != 0
            or replay_counts["decision_change_count"] != 0
        ):
            raise RuntimeError("evaluation Pass B replay accounting 漂移")
        private_inventory = {
            "version": E018_P1_STAGE2A_EVALUATION_RESULT_VERSION,
            "label_count": len(private_inventory_rows),
            "rows": private_inventory_rows,
        }
        private_inventory["inventory_sha256"] = canonical_sha256(
            private_inventory
        )
        _atomic_create_json(
            private_path / "private_label_inventory.json",
            private_inventory,
        )
        scored, summary = score_selected_gain_evaluation(
            branches,
            labels,
            minimum_support=loaded.payload["promotion"][
                "minimum_oracle_recoverable_support"
            ],
            minimum_recovery_rate=loaded.payload["promotion"][
                "minimum_recovery_rate"
            ],
            seeds=mode.seeds,
        )
        independently_recomputed = _recompute_evaluation_summary(
            [
                _validate_scored_evaluation_row(
                    row,
                    row_index=index,
                    branch=_validated_evaluation_branch(
                        branches[index], expected_seed=mode.seeds[index]
                    ),
                    mode=mode,
                    private_labels=labels[index * 3 : index * 3 + 3],
                )
                for index, row in enumerate(scored)
            ]
        )
        if summary != independently_recomputed:
            raise RuntimeError("evaluation Pass B scorer summary 独立重算不一致")
        replay_counts["independent_scored_route_validation_count"] = len(
            mode.seeds
        )
        if (
            _selection_runtime._combined_artifact_bytes(artifact_root)
            + 1_048_576
            > loaded.payload["budgets"]["combined_artifact_bytes_max"]
        ):
            raise RuntimeError("evaluation Pass B artifact budget 缺结果余量")
        marker = _selection_runtime._replace_consumption_marker(
            marker_path,
            marker,
            status="PASS_B_PRIVATE_LABELS_CAPTURED_AND_SCORED_EXACT_ONCE",
            private_inventory_raw_sha256=file_sha256(
                private_path / "private_label_inventory.json"
            ),
            private_inventory_internal_sha256=private_inventory[
                "inventory_sha256"
            ],
            replay_counts=replay_counts,
            replay_environment_identity=replay_environment,
            completed_at_unix_ns=time.time_ns(),
        )
    except Exception as error:
        try:
            _selection_runtime._replace_consumption_marker(
                marker_path,
                marker,
                status="PASS_B_FAILED_IDENTITY_PERMANENTLY_CONSUMED",
                failure={
                    "error_type": type(error).__name__,
                    "message": str(error)[:1024],
                    "failed_at_unix_ns": time.time_ns(),
                },
            )
        finally:
            labels.clear()
        raise
    labels.clear()
    completion_path = result_path / "RESULT_COMPLETE.json"
    try:
        return _publish_evaluation_result(
            evaluation_config_path=evaluation_config_path,
            stage2a_config_path=stage2a_config_path,
            qualification_config_path=qualification_config_path,
            public_root=public_path,
            result_root=result_path,
            artifact_root=artifact_root,
            loaded=loaded,
            public=public,
            mode=mode,
            producer=producer,
            scorer=scorer,
            consumption_marker_path=marker_path,
            scored=scored,
            summary=summary,
            replay_environment_identity=replay_environment,
            replay_counts=replay_counts,
            expected_source_git_commit=expected_source_git_commit,
            expected_source_identity_sha256=expected_source_identity_sha256,
        )
    except Exception as error:
        if completion_path.exists() or completion_path.is_symlink():
            completion_path.unlink()
            _selection_runtime._fsync_directory(result_path)
        _selection_runtime._replace_consumption_marker(
            marker_path,
            marker,
            status="PASS_B_FAILED_IDENTITY_PERMANENTLY_CONSUMED",
            failure={
                "error_type": type(error).__name__,
                "message": str(error)[:1024],
                "failed_at_unix_ns": time.time_ns(),
            },
        )
        raise


__all__ = [
    "Stage2AEvaluationJournal",
    "run_e018_p1_stage2a_evaluation_capture",
    "run_e018_p1_stage2a_evaluation_score_private",
    "verify_e018_p1_stage2a_evaluation_formal_go",
    "verify_e018_p1_stage2a_evaluation_parent_gate",
    "verify_e018_p1_stage2a_evaluation_public",
    "verify_e018_p1_stage2a_evaluation_result",
]
