"""E018-P1 Stage 2A 信息增益选择的两阶段隔离执行与验证。

Pass A 是唯一可加载 provider / ManiSkill context 的进程；它逐 prediction
fsync，随后才读取同帧 object-only private label。Pass A 销毁全部运行时后，
三个 gain 才从公开 typed evidence 做纯逻辑重放。Pass B 必须是新进程，
exact-once 打开 75 个 private labels，且没有 model/checkpoint/provider 参数。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import secrets
import stat as stat_module
import time
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from robot_vla.precision import e018_p1_stage2a as _stage2a
from robot_vla.precision.calibrated_front_provider import canonical_sha256
from robot_vla.precision.e018_p1_g2c_qualification import (
    _atomic_create_json,
    _atomic_replace_json,
    capture_qualification_object_label,
    load_g2c_dynamic_qualification_config,
)
from robot_vla.precision.e018_p1_g2c_training import _git_source_identity
from robot_vla.precision.e018_p1_stage2a import (
    E018_P1_STAGE2A_EXECUTION_VERSION,
    E018_P1_STAGE2A_SELECTION_EXPERIMENT_ID,
    E018_P1_STAGE2A_SELECTION_PREFLIGHT_EXPERIMENT_ID,
    STAGE2A_COLLECT_FRAME_INDICES,
    STAGE2A_PROVIDER_FRAME_INDICES,
    STAGE2A_SELECTION_PREFLIGHT_SEED,
    STAGE2A_SELECTION_SEEDS,
    Stage2AExecutionProgress,
    Stage2AProviderOutputRecord,
    Stage2ARouteTransaction,
    load_e018_p1_stage2a_config,
    verify_stage2a_provider_output_record,
)
from robot_vla.precision.e018_p1_stage2a_selection import (
    E018_P1_STAGE2A_SELECTION_EXECUTION_VERSION,
    E018_P1_STAGE2A_SELECTION_PREFLIGHT_EXECUTION_VERSION,
    E018_P1_STAGE2A_SELECTION_RESULT_VERSION,
    STAGE2A_SELECTION_BRANCH_COUNT,
    STAGE2A_SELECTION_GAINS,
    STAGE2A_SELECTION_GO,
    STAGE2A_SELECTION_LABEL_COUNT,
    STAGE2A_SELECTION_PREDICTION_COUNT,
    STAGE2A_SELECTION_PREFLIGHT_GO,
    CapturedSelectionRoute,
    GainBranchOutcome,
    Stage2ASelectionJournal,
    Stage2ASelectionPreflightJournal,
    _array_sha256,
    _is_sha256,
    _read_json,
    _read_jsonl,
    _require_exact_keys,
    _serialize_baseline,
    _validate_private_label_for_scoring,
    _validated_gain_branch,
    _verify_internal_digest,
    file_sha256,
    load_e018_p1_stage2a_selection_config,
    replay_all_gain_branches,
    score_gain_branches,
    verify_selection_parent_gate,
)

_PUBLIC_FILES = {
    "RUN_STARTED.json",
    "config_snapshot.json",
    "source_identity.json",
    "parent_verification.json",
    "camera_pose_ledger.jsonl",
    "provider_output_ledger.jsonl",
    "route_evidence_ledger.jsonl",
    "gain_branch_ledger.jsonl",
    "private_label_inventory.json",
    "execution_freeze.json",
    "execution_receipt.json",
    "PUBLIC_EXECUTION_COMPLETE.json",
}
_PUBLIC_PRECOMPLETION_FILES = _PUBLIC_FILES - {"PUBLIC_EXECUTION_COMPLETE.json"}
_RESULT_FILES = {
    "scored_gain_branches.jsonl",
    "selection_summary.json",
    "result_receipt.json",
    "RESULT_COMPLETE.json",
}
_RESULT_PRECOMPLETION_FILES = _RESULT_FILES - {"RESULT_COMPLETE.json"}
_ARTIFACT_TOP_LEVEL_DIRECTORIES = {
    "public_execution",
    "private_labels",
    "result",
}
_PREFLIGHT_PUBLIC_FILES = {
    "config_snapshot.json",
    "source_identity.json",
    "parent_verification.json",
    "checkpoint_identity.json",
    "camera_pose_ledger.jsonl",
    "provider_output_ledger.jsonl",
    "route_summary.json",
    "transaction.json",
    "private_label_inventory.json",
    "preflight_receipt.json",
    "PREFLIGHT_COMPLETE.json",
}

_SELECTION_CLASSIFICATION = (
    "formal-development-selection-no-test-no-actuation/v1"
)
_SELECTION_CAPTURE_CLASSIFICATION = (
    "formal-development-selection-capture-only-no-test-no-actuation/v1"
)

_SELECTION_TRANSACTION_KEYS = {
    "version",
    "classification",
    "effect_claim",
    "wrist_capability",
    "seed",
    "episode_id",
    "request_id",
    "trigger_decisions",
    "trigger_wrist_capability_records",
    "trigger_wrist_capability",
    "source_recheck_wrist_capability",
    "capability_absence_trigger_reason",
    "provider_output_digests",
    "provider_frame_indices",
    "primary_frame_digests",
    "candidate_stage_receipt",
    "candidate",
    "candidate_digest",
    "candidate_commit_eligible",
    "collect_rejection_reasons",
    "terminal_reasons",
    "orchestrator_state",
    "home_barrier_frame_indices",
    "home_observation_sequence_ids",
    "home_observation_timestamps_s",
    "home_frame_digests",
    "home_barrier_evidence",
    "observation_v2_window_identity",
    "safety_evidence",
    "camera_command_authorizations",
    "controller_events",
    "controller_receipt",
    "action_history_reset_audit",
    "action_history_resume_audit",
    "memory_write_count",
    "commit_receipt",
    "shadow_action_generation",
    "final_memory_state",
    "route_passed",
    "runtime_object_gt_reads",
    "goal_gt_reads",
    "offline_label_reads",
    "wrist_provider_forward_count",
    "arm_motion_command_count",
    "gripper_close_command_count",
    "fresh_test_reads",
    "checkpoint_writes",
}

_SELECTION_ROUTE_SUMMARY_KEYS = {
    "version",
    "episode_id",
    "seed",
    "alternate_viewpoint_id",
    "alternate_orientation_id",
    "yaw_offset_rad",
    "pitch_offset_rad",
    "roll_offset_rad",
    "status",
    "passed",
    "frame_count",
    "control_hz",
    "motion_ticks_each_leg",
    "route_simulated_duration_s",
    "gates",
    "diagnostics",
    "test_split_status",
    "provider_forward_count",
    "memory_write_count",
    "formal_claim_allowed",
    "classification",
    "offline_segmentation_diagnostics",
    "runtime_object_gt_reads",
    "goal_gt_reads",
    "fresh_test_reads",
}

_SELECTION_ROUTE_DIAGNOSTIC_KEYS = {
    "alternate_rgb_mean_abs_difference",
    "return_home_rgb_mean_abs_difference",
    "alternate_displacement_m",
    "requested_orientation_offset_rad",
    "actual_orientation_offset_rad",
    "alternate_target_orientation_error_rad",
    "object_visible_pixels_collect_min",
    "goal_visible_pixels_collect_min",
}

_SELECTION_ROUTE_GATE_ACTUAL_KEYS: dict[str, set[str] | None] = {
    "nonzero_time_camera_motion": {
        "move_ticks_each_leg",
        "move_duration_s_each_leg",
    },
    "actual_dynamic_pose_observed": {
        "unique_actual_positions",
        "alternate_displacement_m",
    },
    "alternate_orientation_target_reached": {
        "orientation_id",
        "requested_offset_rad",
        "actual_offset_rad",
        "target_error_rad",
    },
    "commanded_actual_tracking": {
        "max_position_error_m",
        "max_orientation_error_rad",
    },
    "camera_velocity_limits": {
        "max_linear_velocity_m_s",
        "max_angular_velocity_rad_s",
    },
    "camera_acceleration_limits": {
        "max_linear_acceleration_m_s2",
        "max_angular_acceleration_rad_s2",
    },
    "settled_collection_window": {
        "collect_frames",
        "eligible_collect_frames",
    },
    "return_home_pose": {
        "position_error_m",
        "orientation_error_rad",
        "final_settled",
    },
    "rendered_view_changed": None,
    "return_home_render_recovered": None,
    "arm_joint_hold": None,
    "tcp_hold": {
        "max_position_drift_m",
        "max_orientation_drift_rad",
    },
    "gripper_safe_hold_open": None,
    "unexpected_contact_absent": None,
    "non_collect_frame_write_invalidation": None,
    "memory_write_disabled": None,
    "episode_remained_nonterminal": None,
    "runtime_gt_control_dependency_absent": None,
}


def _finite_real(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} 必须是有限实数")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{name} 必须是有限实数")
    return result


def _new_process_identity(role: str) -> dict[str, Any]:
    """生成位置无关、隐藏 raw boot id 的 OS process-instance 证据。"""

    if role not in {"pass-a-producer", "pass-b-scorer"}:
        raise ValueError("selection process role 非法")
    try:
        raw_boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="ascii"
        ).strip()
        stat = Path("/proc/self/stat").read_text(encoding="ascii").strip()
        closing = stat.rfind(")")
        fields_after_comm = stat[closing + 2 :].split()
        start_time_ticks = int(fields_after_comm[19])
    except (OSError, UnicodeError, ValueError, IndexError) as error:
        raise RuntimeError("selection 无法建立 /proc process identity") from error
    if not raw_boot_id or closing < 0 or start_time_ticks <= 0:
        raise RuntimeError("selection /proc process identity 非法")
    instance = {
        "version": "e018-p1-stage2a-selection-process-instance/v1",
        "boot_id_sha256": hashlib.sha256(
            raw_boot_id.encode("ascii")
        ).hexdigest(),
        "pid": os.getpid(),
        "proc_start_time_ticks": start_time_ticks,
    }
    process_instance_sha256 = canonical_sha256(instance)
    value = {
        **instance,
        "role": role,
        "run_nonce_sha256": hashlib.sha256(
            secrets.token_bytes(32)
        ).hexdigest(),
        "process_instance_sha256": process_instance_sha256,
    }
    value["process_token_sha256"] = canonical_sha256(value)
    return value


def _verify_process_identity(value: Any, *, role: str) -> dict[str, Any]:
    identity = _require_exact_keys(
        value,
        {
            "version",
            "boot_id_sha256",
            "pid",
            "proc_start_time_ticks",
            "role",
            "run_nonce_sha256",
            "process_instance_sha256",
            "process_token_sha256",
        },
        f"selection {role} process identity",
    )
    unsigned = dict(identity)
    token = unsigned.pop("process_token_sha256")
    instance = {
        key: identity[key]
        for key in (
            "version",
            "boot_id_sha256",
            "pid",
            "proc_start_time_ticks",
        )
    }
    if (
        identity["version"]
        != "e018-p1-stage2a-selection-process-instance/v1"
        or identity["role"] != role
        or not _is_sha256(identity["boot_id_sha256"])
        or type(identity["pid"]) is not int
        or identity["pid"] <= 0
        or type(identity["proc_start_time_ticks"]) is not int
        or identity["proc_start_time_ticks"] <= 0
        or not _is_sha256(identity["run_nonce_sha256"])
        or identity["process_instance_sha256"] != canonical_sha256(instance)
        or token != canonical_sha256(unsigned)
    ):
        raise RuntimeError(f"selection {role} process identity 漂移")
    return identity


def _verify_selection_route_summary_schema(
    value: Any,
    *,
    seed: int,
    episode_id: str,
) -> dict[str, Any]:
    """拒绝 route summary 的未知/privileged nested payload。"""

    summary = _require_exact_keys(
        value,
        _SELECTION_ROUTE_SUMMARY_KEYS,
        "selection route summary",
    )
    diagnostics = _require_exact_keys(
        summary["diagnostics"],
        _SELECTION_ROUTE_DIAGNOSTIC_KEYS,
        "selection route diagnostics",
    )
    gates = _require_exact_keys(
        summary["gates"],
        set(_SELECTION_ROUTE_GATE_ACTUAL_KEYS),
        "selection route gates",
    )
    for name, actual_keys in _SELECTION_ROUTE_GATE_ACTUAL_KEYS.items():
        gate = _require_exact_keys(
            gates[name],
            {"actual", "required", "passed"},
            f"selection route gate {name}",
        )
        if type(gate["passed"]) is not bool or not isinstance(
            gate["required"], str
        ) or not gate["required"]:
            raise RuntimeError(f"selection route gate {name} primitive 漂移")
        if actual_keys is None:
            _finite_real(gate["actual"], f"selection route gate {name}.actual")
        else:
            actual = _require_exact_keys(
                gate["actual"],
                actual_keys,
                f"selection route gate {name}.actual",
            )
            for field, field_value in actual.items():
                if field == "orientation_id":
                    if field_value != "PITCH_UP":
                        raise RuntimeError("selection route orientation gate 漂移")
                elif field == "final_settled":
                    if type(field_value) is not bool:
                        raise RuntimeError("selection route final_settled 类型漂移")
                elif field in {
                    "move_ticks_each_leg",
                    "unique_actual_positions",
                    "collect_frames",
                    "eligible_collect_frames",
                }:
                    if type(field_value) is not int or field_value < 0:
                        raise RuntimeError(
                            f"selection route gate {name}.{field} 类型漂移"
                        )
                else:
                    _finite_real(
                        field_value,
                        f"selection route gate {name}.{field}",
                    )
    for name in (
        "alternate_rgb_mean_abs_difference",
        "return_home_rgb_mean_abs_difference",
        "alternate_displacement_m",
        "requested_orientation_offset_rad",
        "actual_orientation_offset_rad",
        "alternate_target_orientation_error_rad",
    ):
        _finite_real(diagnostics[name], f"selection route diagnostics.{name}")
    if (
        diagnostics["object_visible_pixels_collect_min"] is not None
        or diagnostics["goal_visible_pixels_collect_min"] is not None
        or summary["version"] != E018_P1_STAGE2A_EXECUTION_VERSION
        or summary["seed"] != seed
        or summary["episode_id"] != episode_id
        or summary["alternate_viewpoint_id"]
        != _stage2a.ACTIVE_FRONT_PRIMARY_PRIMITIVE_ID
        or summary["alternate_orientation_id"] != "PITCH_UP"
        or summary["classification"] != _SELECTION_CAPTURE_CLASSIFICATION
        or summary["offline_segmentation_diagnostics"] is not False
        or summary["test_split_status"] != "prohibited-unread"
        or summary["formal_claim_allowed"] is not False
        or type(summary["passed"]) is not bool
        or summary["status"] != ("passed" if summary["passed"] else "failed")
        or summary["passed"] is not all(gate["passed"] for gate in gates.values())
        or type(summary["frame_count"]) is not int
        or summary["frame_count"] != 92
        or type(summary["control_hz"]) is not int
        or summary["control_hz"] != 20
        or type(summary["motion_ticks_each_leg"]) is not int
        or summary["motion_ticks_each_leg"] != 40
        or _finite_real(
            summary["route_simulated_duration_s"],
            "selection route simulated duration",
        )
        != 4.55
        or type(summary["provider_forward_count"]) is not int
        or summary["provider_forward_count"] != 4
        or type(summary["memory_write_count"]) is not int
        or summary["memory_write_count"] != 0
        or any(
            type(summary[name]) is not int or summary[name] != 0
            for name in (
                "runtime_object_gt_reads",
                "goal_gt_reads",
                "fresh_test_reads",
            )
        )
        or gates["actual_dynamic_pose_observed"]["actual"][
            "alternate_displacement_m"
        ]
        != diagnostics["alternate_displacement_m"]
        or gates["alternate_orientation_target_reached"]["actual"][
            "requested_offset_rad"
        ]
        != diagnostics["requested_orientation_offset_rad"]
        or gates["alternate_orientation_target_reached"]["actual"][
            "actual_offset_rad"
        ]
        != diagnostics["actual_orientation_offset_rad"]
        or gates["alternate_orientation_target_reached"]["actual"][
            "target_error_rad"
        ]
        != diagnostics["alternate_target_orientation_error_rad"]
        or gates["rendered_view_changed"]["actual"]
        != diagnostics["alternate_rgb_mean_abs_difference"]
        or gates["return_home_render_recovered"]["actual"]
        != diagnostics["return_home_rgb_mean_abs_difference"]
    ):
        raise RuntimeError("selection route summary identity/accounting 漂移")
    for name in ("yaw_offset_rad", "pitch_offset_rad", "roll_offset_rad"):
        _finite_real(summary[name], f"selection route summary.{name}")
    return summary


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _jsonl_line_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"拒绝覆盖 immutable selection ledger: {path}")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        for row in rows:
            raw = memoryview(_jsonl_line_bytes(row))
            while raw:
                written = os.write(descriptor, raw)
                if written <= 0:
                    raise OSError("selection JSONL 写入失败")
                raw = raw[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _single_velocity_vector(value: Any, name: str) -> np.ndarray:
    array = np.asarray(_stage2a._g0._numpy(value), dtype=np.float64)
    if array.shape == (1, 3):
        array = array[0]
    if array.shape != (3,) or not np.isfinite(array).all():
        raise RuntimeError(f"selection private {name} 必须是唯一有限 [3] 向量")
    return array


def capture_selection_private_label(
    *,
    observation: Mapping[str, Any],
    base_env: Any,
    prediction: Mapping[str, Any],
    data_config: Mapping[str, Any],
) -> dict[str, Any]:
    """读取 object-only label 与 object motion；返回值只能交给 write-only journal。"""

    label = capture_qualification_object_label(
        observation=observation,
        base_env=base_env,
        prediction=prediction,
        data_config=data_config,
    )
    linear = _single_velocity_vector(base_env.cube.linear_velocity, "linear velocity")
    angular = _single_velocity_vector(
        base_env.cube.angular_velocity,
        "angular velocity",
    )
    linear_speed = float(np.linalg.norm(linear))
    angular_speed = float(np.linalg.norm(angular))
    if not math.isfinite(linear_speed) or not math.isfinite(angular_speed):
        raise RuntimeError("selection private object speed 非有限")
    return {
        **label,
        "object_linear_speed_m_s": linear_speed,
        "object_angular_speed_rad_s": angular_speed,
        "object_motion_event": bool(
            linear_speed > 0.01 or angular_speed > 0.5
        ),
    }


def build_route_evidence_row(
    *,
    seed: int,
    camera_row_start: int,
    camera_rows: Sequence[Mapping[str, Any]],
    provider_row_indices: Sequence[int],
    provider_records: Sequence[Stage2AProviderOutputRecord],
    prediction_receipts: Sequence[Mapping[str, Any]],
    private_inventory_rows: Sequence[Mapping[str, Any]],
    capture_transaction: Mapping[str, Any],
    route_summary: Mapping[str, Any],
    captured_route: CapturedSelectionRoute,
) -> dict[str, Any]:
    """构造不含 label 值的完整 route bundle。"""

    if (
        seed not in STAGE2A_SELECTION_SEEDS
        or len(camera_rows) != 92
        or len(provider_row_indices) != 4
        or len(provider_records) != 4
        or len(prediction_receipts) != 4
        or len(private_inventory_rows) != 3
        or captured_route.seed != seed
        or tuple(record.route_frame_index for record in provider_records)
        != STAGE2A_PROVIDER_FRAME_INDICES
    ):
        raise RuntimeError("selection route bundle count/order 漂移")
    transaction = dict(capture_transaction)
    summary = dict(route_summary)
    route = captured_route.to_public_dict()
    row = {
        "version": E018_P1_STAGE2A_SELECTION_EXECUTION_VERSION,
        "seed": seed,
        "episode_id": captured_route.episode_id,
        "request_id": captured_route.request.request_id,
        "camera_row_start": camera_row_start,
        "camera_row_stop_exclusive": camera_row_start + 92,
        "camera_rows_sha256": canonical_sha256(list(camera_rows)),
        "provider_row_indices": list(provider_row_indices),
        "provider_output_digests": [
            value.provider_output_digest for value in provider_records
        ],
        "prediction_commit_receipt_sha256s": [
            value["commit_receipt_sha256"] for value in prediction_receipts
        ],
        "private_label_inventory_row_sha256s": [
            value["row_sha256"] for value in private_inventory_rows
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


def _assert_exact_tree(
    root: Path,
    *,
    expected_files: set[str],
    expected_directory: str | None = None,
    expected_directory_files: set[str] | None = None,
    name: str,
) -> None:
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"{name} root 必须是真实目录")
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise RuntimeError(f"{name} 禁止 symlink: {relative}")
        if path.is_dir():
            actual_directories.add(relative)
        elif path.is_file() and path.stat().st_nlink == 1:
            actual_files.add(relative)
        else:
            raise RuntimeError(f"{name} 禁止 hardlink/special file: {relative}")
    expected_all = set(expected_files)
    directories: set[str] = set()
    if expected_directory is not None:
        directories.add(expected_directory)
        if expected_directory_files is None:
            raise ValueError("expected directory files 未配置")
        expected_all.update(
            f"{expected_directory}/{value}" for value in expected_directory_files
        )
    if actual_files != expected_all or actual_directories != directories:
        raise RuntimeError(
            f"{name} exact tree 漂移: missing={sorted(expected_all-actual_files)}, "
            f"extra={sorted(actual_files-expected_all)}, "
            f"directories={sorted(actual_directories)}"
        )


def _require_common_artifact_parent(
    *,
    public_root: Path,
    result_root: Path,
    private_root: Path | None = None,
) -> Path:
    """绑定位置无关 artifact role 到同一真实、非 symlink 的父目录。"""

    roots = {
        "public_execution": public_root,
        "result": result_root,
    }
    if private_root is not None:
        roots["private_labels"] = private_root
    for expected_name, path in roots.items():
        if path.name != expected_name:
            raise RuntimeError(
                f"selection artifact role basename 漂移: "
                f"expected={expected_name}, actual={path.name}"
            )
    parent_stats: list[tuple[int, int]] = []
    canonical_parents: list[Path] = []
    for path in roots.values():
        parent = path.parent
        if parent.is_symlink() or not parent.is_dir():
            raise RuntimeError("selection artifact parent 必须是真实非 symlink 目录")
        try:
            canonical = parent.resolve(strict=True)
        except OSError as error:
            raise RuntimeError("selection artifact parent 无法解析") from error
        if canonical != Path(os.path.abspath(parent)):
            raise RuntimeError("selection artifact parent 路径禁止 symlink alias")
        metadata = parent.stat()
        parent_stats.append((metadata.st_dev, metadata.st_ino))
        canonical_parents.append(canonical)
    if len(set(parent_stats)) != 1 or len(set(canonical_parents)) != 1:
        raise RuntimeError("selection artifacts 必须位于同一真实父目录")
    return canonical_parents[0]


def _assert_artifact_top_level_directories(
    artifact_root: Path,
    *,
    expected_directories: set[str],
) -> None:
    actual: set[str] = set()
    for path in artifact_root.iterdir():
        if path.is_symlink() or not path.is_dir():
            raise RuntimeError(
                f"selection artifact top-level 禁止文件/symlink: {path.name}"
            )
        actual.add(path.name)
    if actual != expected_directories:
        raise RuntimeError(
            "selection artifact top-level directories 漂移: "
            f"missing={sorted(expected_directories-actual)}, "
            f"extra={sorted(actual-expected_directories)}"
        )


def _combined_artifact_bytes(artifact_root: Path) -> int:
    return sum(
        path.stat().st_size
        for path in artifact_root.rglob("*")
        if path.is_file()
    )


def _verify_prediction_commit_chain(
    public_root: Path,
    provider_rows: Sequence[Mapping[str, Any]],
    *,
    expected_transaction_identity_sha256: str,
) -> list[dict[str, Any]]:
    if not _is_sha256(expected_transaction_identity_sha256):
        raise ValueError("selection expected transaction identity 非法")
    ledger_path = public_root / "provider_output_ledger.jsonl"
    raw_lines = ledger_path.read_bytes().splitlines(keepends=True)
    if len(raw_lines) != STAGE2A_SELECTION_PREDICTION_COUNT or any(
        not line.endswith(b"\n") for line in raw_lines
    ):
        raise RuntimeError("selection provider raw ledger 行数/终止符漂移")
    prefix = hashlib.sha256()
    previous: str | None = None
    receipts: list[dict[str, Any]] = []
    for index, (line, provider_row) in enumerate(
        zip(raw_lines, provider_rows, strict=True)
    ):
        if line != _jsonl_line_bytes(provider_row):
            raise RuntimeError("selection provider ledger serialization 漂移")
        prefix.update(line)
        path = public_root / "prediction_commits" / f"{index:06d}.commit.json"
        receipt = _read_json(path, f"prediction commit[{index}]")
        unsigned = dict(receipt)
        stored = unsigned.pop("commit_receipt_sha256", None)
        seed = STAGE2A_SELECTION_SEEDS[index // 4]
        frame = STAGE2A_PROVIDER_FRAME_INDICES[index % 4]
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
            or receipt["version"] != E018_P1_STAGE2A_SELECTION_EXECUTION_VERSION
            or receipt["row_index"] != index
            or receipt["seed"] != seed
            or receipt["route_frame_index"] != frame
            or receipt["provider_output_digest"]
            != provider_row.get("provider_output_digest")
            or receipt["model_input_digest"]
            != provider_row.get("model_input_digest")
            or receipt["transaction_identity_sha256"]
            != expected_transaction_identity_sha256
            or receipt["provider_ledger_prefix_raw_sha256"] != prefix.hexdigest()
            or receipt["previous_prediction_commit_sha256"] != previous
            or type(receipt["prediction_fsync_completed_at_unix_ns"]) is not int
            or receipt["prediction_fsync_completed_at_unix_ns"] <= 0
        ):
            raise RuntimeError(f"selection prediction commit[{index}] chain 漂移")
        previous = stored
        receipts.append(receipt)
    return receipts


def _file_inventory(
    root: Path,
    relative_paths: Sequence[str],
) -> dict[str, dict[str, Any]]:
    return {
        relative: {
            "raw_sha256": file_sha256(root / relative),
            "size_bytes": (root / relative).stat().st_size,
        }
        for relative in relative_paths
    }


def _verify_private_inventory_public(
    value: Mapping[str, Any],
) -> list[dict[str, Any]]:
    inventory = _require_exact_keys(
        dict(value),
        {"version", "label_count", "rows", "inventory_sha256"},
        "selection private label inventory",
    )
    unsigned = dict(inventory)
    stored = unsigned.pop("inventory_sha256")
    rows = inventory["rows"]
    if (
        stored != canonical_sha256(unsigned)
        or inventory["version"] != E018_P1_STAGE2A_SELECTION_EXECUTION_VERSION
        or inventory["label_count"] != STAGE2A_SELECTION_LABEL_COUNT
        or not isinstance(rows, list)
        or len(rows) != STAGE2A_SELECTION_LABEL_COUNT
    ):
        raise RuntimeError("selection private inventory identity/count 漂移")
    verified: list[dict[str, Any]] = []
    expected_keys = {
        "label_index",
        "prediction_row_index",
        "seed",
        "route_frame_index",
        "path",
        "raw_sha256",
        "size_bytes",
        "scoring_primitive_sha256",
        "row_sha256",
    }
    for index, item in enumerate(rows):
        row = _require_exact_keys(
            dict(item), expected_keys, f"private inventory[{index}]"
        )
        unsigned_row = dict(row)
        row_sha = unsigned_row.pop("row_sha256")
        seed = STAGE2A_SELECTION_SEEDS[index // 3]
        frame = STAGE2A_COLLECT_FRAME_INDICES[index % 3]
        if (
            row_sha != canonical_sha256(unsigned_row)
            or row["label_index"] != index
            or row["prediction_row_index"] != (index // 3) * 4 + 1 + index % 3
            or row["seed"] != seed
            or row["route_frame_index"] != frame
            or row["path"] != f"label_commits/{index:06d}.json"
            or not _is_sha256(row["raw_sha256"])
            or not _is_sha256(row["scoring_primitive_sha256"])
            or type(row["size_bytes"]) is not int
            or row["size_bytes"] <= 0
        ):
            raise RuntimeError(f"private inventory[{index}] identity/order 漂移")
        verified.append(row)
    return verified


def _verify_route_evidence_row(
    value: Mapping[str, Any],
    *,
    route_index: int,
    camera_rows: Sequence[Mapping[str, Any]],
    provider_records: Sequence[Stage2AProviderOutputRecord],
    prediction_receipts: Sequence[Mapping[str, Any]],
    private_inventory_rows: Sequence[Mapping[str, Any]],
    stage2a_config: Any,
    qualification_config: Mapping[str, Any],
) -> CapturedSelectionRoute:
    expected_keys = {
        "version",
        "seed",
        "episode_id",
        "request_id",
        "camera_row_start",
        "camera_row_stop_exclusive",
        "camera_rows_sha256",
        "provider_row_indices",
        "provider_output_digests",
        "prediction_commit_receipt_sha256s",
        "private_label_inventory_row_sha256s",
        "capture_transaction",
        "capture_transaction_sha256",
        "route_summary",
        "route_summary_sha256",
        "captured_route",
        "route_evidence_digest",
        "route_row_sha256",
    }
    row = _require_exact_keys(
        dict(value), expected_keys, f"selection route[{route_index}]"
    )
    unsigned = dict(row)
    stored = unsigned.pop("route_row_sha256")
    seed = STAGE2A_SELECTION_SEEDS[route_index]
    episode = f"e018-p1-stage2a-selection-development-seed-{seed}"
    request = f"{episode}-active-front-01"
    camera_start = route_index * 92
    provider_start = route_index * 4
    label_start = route_index * 3
    route_camera_rows = camera_rows[camera_start : camera_start + 92]
    route_provider_records = provider_records[provider_start : provider_start + 4]
    route_receipts = prediction_receipts[provider_start : provider_start + 4]
    route_private = private_inventory_rows[label_start : label_start + 3]
    if (
        stored != canonical_sha256(unsigned)
        or row["version"] != E018_P1_STAGE2A_SELECTION_EXECUTION_VERSION
        or row["seed"] != seed
        or row["episode_id"] != episode
        or row["request_id"] != request
        or row["camera_row_start"] != camera_start
        or row["camera_row_stop_exclusive"] != camera_start + 92
        or row["camera_rows_sha256"] != canonical_sha256(list(route_camera_rows))
        or row["provider_row_indices"]
        != list(range(provider_start, provider_start + 4))
        or row["provider_output_digests"]
        != [item.provider_output_digest for item in route_provider_records]
        or row["prediction_commit_receipt_sha256s"]
        != [item["commit_receipt_sha256"] for item in route_receipts]
        or row["private_label_inventory_row_sha256s"]
        != [item["row_sha256"] for item in route_private]
        or row["capture_transaction_sha256"]
        != canonical_sha256(row["capture_transaction"])
        or row["route_summary_sha256"] != canonical_sha256(row["route_summary"])
    ):
        raise RuntimeError(f"selection route[{route_index}] outer binding 漂移")
    verified_camera_rows = [
        _stage2a._verify_stage2a_camera_row_identity(
            camera_row,
            episode_id=episode,
            request_id=request,
            frame_index=frame_index,
        )
        for frame_index, camera_row in enumerate(route_camera_rows)
    ]
    maximum_projection_error = float(
        qualification_config["capture_safety"][
            "maximum_rotation_projection_error_frobenius"
        ]
    )
    primary = []
    for record in route_provider_records:
        frame_index = record.route_frame_index
        camera_row = verified_camera_rows[frame_index]
        _stage2a._verify_stage2a_provider_route_pose_binding(
            record.prediction["base_from_external_camera_cv"],
            camera_row["actual_base_from_external_camera_cv"],
            maximum_projection_error=maximum_projection_error,
        )
        if frame_index == 0:
            _stage2a.build_stage2a_home_score_evidence(
                record,
                motion_row=camera_row,
                timestamp_offset_s=Stage2ARouteTransaction._TIMESTAMP_OFFSET_S,
            )
        else:
            primary.append(
                _stage2a.build_stage2a_primary_frame_evidence(
                    record,
                    motion_row=camera_row,
                    timestamp_offset_s=Stage2ARouteTransaction._TIMESTAMP_OFFSET_S,
                )
            )
    captured = CapturedSelectionRoute.from_public_dict(row["captured_route"])
    transaction = _require_exact_keys(
        row["capture_transaction"],
        _SELECTION_TRANSACTION_KEYS,
        f"selection transaction[{route_index}]",
    )
    summary = _verify_selection_route_summary_schema(
        row["route_summary"],
        seed=seed,
        episode_id=episode,
    )
    candidate = transaction.get("candidate")
    candidate_digest = None if candidate is None else canonical_sha256(candidate)
    candidate_eligible = None if candidate is None else candidate.get("commit_eligible")
    candidate_reasons = None if candidate is None else candidate.get("rejection_reasons")
    if (
        captured.route_evidence_digest != row["route_evidence_digest"]
        or [item.frame_digest for item in primary]
        != [item.frame_digest for item in captured.primary_frames]
        or transaction.get("seed") != seed
        or transaction.get("episode_id") != episode
        or transaction.get("request_id") != request
        or transaction.get("classification")
        != "formal-development-selection-capture-only-no-test-no-actuation/v1"
        or transaction.get("effect_claim") != "no-effect-claim"
        or transaction.get("wrist_capability")
        != _stage2a.WRIST_CAPABILITY_ABSENT_STATUS
        or transaction.get("provider_output_digests")
        != row["provider_output_digests"]
        or transaction.get("provider_frame_indices")
        != list(STAGE2A_PROVIDER_FRAME_INDICES)
        or transaction.get("primary_frame_digests")
        != [item.frame_digest for item in primary]
        or transaction.get("candidate_digest") != candidate_digest
        or captured.raw_candidate_digest_at_gain_0_02 != candidate_digest
        or captured.raw_candidate_commit_eligible_at_gain_0_02
        is not candidate_eligible
        or list(captured.raw_candidate_rejection_reasons_at_gain_0_02 or ())
        != (candidate_reasons or [])
        or transaction.get("memory_write_count") != 0
        or transaction.get("commit_receipt") is not None
        or transaction.get("shadow_action_generation") is not None
        or transaction.get("action_history_resume_audit") is not None
        or any(
            transaction.get(name) != 0
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
        or summary.get("seed") != seed
        or summary.get("episode_id") != episode
        or type(summary.get("frame_count")) is not int
        or summary.get("frame_count") != 92
        or type(summary.get("provider_forward_count")) is not int
        or summary.get("provider_forward_count") != 4
        or type(summary.get("memory_write_count")) is not int
        or summary.get("memory_write_count") != 0
        or type(summary.get("passed")) is not bool
        or captured.route_protocol_safety_valid is not summary.get("passed")
        or transaction.get("route_passed")
        is not captured.route_protocol_safety_valid
    ):
        raise RuntimeError(f"selection route[{route_index}] transaction/replay 漂移")
    _verify_capture_only_transaction_deep(
        stage2a_config=stage2a_config,
        transaction=transaction,
        route_summary=summary,
        camera_rows=verified_camera_rows,
        provider_records=route_provider_records,
        primary_frames=primary,
        captured=captured,
    )
    return captured


def _verify_capture_only_transaction_deep(
    *,
    stage2a_config: Any,
    transaction: Mapping[str, Any],
    route_summary: Mapping[str, Any],
    camera_rows: Sequence[Mapping[str, Any]],
    provider_records: Sequence[Stage2AProviderOutputRecord],
    primary_frames: Sequence[Any],
    captured: CapturedSelectionRoute,
) -> None:
    """重放决定 route safety/阶段/返回 HOME 的完整 capture-only 协议。"""

    episode_id = captured.episode_id
    request_id = captured.request.request_id
    controller = _stage2a._new_stage2a_replay_controller(
        stage2a_config,
        episode_id=episode_id,
    )
    trigger_records, source_recheck = _stage2a._verify_stage2a_trigger_replay(
        transaction,
        controller=controller,
        episode_id=episode_id,
    )
    reset_receipt = _stage2a.verify_stage2a_action_history_audit(
        transaction["action_history_reset_audit"]
    )
    if (
        not isinstance(reset_receipt, _stage2a.ActionHistoryResetReceipt)
        or reset_receipt.episode_id != episode_id
        or reset_receipt.request_id != request_id
        or reset_receipt.reset_control_tick != 2
    ):
        raise RuntimeError("selection Action reset audit 漂移")
    home_score = _stage2a.build_stage2a_home_score_evidence(
        provider_records[0],
        motion_row=camera_rows[0],
        timestamp_offset_s=Stage2ARouteTransaction._TIMESTAMP_OFFSET_S,
    )
    expected_baseline = _stage2a.PassiveBaselineEvidence(
        episode_id=episode_id,
        episode_generation=1,
        request_id=request_id,
        timestamp_s=captured.request.trigger_timestamp_s,
        wrist_object_measurement_usable=trigger_records[-1].measurement_usable,
        wrist_evidence_identity_sha256=trigger_records[-1].digest,
        home_front=home_score,
        object_memory_navigation_state_available=(
            trigger_records[-1].memory_resolution_available
        ),
        object_memory_age_s=None,
        object_memory_source_identity=None,
    )
    if _serialize_baseline(expected_baseline) != _serialize_baseline(
        captured.passive_baseline
    ):
        raise RuntimeError("selection HOME provider/passive baseline 未 exact 绑定")
    candidate_receipt = _stage2a._stage2_candidate_receipt_from_dict(
        transaction["candidate_stage_receipt"]
    )
    raw_candidate = transaction["candidate"]
    raw_candidate_digest = (
        None if raw_candidate is None else canonical_sha256(raw_candidate)
    )
    if (
        candidate_receipt.request_id != request_id
        or candidate_receipt.collect_frame_digests
        != tuple(value.frame_digest for value in primary_frames)
        or candidate_receipt.provider_forward_count != 3
        or candidate_receipt.live_memory_write_executed
        or candidate_receipt.commit_eligible
        or candidate_receipt.memory_write_deferred
        or candidate_receipt.candidate_digest
        != transaction["candidate_stage_receipt"].get("candidate_digest")
        or transaction.get("candidate_digest") != raw_candidate_digest
    ):
        raise RuntimeError("selection capture-only candidate receipt 漂移")
    safety_values = transaction["safety_evidence"]
    authorization_values = transaction["camera_command_authorizations"]
    event_values = transaction["controller_events"]
    home_values = transaction["home_barrier_evidence"]
    if (
        not isinstance(safety_values, list)
        or len(safety_values) != 92
        or not isinstance(authorization_values, list)
        or len(authorization_values) != 91
        or not isinstance(event_values, list)
        or not isinstance(home_values, list)
        or len(home_values) != 4
    ):
        raise RuntimeError("selection safety/authorization/event/HOME count 漂移")
    event_index = 0
    home_frames = []
    home_digests: list[str] = []
    home_timestamps: list[float] = []
    for frame_index, camera_row in enumerate(camera_rows):
        if frame_index > 0:
            _stage2a._verify_stage2a_camera_authorization(
                camera_row,
                authorization_values[frame_index - 1],
                controller=controller,
            )
        safety = _stage2a._verify_stage2a_safety_record(
            camera_row,
            safety_values[frame_index],
            controller=controller,
            contact_comparison_tolerance_n=0.0,
        )
        if (
            frame_index > 0
            and frame_index not in _stage2a.STAGE2A_HOME_BARRIER_FRAME_INDICES
            and not controller.observe_safety(
                safety,
                camera_at_home=bool(
                    camera_row["camera_motion_state"]
                    in {
                        _stage2a.ExternalCameraMotionState.RETURN_HOME.value,
                        _stage2a.ExternalCameraMotionState.VERIFY_HOME_AND_ARM_HOLD.value,
                    }
                    and _stage2a._stage2a_pose_at_home(camera_row)
                ),
            )
        ):
            raise RuntimeError("selection safety replay 遇到未宣告 failure")
        replay_events: list[tuple[Any, str | None, Any | None]] = []
        if frame_index == 0:
            controller.begin(reset_receipt)
            replay_events.extend(
                (
                    (_stage2a.ActiveFrontSignal.CAMERA_LEASE_ACQUIRED, None, None),
                    (
                        _stage2a.ActiveFrontSignal.FROZEN_PRIMITIVE_SELECTED,
                        _stage2a.ACTIVE_FRONT_PRIMARY_PRIMITIVE_ID,
                        None,
                    ),
                )
            )
        elif frame_index == 40:
            replay_events.append(
                (_stage2a.ActiveFrontSignal.MOVE_COMPLETE, None, None)
            )
        elif frame_index == 44:
            if camera_row["settled"] is not True:
                raise RuntimeError("selection settle completion帧未 settled")
            replay_events.append(
                (_stage2a.ActiveFrontSignal.SETTLE_COMPLETE, None, None)
            )
        elif frame_index == 47:
            replay_events.extend(
                (
                    (_stage2a.ActiveFrontSignal.COLLECTION_COMPLETE, None, None),
                    (
                        _stage2a.ActiveFrontSignal.SHADOW_CANDIDATE_STAGED,
                        None,
                        candidate_receipt,
                    ),
                )
            )
        elif frame_index == 87:
            replay_events.append(
                (_stage2a.ActiveFrontSignal.RETURN_HOME_COMPLETE, None, None)
            )
        for signal, primitive_id, staged_candidate in replay_events:
            if event_index >= len(event_values):
                raise RuntimeError("selection controller event ledger 提前结束")
            _stage2a._replay_stage2a_controller_event(
                controller,
                event_values[event_index],
                signal=signal,
                frame_index=frame_index,
                safety=safety,
                selected_primitive_id=primitive_id,
                candidate_receipt=staged_candidate,
            )
            event_index += 1
        if frame_index in _stage2a.STAGE2A_HOME_BARRIER_FRAME_INDICES:
            home_index = frame_index - _stage2a.STAGE2A_HOME_BARRIER_FRAME_INDICES[0]
            home_frame, home_digest = _stage2a._verify_stage2a_home_barrier_evidence(
                camera_row,
                home_values[home_index],
                episode_id=episode_id,
                request_id=request_id,
                safety_evidence_sha256=safety_values[frame_index][
                    "evidence_sha256"
                ],
            )
            controller.accept_home_v2_barrier_frame(home_frame)
            home_frames.append(home_frame)
            home_digests.append(home_digest)
            home_timestamps.append(
                float(home_values[home_index]["control_timestamp_s"])
            )
    if event_index != len(event_values):
        raise RuntimeError("selection controller event ledger 存在额外/乱序事件")
    home_ids = [value.observation_sequence_id for value in home_frames]
    if (
        transaction["home_observation_sequence_ids"] != home_ids
        or transaction["home_observation_timestamps_s"] != home_timestamps
        or transaction["home_frame_digests"] != home_digests
    ):
        raise RuntimeError("selection HOME barrier receipt 不能重算")
    observation_window = _stage2a.verify_stage2a_observation_v2_window_identity(
        transaction["observation_v2_window_identity"],
        spec=_stage2a.RobotSpec(),
        home_evidence=home_values,
        home_motion_rows=[
            camera_rows[index]
            for index in _stage2a.STAGE2A_HOME_BARRIER_FRAME_INDICES
        ],
        expected_episode_id=episode_id,
        expected_episode_generation=1,
    )
    if observation_window.frame_timestamp_s.tolist() != home_timestamps:
        raise RuntimeError("selection Observation V2/HOME timestamp 漂移")
    _stage2a._verify_stage2a_source_recheck_identity(
        source_recheck,
        trigger_records=trigger_records,
        final_home_evidence=home_values[-1],
        episode_id=episode_id,
        request_id=request_id,
    )
    final_state = _stage2a._object_state_from_snapshot(
        transaction["final_memory_state"]
    )
    if (
        final_state.valid
        or final_state.accepted_update_count != 0
        or transaction["memory_write_count"] != 0
    ):
        raise RuntimeError("selection capture-only route 不得写 Memory")
    replay_receipt = controller.receipt(
        memory_write_count=0,
        provider_forward_count=4,
    )
    _stage2a._verify_stage2a_controller_receipt(
        transaction["controller_receipt"], replay_receipt
    )
    if transaction["route_passed"] is not route_summary["passed"]:
        raise RuntimeError("selection route summary safety bool 未绑定")


def _verify_public_snapshot_files(
    *,
    public_root: Path,
    selection_config_path: str | Path,
    expected_source_git_commit: str,
    expected_source_identity_sha256: str,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    loaded = load_e018_p1_stage2a_selection_config(selection_config_path)
    snapshot = _read_json(public_root / "config_snapshot.json", "selection config snapshot")
    unsigned = dict(snapshot)
    stored = unsigned.pop("snapshot_sha256", None)
    if (
        set(snapshot)
        != {
            "version",
            "config_raw_sha256",
            "config_canonical_sha256",
            "config",
            "snapshot_sha256",
        }
        or stored != canonical_sha256(unsigned)
        or snapshot["version"] != E018_P1_STAGE2A_SELECTION_EXECUTION_VERSION
        or snapshot["config_raw_sha256"] != loaded.raw_sha256
        or snapshot["config_canonical_sha256"] != loaded.canonical_sha256
        or snapshot["config"] != loaded.payload
    ):
        raise RuntimeError("selection config snapshot 漂移")
    source = _read_json(public_root / "source_identity.json", "selection source")
    source_primitive = _require_exact_keys(
        dict(source),
        {"git_commit", "source_tree_sha256", "identity_sha256"},
        "selection source",
    )
    identity = source_primitive.pop("identity_sha256")
    if (
        identity != canonical_sha256(source_primitive)
        or source["git_commit"] != expected_source_git_commit
        or identity != expected_source_identity_sha256
        or not _is_sha256(source["source_tree_sha256"])
    ):
        raise RuntimeError("selection source identity 漂移")
    parent = _read_json(
        public_root / "parent_verification.json", "selection parent verification"
    )
    _require_exact_keys(
        parent,
        {
            "version",
            "verified",
            "accepted_artifact_id",
            "artifact_manifest_internal_sha256",
            "artifact_file_inventory_sha256",
            "completion_marker_internal_sha256",
            "persistence_verification_internal_sha256",
            "local_canonical_verification_internal_sha256",
            "inventory_record_canonical_sha256",
            "replication_state",
            "stage2a_public_verification_sha256",
            "parent_replay_artifact_id",
            "parent_replay_verification_sha256",
            "parent_replay_drive_marker_internal_sha256",
            "parent_replay_local_verification_internal_sha256",
            "parent_replay_inventory_record_canonical_sha256",
            "parent_replay_replication_state",
            "portability_diagnostics",
            "verification_sha256",
        },
        "selection parent verification",
    )
    parent_config = loaded.payload["parents"]
    expected_parent_fields = {
        "accepted_artifact_id": parent_config["accepted_artifact_id"],
        "artifact_manifest_internal_sha256": parent_config[
            "artifact_manifest_internal_sha256"
        ],
        "artifact_file_inventory_sha256": parent_config[
            "artifact_file_inventory_sha256"
        ],
        "completion_marker_internal_sha256": parent_config[
            "completion_marker_internal_sha256"
        ],
        "persistence_verification_internal_sha256": parent_config[
            "persistence_verification_internal_sha256"
        ],
        "local_canonical_verification_internal_sha256": parent_config[
            "local_canonical_verification_internal_sha256"
        ],
        "inventory_record_canonical_sha256": parent_config[
            "inventory_record_canonical_sha256"
        ],
        "replication_state": "REPLICATED",
        "stage2a_public_verification_sha256": parent_config[
            "stage2a_smoke_public_verification_sha256"
        ],
        "parent_replay_artifact_id": parent_config["parent_replay_artifact_id"],
        "parent_replay_verification_sha256": parent_config[
            "parent_replay_verification_sha256"
        ],
        "parent_replay_drive_marker_internal_sha256": parent_config[
            "parent_replay_drive_marker_internal_sha256"
        ],
        "parent_replay_local_verification_internal_sha256": parent_config[
            "parent_replay_local_verification_internal_sha256"
        ],
        "parent_replay_inventory_record_canonical_sha256": parent_config[
            "parent_replay_inventory_record_canonical_sha256"
        ],
        "parent_replay_replication_state": "REPLICATED",
    }
    if (
        _verify_internal_digest(
            parent,
            digest_key="verification_sha256",
            name="selection parent verification",
        )
        != parent.get("verification_sha256")
        or parent.get("verified") is not True
        or parent.get("version")
        != E018_P1_STAGE2A_SELECTION_EXECUTION_VERSION
        or parent.get("replication_state") != "REPLICATED"
        or parent.get("parent_replay_replication_state") != "REPLICATED"
        or parent.get("portability_diagnostics")
        != [
            {
                "classification": (
                    "non-experiment-local-svd-portability-diagnostic"
                ),
                "environment": "python-3.12.3-numpy-1.26.4-system-blas",
                "exit_code": 1,
                "collect_pose_max_abs_difference": 2.220446049250313e-16,
                "parent_gate_effect": "diagnostic-only-no-rejection",
                "fresh_selection_seed_reads": 0,
            },
            {
                "classification": "non-experiment-operator-path-error",
                "exit_code": 1,
                "cause": (
                    "public_execution-subdirectory-appended-to-flat-source-root"
                ),
                "artifact_read_count": 0,
                "artifact_write_count": 0,
                "parent_gate_effect": (
                    "superseded-by-corrected-exit-zero-replay"
                ),
                "fresh_selection_seed_reads": 0,
            },
        ]
        or any(
            parent.get(name) != expected
            for name, expected in expected_parent_fields.items()
        )
    ):
        raise RuntimeError("selection parent verification receipt 漂移")
    return loaded, source, parent


def _verify_e018_p1_stage2a_selection_public(
    *,
    selection_config_path: str | Path,
    stage2a_config_path: str | Path,
    qualification_config_path: str | Path,
    public_root: str | Path,
    expected_source_git_commit: str,
    expected_source_identity_sha256: str,
    require_complete: bool,
) -> dict[str, Any]:
    """公开验证器：签名刻意不接受 private/model/checkpoint/stats root。"""

    root = Path(public_root)
    commit_names = {
        f"{index:06d}.commit.json"
        for index in range(STAGE2A_SELECTION_PREDICTION_COUNT)
    }
    _assert_exact_tree(
        root,
        expected_files=(
            _PUBLIC_FILES if require_complete else _PUBLIC_PRECOMPLETION_FILES
        ),
        expected_directory="prediction_commits",
        expected_directory_files=commit_names,
        name="selection public artifact",
    )
    loaded, source, parent = _verify_public_snapshot_files(
        public_root=root,
        selection_config_path=selection_config_path,
        expected_source_git_commit=expected_source_git_commit,
        expected_source_identity_sha256=expected_source_identity_sha256,
    )
    stage2a_config = load_e018_p1_stage2a_config(stage2a_config_path)
    qualification = load_g2c_dynamic_qualification_config(
        qualification_config_path
    )
    if (
        stage2a_config.raw_sha256
        != loaded.payload["parents"]["stage2a_config_raw_sha256"]
        or qualification.get("config_sha256")
        != stage2a_config.payload["parents"][
            "d048_qualification_config_internal_sha256"
        ]
    ):
        raise RuntimeError("selection public verifier config parents 漂移")
    started = _read_json(root / "RUN_STARTED.json", "selection RUN_STARTED")
    _require_exact_keys(
        started,
        {
            "version",
            "status",
            "experiment_id",
            "classification",
            "effect_claim",
            "seed_range",
            "gain_order",
            "transaction_identity_sha256",
            "public_artifact_role_identity_sha256",
            "private_artifact_role_identity_sha256",
            "result_root_created",
            "gpu_wall_seconds_max",
            "stop_conditions",
            "fresh_test_reads",
            "started_at_unix_ns",
            "run_started_sha256",
        },
        "selection RUN_STARTED",
    )
    transaction_primitive = {
        "version": E018_P1_STAGE2A_SELECTION_EXECUTION_VERSION,
        "experiment_id": E018_P1_STAGE2A_SELECTION_EXPERIMENT_ID,
        "config_raw_sha256": loaded.raw_sha256,
        "config_canonical_sha256": loaded.canonical_sha256,
        "source_identity_sha256": source["identity_sha256"],
        "parent_verification_sha256": parent["verification_sha256"],
        "public_artifact_role_identity_sha256": _artifact_role_identity_sha256(
            role="public_execution",
            config_canonical_sha256=loaded.canonical_sha256,
            source_identity_sha256=source["identity_sha256"],
            parent_verification_sha256=parent["verification_sha256"],
        ),
        "private_artifact_role_identity_sha256": (
            _artifact_role_identity_sha256(
                role="private_labels",
                config_canonical_sha256=loaded.canonical_sha256,
                source_identity_sha256=source["identity_sha256"],
                parent_verification_sha256=parent["verification_sha256"],
            )
        ),
        "seeds": [77001, 77025],
        "gain_order": list(STAGE2A_SELECTION_GAINS),
    }
    if (
        _verify_internal_digest(
            started,
            digest_key="run_started_sha256",
            name="selection RUN_STARTED",
        )
        != started.get("run_started_sha256")
        or started.get("experiment_id")
        != E018_P1_STAGE2A_SELECTION_EXPERIMENT_ID
        or started.get("version")
        != E018_P1_STAGE2A_SELECTION_EXECUTION_VERSION
        or started.get("status")
        != "PASS_A_IN_PROGRESS_NO_TEST_NO_ACTUATION"
        or started.get("classification") != _SELECTION_CLASSIFICATION
        or started.get("effect_claim") != "no-effect-claim"
        or started.get("seed_range") != [77001, 77025]
        or started.get("gain_order") != list(STAGE2A_SELECTION_GAINS)
        or started.get("fresh_test_reads") != 0
        or started.get("public_artifact_role_identity_sha256")
        != transaction_primitive["public_artifact_role_identity_sha256"]
        or started.get("private_artifact_role_identity_sha256")
        != transaction_primitive["private_artifact_role_identity_sha256"]
        or started.get("transaction_identity_sha256")
        != canonical_sha256(transaction_primitive)
        or started.get("result_root_created") is not False
        or started.get("gpu_wall_seconds_max")
        != loaded.payload["budgets"]["gpu_wall_seconds_max"]
        or started.get("stop_conditions")
        != [
            "hard-protocol-or-safety-exception",
            "gpu-wall-budget-exceeded",
            "artifact-budget-exceeded",
        ]
        or type(started.get("started_at_unix_ns")) is not int
        or started["started_at_unix_ns"] <= 0
    ):
        raise RuntimeError("selection RUN_STARTED identity 漂移")
    provider_rows = _read_jsonl(
        root / "provider_output_ledger.jsonl", "selection provider ledger"
    )
    if len(provider_rows) != STAGE2A_SELECTION_PREDICTION_COUNT:
        raise RuntimeError("selection provider ledger 必须 100 行")
    prediction_receipts = _verify_prediction_commit_chain(
        root,
        provider_rows,
        expected_transaction_identity_sha256=started[
            "transaction_identity_sha256"
        ],
    )
    records = [_stage2a._provider_record_from_dict(row) for row in provider_rows]
    for index, record in enumerate(records):
        expected_seed = STAGE2A_SELECTION_SEEDS[index // 4]
        expected_frame = STAGE2A_PROVIDER_FRAME_INDICES[index % 4]
        if (
            record.prediction.get("seed") != expected_seed
            or record.route_frame_index != expected_frame
        ):
            raise RuntimeError("selection provider seed/frame order 漂移")
        verify_stage2a_provider_output_record(
            record,
            stage2_config=stage2a_config,
            qualification_config=qualification,
            expected_classification=(
                _stage2a.QUALIFICATION_CLASSIFICATION_SELECTION
            ),
        )
    camera_rows = _read_jsonl(
        root / "camera_pose_ledger.jsonl", "selection camera ledger"
    )
    route_rows = _read_jsonl(
        root / "route_evidence_ledger.jsonl", "selection route ledger"
    )
    branch_rows = _read_jsonl(
        root / "gain_branch_ledger.jsonl", "selection gain branch ledger"
    )
    if (
        len(camera_rows) != 2300
        or len(route_rows) != 25
        or len(branch_rows) != STAGE2A_SELECTION_BRANCH_COUNT
    ):
        raise RuntimeError("selection camera/route/branch ledger count 漂移")
    private_inventory_value = _read_json(
        root / "private_label_inventory.json", "selection private inventory"
    )
    private_inventory = _verify_private_inventory_public(private_inventory_value)
    captured_routes = [
        _verify_route_evidence_row(
            route_row,
            route_index=index,
            camera_rows=camera_rows,
            provider_records=records,
            prediction_receipts=prediction_receipts,
            private_inventory_rows=private_inventory,
            stage2a_config=stage2a_config,
            qualification_config=qualification,
        )
        for index, route_row in enumerate(route_rows)
    ]
    verified_branches = [_validated_gain_branch(row) for row in branch_rows]
    for route_index, captured in enumerate(captured_routes):
        expected = [item.to_dict() for item in replay_all_gain_branches(captured)]
        actual = verified_branches[route_index * 3 : route_index * 3 + 3]
        if actual != expected:
            raise RuntimeError(
                f"selection route[{route_index}] gain branches 不能纯逻辑重放"
            )
    receipt = _read_json(root / "execution_receipt.json", "selection execution receipt")
    _require_exact_keys(
        receipt,
        {
            "version",
            "status",
            "classification",
            "effect_claim",
            "experiment_id",
            "config_raw_sha256",
            "config_canonical_sha256",
            "source_identity_sha256",
            "parent_verification_sha256",
            "transaction_identity_sha256",
            "context_destroyed",
            "provider_context_destroyed",
            "environment_closed",
            "pass_b_process_started",
            "counts",
            "gpu_wall_seconds",
            "gpu_wall_seconds_max",
            "gpu_budget_passed",
            "environment_identity",
            "producer_process_identity",
            "execution_freeze_raw_sha256",
            "execution_freeze_internal_sha256",
            "formal_claim_allowed",
            "fresh_test_status",
            "receipt_sha256",
        },
        "selection execution receipt",
    )
    environment = _require_exact_keys(
        receipt["environment_identity"],
        {
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
            "provider_context_destroyed",
            "environment_closed",
            "normalizer_identity",
        },
        "selection execution environment",
    )
    normalizers = _require_exact_keys(
        environment["normalizer_identity"],
        {
            "proprio_stats_sha256",
            "proprio_normalizer_sha256",
            "finger_force_stats_sha256",
            "finger_force_normalizer_sha256",
        },
        "selection normalizer identity",
    )
    expected_normalizers = {
        name: qualification["parents"][name]
        for name in normalizers
    }
    producer_process = _verify_process_identity(
        receipt["producer_process_identity"],
        role="pass-a-producer",
    )
    if (
        _verify_internal_digest(
            receipt,
            digest_key="receipt_sha256",
            name="selection execution receipt",
        )
        != receipt.get("receipt_sha256")
        or receipt.get("experiment_id")
        != E018_P1_STAGE2A_SELECTION_EXPERIMENT_ID
        or receipt.get("version")
        != E018_P1_STAGE2A_SELECTION_EXECUTION_VERSION
        or receipt.get("status") != "PASS_A_COMPLETE_CONTEXT_DESTROYED"
        or receipt.get("classification") != _SELECTION_CLASSIFICATION
        or receipt.get("effect_claim") != "no-effect-claim"
        or receipt.get("config_raw_sha256") != loaded.raw_sha256
        or receipt.get("config_canonical_sha256") != loaded.canonical_sha256
        or receipt.get("source_identity_sha256") != source["identity_sha256"]
        or receipt.get("parent_verification_sha256")
        != parent["verification_sha256"]
        or receipt.get("transaction_identity_sha256")
        != started["transaction_identity_sha256"]
        or receipt.get("context_destroyed") is not True
        or receipt.get("provider_context_destroyed") is not True
        or receipt.get("environment_closed") is not True
        or receipt.get("pass_b_process_started") is not False
        or receipt.get("counts")
        != {
            "seed_count": 25,
            "route_count": 25,
            "camera_frame_count": 2300,
            "provider_forward_count": 100,
            "private_label_capture_count": 75,
            "private_label_open_count": 0,
            "gain_branch_count": 75,
            "branch_provider_forward_count": 0,
            "arm_motion_command_count": 0,
            "gripper_close_command_count": 0,
            "runtime_object_gt_read_count": 0,
            "goal_gt_read_count": 0,
            "fresh_test_read_count": 0,
            "checkpoint_write_count": 0,
        }
        or receipt.get("fresh_test_status") != "prohibited-unread"
        or receipt.get("formal_claim_allowed") is not False
        or _finite_real(receipt.get("gpu_wall_seconds"), "selection GPU wall")
        < 0.0
        or receipt.get("gpu_wall_seconds_max")
        != loaded.payload["budgets"]["gpu_wall_seconds_max"]
        or receipt.get("gpu_budget_passed") is not True
        or receipt["gpu_wall_seconds"] > receipt["gpu_wall_seconds_max"]
        or environment["cuda_available"] is not True
        or environment["python"] != "3.10.12"
        or environment["numpy"] != "1.26.4"
        or environment["torch"] != "2.11.0+cu128"
        or environment["cuda_device"]
        != "NVIDIA RTX 6000 Ada Generation"
        or environment["mani_skill"] != "3.0.1"
        or environment["sapien"] != "3.0.3"
        or environment["external_camera_sensor_class"]
        != "mani_skill.sensors.camera.Camera"
        or environment["external_camera_class"]
        != "mani_skill.utils.structs.render_camera.RenderCamera"
        or environment["external_camera_unmounted"] is not True
        or environment["provider_context_destroyed"] is not True
        or environment["environment_closed"] is not True
        or normalizers != expected_normalizers
    ):
        raise RuntimeError("selection execution receipt identity/accounting 漂移")
    freeze = _read_json(root / "execution_freeze.json", "selection execution freeze")
    _require_exact_keys(
        freeze,
        {
            "version",
            "status",
            "context_destroyed",
            "provider_context_destroyed",
            "environment_closed",
            "prediction_ledger_frozen",
            "route_evidence_frozen",
            "gain_branches_frozen",
            "private_labels_write_only",
            "private_label_open_count",
            "provider_forward_count",
            "branch_provider_forward_count",
            "producer_process_identity",
            "artifact_inventory",
            "frozen_at_unix_ns",
            "freeze_sha256",
        },
        "selection execution freeze",
    )
    if (
        _verify_internal_digest(
            freeze,
            digest_key="freeze_sha256",
            name="selection execution freeze",
        )
        != freeze.get("freeze_sha256")
        or freeze.get("context_destroyed") is not True
        or freeze.get("version")
        != E018_P1_STAGE2A_SELECTION_EXECUTION_VERSION
        or freeze.get("status") != "PASS_A_FROZEN_AFTER_CONTEXT_DESTROY"
        or freeze.get("provider_context_destroyed") is not True
        or freeze.get("environment_closed") is not True
        or freeze.get("prediction_ledger_frozen") is not True
        or freeze.get("route_evidence_frozen") is not True
        or freeze.get("gain_branches_frozen") is not True
        or freeze.get("private_labels_write_only") is not True
        or freeze.get("private_label_open_count") != 0
        or freeze.get("provider_forward_count") != 100
        or freeze.get("branch_provider_forward_count") != 0
        or _verify_process_identity(
            freeze.get("producer_process_identity"),
            role="pass-a-producer",
        )
        != producer_process
        or receipt.get("execution_freeze_raw_sha256")
        != file_sha256(root / "execution_freeze.json")
        or receipt.get("execution_freeze_internal_sha256")
        != freeze["freeze_sha256"]
        or type(freeze.get("frozen_at_unix_ns")) is not int
        or freeze["frozen_at_unix_ns"] <= 0
    ):
        raise RuntimeError("selection execution freeze 漂移")
    frozen_inventory = freeze.get("artifact_inventory")
    if not isinstance(frozen_inventory, dict):
        raise TypeError("selection freeze inventory schema 漂移")
    expected_frozen_paths = {
        "RUN_STARTED.json",
        "config_snapshot.json",
        "source_identity.json",
        "parent_verification.json",
        "camera_pose_ledger.jsonl",
        "provider_output_ledger.jsonl",
        "route_evidence_ledger.jsonl",
        "gain_branch_ledger.jsonl",
        "private_label_inventory.json",
        *{
            f"prediction_commits/{index:06d}.commit.json"
            for index in range(STAGE2A_SELECTION_PREDICTION_COUNT)
        },
    }
    if set(frozen_inventory) != expected_frozen_paths:
        raise RuntimeError("selection freeze artifact inventory key set 漂移")
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
            raise RuntimeError(f"selection frozen artifact 漂移: {relative}")
    precompletion = {
        "version": E018_P1_STAGE2A_SELECTION_EXECUTION_VERSION,
        "verified": True,
        "complete_marker_verified": False,
        "experiment_id": E018_P1_STAGE2A_SELECTION_EXPERIMENT_ID,
        "config_raw_sha256": loaded.raw_sha256,
        "config_canonical_sha256": loaded.canonical_sha256,
        "source_git_commit": source["git_commit"],
        "source_identity_sha256": source["identity_sha256"],
        "parent_verification_sha256": parent["verification_sha256"],
        "execution_receipt_sha256": receipt["receipt_sha256"],
        "execution_freeze_sha256": freeze["freeze_sha256"],
        "provider_prediction_count": len(provider_rows),
        "private_label_inventory_count": len(private_inventory),
        "route_count": len(route_rows),
        "gain_branch_count": len(branch_rows),
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
        "producer_process_identity": producer_process,
    }
    precompletion["verification_sha256"] = canonical_sha256(precompletion)
    if not require_complete:
        return precompletion
    marker = _read_json(
        root / "PUBLIC_EXECUTION_COMPLETE.json",
        "selection public completion marker",
    )
    _require_exact_keys(
        marker,
        {
            "version",
            "status",
            "experiment_id",
            "transaction_identity_sha256",
            "precompletion_verification_sha256",
            "execution_receipt_raw_sha256",
            "execution_receipt_internal_sha256",
            "private_label_open_count",
            "fresh_test_reads",
            "completed_at_unix_ns",
            "marker_sha256",
        },
        "selection public completion marker",
    )
    if (
        _verify_internal_digest(
            marker,
            digest_key="marker_sha256",
            name="selection public completion marker",
        )
        != marker.get("marker_sha256")
        or marker.get("status") != "PUBLIC_EXECUTION_COMPLETE"
        or marker.get("version")
        != E018_P1_STAGE2A_SELECTION_EXECUTION_VERSION
        or marker.get("experiment_id")
        != E018_P1_STAGE2A_SELECTION_EXPERIMENT_ID
        or marker.get("precompletion_verification_sha256")
        != precompletion["verification_sha256"]
        or marker.get("execution_receipt_raw_sha256")
        != file_sha256(root / "execution_receipt.json")
        or marker.get("execution_receipt_internal_sha256")
        != receipt["receipt_sha256"]
        or marker.get("transaction_identity_sha256")
        != started["transaction_identity_sha256"]
        or marker.get("private_label_open_count") != 0
        or marker.get("fresh_test_reads") != 0
        or type(marker.get("completed_at_unix_ns")) is not int
        or marker["completed_at_unix_ns"] <= 0
    ):
        raise RuntimeError("selection public completion marker 漂移")
    result = {
        **precompletion,
        "complete_marker_verified": True,
        "public_completion_marker_sha256": marker["marker_sha256"],
    }
    result.pop("verification_sha256")
    result["verification_sha256"] = canonical_sha256(result)
    return result


def verify_e018_p1_stage2a_selection_public(
    *,
    selection_config_path: str | Path,
    stage2a_config_path: str | Path,
    qualification_config_path: str | Path,
    public_root: str | Path,
    expected_source_git_commit: str,
    expected_source_identity_sha256: str,
) -> dict[str, Any]:
    return _verify_e018_p1_stage2a_selection_public(
        selection_config_path=selection_config_path,
        stage2a_config_path=stage2a_config_path,
        qualification_config_path=qualification_config_path,
        public_root=public_root,
        expected_source_git_commit=expected_source_git_commit,
        expected_source_identity_sha256=expected_source_identity_sha256,
        require_complete=True,
    )


def _artifact_role_identity_sha256(
    *,
    role: str,
    config_canonical_sha256: str,
    source_identity_sha256: str,
    parent_verification_sha256: str,
) -> str:
    """位置无关 artifact role identity；复制后仍能独立重算。"""

    if role not in {"public_execution", "private_labels", "result"}:
        raise ValueError("selection artifact role 非法")
    if any(
        not _is_sha256(value)
        for value in (
            config_canonical_sha256,
            source_identity_sha256,
            parent_verification_sha256,
        )
    ):
        raise ValueError("selection artifact role parent identity 非法")
    return canonical_sha256(
        {
            "version": E018_P1_STAGE2A_SELECTION_EXECUTION_VERSION,
            "experiment_id": E018_P1_STAGE2A_SELECTION_EXPERIMENT_ID,
            "role": role,
            "config_canonical_sha256": config_canonical_sha256,
            "source_identity_sha256": source_identity_sha256,
            "parent_verification_sha256": parent_verification_sha256,
        }
    )


def _assert_capture_authority(
    *,
    selection_config_path: str | Path,
    repository_root: Path,
    expected_config_raw_sha256: str,
    expected_config_canonical_sha256: str,
    expected_source_git_commit: str,
    expected_source_identity_sha256: str,
    exact_go_token: str,
) -> tuple[Any, dict[str, Any]]:
    if exact_go_token != STAGE2A_SELECTION_GO:
        raise PermissionError("selection capture 缺 exact GO token")
    loaded = load_e018_p1_stage2a_selection_config(selection_config_path)
    if (
        loaded.raw_sha256 != expected_config_raw_sha256
        or loaded.canonical_sha256 != expected_config_canonical_sha256
    ):
        raise RuntimeError("selection config 不匹配冻结 exact identity")
    source = _git_source_identity(repository_root)
    if (
        source["git_commit"] != expected_source_git_commit
        or source["identity_sha256"] != expected_source_identity_sha256
    ):
        raise RuntimeError("selection capture 要求 exact-clean source identity")
    return loaded, source


def _assert_preflight_authority(
    *,
    selection_config_path: str | Path,
    repository_root: Path,
    expected_config_raw_sha256: str,
    expected_config_canonical_sha256: str,
    expected_source_git_commit: str,
    expected_source_identity_sha256: str,
    exact_preflight_token: str,
) -> tuple[Any, dict[str, Any]]:
    """只授权固定 76891 preflight；formal GO 在这里必须无效。"""

    if exact_preflight_token != STAGE2A_SELECTION_PREFLIGHT_GO:
        raise PermissionError("Stage 2A preflight 缺 exact preflight token")
    loaded = load_e018_p1_stage2a_selection_config(selection_config_path)
    if (
        loaded.raw_sha256 != expected_config_raw_sha256
        or loaded.canonical_sha256 != expected_config_canonical_sha256
    ):
        raise RuntimeError("preflight config 不匹配冻结 selection config identity")
    source = _git_source_identity(repository_root)
    if (
        source["git_commit"] != expected_source_git_commit
        or source["identity_sha256"] != expected_source_identity_sha256
    ):
        raise RuntimeError("preflight 要求 exact-clean source identity")
    return loaded, source


def _build_preflight_stats_identity(
    stats_root: Path,
    data_config: Mapping[str, Any],
) -> dict[str, Any]:
    """把实际读取的两份 normalizer stats 绑定进 preflight identity。"""

    paths = {
        "proprio_stats_raw_sha256": stats_root / "proprio_stats.json",
        "finger_force_stats_raw_sha256": stats_root / "finger_force_stats.json",
    }
    for path in paths.values():
        if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
            raise RuntimeError("preflight stats 必须是单链接 regular file")
    value = {
        "version": "e018-p1-stage2a-preflight-stats-identity/v1",
        "data_config_canonical_sha256": canonical_sha256(data_config),
        **{name: file_sha256(path) for name, path in paths.items()},
    }
    expected = data_config["data_identity"]
    if (
        value["proprio_stats_raw_sha256"]
        != expected["proprio_stats_sha256"]
        or value["finger_force_stats_raw_sha256"]
        != expected["finger_force_stats_sha256"]
    ):
        raise RuntimeError("preflight stats raw SHA-256 与冻结 data config 不匹配")
    value["stats_identity_sha256"] = canonical_sha256(value)
    return value


def _run_selection_simulator(
    *,
    loaded_selection_config: Any,
    loaded_stage2a_config: Any,
    qualification_config: Mapping[str, Any],
    g0c_config: dict[str, Any],
    data_config: Mapping[str, Any],
    stats_root: Path,
    selected_checkpoint_path: Path,
    public_root: Path,
    journal: Stage2ASelectionJournal | Stage2ASelectionPreflightJournal,
    execution_progress: Stage2AExecutionProgress,
    started_monotonic: float,
    preflight_one_route: bool = False,
) -> tuple[
    list[dict[str, Any]],
    list[CapturedSelectionRoute],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[Stage2AProviderOutputRecord],
    list[list[dict[str, Any]]],
    list[list[dict[str, Any]]],
    dict[str, Any],
    bool,
]:
    """唯一 physical route 进程；返回值在 finally 后不含 env/provider。"""

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
        raise RuntimeError("selection GPU/ManiSkill/SAPIEN environment 漂移")
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
    captured_routes: list[CapturedSelectionRoute] = []
    route_summaries: list[dict[str, Any]] = []
    transactions: list[dict[str, Any]] = []
    provider_records: list[Stage2AProviderOutputRecord] = []
    route_prediction_receipts: list[list[dict[str, Any]]] = []
    route_private_metadata: list[list[dict[str, Any]]] = []
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
            raise RuntimeError("selection G0C primitive order 漂移")
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
            raise RuntimeError("selection environment/control/camera identity 漂移")
        sensor = base_env._sensors[environment["camera_uid"]]
        camera = sensor.camera
        if sensor.entity is not None or not callable(
            getattr(camera, "set_local_pose", None)
        ):
            raise RuntimeError("selection 要求 isolated unmounted external camera")
        route_seeds = (
            (STAGE2A_SELECTION_PREFLIGHT_SEED,)
            if preflight_one_route
            else STAGE2A_SELECTION_SEEDS
        )
        for seed in route_seeds:
            if (
                time.monotonic() - started_monotonic
                > loaded_selection_config.payload["budgets"][
                    "gpu_wall_seconds_max"
                ]
            ):
                raise TimeoutError("selection GPU wall budget 已到停止条件")
            if preflight_one_route:
                execution_progress.begin_information_gain_preflight(
                    seed,
                    experiment_identity=(
                        E018_P1_STAGE2A_SELECTION_PREFLIGHT_EXPERIMENT_ID
                    ),
                )
            else:
                execution_progress.begin_information_gain_selection(
                    seed,
                    experiment_identity=E018_P1_STAGE2A_SELECTION_EXPERIMENT_ID,
                )
            prediction_receipts: list[dict[str, Any]] = []
            private_metadata: list[dict[str, Any]] = []

            def provider_commit_hook(
                record: Stage2AProviderOutputRecord,
                motion_row: Mapping[str, Any],
                rgb: np.ndarray,
                observation: Mapping[str, Any],
                _bound_seed: int = seed,
                _bound_prediction_receipts: list[dict[str, Any]] = (
                    prediction_receipts
                ),
                _bound_private_metadata: list[dict[str, Any]] = private_metadata,
            ) -> None:
                frame_index = record.route_frame_index
                rgb_array = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
                if (
                    rgb_array.shape != (128, 128, 3)
                    or hashlib.sha256(rgb_array.tobytes()).hexdigest()
                    != motion_row["rgb_sha256"]
                ):
                    raise RuntimeError("selection hook RGB/route identity 漂移")
                receipt = journal.commit_prediction(
                    record.to_dict(),
                    seed=_bound_seed,
                    route_frame_index=frame_index,
                    provider_output_digest=record.provider_output_digest,
                    model_input_digest=record.model_input_digest,
                )
                _bound_prediction_receipts.append(receipt)
                if frame_index in STAGE2A_COLLECT_FRAME_INDICES:
                    actual_pose = np.asarray(
                        motion_row["actual_base_from_external_camera_cv"],
                        dtype=np.float64,
                    )
                    metadata = journal.capture_private_label_after_prediction(
                        prediction_receipt=receipt,
                        seed=_bound_seed,
                        route_frame_index=frame_index,
                        rgb_sha256=motion_row["rgb_sha256"],
                        actual_pose_sha256=_array_sha256(actual_pose),
                        provider_output_digest=record.provider_output_digest,
                        privileged_getter=lambda: capture_selection_private_label(
                            observation=observation,
                            base_env=base_env,
                            prediction=record.prediction,
                            data_config=data_config,
                        ),
                    )
                    _bound_private_metadata.append(metadata)

            factory = (
                Stage2ARouteTransaction.for_information_gain_selection_preflight_capture
                if preflight_one_route
                else Stage2ARouteTransaction.for_information_gain_selection_capture
            )
            transaction = factory(
                experiment_identity=(
                    E018_P1_STAGE2A_SELECTION_PREFLIGHT_EXPERIMENT_ID
                    if preflight_one_route
                    else E018_P1_STAGE2A_SELECTION_EXPERIMENT_ID
                ),
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
                episode_prefix=(
                    "stage2a-selection-preflight"
                    if preflight_one_route
                    else "stage2a-selection"
                ),
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
                "classification": (
                    "engineering-preflight-selection-capture-only-no-test-no-actuation/v1"
                    if preflight_one_route
                    else "formal-development-selection-capture-only-no-test-no-actuation/v1"
                ),
                "provider_forward_count": len(transaction.provider_records),
                "memory_write_count": transaction.orchestrator.memory_write_count,
                "offline_segmentation_diagnostics": False,
                "runtime_object_gt_reads": 0,
                "goal_gt_reads": 0,
                "fresh_test_reads": 0,
            }
            transaction_row = transaction.finalize(route_summary)
            captured = (
                None
                if preflight_one_route
                else CapturedSelectionRoute.from_transaction_export(
                    transaction.selection_replay_inputs()
                )
            )
            if (
                len(route_rows) != 92
                or len(prediction_receipts) != 4
                or len(private_metadata) != 3
                or transaction.orchestrator.memory_write_count != 0
            ):
                raise RuntimeError("selection route capture accounting 漂移")
            rows.extend(route_rows)
            route_summaries.append(route_summary)
            transactions.append(transaction_row)
            if captured is not None:
                captured_routes.append(captured)
            provider_records.extend(transaction.provider_records)
            route_prediction_receipts.append(prediction_receipts)
            route_private_metadata.append(private_metadata)
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
    context_destroyed = bool(env_closed and provider_destroyed)
    return (
        rows,
        captured_routes,
        route_summaries,
        transactions,
        provider_records,
        route_prediction_receipts,
        route_private_metadata,
        environment_identity,
        context_destroyed,
    )


def run_e018_p1_stage2a_selection_preflight_one_route(
    *,
    selection_config_path: str | Path,
    stage2a_config_path: str | Path,
    qualification_config_path: str | Path,
    g0c_config_path: str | Path,
    data_config_path: str | Path,
    stats_root: str | Path,
    selected_checkpoint_path: str | Path,
    repository_root: str | Path,
    artifact_root: str | Path,
    stage2a_artifact_root: str | Path,
    stage2a_control_evidence_root: str | Path,
    artifact_inventory_path: str | Path,
    parent_replay_artifact_root: str | Path,
    parent_replay_control_evidence_root: str | Path,
    expected_config_raw_sha256: str,
    expected_config_canonical_sha256: str,
    expected_source_git_commit: str,
    expected_source_identity_sha256: str,
    exact_preflight_token: str,
) -> dict[str, Any]:
    """固定 seed 76891 的单路线 GPU preflight；不触碰正式 split。"""

    artifact = Path(artifact_root)
    if artifact.exists():
        raise FileExistsError(f"preflight artifact root 已存在: {artifact}")
    loaded, source = _assert_preflight_authority(
        selection_config_path=selection_config_path,
        repository_root=Path(repository_root),
        expected_config_raw_sha256=expected_config_raw_sha256,
        expected_config_canonical_sha256=expected_config_canonical_sha256,
        expected_source_git_commit=expected_source_git_commit,
        expected_source_identity_sha256=expected_source_identity_sha256,
        exact_preflight_token=exact_preflight_token,
    )
    stage2a = load_e018_p1_stage2a_config(stage2a_config_path)
    parent = verify_selection_parent_gate(
        selection_config_path=selection_config_path,
        stage2a_config_path=stage2a_config_path,
        qualification_config_path=qualification_config_path,
        stage2a_artifact_root=stage2a_artifact_root,
        stage2a_control_evidence_root=stage2a_control_evidence_root,
        artifact_inventory_path=artifact_inventory_path,
        parent_replay_artifact_root=parent_replay_artifact_root,
        parent_replay_control_evidence_root=(
            parent_replay_control_evidence_root
        ),
    )
    qualification = load_g2c_dynamic_qualification_config(
        qualification_config_path
    )
    g0c = _stage2a._g0c.load_e018_p1_g0c_config(g0c_config_path)
    data = _stage2a.load_e018_p1_g2c_data_config(
        data_config_path,
        parent_g0c_config_path=g0c_config_path,
    )
    stats_identity = _build_preflight_stats_identity(Path(stats_root), data)
    checkpoint = Path(selected_checkpoint_path)
    if (
        not checkpoint.is_file()
        or checkpoint.is_symlink()
        or checkpoint.stat().st_nlink != 1
    ):
        raise RuntimeError("preflight checkpoint 必须是单链接 regular file")
    artifact.mkdir(mode=0o700, parents=True, exist_ok=False)
    public_root = artifact / "public_preflight"
    private_root = artifact / "private_labels"
    transaction_primitive = {
        "version": E018_P1_STAGE2A_SELECTION_PREFLIGHT_EXECUTION_VERSION,
        "experiment_id": E018_P1_STAGE2A_SELECTION_PREFLIGHT_EXPERIMENT_ID,
        "seed": STAGE2A_SELECTION_PREFLIGHT_SEED,
        "config_raw_sha256": loaded.raw_sha256,
        "config_canonical_sha256": loaded.canonical_sha256,
        "source_identity_sha256": source["identity_sha256"],
        "parent_verification_sha256": parent["verification_sha256"],
        "checkpoint_raw_sha256": file_sha256(checkpoint),
        "proprio_stats_raw_sha256": stats_identity[
            "proprio_stats_raw_sha256"
        ],
        "finger_force_stats_raw_sha256": stats_identity[
            "finger_force_stats_raw_sha256"
        ],
        "stats_identity_sha256": stats_identity["stats_identity_sha256"],
    }
    transaction_identity = canonical_sha256(transaction_primitive)
    journal = Stage2ASelectionPreflightJournal(
        public_root=public_root,
        private_root=private_root,
        config_canonical_sha256=loaded.canonical_sha256,
        transaction_identity_sha256=transaction_identity,
    )
    snapshot = {
        "version": E018_P1_STAGE2A_SELECTION_PREFLIGHT_EXECUTION_VERSION,
        "config_raw_sha256": loaded.raw_sha256,
        "config_canonical_sha256": loaded.canonical_sha256,
        "config": loaded.payload,
    }
    snapshot["snapshot_sha256"] = canonical_sha256(snapshot)
    checkpoint_identity = {
        "version": E018_P1_STAGE2A_SELECTION_PREFLIGHT_EXECUTION_VERSION,
        "checkpoint_basename": checkpoint.name,
        "checkpoint_raw_sha256": transaction_primitive[
            "checkpoint_raw_sha256"
        ],
        "checkpoint_write_count": 0,
    }
    checkpoint_identity["identity_sha256"] = canonical_sha256(
        checkpoint_identity
    )
    _atomic_create_json(public_root / "config_snapshot.json", snapshot)
    _atomic_create_json(public_root / "source_identity.json", source)
    _atomic_create_json(public_root / "parent_verification.json", parent)
    _atomic_create_json(
        public_root / "checkpoint_identity.json", checkpoint_identity
    )
    progress = Stage2AExecutionProgress()
    started_monotonic = time.monotonic()
    try:
        (
            camera_rows,
            captured_routes,
            route_summaries,
            transactions,
            provider_records,
            route_receipts,
            route_private,
            environment_identity,
            context_destroyed,
        ) = _run_selection_simulator(
            loaded_selection_config=loaded,
            loaded_stage2a_config=stage2a,
            qualification_config=qualification,
            g0c_config=g0c,
            data_config=data,
            stats_root=Path(stats_root),
            selected_checkpoint_path=checkpoint,
            public_root=public_root,
            journal=journal,
            execution_progress=progress,
            started_monotonic=started_monotonic,
            preflight_one_route=True,
        )
        provider_freeze, private_inventory = (
            journal.finalize_preflight_capture()
        )
        if (
            not context_destroyed
            or captured_routes
            or len(camera_rows) != 92
            or len(route_summaries) != 1
            or len(transactions) != 1
            or len(provider_records) != 4
            or [len(value) for value in route_receipts] != [4]
            or [len(value) for value in route_private] != [3]
            or provider_freeze["row_count"] != 4
            or transactions[0].get("memory_write_count") != 0
            or transactions[0].get("shadow_action_generation") is not None
        ):
            raise RuntimeError("preflight 固定单路线 accounting 漂移")
        _atomic_jsonl(public_root / "camera_pose_ledger.jsonl", camera_rows)
        _atomic_create_json(public_root / "route_summary.json", route_summaries[0])
        _atomic_create_json(public_root / "transaction.json", transactions[0])
        _atomic_create_json(
            public_root / "private_label_inventory.json", private_inventory
        )
        wall_seconds = time.monotonic() - started_monotonic
        counts = {
            "route_count": 1,
            "camera_frame_count": 92,
            "provider_forward_count": 4,
            "private_label_capture_count": 3,
            "memory_write_count": 0,
            "fresh_shadow_action_generation_count": 0,
            "branch_provider_forward_count": 0,
            "fresh_test_read_count": 0,
            "arm_motion_command_count": 0,
            "tcp_motion_command_count": 0,
            "gripper_command_count": 0,
        }
        receipt = {
            "version": E018_P1_STAGE2A_SELECTION_PREFLIGHT_EXECUTION_VERSION,
            "status": "PREFLIGHT_ROUTE_COMPLETE_CONTEXT_DESTROYED",
            "classification": (
                "engineering-preflight-selection-capture-only-no-test-no-actuation/v1"
            ),
            "effect_claim": "no-effect-claim",
            "experiment_id": E018_P1_STAGE2A_SELECTION_PREFLIGHT_EXPERIMENT_ID,
            "seed": STAGE2A_SELECTION_PREFLIGHT_SEED,
            "config_raw_sha256": loaded.raw_sha256,
            "config_canonical_sha256": loaded.canonical_sha256,
            "source_git_commit": source["git_commit"],
            "source_identity_sha256": source["identity_sha256"],
            "parent_verification_sha256": parent["verification_sha256"],
            "checkpoint_identity_sha256": checkpoint_identity[
                "identity_sha256"
            ],
            "stats_identity": stats_identity,
            "transaction_identity_sha256": transaction_identity,
            "context_destroyed": True,
            "provider_context_destroyed": True,
            "environment_closed": True,
            "counts": counts,
            "environment_identity": environment_identity,
            "gpu_wall_seconds": wall_seconds,
            "formal_selection_identity_consumed": False,
            "formal_completion_created": False,
            "scoring_consumption_marker_created": False,
            "result_root_created": False,
            "fresh_test_status": "prohibited-unread",
        }
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        _atomic_create_json(public_root / "preflight_receipt.json", receipt)
        precompletion_paths = sorted(
            _PREFLIGHT_PUBLIC_FILES
            - {"PREFLIGHT_COMPLETE.json"}
            | {
                f"prediction_commits/{index:06d}.commit.json"
                for index in range(4)
            }
        )
        completion = {
            "version": E018_P1_STAGE2A_SELECTION_PREFLIGHT_EXECUTION_VERSION,
            "status": "PREFLIGHT_COMPLETE",
            "experiment_id": E018_P1_STAGE2A_SELECTION_PREFLIGHT_EXPERIMENT_ID,
            "seed": STAGE2A_SELECTION_PREFLIGHT_SEED,
            "transaction_identity_sha256": transaction_identity,
            "preflight_receipt_internal_sha256": receipt["receipt_sha256"],
            "private_inventory_internal_sha256": private_inventory[
                "inventory_sha256"
            ],
            "stats_identity_sha256": stats_identity[
                "stats_identity_sha256"
            ],
            "counts": counts,
            "context_destroyed": True,
            "formal_selection_identity_consumed": False,
            "public_artifact_inventory": _file_inventory(
                public_root, precompletion_paths
            ),
            "completed_at_unix_ns": time.time_ns(),
        }
        completion["completion_sha256"] = canonical_sha256(completion)
        _atomic_create_json(public_root / "PREFLIGHT_COMPLETE.json", completion)
        return verify_e018_p1_stage2a_selection_preflight(
            selection_config_path=selection_config_path,
            data_config_path=data_config_path,
            artifact_root=artifact,
            expected_source_git_commit=expected_source_git_commit,
            expected_source_identity_sha256=expected_source_identity_sha256,
        )
    except Exception as error:
        trace = "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )
        failure = {
            "version": E018_P1_STAGE2A_SELECTION_PREFLIGHT_EXECUTION_VERSION,
            "status": "PREFLIGHT_FAILED_EVIDENCE_PRESERVED",
            "experiment_id": E018_P1_STAGE2A_SELECTION_PREFLIGHT_EXPERIMENT_ID,
            "seed": STAGE2A_SELECTION_PREFLIGHT_SEED,
            "transaction_identity_sha256": transaction_identity,
            "error_type": type(error).__name__,
            "error": str(error)[:1024],
            "traceback_tail": trace[-8192:],
            "traceback_sha256": hashlib.sha256(trace.encode()).hexdigest(),
            "progress": progress.as_dict(),
            "formal_selection_identity_consumed": False,
            "fresh_test_reads": 0,
            "failed_at_unix_ns": time.time_ns(),
        }
        failure["failure_sha256"] = canonical_sha256(failure)
        _atomic_create_json(artifact / "PREFLIGHT_FAILURE.json", failure)
        raise


def verify_e018_p1_stage2a_selection_preflight(
    *,
    selection_config_path: str | Path,
    data_config_path: str | Path,
    artifact_root: str | Path,
    expected_source_git_commit: str,
    expected_source_identity_sha256: str,
) -> dict[str, Any]:
    """验证固定单路线 preflight；不打开 private label payload。"""

    artifact = Path(artifact_root)
    _assert_artifact_top_level_directories(
        artifact,
        expected_directories={"public_preflight", "private_labels"},
    )
    public_root = artifact / "public_preflight"
    private_root = artifact / "private_labels"
    commit_names = {f"{index:06d}.commit.json" for index in range(4)}
    _assert_exact_tree(
        public_root,
        expected_files=_PREFLIGHT_PUBLIC_FILES,
        expected_directory="prediction_commits",
        expected_directory_files=commit_names,
        name="selection preflight public artifact",
    )
    _assert_exact_tree(
        private_root,
        expected_files={"capture_state.json"},
        expected_directory="label_commits",
        expected_directory_files={
            f"{index:06d}.json" for index in range(3)
        },
        name="selection preflight private artifact",
    )
    loaded = load_e018_p1_stage2a_selection_config(selection_config_path)
    data_config = _stage2a.load_e018_p1_g2c_data_config(data_config_path)
    stats_identity = {
        "version": "e018-p1-stage2a-preflight-stats-identity/v1",
        "data_config_canonical_sha256": canonical_sha256(data_config),
        "proprio_stats_raw_sha256": data_config["data_identity"][
            "proprio_stats_sha256"
        ],
        "finger_force_stats_raw_sha256": data_config["data_identity"][
            "finger_force_stats_sha256"
        ],
    }
    stats_identity["stats_identity_sha256"] = canonical_sha256(stats_identity)
    snapshot = _read_json(
        public_root / "config_snapshot.json", "preflight config snapshot"
    )
    source = _read_json(
        public_root / "source_identity.json", "preflight source identity"
    )
    parent = _read_json(
        public_root / "parent_verification.json",
        "preflight parent verification",
    )
    checkpoint = _read_json(
        public_root / "checkpoint_identity.json",
        "preflight checkpoint identity",
    )
    if (
        snapshot.get("version")
        != E018_P1_STAGE2A_SELECTION_PREFLIGHT_EXECUTION_VERSION
        or snapshot.get("config_raw_sha256") != loaded.raw_sha256
        or snapshot.get("config_canonical_sha256") != loaded.canonical_sha256
        or snapshot.get("config") != loaded.payload
        or _verify_internal_digest(
            snapshot,
            digest_key="snapshot_sha256",
            name="preflight config snapshot",
        )
        != snapshot.get("snapshot_sha256")
        or source.get("git_commit") != expected_source_git_commit
        or source.get("identity_sha256") != expected_source_identity_sha256
    ):
        raise RuntimeError("preflight config/source identity 漂移")
    receipt = _read_json(
        public_root / "preflight_receipt.json", "preflight receipt"
    )
    completion = _read_json(
        public_root / "PREFLIGHT_COMPLETE.json", "preflight completion"
    )
    expected_counts = {
        "route_count": 1,
        "camera_frame_count": 92,
        "provider_forward_count": 4,
        "private_label_capture_count": 3,
        "memory_write_count": 0,
        "fresh_shadow_action_generation_count": 0,
        "branch_provider_forward_count": 0,
        "fresh_test_read_count": 0,
        "arm_motion_command_count": 0,
        "tcp_motion_command_count": 0,
        "gripper_command_count": 0,
    }
    transaction_primitive = {
        "version": E018_P1_STAGE2A_SELECTION_PREFLIGHT_EXECUTION_VERSION,
        "experiment_id": E018_P1_STAGE2A_SELECTION_PREFLIGHT_EXPERIMENT_ID,
        "seed": STAGE2A_SELECTION_PREFLIGHT_SEED,
        "config_raw_sha256": loaded.raw_sha256,
        "config_canonical_sha256": loaded.canonical_sha256,
        "source_identity_sha256": source["identity_sha256"],
        "parent_verification_sha256": parent["verification_sha256"],
        "checkpoint_raw_sha256": checkpoint["checkpoint_raw_sha256"],
        "proprio_stats_raw_sha256": stats_identity[
            "proprio_stats_raw_sha256"
        ],
        "finger_force_stats_raw_sha256": stats_identity[
            "finger_force_stats_raw_sha256"
        ],
        "stats_identity_sha256": stats_identity["stats_identity_sha256"],
    }
    if (
        _verify_internal_digest(
            receipt,
            digest_key="receipt_sha256",
            name="preflight receipt",
        )
        != receipt.get("receipt_sha256")
        or receipt.get("version")
        != E018_P1_STAGE2A_SELECTION_PREFLIGHT_EXECUTION_VERSION
        or receipt.get("status")
        != "PREFLIGHT_ROUTE_COMPLETE_CONTEXT_DESTROYED"
        or receipt.get("experiment_id")
        != E018_P1_STAGE2A_SELECTION_PREFLIGHT_EXPERIMENT_ID
        or receipt.get("seed") != STAGE2A_SELECTION_PREFLIGHT_SEED
        or receipt.get("stats_identity") != stats_identity
        or receipt.get("transaction_identity_sha256")
        != canonical_sha256(transaction_primitive)
        or receipt.get("counts") != expected_counts
        or receipt.get("context_destroyed") is not True
        or receipt.get("provider_context_destroyed") is not True
        or receipt.get("environment_closed") is not True
        or receipt.get("formal_selection_identity_consumed") is not False
        or receipt.get("formal_completion_created") is not False
        or receipt.get("scoring_consumption_marker_created") is not False
        or receipt.get("result_root_created") is not False
        or receipt.get("fresh_test_status") != "prohibited-unread"
    ):
        raise RuntimeError("preflight receipt identity/accounting 漂移")
    if (
        _verify_internal_digest(
            completion,
            digest_key="completion_sha256",
            name="preflight completion",
        )
        != completion.get("completion_sha256")
        or completion.get("version")
        != E018_P1_STAGE2A_SELECTION_PREFLIGHT_EXECUTION_VERSION
        or completion.get("status") != "PREFLIGHT_COMPLETE"
        or completion.get("experiment_id")
        != E018_P1_STAGE2A_SELECTION_PREFLIGHT_EXPERIMENT_ID
        or completion.get("seed") != STAGE2A_SELECTION_PREFLIGHT_SEED
        or completion.get("counts") != expected_counts
        or completion.get("context_destroyed") is not True
        or completion.get("formal_selection_identity_consumed") is not False
        or completion.get("preflight_receipt_internal_sha256")
        != receipt["receipt_sha256"]
        or completion.get("stats_identity_sha256")
        != stats_identity["stats_identity_sha256"]
    ):
        raise RuntimeError("preflight completion identity/accounting 漂移")
    expected_inventory_paths = (
        _PREFLIGHT_PUBLIC_FILES
        - {"PREFLIGHT_COMPLETE.json"}
        | {
            f"prediction_commits/{index:06d}.commit.json"
            for index in range(4)
        }
    )
    inventory = completion.get("public_artifact_inventory")
    if not isinstance(inventory, dict) or set(inventory) != expected_inventory_paths:
        raise RuntimeError("preflight completion inventory key set 漂移")
    for relative, identity in inventory.items():
        path = public_root / relative
        if (
            not isinstance(identity, dict)
            or set(identity) != {"raw_sha256", "size_bytes"}
            or file_sha256(path) != identity["raw_sha256"]
            or path.stat().st_size != identity["size_bytes"]
        ):
            raise RuntimeError(f"preflight frozen public artifact 漂移: {relative}")
    camera_rows = _read_jsonl(
        public_root / "camera_pose_ledger.jsonl", "preflight camera ledger"
    )
    provider_rows = _read_jsonl(
        public_root / "provider_output_ledger.jsonl", "preflight provider ledger"
    )
    private_inventory = _read_json(
        public_root / "private_label_inventory.json",
        "preflight private inventory",
    )
    if (
        len(camera_rows) != 92
        or len(provider_rows) != 4
        or private_inventory.get("version")
        != E018_P1_STAGE2A_SELECTION_PREFLIGHT_EXECUTION_VERSION
        or private_inventory.get("experiment_id")
        != E018_P1_STAGE2A_SELECTION_PREFLIGHT_EXPERIMENT_ID
        or private_inventory.get("seed") != STAGE2A_SELECTION_PREFLIGHT_SEED
        or private_inventory.get("label_count") != 3
        or not isinstance(private_inventory.get("rows"), list)
        or len(private_inventory["rows"]) != 3
        or _verify_internal_digest(
            private_inventory,
            digest_key="inventory_sha256",
            name="preflight private inventory",
        )
        != private_inventory.get("inventory_sha256")
        or completion.get("private_inventory_internal_sha256")
        != private_inventory["inventory_sha256"]
    ):
        raise RuntimeError("preflight ledger/private inventory accounting 漂移")
    for index, row in enumerate(private_inventory["rows"]):
        path = private_root / f"label_commits/{index:06d}.json"
        if (
            row.get("label_index") != index
            or row.get("prediction_row_index") != index + 1
            or row.get("seed") != STAGE2A_SELECTION_PREFLIGHT_SEED
            or row.get("route_frame_index") != STAGE2A_COLLECT_FRAME_INDICES[index]
            or row.get("path") != f"label_commits/{index:06d}.json"
            or row.get("raw_sha256") != file_sha256(path)
            or row.get("size_bytes") != path.stat().st_size
        ):
            raise RuntimeError(f"preflight private inventory[{index}] 漂移")
    transaction = _read_json(
        public_root / "transaction.json", "preflight transaction"
    )
    if (
        transaction.get("classification")
        != "engineering-preflight-selection-capture-only-no-test-no-actuation/v1"
        or transaction.get("seed") != STAGE2A_SELECTION_PREFLIGHT_SEED
        or transaction.get("memory_write_count") != 0
        or transaction.get("shadow_action_generation") is not None
        or transaction.get("fresh_test_reads") != 0
        or transaction.get("arm_motion_command_count") != 0
        or transaction.get("gripper_close_command_count") != 0
    ):
        raise RuntimeError("preflight transaction 权限/accounting 漂移")
    result = {
        "version": E018_P1_STAGE2A_SELECTION_PREFLIGHT_EXECUTION_VERSION,
        "verified": True,
        "experiment_id": E018_P1_STAGE2A_SELECTION_PREFLIGHT_EXPERIMENT_ID,
        "seed": STAGE2A_SELECTION_PREFLIGHT_SEED,
        "config_raw_sha256": loaded.raw_sha256,
        "config_canonical_sha256": loaded.canonical_sha256,
        "source_git_commit": source["git_commit"],
        "source_identity_sha256": source["identity_sha256"],
        "counts": expected_counts,
        "context_destroyed": True,
        "formal_selection_identity_consumed": False,
        "stats_identity_sha256": stats_identity["stats_identity_sha256"],
        "completion_sha256": completion["completion_sha256"],
    }
    result["verification_sha256"] = canonical_sha256(result)
    return result


def _record_capture_failure(
    *,
    artifact_root: Path,
    error: Exception,
    progress: Stage2AExecutionProgress,
    transaction_identity_sha256: str,
) -> None:
    path = artifact_root / "FAILURE.json"
    if path.exists():
        return
    trace = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    value = {
        "version": E018_P1_STAGE2A_SELECTION_EXECUTION_VERSION,
        "status": "FAILED_EVIDENCE_PRESERVED_NO_RERUN_SAME_IDENTITY",
        "experiment_id": E018_P1_STAGE2A_SELECTION_EXPERIMENT_ID,
        "transaction_identity_sha256": transaction_identity_sha256,
        "error_type": type(error).__name__,
        "error": str(error)[:1024],
        "traceback_tail": trace[-8192:],
        "traceback_sha256": hashlib.sha256(trace.encode("utf-8")).hexdigest(),
        "progress": progress.as_dict(),
        "fresh_test_reads": 0,
        "rerun_under_same_identity_allowed": False,
        "failed_at_unix_ns": time.time_ns(),
    }
    value["failure_sha256"] = canonical_sha256(value)
    _atomic_create_json(path, value)


def run_e018_p1_stage2a_selection_capture(
    *,
    selection_config_path: str | Path,
    stage2a_config_path: str | Path,
    qualification_config_path: str | Path,
    g0c_config_path: str | Path,
    data_config_path: str | Path,
    stats_root: str | Path,
    selected_checkpoint_path: str | Path,
    repository_root: str | Path,
    artifact_root: str | Path,
    stage2a_artifact_root: str | Path,
    stage2a_control_evidence_root: str | Path,
    artifact_inventory_path: str | Path,
    parent_replay_artifact_root: str | Path,
    parent_replay_control_evidence_root: str | Path,
    expected_config_raw_sha256: str,
    expected_config_canonical_sha256: str,
    expected_source_git_commit: str,
    expected_source_identity_sha256: str,
    exact_go_token: str,
) -> dict[str, Any]:
    """Pass A：25 条 physical route + context destroy 后 75 条 logic branch。"""

    artifact = Path(artifact_root)
    if artifact.exists():
        raise FileExistsError(f"selection artifact root 已存在: {artifact}")
    loaded, source = _assert_capture_authority(
        selection_config_path=selection_config_path,
        repository_root=Path(repository_root),
        expected_config_raw_sha256=expected_config_raw_sha256,
        expected_config_canonical_sha256=expected_config_canonical_sha256,
        expected_source_git_commit=expected_source_git_commit,
        expected_source_identity_sha256=expected_source_identity_sha256,
        exact_go_token=exact_go_token,
    )
    stage2a = load_e018_p1_stage2a_config(stage2a_config_path)
    parent = verify_selection_parent_gate(
        selection_config_path=selection_config_path,
        stage2a_config_path=stage2a_config_path,
        qualification_config_path=qualification_config_path,
        stage2a_artifact_root=stage2a_artifact_root,
        stage2a_control_evidence_root=stage2a_control_evidence_root,
        artifact_inventory_path=artifact_inventory_path,
        parent_replay_artifact_root=parent_replay_artifact_root,
        parent_replay_control_evidence_root=(
            parent_replay_control_evidence_root
        ),
    )
    qualification = load_g2c_dynamic_qualification_config(
        qualification_config_path
    )
    g0c = _stage2a._g0c.load_e018_p1_g0c_config(g0c_config_path)
    data = _stage2a.load_e018_p1_g2c_data_config(
        data_config_path,
        parent_g0c_config_path=g0c_config_path,
    )
    producer_process = _new_process_identity("pass-a-producer")
    artifact.mkdir(mode=0o700, parents=True, exist_ok=False)
    public_root = artifact / "public_execution"
    private_root = artifact / "private_labels"
    public_identity = _artifact_role_identity_sha256(
        role="public_execution",
        config_canonical_sha256=loaded.canonical_sha256,
        source_identity_sha256=source["identity_sha256"],
        parent_verification_sha256=parent["verification_sha256"],
    )
    private_identity = _artifact_role_identity_sha256(
        role="private_labels",
        config_canonical_sha256=loaded.canonical_sha256,
        source_identity_sha256=source["identity_sha256"],
        parent_verification_sha256=parent["verification_sha256"],
    )
    transaction_primitive = {
        "version": E018_P1_STAGE2A_SELECTION_EXECUTION_VERSION,
        "experiment_id": E018_P1_STAGE2A_SELECTION_EXPERIMENT_ID,
        "config_raw_sha256": loaded.raw_sha256,
        "config_canonical_sha256": loaded.canonical_sha256,
        "source_identity_sha256": source["identity_sha256"],
        "parent_verification_sha256": parent["verification_sha256"],
        "public_artifact_role_identity_sha256": public_identity,
        "private_artifact_role_identity_sha256": private_identity,
        "seeds": [77001, 77025],
        "gain_order": list(STAGE2A_SELECTION_GAINS),
    }
    transaction_identity = canonical_sha256(transaction_primitive)
    journal = Stage2ASelectionJournal(
        public_root=public_root,
        private_root=private_root,
        config_canonical_sha256=loaded.canonical_sha256,
        transaction_identity_sha256=transaction_identity,
    )
    snapshot = {
        "version": E018_P1_STAGE2A_SELECTION_EXECUTION_VERSION,
        "config_raw_sha256": loaded.raw_sha256,
        "config_canonical_sha256": loaded.canonical_sha256,
        "config": loaded.payload,
    }
    snapshot["snapshot_sha256"] = canonical_sha256(snapshot)
    _atomic_create_json(public_root / "config_snapshot.json", snapshot)
    _atomic_create_json(public_root / "source_identity.json", source)
    _atomic_create_json(public_root / "parent_verification.json", parent)
    started_record = {
        "version": E018_P1_STAGE2A_SELECTION_EXECUTION_VERSION,
        "status": "PASS_A_IN_PROGRESS_NO_TEST_NO_ACTUATION",
        "experiment_id": E018_P1_STAGE2A_SELECTION_EXPERIMENT_ID,
        "classification": "formal-development-selection-no-test-no-actuation/v1",
        "effect_claim": "no-effect-claim",
        "seed_range": [77001, 77025],
        "gain_order": list(STAGE2A_SELECTION_GAINS),
        "transaction_identity_sha256": transaction_identity,
        "public_artifact_role_identity_sha256": public_identity,
        "private_artifact_role_identity_sha256": private_identity,
        "result_root_created": False,
        "gpu_wall_seconds_max": loaded.payload["budgets"]["gpu_wall_seconds_max"],
        "stop_conditions": [
            "hard-protocol-or-safety-exception",
            "gpu-wall-budget-exceeded",
            "artifact-budget-exceeded",
        ],
        "fresh_test_reads": 0,
        "started_at_unix_ns": time.time_ns(),
    }
    started_record["run_started_sha256"] = canonical_sha256(started_record)
    _atomic_create_json(public_root / "RUN_STARTED.json", started_record)
    progress = Stage2AExecutionProgress()
    started_monotonic = time.monotonic()
    try:
        (
            camera_rows,
            captured_routes,
            route_summaries,
            transactions,
            provider_records,
            route_receipts,
            route_private,
            environment_identity,
            context_destroyed,
        ) = _run_selection_simulator(
            loaded_selection_config=loaded,
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
        )
        if not context_destroyed:
            raise RuntimeError("selection context 未销毁，禁止 logic replay")
        provider_freeze, private_inventory = journal.freeze()
        branches: list[GainBranchOutcome] = []
        for captured in captured_routes:
            branches.extend(replay_all_gain_branches(captured))
        if (
            len(branches) != STAGE2A_SELECTION_BRANCH_COUNT
            or any(branch.provider_forward_count != 0 for branch in branches)
        ):
            raise RuntimeError("selection pure-logic branch accounting 漂移")
        route_rows = [
            build_route_evidence_row(
                seed=captured.seed,
                camera_row_start=index * 92,
                camera_rows=camera_rows[index * 92 : (index + 1) * 92],
                provider_row_indices=range(index * 4, index * 4 + 4),
                provider_records=provider_records[index * 4 : index * 4 + 4],
                prediction_receipts=route_receipts[index],
                private_inventory_rows=route_private[index],
                capture_transaction=transactions[index],
                route_summary=route_summaries[index],
                captured_route=captured,
            )
            for index, captured in enumerate(captured_routes)
        ]
        _atomic_jsonl(public_root / "camera_pose_ledger.jsonl", camera_rows)
        _atomic_jsonl(public_root / "route_evidence_ledger.jsonl", route_rows)
        _atomic_jsonl(
            public_root / "gain_branch_ledger.jsonl",
            [value.to_dict() for value in branches],
        )
        _atomic_create_json(
            public_root / "private_label_inventory.json", private_inventory
        )
        prefreeze_paths = sorted(
            [
                "RUN_STARTED.json",
                "config_snapshot.json",
                "source_identity.json",
                "parent_verification.json",
                "camera_pose_ledger.jsonl",
                "provider_output_ledger.jsonl",
                "route_evidence_ledger.jsonl",
                "gain_branch_ledger.jsonl",
                "private_label_inventory.json",
            ]
            + [
                f"prediction_commits/{index:06d}.commit.json"
                for index in range(STAGE2A_SELECTION_PREDICTION_COUNT)
            ]
        )
        freeze = {
            "version": E018_P1_STAGE2A_SELECTION_EXECUTION_VERSION,
            "status": "PASS_A_FROZEN_AFTER_CONTEXT_DESTROY",
            "context_destroyed": True,
            "provider_context_destroyed": True,
            "environment_closed": True,
            "prediction_ledger_frozen": True,
            "route_evidence_frozen": True,
            "gain_branches_frozen": True,
            "private_labels_write_only": True,
            "private_label_open_count": 0,
            "provider_forward_count": provider_freeze["row_count"],
            "branch_provider_forward_count": 0,
            "producer_process_identity": producer_process,
            "artifact_inventory": _file_inventory(public_root, prefreeze_paths),
            "frozen_at_unix_ns": time.time_ns(),
        }
        freeze["freeze_sha256"] = canonical_sha256(freeze)
        _atomic_create_json(public_root / "execution_freeze.json", freeze)
        wall_seconds = time.monotonic() - started_monotonic
        counts = {
            "seed_count": len(captured_routes),
            "route_count": len(route_rows),
            "camera_frame_count": len(camera_rows),
            "provider_forward_count": len(provider_records),
            "private_label_capture_count": journal.label_count,
            "private_label_open_count": 0,
            "gain_branch_count": len(branches),
            "branch_provider_forward_count": sum(
                value.provider_forward_count for value in branches
            ),
            "arm_motion_command_count": sum(
                value.arm_motion_command_count for value in branches
            ),
            "gripper_close_command_count": sum(
                value.gripper_close_command_count for value in branches
            ),
            "runtime_object_gt_read_count": 0,
            "goal_gt_read_count": 0,
            "fresh_test_read_count": 0,
            "checkpoint_write_count": 0,
        }
        receipt = {
            "version": E018_P1_STAGE2A_SELECTION_EXECUTION_VERSION,
            "status": "PASS_A_COMPLETE_CONTEXT_DESTROYED",
            "classification": "formal-development-selection-no-test-no-actuation/v1",
            "effect_claim": "no-effect-claim",
            "experiment_id": E018_P1_STAGE2A_SELECTION_EXPERIMENT_ID,
            "config_raw_sha256": loaded.raw_sha256,
            "config_canonical_sha256": loaded.canonical_sha256,
            "source_identity_sha256": source["identity_sha256"],
            "parent_verification_sha256": parent["verification_sha256"],
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
            "producer_process_identity": producer_process,
            "execution_freeze_raw_sha256": file_sha256(
                public_root / "execution_freeze.json"
            ),
            "execution_freeze_internal_sha256": freeze["freeze_sha256"],
            "formal_claim_allowed": False,
            "fresh_test_status": "prohibited-unread",
        }
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        _atomic_create_json(public_root / "execution_receipt.json", receipt)
        bytes_before_marker = sum(
            path.stat().st_size
            for path in artifact.rglob("*")
            if path.is_file()
        )
        artifact_budget = loaded.payload["budgets"]["combined_artifact_bytes_max"]
        if bytes_before_marker + 1_048_576 > artifact_budget:
            raise RuntimeError("selection artifact budget 缺少 completion marker 余量")
        precompletion = _verify_e018_p1_stage2a_selection_public(
            selection_config_path=selection_config_path,
            stage2a_config_path=stage2a_config_path,
            qualification_config_path=qualification_config_path,
            public_root=public_root,
            expected_source_git_commit=expected_source_git_commit,
            expected_source_identity_sha256=expected_source_identity_sha256,
            require_complete=False,
        )
        marker = {
            "version": E018_P1_STAGE2A_SELECTION_EXECUTION_VERSION,
            "status": "PUBLIC_EXECUTION_COMPLETE",
            "experiment_id": E018_P1_STAGE2A_SELECTION_EXPERIMENT_ID,
            "transaction_identity_sha256": transaction_identity,
            "precompletion_verification_sha256": precompletion[
                "verification_sha256"
            ],
            "execution_receipt_raw_sha256": file_sha256(
                public_root / "execution_receipt.json"
            ),
            "execution_receipt_internal_sha256": receipt["receipt_sha256"],
            "private_label_open_count": 0,
            "fresh_test_reads": 0,
            "completed_at_unix_ns": time.time_ns(),
        }
        marker["marker_sha256"] = canonical_sha256(marker)
        _atomic_create_json(public_root / "PUBLIC_EXECUTION_COMPLETE.json", marker)
        final = verify_e018_p1_stage2a_selection_public(
            selection_config_path=selection_config_path,
            stage2a_config_path=stage2a_config_path,
            qualification_config_path=qualification_config_path,
            public_root=public_root,
            expected_source_git_commit=expected_source_git_commit,
            expected_source_identity_sha256=expected_source_identity_sha256,
        )
        if sum(
            path.stat().st_size
            for path in artifact.rglob("*")
            if path.is_file()
        ) > artifact_budget:
            raise RuntimeError("selection combined artifact budget 超限")
        return final
    except Exception as error:
        _record_capture_failure(
            artifact_root=artifact,
            error=error,
            progress=progress,
            transaction_identity_sha256=transaction_identity,
        )
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _durable_create_json_exclusive(
    path: Path,
    value: Mapping[str, Any],
) -> tuple[str, int]:
    """以 O_EXCL 创建一次性 marker，并 fsync 文件与父目录。"""

    raw = _json_bytes(value)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("selection consumption marker 写入失败")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)
    return hashlib.sha256(raw).hexdigest(), time.time_ns()


def _signed_consumption_marker(value: Mapping[str, Any]) -> dict[str, Any]:
    marker = dict(value)
    marker.pop("marker_sha256", None)
    marker["marker_sha256"] = canonical_sha256(marker)
    return marker


def _replace_consumption_marker(
    path: Path,
    marker: Mapping[str, Any],
    **changes: Any,
) -> dict[str, Any]:
    updated = {
        **dict(marker),
        **changes,
        "update_sequence": int(marker["update_sequence"]) + 1,
        "last_updated_at_unix_ns": time.time_ns(),
    }
    updated = _signed_consumption_marker(updated)
    _atomic_replace_json(path, updated)
    return updated


def _verify_capture_state_before_scoring(
    private_root: Path,
    *,
    transaction_identity_sha256: str,
) -> dict[str, Any]:
    state = _read_json(
        private_root / "capture_state.json",
        "selection private capture state",
    )
    _require_exact_keys(
        state,
        {
            "version",
            "status",
            "transaction_identity_sha256",
            "prediction_commit_count",
            "privileged_access_started_count",
            "privileged_capture_count",
            "rerun_under_same_identity_allowed",
            "state_sha256",
        },
        "selection private capture state",
    )
    if (
        _verify_internal_digest(
            state,
            digest_key="state_sha256",
            name="selection private capture state",
        )
        != state["state_sha256"]
        or state["version"] != E018_P1_STAGE2A_SELECTION_EXECUTION_VERSION
        or state["status"] != "capture-complete-write-only-not-opened"
        or state["transaction_identity_sha256"]
        != transaction_identity_sha256
        or type(state["prediction_commit_count"]) is not int
        or state["prediction_commit_count"] != 100
        or type(state["privileged_access_started_count"]) is not int
        or state["privileged_access_started_count"] != 75
        or type(state["privileged_capture_count"]) is not int
        or state["privileged_capture_count"] != 75
        or state["rerun_under_same_identity_allowed"] is not False
    ):
        raise RuntimeError("selection private capture state 漂移")
    return state


def _assert_private_tree_before_consumption(private_root: Path) -> None:
    marker_path = private_root / "SCORING_CONSUMED.json"
    if marker_path.exists() or marker_path.is_symlink():
        raise RuntimeError(
            "selection Pass B identity 已消费，禁止在同一 identity 下重跑"
        )
    _assert_exact_tree(
        private_root,
        expected_files={"capture_state.json"},
        expected_directory="label_commits",
        expected_directory_files={
            f"{index:06d}.json"
            for index in range(STAGE2A_SELECTION_LABEL_COUNT)
        },
        name="selection private artifact before consumption",
    )


def _read_private_label_exactly_once(path: Path) -> tuple[dict[str, Any], str, int]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat_module.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise RuntimeError("selection private label 必须是 single-link regular file")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            digest.update(chunk)
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("selection private label JSON 非法") from error
    if not isinstance(value, dict) or raw != _json_bytes(value):
        raise RuntimeError("selection private label serialization 漂移")
    return value, digest.hexdigest(), len(raw)


def _scoring_primitive_sha256(label: Mapping[str, Any]) -> str:
    return canonical_sha256(
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


def _verify_final_consumption_marker(
    path: Path,
    *,
    loaded: Any,
    public: Mapping[str, Any],
    producer: Mapping[str, Any],
    scorer: Mapping[str, Any],
    result_role_identity: str,
) -> dict[str, Any]:
    marker = _read_json(path, "selection final consumption marker")
    _require_exact_keys(
        marker,
        {
            "version",
            "status",
            "experiment_id",
            "classification",
            "effect_claim",
            "config_raw_sha256",
            "config_canonical_sha256",
            "source_identity_sha256",
            "parent_verification_sha256",
            "transaction_identity_sha256",
            "public_verification_sha256",
            "public_completion_marker_sha256",
            "private_artifact_role_identity_sha256",
            "result_artifact_role_identity_sha256",
            "producer_process_identity",
            "scorer_process_identity",
            "process_boundary_verified",
            "rerun_under_same_identity_allowed",
            "label_open_started_count",
            "label_open_completed_count",
            "provider_forward_count",
            "checkpoint_load_count",
            "result_root_created",
            "update_sequence",
            "failure",
            "started_at_unix_ns",
            "last_updated_at_unix_ns",
            "completed_at_unix_ns",
            "marker_sha256",
        },
        "selection final consumption marker",
    )
    if (
        _verify_internal_digest(
            marker,
            digest_key="marker_sha256",
            name="selection final consumption marker",
        )
        != marker["marker_sha256"]
        or marker["version"] != E018_P1_STAGE2A_SELECTION_RESULT_VERSION
        or marker["status"]
        != "PASS_B_PRIVATE_LABELS_CONSUMED_EXACT_ONCE"
        or marker["experiment_id"]
        != E018_P1_STAGE2A_SELECTION_EXPERIMENT_ID
        or marker["classification"] != _SELECTION_CLASSIFICATION
        or marker["effect_claim"] != "no-effect-claim"
        or marker["config_raw_sha256"] != loaded.raw_sha256
        or marker["config_canonical_sha256"] != loaded.canonical_sha256
        or marker["source_identity_sha256"]
        != public["source_identity_sha256"]
        or marker["parent_verification_sha256"]
        != public["parent_verification_sha256"]
        or marker["transaction_identity_sha256"]
        != public["transaction_identity_sha256"]
        or marker["public_verification_sha256"]
        != public["verification_sha256"]
        or marker["public_completion_marker_sha256"]
        != public["public_completion_marker_sha256"]
        or marker["private_artifact_role_identity_sha256"]
        != public["private_artifact_role_identity_sha256"]
        or marker["result_artifact_role_identity_sha256"]
        != result_role_identity
        or marker["producer_process_identity"] != producer
        or marker["scorer_process_identity"] != scorer
        or marker["process_boundary_verified"] is not True
        or producer["process_instance_sha256"]
        == scorer["process_instance_sha256"]
        or marker["rerun_under_same_identity_allowed"] is not False
        or type(marker["label_open_started_count"]) is not int
        or marker["label_open_started_count"] != 75
        or type(marker["label_open_completed_count"]) is not int
        or marker["label_open_completed_count"] != 75
        or marker["provider_forward_count"] != 0
        or marker["checkpoint_load_count"] != 0
        or marker["result_root_created"] is not True
        or type(marker["update_sequence"]) is not int
        or marker["update_sequence"] != 152
        or marker["failure"] is not None
        or type(marker["started_at_unix_ns"]) is not int
        or type(marker["last_updated_at_unix_ns"]) is not int
        or type(marker["completed_at_unix_ns"]) is not int
        or not (
            0
            < marker["started_at_unix_ns"]
            <= marker["completed_at_unix_ns"]
            <= marker["last_updated_at_unix_ns"]
        )
    ):
        raise RuntimeError("selection final consumption marker 漂移")
    return marker


def _validate_scored_selection_row(
    value: Mapping[str, Any],
    *,
    row_index: int,
) -> dict[str, Any]:
    row = _require_exact_keys(
        value,
        {
            "version",
            "seed",
            "gain",
            "oracle_common_denominator_eligible",
            "memory_commit_count",
            "navigation_state_available",
            "fresh_shadow_action_generation_count",
            "xyz_error_m",
            "recovered",
            "false_recovery",
            "catastrophic_recovery",
            "unsafe_recovery",
            "protocol_violation_count",
            "scored_row_sha256",
        },
        f"selection scored row[{row_index}]",
    )
    unsigned = dict(row)
    stored = unsigned.pop("scored_row_sha256")
    expected_seed = STAGE2A_SELECTION_SEEDS[row_index // 3]
    expected_gain = STAGE2A_SELECTION_GAINS[row_index % 3]
    error = row["xyz_error_m"]
    if error is not None:
        error = _finite_real(error, f"selection scored row[{row_index}].xyz_error_m")
        if error < 0.0:
            raise RuntimeError("selection scored XYZ error 不得为负")
    for name in (
        "oracle_common_denominator_eligible",
        "navigation_state_available",
        "recovered",
        "false_recovery",
        "catastrophic_recovery",
        "unsafe_recovery",
    ):
        if type(row[name]) is not bool:
            raise RuntimeError(f"selection scored row {name} 必须是 bool")
    committed = row["memory_commit_count"] == 1
    expected_recovered = bool(
        row["oracle_common_denominator_eligible"]
        and committed
        and row["navigation_state_available"]
        and row["fresh_shadow_action_generation_count"] == 1
        and error is not None
        and error <= 0.005
    )
    expected_false = bool(
        committed
        and (
            not row["oracle_common_denominator_eligible"]
            or error is None
            or error > 0.005
        )
    )
    expected_catastrophic = bool(
        committed and error is not None and error > 0.020
    )
    if (
        stored != canonical_sha256(unsigned)
        or row["version"] != E018_P1_STAGE2A_SELECTION_RESULT_VERSION
        or row["seed"] != expected_seed
        or row["gain"] != expected_gain
        or type(row["memory_commit_count"]) is not int
        or row["memory_commit_count"] not in {0, 1}
        or type(row["fresh_shadow_action_generation_count"]) is not int
        or row["fresh_shadow_action_generation_count"]
        != row["memory_commit_count"]
        or type(row["protocol_violation_count"]) is not int
        or row["protocol_violation_count"] < 0
        or (not committed and error is not None)
        or (
            committed
            and (
                error is None
                or row["navigation_state_available"] is not True
            )
        )
        or (
            row["oracle_common_denominator_eligible"]
            and row["protocol_violation_count"] != 0
        )
        or row["recovered"] is not expected_recovered
        or row["false_recovery"] is not expected_false
        or row["catastrophic_recovery"] is not expected_catastrophic
        or row["unsafe_recovery"] is not expected_false
    ):
        raise RuntimeError(f"selection scored row[{row_index}] mechanics 漂移")
    return row


def _recompute_selection_summary_from_scored(
    rows: Sequence[Mapping[str, Any]],
    *,
    minimum_support: int,
) -> dict[str, Any]:
    denominator_seeds = {
        row["seed"]
        for row in rows
        if row["oracle_common_denominator_eligible"]
    }
    denominator = len(denominator_seeds)
    per_gain: list[dict[str, Any]] = []
    for gain in STAGE2A_SELECTION_GAINS:
        gain_rows = [row for row in rows if row["gain"] == gain]
        value = {
            "gain": gain,
            "common_denominator_count": denominator,
            "recovered_count": sum(row["recovered"] for row in gain_rows),
            "false_recovery_count": sum(
                row["false_recovery"] for row in gain_rows
            ),
            "catastrophic_recovery_count": sum(
                row["catastrophic_recovery"] for row in gain_rows
            ),
            "unsafe_recovery_count": sum(
                row["unsafe_recovery"] for row in gain_rows
            ),
            "protocol_violation_count": sum(
                row["protocol_violation_count"] for row in gain_rows
            ),
        }
        value["eligible"] = bool(
            value["false_recovery_count"] == 0
            and value["catastrophic_recovery_count"] == 0
            and value["unsafe_recovery_count"] == 0
            and value["protocol_violation_count"] == 0
        )
        per_gain.append(value)
    eligible = [value for value in per_gain if value["eligible"]]
    if denominator < minimum_support:
        selected_gain = None
        reason = "insufficient-common-denominator-support"
    elif not eligible:
        selected_gain = None
        reason = "no-safe-eligible-gain"
    else:
        selected = max(
            eligible,
            key=lambda value: (value["recovered_count"], value["gain"]),
        )
        selected_gain = float(selected["gain"])
        reason = "max-recovered-count-then-larger-gain"
    summary = {
        "version": E018_P1_STAGE2A_SELECTION_RESULT_VERSION,
        "status": "complete-development-selection",
        "classification": _SELECTION_CLASSIFICATION,
        "effect_claim": "no-effect-claim",
        "common_denominator_count": denominator,
        "minimum_support_required": minimum_support,
        "per_gain": per_gain,
        "selected_gain": selected_gain,
        "selection_reason": reason,
        "evaluation_config_generation_allowed": selected_gain is not None,
        "stage2b_continuation_required": True,
        "fresh_test_reads": 0,
        "runtime_object_gt_reads": 0,
        "goal_gt_reads": 0,
        "arm_motion_command_count": 0,
        "gripper_close_command_count": 0,
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    return summary


def _verify_e018_p1_stage2a_selection_result(
    *,
    selection_config_path: str | Path,
    stage2a_config_path: str | Path,
    qualification_config_path: str | Path,
    public_root: str | Path,
    result_root: str | Path,
    expected_source_git_commit: str,
    expected_source_identity_sha256: str,
    require_complete: bool,
) -> dict[str, Any]:
    """结果验证器不接受也不打开 private/model/checkpoint/stats。"""

    public = verify_e018_p1_stage2a_selection_public(
        selection_config_path=selection_config_path,
        stage2a_config_path=stage2a_config_path,
        qualification_config_path=qualification_config_path,
        public_root=public_root,
        expected_source_git_commit=expected_source_git_commit,
        expected_source_identity_sha256=expected_source_identity_sha256,
    )
    loaded = load_e018_p1_stage2a_selection_config(selection_config_path)
    public_path = Path(public_root)
    root = Path(result_root)
    _require_common_artifact_parent(
        public_root=public_path,
        result_root=root,
    )
    _assert_exact_tree(
        root,
        expected_files=(
            _RESULT_FILES if require_complete else _RESULT_PRECOMPLETION_FILES
        ),
        name="selection result artifact",
    )
    scored_path = root / "scored_gain_branches.jsonl"
    raw_lines = scored_path.read_bytes().splitlines(keepends=True)
    scored_values = _read_jsonl(scored_path, "selection scored branches")
    if (
        len(raw_lines) != STAGE2A_SELECTION_BRANCH_COUNT
        or len(scored_values) != STAGE2A_SELECTION_BRANCH_COUNT
        or any(not line.endswith(b"\n") for line in raw_lines)
        or any(
            line != _jsonl_line_bytes(value)
            for line, value in zip(raw_lines, scored_values, strict=True)
        )
    ):
        raise RuntimeError("selection scored ledger 行数/serialization 漂移")
    scored = [
        _validate_scored_selection_row(value, row_index=index)
        for index, value in enumerate(scored_values)
    ]
    expected_summary = _recompute_selection_summary_from_scored(
        scored,
        minimum_support=loaded.payload["gain_selection"][
            "minimum_common_denominator_routes"
        ],
    )
    summary_path = root / "selection_summary.json"
    summary = _read_json(summary_path, "selection summary")
    if summary != expected_summary:
        raise RuntimeError("selection summary 不能从 scored rows 独立重算")
    receipt_path = root / "result_receipt.json"
    receipt = _read_json(receipt_path, "selection result receipt")
    _require_exact_keys(
        receipt,
        {
            "version",
            "status",
            "classification",
            "effect_claim",
            "experiment_id",
            "config_raw_sha256",
            "config_canonical_sha256",
            "source_identity_sha256",
            "parent_verification_sha256",
            "transaction_identity_sha256",
            "public_verification_sha256",
            "public_completion_marker_sha256",
            "result_artifact_role_identity_sha256",
            "producer_process_identity",
            "scorer_process_identity",
            "process_boundary_verified",
            "consumption_marker_raw_sha256",
            "consumption_marker_internal_sha256",
            "rerun_under_same_identity_allowed",
            "scored_ledger_raw_sha256",
            "scored_row_count",
            "selection_summary_raw_sha256",
            "selection_summary_internal_sha256",
            "counts",
            "formal_claim_allowed",
            "fresh_test_status",
            "stage2b_continuation_required",
            "completed_at_unix_ns",
            "receipt_sha256",
        },
        "selection result receipt",
    )
    producer = _verify_process_identity(
        receipt["producer_process_identity"], role="pass-a-producer"
    )
    scorer = _verify_process_identity(
        receipt["scorer_process_identity"], role="pass-b-scorer"
    )
    result_role_identity = _artifact_role_identity_sha256(
        role="result",
        config_canonical_sha256=loaded.canonical_sha256,
        source_identity_sha256=public["source_identity_sha256"],
        parent_verification_sha256=public["parent_verification_sha256"],
    )
    expected_counts = {
        "private_label_open_started_count": 75,
        "private_label_open_completed_count": 75,
        "provider_forward_count": 0,
        "checkpoint_load_count": 0,
        "gain_branch_count": 75,
        "fresh_test_read_count": 0,
        "runtime_object_gt_read_count": 0,
        "goal_gt_read_count": 0,
        "arm_motion_command_count": 0,
        "gripper_close_command_count": 0,
    }
    if (
        _verify_internal_digest(
            receipt,
            digest_key="receipt_sha256",
            name="selection result receipt",
        )
        != receipt["receipt_sha256"]
        or receipt["version"] != E018_P1_STAGE2A_SELECTION_RESULT_VERSION
        or receipt["status"] != "PASS_B_COMPLETE_EXACT_ONCE"
        or receipt["classification"] != _SELECTION_CLASSIFICATION
        or receipt["effect_claim"] != "no-effect-claim"
        or receipt["experiment_id"]
        != E018_P1_STAGE2A_SELECTION_EXPERIMENT_ID
        or receipt["config_raw_sha256"] != loaded.raw_sha256
        or receipt["config_canonical_sha256"] != loaded.canonical_sha256
        or receipt["source_identity_sha256"]
        != public["source_identity_sha256"]
        or receipt["parent_verification_sha256"]
        != public["parent_verification_sha256"]
        or receipt["transaction_identity_sha256"]
        != public["transaction_identity_sha256"]
        or receipt["public_verification_sha256"]
        != public["verification_sha256"]
        or receipt["public_completion_marker_sha256"]
        != public["public_completion_marker_sha256"]
        or receipt["result_artifact_role_identity_sha256"]
        != result_role_identity
        or producer != public["producer_process_identity"]
        or producer["process_instance_sha256"]
        == scorer["process_instance_sha256"]
        or receipt["process_boundary_verified"] is not True
        or not _is_sha256(receipt["consumption_marker_raw_sha256"])
        or not _is_sha256(receipt["consumption_marker_internal_sha256"])
        or receipt["rerun_under_same_identity_allowed"] is not False
        or receipt["scored_ledger_raw_sha256"] != file_sha256(scored_path)
        or receipt["scored_row_count"] != len(scored)
        or receipt["selection_summary_raw_sha256"]
        != file_sha256(summary_path)
        or receipt["selection_summary_internal_sha256"]
        != summary["summary_sha256"]
        or receipt["counts"] != expected_counts
        or receipt["formal_claim_allowed"] is not False
        or receipt["fresh_test_status"] != "prohibited-unread"
        or receipt["stage2b_continuation_required"] is not True
        or type(receipt["completed_at_unix_ns"]) is not int
        or receipt["completed_at_unix_ns"] <= 0
    ):
        raise RuntimeError("selection result receipt identity/accounting 漂移")
    precompletion = {
        "version": E018_P1_STAGE2A_SELECTION_RESULT_VERSION,
        "verified": True,
        "complete_marker_verified": False,
        "experiment_id": E018_P1_STAGE2A_SELECTION_EXPERIMENT_ID,
        "config_raw_sha256": loaded.raw_sha256,
        "config_canonical_sha256": loaded.canonical_sha256,
        "source_git_commit": public["source_git_commit"],
        "source_identity_sha256": public["source_identity_sha256"],
        "parent_verification_sha256": public["parent_verification_sha256"],
        "transaction_identity_sha256": public["transaction_identity_sha256"],
        "public_verification_sha256": public["verification_sha256"],
        "consumption_marker_raw_sha256": receipt[
            "consumption_marker_raw_sha256"
        ],
        "consumption_marker_internal_sha256": receipt[
            "consumption_marker_internal_sha256"
        ],
        "scorer_process_token_sha256": scorer["process_token_sha256"],
        "scored_row_count": len(scored),
        "selection_summary_sha256": summary["summary_sha256"],
        "selected_gain": summary["selected_gain"],
        "stage2b_continuation_required": True,
        "fresh_test_reads": 0,
    }
    precompletion["verification_sha256"] = canonical_sha256(precompletion)
    if not require_complete:
        return precompletion
    marker_path = root / "RESULT_COMPLETE.json"
    marker = _read_json(marker_path, "selection result completion marker")
    _require_exact_keys(
        marker,
        {
            "version",
            "status",
            "experiment_id",
            "transaction_identity_sha256",
            "precompletion_verification_sha256",
            "result_receipt_raw_sha256",
            "result_receipt_internal_sha256",
            "scored_ledger_raw_sha256",
            "selection_summary_raw_sha256",
            "selection_summary_internal_sha256",
            "completed_at_unix_ns",
            "marker_sha256",
        },
        "selection result completion marker",
    )
    if (
        _verify_internal_digest(
            marker,
            digest_key="marker_sha256",
            name="selection result completion marker",
        )
        != marker["marker_sha256"]
        or marker["version"] != E018_P1_STAGE2A_SELECTION_RESULT_VERSION
        or marker["status"] != "RESULT_COMPLETE"
        or marker["experiment_id"]
        != E018_P1_STAGE2A_SELECTION_EXPERIMENT_ID
        or marker["transaction_identity_sha256"]
        != public["transaction_identity_sha256"]
        or marker["precompletion_verification_sha256"]
        != precompletion["verification_sha256"]
        or marker["result_receipt_raw_sha256"] != file_sha256(receipt_path)
        or marker["result_receipt_internal_sha256"]
        != receipt["receipt_sha256"]
        or marker["scored_ledger_raw_sha256"] != file_sha256(scored_path)
        or marker["selection_summary_raw_sha256"]
        != file_sha256(summary_path)
        or marker["selection_summary_internal_sha256"]
        != summary["summary_sha256"]
        or type(marker["completed_at_unix_ns"]) is not int
        or marker["completed_at_unix_ns"] <= 0
    ):
        raise RuntimeError("selection result completion marker 漂移")
    result = {
        **precompletion,
        "complete_marker_verified": True,
        "result_completion_marker_sha256": marker["marker_sha256"],
    }
    result.pop("verification_sha256")
    result["verification_sha256"] = canonical_sha256(result)
    return result


def verify_e018_p1_stage2a_selection_result(
    *,
    selection_config_path: str | Path,
    stage2a_config_path: str | Path,
    qualification_config_path: str | Path,
    public_root: str | Path,
    result_root: str | Path,
    expected_source_git_commit: str,
    expected_source_identity_sha256: str,
) -> dict[str, Any]:
    return _verify_e018_p1_stage2a_selection_result(
        selection_config_path=selection_config_path,
        stage2a_config_path=stage2a_config_path,
        qualification_config_path=qualification_config_path,
        public_root=public_root,
        result_root=result_root,
        expected_source_git_commit=expected_source_git_commit,
        expected_source_identity_sha256=expected_source_identity_sha256,
        require_complete=True,
    )


def _publish_pass_b_selection_result(
    *,
    selection_config_path: str | Path,
    stage2a_config_path: str | Path,
    qualification_config_path: str | Path,
    public_root: str | Path,
    result_path: Path,
    artifact_root: Path,
    marker_path: Path,
    loaded: Any,
    public: Mapping[str, Any],
    producer: Mapping[str, Any],
    scorer: Mapping[str, Any],
    result_role_identity: str,
    scored: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    expected_source_git_commit: str,
    expected_source_identity_sha256: str,
) -> dict[str, Any]:
    """发布 Pass B 结果；异常由 caller 永久冻结 consumption identity。"""

    verified_marker = _verify_final_consumption_marker(
        marker_path,
        loaded=loaded,
        public=public,
        producer=producer,
        scorer=scorer,
        result_role_identity=result_role_identity,
    )
    consumption_raw_sha256 = file_sha256(marker_path)
    _atomic_jsonl(result_path / "scored_gain_branches.jsonl", scored)
    _atomic_create_json(result_path / "selection_summary.json", dict(summary))
    counts = {
        "private_label_open_started_count": 75,
        "private_label_open_completed_count": 75,
        "provider_forward_count": 0,
        "checkpoint_load_count": 0,
        "gain_branch_count": 75,
        "fresh_test_read_count": 0,
        "runtime_object_gt_read_count": 0,
        "goal_gt_read_count": 0,
        "arm_motion_command_count": 0,
        "gripper_close_command_count": 0,
    }
    receipt = {
        "version": E018_P1_STAGE2A_SELECTION_RESULT_VERSION,
        "status": "PASS_B_COMPLETE_EXACT_ONCE",
        "classification": _SELECTION_CLASSIFICATION,
        "effect_claim": "no-effect-claim",
        "experiment_id": E018_P1_STAGE2A_SELECTION_EXPERIMENT_ID,
        "config_raw_sha256": loaded.raw_sha256,
        "config_canonical_sha256": loaded.canonical_sha256,
        "source_identity_sha256": public["source_identity_sha256"],
        "parent_verification_sha256": public["parent_verification_sha256"],
        "transaction_identity_sha256": public["transaction_identity_sha256"],
        "public_verification_sha256": public["verification_sha256"],
        "public_completion_marker_sha256": public[
            "public_completion_marker_sha256"
        ],
        "result_artifact_role_identity_sha256": result_role_identity,
        "producer_process_identity": dict(producer),
        "scorer_process_identity": dict(scorer),
        "process_boundary_verified": True,
        "consumption_marker_raw_sha256": consumption_raw_sha256,
        "consumption_marker_internal_sha256": verified_marker["marker_sha256"],
        "rerun_under_same_identity_allowed": False,
        "scored_ledger_raw_sha256": file_sha256(
            result_path / "scored_gain_branches.jsonl"
        ),
        "scored_row_count": len(scored),
        "selection_summary_raw_sha256": file_sha256(
            result_path / "selection_summary.json"
        ),
        "selection_summary_internal_sha256": summary["summary_sha256"],
        "counts": counts,
        "formal_claim_allowed": False,
        "fresh_test_status": "prohibited-unread",
        "stage2b_continuation_required": True,
        "completed_at_unix_ns": time.time_ns(),
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    _atomic_create_json(result_path / "result_receipt.json", receipt)
    precompletion = _verify_e018_p1_stage2a_selection_result(
        selection_config_path=selection_config_path,
        stage2a_config_path=stage2a_config_path,
        qualification_config_path=qualification_config_path,
        public_root=public_root,
        result_root=result_path,
        expected_source_git_commit=expected_source_git_commit,
        expected_source_identity_sha256=expected_source_identity_sha256,
        require_complete=False,
    )
    completion = {
        "version": E018_P1_STAGE2A_SELECTION_RESULT_VERSION,
        "status": "RESULT_COMPLETE",
        "experiment_id": E018_P1_STAGE2A_SELECTION_EXPERIMENT_ID,
        "transaction_identity_sha256": public["transaction_identity_sha256"],
        "precompletion_verification_sha256": precompletion[
            "verification_sha256"
        ],
        "result_receipt_raw_sha256": file_sha256(
            result_path / "result_receipt.json"
        ),
        "result_receipt_internal_sha256": receipt["receipt_sha256"],
        "scored_ledger_raw_sha256": receipt["scored_ledger_raw_sha256"],
        "selection_summary_raw_sha256": receipt[
            "selection_summary_raw_sha256"
        ],
        "selection_summary_internal_sha256": summary["summary_sha256"],
        "completed_at_unix_ns": time.time_ns(),
    }
    completion["marker_sha256"] = canonical_sha256(completion)
    completion_path = result_path / "RESULT_COMPLETE.json"
    _atomic_create_json(completion_path, completion)
    artifact_budget = loaded.payload["budgets"]["combined_artifact_bytes_max"]
    if _combined_artifact_bytes(artifact_root) > artifact_budget:
        completion_path.unlink()
        _fsync_directory(result_path)
        raise RuntimeError("selection Pass B combined artifact budget 超限")
    return verify_e018_p1_stage2a_selection_result(
        selection_config_path=selection_config_path,
        stage2a_config_path=stage2a_config_path,
        qualification_config_path=qualification_config_path,
        public_root=public_root,
        result_root=result_path,
        expected_source_git_commit=expected_source_git_commit,
        expected_source_identity_sha256=expected_source_identity_sha256,
    )


def run_e018_p1_stage2a_selection_score_private(
    *,
    selection_config_path: str | Path,
    stage2a_config_path: str | Path,
    qualification_config_path: str | Path,
    public_root: str | Path,
    private_root: str | Path,
    result_root: str | Path,
    expected_source_git_commit: str,
    expected_source_identity_sha256: str,
    exact_go_token: str,
) -> dict[str, Any]:
    """Pass B：新进程 exact-once 打开 75 labels；无 provider/checkpoint 参数。"""

    if exact_go_token != STAGE2A_SELECTION_GO:
        raise PermissionError("selection Pass B 缺 exact GO token")
    public = verify_e018_p1_stage2a_selection_public(
        selection_config_path=selection_config_path,
        stage2a_config_path=stage2a_config_path,
        qualification_config_path=qualification_config_path,
        public_root=public_root,
        expected_source_git_commit=expected_source_git_commit,
        expected_source_identity_sha256=expected_source_identity_sha256,
    )
    loaded = load_e018_p1_stage2a_selection_config(selection_config_path)
    producer = _verify_process_identity(
        public["producer_process_identity"], role="pass-a-producer"
    )
    scorer = _new_process_identity("pass-b-scorer")
    if (
        producer["process_instance_sha256"]
        == scorer["process_instance_sha256"]
    ):
        raise RuntimeError("selection Pass B 必须在不同 OS 进程执行")
    public_path = Path(public_root)
    private_path = Path(private_root)
    result_path = Path(result_root)
    artifact_root = _require_common_artifact_parent(
        public_root=public_path,
        private_root=private_path,
        result_root=result_path,
    )
    if result_path.exists() or result_path.is_symlink():
        raise FileExistsError("selection result root 必须全新")
    _assert_artifact_top_level_directories(
        artifact_root,
        expected_directories={"public_execution", "private_labels"},
    )
    _assert_private_tree_before_consumption(private_path)
    _verify_capture_state_before_scoring(
        private_path,
        transaction_identity_sha256=public["transaction_identity_sha256"],
    )
    inventory = _verify_private_inventory_public(
        _read_json(
            public_path / "private_label_inventory.json",
            "selection private inventory",
        )
    )
    branches = _read_jsonl(
        public_path / "gain_branch_ledger.jsonl",
        "selection gain branch ledger",
    )
    provider_rows = _read_jsonl(
        public_path / "provider_output_ledger.jsonl",
        "selection provider output ledger",
    )
    camera_rows = _read_jsonl(
        public_path / "camera_pose_ledger.jsonl",
        "selection camera pose ledger",
    )
    result_role_identity = _artifact_role_identity_sha256(
        role="result",
        config_canonical_sha256=loaded.canonical_sha256,
        source_identity_sha256=public["source_identity_sha256"],
        parent_verification_sha256=public["parent_verification_sha256"],
    )
    marker_path = private_path / "SCORING_CONSUMED.json"
    marker = _signed_consumption_marker(
        {
            "version": E018_P1_STAGE2A_SELECTION_RESULT_VERSION,
            "status": "PASS_B_CONSUMPTION_MARKER_CREATED_BEFORE_LABEL_OPEN",
            "experiment_id": E018_P1_STAGE2A_SELECTION_EXPERIMENT_ID,
            "classification": _SELECTION_CLASSIFICATION,
            "effect_claim": "no-effect-claim",
            "config_raw_sha256": loaded.raw_sha256,
            "config_canonical_sha256": loaded.canonical_sha256,
            "source_identity_sha256": public["source_identity_sha256"],
            "parent_verification_sha256": public[
                "parent_verification_sha256"
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
            "label_open_started_count": 0,
            "label_open_completed_count": 0,
            "provider_forward_count": 0,
            "checkpoint_load_count": 0,
            "result_root_created": False,
            "update_sequence": 0,
            "failure": None,
            "started_at_unix_ns": time.time_ns(),
            "last_updated_at_unix_ns": time.time_ns(),
            "completed_at_unix_ns": None,
        }
    )
    _durable_create_json_exclusive(marker_path, marker)
    labels: list[dict[str, Any]] = []
    try:
        result_path.mkdir(mode=0o700, parents=True, exist_ok=False)
        _assert_artifact_top_level_directories(
            artifact_root,
            expected_directories=_ARTIFACT_TOP_LEVEL_DIRECTORIES,
        )
        marker = _replace_consumption_marker(
            marker_path,
            marker,
            status="PASS_B_RESULT_ROOT_CREATED_BEFORE_LABEL_OPEN",
            result_root_created=True,
        )
        for index, identity in enumerate(inventory):
            marker = _replace_consumption_marker(
                marker_path,
                marker,
                status="PASS_B_PRIVATE_LABEL_OPEN_IN_PROGRESS",
                label_open_started_count=index + 1,
            )
            label_path = private_path / identity["path"]
            label, raw_sha256, size_bytes = _read_private_label_exactly_once(
                label_path
            )
            _validate_private_label_for_scoring(
                label,
                expected_label_index=index,
            )
            prediction_index = (index // 3) * 4 + 1 + index % 3
            receipt = _read_json(
                public_path
                / "prediction_commits"
                / f"{prediction_index:06d}.commit.json",
                f"selection prediction receipt for label[{index}]",
            )
            camera_index = (index // 3) * 92 + STAGE2A_COLLECT_FRAME_INDICES[
                index % 3
            ]
            camera = camera_rows[camera_index]
            if (
                identity["label_index"] != index
                or identity["path"] != f"label_commits/{index:06d}.json"
                or identity["raw_sha256"] != raw_sha256
                or identity["size_bytes"] != size_bytes
                or label["transaction_identity_sha256"]
                != public["transaction_identity_sha256"]
                or label["prediction_row_index"] != prediction_index
                or label["prediction_commit_receipt_sha256"]
                != receipt["commit_receipt_sha256"]
                or label["provider_output_digest"]
                != provider_rows[prediction_index]["provider_output_digest"]
                or label["provider_output_digest"]
                != receipt["provider_output_digest"]
                or label["seed"] != STAGE2A_SELECTION_SEEDS[index // 3]
                or label["route_frame_index"]
                != STAGE2A_COLLECT_FRAME_INDICES[index % 3]
                or label["rgb_sha256"] != camera["rgb_sha256"]
                or label["actual_pose_sha256"]
                != _array_sha256(
                    np.asarray(
                        camera["actual_base_from_external_camera_cv"],
                        dtype=np.float64,
                    )
                )
                or identity["scoring_primitive_sha256"]
                != _scoring_primitive_sha256(label)
                or label["privileged_captured_at_unix_ns"]
                <= receipt["prediction_fsync_completed_at_unix_ns"]
            ):
                raise RuntimeError(
                    f"selection private label[{index}] public binding 漂移"
                )
            labels.append(label)
            marker = _replace_consumption_marker(
                marker_path,
                marker,
                status="PASS_B_PRIVATE_LABEL_OPEN_IN_PROGRESS",
                label_open_completed_count=index + 1,
            )
        scored, summary = score_gain_branches(branches, labels)
        independently_recomputed = _recompute_selection_summary_from_scored(
            [
                _validate_scored_selection_row(value, row_index=index)
                for index, value in enumerate(scored)
            ],
            minimum_support=loaded.payload["gain_selection"][
                "minimum_common_denominator_routes"
            ],
        )
        if summary != independently_recomputed:
            raise RuntimeError("selection scorer summary 独立重算不一致")
        artifact_budget = loaded.payload["budgets"][
            "combined_artifact_bytes_max"
        ]
        if _combined_artifact_bytes(artifact_root) + 1_048_576 > artifact_budget:
            raise RuntimeError("selection Pass B artifact budget 缺少结果完成余量")
        marker = _replace_consumption_marker(
            marker_path,
            marker,
            status="PASS_B_PRIVATE_LABELS_CONSUMED_EXACT_ONCE",
            completed_at_unix_ns=time.time_ns(),
        )
    except Exception as error:
        try:
            _replace_consumption_marker(
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
        return _publish_pass_b_selection_result(
            selection_config_path=selection_config_path,
            stage2a_config_path=stage2a_config_path,
            qualification_config_path=qualification_config_path,
            public_root=public_root,
            result_path=result_path,
            artifact_root=artifact_root,
            marker_path=marker_path,
            loaded=loaded,
            public=public,
            producer=producer,
            scorer=scorer,
            result_role_identity=result_role_identity,
            scored=scored,
            summary=summary,
            expected_source_git_commit=expected_source_git_commit,
            expected_source_identity_sha256=expected_source_identity_sha256,
        )
    except Exception as error:
        if completion_path.exists() or completion_path.is_symlink():
            completion_path.unlink()
            _fsync_directory(result_path)
        _replace_consumption_marker(
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
