"""E018-P1 G2B-CAL-v2 的 pregrasp cohort 协议检查。"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from robot_vla.contracts import OUTCOME_PREDICATE_VERSION, PICK_AND_PLACE_SKILLS
from robot_vla.data.trajectory import TrajectoryStore
from robot_vla.precision.calibrated_front_provider import canonical_sha256
from robot_vla.precision.data import PrecisionLabelStore, file_sha256

G2B_TASK_TAXONOMY_VERSION = "robot-vla-task-spec-skill-taxonomy/v1"
G2B_CAL_RESULT_VERSION = (
    "e018-p1-g2b-covariance-calibrated-provider-requalification-result/v2"
)
G2B_TASK_TAXONOMY = {
    "version": G2B_TASK_TAXONOMY_VERSION,
    "task_id": "pick-cube-to-region",
    "task_group_id": "pick-and-place",
    "skill_names": list(PICK_AND_PLACE_SKILLS),
    "outcome_predicate_version": OUTCOME_PREDICATE_VERSION,
}
G2B_TASK_TAXONOMY_SHA256 = canonical_sha256(G2B_TASK_TAXONOMY)

_PREDICTION_FORBIDDEN_KEYS = {
    "gt_observable",
    "gt_object_position_base_m",
    "gt_projected_normalized_uv",
    "object_position_base_m",
    "object_mask",
    "goal_mask",
    "is_grasped",
    "finger_force_valid",
    "left_finger_force_n",
    "right_finger_force_n",
    "gripper_opening_ratio",
    "world_xy_error_vector_m",
    "mahalanobis_squared",
    "calibration_selected",
    "segmentation",
}


def task_taxonomy_from_manifest_entry(entry: Any) -> dict[str, Any]:
    return {
        "version": G2B_TASK_TAXONOMY_VERSION,
        "task_id": entry.task.task_id,
        "task_group_id": entry.task.task_group_id,
        "skill_names": list(entry.task.skill_names),
        "outcome_predicate_version": entry.task.outcome_predicate_version,
    }


def _read_json(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{name} 不存在: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{name} 必须是 JSON object")
    return value


def verify_failed_v1_lineage(root: Path, *, config: dict[str, Any]) -> dict[str, Any]:
    """把 CAL-v2 绑定到只读的 v1 protocol-invalid artifact。"""

    lineage = config["failed_v1_lineage"]
    required = {
        "config_snapshot.json": lineage["config_snapshot_raw_sha256"],
        "calibration_prediction_ledger.jsonl": lineage["prediction_ledger_raw_sha256"],
        "calibration_prediction_freeze.json": lineage["prediction_freeze_raw_sha256"],
        "failure.json": lineage["failure_raw_sha256"],
    }
    for relative, expected_sha256 in required.items():
        path = root / relative
        if not path.is_file() or file_sha256(path) != expected_sha256:
            raise RuntimeError(f"G2B CAL-v2 failed-v1 lineage 漂移: {relative}")
    v1_config = _read_json(root / "config_snapshot.json", "G2B CAL-v1 config snapshot")
    if (
        v1_config.get("version") != lineage["config_version"]
        or canonical_sha256(v1_config) != lineage["config_canonical_sha256"]
    ):
        raise RuntimeError("G2B CAL-v2 failed-v1 config semantic identity 漂移")
    freeze = _read_json(
        root / "calibration_prediction_freeze.json",
        "G2B CAL-v1 prediction freeze",
    )
    freeze_marker_sha256 = freeze.get("freeze_marker_sha256")
    freeze_payload = dict(freeze)
    freeze_payload.pop("freeze_marker_sha256", None)
    if (
        freeze_marker_sha256 != lineage["prediction_freeze_marker_sha256"]
        or canonical_sha256(freeze_payload) != freeze_marker_sha256
        or freeze.get("prediction_ledger_sha256")
        != lineage["prediction_ledger_raw_sha256"]
        or freeze.get("prediction_count") != 4154
        or freeze.get("status") != "frozen-before-validation-label-read"
    ):
        raise RuntimeError("G2B CAL-v2 failed-v1 freeze semantic identity 漂移")
    failure = _read_json(root / "failure.json", "G2B CAL-v1 failure")
    if (
        failure.get("status") != "failed-preserved"
        or failure.get("error_type") != "RuntimeError"
        or failure.get("error_message") != "G2B CAL object position 不符合冻结 task plane"
        or failure.get("prediction_ledger_exists") is not True
        or any(
            failure.get(name) != 0
            for name in (
                "test_trajectory_array_read_count",
                "test_label_array_read_count",
                "live_memory_read_count",
                "live_memory_write_count",
                "runtime_camera_actuation_count",
                "arm_actuation_count",
                "manipulation_progression_count",
            )
        )
    ):
        raise RuntimeError("G2B CAL-v2 failed-v1 failure semantic identity 漂移")
    if any(
        (root / name).exists()
        for name in (
            "calibration_receipt.json",
            "calibration_summary.json",
            "calibration.json",
            "calibration_scoring_ledger.jsonl",
        )
    ):
        raise RuntimeError("G2B CAL-v1 protocol-invalid artifact 不得补写完成产物")
    source_identity = _read_json(root / "source_identity.json", "G2B CAL-v1 source identity")
    if source_identity.get("git_commit") != lineage["source_git_commit"]:
        raise RuntimeError("G2B CAL-v2 failed-v1 source commit 漂移")
    binding = {
        "version": G2B_CAL_RESULT_VERSION,
        "classification": lineage["classification"],
        "source_git_commit": lineage["source_git_commit"],
        "config_canonical_sha256": lineage["config_canonical_sha256"],
        "config_snapshot_raw_sha256": lineage["config_snapshot_raw_sha256"],
        "prediction_ledger_raw_sha256": lineage["prediction_ledger_raw_sha256"],
        "prediction_freeze_raw_sha256": lineage["prediction_freeze_raw_sha256"],
        "prediction_freeze_marker_sha256": lineage[
            "prediction_freeze_marker_sha256"
        ],
        "failure_raw_sha256": lineage["failure_raw_sha256"],
        "v1_completion_receipt_exists": False,
    }
    binding["binding_sha256"] = canonical_sha256(binding)
    return binding


def verify_e018_p0_pregrasp_binding(
    repository_root: Path,
    *,
    config: dict[str, Any],
) -> None:
    binding = config["calibration_data"]["e018_p0_pregrasp_binding"]
    path = repository_root / "configs/e018_p0_dual_memory_development_v1.json"
    payload = _read_json(path, "E018-P0 pregrasp parent config")
    if (
        payload.get("version") != binding["config_version"]
        or canonical_sha256(payload) != binding["config_sha256"]
        or payload.get("source", {}).get("pregrasp_skill_id")
        != binding["pregrasp_skill_id"]
    ):
        raise RuntimeError("G2B CAL-v2 E018-P0 pregrasp binding 漂移")


def assert_prediction_ledger_deployable_only(rows: list[dict[str, Any]]) -> None:
    def walk(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in _PREDICTION_FORBIDDEN_KEYS or key.startswith("gt_"):
                    raise ValueError(
                        f"G2B CAL prediction ledger 含 privileged field: {path}.{key}"
                    )
                walk(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")

    for index, row in enumerate(rows):
        walk(row, f"rows[{index}]")


def audit_prediction_applicability(
    rows: list[dict[str, Any]],
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    """核对 Phase-A deployable skill cohort，禁止在 Phase-B 重选 cohort。"""

    data = config["calibration_data"]
    pregrasp_skill_id = int(data["pregrasp_skill_id"])
    expected_skill_counts = {
        int(skill_id): int(count) for skill_id, count in data["expected_skill_counts"].items()
    }
    skill_counts: Counter[int] = Counter(
        {skill_id: 0 for skill_id in expected_skill_counts}
    )
    applicable_count = 0
    applicable_trajectories: set[str] = set()
    for index, row in enumerate(rows):
        skill_id = row.get("skill_id")
        applicable = row.get("calibration_applicable")
        taxonomy_sha256 = row.get("task_spec_taxonomy_sha256")
        if type(skill_id) is not int or skill_id not in expected_skill_counts:
            raise RuntimeError(f"G2B CAL-v2 row[{index}] skill_id 漂移")
        if type(applicable) is not bool or applicable != (skill_id == pregrasp_skill_id):
            raise RuntimeError(f"G2B CAL-v2 row[{index}] applicability 漂移")
        if taxonomy_sha256 != data["task_spec_taxonomy"]["identity_sha256"]:
            raise RuntimeError(f"G2B CAL-v2 row[{index}] taxonomy identity 漂移")
        skill_counts[skill_id] += 1
        if applicable:
            applicable_count += 1
            applicable_trajectories.add(str(row["trajectory_id"]))
    nonapplicable_count = len(rows) - applicable_count
    actual_skill_counts = dict(sorted(skill_counts.items()))
    if actual_skill_counts != expected_skill_counts:
        raise RuntimeError(
            "G2B CAL-v2 Phase-A skill counts 漂移: "
            f"expected={expected_skill_counts}, actual={actual_skill_counts}"
        )
    if (
        len(rows) != data["manifest_sample_count"]
        or applicable_count != data["expected_applicable_frame_count"]
        or nonapplicable_count != data["expected_nonapplicable_frame_count"]
        or len(applicable_trajectories)
        != data["expected_trajectories_with_applicable_frames"]
    ):
        raise RuntimeError("G2B CAL-v2 Phase-A applicability cohort identity 漂移")
    audit = {
        "version": G2B_CAL_RESULT_VERSION,
        "applicability_source": data["applicability_source"],
        "applicability_predicate": config["calibration"]["applicability_predicate"],
        "task_spec_taxonomy_sha256": data["task_spec_taxonomy"]["identity_sha256"],
        "pregrasp_skill_id": pregrasp_skill_id,
        "frame_count": len(rows),
        "skill_counts": {str(key): value for key, value in actual_skill_counts.items()},
        "applicable_frame_count": applicable_count,
        "nonapplicable_frame_count": nonapplicable_count,
        "trajectories_with_applicable_frames": len(applicable_trajectories),
    }
    audit["audit_sha256"] = canonical_sha256(audit)
    return audit


def audit_cohort_invariants(
    *,
    predictions: list[dict[str, Any]],
    config: dict[str, Any],
    label_by_trajectory: dict[str, Any],
    label_store: PrecisionLabelStore,
    deployable_by_trajectory: dict[str, Any],
    deployable_store: TrajectoryStore,
    phase_a_applicability_audit: dict[str, Any],
) -> dict[str, Any]:
    """对完整 skill-0 cohort 做 fail-whole invariant，不执行 calibration selection。"""

    invariants = config["calibration"]["cohort_invariants"]
    violation_counts: Counter[str] = Counter(
        {
            "object_position_finite": 0,
            "task_plane": 0,
            "is_grasped": 0,
            "finger_force_valid": 0,
            "left_finger_force": 0,
            "right_finger_force": 0,
            "raw_gripper_opening_ratio": 0,
        }
    )
    violation_examples: dict[str, list[dict[str, Any]]] = {
        name: [] for name in violation_counts
    }
    applicable_count = 0
    applicable_trajectories: set[str] = set()
    label_files_read: set[str] = set()

    def record(name: str, *, prediction: dict[str, Any], value: Any) -> None:
        violation_counts[name] += 1
        if len(violation_examples[name]) < 10:
            violation_examples[name].append(
                {
                    "trajectory_id": str(prediction["trajectory_id"]),
                    "timestep": int(prediction["timestep"]),
                    "value": value,
                }
            )

    for prediction in predictions:
        trajectory_id = str(prediction["trajectory_id"])
        label_meta = label_by_trajectory.get(trajectory_id)
        deployable_meta = deployable_by_trajectory.get(trajectory_id)
        if label_meta is None or deployable_meta is None:
            raise RuntimeError("G2B CAL-v2 cohort 缺少 source/label trajectory")
        timestep = int(prediction["timestep"])
        deployable_arrays = deployable_store.get(deployable_meta)
        if not 0 <= timestep < deployable_arrays.num_steps:
            raise RuntimeError("G2B CAL-v2 cohort timestep 超出 deployable trajectory")
        raw_skill_id = int(deployable_arrays.skill_id[timestep])
        if raw_skill_id != prediction["skill_id"]:
            raise RuntimeError("G2B CAL-v2 frozen skill_id 与 deployable source 漂移")
        if not prediction["calibration_applicable"]:
            continue

        applicable_count += 1
        applicable_trajectories.add(trajectory_id)
        label_arrays = label_store.get(label_meta)
        label_files_read.add(label_meta.file)
        if not 0 <= timestep < label_arrays.num_steps:
            raise RuntimeError("G2B CAL-v2 cohort timestep 超出 label trajectory")
        object_position = np.asarray(
            label_arrays.object_position_base_m[timestep],
            dtype=np.float64,
        )
        finite_position = bool(
            object_position.shape == (3,) and np.isfinite(object_position).all()
        )
        if not finite_position:
            record(
                "object_position_finite",
                prediction=prediction,
                value=object_position.tolist(),
            )
        task_plane_ok = bool(
            finite_position
            and abs(float(object_position[2]) - invariants["task_plane_base_z_m"])
            <= invariants["task_plane_tolerance_m"]
        )
        if not task_plane_ok:
            record(
                "task_plane",
                prediction=prediction,
                value=None if not finite_position else float(object_position[2]),
            )
        is_grasped = bool(deployable_arrays.is_grasped[timestep])
        if is_grasped != invariants["is_grasped_required"]:
            record("is_grasped", prediction=prediction, value=is_grasped)
        force_valid = bool(deployable_arrays.finger_force_valid[timestep])
        if force_valid != invariants["finger_force_valid_required"]:
            record("finger_force_valid", prediction=prediction, value=force_valid)
        left_force = float(deployable_arrays.left_finger_force_n[timestep])
        if not math.isfinite(left_force) or left_force > invariants[
            "maximum_left_finger_force_n"
        ]:
            record("left_finger_force", prediction=prediction, value=left_force)
        right_force = float(deployable_arrays.right_finger_force_n[timestep])
        if not math.isfinite(right_force) or right_force > invariants[
            "maximum_right_finger_force_n"
        ]:
            record("right_finger_force", prediction=prediction, value=right_force)
        gripper_ratio = float(deployable_arrays.proprio[timestep, -1])
        if (
            not math.isfinite(gripper_ratio)
            or gripper_ratio < invariants["minimum_raw_gripper_opening_ratio"]
        ):
            record(
                "raw_gripper_opening_ratio",
                prediction=prediction,
                value=gripper_ratio,
            )

    data = config["calibration_data"]
    if (
        applicable_count != data["expected_applicable_frame_count"]
        or len(applicable_trajectories)
        != data["expected_trajectories_with_applicable_frames"]
    ):
        raise RuntimeError("G2B CAL-v2 Phase-B cohort identity 漂移")
    total_violations = sum(violation_counts.values())
    audit = {
        "version": G2B_CAL_RESULT_VERSION,
        "status": "cohort-valid" if total_violations == 0 else "protocol-invalid",
        "applicability_source": data["applicability_source"],
        "applicability_predicate": config["calibration"]["applicability_predicate"],
        "phase_a_applicability_audit_sha256": phase_a_applicability_audit["audit_sha256"],
        "frame_count": len(predictions),
        "applicable_frame_count": applicable_count,
        "nonapplicable_frame_count": len(predictions) - applicable_count,
        "applicable_trajectory_count": len(applicable_trajectories),
        "invariants": invariants,
        "invariant_violation_counts": dict(sorted(violation_counts.items())),
        "invariant_total_violation_events": total_violations,
        "invariant_violation_examples": violation_examples,
        "cohort_passed": total_violations == 0,
        "selection_evaluated": total_violations == 0,
        "validation_label_array_file_read_count": len(label_files_read),
        "test_trajectory_array_read_count": 0,
        "test_label_array_read_count": 0,
        "live_memory_read_count": 0,
        "live_memory_write_count": 0,
        "runtime_camera_actuation_count": 0,
        "arm_actuation_count": 0,
        "manipulation_progression_count": 0,
        "provider_training_count": 0,
    }
    audit["audit_sha256"] = canonical_sha256(audit)
    return audit


__all__ = [
    "G2B_CAL_RESULT_VERSION",
    "G2B_TASK_TAXONOMY",
    "G2B_TASK_TAXONOMY_SHA256",
    "G2B_TASK_TAXONOMY_VERSION",
    "assert_prediction_ledger_deployable_only",
    "audit_cohort_invariants",
    "audit_prediction_applicability",
    "task_taxonomy_from_manifest_entry",
    "verify_e018_p0_pregrasp_binding",
    "verify_failed_v1_lineage",
]
