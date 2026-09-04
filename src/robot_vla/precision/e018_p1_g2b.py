"""E018-P1 G2B：covariance calibration 与 provider requalification。

G2B-CAL 先只用 E016 fresh-validation 的 deployable wrist 输入推理并冻结
prediction ledger，随后才从磁盘重载 ledger 并读取 validation label。通过 CAL
后，资格验证沿用 G2A 的 seeds、viewpoints、绝对门槛、native cascade 和 PRIMARY
规则，但使用稳定 identity 以及真正进入 provider/write evidence 的 calibrated sigma。

本模块不读取 test trajectory/label arrays，不训练模型，不访问 Memory，也不执行
机器人或运行时相机动作。
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from robot_vla.contracts import RobotSpec
from robot_vla.data.dataset import ObservationV2ActionChunkDataset
from robot_vla.data.trajectory import load_manifest
from robot_vla.observation import rotation_6d_to_matrix, validate_se3
from robot_vla.precision.calibrated_front_provider import (
    SCALAR_COVARIANCE_CALIBRATION_METHOD,
    ScalarCovarianceCalibration,
    array_sha256,
    build_calibrated_object_write_evidence,
    build_calibrated_provider_identity,
    build_stable_camera_calibration_identity,
    canonical_sha256,
)
from robot_vla.precision.data import (
    PrecisionLabelStore,
    file_sha256,
    load_precision_label_manifest,
)
from robot_vla.precision.e018_p1_g2a import (
    FRONT_ALTERNATE_IDS,
    FRONT_HOME_ID,
    NATIVE_WRIST_CONTROL_ID,
    PER_SCENE_CAPTURE_ORDER,
    _artifact_hashes,
    _atomic_text,
    _base_point,
    _base_transforms,
    _capture_deployable_phase,
    _load_model_context,
    _mahalanobis_squared_psd,
    _numpy,
    _report_markdown,
    _score_prediction,
    _verify_g0c_receipt,
    _viewpoint_map,
    assert_prediction_ledger_deployable_only,
    audit_g2a_seed_disjointness,
    build_e013_wrist_pose_envelope,
    finalize_qualification_summaries,
    load_e018_p1_g2a_config,
    summarize_qualification_rows,
)
from robot_vla.precision.geometry import project_base_point_to_normalized_uv
from robot_vla.precision.object_observability import derive_object_observability
from robot_vla.precision.observability import mask_probability_at_normalized_uv
from robot_vla.precision.outliers import geometry_conditioning

E018_P1_G2B_CONFIG_VERSION = (
    "e018-p1-g2b-covariance-calibrated-provider-requalification-development/v1"
)
E018_P1_G2B_RESULT_VERSION = "e018-p1-g2b-covariance-calibrated-provider-requalification-result/v1"
E018_P1_G2B_CAL_GATE = "G2B_CAL_COVARIANCE_CALIBRATION"
E018_P1_G2B_QUALIFICATION_GATE = "G2B_PROVIDER_REQUALIFICATION"
G2B_CAL_PRIMITIVE_ID = "WRIST_VALIDATION_CALIBRATION"
_SOURCE_FILES = (
    "src/robot_vla/precision/calibrated_front_provider.py",
    "src/robot_vla/precision/e018_p1_g2b.py",
    "src/robot_vla/cli/run_e018_p1_g2b.py",
    "configs/e018_p1_g2b_covariance_calibrated_provider_requalification_development_v1.json",
)
_CAL_ARTIFACT_NAMES = (
    "config_snapshot.json",
    "source_identity.json",
    "parent_g2a_receipt_binding.json",
    "manifest_audit.json",
    "calibration_inference_audit.json",
    "calibration_prediction_ledger.jsonl",
    "calibration_prediction_freeze.json",
    "calibration_scoring_ledger.jsonl",
    "calibration.json",
    "calibration_summary.json",
)
_QUALIFICATION_ARTIFACT_NAMES = (
    "config_snapshot.json",
    "source_identity.json",
    "parent_g2a_config_snapshot.json",
    "calibration_binding.json",
    "seed_audit.json",
    "wrist_pose_envelope.json",
    "deployable_capture_audit.json",
    "inference_audit.json",
    "prediction_ledger.jsonl",
    "prediction_freeze.json",
    "offline_scoring_ledger.jsonl",
    "offline_scoring_audit.json",
    "summary.json",
    "report.md",
)


def _require_keys(value: Any, expected: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} 必须是 JSON object")
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{name} keys 漂移: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _require_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} 必须是 64 位小写 SHA-256")
    return value


def _require_git_commit(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} 必须是 40 位小写 Git commit")
    return value


def _read_json(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{name} 不存在: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{name} 必须是 JSON object")
    return value


def _fsync_parent_directory(path: Path) -> None:
    descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        _fsync_parent_directory(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        _fsync_parent_directory(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_jsonl(path: Path, name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise RuntimeError(f"{name} 第 {line_number} 行为空")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{name} 第 {line_number} 行不是 object")
            rows.append(value)
    return rows


def _source_identity(
    repository_root: Path,
    *,
    source_parent_git_commit: str,
) -> dict[str, Any]:
    safe_repository = str(repository_root.resolve())
    git = ("git", "-c", f"safe.directory={safe_repository}")
    _require_git_commit(source_parent_git_commit, "source_parent_git_commit")
    missing = [relative for relative in _SOURCE_FILES if not (repository_root / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"G2B source files 缺失: {missing}")
    parent_is_ancestor = (
        subprocess.run(
            [
                *git,
                "merge-base",
                "--is-ancestor",
                source_parent_git_commit,
                "HEAD",
            ],
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
        ).returncode
        == 0
    )
    identity = {
        "git_commit": subprocess.run(
            [*git, "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "source_parent_git_commit": source_parent_git_commit,
        "source_parent_is_ancestor": parent_is_ancestor,
        "git_parent_commit": subprocess.run(
            [*git, "rev-parse", "HEAD^"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "git_status": subprocess.run(
            [*git, "status", "--porcelain", "--untracked-files=all"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines(),
        "source_file_sha256": {
            relative: file_sha256(repository_root / relative) for relative in _SOURCE_FILES
        },
    }
    identity["worktree_clean"] = not identity["git_status"]
    identity["identity_sha256"] = canonical_sha256(identity)
    return identity


def load_e018_p1_g2b_config(
    path: str | Path,
    *,
    parent_g2a_config_path: str | Path,
) -> dict[str, Any]:
    """严格加载 G2B 配置，并确认没有改变 G2A 资格门槛。"""

    config = _require_keys(
        _read_json(Path(path), "E018-P1 G2B config"),
        {
            "version",
            "status",
            "scope",
            "parents",
            "calibration_data",
            "calibration",
            "qualification",
            "execution",
        },
        "E018-P1 G2B config",
    )
    if config["version"] != E018_P1_G2B_CONFIG_VERSION:
        raise ValueError("G2B config version 漂移")
    if (
        config["status"]
        != "development-only-calibration-and-provider-requalification-no-formal-claim"
    ):
        raise ValueError("G2B 只能运行 development-only calibration/requalification")
    expected_scope = {
        "calibration_gate": E018_P1_G2B_CAL_GATE,
        "qualification_gate": E018_P1_G2B_QUALIFICATION_GATE,
        "allowed_label_split": "val",
        "test_manifest_metadata_read_allowed": True,
        "test_trajectory_array_read_allowed": False,
        "test_label_array_read_allowed": False,
        "formal_claim_allowed": False,
        "runtime_gt_control_allowed": False,
        "live_memory_read_allowed": False,
        "live_memory_write_allowed": False,
        "runtime_camera_actuation_allowed": False,
        "arm_actuation_allowed": False,
        "manipulation_progression_allowed": False,
        "provider_training_allowed": False,
    }
    if config["scope"] != expected_scope:
        raise ValueError("G2B scope 必须保持 val-only/no-test-array/no-memory/no-actuation")

    parents = _require_keys(
        config["parents"],
        {
            "source_parent_git_commit",
            "g2a_config_version",
            "g2a_config_sha256",
            "g2a_receipt_raw_sha256",
            "g2a_receipt_internal_sha256",
            "g2a_gate_passed",
            "g2a_status",
            "e016_config_sha256",
            "checkpoint_sha256",
            "checkpoint_parameter_sha256",
            "checkpoint_provenance_sha256",
            "model_config_sha256",
            "selected_epoch",
            "source_training_camera",
        },
        "parents",
    )
    _require_git_commit(
        parents["source_parent_git_commit"],
        "parents.source_parent_git_commit",
    )
    for name in (
        "g2a_config_sha256",
        "g2a_receipt_raw_sha256",
        "g2a_receipt_internal_sha256",
        "e016_config_sha256",
        "checkpoint_sha256",
        "checkpoint_parameter_sha256",
        "checkpoint_provenance_sha256",
        "model_config_sha256",
    ):
        _require_sha256(parents[name], f"parents.{name}")
    if (
        parents["g2a_config_version"] != "e018-p1-g2a-front-provider-qualification-development/v1"
        or parents["g2a_gate_passed"] is not False
        or parents["g2a_status"] != "inconclusive_parent_health"
        or parents["selected_epoch"] != 12
        or parents["source_training_camera"] != "hand_camera"
    ):
        raise ValueError("G2B parent G2A/checkpoint identity 漂移")
    parent_g2a_path = Path(parent_g2a_config_path)
    parent_g2a = _read_json(parent_g2a_path, "parent G2A config")
    if (
        canonical_sha256(parent_g2a) != parents["g2a_config_sha256"]
        or parent_g2a.get("version") != parents["g2a_config_version"]
    ):
        raise ValueError("G2B parent G2A config SHA/version 漂移")
    for name in (
        "e016_config_sha256",
        "checkpoint_sha256",
        "checkpoint_parameter_sha256",
        "checkpoint_provenance_sha256",
        "model_config_sha256",
    ):
        if parent_g2a["parents"][name] != parents[name]:
            raise ValueError(f"G2B {name} 未精确继承 G2A")

    data = _require_keys(
        config["calibration_data"],
        {
            "deployable_manifest_sha256",
            "privileged_label_manifest_sha256",
            "validation_files_identity_sha256",
            "split",
            "camera_uid",
            "trajectory_count",
            "manifest_sample_count",
            "seed_source",
            "require_disjoint_from_g0_g0b_g0c_g1a_g2a_and_all_other_manifest_splits",
            "normalizer_source",
            "proprio_stats_sha256",
            "proprio_normalizer_sha256",
            "finger_force_stats_sha256",
            "finger_force_normalizer_sha256",
            "phase_a_policy",
            "phase_b_policy",
        },
        "calibration_data",
    )
    for name in (
        "deployable_manifest_sha256",
        "privileged_label_manifest_sha256",
        "validation_files_identity_sha256",
        "proprio_stats_sha256",
        "proprio_normalizer_sha256",
        "finger_force_stats_sha256",
        "finger_force_normalizer_sha256",
    ):
        _require_sha256(data[name], f"calibration_data.{name}")
    if (
        data["split"] != "val"
        or data["camera_uid"] != "hand_camera"
        or data["trajectory_count"] != 20
        or data["manifest_sample_count"] != 4154
        or data["seed_source"] != "e016-fresh-manifest-randomization.seed/v1"
        or data["require_disjoint_from_g0_g0b_g0c_g1a_g2a_and_all_other_manifest_splits"]
        is not True
        or data["normalizer_source"] != "frozen-e013-training-provider-stats/v1"
        or data["phase_a_policy"]
        != "deployable-val-only-predict-fsync-parent-fsync-freeze-delete-context/v1"
        or data["phase_b_policy"] != "reload-frozen-ledger-before-opening-val-label-arrays/v1"
    ):
        raise ValueError("G2B CAL 数据、split 或两阶段策略漂移")
    inherited_data = parent_g2a["data_identity"]
    if (
        data["deployable_manifest_sha256"] != inherited_data["e016_fresh_manifest_sha256"]
        or data["proprio_stats_sha256"] != inherited_data["proprio_stats_sha256"]
        or data["proprio_normalizer_sha256"] != inherited_data["proprio_normalizer_sha256"]
        or data["finger_force_stats_sha256"] != inherited_data["finger_force_stats_sha256"]
        or data["finger_force_normalizer_sha256"]
        != inherited_data["finger_force_normalizer_sha256"]
    ):
        raise ValueError("G2B CAL manifest/stats 未精确继承 provider identity")

    calibration = _require_keys(
        config["calibration"],
        {
            "method",
            "alpha",
            "target_coverage",
            "chi_square_threshold",
            "order_statistic",
            "scale_rule",
            "selection_predicate",
            "selection_must_not_use",
            "minimum_support_count",
            "maximum_calibrated_position_std_m",
            "singular_nullspace_nonzero_error_policy",
            "failure_policy",
        },
        "calibration",
    )
    if calibration != {
        "method": SCALAR_COVARIANCE_CALIBRATION_METHOD,
        "alpha": 0.05,
        "target_coverage": 0.95,
        "chi_square_threshold": 5.991,
        "order_statistic": "ceil((n+1)*0.95)-one-based/v1",
        "scale_rule": "max(1,q/5.991)/v1",
        "selection_predicate": (
            "gt-object-observable-and-geometry-valid-and-raw-covariance-finite-symmetric-psd/v1"
        ),
        "selection_must_not_use": [
            "confidence",
            "write_accepted",
            "prediction_error_magnitude",
        ],
        "minimum_support_count": 30,
        "maximum_calibrated_position_std_m": 0.02,
        "singular_nullspace_nonzero_error_policy": "calibration-no-go/v1",
        "failure_policy": "freeze-negative-receipt-stop-no-complex-model-switch/v1",
    }:
        raise ValueError("G2B CAL 算法/样本选择/失败策略漂移")

    qualification = _require_keys(
        config["qualification"],
        {
            "inherit_all_sampling_viewpoints_thresholds_primary_and_native_cascade_from_g2a",
            "same_seed_range",
            "required_seed_count",
            "calibrated_covariance_must_reach_provider_measurement",
            "calibrated_sigma_must_reach_object_write_evidence",
            "preflight_required_before_full_run",
            "full_run_requires_decision_agent_exit_go",
            "same_identity_rerun_allowed",
        },
        "qualification",
    )
    if qualification != {
        "inherit_all_sampling_viewpoints_thresholds_primary_and_native_cascade_from_g2a": True,
        "same_seed_range": [75001, 75050],
        "required_seed_count": 50,
        "calibrated_covariance_must_reach_provider_measurement": True,
        "calibrated_sigma_must_reach_object_write_evidence": True,
        "preflight_required_before_full_run": True,
        "full_run_requires_decision_agent_exit_go": True,
        "same_identity_rerun_allowed": False,
    }:
        raise ValueError("G2B qualification inheritance/exit gate 漂移")

    execution = _require_keys(
        config["execution"],
        {
            "device",
            "use_bf16",
            "batch_size",
            "num_workers",
            "save_rgb",
            "require_clean_worktree",
            "calibration_prediction_before_label_required",
            "qualification_prediction_before_gt_required",
            "environment_step_allowed",
            "runtime_camera_actuation_allowed",
            "provider_training_allowed",
            "memory_read_allowed",
            "memory_write_allowed",
            "manipulation_progression_allowed",
        },
        "execution",
    )
    if execution != {
        "device": "cuda",
        "use_bf16": True,
        "batch_size": 32,
        "num_workers": 0,
        "save_rgb": False,
        "require_clean_worktree": True,
        "calibration_prediction_before_label_required": True,
        "qualification_prediction_before_gt_required": True,
        "environment_step_allowed": False,
        "runtime_camera_actuation_allowed": False,
        "provider_training_allowed": False,
        "memory_read_allowed": False,
        "memory_write_allowed": False,
        "manipulation_progression_allowed": False,
    }:
        raise ValueError("G2B execution 必须保持 CUDA/no-test/no-training/no-memory/no-actuation")
    return config


def _verify_g2a_parent_receipt(path: Path, *, config: dict[str, Any]) -> dict[str, Any]:
    parents = config["parents"]
    if file_sha256(path) != parents["g2a_receipt_raw_sha256"]:
        raise RuntimeError("G2B parent G2A receipt raw SHA-256 漂移")
    receipt = _read_json(path, "parent G2A receipt")
    if (
        receipt.get("version") != "e018-p1-g2a-front-provider-qualification-result/v1"
        or receipt.get("status") != "complete-development-only"
        or receipt.get("gate_evaluated") is not True
        or receipt.get("gate_passed") is not False
        or receipt.get("config_sha256") != parents["g2a_config_sha256"]
        or receipt.get("receipt_sha256") != parents["g2a_receipt_internal_sha256"]
        or receipt.get("test_trajectory_array_read_count") != 0
        or receipt.get("test_label_array_read_count") != 0
    ):
        raise RuntimeError("G2B parent G2A receipt 内容/权限 identity 漂移")
    return receipt


def audit_g2b_calibration_manifests(
    *,
    config: dict[str, Any],
    parent_g2a_config: dict[str, Any],
    deployable_root: str | Path,
    label_root: str | Path,
) -> dict[str, Any]:
    """只读 manifest metadata；此函数不打开任何 trajectory/label NPZ。"""

    deployable = Path(deployable_root)
    labels = Path(label_root)
    data = config["calibration_data"]
    deployable_manifest = deployable / "manifest.jsonl"
    label_manifest = labels / "manifest.jsonl"
    if (
        not deployable_manifest.is_file()
        or file_sha256(deployable_manifest) != data["deployable_manifest_sha256"]
    ):
        raise RuntimeError("G2B E016 fresh deployable manifest identity 漂移")
    if (
        not label_manifest.is_file()
        or file_sha256(label_manifest) != data["privileged_label_manifest_sha256"]
    ):
        raise RuntimeError("G2B E016 fresh label manifest identity 漂移")

    deployable_entries = load_manifest(deployable)
    label_entries = load_precision_label_manifest(labels)
    deployable_by_id = {entry.trajectory_id: entry for entry in deployable_entries}
    label_by_id = {entry.trajectory_id: entry for entry in label_entries}
    if (
        len(deployable_by_id) != len(deployable_entries)
        or len(label_by_id) != len(label_entries)
        or set(deployable_by_id) != set(label_by_id)
    ):
        raise RuntimeError("G2B E016 source/label manifest trajectory identity 不一致")
    split_counts = Counter(entry.split for entry in deployable_entries)
    if split_counts != Counter({"train": 1, "val": 20, "test": 100}):
        raise RuntimeError(f"G2B E016 fresh split counts 漂移: {dict(split_counts)}")
    seeds_by_split: dict[str, set[int]] = {name: set() for name in split_counts}
    for entry in deployable_entries:
        seed = entry.randomization.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise RuntimeError("G2B E016 manifest randomization.seed 无效")
        if seed in seeds_by_split[entry.split]:
            raise RuntimeError("G2B E016 manifest split 内 seed 重复")
        seeds_by_split[entry.split].add(seed)
        label = label_by_id[entry.trajectory_id]
        if (
            label.split != entry.split
            or label.scene_id != entry.scene_id
            or label.num_steps != entry.num_steps
            or label.camera_name != data["camera_uid"]
        ):
            raise RuntimeError("G2B E016 source/label metadata 对齐或 camera identity 漂移")
    split_pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    split_overlaps = {
        f"{left}_{right}": sorted(seeds_by_split[left] & seeds_by_split[right])
        for left, right in split_pairs
    }
    if any(split_overlaps.values()):
        raise RuntimeError(f"G2B E016 split seed 泄漏: {split_overlaps}")

    val_entries = [entry for entry in deployable_entries if entry.split == "val"]
    val_seeds = {int(entry.randomization["seed"]) for entry in val_entries}
    known = {int(seed) for seed in parent_g2a_config["sampling"]["seeds"]}
    for values in parent_g2a_config["sampling"]["known_development_seeds"].values():
        known.update(int(seed) for seed in values)
    known_overlap = sorted(val_seeds & known)
    if known_overlap:
        raise RuntimeError(f"G2B CAL val seeds 与 development seeds 重叠: {known_overlap}")
    val_steps = sum(entry.num_steps for entry in val_entries)
    if len(val_entries) != data["trajectory_count"] or val_steps != data["manifest_sample_count"]:
        raise RuntimeError("G2B CAL val trajectory/sample count 漂移")

    result = {
        "version": E018_P1_G2B_RESULT_VERSION,
        "passed": True,
        "deployable_manifest_sha256": data["deployable_manifest_sha256"],
        "privileged_label_manifest_sha256": data["privileged_label_manifest_sha256"],
        "manifest_metadata_read_count": 2,
        "split_trajectory_counts": dict(sorted(split_counts.items())),
        "val_trajectory_count": len(val_entries),
        "val_manifest_sample_count": val_steps,
        "val_seed_count": len(val_seeds),
        "val_seed_min": min(val_seeds),
        "val_seed_max": max(val_seeds),
        "split_seed_overlaps": split_overlaps,
        "known_development_seed_overlap": known_overlap,
        "source_label_trajectory_sets_equal": True,
        "test_trajectory_array_read_count": 0,
        "test_label_array_read_count": 0,
    }
    result["validation_data_identity_sha256"] = canonical_sha256(result)
    return result


class _DeployableWristCalibrationDataset:
    """只构建 frozen provider 输入；构造和取样均不接收 label root。"""

    def __init__(
        self,
        deployable_root: str | Path,
        *,
        spec: RobotSpec,
        proprio_normalizer: Any,
        finger_force_normalizer: Any,
        cache_size: int = 32,
    ) -> None:
        self.root = Path(deployable_root)
        self.base = ObservationV2ActionChunkDataset(
            str(self.root),
            load_manifest(self.root, split="val"),
            spec,
            proprio_normalizer,
            finger_force_normalizer=finger_force_normalizer,
            cache_size=cache_size,
        )
        self.source_by_trajectory = {entry.trajectory_id: entry for entry in self.base.entries}

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.base[index]
        trajectory_id = str(sample["trajectory_id"])
        timestep = int(sample["timestep"])
        meta = self.source_by_trajectory[trajectory_id]
        arrays = self.base.store.get(meta)
        if not bool(sample["state_history_mask"][-1]):
            raise RuntimeError("G2B CAL current Observation V2 row 必须有效")
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation_6d_to_matrix(arrays.wrist_camera_rotation_6d_base[timestep])
        transform[:3, 3] = arrays.wrist_camera_position_base_m[timestep]
        transform = validate_se3(transform, "base_from_wrist_camera_cv")
        intrinsic = np.asarray(
            meta.camera_calibration.intrinsic_wrist,
            dtype=np.float64,
        ).reshape(3, 3)
        rgb = np.ascontiguousarray(sample["rgb_wrist"])
        state = np.ascontiguousarray(sample["state_history"][-1].copy())
        motion = np.zeros(4, dtype=np.float32)
        return {
            "model_inputs": {
                "rgb_wrist": rgb,
                "structured_state": state,
                "geometric_motion": motion,
            },
            "audit": {
                "trajectory_id": trajectory_id,
                "scene_id": meta.scene_id,
                "split": meta.split,
                "timestep": timestep,
                "timestamp_s": float(arrays.timestamp_wrist[timestep]),
                "base_from_camera_cv": transform,
                "intrinsic_cv": intrinsic,
            },
        }


def _calibration_collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    model_inputs = [sample["model_inputs"] for sample in samples]
    image = np.stack([item["rgb_wrist"] for item in model_inputs])
    return {
        "image": torch.from_numpy(
            np.ascontiguousarray(
                image.transpose(0, 3, 1, 2),
                dtype=np.float32,
            )
            / np.float32(255.0)
        ),
        "structured_state": torch.from_numpy(
            np.stack([item["structured_state"] for item in model_inputs])
        ),
        "geometric_motion": torch.from_numpy(
            np.stack([item["geometric_motion"] for item in model_inputs])
        ),
        "audit": [sample["audit"] for sample in samples],
    }


def _measurement_covariance(
    local_jacobian_xy_m_per_px: list[list[float]],
    sigma_xy_px: np.ndarray,
) -> np.ndarray:
    jacobian = np.asarray(local_jacobian_xy_m_per_px, dtype=np.float64)
    sigma = np.asarray(sigma_xy_px, dtype=np.float64)
    if (
        jacobian.shape != (2, 2)
        or sigma.shape != (2,)
        or not np.isfinite(jacobian).all()
        or not np.isfinite(sigma).all()
        or np.any(sigma < 0.0)
    ):
        raise ValueError("G2B measurement covariance 输入无效")
    covariance = np.zeros((3, 3), dtype=np.float64)
    covariance[:2, :2] = jacobian @ np.diag(np.square(sigma)) @ jacobian.T
    if not np.isfinite(covariance).all():
        raise ValueError("G2B measurement covariance 非有限")
    return covariance


def _covariance_structure(covariance: Any) -> dict[str, Any]:
    value = np.asarray(covariance, dtype=np.float64)
    finite = bool(value.shape == (3, 3) and np.isfinite(value).all())
    symmetric = bool(finite and np.allclose(value, value.T, rtol=0.0, atol=1e-12))
    eigenvalues = (
        np.linalg.eigvalsh((value + value.T) * 0.5)
        if finite and symmetric
        else np.asarray([], dtype=np.float64)
    )
    psd = bool(eigenvalues.size == 3 and float(eigenvalues.min()) >= -1e-12)
    return {
        "finite": finite,
        "symmetric": symmetric,
        "positive_semidefinite": psd,
        "valid": bool(finite and symmetric and psd),
        "maximum_position_std_m": (
            float(np.sqrt(max(float(eigenvalues.max()), 0.0))) if psd else None
        ),
    }


def fit_covariance_scale(
    scores: list[float],
    *,
    minimum_support_count: int = 30,
    target_coverage: float = 0.95,
    chi_square_threshold: float = 5.991,
) -> dict[str, Any]:
    """执行冻结的 split-conformal order statistic；失败时返回 no-go。"""

    if minimum_support_count != 30 or target_coverage != 0.95:
        raise ValueError("G2B support/target coverage 漂移")
    if chi_square_threshold != 5.991:
        raise ValueError("G2B chi-square threshold 漂移")
    values = np.asarray(scores, dtype=np.float64)
    support = int(values.size)
    k = math.ceil((support + 1) * target_coverage)
    reasons: list[str] = []
    if support < minimum_support_count:
        reasons.append("insufficient_support")
    if k > support:
        reasons.append("order_statistic_out_of_range")
    if not np.isfinite(values).all():
        reasons.append("nonfinite_calibration_score")
    quantile = None if reasons else float(np.sort(values, kind="stable")[k - 1])
    scale = None if quantile is None else float(max(1.0, quantile / chi_square_threshold))
    if quantile is not None and not math.isfinite(quantile):
        reasons.append("nonfinite_quantile")
    if scale is not None and (not math.isfinite(scale) or scale < 1.0):
        reasons.append("invalid_scale")
    return {
        "passed": not reasons,
        "support_count": support,
        "order_statistic_k": k,
        "quantile_score": quantile,
        "scale_factor": scale,
        "sigma_scale_factor": None if scale is None else float(math.sqrt(scale)),
        "minimum_support_count": minimum_support_count,
        "target_coverage": target_coverage,
        "chi_square_threshold": chi_square_threshold,
        "failure_reasons": reasons,
    }


_CAL_PREDICTION_FORBIDDEN_KEYS = {
    "gt_observable",
    "gt_object_position_base_m",
    "gt_projected_normalized_uv",
    "object_mask",
    "goal_mask",
    "world_xy_error_vector_m",
    "mahalanobis_squared",
    "calibration_selected",
    "segmentation",
}


def assert_calibration_prediction_ledger_deployable_only(
    rows: list[dict[str, Any]],
) -> None:
    def walk(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in _CAL_PREDICTION_FORBIDDEN_KEYS or key.startswith("gt_"):
                    raise ValueError(f"G2B CAL prediction ledger 含 privileged field: {path}.{key}")
                walk(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")

    for index, row in enumerate(rows):
        walk(row, f"rows[{index}]")


def _predict_calibration_validation(
    *,
    context: Any,
    dataset: _DeployableWristCalibrationDataset,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Phase A：只消费 deployable val，输出不含 label/GT 的 ledger。"""

    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=int(config["execution"]["batch_size"]),
        shuffle=False,
        num_workers=0,
        collate_fn=_calibration_collate,
        drop_last=False,
        pin_memory=False,
    )
    model = context.model
    model.eval()
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    batch_count = 0
    with torch.inference_mode():
        for raw_batch in loader:
            batch_count += 1
            image = raw_batch["image"].to(torch.device("cuda"))
            state = raw_batch["structured_state"].to(torch.device("cuda"))
            motion = raw_batch["geometric_motion"].to(torch.device("cuda"))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                output = model(image, state, motion)
            decoded = output.decode_for_control(temperature=context.keypoint_temperature)
            predicted_uv = decoded.keypoints.normalized_uv.detach().float().cpu().numpy()
            raw_sigma = decoded.keypoint_sigma_px.detach().float().cpu().numpy()
            if predicted_uv.shape != (len(raw_batch["audit"]), 2, 2) or raw_sigma.shape != (
                len(raw_batch["audit"]),
                2,
                2,
            ):
                raise RuntimeError("G2B CAL model object/goal output shape 漂移")
            for index, audit in enumerate(raw_batch["audit"]):
                transform = validate_se3(
                    audit["base_from_camera_cv"],
                    "base_from_wrist_camera_cv",
                )
                intrinsic = np.asarray(audit["intrinsic_cv"], dtype=np.float64)
                object_uv = np.asarray(predicted_uv[index, 0], dtype=np.float64)
                sigma = np.asarray(raw_sigma[index, 0], dtype=np.float64)
                try:
                    geometry = geometry_conditioning(
                        normalized_uv=object_uv,
                        intrinsic_cv=intrinsic,
                        base_from_camera_cv=transform,
                        image_size_hw=tuple(raw_batch["image"].shape[-2:]),
                        plane_base_z_m=0.02,
                    )
                    predicted_world = np.asarray(
                        geometry["predicted_world_point_base_m"],
                        dtype=np.float64,
                    )
                    covariance = _measurement_covariance(
                        geometry["local_jacobian_xy_m_per_px"],
                        sigma,
                    )
                    geometry_payload: dict[str, Any] = {
                        "valid": True,
                        "predicted_object_position_base_m": predicted_world.tolist(),
                        "raw_measurement_covariance_base_m2": covariance.tolist(),
                        "local_jacobian_xy_m_per_px": geometry["local_jacobian_xy_m_per_px"],
                    }
                except ValueError as error:
                    geometry_payload = {
                        "valid": False,
                        "predicted_object_position_base_m": None,
                        "raw_measurement_covariance_base_m2": None,
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    }
                identity, pose_record = build_stable_camera_calibration_identity(
                    camera_uid="hand_camera",
                    primitive_id=G2B_CAL_PRIMITIVE_ID,
                    intrinsic_cv=intrinsic,
                    actual_base_from_camera_cv=transform,
                    # CAL Phase A 还没有 fitted identity；固定使用 method/config identity。
                    covariance_calibration_identity_sha256=canonical_sha256(
                        {
                            "method": config["calibration"]["method"],
                            "config_sha256": canonical_sha256(config),
                            "stage": "pre-fit-validation-prediction",
                        }
                    ),
                    source_training_camera="hand_camera",
                    target_camera="hand_camera",
                    frame_convention="robot-base-from-opencv-optical-camera/v1",
                )
                rows.append(
                    {
                        "version": E018_P1_G2B_RESULT_VERSION,
                        "phase": "calibration-prediction-before-label/v1",
                        "trajectory_id": audit["trajectory_id"],
                        "scene_id": audit["scene_id"],
                        "split": audit["split"],
                        "timestep": int(audit["timestep"]),
                        "timestamp_s": float(audit["timestamp_s"]),
                        "source_camera": "hand_camera",
                        "primitive_id": G2B_CAL_PRIMITIVE_ID,
                        "image_input_sha256": array_sha256(raw_batch["image"][index].numpy()),
                        "structured_state_sha256": array_sha256(
                            raw_batch["structured_state"][index].numpy()
                        ),
                        "geometric_motion_sha256": array_sha256(
                            raw_batch["geometric_motion"][index].numpy()
                        ),
                        **pose_record,
                        "intrinsic_cv": intrinsic.tolist(),
                        "intrinsic_cv_sha256": array_sha256(intrinsic),
                        "stable_camera_calibration_identity": identity,
                        "stable_camera_calibration_identity_sha256": identity["identity_sha256"],
                        "checkpoint_sha256": context.checkpoint_sha256,
                        "checkpoint_parameter_sha256": (context.checkpoint_parameter_sha256),
                        "model_config_sha256": context.model_config_sha256,
                        "predicted_object_normalized_uv": object_uv.tolist(),
                        "raw_object_sigma_xy_px": sigma.tolist(),
                        "geometry": geometry_payload,
                        "label_root_received": False,
                        "privileged_array_read_count": 0,
                    }
                )
    assert_calibration_prediction_ledger_deployable_only(rows)
    prediction_keys = {(str(row["trajectory_id"]), int(row["timestep"])) for row in rows}
    if len(prediction_keys) != len(rows):
        raise RuntimeError("G2B CAL deployable prediction trajectory/timestep identity 重复")
    stable_identity_count = len(
        {str(row["stable_camera_calibration_identity_sha256"]) for row in rows}
    )
    if stable_identity_count != 1:
        raise RuntimeError("G2B CAL hand_camera static calibration identity 漂移")
    audit = {
        "version": E018_P1_G2B_RESULT_VERSION,
        "phase": "deployable-val-prediction-before-label/v1",
        "prediction_count": len(rows),
        "trajectory_array_file_count": len(dataset.base.entries),
        "label_manifest_received": False,
        "label_root_received": False,
        "privileged_label_array_read_count": 0,
        "model_forward_batch_count": batch_count,
        "model_forward_sample_count": len(rows),
        "unique_trajectory_timestep_count": len(prediction_keys),
        "stable_camera_calibration_identity_count": stable_identity_count,
        "elapsed_s": time.perf_counter() - started,
        "test_trajectory_array_read_count": 0,
        "test_label_array_read_count": 0,
        "live_memory_read_count": 0,
        "live_memory_write_count": 0,
        "actuation_count": 0,
    }
    audit["audit_sha256"] = canonical_sha256(audit)
    return rows, audit


def freeze_calibration_prediction_ledger(
    output_root: str | Path,
    *,
    rows: list[dict[str, Any]],
    config_sha256: str,
) -> dict[str, Any]:
    output = Path(output_root)
    assert_calibration_prediction_ledger_deployable_only(rows)
    ledger = output / "calibration_prediction_ledger.jsonl"
    _atomic_jsonl(ledger, rows)
    ledger_sha = file_sha256(ledger)
    marker = {
        "version": E018_P1_G2B_RESULT_VERSION,
        "status": "frozen-before-validation-label-read",
        "config_sha256": config_sha256,
        "prediction_ledger_sha256": ledger_sha,
        "prediction_count": len(rows),
        "privileged_field_scan_passed": True,
        "validation_label_array_read_count_before_freeze": 0,
        "test_trajectory_array_read_count": 0,
        "test_label_array_read_count": 0,
    }
    marker["freeze_marker_sha256"] = canonical_sha256(marker)
    _atomic_json(output / "calibration_prediction_freeze.json", marker)
    if file_sha256(ledger) != ledger_sha:
        raise RuntimeError("G2B CAL prediction ledger freeze 后漂移")
    return marker


def load_frozen_calibration_prediction_ledger(
    output_root: str | Path,
    *,
    config_sha256: str,
    expected_prediction_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output = Path(output_root)
    marker = _read_json(
        output / "calibration_prediction_freeze.json",
        "G2B CAL prediction freeze marker",
    )
    marker_sha = _require_sha256(
        marker.get("freeze_marker_sha256"),
        "freeze_marker_sha256",
    )
    marker_payload = dict(marker)
    del marker_payload["freeze_marker_sha256"]
    if canonical_sha256(marker_payload) != marker_sha:
        raise RuntimeError("G2B CAL freeze marker internal SHA-256 漂移")
    if (
        marker.get("version") != E018_P1_G2B_RESULT_VERSION
        or marker.get("status") != "frozen-before-validation-label-read"
        or marker.get("config_sha256") != config_sha256
        or marker.get("prediction_count") != expected_prediction_count
        or marker.get("privileged_field_scan_passed") is not True
        or marker.get("validation_label_array_read_count_before_freeze") != 0
        or marker.get("test_trajectory_array_read_count") != 0
        or marker.get("test_label_array_read_count") != 0
    ):
        raise RuntimeError("G2B CAL freeze marker 内容漂移")
    ledger_sha = _require_sha256(
        marker.get("prediction_ledger_sha256"),
        "prediction_ledger_sha256",
    )
    ledger = output / "calibration_prediction_ledger.jsonl"
    if not ledger.is_file() or file_sha256(ledger) != ledger_sha:
        raise RuntimeError("G2B CAL frozen prediction ledger identity 漂移")
    rows = _load_jsonl(ledger, "G2B CAL frozen prediction ledger")
    if len(rows) != expected_prediction_count:
        raise RuntimeError("G2B CAL frozen prediction count 漂移")
    assert_calibration_prediction_ledger_deployable_only(rows)
    if file_sha256(ledger) != ledger_sha:
        raise RuntimeError("G2B CAL frozen ledger 读取期间漂移")
    return rows, marker


def _score_calibration_after_prediction_freeze(
    *,
    config: dict[str, Any],
    deployable_root: str | Path,
    label_root: str | Path,
    output_root: str | Path,
    source_identity_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Phase B：重载 frozen ledger 后才打开 E016 validation labels。"""

    output = Path(output_root)
    expected_count = int(config["calibration_data"]["manifest_sample_count"])
    predictions, marker = load_frozen_calibration_prediction_ledger(
        output,
        config_sha256=canonical_sha256(config),
        expected_prediction_count=expected_count,
    )
    prediction_ledger = output / "calibration_prediction_ledger.jsonl"
    prediction_sha = marker["prediction_ledger_sha256"]

    # 从这里起才允许接触 privileged validation files；split 被固定为 val。
    from robot_vla.precision.e016_evaluation import _validate_non_test_split_files

    labels = Path(label_root)
    validation_data_identity = _validate_non_test_split_files(
        deployable_root=Path(deployable_root),
        label_root=labels,
        split="val",
    )
    if validation_data_identity != config["calibration_data"]["validation_files_identity_sha256"]:
        raise RuntimeError("G2B CAL validation source/label file identity 漂移")
    label_entries = load_precision_label_manifest(labels, split="val")
    label_by_trajectory = {entry.trajectory_id: entry for entry in label_entries}
    if len(label_by_trajectory) != config["calibration_data"]["trajectory_count"]:
        raise RuntimeError("G2B CAL validation label trajectory count 漂移")
    store = PrecisionLabelStore(labels, cache_size=32)
    scored: list[dict[str, Any]] = []
    label_files_read: set[str] = set()
    selection_counts = Counter()
    scores: list[float] = []
    provider_covariances: list[np.ndarray] = []
    for prediction in predictions:
        if prediction.get("split") != "val":
            raise RuntimeError("G2B CAL frozen ledger 出现非 val row")
        trajectory_id = str(prediction["trajectory_id"])
        meta = label_by_trajectory.get(trajectory_id)
        if meta is None:
            raise RuntimeError("G2B CAL prediction 缺少对应 validation label")
        arrays = store.get(meta)
        label_files_read.add(meta.file)
        timestep = int(prediction["timestep"])
        if not 0 <= timestep < arrays.num_steps:
            raise RuntimeError("G2B CAL prediction timestep 超出 label 范围")
        object_position = arrays.object_position_base_m[timestep].astype(np.float64)
        if abs(float(object_position[2]) - 0.02) > 1e-5:
            raise RuntimeError("G2B CAL object position 不符合冻结 task plane")
        transform = validate_se3(
            prediction["actual_base_from_camera_cv"],
            "actual_base_from_camera_cv",
        )
        if array_sha256(transform) != prediction["actual_base_from_camera_cv_sha256"]:
            raise RuntimeError("G2B CAL actual camera pose row identity 漂移")
        intrinsic = np.asarray(prediction["intrinsic_cv"], dtype=np.float64)
        if array_sha256(intrinsic) != prediction["intrinsic_cv_sha256"]:
            raise RuntimeError("G2B CAL intrinsic row identity 漂移")
        try:
            gt_uv = project_base_point_to_normalized_uv(
                object_position,
                intrinsic,
                transform,
                arrays.object_mask[timestep].shape,
            )
            projection_valid = True
        except ValueError:
            gt_uv = None
            projection_valid = False
        observability = derive_object_observability(
            object_exists=True,
            projection_valid=projection_valid,
            projected_normalized_uv=gt_uv,
            object_mask=arrays.object_mask[timestep],
            goal_mask=arrays.goal_mask[timestep],
            legacy_visible=bool(arrays.keypoint_visible[timestep, 0]),
            support_radius_px=2,
        )
        geometry = prediction["geometry"]
        geometry_valid = bool(geometry["valid"])
        raw_covariance_value = geometry["raw_measurement_covariance_base_m2"]
        covariance_structure = (
            {
                "finite": False,
                "symmetric": False,
                "positive_semidefinite": False,
                "valid": False,
                "maximum_position_std_m": None,
            }
            if raw_covariance_value is None
            else _covariance_structure(raw_covariance_value)
        )
        selected = bool(
            observability.observable and geometry_valid and covariance_structure["valid"]
        )
        if not observability.observable:
            selection_counts["gt_object_unobservable"] += 1
        if not geometry_valid:
            selection_counts["geometry_invalid"] += 1
        if not covariance_structure["valid"]:
            selection_counts["raw_covariance_invalid"] += 1
            if geometry_valid:
                selection_counts["geometry_valid_raw_covariance_invalid"] += 1
        elif geometry_valid:
            provider_covariances.append(np.asarray(raw_covariance_value, dtype=np.float64))
        predicted_position_value = geometry["predicted_object_position_base_m"]
        predicted_position = (
            None
            if predicted_position_value is None
            else np.asarray(predicted_position_value, dtype=np.float64)
        )
        error_xy = (
            None if predicted_position is None else predicted_position[:2] - object_position[:2]
        )
        mahalanobis = None
        if selected:
            if error_xy is None:
                raise RuntimeError("G2B CAL selected row 缺少 XY error")
            covariance = np.asarray(raw_covariance_value, dtype=np.float64)
            mahalanobis = _mahalanobis_squared_psd(error_xy, covariance[:2, :2])
            scores.append(float(mahalanobis))
            selection_counts["selected"] += 1
            if not math.isfinite(mahalanobis):
                selection_counts["singular_nullspace_nonzero_error"] += 1
        scored.append(
            {
                "version": E018_P1_G2B_RESULT_VERSION,
                "phase": "validation-label-scoring-after-prediction-freeze/v1",
                "prediction_ledger_sha256": prediction_sha,
                "trajectory_id": trajectory_id,
                "scene_id": prediction["scene_id"],
                "split": "val",
                "timestep": timestep,
                "source_camera": "hand_camera",
                "gt_object_position_base_m": object_position.tolist(),
                "gt_projected_normalized_uv": (
                    None if gt_uv is None else gt_uv.astype(float).tolist()
                ),
                "gt_object_observable": bool(observability.observable),
                "observability": observability.to_dict(),
                "geometry_valid": geometry_valid,
                "raw_covariance_structure": covariance_structure,
                "predicted_object_position_base_m": predicted_position_value,
                "world_xy_error_vector_m": (
                    None if error_xy is None else error_xy.astype(float).tolist()
                ),
                "calibration_selected": selected,
                "mahalanobis_squared": mahalanobis,
                "selection_inputs": {
                    "gt_object_observable": bool(observability.observable),
                    "geometry_valid": geometry_valid,
                    "raw_covariance_finite_symmetric_psd": bool(covariance_structure["valid"]),
                },
                "confidence_used_for_selection": False,
                "write_acceptance_used_for_selection": False,
                "prediction_error_magnitude_used_for_selection": False,
                "test_data_read": False,
            }
        )
    if len(scored) != expected_count:
        raise RuntimeError("G2B CAL scoring row count 漂移")
    _atomic_jsonl(output / "calibration_scoring_ledger.jsonl", scored)
    scoring_sha = file_sha256(output / "calibration_scoring_ledger.jsonl")
    fit = fit_covariance_scale(
        scores,
        minimum_support_count=int(config["calibration"]["minimum_support_count"]),
        target_coverage=float(config["calibration"]["target_coverage"]),
        chi_square_threshold=float(config["calibration"]["chi_square_threshold"]),
    )
    gate_reasons = list(fit["failure_reasons"])
    if selection_counts["geometry_valid_raw_covariance_invalid"]:
        gate_reasons.append("raw_provider_covariance_invalid")
    if (
        selection_counts["singular_nullspace_nonzero_error"]
        and "singular_nullspace_nonzero_error" not in gate_reasons
    ):
        gate_reasons.append("singular_nullspace_nonzero_error")

    maximum_calibrated_std: float | None = None
    calibrated_all_valid = False
    empirical_coverage: float | None = None
    calibration: ScalarCovarianceCalibration | None = None
    if not gate_reasons:
        calibration = ScalarCovarianceCalibration(
            scale_factor=float(fit["scale_factor"]),
            support_count=int(fit["support_count"]),
            order_statistic_k=int(fit["order_statistic_k"]),
            quantile_score=float(fit["quantile_score"]),
            scoring_ledger_sha256=scoring_sha,
            config_sha256=canonical_sha256(config),
            validation_data_identity_sha256=validation_data_identity,
            source_identity_sha256=source_identity_sha256,
        )
        calibrated_qualities = [
            _covariance_structure(calibration.calibrate_covariance(value))
            for value in provider_covariances
        ]
        calibrated_all_valid = all(bool(item["valid"]) for item in calibrated_qualities)
        maximum_calibrated_std = max(
            float(item["maximum_position_std_m"])
            for item in calibrated_qualities
            if item["maximum_position_std_m"] is not None
        )
        if not calibrated_all_valid:
            gate_reasons.append("calibrated_covariance_invalid")
        if maximum_calibrated_std > float(
            config["calibration"]["maximum_calibrated_position_std_m"]
        ):
            gate_reasons.append("calibrated_covariance_std_exceeds_20mm")
        empirical_coverage = float(
            np.mean(
                np.asarray(scores, dtype=np.float64)
                <= calibration.scale_factor * float(config["calibration"]["chi_square_threshold"])
            )
        )
    passed = not gate_reasons
    calibration_payload = {
        "version": E018_P1_G2B_RESULT_VERSION,
        "status": "calibration-pass" if passed else "calibration-no-go",
        "gate": E018_P1_G2B_CAL_GATE,
        "gate_passed": passed,
        "calibration": None if calibration is None else calibration.to_dict(),
        "failure_reasons": gate_reasons,
        "calibration_scoring_ledger_sha256": scoring_sha,
        "prediction_ledger_sha256": prediction_sha,
        "selection_predicate": config["calibration"]["selection_predicate"],
    }
    _atomic_json(output / "calibration.json", calibration_payload)
    summary = {
        "version": E018_P1_G2B_RESULT_VERSION,
        "status": "calibration-pass" if passed else "calibration-no-go",
        "gate": E018_P1_G2B_CAL_GATE,
        "gate_evaluated": True,
        "gate_passed": passed,
        "frame_count": len(scored),
        "validation_trajectory_count": len(label_files_read),
        "validation_data_identity_sha256": validation_data_identity,
        "selection_counts": dict(sorted(selection_counts.items())),
        "fit": fit,
        "calibration_identity_sha256": (
            None if calibration is None else calibration.identity_sha256
        ),
        "calibrated_covariance_all_finite_symmetric_psd": calibrated_all_valid,
        "maximum_calibrated_position_std_m": maximum_calibrated_std,
        "maximum_allowed_position_std_m": config["calibration"][
            "maximum_calibrated_position_std_m"
        ],
        "validation_empirical_coverage_after_calibration": empirical_coverage,
        "failure_reasons": gate_reasons,
        "prediction_source": "fsynced-frozen-ledger-reload/v1",
        "prediction_freeze_marker_sha256": marker["freeze_marker_sha256"],
        "prediction_ledger_sha256_before": prediction_sha,
        "prediction_ledger_sha256_after": file_sha256(prediction_ledger),
        "validation_label_array_file_read_count": len(label_files_read),
        "test_trajectory_array_read_count": 0,
        "test_label_array_read_count": 0,
        "model_context_received_by_phase_b": False,
        "provider_forward_count_in_phase_b": 0,
        "live_memory_read_count": 0,
        "live_memory_write_count": 0,
        "runtime_camera_actuation_count": 0,
        "arm_actuation_count": 0,
        "manipulation_progression_count": 0,
        "provider_training_count": 0,
    }
    if summary["prediction_ledger_sha256_after"] != prediction_sha:
        raise RuntimeError("G2B CAL privileged scoring 后 prediction ledger 漂移")
    _atomic_json(output / "calibration_summary.json", summary)
    return summary, calibration_payload, scored


def verify_g2b_calibration_receipt(output_root: str | Path) -> dict[str, Any]:
    root = Path(output_root)
    receipt = _read_json(root / "calibration_receipt.json", "G2B CAL receipt")
    receipt_sha = _require_sha256(receipt.get("receipt_sha256"), "receipt_sha256")
    payload = dict(receipt)
    del payload["receipt_sha256"]
    if canonical_sha256(payload) != receipt_sha:
        raise RuntimeError("G2B CAL receipt internal SHA-256 漂移")
    if (
        receipt.get("version") != E018_P1_G2B_RESULT_VERSION
        or receipt.get("gate") != E018_P1_G2B_CAL_GATE
        or receipt.get("status") not in {"complete-calibration-pass", "complete-calibration-no-go"}
        or not isinstance(receipt.get("gate_passed"), bool)
        or receipt.get("prediction_frozen_before_validation_label") is not True
        or receipt.get("allowed_label_split") != "val"
        or any(
            receipt.get(name) != 0
            for name in (
                "test_trajectory_array_read_count",
                "test_label_array_read_count",
                "live_memory_read_count",
                "live_memory_write_count",
                "runtime_camera_actuation_count",
                "arm_actuation_count",
                "manipulation_progression_count",
                "provider_training_count",
            )
        )
    ):
        raise RuntimeError("G2B CAL receipt scope/status 漂移")
    if receipt["gate_passed"] != (receipt["status"] == "complete-calibration-pass"):
        raise RuntimeError("G2B CAL receipt status/gate 不一致")
    files = receipt.get("files")
    if not isinstance(files, list):
        raise TypeError("G2B CAL receipt files 必须是 list")
    by_path = {str(record.get("path")): record for record in files}
    if len(by_path) != len(files) or set(by_path) != set(_CAL_ARTIFACT_NAMES):
        raise RuntimeError("G2B CAL receipt artifact 集合漂移")
    for relative, record in by_path.items():
        path = root / relative
        if (
            not path.is_file()
            or path.stat().st_size != record.get("size_bytes")
            or file_sha256(path) != record.get("sha256")
        ):
            raise RuntimeError(f"G2B CAL artifact identity 漂移: {relative}")
    summary = _read_json(root / "calibration_summary.json", "G2B CAL summary")
    calibration = _read_json(root / "calibration.json", "G2B calibration")
    if (
        summary.get("gate_passed") != receipt["gate_passed"]
        or calibration.get("gate_passed") != receipt["gate_passed"]
        or summary.get("calibration_identity_sha256") != receipt.get("calibration_identity_sha256")
    ):
        raise RuntimeError("G2B CAL receipt/summary/calibration 绑定漂移")
    return receipt


def load_passed_covariance_calibration(
    output_root: str | Path,
) -> tuple[ScalarCovarianceCalibration, dict[str, Any]]:
    root = Path(output_root)
    receipt = verify_g2b_calibration_receipt(root)
    if receipt["gate_passed"] is not True:
        raise RuntimeError("G2B qualification 禁止使用 calibration-no-go artifact")
    payload = _read_json(root / "calibration.json", "G2B calibration")
    value = payload.get("calibration")
    if not isinstance(value, dict):
        raise TypeError("G2B passed calibration 缺少 calibration payload")
    expected_keys = {
        "scale_factor",
        "support_count",
        "order_statistic_k",
        "quantile_score",
        "scoring_ledger_sha256",
        "config_sha256",
        "validation_data_identity_sha256",
        "source_identity_sha256",
        "alpha",
        "chi_square_threshold",
        "method",
        "sigma_scale_factor",
        "identity_sha256",
    }
    if set(value) != expected_keys:
        raise RuntimeError("G2B calibration payload keys 漂移")
    calibration = ScalarCovarianceCalibration(
        scale_factor=float(value["scale_factor"]),
        support_count=int(value["support_count"]),
        order_statistic_k=int(value["order_statistic_k"]),
        quantile_score=float(value["quantile_score"]),
        scoring_ledger_sha256=str(value["scoring_ledger_sha256"]),
        config_sha256=str(value["config_sha256"]),
        validation_data_identity_sha256=str(value["validation_data_identity_sha256"]),
        source_identity_sha256=str(value["source_identity_sha256"]),
        alpha=float(value["alpha"]),
        chi_square_threshold=float(value["chi_square_threshold"]),
        method=str(value["method"]),
    )
    if (
        calibration.identity_sha256 != value["identity_sha256"]
        or calibration.sigma_scale_factor != value["sigma_scale_factor"]
        or receipt["calibration_identity_sha256"] != calibration.identity_sha256
    ):
        raise RuntimeError("G2B calibration semantic identity 漂移")
    return calibration, receipt


def run_e018_p1_g2b_calibration(
    *,
    config_path: str | Path,
    parent_g2a_config_path: str | Path,
    parent_g2a_receipt_path: str | Path,
    e016_config_path: str | Path,
    e013_deployable_root: str | Path,
    e016_fresh_deployable_root: str | Path,
    e016_fresh_label_root: str | Path,
    training_output: str | Path,
    repository_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """运行一次 G2B-CAL；算法 no-go 也会冻结完成 receipt。"""

    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"G2B CAL output 已存在: {output}")
    config = load_e018_p1_g2b_config(
        config_path,
        parent_g2a_config_path=parent_g2a_config_path,
    )
    config_sha = canonical_sha256(config)
    parent_g2a = _read_json(Path(parent_g2a_config_path), "parent G2A config")
    repository = Path(repository_root)
    source_identity = _source_identity(
        repository,
        source_parent_git_commit=config["parents"]["source_parent_git_commit"],
    )
    if config["execution"]["require_clean_worktree"] and not source_identity["worktree_clean"]:
        raise RuntimeError("G2B CAL 要求 clean worktree 与精确 commit identity")
    if not source_identity["source_parent_is_ancestor"]:
        raise RuntimeError("G2B source commit 不基于冻结 G2A parent")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    _atomic_json(
        output / "run_state.json",
        {
            "version": E018_P1_G2B_RESULT_VERSION,
            "status": "in-progress-calibration-phase-a",
            "gate": E018_P1_G2B_CAL_GATE,
            "config_sha256": config_sha,
            "validation_label_array_read_count": 0,
            "test_trajectory_array_read_count": 0,
            "test_label_array_read_count": 0,
        },
    )
    try:
        _atomic_json(output / "config_snapshot.json", config)
        _atomic_json(output / "source_identity.json", source_identity)
        parent_receipt = _verify_g2a_parent_receipt(
            Path(parent_g2a_receipt_path),
            config=config,
        )
        _atomic_json(
            output / "parent_g2a_receipt_binding.json",
            {
                "raw_sha256": file_sha256(parent_g2a_receipt_path),
                "internal_sha256": parent_receipt["receipt_sha256"],
                "gate_passed": parent_receipt["gate_passed"],
                "config_sha256": parent_receipt["config_sha256"],
            },
        )
        manifest_audit = audit_g2b_calibration_manifests(
            config=config,
            parent_g2a_config=parent_g2a,
            deployable_root=e016_fresh_deployable_root,
            label_root=e016_fresh_label_root,
        )
        _atomic_json(output / "manifest_audit.json", manifest_audit)
        context = _load_model_context(
            config=parent_g2a,
            e016_config_path=Path(e016_config_path),
            training_output=Path(training_output),
            stats_root=Path(e013_deployable_root),
        )
        data = config["calibration_data"]
        if (
            context.proprio_stats_sha256 != data["proprio_stats_sha256"]
            or context.proprio_normalizer_sha256 != data["proprio_normalizer_sha256"]
            or context.finger_force_stats_sha256 != data["finger_force_stats_sha256"]
            or context.finger_force_normalizer_sha256 != data["finger_force_normalizer_sha256"]
        ):
            raise RuntimeError("G2B CAL provider normalizer identity 漂移")
        dataset = _DeployableWristCalibrationDataset(
            e016_fresh_deployable_root,
            spec=context.spec,
            proprio_normalizer=context.proprio_normalizer,
            finger_force_normalizer=context.finger_force_normalizer,
        )
        if len(dataset) != data["manifest_sample_count"]:
            raise RuntimeError("G2B CAL deployable valid sample count 与 manifest identity 漂移")
        predictions, inference_audit = _predict_calibration_validation(
            context=context,
            dataset=dataset,
            config=config,
        )
        _atomic_json(
            output / "calibration_inference_audit.json",
            inference_audit,
        )
        freeze = freeze_calibration_prediction_ledger(
            output,
            rows=predictions,
            config_sha256=config_sha,
        )
        # Phase B 不接收 dataset/model/provider context，只从冻结 ledger 恢复预测。
        del predictions
        del dataset
        del context
        _atomic_json(
            output / "run_state.json",
            {
                "version": E018_P1_G2B_RESULT_VERSION,
                "status": "in-progress-calibration-phase-b",
                "prediction_ledger_sha256": freeze["prediction_ledger_sha256"],
                "prediction_freeze_marker_sha256": freeze["freeze_marker_sha256"],
                "validation_label_array_read_count_before_freeze": 0,
                "test_trajectory_array_read_count": 0,
                "test_label_array_read_count": 0,
            },
        )
        summary, calibration_payload, _ = _score_calibration_after_prediction_freeze(
            config=config,
            deployable_root=e016_fresh_deployable_root,
            label_root=e016_fresh_label_root,
            output_root=output,
            source_identity_sha256=source_identity["identity_sha256"],
        )
        passed = bool(summary["gate_passed"])
        receipt = {
            "version": E018_P1_G2B_RESULT_VERSION,
            "status": ("complete-calibration-pass" if passed else "complete-calibration-no-go"),
            "gate": E018_P1_G2B_CAL_GATE,
            "gate_evaluated": True,
            "gate_passed": passed,
            "config_sha256": config_sha,
            "source_identity_sha256": source_identity["identity_sha256"],
            "parent_g2a_receipt_internal_sha256": parent_receipt["receipt_sha256"],
            "validation_data_identity_sha256": summary["validation_data_identity_sha256"],
            "prediction_ledger_sha256": freeze["prediction_ledger_sha256"],
            "prediction_frozen_before_validation_label": True,
            "allowed_label_split": "val",
            "calibration_identity_sha256": summary["calibration_identity_sha256"],
            "failure_reasons": summary["failure_reasons"],
            "test_trajectory_array_read_count": 0,
            "test_label_array_read_count": 0,
            "live_memory_read_count": 0,
            "live_memory_write_count": 0,
            "runtime_camera_actuation_count": 0,
            "arm_actuation_count": 0,
            "manipulation_progression_count": 0,
            "provider_training_count": 0,
            "files": _artifact_hashes(output, list(_CAL_ARTIFACT_NAMES)),
        }
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        _atomic_json(output / "calibration_receipt.json", receipt)
        receipt = verify_g2b_calibration_receipt(output)
        _atomic_json(
            output / "run_state.json",
            {
                "version": E018_P1_G2B_RESULT_VERSION,
                "status": "complete",
                "summary_status": summary["status"],
                "receipt_sha256": receipt["receipt_sha256"],
                "receipt_raw_sha256": file_sha256(output / "calibration_receipt.json"),
            },
        )
        return {
            "summary": summary,
            "calibration": calibration_payload,
            "receipt": receipt,
        }
    except Exception as error:
        _atomic_json(
            output / "failure.json",
            {
                "version": E018_P1_G2B_RESULT_VERSION,
                "status": "failed-preserved",
                "error_type": type(error).__name__,
                "error_message": str(error),
                "prediction_ledger_exists": (
                    output / "calibration_prediction_ledger.jsonl"
                ).is_file(),
                "test_trajectory_array_read_count": 0,
                "test_label_array_read_count": 0,
                "live_memory_read_count": 0,
                "live_memory_write_count": 0,
                "runtime_camera_actuation_count": 0,
                "arm_actuation_count": 0,
                "manipulation_progression_count": 0,
            },
        )
        _atomic_json(
            output / "run_state.json",
            {
                "version": E018_P1_G2B_RESULT_VERSION,
                "status": "failed-preserved",
                "error_type": type(error).__name__,
            },
        )
        raise


def _stabilize_qualification_captures(
    captures: list[Any],
    *,
    context: Any,
    calibration: ScalarCovarianceCalibration,
    parent_g2a_config: dict[str, Any],
) -> tuple[list[Any], dict[str, Any]]:
    """用静态 camera/calibration/provider provenance 替换 G2A 动态 identity。"""

    native_adapter_sha = canonical_sha256(
        {
            "version": "e018-p1-g2b-native-wrist-input-adapter/v1",
            "source_training_camera": "hand_camera",
            "target_camera": "hand_camera",
            "frame_convention": "robot-base-from-opencv-optical-camera/v1",
        }
    )
    result: list[Any] = []
    stable_calibration_ids: dict[str, set[str]] = defaultdict(set)
    provider_ids: dict[str, set[str]] = defaultdict(set)
    actual_pose_ids: dict[str, set[str]] = defaultdict(set)
    for capture in captures:
        is_native = capture.primitive_id == NATIVE_WRIST_CONTROL_ID
        target_camera = "hand_camera" if is_native else "base_camera"
        expected_camera = target_camera
        if capture.source_camera != expected_camera:
            raise RuntimeError("G2B capture camera/primitive identity 漂移")
        stable_calibration, pose_record = build_stable_camera_calibration_identity(
            camera_uid=capture.source_camera,
            primitive_id=capture.primitive_id,
            intrinsic_cv=capture.intrinsic_cv,
            actual_base_from_camera_cv=capture.base_from_camera_cv,
            covariance_calibration_identity_sha256=calibration.identity_sha256,
            source_training_camera="hand_camera",
            target_camera=target_camera,
            frame_convention="robot-base-from-opencv-optical-camera/v1",
        )
        provider = build_calibrated_provider_identity(
            checkpoint_sha256=context.checkpoint_sha256,
            checkpoint_parameter_sha256=context.checkpoint_parameter_sha256,
            checkpoint_provenance_sha256=context.checkpoint_provenance_sha256,
            model_config_sha256=context.model_config_sha256,
            proprio_stats_sha256=context.proprio_stats_sha256,
            proprio_normalizer_sha256=context.proprio_normalizer_sha256,
            finger_force_stats_sha256=context.finger_force_stats_sha256,
            finger_force_normalizer_sha256=context.finger_force_normalizer_sha256,
            adapter_config_sha256=(
                native_adapter_sha if is_native else context.adapter_config.sha256
            ),
            stable_camera_calibration_identity_sha256=stable_calibration["identity_sha256"],
            covariance_calibration_identity_sha256=calibration.identity_sha256,
            primitive_id=capture.primitive_id,
            geometric_motion_provider_id=parent_g2a_config["adapter"][
                "geometric_motion_provider_id"
            ],
            source_training_camera="hand_camera",
            target_camera=target_camera,
            frame_convention="robot-base-from-opencv-optical-camera/v1",
        )
        stabilized = replace(
            capture,
            calibration_identity_sha256=stable_calibration["identity_sha256"],
            provider_identity=provider,
        )
        result.append(stabilized)
        stable_calibration_ids[capture.primitive_id].add(stable_calibration["identity_sha256"])
        provider_ids[capture.primitive_id].add(provider["identity_sha256"])
        actual_pose_ids[capture.primitive_id].add(pose_record["actual_base_from_camera_cv_sha256"])
    per_primitive = {
        primitive_id: {
            "stable_calibration_identity_count": len(stable_calibration_ids[primitive_id]),
            "provider_identity_count": len(provider_ids[primitive_id]),
            "actual_pose_identity_count": len(actual_pose_ids[primitive_id]),
        }
        for primitive_id in PER_SCENE_CAPTURE_ORDER
    }
    if any(
        item["stable_calibration_identity_count"] != 1 or item["provider_identity_count"] != 1
        for item in per_primitive.values()
    ):
        raise RuntimeError("G2B stable calibration/provider identity 仍随帧漂移")
    audit = {
        "version": E018_P1_G2B_RESULT_VERSION,
        "capture_count": len(result),
        "per_primitive": per_primitive,
        "actual_pose_excluded_from_stable_identity": True,
        "calibration_identity_sha256": calibration.identity_sha256,
    }
    audit["audit_sha256"] = canonical_sha256(audit)
    return result, audit


def _predict_calibrated_qualification_captures(
    captures: list[Any],
    *,
    context: Any,
    calibration: ScalarCovarianceCalibration,
    parent_g2a_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """G2B qualification Phase A；calibrated sigma 同时进入 covariance/evidence。"""

    torch = context.torch
    model = context.model
    batch_size = int(parent_g2a_config["execution"]["batch_size"])
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    batch_count = 0
    with torch.inference_mode():
        for start in range(0, len(captures), batch_size):
            batch_count += 1
            batch_captures = captures[start : start + batch_size]
            image_numpy = np.stack(
                [capture.rgb.transpose(2, 0, 1) for capture in batch_captures]
            ).astype(np.float32)
            image_numpy /= np.float32(255.0)
            state_numpy = np.stack([capture.structured_state for capture in batch_captures])
            motion_numpy = np.stack([capture.geometric_motion for capture in batch_captures])
            image = torch.from_numpy(image_numpy).to(torch.device("cuda"))
            state = torch.from_numpy(state_numpy).to(torch.device("cuda"))
            motion = torch.from_numpy(motion_numpy).to(torch.device("cuda"))
            torch.cuda.synchronize()
            batch_started = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                output = model(image, state, motion)
            decoded = output.decode_for_control(temperature=context.keypoint_temperature)
            torch.cuda.synchronize()
            batch_latency = time.perf_counter() - batch_started
            predicted_uv = decoded.keypoints.normalized_uv.detach().float().cpu().numpy()
            visibility = decoded.visibility_probability.detach().float().cpu().numpy()
            projection = decoded.projection_validity_probability.detach().float().cpu().numpy()
            entropy = decoded.keypoints.normalized_entropy.detach().float().cpu().numpy()
            raw_sigma = decoded.keypoint_sigma_px.detach().float().cpu().numpy()
            mask_probability = torch.sigmoid(output.mask_logits.detach().float()).cpu().numpy()
            if (
                predicted_uv.shape != (len(batch_captures), 2, 2)
                or visibility.shape != (len(batch_captures), 2)
                or projection.shape != (len(batch_captures),)
                or entropy.shape != (len(batch_captures), 2)
                or raw_sigma.shape != (len(batch_captures), 2, 2)
                or mask_probability.shape[:2] != (len(batch_captures), 2)
            ):
                raise RuntimeError("G2B qualification model output shape 漂移")
            for index, capture in enumerate(batch_captures):
                object_uv = predicted_uv[index, 0]
                object_mask_probability = mask_probability_at_normalized_uv(
                    mask_probability[index, 0],
                    object_uv,
                )
                goal_mask_probability = mask_probability_at_normalized_uv(
                    mask_probability[index, 1],
                    object_uv,
                )
                calibrated_sigma = calibration.calibrate_sigma_xy_px(raw_sigma[index, 0])
                try:
                    geometry = geometry_conditioning(
                        normalized_uv=object_uv,
                        intrinsic_cv=capture.intrinsic_cv,
                        base_from_camera_cv=capture.base_from_camera_cv,
                        image_size_hw=tuple(capture.rgb.shape[:2]),
                        plane_base_z_m=float(
                            parent_g2a_config["geometry"]["object_center_plane_base_z_m"]
                        ),
                    )
                    predicted_world = np.asarray(
                        geometry["predicted_world_point_base_m"],
                        dtype=np.float64,
                    )
                    raw_covariance = _measurement_covariance(
                        geometry["local_jacobian_xy_m_per_px"],
                        raw_sigma[index, 0],
                    )
                    calibrated_covariance = _measurement_covariance(
                        geometry["local_jacobian_xy_m_per_px"],
                        calibrated_sigma,
                    )
                    expected_covariance = calibration.calibrate_covariance(raw_covariance)
                    if not np.allclose(
                        calibrated_covariance,
                        expected_covariance,
                        rtol=1e-12,
                        atol=1e-18,
                    ):
                        raise RuntimeError("G2B calibrated sigma/covariance 路径不一致")
                    geometry_valid = True
                    geometry_payload: dict[str, Any] = {
                        "valid": True,
                        "predicted_object_position_base_m": predicted_world.tolist(),
                        "raw_measurement_covariance_base_m2": raw_covariance.tolist(),
                        "measurement_covariance_base_m2": calibrated_covariance.tolist(),
                        "abs_n_dot_unit_ray": geometry["abs_n_dot_unit_ray"],
                        "jacobian_sigma_max_mm_per_px": geometry["jacobian_sigma_max_mm_per_px"],
                    }
                except ValueError as error:
                    geometry_valid = False
                    geometry_payload = {
                        "valid": False,
                        "predicted_object_position_base_m": None,
                        "raw_measurement_covariance_base_m2": None,
                        "measurement_covariance_base_m2": None,
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    }
                evidence, evidence_sigma = build_calibrated_object_write_evidence(
                    calibration=calibration,
                    raw_sigma_xy_px=raw_sigma[index, 0],
                    visibility_probability=float(visibility[index, 0]),
                    projection_validity_probability=float(projection[index]),
                    object_mask_probability=float(object_mask_probability),
                    goal_mask_probability=float(goal_mask_probability),
                    normalized_entropy=float(entropy[index, 0]),
                    geometry_valid=geometry_valid,
                    min_object_mask_probability=float(
                        parent_g2a_config["observability"]["min_object_mask_probability"]
                    ),
                    max_goal_mask_probability=float(
                        parent_g2a_config["observability"]["max_goal_mask_probability"]
                    ),
                )
                if not np.array_equal(evidence_sigma, calibrated_sigma):
                    raise RuntimeError("G2B write evidence 未使用 provider calibrated sigma")
                provider_sha = str(capture.provider_identity["identity_sha256"])
                stable_camera_identity, _ = build_stable_camera_calibration_identity(
                    camera_uid=capture.source_camera,
                    primitive_id=capture.primitive_id,
                    intrinsic_cv=capture.intrinsic_cv,
                    actual_base_from_camera_cv=capture.base_from_camera_cv,
                    covariance_calibration_identity_sha256=(calibration.identity_sha256),
                    source_training_camera="hand_camera",
                    target_camera=(
                        "hand_camera"
                        if capture.primitive_id == NATIVE_WRIST_CONTROL_ID
                        else "base_camera"
                    ),
                    frame_convention="robot-base-from-opencv-optical-camera/v1",
                )
                if stable_camera_identity["identity_sha256"] != capture.calibration_identity_sha256:
                    raise RuntimeError("G2B stable camera identity 重算漂移")
                rows.append(
                    {
                        "version": E018_P1_G2B_RESULT_VERSION,
                        "phase": "prediction-before-gt/v1",
                        "seed": capture.seed,
                        "scene_id": capture.scene_id,
                        "primitive_id": capture.primitive_id,
                        "source_camera": capture.source_camera,
                        "input_digest": capture.input_digest,
                        "rgb_sha256": array_sha256(capture.rgb),
                        "structured_state_sha256": array_sha256(capture.structured_state),
                        "geometric_motion_sha256": array_sha256(capture.geometric_motion),
                        "actual_base_from_camera_cv": (capture.base_from_camera_cv.tolist()),
                        "actual_base_from_camera_cv_sha256": array_sha256(
                            capture.base_from_camera_cv
                        ),
                        "intrinsic_cv": capture.intrinsic_cv.tolist(),
                        "intrinsic_cv_sha256": array_sha256(capture.intrinsic_cv),
                        "calibration_identity_sha256": (capture.calibration_identity_sha256),
                        "stable_camera_calibration_identity": stable_camera_identity,
                        "covariance_calibration_identity_sha256": (calibration.identity_sha256),
                        "provider_identity": capture.provider_identity,
                        "provider_identity_sha256": provider_sha,
                        "camera_pose_ood": capture.camera_pose_ood,
                        "rotation_projection_audit": (capture.rotation_projection_audit),
                        "physical_safety": capture.physical_safety,
                        "predicted_object_normalized_uv": object_uv.astype(float).tolist(),
                        "predicted_goal_normalized_uv": predicted_uv[index, 1]
                        .astype(float)
                        .tolist(),
                        "object_visibility_probability": float(visibility[index, 0]),
                        "goal_visibility_probability": float(visibility[index, 1]),
                        "projection_validity_probability": float(projection[index]),
                        "object_normalized_entropy": float(entropy[index, 0]),
                        "raw_object_sigma_xy_px": raw_sigma[index, 0].astype(float).tolist(),
                        "object_sigma_xy_px": calibrated_sigma.astype(float).tolist(),
                        "object_mask_probability_at_predicted_object": float(
                            object_mask_probability
                        ),
                        "goal_mask_probability_at_predicted_object": float(goal_mask_probability),
                        "predicted_observable": bool(
                            float(visibility[index, 0])
                            >= float(parent_g2a_config["observability"]["visibility_threshold"])
                        ),
                        "geometry": geometry_payload,
                        "write_evidence": evidence.to_dict(),
                        "write_accepted": evidence.accepted(
                            threshold=float(
                                parent_g2a_config["observability"][
                                    "write_score_diagnostic_threshold"
                                ]
                            )
                        ),
                        "calibrated_sigma_reaches_write_evidence": True,
                        "calibrated_covariance_reaches_measurement": True,
                        "batch_latency_s": batch_latency,
                        "batch_size": len(batch_captures),
                        "qualification_only": True,
                        "memory_write_allowed": False,
                        "actuation_allowed": False,
                    }
                )
    assert_prediction_ledger_deployable_only(rows)
    audit = {
        "version": E018_P1_G2B_RESULT_VERSION,
        "prediction_count": len(rows),
        "model_forward_batch_count": batch_count,
        "model_forward_sample_count": len(rows),
        "elapsed_s": time.perf_counter() - started,
        "privileged_field_scan_passed": True,
        "calibrated_sigma_reaches_write_evidence": all(
            row["calibrated_sigma_reaches_write_evidence"] for row in rows
        ),
        "calibrated_covariance_reaches_measurement": all(
            row["calibrated_covariance_reaches_measurement"] for row in rows
        ),
        "gt_read_count_before_prediction_freeze": 0,
        "live_memory_read_count": 0,
        "live_memory_write_count": 0,
        "actuation_count": 0,
        "test_trajectory_array_read_count": 0,
        "test_label_array_read_count": 0,
    }
    audit["audit_sha256"] = canonical_sha256(audit)
    return rows, audit


def freeze_qualification_prediction_ledger(
    output_root: str | Path,
    *,
    rows: list[dict[str, Any]],
    config_sha256: str,
    calibration_identity_sha256: str,
) -> dict[str, Any]:
    output = Path(output_root)
    assert_prediction_ledger_deployable_only(rows)
    ledger = output / "prediction_ledger.jsonl"
    _atomic_jsonl(ledger, rows)
    ledger_sha = file_sha256(ledger)
    marker = {
        "version": E018_P1_G2B_RESULT_VERSION,
        "status": "frozen-before-privileged-gt-read",
        "config_sha256": config_sha256,
        "calibration_identity_sha256": calibration_identity_sha256,
        "prediction_ledger_sha256": ledger_sha,
        "prediction_count": len(rows),
        "privileged_field_scan_passed": True,
        "gt_read_count_before_freeze": 0,
        "test_trajectory_array_read_count": 0,
        "test_label_array_read_count": 0,
    }
    marker["freeze_marker_sha256"] = canonical_sha256(marker)
    _atomic_json(output / "prediction_freeze.json", marker)
    if file_sha256(ledger) != ledger_sha:
        raise RuntimeError("G2B qualification prediction ledger freeze 后漂移")
    return marker


def load_frozen_qualification_prediction_ledger(
    output_root: str | Path,
    *,
    config_sha256: str,
    calibration_identity_sha256: str,
    expected_prediction_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output = Path(output_root)
    marker = _read_json(output / "prediction_freeze.json", "G2B prediction freeze")
    marker_sha = _require_sha256(
        marker.get("freeze_marker_sha256"),
        "freeze_marker_sha256",
    )
    marker_payload = dict(marker)
    del marker_payload["freeze_marker_sha256"]
    if canonical_sha256(marker_payload) != marker_sha:
        raise RuntimeError("G2B qualification freeze marker internal SHA 漂移")
    if (
        marker.get("version") != E018_P1_G2B_RESULT_VERSION
        or marker.get("status") != "frozen-before-privileged-gt-read"
        or marker.get("config_sha256") != config_sha256
        or marker.get("calibration_identity_sha256") != calibration_identity_sha256
        or marker.get("prediction_count") != expected_prediction_count
        or marker.get("privileged_field_scan_passed") is not True
        or marker.get("gt_read_count_before_freeze") != 0
        or marker.get("test_trajectory_array_read_count") != 0
        or marker.get("test_label_array_read_count") != 0
    ):
        raise RuntimeError("G2B qualification freeze marker 内容漂移")
    ledger_sha = _require_sha256(
        marker.get("prediction_ledger_sha256"),
        "prediction_ledger_sha256",
    )
    ledger = output / "prediction_ledger.jsonl"
    if not ledger.is_file() or file_sha256(ledger) != ledger_sha:
        raise RuntimeError("G2B frozen qualification ledger identity 漂移")
    rows = _load_jsonl(ledger, "G2B frozen qualification ledger")
    if len(rows) != expected_prediction_count:
        raise RuntimeError("G2B frozen qualification prediction count 漂移")
    assert_prediction_ledger_deployable_only(rows)
    if file_sha256(ledger) != ledger_sha:
        raise RuntimeError("G2B qualification ledger 读取期间漂移")
    return rows, marker


def _load_qualification_calibration_binding(
    output_root: Path,
) -> ScalarCovarianceCalibration:
    binding = _read_json(
        output_root / "calibration_binding.json",
        "G2B qualification calibration binding",
    )
    binding_sha = _require_sha256(
        binding.get("binding_sha256"),
        "calibration_binding.binding_sha256",
    )
    payload = dict(binding)
    del payload["binding_sha256"]
    if canonical_sha256(payload) != binding_sha:
        raise RuntimeError("G2B calibration binding internal SHA 漂移")
    value = binding.get("calibration")
    if not isinstance(value, dict):
        raise TypeError("G2B calibration binding 缺少 calibration")
    calibration = ScalarCovarianceCalibration(
        scale_factor=float(value["scale_factor"]),
        support_count=int(value["support_count"]),
        order_statistic_k=int(value["order_statistic_k"]),
        quantile_score=float(value["quantile_score"]),
        scoring_ledger_sha256=str(value["scoring_ledger_sha256"]),
        config_sha256=str(value["config_sha256"]),
        validation_data_identity_sha256=str(value["validation_data_identity_sha256"]),
        source_identity_sha256=str(value["source_identity_sha256"]),
        alpha=float(value["alpha"]),
        chi_square_threshold=float(value["chi_square_threshold"]),
        method=str(value["method"]),
    )
    if calibration.identity_sha256 != binding.get(
        "calibration_identity_sha256"
    ) or calibration.identity_sha256 != value.get("identity_sha256"):
        raise RuntimeError("G2B calibration binding semantic identity 漂移")
    return calibration


def _score_calibrated_qualification_prediction(
    prediction: dict[str, Any],
    *,
    calibration: ScalarCovarianceCalibration,
    observation: dict[str, Any],
    camera_uid: str,
    base_env: Any,
    object_actor_id: int,
    goal_actor_id: int,
    object_position_base_m: np.ndarray,
    spec: RobotSpec,
    config: dict[str, Any],
    parent_g0c: dict[str, Any],
    pose_envelope: dict[str, Any],
    anchor_arm_q: np.ndarray,
    anchor_base_from_tcp: np.ndarray,
    prediction_ledger_sha256: str,
) -> dict[str, Any]:
    primitive_id = str(prediction["primitive_id"])
    # G2A replay helper 的 capture-integrity 仍期待旧动态 calibration digest；这里只为
    # 兼容调用构造临时副本，冻结 G2B ledger 本身始终保留稳定 identity。
    legacy_calibration = {
        "version": "e018-p1-g2a-front-provider-qualification-result/v1",
        "camera_uid": camera_uid,
        "primitive_id": primitive_id,
        "frame_convention": "robot-base-from-opencv-optical-camera/v1",
        "actual_base_from_camera_cv_sha256": prediction["actual_base_from_camera_cv_sha256"],
        "intrinsic_cv_sha256": prediction["intrinsic_cv_sha256"],
    }
    compatible_prediction = dict(prediction)
    compatible_prediction["calibration_identity_sha256"] = canonical_sha256(legacy_calibration)
    row = _score_prediction(
        compatible_prediction,
        observation=observation,
        camera_uid=camera_uid,
        base_env=base_env,
        object_actor_id=object_actor_id,
        goal_actor_id=goal_actor_id,
        object_position_base_m=object_position_base_m,
        spec=spec,
        config=config,
        parent_g0c=parent_g0c,
        pose_envelope=pose_envelope,
        anchor_arm_q=anchor_arm_q,
        anchor_base_from_tcp=anchor_base_from_tcp,
        prediction_ledger_sha256=prediction_ledger_sha256,
    )
    stable_identity, _ = build_stable_camera_calibration_identity(
        camera_uid=camera_uid,
        primitive_id=primitive_id,
        intrinsic_cv=np.asarray(prediction["intrinsic_cv"], dtype=np.float64),
        actual_base_from_camera_cv=np.asarray(
            prediction["actual_base_from_camera_cv"],
            dtype=np.float64,
        ),
        covariance_calibration_identity_sha256=calibration.identity_sha256,
        source_training_camera="hand_camera",
        target_camera=("hand_camera" if primitive_id == NATIVE_WRIST_CONTROL_ID else "base_camera"),
        frame_convention="robot-base-from-opencv-optical-camera/v1",
    )
    provider = dict(prediction["provider_identity"])
    provider_sha = provider.pop("identity_sha256", None)
    raw_covariance = prediction["geometry"]["raw_measurement_covariance_base_m2"]
    calibrated_covariance = prediction["geometry"]["measurement_covariance_base_m2"]
    covariance_scale_match = bool(
        raw_covariance is None
        and calibrated_covariance is None
        or raw_covariance is not None
        and calibrated_covariance is not None
        and np.allclose(
            np.asarray(calibrated_covariance, dtype=np.float64),
            calibration.calibrate_covariance(np.asarray(raw_covariance, dtype=np.float64)),
            rtol=1e-12,
            atol=1e-18,
        )
    )
    extra_integrity = {
        "stable_calibration_identity_match": (
            stable_identity == prediction["stable_camera_calibration_identity"]
            and stable_identity["identity_sha256"] == prediction["calibration_identity_sha256"]
        ),
        "provider_identity_internal_hash_match": (
            provider_sha == canonical_sha256(provider)
            and provider_sha == prediction["provider_identity_sha256"]
        ),
        "provider_calibration_binding_match": (
            provider["covariance_calibration_identity_sha256"] == calibration.identity_sha256
        ),
        "covariance_scale_match": covariance_scale_match,
        "calibrated_sigma_reaches_write_evidence": prediction[
            "calibrated_sigma_reaches_write_evidence"
        ]
        is True,
        "calibrated_covariance_reaches_measurement": prediction[
            "calibrated_covariance_reaches_measurement"
        ]
        is True,
    }
    legacy_match = bool(row["capture_integrity"].pop("calibration_identity_match"))
    row["capture_integrity"]["legacy_replay_calibration_match"] = legacy_match
    row["capture_integrity"].update(extra_integrity)
    row["capture_integrity"]["passed"] = all(
        value for key, value in row["capture_integrity"].items() if key != "passed"
    )
    row.update(
        {
            "version": E018_P1_G2B_RESULT_VERSION,
            "calibration_identity_sha256": prediction["calibration_identity_sha256"],
            "covariance_calibration_identity_sha256": calibration.identity_sha256,
            "capture_integrity_passed": bool(row["capture_integrity"]["passed"]),
            "raw_measurement_covariance_base_m2": raw_covariance,
            "covariance_scale_match": covariance_scale_match,
        }
    )
    return row


def _score_qualification_after_prediction_freeze(
    *,
    config: dict[str, Any],
    parent_g0c: dict[str, Any],
    spec: RobotSpec,
    pose_envelope: dict[str, Any],
    seeds: list[int],
    output_root: Path,
    g2b_config_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Phase B：不接收 model/provider context，只重载冻结 prediction/calibration。"""

    import gymnasium as gym
    import sapien
    from mani_skill.utils import sapien_utils

    from robot_vla.precision.e018_p1_viewpoint_screen import (
        _capture_sensor_observation,
        _set_static_camera_pose,
    )
    from robot_vla.sim import register_robot_vla_maniskill_envs

    calibration = _load_qualification_calibration_binding(output_root)
    expected_count = len(seeds) * len(PER_SCENE_CAPTURE_ORDER)
    predictions, marker = load_frozen_qualification_prediction_ledger(
        output_root,
        config_sha256=g2b_config_sha256,
        calibration_identity_sha256=calibration.identity_sha256,
        expected_prediction_count=expected_count,
    )
    ledger = output_root / "prediction_ledger.jsonl"
    ledger_sha = marker["prediction_ledger_sha256"]
    by_key = {(int(row["seed"]), str(row["primitive_id"])): row for row in predictions}
    if len(by_key) != len(predictions):
        raise RuntimeError("G2B qualification prediction seed/viewpoint identity 重复")
    environment = config["environment"]
    viewpoint_by_id = _viewpoint_map(parent_g0c)
    register_robot_vla_maniskill_envs()
    env = gym.make(
        environment["environment_id"],
        obs_mode=environment["obs_mode"],
        control_mode=environment["control_mode"],
        num_envs=environment["num_envs"],
        robot_uids=environment["robot_uid"],
    )
    rows: list[dict[str, Any]] = []
    static_pose_configurations = 0
    try:
        base_env = env.unwrapped
        sensor = base_env._sensors.get(environment["external_camera_uid"])
        if sensor is None:
            raise RuntimeError("G2B Phase B external camera 缺失")
        camera = sensor.camera
        for seed in seeds:
            env.reset(seed=seed)
            qpos = _numpy(base_env.agent.robot.get_qpos())
            anchor_arm_q = np.asarray(qpos[0, :7], dtype=np.float64).copy()
            _, anchor_tcp, _ = _base_transforms(base_env, config=config)
            object_actor_id = int(_numpy(base_env.cube.per_scene_id).reshape(-1)[0])
            goal_actor_id = int(_numpy(base_env.goal_site.per_scene_id).reshape(-1)[0])
            object_position = _base_point(base_env, base_env.cube, config=config)
            if abs(
                float(object_position[2])
                - float(config["geometry"]["object_center_plane_base_z_m"])
            ) > float(config["geometry"]["object_center_plane_tolerance_m"]):
                raise RuntimeError("G2B object center 不符合冻结 task plane")
            native_observation = _capture_sensor_observation(base_env)
            rows.append(
                _score_calibrated_qualification_prediction(
                    by_key[(seed, NATIVE_WRIST_CONTROL_ID)],
                    calibration=calibration,
                    observation=native_observation,
                    camera_uid=environment["wrist_camera_uid"],
                    base_env=base_env,
                    object_actor_id=object_actor_id,
                    goal_actor_id=goal_actor_id,
                    object_position_base_m=object_position,
                    spec=spec,
                    config=config,
                    parent_g0c=parent_g0c,
                    pose_envelope=pose_envelope,
                    anchor_arm_q=anchor_arm_q,
                    anchor_base_from_tcp=anchor_tcp,
                    prediction_ledger_sha256=ledger_sha,
                )
            )
            for primitive_id in (FRONT_HOME_ID, *FRONT_ALTERNATE_IDS):
                viewpoint, orientation = viewpoint_by_id[primitive_id]
                _set_static_camera_pose(
                    camera,
                    viewpoint,
                    orientation,
                    sapien_module=sapien,
                    sapien_utils_module=sapien_utils,
                )
                static_pose_configurations += 1
                observation = _capture_sensor_observation(base_env)
                rows.append(
                    _score_calibrated_qualification_prediction(
                        by_key[(seed, primitive_id)],
                        calibration=calibration,
                        observation=observation,
                        camera_uid=environment["external_camera_uid"],
                        base_env=base_env,
                        object_actor_id=object_actor_id,
                        goal_actor_id=goal_actor_id,
                        object_position_base_m=object_position,
                        spec=spec,
                        config=config,
                        parent_g0c=parent_g0c,
                        pose_envelope=pose_envelope,
                        anchor_arm_q=anchor_arm_q,
                        anchor_base_from_tcp=anchor_tcp,
                        prediction_ledger_sha256=ledger_sha,
                    )
                )
    finally:
        env.close()
    if len(rows) != expected_count:
        raise RuntimeError("G2B Phase B scoring count 漂移")
    if file_sha256(ledger) != ledger_sha:
        raise RuntimeError("G2B privileged scoring 后 prediction ledger hash 漂移")
    audit = {
        "version": E018_P1_G2B_RESULT_VERSION,
        "phase": "privileged-scoring-after-prediction-freeze/v1",
        "prediction_source": "fsynced-frozen-ledger-reload/v1",
        "prediction_freeze_marker_sha256": marker["freeze_marker_sha256"],
        "prediction_ledger_sha256_before": ledger_sha,
        "prediction_ledger_sha256_after": file_sha256(ledger),
        "calibration_source": "fsynced-frozen-calibration-binding-reload/v1",
        "calibration_identity_sha256": calibration.identity_sha256,
        "seed_count": len(seeds),
        "scored_row_count": len(rows),
        "static_qualification_pose_configuration_count": static_pose_configurations,
        "environment_step_count": 0,
        "runtime_dynamic_camera_actuation_count": 0,
        "arm_actuation_count": 0,
        "manipulation_progression_count": 0,
        "segmentation_array_read_count": len(rows),
        "object_pose_read_count": len(seeds),
        "runtime_gt_control_read_count": 0,
        "model_context_received": False,
        "provider_forward_count": 0,
        "live_memory_read_count": 0,
        "live_memory_write_count": 0,
        "test_trajectory_array_read_count": 0,
        "test_label_array_read_count": 0,
        "capture_integrity_passed": all(bool(row["capture_integrity_passed"]) for row in rows),
        "physical_safety_passed": all(bool(row["physical_safety_passed"]) for row in rows),
        "covariance_scale_match": all(bool(row["covariance_scale_match"]) for row in rows),
    }
    audit["audit_sha256"] = canonical_sha256(audit)
    return rows, audit


def verify_g2b_qualification_receipt(output_root: str | Path) -> dict[str, Any]:
    root = Path(output_root)
    receipt = _read_json(root / "receipt.json", "G2B qualification receipt")
    receipt_sha = _require_sha256(receipt.get("receipt_sha256"), "receipt_sha256")
    payload = dict(receipt)
    del payload["receipt_sha256"]
    if canonical_sha256(payload) != receipt_sha:
        raise RuntimeError("G2B qualification receipt internal SHA 漂移")
    if (
        receipt.get("version") != E018_P1_G2B_RESULT_VERSION
        or receipt.get("gate") != E018_P1_G2B_QUALIFICATION_GATE
        or receipt.get("status")
        not in {
            "complete-preflight-no-qualification-claim",
            "complete-development-only",
        }
        or receipt.get("prediction_frozen_before_gt") is not True
        or any(
            receipt.get(name) != 0
            for name in (
                "test_trajectory_array_read_count",
                "test_label_array_read_count",
                "live_memory_read_count",
                "live_memory_write_count",
                "runtime_camera_actuation_count",
                "arm_actuation_count",
                "manipulation_progression_count",
                "provider_training_count",
            )
        )
    ):
        raise RuntimeError("G2B qualification receipt scope/status 漂移")
    if receipt["status"] == "complete-preflight-no-qualification-claim":
        if receipt.get("gate_evaluated") is not False or receipt.get("gate_passed") is not None:
            raise RuntimeError("G2B preflight receipt gate 语义漂移")
    elif receipt.get("gate_evaluated") is not True or not isinstance(
        receipt.get("gate_passed"), bool
    ):
        raise RuntimeError("G2B full qualification receipt gate 语义漂移")
    files = receipt.get("files")
    if not isinstance(files, list):
        raise TypeError("G2B qualification receipt files 必须是 list")
    by_path = {str(record.get("path")): record for record in files}
    if len(by_path) != len(files) or set(by_path) != set(_QUALIFICATION_ARTIFACT_NAMES):
        raise RuntimeError("G2B qualification receipt artifact 集合漂移")
    for relative, record in by_path.items():
        path = root / relative
        if (
            not path.is_file()
            or path.stat().st_size != record.get("size_bytes")
            or file_sha256(path) != record.get("sha256")
        ):
            raise RuntimeError(f"G2B qualification artifact identity 漂移: {relative}")
    binding = _read_json(root / "calibration_binding.json", "calibration binding")
    if binding.get("calibration_identity_sha256") != receipt.get("calibration_identity_sha256"):
        raise RuntimeError("G2B qualification receipt/calibration binding 漂移")
    return receipt


def run_e018_p1_g2b_qualification(
    *,
    config_path: str | Path,
    parent_g2a_config_path: str | Path,
    parent_g2a_receipt_path: str | Path,
    parent_g0c_config_path: str | Path,
    parent_g0c_receipt_path: str | Path,
    calibration_output: str | Path,
    e016_config_path: str | Path,
    e013_deployable_root: str | Path,
    e016_fresh_deployable_root: str | Path,
    training_output: str | Path,
    repository_root: str | Path,
    output_root: str | Path,
    preflight_only: bool,
    decision_exit_go: bool = False,
) -> dict[str, Any]:
    """运行 G2B requalification；full run 需要显式 decision-exit GO。"""

    if not isinstance(preflight_only, bool) or not isinstance(decision_exit_go, bool):
        raise TypeError("preflight_only/decision_exit_go 必须是 bool")
    if not preflight_only and not decision_exit_go:
        raise RuntimeError("G2B 完整 50-seed 运行缺少 decision Agent 出口 GO")
    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"G2B qualification output 已存在: {output}")
    config = load_e018_p1_g2b_config(
        config_path,
        parent_g2a_config_path=parent_g2a_config_path,
    )
    config_sha = canonical_sha256(config)
    parent_g2a = load_e018_p1_g2a_config(
        parent_g2a_config_path,
        parent_g0c_config_path=parent_g0c_config_path,
    )
    from robot_vla.precision.e018_p1_g0c import load_e018_p1_g0c_config

    parent_g0c = load_e018_p1_g0c_config(parent_g0c_config_path)
    source_identity = _source_identity(
        Path(repository_root),
        source_parent_git_commit=config["parents"]["source_parent_git_commit"],
    )
    if config["execution"]["require_clean_worktree"] and not source_identity["worktree_clean"]:
        raise RuntimeError("G2B qualification 要求 clean worktree")
    if not source_identity["source_parent_is_ancestor"]:
        raise RuntimeError("G2B source commit 不基于冻结 G2A parent")
    parent_receipt = _verify_g2a_parent_receipt(
        Path(parent_g2a_receipt_path),
        config=config,
    )
    _verify_g0c_receipt(Path(parent_g0c_receipt_path), config=parent_g2a)
    calibration, calibration_receipt = load_passed_covariance_calibration(calibration_output)
    if calibration.config_sha256 != config_sha:
        raise RuntimeError("G2B qualification calibration 来自不同 config identity")
    if calibration.source_identity_sha256 != source_identity["identity_sha256"]:
        raise RuntimeError("G2B qualification source identity 与 CAL 运行不一致")
    seeds = (
        [int(parent_g2a["sampling"]["seeds"][0])]
        if preflight_only
        else [int(seed) for seed in parent_g2a["sampling"]["seeds"]]
    )
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    _atomic_json(
        output / "run_state.json",
        {
            "version": E018_P1_G2B_RESULT_VERSION,
            "status": "in-progress-preflight" if preflight_only else "in-progress",
            "gate": E018_P1_G2B_QUALIFICATION_GATE,
            "config_sha256": config_sha,
            "calibration_identity_sha256": calibration.identity_sha256,
            "decision_exit_go": decision_exit_go,
            "test_trajectory_array_read_count": 0,
            "test_label_array_read_count": 0,
        },
    )
    try:
        _atomic_json(output / "config_snapshot.json", config)
        _atomic_json(output / "source_identity.json", source_identity)
        _atomic_json(output / "parent_g2a_config_snapshot.json", parent_g2a)
        binding = {
            "version": E018_P1_G2B_RESULT_VERSION,
            "calibration": calibration.to_dict(),
            "calibration_identity_sha256": calibration.identity_sha256,
            "calibration_receipt_raw_sha256": file_sha256(
                Path(calibration_output) / "calibration_receipt.json"
            ),
            "calibration_receipt_internal_sha256": calibration_receipt["receipt_sha256"],
            "calibration_gate_passed": True,
            "config_sha256": config_sha,
        }
        binding["binding_sha256"] = canonical_sha256(binding)
        _atomic_json(output / "calibration_binding.json", binding)
        seed_audit = audit_g2a_seed_disjointness(
            config=parent_g2a,
            e013_deployable_root=e013_deployable_root,
            e016_fresh_deployable_root=e016_fresh_deployable_root,
        )
        _atomic_json(output / "seed_audit.json", seed_audit)
        pose_envelope = build_e013_wrist_pose_envelope(
            config=parent_g2a,
            e013_deployable_root=e013_deployable_root,
        )
        _atomic_json(output / "wrist_pose_envelope.json", pose_envelope)
        context = _load_model_context(
            config=parent_g2a,
            e016_config_path=Path(e016_config_path),
            training_output=Path(training_output),
            stats_root=Path(e013_deployable_root),
        )
        captures, capture_audit = _capture_deployable_phase(
            config=parent_g2a,
            parent_g0c=parent_g0c,
            context=context,
            pose_envelope=pose_envelope,
            seeds=seeds,
        )
        captures, identity_audit = _stabilize_qualification_captures(
            captures,
            context=context,
            calibration=calibration,
            parent_g2a_config=parent_g2a,
        )
        capture_audit = dict(capture_audit)
        capture_audit["version"] = E018_P1_G2B_RESULT_VERSION
        capture_audit["stable_identity_audit"] = identity_audit
        _atomic_json(output / "deployable_capture_audit.json", capture_audit)
        predictions, inference_audit = _predict_calibrated_qualification_captures(
            captures,
            context=context,
            calibration=calibration,
            parent_g2a_config=parent_g2a,
        )
        _atomic_json(output / "inference_audit.json", inference_audit)
        freeze = freeze_qualification_prediction_ledger(
            output,
            rows=predictions,
            config_sha256=config_sha,
            calibration_identity_sha256=calibration.identity_sha256,
        )
        # Phase B 只保留 RobotSpec；model/provider/captures/predictions 均不可传入。
        spec = context.spec
        del captures
        del predictions
        del context
        scored_rows, scoring_audit = _score_qualification_after_prediction_freeze(
            config=parent_g2a,
            parent_g0c=parent_g0c,
            spec=spec,
            pose_envelope=pose_envelope,
            seeds=seeds,
            output_root=output,
            g2b_config_sha256=config_sha,
        )
        _atomic_jsonl(output / "offline_scoring_ledger.jsonl", scored_rows)
        _atomic_json(output / "offline_scoring_audit.json", scoring_audit)
        if preflight_only:
            if not (
                scoring_audit["capture_integrity_passed"]
                and scoring_audit["physical_safety_passed"]
                and scoring_audit["covariance_scale_match"]
            ):
                raise RuntimeError("G2B preflight integrity/safety/calibrated-path 未通过")
            summary = {
                "version": E018_P1_G2B_RESULT_VERSION,
                "status": "preflight-pass-no-qualification-claim",
                "gate": E018_P1_G2B_QUALIFICATION_GATE,
                "gate_evaluated": False,
                "gate_passed": None,
                "config_sha256": config_sha,
                "source_identity_sha256": source_identity["identity_sha256"],
                "calibration_identity_sha256": calibration.identity_sha256,
                "seed_count": len(seeds),
                "sample_count": len(scored_rows),
                "prediction_ledger_sha256": freeze["prediction_ledger_sha256"],
                "capture_integrity_passed": True,
                "physical_safety_passed": True,
                "covariance_scale_match": True,
                "native_wrist_control_passed": None,
                "primary": None,
                "viewpoint_summaries": [],
                "allowed_conclusion": (
                    "G2B calibrated provider/schema/two-phase replay works for one "
                    "frozen seed; no provider qualification claim"
                ),
            }
        else:
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in scored_rows:
                grouped[str(row["primitive_id"])].append(row)
            viewpoint_summaries = []
            for primitive_id in PER_SCENE_CAPTURE_ORDER:
                item = summarize_qualification_rows(
                    grouped[primitive_id],
                    config=parent_g2a,
                )
                item["version"] = E018_P1_G2B_RESULT_VERSION
                viewpoint_summaries.append(item)
            final = finalize_qualification_summaries(viewpoint_summaries)
            if final["status"] == "pass":
                allowed_conclusion = (
                    "at least one frozen front alternate passed development-only "
                    "requalification with the validation-fitted scalar calibration"
                )
            elif final["status"] == "inconclusive_parent_health":
                allowed_conclusion = (
                    "calibrated native wrist control still failed; front results remain "
                    "inconclusive and cannot be attributed to camera-role shift"
                )
            else:
                allowed_conclusion = (
                    "calibrated native control passed but no front alternate passed; "
                    "this rejects the current frozen wrist checkpoint role-substitution "
                    "provider, not active vision in general"
                )
            summary = {
                "version": E018_P1_G2B_RESULT_VERSION,
                "status": final["status"],
                "gate": E018_P1_G2B_QUALIFICATION_GATE,
                "gate_evaluated": True,
                "gate_passed": final["status"] == "pass",
                "config_sha256": config_sha,
                "source_identity_sha256": source_identity["identity_sha256"],
                "calibration_identity_sha256": calibration.identity_sha256,
                "seed_count": len(seeds),
                "sample_count": len(scored_rows),
                "prediction_ledger_sha256": freeze["prediction_ledger_sha256"],
                "native_wrist_control_passed": final["native_wrist_control_passed"],
                "qualified_front_alternate_ids": final["qualified_front_alternate_ids"],
                "primary": final["primary"],
                "viewpoint_summaries": final["summaries"],
                "allowed_conclusion": allowed_conclusion,
                "test_manifest_metadata_read_count": 2,
                "test_trajectory_array_read_count": 0,
                "test_label_array_read_count": 0,
                "live_memory_read_count": 0,
                "live_memory_write_count": 0,
                "runtime_camera_actuation_count": 0,
                "arm_actuation_count": 0,
                "manipulation_progression_count": 0,
                "provider_training_count": 0,
                "environment_step_count": 0,
            }
        _atomic_json(output / "summary.json", summary)
        _atomic_text(output / "report.md", _report_markdown(summary))
        receipt = {
            "version": E018_P1_G2B_RESULT_VERSION,
            "status": (
                "complete-preflight-no-qualification-claim"
                if preflight_only
                else "complete-development-only"
            ),
            "gate": E018_P1_G2B_QUALIFICATION_GATE,
            "gate_evaluated": not preflight_only,
            "gate_passed": None if preflight_only else bool(summary["gate_passed"]),
            "config_sha256": config_sha,
            "source_identity_sha256": source_identity["identity_sha256"],
            "parent_g2a_receipt_internal_sha256": parent_receipt["receipt_sha256"],
            "calibration_receipt_internal_sha256": calibration_receipt["receipt_sha256"],
            "calibration_identity_sha256": calibration.identity_sha256,
            "prediction_ledger_sha256": freeze["prediction_ledger_sha256"],
            "prediction_frozen_before_gt": True,
            "test_split_status": "manifest-metadata-read-arrays-prohibited-unread",
            "test_trajectory_array_read_count": 0,
            "test_label_array_read_count": 0,
            "live_memory_read_count": 0,
            "live_memory_write_count": 0,
            "runtime_camera_actuation_count": 0,
            "arm_actuation_count": 0,
            "manipulation_progression_count": 0,
            "provider_training_count": 0,
            "decision_exit_go_for_full_run": (decision_exit_go if not preflight_only else False),
            "files": _artifact_hashes(output, list(_QUALIFICATION_ARTIFACT_NAMES)),
        }
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        _atomic_json(output / "receipt.json", receipt)
        receipt = verify_g2b_qualification_receipt(output)
        _atomic_json(
            output / "run_state.json",
            {
                "version": E018_P1_G2B_RESULT_VERSION,
                "status": "complete",
                "summary_status": summary["status"],
                "receipt_sha256": receipt["receipt_sha256"],
                "receipt_raw_sha256": file_sha256(output / "receipt.json"),
            },
        )
        return {"summary": summary, "receipt": receipt}
    except Exception as error:
        _atomic_json(
            output / "failure.json",
            {
                "version": E018_P1_G2B_RESULT_VERSION,
                "status": "failed-preserved",
                "error_type": type(error).__name__,
                "error_message": str(error),
                "prediction_ledger_exists": (output / "prediction_ledger.jsonl").is_file(),
                "test_trajectory_array_read_count": 0,
                "test_label_array_read_count": 0,
                "live_memory_read_count": 0,
                "live_memory_write_count": 0,
                "runtime_camera_actuation_count": 0,
                "arm_actuation_count": 0,
                "manipulation_progression_count": 0,
            },
        )
        _atomic_json(
            output / "run_state.json",
            {
                "version": E018_P1_G2B_RESULT_VERSION,
                "status": "failed-preserved",
                "error_type": type(error).__name__,
            },
        )
        raise


__all__ = [
    "E018_P1_G2B_CAL_GATE",
    "E018_P1_G2B_CONFIG_VERSION",
    "E018_P1_G2B_QUALIFICATION_GATE",
    "E018_P1_G2B_RESULT_VERSION",
    "assert_calibration_prediction_ledger_deployable_only",
    "audit_g2b_calibration_manifests",
    "fit_covariance_scale",
    "freeze_calibration_prediction_ledger",
    "freeze_qualification_prediction_ledger",
    "load_e018_p1_g2b_config",
    "load_frozen_calibration_prediction_ledger",
    "load_frozen_qualification_prediction_ledger",
    "load_passed_covariance_calibration",
    "run_e018_p1_g2b_calibration",
    "run_e018_p1_g2b_qualification",
    "verify_g2b_calibration_receipt",
    "verify_g2b_qualification_receipt",
]
