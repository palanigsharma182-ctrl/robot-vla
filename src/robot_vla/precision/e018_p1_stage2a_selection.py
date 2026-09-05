"""E018-P1 Stage 2A 最小信息增益选择（development、no-test、no-actuation）。

本模块把一次真实 PRIMARY 路线的四次 provider 输出先持久化，再从同一
pre-state 对三个冻结 gain 做纯逻辑重放。私有 simulator object label 只能在
对应 prediction ledger 行 fsync 后写入，且 Pass A 不允许重开评分。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from robot_vla.executive.contracts import PhaseId
from robot_vla.precision.active_front_camera import ExternalCameraMotionState
from robot_vla.precision.active_front_memory import (
    ActiveFrontSourceRecheckEvidence,
    ActiveFrontStage2MemoryOrchestrator,
    PendingActiveViewState,
)
from robot_vla.precision.active_front_memory_provider import (
    ActiveFrontScoreComponents,
    ActiveFrontStage2Config,
    ActiveFrontStage2FrameEvidence,
    ActiveFrontStage2ProviderIdentity,
    PassiveBaselineEvidence,
    PassiveHomeScoreEvidence,
    build_stage2_object_memory_config,
    d049_primary_provider_identity,
)
from robot_vla.precision.active_front_reobserve import (
    ActiveFrontReobserveRequest,
    ActiveFrontSafetyEvidence,
    ActiveFrontTriggerReason,
    HomeV2BarrierFrame,
)
from robot_vla.precision.calibrated_front_provider import canonical_sha256
from robot_vla.precision.e018_p1_g2a import file_sha256
from robot_vla.precision.e018_p1_g2c_qualification import (
    _AppendOnlyJsonl,
    _atomic_create_json,
    _atomic_replace_json,
    _validate_qualification_object_label,
    assert_qualification_prediction_deployable_only,
    load_g2c_dynamic_qualification_config,
)
from robot_vla.precision.e018_p1_stage2a import (
    E018_P1_STAGE2A_SELECTION_EXPERIMENT_ID,
    E018_P1_STAGE2A_SELECTION_PREFLIGHT_EXPERIMENT_ID,
    STAGE2A_COLLECT_FRAME_INDICES,
    STAGE2A_PROVIDER_FRAME_INDICES,
    STAGE2A_SELECTION_PREFLIGHT_SEED,
    STAGE2A_SELECTION_SEEDS,
    Stage2AActionHistoryRuntime,
    load_e018_p1_stage2a_config,
)
from robot_vla.precision.object_memory import (
    ExplicitObjectStateMemory,
    ObjectMemoryMode,
    ObjectMemorySafetyContext,
    ObjectStateRequirement,
    resolve_object_state,
)

E018_P1_STAGE2A_SELECTION_CONFIG_VERSION = (
    "e018-p1-stage2a-information-gain-selection-development/v1"
)
E018_P1_STAGE2A_SELECTION_EXECUTION_VERSION = (
    "e018-p1-stage2a-min-information-gain-selection-execution/v1"
)
E018_P1_STAGE2A_SELECTION_RESULT_VERSION = (
    "e018-p1-stage2a-min-information-gain-selection-result/v1"
)
E018_P1_STAGE2A_SELECTION_PREFLIGHT_EXECUTION_VERSION = (
    "e018-p1-stage2a-pass-a-one-route-preflight-execution/v1"
)
STAGE2A_SELECTION_GO = (
    "E018_P1_STAGE2A_MIN_INFORMATION_GAIN_SELECTION_GO_77001_77025_V1"
)
STAGE2A_SELECTION_PREFLIGHT_GO = (
    "E018_P1_STAGE2A_PASS_A_ONE_ROUTE_PREFLIGHT_GO_76891_V1"
)
STAGE2A_SELECTION_GAINS = (0.02, 0.05, 0.10)
STAGE2A_SELECTION_PREDICTION_COUNT = 100
STAGE2A_SELECTION_LABEL_COUNT = 75
STAGE2A_SELECTION_BRANCH_COUNT = 75
STAGE2A_SELECTION_MIN_SUPPORT = 1
STAGE2A_SELECTION_PREFLIGHT_PREDICTION_COUNT = 4
STAGE2A_SELECTION_PREFLIGHT_LABEL_COUNT = 3

_QUALIFICATION_OBJECT_LABEL_KEYS = {
    "gt_object_exists",
    "gt_observable",
    "gt_object_position_base_m",
    "gt_object_projection_valid",
    "gt_object_projected_normalized_uv",
    "gt_object_mask_sha256",
    "gt_object_visible_pixel_count",
    "gt_object_observability",
    "is_grasped",
    "robot_object_contact_force_n",
    "goal_gt_read_count",
    "test_data_read",
}
_SELECTION_PRIVATE_CAPTURE_KEYS = _QUALIFICATION_OBJECT_LABEL_KEYS | {
    "object_linear_speed_m_s",
    "object_angular_speed_rad_s",
    "object_motion_event",
}

_CONFIG_TOP_LEVEL_KEYS = {
    "version",
    "status",
    "experiment",
    "parents",
    "split",
    "gain_selection",
    "oracle",
    "journaling",
    "artifact_layout",
    "phase_accounting",
    "budgets",
    "permissions",
}

_PARENT_REPLAY_EVIDENCE_FILES = {
    "ARTIFACT_EXACT_TREE.txt",
    "ARTIFACT_FILE_SHA256SUMS.txt",
    "ARTIFACT_INPUT_PATH.txt",
    "COMMAND.txt",
    "ENVIRONMENT.txt",
    "EVIDENCE_SHA256SUMS.txt",
    "EXIT_CODE.txt",
    "FINISHED_AT_UTC.txt",
    "GPU_ENVIRONMENT.txt",
    "OUTPUT_ASSERTIONS.txt",
    "SOURCE_GIT_COMMIT.txt",
    "SOURCE_GIT_STATUS.txt",
    "SOURCE_GIT_TREE.txt",
    "SOURCE_PATH.txt",
    "STARTED_AT_UTC.txt",
    "STDERR.log",
    "STDOUT.json",
}
_PARENT_REPLAY_STAGE2A_FILES = {
    "RUN_STARTED.json",
    "camera_pose_ledger.jsonl",
    "execution_freeze.json",
    "execution_receipt.json",
    "provider_output_ledger.jsonl",
    "route_summaries.jsonl",
    "transaction_ledger.jsonl",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_exact_keys(value: Any, expected: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(f"{name} keys 漂移: {actual}")
    return value


def _read_json(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise RuntimeError(f"{name} 必须是单链接 regular file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{name} 必须是 JSON object")
    return value


def _read_jsonl(path: Path, name: str) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise RuntimeError(f"{name} 必须是单链接 regular file")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith("\n"):
                raise RuntimeError(f"{name} 第 {line_number} 行缺少换行终止")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{name} 第 {line_number} 行必须是 object")
            rows.append(value)
    return rows


@dataclass(frozen=True)
class LoadedStage2ASelectionConfig:
    canonical_json: str
    raw_sha256: str
    canonical_sha256: str

    @property
    def payload(self) -> dict[str, Any]:
        return json.loads(self.canonical_json)


def _validate_selection_config(config: dict[str, Any]) -> None:
    _require_exact_keys(config, _CONFIG_TOP_LEVEL_KEYS, "selection config")
    if (
        config["version"] != E018_P1_STAGE2A_SELECTION_CONFIG_VERSION
        or config["status"]
        != "preregistered-development-selection-no-test-no-actuation"
    ):
        raise ValueError("selection config version/status 漂移")
    experiment = _require_exact_keys(
        config["experiment"],
        {
            "id",
            "gate",
            "classification",
            "exact_go_token",
            "rerun_under_same_identity_allowed",
        },
        "selection experiment",
    )
    if experiment != {
        "id": E018_P1_STAGE2A_SELECTION_EXPERIMENT_ID,
        "gate": "D049",
        "classification": "formal-development-selection-no-test-no-actuation/v1",
        "exact_go_token": STAGE2A_SELECTION_GO,
        "rerun_under_same_identity_allowed": False,
    }:
        raise ValueError("selection experiment identity 漂移")
    parents = _require_exact_keys(
        config["parents"],
        {
            "stage2a_config_path",
            "stage2a_config_raw_sha256",
            "stage2a_config_canonical_sha256",
            "stage2a_smoke_source_git_commit",
            "stage2a_smoke_source_identity_sha256",
            "stage2a_smoke_public_verification_raw_sha256",
            "stage2a_smoke_public_verification_sha256",
            "parent_replay_artifact_id",
            "parent_replay_evidence_manifest_raw_sha256",
            "parent_replay_verification_sha256",
            "parent_replay_drive_marker_raw_sha256",
            "parent_replay_drive_marker_internal_sha256",
            "parent_replay_local_verification_raw_sha256",
            "parent_replay_local_verification_internal_sha256",
            "parent_replay_inventory_record_canonical_sha256",
            "parent_replay_replication_state",
            "accepted_artifact_id",
            "artifact_manifest_raw_sha256",
            "artifact_manifest_internal_sha256",
            "completion_marker_raw_sha256",
            "completion_marker_internal_sha256",
            "persistence_verification_raw_sha256",
            "persistence_verification_internal_sha256",
            "local_canonical_verification_raw_sha256",
            "local_canonical_verification_internal_sha256",
            "artifact_file_inventory_sha256",
            "inventory_record_canonical_sha256",
            "replication_state",
        },
        "selection parents",
    )
    if (
        parents["stage2a_config_path"]
        != "configs/e018_p1_stage2a_primary_memory_development_v1.json"
        or parents["stage2a_config_raw_sha256"]
        != "12794f1cc08f45d9dc5acae01c46d5f760f9456f8d73a308ae981a7b65a27512"
        or parents["stage2a_config_canonical_sha256"]
        != "b9f33d39c668bb204754140c501996a13af20672ced19ec669503803bb9eb767"
        or parents["stage2a_smoke_source_git_commit"]
        != "1769eeaf0edfed7dacb3e4c2c1cecada24274e13"
        or parents["accepted_artifact_id"]
        != "stage2a-integration-smoke-1769eea-20260906-v1"
        or parents["parent_replay_artifact_id"]
        != "e018-stage2a-parent-replay-1769eea-20260906-v1"
        or parents["parent_replay_replication_state"] != "REPLICATED"
        or parents["replication_state"] != "REPLICATED"
        or any(
            not _is_sha256(parents[name])
            for name in (
                "stage2a_config_raw_sha256",
                "stage2a_config_canonical_sha256",
                "stage2a_smoke_source_identity_sha256",
                "stage2a_smoke_public_verification_raw_sha256",
                "stage2a_smoke_public_verification_sha256",
                "parent_replay_evidence_manifest_raw_sha256",
                "parent_replay_verification_sha256",
                "parent_replay_drive_marker_raw_sha256",
                "parent_replay_drive_marker_internal_sha256",
                "parent_replay_local_verification_raw_sha256",
                "parent_replay_local_verification_internal_sha256",
                "parent_replay_inventory_record_canonical_sha256",
                "artifact_manifest_raw_sha256",
                "artifact_manifest_internal_sha256",
                "completion_marker_raw_sha256",
                "completion_marker_internal_sha256",
                "persistence_verification_raw_sha256",
                "persistence_verification_internal_sha256",
                "local_canonical_verification_raw_sha256",
                "local_canonical_verification_internal_sha256",
                "artifact_file_inventory_sha256",
                "inventory_record_canonical_sha256",
            )
        )
    ):
        raise ValueError("selection parent identity 漂移")
    split = _require_exact_keys(
        config["split"],
        {
            "seeds",
            "seed_count",
            "route_count",
            "camera_frames_per_route",
            "provider_frames_per_route",
            "private_label_frames_per_route",
            "provider_prediction_count",
            "private_label_count",
            "gain_branch_count",
            "execution_order",
            "test_once",
        },
        "selection split",
    )
    if split != {
        "seeds": [77001, 77025],
        "seed_count": 25,
        "route_count": 25,
        "camera_frames_per_route": 92,
        "provider_frames_per_route": list(STAGE2A_PROVIDER_FRAME_INDICES),
        "private_label_frames_per_route": list(STAGE2A_COLLECT_FRAME_INDICES),
        "provider_prediction_count": STAGE2A_SELECTION_PREDICTION_COUNT,
        "private_label_count": STAGE2A_SELECTION_LABEL_COUNT,
        "gain_branch_count": STAGE2A_SELECTION_BRANCH_COUNT,
        "execution_order": "ascending-seed-then-fixed-gain-order/v1",
        "test_once": "selection-split-consume-once-no-rerun-same-identity/v1",
    }:
        raise ValueError("selection split/order/count 漂移")
    gain = config["gain_selection"]
    if (
        gain.get("candidates_in_fixed_order") != list(STAGE2A_SELECTION_GAINS)
        or gain.get("information_gain_comparison_tolerance") != 0.0
        or gain.get("branch_initial_state")
        != "fresh-memory-controller-action-state-from-identical-pre-state/v1"
        or gain.get("provider_forward_reuse")
        != "same-four-frozen-provider-outputs-no-reforward/v1"
        or gain.get("tie_break")
        != "maximize-integer-recovered-count-then-larger-gain/v1"
        or any(
            gain.get(name) != 0
            for name in (
                "unsafe_count_max",
                "catastrophic_count_max",
                "false_recovery_count_max",
                "protocol_violation_count_max",
            )
        )
        or gain.get("minimum_common_denominator_routes")
        != STAGE2A_SELECTION_MIN_SUPPORT
        or gain.get("zero_denominator_selection") is not None
        or gain.get("null_selection_policy")
        != "no-evaluation-config-continue-stage2b/v1"
    ):
        raise ValueError("selection gain rule 漂移")
    oracle = config["oracle"]
    if (
        oracle.get("common_denominator")
        != "route-protocol-safety-valid-and-all-three-primary-object-exists-own-mask-observable-no-contact-no-motion/v1"
        or oracle.get("object_observability") != "d048-own-mask-not-goal-mask/v1"
        or oracle.get("contact_unsafe_if_strictly_greater_n") != 0.01
        or oracle.get("motion_predicate_version") != "pick-and-place-predicates/v1"
        or oracle.get("motion_if_linear_strictly_greater_m_s") != 0.01
        or oracle.get("motion_if_angular_strictly_greater_rad_s") != 0.5
        or oracle.get("recovered_xyz_error_max_m") != 0.005
        or oracle.get("catastrophic_xyz_error_strictly_greater_m") != 0.02
    ):
        raise ValueError("selection oracle/threshold 漂移")
    budgets = config["budgets"]
    if budgets != {
        "gpu_wall_seconds_max": 1800,
        "combined_artifact_bytes_max": 2147483648,
        "stage2_cumulative_gpu_wall_seconds_max": 7200,
        "stage2_cumulative_artifact_bytes_max": 8589934592,
    }:
        raise ValueError("selection budget 漂移")
    permissions = config["permissions"]
    if (
        permissions.get("isolated_maniskill") is not True
        or permissions.get("object_labels_after_prediction_fsync_only") is not True
        or permissions.get("effect_claim_allowed") is not False
        or any(
            permissions.get(name) != 0
            for name in (
                "fresh_test_reads",
                "runtime_object_gt_reads",
                "goal_gt_reads",
                "checkpoint_writes",
                "wrist_provider_forwards",
                "arm_tcp_actuation",
                "gripper_close",
                "canonical_runtime_mutation",
            )
        )
    ):
        raise ValueError("selection permission 漂移")
    journaling = config["journaling"]
    if (
        journaling.get("prediction_before_label") is not True
        or journaling.get("private_labels_write_only_during_pass_a") is not True
        or journaling.get("private_labels_read_only_during_new_process_pass_b")
        is not True
        or journaling.get("pass_b_new_process_exact_once") is not True
        or journaling.get("durable_consumption_marker_location")
        != "private_labels/SCORING_CONSUMED.json"
        or journaling.get("public_verifier_accepts_private_root") is not False
        or journaling.get("result_verifier_accepts_model_or_private_root") is not False
    ):
        raise ValueError("selection journal phase boundary 漂移")
    layout = config["artifact_layout"]
    if (
        layout.get("exact_top_level_directories")
        != ["public_execution", "private_labels", "result"]
        or layout.get("public_execution_directories") != {"prediction_commits": 100}
        or layout.get("private_label_files")
        != ["capture_state.json", "SCORING_CONSUMED.json"]
        or layout.get("private_label_directories") != {"label_commits": 75}
        or layout.get("ledger_row_counts")
        != {
            "camera_pose_ledger.jsonl": 2300,
            "provider_output_ledger.jsonl": 100,
            "route_evidence_ledger.jsonl": 25,
            "gain_branch_ledger.jsonl": 75,
            "scored_gain_branches.jsonl": 75,
        }
        or len(layout.get("public_execution_files", [])) != 12
        or len(layout.get("result_files", [])) != 4
    ):
        raise ValueError("selection exact artifact layout 漂移")
    if config["phase_accounting"] != {
        "pass_a_provider_prediction_count": 100,
        "pass_a_privileged_capture_count": 75,
        "pass_a_private_label_open_count": 0,
        "pass_a_runtime_object_gt_reads": 0,
        "pass_b_label_open_count": 75,
        "pass_b_provider_forward_count": 0,
        "pass_b_checkpoint_load_count": 0,
    }:
        raise ValueError("selection Pass A/B accounting 漂移")


def load_e018_p1_stage2a_selection_config(
    path: str | Path,
) -> LoadedStage2ASelectionConfig:
    config_path = Path(path)
    raw = config_path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError("selection config 必须是 JSON object")
    _validate_selection_config(value)
    canonical = _canonical_json(value)
    return LoadedStage2ASelectionConfig(
        canonical_json=canonical,
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        canonical_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )


def _verify_internal_digest(
    value: Mapping[str, Any],
    *,
    digest_key: str,
    name: str,
) -> str:
    unsigned = dict(value)
    stored = unsigned.pop(digest_key, None)
    if not _is_sha256(stored) or canonical_sha256(unsigned) != stored:
        raise RuntimeError(f"{name} internal digest 漂移")
    return stored


def _verify_stage2a_replica_manifest(
    artifact_root: Path,
    *,
    expected_raw_sha256: str,
    expected_internal_sha256: str,
    expected_inventory_sha256: str,
) -> dict[str, Any]:
    if (
        not artifact_root.is_dir()
        or artifact_root.is_symlink()
    ):
        raise RuntimeError("Stage 2A replica root 必须是真实目录")
    manifest_path = artifact_root / "ARTIFACT_MANIFEST.json"
    if file_sha256(manifest_path) != expected_raw_sha256:
        raise RuntimeError("Stage 2A parent manifest raw SHA 漂移")
    manifest = _read_json(manifest_path, "Stage 2A parent manifest")
    internal = _verify_internal_digest(
        manifest,
        digest_key="manifest_sha256",
        name="Stage 2A parent manifest",
    )
    inventory = manifest.get("artifact_inventory")
    if (
        internal != expected_internal_sha256
        or not isinstance(inventory, list)
        or len(inventory) != 58
        or canonical_sha256(inventory) != expected_inventory_sha256
        or manifest.get("artifact_inventory_sha256") != expected_inventory_sha256
        or manifest.get("payload_file_count") != len(inventory)
    ):
        raise RuntimeError("Stage 2A parent manifest identity/count 漂移")
    expected_paths: set[str] = set()
    total_bytes = 0
    for index, row in enumerate(inventory):
        item = _require_exact_keys(
            row,
            {"relative_path", "raw_sha256", "size_bytes"},
            f"Stage 2A manifest inventory[{index}]",
        )
        relative = item["relative_path"]
        if (
            not isinstance(relative, str)
            or not relative
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or relative in expected_paths
            or not _is_sha256(item["raw_sha256"])
            or type(item["size_bytes"]) is not int
            or item["size_bytes"] < 0
        ):
            raise RuntimeError("Stage 2A manifest inventory row 非法")
        path = artifact_root / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_nlink != 1
            or path.stat().st_size != item["size_bytes"]
            or file_sha256(path) != item["raw_sha256"]
        ):
            raise RuntimeError(f"Stage 2A replica file/hash 漂移: {relative}")
        expected_paths.add(relative)
        total_bytes += item["size_bytes"]
    expected_directories: set[str] = set()
    for relative in expected_paths:
        parent = Path(relative).parent
        while parent != Path("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    actual_paths: set[str] = set()
    actual_directories: set[str] = set()
    for path in artifact_root.rglob("*"):
        relative = path.relative_to(artifact_root).as_posix()
        if path.is_symlink():
            raise RuntimeError(f"Stage 2A replica 禁止 symlink: {relative}")
        if path.is_dir():
            actual_directories.add(relative)
        elif path.is_file():
            if path.name not in {
                "ARTIFACT_MANIFEST.json",
                "DRIVE_BACKUP_COMPLETE.json",
            }:
                actual_paths.add(relative)
        else:
            raise RuntimeError(f"Stage 2A replica 禁止 special file: {relative}")
    if (
        actual_paths != expected_paths
        or actual_directories != expected_directories
        or total_bytes != manifest.get("payload_bytes")
    ):
        raise RuntimeError("Stage 2A replica exact file/directory tree/accounting 漂移")
    return manifest


def _parse_sha256_manifest(
    path: Path,
    *,
    expected_names: set[str],
    name: str,
) -> dict[str, str]:
    rows: dict[str, str] = {}
    raw = path.read_text(encoding="utf-8")
    if not raw.endswith("\n"):
        raise RuntimeError(f"{name} 缺少换行终止")
    for line_number, line in enumerate(raw.splitlines(), start=1):
        match = re.fullmatch(r"([0-9a-f]{64})  (/.+)", line)
        if match is None:
            raise RuntimeError(f"{name} 第 {line_number} 行格式漂移")
        digest, absolute = match.groups()
        basename = Path(absolute).name
        if basename in rows or basename not in expected_names:
            raise RuntimeError(f"{name} 文件名重复或不在冻结集合: {basename}")
        rows[basename] = digest
    if set(rows) != expected_names:
        raise RuntimeError(f"{name} 文件集合漂移")
    return rows


def verify_stage2a_parent_replay_evidence(
    *,
    replay_evidence_root: str | Path,
    stage2a_artifact_root: str | Path,
    expected_evidence_manifest_raw_sha256: str,
) -> dict[str, Any]:
    """验证原执行环境的 exact Stage2A verifier 重放证据。

    该证据只证明冻结 ``1769eea`` verifier 在原 RTX 6000 Ada 数值环境
    对 accepted public artifact 重得同一摘要；不会把 worker 私有路径复制到
    返回 receipt，也不会替代当前 source 对 manifest/replica 的机械检查。
    """

    root = Path(replay_evidence_root)
    artifact_root = Path(stage2a_artifact_root)
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("Stage 2A parent replay evidence root 必须是真实目录")
    actual_names: set[str] = set()
    total_bytes = 0
    for path in root.iterdir():
        if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
            raise RuntimeError("Stage 2A parent replay evidence 禁止目录/链接/特殊文件")
        actual_names.add(path.name)
        total_bytes += path.stat().st_size
    if actual_names != _PARENT_REPLAY_EVIDENCE_FILES:
        raise RuntimeError("Stage 2A parent replay evidence exact tree 漂移")

    evidence_manifest_path = root / "EVIDENCE_SHA256SUMS.txt"
    if (
        not _is_sha256(expected_evidence_manifest_raw_sha256)
        or file_sha256(evidence_manifest_path)
        != expected_evidence_manifest_raw_sha256
    ):
        raise RuntimeError("Stage 2A parent replay evidence manifest raw SHA 漂移")
    evidence_hashes = _parse_sha256_manifest(
        evidence_manifest_path,
        expected_names=_PARENT_REPLAY_EVIDENCE_FILES
        - {"EVIDENCE_SHA256SUMS.txt"},
        name="Stage 2A parent replay evidence manifest",
    )
    for basename, digest in evidence_hashes.items():
        if file_sha256(root / basename) != digest:
            raise RuntimeError(f"Stage 2A parent replay evidence 文件 SHA 漂移: {basename}")

    source_commit = (root / "SOURCE_GIT_COMMIT.txt").read_text(encoding="utf-8")
    source_tree = (root / "SOURCE_GIT_TREE.txt").read_text(encoding="utf-8")
    source_status = (root / "SOURCE_GIT_STATUS.txt").read_bytes()
    environment = (root / "ENVIRONMENT.txt").read_text(encoding="utf-8")
    gpu_environment = (root / "GPU_ENVIRONMENT.txt").read_text(encoding="utf-8")
    artifact_input_path = (root / "ARTIFACT_INPUT_PATH.txt").read_text(
        encoding="utf-8"
    )
    command = (root / "COMMAND.txt").read_text(encoding="utf-8")
    exact_tree = (root / "ARTIFACT_EXACT_TREE.txt").read_text(encoding="utf-8")
    exit_code = (root / "EXIT_CODE.txt").read_text(encoding="utf-8")
    stderr = (root / "STDERR.log").read_bytes()
    assertions = (root / "OUTPUT_ASSERTIONS.txt").read_text(encoding="utf-8")
    if (
        source_commit != "1769eeaf0edfed7dacb3e4c2c1cecada24274e13\n"
        or source_tree != "41933676c6173704802515d1f169b90298b0fa8b\n"
        or source_status != b""
        or not environment.startswith(
            "python=3.10.12\n"
            "python_executable=/opt/robot-vla/env/bin/python\n"
            "numpy=1.26.4\n"
        )
        or "name: openblas64\n" not in environment
        or "version: 0.3.23.dev\n" not in environment
        or gpu_environment
        != "NVIDIA RTX 6000 Ada Generation, 49140 MiB, 580.126.09\n"
        or not artifact_input_path.endswith("/public_execution\n")
        or "stage2a-integration-smoke-1769eea-20260906-v1" not in artifact_input_path
        or exit_code != "0\n"
        or stderr != b""
        or assertions
        != (
            "verified=true\n"
            "verification_sha256="
            "c85fafaa91a9e9f40fb019c854830ff45602a0e94c5e670815432604e9193705\n"
        )
        or "robot_vla.cli.run_e018_p1_stage2a verify" not in command
        or "--output <drive-restored-canonical>/public_execution" not in command
        or "--expected-source-git-commit 1769eeaf0edfed7dacb3e4c2c1cecada24274e13" not in command
        or "--expected-source-identity-sha256 d4d36c9b8a584b72c6c1a3fed79e854043f1bfcaac8cd1a2c056e8b7c18f24d8" not in command
    ):
        raise RuntimeError("Stage 2A parent replay source/env/command/exit 语义漂移")
    expected_tree = "".join(
        f"f {name}\n" for name in sorted(_PARENT_REPLAY_STAGE2A_FILES)
    )
    if exact_tree != expected_tree:
        raise RuntimeError("Stage 2A parent replay artifact exact tree 漂移")

    artifact_hashes = _parse_sha256_manifest(
        root / "ARTIFACT_FILE_SHA256SUMS.txt",
        expected_names=_PARENT_REPLAY_STAGE2A_FILES,
        name="Stage 2A parent replay artifact file manifest",
    )
    public_execution = artifact_root / "public_execution"
    for basename, digest in artifact_hashes.items():
        path = public_execution / basename
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_nlink != 1
            or file_sha256(path) != digest
        ):
            raise RuntimeError(
                f"Stage 2A parent replay input 未绑定 canonical replica: {basename}"
            )

    stdout = _read_json(root / "STDOUT.json", "Stage 2A parent replay stdout")
    stdout_unsigned = dict(stdout)
    stdout_internal = stdout_unsigned.pop("verification_sha256", None)
    counts = stdout.get("counts")
    expected_zero_counts = {
        "arm_motion_command_count",
        "checkpoint_writes",
        "fresh_test_reads",
        "goal_gt_reads",
        "gripper_close_command_count",
        "offline_label_reads",
        "runtime_object_gt_reads",
        "wrist_provider_forward_count",
    }
    if (
        stdout_internal
        != "c85fafaa91a9e9f40fb019c854830ff45602a0e94c5e670815432604e9193705"
        or canonical_sha256(stdout_unsigned) != stdout_internal
        or stdout.get("source_git_commit")
        != "1769eeaf0edfed7dacb3e4c2c1cecada24274e13"
        or stdout.get("source_identity_sha256")
        != "d4d36c9b8a584b72c6c1a3fed79e854043f1bfcaac8cd1a2c056e8b7c18f24d8"
        or stdout.get("stage2_config_raw_sha256")
        != "12794f1cc08f45d9dc5acae01c46d5f760f9456f8d73a308ae981a7b65a27512"
        or stdout.get("stage2_config_canonical_sha256")
        != "b9f33d39c668bb204754140c501996a13af20672ced19ec669503803bb9eb767"
        or stdout.get("integration_plumbing_passed") is not True
        or stdout.get("success_path_exercised") is not True
        or not isinstance(counts, dict)
        or counts.get("frame_count") != 920
        or counts.get("provider_forward_count") != 40
        or counts.get("memory_commit_count") != 3
        or counts.get("fresh_shadow_action_generation_count") != 3
        or counts.get("route_count") != 10
        or any(counts.get(name) != 0 for name in expected_zero_counts)
    ):
        raise RuntimeError("Stage 2A parent replay verifier output 漂移")

    receipt = {
        "version": "e018-p1-stage2a-parent-exact-environment-replay/v1",
        "status": "verified-original-environment-exact-replay",
        "verified": True,
        "accepted_artifact_id": "stage2a-integration-smoke-1769eea-20260906-v1",
        "source_git_commit": source_commit.strip(),
        "source_git_tree": source_tree.strip(),
        "source_status_clean": True,
        "python_version": "3.10.12",
        "numpy_version": "1.26.4",
        "blas_identity": "openblas64-0.3.23.dev",
        "gpu_identity": "NVIDIA RTX 6000 Ada Generation-49140MiB-driver580.126.09",
        "evidence_manifest_raw_sha256": expected_evidence_manifest_raw_sha256,
        "evidence_file_inventory_sha256": canonical_sha256(
            [
                {"basename": basename, "raw_sha256": evidence_hashes[basename]}
                for basename in sorted(evidence_hashes)
            ]
        ),
        "evidence_file_count": len(_PARENT_REPLAY_EVIDENCE_FILES),
        "evidence_total_bytes": total_bytes,
        "artifact_exact_tree_raw_sha256": evidence_hashes[
            "ARTIFACT_EXACT_TREE.txt"
        ],
        "artifact_file_manifest_raw_sha256": evidence_hashes[
            "ARTIFACT_FILE_SHA256SUMS.txt"
        ],
        "command_raw_sha256": evidence_hashes["COMMAND.txt"],
        "stdout_raw_sha256": evidence_hashes["STDOUT.json"],
        "stderr_raw_sha256": evidence_hashes["STDERR.log"],
        "exit_code": 0,
        "stage2a_public_verification_sha256": stdout_internal,
        "fresh_selection_seed_reads": 0,
    }
    receipt["verification_sha256"] = canonical_sha256(receipt)
    return receipt


def verify_selection_parent_gate(
    *,
    selection_config_path: str | Path,
    stage2a_config_path: str | Path,
    qualification_config_path: str | Path,
    stage2a_artifact_root: str | Path,
    stage2a_control_evidence_root: str | Path,
    artifact_inventory_path: str | Path,
    parent_replay_artifact_root: str | Path,
    parent_replay_control_evidence_root: str | Path,
) -> dict[str, Any]:
    """重算 immutable replica，并验证原数值环境的 exact smoke replay。"""

    loaded = load_e018_p1_stage2a_selection_config(selection_config_path)
    parent = loaded.payload["parents"]
    stage2a = load_e018_p1_stage2a_config(stage2a_config_path)
    if (
        stage2a.raw_sha256 != parent["stage2a_config_raw_sha256"]
        or stage2a.canonical_sha256
        != parent["stage2a_config_canonical_sha256"]
    ):
        raise RuntimeError("selection Stage 2A config parent 漂移")
    qualification_path = Path(qualification_config_path)
    qualification = load_g2c_dynamic_qualification_config(qualification_path)
    if (
        file_sha256(qualification_path)
        != stage2a.payload["parents"]["d048_qualification_config_raw_sha256"]
        or qualification.get("config_sha256")
        != stage2a.payload["parents"][
            "d048_qualification_config_internal_sha256"
        ]
    ):
        raise RuntimeError("selection D048 qualification config parent 漂移")
    artifact_root = Path(stage2a_artifact_root)
    _verify_stage2a_replica_manifest(
        artifact_root,
        expected_raw_sha256=parent["artifact_manifest_raw_sha256"],
        expected_internal_sha256=parent["artifact_manifest_internal_sha256"],
        expected_inventory_sha256=parent["artifact_file_inventory_sha256"],
    )
    marker_path = artifact_root / "DRIVE_BACKUP_COMPLETE.json"
    marker = _read_json(marker_path, "Stage 2A completion marker")
    if (
        file_sha256(marker_path) != parent["completion_marker_raw_sha256"]
        or _verify_internal_digest(
            marker,
            digest_key="marker_sha256",
            name="Stage 2A completion marker",
        )
        != parent["completion_marker_internal_sha256"]
        or marker.get("status") != "DRIVE_VERIFIED"
        or marker.get("artifact_id") != parent["accepted_artifact_id"]
    ):
        raise RuntimeError("Stage 2A completion marker identity 漂移")
    control_root = Path(stage2a_control_evidence_root)
    persistence_path = control_root / "PERSISTENCE_VERIFICATION.json"
    persistence = _read_json(persistence_path, "Stage 2A persistence verification")
    if (
        file_sha256(persistence_path)
        != parent["persistence_verification_raw_sha256"]
        or _verify_internal_digest(
            persistence,
            digest_key="verification_sha256",
            name="Stage 2A persistence verification",
        )
        != parent["persistence_verification_internal_sha256"]
        or persistence.get("verified") is not True
        or persistence.get("replication_state") != "DRIVE_VERIFIED"
    ):
        raise RuntimeError("Stage 2A persistence verification 漂移")
    local_path = control_root / "LOCAL_CANONICAL_REPLICA_VERIFICATION.json"
    local = _read_json(local_path, "Stage 2A local canonical verification")
    if (
        file_sha256(local_path)
        != parent["local_canonical_verification_raw_sha256"]
        or _verify_internal_digest(
            local,
            digest_key="verification_sha256",
            name="Stage 2A local canonical verification",
        )
        != parent["local_canonical_verification_internal_sha256"]
        or local.get("verified") is not True
        or local.get("replication_state") != "REPLICATED"
    ):
        raise RuntimeError("Stage 2A local canonical verification 漂移")
    inventory = _read_json(Path(artifact_inventory_path), "artifact inventory")
    records = inventory.get("backups")
    if not isinstance(records, list):
        raise TypeError("artifact inventory backups schema 漂移")
    matches = [
        value
        for value in records
        if isinstance(value, dict)
        and value.get("id") == parent["accepted_artifact_id"]
    ]
    if (
        len(matches) != 1
        or canonical_sha256(matches[0])
        != parent["inventory_record_canonical_sha256"]
        or matches[0].get("replication_state") != "REPLICATED"
        or matches[0].get("artifact_inventory_sha256")
        != parent["artifact_file_inventory_sha256"]
    ):
        raise RuntimeError("Stage 2A REPLICATED inventory record 漂移")

    replay_artifact_root = Path(parent_replay_artifact_root)
    if replay_artifact_root.is_symlink() or not replay_artifact_root.is_dir():
        raise RuntimeError("Stage 2A parent replay artifact root 必须是真实目录")
    replay_entries = {
        path.name: path for path in replay_artifact_root.iterdir()
    }
    if set(replay_entries) != {"evidence", "DRIVE_BACKUP_COMPLETE.json"}:
        raise RuntimeError("Stage 2A parent replay artifact exact tree 漂移")
    if (
        replay_entries["evidence"].is_symlink()
        or not replay_entries["evidence"].is_dir()
    ):
        raise RuntimeError("Stage 2A parent replay evidence 必须是真实目录")
    replay_marker_path = replay_entries["DRIVE_BACKUP_COMPLETE.json"]
    replay_marker = _read_json(
        replay_marker_path,
        "Stage 2A parent replay Drive marker",
    )
    if (
        file_sha256(replay_marker_path)
        != parent["parent_replay_drive_marker_raw_sha256"]
        or _verify_internal_digest(
            replay_marker,
            digest_key="marker_sha256",
            name="Stage 2A parent replay Drive marker",
        )
        != parent["parent_replay_drive_marker_internal_sha256"]
        or replay_marker.get("artifact_id")
        != parent["parent_replay_artifact_id"]
        or replay_marker.get("status") != "DRIVE_VERIFIED"
        or replay_marker.get("evidence_manifest_raw_sha256")
        != parent["parent_replay_evidence_manifest_raw_sha256"]
        or replay_marker.get("public_verification_sha256")
        != parent["stage2a_smoke_public_verification_sha256"]
        or replay_marker.get("source_git_commit")
        != parent["stage2a_smoke_source_git_commit"]
    ):
        raise RuntimeError("Stage 2A parent replay Drive marker identity 漂移")

    replay_control_root = Path(parent_replay_control_evidence_root)
    replay_local_path = replay_control_root / "LOCAL_VERIFICATION.json"
    replay_local = _read_json(
        replay_local_path,
        "Stage 2A parent replay local verification",
    )
    if (
        file_sha256(replay_local_path)
        != parent["parent_replay_local_verification_raw_sha256"]
        or _verify_internal_digest(
            replay_local,
            digest_key="verification_sha256",
            name="Stage 2A parent replay local verification",
        )
        != parent["parent_replay_local_verification_internal_sha256"]
        or replay_local.get("verified") is not True
        or replay_local.get("replication_state") != "REPLICATED"
        or replay_local.get("transfer_route")
        != "worker-direct-local-replica-after-drive-primary-verified"
        or replay_local.get("artifact_id")
        != parent["parent_replay_artifact_id"]
        or replay_local.get("drive_marker_raw_sha256")
        != parent["parent_replay_drive_marker_raw_sha256"]
        or replay_local.get("drive_marker_internal_sha256")
        != parent["parent_replay_drive_marker_internal_sha256"]
        or replay_local.get("evidence_manifest_raw_sha256")
        != parent["parent_replay_evidence_manifest_raw_sha256"]
        or replay_local.get("public_verification_sha256")
        != parent["stage2a_smoke_public_verification_sha256"]
    ):
        raise RuntimeError("Stage 2A parent replay local verification identity 漂移")

    replay_inventory_matches = [
        value
        for value in records
        if isinstance(value, dict)
        and value.get("id") == parent["parent_replay_artifact_id"]
    ]
    if len(replay_inventory_matches) != 1:
        raise RuntimeError("Stage 2A parent replay inventory 必须 exact-one")
    replay_inventory_record = replay_inventory_matches[0]
    if (
        canonical_sha256(replay_inventory_record)
        != parent["parent_replay_inventory_record_canonical_sha256"]
        or replay_inventory_record.get("replication_state") != "REPLICATED"
        or replay_inventory_record.get("evidence_manifest_raw_sha256")
        != parent["parent_replay_evidence_manifest_raw_sha256"]
        or replay_inventory_record.get("completion_marker_raw_sha256")
        != parent["parent_replay_drive_marker_raw_sha256"]
        or replay_inventory_record.get("completion_marker_internal_sha256")
        != parent["parent_replay_drive_marker_internal_sha256"]
        or replay_inventory_record.get("local_replica_verification_raw_sha256")
        != parent["parent_replay_local_verification_raw_sha256"]
        or replay_inventory_record.get(
            "local_replica_verification_internal_sha256"
        )
        != parent["parent_replay_local_verification_internal_sha256"]
        or replay_inventory_record.get("replay_verification_internal_sha256")
        != parent["parent_replay_verification_sha256"]
        or replay_inventory_record.get("public_verification_internal_sha256")
        != parent["stage2a_smoke_public_verification_sha256"]
        or replay_inventory_record.get("source_git_commit")
        != parent["stage2a_smoke_source_git_commit"]
        or replay_inventory_record.get("source_identity_sha256")
        != parent["stage2a_smoke_source_identity_sha256"]
    ):
        raise RuntimeError("Stage 2A parent replay REPLICATED inventory record 漂移")
    # 存储中的 public verification 先做环境无关的 raw/internal identity
    # 校验。SVD exact replay 由下方原运行时证据证明，当前机器不重新执行
    # ``np.array_equal``，避免把 BLAS/LAPACK 末位差异误写成 parent 失败。
    stored_public_path = artifact_root / "control" / "public_verification.json"
    stored_public = _read_json(
        stored_public_path,
        "Stage 2A stored public verification",
    )
    if (
        file_sha256(stored_public_path)
        != parent["stage2a_smoke_public_verification_raw_sha256"]
        or _verify_internal_digest(
            stored_public,
            digest_key="verification_sha256",
            name="Stage 2A stored public verification",
        )
        != parent["stage2a_smoke_public_verification_sha256"]
        or stored_public.get("source_git_commit")
        != parent["stage2a_smoke_source_git_commit"]
        or stored_public.get("source_identity_sha256")
        != parent["stage2a_smoke_source_identity_sha256"]
        or stored_public.get("integration_plumbing_passed") is not True
    ):
        raise RuntimeError("Stage 2A stored public verification identity 漂移")
    replay = verify_stage2a_parent_replay_evidence(
        replay_evidence_root=replay_entries["evidence"],
        stage2a_artifact_root=artifact_root,
        expected_evidence_manifest_raw_sha256=parent[
            "parent_replay_evidence_manifest_raw_sha256"
        ],
    )
    if (
        replay.get("verification_sha256")
        != parent["parent_replay_verification_sha256"]
        or replay.get("stage2a_public_verification_sha256")
        != parent["stage2a_smoke_public_verification_sha256"]
    ):
        raise RuntimeError("Stage 2A original-environment replay identity 漂移")
    result = {
        "version": E018_P1_STAGE2A_SELECTION_EXECUTION_VERSION,
        "verified": True,
        "accepted_artifact_id": parent["accepted_artifact_id"],
        "artifact_manifest_internal_sha256": parent[
            "artifact_manifest_internal_sha256"
        ],
        "artifact_file_inventory_sha256": parent[
            "artifact_file_inventory_sha256"
        ],
        "completion_marker_internal_sha256": parent[
            "completion_marker_internal_sha256"
        ],
        "persistence_verification_internal_sha256": parent[
            "persistence_verification_internal_sha256"
        ],
        "local_canonical_verification_internal_sha256": parent[
            "local_canonical_verification_internal_sha256"
        ],
        "inventory_record_canonical_sha256": parent[
            "inventory_record_canonical_sha256"
        ],
        "replication_state": "REPLICATED",
        "stage2a_public_verification_sha256": stored_public[
            "verification_sha256"
        ],
        "parent_replay_artifact_id": parent["parent_replay_artifact_id"],
        "parent_replay_verification_sha256": replay["verification_sha256"],
        "parent_replay_drive_marker_internal_sha256": replay_marker[
            "marker_sha256"
        ],
        "parent_replay_local_verification_internal_sha256": replay_local[
            "verification_sha256"
        ],
        "parent_replay_inventory_record_canonical_sha256": parent[
            "parent_replay_inventory_record_canonical_sha256"
        ],
        "parent_replay_replication_state": parent[
            "parent_replay_replication_state"
        ],
        "portability_diagnostics": [
            {
                "classification": "non-experiment-local-svd-portability-diagnostic",
                "environment": "python-3.12.3-numpy-1.26.4-system-blas",
                "exit_code": 1,
                "collect_pose_max_abs_difference": 2.220446049250313e-16,
                "parent_gate_effect": "diagnostic-only-no-rejection",
                "fresh_selection_seed_reads": 0,
            },
            {
                "classification": "non-experiment-operator-path-error",
                "exit_code": 1,
                "cause": "public_execution-subdirectory-appended-to-flat-source-root",
                "artifact_read_count": 0,
                "artifact_write_count": 0,
                "parent_gate_effect": "superseded-by-corrected-exit-zero-replay",
                "fresh_selection_seed_reads": 0,
            },
        ],
    }
    result["verification_sha256"] = canonical_sha256(result)
    return result


@dataclass
class SelectionExecutionProgress:
    current_seed: int | None = None
    prediction_count: int = 0
    private_label_count: int = 0
    route_count: int = 0
    branch_count: int = 0
    provider_forward_count: int = 0
    memory_commit_count: int = 0
    fresh_shadow_action_generation_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


class Stage2ASelectionJournal:
    """100 个 public prediction commit 与 75 个 write-only private label。"""

    def __init__(
        self,
        *,
        public_root: str | Path,
        private_root: str | Path,
        config_canonical_sha256: str,
        transaction_identity_sha256: str,
    ) -> None:
        self.public_root = Path(public_root)
        self.private_root = Path(private_root)
        if self.public_root.exists() or self.private_root.exists():
            raise FileExistsError("selection public/private roots 必须同时全新")
        if not _is_sha256(config_canonical_sha256) or not _is_sha256(
            transaction_identity_sha256
        ):
            raise ValueError("selection journal identity SHA 非法")
        self.public_root.mkdir(mode=0o700, parents=True, exist_ok=False)
        self.private_root.mkdir(mode=0o700, parents=True, exist_ok=False)
        (self.public_root / "prediction_commits").mkdir(mode=0o700)
        (self.private_root / "label_commits").mkdir(mode=0o700)
        self.config_canonical_sha256 = config_canonical_sha256
        self.transaction_identity_sha256 = transaction_identity_sha256
        self.provider_writer = _AppendOnlyJsonl(
            self.public_root / "provider_output_ledger.jsonl"
        )
        self.prediction_count = 0
        self.label_count = 0
        self.privileged_access_started_count = 0
        self.previous_commit_receipt_sha256: str | None = None
        self.private_inventory_rows: list[dict[str, Any]] = []
        self._write_capture_state("initialized-before-privileged-read")

    def _write_capture_state(self, status: str) -> None:
        state = {
            "version": E018_P1_STAGE2A_SELECTION_EXECUTION_VERSION,
            "status": status,
            "transaction_identity_sha256": self.transaction_identity_sha256,
            "prediction_commit_count": self.prediction_count,
            "privileged_access_started_count": self.privileged_access_started_count,
            "privileged_capture_count": self.label_count,
            "rerun_under_same_identity_allowed": False,
        }
        state["state_sha256"] = canonical_sha256(state)
        _atomic_replace_json(self.private_root / "capture_state.json", state)

    def commit_prediction(
        self,
        row: Mapping[str, Any],
        *,
        seed: int,
        route_frame_index: int,
        provider_output_digest: str,
        model_input_digest: str,
    ) -> dict[str, Any]:
        row_index = self.prediction_count
        if seed not in STAGE2A_SELECTION_SEEDS:
            raise ValueError("selection prediction seed 不在冻结 split")
        if route_frame_index not in STAGE2A_PROVIDER_FRAME_INDICES:
            raise ValueError("selection prediction frame 不在冻结 provider frames")
        if not _is_sha256(provider_output_digest) or not _is_sha256(model_input_digest):
            raise ValueError("selection provider/input digest 非法")
        expected_seed = STAGE2A_SELECTION_SEEDS[row_index // 4]
        expected_frame = STAGE2A_PROVIDER_FRAME_INDICES[row_index % 4]
        if seed != expected_seed or route_frame_index != expected_frame:
            raise RuntimeError("selection prediction seed/frame exact order 漂移")
        public_row = dict(row)
        assert_qualification_prediction_deployable_only(public_row)
        if (
            public_row.get("provider_output_digest") != provider_output_digest
            or public_row.get("model_input_digest") != model_input_digest
        ):
            raise RuntimeError("selection provider ledger row/digest 未绑定")
        self.provider_writer.append([public_row])
        prediction_fsync_completed_at = time.time_ns()
        prefix_sha256 = file_sha256(self.provider_writer.path)
        receipt = {
            "version": E018_P1_STAGE2A_SELECTION_EXECUTION_VERSION,
            "row_index": row_index,
            "seed": seed,
            "route_frame_index": route_frame_index,
            "provider_output_digest": provider_output_digest,
            "model_input_digest": model_input_digest,
            "transaction_identity_sha256": self.transaction_identity_sha256,
            "provider_ledger_prefix_raw_sha256": prefix_sha256,
            "previous_prediction_commit_sha256": (
                self.previous_commit_receipt_sha256
            ),
            "prediction_fsync_completed_at_unix_ns": (
                prediction_fsync_completed_at
            ),
        }
        receipt["commit_receipt_sha256"] = canonical_sha256(receipt)
        path = self.public_root / "prediction_commits" / f"{row_index:06d}.commit.json"
        _atomic_create_json(path, receipt)
        self.previous_commit_receipt_sha256 = receipt["commit_receipt_sha256"]
        self.prediction_count += 1
        return receipt

    def capture_private_label_after_prediction(
        self,
        *,
        prediction_receipt: Mapping[str, Any],
        seed: int,
        route_frame_index: int,
        rgb_sha256: str,
        actual_pose_sha256: str,
        provider_output_digest: str,
        privileged_getter: Callable[[], Mapping[str, Any]],
    ) -> dict[str, Any]:
        if route_frame_index not in STAGE2A_COLLECT_FRAME_INDICES:
            raise ValueError("HOME prediction 禁止打开 private label")
        expected_label_seed = STAGE2A_SELECTION_SEEDS[self.label_count // 3]
        expected_label_frame = STAGE2A_COLLECT_FRAME_INDICES[self.label_count % 3]
        if seed != expected_label_seed or route_frame_index != expected_label_frame:
            raise RuntimeError("selection private label seed/frame order 漂移")
        if (
            prediction_receipt.get("row_index") != self.prediction_count - 1
            or prediction_receipt.get("seed") != seed
            or prediction_receipt.get("route_frame_index") != route_frame_index
            or prediction_receipt.get("provider_output_digest")
            != provider_output_digest
            or canonical_sha256(
                {
                    key: value
                    for key, value in prediction_receipt.items()
                    if key != "commit_receipt_sha256"
                }
            )
            != prediction_receipt.get("commit_receipt_sha256")
        ):
            raise RuntimeError("selection label 早于或未绑定 prediction commit")
        for value, name in (
            (rgb_sha256, "rgb_sha256"),
            (actual_pose_sha256, "actual_pose_sha256"),
            (provider_output_digest, "provider_output_digest"),
        ):
            if not _is_sha256(value):
                raise ValueError(f"selection label {name} 非法")
        # 先持久化“privileged access 已开始”。getter 即使读取后抛错，
        # 当前 experiment identity 也不能再被声明为未消费。
        self.privileged_access_started_count += 1
        self._write_capture_state("privileged-access-in-progress")
        privileged = dict(privileged_getter())
        if set(privileged) != _SELECTION_PRIVATE_CAPTURE_KEYS:
            raise RuntimeError("selection private label exact keys 漂移")
        qualification_label = {
            key: privileged[key] for key in _QUALIFICATION_OBJECT_LABEL_KEYS
        }
        _validate_qualification_object_label(qualification_label, committed=False)
        position = np.asarray(privileged["gt_object_position_base_m"], dtype=np.float64)
        contact = float(privileged["robot_object_contact_force_n"])
        linear = float(privileged["object_linear_speed_m_s"])
        angular = float(privileged["object_angular_speed_rad_s"])
        expected_motion = bool(linear > 0.01 or angular > 0.5)
        if (
            type(privileged["gt_object_exists"]) is not bool
            or type(privileged["gt_observable"]) is not bool
            or position.shape != (3,)
            or not np.isfinite(position).all()
            or any(not math.isfinite(value) or value < 0.0 for value in (contact, linear, angular))
            or privileged["object_motion_event"] is not expected_motion
            or type(privileged["is_grasped"]) is not bool
            or privileged["goal_gt_read_count"] != 0
            or privileged["test_data_read"] is not False
        ):
            raise RuntimeError("selection private scoring primitive 语义漂移")
        captured_at = time.time_ns()
        if captured_at <= int(
            prediction_receipt["prediction_fsync_completed_at_unix_ns"]
        ):
            raise RuntimeError("selection label capture 必须晚于 prediction fsync")
        label_index = self.label_count
        label = {
            **privileged,
            "version": E018_P1_STAGE2A_SELECTION_EXECUTION_VERSION,
            "label_index": label_index,
            "prediction_row_index": prediction_receipt["row_index"],
            "seed": seed,
            "route_frame_index": route_frame_index,
            "rgb_sha256": rgb_sha256,
            "actual_pose_sha256": actual_pose_sha256,
            "provider_output_digest": provider_output_digest,
            "prediction_commit_receipt_sha256": prediction_receipt[
                "commit_receipt_sha256"
            ],
            "transaction_identity_sha256": self.transaction_identity_sha256,
            "motion_predicate_version": "pick-and-place-predicates/v1",
            "motion_linear_threshold_m_s": 0.01,
            "motion_angular_threshold_rad_s": 0.5,
            "contact_threshold_n": 0.01,
            "privileged_captured_at_unix_ns": captured_at,
        }
        label["label_sha256"] = canonical_sha256(label)
        path = self.private_root / "label_commits" / f"{label_index:06d}.json"
        raw_sha256, _ = _atomic_create_json(path, label)
        scoring_primitive = {
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
            "label_index": label_index,
            "prediction_row_index": prediction_receipt["row_index"],
            "seed": seed,
            "route_frame_index": route_frame_index,
            "path": f"label_commits/{label_index:06d}.json",
            "raw_sha256": raw_sha256,
            "size_bytes": path.stat().st_size,
            "scoring_primitive_sha256": canonical_sha256(scoring_primitive),
        }
        inventory_row["row_sha256"] = canonical_sha256(inventory_row)
        self.private_inventory_rows.append(inventory_row)
        self.label_count += 1
        self._write_capture_state("capture-in-progress")
        # Phase A 只把不可逆 metadata 交还 caller；privileged 值不能进入
        # branch 决策或公开 ledger。
        public_commit_metadata = dict(inventory_row)
        privileged.clear()
        qualification_label.clear()
        label.clear()
        return public_commit_metadata

    def freeze(self) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.prediction_count != STAGE2A_SELECTION_PREDICTION_COUNT:
            raise RuntimeError("selection prediction count 未达到 100")
        if self.label_count != STAGE2A_SELECTION_LABEL_COUNT:
            raise RuntimeError("selection private label count 未达到 75")
        if self.privileged_access_started_count != self.label_count:
            raise RuntimeError("selection privileged access/capture count 不一致")
        provider = self.provider_writer.freeze()
        inventory = {
            "version": E018_P1_STAGE2A_SELECTION_EXECUTION_VERSION,
            "label_count": self.label_count,
            "rows": self.private_inventory_rows,
        }
        inventory["inventory_sha256"] = canonical_sha256(inventory)
        self._write_capture_state("capture-complete-write-only-not-opened")
        return provider, inventory


class Stage2ASelectionPreflightJournal:
    """固定 76891 的 4 prediction / 3 write-only label preflight journal。"""

    def __init__(
        self,
        *,
        public_root: str | Path,
        private_root: str | Path,
        config_canonical_sha256: str,
        transaction_identity_sha256: str,
    ) -> None:
        self.public_root = Path(public_root)
        self.private_root = Path(private_root)
        if self.public_root.exists() or self.private_root.exists():
            raise FileExistsError("preflight public/private roots 必须同时全新")
        if not _is_sha256(config_canonical_sha256) or not _is_sha256(
            transaction_identity_sha256
        ):
            raise ValueError("preflight journal identity SHA 非法")
        self.public_root.mkdir(mode=0o700, parents=True, exist_ok=False)
        self.private_root.mkdir(mode=0o700, parents=True, exist_ok=False)
        (self.public_root / "prediction_commits").mkdir(mode=0o700)
        (self.private_root / "label_commits").mkdir(mode=0o700)
        self.config_canonical_sha256 = config_canonical_sha256
        self.transaction_identity_sha256 = transaction_identity_sha256
        self.provider_writer = _AppendOnlyJsonl(
            self.public_root / "provider_output_ledger.jsonl"
        )
        self.prediction_count = 0
        self.label_count = 0
        self.privileged_access_started_count = 0
        self.previous_commit_receipt_sha256: str | None = None
        self.private_inventory_rows: list[dict[str, Any]] = []
        self._write_capture_state("initialized-before-privileged-read")

    def _write_capture_state(self, status: str) -> None:
        state = {
            "version": E018_P1_STAGE2A_SELECTION_PREFLIGHT_EXECUTION_VERSION,
            "experiment_id": E018_P1_STAGE2A_SELECTION_PREFLIGHT_EXPERIMENT_ID,
            "status": status,
            "seed": STAGE2A_SELECTION_PREFLIGHT_SEED,
            "transaction_identity_sha256": self.transaction_identity_sha256,
            "prediction_commit_count": self.prediction_count,
            "privileged_access_started_count": self.privileged_access_started_count,
            "privileged_capture_count": self.label_count,
            "formal_selection_identity_consumed": False,
        }
        state["state_sha256"] = canonical_sha256(state)
        _atomic_replace_json(self.private_root / "capture_state.json", state)

    def commit_prediction(
        self,
        row: Mapping[str, Any],
        *,
        seed: int,
        route_frame_index: int,
        provider_output_digest: str,
        model_input_digest: str,
    ) -> dict[str, Any]:
        row_index = self.prediction_count
        if seed != STAGE2A_SELECTION_PREFLIGHT_SEED:
            raise ValueError("preflight prediction 只接受固定 seed 76891")
        if row_index >= STAGE2A_SELECTION_PREFLIGHT_PREDICTION_COUNT:
            raise RuntimeError("preflight prediction 超过固定 4 次")
        expected_frame = STAGE2A_PROVIDER_FRAME_INDICES[row_index]
        if route_frame_index != expected_frame:
            raise RuntimeError("preflight prediction frame exact order 漂移")
        if not _is_sha256(provider_output_digest) or not _is_sha256(
            model_input_digest
        ):
            raise ValueError("preflight provider/input digest 非法")
        public_row = dict(row)
        assert_qualification_prediction_deployable_only(public_row)
        if (
            public_row.get("provider_output_digest") != provider_output_digest
            or public_row.get("model_input_digest") != model_input_digest
        ):
            raise RuntimeError("preflight provider ledger row/digest 未绑定")
        self.provider_writer.append([public_row])
        completed_at = time.time_ns()
        receipt = {
            "version": E018_P1_STAGE2A_SELECTION_PREFLIGHT_EXECUTION_VERSION,
            "row_index": row_index,
            "seed": seed,
            "route_frame_index": route_frame_index,
            "provider_output_digest": provider_output_digest,
            "model_input_digest": model_input_digest,
            "transaction_identity_sha256": self.transaction_identity_sha256,
            "provider_ledger_prefix_raw_sha256": file_sha256(
                self.provider_writer.path
            ),
            "previous_prediction_commit_sha256": (
                self.previous_commit_receipt_sha256
            ),
            "prediction_fsync_completed_at_unix_ns": completed_at,
        }
        receipt["commit_receipt_sha256"] = canonical_sha256(receipt)
        path = self.public_root / "prediction_commits" / f"{row_index:06d}.commit.json"
        _atomic_create_json(path, receipt)
        self.previous_commit_receipt_sha256 = receipt["commit_receipt_sha256"]
        self.prediction_count += 1
        return receipt

    def capture_private_label_after_prediction(
        self,
        *,
        prediction_receipt: Mapping[str, Any],
        seed: int,
        route_frame_index: int,
        rgb_sha256: str,
        actual_pose_sha256: str,
        provider_output_digest: str,
        privileged_getter: Callable[[], Mapping[str, Any]],
    ) -> dict[str, Any]:
        label_index = self.label_count
        if seed != STAGE2A_SELECTION_PREFLIGHT_SEED:
            raise ValueError("preflight private label 只接受固定 seed 76891")
        if label_index >= STAGE2A_SELECTION_PREFLIGHT_LABEL_COUNT:
            raise RuntimeError("preflight private label 超过固定 3 次")
        expected_frame = STAGE2A_COLLECT_FRAME_INDICES[label_index]
        if route_frame_index != expected_frame:
            raise RuntimeError("preflight private label frame exact order 漂移")
        unsigned_receipt = dict(prediction_receipt)
        stored_receipt_sha256 = unsigned_receipt.pop(
            "commit_receipt_sha256", None
        )
        if (
            prediction_receipt.get("row_index") != self.prediction_count - 1
            or prediction_receipt.get("seed") != seed
            or prediction_receipt.get("route_frame_index") != route_frame_index
            or prediction_receipt.get("provider_output_digest")
            != provider_output_digest
            or stored_receipt_sha256 != canonical_sha256(unsigned_receipt)
        ):
            raise RuntimeError("preflight label 早于或未绑定 prediction commit")
        for value, name in (
            (rgb_sha256, "rgb_sha256"),
            (actual_pose_sha256, "actual_pose_sha256"),
            (provider_output_digest, "provider_output_digest"),
        ):
            if not _is_sha256(value):
                raise ValueError(f"preflight label {name} 非法")
        self.privileged_access_started_count += 1
        self._write_capture_state("privileged-access-in-progress")
        privileged = dict(privileged_getter())
        if set(privileged) != _SELECTION_PRIVATE_CAPTURE_KEYS:
            raise RuntimeError("preflight private label exact keys 漂移")
        qualification_label = {
            key: privileged[key] for key in _QUALIFICATION_OBJECT_LABEL_KEYS
        }
        _validate_qualification_object_label(qualification_label, committed=False)
        position = np.asarray(
            privileged["gt_object_position_base_m"], dtype=np.float64
        )
        contact = float(privileged["robot_object_contact_force_n"])
        linear = float(privileged["object_linear_speed_m_s"])
        angular = float(privileged["object_angular_speed_rad_s"])
        if (
            type(privileged["gt_object_exists"]) is not bool
            or type(privileged["gt_observable"]) is not bool
            or position.shape != (3,)
            or not np.isfinite(position).all()
            or any(
                not math.isfinite(value) or value < 0.0
                for value in (contact, linear, angular)
            )
            or privileged["object_motion_event"]
            is not bool(linear > 0.01 or angular > 0.5)
            or type(privileged["is_grasped"]) is not bool
            or privileged["goal_gt_read_count"] != 0
            or privileged["test_data_read"] is not False
        ):
            raise RuntimeError("preflight private scoring primitive 语义漂移")
        captured_at = time.time_ns()
        if captured_at <= int(
            prediction_receipt["prediction_fsync_completed_at_unix_ns"]
        ):
            raise RuntimeError("preflight label capture 必须晚于 prediction fsync")
        label = {
            **privileged,
            "version": E018_P1_STAGE2A_SELECTION_PREFLIGHT_EXECUTION_VERSION,
            "label_index": label_index,
            "prediction_row_index": prediction_receipt["row_index"],
            "seed": seed,
            "route_frame_index": route_frame_index,
            "rgb_sha256": rgb_sha256,
            "actual_pose_sha256": actual_pose_sha256,
            "provider_output_digest": provider_output_digest,
            "prediction_commit_receipt_sha256": stored_receipt_sha256,
            "transaction_identity_sha256": self.transaction_identity_sha256,
            "motion_predicate_version": "pick-and-place-predicates/v1",
            "motion_linear_threshold_m_s": 0.01,
            "motion_angular_threshold_rad_s": 0.5,
            "contact_threshold_n": 0.01,
            "privileged_captured_at_unix_ns": captured_at,
        }
        label["label_sha256"] = canonical_sha256(label)
        path = self.private_root / "label_commits" / f"{label_index:06d}.json"
        raw_sha256, _ = _atomic_create_json(path, label)
        scoring_primitive = {
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
            "label_index": label_index,
            "prediction_row_index": prediction_receipt["row_index"],
            "seed": seed,
            "route_frame_index": route_frame_index,
            "path": f"label_commits/{label_index:06d}.json",
            "raw_sha256": raw_sha256,
            "size_bytes": path.stat().st_size,
            "scoring_primitive_sha256": canonical_sha256(scoring_primitive),
        }
        inventory_row["row_sha256"] = canonical_sha256(inventory_row)
        self.private_inventory_rows.append(inventory_row)
        self.label_count += 1
        self._write_capture_state("capture-in-progress")
        privileged.clear()
        qualification_label.clear()
        label.clear()
        return dict(inventory_row)

    def freeze(self) -> None:
        """preflight 不能调用 formal selection freeze。"""

        raise PermissionError("preflight journal 禁止 formal freeze")

    def finalize_preflight_capture(self) -> tuple[dict[str, Any], dict[str, Any]]:
        if (
            self.prediction_count
            != STAGE2A_SELECTION_PREFLIGHT_PREDICTION_COUNT
            or self.label_count != STAGE2A_SELECTION_PREFLIGHT_LABEL_COUNT
            or self.privileged_access_started_count != self.label_count
        ):
            raise RuntimeError("preflight journal 未达到固定 4/3 accounting")
        provider = self.provider_writer.freeze()
        inventory = {
            "version": E018_P1_STAGE2A_SELECTION_PREFLIGHT_EXECUTION_VERSION,
            "experiment_id": E018_P1_STAGE2A_SELECTION_PREFLIGHT_EXPERIMENT_ID,
            "seed": STAGE2A_SELECTION_PREFLIGHT_SEED,
            "label_count": self.label_count,
            "rows": self.private_inventory_rows,
        }
        inventory["inventory_sha256"] = canonical_sha256(inventory)
        self._write_capture_state("preflight-capture-complete-write-only")
        return provider, inventory


@dataclass(frozen=True)
class GainBranchOutcome:
    seed: int
    gain: float
    route_evidence_digest: str
    route_protocol_safety_valid: bool
    candidate_commit_eligible: bool
    memory_commit_count: int
    navigation_state_available: bool
    fresh_shadow_action_generation_count: int
    committed_position_base_m: tuple[float, float, float] | None
    provider_forward_count: int
    arm_motion_command_count: int = 0
    gripper_close_command_count: int = 0
    protocol_violation_count: int = 0

    def __post_init__(self) -> None:
        if self.seed not in STAGE2A_SELECTION_SEEDS:
            raise ValueError("gain branch seed 不在 selection split")
        if self.gain not in STAGE2A_SELECTION_GAINS:
            raise ValueError("gain branch 未使用冻结候选")
        if not _is_sha256(self.route_evidence_digest):
            raise ValueError("gain branch route evidence digest 非法")
        for name in (
            "route_protocol_safety_valid",
            "candidate_commit_eligible",
            "navigation_state_available",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"gain branch {name} 必须是 bool")
        for name in (
            "memory_commit_count",
            "fresh_shadow_action_generation_count",
            "provider_forward_count",
            "arm_motion_command_count",
            "gripper_close_command_count",
        ):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"gain branch {name} 必须是整数")
        if self.memory_commit_count not in {0, 1}:
            raise ValueError("gain branch Memory commit count 只能是 0/1")
        if self.fresh_shadow_action_generation_count != self.memory_commit_count:
            raise ValueError("gain branch commit 与 fresh Action 必须一一对应")
        if self.provider_forward_count != 0:
            raise ValueError("纯逻辑 gain branch 禁止 provider re-forward")
        if self.arm_motion_command_count != 0 or self.gripper_close_command_count != 0:
            raise ValueError("gain branch 禁止 manipulation actuator")
        if (
            type(self.protocol_violation_count) is not int
            or self.protocol_violation_count < 0
        ):
            raise ValueError("gain branch protocol_violation_count 必须是非负整数")
        if self.memory_commit_count == 1:
            if (
                not self.candidate_commit_eligible
                or not self.route_protocol_safety_valid
                or not self.navigation_state_available
                or self.fresh_shadow_action_generation_count != 1
            ):
                raise ValueError("gain branch commit/eligibility/route/navigation/Action 不一致")
            position = np.asarray(self.committed_position_base_m, dtype=np.float64)
            if position.shape != (3,) or not np.isfinite(position).all():
                raise ValueError("committed gain branch 缺少有限 base-frame XYZ")
        elif self.committed_position_base_m is not None or self.navigation_state_available:
            raise ValueError("no-commit gain branch 不得携带 committed XYZ/navigation")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["committed_position_base_m"] = (
            None
            if self.committed_position_base_m is None
            else list(self.committed_position_base_m)
        )
        value["branch_sha256"] = canonical_sha256(value)
        return value


def _serialize_request(value: ActiveFrontReobserveRequest) -> dict[str, Any]:
    return {
        "episode_id": value.episode_id,
        "episode_generation": value.episode_generation,
        "request_id": value.request_id,
        "source_phase": value.source_phase.value,
        "resume_phase": value.resume_phase.value,
        "trigger_tick": value.trigger_tick,
        "trigger_timestamp_s": value.trigger_timestamp_s,
        "trigger_reason": value.trigger_reason.value,
        "attempt_index": value.attempt_index,
        "selected_primitive_id": value.selected_primitive_id,
        "camera_command_sequence_id": value.camera_command_sequence_id,
    }


def _deserialize_request(value: Mapping[str, Any]) -> ActiveFrontReobserveRequest:
    expected = {
        "episode_id",
        "episode_generation",
        "request_id",
        "source_phase",
        "resume_phase",
        "trigger_tick",
        "trigger_timestamp_s",
        "trigger_reason",
        "attempt_index",
        "selected_primitive_id",
        "camera_command_sequence_id",
    }
    row = _require_exact_keys(dict(value), expected, "selection request")
    try:
        return ActiveFrontReobserveRequest(
            **{
                **row,
                "source_phase": PhaseId(row["source_phase"]),
                "resume_phase": PhaseId(row["resume_phase"]),
                "trigger_reason": ActiveFrontTriggerReason(row["trigger_reason"]),
            }
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError("selection request 类型/枚举漂移") from error


def _serialize_home_score(value: PassiveHomeScoreEvidence) -> dict[str, Any]:
    return {
        "episode_id": value.episode_id,
        "episode_generation": value.episode_generation,
        "request_id": value.request_id,
        "observation_sequence_id": value.observation_sequence_id,
        "model_input_digest": value.model_input_digest,
        "provider_output_digest": value.provider_output_digest,
        "provider_identity": value.provider_identity.to_dict(),
        "viewpoint_primitive_id": value.viewpoint_primitive_id,
        "camera_motion_state": value.camera_motion_state.value,
        "settled": value.settled,
        "score_components": asdict(value.score_components),
        "stored_write_score": value.stored_write_score,
        "geometry_valid": value.geometry_valid,
        "control_timestamp_s": value.control_timestamp_s,
        "rgb_timestamp_s": value.rgb_timestamp_s,
        "camera_pose_timestamp_s": value.camera_pose_timestamp_s,
        "tcp_pose_timestamp_s": value.tcp_pose_timestamp_s,
        "base_from_external_camera_cv": (
            None
            if value.base_from_external_camera_cv is None
            else value.base_from_external_camera_cv.tolist()
        ),
        "actual_pose_source": value.actual_pose_source,
        "score_semantics": value.score_semantics,
        "object_measurement_usable": value.object_measurement_usable,
        "version": value.version,
    }


def _deserialize_home_score(value: Mapping[str, Any]) -> PassiveHomeScoreEvidence:
    expected = {
        "episode_id",
        "episode_generation",
        "request_id",
        "observation_sequence_id",
        "model_input_digest",
        "provider_output_digest",
        "provider_identity",
        "viewpoint_primitive_id",
        "camera_motion_state",
        "settled",
        "score_components",
        "stored_write_score",
        "geometry_valid",
        "control_timestamp_s",
        "rgb_timestamp_s",
        "camera_pose_timestamp_s",
        "tcp_pose_timestamp_s",
        "base_from_external_camera_cv",
        "actual_pose_source",
        "score_semantics",
        "object_measurement_usable",
        "version",
    }
    row = _require_exact_keys(dict(value), expected, "selection HOME baseline")
    try:
        return PassiveHomeScoreEvidence(
            **{
                **row,
                "provider_identity": ActiveFrontStage2ProviderIdentity(
                    **row["provider_identity"]
                ),
                "camera_motion_state": ExternalCameraMotionState(
                    row["camera_motion_state"]
                ),
                "score_components": ActiveFrontScoreComponents(
                    **row["score_components"]
                ),
                "base_from_external_camera_cv": (
                    None
                    if row["base_from_external_camera_cv"] is None
                    else np.asarray(
                        row["base_from_external_camera_cv"], dtype=np.float64
                    )
                ),
            }
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError("selection HOME baseline 类型/语义漂移") from error


def _serialize_baseline(value: PassiveBaselineEvidence) -> dict[str, Any]:
    return {
        "episode_id": value.episode_id,
        "episode_generation": value.episode_generation,
        "request_id": value.request_id,
        "timestamp_s": value.timestamp_s,
        "wrist_object_measurement_usable": value.wrist_object_measurement_usable,
        "wrist_evidence_identity_sha256": value.wrist_evidence_identity_sha256,
        "home_front": (
            None if value.home_front is None else _serialize_home_score(value.home_front)
        ),
        "object_memory_navigation_state_available": (
            value.object_memory_navigation_state_available
        ),
        "object_memory_age_s": value.object_memory_age_s,
        "object_memory_source_identity": value.object_memory_source_identity,
        "version": value.version,
    }


def _deserialize_baseline(value: Mapping[str, Any]) -> PassiveBaselineEvidence:
    expected = {
        "episode_id",
        "episode_generation",
        "request_id",
        "timestamp_s",
        "wrist_object_measurement_usable",
        "wrist_evidence_identity_sha256",
        "home_front",
        "object_memory_navigation_state_available",
        "object_memory_age_s",
        "object_memory_source_identity",
        "version",
    }
    row = _require_exact_keys(dict(value), expected, "selection passive baseline")
    try:
        return PassiveBaselineEvidence(
            **{
                **row,
                "home_front": (
                    None
                    if row["home_front"] is None
                    else _deserialize_home_score(row["home_front"])
                ),
            }
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError("selection passive baseline 类型/语义漂移") from error


def _serialize_primary_frame(value: ActiveFrontStage2FrameEvidence) -> dict[str, Any]:
    return {
        "episode_id": value.episode_id,
        "episode_generation": value.episode_generation,
        "request_id": value.request_id,
        "source_phase": value.source_phase.value,
        "observation_sequence_id": value.observation_sequence_id,
        "model_input_digest": value.model_input_digest,
        "provider_output_digest": value.provider_output_digest,
        "provider_identity": value.provider_identity.to_dict(),
        "camera_motion_state": value.camera_motion_state.value,
        "settled": value.settled,
        "control_timestamp_s": value.control_timestamp_s,
        "rgb_timestamp_s": value.rgb_timestamp_s,
        "camera_pose_timestamp_s": value.camera_pose_timestamp_s,
        "tcp_pose_timestamp_s": value.tcp_pose_timestamp_s,
        "base_from_external_camera_cv": value.base_from_external_camera_cv.tolist(),
        "position_base_m": (
            None if value.position_base_m is None else list(value.position_base_m)
        ),
        "covariance_base_m2": (
            None
            if value.covariance_base_m2 is None
            else [list(row) for row in value.covariance_base_m2]
        ),
        "measurement_confidence": value.measurement_confidence,
        "write_score": value.write_score,
        "score_components": asdict(value.score_components),
        "projection_valid": value.projection_valid,
        "in_fov": value.in_fov,
        "observable": value.observable,
        "geometry_valid": value.geometry_valid,
        "structurally_eligible": value.structurally_eligible,
        "deployable_free_static_safe": value.deployable_free_static_safe,
        "source_camera": value.source_camera,
        "actual_pose_source": value.actual_pose_source,
        "input_schema_version": value.input_schema_version,
        "score_semantics": value.score_semantics,
        "execution_mode": value.execution_mode,
        "qualification_only": value.qualification_only,
        "version": value.version,
    }


def _deserialize_primary_frame(
    value: Mapping[str, Any],
) -> ActiveFrontStage2FrameEvidence:
    expected = {
        "episode_id",
        "episode_generation",
        "request_id",
        "source_phase",
        "observation_sequence_id",
        "model_input_digest",
        "provider_output_digest",
        "provider_identity",
        "camera_motion_state",
        "settled",
        "control_timestamp_s",
        "rgb_timestamp_s",
        "camera_pose_timestamp_s",
        "tcp_pose_timestamp_s",
        "base_from_external_camera_cv",
        "position_base_m",
        "covariance_base_m2",
        "measurement_confidence",
        "write_score",
        "score_components",
        "projection_valid",
        "in_fov",
        "observable",
        "geometry_valid",
        "structurally_eligible",
        "deployable_free_static_safe",
        "source_camera",
        "actual_pose_source",
        "input_schema_version",
        "score_semantics",
        "execution_mode",
        "qualification_only",
        "version",
    }
    row = _require_exact_keys(dict(value), expected, "selection PRIMARY frame")
    try:
        return ActiveFrontStage2FrameEvidence(
            **{
                **row,
                "source_phase": PhaseId(row["source_phase"]),
                "provider_identity": ActiveFrontStage2ProviderIdentity(
                    **row["provider_identity"]
                ),
                "camera_motion_state": ExternalCameraMotionState(
                    row["camera_motion_state"]
                ),
                "base_from_external_camera_cv": np.asarray(
                    row["base_from_external_camera_cv"], dtype=np.float64
                ),
                "score_components": ActiveFrontScoreComponents(
                    **row["score_components"]
                ),
            }
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError("selection PRIMARY frame 类型/语义漂移") from error


@dataclass(frozen=True)
class CapturedSelectionRoute:
    """env/provider 销毁后仍可使用的 deployable-only route freeze。"""

    seed: int
    episode_id: str
    request: ActiveFrontReobserveRequest
    passive_baseline: PassiveBaselineEvidence
    primary_frames: tuple[ActiveFrontStage2FrameEvidence, ...]
    collect_memory_safety: tuple[ObjectMemorySafetyContext, ...]
    home_frames: tuple[HomeV2BarrierFrame, ...]
    home_timestamps_s: tuple[float, ...]
    home_memory_safety: tuple[ObjectMemorySafetyContext, ...]
    home_active_safety: tuple[ActiveFrontSafetyEvidence, ...]
    final_active_safety: ActiveFrontSafetyEvidence
    final_memory_safety: ObjectMemorySafetyContext
    source_recheck_evidence_digest: str
    source_recheck_timestamp_s: float
    return_home_timestamp_s: float
    home_evidence: tuple[Mapping[str, Any], ...]
    observation_v2_window_identity: Mapping[str, Any]
    route_protocol_safety_valid: bool
    physical_route_count: int
    captured_provider_forward_count: int
    raw_candidate_digest_at_gain_0_02: str | None
    raw_candidate_commit_eligible_at_gain_0_02: bool | None
    raw_candidate_rejection_reasons_at_gain_0_02: tuple[str, ...] | None

    def __post_init__(self) -> None:
        if self.seed not in STAGE2A_SELECTION_SEEDS:
            raise ValueError("captured route seed 不在 selection split")
        if (
            self.episode_id
            != f"e018-p1-stage2a-selection-development-seed-{self.seed}"
            or self.request.episode_id != self.episode_id
            or self.request.episode_generation != 1
        ):
            raise ValueError("captured route Episode/request identity 漂移")
        if (
            self.passive_baseline.episode_id != self.episode_id
            or self.passive_baseline.request_id != self.request.request_id
            or len(self.primary_frames) != 3
            or len(self.collect_memory_safety) != 3
            or len(self.home_frames) != 4
            or len(self.home_timestamps_s) != 4
            or len(self.home_memory_safety) != 4
            or len(self.home_active_safety) != 4
            or len(self.home_evidence) != 4
        ):
            raise ValueError("captured route replay evidence 数量/identity 漂移")
        if tuple(frame.control_timestamp_s for frame in self.primary_frames) != tuple(
            sorted(frame.control_timestamp_s for frame in self.primary_frames)
        ):
            raise ValueError("captured PRIMARY timestamps 未按固定顺序")
        if any(
            later <= earlier + 1e-12
            for earlier, later in zip(
                self.home_timestamps_s, self.home_timestamps_s[1:]
            )
        ):
            raise ValueError("captured HOME timestamps 必须严格递增")
        if self.physical_route_count != 1 or self.captured_provider_forward_count != 4:
            raise ValueError("每个 selection seed 必须恰好一条 route/四次 provider forward")
        if type(self.route_protocol_safety_valid) is not bool:
            raise TypeError("captured route protocol safety 必须是 exact bool")
        if not _is_sha256(self.source_recheck_evidence_digest):
            raise ValueError("source recheck evidence digest 非法")
        candidate_digest = self.raw_candidate_digest_at_gain_0_02
        candidate_eligible = self.raw_candidate_commit_eligible_at_gain_0_02
        candidate_reasons = self.raw_candidate_rejection_reasons_at_gain_0_02
        if candidate_digest is None:
            if candidate_eligible is not None or candidate_reasons is not None:
                raise ValueError("absent raw candidate 必须完整冻结为 None")
        else:
            if not _is_sha256(candidate_digest):
                raise ValueError("raw candidate digest 非法")
            if type(candidate_eligible) is not bool:
                raise TypeError("raw candidate eligibility 必须是 exact bool")
            if (
                type(candidate_reasons) is not tuple
                or any(
                    not isinstance(reason, str) or not reason
                    for reason in candidate_reasons
                )
                or len(set(candidate_reasons)) != len(candidate_reasons)
                or candidate_eligible != (not candidate_reasons)
            ):
                raise ValueError("raw candidate rejection reasons/eligibility 漂移")

    @classmethod
    def from_transaction_export(
        cls,
        value: Mapping[str, Any],
    ) -> CapturedSelectionRoute:
        source_recheck = value.get("source_recheck_record")
        if source_recheck is None:
            raise RuntimeError("selection transaction export 缺 source recheck")
        return cls(
            seed=int(value["seed"]),
            episode_id=str(value["episode_id"]),
            request=value["request"],
            passive_baseline=value["passive_baseline"],
            primary_frames=tuple(value["primary_frames"]),
            collect_memory_safety=tuple(value["collect_memory_safety"]),
            home_frames=tuple(value["home_frames"]),
            home_timestamps_s=tuple(float(item) for item in value["home_timestamps_s"]),
            home_memory_safety=tuple(value["home_memory_safety"]),
            home_active_safety=tuple(value["home_active_safety"]),
            final_active_safety=value["final_active_safety"],
            final_memory_safety=value["final_memory_safety"],
            source_recheck_evidence_digest=source_recheck.digest,
            source_recheck_timestamp_s=float(source_recheck.timestamp_s),
            return_home_timestamp_s=float(value["return_home_timestamp_s"]),
            home_evidence=tuple(value["home_evidence"]),
            observation_v2_window_identity=value[
                "observation_v2_window_identity"
            ],
            route_protocol_safety_valid=value["route_protocol_safety_valid"],
            physical_route_count=1,
            captured_provider_forward_count=int(value["provider_forward_count"]),
            raw_candidate_digest_at_gain_0_02=value["raw_candidate_digest"],
            raw_candidate_commit_eligible_at_gain_0_02=value[
                "raw_candidate_commit_eligible_at_gain_0_02"
            ],
            raw_candidate_rejection_reasons_at_gain_0_02=(
                None
                if value["raw_candidate_rejection_reasons_at_gain_0_02"] is None
                else tuple(value["raw_candidate_rejection_reasons_at_gain_0_02"])
            ),
        )

    def to_public_dict(self) -> dict[str, Any]:
        """冻结纯逻辑 replay 所需的 deployable-only typed evidence。"""

        row = {
            "version": E018_P1_STAGE2A_SELECTION_EXECUTION_VERSION,
            "seed": self.seed,
            "episode_id": self.episode_id,
            "request": _serialize_request(self.request),
            "passive_baseline": _serialize_baseline(self.passive_baseline),
            "primary_frames": [
                _serialize_primary_frame(value) for value in self.primary_frames
            ],
            "collect_memory_safety": [
                asdict(value) for value in self.collect_memory_safety
            ],
            "home_frames": [asdict(value) for value in self.home_frames],
            "home_timestamps_s": list(self.home_timestamps_s),
            "home_memory_safety": [
                asdict(value) for value in self.home_memory_safety
            ],
            "home_active_safety": [
                asdict(value) for value in self.home_active_safety
            ],
            "final_active_safety": asdict(self.final_active_safety),
            "final_memory_safety": asdict(self.final_memory_safety),
            "source_recheck_evidence_digest": self.source_recheck_evidence_digest,
            "source_recheck_timestamp_s": self.source_recheck_timestamp_s,
            "return_home_timestamp_s": self.return_home_timestamp_s,
            "home_evidence": [dict(value) for value in self.home_evidence],
            "observation_v2_window_identity": dict(
                self.observation_v2_window_identity
            ),
            "route_protocol_safety_valid": self.route_protocol_safety_valid,
            "physical_route_count": self.physical_route_count,
            "captured_provider_forward_count": self.captured_provider_forward_count,
            "raw_candidate_digest_at_gain_0_02": (
                self.raw_candidate_digest_at_gain_0_02
            ),
            "raw_candidate_commit_eligible_at_gain_0_02": (
                self.raw_candidate_commit_eligible_at_gain_0_02
            ),
            "raw_candidate_rejection_reasons_at_gain_0_02": (
                None
                if self.raw_candidate_rejection_reasons_at_gain_0_02 is None
                else list(self.raw_candidate_rejection_reasons_at_gain_0_02)
            ),
            "route_evidence_digest": self.route_evidence_digest,
        }
        row["route_row_sha256"] = canonical_sha256(row)
        return row

    @classmethod
    def from_public_dict(
        cls,
        value: Mapping[str, Any],
    ) -> CapturedSelectionRoute:
        expected = {
            "version",
            "seed",
            "episode_id",
            "request",
            "passive_baseline",
            "primary_frames",
            "collect_memory_safety",
            "home_frames",
            "home_timestamps_s",
            "home_memory_safety",
            "home_active_safety",
            "final_active_safety",
            "final_memory_safety",
            "source_recheck_evidence_digest",
            "source_recheck_timestamp_s",
            "return_home_timestamp_s",
            "home_evidence",
            "observation_v2_window_identity",
            "route_protocol_safety_valid",
            "physical_route_count",
            "captured_provider_forward_count",
            "raw_candidate_digest_at_gain_0_02",
            "raw_candidate_commit_eligible_at_gain_0_02",
            "raw_candidate_rejection_reasons_at_gain_0_02",
            "route_evidence_digest",
            "route_row_sha256",
        }
        row = _require_exact_keys(dict(value), expected, "selection route evidence")
        unsigned = dict(row)
        route_row_sha256 = unsigned.pop("route_row_sha256")
        if (
            row["version"] != E018_P1_STAGE2A_SELECTION_EXECUTION_VERSION
            or route_row_sha256 != canonical_sha256(unsigned)
        ):
            raise RuntimeError("selection route evidence row identity 漂移")
        try:
            route = cls(
                seed=row["seed"],
                episode_id=row["episode_id"],
                request=_deserialize_request(row["request"]),
                passive_baseline=_deserialize_baseline(row["passive_baseline"]),
                primary_frames=tuple(
                    _deserialize_primary_frame(item)
                    for item in row["primary_frames"]
                ),
                collect_memory_safety=tuple(
                    ObjectMemorySafetyContext(**item)
                    for item in row["collect_memory_safety"]
                ),
                home_frames=tuple(
                    HomeV2BarrierFrame(**item) for item in row["home_frames"]
                ),
                home_timestamps_s=tuple(row["home_timestamps_s"]),
                home_memory_safety=tuple(
                    ObjectMemorySafetyContext(**item)
                    for item in row["home_memory_safety"]
                ),
                home_active_safety=tuple(
                    ActiveFrontSafetyEvidence(**item)
                    for item in row["home_active_safety"]
                ),
                final_active_safety=ActiveFrontSafetyEvidence(
                    **row["final_active_safety"]
                ),
                final_memory_safety=ObjectMemorySafetyContext(
                    **row["final_memory_safety"]
                ),
                source_recheck_evidence_digest=row[
                    "source_recheck_evidence_digest"
                ],
                source_recheck_timestamp_s=row["source_recheck_timestamp_s"],
                return_home_timestamp_s=row["return_home_timestamp_s"],
                home_evidence=tuple(row["home_evidence"]),
                observation_v2_window_identity=row[
                    "observation_v2_window_identity"
                ],
                route_protocol_safety_valid=row[
                    "route_protocol_safety_valid"
                ],
                physical_route_count=row["physical_route_count"],
                captured_provider_forward_count=row[
                    "captured_provider_forward_count"
                ],
                raw_candidate_digest_at_gain_0_02=row[
                    "raw_candidate_digest_at_gain_0_02"
                ],
                raw_candidate_commit_eligible_at_gain_0_02=row[
                    "raw_candidate_commit_eligible_at_gain_0_02"
                ],
                raw_candidate_rejection_reasons_at_gain_0_02=(
                    None
                    if row["raw_candidate_rejection_reasons_at_gain_0_02"]
                    is None
                    else tuple(
                        row["raw_candidate_rejection_reasons_at_gain_0_02"]
                    )
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("selection route evidence 类型/状态语义漂移") from error
        if (
            route.route_evidence_digest != row["route_evidence_digest"]
            or route.to_public_dict() != row
        ):
            raise RuntimeError("selection route evidence 不能从 typed evidence 重算")
        return route

    @property
    def route_evidence_digest(self) -> str:
        """绑定 replay 实际消费的完整 typed 输入，而非选择性摘要。"""

        payload = {
            "version": E018_P1_STAGE2A_SELECTION_EXECUTION_VERSION,
            "seed": self.seed,
            "episode_id": self.episode_id,
            "request": _serialize_request(self.request),
            "passive_baseline": _serialize_baseline(self.passive_baseline),
            "physical_route_count": self.physical_route_count,
            "captured_provider_forward_count": self.captured_provider_forward_count,
            "primary_frames": [
                _serialize_primary_frame(frame) for frame in self.primary_frames
            ],
            "collect_memory_safety": [
                asdict(value) for value in self.collect_memory_safety
            ],
            "home_frames": [asdict(value) for value in self.home_frames],
            "home_timestamps_s": list(self.home_timestamps_s),
            "home_memory_safety": [
                asdict(value) for value in self.home_memory_safety
            ],
            "home_active_safety": [
                asdict(value) for value in self.home_active_safety
            ],
            "final_active_safety": asdict(self.final_active_safety),
            "final_memory_safety": asdict(self.final_memory_safety),
            "source_recheck_evidence_digest": self.source_recheck_evidence_digest,
            "source_recheck_timestamp_s": self.source_recheck_timestamp_s,
            "return_home_timestamp_s": self.return_home_timestamp_s,
            "home_evidence": [dict(value) for value in self.home_evidence],
            "observation_v2_window_identity": dict(
                self.observation_v2_window_identity
            ),
            "route_protocol_safety_valid": self.route_protocol_safety_valid,
            "raw_candidate_digest_at_gain_0_02": (
                self.raw_candidate_digest_at_gain_0_02
            ),
            "raw_candidate_commit_eligible_at_gain_0_02": (
                self.raw_candidate_commit_eligible_at_gain_0_02
            ),
            "raw_candidate_rejection_reasons_at_gain_0_02": (
                self.raw_candidate_rejection_reasons_at_gain_0_02
            ),
        }
        return canonical_sha256(payload)


def _begin_gain_replay(
    captured: CapturedSelectionRoute,
    gain: float,
) -> tuple[
    ExplicitObjectStateMemory,
    ActiveFrontStage2MemoryOrchestrator,
    Stage2AActionHistoryRuntime,
]:
    """重建一个 gain 的 pre-HOME 状态；不接 env/provider/label。"""

    if gain not in STAGE2A_SELECTION_GAINS:
        raise ValueError("gain 不属于冻结候选")
    memory = ExplicitObjectStateMemory(
        build_stage2_object_memory_config(d049_primary_provider_identity())
    )
    orchestrator = ActiveFrontStage2MemoryOrchestrator(
        memory,
        config=ActiveFrontStage2Config.development(
            min_information_gain=gain,
            information_gain_comparison_tolerance=0.0,
        ),
    )
    orchestrator.reset_episode(
        captured.episode_id,
        episode_generation=1,
        timestamp_s=0.0,
    )
    action_history = Stage2AActionHistoryRuntime(captured.episode_id)
    reset_receipt, _ = action_history.invalidate_for_active_request(
        captured.request
    )
    orchestrator.begin_collection(
        captured.request,
        reset_receipt=reset_receipt,
        baseline=captured.passive_baseline,
    )
    for frame, safety in zip(
        captured.primary_frames,
        captured.collect_memory_safety,
        strict=True,
    ):
        if orchestrator.state is PendingActiveViewState.COLLECTING:
            orchestrator.observe_collect_frame(frame, safety=safety)
    return memory, orchestrator, action_history


def replay_gain_branch(
    captured: CapturedSelectionRoute,
    gain: float,
) -> GainBranchOutcome:
    """从全新 Memory/Action state 重放一个 gain；不接 env/provider/label。"""

    memory, orchestrator, action_history = _begin_gain_replay(captured, gain)
    candidate = orchestrator.pending_candidate
    candidate_digest = None if candidate is None else candidate.digest
    candidate_commit_eligible = bool(
        candidate is not None and candidate.commit_eligible
    )
    if gain == STAGE2A_SELECTION_GAINS[0] and (
        (candidate is None)
        != (captured.raw_candidate_digest_at_gain_0_02 is None)
        or candidate_digest != captured.raw_candidate_digest_at_gain_0_02
        or (None if candidate is None else candidate.commit_eligible)
        is not captured.raw_candidate_commit_eligible_at_gain_0_02
        or (None if candidate is None else candidate.rejection_reasons)
        != captured.raw_candidate_rejection_reasons_at_gain_0_02
    ):
        raise RuntimeError(
            "gain=0.02 reconstructed candidate 与 capture-time raw identity 漂移"
        )
    orchestrator.mark_returning_home(
        timestamp_s=captured.return_home_timestamp_s,
        candidate_digest=candidate_digest,
    )
    for frame, timestamp_s, safety in zip(
        captured.home_frames,
        captured.home_timestamps_s,
        captured.home_active_safety,
        strict=True,
    ):
        orchestrator.accept_home_v2_barrier_frame(
            frame,
            timestamp_s=timestamp_s,
            safety=safety,
        )
    if orchestrator.state is PendingActiveViewState.HOME_BARRIER_PASSED:
        if candidate is None:
            raise RuntimeError("HOME barrier passed 却缺 candidate")
        recheck = ActiveFrontSourceRecheckEvidence(
            episode_id=captured.episode_id,
            episode_generation=1,
            request_id=captured.request.request_id,
            candidate_digest=candidate.digest,
            timestamp_s=captured.source_recheck_timestamp_s,
            source_phase=captured.request.source_phase,
            camera_at_home=True,
            source_invariants_passed=captured.route_protocol_safety_valid,
            active_window_open=captured.final_active_safety.active_window_open,
            qualified_direct_wrist_measurement_usable=False,
            qualified_direct_wrist_evidence_identity_sha256=(
                captured.source_recheck_evidence_digest
            ),
        )
        if orchestrator.recheck_source(
            recheck,
            safety=captured.final_active_safety,
        ):
            commit_receipt = orchestrator.commit(
                candidate_digest=candidate.digest,
                commit_timestamp_s=captured.source_recheck_timestamp_s + 0.001,
                safety=captured.final_memory_safety,
            )
            resume_receipt, _ = action_history.generate_fresh_shadow_replan(
                captured.request,
                home_evidence=captured.home_evidence,
                observation_v2_window_identity=(
                    captured.observation_v2_window_identity
                ),
                memory_state=memory.state,
                source_phase=captured.request.source_phase,
            )
            orchestrator.create_shadow_action_generation(
                resume_receipt,
                source_phase=captured.request.source_phase,
                source_phase_stability_reset=True,
                source_phase_stability_ticks=0,
            )
            if commit_receipt.memory_write_count != 1:
                raise RuntimeError("gain branch commit count 漂移")
    commit_count = orchestrator.memory_write_count
    action_count = 1 if orchestrator.shadow_action_receipt is not None else 0
    navigation_available = False
    if orchestrator.commit_update is not None:
        navigation_available = resolve_object_state(
            orchestrator.commit_update,
            requirement=ObjectStateRequirement.NAVIGATION,
        ).available
    position = (
        memory.state.position_base_m
        if commit_count == 1
        and memory.state.mode is ObjectMemoryMode.FREE_STATIC
        else None
    )
    return GainBranchOutcome(
        seed=captured.seed,
        gain=gain,
        route_evidence_digest=captured.route_evidence_digest,
        route_protocol_safety_valid=captured.route_protocol_safety_valid,
        candidate_commit_eligible=candidate_commit_eligible,
        memory_commit_count=commit_count,
        navigation_state_available=navigation_available,
        fresh_shadow_action_generation_count=action_count,
        committed_position_base_m=position,
        provider_forward_count=0,
        protocol_violation_count=int(not captured.route_protocol_safety_valid),
    )


def replay_all_gain_branches(
    captured: CapturedSelectionRoute,
    *,
    gain_order: Sequence[float] = STAGE2A_SELECTION_GAINS,
) -> tuple[GainBranchOutcome, ...]:
    """每个 gain 都新建独立状态；输出统一按冻结 gain 顺序归一。"""

    if sorted(gain_order) != sorted(STAGE2A_SELECTION_GAINS) or len(gain_order) != 3:
        raise ValueError("gain replay order 必须恰好包含三个冻结候选")
    by_gain = {gain: replay_gain_branch(captured, gain) for gain in gain_order}
    return tuple(by_gain[gain] for gain in STAGE2A_SELECTION_GAINS)


def _validate_private_label_for_scoring(
    label: Mapping[str, Any],
    *,
    expected_label_index: int,
) -> None:
    expected_augmented_keys = _SELECTION_PRIVATE_CAPTURE_KEYS | {
        "version",
        "label_index",
        "prediction_row_index",
        "seed",
        "route_frame_index",
        "rgb_sha256",
        "actual_pose_sha256",
        "provider_output_digest",
        "prediction_commit_receipt_sha256",
        "transaction_identity_sha256",
        "motion_predicate_version",
        "motion_linear_threshold_m_s",
        "motion_angular_threshold_rad_s",
        "contact_threshold_n",
        "privileged_captured_at_unix_ns",
        "label_sha256",
    }
    if set(label) != expected_augmented_keys:
        raise RuntimeError("selection scored private label exact keys 漂移")
    qualification_label = {
        key: label[key] for key in _QUALIFICATION_OBJECT_LABEL_KEYS
    }
    _validate_qualification_object_label(qualification_label, committed=False)
    try:
        linear_speed = float(label["object_linear_speed_m_s"])
        angular_speed = float(label["object_angular_speed_rad_s"])
        captured_at = int(label["privileged_captured_at_unix_ns"])
    except (TypeError, ValueError) as error:
        raise RuntimeError("selection private label numeric primitive 非法") from error
    seed = STAGE2A_SELECTION_SEEDS[expected_label_index // 3]
    frame = STAGE2A_COLLECT_FRAME_INDICES[expected_label_index % 3]
    unsigned = dict(label)
    internal = unsigned.pop("label_sha256", None)
    if (
        label.get("label_index") != expected_label_index
        or label.get("seed") != seed
        or label.get("route_frame_index") != frame
        or label.get("prediction_row_index") != (expected_label_index // 3) * 4 + 1 + expected_label_index % 3
        or internal != canonical_sha256(unsigned)
        or label.get("goal_gt_read_count") != 0
        or label.get("test_data_read") is not False
        or label.get("motion_predicate_version") != "pick-and-place-predicates/v1"
        or label.get("motion_linear_threshold_m_s") != 0.01
        or label.get("motion_angular_threshold_rad_s") != 0.5
        or label.get("contact_threshold_n") != 0.01
        or not math.isfinite(linear_speed)
        or linear_speed < 0.0
        or not math.isfinite(angular_speed)
        or angular_speed < 0.0
        or label.get("object_motion_event")
        is not bool(linear_speed > 0.01 or angular_speed > 0.5)
        or captured_at <= 0
        or any(
            not _is_sha256(label.get(name))
            for name in (
                "rgb_sha256",
                "actual_pose_sha256",
                "provider_output_digest",
                "prediction_commit_receipt_sha256",
                "transaction_identity_sha256",
            )
        )
    ):
        raise RuntimeError("selection private label identity/order/hash 漂移")


def _validated_gain_branch(value: Mapping[str, Any]) -> dict[str, Any]:
    expected_keys = {
        "seed",
        "gain",
        "route_evidence_digest",
        "route_protocol_safety_valid",
        "candidate_commit_eligible",
        "memory_commit_count",
        "navigation_state_available",
        "fresh_shadow_action_generation_count",
        "committed_position_base_m",
        "provider_forward_count",
        "arm_motion_command_count",
        "gripper_close_command_count",
        "protocol_violation_count",
        "branch_sha256",
    }
    payload = _require_exact_keys(dict(value), expected_keys, "gain branch")
    try:
        branch = GainBranchOutcome(
            seed=payload["seed"],
            gain=payload["gain"],
            route_evidence_digest=payload["route_evidence_digest"],
            route_protocol_safety_valid=payload["route_protocol_safety_valid"],
            candidate_commit_eligible=payload["candidate_commit_eligible"],
            memory_commit_count=payload["memory_commit_count"],
            navigation_state_available=payload["navigation_state_available"],
            fresh_shadow_action_generation_count=payload[
                "fresh_shadow_action_generation_count"
            ],
            committed_position_base_m=(
                None
                if payload["committed_position_base_m"] is None
                else tuple(payload["committed_position_base_m"])
            ),
            provider_forward_count=payload["provider_forward_count"],
            arm_motion_command_count=payload["arm_motion_command_count"],
            gripper_close_command_count=payload["gripper_close_command_count"],
            protocol_violation_count=payload["protocol_violation_count"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("gain branch 类型/状态语义漂移") from error
    normalized = branch.to_dict()
    if normalized != payload:
        raise RuntimeError("gain branch exact fields/hash 不能重算")
    return normalized


def score_gain_branches(
    branches: Sequence[Mapping[str, Any]],
    private_labels: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pass B 核心：只消费冻结 branch 和 exact-once private labels。"""

    if len(branches) != STAGE2A_SELECTION_BRANCH_COUNT:
        raise RuntimeError("selection scoring 必须有 75 个 gain branches")
    if len(private_labels) != STAGE2A_SELECTION_LABEL_COUNT:
        raise RuntimeError("selection scoring 必须有 75 个 private labels")
    verified_branches = [_validated_gain_branch(value) for value in branches]
    for index, label in enumerate(private_labels):
        _validate_private_label_for_scoring(label, expected_label_index=index)

    labels_by_seed = {
        seed: private_labels[offset * 3 : offset * 3 + 3]
        for offset, seed in enumerate(STAGE2A_SELECTION_SEEDS)
    }
    oracle_by_seed: dict[int, bool] = {}
    for offset, seed in enumerate(STAGE2A_SELECTION_SEEDS):
        seed_branches = verified_branches[offset * 3 : offset * 3 + 3]
        route_valid_values = {
            value.get("route_protocol_safety_valid") for value in seed_branches
        }
        route_digests = {value.get("route_evidence_digest") for value in seed_branches}
        if (
            route_valid_values not in ({True}, {False})
            or len(route_digests) != 1
            or not _is_sha256(next(iter(route_digests)))
        ):
            raise RuntimeError("三个 gain 未绑定同一 route/protocol/safety evidence")
        labels = labels_by_seed[seed]
        oracle_by_seed[seed] = bool(
            next(iter(route_valid_values))
            and all(label.get("gt_object_exists") is True for label in labels)
            and all(label.get("gt_observable") is True for label in labels)
            and all(
                float(label.get("robot_object_contact_force_n")) <= 0.01
                for label in labels
            )
            and all(label.get("object_motion_event") is False for label in labels)
            and all(label.get("is_grasped") is False for label in labels)
        )
    scored: list[dict[str, Any]] = []
    for index, branch in enumerate(verified_branches):
        seed = STAGE2A_SELECTION_SEEDS[index // 3]
        gain = STAGE2A_SELECTION_GAINS[index % 3]
        if branch.get("seed") != seed or branch.get("gain") != gain:
            raise RuntimeError("selection gain branch seed/gain order 漂移")
        labels = labels_by_seed[seed]
        oracle_eligible = oracle_by_seed[seed]
        committed = branch.get("memory_commit_count") == 1
        error_m: float | None = None
        if committed:
            predicted = np.asarray(
                branch.get("committed_position_base_m"), dtype=np.float64
            )
            target = np.asarray(
                labels[-1].get("gt_object_position_base_m"), dtype=np.float64
            )
            if predicted.shape != (3,) or target.shape != (3,):
                raise RuntimeError("selection scoring XYZ shape 漂移")
            error_m = float(np.linalg.norm(predicted - target))
        recovered = bool(
            oracle_eligible
            and committed
            and branch.get("navigation_state_available") is True
            and branch.get("fresh_shadow_action_generation_count") == 1
            and error_m is not None
            and error_m <= 0.005
        )
        committed_without_oracle_evidence = bool(
            branch.get("route_protocol_safety_valid") is not True
            or any(label.get("gt_object_exists") is not True for label in labels)
            or any(label.get("gt_observable") is not True for label in labels)
            or any(
                float(label.get("robot_object_contact_force_n")) > 0.01
                or label.get("object_motion_event") is True
                or label.get("is_grasped") is True
                for label in labels
            )
        )
        false_recovery = bool(
            committed
            and (
                committed_without_oracle_evidence
                or error_m is None
                or error_m > 0.005
            )
        )
        catastrophic = bool(committed and error_m is not None and error_m > 0.020)
        unsafe = bool(
            false_recovery
            or branch.get("arm_motion_command_count") != 0
            or branch.get("gripper_close_command_count") != 0
        )
        row = {
            "version": E018_P1_STAGE2A_SELECTION_RESULT_VERSION,
            "seed": seed,
            "gain": gain,
            "oracle_common_denominator_eligible": oracle_eligible,
            "memory_commit_count": branch.get("memory_commit_count"),
            "navigation_state_available": branch.get("navigation_state_available"),
            "fresh_shadow_action_generation_count": branch.get(
                "fresh_shadow_action_generation_count"
            ),
            "xyz_error_m": error_m,
            "recovered": recovered,
            "false_recovery": false_recovery,
            "catastrophic_recovery": catastrophic,
            "unsafe_recovery": unsafe,
            "protocol_violation_count": branch.get("protocol_violation_count"),
        }
        row["scored_row_sha256"] = canonical_sha256(row)
        scored.append(row)

    denominator_seeds = {
        row["seed"]
        for row in scored
        if row["oracle_common_denominator_eligible"]
    }
    denominator = len(denominator_seeds)
    per_gain: list[dict[str, Any]] = []
    for gain in STAGE2A_SELECTION_GAINS:
        rows = [row for row in scored if row["gain"] == gain]
        summary = {
            "gain": gain,
            "common_denominator_count": denominator,
            "recovered_count": sum(row["recovered"] for row in rows),
            "false_recovery_count": sum(row["false_recovery"] for row in rows),
            "catastrophic_recovery_count": sum(
                row["catastrophic_recovery"] for row in rows
            ),
            "unsafe_recovery_count": sum(row["unsafe_recovery"] for row in rows),
            "protocol_violation_count": sum(
                int(row["protocol_violation_count"]) for row in rows
            ),
        }
        summary["eligible"] = bool(
            summary["false_recovery_count"] == 0
            and summary["catastrophic_recovery_count"] == 0
            and summary["unsafe_recovery_count"] == 0
            and summary["protocol_violation_count"] == 0
        )
        per_gain.append(summary)
    eligible = [row for row in per_gain if row["eligible"]]
    selected_gain: float | None
    selection_reason: str
    if denominator < STAGE2A_SELECTION_MIN_SUPPORT:
        selected_gain = None
        selection_reason = "insufficient-common-denominator-support"
    elif not eligible:
        selected_gain = None
        selection_reason = "no-safe-eligible-gain"
    else:
        # denominator 对所有 gain 完全相同，因此整数 recovered_count 足以做
        # exact comparison；同 count 的较大 gain 由 tuple 第二项决定。
        selected = max(eligible, key=lambda row: (row["recovered_count"], row["gain"]))
        selected_gain = float(selected["gain"])
        selection_reason = "max-recovered-count-then-larger-gain"
    summary = {
        "version": E018_P1_STAGE2A_SELECTION_RESULT_VERSION,
        "status": "complete-development-selection",
        "classification": "formal-development-selection-no-test-no-actuation/v1",
        "effect_claim": "no-effect-claim",
        "common_denominator_count": denominator,
        "minimum_support_required": STAGE2A_SELECTION_MIN_SUPPORT,
        "per_gain": per_gain,
        "selected_gain": selected_gain,
        "selection_reason": selection_reason,
        "evaluation_config_generation_allowed": selected_gain is not None,
        "stage2b_continuation_required": True,
        "fresh_test_reads": 0,
        "runtime_object_gt_reads": 0,
        "goal_gt_reads": 0,
        "arm_motion_command_count": 0,
        "gripper_close_command_count": 0,
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    return scored, summary


__all__ = [
    "E018_P1_STAGE2A_SELECTION_CONFIG_VERSION",
    "E018_P1_STAGE2A_SELECTION_EXECUTION_VERSION",
    "E018_P1_STAGE2A_SELECTION_RESULT_VERSION",
    "STAGE2A_SELECTION_BRANCH_COUNT",
    "STAGE2A_SELECTION_GAINS",
    "STAGE2A_SELECTION_GO",
    "STAGE2A_SELECTION_LABEL_COUNT",
    "STAGE2A_SELECTION_PREDICTION_COUNT",
    "GainBranchOutcome",
    "LoadedStage2ASelectionConfig",
    "SelectionExecutionProgress",
    "Stage2ASelectionJournal",
    "load_e018_p1_stage2a_selection_config",
    "score_gain_branches",
]
