"""E018-P1 Stage 2A：PRIMARY 主动观察的隔离仿真 integration runner。

本模块只实现 D049/D050 已放行的 ``76901..76910`` engineering smoke。
它不读取在线 object/goal GT，不打开离线 oracle label，也不把缺少合格 wrist
provider 写成“看不清”。HOME 输出始终只是 raw-score baseline。
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import time
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robot_vla.adapters import FrankaObservationAdapter
from robot_vla.contracts import RobotSpec
from robot_vla.executive.contracts import PhaseId
from robot_vla.observation import (
    OBSERVATION_MODALITIES,
    ObservationV2Frame,
    ObservationV2History,
    ObservationV2Window,
    invert_se3,
    opengl_camera_to_opencv,
    rotation_matrix_to_6d,
    validate_se3,
)
from robot_vla.precision import e018_p1_g0 as _g0
from robot_vla.precision import e018_p1_g0c as _g0c
from robot_vla.precision.active_front_camera import ExternalCameraMotionState
from robot_vla.precision.active_front_memory import (
    ActiveFrontSourceRecheckEvidence,
    ActiveFrontStage2MemoryOrchestrator,
    PendingActiveViewState,
)
from robot_vla.precision.active_front_memory_provider import (
    ACTIVE_FRONT_HOME_BASE_FROM_EXTERNAL_CAMERA_CV,
    ACTIVE_FRONT_HOME_ORIENTATION_TOLERANCE_RAD,
    ACTIVE_FRONT_HOME_POSITION_TOLERANCE_M,
    ACTIVE_FRONT_HOME_PRIMITIVE_ID,
    ACTIVE_FRONT_INFORMATION_GAIN_CANDIDATES,
    ACTIVE_FRONT_PRIMARY_PRIMITIVE_ID,
    ACTIVE_FRONT_SCORE_SEMANTICS,
    ActiveFrontScoreComponents,
    ActiveFrontStage2Config,
    ActiveFrontStage2FrameEvidence,
    ActiveFrontStage2ProviderIdentity,
    PassiveBaselineEvidence,
    PassiveHomeScoreEvidence,
    build_stage2_object_memory_config,
    d049_home_baseline_provider_identity,
    d049_primary_provider_identity,
)
from robot_vla.precision.active_front_reobserve import (
    ActionHistoryResetReceipt,
    ActionHistoryResumeReceipt,
    ActiveFrontReobserveConfig,
    ActiveFrontReobserveController,
    ActiveFrontReobserveReceipt,
    ActiveFrontReobserveState,
    ActiveFrontSafetyEvidence,
    ActiveFrontSignal,
    ActiveFrontTriggerEvidence,
    ActiveFrontTriggerReason,
    ExternalCameraControllerOwner,
    HomeV2BarrierFrame,
    Stage2MemoryCandidateReceipt,
)
from robot_vla.precision.calibrated_front_provider import canonical_sha256
from robot_vla.precision.e018_p1_g2a import file_sha256
from robot_vla.precision.e018_p1_g2c_data import (
    FRONT_ALTERNATE_IDS,
    FRONT_HOME_ID,
    _load_normalizers,
    load_e018_p1_g2c_data_config,
)
from robot_vla.precision.e018_p1_g2c_qualification import (
    QUALIFICATION_CLASSIFICATION_SMOKE,
    QualificationProvider,
    build_qualification_deployable_capture,
    load_g2c_dynamic_qualification_config,
    validate_qualification_prediction_mechanics,
    verify_g2c_qualification_result,
)
from robot_vla.precision.e018_p1_g2c_training import _git_source_identity
from robot_vla.precision.object_memory import (
    ExplicitObjectStateMemory,
    ObjectMemoryMode,
    ObjectMemorySafetyContext,
    ObjectState,
)

E018_P1_STAGE2A_CONFIG_VERSION = (
    "e018-p1-stage2a-primary-memory-development/v1"
)
E018_P1_STAGE2A_EXECUTION_VERSION = (
    "e018-p1-stage2a-primary-memory-integration-smoke/v1"
)
E018_P1_STAGE2A_PROVIDER_RECORD_VERSION = (
    "e018-p1-stage2a-canonical-provider-output/v1"
)
E018_P1_STAGE2A_WRIST_CAPABILITY_VERSION = (
    "e018-p1-stage2a-wrist-capability-evidence/v1"
)
WRIST_CAPABILITY_ABSENT_STATUS = "NO_QUALIFIED_WRIST_PROVIDER_IN_D049_PARENT"
STAGE2A_INTEGRATION_SMOKE_SEEDS = tuple(range(76901, 76911))
STAGE2A_COLLECT_FRAME_INDICES = (45, 46, 47)
_STAGE2A_FAILURE_TRACEBACK_MAX_CHARS = 8192
_STAGE2A_FAILURE_ERROR_MAX_CHARS = 1024


@dataclass
class Stage2AExecutionProgress:
    """失败路径只保留可恢复控制进度，不包含 RGB、GT 或私有 label。"""

    current_seed: int | None = None
    episode_id: str | None = None
    request_id: str | None = None
    current_frame_index: int | None = None
    last_processed_frame_index: int | None = None
    last_authorized_frame_index: int | None = None
    controller_state: str | None = None
    orchestrator_state: str | None = None
    provider_forward_count: int = 0
    memory_write_count: int = 0

    def begin_seed(self, seed: int) -> None:
        if seed not in STAGE2A_INTEGRATION_SMOKE_SEEDS:
            raise ValueError("Stage 2A progress 只接受 integration smoke seed")
        self.current_seed = seed
        self.episode_id = _stage2a_episode_id(seed)
        self.request_id = None
        self.current_frame_index = None
        self.last_processed_frame_index = None
        self.last_authorized_frame_index = None
        self.controller_state = None
        self.orchestrator_state = None
        self.provider_forward_count = 0
        self.memory_write_count = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "current_seed": self.current_seed,
            "episode_id": self.episode_id,
            "request_id": self.request_id,
            "current_frame_index": self.current_frame_index,
            "last_processed_frame_index": self.last_processed_frame_index,
            "last_authorized_frame_index": self.last_authorized_frame_index,
            "controller_state": self.controller_state,
            "orchestrator_state": self.orchestrator_state,
            "provider_forward_count": self.provider_forward_count,
            "memory_write_count": self.memory_write_count,
        }
STAGE2A_HOME_BARRIER_FRAME_INDICES = (88, 89, 90, 91)
STAGE2A_PROVIDER_FRAME_INDICES = (0, *STAGE2A_COLLECT_FRAME_INDICES)
STAGE2A_TRIGGER_WARMUP_INDICES = (2, 3, 4)
STAGE2A_SOURCE_PHASE = PhaseId.ACQUIRE_TRACK
STAGE2A_CAMERA_OWNER = "ACTIVE_REOBSERVE_STAGE2A_INTEGRATION_SMOKE"
_D049_GATE_COMMIT = "de48f1305098c86d7d49ab4a487eb1f36aea544c"
_D049_HOME_CLARIFICATION_COMMIT = "22d2719c2614dee2b02ebf396a55817b644810aa"
_D050_ABSENT_WRIST_CAPABILITY_COMMIT = (
    "7f31a1392d64d71f51115d2b9e28a7f8be3b3260"
)
_D050_EXPERIMENT_ID = "E018-P1-S2A-ABSENT-WRIST-CAPABILITY-BASELINE/v1"
_STAGE2A_CONFIG_TOP_LEVEL_KEYS = {
    "version",
    "status",
    "decision",
    "parents",
    "provider",
    "candidate",
    "information_gain",
    "home_baseline",
    "memory",
    "execution",
    "splits",
    "budgets",
    "permissions",
}


def _require_exact_keys(
    value: Any,
    expected: set[str],
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(f"{name} keys 漂移: {actual}")
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _array_sha256(value: Any) -> str:
    array = np.asarray(value)
    if not np.isfinite(array).all() and np.issubdtype(array.dtype, np.number):
        raise ValueError("array digest 不接受非有限数值")
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(contiguous.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class LoadedStage2AConfig:
    """保留 raw/canonical 双身份，避免 runner 接受临时修改的 config。"""

    canonical_json: str
    raw_sha256: str
    canonical_sha256: str

    def __post_init__(self) -> None:
        if not _is_sha256(self.raw_sha256) or not _is_sha256(self.canonical_sha256):
            raise ValueError("Stage 2A config SHA-256 非法")
        payload = json.loads(self.canonical_json)
        if _canonical_json(payload) != self.canonical_json:
            raise ValueError("Stage 2A config canonical JSON 漂移")
        if canonical_sha256(payload) != self.canonical_sha256:
            raise ValueError("Stage 2A config canonical SHA-256 漂移")

    @property
    def payload(self) -> dict[str, Any]:
        return json.loads(self.canonical_json)

    @property
    def runtime_config(self) -> ActiveFrontStage2Config:
        config = self.payload
        return ActiveFrontStage2Config.development(
            min_information_gain=float(
                config["information_gain"][
                    "non_promotional_integration_smoke_provisional_value"
                ]
            )
        )


def _validate_stage2a_config_payload(config: dict[str, Any]) -> None:
    _require_exact_keys(config, _STAGE2A_CONFIG_TOP_LEVEL_KEYS, "Stage 2A config")
    if (
        config["version"] != E018_P1_STAGE2A_CONFIG_VERSION
        or config["status"]
        != "implementation-and-integration-smoke-go-development-only"
    ):
        raise ValueError("Stage 2A config version/status 漂移")
    decision = _require_exact_keys(
        config["decision"],
        {
            "gate",
            "d049_gate_commit",
            "d049_home_clarification_commit",
            "d050_absent_wrist_capability_commit",
            "d050_experiment_id",
            "fresh_test",
            "canonical_runtime",
            "physical_and_manipulation_actuator",
        },
        "Stage 2A decision",
    )
    if decision != {
        "gate": "D049",
        "d049_gate_commit": _D049_GATE_COMMIT,
        "d049_home_clarification_commit": _D049_HOME_CLARIFICATION_COMMIT,
        "d050_absent_wrist_capability_commit": (
            _D050_ABSENT_WRIST_CAPABILITY_COMMIT
        ),
        "d050_experiment_id": _D050_EXPERIMENT_ID,
        "fresh_test": "HOLD",
        "canonical_runtime": "HOLD",
        "physical_and_manipulation_actuator": "HOLD",
    }:
        raise ValueError("Stage 2A Decision identity/权限漂移")

    primary = d049_primary_provider_identity()
    home = d049_home_baseline_provider_identity()
    parents = config["parents"]
    parent_expectations = {
        "d048_artifact_id": primary.qualification_artifact_id,
        "d048_source_identity_sha256": primary.qualification_source_identity_sha256,
        "d048_qualification_config_raw_sha256": primary.qualification_config_raw_sha256,
        "d048_qualification_config_internal_sha256": (
            primary.qualification_config_internal_sha256
        ),
        "d048_result_receipt_raw_sha256": (
            primary.qualification_result_receipt_raw_sha256
        ),
        "d048_result_receipt_internal_sha256": (
            primary.qualification_result_receipt_internal_sha256
        ),
        "d048_result_verification_sha256": (
            primary.qualification_result_verification_sha256
        ),
        "d046_calibration_config_raw_sha256": primary.calibration_config_raw_sha256,
        "d046_calibration_config_internal_sha256": (
            primary.calibration_config_internal_sha256
        ),
        "d046_calibration_result_receipt_raw_sha256": (
            primary.calibration_result_receipt_raw_sha256
        ),
        "d046_calibration_result_receipt_internal_sha256": (
            primary.calibration_result_receipt_internal_sha256
        ),
        "d046_calibration_viewpoints_raw_sha256": (
            primary.calibration_viewpoints_raw_sha256
        ),
    }
    if parents != parent_expectations:
        raise ValueError("Stage 2A D048/D046 parent identity 漂移")

    provider = config["provider"]
    provider_expectations = {
        "source_training_camera": primary.source_training_camera,
        "source_camera": primary.source_camera,
        "primary_primitive_id": primary.primitive_id,
        "candidate_id": primary.candidate_id,
        "checkpoint_epoch": primary.checkpoint_epoch,
        "checkpoint_sha256": primary.checkpoint_sha256,
        "checkpoint_parameter_sha256": primary.checkpoint_parameter_sha256,
        "checkpoint_provenance_sha256": primary.checkpoint_provenance_sha256,
        "model_config_sha256": primary.model_config_sha256,
        "calibration_identity_sha256": primary.calibration_identity_sha256,
        "calibration_scale_factor": primary.calibration_scale_factor,
        "write_threshold": primary.write_threshold,
        "score_semantics": primary.score_semantics,
        "actual_pose_required": True,
        "qualification_only_adapter_memory_write_allowed": False,
    }
    if provider != provider_expectations:
        raise ValueError("Stage 2A PRIMARY provider identity 漂移")

    candidate = config["candidate"]
    candidate_expectations = {
        "settled_collect_frame_count": 3,
        "final_measurement_frame_index": 2,
        "average_position": False,
        "divide_covariance_by_frame_count": False,
        "max_frame_gap_s": 0.075,
        "max_position_spread_m": 0.005,
        "max_innovation_m": 0.01,
        "max_position_std_m": 0.02,
        "max_sensor_skew_s": 0.01,
        "require_covariance": True,
        "max_pending_age_s": 2.5,
    }
    if candidate != candidate_expectations:
        raise ValueError("Stage 2A candidate contract 漂移")

    gain = config["information_gain"]
    if (
        gain.get("selection_candidates") != list(ACTIVE_FRONT_INFORMATION_GAIN_CANDIDATES)
        or gain.get("non_promotional_integration_smoke_provisional_value") != 0.05
        or gain.get("frozen_evaluation_value") is not None
        or gain.get("evaluation_config_sha256") is not None
        or gain.get("home_raw_score_can_form_measurement") is not False
        or gain.get("baseline_unavailable_policy")
        != "shadow-only-no-zero-imputation-no-commit/v1"
    ):
        raise ValueError("Stage 2A information-gain contract 漂移")
    if set(gain) != {
        "selection_candidates",
        "non_promotional_integration_smoke_provisional_value",
        "frozen_evaluation_value",
        "evaluation_config_sha256",
        "selection_protocol",
        "comparison",
        "home_raw_score_can_form_measurement",
        "baseline_unavailable_policy",
    }:
        raise ValueError("Stage 2A information-gain keys 漂移")

    home_config = config["home_baseline"]
    if (
        home_config.get("role")
        != "raw-score-comparison-only-no-measurement-no-memory-write-no-state-resolution/v1"
        or home_config.get("primitive_id") != ACTIVE_FRONT_HOME_PRIMITIVE_ID
        or home_config.get("camera_motion_state") != "home_anchor"
        or home_config.get("settled_required") is not True
        or home_config.get("provider_identity_sha256") != home.sha256
        or home_config.get("provider_family_sha256") != home.provider_family_sha256
        or home_config.get("calibration_identity_sha256")
        != home.calibration_identity_sha256
        or home_config.get("calibration_scale_factor") != home.calibration_scale_factor
        or home_config.get("write_threshold") != home.write_threshold
        or home_config.get("score_semantics") != ACTIVE_FRONT_SCORE_SEMANTICS
        or home_config.get("actual_pose_source") != home.actual_pose_source
        or home_config.get("expected_base_from_external_camera_cv")
        != [list(row) for row in ACTIVE_FRONT_HOME_BASE_FROM_EXTERNAL_CAMERA_CV]
        or home_config.get("maximum_position_error_m")
        != ACTIVE_FRONT_HOME_POSITION_TOLERANCE_M
        or home_config.get("maximum_orientation_error_rad")
        != ACTIVE_FRONT_HOME_ORIENTATION_TOLERANCE_RAD
        or home_config.get("model_input_digest_required") is not True
        or home_config.get("provider_output_digest_required") is not True
    ):
        raise ValueError("Stage 2A HOME raw-score identity/边界漂移")

    if config["memory"] != {
        "mode": "free_static",
        "frame": "robot_base",
        "position_only": True,
        "max_unobserved_age_s": 2.5,
        "commit_after_home_barrier_only": True,
        "home_v2_barrier_frames": 4,
        "observable_now_after_commit": False,
        "navigation_memory_only": True,
        "contact_authorized": False,
        "maximum_writes_per_candidate": 1,
    }:
        raise ValueError("Stage 2A Memory contract 漂移")
    if config["execution"] != {
        "library_default_enabled": False,
        "experiment_feature_enabled": True,
        "memory_write_allowed": True,
        "maximum_attempts_per_episode": 1,
        "isolated_maniskill_only": True,
        "new_shadow_action_generation_required": True,
        "source_phase_stability_reset_ticks": 0,
        "allow_capability_absent_trigger": True,
        "capability_absent_trigger_reason": (
            ActiveFrontTriggerReason.NO_QUALIFIED_WRIST_PROVIDER_IN_PARENT.value
        ),
        "capability_absent_claim_boundary": (
            "capability-not-evaluated-not-low-confidence-not-uncertainty/v1"
        ),
        "capability_absent_consecutive_trigger_ticks": 3,
        "wrist_capability_status": WRIST_CAPABILITY_ABSENT_STATUS,
        "wrist_capability_record_version": (
            E018_P1_STAGE2A_WRIST_CAPABILITY_VERSION
        ),
        "wrist_provider_forward_count": 0,
        "independent_trigger_and_source_recheck_records_required": True,
    }:
        raise ValueError("Stage 2A execution contract 漂移")
    if config["splits"] != {
        "integration_smoke": [76901, 76910],
        "selection": [77001, 77025],
        "evaluation": [77026, 77050],
        "stage2b_shadow": [77101, 77150],
        "stage3_reserved": [77201, 77250],
    }:
        raise ValueError("Stage 2A split identity 漂移")
    if config["budgets"] != {
        "total_gpu_wall_seconds_max": 7200,
        "combined_artifact_bytes_max": 8589934592,
        "integration_smoke_gpu_wall_seconds_max": 900,
        "integration_smoke_artifact_bytes_max": 1073741824,
        "selection_gpu_wall_seconds_max": 1800,
        "selection_artifact_bytes_max": 2147483648,
        "evaluation_gpu_wall_seconds_max": 1800,
        "evaluation_artifact_bytes_max": 2147483648,
        "stage2b_shadow_gpu_wall_seconds_max": 2700,
        "stage2b_shadow_artifact_bytes_max": 3221225472,
    }:
        raise ValueError("Stage 2A budget contract 漂移")
    if config["permissions"] != {
        "fresh_test_reads": 0,
        "runtime_object_gt_reads": 0,
        "offline_label_reads": "only-after-prediction-and-decision-freeze",
        "goal_gt_reads": 0,
        "physical_camera_actuation": 0,
        "canonical_runtime_camera_actuation": 0,
        "arm_gripper_actuation": 0,
        "manipulation_progression": 0,
        "checkpoint_writes": 0,
    }:
        raise ValueError("Stage 2A permission contract 漂移")
    ActiveFrontStage2Config.development(min_information_gain=0.05)
    build_stage2_object_memory_config(primary)


def load_e018_p1_stage2a_config(path: str | Path) -> LoadedStage2AConfig:
    config_path = Path(path)
    raw = config_path.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Stage 2A config 不是有效 UTF-8 JSON") from error
    _validate_stage2a_config_payload(payload)
    return LoadedStage2AConfig(
        canonical_json=_canonical_json(payload),
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        canonical_sha256=canonical_sha256(payload),
    )


_PROVIDER_RECORD_ROLES = {
    "home-raw-score-baseline/v1",
    "primary-settled-collect/v1",
}
_WRIST_CAPABILITY_ROLES = {"trigger", "source_recheck"}
_STAGE2A_MEMORY_RESOLUTION_POLICY = {
    "version": "e018-p1-stage2a-memory-resolution-policy/v1",
    "requirement": "navigation",
    "maximum_unobserved_age_s": 2.5,
    "maximum_position_std_m": 0.020,
    "contact_authorized": False,
}
_STAGE2A_MEMORY_RESOLUTION_POLICY_SHA256 = _sha256_text(
    _canonical_json(_STAGE2A_MEMORY_RESOLUTION_POLICY)
)


@dataclass(frozen=True)
class Stage2AProviderOutputRecord:
    """把实际 model input 与 deployable prediction 绑定成不可变输出证据。"""

    episode_id: str
    episode_generation: int
    request_id: str
    observation_sequence_id: str
    route_frame_index: int
    record_role: str
    model_input_digest: str
    provider_identity: ActiveFrontStage2ProviderIdentity
    stage2_config_raw_sha256: str
    stage2_config_canonical_sha256: str
    qualification_config_raw_sha256: str
    qualification_config_internal_sha256: str
    prediction_canonical_json: str
    version: str = E018_P1_STAGE2A_PROVIDER_RECORD_VERSION

    def __post_init__(self) -> None:
        for name in ("episode_id", "request_id", "observation_sequence_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"provider record {name} 必须是非空字符串")
        if (
            type(self.episode_generation) is not int
            or self.episode_generation <= 0
            or type(self.route_frame_index) is not int
            or self.route_frame_index not in STAGE2A_PROVIDER_FRAME_INDICES
        ):
            raise ValueError("provider record generation/frame identity 非法")
        if self.record_role not in _PROVIDER_RECORD_ROLES:
            raise ValueError("provider record role 非法")
        expected_role = (
            "home-raw-score-baseline/v1"
            if self.route_frame_index == 0
            else "primary-settled-collect/v1"
        )
        if self.record_role != expected_role:
            raise ValueError("provider record role/frame identity 漂移")
        if not isinstance(self.provider_identity, ActiveFrontStage2ProviderIdentity):
            raise TypeError("provider record identity 类型错误")
        expected_provider = (
            d049_home_baseline_provider_identity()
            if self.route_frame_index == 0
            else d049_primary_provider_identity()
        )
        if self.provider_identity.sha256 != expected_provider.sha256:
            raise ValueError("provider record 完整 provider identity 漂移")
        for name in (
            "model_input_digest",
            "stage2_config_raw_sha256",
            "stage2_config_canonical_sha256",
            "qualification_config_raw_sha256",
            "qualification_config_internal_sha256",
        ):
            if not _is_sha256(getattr(self, name)):
                raise ValueError(f"provider record {name} 不是 SHA-256")
        try:
            prediction = json.loads(self.prediction_canonical_json)
        except json.JSONDecodeError as error:
            raise ValueError("provider prediction canonical JSON 非法") from error
        if (
            not isinstance(prediction, dict)
            or _canonical_json(prediction) != self.prediction_canonical_json
            or prediction.get("input_sha256") != self.model_input_digest
            or prediction.get("route_frame_index") != self.route_frame_index
            or prediction.get("frame_role") != self.record_role
            or prediction.get("viewpoint_id") != self.provider_identity.primitive_id
        ):
            raise ValueError("provider prediction 与 input/frame/provider 绑定漂移")
        if self.version != E018_P1_STAGE2A_PROVIDER_RECORD_VERSION:
            raise ValueError("provider output record version 漂移")

    @property
    def prediction(self) -> dict[str, Any]:
        return json.loads(self.prediction_canonical_json)

    def _payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "episode_id": self.episode_id,
            "episode_generation": self.episode_generation,
            "request_id": self.request_id,
            "observation_sequence_id": self.observation_sequence_id,
            "route_frame_index": self.route_frame_index,
            "record_role": self.record_role,
            "model_input_digest": self.model_input_digest,
            "provider_identity": self.provider_identity.to_dict(),
            "provider_identity_sha256": self.provider_identity.sha256,
            "stage2_config_raw_sha256": self.stage2_config_raw_sha256,
            "stage2_config_canonical_sha256": self.stage2_config_canonical_sha256,
            "qualification_config_raw_sha256": self.qualification_config_raw_sha256,
            "qualification_config_internal_sha256": (
                self.qualification_config_internal_sha256
            ),
            "prediction": self.prediction,
        }

    @property
    def provider_output_digest(self) -> str:
        return _sha256_text(_canonical_json(self._payload()))

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "provider_output_digest": self.provider_output_digest}


def verify_stage2a_provider_output_record(
    record: Stage2AProviderOutputRecord,
    *,
    stage2_config: LoadedStage2AConfig,
    qualification_config: Mapping[str, Any],
) -> None:
    """独立重算 provider mechanics 与所有 parent/config identity。"""

    if not isinstance(record, Stage2AProviderOutputRecord):
        raise TypeError("record 必须是 Stage2AProviderOutputRecord")
    parents = stage2_config.payload["parents"]
    if (
        record.stage2_config_raw_sha256 != stage2_config.raw_sha256
        or record.stage2_config_canonical_sha256 != stage2_config.canonical_sha256
        or record.qualification_config_raw_sha256
        != parents["d048_qualification_config_raw_sha256"]
        or record.qualification_config_internal_sha256
        != parents["d048_qualification_config_internal_sha256"]
        or qualification_config.get("config_sha256")
        != record.qualification_config_internal_sha256
    ):
        raise ValueError("provider record Stage2/D048 config identity 漂移")
    prediction = record.prediction
    validate_qualification_prediction_mechanics(
        prediction,
        config=qualification_config,
    )
    identity = record.provider_identity
    if (
        prediction.get("classification") != QUALIFICATION_CLASSIFICATION_SMOKE
        or prediction.get("candidate_id") != identity.candidate_id
        or prediction.get("epoch") != identity.checkpoint_epoch
        or prediction.get("checkpoint_sha256") != identity.checkpoint_sha256
        or prediction.get("checkpoint_parameter_sha256")
        != identity.checkpoint_parameter_sha256
        or prediction.get("checkpoint_provenance_sha256")
        != identity.checkpoint_provenance_sha256
        or prediction.get("checkpoint_model_config_sha256")
        != identity.model_config_sha256
        or prediction.get("calibration_scale_factor")
        != identity.calibration_scale_factor
        or prediction.get("write_threshold") != identity.write_threshold
        or prediction.get("memory_write_allowed") is not False
        or prediction.get("memory_write_executed") is not False
        or prediction.get("actuation_allowed") is not False
        or prediction.get("test_data_read") is not False
    ):
        raise ValueError("provider prediction checkpoint/calibration/权限 identity 漂移")


def build_stage2a_provider_output_record(
    *,
    capture: Mapping[str, Any],
    prediction: Mapping[str, Any],
    episode_id: str,
    episode_generation: int,
    request_id: str,
    observation_sequence_id: str,
    route_frame_index: int,
    stage2_config: LoadedStage2AConfig,
    qualification_config: Mapping[str, Any],
) -> Stage2AProviderOutputRecord:
    """唯一构造入口；调用方不能分别注入任意 input/output digest。"""

    input_digest = capture.get("input_sha256")
    capture_identity = capture.get("identity")
    if not _is_sha256(input_digest) or not isinstance(capture_identity, Mapping):
        raise ValueError("provider capture identity/input digest 非法")
    if (
        prediction.get("input_sha256") != input_digest
        or capture_identity.get("route_frame_index") != route_frame_index
        or prediction.get("route_frame_index") != route_frame_index
        or prediction.get("frame_role") != capture_identity.get("frame_role")
        or prediction.get("viewpoint_id") != capture_identity.get("viewpoint_id")
    ):
        raise ValueError("实际 capture 与 provider prediction identity 漂移")
    provider_identity = (
        d049_home_baseline_provider_identity()
        if route_frame_index == 0
        else d049_primary_provider_identity()
    )
    record = Stage2AProviderOutputRecord(
        episode_id=episode_id,
        episode_generation=episode_generation,
        request_id=request_id,
        observation_sequence_id=observation_sequence_id,
        route_frame_index=route_frame_index,
        record_role=str(capture_identity["frame_role"]),
        model_input_digest=str(input_digest),
        provider_identity=provider_identity,
        stage2_config_raw_sha256=stage2_config.raw_sha256,
        stage2_config_canonical_sha256=stage2_config.canonical_sha256,
        qualification_config_raw_sha256=stage2_config.payload["parents"][
            "d048_qualification_config_raw_sha256"
        ],
        qualification_config_internal_sha256=stage2_config.payload["parents"][
            "d048_qualification_config_internal_sha256"
        ],
        prediction_canonical_json=_canonical_json(dict(prediction)),
    )
    verify_stage2a_provider_output_record(
        record,
        stage2_config=stage2_config,
        qualification_config=qualification_config,
    )
    return record


def _object_state_snapshot(state: ObjectState) -> dict[str, Any]:
    value = asdict(state)
    value["mode"] = state.mode.value
    return value


def _object_state_from_snapshot(snapshot: Mapping[str, Any]) -> ObjectState:
    payload = dict(snapshot)
    reasons = payload.get("invalid_reasons")
    if not isinstance(reasons, (list, tuple)):
        raise TypeError("ObjectState snapshot invalid_reasons 必须是 sequence")
    payload["invalid_reasons"] = tuple(reasons)
    return ObjectState(**payload)


def _derive_memory_resolution(
    snapshot: Mapping[str, Any],
    *,
    timestamp_s: float,
) -> dict[str, Any]:
    """从冻结 ObjectState primitive 机械重算 NAVIGATION availability。"""

    required = {
        "episode_id",
        "mode",
        "position_base_m",
        "covariance_base_m2",
        "measurement_confidence",
        "last_observed_timestamp_s",
        "state_timestamp_s",
        "observable_now",
        "valid",
        "accepted_update_count",
        "source_camera",
        "source_model_identity",
        "invalid_reasons",
        "frame_semantics",
        "version",
    }
    _require_exact_keys(dict(snapshot), required, "wrist capability Memory snapshot")
    try:
        state = _object_state_from_snapshot(snapshot)
    except (TypeError, ValueError) as error:
        raise ValueError("wrist capability ObjectState snapshot contract 非法") from error
    if state.state_timestamp_s > timestamp_s + 1e-12:
        raise ValueError("wrist capability Memory state timestamp 晚于 observation")
    mode = state.mode.value
    position_value = state.position_base_m
    covariance_value = state.covariance_base_m2
    last_observed = state.last_observed_timestamp_s
    reasons: list[str] = []
    if mode != ObjectMemoryMode.FREE_STATIC.value:
        reasons.append(f"memory_mode:{mode}")
    if not state.valid:
        reasons.append("memory_not_valid")
    if position_value is None:
        reasons.append("memory_position_missing")
    if covariance_value is None:
        reasons.append("memory_covariance_missing")
    if last_observed is None:
        reasons.append("memory_last_observed_timestamp_missing")
    else:
        age = timestamp_s - float(last_observed)
        if not math.isfinite(age) or age < -1e-12:
            reasons.append("memory_timestamp_invalid")
        elif age > (
            float(_STAGE2A_MEMORY_RESOLUTION_POLICY["maximum_unobserved_age_s"])
            + 1e-12
        ):
            reasons.append("memory_stale")
    if covariance_value is not None:
        covariance = np.asarray(covariance_value, dtype=np.float64)
        if (
            covariance.shape != (3, 3)
            or not np.isfinite(covariance).all()
            or not np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-12)
        ):
            reasons.append("memory_covariance_invalid")
        else:
            maximum_eigenvalue = float(np.linalg.eigvalsh(covariance).max())
            if maximum_eigenvalue < -1e-12:
                reasons.append("memory_covariance_invalid")
            elif math.sqrt(max(0.0, maximum_eigenvalue)) > (
                float(_STAGE2A_MEMORY_RESOLUTION_POLICY["maximum_position_std_m"])
                + 1e-12
            ):
                reasons.append("memory_uncertain")
    for value in state.invalid_reasons:
        reason = str(value)
        if not reason:
            raise ValueError("Memory invalid reason 不能为空")
        reasons.append(f"memory_state_reason:{reason}")
    unavailable = tuple(dict.fromkeys(reasons))
    return {
        "mode": mode,
        "available": not unavailable,
        "unavailable_reasons": unavailable,
    }


@dataclass(frozen=True)
class WristCapabilityEvidenceRecord:
    """D050 前的显式 capability-absent 事实，不冒充视觉失败判断。"""

    episode_id: str
    episode_generation: int
    request_id: str
    record_role: str
    source_phase: PhaseId
    observation_sequence_id: str
    timestamp_s: float
    home_observation_payload_digest: str
    memory_state_canonical_json: str
    memory_mode: str
    memory_resolution_available: bool
    memory_unavailable_reasons: tuple[str, ...]
    memory_state_revision: str
    memory_resolution_policy_sha256: str
    home_observation_payload_canonical_json: str
    reason: str = "no-qualified-wrist-provider-in-d049-parent/v1"
    status: str = WRIST_CAPABILITY_ABSENT_STATUS
    provider_identity: str | None = None
    inference_attempt_count: int = 0
    frame_evaluated: bool = False
    measurement_usable: bool = False
    state_authorized: bool = False
    supersede_authorized: bool = False
    contact_authorized: bool = False
    version: str = E018_P1_STAGE2A_WRIST_CAPABILITY_VERSION

    def __post_init__(self) -> None:
        for name in ("episode_id", "request_id", "observation_sequence_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"wrist capability {name} 必须是非空字符串")
        if type(self.episode_generation) is not int or self.episode_generation <= 0:
            raise ValueError("wrist capability episode_generation 必须是正整数")
        if self.record_role not in _WRIST_CAPABILITY_ROLES:
            raise ValueError("wrist capability role 非法")
        if self.source_phase is not STAGE2A_SOURCE_PHASE:
            raise ValueError("wrist capability source phase 漂移")
        if not math.isfinite(self.timestamp_s) or self.timestamp_s < 0.0:
            raise ValueError("wrist capability timestamp 非法")
        if not _is_sha256(self.home_observation_payload_digest):
            raise ValueError("wrist capability HOME payload digest 非法")
        try:
            home_payload = json.loads(self.home_observation_payload_canonical_json)
        except json.JSONDecodeError as error:
            raise ValueError("wrist capability HOME payload identity 非法") from error
        if (
            not isinstance(home_payload, dict)
            or _canonical_json(home_payload)
            != self.home_observation_payload_canonical_json
            or _sha256_text(self.home_observation_payload_canonical_json)
            != self.home_observation_payload_digest
        ):
            raise ValueError("wrist capability HOME payload完整 identity 漂移")
        _verify_home_observation_payload_identity(home_payload)
        try:
            memory_snapshot = json.loads(self.memory_state_canonical_json)
        except json.JSONDecodeError as error:
            raise ValueError("wrist capability Memory snapshot 非法") from error
        if (
            not isinstance(memory_snapshot, dict)
            or _canonical_json(memory_snapshot) != self.memory_state_canonical_json
        ):
            raise ValueError("wrist capability Memory snapshot canonical identity 漂移")
        expected_memory = _derive_memory_resolution(
            memory_snapshot,
            timestamp_s=self.timestamp_s,
        )
        if (
            memory_snapshot.get("episode_id") != self.episode_id
            or self.memory_resolution_policy_sha256
            != _STAGE2A_MEMORY_RESOLUTION_POLICY_SHA256
            or self.memory_mode != expected_memory["mode"]
            or self.memory_resolution_available
            is not expected_memory["available"]
            or self.memory_unavailable_reasons
            != tuple(expected_memory["unavailable_reasons"])
            or self.memory_state_revision
            != _sha256_text(self.memory_state_canonical_json)
        ):
            raise ValueError("wrist capability Memory resolution/revision 漂移")
        if (
            self.reason != "no-qualified-wrist-provider-in-d049-parent/v1"
            or self.status != WRIST_CAPABILITY_ABSENT_STATUS
            or self.provider_identity is not None
            or self.inference_attempt_count != 0
            or self.frame_evaluated
            or self.measurement_usable
            or self.state_authorized
            or self.supersede_authorized
            or self.contact_authorized
            or self.memory_resolution_available
        ):
            raise ValueError("D050 前 wrist capability-absent contract 漂移")
        if self.version != E018_P1_STAGE2A_WRIST_CAPABILITY_VERSION:
            raise ValueError("wrist capability evidence version 漂移")

    def _payload(self) -> dict[str, Any]:
        value = asdict(self)
        value["source_phase"] = self.source_phase.value
        value["memory_state"] = json.loads(value.pop("memory_state_canonical_json"))
        value["home_observation_payload"] = json.loads(
            value.pop("home_observation_payload_canonical_json")
        )
        return value

    @property
    def digest(self) -> str:
        return _sha256_text(_canonical_json(self._payload()))

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "evidence_identity_sha256": self.digest}


def home_observation_payload_identity(
    observation: Mapping[str, Any],
    *,
    timestamp_s: float,
    wrist_camera_uid: str = "hand_camera",
    front_camera_uid: str = "base_camera",
) -> dict[str, Any]:
    """只读取两相机 RGB/pose；不触碰 segmentation、actor 或 object/goal GT。"""

    if not math.isfinite(timestamp_s) or timestamp_s < 0.0:
        raise ValueError("HOME observation timestamp 非法")
    sensor_data = observation["sensor_data"]
    sensor_param = observation["sensor_param"]
    cameras: dict[str, Any] = {}
    for role, uid in (("wrist", wrist_camera_uid), ("front", front_camera_uid)):
        rgb = np.ascontiguousarray(_g0._numpy(sensor_data[uid]["rgb"]))
        pose = np.ascontiguousarray(_g0._numpy(sensor_param[uid]["cam2world_gl"]))
        if rgb.ndim != 4 or rgb.shape[0] != 1 or rgb.dtype != np.uint8:
            raise ValueError(f"{role} HOME RGB shape/dtype 漂移")
        if pose.shape not in {(1, 4, 4), (4, 4)} or not np.isfinite(pose).all():
            raise ValueError(f"{role} HOME camera pose 漂移")
        pose_matrix = pose[0] if pose.shape == (1, 4, 4) else pose
        cameras[role] = {
            "camera_uid": uid,
            "rgb_shape": list(rgb.shape),
            "rgb_dtype": str(rgb.dtype),
            "rgb_bytes_sha256": hashlib.sha256(rgb.tobytes(order="C")).hexdigest(),
            "cam2world_gl_shape": list(pose.shape),
            "cam2world_gl_dtype": str(pose.dtype),
            "cam2world_gl_sha256": _array_sha256(pose),
            "cam2world_gl": pose_matrix.tolist(),
        }
    payload = {
        "version": "e018-p1-stage2a-home-observation-payload/v2",
        "timestamp_s": float(timestamp_s),
        "cameras": cameras,
    }
    _verify_home_observation_payload_identity(payload)
    return payload


def _verify_home_observation_payload_identity(
    payload: Mapping[str, Any],
) -> None:
    """验证可公开重放的两相机 RGB/pose identity；不读取图像或 GT。"""

    value = _require_exact_keys(
        dict(payload),
        {"version", "timestamp_s", "cameras"},
        "HOME observation payload",
    )
    if (
        value["version"] != "e018-p1-stage2a-home-observation-payload/v2"
        or not isinstance(value["timestamp_s"], (int, float))
        or isinstance(value["timestamp_s"], bool)
        or not math.isfinite(float(value["timestamp_s"]))
        or float(value["timestamp_s"]) < 0.0
    ):
        raise ValueError("HOME observation payload version/timestamp 漂移")
    cameras = _require_exact_keys(
        value["cameras"], {"wrist", "front"}, "HOME observation cameras"
    )
    for role, expected_uid in (("wrist", "hand_camera"), ("front", "base_camera")):
        camera = _require_exact_keys(
            cameras[role],
            {
                "camera_uid",
                "rgb_shape",
                "rgb_dtype",
                "rgb_bytes_sha256",
                "cam2world_gl_shape",
                "cam2world_gl_dtype",
                "cam2world_gl_sha256",
                "cam2world_gl",
            },
            f"HOME observation {role} camera",
        )
        if (
            camera["camera_uid"] != expected_uid
            or camera["rgb_dtype"] != "uint8"
            or not isinstance(camera["rgb_shape"], list)
            or len(camera["rgb_shape"]) != 4
            or camera["rgb_shape"][0] != 1
            or camera["rgb_shape"][-1] != 3
            or any(type(size) is not int or size <= 0 for size in camera["rgb_shape"])
            or not _is_sha256(camera["rgb_bytes_sha256"])
            or not _is_sha256(camera["cam2world_gl_sha256"])
            or camera["cam2world_gl_shape"] not in ([1, 4, 4], [4, 4])
        ):
            raise ValueError(f"HOME observation {role} RGB/pose identity 漂移")
        try:
            pose_dtype = np.dtype(camera["cam2world_gl_dtype"])
            pose_matrix = np.asarray(camera["cam2world_gl"], dtype=pose_dtype)
            pose = np.ascontiguousarray(
                pose_matrix.reshape(tuple(camera["cam2world_gl_shape"]))
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"HOME observation {role} pose payload 非法") from error
        if (
            pose_dtype.kind != "f"
            or pose_matrix.shape != (4, 4)
            or not np.isfinite(pose_matrix).all()
            or not np.allclose(pose_matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9)
            or _array_sha256(pose) != camera["cam2world_gl_sha256"]
        ):
            raise ValueError(f"HOME observation {role} pose digest/SE3 漂移")


def home_observation_payload_digest(
    observation: Mapping[str, Any],
    *,
    timestamp_s: float,
    wrist_camera_uid: str = "hand_camera",
    front_camera_uid: str = "base_camera",
) -> str:
    return _sha256_text(
        _canonical_json(
            home_observation_payload_identity(
                observation,
                timestamp_s=timestamp_s,
                wrist_camera_uid=wrist_camera_uid,
                front_camera_uid=front_camera_uid,
            )
        )
    )


_STAGE2A_OBSERVATION_V2_WINDOW_IDENTITY_VERSION = (
    "e018-p1-stage2a-fresh-home-observation-v2-window/v1"
)
_STAGE2A_SHADOW_REPLAN_INSTRUCTION = (
    "E018 Stage 2A fresh HOME shadow replan; no actuator"
)


def _build_stage2a_observation_v2_frame(
    observation: Mapping[str, Any],
    *,
    base_env: Any,
    observation_adapter: FrankaObservationAdapter,
    spec: Any,
    timestamp_s: float,
) -> ObservationV2Frame:
    """从同一仿真 Tick 构造完整 V2 frame；不读取 object/goal pose。"""

    if not math.isfinite(timestamp_s) or timestamp_s < 0.0:
        raise ValueError("Stage 2A Observation V2 timestamp 非法")
    sensor_data = observation["sensor_data"]
    sensor_param = observation["sensor_param"]
    external = np.asarray(_g0._numpy(sensor_data["base_camera"]["rgb"]))
    wrist = np.asarray(_g0._numpy(sensor_data["hand_camera"]["rgb"]))
    if (
        external.ndim != 4
        or wrist.ndim != 4
        or external.shape[0] != 1
        or wrist.shape[0] != 1
        or external.dtype != np.uint8
        or wrist.dtype != np.uint8
    ):
        raise ValueError("Stage 2A Observation V2 RGB shape/dtype 漂移")

    robot = base_env.agent.robot
    qpos = np.asarray(_g0._numpy(robot.get_qpos()))
    qvel = np.asarray(_g0._numpy(robot.get_qvel()))
    joint_names = tuple(joint.name for joint in robot.active_joints)
    if qpos.shape != (1, 9) or qvel.shape != qpos.shape:
        raise ValueError("Stage 2A Observation V2 Panda qpos/qvel 必须是 [1,9]")
    proprio = observation_adapter.from_maniskill(qpos[0], qvel[0], joint_names)

    world_from_base = _g0._single_matrix(
        robot.pose, "stage2a_world_from_robot_base"
    )
    world_from_tcp = _g0._single_matrix(
        base_env.agent.tcp_pose, "stage2a_world_from_tcp"
    )
    wrist_gl = _g0._single_matrix(
        sensor_param["hand_camera"]["cam2world_gl"],
        "stage2a_world_from_wrist_camera_gl",
    )
    base_from_world = invert_se3(world_from_base, "stage2a_world_from_robot_base")
    base_from_tcp = validate_se3(
        base_from_world @ world_from_tcp, "stage2a_base_from_tcp"
    )
    base_from_wrist = validate_se3(
        base_from_world @ opengl_camera_to_opencv(wrist_gl),
        "stage2a_base_from_wrist_camera_cv",
    )

    scene = base_env.scene
    left = np.asarray(
        _g0._numpy(
            scene.get_pairwise_contact_forces(
                base_env.agent.finger1_link, base_env.cube
            )
        )
    )
    right = np.asarray(
        _g0._numpy(
            scene.get_pairwise_contact_forces(
                base_env.agent.finger2_link, base_env.cube
            )
        )
    )
    if (
        left.shape != (1, 3)
        or right.shape != (1, 3)
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
    ):
        raise ValueError("Stage 2A Observation V2 F_L/F_R sensor shape 漂移")
    finger_force = np.asarray(
        (float(np.linalg.norm(left[0])), float(np.linalg.norm(right[0]))),
        dtype=np.float32,
    )
    return ObservationV2Frame(
        rgb_external=np.ascontiguousarray(external[0]),
        rgb_wrist=np.ascontiguousarray(wrist[0]),
        physical_proprio=proprio.astype(np.float32, copy=False),
        base_from_tcp=base_from_tcp.astype(np.float32),
        base_from_wrist_camera=base_from_wrist.astype(np.float32),
        finger_force_n=finger_force,
        timestamp_s=float(timestamp_s),
        modality_timestamp_s=np.full(
            len(OBSERVATION_MODALITIES), float(timestamp_s), dtype=np.float64
        ),
        modality_valid=np.ones(len(OBSERVATION_MODALITIES), dtype=np.bool_),
    )


def _array_value_identity(value: Any) -> dict[str, Any]:
    array = np.ascontiguousarray(np.asarray(value))
    if array.dtype.kind not in "fiu b".replace(" ", ""):
        raise ValueError("Observation V2 identity 只接受数值/bool array")
    if array.dtype.kind in "f" and not np.isfinite(array).all():
        raise ValueError("Observation V2 identity 不接受非有限 array")
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "values": array.tolist(),
        "array_sha256": _array_sha256(array),
    }


def _array_from_value_identity(
    value: Any,
    *,
    name: str,
    dtype: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    record = _require_exact_keys(
        value, {"dtype", "shape", "values", "array_sha256"}, name
    )
    if record["dtype"] != dtype or record["shape"] != list(shape):
        raise ValueError(f"{name} dtype/shape 漂移")
    try:
        array = np.ascontiguousarray(np.asarray(record["values"], dtype=np.dtype(dtype)))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} values 非法") from error
    if array.shape != shape or _array_sha256(array) != record["array_sha256"]:
        raise ValueError(f"{name} values/digest 漂移")
    return array


def _image_history_identity(value: Any) -> dict[str, Any]:
    images = np.ascontiguousarray(np.asarray(value))
    if images.ndim != 4 or images.shape[0] != 4 or images.dtype != np.uint8:
        raise ValueError("Observation V2 image history 必须是 uint8 [4,H,W,3]")
    return {
        "dtype": "uint8",
        "shape": list(images.shape),
        "frame_bytes_sha256": [
            hashlib.sha256(np.ascontiguousarray(frame).tobytes(order="C")).hexdigest()
            for frame in images
        ],
    }


def _verify_image_history_identity(value: Any, *, name: str) -> tuple[int, ...]:
    record = _require_exact_keys(
        value, {"dtype", "shape", "frame_bytes_sha256"}, name
    )
    shape = record["shape"]
    digests = record["frame_bytes_sha256"]
    if (
        record["dtype"] != "uint8"
        or not isinstance(shape, list)
        or len(shape) != 4
        or shape[0] != 4
        or shape[-1] != 3
        or any(type(size) is not int or size <= 0 for size in shape)
        or not isinstance(digests, list)
        or len(digests) != 4
        or any(not _is_sha256(digest) for digest in digests)
    ):
        raise ValueError(f"{name} identity 漂移")
    return tuple(shape)


def _build_observation_v2_window_identity(
    window: ObservationV2Window,
    *,
    spec: Any,
    episode_id: str,
    episode_generation: int,
    observation_sequence_ids: Sequence[str],
    home_evidence: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    window.validate(spec, require_current_complete=True)
    if (
        list(np.asarray(window.history_valid, dtype=np.bool_)) != [True] * 4
        or not bool(np.asarray(window.modality_valid, dtype=np.bool_).all())
        or list(np.asarray(window.controller_valid, dtype=np.bool_)) != [False, False]
        or np.any(window.previous_command_q)
        or np.any(window.tracking_error)
        or np.any(window.previous_action)
        or len(observation_sequence_ids) != 4
        or len(set(observation_sequence_ids)) != 4
        or len(home_evidence) != 4
    ):
        raise ValueError("fresh HOME Observation V2 window 完整性/controller reset 漂移")
    external_identity = _image_history_identity(window.rgb_external)
    wrist_identity = _image_history_identity(window.rgb_wrist)
    home_digests: list[str] = []
    for index, evidence in enumerate(home_evidence):
        payload = evidence.get("home_observation_payload")
        _verify_home_observation_payload_identity(payload)
        if (
            evidence.get("observation_sequence_id") != observation_sequence_ids[index]
            or float(evidence.get("control_timestamp_s", math.nan))
            != float(window.frame_timestamp_s[index])
            or payload["cameras"]["front"]["rgb_bytes_sha256"]
            != external_identity["frame_bytes_sha256"][index]
            or payload["cameras"]["wrist"]["rgb_bytes_sha256"]
            != wrist_identity["frame_bytes_sha256"][index]
            or not _is_sha256(evidence.get("evidence_sha256"))
        ):
            raise ValueError("fresh HOME evidence 与 Observation V2 window 不一致")
        home_digests.append(str(evidence["evidence_sha256"]))
    arrays = {
        name: _array_value_identity(getattr(window, name))
        for name in (
            "physical_proprio",
            "tcp_position",
            "tcp_rotation_6d",
            "wrist_position",
            "wrist_rotation_6d",
            "finger_force_n",
            "frame_age_s",
            "modality_age_s",
            "frame_timestamp_s",
            "modality_timestamp_s",
            "history_valid",
            "modality_valid",
            "previous_command_q",
            "tracking_error",
            "previous_action",
            "controller_valid",
        )
    }
    identity = {
        "version": _STAGE2A_OBSERVATION_V2_WINDOW_IDENTITY_VERSION,
        "episode_id": episode_id,
        "episode_generation": episode_generation,
        "observation_sequence_ids": list(observation_sequence_ids),
        "observation_version": window.version,
        "instruction": window.instruction,
        "timestamp_s": float(window.timestamp_s),
        "rgb_external": external_identity,
        "rgb_wrist": wrist_identity,
        "arrays": arrays,
        "home_evidence_digests": home_digests,
        "require_current_complete_validated": True,
        "oldest_to_newest_contiguous_validated": True,
        "controller_state_invalidated_before_replan": True,
    }
    identity["window_sha256"] = canonical_sha256(identity)
    return identity


def verify_stage2a_observation_v2_window_identity(
    identity: Mapping[str, Any],
    *,
    spec: Any,
    home_evidence: Sequence[Mapping[str, Any]],
    home_motion_rows: Sequence[Mapping[str, Any]] | None = None,
    expected_episode_id: str,
    expected_episode_generation: int,
) -> ObservationV2Window:
    """由公开 primitive 重建 V2 数值窗口并重新执行稳定合同校验。"""

    value = _require_exact_keys(
        dict(identity),
        {
            "version",
            "episode_id",
            "episode_generation",
            "observation_sequence_ids",
            "observation_version",
            "instruction",
            "timestamp_s",
            "rgb_external",
            "rgb_wrist",
            "arrays",
            "home_evidence_digests",
            "require_current_complete_validated",
            "oldest_to_newest_contiguous_validated",
            "controller_state_invalidated_before_replan",
            "window_sha256",
        },
        "Observation V2 window identity",
    )
    primitive = dict(value)
    stored_digest = primitive.pop("window_sha256")
    if (
        stored_digest != canonical_sha256(primitive)
        or value["version"] != _STAGE2A_OBSERVATION_V2_WINDOW_IDENTITY_VERSION
        or value["episode_id"] != expected_episode_id
        or value["episode_generation"] != expected_episode_generation
        or value["observation_version"] != "robot-vla-observation/v2"
        or value["instruction"] != _STAGE2A_SHADOW_REPLAN_INSTRUCTION
        or value["require_current_complete_validated"] is not True
        or value["oldest_to_newest_contiguous_validated"] is not True
        or value["controller_state_invalidated_before_replan"] is not True
        or not isinstance(value["observation_sequence_ids"], list)
        or len(value["observation_sequence_ids"]) != 4
        or len(set(value["observation_sequence_ids"])) != 4
        or len(home_evidence) != 4
    ):
        raise ValueError("Observation V2 window顶层 identity 漂移")
    external_shape = _verify_image_history_identity(
        value["rgb_external"], name="Observation V2 external RGB"
    )
    wrist_shape = _verify_image_history_identity(
        value["rgb_wrist"], name="Observation V2 wrist RGB"
    )
    arrays = _require_exact_keys(
        value["arrays"],
        {
            "physical_proprio",
            "tcp_position",
            "tcp_rotation_6d",
            "wrist_position",
            "wrist_rotation_6d",
            "finger_force_n",
            "frame_age_s",
            "modality_age_s",
            "frame_timestamp_s",
            "modality_timestamp_s",
            "history_valid",
            "modality_valid",
            "previous_command_q",
            "tracking_error",
            "previous_action",
            "controller_valid",
        },
        "Observation V2 window arrays",
    )
    field_specs = {
        "physical_proprio": ("float32", (4, spec.proprio_dim)),
        "tcp_position": ("float32", (4, 3)),
        "tcp_rotation_6d": ("float32", (4, 6)),
        "wrist_position": ("float32", (4, 3)),
        "wrist_rotation_6d": ("float32", (4, 6)),
        "finger_force_n": ("float32", (4, 2)),
        "frame_age_s": ("float32", (4,)),
        "modality_age_s": ("float32", (4, len(OBSERVATION_MODALITIES))),
        "frame_timestamp_s": ("float64", (4,)),
        "modality_timestamp_s": ("float64", (4, len(OBSERVATION_MODALITIES))),
        "history_valid": ("bool", (4,)),
        "modality_valid": ("bool", (4, len(OBSERVATION_MODALITIES))),
        "previous_command_q": ("float32", (spec.arm_dof,)),
        "tracking_error": ("float32", (spec.arm_dof,)),
        "previous_action": ("float32", (spec.action_dim,)),
        "controller_valid": ("bool", (2,)),
    }
    parsed = {
        name: _array_from_value_identity(
            arrays[name], name=f"Observation V2 {name}", dtype=dtype, shape=shape
        )
        for name, (dtype, shape) in field_specs.items()
    }
    window = ObservationV2Window(
        rgb_external=np.zeros(external_shape, dtype=np.uint8),
        rgb_wrist=np.zeros(wrist_shape, dtype=np.uint8),
        instruction=value["instruction"],
        timestamp_s=value["timestamp_s"],
        version=value["observation_version"],
        **parsed,
    )
    window.validate(spec, require_current_complete=True)
    if (
        window.history_valid.tolist() != [True] * 4
        or not bool(window.modality_valid.all())
        or window.controller_valid.tolist() != [False, False]
        or np.any(window.previous_command_q)
        or np.any(window.tracking_error)
        or np.any(window.previous_action)
    ):
        raise ValueError("Observation V2 HOME/controller state 语义漂移")
    expected_home_digests: list[str] = []
    for index, evidence in enumerate(home_evidence):
        primitive_evidence = dict(evidence)
        evidence_digest = primitive_evidence.pop("evidence_sha256", None)
        payload = evidence.get("home_observation_payload")
        _verify_home_observation_payload_identity(payload)
        if (
            evidence_digest != canonical_sha256(primitive_evidence)
            or evidence.get("observation_sequence_id")
            != value["observation_sequence_ids"][index]
            or float(evidence.get("control_timestamp_s", math.nan))
            != float(window.frame_timestamp_s[index])
            or payload["cameras"]["front"]["rgb_bytes_sha256"]
            != value["rgb_external"]["frame_bytes_sha256"][index]
            or payload["cameras"]["wrist"]["rgb_bytes_sha256"]
            != value["rgb_wrist"]["frame_bytes_sha256"][index]
        ):
            raise ValueError("Observation V2 window 与 HOME evidence 绑定漂移")
        expected_home_digests.append(str(evidence_digest))
    if value["home_evidence_digests"] != expected_home_digests:
        raise ValueError("Observation V2 window HOME evidence digest顺序漂移")
    if home_motion_rows is not None:
        if len(home_motion_rows) != 4:
            raise ValueError("Observation V2 window 必须绑定四条 HOME motion row")
        for index, (evidence, row) in enumerate(
            zip(home_evidence, home_motion_rows, strict=True)
        ):
            frame_index = STAGE2A_HOME_BARRIER_FRAME_INDICES[index]
            if (
                row.get("frame_index") != frame_index
                or row.get("episode_id") != expected_episode_id
                or evidence.get("route_frame_index") != frame_index
                or evidence.get("motion_row_sha256")
                != canonical_sha256(dict(row))
            ):
                raise ValueError("Observation V2 window HOME motion row identity 漂移")
            world_from_base = validate_se3(
                np.asarray(row.get("world_from_robot_base"), dtype=np.float64),
                "Observation V2 world_from_robot_base",
            )
            world_from_tcp = validate_se3(
                np.asarray(row.get("tcp_current_world"), dtype=np.float64),
                "Observation V2 world_from_tcp",
            )
            wrist_payload = evidence["home_observation_payload"]["cameras"][
                "wrist"
            ]
            world_from_wrist_gl = validate_se3(
                np.asarray(wrist_payload["cam2world_gl"], dtype=np.float64),
                "Observation V2 world_from_wrist_camera_gl",
            )
            base_from_world = invert_se3(
                world_from_base, "Observation V2 world_from_robot_base"
            )
            base_from_tcp = validate_se3(
                base_from_world @ world_from_tcp,
                "Observation V2 base_from_tcp",
            )
            base_from_wrist = validate_se3(
                base_from_world @ opengl_camera_to_opencv(world_from_wrist_gl),
                "Observation V2 base_from_wrist_camera_cv",
            )
            expected_tcp_position = base_from_tcp[:3, 3].astype(np.float32)
            expected_tcp_rotation = rotation_matrix_to_6d(base_from_tcp[:3, :3])
            expected_wrist_position = base_from_wrist[:3, 3].astype(np.float32)
            expected_wrist_rotation = rotation_matrix_to_6d(
                base_from_wrist[:3, :3]
            )
            arm_q = np.asarray(row.get("arm_current_q_rad"), dtype=np.float32)
            arm_dq = np.asarray(
                row.get("arm_current_dq_rad_s"), dtype=np.float32
            )
            finger_q = np.asarray(
                row.get("finger_joint_positions_m"), dtype=np.float32
            )
            force_pair = np.asarray(
                (row.get("finger_force_left_n"), row.get("finger_force_right_n")),
                dtype=np.float32,
            )
            if (
                arm_q.shape != (spec.arm_dof,)
                or arm_dq.shape != (spec.arm_dof,)
                or finger_q.shape != (2,)
                or force_pair.shape != (2,)
                or not np.isfinite(arm_q).all()
                or not np.isfinite(arm_dq).all()
                or not np.isfinite(finger_q).all()
                or not np.isfinite(force_pair).all()
            ):
                raise ValueError("Observation V2 HOME proprio/F_L/F_R raw witness 非法")
            lower, upper = spec.gripper_joint_position_range_m
            expected_gripper = float(
                np.mean(np.clip((finger_q - lower) / (upper - lower), 0.0, 1.0))
            )
            timestamp = float(evidence["control_timestamp_s"])
            if (
                not np.allclose(
                    window.physical_proprio[index, : spec.arm_dof],
                    arm_q,
                    rtol=0.0,
                    atol=1e-6,
                )
                or not np.allclose(
                    window.physical_proprio[index, spec.arm_dof : 2 * spec.arm_dof],
                    arm_dq,
                    rtol=0.0,
                    atol=1e-6,
                )
                or not math.isclose(
                    float(window.physical_proprio[index, -1]),
                    expected_gripper,
                    rel_tol=0.0,
                    abs_tol=1e-6,
                )
                or not np.allclose(
                    window.tcp_position[index],
                    expected_tcp_position,
                    rtol=0.0,
                    atol=1e-6,
                )
                or not np.allclose(
                    window.tcp_rotation_6d[index],
                    expected_tcp_rotation,
                    rtol=0.0,
                    atol=1e-6,
                )
                or not np.allclose(
                    window.wrist_position[index],
                    expected_wrist_position,
                    rtol=0.0,
                    atol=1e-6,
                )
                or not np.allclose(
                    window.wrist_rotation_6d[index],
                    expected_wrist_rotation,
                    rtol=0.0,
                    atol=1e-6,
                )
                or not np.allclose(
                    window.finger_force_n[index],
                    force_pair,
                    rtol=0.0,
                    atol=1e-6,
                )
                or not math.isclose(
                    float(window.frame_timestamp_s[index]),
                    timestamp,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or not np.allclose(
                    window.modality_timestamp_s[index],
                    np.full(len(OBSERVATION_MODALITIES), timestamp),
                    rtol=0.0,
                    atol=1e-12,
                )
            ):
                raise ValueError(
                    "Observation V2 数值窗口不能由 HOME raw witness 重算"
                )
    return window


def build_absent_wrist_capability_record(
    *,
    observation: Mapping[str, Any],
    episode_id: str,
    episode_generation: int,
    request_id: str,
    record_role: str,
    observation_sequence_id: str,
    timestamp_s: float,
    memory_state: ObjectState,
    source_phase: PhaseId = STAGE2A_SOURCE_PHASE,
) -> WristCapabilityEvidenceRecord:
    if not isinstance(memory_state, ObjectState):
        raise TypeError("wrist capability memory_state 类型错误")
    memory_snapshot = _object_state_snapshot(memory_state)
    resolution = _derive_memory_resolution(memory_snapshot, timestamp_s=timestamp_s)
    home_payload = home_observation_payload_identity(
        observation,
        timestamp_s=timestamp_s,
    )
    home_payload_canonical = _canonical_json(home_payload)
    return WristCapabilityEvidenceRecord(
        episode_id=episode_id,
        episode_generation=episode_generation,
        request_id=request_id,
        record_role=record_role,
        source_phase=source_phase,
        observation_sequence_id=observation_sequence_id,
        timestamp_s=timestamp_s,
        home_observation_payload_digest=_sha256_text(home_payload_canonical),
        home_observation_payload_canonical_json=home_payload_canonical,
        memory_state_canonical_json=_canonical_json(memory_snapshot),
        memory_mode=str(resolution["mode"]),
        memory_resolution_available=bool(resolution["available"]),
        memory_unavailable_reasons=tuple(resolution["unavailable_reasons"]),
        memory_state_revision=_sha256_text(_canonical_json(memory_snapshot)),
        memory_resolution_policy_sha256=(
            _STAGE2A_MEMORY_RESOLUTION_POLICY_SHA256
        ),
    )


def _score_components(prediction: Mapping[str, Any]) -> ActiveFrontScoreComponents:
    return ActiveFrontScoreComponents(
        object_visibility_probability=float(
            prediction["object_visibility_probability"]
        ),
        projection_validity_probability=float(
            prediction["projection_validity_probability"]
        ),
        object_mask_probability=float(
            prediction["object_mask_probability_at_prediction"]
        ),
        goal_mask_probability=float(
            prediction["goal_mask_probability_at_prediction"]
        ),
        object_normalized_entropy=float(prediction["object_normalized_entropy"]),
        object_sigma_xy_px=tuple(float(value) for value in prediction["object_sigma_xy_px"]),
    )


def build_stage2a_home_score_evidence(
    record: Stage2AProviderOutputRecord,
    *,
    motion_row: Mapping[str, Any],
    timestamp_offset_s: float,
) -> PassiveHomeScoreEvidence:
    if record.record_role != "home-raw-score-baseline/v1":
        raise ValueError("HOME evidence 只接受 HOME provider record")
    prediction = record.prediction
    timestamp = float(motion_row["timestamp_s"]) + timestamp_offset_s
    components = _score_components(prediction)
    score = components.to_object_write_evidence(
        geometry_valid=bool(prediction["geometry_valid"])
    ).score
    return PassiveHomeScoreEvidence(
        episode_id=record.episode_id,
        episode_generation=record.episode_generation,
        request_id=record.request_id,
        observation_sequence_id=record.observation_sequence_id,
        model_input_digest=record.model_input_digest,
        provider_output_digest=record.provider_output_digest,
        provider_identity=record.provider_identity,
        viewpoint_primitive_id=ACTIVE_FRONT_HOME_PRIMITIVE_ID,
        camera_motion_state=ExternalCameraMotionState.HOME_ANCHOR,
        settled=bool(motion_row["settled"]),
        score_components=components,
        stored_write_score=score,
        geometry_valid=bool(prediction["geometry_valid"]),
        control_timestamp_s=timestamp,
        rgb_timestamp_s=float(motion_row["external_rgb_timestamp_s"])
        + timestamp_offset_s,
        camera_pose_timestamp_s=float(motion_row["external_pose_timestamp_s"])
        + timestamp_offset_s,
        tcp_pose_timestamp_s=timestamp,
        base_from_external_camera_cv=np.asarray(
            prediction["base_from_external_camera_cv"], dtype=np.float64
        ),
    )


def build_stage2a_primary_frame_evidence(
    record: Stage2AProviderOutputRecord,
    *,
    motion_row: Mapping[str, Any],
    timestamp_offset_s: float,
) -> ActiveFrontStage2FrameEvidence:
    if record.record_role != "primary-settled-collect/v1":
        raise ValueError("PRIMARY frame 只接受 PRIMARY provider record")
    prediction = record.prediction
    components = _score_components(prediction)
    write_evidence = components.to_object_write_evidence(
        geometry_valid=bool(prediction["geometry_valid"])
    )
    normalized_uv = np.asarray(
        prediction["predicted_object_normalized_uv"], dtype=np.float64
    )
    projection_valid = bool(prediction["geometry_valid"])
    in_fov = bool(
        projection_valid
        and normalized_uv.shape == (2,)
        and np.isfinite(normalized_uv).all()
        and np.all((normalized_uv >= 0.0) & (normalized_uv <= 1.0))
    )
    observable = bool(write_evidence.observable and projection_valid and in_fov)
    timestamp = float(motion_row["timestamp_s"]) + timestamp_offset_s
    return ActiveFrontStage2FrameEvidence(
        episode_id=record.episode_id,
        episode_generation=record.episode_generation,
        request_id=record.request_id,
        source_phase=STAGE2A_SOURCE_PHASE,
        observation_sequence_id=record.observation_sequence_id,
        model_input_digest=record.model_input_digest,
        provider_output_digest=record.provider_output_digest,
        provider_identity=record.provider_identity,
        camera_motion_state=ExternalCameraMotionState.COLLECT,
        settled=bool(motion_row["settled"]),
        control_timestamp_s=timestamp,
        rgb_timestamp_s=float(motion_row["external_rgb_timestamp_s"])
        + timestamp_offset_s,
        camera_pose_timestamp_s=float(motion_row["external_pose_timestamp_s"])
        + timestamp_offset_s,
        tcp_pose_timestamp_s=timestamp,
        base_from_external_camera_cv=np.asarray(
            prediction["base_from_external_camera_cv"], dtype=np.float64
        ),
        position_base_m=prediction["predicted_object_position_base_m"],
        covariance_base_m2=prediction["calibrated_covariance_base_m2"],
        measurement_confidence=write_evidence.score,
        write_score=write_evidence.score,
        score_components=components,
        projection_valid=projection_valid,
        in_fov=in_fov,
        observable=observable,
        geometry_valid=bool(prediction["geometry_valid"]),
        structurally_eligible=bool(observable and prediction["geometry_valid"]),
        deployable_free_static_safe=bool(prediction["deployable_free_static_safe"]),
        qualification_only=False,
    )


def build_trigger_evidence_from_capability(
    record: WristCapabilityEvidenceRecord,
    *,
    control_tick: int,
    arm_hold_prerequisites_pass: bool,
    camera_home_prerequisites_pass: bool,
) -> ActiveFrontTriggerEvidence:
    """触发 bool 只从 capability/Memory record 派生，不接受裸 bool 注入。"""

    if record.record_role != "trigger":
        raise ValueError("trigger evidence 必须来自 trigger capability record")
    return ActiveFrontTriggerEvidence(
        episode_id=record.episode_id,
        episode_generation=record.episode_generation,
        control_tick=control_tick,
        timestamp_s=record.timestamp_s,
        source_phase=record.source_phase,
        wrist_object_measurement_usable=record.measurement_usable,
        front_home_object_measurement_usable=False,
        object_memory_navigation_state_available=(
            record.memory_resolution_available
        ),
        arm_hold_prerequisites_pass=arm_hold_prerequisites_pass,
        camera_home_prerequisites_pass=camera_home_prerequisites_pass,
        failure_reason=(
            ActiveFrontTriggerReason.NO_QUALIFIED_WRIST_PROVIDER_IN_PARENT
        ),
        object_contact=record.contact_authorized,
        gripper_close_commanded=False,
        grasp_candidate=False,
        grasp_verified=False,
        object_motion_risk=False,
    )


def build_source_recheck_from_capability(
    record: WristCapabilityEvidenceRecord,
    *,
    candidate_digest: str,
    camera_at_home: bool,
    source_invariants_passed: bool,
    active_window_open: bool,
) -> ActiveFrontSourceRecheckEvidence:
    if record.record_role != "source_recheck":
        raise ValueError("source recheck 必须来自独立 recheck capability record")
    return ActiveFrontSourceRecheckEvidence(
        episode_id=record.episode_id,
        episode_generation=record.episode_generation,
        request_id=record.request_id,
        candidate_digest=candidate_digest,
        timestamp_s=record.timestamp_s,
        source_phase=record.source_phase,
        camera_at_home=camera_at_home,
        source_invariants_passed=source_invariants_passed,
        active_window_open=active_window_open,
        qualified_direct_wrist_measurement_usable=record.measurement_usable,
        qualified_direct_wrist_evidence_identity_sha256=record.digest,
    )


def _stage2a_episode_id(seed: int) -> str:
    suffix = ACTIVE_FRONT_PRIMARY_PRIMITIVE_ID.lower().replace("_", "-")
    return f"stage2a-seed-{seed:06d}-{suffix}"


def _stage2a_capture_identity(
    *,
    seed: int,
    route_frame_index: int,
    row_index: int,
) -> dict[str, Any]:
    if route_frame_index not in STAGE2A_PROVIDER_FRAME_INDICES:
        raise ValueError("Stage 2A provider frame index 非法")
    alternate_index = FRONT_ALTERNATE_IDS.index(ACTIVE_FRONT_PRIMARY_PRIMITIVE_ID)
    home = route_frame_index == 0
    return {
        "seed": seed,
        "sample_index": STAGE2A_PROVIDER_FRAME_INDICES.index(route_frame_index),
        "viewpoint_id": (
            FRONT_HOME_ID if home else ACTIVE_FRONT_PRIMARY_PRIMITIVE_ID
        ),
        "frame_role": (
            "home-raw-score-baseline/v1"
            if home
            else "primary-settled-collect/v1"
        ),
        "route_alternate_index": alternate_index,
        "route_alternate_viewpoint_id": ACTIVE_FRONT_PRIMARY_PRIMITIVE_ID,
        "route_frame_index": route_frame_index,
        "row_index": row_index,
    }


def _rotation_distance_rad(left: np.ndarray, right: np.ndarray) -> float:
    relative = left.T @ right
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(math.acos(cosine))


def _stage2a_camera_at_home(row: Mapping[str, Any]) -> bool:
    actual = np.asarray(row["actual_base_from_external_camera_cv"], dtype=np.float64)
    expected = np.asarray(
        ACTIVE_FRONT_HOME_BASE_FROM_EXTERNAL_CAMERA_CV,
        dtype=np.float64,
    )
    return bool(
        row.get("camera_motion_state")
        == ExternalCameraMotionState.VERIFY_HOME_AND_ARM_HOLD.value
        and actual.shape == (4, 4)
        and np.isfinite(actual).all()
        and float(np.linalg.norm(actual[:3, 3] - expected[:3, 3]))
        <= ACTIVE_FRONT_HOME_POSITION_TOLERANCE_M + 1e-12
        and _rotation_distance_rad(expected[:3, :3], actual[:3, :3])
        <= ACTIVE_FRONT_HOME_ORIENTATION_TOLERANCE_RAD + 1e-12
    )


def _stage2a_pose_at_home(row: Mapping[str, Any]) -> bool:
    """不信任 stored bool；直接从 actual base<-camera pose 重算 HOME。"""

    actual = np.asarray(row["actual_base_from_external_camera_cv"], dtype=np.float64)
    expected = np.asarray(
        ACTIVE_FRONT_HOME_BASE_FROM_EXTERNAL_CAMERA_CV,
        dtype=np.float64,
    )
    return bool(
        actual.shape == (4, 4)
        and np.isfinite(actual).all()
        and float(np.linalg.norm(actual[:3, 3] - expected[:3, 3]))
        <= ACTIVE_FRONT_HOME_POSITION_TOLERANCE_M + 1e-12
        and _rotation_distance_rad(expected[:3, :3], actual[:3, :3])
        <= ACTIVE_FRONT_HOME_ORIENTATION_TOLERANCE_RAD + 1e-12
    )


def _stage2a_safety_scalars(row: Mapping[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for name in (
        "arm_joint_max_drift_rad",
        "tcp_position_drift_m",
        "tcp_orientation_drift_rad",
        "minimum_finger_joint_position_m",
        "finger_object_contact_force_n",
    ):
        raw = row.get(name)
        if (
            not isinstance(raw, (int, float))
            or isinstance(raw, bool)
            or not math.isfinite(float(raw))
        ):
            raise ValueError(f"Stage 2A raw safety witness 非法: {name}")
        values[name] = float(raw)
    return values


def _stage2a_active_safety(
    row: Mapping[str, Any],
    *,
    controller: ActiveFrontReobserveController,
) -> ActiveFrontSafetyEvidence:
    raw = _stage2a_safety_scalars(row)
    return ActiveFrontSafetyEvidence(
        arm_hold_pass=raw["arm_joint_max_drift_rad"] <= 1e-5 + 1e-12,
        tcp_hold_pass=bool(
            raw["tcp_position_drift_m"] <= 1e-5 + 1e-12
            and raw["tcp_orientation_drift_rad"] <= 1e-4 + 1e-12
        ),
        gripper_open_hold_pass=raw["minimum_finger_joint_position_m"]
        >= 0.039 - 1e-12,
        contact_absent=raw["finger_object_contact_force_n"] <= 0.01 + 1e-12,
        active_window_open=controller.active_window_open,
    )


def _stage2a_memory_safety(
    row: Mapping[str, Any],
    *,
    controller: ActiveFrontReobserveController,
) -> ObjectMemorySafetyContext:
    active = _stage2a_active_safety(row, controller=controller)
    return ObjectMemorySafetyContext(
        pregrasp_window_open=active.active_window_open,
        gripper_open=active.gripper_open_hold_pass,
        controller_tracking_valid=active.arm_hold_pass and active.tcp_hold_pass,
        object_contact_detected=not active.contact_absent,
        gripper_close_commanded=False,
        grasp_candidate=False,
        grasp_verified=False,
        object_maybe_moved=False,
    )


def _stage2a_safety_evidence_record(
    row: Mapping[str, Any],
    *,
    controller: ActiveFrontReobserveController,
) -> dict[str, Any]:
    raw = _stage2a_safety_scalars(row)
    evidence = _stage2a_active_safety(row, controller=controller)
    record = {
        "version": "e018-p1-stage2a-derived-safety-evidence/v1",
        "frame_index": int(row["frame_index"]),
        "motion_row_sha256": canonical_sha256(dict(row)),
        "raw": raw,
        "controller_active_window_open": controller.active_window_open,
        "derived": asdict(evidence),
    }
    record["evidence_sha256"] = canonical_sha256(record)
    return record


def _stage2a_home_barrier_evidence(
    row: Mapping[str, Any],
    observation: Mapping[str, Any],
    *,
    episode_id: str,
    request_id: str,
    observation_sequence_id: str,
    timestamp_offset_s: float,
    controller: ActiveFrontReobserveController,
) -> dict[str, Any]:
    control_timestamp = float(row["timestamp_s"]) + timestamp_offset_s
    payload = home_observation_payload_identity(
        observation,
        timestamp_s=control_timestamp,
    )
    actual_pose = np.asarray(
        row["actual_base_from_external_camera_cv"], dtype=np.float64
    )
    safety = _stage2a_safety_evidence_record(row, controller=controller)
    evidence = {
        "version": "e018-p1-stage2a-home-barrier-evidence/v1",
        "episode_id": episode_id,
        "episode_generation": 1,
        "request_id": request_id,
        "observation_sequence_id": observation_sequence_id,
        "route_frame_index": int(row["frame_index"]),
        "camera_motion_state": row["camera_motion_state"],
        "viewpoint_primitive_id": row["viewpoint_primitive_id"],
        "timestamp_source": row["timestamp_source"],
        "control_timestamp_s": control_timestamp,
        "rgb_timestamp_s": float(row["external_rgb_timestamp_s"])
        + timestamp_offset_s,
        "camera_pose_timestamp_s": float(row["external_pose_timestamp_s"])
        + timestamp_offset_s,
        "actual_pose_source": "same-observation.sensor_param.base_camera.cam2world_gl/v1",
        "actual_base_from_external_camera_cv": actual_pose.tolist(),
        "actual_pose_sha256": _array_sha256(actual_pose),
        "camera_at_home": _stage2a_camera_at_home(row),
        "home_observation_payload": payload,
        "home_observation_payload_digest": canonical_sha256(payload),
        "motion_row_sha256": canonical_sha256(dict(row)),
        "safety_evidence_sha256": safety["evidence_sha256"],
    }
    evidence["evidence_sha256"] = canonical_sha256(evidence)
    return evidence


def _serialize_receipt(value: Any) -> dict[str, Any]:
    payload = asdict(value)
    for key in ("source_phase", "resume_phase"):
        if isinstance(payload.get(key), PhaseId):
            payload[key] = payload[key].value
    payload["digest"] = value.digest
    return payload


_STAGE2A_ACTION_HISTORY_VERSION = "e018-p1-stage2a-action-history-shadow/v1"
_STAGE2A_ACTION_HISTORY_AUDIT_VERSION = (
    "e018-p1-stage2a-action-history-transition-audit/v1"
)


def _action_history_snapshot_digest(snapshot: Mapping[str, Any]) -> str:
    _require_exact_keys(
        dict(snapshot),
        {
            "version",
            "episode_id",
            "generation",
            "mode",
            "action_chunk_identity_sha256",
            "temporal_ensemble_identity_sha256",
            "rtc_overlap_identity_sha256",
            "command_reference_identity_sha256",
            "fresh_home_bundle_sha256",
        },
        "Action history snapshot",
    )
    if snapshot["version"] != _STAGE2A_ACTION_HISTORY_VERSION:
        raise ValueError("Action history snapshot version 漂移")
    episode_id = snapshot["episode_id"]
    generation = snapshot["generation"]
    mode = snapshot["mode"]
    if (
        not isinstance(episode_id, str)
        or not episode_id
        or type(generation) is not int
        or generation < 0
        or mode
        not in {
            "ready-shadow-history",
            "invalidated-for-active-reobserve",
            "fresh-shadow-replan",
        }
    ):
        raise ValueError("Action history snapshot identity/generation/mode 非法")
    identity_fields = (
        "action_chunk_identity_sha256",
        "temporal_ensemble_identity_sha256",
        "rtc_overlap_identity_sha256",
        "command_reference_identity_sha256",
    )
    for name in (*identity_fields, "fresh_home_bundle_sha256"):
        value = snapshot[name]
        if value is not None and not _is_sha256(value):
            raise ValueError(f"Action history snapshot {name} 非法")
    if mode == "ready-shadow-history":
        expected = (
            _sha256_text(f"{episode_id}:action-chunk:{generation}"),
            _sha256_text(f"{episode_id}:temporal-ensemble:{generation}"),
            _sha256_text(f"{episode_id}:rtc-overlap:{generation}"),
            _sha256_text(f"{episode_id}:command-reference:{generation}"),
        )
        if (
            tuple(snapshot[name] for name in identity_fields) != expected
            or snapshot["fresh_home_bundle_sha256"] is not None
        ):
            raise ValueError("Action history ready sentinel identity 漂移")
    elif mode == "invalidated-for-active-reobserve" and (
        any(snapshot[name] is not None for name in identity_fields)
        or snapshot["fresh_home_bundle_sha256"] is not None
    ):
        raise ValueError("Action history invalidated state 仍含 stale identity")
    return canonical_sha256(dict(snapshot))


def _action_receipt_payload(
    value: ActionHistoryResetReceipt | ActionHistoryResumeReceipt,
) -> dict[str, Any]:
    payload = asdict(value)
    return payload


def _build_action_history_audit(
    *,
    transition: str,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    receipt: ActionHistoryResetReceipt | ActionHistoryResumeReceipt,
    home_evidence_digests: Sequence[str] = (),
    memory_state_sha256: str | None = None,
    fresh_home_bundle: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    audit = {
        "version": _STAGE2A_ACTION_HISTORY_AUDIT_VERSION,
        "transition": transition,
        "before": dict(before),
        "after": dict(after),
        "before_sha256": _action_history_snapshot_digest(before),
        "after_sha256": _action_history_snapshot_digest(after),
        "receipt": _action_receipt_payload(receipt),
        "home_evidence_digests": list(home_evidence_digests),
        "memory_state_sha256": memory_state_sha256,
        "fresh_home_bundle": (
            None if fresh_home_bundle is None else dict(fresh_home_bundle)
        ),
    }
    audit["audit_sha256"] = canonical_sha256(audit)
    return audit


class Stage2AActionHistoryRuntime:
    """隔离 shadow runtime：receipt 只能由实际状态清空/新建动作产生。"""

    def __init__(self, episode_id: str) -> None:
        if not episode_id:
            raise ValueError("Action history episode_id 不能为空")
        self.episode_id = episode_id
        self.generation = 0
        self.mode = "ready-shadow-history"
        self.action_chunk_identity_sha256 = _sha256_text(f"{episode_id}:action-chunk:0")
        self.temporal_ensemble_identity_sha256 = _sha256_text(
            f"{episode_id}:temporal-ensemble:0"
        )
        self.rtc_overlap_identity_sha256 = _sha256_text(f"{episode_id}:rtc-overlap:0")
        self.command_reference_identity_sha256 = _sha256_text(
            f"{episode_id}:command-reference:0"
        )
        self.fresh_home_bundle_sha256: str | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "version": _STAGE2A_ACTION_HISTORY_VERSION,
            "episode_id": self.episode_id,
            "generation": self.generation,
            "mode": self.mode,
            "action_chunk_identity_sha256": self.action_chunk_identity_sha256,
            "temporal_ensemble_identity_sha256": (
                self.temporal_ensemble_identity_sha256
            ),
            "rtc_overlap_identity_sha256": self.rtc_overlap_identity_sha256,
            "command_reference_identity_sha256": (
                self.command_reference_identity_sha256
            ),
            "fresh_home_bundle_sha256": self.fresh_home_bundle_sha256,
        }

    def invalidate_for_active_request(
        self,
        request: Any,
    ) -> tuple[ActionHistoryResetReceipt, dict[str, Any]]:
        if request.episode_id != self.episode_id or self.mode != "ready-shadow-history":
            raise RuntimeError("Action history 只能从同 Episode ready 状态失效")
        before = self.snapshot()
        generation_before = self.generation
        self.action_chunk_identity_sha256 = None
        self.temporal_ensemble_identity_sha256 = None
        self.rtc_overlap_identity_sha256 = None
        self.command_reference_identity_sha256 = None
        self.fresh_home_bundle_sha256 = None
        self.generation += 1
        self.mode = "invalidated-for-active-reobserve"
        after = self.snapshot()
        receipt = ActionHistoryResetReceipt(
            episode_id=self.episode_id,
            request_id=request.request_id,
            reset_control_tick=request.trigger_tick,
            generation_before=generation_before,
            generation_after=self.generation,
            action_chunk_cleared=bool(
                before["action_chunk_identity_sha256"] is not None
                and after["action_chunk_identity_sha256"] is None
            ),
            temporal_ensemble_cleared=bool(
                before["temporal_ensemble_identity_sha256"] is not None
                and after["temporal_ensemble_identity_sha256"] is None
            ),
            rtc_overlap_cleared=bool(
                before["rtc_overlap_identity_sha256"] is not None
                and after["rtc_overlap_identity_sha256"] is None
            ),
            command_reference_invalidated=bool(
                before["command_reference_identity_sha256"] is not None
                and after["command_reference_identity_sha256"] is None
            ),
        )
        receipt.validate_for(request)
        return receipt, _build_action_history_audit(
            transition="atomic-invalidate-for-active-reobserve",
            before=before,
            after=after,
            receipt=receipt,
        )

    def generate_fresh_shadow_replan(
        self,
        request: Any,
        *,
        home_evidence: Sequence[Mapping[str, Any]],
        observation_v2_window_identity: Mapping[str, Any],
        memory_state: ObjectState,
        source_phase: PhaseId,
    ) -> tuple[ActionHistoryResumeReceipt, dict[str, Any]]:
        if (
            request.episode_id != self.episode_id
            or source_phase is not request.resume_phase
            or self.mode != "invalidated-for-active-reobserve"
            or any(
                getattr(self, name) is not None
                for name in (
                    "action_chunk_identity_sha256",
                    "temporal_ensemble_identity_sha256",
                    "rtc_overlap_identity_sha256",
                    "command_reference_identity_sha256",
                )
            )
        ):
            raise RuntimeError("fresh shadow replan 前 Action history 状态非法")
        if len(home_evidence) != 4:
            raise ValueError("fresh shadow replan 必须绑定四帧 HOME evidence")
        ids = tuple(str(value.get("observation_sequence_id", "")) for value in home_evidence)
        digests = tuple(str(value.get("evidence_sha256", "")) for value in home_evidence)
        timestamps = tuple(value.get("control_timestamp_s") for value in home_evidence)
        if (
            any(not value for value in ids)
            or len(set(ids)) != 4
            or any(not _is_sha256(value) for value in digests)
            or len(set(digests)) != 4
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                for value in timestamps
            )
            or any(
                float(timestamps[index]) >= float(timestamps[index + 1])
                for index in range(3)
            )
        ):
            raise ValueError("fresh HOME evidence identity/timestamp 非法")
        if memory_state.episode_id != self.episode_id or not memory_state.valid:
            raise ValueError("fresh shadow replan 要求同 Episode valid Memory")
        window_identity = dict(observation_v2_window_identity)
        window_sha256 = window_identity.get("window_sha256")
        window_primitive = dict(window_identity)
        window_primitive.pop("window_sha256", None)
        if (
            not _is_sha256(window_sha256)
            or canonical_sha256(window_primitive) != window_sha256
            or window_identity.get("episode_id") != self.episode_id
            or window_identity.get("episode_generation")
            != request.episode_generation
            or window_identity.get("observation_sequence_ids") != list(ids)
            or window_identity.get("home_evidence_digests") != list(digests)
        ):
            raise ValueError("fresh shadow replan Observation V2 window identity 漂移")
        memory_state_sha256 = canonical_sha256(_object_state_snapshot(memory_state))
        home_bundle = {
            "version": "e018-p1-stage2a-fresh-home-bundle/v1",
            "episode_id": self.episode_id,
            "request_id": request.request_id,
            "observation_sequence_ids": list(ids),
            "home_evidence_digests": list(digests),
            "control_timestamps_s": [float(value) for value in timestamps],
            "memory_state_sha256": memory_state_sha256,
            "source_phase": source_phase.value,
            "observation_v2_window": window_identity,
            "observation_v2_window_sha256": window_sha256,
        }
        bundle_sha256 = canonical_sha256(home_bundle)
        before = self.snapshot()
        next_generation = self.generation + 1
        self.action_chunk_identity_sha256 = _sha256_text(
            f"{bundle_sha256}:fresh-shadow-action-chunk:{next_generation}"
        )
        self.temporal_ensemble_identity_sha256 = _sha256_text(
            f"{bundle_sha256}:fresh-temporal-ensemble:{next_generation}"
        )
        self.rtc_overlap_identity_sha256 = _sha256_text(
            f"{bundle_sha256}:fresh-rtc-overlap:{next_generation}"
        )
        self.command_reference_identity_sha256 = _sha256_text(
            f"{bundle_sha256}:fresh-command-reference:{next_generation}"
        )
        self.fresh_home_bundle_sha256 = bundle_sha256
        self.generation = next_generation
        self.mode = "fresh-shadow-replan"
        after = self.snapshot()
        receipt = ActionHistoryResumeReceipt(
            episode_id=self.episode_id,
            request_id=request.request_id,
            generation=self.generation,
            home_observation_sequence_ids=ids,
            generated_from_fresh_home_v2=bool(
                self.fresh_home_bundle_sha256 == bundle_sha256
                and all(after[name] is not None for name in (
                    "action_chunk_identity_sha256",
                    "temporal_ensemble_identity_sha256",
                    "rtc_overlap_identity_sha256",
                    "command_reference_identity_sha256",
                ))
            ),
            stale_action_chunk_resumed=bool(
                after["action_chunk_identity_sha256"]
                == before["action_chunk_identity_sha256"]
            ),
            observation_v2_window_sha256=window_sha256,
        )
        return receipt, _build_action_history_audit(
            transition="fresh-home-shadow-replan",
            before=before,
            after=after,
            receipt=receipt,
            home_evidence_digests=digests,
            memory_state_sha256=memory_state_sha256,
            fresh_home_bundle=home_bundle,
        )


def verify_stage2a_action_history_audit(
    audit: Mapping[str, Any],
) -> ActionHistoryResetReceipt | ActionHistoryResumeReceipt:
    """从 before/after state 重算 receipt；拒绝调用方自报的成功布尔。"""

    value = _require_exact_keys(
        dict(audit),
        {
            "version",
            "transition",
            "before",
            "after",
            "before_sha256",
            "after_sha256",
            "receipt",
            "home_evidence_digests",
            "memory_state_sha256",
            "fresh_home_bundle",
            "audit_sha256",
        },
        "Action history audit",
    )
    primitive = dict(value)
    stored_audit_sha256 = primitive.pop("audit_sha256")
    if (
        value["version"] != _STAGE2A_ACTION_HISTORY_AUDIT_VERSION
        or stored_audit_sha256 != canonical_sha256(primitive)
        or value["before_sha256"]
        != _action_history_snapshot_digest(value["before"])
        or value["after_sha256"]
        != _action_history_snapshot_digest(value["after"])
    ):
        raise ValueError("Action history audit identity 漂移")
    before = value["before"]
    after = value["after"]
    if (
        before["episode_id"] != after["episode_id"]
        or type(before["generation"]) is not int
        or type(after["generation"]) is not int
        or after["generation"] != before["generation"] + 1
    ):
        raise ValueError("Action history generation/Episode 漂移")
    receipt_value = value["receipt"]
    if value["transition"] == "atomic-invalidate-for-active-reobserve":
        _require_exact_keys(
            receipt_value,
            {
                "episode_id",
                "request_id",
                "reset_control_tick",
                "generation_before",
                "generation_after",
                "action_chunk_cleared",
                "temporal_ensemble_cleared",
                "rtc_overlap_cleared",
                "command_reference_invalidated",
            },
            "Action history reset receipt",
        )
        receipt = ActionHistoryResetReceipt(**receipt_value)
        identity_fields = (
            "action_chunk_identity_sha256",
            "temporal_ensemble_identity_sha256",
            "rtc_overlap_identity_sha256",
            "command_reference_identity_sha256",
        )
        derived = {
            "action_chunk_cleared": bool(
                before[identity_fields[0]] is not None
                and after[identity_fields[0]] is None
            ),
            "temporal_ensemble_cleared": bool(
                before[identity_fields[1]] is not None
                and after[identity_fields[1]] is None
            ),
            "rtc_overlap_cleared": bool(
                before[identity_fields[2]] is not None
                and after[identity_fields[2]] is None
            ),
            "command_reference_invalidated": bool(
                before[identity_fields[3]] is not None
                and after[identity_fields[3]] is None
            ),
        }
        if (
            before["mode"] != "ready-shadow-history"
            or after["mode"] != "invalidated-for-active-reobserve"
            or before["fresh_home_bundle_sha256"] is not None
            or after["fresh_home_bundle_sha256"] is not None
            or receipt.episode_id != before["episode_id"]
            or receipt.generation_before != before["generation"]
            or receipt.generation_after != after["generation"]
            or any(getattr(receipt, name) is not expected for name, expected in derived.items())
            or not all(derived.values())
            or value["home_evidence_digests"] != []
            or value["memory_state_sha256"] is not None
            or value["fresh_home_bundle"] is not None
        ):
            raise ValueError("Action history reset 不是实际原子失效")
        return receipt
    if value["transition"] != "fresh-home-shadow-replan":
        raise ValueError("Action history transition 非法")
    _require_exact_keys(
        receipt_value,
        {
            "episode_id",
            "request_id",
            "generation",
            "home_observation_sequence_ids",
            "generated_from_fresh_home_v2",
            "stale_action_chunk_resumed",
            "observation_v2_window_sha256",
        },
        "Action history resume receipt",
    )
    receipt_payload = dict(receipt_value)
    receipt_payload["home_observation_sequence_ids"] = tuple(
        receipt_payload["home_observation_sequence_ids"]
    )
    receipt = ActionHistoryResumeReceipt(**receipt_payload)
    home_bundle = _require_exact_keys(
        value["fresh_home_bundle"],
        {
            "version",
            "episode_id",
            "request_id",
            "observation_sequence_ids",
            "home_evidence_digests",
            "control_timestamps_s",
            "memory_state_sha256",
            "source_phase",
            "observation_v2_window",
            "observation_v2_window_sha256",
        },
        "fresh HOME bundle",
    )
    bundle_sha256 = canonical_sha256(home_bundle)
    window_identity = home_bundle["observation_v2_window"]
    if not isinstance(window_identity, dict):
        raise TypeError("fresh HOME bundle 缺 Observation V2 window identity")
    window_primitive = dict(window_identity)
    window_digest = window_primitive.pop("window_sha256", None)
    identity_fields = (
        "action_chunk_identity_sha256",
        "temporal_ensemble_identity_sha256",
        "rtc_overlap_identity_sha256",
        "command_reference_identity_sha256",
    )
    expected_identities = (
        _sha256_text(f"{bundle_sha256}:fresh-shadow-action-chunk:{after['generation']}"),
        _sha256_text(f"{bundle_sha256}:fresh-temporal-ensemble:{after['generation']}"),
        _sha256_text(f"{bundle_sha256}:fresh-rtc-overlap:{after['generation']}"),
        _sha256_text(f"{bundle_sha256}:fresh-command-reference:{after['generation']}"),
    )
    timestamps = home_bundle["control_timestamps_s"]
    evidence_digests = home_bundle["home_evidence_digests"]
    if (
        before["mode"] != "invalidated-for-active-reobserve"
        or after["mode"] != "fresh-shadow-replan"
        or home_bundle["version"] != "e018-p1-stage2a-fresh-home-bundle/v1"
        or home_bundle["episode_id"] != before["episode_id"]
        or home_bundle["source_phase"] != STAGE2A_SOURCE_PHASE.value
        or not isinstance(home_bundle["observation_sequence_ids"], list)
        or len(home_bundle["observation_sequence_ids"]) != 4
        or len(set(home_bundle["observation_sequence_ids"])) != 4
        or not isinstance(evidence_digests, list)
        or len(evidence_digests) != 4
        or len(set(evidence_digests)) != 4
        or any(not _is_sha256(value) for value in evidence_digests)
        or not isinstance(timestamps, list)
        or len(timestamps) != 4
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            for value in timestamps
        )
        or any(
            float(timestamps[index + 1]) <= float(timestamps[index])
            for index in range(3)
        )
        or any(
            not math.isclose(
                float(timestamps[index + 1]) - float(timestamps[index]),
                1.0 / RobotSpec().control_hz,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            for index in range(3)
        )
        or not _is_sha256(home_bundle["memory_state_sha256"])
        or any(before[name] is not None for name in identity_fields)
        or tuple(after[name] for name in identity_fields) != expected_identities
        or after["fresh_home_bundle_sha256"] != bundle_sha256
        or value["home_evidence_digests"] != home_bundle["home_evidence_digests"]
        or value["memory_state_sha256"] != home_bundle["memory_state_sha256"]
        or not _is_sha256(window_digest)
        or canonical_sha256(window_primitive) != window_digest
        or home_bundle["observation_v2_window_sha256"] != window_digest
        or window_identity.get("episode_id") != before["episode_id"]
        or window_identity.get("observation_sequence_ids")
        != home_bundle["observation_sequence_ids"]
        or window_identity.get("home_evidence_digests")
        != home_bundle["home_evidence_digests"]
        or receipt.episode_id != before["episode_id"]
        or receipt.episode_id != home_bundle["episode_id"]
        or receipt.request_id != home_bundle["request_id"]
        or receipt.generation != after["generation"]
        or list(receipt.home_observation_sequence_ids)
        != home_bundle["observation_sequence_ids"]
        or receipt.generated_from_fresh_home_v2 is not True
        or receipt.stale_action_chunk_resumed is not False
        or receipt.observation_v2_window_sha256 != window_digest
    ):
        raise ValueError("Action history resume 不是 fresh HOME shadow replan")
    return receipt


class Stage2ARouteTransaction:
    """把单条 92-frame route 接到 trigger→Memory→fresh shadow Action。"""

    _TIMESTAMP_OFFSET_S = 0.10

    def __init__(
        self,
        *,
        seed: int,
        provider: QualificationProvider,
        stage2_config: LoadedStage2AConfig,
        qualification_config: Mapping[str, Any],
        data_config: Mapping[str, Any],
        base_env: Any,
        spec: Any,
        proprio_normalizer: Any,
        finger_force_normalizer: Any,
        execution_progress: Stage2AExecutionProgress,
    ) -> None:
        if seed not in STAGE2A_INTEGRATION_SMOKE_SEEDS:
            raise ValueError("Stage 2A transaction 只接受 76901..76910")
        self.seed = seed
        self.episode_id = _stage2a_episode_id(seed)
        self.provider = provider
        self.stage2_config = stage2_config
        self.qualification_config = qualification_config
        self.data_config = data_config
        self.base_env = base_env
        self.spec = spec
        self.proprio_normalizer = proprio_normalizer
        self.finger_force_normalizer = finger_force_normalizer
        self.execution_progress = execution_progress
        self.observation_adapter = FrankaObservationAdapter(self.spec)
        self.home_v2_history = ObservationV2History(self.spec)
        self.home_v2_window_identity: dict[str, Any] | None = None
        execution = stage2_config.payload["execution"]
        self.trigger_controller = ActiveFrontReobserveController(
            ActiveFrontReobserveConfig(
                enabled=True,
                selected_primitive_id=ACTIVE_FRONT_PRIMARY_PRIMITIVE_ID,
                consecutive_unusable_ticks=execution[
                    "capability_absent_consecutive_trigger_ticks"
                ],
                maximum_attempts_per_episode=1,
                home_v2_barrier_frames=4,
                allow_capability_absent_trigger=execution[
                    "allow_capability_absent_trigger"
                ],
            )
        )
        self.memory = ExplicitObjectStateMemory(
            build_stage2_object_memory_config(d049_primary_provider_identity())
        )
        self.orchestrator = ActiveFrontStage2MemoryOrchestrator(
            self.memory,
            config=stage2_config.runtime_config,
        )
        self.orchestrator.reset_episode(
            self.episode_id,
            episode_generation=1,
            timestamp_s=0.0,
        )
        self.trigger_controller.reset_episode(self.episode_id, episode_generation=1)
        self.action_history = Stage2AActionHistoryRuntime(self.episode_id)
        self.trigger_record: WristCapabilityEvidenceRecord | None = None
        self.trigger_capability_records: list[WristCapabilityEvidenceRecord] = []
        self.source_recheck_record: WristCapabilityEvidenceRecord | None = None
        self.reset_receipt: ActionHistoryResetReceipt | None = None
        self.reset_audit: dict[str, Any] | None = None
        self.resume_audit: dict[str, Any] | None = None
        self.controller_receipt: ActiveFrontReobserveReceipt | None = None
        self.candidate_stage_receipt: Stage2MemoryCandidateReceipt | None = None
        self.provider_records: list[Stage2AProviderOutputRecord] = []
        self.primary_frames: list[ActiveFrontStage2FrameEvidence] = []
        self.home_barrier_rows: list[Mapping[str, Any]] = []
        self.home_barrier_observations: list[Mapping[str, Any]] = []
        self.home_barrier_evidence: list[dict[str, Any]] = []
        self.safety_evidence: list[dict[str, Any]] = []
        self.camera_command_authorizations: list[dict[str, Any]] = []
        self.controller_events: list[dict[str, Any]] = []
        self.trigger_decisions: list[dict[str, Any]] = []
        self._request: Any | None = None
        self._last_warmup_observation: Mapping[str, Any] | None = None
        self._return_marked = False
        self._collect_orchestrator_rejections: list[str] = []
        self._sync_execution_progress()

    def _sync_execution_progress(self) -> None:
        """把当前 supervisor 状态复制到失败恢复 receipt。"""

        self.execution_progress.current_seed = self.seed
        self.execution_progress.episode_id = self.episode_id
        self.execution_progress.request_id = (
            None if self._request is None else self._request.request_id
        )
        self.execution_progress.controller_state = self.trigger_controller.state.value
        self.execution_progress.orchestrator_state = self.orchestrator.state.value
        self.execution_progress.provider_forward_count = len(self.provider_records)
        self.execution_progress.memory_write_count = self.orchestrator.memory_write_count

    def _advance_controller(
        self,
        signal: ActiveFrontSignal,
        *,
        frame_index: int,
        expected_state: ActiveFrontReobserveState,
        safety: ActiveFrontSafetyEvidence,
        selected_primitive_id: str | None = None,
        candidate_receipt: Stage2MemoryCandidateReceipt | None = None,
        source_phase: PhaseId | None = None,
        source_invariants_passed: bool | None = None,
    ) -> None:
        before = self.trigger_controller.state
        self.trigger_controller.advance(
            signal,
            safety=safety,
            selected_primitive_id=selected_primitive_id,
            shadow_candidate_receipt=candidate_receipt,
            source_phase=source_phase,
            source_invariants_passed=source_invariants_passed,
        )
        after = self.trigger_controller.state
        event = {
            "version": "e018-p1-stage2a-controller-event/v1",
            "frame_index": frame_index,
            "signal": signal.value,
            "state_before": before.value,
            "state_after": after.value,
            "selected_primitive_id": selected_primitive_id,
            "candidate_receipt_sha256": (
                None
                if candidate_receipt is None
                else canonical_sha256(asdict(candidate_receipt))
            ),
            "source_phase": None if source_phase is None else source_phase.value,
            "source_invariants_passed": source_invariants_passed,
            "safety": asdict(safety),
        }
        event["event_sha256"] = canonical_sha256(event)
        self.controller_events.append(event)
        self._sync_execution_progress()
        if after is not expected_state:
            raise RuntimeError(
                f"Stage 2A controller {signal.value} 未到达 {expected_state.value}: "
                f"{after.value}"
            )

    def pre_command_hook(
        self,
        state: ExternalCameraMotionState,
        frame_index: int,
        viewpoint_id: str,
    ) -> None:
        """在任何非 HOME pose mutation 前机械检查 lease/owner/controller 前态。"""

        self.execution_progress.current_frame_index = frame_index
        self._sync_execution_progress()

        expected = {
            ExternalCameraMotionState.MOVE_TO_VIEW: (
                ActiveFrontReobserveState.MOVE_TO_VIEW,
                ACTIVE_FRONT_PRIMARY_PRIMITIVE_ID,
            ),
            ExternalCameraMotionState.SETTLE_AT_VIEW: (
                ActiveFrontReobserveState.SETTLE_AT_VIEW,
                ACTIVE_FRONT_PRIMARY_PRIMITIVE_ID,
            ),
            ExternalCameraMotionState.COLLECT: (
                ActiveFrontReobserveState.COLLECT,
                ACTIVE_FRONT_PRIMARY_PRIMITIVE_ID,
            ),
            ExternalCameraMotionState.RETURN_HOME: (
                ActiveFrontReobserveState.RETURN_HOME,
                ACTIVE_FRONT_HOME_PRIMITIVE_ID,
            ),
            ExternalCameraMotionState.VERIFY_HOME_AND_ARM_HOLD: (
                ActiveFrontReobserveState.VERIFY_HOME_AND_ARM_HOLD,
                ACTIVE_FRONT_HOME_PRIMITIVE_ID,
            ),
        }
        expected_state, expected_viewpoint = expected[state]
        authorized = bool(
            frame_index >= 1
            and self._request is not None
            and self._request.selected_primitive_id
            == ACTIVE_FRONT_PRIMARY_PRIMITIVE_ID
            and self.trigger_controller.state is expected_state
            and self.trigger_controller.external_camera_owner
            is ExternalCameraControllerOwner.ACTIVE_REOBSERVE
            and self.trigger_controller.active_window_open
            and self.orchestrator.camera_lease_held
            and viewpoint_id == expected_viewpoint
            and (state is not ExternalCameraMotionState.RETURN_HOME or self._return_marked)
        )
        if not authorized:
            raise RuntimeError(
                "Stage 2A camera command 在 pose mutation 前未获 controller/lease 授权"
            )
        record = {
            "version": "e018-p1-stage2a-camera-command-authorization/v1",
            "frame_index": frame_index,
            "camera_motion_state": state.value,
            "viewpoint_primitive_id": viewpoint_id,
            "controller_state_before_command": self.trigger_controller.state.value,
            "external_camera_owner": (
                self.trigger_controller.external_camera_owner.value
            ),
            "camera_lease_held": self.orchestrator.camera_lease_held,
            "active_window_open": self.trigger_controller.active_window_open,
            "request_id": self._request.request_id,
            "camera_command_sequence_id": self._request.camera_command_sequence_id,
            "selected_primitive_id": self._request.selected_primitive_id,
            "authorized": authorized,
        }
        record["authorization_sha256"] = canonical_sha256(record)
        self.camera_command_authorizations.append(record)
        self.execution_progress.last_authorized_frame_index = frame_index
        self._sync_execution_progress()

    def warmup_hook(self, warmup_index: int, observation: Mapping[str, Any]) -> None:
        if warmup_index not in STAGE2A_TRIGGER_WARMUP_INDICES:
            return
        trigger_tick = STAGE2A_TRIGGER_WARMUP_INDICES.index(warmup_index)
        timestamp = trigger_tick * 0.05
        placeholder_request_id = f"{self.episode_id}-active-front-01"
        capability = build_absent_wrist_capability_record(
            observation=observation,
            episode_id=self.episode_id,
            episode_generation=1,
            request_id=placeholder_request_id,
            record_role="trigger",
            observation_sequence_id=(
                f"{self.episode_id}-trigger-home-{trigger_tick:02d}"
            ),
            timestamp_s=timestamp,
            memory_state=self.memory.state,
        )
        evidence = build_trigger_evidence_from_capability(
            capability,
            control_tick=trigger_tick,
            arm_hold_prerequisites_pass=True,
            camera_home_prerequisites_pass=True,
        )
        decision = self.trigger_controller.consider_trigger(evidence)
        self.trigger_capability_records.append(capability)
        decision_record = {
            "version": "e018-p1-stage2a-capability-trigger-decision/v1",
            "control_tick": trigger_tick,
            "timestamp_s": timestamp,
            "capability_evidence_identity_sha256": capability.digest,
            "requestable": decision.requestable,
            "reason": decision.reason.value,
            "consecutive_unusable_ticks": decision.consecutive_unusable_ticks,
        }
        decision_record["decision_sha256"] = canonical_sha256(decision_record)
        self.trigger_decisions.append(decision_record)
        if decision.requestable:
            if decision.request is None or decision.request.request_id != placeholder_request_id:
                raise RuntimeError("Stage 2A trigger request identity 漂移")
            self._request = decision.request
            self.trigger_record = capability
            self._last_warmup_observation = observation
        self._sync_execution_progress()

    def _provider_frame(
        self,
        row: Mapping[str, Any],
        rgb: np.ndarray,
    ) -> Stage2AProviderOutputRecord:
        frame_index = int(row["frame_index"])
        if self._request is None:
            raise RuntimeError("provider frame 早于三 Tick trigger")
        identity = _stage2a_capture_identity(
            seed=self.seed,
            route_frame_index=frame_index,
            row_index=self.provider.forward_count,
        )
        capture = build_qualification_deployable_capture(
            identity=identity,
            motion_row=row,
            rgb=rgb,
            base_env=self.base_env,
            spec=self.spec,
            proprio_normalizer=self.proprio_normalizer,
            finger_force_normalizer=self.finger_force_normalizer,
            data_config=self.data_config,
            eligible_capture_frame_indices=STAGE2A_PROVIDER_FRAME_INDICES,
        )
        prediction = self.provider.predict(capture)
        observation_sequence_id = (
            f"{self.episode_id}-route-frame-{frame_index:02d}"
        )
        record = build_stage2a_provider_output_record(
            capture=capture,
            prediction=prediction,
            episode_id=self.episode_id,
            episode_generation=1,
            request_id=self._request.request_id,
            observation_sequence_id=observation_sequence_id,
            route_frame_index=frame_index,
            stage2_config=self.stage2_config,
            qualification_config=self.qualification_config,
        )
        self.provider_records.append(record)
        return record

    def frame_hook(
        self,
        row: dict[str, Any],
        rgb: np.ndarray,
        observation: Mapping[str, Any],
    ) -> None:
        """记录成功处理边界；异常时保留失败帧与最近完成帧。"""

        frame_index = int(row["frame_index"])
        self.execution_progress.current_frame_index = frame_index
        self._sync_execution_progress()
        try:
            self._process_frame(row, rgb, observation)
        except Exception:
            self._sync_execution_progress()
            raise
        self.execution_progress.last_processed_frame_index = frame_index
        self._sync_execution_progress()

    def _process_frame(
        self,
        row: dict[str, Any],
        rgb: np.ndarray,
        observation: Mapping[str, Any],
    ) -> None:
        frame_index = int(row["frame_index"])
        if row.get("offline_segmentation_diagnostics") is not None:
            raise RuntimeError("Stage 2A hook 禁止 segmentation diagnostics")
        if row.get("is_grasping") is not None:
            raise RuntimeError("Stage 2A hook 禁止 is_grasping witness")
        if row.get("robot_object_contact_force_n") is not None:
            raise RuntimeError("Stage 2A hook 禁止 robot-object privileged witness")

        safety_record = _stage2a_safety_evidence_record(
            row,
            controller=self.trigger_controller,
        )
        self.safety_evidence.append(safety_record)
        safety = _stage2a_active_safety(
            row,
            controller=self.trigger_controller,
        )

        if (
            frame_index > 0
            and frame_index not in STAGE2A_HOME_BARRIER_FRAME_INDICES
            and not self.trigger_controller.observe_safety(
                safety,
                camera_at_home=bool(
                    row["camera_motion_state"]
                    in {
                        ExternalCameraMotionState.RETURN_HOME.value,
                        ExternalCameraMotionState.VERIFY_HOME_AND_ARM_HOLD.value,
                    }
                    and _stage2a_pose_at_home(row)
                ),
            )
        ):
            raise RuntimeError("Stage 2A route safety witness fail-closed")

        if frame_index in STAGE2A_PROVIDER_FRAME_INDICES:
            record = self._provider_frame(row, rgb)
            if frame_index == 0:
                if self.trigger_record is None or self._request is None:
                    raise RuntimeError("Stage 2A HOME baseline 缺少 trigger record")
                home = build_stage2a_home_score_evidence(
                    record,
                    motion_row=row,
                    timestamp_offset_s=self._TIMESTAMP_OFFSET_S,
                )
                baseline = PassiveBaselineEvidence(
                    episode_id=self.episode_id,
                    episode_generation=1,
                    request_id=self._request.request_id,
                    timestamp_s=self._request.trigger_timestamp_s,
                    wrist_object_measurement_usable=(
                        self.trigger_record.measurement_usable
                    ),
                    wrist_evidence_identity_sha256=self.trigger_record.digest,
                    home_front=home,
                    object_memory_navigation_state_available=(
                        self.trigger_record.memory_resolution_available
                    ),
                    object_memory_age_s=None,
                    object_memory_source_identity=None,
                )
                self.reset_receipt, self.reset_audit = (
                    self.action_history.invalidate_for_active_request(self._request)
                )
                self.trigger_controller.begin(self.reset_receipt)
                self._advance_controller(
                    ActiveFrontSignal.CAMERA_LEASE_ACQUIRED,
                    frame_index=frame_index,
                    expected_state=ActiveFrontReobserveState.SELECT_FROZEN_PRIMITIVE,
                    safety=safety,
                )
                self._advance_controller(
                    ActiveFrontSignal.FROZEN_PRIMITIVE_SELECTED,
                    frame_index=frame_index,
                    expected_state=ActiveFrontReobserveState.MOVE_TO_VIEW,
                    safety=safety,
                    selected_primitive_id=ACTIVE_FRONT_PRIMARY_PRIMITIVE_ID,
                )
                self.orchestrator.begin_collection(
                    self._request,
                    reset_receipt=self.reset_receipt,
                    baseline=baseline,
                )
            else:
                frame = build_stage2a_primary_frame_evidence(
                    record,
                    motion_row=row,
                    timestamp_offset_s=self._TIMESTAMP_OFFSET_S,
                )
                self.primary_frames.append(frame)
                if self.orchestrator.state is PendingActiveViewState.COLLECTING:
                    adaptation = self.orchestrator.observe_collect_frame(
                        frame,
                        safety=_stage2a_memory_safety(
                            row,
                            controller=self.trigger_controller,
                        ),
                    )
                    self._collect_orchestrator_rejections.extend(
                        adaptation.rejection_reasons
                    )

        if frame_index == 40:
            self._advance_controller(
                ActiveFrontSignal.MOVE_COMPLETE,
                frame_index=frame_index,
                expected_state=ActiveFrontReobserveState.SETTLE_AT_VIEW,
                safety=safety,
            )
        elif frame_index == 44:
            if row.get("settled") is not True:
                raise RuntimeError("Stage 2A settle completion 缺少实际 settled evidence")
            self._advance_controller(
                ActiveFrontSignal.SETTLE_COMPLETE,
                frame_index=frame_index,
                expected_state=ActiveFrontReobserveState.COLLECT,
                safety=safety,
            )
        elif frame_index == 47:
            if len(self.primary_frames) != 3:
                raise RuntimeError("Stage 2A collection complete 缺少三帧 evidence")
            self._advance_controller(
                ActiveFrontSignal.COLLECTION_COMPLETE,
                frame_index=frame_index,
                expected_state=ActiveFrontReobserveState.STAGE_CANDIDATE,
                safety=safety,
            )
            candidate = self.orchestrator.pending_candidate
            rejection_reasons = tuple(self.orchestrator.terminal_reasons)
            if candidate is None:
                rejection_reasons = tuple(
                    dict.fromkeys((*rejection_reasons, "candidate_not_constructed"))
                )
                candidate_digest = canonical_sha256(
                    {
                        "version": "e018-p1-stage2a-rejected-candidate-outcome/v1",
                        "request_id": self._request.request_id,
                        "collect_frame_digests": [
                            value.frame_digest for value in self.primary_frames
                        ],
                        "rejection_reasons": list(rejection_reasons),
                    }
                )
                commit_eligible = False
            else:
                candidate_digest = candidate.digest
                commit_eligible = candidate.commit_eligible
                rejection_reasons = candidate.rejection_reasons
            self.candidate_stage_receipt = Stage2MemoryCandidateReceipt(
                request_id=self._request.request_id,
                candidate_digest=candidate_digest,
                commit_eligible=commit_eligible,
                rejection_reasons=rejection_reasons,
                memory_write_deferred=commit_eligible,
                live_memory_write_executed=False,
                provider_forward_count=3,
                collect_frame_digests=tuple(
                    value.frame_digest for value in self.primary_frames
                ),
            )
            self._advance_controller(
                ActiveFrontSignal.SHADOW_CANDIDATE_STAGED,
                frame_index=frame_index,
                expected_state=ActiveFrontReobserveState.RETURN_HOME,
                safety=safety,
                candidate_receipt=self.candidate_stage_receipt,
            )
            self.orchestrator.mark_returning_home(
                timestamp_s=(
                    float(row["timestamp_s"]) + self._TIMESTAMP_OFFSET_S + 0.001
                ),
                candidate_digest=None if candidate is None else candidate.digest,
            )
            self._return_marked = True
        elif frame_index == 87:
            if not _stage2a_pose_at_home(row):
                raise RuntimeError("Stage 2A return complete 缺少 actual HOME pose")
            self._advance_controller(
                ActiveFrontSignal.RETURN_HOME_COMPLETE,
                frame_index=frame_index,
                expected_state=ActiveFrontReobserveState.VERIFY_HOME_AND_ARM_HOLD,
                safety=safety,
            )

        if frame_index in STAGE2A_HOME_BARRIER_FRAME_INDICES:
            if not self._return_marked:
                raise RuntimeError("HOME barrier 早于 return-home transaction")
            sequence_id = f"{self.episode_id}-home-v2-{frame_index:02d}"
            control_timestamp = (
                float(row["timestamp_s"]) + self._TIMESTAMP_OFFSET_S
            )
            observation_v2_frame = _build_stage2a_observation_v2_frame(
                observation,
                base_env=self.base_env,
                observation_adapter=self.observation_adapter,
                spec=self.spec,
                timestamp_s=control_timestamp,
            )
            if not math.isclose(
                float(np.max(observation_v2_frame.finger_force_n)),
                float(row["finger_object_contact_force_n"]),
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise RuntimeError("Observation V2 F_L/F_R 与 safety contact witness 漂移")
            self.home_v2_history.append(observation_v2_frame)
            home_frame = HomeV2BarrierFrame(
                observation_sequence_id=sequence_id,
                camera_at_home=_stage2a_camera_at_home(row),
                fresh_observation_v2_frame=True,
                captured_after_return=True,
                contains_alternate_or_motion_rgb=False,
            )
            home_evidence = _stage2a_home_barrier_evidence(
                row,
                observation,
                episode_id=self.episode_id,
                request_id=self._request.request_id,
                observation_sequence_id=sequence_id,
                timestamp_offset_s=self._TIMESTAMP_OFFSET_S,
                controller=self.trigger_controller,
            )
            prospective_home_evidence = [*self.home_barrier_evidence, home_evidence]
            if frame_index == STAGE2A_HOME_BARRIER_FRAME_INDICES[-1]:
                window = self.home_v2_history.snapshot(
                    _STAGE2A_SHADOW_REPLAN_INSTRUCTION,
                    previous_command_q=None,
                    previous_action=None,
                )
                self.home_v2_window_identity = (
                    _build_observation_v2_window_identity(
                        window,
                        spec=self.spec,
                        episode_id=self.episode_id,
                        episode_generation=1,
                        observation_sequence_ids=(
                            *self.orchestrator.home_observation_sequence_ids,
                            sequence_id,
                        ),
                        home_evidence=prospective_home_evidence,
                    )
                )
            self.orchestrator.accept_home_v2_barrier_frame(
                home_frame,
                timestamp_s=control_timestamp,
                safety=safety,
            )
            if safety.failure() is not None:
                self.trigger_controller.observe_safety(safety, camera_at_home=True)
                raise RuntimeError("Stage 2A HOME barrier safety fail-closed")
            self.trigger_controller.accept_home_v2_barrier_frame(home_frame)
            self.home_barrier_rows.append(row)
            self.home_barrier_observations.append(observation)
            self.home_barrier_evidence.append(home_evidence)
            if frame_index == STAGE2A_HOME_BARRIER_FRAME_INDICES[-1]:
                if self.candidate_stage_receipt is None:
                    raise RuntimeError("Stage 2A HOME barrier 缺少 staged candidate")
                if self.candidate_stage_receipt.commit_eligible:
                    if (
                        self.orchestrator.state
                        is not PendingActiveViewState.HOME_BARRIER_PASSED
                        or self.trigger_controller.state
                        is not ActiveFrontReobserveState.RECHECK_SOURCE_INVARIANTS
                    ):
                        raise RuntimeError("Stage 2A HOME barrier controller/orchestrator 不同步")
                elif (
                    self.orchestrator.state
                    is not PendingActiveViewState.HOME_VERIFIED_FAILED_SAFE_HOLD
                    or self.trigger_controller.state
                    is not ActiveFrontReobserveState.FAILED_SAFE_HOLD
                ):
                    raise RuntimeError("Stage 2A rejected candidate HOME recovery 不同步")

    def finalize(self, route_summary: Mapping[str, Any]) -> dict[str, Any]:
        if (
            self._request is None
            or self.trigger_record is None
            or len(self.trigger_capability_records) != 3
            or self.reset_receipt is None
            or self.reset_audit is None
            or self.candidate_stage_receipt is None
            or tuple(record.route_frame_index for record in self.provider_records)
            != STAGE2A_PROVIDER_FRAME_INDICES
            or len(self.primary_frames) != 3
            or len(self.home_barrier_rows) != 4
            or len(self.home_barrier_observations) != 4
            or len(self.home_barrier_evidence) != 4
            or self.home_v2_window_identity is None
            or len(self.safety_evidence) != 92
            or len(self.camera_command_authorizations) != 91
        ):
            raise RuntimeError("Stage 2A route transaction evidence 不完整")
        final_row = self.home_barrier_rows[-1]
        final_observation = self.home_barrier_observations[-1]
        recheck_timestamp = (
            float(final_row["timestamp_s"]) + self._TIMESTAMP_OFFSET_S + 0.001
        )
        self.source_recheck_record = build_absent_wrist_capability_record(
            observation=final_observation,
            episode_id=self.episode_id,
            episode_generation=1,
            request_id=self._request.request_id,
            record_role="source_recheck",
            observation_sequence_id=f"{self.episode_id}-source-recheck-home",
            timestamp_s=recheck_timestamp,
            memory_state=self.memory.state,
        )
        if self.source_recheck_record.digest == self.trigger_record.digest:
            raise RuntimeError("trigger/source recheck capability record 必须独立")

        commit_receipt = None
        shadow_action_receipt = None
        if (
            self.orchestrator.state is PendingActiveViewState.HOME_BARRIER_PASSED
            and self.trigger_controller.state
            is ActiveFrontReobserveState.RECHECK_SOURCE_INVARIANTS
        ):
            candidate = self.orchestrator.pending_candidate
            if candidate is None:
                raise RuntimeError("HOME barrier passed 却缺少 pending candidate")
            recheck = build_source_recheck_from_capability(
                self.source_recheck_record,
                candidate_digest=candidate.digest,
                camera_at_home=_stage2a_camera_at_home(final_row),
                source_invariants_passed=bool(route_summary.get("passed")),
                active_window_open=self.trigger_controller.active_window_open,
            )
            recheck_passed = self.orchestrator.recheck_source(
                recheck,
                safety=_stage2a_active_safety(
                    final_row,
                    controller=self.trigger_controller,
                ),
            )
            expected_controller_state = (
                ActiveFrontReobserveState.COMMIT_AND_RESUME
                if recheck_passed
                else ActiveFrontReobserveState.FAILED_SAFE_HOLD
            )
            self._advance_controller(
                ActiveFrontSignal.SOURCE_INVARIANTS_VERIFIED,
                frame_index=STAGE2A_HOME_BARRIER_FRAME_INDICES[-1],
                expected_state=expected_controller_state,
                safety=_stage2a_active_safety(
                    final_row,
                    controller=self.trigger_controller,
                ),
                source_phase=STAGE2A_SOURCE_PHASE,
                source_invariants_passed=bool(route_summary.get("passed")),
            )
            if recheck_passed != (
                self.trigger_controller.state
                is ActiveFrontReobserveState.COMMIT_AND_RESUME
            ):
                raise RuntimeError("Stage 2A source recheck controller/orchestrator 不同步")
            if recheck_passed:
                commit_receipt = self.orchestrator.commit(
                    candidate_digest=candidate.digest,
                    commit_timestamp_s=recheck_timestamp + 0.001,
                    safety=_stage2a_memory_safety(
                        final_row,
                        controller=self.trigger_controller,
                    ),
                )
                resume, self.resume_audit = (
                    self.action_history.generate_fresh_shadow_replan(
                        self._request,
                        home_evidence=self.home_barrier_evidence,
                        observation_v2_window_identity=(
                            self.home_v2_window_identity
                        ),
                        memory_state=self.memory.state,
                        source_phase=STAGE2A_SOURCE_PHASE,
                    )
                )
                shadow_action_receipt = (
                    self.orchestrator.create_shadow_action_generation(
                        resume,
                        source_phase=STAGE2A_SOURCE_PHASE,
                        source_phase_stability_reset=True,
                        source_phase_stability_ticks=0,
                    )
                )
                self.controller_receipt = (
                    self.trigger_controller.complete_stage2_memory_write(
                        resume,
                        memory_write_count=self.orchestrator.memory_write_count,
                        provider_forward_count=len(self.provider_records),
                    )
                )
                if (
                    self.trigger_controller.state
                    is not ActiveFrontReobserveState.COMPLETE_STAGE2_MEMORY_WRITE
                    or self.orchestrator.state is not PendingActiveViewState.COMMITTED
                    or self.controller_receipt.memory_write_count != 1
                ):
                    raise RuntimeError("Stage 2A commit/fresh shadow replan 终态不一致")

        if self.controller_receipt is None:
            self.controller_receipt = self.trigger_controller.receipt(
                memory_write_count=self.orchestrator.memory_write_count,
                provider_forward_count=len(self.provider_records),
            )

        candidate = self.orchestrator.pending_candidate
        final_memory = _object_state_snapshot(self.memory.state)
        candidate_stage = asdict(self.candidate_stage_receipt)
        candidate_stage["receipt_sha256"] = canonical_sha256(candidate_stage)
        return {
            "version": E018_P1_STAGE2A_EXECUTION_VERSION,
            "classification": "engineering-integration-smoke",
            "effect_claim": "no-effect-claim",
            "wrist_capability": "not-evaluated",
            "seed": self.seed,
            "episode_id": self.episode_id,
            "request_id": self._request.request_id,
            "trigger_decisions": self.trigger_decisions,
            "trigger_wrist_capability_records": [
                value.to_dict() for value in self.trigger_capability_records
            ],
            "trigger_wrist_capability": self.trigger_record.to_dict(),
            "source_recheck_wrist_capability": self.source_recheck_record.to_dict(),
            "capability_absence_trigger_reason": (
                ActiveFrontTriggerReason.NO_QUALIFIED_WRIST_PROVIDER_IN_PARENT.value
            ),
            "provider_output_digests": [
                value.provider_output_digest for value in self.provider_records
            ],
            "provider_frame_indices": [
                value.route_frame_index for value in self.provider_records
            ],
            "primary_frame_digests": [
                value.frame_digest for value in self.primary_frames
            ],
            "candidate_stage_receipt": candidate_stage,
            "candidate": None if candidate is None else candidate.as_dict(),
            "candidate_digest": None if candidate is None else candidate.digest,
            "candidate_commit_eligible": bool(
                candidate is not None and candidate.commit_eligible
            ),
            "collect_rejection_reasons": list(
                dict.fromkeys(self._collect_orchestrator_rejections)
            ),
            "terminal_reasons": list(self.orchestrator.terminal_reasons),
            "orchestrator_state": self.orchestrator.state.value,
            "home_barrier_frame_indices": list(STAGE2A_HOME_BARRIER_FRAME_INDICES),
            "home_observation_sequence_ids": list(
                self.orchestrator.home_observation_sequence_ids
            ),
            "home_observation_timestamps_s": list(
                self.orchestrator.home_observation_timestamps_s
            ),
            "home_frame_digests": list(self.orchestrator.home_frame_digests),
            "home_barrier_evidence": self.home_barrier_evidence,
            "observation_v2_window_identity": self.home_v2_window_identity,
            "safety_evidence": self.safety_evidence,
            "camera_command_authorizations": self.camera_command_authorizations,
            "controller_events": self.controller_events,
            "controller_receipt": {
                **self.controller_receipt.as_dict(),
                "audit_digest": self.controller_receipt.audit_digest,
            },
            "action_history_reset_audit": self.reset_audit,
            "action_history_resume_audit": self.resume_audit,
            "memory_write_count": self.orchestrator.memory_write_count,
            "commit_receipt": (
                None if commit_receipt is None else _serialize_receipt(commit_receipt)
            ),
            "shadow_action_generation": (
                None
                if shadow_action_receipt is None
                else _serialize_receipt(shadow_action_receipt)
            ),
            "final_memory_state": final_memory,
            "route_passed": bool(route_summary.get("passed")),
            "runtime_object_gt_reads": 0,
            "goal_gt_reads": 0,
            "offline_label_reads": 0,
            "wrist_provider_forward_count": 0,
            "arm_motion_command_count": 0,
            "gripper_close_command_count": 0,
            "fresh_test_reads": 0,
            "checkpoint_writes": 0,
        }


STAGE2A_INTEGRATION_SMOKE_GO = (
    "GO-E018-P1-STAGE2A-76901-76910-INTEGRATION-SMOKE"
)
_STAGE2A_ARTIFACT_FILES_BEFORE_FREEZE = (
    "camera_pose_ledger.jsonl",
    "provider_output_ledger.jsonl",
    "route_summaries.jsonl",
    "transaction_ledger.jsonl",
)


def verify_stage2a_parent_gate(
    *,
    stage2_config_path: str | Path,
    qualification_config_path: str | Path,
    qualification_public_execution_root: str | Path,
    qualification_result_root: str | Path,
    g0c_config_path: str | Path,
    data_config_path: str | Path,
) -> dict[str, Any]:
    loaded = load_e018_p1_stage2a_config(stage2_config_path)
    qualification_path = Path(qualification_config_path)
    if file_sha256(qualification_path) != loaded.payload["parents"][
        "d048_qualification_config_raw_sha256"
    ]:
        raise RuntimeError("Stage 2A D048 qualification config raw SHA 漂移")
    qualification = load_g2c_dynamic_qualification_config(qualification_path)
    if qualification["config_sha256"] != loaded.payload["parents"][
        "d048_qualification_config_internal_sha256"
    ]:
        raise RuntimeError("Stage 2A D048 qualification config internal SHA 漂移")
    result = verify_g2c_qualification_result(
        qualification_config_path=qualification_path,
        public_execution_root=qualification_public_execution_root,
        result_root=qualification_result_root,
    )
    parents = loaded.payload["parents"]
    if (
        result.get("gate_passed") is not True
        or result.get("classification")
        != "formal-dynamic-qualification-no-test-no-memory-no-manipulation/v1"
        or result.get("config_sha256")
        != parents["d048_qualification_config_internal_sha256"]
        or result.get("source_identity_sha256")
        != parents["d048_source_identity_sha256"]
        or result.get("receipt_raw_sha256")
        != parents["d048_result_receipt_raw_sha256"]
        or result.get("receipt_internal_sha256")
        != parents["d048_result_receipt_internal_sha256"]
        or result.get("verification_sha256")
        != parents["d048_result_verification_sha256"]
        or result.get("primary_viewpoint_id") != ACTIVE_FRONT_PRIMARY_PRIMITIVE_ID
    ):
        raise RuntimeError("Stage 2A D048 formal result parent gate 漂移")
    g0c = _g0c.load_e018_p1_g0c_config(g0c_config_path)
    data = load_e018_p1_g2c_data_config(
        data_config_path,
        parent_g0c_config_path=g0c_config_path,
    )
    if (
        _g0._canonical_sha256(g0c) != data["parents"]["g0c_config_sha256"]
        or data["environment"]["external_camera_uid"]
        != g0c["environment"]["camera_uid"]
    ):
        raise RuntimeError("Stage 2A G0C/DATA parent identity 漂移")
    verification = {
        "version": "e018-p1-stage2a-parent-verification/v1",
        "stage2_config_raw_sha256": loaded.raw_sha256,
        "stage2_config_canonical_sha256": loaded.canonical_sha256,
        "qualification_config_raw_sha256": file_sha256(qualification_path),
        "qualification_config_internal_sha256": qualification["config_sha256"],
        "d048_result_verification_sha256": result["verification_sha256"],
        "d048_primary_viewpoint_id": result["primary_viewpoint_id"],
        "g0c_config_sha256": _g0._canonical_sha256(g0c),
        "data_config_sha256": canonical_sha256(data),
        "d050_absent_wrist_capability_commit": (
            _D050_ABSENT_WRIST_CAPABILITY_COMMIT
        ),
        "d050_experiment_id": _D050_EXPERIMENT_ID,
    }
    verification["verification_sha256"] = canonical_sha256(verification)
    return verification


def _assert_stage2a_run_authority(
    *,
    stage2_config: LoadedStage2AConfig,
    repository_root: Path,
    expected_stage2_config_raw_sha256: str,
    expected_stage2_config_canonical_sha256: str,
    expected_source_git_commit: str,
    expected_source_identity_sha256: str,
    integration_smoke_go: str,
) -> dict[str, Any]:
    if integration_smoke_go != STAGE2A_INTEGRATION_SMOKE_GO:
        raise PermissionError("Stage 2A integration smoke 缺少 exact GO token")
    for value, name in (
        (expected_stage2_config_raw_sha256, "expected Stage2 raw SHA"),
        (expected_stage2_config_canonical_sha256, "expected Stage2 canonical SHA"),
        (expected_source_identity_sha256, "expected source identity SHA"),
    ):
        if not _is_sha256(value):
            raise ValueError(f"{name} 非法")
    if (
        stage2_config.raw_sha256 != expected_stage2_config_raw_sha256
        or stage2_config.canonical_sha256
        != expected_stage2_config_canonical_sha256
    ):
        raise RuntimeError("Stage 2A config 不匹配预审 exact identity")
    source = _git_source_identity(repository_root)
    if (
        source["git_commit"] != expected_source_git_commit
        or source["identity_sha256"] != expected_source_identity_sha256
    ):
        raise RuntimeError("Stage 2A exact-clean source identity 漂移")
    return source


def _run_stage2a_simulator(
    *,
    loaded_stage2_config: LoadedStage2AConfig,
    qualification_config: Mapping[str, Any],
    g0c_config: dict[str, Any],
    data_config: Mapping[str, Any],
    stats_root: Path,
    selected_checkpoint_path: Path,
    output_root: Path,
    execution_progress: Stage2AExecutionProgress,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[Stage2AProviderOutputRecord],
    dict[str, Any],
    bool,
]:
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
        raise RuntimeError("Stage 2A GPU/ManiSkill/SAPIEN environment 漂移")
    spec, proprio, force, normalizer_identity = _load_normalizers(
        stats_root=stats_root,
        config=data_config,
    )
    provider = QualificationProvider(
        checkpoint_path=selected_checkpoint_path,
        qualification_config=qualification_config,
        data_config=data_config,
        classification=QUALIFICATION_CLASSIFICATION_SMOKE,
    )
    home, anchors, orientations = _g0c._parse_library(g0c_config)
    primitives = _g0c._expand_primitives(anchors, orientations)
    by_id = {item.viewpoint_id: (item, orientation) for item, orientation in primitives}
    if tuple(by_id) != FRONT_ALTERNATE_IDS:
        provider.destroy()
        raise RuntimeError("Stage 2A G0C primitive order 漂移")
    primary, primary_orientation = by_id[ACTIVE_FRONT_PRIMARY_PRIMITIVE_ID]
    route_config = json.loads(json.dumps(g0c_config))
    route_config["experiment"]["offline_segmentation_diagnostics"] = False
    route_config["experiment"]["save_settled_rgb"] = False
    register_robot_vla_maniskill_envs()
    environment = route_config["environment"]
    env: Any | None = None
    env_closed = False
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    transactions: list[dict[str, Any]] = []
    provider_records: list[Stage2AProviderOutputRecord] = []
    sensor: Any | None = None
    camera: Any | None = None
    try:
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
            raise RuntimeError("Stage 2A environment/control/camera identity 漂移")
        sensor = base_env._sensors[environment["camera_uid"]]
        camera = sensor.camera
        if sensor.entity is not None or not callable(getattr(camera, "set_local_pose", None)):
            raise RuntimeError("Stage 2A 要求 isolated unmounted external camera")
        for seed in STAGE2A_INTEGRATION_SMOKE_SEEDS:
            execution_progress.begin_seed(seed)
            transaction = Stage2ARouteTransaction(
                seed=seed,
                provider=provider,
                stage2_config=loaded_stage2_config,
                qualification_config=qualification_config,
                data_config=data_config,
                base_env=base_env,
                spec=spec,
                proprio_normalizer=proprio,
                finger_force_normalizer=force,
                execution_progress=execution_progress,
            )
            route_rows, route_summary, _ = _g0._run_route(
                env=env,
                base_env=base_env,
                camera=camera,
                config=route_config,
                seed=seed,
                home=home,
                alternate=primary,
                output_root=output_root,
                sapien_module=sapien,
                sapien_utils_module=sapien_utils,
                alternate_orientation=primary_orientation,
                result_version=E018_P1_STAGE2A_EXECUTION_VERSION,
                episode_prefix="stage2a",
                source_phase=STAGE2A_SOURCE_PHASE.value,
                camera_owner=STAGE2A_CAMERA_OWNER,
                frame_hook=transaction.frame_hook,
                warmup_hook=transaction.warmup_hook,
                pre_command_hook=transaction.pre_command_hook,
                episode_id_override=transaction.episode_id,
                request_id_override=(
                    f"{transaction.episode_id}-active-front-01"
                ),
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
                "classification": "engineering-integration-smoke",
                "provider_forward_count": len(transaction.provider_records),
                "memory_write_count": transaction.orchestrator.memory_write_count,
                "offline_segmentation_diagnostics": False,
                "runtime_object_gt_reads": 0,
                "goal_gt_reads": 0,
            }
            try:
                transaction_row = transaction.finalize(route_summary)
            finally:
                transaction._sync_execution_progress()
            route_summary["memory_write_count"] = transaction_row["memory_write_count"]
            rows.extend(route_rows)
            summaries.append(route_summary)
            transactions.append(transaction_row)
            provider_records.extend(transaction.provider_records)
    finally:
        if env is not None:
            env.close()
            env_closed = True
        provider.destroy()
    environment_identity = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(torch.device("cuda")),
        "mani_skill": mani_skill.__version__,
        "sapien": sapien.__version__,
        "external_camera_sensor_class": (
            None if sensor is None else type(sensor).__module__ + "." + type(sensor).__name__
        ),
        "external_camera_class": (
            None if camera is None else type(camera).__module__ + "." + type(camera).__name__
        ),
        "external_camera_unmounted": bool(sensor is not None and sensor.entity is None),
        "provider_context_destroyed": provider.destroyed,
        "environment_closed": env_closed,
        "normalizer_identity": normalizer_identity,
    }
    context_destroyed = bool(env_closed and provider.destroyed)
    return (
        rows,
        summaries,
        transactions,
        provider_records,
        environment_identity,
        context_destroyed,
    )


def _publish_stage2a_frozen_artifacts(
    *,
    output_root: Path,
    camera_rows: list[dict[str, Any]],
    route_summaries: list[dict[str, Any]],
    transactions: list[dict[str, Any]],
    provider_records: list[Stage2AProviderOutputRecord],
    context_destroyed: bool,
) -> dict[str, Any]:
    """只有 env/provider 都销毁后才能发布 prediction/decision freeze。"""

    if not context_destroyed:
        raise RuntimeError("Stage 2A context destroy 前禁止发布最终 ledgers/receipt")
    _g0._atomic_jsonl(output_root / "camera_pose_ledger.jsonl", camera_rows)
    _g0._atomic_jsonl(
        output_root / "provider_output_ledger.jsonl",
        [value.to_dict() for value in provider_records],
    )
    _g0._atomic_jsonl(output_root / "route_summaries.jsonl", route_summaries)
    _g0._atomic_jsonl(output_root / "transaction_ledger.jsonl", transactions)
    inventory = {
        name: {
            "raw_sha256": file_sha256(output_root / name),
            "size_bytes": (output_root / name).stat().st_size,
        }
        for name in _STAGE2A_ARTIFACT_FILES_BEFORE_FREEZE
    }
    freeze = {
        "version": "e018-p1-stage2a-execution-freeze/v1",
        "status": "prediction-and-decision-ledgers-frozen-context-destroyed",
        "context_destroyed": True,
        "provider_context_destroyed": True,
        "environment_closed": True,
        "prediction_ledger_frozen": True,
        "decision_ledger_frozen": True,
        "offline_oracle_labels_opened": False,
        "offline_label_reads": 0,
        "runtime_object_gt_reads": 0,
        "goal_gt_reads": 0,
        "fresh_test_reads": 0,
        "checkpoint_writes": 0,
        "artifact_inventory": inventory,
        "frozen_at_unix_ns": time.time_ns(),
    }
    freeze["freeze_sha256"] = canonical_sha256(freeze)
    _g0._atomic_json(output_root / "execution_freeze.json", freeze)
    return freeze


def _verify_stage2a_execution_progress(
    progress: Mapping[str, Any],
) -> dict[str, Any]:
    value = _require_exact_keys(
        dict(progress),
        {
            "current_seed",
            "episode_id",
            "request_id",
            "current_frame_index",
            "last_processed_frame_index",
            "last_authorized_frame_index",
            "controller_state",
            "orchestrator_state",
            "provider_forward_count",
            "memory_write_count",
        },
        "Stage 2A failure progress",
    )
    seed = value["current_seed"]
    if seed is not None and (type(seed) is not int or seed not in STAGE2A_INTEGRATION_SMOKE_SEEDS):
        raise ValueError("Stage 2A failure current seed 非法")
    episode_id = value["episode_id"]
    request_id = value["request_id"]
    if seed is None:
        if any(
            item is not None
            for item in (
                episode_id,
                request_id,
                value["current_frame_index"],
                value["last_processed_frame_index"],
                value["last_authorized_frame_index"],
                value["controller_state"],
                value["orchestrator_state"],
            )
        ):
            raise ValueError("Stage 2A failure pre-seed progress 不得伪造 route 状态")
    elif episode_id != _stage2a_episode_id(seed):
        raise ValueError("Stage 2A failure seed/Episode identity 漂移")
    if request_id is not None and request_id != f"{episode_id}-active-front-01":
        raise ValueError("Stage 2A failure request identity 漂移")

    for name, lower in (
        ("current_frame_index", 0),
        ("last_processed_frame_index", 0),
        ("last_authorized_frame_index", 1),
    ):
        item = value[name]
        if item is not None and (
            type(item) is not int or not lower <= item <= 91
        ):
            raise ValueError(f"Stage 2A failure {name} 非法")
    current = value["current_frame_index"]
    for name in ("last_processed_frame_index", "last_authorized_frame_index"):
        item = value[name]
        if current is not None and item is not None and item > current:
            raise ValueError(f"Stage 2A failure {name} 晚于 current frame")

    controller_state = value["controller_state"]
    orchestrator_state = value["orchestrator_state"]
    if controller_state is not None and controller_state not in {
        item.value for item in ActiveFrontReobserveState
    }:
        raise ValueError("Stage 2A failure controller state 非法")
    if orchestrator_state is not None and orchestrator_state not in {
        item.value for item in PendingActiveViewState
    }:
        raise ValueError("Stage 2A failure orchestrator state 非法")
    if (
        type(value["provider_forward_count"]) is not int
        or not 0 <= value["provider_forward_count"] <= 4
        or type(value["memory_write_count"]) is not int
        or value["memory_write_count"] not in {0, 1}
    ):
        raise ValueError("Stage 2A failure provider/Memory counter 非法")
    return value


def verify_stage2a_failure_evidence(
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """机械校验失败证据的范围、identity 与可恢复进度。"""

    value = _require_exact_keys(
        dict(evidence),
        {
            "version",
            "status",
            "classification",
            "effect_claim",
            "stage2_config_raw_sha256",
            "stage2_config_canonical_sha256",
            "source_identity",
            "error_type",
            "error",
            "progress",
            "traceback",
            "fresh_test_reads",
            "runtime_object_gt_reads",
            "goal_gt_reads",
            "offline_label_reads",
            "failure_sha256",
        },
        "Stage 2A failure evidence",
    )
    primitive = dict(value)
    stored_digest = primitive.pop("failure_sha256")
    source = _require_exact_keys(
        value["source_identity"],
        {"git_commit", "source_tree_sha256", "identity_sha256"},
        "Stage 2A failure source identity",
    )
    source_primitive = dict(source)
    source_digest = source_primitive.pop("identity_sha256")
    trace = _require_exact_keys(
        value["traceback"],
        {
            "encoding",
            "tail",
            "tail_sha256",
            "full_sha256",
            "original_char_count",
            "maximum_tail_chars",
            "truncated",
        },
        "Stage 2A failure traceback",
    )
    progress = _verify_stage2a_execution_progress(value["progress"])
    tail = trace["tail"]
    if (
        stored_digest != canonical_sha256(primitive)
        or value["version"] != E018_P1_STAGE2A_EXECUTION_VERSION
        or value["status"]
        != "failed-engineering-integration-smoke-evidence-preserved"
        or value["classification"] != "engineering-integration-smoke"
        or value["effect_claim"] != "no-effect-claim"
        or not _is_sha256(value["stage2_config_raw_sha256"])
        or not _is_sha256(value["stage2_config_canonical_sha256"])
        or not isinstance(source["git_commit"], str)
        or len(source["git_commit"]) != 40
        or any(item not in "0123456789abcdef" for item in source["git_commit"])
        or not _is_sha256(source["source_tree_sha256"])
        or source_digest != canonical_sha256(source_primitive)
        or not isinstance(value["error_type"], str)
        or not 0 < len(value["error_type"]) <= 128
        or not isinstance(value["error"], str)
        or len(value["error"]) > _STAGE2A_FAILURE_ERROR_MAX_CHARS
        or trace["encoding"] != "utf-8"
        or not isinstance(tail, str)
        or len(tail) > _STAGE2A_FAILURE_TRACEBACK_MAX_CHARS
        or trace["tail_sha256"] != hashlib.sha256(tail.encode("utf-8")).hexdigest()
        or not _is_sha256(trace["full_sha256"])
        or type(trace["original_char_count"]) is not int
        or trace["original_char_count"] < len(tail)
        or trace["maximum_tail_chars"] != _STAGE2A_FAILURE_TRACEBACK_MAX_CHARS
        or trace["truncated"]
        is not (trace["original_char_count"] > len(tail))
        or any(
            value[name] != 0
            for name in (
                "fresh_test_reads",
                "runtime_object_gt_reads",
                "goal_gt_reads",
                "offline_label_reads",
            )
        )
    ):
        raise ValueError("Stage 2A failure evidence identity/scope 漂移")
    return {
        "failure_sha256": stored_digest,
        "error_type": value["error_type"],
        "progress": progress,
        "traceback_full_sha256": trace["full_sha256"],
    }


def _record_stage2a_failure_evidence(
    *,
    output_root: Path,
    error: Exception,
    progress: Stage2AExecutionProgress,
    stage2_config: LoadedStage2AConfig,
    source_identity: Mapping[str, Any],
) -> dict[str, Any]:
    full_traceback = "".join(
        traceback.format_exception(type(error), error, error.__traceback__)
    )
    traceback_tail = full_traceback[-_STAGE2A_FAILURE_TRACEBACK_MAX_CHARS:]
    error_text = str(error)[:_STAGE2A_FAILURE_ERROR_MAX_CHARS]
    evidence = {
        "version": E018_P1_STAGE2A_EXECUTION_VERSION,
        "status": "failed-engineering-integration-smoke-evidence-preserved",
        "classification": "engineering-integration-smoke",
        "effect_claim": "no-effect-claim",
        "stage2_config_raw_sha256": stage2_config.raw_sha256,
        "stage2_config_canonical_sha256": stage2_config.canonical_sha256,
        "source_identity": dict(source_identity),
        "error_type": type(error).__name__,
        "error": error_text,
        "progress": progress.as_dict(),
        "traceback": {
            "encoding": "utf-8",
            "tail": traceback_tail,
            "tail_sha256": hashlib.sha256(traceback_tail.encode("utf-8")).hexdigest(),
            "full_sha256": hashlib.sha256(full_traceback.encode("utf-8")).hexdigest(),
            "original_char_count": len(full_traceback),
            "maximum_tail_chars": _STAGE2A_FAILURE_TRACEBACK_MAX_CHARS,
            "truncated": len(full_traceback) > len(traceback_tail),
        },
        "fresh_test_reads": 0,
        "runtime_object_gt_reads": 0,
        "goal_gt_reads": 0,
        "offline_label_reads": 0,
    }
    evidence["failure_sha256"] = canonical_sha256(evidence)
    verify_stage2a_failure_evidence(evidence)
    _g0._atomic_json(output_root / "FAILURE.json", evidence)
    return evidence


def run_e018_p1_stage2a_integration_smoke(
    *,
    stage2_config_path: str | Path,
    qualification_config_path: str | Path,
    qualification_public_execution_root: str | Path,
    qualification_result_root: str | Path,
    g0c_config_path: str | Path,
    data_config_path: str | Path,
    stats_root: str | Path,
    selected_checkpoint_path: str | Path,
    repository_root: str | Path,
    output_root: str | Path,
    expected_stage2_config_raw_sha256: str,
    expected_stage2_config_canonical_sha256: str,
    expected_source_git_commit: str,
    expected_source_identity_sha256: str,
    integration_smoke_go: str,
) -> dict[str, Any]:
    if integration_smoke_go != STAGE2A_INTEGRATION_SMOKE_GO:
        raise PermissionError("Stage 2A integration smoke 缺少 exact GO token")
    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"Stage 2A output 已存在: {output}")
    loaded = load_e018_p1_stage2a_config(stage2_config_path)
    source = _assert_stage2a_run_authority(
        stage2_config=loaded,
        repository_root=Path(repository_root),
        expected_stage2_config_raw_sha256=expected_stage2_config_raw_sha256,
        expected_stage2_config_canonical_sha256=(
            expected_stage2_config_canonical_sha256
        ),
        expected_source_git_commit=expected_source_git_commit,
        expected_source_identity_sha256=expected_source_identity_sha256,
        integration_smoke_go=integration_smoke_go,
    )
    parent = verify_stage2a_parent_gate(
        stage2_config_path=stage2_config_path,
        qualification_config_path=qualification_config_path,
        qualification_public_execution_root=qualification_public_execution_root,
        qualification_result_root=qualification_result_root,
        g0c_config_path=g0c_config_path,
        data_config_path=data_config_path,
    )
    qualification = load_g2c_dynamic_qualification_config(
        qualification_config_path
    )
    g0c = _g0c.load_e018_p1_g0c_config(g0c_config_path)
    data = load_e018_p1_g2c_data_config(
        data_config_path,
        parent_g0c_config_path=g0c_config_path,
    )
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    _g0._atomic_json(
        output / "RUN_STARTED.json",
        {
            "version": E018_P1_STAGE2A_EXECUTION_VERSION,
            "status": "in-progress-engineering-integration-smoke",
            "classification": "engineering-integration-smoke",
            "effect_claim": "no-effect-claim",
            "wrist_capability": "not-evaluated",
            "seed_range": [76901, 76910],
            "fresh_test_reads": 0,
            "runtime_object_gt_reads": 0,
            "goal_gt_reads": 0,
            "offline_label_reads": 0,
        },
    )
    started = time.monotonic()
    progress = Stage2AExecutionProgress()
    try:
        (
            camera_rows,
            route_summaries,
            transactions,
            provider_records,
            environment_identity,
            context_destroyed,
        ) = _run_stage2a_simulator(
            loaded_stage2_config=loaded,
            qualification_config=qualification,
            g0c_config=g0c,
            data_config=data,
            stats_root=Path(stats_root),
            selected_checkpoint_path=Path(selected_checkpoint_path),
            output_root=output,
            execution_progress=progress,
        )
        wall_seconds = time.monotonic() - started
        freeze = _publish_stage2a_frozen_artifacts(
            output_root=output,
            camera_rows=camera_rows,
            route_summaries=route_summaries,
            transactions=transactions,
            provider_records=provider_records,
            context_destroyed=context_destroyed,
        )
        commit_count = sum(value["memory_write_count"] for value in transactions)
        action_count = sum(
            value["shadow_action_generation"] is not None for value in transactions
        )
        route_pass_count = sum(bool(value["passed"]) for value in route_summaries)
        terminal_count = sum(
            value["orchestrator_state"]
            in {
                PendingActiveViewState.COMMITTED.value,
                PendingActiveViewState.HOME_VERIFIED_FAILED_SAFE_HOLD.value,
            }
            for value in transactions
        )
        rejection_counts: dict[str, int] = {}
        for value in transactions:
            candidate = value["candidate"]
            reasons = (
                value["terminal_reasons"]
                if candidate is None
                else candidate["rejection_reasons"]
            )
            for reason in reasons:
                rejection_counts[str(reason)] = rejection_counts.get(str(reason), 0) + 1
        counts = {
            "seed_count": len(transactions),
            "route_count": len(route_summaries),
            "frame_count": len(camera_rows),
            "provider_forward_count": len(provider_records),
            "home_raw_score_forward_count": sum(
                value.route_frame_index == 0 for value in provider_records
            ),
            "primary_collect_forward_count": sum(
                value.route_frame_index in STAGE2A_COLLECT_FRAME_INDICES
                for value in provider_records
            ),
            "wrist_provider_forward_count": 0,
            "memory_commit_count": commit_count,
            "fresh_shadow_action_generation_count": action_count,
            "route_pass_count": route_pass_count,
            "terminal_transaction_count": terminal_count,
            "runtime_object_gt_reads": 0,
            "goal_gt_reads": 0,
            "offline_label_reads": 0,
            "fresh_test_reads": 0,
            "checkpoint_writes": 0,
            "arm_motion_command_count": 0,
            "gripper_close_command_count": 0,
        }
        plumbing_passed = bool(
            counts["seed_count"] == 10
            and counts["route_count"] == 10
            and counts["frame_count"] == 920
            and counts["provider_forward_count"] == 40
            and counts["home_raw_score_forward_count"] == 10
            and counts["primary_collect_forward_count"] == 30
            and counts["route_pass_count"] == 10
            and counts["terminal_transaction_count"] == 10
            and commit_count >= 1
            and action_count == commit_count
            and wall_seconds
            <= loaded.payload["budgets"]["integration_smoke_gpu_wall_seconds_max"]
        )
        receipt = {
            "version": E018_P1_STAGE2A_EXECUTION_VERSION,
            "status": "complete-engineering-integration-smoke",
            "classification": "engineering-integration-smoke",
            "effect_claim": "no-effect-claim",
            "wrist_capability": "not-evaluated",
            "integration_plumbing_passed": plumbing_passed,
            "success_path_exercised": commit_count >= 1,
            "negative_results_preserved": True,
            "stage2_config_raw_sha256": loaded.raw_sha256,
            "stage2_config_canonical_sha256": loaded.canonical_sha256,
            "source_identity": source,
            "parent_verification": parent,
            "d050_absent_wrist_capability_commit": (
                _D050_ABSENT_WRIST_CAPABILITY_COMMIT
            ),
            "d050_experiment_id": _D050_EXPERIMENT_ID,
            "wrist_capability_status": WRIST_CAPABILITY_ABSENT_STATUS,
            "wrist_capability_record_version": (
                E018_P1_STAGE2A_WRIST_CAPABILITY_VERSION
            ),
            "provider_record_version": E018_P1_STAGE2A_PROVIDER_RECORD_VERSION,
            "seed_range": [76901, 76910],
            "provider_frame_indices": list(STAGE2A_PROVIDER_FRAME_INDICES),
            "primary_collect_frame_indices": list(STAGE2A_COLLECT_FRAME_INDICES),
            "home_barrier_frame_indices": list(STAGE2A_HOME_BARRIER_FRAME_INDICES),
            "counts": counts,
            "candidate_rejection_reason_counts": rejection_counts,
            "gpu_wall_seconds": wall_seconds,
            "environment_identity": environment_identity,
            "execution_freeze_raw_sha256": file_sha256(
                output / "execution_freeze.json"
            ),
            "execution_freeze_internal_sha256": freeze["freeze_sha256"],
            "formal_claim_allowed": False,
            "fresh_test_status": "prohibited-unread",
        }
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        _g0._atomic_json(output / "execution_receipt.json", receipt)
        artifact_bytes = sum(
            path.stat().st_size for path in output.iterdir() if path.is_file()
        )
        if artifact_bytes > loaded.payload["budgets"][
            "integration_smoke_artifact_bytes_max"
        ]:
            raise RuntimeError("Stage 2A integration artifact budget 超限")
        return receipt
    except Exception as error:
        _record_stage2a_failure_evidence(
            output_root=output,
            error=error,
            progress=progress,
            stage2_config=loaded,
            source_identity=source,
        )
        raise


def _read_stage2a_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{name} 不是有效 JSON") from error
    if not isinstance(value, dict):
        raise TypeError(f"{name} 必须是 JSON object")
    return value


def _read_stage2a_jsonl(path: Path, name: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        values = [json.loads(line) for line in lines]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{name} 不是有效 JSONL") from error
    if any(not isinstance(value, dict) for value in values):
        raise RuntimeError(f"{name} 每行必须是 JSON object")
    return values


_STAGE2A_COMPLETE_ARTIFACT_FILES = {
    "RUN_STARTED.json",
    *_STAGE2A_ARTIFACT_FILES_BEFORE_FREEZE,
    "execution_freeze.json",
    "execution_receipt.json",
}

_STAGE2A_CAMERA_ROW_KEYS = {
    "version",
    "episode_id",
    "request_id",
    "camera_command_sequence_id",
    "frame_index",
    "control_tick",
    "timestamp_s",
    "external_rgb_timestamp_s",
    "external_pose_timestamp_s",
    "timestamp_source",
    "external_rgb_pose_skew_s",
    "source_phase",
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
    "actual_external_position_world_m",
    "actual_external_quaternion_sapien",
    "commanded_world_from_external_camera_gl",
    "actual_world_from_external_camera_gl",
    "commanded_base_from_external_camera_cv",
    "actual_base_from_external_camera_cv",
    "external_intrinsic_cv",
    "external_pose_valid",
    "external_position_tracking_error_m",
    "external_orientation_tracking_error_rad",
    "external_linear_velocity_m_s",
    "external_linear_speed_m_s",
    "external_linear_acceleration_m_s2",
    "external_angular_speed_rad_s",
    "external_angular_acceleration_rad_s2",
    "settle_evidence_passed",
    "settle_streak",
    "settled",
    "measurement_write_eligible",
    "memory_write_executed",
    "arm_joint_max_drift_rad",
    "tcp_position_drift_m",
    "tcp_orientation_drift_rad",
    "minimum_finger_joint_position_m",
    "finger_object_contact_force_n",
    "is_grasping",
    "terminated",
    "truncated",
    "rgb_sha256",
    "offline_segmentation_diagnostics",
    "arm_anchor_q_rad",
    "arm_current_q_rad",
    "arm_current_dq_rad_s",
    "tcp_anchor_world",
    "tcp_current_world",
    "world_from_robot_base",
    "finger_joint_positions_m",
    "finger_force_left_n",
    "finger_force_right_n",
    "robot_object_contact_force_n",
    "robot_object_contact_by_link",
}


def _verify_stage2a_exact_file_tree(root: Path) -> None:
    """成功 artifact 必须是扁平 exact tree，拒绝链接、目录和额外 payload。"""

    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("Stage 2A artifact root 必须是真实目录")
    actual: set[str] = set()
    for path in root.iterdir():
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise RuntimeError(f"Stage 2A artifact 禁止 symlink: {relative}")
        stat = path.stat()
        if not path.is_file():
            raise RuntimeError(f"Stage 2A artifact 禁止额外目录/特殊文件: {relative}")
        if stat.st_nlink != 1:
            raise RuntimeError(f"Stage 2A artifact 禁止 hardlink: {relative}")
        actual.add(relative)
    if actual != _STAGE2A_COMPLETE_ARTIFACT_FILES:
        raise RuntimeError(
            "Stage 2A artifact exact file tree 漂移: "
            f"missing={sorted(_STAGE2A_COMPLETE_ARTIFACT_FILES - actual)}, "
            f"extra={sorted(actual - _STAGE2A_COMPLETE_ARTIFACT_FILES)}"
        )


def _verify_stage2a_camera_row_identity(
    value: Mapping[str, Any],
    *,
    episode_id: str,
    request_id: str,
    frame_index: int,
) -> dict[str, Any]:
    """重算单帧路由/坐标身份；不信任 stored HOME/safety bool。"""

    row = _require_exact_keys(
        dict(value), _STAGE2A_CAMERA_ROW_KEYS, "Stage 2A camera row"
    )
    expected_state, expected_viewpoint = _expected_stage2a_motion_identity(
        frame_index
    )
    timestamp = frame_index / 20.0
    numeric_scalars = (
        "orientation_progress",
        "commanded_yaw_offset_rad",
        "commanded_pitch_offset_rad",
        "commanded_roll_offset_rad",
        "external_position_tracking_error_m",
        "external_orientation_tracking_error_rad",
        "external_linear_speed_m_s",
        "external_linear_acceleration_m_s2",
        "external_angular_speed_rad_s",
        "external_angular_acceleration_rad_s2",
    )
    if any(
        not isinstance(row[name], (int, float))
        or isinstance(row[name], bool)
        or not math.isfinite(float(row[name]))
        for name in numeric_scalars
    ):
        raise ValueError("Stage 2A camera row 含非法数值")
    if (
        row["version"] != E018_P1_STAGE2A_EXECUTION_VERSION
        or row["episode_id"] != episode_id
        or row["request_id"] != request_id
        or row["camera_command_sequence_id"]
        != f"{request_id}-camera-command-00"
        or row["frame_index"] != frame_index
        or row["control_tick"] != frame_index
        or not math.isclose(
            float(row["timestamp_s"]), timestamp, rel_tol=0.0, abs_tol=1e-12
        )
        or not math.isclose(
            float(row["external_rgb_timestamp_s"]),
            timestamp,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(row["external_pose_timestamp_s"]),
            timestamp,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or row["timestamp_source"]
        != "synchronous-simulator-control-tick-derived/v1"
        or row["external_rgb_pose_skew_s"] != 0.0
        or row["source_phase"] != STAGE2A_SOURCE_PHASE.value
        or row["camera_motion_state"] != expected_state.value
        or row["viewpoint_primitive_id"] != expected_viewpoint
        or not 0.0 <= float(row["orientation_progress"]) <= 1.0
        or row["arm_owner"] != "SAFE_HOLD"
        or row["gripper_owner"] != "SAFE_HOLD_OPEN"
        or row["external_camera_owner"] != STAGE2A_CAMERA_OWNER
        or row["arm_motion_command_max_abs"] != 0.0
        or row["gripper_hold_open_command"] != 1.0
        or row["external_pose_valid"] is not True
        or row["measurement_write_eligible"]
        is not _g0.measurement_write_eligible(
            expected_state, settled=bool(row["settled"])
        )
        or row["memory_write_executed"] is not False
        or row["is_grasping"] is not None
        or row["offline_segmentation_diagnostics"] is not None
        or row["robot_object_contact_force_n"] is not None
        or row["robot_object_contact_by_link"] is not None
        or row["terminated"] is not False
        or row["truncated"] is not False
        or not _is_sha256(row["rgb_sha256"])
    ):
        raise ValueError("Stage 2A camera row identity/permission 漂移")
    vectors = {
        "commanded_external_position_world_m": 3,
        "commanded_external_quaternion_sapien": 4,
        "actual_external_position_world_m": 3,
        "actual_external_quaternion_sapien": 4,
        "external_linear_velocity_m_s": 3,
        "arm_anchor_q_rad": 7,
        "arm_current_q_rad": 7,
        "arm_current_dq_rad_s": 7,
        "finger_joint_positions_m": 2,
    }
    parsed_vectors: dict[str, np.ndarray] = {}
    for name, size in vectors.items():
        array = np.asarray(row[name], dtype=np.float64)
        if array.shape != (size,) or not np.isfinite(array).all():
            raise ValueError(f"Stage 2A camera row {name} shape/value 漂移")
        parsed_vectors[name] = array
    for name in (
        "commanded_external_quaternion_sapien",
        "actual_external_quaternion_sapien",
    ):
        if not math.isclose(
            float(np.linalg.norm(parsed_vectors[name])),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError(f"Stage 2A camera row {name} 不是单位四元数")
    velocity_limits = np.asarray(
        RobotSpec().joint_velocity_limits_rad_s, dtype=np.float64
    )
    if np.any(
        np.abs(parsed_vectors["arm_current_dq_rad_s"])
        > velocity_limits + 1e-5
    ):
        raise ValueError("Stage 2A camera row arm_current_dq_rad_s 超出 Franka 限制")
    matrices = {
        name: validate_se3(np.asarray(row[name], dtype=np.float64), name)
        for name in (
            "commanded_world_from_external_camera_gl",
            "actual_world_from_external_camera_gl",
            "commanded_base_from_external_camera_cv",
            "actual_base_from_external_camera_cv",
            "tcp_anchor_world",
            "tcp_current_world",
            "world_from_robot_base",
        )
    }
    intrinsic = np.asarray(row["external_intrinsic_cv"], dtype=np.float64)
    base_from_world = invert_se3(
        matrices["world_from_robot_base"], "world_from_robot_base"
    )
    expected_commanded_base = validate_se3(
        base_from_world
        @ opengl_camera_to_opencv(
            matrices["commanded_world_from_external_camera_gl"]
        ),
        "expected_commanded_base_from_external_camera_cv",
    )
    expected_actual_base = validate_se3(
        base_from_world
        @ opengl_camera_to_opencv(
            matrices["actual_world_from_external_camera_gl"]
        ),
        "expected_actual_base_from_external_camera_cv",
    )
    if (
        intrinsic.shape != (3, 3)
        or not np.isfinite(intrinsic).all()
        or not np.allclose(
            parsed_vectors["commanded_external_position_world_m"],
            matrices["commanded_world_from_external_camera_gl"][:3, 3],
            rtol=0.0,
            atol=1e-9,
        )
        or not np.allclose(
            parsed_vectors["actual_external_position_world_m"],
            matrices["actual_world_from_external_camera_gl"][:3, 3],
            rtol=0.0,
            atol=1e-9,
        )
        or not np.allclose(
            matrices["commanded_base_from_external_camera_cv"],
            expected_commanded_base,
            rtol=0.0,
            atol=1e-9,
        )
        or not np.allclose(
            matrices["actual_base_from_external_camera_cv"],
            expected_actual_base,
            rtol=0.0,
            atol=1e-9,
        )
        or (frame_index == 87 and not _stage2a_pose_at_home(row))
        or (
            frame_index in STAGE2A_HOME_BARRIER_FRAME_INDICES
            and not _stage2a_camera_at_home(row)
        )
    ):
        raise ValueError("Stage 2A camera row pose/geometry 不能重算")
    return row


def _provider_record_from_dict(value: Mapping[str, Any]) -> Stage2AProviderOutputRecord:
    expected = {
        "version",
        "episode_id",
        "episode_generation",
        "request_id",
        "observation_sequence_id",
        "route_frame_index",
        "record_role",
        "model_input_digest",
        "provider_identity",
        "provider_identity_sha256",
        "stage2_config_raw_sha256",
        "stage2_config_canonical_sha256",
        "qualification_config_raw_sha256",
        "qualification_config_internal_sha256",
        "prediction",
        "provider_output_digest",
    }
    row = _require_exact_keys(dict(value), expected, "Stage 2A provider ledger row")
    identity_value = row["provider_identity"]
    if not isinstance(identity_value, dict):
        raise TypeError("provider ledger full identity 缺失")
    identity = ActiveFrontStage2ProviderIdentity(**identity_value)
    if row["provider_identity_sha256"] != identity.sha256:
        raise ValueError("provider ledger identity digest 漂移")
    record = Stage2AProviderOutputRecord(
        episode_id=row["episode_id"],
        episode_generation=row["episode_generation"],
        request_id=row["request_id"],
        observation_sequence_id=row["observation_sequence_id"],
        route_frame_index=row["route_frame_index"],
        record_role=row["record_role"],
        model_input_digest=row["model_input_digest"],
        provider_identity=identity,
        stage2_config_raw_sha256=row["stage2_config_raw_sha256"],
        stage2_config_canonical_sha256=row["stage2_config_canonical_sha256"],
        qualification_config_raw_sha256=row["qualification_config_raw_sha256"],
        qualification_config_internal_sha256=row[
            "qualification_config_internal_sha256"
        ],
        prediction_canonical_json=_canonical_json(row["prediction"]),
        version=row["version"],
    )
    if row["provider_output_digest"] != record.provider_output_digest:
        raise ValueError("provider ledger canonical output digest 漂移")
    return record


def _wrist_capability_from_dict(
    value: Mapping[str, Any],
) -> WristCapabilityEvidenceRecord:
    expected = {
        "episode_id",
        "episode_generation",
        "request_id",
        "record_role",
        "source_phase",
        "observation_sequence_id",
        "timestamp_s",
        "home_observation_payload_digest",
        "home_observation_payload",
        "memory_state",
        "memory_mode",
        "memory_resolution_available",
        "memory_unavailable_reasons",
        "memory_state_revision",
        "memory_resolution_policy_sha256",
        "reason",
        "status",
        "provider_identity",
        "inference_attempt_count",
        "frame_evaluated",
        "measurement_usable",
        "state_authorized",
        "supersede_authorized",
        "contact_authorized",
        "version",
        "evidence_identity_sha256",
    }
    row = _require_exact_keys(dict(value), expected, "wrist capability record")
    record = WristCapabilityEvidenceRecord(
        episode_id=row["episode_id"],
        episode_generation=row["episode_generation"],
        request_id=row["request_id"],
        record_role=row["record_role"],
        source_phase=PhaseId(row["source_phase"]),
        observation_sequence_id=row["observation_sequence_id"],
        timestamp_s=row["timestamp_s"],
        home_observation_payload_digest=row["home_observation_payload_digest"],
        home_observation_payload_canonical_json=_canonical_json(
            row["home_observation_payload"]
        ),
        memory_state_canonical_json=_canonical_json(row["memory_state"]),
        memory_mode=row["memory_mode"],
        memory_resolution_available=row["memory_resolution_available"],
        memory_unavailable_reasons=tuple(row["memory_unavailable_reasons"]),
        memory_state_revision=row["memory_state_revision"],
        memory_resolution_policy_sha256=row[
            "memory_resolution_policy_sha256"
        ],
        reason=row["reason"],
        status=row["status"],
        provider_identity=row["provider_identity"],
        inference_attempt_count=row["inference_attempt_count"],
        frame_evaluated=row["frame_evaluated"],
        measurement_usable=row["measurement_usable"],
        state_authorized=row["state_authorized"],
        supersede_authorized=row["supersede_authorized"],
        contact_authorized=row["contact_authorized"],
        version=row["version"],
    )
    if row["evidence_identity_sha256"] != record.digest:
        raise ValueError("wrist capability record digest 漂移")
    return record


def _stage2_candidate_receipt_from_dict(
    value: Mapping[str, Any],
) -> Stage2MemoryCandidateReceipt:
    row = _require_exact_keys(
        dict(value),
        {
            "request_id",
            "candidate_digest",
            "commit_eligible",
            "rejection_reasons",
            "memory_write_deferred",
            "live_memory_write_executed",
            "provider_forward_count",
            "collect_frame_digests",
            "receipt_sha256",
        },
        "Stage 2 candidate stage receipt",
    )
    primitive = dict(row)
    stored_digest = primitive.pop("receipt_sha256")
    if stored_digest != canonical_sha256(primitive):
        raise ValueError("Stage 2 candidate stage receipt digest 漂移")
    return Stage2MemoryCandidateReceipt(
        request_id=row["request_id"],
        candidate_digest=row["candidate_digest"],
        commit_eligible=row["commit_eligible"],
        rejection_reasons=tuple(row["rejection_reasons"]),
        memory_write_deferred=row["memory_write_deferred"],
        live_memory_write_executed=row["live_memory_write_executed"],
        provider_forward_count=row["provider_forward_count"],
        collect_frame_digests=tuple(row["collect_frame_digests"]),
    )


def _expected_stage2a_motion_identity(
    frame_index: int,
) -> tuple[ExternalCameraMotionState, str]:
    if frame_index == 0:
        return ExternalCameraMotionState.HOME_ANCHOR, ACTIVE_FRONT_HOME_PRIMITIVE_ID
    if 1 <= frame_index <= 40:
        return (
            ExternalCameraMotionState.MOVE_TO_VIEW,
            ACTIVE_FRONT_PRIMARY_PRIMITIVE_ID,
        )
    if 41 <= frame_index <= 44:
        return (
            ExternalCameraMotionState.SETTLE_AT_VIEW,
            ACTIVE_FRONT_PRIMARY_PRIMITIVE_ID,
        )
    if 45 <= frame_index <= 47:
        return (
            ExternalCameraMotionState.COLLECT,
            ACTIVE_FRONT_PRIMARY_PRIMITIVE_ID,
        )
    if 48 <= frame_index <= 87:
        return ExternalCameraMotionState.RETURN_HOME, ACTIVE_FRONT_HOME_PRIMITIVE_ID
    if 88 <= frame_index <= 91:
        return (
            ExternalCameraMotionState.VERIFY_HOME_AND_ARM_HOLD,
            ACTIVE_FRONT_HOME_PRIMITIVE_ID,
        )
    raise ValueError(f"Stage 2A frame index 非法: {frame_index}")


def _verify_stage2a_safety_record(
    row: Mapping[str, Any],
    value: Mapping[str, Any],
    *,
    controller: ActiveFrontReobserveController,
) -> ActiveFrontSafetyEvidence:
    record = _require_exact_keys(
        dict(value),
        {
            "version",
            "frame_index",
            "motion_row_sha256",
            "raw",
            "controller_active_window_open",
            "derived",
            "evidence_sha256",
        },
        "Stage 2A safety evidence",
    )
    primitive = dict(record)
    stored_digest = primitive.pop("evidence_sha256")
    raw = _stage2a_safety_scalars(row)
    arm_anchor = np.asarray(row.get("arm_anchor_q_rad"), dtype=np.float64)
    arm_current = np.asarray(row.get("arm_current_q_rad"), dtype=np.float64)
    tcp_anchor = np.asarray(row.get("tcp_anchor_world"), dtype=np.float64)
    tcp_current = np.asarray(row.get("tcp_current_world"), dtype=np.float64)
    finger_positions = np.asarray(
        row.get("finger_joint_positions_m"), dtype=np.float64
    )
    finger_force_left = row.get("finger_force_left_n")
    finger_force_right = row.get("finger_force_right_n")
    if (
        arm_anchor.shape != (7,)
        or arm_current.shape != (7,)
        or not np.isfinite(arm_anchor).all()
        or not np.isfinite(arm_current).all()
        or finger_positions.shape != (2,)
        or not np.isfinite(finger_positions).all()
        or not isinstance(finger_force_left, (int, float))
        or isinstance(finger_force_left, bool)
        or not isinstance(finger_force_right, (int, float))
        or isinstance(finger_force_right, bool)
        or not math.isfinite(float(finger_force_left))
        or not math.isfinite(float(finger_force_right))
        or float(finger_force_left) < 0.0
        or float(finger_force_right) < 0.0
    ):
        raise ValueError("Stage 2A raw joint/finger safety witness 非法")
    tcp_anchor = validate_se3(tcp_anchor, "Stage 2A tcp_anchor_world")
    tcp_current = validate_se3(tcp_current, "Stage 2A tcp_current_world")
    recomputed_raw = {
        "arm_joint_max_drift_rad": float(
            np.max(np.abs(arm_current - arm_anchor))
        ),
        "tcp_position_drift_m": float(
            np.linalg.norm(tcp_current[:3, 3] - tcp_anchor[:3, 3])
        ),
        "tcp_orientation_drift_rad": _rotation_distance_rad(
            tcp_anchor[:3, :3], tcp_current[:3, :3]
        ),
        "minimum_finger_joint_position_m": float(np.min(finger_positions)),
        "finger_object_contact_force_n": max(
            float(finger_force_left), float(finger_force_right)
        ),
    }
    expected = _stage2a_active_safety(row, controller=controller)
    if (
        stored_digest != canonical_sha256(primitive)
        or record["version"] != "e018-p1-stage2a-derived-safety-evidence/v1"
        or record["frame_index"] != row["frame_index"]
        or record["motion_row_sha256"] != canonical_sha256(dict(row))
        or record["raw"] != raw
        or any(
            not math.isclose(raw[name], recomputed_raw[name], rel_tol=0.0, abs_tol=1e-12)
            for name in raw
        )
        or record["controller_active_window_open"]
        is not controller.active_window_open
        or record["derived"] != asdict(expected)
    ):
        raise ValueError("Stage 2A safety evidence 不能由 raw witness 重算")
    return expected


def _verify_stage2a_camera_authorization(
    row: Mapping[str, Any],
    value: Mapping[str, Any],
    *,
    controller: ActiveFrontReobserveController,
) -> None:
    record = _require_exact_keys(
        dict(value),
        {
            "version",
            "frame_index",
            "camera_motion_state",
            "viewpoint_primitive_id",
            "controller_state_before_command",
            "external_camera_owner",
            "camera_lease_held",
            "active_window_open",
            "request_id",
            "camera_command_sequence_id",
            "selected_primitive_id",
            "authorized",
            "authorization_sha256",
        },
        "Stage 2A camera command authorization",
    )
    primitive = dict(record)
    stored_digest = primitive.pop("authorization_sha256")
    frame_index = int(row["frame_index"])
    expected_state, expected_viewpoint = _expected_stage2a_motion_identity(frame_index)
    request = controller.request
    if (
        frame_index == 0
        or request is None
        or stored_digest != canonical_sha256(primitive)
        or record["version"]
        != "e018-p1-stage2a-camera-command-authorization/v1"
        or record["frame_index"] != frame_index
        or record["camera_motion_state"] != expected_state.value
        or record["camera_motion_state"] != row["camera_motion_state"]
        or record["viewpoint_primitive_id"] != expected_viewpoint
        or record["viewpoint_primitive_id"] != row["viewpoint_primitive_id"]
        or record["controller_state_before_command"] != controller.state.value
        or record["external_camera_owner"]
        != ExternalCameraControllerOwner.ACTIVE_REOBSERVE.value
        or controller.external_camera_owner
        is not ExternalCameraControllerOwner.ACTIVE_REOBSERVE
        or record["camera_lease_held"] is not True
        or record["active_window_open"] is not True
        or not controller.active_window_open
        or record["request_id"] != request.request_id
        or record["request_id"] != row["request_id"]
        or record["camera_command_sequence_id"]
        != request.camera_command_sequence_id
        or record["camera_command_sequence_id"]
        != row["camera_command_sequence_id"]
        or record["selected_primitive_id"] != request.selected_primitive_id
        or record["authorized"] is not True
    ):
        raise ValueError("Stage 2A pose mutation 缺少可重放 controller authorization")


def _verify_stage2a_home_barrier_evidence(
    row: Mapping[str, Any],
    value: Mapping[str, Any],
    *,
    episode_id: str,
    request_id: str,
    safety_evidence_sha256: str,
) -> tuple[HomeV2BarrierFrame, str]:
    evidence = _require_exact_keys(
        dict(value),
        {
            "version",
            "episode_id",
            "episode_generation",
            "request_id",
            "observation_sequence_id",
            "route_frame_index",
            "camera_motion_state",
            "viewpoint_primitive_id",
            "timestamp_source",
            "control_timestamp_s",
            "rgb_timestamp_s",
            "camera_pose_timestamp_s",
            "actual_pose_source",
            "actual_base_from_external_camera_cv",
            "actual_pose_sha256",
            "camera_at_home",
            "home_observation_payload",
            "home_observation_payload_digest",
            "motion_row_sha256",
            "safety_evidence_sha256",
            "evidence_sha256",
        },
        "Stage 2A HOME barrier evidence",
    )
    primitive = dict(evidence)
    stored_digest = primitive.pop("evidence_sha256")
    payload = evidence["home_observation_payload"]
    _verify_home_observation_payload_identity(payload)
    actual_pose = np.asarray(
        evidence["actual_base_from_external_camera_cv"], dtype=np.float64
    )
    row_pose = np.asarray(row["actual_base_from_external_camera_cv"], dtype=np.float64)
    front_pose = np.asarray(
        payload["cameras"]["front"]["cam2world_gl"], dtype=np.float64
    )
    row_world_pose = np.asarray(
        row["actual_world_from_external_camera_gl"], dtype=np.float64
    )
    control_timestamp = float(row["timestamp_s"]) + 0.10
    frame = HomeV2BarrierFrame(
        observation_sequence_id=evidence["observation_sequence_id"],
        camera_at_home=_stage2a_camera_at_home(row),
        fresh_observation_v2_frame=True,
        captured_after_return=True,
        contains_alternate_or_motion_rgb=False,
    )
    if (
        stored_digest != canonical_sha256(primitive)
        or evidence["version"]
        != "e018-p1-stage2a-home-barrier-evidence/v1"
        or evidence["episode_id"] != episode_id
        or evidence["episode_generation"] != 1
        or evidence["request_id"] != request_id
        or evidence["route_frame_index"] != row["frame_index"]
        or evidence["observation_sequence_id"]
        != f"{episode_id}-home-v2-{int(row['frame_index']):02d}"
        or evidence["camera_motion_state"]
        != ExternalCameraMotionState.VERIFY_HOME_AND_ARM_HOLD.value
        or evidence["camera_motion_state"] != row["camera_motion_state"]
        or evidence["viewpoint_primitive_id"] != ACTIVE_FRONT_HOME_PRIMITIVE_ID
        or evidence["viewpoint_primitive_id"] != row["viewpoint_primitive_id"]
        or evidence["timestamp_source"] != row["timestamp_source"]
        or not math.isclose(
            float(evidence["control_timestamp_s"]),
            control_timestamp,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(evidence["rgb_timestamp_s"]),
            float(row["external_rgb_timestamp_s"]) + 0.10,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(evidence["camera_pose_timestamp_s"]),
            float(row["external_pose_timestamp_s"]) + 0.10,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or evidence["actual_pose_source"]
        != "same-observation.sensor_param.base_camera.cam2world_gl/v1"
        or actual_pose.shape != (4, 4)
        or not np.array_equal(actual_pose, row_pose)
        or evidence["actual_pose_sha256"] != _array_sha256(actual_pose)
        or evidence["camera_at_home"] is not True
        or not frame.valid()
        or payload["timestamp_s"] != evidence["control_timestamp_s"]
        or payload["cameras"]["front"]["rgb_bytes_sha256"]
        != row["rgb_sha256"]
        or front_pose.shape != (4, 4)
        or not np.array_equal(front_pose, row_world_pose)
        or evidence["home_observation_payload_digest"]
        != canonical_sha256(payload)
        or evidence["motion_row_sha256"] != canonical_sha256(dict(row))
        or evidence["safety_evidence_sha256"] != safety_evidence_sha256
    ):
        raise ValueError("Stage 2A HOME barrier raw row/payload identity 漂移")
    home_frame_digest = canonical_sha256(
        {"frame": asdict(frame), "timestamp_s": control_timestamp}
    )
    return frame, home_frame_digest


def _replay_stage2a_controller_event(
    controller: ActiveFrontReobserveController,
    value: Mapping[str, Any],
    *,
    signal: ActiveFrontSignal,
    frame_index: int,
    safety: ActiveFrontSafetyEvidence,
    selected_primitive_id: str | None = None,
    candidate_receipt: Stage2MemoryCandidateReceipt | None = None,
    source_phase: PhaseId | None = None,
    source_invariants_passed: bool | None = None,
) -> None:
    event = _require_exact_keys(
        dict(value),
        {
            "version",
            "frame_index",
            "signal",
            "state_before",
            "state_after",
            "selected_primitive_id",
            "candidate_receipt_sha256",
            "source_phase",
            "source_invariants_passed",
            "safety",
            "event_sha256",
        },
        "Stage 2A controller event",
    )
    before = controller.state
    controller.advance(
        signal,
        safety=safety,
        selected_primitive_id=selected_primitive_id,
        shadow_candidate_receipt=candidate_receipt,
        source_phase=source_phase,
        source_invariants_passed=source_invariants_passed,
    )
    expected = {
        "version": "e018-p1-stage2a-controller-event/v1",
        "frame_index": frame_index,
        "signal": signal.value,
        "state_before": before.value,
        "state_after": controller.state.value,
        "selected_primitive_id": selected_primitive_id,
        "candidate_receipt_sha256": (
            None
            if candidate_receipt is None
            else canonical_sha256(asdict(candidate_receipt))
        ),
        "source_phase": None if source_phase is None else source_phase.value,
        "source_invariants_passed": source_invariants_passed,
        "safety": asdict(safety),
    }
    expected["event_sha256"] = canonical_sha256(expected)
    if event != expected:
        raise ValueError(f"Stage 2A controller event replay 漂移: {signal.value}")


def _controller_receipt_payload_from_public(value: Mapping[str, Any]) -> dict[str, Any]:
    return _require_exact_keys(
        dict(value),
        {
            "version",
            "episode_id",
            "request_id",
            "status",
            "source_phase",
            "resume_phase",
            "selected_primitive_id",
            "state_trace",
            "home_observation_sequence_ids",
            "action_history_generation_before",
            "action_history_generation_after_reset",
            "resumed_action_history_generation",
            "memory_read_count",
            "memory_write_count",
            "test_read_count",
            "provider_forward_count",
            "failure",
            "audit_digest",
        },
        "Stage 2A controller receipt",
    )


def _new_stage2a_replay_controller(
    stage2_config: LoadedStage2AConfig,
    *,
    episode_id: str,
) -> ActiveFrontReobserveController:
    execution = stage2_config.payload["execution"]
    controller = ActiveFrontReobserveController(
        ActiveFrontReobserveConfig(
            enabled=True,
            selected_primitive_id=ACTIVE_FRONT_PRIMARY_PRIMITIVE_ID,
            consecutive_unusable_ticks=execution[
                "capability_absent_consecutive_trigger_ticks"
            ],
            maximum_attempts_per_episode=1,
            home_v2_barrier_frames=4,
            allow_capability_absent_trigger=execution[
                "allow_capability_absent_trigger"
            ],
        )
    )
    controller.reset_episode(episode_id, episode_generation=1)
    return controller


def _verify_stage2a_trigger_replay(
    transaction: Mapping[str, Any],
    *,
    controller: ActiveFrontReobserveController,
    episode_id: str,
) -> tuple[list[WristCapabilityEvidenceRecord], WristCapabilityEvidenceRecord]:
    records_value = transaction["trigger_wrist_capability_records"]
    decisions_value = transaction["trigger_decisions"]
    request_id = transaction["request_id"]
    if (
        not isinstance(records_value, list)
        or len(records_value) != 3
        or not isinstance(decisions_value, list)
        or len(decisions_value) != 3
        or transaction["capability_absence_trigger_reason"]
        != ActiveFrontTriggerReason.NO_QUALIFIED_WRIST_PROVIDER_IN_PARENT.value
    ):
        raise ValueError("Stage 2A 三 Tick capability trigger ledger 不完整")
    records: list[WristCapabilityEvidenceRecord] = []
    for tick, (record_value, public_decision) in enumerate(
        zip(records_value, decisions_value, strict=True)
    ):
        record = _wrist_capability_from_dict(record_value)
        expected_timestamp = tick * 0.05
        if (
            record.episode_id != episode_id
            or record.episode_generation != 1
            or record.request_id != request_id
            or record.record_role != "trigger"
            or record.observation_sequence_id
            != f"{episode_id}-trigger-home-{tick:02d}"
            or not math.isclose(
                record.timestamp_s,
                expected_timestamp,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or record.provider_identity is not None
            or record.inference_attempt_count != 0
            or record.frame_evaluated
            or record.measurement_usable
            or record.reason
            in {
                ActiveFrontTriggerReason.LOW_VISUAL_CONFIDENCE.value,
                ActiveFrontTriggerReason.HIGH_LOCALIZATION_UNCERTAINTY.value,
            }
        ):
            raise ValueError("Stage 2A capability absence 被冒充为视觉失败")
        evidence = build_trigger_evidence_from_capability(
            record,
            control_tick=tick,
            arm_hold_prerequisites_pass=True,
            camera_home_prerequisites_pass=True,
        )
        decision = controller.consider_trigger(evidence)
        expected_decision = {
            "version": "e018-p1-stage2a-capability-trigger-decision/v1",
            "control_tick": tick,
            "timestamp_s": expected_timestamp,
            "capability_evidence_identity_sha256": record.digest,
            "requestable": decision.requestable,
            "reason": decision.reason.value,
            "consecutive_unusable_ticks": decision.consecutive_unusable_ticks,
        }
        expected_decision["decision_sha256"] = canonical_sha256(
            expected_decision
        )
        if public_decision != expected_decision:
            raise ValueError("Stage 2A capability trigger decision 不能重放")
        records.append(record)
    request = controller.request
    if request is None or request.request_id != request_id:
        raise ValueError("Stage 2A 第三 Tick 未产生预期 request")
    if [value["requestable"] for value in decisions_value] != [False, False, True]:
        raise ValueError("Stage 2A capability trigger 必须恰好三 Tick")
    if transaction["trigger_wrist_capability"] != records[-1].to_dict():
        raise ValueError("Stage 2A trigger alias 不是第三条完整 record")
    if len({value.digest for value in records}) != 3:
        raise ValueError("Stage 2A 三条 trigger capability identity 必须互异")
    source_recheck = _wrist_capability_from_dict(
        transaction["source_recheck_wrist_capability"]
    )
    return records, source_recheck


def _verify_stage2a_provider_transaction_binding(
    transaction: Mapping[str, Any],
    *,
    records: Sequence[Stage2AProviderOutputRecord],
    rows: Sequence[Mapping[str, Any]],
    episode_id: str,
    request_id: str,
) -> tuple[ActiveFrontStage2FrameEvidence, ...]:
    if (
        len(records) != 4
        or tuple(record.route_frame_index for record in records)
        != STAGE2A_PROVIDER_FRAME_INDICES
        or transaction["provider_output_digests"]
        != [record.provider_output_digest for record in records]
        or transaction["provider_frame_indices"]
        != list(STAGE2A_PROVIDER_FRAME_INDICES)
    ):
        raise ValueError("Stage 2A provider ledger/transaction order 漂移")
    primary: list[ActiveFrontStage2FrameEvidence] = []
    for record in records:
        frame_index = record.route_frame_index
        row = rows[frame_index]
        prediction_pose = np.asarray(
            record.prediction["base_from_external_camera_cv"], dtype=np.float64
        )
        actual_pose = np.asarray(
            row["actual_base_from_external_camera_cv"], dtype=np.float64
        )
        if (
            record.episode_id != episode_id
            or record.episode_generation != 1
            or record.request_id != request_id
            or record.observation_sequence_id
            != f"{episode_id}-route-frame-{frame_index:02d}"
            or record.prediction.get("seed") != transaction["seed"]
            or prediction_pose.shape != (4, 4)
            or not np.allclose(
                prediction_pose, actual_pose, rtol=0.0, atol=1e-9
            )
        ):
            raise ValueError("Stage 2A provider output 未绑定实际 route frame")
        if frame_index == 0:
            build_stage2a_home_score_evidence(
                record,
                motion_row=row,
                timestamp_offset_s=Stage2ARouteTransaction._TIMESTAMP_OFFSET_S,
            )
        else:
            primary.append(
                build_stage2a_primary_frame_evidence(
                    record,
                    motion_row=row,
                    timestamp_offset_s=Stage2ARouteTransaction._TIMESTAMP_OFFSET_S,
                )
            )
    expected_digests = [value.frame_digest for value in primary]
    if transaction["primary_frame_digests"] != expected_digests:
        raise ValueError("Stage 2A PRIMARY frame digest 不能从 provider/route 重算")
    return tuple(primary)


def _verify_stage2a_candidate_binding(
    transaction: Mapping[str, Any],
    *,
    primary_frames: Sequence[ActiveFrontStage2FrameEvidence],
) -> Stage2MemoryCandidateReceipt:
    receipt = _stage2_candidate_receipt_from_dict(
        transaction["candidate_stage_receipt"]
    )
    frame_digests = tuple(value.frame_digest for value in primary_frames)
    candidate = transaction["candidate"]
    candidate_digest = transaction["candidate_digest"]
    if (
        receipt.request_id != transaction["request_id"]
        or receipt.collect_frame_digests != frame_digests
        or receipt.commit_eligible is not transaction["candidate_commit_eligible"]
        or receipt.provider_forward_count != 3
        or receipt.live_memory_write_executed
    ):
        raise ValueError("Stage 2A staged candidate receipt 不能重算")
    if candidate is None:
        expected_rejected_digest = canonical_sha256(
            {
                "version": "e018-p1-stage2a-rejected-candidate-outcome/v1",
                "request_id": transaction["request_id"],
                "collect_frame_digests": list(frame_digests),
                "rejection_reasons": list(receipt.rejection_reasons),
            }
        )
        if (
            candidate_digest is not None
            or receipt.commit_eligible
            or receipt.candidate_digest != expected_rejected_digest
        ):
            raise ValueError("Stage 2A rejected candidate identity 漂移")
        return receipt
    if not isinstance(candidate, dict):
        raise TypeError("Stage 2A candidate 必须是 object 或 None")
    candidate = _require_exact_keys(
        candidate,
        {
            "version",
            "episode_id",
            "episode_generation",
            "request_id",
            "window_id",
            "source_phase",
            "resume_phase",
            "frame_sequence_ids",
            "frame_timestamps_s",
            "frame_digests",
            "model_input_digests",
            "provider_output_digests",
            "actual_pose_sha256s",
            "write_scores",
            "score_components",
            "minimum_candidate_score",
            "information_gain",
            "position_spread_m",
            "innovation_m",
            "final_measurement",
            "final_measurement_digest",
            "provider_identity",
            "provider_identity_sha256",
            "baseline_digest",
            "created_timestamp_s",
            "maximum_age_s",
            "commit_eligible",
            "rejection_reasons",
        },
        "Stage 2A pending candidate",
    )
    if (
        canonical_sha256(candidate) != candidate_digest
        or candidate_digest != receipt.candidate_digest
        or candidate["episode_id"] != transaction["episode_id"]
        or candidate["episode_generation"] != 1
        or candidate["request_id"] != transaction["request_id"]
        or candidate["source_phase"] != STAGE2A_SOURCE_PHASE.value
        or candidate["resume_phase"] != STAGE2A_SOURCE_PHASE.value
        or candidate["frame_sequence_ids"]
        != [value.observation_sequence_id for value in primary_frames]
        or candidate["frame_timestamps_s"]
        != [value.control_timestamp_s for value in primary_frames]
        or candidate["frame_digests"] != list(frame_digests)
        or candidate["model_input_digests"]
        != [value.model_input_digest for value in primary_frames]
        or candidate["provider_output_digests"]
        != [value.provider_output_digest for value in primary_frames]
        or candidate["actual_pose_sha256s"]
        != [value.actual_pose_sha256 for value in primary_frames]
        or candidate["write_scores"]
        != [value.write_score for value in primary_frames]
        or candidate["commit_eligible"] is not receipt.commit_eligible
        or candidate["rejection_reasons"] != list(receipt.rejection_reasons)
    ):
        raise ValueError("Stage 2A pending candidate 不能从三帧 PRIMARY 重算")
    return receipt


def _verify_stage2a_source_recheck_identity(
    source_recheck: WristCapabilityEvidenceRecord,
    *,
    trigger_records: Sequence[WristCapabilityEvidenceRecord],
    final_home_evidence: Mapping[str, Any],
    episode_id: str,
    request_id: str,
) -> None:
    expected_payload = json.loads(
        _canonical_json(final_home_evidence["home_observation_payload"])
    )
    expected_timestamp = float(final_home_evidence["control_timestamp_s"]) + 0.001
    expected_payload["timestamp_s"] = expected_timestamp
    if (
        source_recheck.episode_id != episode_id
        or source_recheck.episode_generation != 1
        or source_recheck.request_id != request_id
        or source_recheck.record_role != "source_recheck"
        or source_recheck.observation_sequence_id
        != f"{episode_id}-source-recheck-home"
        or not math.isclose(
            source_recheck.timestamp_s,
            expected_timestamp,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or json.loads(source_recheck.home_observation_payload_canonical_json)
        != expected_payload
        or source_recheck.memory_state_revision
        != trigger_records[-1].memory_state_revision
        or source_recheck.digest in {value.digest for value in trigger_records}
    ):
        raise ValueError("Stage 2A source recheck 不是独立 fresh HOME record")


def _verify_stage2a_controller_receipt(
    public_value: Mapping[str, Any],
    replay: ActiveFrontReobserveReceipt,
) -> None:
    public = _controller_receipt_payload_from_public(public_value)
    expected = {**replay.as_dict(), "audit_digest": replay.audit_digest}
    if public != expected:
        raise ValueError("Stage 2A public controller receipt 不等于重放结果")


def verify_e018_p1_stage2a_integration_smoke(
    *,
    stage2_config_path: str | Path,
    qualification_config_path: str | Path,
    output_root: str | Path,
    expected_source_git_commit: str | None = None,
    expected_source_identity_sha256: str | None = None,
) -> dict[str, Any]:
    """离线重算 Stage 2A smoke；不接模型、环境、label 或 test。"""

    root = Path(output_root)
    _verify_stage2a_exact_file_tree(root)
    loaded = load_e018_p1_stage2a_config(stage2_config_path)
    qualification_path = Path(qualification_config_path)
    qualification = load_g2c_dynamic_qualification_config(qualification_path)
    if (
        file_sha256(qualification_path)
        != loaded.payload["parents"]["d048_qualification_config_raw_sha256"]
        or qualification["config_sha256"]
        != loaded.payload["parents"]["d048_qualification_config_internal_sha256"]
    ):
        raise RuntimeError("Stage 2A verifier qualification config identity 漂移")
    started = _read_stage2a_json(root / "RUN_STARTED.json", "RUN_STARTED")
    _require_exact_keys(
        started,
        {
            "version",
            "status",
            "classification",
            "effect_claim",
            "wrist_capability",
            "seed_range",
            "fresh_test_reads",
            "runtime_object_gt_reads",
            "goal_gt_reads",
            "offline_label_reads",
        },
        "RUN_STARTED",
    )
    if (
        started["version"] != E018_P1_STAGE2A_EXECUTION_VERSION
        or started["status"] != "in-progress-engineering-integration-smoke"
        or started["classification"] != "engineering-integration-smoke"
        or started["effect_claim"] != "no-effect-claim"
        or started["wrist_capability"] != "not-evaluated"
        or started["seed_range"] != [76901, 76910]
        or any(
            started[name] != 0
            for name in (
                "fresh_test_reads",
                "runtime_object_gt_reads",
                "goal_gt_reads",
                "offline_label_reads",
            )
        )
    ):
        raise RuntimeError("RUN_STARTED scope/permission 漂移")

    freeze = _read_stage2a_json(root / "execution_freeze.json", "execution freeze")
    _require_exact_keys(
        freeze,
        {
            "version",
            "status",
            "context_destroyed",
            "provider_context_destroyed",
            "environment_closed",
            "prediction_ledger_frozen",
            "decision_ledger_frozen",
            "offline_oracle_labels_opened",
            "offline_label_reads",
            "runtime_object_gt_reads",
            "goal_gt_reads",
            "fresh_test_reads",
            "checkpoint_writes",
            "artifact_inventory",
            "frozen_at_unix_ns",
            "freeze_sha256",
        },
        "execution freeze",
    )
    freeze_without_sha = dict(freeze)
    stored_freeze_sha = freeze_without_sha.pop("freeze_sha256")
    if (
        stored_freeze_sha != canonical_sha256(freeze_without_sha)
        or freeze["status"]
        != "prediction-and-decision-ledgers-frozen-context-destroyed"
        or any(
            freeze[name] is not True
            for name in (
                "context_destroyed",
                "provider_context_destroyed",
                "environment_closed",
                "prediction_ledger_frozen",
                "decision_ledger_frozen",
            )
        )
        or freeze["offline_oracle_labels_opened"] is not False
        or any(
            freeze[name] != 0
            for name in (
                "offline_label_reads",
                "runtime_object_gt_reads",
                "goal_gt_reads",
                "fresh_test_reads",
                "checkpoint_writes",
            )
        )
    ):
        raise RuntimeError("execution freeze identity/order/permission 漂移")
    inventory = _require_exact_keys(
        freeze["artifact_inventory"],
        set(_STAGE2A_ARTIFACT_FILES_BEFORE_FREEZE),
        "execution freeze artifact inventory",
    )
    for name in _STAGE2A_ARTIFACT_FILES_BEFORE_FREEZE:
        item = _require_exact_keys(
            inventory[name], {"raw_sha256", "size_bytes"}, f"inventory.{name}"
        )
        path = root / name
        if (
            file_sha256(path) != item["raw_sha256"]
            or path.stat().st_size != item["size_bytes"]
        ):
            raise RuntimeError(f"Stage 2A frozen artifact 漂移: {name}")

    provider_rows = _read_stage2a_jsonl(
        root / "provider_output_ledger.jsonl", "provider output ledger"
    )
    records = [_provider_record_from_dict(value) for value in provider_rows]
    if len(records) != 40:
        raise RuntimeError("Stage 2A provider output count 必须是 40")
    for record in records:
        verify_stage2a_provider_output_record(
            record,
            stage2_config=loaded,
            qualification_config=qualification,
        )
    grouped_records: dict[int, list[Stage2AProviderOutputRecord]] = {}
    for record in records:
        seed = int(record.prediction["seed"])
        grouped_records.setdefault(seed, []).append(record)
    if tuple(sorted(grouped_records)) != STAGE2A_INTEGRATION_SMOKE_SEEDS or any(
        tuple(value.route_frame_index for value in grouped_records[seed])
        != STAGE2A_PROVIDER_FRAME_INDICES
        for seed in STAGE2A_INTEGRATION_SMOKE_SEEDS
    ):
        raise RuntimeError("Stage 2A provider seed/frame order 漂移")

    camera_rows = _read_stage2a_jsonl(
        root / "camera_pose_ledger.jsonl", "camera pose ledger"
    )
    route_summaries = _read_stage2a_jsonl(
        root / "route_summaries.jsonl", "route summaries"
    )
    transactions = _read_stage2a_jsonl(
        root / "transaction_ledger.jsonl", "transaction ledger"
    )
    if len(camera_rows) != 920 or len(route_summaries) != 10 or len(transactions) != 10:
        raise RuntimeError("Stage 2A route/frame/transaction count 漂移")
    for row in camera_rows:
        if (
            row.get("offline_segmentation_diagnostics") is not None
            or row.get("is_grasping") is not None
            or row.get("robot_object_contact_force_n") is not None
            or row.get("memory_write_executed") is not False
            or row.get("arm_motion_command_max_abs") != 0.0
            or row.get("gripper_hold_open_command") != 1.0
        ):
            raise RuntimeError("Stage 2A camera ledger privileged/actuation witness 漂移")

    expected_transaction_keys = {
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
    commit_count = 0
    action_count = 0
    spec = RobotSpec()
    for index, transaction in enumerate(transactions):
        _require_exact_keys(
            transaction,
            expected_transaction_keys,
            f"transaction[{index}]",
        )
        seed = STAGE2A_INTEGRATION_SMOKE_SEEDS[index]
        episode_id = _stage2a_episode_id(seed)
        request_id = f"{episode_id}-active-front-01"
        rows = camera_rows[index * 92 : (index + 1) * 92]
        route_summary = route_summaries[index]
        controller = _new_stage2a_replay_controller(
            loaded, episode_id=episode_id
        )
        if (
            transaction["seed"] != seed
            or transaction["episode_id"] != episode_id
            or transaction["request_id"] != request_id
            or transaction["version"] != E018_P1_STAGE2A_EXECUTION_VERSION
            or transaction["classification"]
            != "engineering-integration-smoke"
            or transaction["effect_claim"] != "no-effect-claim"
            or transaction["wrist_capability"] != "not-evaluated"
            or transaction["home_barrier_frame_indices"]
            != list(STAGE2A_HOME_BARRIER_FRAME_INDICES)
            or route_summary.get("seed") != seed
            or route_summary.get("episode_id") != episode_id
            or route_summary.get("frame_count") != 92
            or route_summary.get("provider_forward_count") != 4
            or route_summary.get("memory_write_count")
            != transaction["memory_write_count"]
            or transaction["route_passed"] is not bool(route_summary.get("passed"))
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
        ):
            raise RuntimeError(f"Stage 2A transaction[{index}] identity/permission 漂移")

        trigger_records, source_recheck = _verify_stage2a_trigger_replay(
            transaction,
            controller=controller,
            episode_id=episode_id,
        )
        reset_receipt = verify_stage2a_action_history_audit(
            transaction["action_history_reset_audit"]
        )
        if (
            not isinstance(reset_receipt, ActionHistoryResetReceipt)
            or reset_receipt.episode_id != episode_id
            or reset_receipt.request_id != request_id
            or reset_receipt.reset_control_tick != 2
        ):
            raise RuntimeError("Stage 2A Action history reset receipt 漂移")

        verified_rows = [
            _verify_stage2a_camera_row_identity(
                row,
                episode_id=episode_id,
                request_id=request_id,
                frame_index=frame_index,
            )
            for frame_index, row in enumerate(rows)
        ]
        primary_frames = _verify_stage2a_provider_transaction_binding(
            transaction,
            records=grouped_records[seed],
            rows=verified_rows,
            episode_id=episode_id,
            request_id=request_id,
        )
        candidate_receipt = _verify_stage2a_candidate_binding(
            transaction,
            primary_frames=primary_frames,
        )

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
            raise RuntimeError("Stage 2A controller/safety/HOME ledger count 漂移")
        event_index = 0
        home_frames: list[HomeV2BarrierFrame] = []
        home_frame_digests: list[str] = []
        home_timestamps: list[float] = []
        for frame_index, row in enumerate(verified_rows):
            if frame_index > 0:
                _verify_stage2a_camera_authorization(
                    row,
                    authorization_values[frame_index - 1],
                    controller=controller,
                )
            safety = _verify_stage2a_safety_record(
                row,
                safety_values[frame_index],
                controller=controller,
            )
            if (
                frame_index > 0
                and frame_index not in STAGE2A_HOME_BARRIER_FRAME_INDICES
                and not controller.observe_safety(
                    safety,
                    camera_at_home=bool(
                        row["camera_motion_state"]
                        in {
                            ExternalCameraMotionState.RETURN_HOME.value,
                            ExternalCameraMotionState.VERIFY_HOME_AND_ARM_HOLD.value,
                        }
                        and _stage2a_pose_at_home(row)
                    ),
                )
            ):
                raise RuntimeError("Stage 2A replay 遇到未宣告 safety failure")

            replay_events: list[
                tuple[
                    ActiveFrontSignal,
                    str | None,
                    Stage2MemoryCandidateReceipt | None,
                ]
            ] = []
            if frame_index == 0:
                controller.begin(reset_receipt)
                replay_events.extend(
                    (
                        (ActiveFrontSignal.CAMERA_LEASE_ACQUIRED, None, None),
                        (
                            ActiveFrontSignal.FROZEN_PRIMITIVE_SELECTED,
                            ACTIVE_FRONT_PRIMARY_PRIMITIVE_ID,
                            None,
                        ),
                    )
                )
            elif frame_index == 40:
                replay_events.append((ActiveFrontSignal.MOVE_COMPLETE, None, None))
            elif frame_index == 44:
                if row["settled"] is not True:
                    raise RuntimeError("Stage 2A settle completion帧未 settled")
                replay_events.append((ActiveFrontSignal.SETTLE_COMPLETE, None, None))
            elif frame_index == 47:
                replay_events.extend(
                    (
                        (ActiveFrontSignal.COLLECTION_COMPLETE, None, None),
                        (
                            ActiveFrontSignal.SHADOW_CANDIDATE_STAGED,
                            None,
                            candidate_receipt,
                        ),
                    )
                )
            elif frame_index == 87:
                replay_events.append(
                    (ActiveFrontSignal.RETURN_HOME_COMPLETE, None, None)
                )
            for signal, primitive_id, staged_candidate in replay_events:
                if event_index >= len(event_values):
                    raise RuntimeError("Stage 2A controller event ledger 提前结束")
                _replay_stage2a_controller_event(
                    controller,
                    event_values[event_index],
                    signal=signal,
                    frame_index=frame_index,
                    safety=safety,
                    selected_primitive_id=primitive_id,
                    candidate_receipt=staged_candidate,
                )
                event_index += 1

            if frame_index in STAGE2A_HOME_BARRIER_FRAME_INDICES:
                home_index = frame_index - STAGE2A_HOME_BARRIER_FRAME_INDICES[0]
                frame, frame_digest = _verify_stage2a_home_barrier_evidence(
                    row,
                    home_values[home_index],
                    episode_id=episode_id,
                    request_id=request_id,
                    safety_evidence_sha256=safety_values[frame_index][
                        "evidence_sha256"
                    ],
                )
                controller.accept_home_v2_barrier_frame(frame)
                home_frames.append(frame)
                home_frame_digests.append(frame_digest)
                home_timestamps.append(
                    float(home_values[home_index]["control_timestamp_s"])
                )

        home_ids = [value.observation_sequence_id for value in home_frames]
        if (
            transaction["home_observation_sequence_ids"] != home_ids
            or transaction["home_observation_timestamps_s"] != home_timestamps
            or transaction["home_frame_digests"] != home_frame_digests
        ):
            raise RuntimeError("Stage 2A HOME barrier receipt 不能从四帧重算")
        observation_window = verify_stage2a_observation_v2_window_identity(
            transaction["observation_v2_window_identity"],
            spec=spec,
            home_evidence=home_values,
            home_motion_rows=[
                verified_rows[value]
                for value in STAGE2A_HOME_BARRIER_FRAME_INDICES
            ],
            expected_episode_id=episode_id,
            expected_episode_generation=1,
        )
        if observation_window.frame_timestamp_s.tolist() != home_timestamps:
            raise RuntimeError("Stage 2A Observation V2/HOME timestamp 漂移")
        _verify_stage2a_source_recheck_identity(
            source_recheck,
            trigger_records=trigger_records,
            final_home_evidence=home_values[-1],
            episode_id=episode_id,
            request_id=request_id,
        )

        writes = transaction["memory_write_count"]
        if type(writes) is not int or writes not in {0, 1}:
            raise RuntimeError("Stage 2A transaction write count 非法")
        commit = transaction["commit_receipt"]
        action = transaction["shadow_action_generation"]
        route_passed = bool(route_summary["passed"])
        if candidate_receipt.commit_eligible:
            if event_index >= len(event_values):
                raise RuntimeError("Stage 2A source recheck controller event 缺失")
            final_safety = _stage2a_active_safety(
                verified_rows[-1], controller=controller
            )
            _replay_stage2a_controller_event(
                controller,
                event_values[event_index],
                signal=ActiveFrontSignal.SOURCE_INVARIANTS_VERIFIED,
                frame_index=STAGE2A_HOME_BARRIER_FRAME_INDICES[-1],
                safety=final_safety,
                source_phase=STAGE2A_SOURCE_PHASE,
                source_invariants_passed=route_passed,
            )
            event_index += 1
        if event_index != len(event_values):
            raise RuntimeError("Stage 2A controller event ledger 存在额外或乱序记录")

        if writes == 1:
            resume_audit = transaction["action_history_resume_audit"]
            if not isinstance(resume_audit, dict):
                raise RuntimeError("Stage 2A success 缺 fresh Action history audit")
            resume_receipt = verify_stage2a_action_history_audit(resume_audit)
            window_sha256 = transaction["observation_v2_window_identity"][
                "window_sha256"
            ]
            if (
                not candidate_receipt.commit_eligible
                or not route_passed
                or not isinstance(resume_receipt, ActionHistoryResumeReceipt)
                or transaction["action_history_reset_audit"]["after"]
                != resume_audit["before"]
                or resume_receipt.episode_id != episode_id
                or resume_receipt.request_id != request_id
                or resume_receipt.generation
                != reset_receipt.generation_after + 1
                or list(resume_receipt.home_observation_sequence_ids) != home_ids
                or resume_receipt.observation_v2_window_sha256 != window_sha256
                or resume_audit["fresh_home_bundle"]["observation_v2_window"]
                != transaction["observation_v2_window_identity"]
                or resume_audit["home_evidence_digests"]
                != [value["evidence_sha256"] for value in home_values]
                or not isinstance(commit, dict)
                or not isinstance(action, dict)
                or commit.get("source_recheck_wrist_evidence_identity_sha256")
                != source_recheck.digest
                or commit.get("memory_write_count") != 1
                or commit.get("observable_now") is not False
                or commit.get("memory_only") is not True
                or commit.get("contact_authorized") is not False
                or len(commit.get("home_observation_sequence_ids", [])) != 4
                or action.get("generated_from_fresh_home_v2") is not True
                or action.get("stale_action_chunk_resumed") is not False
                or action.get("memory_only") is not True
                or action.get("contact_authorized") is not False
                or action.get("shadow_only") is not True
            ):
                raise RuntimeError("Stage 2A commit/fresh Action receipt 语义漂移")
            commit_primitive = dict(commit)
            commit_digest = commit_primitive.pop("digest", None)
            action_primitive = dict(action)
            action_digest = action_primitive.pop("digest", None)
            if (
                canonical_sha256(commit_primitive) != commit_digest
                or canonical_sha256(action_primitive) != action_digest
                or commit.get("episode_id") != episode_id
                or commit.get("request_id") != request_id
                or commit.get("candidate_digest")
                != transaction["candidate_digest"]
                or commit.get("pre_state_digest")
                != trigger_records[-1].memory_state_revision
                or commit.get("home_observation_sequence_ids") != home_ids
                or commit.get("home_observation_timestamps_s") != home_timestamps
                or commit.get("home_frame_digests") != home_frame_digests
                or commit.get("post_state_digest")
                != canonical_sha256(transaction["final_memory_state"])
                or action.get("candidate_digest")
                != transaction["candidate_digest"]
                or action.get("commit_receipt_digest") != commit_digest
                or action.get("action_generation_before")
                != reset_receipt.generation_after
                or action.get("action_generation_after") != resume_receipt.generation
                or action.get("source_phase") != STAGE2A_SOURCE_PHASE.value
                or action.get("resume_phase") != STAGE2A_SOURCE_PHASE.value
            ):
                raise RuntimeError("Stage 2A commit/Action receipt digest 漂移")
            final_state = _object_state_from_snapshot(
                transaction["final_memory_state"]
            )
            if (
                not final_state.valid
                or final_state.mode is not ObjectMemoryMode.FREE_STATIC
                or final_state.episode_id != episode_id
                or final_state.accepted_update_count != 1
                or resume_audit["memory_state_sha256"]
                != canonical_sha256(transaction["final_memory_state"])
            ):
                raise RuntimeError("Stage 2A exactly-once final Memory state 漂移")
            replay_receipt = controller.complete_stage2_memory_write(
                resume_receipt,
                memory_write_count=1,
                provider_forward_count=4,
            )
            commit_count += 1
            action_count += 1
        else:
            if (
                commit is not None
                or action is not None
                or transaction["action_history_resume_audit"] is not None
                or (candidate_receipt.commit_eligible and route_passed)
            ):
                raise RuntimeError("Stage 2A no-write transaction 不得生成 commit/Action")
            final_state = _object_state_from_snapshot(
                transaction["final_memory_state"]
            )
            if final_state.valid or final_state.accepted_update_count != 0:
                raise RuntimeError("Stage 2A rejected/failed route 不得写 Memory")
            replay_receipt = controller.receipt(
                memory_write_count=0,
                provider_forward_count=4,
            )
        _verify_stage2a_controller_receipt(
            transaction["controller_receipt"], replay_receipt
        )

    receipt = _read_stage2a_json(
        root / "execution_receipt.json", "Stage 2A execution receipt"
    )
    expected_receipt_keys = {
        "version",
        "status",
        "classification",
        "effect_claim",
        "wrist_capability",
        "integration_plumbing_passed",
        "success_path_exercised",
        "negative_results_preserved",
        "stage2_config_raw_sha256",
        "stage2_config_canonical_sha256",
        "source_identity",
        "parent_verification",
        "d050_absent_wrist_capability_commit",
        "d050_experiment_id",
        "wrist_capability_status",
        "wrist_capability_record_version",
        "provider_record_version",
        "seed_range",
        "provider_frame_indices",
        "primary_collect_frame_indices",
        "home_barrier_frame_indices",
        "counts",
        "candidate_rejection_reason_counts",
        "gpu_wall_seconds",
        "environment_identity",
        "execution_freeze_raw_sha256",
        "execution_freeze_internal_sha256",
        "formal_claim_allowed",
        "fresh_test_status",
        "receipt_sha256",
    }
    _require_exact_keys(receipt, expected_receipt_keys, "execution receipt")
    receipt_primitive = dict(receipt)
    receipt_digest = receipt_primitive.pop("receipt_sha256")
    source = receipt["source_identity"]
    counts = receipt["counts"]
    if (
        canonical_sha256(receipt_primitive) != receipt_digest
        or receipt["version"] != E018_P1_STAGE2A_EXECUTION_VERSION
        or receipt["status"] != "complete-engineering-integration-smoke"
        or receipt["classification"] != "engineering-integration-smoke"
        or receipt["effect_claim"] != "no-effect-claim"
        or receipt["wrist_capability"] != "not-evaluated"
        or receipt["negative_results_preserved"] is not True
        or receipt["stage2_config_raw_sha256"] != loaded.raw_sha256
        or receipt["stage2_config_canonical_sha256"] != loaded.canonical_sha256
        or receipt["d050_absent_wrist_capability_commit"]
        != _D050_ABSENT_WRIST_CAPABILITY_COMMIT
        or receipt["d050_experiment_id"] != _D050_EXPERIMENT_ID
        or receipt["wrist_capability_status"] != WRIST_CAPABILITY_ABSENT_STATUS
        or receipt["wrist_capability_record_version"]
        != E018_P1_STAGE2A_WRIST_CAPABILITY_VERSION
        or receipt["provider_record_version"]
        != E018_P1_STAGE2A_PROVIDER_RECORD_VERSION
        or receipt["seed_range"] != [76901, 76910]
        or receipt["provider_frame_indices"] != list(STAGE2A_PROVIDER_FRAME_INDICES)
        or receipt["primary_collect_frame_indices"]
        != list(STAGE2A_COLLECT_FRAME_INDICES)
        or receipt["home_barrier_frame_indices"]
        != list(STAGE2A_HOME_BARRIER_FRAME_INDICES)
        or receipt["execution_freeze_raw_sha256"]
        != file_sha256(root / "execution_freeze.json")
        or receipt["execution_freeze_internal_sha256"] != stored_freeze_sha
        or receipt["formal_claim_allowed"] is not False
        or receipt["fresh_test_status"] != "prohibited-unread"
    ):
        raise RuntimeError("Stage 2A execution receipt identity/scope 漂移")
    if not isinstance(source, dict) or (
        expected_source_git_commit is not None
        and source.get("git_commit") != expected_source_git_commit
    ) or (
        expected_source_identity_sha256 is not None
        and source.get("identity_sha256") != expected_source_identity_sha256
    ):
        raise RuntimeError("Stage 2A receipt source identity 漂移")
    recomputed_counts = {
        "seed_count": len(transactions),
        "route_count": len(route_summaries),
        "frame_count": len(camera_rows),
        "provider_forward_count": len(records),
        "home_raw_score_forward_count": sum(
            value.route_frame_index == 0 for value in records
        ),
        "primary_collect_forward_count": sum(
            value.route_frame_index in STAGE2A_COLLECT_FRAME_INDICES
            for value in records
        ),
        "wrist_provider_forward_count": 0,
        "memory_commit_count": commit_count,
        "fresh_shadow_action_generation_count": action_count,
        "route_pass_count": sum(bool(value.get("passed")) for value in route_summaries),
        "terminal_transaction_count": sum(
            value["orchestrator_state"]
            in {
                PendingActiveViewState.COMMITTED.value,
                PendingActiveViewState.HOME_VERIFIED_FAILED_SAFE_HOLD.value,
            }
            for value in transactions
        ),
        "runtime_object_gt_reads": 0,
        "goal_gt_reads": 0,
        "offline_label_reads": 0,
        "fresh_test_reads": 0,
        "checkpoint_writes": 0,
        "arm_motion_command_count": 0,
        "gripper_close_command_count": 0,
    }
    if counts != recomputed_counts:
        raise RuntimeError("Stage 2A receipt counters 漂移")
    recomputed_plumbing = bool(
        recomputed_counts["seed_count"] == 10
        and recomputed_counts["route_count"] == 10
        and recomputed_counts["frame_count"] == 920
        and recomputed_counts["provider_forward_count"] == 40
        and recomputed_counts["home_raw_score_forward_count"] == 10
        and recomputed_counts["primary_collect_forward_count"] == 30
        and recomputed_counts["route_pass_count"] == 10
        and recomputed_counts["terminal_transaction_count"] == 10
        and commit_count >= 1
        and action_count == commit_count
        and float(receipt["gpu_wall_seconds"])
        <= loaded.payload["budgets"]["integration_smoke_gpu_wall_seconds_max"]
    )
    if (
        receipt["integration_plumbing_passed"] is not recomputed_plumbing
        or receipt["success_path_exercised"] is not (commit_count >= 1)
    ):
        raise RuntimeError("Stage 2A integration plumbing gate 漂移")
    verification = {
        "version": "e018-p1-stage2a-integration-smoke-verification/v1",
        "status": "verified-engineering-integration-smoke",
        "classification": "engineering-integration-smoke",
        "effect_claim": "no-effect-claim",
        "wrist_capability": "not-evaluated",
        "integration_plumbing_passed": recomputed_plumbing,
        "success_path_exercised": commit_count >= 1,
        "source_git_commit": source.get("git_commit"),
        "source_identity_sha256": source.get("identity_sha256"),
        "stage2_config_raw_sha256": loaded.raw_sha256,
        "stage2_config_canonical_sha256": loaded.canonical_sha256,
        "receipt_raw_sha256": file_sha256(root / "execution_receipt.json"),
        "receipt_internal_sha256": receipt_digest,
        "execution_freeze_internal_sha256": stored_freeze_sha,
        "counts": recomputed_counts,
        "fresh_test_reads": 0,
        "runtime_object_gt_reads": 0,
        "goal_gt_reads": 0,
        "offline_label_reads": 0,
    }
    verification["verification_sha256"] = canonical_sha256(verification)
    return verification
