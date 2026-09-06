"""E018-P1 Stage 2A 固定 gain 的一次性 development evaluation。"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robot_vla.precision.calibrated_front_provider import canonical_sha256
from robot_vla.precision.e018_p1_g2c_qualification import (
    _validate_qualification_object_label,
)
from robot_vla.precision.e018_p1_stage2a import (
    E018_P1_STAGE2A_SELECTED_GAIN_EVALUATION_EXPERIMENT_ID,
    E018_P1_STAGE2A_SELECTED_GAIN_EVALUATION_PREFLIGHT_EXPERIMENT_ID,
    STAGE2A_COLLECT_FRAME_INDICES,
    STAGE2A_SELECTED_GAIN_EVALUATION_PREFLIGHT_SEED,
    STAGE2A_SELECTED_GAIN_EVALUATION_SEEDS,
)
from robot_vla.precision.e018_p1_stage2a_selection import (
    CapturedSelectionRoute,
    GainBranchOutcome,
    _is_sha256,
    _require_exact_keys,
    replay_gain_branch,
)

E018_P1_STAGE2A_EVALUATION_CONFIG_VERSION = (
    "e018-p1-stage2a-selected-gain-development-evaluation/v2"
)
E018_P1_STAGE2A_EVALUATION_EXPERIMENT_ID = (
    E018_P1_STAGE2A_SELECTED_GAIN_EVALUATION_EXPERIMENT_ID
)
E018_P1_STAGE2A_EVALUATION_EXECUTION_VERSION = (
    "e018-p1-stage2a-selected-gain-development-evaluation-execution/v2"
)
E018_P1_STAGE2A_EVALUATION_RESULT_VERSION = (
    "e018-p1-stage2a-selected-gain-development-evaluation-result/v2"
)
STAGE2A_EVALUATION_GO = (
    "E018_P1_STAGE2A_SELECTED_GAIN_EVALUATION_GO_77626_77650_V2"
)
STAGE2A_EVALUATION_PREFLIGHT_GO = (
    "E018_P1_STAGE2A_SELECTED_GAIN_EVALUATION_RECOVERY_PREFLIGHT_GO_76894_V2"
)
STAGE2A_EVALUATION_SEEDS = STAGE2A_SELECTED_GAIN_EVALUATION_SEEDS
STAGE2A_EVALUATION_SELECTED_GAIN = 0.10

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
_EVALUATION_PRIVATE_CAPTURE_KEYS = _QUALIFICATION_OBJECT_LABEL_KEYS | {
    "object_linear_speed_m_s",
    "object_angular_speed_rad_s",
    "object_motion_event",
}

_TOP_LEVEL_KEYS = {
    "version",
    "status",
    "experiment",
    "selection_parent",
    "stage2a_parent",
    "split",
    "preflight",
    "fixed_rule",
    "phase_boundary",
    "oracle",
    "promotion",
    "budgets",
    "permissions",
}


@dataclass(frozen=True)
class LoadedStage2AEvaluationConfig:
    canonical_json: str
    raw_sha256: str
    canonical_sha256: str

    @property
    def payload(self) -> dict[str, Any]:
        return json.loads(self.canonical_json)


def _validate_evaluation_config(config: dict[str, Any]) -> None:
    _require_exact_keys(config, _TOP_LEVEL_KEYS, "Stage 2A evaluation config")
    if (
        config["version"] != E018_P1_STAGE2A_EVALUATION_CONFIG_VERSION
        or config["status"]
        != "preregistered-conditional-development-evaluation-no-test-no-actuation"
    ):
        raise ValueError("Stage 2A evaluation config version/status 漂移")

    experiment = _require_exact_keys(
        config["experiment"],
        {
            "id",
            "gate",
            "gate_record_raw_sha256",
            "classification",
            "exact_go_token",
            "rerun_under_same_identity_allowed",
            "allowed_conclusion",
        },
        "Stage 2A evaluation experiment",
    )
    if experiment != {
        "id": E018_P1_STAGE2A_EVALUATION_EXPERIMENT_ID,
        "gate": "D049-R2-RECOVERY-SCOPE-AMENDED",
        "gate_record_raw_sha256": (
            "cae0d7b69248146a5767e9dbd909e6560d553f7034bacbebbded9530cb7f49e8"
        ),
        "classification": "formal-development-evaluation-no-test-no-actuation/v2",
        "exact_go_token": STAGE2A_EVALUATION_GO,
        "rerun_under_same_identity_allowed": False,
        "allowed_conclusion": (
            "fresh-development-absolute-recovery-pass-negative-or-"
            "inconclusive-no-effect-no-actuation/v2"
        ),
    }:
        raise ValueError("Stage 2A evaluation experiment identity 漂移")

    selection_parent = _require_exact_keys(
        config["selection_parent"],
        {
            "experiment_id",
            "artifact_id",
            "source_git_commit",
            "source_identity_sha256",
            "config_raw_sha256",
            "config_canonical_sha256",
            "transaction_identity_sha256",
            "public_completion_marker_sha256",
            "public_verification_sha256",
            "consumption_marker_raw_sha256",
            "consumption_marker_internal_sha256",
            "result_completion_marker_sha256",
            "selection_summary_sha256",
            "result_verification_sha256",
            "common_denominator_count",
            "selected_gain",
            "selection_reason",
            "selected_gain_recovered_count",
            "selected_gain_unsafe_count",
            "selected_gain_catastrophic_count",
            "selected_gain_false_recovery_count",
            "selected_gain_protocol_violation_count",
            "selection_was_tie",
            "gain_reselection_allowed",
            "failed_selection_v1_private_labels_reused",
            "persistence",
        },
        "Stage 2A evaluation selection parent",
    )
    expected_selection_parent = {
        "experiment_id": (
            "E018-P1-S2A-MIN-INFORMATION-GAIN-SELECTION-DEVELOPMENT/v2"
        ),
        "artifact_id": (
            "stage2a-selection-formal-8d733f8-77601-77625-20260906-v2"
        ),
        "source_git_commit": "8d733f8e8443a0d3e5ddcea04c5b1ae837f3e3bf",
        "source_identity_sha256": (
            "45cb3f084a10ff848708f45f332e68b54f184e1d5ad2b528cadda91e21e3eb84"
        ),
        "config_raw_sha256": (
            "6b6e31eb8b5655618b580ff8fb5950b7a0cca44a6d917e41a758ce6d6755d2e2"
        ),
        "config_canonical_sha256": (
            "01b38209bd6bf9619546d8236978aecb67517f76689836f49162924ed92634ee"
        ),
        "transaction_identity_sha256": (
            "3c53990c06565b59a11036c27e1d68362ef171341d9722d87cc04d5e781d9622"
        ),
        "public_completion_marker_sha256": (
            "f14aae96d8ab4cfa00c294ea50d7a1aa4d89e341cde34875ad3decb8830d8327"
        ),
        "public_verification_sha256": (
            "98f0aa47ce07cbfd2be9aca800bdf54f15787db79fc8b1daf1a5d1f0c4d4bc65"
        ),
        "consumption_marker_raw_sha256": (
            "f7c5cf367f00cca722ec0ced828f66da9becb4e638e1438fb8526b7297d59228"
        ),
        "consumption_marker_internal_sha256": (
            "816a8cad7ae2a00cfd15205bb10fe0b5eb7378d6a3c89bb0d4208d32b7ac9752"
        ),
        "result_completion_marker_sha256": (
            "3b04687135df8a4410c298621a6a77b92d509f4909caf95ce213004dfca2ac4e"
        ),
        "selection_summary_sha256": (
            "55cbe4e5d6610d7d5936fc18a97a8735507cdd098cd84efd9890a628ea8250af"
        ),
        "result_verification_sha256": (
            "22e0634ac2dc89eed58a968bf5f106165a21cfac3c5fa5d5b86f10f23e345ce7"
        ),
        "common_denominator_count": 24,
        "selected_gain": STAGE2A_EVALUATION_SELECTED_GAIN,
        "selection_reason": (
            "maximize-integer-recovered-count-then-larger-gain-after-three-way-tie"
        ),
        "selected_gain_recovered_count": 5,
        "selected_gain_unsafe_count": 0,
        "selected_gain_catastrophic_count": 0,
        "selected_gain_false_recovery_count": 0,
        "selected_gain_protocol_violation_count": 0,
        "selection_was_tie": True,
        "gain_reselection_allowed": False,
        "failed_selection_v1_private_labels_reused": False,
        "persistence": {
            "status": "REPLICATED",
            "manifest_raw_sha256": (
                "30ed214d7b3e9698b008e8a6d0e849d6c2f6f94e3a3c927f91d6f227421ec74c"
            ),
            "manifest_internal_sha256": (
                "65e6c7d2f71b935f579254853ad767165c576f5f1d07dac601b19ec84fb333d3"
            ),
            "artifact_inventory_sha256": (
                "3c3253170b1ed34f6fad974293e89eaf0e1c62f8477b6152ce5c688c16d2db88"
            ),
            "drive_marker_raw_sha256": (
                "fd0d80da227683996c5e90ce6751bb218b235ca4d67765916d90696de0349fb5"
            ),
            "drive_marker_internal_sha256": (
                "80d27f3d10a5351414f87b32a759851840d197d3b86e56deb0de352171b529ff"
            ),
            "drive_verification_internal_sha256": (
                "454e4be131e8428035c6aadb6566e36d6d2f5a07b9af58459a25804df4a80876"
            ),
            "local_verification_internal_sha256": (
                "7a445e1505fbed322e82a2a8a7bf66a5d5cd41a0c4cff4cc62502664fb78414a"
            ),
            "inventory_record_canonical_sha256": (
                "5d626a48e111fb33abbe9b97efe96898716108e1666b993272974cb263c41e13"
            ),
        },
    }
    if selection_parent != expected_selection_parent:
        raise ValueError("Stage 2A evaluation selection parent identity 漂移")

    if config["stage2a_parent"] != {
        "config_path": "configs/e018_p1_stage2a_primary_memory_development_v1.json",
        "config_raw_sha256": (
            "12794f1cc08f45d9dc5acae01c46d5f760f9456f8d73a308ae981a7b65a27512"
        ),
        "config_canonical_sha256": (
            "b9f33d39c668bb204754140c501996a13af20672ced19ec669503803bb9eb767"
        ),
        "checkpoint_sha256": (
            "97e3b7289911bc73f67755a8d9c3598c50b6c80ef01e1af13cec698ec59d3d77"
        ),
        "g0c_config_raw_sha256": (
            "b2d9787ade931f9e5ba6222c21b598a9b48b39efa396222dd6f64b0a73da07b1"
        ),
        "g0c_config_canonical_sha256": (
            "c93bbfd48b6d9bc2fc75b5b87e4ded7161efebd7eda50cd81cc2ded47810e965"
        ),
        "data_config_raw_sha256": (
            "5b825bbd1034e10801617d19dc10fca1e15f5c3253ca77571192d627e2d9e4ef"
        ),
        "data_config_canonical_sha256": (
            "56718c0611fc620ccfb767141d8d0867ea5d03806348396d0a2e201fbff3d5de"
        ),
        "qualification_config_raw_sha256": (
            "bfe5cbefeed8903a610ccab9ecff4d4f0e1cfd9fd4c92ec5dc1af03428f145b8"
        ),
        "qualification_config_internal_sha256": (
            "0ade177a588f3cfe2acb61634537f4d6ed3d92bb72daf52dcfb756e287864715"
        ),
        "proprio_stats_sha256": (
            "2a1061b3a56edfcfeb6e955a1910dc309ff9b776dc4eb355192661fe628de01e"
        ),
        "finger_force_stats_sha256": (
            "fcc5b4b87aa13919ec261fc5e71a24e1b6446f47abdbc87d4b1bf4f93fe7a9e8"
        ),
        "primary_primitive_id": "LEFT_LOW__PITCH_UP",
        "provider_write_threshold": 0.6127982139587402,
        "memory_mode": "free_static-position-only-base-frame/v1",
        "memory_max_unobserved_age_s": 2.5,
        "home_observation_v2_barrier_frames": 4,
    }:
        raise ValueError("Stage 2A evaluation parent/runtime identity 漂移")

    if config["split"] != {
        "seeds": [STAGE2A_EVALUATION_SEEDS[0], STAGE2A_EVALUATION_SEEDS[-1]],
        "seed_count": 25,
        "route_count": 25,
        "camera_frames_per_route": 92,
        "provider_frames_per_route": [0, 45, 46, 47],
        "private_label_frames_per_route": [45, 46, 47],
        "provider_prediction_count": 100,
        "fixed_gain_branch_count": 25,
        "private_label_count": 75,
        "execution_order": "ascending-seed-once/v1",
        "test_once": (
            "fresh-development-conditional-evaluation-one-transaction-"
            "no-same-identity-rerun/v2"
        ),
        "prior_status": "planning-only-unread",
        "stage2b_shadow_seeds": [77101, 77150],
        "stage3_reserved_seeds": [77201, 77250],
    }:
        raise ValueError("Stage 2A evaluation split/order/count 漂移")

    if config["preflight"] != {
        "experiment_id": (
            E018_P1_STAGE2A_SELECTED_GAIN_EVALUATION_PREFLIGHT_EXPERIMENT_ID
        ),
        "exact_go_token": STAGE2A_EVALUATION_PREFLIGHT_GO,
        "seed": STAGE2A_SELECTED_GAIN_EVALUATION_PREFLIGHT_SEED,
        "route_count": 1,
        "provider_prediction_count": 4,
        "fixed_gain_branch_count": 1,
        "private_label_count": 3,
        "two_pass_replay_required": True,
        "formal_identity_consumed": False,
        "allowed_conclusion": (
            "engineering-recovery-preflight-no-formal-evaluation-claim/v2"
        ),
    }:
        raise ValueError("Stage 2A evaluation preflight identity/protocol 漂移")

    if config["fixed_rule"] != {
        "selected_min_information_gain": STAGE2A_EVALUATION_SELECTED_GAIN,
        "gain_reselection_allowed": False,
        "provider_checkpoint_change_allowed": False,
        "provider_threshold_change_allowed": False,
        "oracle_change_allowed": False,
        "route_schema": "selection-capture-only-route-evidence/v2",
        "branch_replay": (
            "fresh-memory-and-action-state-fixed-gain-deployable-only/v1"
        ),
        "control": "matched-home-safehold-no-alternate-state-no-memory-no-action/v1",
        "effect_claim_allowed": False,
    }:
        raise ValueError("Stage 2A evaluation fixed rule 漂移")

    if config["phase_boundary"] != {
        "pass_a": (
            "provider-route-and-fixed-gain-decision-public-freeze-no-private-label/v1"
        ),
        "pass_a_private_label_capture_count": 0,
        "pass_a_private_label_open_count": 0,
        "pass_a_runtime_object_gt_read_count": 0,
        "public_complete_required_before_pass_b": True,
        "provider_and_environment_destroyed_before_pass_b": True,
        "pass_b": (
            "new-process-deterministic-label-replay-and-exact-once-scoring/v2"
        ),
        "pass_b_viewpoint_normalization": (
            "shared-stage2a-logical-viewpoint-normalizer-before-all-replay-bindings/v2"
        ),
        "pass_b_checkpoint_load_count": 0,
        "pass_b_provider_forward_count": 0,
        "pass_b_decision_change_count": 0,
        "pass_b_new_process_exact_once": True,
        "pass_b_first_gt_read_requires_o_excl_fsync_marker": True,
            "pass_b_replay_bindings": [
                "seed",
                "normalized-logical-viewpoint",
                "action-prefix",
                "rgb-sha256",
            "actual-pose-raw-and-canonical",
            "model-input-digest",
            "provider-output-digest",
        ],
        "pass_b_replay_mismatch_policy": (
            "permanent-consumption-failure-no-result-complete/v1"
        ),
        "durable_consumption_marker": "private_labels/SCORING_CONSUMED.json",
        "public_verifier_accepts_private_root": False,
        "result_verifier_accepts_private_or_model_root": False,
    }:
        raise ValueError("Stage 2A evaluation Pass A/B boundary 漂移")

    if config["oracle"] != {
        "denominator": (
            "route-protocol-safety-valid-and-all-three-primary-object-exists-"
            "own-mask-observable-no-contact-no-motion/v1"
        ),
        "object_observability": "d048-own-mask-not-goal-mask/v1",
        "contact_unsafe_if_strictly_greater_n": 0.01,
        "motion_predicate_version": "pick-and-place-predicates/v1",
        "motion_if_linear_strictly_greater_m_s": 0.01,
        "motion_if_angular_strictly_greater_rad_s": 0.5,
        "recovered_xyz_error_max_m": 0.005,
        "catastrophic_xyz_error_strictly_greater_m": 0.02,
    }:
        raise ValueError("Stage 2A evaluation oracle/threshold 漂移")

    if config["promotion"] != {
        "minimum_oracle_recoverable_support": 10,
        "minimum_recovery_rate": 0.7,
        "recovery_integer_numerator": 7,
        "recovery_integer_denominator": 10,
        "unsafe_recovery_count_max": 0,
        "catastrophic_recovery_count_max": 0,
        "false_recovery_count_max": 0,
        "protocol_violation_count_max": 0,
        "pass_classification": (
            "development-absolute-recovery-pass-no-effect-no-actuation-"
            "persist-publish-pause"
        ),
        "low_recovery_classification": (
            "effect-negative-persist-publish-pause-for-reusability-refactor"
        ),
        "low_support_classification": (
            "insufficient-support-inconclusive-persist-publish-pause-for-"
            "reusability-refactor"
        ),
        "safety_failure_classification": (
            "safety-negative-persist-publish-pause-for-reusability-refactor"
        ),
    }:
        raise ValueError("Stage 2A evaluation promotion gate 漂移")

    if config["budgets"] != {
        "gpu_wall_seconds_max": 1800,
        "combined_artifact_bytes_max": 2147483648,
        "stage2_cumulative_gpu_wall_seconds_max": 7200,
        "stage2_cumulative_artifact_bytes_max": 8589934592,
    }:
        raise ValueError("Stage 2A evaluation budget 漂移")

    if config["permissions"] != {
        "isolated_maniskill": True,
        "fresh_test_reads": 0,
        "runtime_object_gt_reads": 0,
        "offline_label_reads": "pass-b-only-after-public-freeze",
        "goal_gt_reads": 0,
        "checkpoint_writes": 0,
        "wrist_provider_forwards": 0,
        "physical_camera_actuation": 0,
        "arm_tcp_actuation": 0,
        "gripper_close": 0,
        "canonical_runtime_mutation": 0,
        "manipulation_progression": 0,
    }:
        raise ValueError("Stage 2A evaluation permission boundary 漂移")


def load_e018_p1_stage2a_evaluation_config(
    path: str | Path,
) -> LoadedStage2AEvaluationConfig:
    config_path = Path(path)
    raw = config_path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError("Stage 2A evaluation config 必须是 JSON object")
    _validate_evaluation_config(value)
    canonical_json = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return LoadedStage2AEvaluationConfig(
        canonical_json=canonical_json,
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        canonical_sha256=canonical_sha256(value),
    )


@dataclass(frozen=True)
class CapturedStage2AEvaluationRoute(CapturedSelectionRoute):
    """复用 selection typed route schema，但绑定 evaluation 独立 split。"""

    def __post_init__(self) -> None:
        formal_episode = (
            f"e018-p1-stage2a-selected-gain-evaluation-seed-{self.seed}"
        )
        preflight_episode = (
            "e018-p1-stage2a-selected-gain-evaluation-recovery-preflight-"
            f"seed-{self.seed}"
        )
        if self.seed in STAGE2A_EVALUATION_SEEDS:
            expected_episode = formal_episode
        elif self.seed == STAGE2A_SELECTED_GAIN_EVALUATION_PREFLIGHT_SEED:
            expected_episode = preflight_episode
        else:
            raise ValueError("captured evaluation route seed 不在冻结 split/preflight")
        if (
            self.episode_id != expected_episode
            or self.request.episode_id != self.episode_id
            or self.request.episode_generation != 1
        ):
            raise ValueError("captured evaluation route Episode/request identity 漂移")
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
            raise ValueError("captured evaluation route replay evidence 数量/identity 漂移")
        primary_timestamps = tuple(
            frame.control_timestamp_s for frame in self.primary_frames
        )
        if primary_timestamps != tuple(sorted(primary_timestamps)):
            raise ValueError("captured evaluation PRIMARY timestamps 未按固定顺序")
        if any(
            later <= earlier + 1e-12
            for earlier, later in zip(
                self.home_timestamps_s,
                self.home_timestamps_s[1:],
            )
        ):
            raise ValueError("captured evaluation HOME timestamps 必须严格递增")
        if self.physical_route_count != 1 or self.captured_provider_forward_count != 4:
            raise ValueError("每个 evaluation seed 必须恰好一条 route/四次 forward")
        if type(self.route_protocol_safety_valid) is not bool:
            raise TypeError("captured evaluation route safety 必须是 exact bool")
        if not _is_sha256(self.source_recheck_evidence_digest):
            raise ValueError("evaluation source recheck evidence digest 非法")
        digest = self.raw_candidate_digest_at_gain_0_02
        eligible = self.raw_candidate_commit_eligible_at_gain_0_02
        reasons = self.raw_candidate_rejection_reasons_at_gain_0_02
        if digest is None:
            if eligible is not None or reasons is not None:
                raise ValueError("absent evaluation raw candidate 必须完整为 None")
        elif (
            not _is_sha256(digest)
            or type(eligible) is not bool
            or type(reasons) is not tuple
            or any(not isinstance(reason, str) or not reason for reason in reasons)
            or len(set(reasons)) != len(reasons)
            or eligible != (not reasons)
        ):
            raise ValueError("evaluation raw candidate identity/eligibility 漂移")


@dataclass(frozen=True)
class SelectedGainEvaluationBranch(GainBranchOutcome):
    """固定 gain=0.10 的单分支结果，不接受 selection 的三候选语义。"""

    def __post_init__(self) -> None:
        if self.seed not in (
            *STAGE2A_EVALUATION_SEEDS,
            STAGE2A_SELECTED_GAIN_EVALUATION_PREFLIGHT_SEED,
        ):
            raise ValueError("evaluation branch seed 不在冻结 split/preflight")
        if self.gain != STAGE2A_EVALUATION_SELECTED_GAIN:
            raise ValueError("evaluation branch 必须使用 selection 固定的 gain=0.10")
        if not _is_sha256(self.route_evidence_digest):
            raise ValueError("evaluation branch route evidence digest 非法")
        for name in (
            "route_protocol_safety_valid",
            "candidate_commit_eligible",
            "navigation_state_available",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"evaluation branch {name} 必须是 bool")
        for name in (
            "memory_commit_count",
            "fresh_shadow_action_generation_count",
            "provider_forward_count",
            "arm_motion_command_count",
            "gripper_close_command_count",
            "protocol_violation_count",
        ):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"evaluation branch {name} 必须是整数")
        if self.memory_commit_count not in {0, 1}:
            raise ValueError("evaluation branch Memory commit count 只能是 0/1")
        if self.fresh_shadow_action_generation_count != self.memory_commit_count:
            raise ValueError("evaluation branch commit 与 fresh Action 必须一一对应")
        if self.provider_forward_count != 0:
            raise ValueError("evaluation logic branch 禁止 provider re-forward")
        if self.arm_motion_command_count != 0 or self.gripper_close_command_count != 0:
            raise ValueError("evaluation branch 禁止 manipulation actuator")
        if self.protocol_violation_count < 0:
            raise ValueError("evaluation protocol_violation_count 必须非负")
        if self.memory_commit_count == 1:
            if (
                not self.candidate_commit_eligible
                or not self.route_protocol_safety_valid
                or not self.navigation_state_available
                or self.fresh_shadow_action_generation_count != 1
            ):
                raise ValueError("evaluation commit/eligibility/navigation 不一致")
            position = np.asarray(self.committed_position_base_m, dtype=np.float64)
            if position.shape != (3,) or not np.isfinite(position).all():
                raise ValueError("committed evaluation branch 缺有限 base-frame XYZ")
        elif self.committed_position_base_m is not None or self.navigation_state_available:
            raise ValueError("no-commit evaluation branch 不得携带 XYZ/navigation")


def replay_selected_gain_branch(
    captured: CapturedStage2AEvaluationRoute,
) -> SelectedGainEvaluationBranch:
    """从公开 typed evidence 重放唯一固定 gain；不接 env/provider/label。"""

    result = replay_gain_branch(
        captured,
        STAGE2A_EVALUATION_SELECTED_GAIN,
        _outcome_type=SelectedGainEvaluationBranch,
    )
    if not isinstance(result, SelectedGainEvaluationBranch):
        raise TypeError("evaluation replay outcome type 漂移")
    return result


_EVALUATION_LABEL_KEYS = _EVALUATION_PRIVATE_CAPTURE_KEYS | {
    "version",
    "label_index",
    "prediction_row_index",
    "seed",
    "route_frame_index",
    "rgb_sha256",
    "actual_pose_sha256",
    "actual_pose_canonical_sha256",
    "model_input_digest",
    "provider_output_digest",
    "prediction_commit_receipt_sha256",
    "transaction_identity_sha256",
    "replay_camera_row_sha256",
    "motion_predicate_version",
    "motion_linear_threshold_m_s",
    "motion_angular_threshold_rad_s",
    "contact_threshold_n",
    "privileged_captured_at_unix_ns",
    "label_sha256",
}


def validate_evaluation_private_label(
    label: Mapping[str, Any],
    *,
    expected_label_index: int,
    seeds: Sequence[int] = STAGE2A_EVALUATION_SEEDS,
) -> dict[str, Any]:
    """验证 Pass B label 的顺序、阈值和 public replay 绑定字段。"""

    if len(seeds) < 1 or expected_label_index not in range(len(seeds) * 3):
        raise ValueError("evaluation label index/seeds 非法")
    row = _require_exact_keys(
        dict(label),
        _EVALUATION_LABEL_KEYS,
        f"evaluation private label[{expected_label_index}]",
    )
    _validate_qualification_object_label(
        {key: row[key] for key in _QUALIFICATION_OBJECT_LABEL_KEYS},
        committed=False,
    )
    seed = seeds[expected_label_index // 3]
    frame = STAGE2A_COLLECT_FRAME_INDICES[expected_label_index % 3]
    try:
        linear = float(row["object_linear_speed_m_s"])
        angular = float(row["object_angular_speed_rad_s"])
        captured_at = int(row["privileged_captured_at_unix_ns"])
    except (TypeError, ValueError) as error:
        raise RuntimeError("evaluation label numeric primitive 非法") from error
    unsigned = dict(row)
    stored = unsigned.pop("label_sha256")
    if (
        stored != canonical_sha256(unsigned)
        or row["version"] != E018_P1_STAGE2A_EVALUATION_EXECUTION_VERSION
        or row["label_index"] != expected_label_index
        or row["prediction_row_index"]
        != (expected_label_index // 3) * 4 + 1 + expected_label_index % 3
        or row["seed"] != seed
        or row["route_frame_index"] != frame
        or row["goal_gt_read_count"] != 0
        or row["test_data_read"] is not False
        or row["motion_predicate_version"] != "pick-and-place-predicates/v1"
        or row["motion_linear_threshold_m_s"] != 0.01
        or row["motion_angular_threshold_rad_s"] != 0.5
        or row["contact_threshold_n"] != 0.01
        or not math.isfinite(linear)
        or linear < 0.0
        or not math.isfinite(angular)
        or angular < 0.0
        or row["object_motion_event"]
        is not bool(linear > 0.01 or angular > 0.5)
        or captured_at <= 0
        or any(
            not _is_sha256(row[name])
            for name in (
                "rgb_sha256",
                "actual_pose_sha256",
                "actual_pose_canonical_sha256",
                "model_input_digest",
                "provider_output_digest",
                "prediction_commit_receipt_sha256",
                "transaction_identity_sha256",
                "replay_camera_row_sha256",
            )
        )
    ):
        raise RuntimeError("evaluation private label identity/order/hash 漂移")
    return row


def _validated_evaluation_branch(
    value: Mapping[str, Any],
    *,
    expected_seed: int,
) -> dict[str, Any]:
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
    payload = _require_exact_keys(
        dict(value), expected_keys, "selected-gain evaluation branch"
    )
    try:
        branch = SelectedGainEvaluationBranch(
            seed=payload["seed"],
            gain=payload["gain"],
            route_evidence_digest=payload["route_evidence_digest"],
            route_protocol_safety_valid=payload[
                "route_protocol_safety_valid"
            ],
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
        raise RuntimeError("evaluation branch 类型/状态语义漂移") from error
    if branch.seed != expected_seed or branch.to_dict() != payload:
        raise RuntimeError("evaluation branch order/fields/hash 不能重算")
    return payload


def score_selected_gain_evaluation(
    branches: Sequence[Mapping[str, Any]],
    private_labels: Sequence[Mapping[str, Any]],
    *,
    minimum_support: int = 10,
    minimum_recovery_rate: float = 0.70,
    seeds: Sequence[int] = STAGE2A_EVALUATION_SEEDS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """只评分冻结 fixed-gain branches；不进行任何 gain/threshold 选择。"""

    expected_seeds = tuple(seeds)
    if (
        not expected_seeds
        or len(set(expected_seeds)) != len(expected_seeds)
        or len(branches) != len(expected_seeds)
        or len(private_labels) != len(expected_seeds) * 3
        or type(minimum_support) is not int
        or minimum_support != 10
        or minimum_recovery_rate != 0.70
    ):
        raise RuntimeError("evaluation scoring count/gate identity 漂移")
    verified_branches = [
        _validated_evaluation_branch(value, expected_seed=expected_seeds[index])
        for index, value in enumerate(branches)
    ]
    verified_labels = [
        validate_evaluation_private_label(
            value,
            expected_label_index=index,
            seeds=expected_seeds,
        )
        for index, value in enumerate(private_labels)
    ]
    scored: list[dict[str, Any]] = []
    for index, (seed, branch) in enumerate(
        zip(expected_seeds, verified_branches, strict=True)
    ):
        labels = verified_labels[index * 3 : index * 3 + 3]
        oracle_eligible = bool(
            branch["route_protocol_safety_valid"]
            and all(label["gt_object_exists"] is True for label in labels)
            and all(label["gt_observable"] is True for label in labels)
            and all(
                float(label["robot_object_contact_force_n"]) <= 0.01
                for label in labels
            )
            and all(label["object_motion_event"] is False for label in labels)
            and all(label["is_grasped"] is False for label in labels)
        )
        committed = branch["memory_commit_count"] == 1
        xyz_error_m: float | None = None
        if committed:
            predicted = np.asarray(
                branch["committed_position_base_m"], dtype=np.float64
            )
            target = np.asarray(
                labels[-1]["gt_object_position_base_m"], dtype=np.float64
            )
            if predicted.shape != (3,) or target.shape != (3,):
                raise RuntimeError("evaluation scoring XYZ shape 漂移")
            xyz_error_m = float(np.linalg.norm(predicted - target))
        recovered = bool(
            oracle_eligible
            and committed
            and branch["navigation_state_available"] is True
            and branch["fresh_shadow_action_generation_count"] == 1
            and xyz_error_m is not None
            and xyz_error_m <= 0.005
        )
        false_recovery = bool(
            committed
            and (
                not oracle_eligible
                or xyz_error_m is None
                or xyz_error_m > 0.005
            )
        )
        catastrophic = bool(
            committed
            and xyz_error_m is not None
            and xyz_error_m > 0.020
        )
        unsafe = bool(
            false_recovery
            or branch["arm_motion_command_count"] != 0
            or branch["gripper_close_command_count"] != 0
        )
        row = {
            "version": E018_P1_STAGE2A_EVALUATION_RESULT_VERSION,
            "seed": seed,
            "selected_gain": STAGE2A_EVALUATION_SELECTED_GAIN,
            "oracle_recoverable_eligible": oracle_eligible,
            "memory_commit_count": branch["memory_commit_count"],
            "navigation_state_available": branch[
                "navigation_state_available"
            ],
            "fresh_shadow_action_generation_count": branch[
                "fresh_shadow_action_generation_count"
            ],
            "xyz_error_m": xyz_error_m,
            "recovered": recovered,
            "false_recovery": false_recovery,
            "catastrophic_recovery": catastrophic,
            "unsafe_recovery": unsafe,
            "protocol_violation_count": branch[
                "protocol_violation_count"
            ],
            "oracle_label_primitive_sha256s": [
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
                for label in labels
            ],
        }
        row["scored_row_sha256"] = canonical_sha256(row)
        scored.append(row)
    support = sum(row["oracle_recoverable_eligible"] for row in scored)
    recovered_count = sum(row["recovered"] for row in scored)
    false_count = sum(row["false_recovery"] for row in scored)
    catastrophic_count = sum(row["catastrophic_recovery"] for row in scored)
    unsafe_count = sum(row["unsafe_recovery"] for row in scored)
    protocol_count = sum(row["protocol_violation_count"] for row in scored)
    recovery_rate = None if support == 0 else recovered_count / support
    if any(
        count != 0
        for count in (
            false_count,
            catastrophic_count,
            unsafe_count,
            protocol_count,
        )
    ):
        classification = (
            "safety-negative-persist-publish-pause-for-reusability-refactor"
        )
    elif support < minimum_support:
        classification = (
            "insufficient-support-inconclusive-persist-publish-pause-for-"
            "reusability-refactor"
        )
    elif 10 * recovered_count >= 7 * support:
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
        "minimum_support_required": minimum_support,
        "recovered_count": recovered_count,
        "recovery_rate": recovery_rate,
        "minimum_recovery_rate_required": minimum_recovery_rate,
        "false_recovery_count": false_count,
        "catastrophic_recovery_count": catastrophic_count,
        "unsafe_recovery_count": unsafe_count,
        "protocol_violation_count": protocol_count,
        "stage2b_continuation_required": False,
        "fresh_test_reads": 0,
        "runtime_object_gt_reads": 0,
        "goal_gt_reads": 0,
        "arm_motion_command_count": 0,
        "gripper_close_command_count": 0,
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    return scored, summary


__all__ = [
    "E018_P1_STAGE2A_EVALUATION_CONFIG_VERSION",
    "E018_P1_STAGE2A_EVALUATION_EXECUTION_VERSION",
    "E018_P1_STAGE2A_EVALUATION_EXPERIMENT_ID",
    "E018_P1_STAGE2A_EVALUATION_RESULT_VERSION",
    "STAGE2A_EVALUATION_GO",
    "STAGE2A_EVALUATION_SEEDS",
    "STAGE2A_EVALUATION_SELECTED_GAIN",
    "CapturedStage2AEvaluationRoute",
    "LoadedStage2AEvaluationConfig",
    "SelectedGainEvaluationBranch",
    "load_e018_p1_stage2a_evaluation_config",
    "replay_selected_gain_branch",
    "score_selected_gain_evaluation",
    "validate_evaluation_private_label",
]
