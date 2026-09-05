from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from robot_vla.cli.run_e018_p1_stage2a_evaluation import build_parser
from robot_vla.precision import (
    e018_p1_stage2a_evaluation_runtime as evaluation_runtime,
)
from robot_vla.precision import (
    e018_p1_stage2a_selection_runtime as selection_runtime,
)
from robot_vla.precision.calibrated_front_provider import canonical_sha256
from robot_vla.precision.e018_p1_stage2a_evaluation import (
    E018_P1_STAGE2A_EVALUATION_EXECUTION_VERSION,
    STAGE2A_EVALUATION_GO,
    STAGE2A_EVALUATION_PREFLIGHT_GO,
    STAGE2A_EVALUATION_SEEDS,
    SelectedGainEvaluationBranch,
    load_e018_p1_stage2a_evaluation_config,
    score_selected_gain_evaluation,
    validate_evaluation_private_label,
)
from robot_vla.precision.object_observability import ObjectObservabilityLabel

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_CONFIG = (
    REPOSITORY_ROOT
    / "configs"
    / "e018_p1_stage2a_selected_gain_evaluation_development_v1.json"
)
D049_GATE = Path(
    "/home/czw/vla-control/gates/e018-p1-stage2a-d049/"
    "D049_CONDITIONAL_EVALUATION_GATE.json"
)


def _private_capture(
    *,
    observable: bool = True,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    contact_n: float = 0.0,
    linear_speed_m_s: float = 0.01,
    angular_speed_rad_s: float = 0.5,
) -> dict[str, object]:
    observability = ObjectObservabilityLabel(
        object_exists=True,
        projection_valid=True,
        in_fov=True,
        observable=observable,
        legacy_visible=True,
        center_inside_object_mask=observable,
        center_inside_goal_mask=False,
        local_object_visible_fraction=1.0 if observable else 0.0,
        object_mask_area_fraction=1.0 / (128 * 128),
        occlusion_type=(
            "observable" if observable else "other_occlusion_or_background"
        ),
    )
    return {
        "gt_object_exists": True,
        "gt_observable": observable,
        "gt_object_position_base_m": list(position),
        "gt_object_projection_valid": True,
        "gt_object_projected_normalized_uv": [0.5, 0.5],
        "gt_object_mask_sha256": "a" * 64,
        "gt_object_visible_pixel_count": 1,
        "gt_object_observability": observability.to_dict(),
        "is_grasped": False,
        "robot_object_contact_force_n": contact_n,
        "goal_gt_read_count": 0,
        "test_data_read": False,
        "object_linear_speed_m_s": linear_speed_m_s,
        "object_angular_speed_rad_s": angular_speed_rad_s,
        "object_motion_event": bool(
            linear_speed_m_s > 0.01 or angular_speed_rad_s > 0.5
        ),
    }


def _private_label(
    label_index: int,
    *,
    seeds: tuple[int, ...],
    observable: bool = True,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    contact_n: float = 0.0,
) -> dict[str, object]:
    seed = seeds[label_index // 3]
    frame = (45, 46, 47)[label_index % 3]
    value = {
        **_private_capture(
            observable=observable,
            position=position,
            contact_n=contact_n,
        ),
        "version": E018_P1_STAGE2A_EVALUATION_EXECUTION_VERSION,
        "label_index": label_index,
        "prediction_row_index": (label_index // 3) * 4 + 1 + label_index % 3,
        "seed": seed,
        "route_frame_index": frame,
        "rgb_sha256": "b" * 64,
        "actual_pose_sha256": "c" * 64,
        "actual_pose_canonical_sha256": "d" * 64,
        "model_input_digest": "e" * 64,
        "provider_output_digest": "f" * 64,
        "prediction_commit_receipt_sha256": "1" * 64,
        "transaction_identity_sha256": "2" * 64,
        "replay_camera_row_sha256": "3" * 64,
        "motion_predicate_version": "pick-and-place-predicates/v1",
        "motion_linear_threshold_m_s": 0.01,
        "motion_angular_threshold_rad_s": 0.5,
        "contact_threshold_n": 0.01,
        "privileged_captured_at_unix_ns": label_index + 1,
    }
    value["label_sha256"] = canonical_sha256(value)
    return value


def _branch(
    seed: int,
    *,
    committed_position: tuple[float, float, float] | None = None,
    protocol_violation_count: int = 0,
) -> dict[str, object]:
    committed = committed_position is not None
    return SelectedGainEvaluationBranch(
        seed=seed,
        gain=0.10,
        route_evidence_digest=hashlib.sha256(f"route-{seed}".encode()).hexdigest(),
        route_protocol_safety_valid=True,
        candidate_commit_eligible=committed,
        memory_commit_count=int(committed),
        navigation_state_available=committed,
        fresh_shadow_action_generation_count=int(committed),
        committed_position_base_m=committed_position,
        provider_forward_count=0,
        arm_motion_command_count=0,
        gripper_close_command_count=0,
        protocol_violation_count=protocol_violation_count,
    ).to_dict()


def _score_case(
    *,
    recovered_count: int,
    support: int = 10,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    seeds = tuple(STAGE2A_EVALUATION_SEEDS[:10])
    branches = [
        _branch(
            seed,
            committed_position=(0.0, 0.0, 0.0)
            if index < recovered_count
            else None,
        )
        for index, seed in enumerate(seeds)
    ]
    labels = [
        _private_label(
            index,
            seeds=seeds,
            observable=(index // 3) < support,
        )
        for index in range(len(seeds) * 3)
    ]
    return score_selected_gain_evaluation(branches, labels, seeds=seeds)


def _source_identity() -> dict[str, str]:
    source = {
        "git_commit": "a" * 40,
        "source_tree_sha256": "b" * 64,
    }
    source["identity_sha256"] = canonical_sha256(source)
    return source


def _formal_go_receipt(
    *,
    execution_id: str = "stage2a-selected-gain-evaluation-formal-abc-77626-77650-20260906-v1",
) -> tuple[object, dict[str, str], str, dict[str, object]]:
    loaded = load_e018_p1_stage2a_evaluation_config(EVALUATION_CONFIG)
    source = _source_identity()
    conditional_parent = "c" * 64
    preflight_mode = evaluation_runtime._evaluation_mode(True)
    formal_mode = evaluation_runtime._evaluation_mode(False)

    def roles(mode: object) -> dict[str, str]:
        return {
            role: evaluation_runtime._evaluation_artifact_role_identity_sha256(
                role=role,
                mode=mode,
                config_canonical_sha256=loaded.canonical_sha256,
                source_identity_sha256=source["identity_sha256"],
                conditional_parent_verification_sha256=conditional_parent,
            )
            for role in ("public_execution", "private_labels", "result")
        }

    preflight_roles = roles(preflight_mode)
    formal_roles = roles(formal_mode)
    worker_root = Path("/root/robot-vla-runs/e018-active-front-reobserve") / execution_id
    receipt: dict[str, object] = {
        "version": "e018-p1-stage2a-d049-final-formal-go-receipt/v1",
        "decision_id": "D049",
        "status": (
            "GO-exactly-once-selected-gain-development-evaluation-"
            "no-test-no-actuation"
        ),
        "authority": "user-authorized-b-level-offline-no-actuation-decision-agent",
        "conditional_gate": {
            "filename": "D049_CONDITIONAL_EVALUATION_GATE.json",
            "raw_sha256": loaded.payload["experiment"]["gate_record_raw_sha256"],
            "status": (
                "implementation-go-formal-hold-until-final-source-r2-and-"
                "preflight"
            ),
        },
        "evaluation_config": {
            "path": (
                "configs/"
                "e018_p1_stage2a_selected_gain_evaluation_development_v1.json"
            ),
            "version": loaded.payload["version"],
            "raw_sha256": loaded.raw_sha256,
            "canonical_sha256": loaded.canonical_sha256,
        },
        "source": {
            **source,
            "github_remote_branch_commit": source["git_commit"],
            "worker_checkout_commit": source["git_commit"],
            "worker_checkout_exact_clean": True,
        },
        "code_validation": {
            "targeted_tests_exit_code": 0,
            "targeted_tests_receipt_sha256": "4" * 64,
            "r2_sample_verification_sha256": "5" * 64,
            "formal_seed_read_count_before_issue": 0,
        },
        "selection_parent": {
            "artifact_id": loaded.payload["selection_parent"]["artifact_id"],
            "replication_state": "REPLICATED",
            "inventory_record_canonical_sha256": loaded.payload[
                "selection_parent"
            ]["persistence"]["inventory_record_canonical_sha256"],
        },
        "preflight": {
            "experiment_id": preflight_mode.experiment_id,
            "seed": 76892,
            "transaction_identity_sha256": "6" * 64,
            "public_output_identity_sha256": preflight_roles["public_execution"],
            "private_output_identity_sha256": preflight_roles["private_labels"],
            "result_output_identity_sha256": preflight_roles["result"],
            "public_verification_sha256": "7" * 64,
            "public_completion_marker_raw_sha256": "8" * 64,
            "public_completion_marker_internal_sha256": "9" * 64,
            "result_verification_sha256": "a" * 64,
            "result_completion_marker_raw_sha256": "b" * 64,
            "result_completion_marker_internal_sha256": "c" * 64,
            "two_pass_verification_passed": True,
            "process_boundary_verified": True,
            "provider_prediction_count": 4,
            "private_label_count": 3,
            "formal_identity_consumed": False,
            "formal_split_consumed": False,
            "fresh_test_reads": 0,
            "physical_camera_actuation": 0,
            "arm_tcp_actuation": 0,
            "gripper_close": 0,
        },
        "formal_execution": {
            "experiment_id": formal_mode.experiment_id,
            "execution_id": execution_id,
            "classification": formal_mode.classification,
            "exact_go_token": formal_mode.go_token,
            "seed_start": formal_mode.seeds[0],
            "seed_end": formal_mode.seeds[-1],
            "seed_count": len(formal_mode.seeds),
            "execution_order": loaded.payload["split"]["execution_order"],
            "capture_attempt_count": 1,
            "scoring_attempt_count": 1,
            "same_identity_rerun_allowed": False,
            "conditional_parent_verification_sha256": conditional_parent,
            "artifact_role_identity_version": (
                "e018-p1-stage2a-evaluation-artifact-role/v1"
            ),
            "worker_artifact_root_identity_version": (
                "e018-p1-stage2a-worker-artifact-root/v1"
            ),
            "worker_artifact_root_identity_sha256": (
                evaluation_runtime._worker_artifact_root_identity_sha256(
                    worker_root
                )
            ),
            "worker_artifact_root_basename": execution_id,
            "public_output_identity_sha256": formal_roles["public_execution"],
            "private_output_identity_sha256": formal_roles["private_labels"],
            "result_output_identity_sha256": formal_roles["result"],
            "formal_output_roots_absent_before_issue": True,
            "formal_consumption_marker_absent_before_issue": True,
        },
        "permissions": {
            "fresh_test_reads": 0,
            "canonical_runtime_mutation": 0,
            "physical_camera_actuation": 0,
            "arm_tcp_actuation": 0,
            "gripper_close": 0,
            "manipulation_progression": 0,
            "checkpoint_writes": 0,
        },
        "continuation": {
            "stage2b_required_for_every_complete_outcome": True,
            "integrity_failure_policy": (
                "freeze-current-identity-preserve-evidence-and-recover-under-"
                "new-experiment-config-and-unused-seed-identity"
            ),
            "substantive_negative_result_policy": (
                "freeze-negative-result-and-continue-stage2b-without-"
                "threshold-or-seed-retuning"
            ),
        },
        "issued_at_unix_ns": 200,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return loaded, source, conditional_parent, receipt


def test_config_is_strict_and_has_frozen_identity(tmp_path: Path) -> None:
    loaded = load_e018_p1_stage2a_evaluation_config(EVALUATION_CONFIG)
    assert loaded.raw_sha256 == (
        "de17c5d4471b47eff2fda9e899fc082a1c854025e673ca481adfadafd69b7358"
    )
    assert loaded.canonical_sha256 == (
        "1f058f95689d9371971f30feb7946a34fa97c9189ec8238896bb7f5c64b1deee"
    )
    assert loaded.payload["selection_parent"]["selection_reason"] == (
        "maximize-integer-recovered-count-then-larger-gain-after-three-way-tie"
    )

    drifted = loaded.payload
    drifted["fixed_rule"]["selected_min_information_gain"] = 0.05
    path = tmp_path / "drifted.json"
    path.write_text(json.dumps(drifted), encoding="utf-8")
    with pytest.raises(ValueError, match="fixed rule"):
        load_e018_p1_stage2a_evaluation_config(path)


def test_gate_verifier_exactly_binds_selection_reason(tmp_path: Path) -> None:
    loaded = load_e018_p1_stage2a_evaluation_config(EVALUATION_CONFIG)
    assert evaluation_runtime._verify_gate_record(
        gate_path=D049_GATE,
        config=loaded.payload,
    )["formal_execution_requires_final_go"] is True

    gate = json.loads(D049_GATE.read_text())
    gate["selection_parent"]["selection_reason"] = (
        "max-recovered-count-then-larger-gain"
    )
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(gate), encoding="utf-8")
    config = loaded.payload
    config["experiment"]["gate_record_raw_sha256"] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    with pytest.raises(RuntimeError, match="Gate identity"):
        evaluation_runtime._verify_gate_record(gate_path=path, config=config)


@pytest.mark.parametrize(
    ("recovered_count", "expected"),
    [
        (7, "development-absolute-recovery-pass-no-effect-no-actuation"),
        (6, "effect-negative-continue-stage2b"),
    ],
)
def test_integer_recovery_gate_boundaries(
    recovered_count: int,
    expected: str,
) -> None:
    _, summary = _score_case(recovered_count=recovered_count)
    assert summary["oracle_recoverable_support"] == 10
    assert summary["recovered_count"] == recovered_count
    assert summary["classification"] == expected
    assert summary["stage2b_continuation_required"] is True


def test_low_support_and_safety_failure_both_continue_stage2b() -> None:
    _, low_support = _score_case(recovered_count=7, support=9)
    assert low_support["classification"] == (
        "insufficient-support-inconclusive-continue-stage2b"
    )

    seeds = tuple(STAGE2A_EVALUATION_SEEDS[:10])
    branches = [_branch(seed) for seed in seeds]
    branches[0] = _branch(seeds[0], committed_position=(0.006, 0.0, 0.0))
    labels = [
        _private_label(index, seeds=seeds) for index in range(len(seeds) * 3)
    ]
    _, safety = score_selected_gain_evaluation(branches, labels, seeds=seeds)
    assert safety["false_recovery_count"] == 1
    assert safety["classification"] == "safety-negative-continue-stage2b"
    assert safety["stage2b_continuation_required"] is True


@pytest.mark.parametrize(
    ("error_m", "recovered", "catastrophic"),
    [(0.005, True, False), (0.020, False, False), (0.0200001, False, True)],
)
def test_xyz_thresholds_are_5mm_inclusive_and_20mm_strict(
    error_m: float,
    recovered: bool,
    catastrophic: bool,
) -> None:
    seeds = (STAGE2A_EVALUATION_SEEDS[0],)
    branches = [_branch(seeds[0], committed_position=(error_m, 0.0, 0.0))]
    labels = [_private_label(index, seeds=seeds) for index in range(3)]
    scored, _ = score_selected_gain_evaluation(branches, labels, seeds=seeds)
    assert scored[0]["recovered"] is recovered
    assert scored[0]["catastrophic_recovery"] is catastrophic


def test_nonfrozen_scoring_parameters_and_label_identity_are_rejected() -> None:
    seeds = (STAGE2A_EVALUATION_SEEDS[0],)
    branches = [_branch(seeds[0])]
    labels = [_private_label(index, seeds=seeds) for index in range(3)]
    with pytest.raises(RuntimeError, match="gate identity"):
        score_selected_gain_evaluation(
            branches,
            labels,
            seeds=seeds,
            minimum_support=9,
        )
    with pytest.raises(RuntimeError, match="gate identity"):
        score_selected_gain_evaluation(
            branches,
            labels,
            seeds=seeds,
            minimum_recovery_rate=0.71,
        )

    drifted = copy.deepcopy(labels[0])
    drifted["model_input_digest"] = "9" * 64
    with pytest.raises(RuntimeError, match="identity/order/hash"):
        validate_evaluation_private_label(
            drifted,
            expected_label_index=0,
            seeds=seeds,
        )


@pytest.mark.parametrize("tamper", ["oracle", "xyz", "primitive_hash"])
def test_private_labels_independently_reject_rehashed_scored_row_tampering(
    tamper: str,
) -> None:
    seeds = (76892,)
    branch = _branch(seeds[0], committed_position=(0.0, 0.0, 0.0))
    labels = [_private_label(index, seeds=seeds) for index in range(3)]
    scored, _ = score_selected_gain_evaluation([branch], labels, seeds=seeds)
    row = copy.deepcopy(scored[0])
    if tamper == "oracle":
        row["oracle_recoverable_eligible"] = False
        row["recovered"] = False
        row["false_recovery"] = True
        row["unsafe_recovery"] = True
    elif tamper == "xyz":
        row["xyz_error_m"] = 0.006
        row["recovered"] = False
        row["false_recovery"] = True
        row["unsafe_recovery"] = True
    else:
        row["oracle_label_primitive_sha256s"][0] = "9" * 64
    row.pop("scored_row_sha256")
    row["scored_row_sha256"] = canonical_sha256(row)

    with pytest.raises(RuntimeError, match="private-label derivation"):
        evaluation_runtime._validate_scored_evaluation_row(
            row,
            row_index=0,
            branch=branch,
            mode=evaluation_runtime._evaluation_mode(True),
            private_labels=labels,
        )


def test_evaluation_role_identities_are_isolated_from_preflight_and_selection() -> None:
    loaded = load_e018_p1_stage2a_evaluation_config(EVALUATION_CONFIG)
    source = "1" * 64
    parent = "2" * 64
    formal = evaluation_runtime._evaluation_artifact_role_identity_sha256(
        role="public_execution",
        mode=evaluation_runtime._evaluation_mode(False),
        config_canonical_sha256=loaded.canonical_sha256,
        source_identity_sha256=source,
        conditional_parent_verification_sha256=parent,
    )
    preflight = evaluation_runtime._evaluation_artifact_role_identity_sha256(
        role="public_execution",
        mode=evaluation_runtime._evaluation_mode(True),
        config_canonical_sha256=loaded.canonical_sha256,
        source_identity_sha256=source,
        conditional_parent_verification_sha256=parent,
    )
    selection = selection_runtime._artifact_role_identity_sha256(
        role="public_execution",
        config_canonical_sha256=loaded.canonical_sha256,
        source_identity_sha256=source,
        parent_verification_sha256=parent,
    )
    assert len({formal, preflight, selection}) == 3


def test_final_go_contract_binds_exact_source_config_preflight_and_roles() -> None:
    loaded, source, parent, receipt = _formal_go_receipt()
    internal = evaluation_runtime._validate_formal_execution_go_receipt(
        receipt,
        loaded=loaded,
        source=source,
        conditional_parent_verification_sha256=parent,
        expected_execution_id=receipt["formal_execution"]["execution_id"],
    )
    assert internal == receipt["receipt_sha256"]

    worker_root = (
        Path("/root/robot-vla-runs/e018-active-front-reobserve")
        / receipt["formal_execution"]["execution_id"]
    )
    assert (
        evaluation_runtime._validate_formal_execution_go_receipt(
            receipt,
            loaded=loaded,
            source=source,
            conditional_parent_verification_sha256=parent,
            expected_execution_id=receipt["formal_execution"]["execution_id"],
            expected_worker_artifact_root=worker_root,
        )
        == receipt["receipt_sha256"]
    )
    with pytest.raises(RuntimeError, match="contract/identity"):
        evaluation_runtime._validate_formal_execution_go_receipt(
            receipt,
            loaded=loaded,
            source=source,
            conditional_parent_verification_sha256=parent,
            expected_execution_id=receipt["formal_execution"]["execution_id"],
            expected_worker_artifact_root=(
                Path("/different-parent") / worker_root.name
            ),
        )

    for path, value in (
        (("source", "source_tree_sha256"), "0" * 64),
        (("preflight", "public_verification_sha256"), "0" * 63),
        (("formal_execution", "public_output_identity_sha256"), "0" * 64),
        (("permissions", "fresh_test_reads"), 1),
    ):
        drifted = copy.deepcopy(receipt)
        drifted[path[0]][path[1]] = value
        drifted.pop("receipt_sha256")
        drifted["receipt_sha256"] = canonical_sha256(drifted)
        with pytest.raises(RuntimeError, match="contract/identity"):
            evaluation_runtime._validate_formal_execution_go_receipt(
                drifted,
                loaded=loaded,
                source=source,
                conditional_parent_verification_sha256=parent,
                expected_execution_id=receipt["formal_execution"]["execution_id"],
            )


def _formal_capture_kwargs(tmp_path: Path) -> dict[str, object]:
    missing = tmp_path / "must-not-be-read"
    return {
        "evaluation_config_path": missing,
        "stage2a_config_path": missing,
        "qualification_config_path": missing,
        "g0c_config_path": missing,
        "data_config_path": missing,
        "selected_checkpoint_path": missing,
        "stats_root": missing,
        "selection_config_path": missing,
        "selection_public_root": missing,
        "selection_result_root": missing,
        "decision_gate_path": missing,
        "artifact_inventory_path": missing,
        "repository_root": missing,
        "artifact_root": missing,
        "expected_config_raw_sha256": "1" * 64,
        "expected_config_canonical_sha256": "2" * 64,
        "expected_source_git_commit": "3" * 40,
        "expected_source_identity_sha256": "4" * 64,
        "exact_go_token": STAGE2A_EVALUATION_GO,
    }


def _formal_score_kwargs(tmp_path: Path) -> dict[str, object]:
    missing = tmp_path / "must-not-be-read"
    return {
        "evaluation_config_path": missing,
        "stage2a_config_path": missing,
        "qualification_config_path": missing,
        "g0c_config_path": missing,
        "data_config_path": missing,
        "stats_root": missing,
        "public_root": missing,
        "private_root": tmp_path / "private-must-not-exist",
        "result_root": tmp_path / "result-must-not-exist",
        "repository_root": missing,
        "expected_source_git_commit": "3" * 40,
        "expected_source_identity_sha256": "4" * 64,
        "exact_go_token": STAGE2A_EVALUATION_GO,
    }


def test_formal_paths_fail_closed_before_any_input_or_artifact_read(
    tmp_path: Path,
) -> None:
    with pytest.raises(PermissionError, match="final-GO"):
        evaluation_runtime.run_e018_p1_stage2a_evaluation_capture(
            **_formal_capture_kwargs(tmp_path)
        )
    with pytest.raises(PermissionError, match="final-GO"):
        evaluation_runtime.run_e018_p1_stage2a_evaluation_score_private(
            **_formal_score_kwargs(tmp_path)
        )
    assert not (tmp_path / "private-must-not-exist").exists()
    assert not (tmp_path / "result-must-not-exist").exists()


def test_pass_a_post_root_initialization_failure_is_permanently_recorded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = load_e018_p1_stage2a_evaluation_config(EVALUATION_CONFIG)
    source = _source_identity()
    artifact_root = tmp_path / "preflight-init-failure"
    conditional_parent = {"verification_sha256": "3" * 64}
    monkeypatch.setattr(
        evaluation_runtime,
        "_git_source_identity",
        lambda _: copy.deepcopy(source),
    )
    monkeypatch.setattr(
        evaluation_runtime,
        "verify_e018_p1_stage2a_evaluation_parent_gate",
        lambda **_: copy.deepcopy(conditional_parent),
    )
    monkeypatch.setattr(
        evaluation_runtime,
        "load_e018_p1_stage2a_config",
        lambda _: {"identity": "stage2a"},
    )
    monkeypatch.setattr(
        evaluation_runtime,
        "load_g2c_dynamic_qualification_config",
        lambda _: {"identity": "qualification"},
    )
    monkeypatch.setattr(
        evaluation_runtime._stage2a._g0c,
        "load_e018_p1_g0c_config",
        lambda _: {"identity": "g0c"},
    )
    monkeypatch.setattr(
        evaluation_runtime._stage2a,
        "load_e018_p1_g2c_data_config",
        lambda *_args, **_kwargs: {"identity": "data"},
    )
    monkeypatch.setattr(
        evaluation_runtime._selection_runtime,
        "_new_process_identity",
        lambda _: {"role": "pass-a-producer", "process_instance_sha256": "4" * 64},
    )

    def fail_journal(**_: object) -> None:
        raise RuntimeError("synthetic post-root journal initialization failure")

    monkeypatch.setattr(
        evaluation_runtime,
        "Stage2AEvaluationJournal",
        fail_journal,
    )
    with pytest.raises(RuntimeError, match="synthetic post-root"):
        evaluation_runtime.run_e018_p1_stage2a_evaluation_capture(
            evaluation_config_path=EVALUATION_CONFIG,
            stage2a_config_path=tmp_path / "stage2a",
            qualification_config_path=tmp_path / "qualification",
            g0c_config_path=tmp_path / "g0c",
            data_config_path=tmp_path / "data",
            selected_checkpoint_path=tmp_path / "checkpoint",
            stats_root=tmp_path / "stats",
            selection_config_path=tmp_path / "selection",
            selection_public_root=tmp_path / "selection-public",
            selection_result_root=tmp_path / "selection-result",
            decision_gate_path=tmp_path / "gate",
            artifact_inventory_path=tmp_path / "inventory",
            repository_root=tmp_path,
            artifact_root=artifact_root,
            expected_config_raw_sha256=loaded.raw_sha256,
            expected_config_canonical_sha256=loaded.canonical_sha256,
            expected_source_git_commit=source["git_commit"],
            expected_source_identity_sha256=source["identity_sha256"],
            exact_go_token=STAGE2A_EVALUATION_PREFLIGHT_GO,
            preflight=True,
        )

    failure = json.loads((artifact_root / "PASS_A_FAILURE.json").read_text())
    assert failure["status"] == "PASS_A_FAILED_IDENTITY_NOT_RERUNNABLE"
    assert failure["private_label_capture_count"] == 0
    assert failure["private_label_open_count"] == 0
    assert failure["fresh_test_reads"] == 0
    assert failure["error_type"] == "RuntimeError"
    assert not (artifact_root / "PUBLIC_EXECUTION_COMPLETE.json").exists()
    assert not (
        artifact_root / "public_execution" / "PUBLIC_EXECUTION_COMPLETE.json"
    ).exists()


def test_preflight_rejects_formal_authority_and_crossed_go_token(
    tmp_path: Path,
) -> None:
    kwargs = _formal_capture_kwargs(tmp_path)
    kwargs.update(
        {
            "exact_go_token": STAGE2A_EVALUATION_PREFLIGHT_GO,
            "preflight": True,
            "formal_go_receipt_path": tmp_path / "receipt",
            "expected_formal_go_raw_sha256": "5" * 64,
            "expected_formal_go_internal_sha256": "6" * 64,
            "preflight_public_root": tmp_path / "public",
            "preflight_result_root": tmp_path / "result",
        }
    )
    with pytest.raises(PermissionError, match="禁止携带"):
        evaluation_runtime.run_e018_p1_stage2a_evaluation_capture(**kwargs)

    crossed = _formal_capture_kwargs(tmp_path)
    crossed["exact_go_token"] = STAGE2A_EVALUATION_PREFLIGHT_GO
    with pytest.raises(PermissionError, match="corresponding|对应"):
        evaluation_runtime.run_e018_p1_stage2a_evaluation_capture(**crossed)


def test_pass_b_rejects_wrong_or_dirty_checkout_before_private_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    public = {
        "source_git_commit": "1" * 40,
        "source_tree_sha256": "2" * 64,
        "source_identity_sha256": "3" * 64,
    }
    monkeypatch.setattr(
        evaluation_runtime,
        "verify_e018_p1_stage2a_evaluation_public",
        lambda **_: public,
    )
    monkeypatch.setattr(
        evaluation_runtime,
        "_git_source_identity",
        lambda _: {
            "git_commit": "4" * 40,
            "source_tree_sha256": "5" * 64,
            "identity_sha256": "6" * 64,
        },
    )
    kwargs = _formal_score_kwargs(tmp_path)
    kwargs.update(
        {
            "exact_go_token": STAGE2A_EVALUATION_PREFLIGHT_GO,
            "preflight": True,
        }
    )
    with pytest.raises(RuntimeError, match="exact-clean source identity"):
        evaluation_runtime.run_e018_p1_stage2a_evaluation_score_private(**kwargs)
    assert not Path(kwargs["private_root"]).exists()

    def reject_dirty(_: Path) -> dict[str, str]:
        raise RuntimeError("G2C formal TRAIN 要求 exact-clean Git worktree")

    monkeypatch.setattr(evaluation_runtime, "_git_source_identity", reject_dirty)
    dirty = dict(kwargs)
    dirty["private_root"] = tmp_path / "dirty-private-must-not-exist"
    dirty["result_root"] = tmp_path / "dirty-result-must-not-exist"
    with pytest.raises(RuntimeError, match="exact-clean Git worktree"):
        evaluation_runtime.run_e018_p1_stage2a_evaluation_score_private(**dirty)
    assert not Path(dirty["private_root"]).exists()


def test_post_consumption_public_ledger_failure_is_permanently_marked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = load_e018_p1_stage2a_evaluation_config(EVALUATION_CONFIG)
    frozen = loaded.payload["stage2a_parent"]
    artifact = tmp_path / "preflight-artifact"
    public_root = artifact / "public_execution"
    private_root = artifact / "private_labels"
    result_root = artifact / "result"
    public_root.mkdir(parents=True)
    producer = {"process_instance_sha256": "8" * 64, "role": "pass-a-producer"}
    scorer = {"process_instance_sha256": "9" * 64, "role": "pass-b-scorer"}
    public = {
        "source_git_commit": "1" * 40,
        "source_tree_sha256": "2" * 64,
        "source_identity_sha256": "3" * 64,
        "producer_process_identity": producer,
        "parent_verification_sha256": "4" * 64,
        "conditional_parent_verification_sha256": "5" * 64,
        "formal_execution_go_verification_sha256": None,
        "transaction_identity_sha256": "6" * 64,
        "verification_sha256": "7" * 64,
        "public_completion_marker_sha256": "a" * 64,
        "private_artifact_role_identity_sha256": "b" * 64,
    }
    monkeypatch.setattr(
        evaluation_runtime,
        "verify_e018_p1_stage2a_evaluation_public",
        lambda **_: copy.deepcopy(public),
    )
    monkeypatch.setattr(
        evaluation_runtime,
        "_git_source_identity",
        lambda _: {
            "git_commit": public["source_git_commit"],
            "source_tree_sha256": public["source_tree_sha256"],
            "identity_sha256": public["source_identity_sha256"],
        },
    )
    monkeypatch.setattr(
        evaluation_runtime,
        "load_e018_p1_stage2a_evaluation_config",
        lambda _: loaded,
    )
    g0c = {"identity": "g0c"}
    data = {"identity": "data"}
    qualification = {
        "config_sha256": frozen["qualification_config_internal_sha256"]
    }
    monkeypatch.setattr(
        evaluation_runtime._stage2a._g0c,
        "load_e018_p1_g0c_config",
        lambda _: g0c,
    )
    monkeypatch.setattr(
        evaluation_runtime._stage2a,
        "load_e018_p1_g2c_data_config",
        lambda *_args, **_kwargs: data,
    )
    monkeypatch.setattr(
        evaluation_runtime,
        "load_g2c_dynamic_qualification_config",
        lambda _: qualification,
    )
    path_hashes = {
        "g0c": frozen["g0c_config_raw_sha256"],
        "data": frozen["data_config_raw_sha256"],
        "qualification": frozen["qualification_config_raw_sha256"],
    }
    monkeypatch.setattr(
        evaluation_runtime,
        "file_sha256",
        lambda path: path_hashes[Path(path).name],
    )
    real_canonical_sha256 = canonical_sha256

    def controlled_canonical_sha256(value: object) -> str:
        if value is g0c:
            return frozen["g0c_config_canonical_sha256"]
        if value is data:
            return frozen["data_config_canonical_sha256"]
        return real_canonical_sha256(value)

    monkeypatch.setattr(
        evaluation_runtime,
        "canonical_sha256",
        controlled_canonical_sha256,
    )
    monkeypatch.setattr(
        evaluation_runtime._selection_runtime,
        "_verify_process_identity",
        lambda *_args, **_kwargs: producer,
    )
    monkeypatch.setattr(
        evaluation_runtime._selection_runtime,
        "_new_process_identity",
        lambda *_args, **_kwargs: scorer,
    )
    monkeypatch.setattr(
        evaluation_runtime,
        "_read_jsonl",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic public ledger corruption")
        ),
    )
    with pytest.raises(RuntimeError, match="synthetic public ledger corruption"):
        evaluation_runtime.run_e018_p1_stage2a_evaluation_score_private(
            evaluation_config_path=EVALUATION_CONFIG,
            stage2a_config_path=tmp_path / "stage2a",
            qualification_config_path=tmp_path / "qualification",
            g0c_config_path=tmp_path / "g0c",
            data_config_path=tmp_path / "data",
            stats_root=tmp_path / "stats",
            public_root=public_root,
            private_root=private_root,
            result_root=result_root,
            repository_root=tmp_path,
            expected_source_git_commit=public["source_git_commit"],
            expected_source_identity_sha256=public["source_identity_sha256"],
            exact_go_token=STAGE2A_EVALUATION_PREFLIGHT_GO,
            preflight=True,
        )
    marker = json.loads((private_root / "SCORING_CONSUMED.json").read_text())
    assert marker["status"] == "PASS_B_FAILED_IDENTITY_PERMANENTLY_CONSUMED"
    assert marker["failure"]["error_type"] == "RuntimeError"
    assert not (result_root / "RESULT_COMPLETE.json").exists()


def test_cli_separates_formal_authority_and_keeps_pass_b_model_free() -> None:
    parser = build_parser()
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )

    def destinations(command: str) -> set[str]:
        return {
            action.dest
            for action in subparsers.choices[command]._actions
            if action.dest != "help"
        }

    formal_capture = destinations("capture-public")
    preflight_capture = destinations("preflight-capture-public")
    formal_score = destinations("score-private")
    result_verifier = destinations("verify-result")
    assert {
        "formal_go_receipt",
        "expected_formal_go_raw_sha256",
        "expected_formal_go_internal_sha256",
        "preflight_public_root",
        "preflight_result_root",
    } <= formal_capture
    assert "formal_go_receipt" not in preflight_capture
    assert {
        "formal_go_receipt",
        "expected_formal_go_raw_sha256",
        "expected_formal_go_internal_sha256",
        "repository_root",
    } <= formal_score
    assert "selected_checkpoint" not in formal_score
    assert "model" not in formal_score
    assert "private_root" not in result_verifier
    assert "stats_root" not in result_verifier


def test_evaluation_journal_enforces_four_prediction_order_and_no_private_api(
    tmp_path: Path,
) -> None:
    mode = evaluation_runtime._evaluation_mode(True)
    journal = evaluation_runtime.Stage2AEvaluationJournal(
        public_root=tmp_path / "public",
        config_canonical_sha256="1" * 64,
        transaction_identity_sha256="2" * 64,
        mode=mode,
    )
    with pytest.raises(RuntimeError, match="exact order"):
        journal.commit_prediction(
            {
                "provider_output_digest": "3" * 64,
                "model_input_digest": "4" * 64,
            },
            seed=STAGE2A_EVALUATION_SEEDS[0],
            route_frame_index=0,
            provider_output_digest="3" * 64,
            model_input_digest="4" * 64,
        )
    for index, frame in enumerate((0, 45, 46, 47)):
        provider_digest = f"{index + 3:x}" * 64
        input_digest = f"{index + 7:x}" * 64
        journal.commit_prediction(
            {
                "provider_output_digest": provider_digest,
                "model_input_digest": input_digest,
            },
            seed=76892,
            route_frame_index=frame,
            provider_output_digest=provider_digest,
            model_input_digest=input_digest,
        )
    assert journal.freeze()["row_count"] == 4
    assert not hasattr(journal, "private_root")
    assert not hasattr(journal, "capture_private_label")
    with pytest.raises(FileExistsError, match="必须全新"):
        evaluation_runtime.Stage2AEvaluationJournal(
            public_root=tmp_path / "public",
            config_canonical_sha256="1" * 64,
            transaction_identity_sha256="2" * 64,
            mode=mode,
        )


def _camera_row(rgb: np.ndarray) -> dict[str, object]:
    pose = np.eye(4, dtype=np.float64).tolist()
    return {
        "episode_id": "episode",
        "request_id": "request",
        "camera_command_sequence_id": "command",
        "frame_index": 0,
        "control_tick": 1,
        "timestamp_s": 0.0,
        "camera_motion_state": "HOME_ANCHOR",
        "viewpoint_primitive_id": "HOME__CENTER",
        "target_orientation_id": "CENTER",
        "orientation_progress": 1.0,
        "commanded_yaw_offset_rad": 0.0,
        "commanded_pitch_offset_rad": 0.0,
        "commanded_roll_offset_rad": 0.0,
        "arm_owner": "SafeHold",
        "gripper_owner": "SafeHoldOpen",
        "external_camera_owner": "ActiveFrontReobserveController",
        "arm_motion_command_max_abs": 0.0,
        "gripper_hold_open_command": True,
        "commanded_external_position_world_m": [0.0, 0.0, 0.0],
        "commanded_external_quaternion_sapien": [1.0, 0.0, 0.0, 0.0],
        "commanded_world_from_external_camera_gl": pose,
        "commanded_base_from_external_camera_cv": pose,
        "actual_base_from_external_camera_cv": pose,
        "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
    }


def test_replay_binding_rejects_action_rgb_and_pose_drift() -> None:
    rgb = np.zeros((128, 128, 3), dtype=np.uint8)
    public = _camera_row(rgb)
    expected_action = evaluation_runtime._action_prefix_sha256s([public])[-1]
    binding = evaluation_runtime._verify_replay_frame_binding(
        replay_row=copy.deepcopy(public),
        public_row=public,
        replay_prefix_rows=[copy.deepcopy(public)],
        expected_action_prefix_sha256=expected_action,
        rgb=rgb,
    )
    assert binding["rgb_sha256"] == public["rgb_sha256"]

    with pytest.raises(RuntimeError, match="replay 漂移"):
        evaluation_runtime._verify_replay_frame_binding(
            replay_row=copy.deepcopy(public),
            public_row=public,
            replay_prefix_rows=[copy.deepcopy(public)],
            expected_action_prefix_sha256="0" * 64,
            rgb=rgb,
        )
    changed_rgb = rgb.copy()
    changed_rgb[0, 0, 0] = 1
    with pytest.raises(RuntimeError, match="replay 漂移"):
        evaluation_runtime._verify_replay_frame_binding(
            replay_row=copy.deepcopy(public),
            public_row=public,
            replay_prefix_rows=[copy.deepcopy(public)],
            expected_action_prefix_sha256=expected_action,
            rgb=changed_rgb,
        )
    changed_pose = copy.deepcopy(public)
    changed_pose["actual_base_from_external_camera_cv"][0][3] = 0.001
    with pytest.raises(RuntimeError, match="replay 漂移"):
        evaluation_runtime._verify_replay_frame_binding(
            replay_row=changed_pose,
            public_row=public,
            replay_prefix_rows=[changed_pose],
            expected_action_prefix_sha256=expected_action,
            rgb=rgb,
        )


def test_result_verifier_has_no_private_model_checkpoint_or_stats_input() -> None:
    parameters = inspect.signature(
        evaluation_runtime.verify_e018_p1_stage2a_evaluation_result
    ).parameters
    assert "private_root" not in parameters
    assert "model" not in parameters
    assert "selected_checkpoint_path" not in parameters
    assert "stats_root" not in parameters
