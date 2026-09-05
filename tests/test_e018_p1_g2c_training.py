from __future__ import annotations

import copy
import inspect
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import robot_vla.cli.freeze_e018_p1_g2c_model_val as freeze_cli
import robot_vla.precision.e018_p1_g2c_model_val as model_val
import robot_vla.precision.e018_p1_g2c_training as training
from robot_vla.precision.e018_p1_g2a import canonical_sha256, file_sha256
from robot_vla.precision.e018_p1_g2c import (
    G2C_CANDIDATE_EPOCHS,
    G2C_CANDIDATE_IDS,
    select_g2c_checkpoint,
)
from robot_vla.precision.e018_p1_g2c_data import (
    G2C_STATIC_SPLITS,
    G2C_VIEW_ORDER,
    _atomic_json,
    _atomic_jsonl,
)
from robot_vla.precision.e018_p1_g2c_training import (
    D038_ACCEPTED_DATA,
    E018_P1_G2C_FORMAL_TRAIN_CONFIG_VERSION,
    E018_P1_G2C_SELECTION_RESULT_VERSION,
    G2C_DIAGNOSTIC_CONTROL_ID,
    _state_identity_sha256,
    _train_one_formal_candidate,
    _validate_resume_progress_semantics,
    _validate_zero_epoch_restart_candidate,
    _verify_exact_regular_file_tree,
    g2c_formal_training_protocol,
    load_g2c_formal_training_config,
    remaining_g2c_active_gpu_budget_seconds,
    run_g2c_formal_training,
    validate_g2c_input_view,
)


def _sha(character: str) -> str:
    return character * 64


def _formal_config() -> dict[str, object]:
    split_inventory = {
        split: {
            "split": split,
            "seed_count": {"train": 400, "model_val": 100, "calibration": 50}[
                split
            ],
            "sample_count": {
                "train": 4400,
                "model_val": 1100,
                "calibration": 550,
            }[split],
            "deployable_inventory_sha256": _sha("1"),
            "privileged_inventory_sha256": _sha("2"),
            "paired_inventory_sha256": _sha("3"),
        }
        for split in G2C_STATIC_SPLITS
    }
    payload: dict[str, object] = {
        "version": E018_P1_G2C_FORMAL_TRAIN_CONFIG_VERSION,
        "status": "frozen-pre-formal-train-awaiting-source-r2-go/v1",
        "decision": {
            "data_acceptance": "D038",
            "formal_train_execution": "HOLD-until-new-source-r2-go",
            "model_val_execution": "HOLD-until-separate-go",
        },
        "data_parent": dict(D038_ACCEPTED_DATA),
        "input_inventories": {
            "all_inventory_sha256": canonical_sha256(
                [
                    {"split": split, **split_inventory[split]}
                    for split in G2C_STATIC_SPLITS
                ]
            ),
            "total_seed_count": 550,
            "total_sample_count": 6050,
            "splits": split_inventory,
        },
        "model_parent": {
            "e016_config_sha256": _sha("4"),
            "e016_checkpoint_sha256": _sha("5"),
            "e016_checkpoint_parameter_sha256": _sha("6"),
            "e016_checkpoint_provenance_sha256": _sha("7"),
            "e016_checkpoint_model_config_sha256": _sha("8"),
            "source_training_camera": "wrist",
            "target_training_camera": "external/front",
        },
        "protocol": g2c_formal_training_protocol(),
        "permissions": {
            "test_array_reads": 0,
            "memory_reads": 0,
            "memory_writes": 0,
            "runtime_camera_actuation": 0,
            "physical_camera_actuation": 0,
            "arm_motion_commands": 0,
            "gripper_close_commands": 0,
            "manipulation_progression": 0,
        },
    }
    payload["config_sha256"] = canonical_sha256(payload)
    return payload


def _resign_config(config: dict[str, object]) -> None:
    config.pop("config_sha256", None)
    config["config_sha256"] = canonical_sha256(config)


def test_protocol_freezes_diagnostic_control_outside_selection_pool() -> None:
    protocol = g2c_formal_training_protocol()
    validation = protocol["model_validation"]

    assert protocol["candidate_ids"] == list(G2C_CANDIDATE_IDS)
    assert protocol["checkpoint_epochs"] == list(G2C_CANDIDATE_EPOCHS)
    assert validation["selection_checkpoint_count"] == 8
    assert validation["candidate_prediction_row_count"] == 8800
    assert validation["candidate_loss_output_shard_count"] == 280
    assert validation["diagnostic_control"] == {
        "control_id": G2C_DIAGNOSTIC_CONTROL_ID,
        "source": "exact-e016-selected-epoch12-role-substitution/v1",
        "prediction_ledger_count": 1,
        "prediction_row_count": 1100,
        "loss_output_shard_count": 0,
        "validation_loss_count": 0,
        "eligible_for_selection": False,
    }
    assert validation["total_prediction_row_count"] == 9900
    assert validation["model_val_deployable_bundle_open_count"] == 900


def test_phase_a_and_phase_b_count_contracts_reject_legacy_totals() -> None:
    freeze_counts = {
        "selection_checkpoint_count": 8,
        "candidate_prediction_ledger_count": 8,
        "candidate_prediction_row_count": 8800,
        "candidate_loss_output_shard_count": 280,
        "diagnostic_control_prediction_ledger_count": 1,
        "diagnostic_control_prediction_row_count": 1100,
        "diagnostic_control_loss_output_shard_count": 0,
        "total_prediction_ledger_count": 9,
        "total_prediction_row_count": 9900,
        "model_val_unique_deployable_bundle_count": 100,
        "model_val_deployable_bundle_open_count": 900,
        "model_val_deployable_sample_read_count": 9900,
        "privileged_label_open_count_before_freeze": 0,
    }
    model_val._assert_prediction_freeze_count_contract(freeze_counts)
    legacy_freeze = dict(freeze_counts)
    legacy_freeze["total_prediction_row_count"] = 8800
    with pytest.raises(RuntimeError, match="count contract"):
        model_val._assert_prediction_freeze_count_contract(legacy_freeze)

    selection_counts = {
        "selection_checkpoint_count": 8,
        "candidate_prediction_row_count": 8800,
        "candidate_scoring_row_count": 8800,
        "candidate_loss_output_shard_count": 280,
        "validation_loss_count": 8,
        "diagnostic_control_prediction_row_count": 1100,
        "diagnostic_control_scoring_row_count": 1100,
        "diagnostic_control_validation_loss_count": 0,
        "total_prediction_row_count": 9900,
        "total_scoring_row_count": 9900,
        "model_val_privileged_label_bundle_open_count": 100,
    }
    model_val._assert_selection_count_contract(selection_counts)
    legacy_selection = dict(selection_counts)
    legacy_selection.pop("diagnostic_control_scoring_row_count")
    with pytest.raises(RuntimeError, match="count contract"):
        model_val._assert_selection_count_contract(legacy_selection)


def test_checkpoint_selection_accepts_finite_negative_validation_loss() -> None:
    rows = [
        {
            "candidate_id": candidate_id,
            "epoch": epoch,
            "viewpoint_id": viewpoint_id,
            "gt_observable": True,
            "predicted_observable": True,
            "geometry_valid": True,
            "world_xyz_error_m": 0.001,
        }
        for candidate_id in G2C_CANDIDATE_IDS
        for epoch in G2C_CANDIDATE_EPOCHS
        for viewpoint_id in G2C_VIEW_ORDER
        for _ in range(30)
    ]
    losses = {
        (candidate_id, epoch): -0.25
        for candidate_id in G2C_CANDIDATE_IDS
        for epoch in G2C_CANDIDATE_EPOCHS
    }
    losses[("S", 10)] = -1.25

    selection = select_g2c_checkpoint(rows, validation_losses=losses)

    assert selection["status"] == "complete-model-val-pass"
    assert selection["selected"]["candidate_id"] == "S"
    assert selection["selected"]["epoch"] == 10


def test_freeze_cli_requires_and_parses_e016_control_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "freeze-e018",
            "freeze",
            "--config",
            "config.json",
            "--training-output",
            "training",
            "--e016-training-output",
            "e016-training",
            "--model-val-deployable-input",
            "model-val",
            "--repository-root",
            "repo",
            "--output",
            "freeze",
        ],
    )

    args = freeze_cli._parse_args()

    assert args.phase == "freeze"
    assert args.e016_training_output == Path("e016-training")


def test_config_survives_sorted_json_write_read_without_dict_order_dependency(
    tmp_path: Path,
) -> None:
    config = _formal_config()
    path = tmp_path / "config.json"
    _atomic_json(path, config)

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert list(raw["input_inventories"]["splits"]) == sorted(G2C_STATIC_SPLITS)
    assert load_g2c_formal_training_config(path) == raw


@pytest.mark.parametrize(
    "case",
    [
        "permission-missing",
        "permission-extra",
        "permission-bool",
        "model-parent-missing",
        "model-parent-extra",
        "model-parent-bad-sha",
        "model-parent-camera-swap",
        "inventories-missing",
        "inventories-extra",
        "split-item-missing",
        "split-item-extra",
    ],
)
def test_config_loader_rejects_nested_schema_drift(
    tmp_path: Path, case: str
) -> None:
    config = _formal_config()
    permissions = config["permissions"]
    model_parent = config["model_parent"]
    inventories = config["input_inventories"]
    assert isinstance(permissions, dict)
    assert isinstance(model_parent, dict)
    assert isinstance(inventories, dict)
    splits = inventories["splits"]
    assert isinstance(splits, dict)
    train_inventory = splits["train"]
    assert isinstance(train_inventory, dict)

    if case == "permission-missing":
        permissions.pop("test_array_reads")
    elif case == "permission-extra":
        permissions["model_val_label_reads"] = 0
    elif case == "permission-bool":
        permissions["test_array_reads"] = False
    elif case == "model-parent-missing":
        model_parent.pop("e016_checkpoint_sha256")
    elif case == "model-parent-extra":
        model_parent["unfrozen_backbone"] = False
    elif case == "model-parent-bad-sha":
        model_parent["e016_checkpoint_sha256"] = "not-a-sha"
    elif case == "model-parent-camera-swap":
        model_parent["source_training_camera"] = "external/front"
    elif case == "inventories-missing":
        inventories.pop("total_sample_count")
    elif case == "inventories-extra":
        inventories["test_sample_count"] = 0
    elif case == "split-item-missing":
        train_inventory.pop("paired_inventory_sha256")
    elif case == "split-item-extra":
        train_inventory["test_inventory_sha256"] = _sha("a")
    else:
        raise AssertionError(case)

    _resign_config(config)
    path = tmp_path / f"{case}.json"
    _atomic_json(path, config)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        load_g2c_formal_training_config(path)


def test_formal_interfaces_keep_phase_boundaries_and_control_parent() -> None:
    freeze_parameters = inspect.signature(
        model_val.run_g2c_model_val_prediction_freeze
    ).parameters
    score_parameters = inspect.signature(model_val.score_select_g2c_model_val).parameters
    verify_parameters = inspect.signature(
        model_val.verify_g2c_model_val_selection
    ).parameters

    assert "e016_training_output" in freeze_parameters
    assert "model_val_deployable_input_root" in freeze_parameters
    assert "model" not in score_parameters
    assert "checkpoint" not in score_parameters
    assert "model_val_deployable_input_root" not in score_parameters
    assert "model_val_label_input_root" not in verify_parameters


def test_decision_hold_is_checked_before_any_input_read(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("HOLD 后不得读取输入")

    monkeypatch.setattr(model_val, "load_g2c_formal_training_config", forbidden)
    with pytest.raises(PermissionError, match="HOLD"):
        model_val.run_g2c_model_val_prediction_freeze(
            config_path="missing",
            training_output_root="missing",
            e016_training_output="missing",
            model_val_deployable_input_root="missing",
            repository_root="missing",
            output_root="missing",
            decision_exit_go=False,
        )
    with pytest.raises(PermissionError, match="HOLD"):
        model_val.score_select_g2c_model_val(
            config_path="missing",
            prediction_freeze_root="missing",
            model_val_label_input_root="missing",
            repository_root="missing",
            output_root="missing",
            decision_exit_go=False,
        )
    with pytest.raises(PermissionError, match="HOLD"):
        run_g2c_formal_training(
            config_path="missing",
            train_input_root="missing",
            e016_config_path="missing",
            e016_training_output="missing",
            repository_root="missing",
            output_root="missing",
            decision_exit_go=False,
        )


def test_exact_tree_rejects_extra_file_directory_and_symlink(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    (root / "nested").mkdir(parents=True)
    (root / "nested" / "value.json").write_text("{}\n", encoding="utf-8")
    expected = {"nested/value.json"}
    assert (
        _verify_exact_regular_file_tree(root, expected_files=expected, name="test")
        == 3
    )

    (root / "extra.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="白名单"):
        _verify_exact_regular_file_tree(root, expected_files=expected, name="test")
    (root / "extra.json").unlink()
    (root / "extra-dir").mkdir()
    with pytest.raises(RuntimeError, match="白名单"):
        _verify_exact_regular_file_tree(root, expected_files=expected, name="test")
    (root / "extra-dir").rmdir()
    (root / "link.json").symlink_to(root / "nested" / "value.json")
    with pytest.raises(RuntimeError, match="symlink"):
        _verify_exact_regular_file_tree(root, expected_files=expected, name="test")


def test_input_view_validation_rejects_hardlinked_receipt_before_read(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.json"
    _atomic_json(config_path, _formal_config())
    canonical_receipt = tmp_path / "canonical-receipt.json"
    canonical_receipt.write_text("{}\n", encoding="utf-8")
    view = tmp_path / "view"
    view.mkdir()
    os.link(canonical_receipt, view / "input_view_receipt.json")

    with pytest.raises(RuntimeError, match="hardlink"):
        validate_g2c_input_view(
            config_path=config_path,
            input_root=view,
            expected_role="train-paired",
        )


def test_state_identity_preserves_mapping_key_types() -> None:
    assert _state_identity_sha256({1: "value"}) != _state_identity_sha256(
        {"1": "value"}
    )
    assert _state_identity_sha256({True: "value"}) != _state_identity_sha256(
        {1: "value"}
    )


def test_active_gpu_budget_uses_persisted_high_water_mark() -> None:
    consumed, remaining = remaining_g2c_active_gpu_budget_seconds(
        budget_hours=10.0, persisted_active_elapsed_s=[]
    )
    assert consumed == 0.0
    assert remaining == 36000.0

    consumed, remaining = remaining_g2c_active_gpu_budget_seconds(
        budget_hours=10.0,
        persisted_active_elapsed_s=[120.0, 35999.0, 240.0],
    )
    assert consumed == 35999.0
    assert remaining == 1.0
    with pytest.raises(TimeoutError, match="耗尽"):
        remaining_g2c_active_gpu_budget_seconds(
            budget_hours=10.0, persisted_active_elapsed_s=[36000.0]
        )


def test_zero_epoch_restart_preserves_budget_and_rejects_unknown_state(
    tmp_path: Path,
) -> None:
    config = _formal_config()
    source_identity = {"identity_sha256": _sha("d")}
    candidate_root = tmp_path / "W-KV0"
    candidate_root.mkdir()
    budget = {
        "version": "e018-p1-g2c-train-result/v1",
        "candidate_id": "W-KV0",
        "config_sha256": config["config_sha256"],
        "source_identity_sha256": source_identity["identity_sha256"],
        "last_fully_resumable_epoch": 0,
        "active_attempt_epoch": 1,
        "active_attempt_batch_count": 17,
        "active_gpu_elapsed_s": 123.0,
        "gpu_budget_hours": 10.0,
    }
    _atomic_json(candidate_root / "budget_state.json", budget)

    assert (
        _validate_zero_epoch_restart_candidate(
            candidate_root=candidate_root,
            candidate_id="W-KV0",
            config=config,
            source_identity=source_identity,
        )
        is None
    )

    initialization = {"candidate_id": "W-KV0", "identity": _sha("a")}
    _atomic_json(candidate_root / "initialization.json", initialization)
    assert _validate_zero_epoch_restart_candidate(
        candidate_root=candidate_root,
        candidate_id="W-KV0",
        config=config,
        source_identity=source_identity,
    ) == json.loads(
        (candidate_root / "initialization.json").read_text(encoding="utf-8")
    )

    (candidate_root / "unexpected.pt").write_bytes(b"not-allowed")
    with pytest.raises(RuntimeError, match="白名单"):
        _validate_zero_epoch_restart_candidate(
            candidate_root=candidate_root,
            candidate_id="W-KV0",
            config=config,
            source_identity=source_identity,
        )


def _resume_trace_row(epoch: int = 1) -> dict[str, object]:
    return {
        "candidate_id": "W-KV0",
        "epoch": epoch,
        "sample_count": 4400,
        "batch_count": 138,
        "sample_order_sha256": _sha("1"),
        "sampler_generator_state_before_sha256": _sha("2"),
        "sampler_generator_state_after_sha256": _sha("3"),
        "loss": {
            "loss": -0.1,
            "heatmap_loss": 0.1,
            "mask_loss": 0.1,
            "coordinate_loss": 0.1,
            "motion_loss": 0.0,
            "uncertainty_loss": -0.2,
            "visibility_loss": 0.1,
            "projection_loss": 0.1,
        },
        "maximum_gradient_norm_pre_clip": 1.2,
        "maximum_gradient_norm_post_clip": 1.0,
        "learning_rate_after_scheduler_step": 2.9e-4,
        "examples_seen_total": epoch * 4400,
        "optimizer_steps_total": epoch * 138,
        "parameter_state_sha256": _sha("4"),
        "motion_head_parameter_sha256": _sha("5"),
        "optimizer_state_identity_sha256": _sha("6"),
        "scheduler_state_identity_sha256": _sha("7"),
        "rng_state_identity_sha256": _sha("8"),
    }


def _resume_progress_state(config: dict[str, object]) -> dict[str, object]:
    return {
        "version": "e018-p1-g2c-train-result/v1",
        "candidate_id": "W-KV0",
        "config_sha256": config["config_sha256"],
        "source_identity_sha256": _sha("d"),
        "completed_epoch": 1,
        "examples_seen": 4400,
        "optimizer_steps": 138,
        "model_state": {},
        "optimizer_state": {},
        "scheduler_state": {},
        "rng_state": {},
        "initialization": {"candidate_id": "W-KV0"},
        "epoch_trace": [_resume_trace_row()],
        "checkpoint_inventory": [],
        "active_gpu_elapsed_s": 12.0,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("examples_seen", 4399),
        ("optimizer_steps", 137),
        ("completed_epoch", 2),
    ],
)
def test_resume_progress_rejects_count_or_trace_length_drift(
    field: str, value: int
) -> None:
    config = _formal_config()
    source_identity = {"identity_sha256": _sha("d")}
    state = _resume_progress_state(config)
    _validate_resume_progress_semantics(
        state=state,
        candidate_id="W-KV0",
        config=config,
        source_identity=source_identity,
    )
    corrupted = copy.deepcopy(state)
    corrupted[field] = value

    with pytest.raises(RuntimeError, match="resume"):
        _validate_resume_progress_semantics(
            state=corrupted,
            candidate_id="W-KV0",
            config=config,
            source_identity=source_identity,
        )


def test_corrupt_resume_is_rejected_before_model_load_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    config = _formal_config()
    source_identity = {
        "git_commit": "commit",
        "source_tree_sha256": _sha("c"),
        "identity_sha256": _sha("d"),
    }
    candidate_root = tmp_path / "W-KV0"
    candidate_root.mkdir()
    state = _resume_progress_state(config)
    state["examples_seen"] = 4399
    torch.save(state, candidate_root / "resume_state.pt")
    _atomic_json(
        candidate_root / "initialization.json", state["initialization"]
    )
    _atomic_json(candidate_root / "epoch_trace.json", state["epoch_trace"])
    _atomic_json(
        candidate_root / "checkpoint_inventory.json",
        state["checkpoint_inventory"],
    )
    _atomic_json(
        candidate_root / "budget_state.json",
        {
            "version": "e018-p1-g2c-train-result/v1",
            "candidate_id": "W-KV0",
            "config_sha256": config["config_sha256"],
            "source_identity_sha256": source_identity["identity_sha256"],
            "last_fully_resumable_epoch": 1,
            "active_attempt_epoch": 2,
            "active_attempt_batch_count": 3,
            "active_gpu_elapsed_s": 13.0,
            "gpu_budget_hours": 10.0,
        },
    )
    before = {
        path.name: file_sha256(path)
        for path in candidate_root.iterdir()
        if path.is_file()
    }
    calls = {"model_load": 0}

    def forbidden_model_load(**_: object) -> object:
        calls["model_load"] += 1
        raise AssertionError("corrupt resume 后不得加载 model")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(training, "_load_formal_candidate_model", forbidden_model_load)

    with pytest.raises(RuntimeError, match="identity/count"):
        _train_one_formal_candidate(
            candidate_id="W-KV0",
            samples=[{}] * 4400,
            config=config,
            source_identity=source_identity,
            e016_config_path=tmp_path / "e016.json",
            e016_training_output=tmp_path / "e016",
            candidate_root=candidate_root,
            resume=True,
            deadline_monotonic_s=float("inf"),
            active_gpu_elapsed_s_before_process=13.0,
            active_process_started_monotonic_s=0.0,
        )
    after = {
        path.name: file_sha256(path)
        for path in candidate_root.iterdir()
        if path.is_file()
    }
    assert calls["model_load"] == 0
    assert after == before


def test_outer_runner_does_not_rewrite_budget_on_resume_preflight_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _formal_config()
    source_identity = {
        "git_commit": "commit",
        "source_tree_sha256": _sha("c"),
        "identity_sha256": _sha("d"),
    }
    input_verification = {"verified": True}
    output = tmp_path / "formal-output"
    candidate_root = output / "candidates" / "W-KV0"
    candidate_root.mkdir(parents=True)
    _atomic_json(output / "config_snapshot.json", config)
    _atomic_json(output / "source_identity.json", source_identity)
    _atomic_json(output / "train_input_verification.json", input_verification)
    _atomic_json(
        output / "run_state.json",
        {"status": "formal-training-in-progress"},
    )
    budget_path = candidate_root / "budget_state.json"
    _atomic_json(
        budget_path,
        {
            "version": "e018-p1-g2c-train-result/v1",
            "candidate_id": "W-KV0",
            "config_sha256": config["config_sha256"],
            "source_identity_sha256": source_identity["identity_sha256"],
            "last_fully_resumable_epoch": 1,
            "active_attempt_epoch": 2,
            "active_attempt_batch_count": 3,
            "active_gpu_elapsed_s": 13.0,
            "gpu_budget_hours": 10.0,
        },
    )
    budget_sha_before = file_sha256(budget_path)

    class FakeDataset:
        def __init__(self, *_: object) -> None:
            pass

        def __len__(self) -> int:
            return 4400

        def __getitem__(self, _: int) -> dict[str, object]:
            return {}

    monkeypatch.setattr(training, "load_g2c_formal_training_config", lambda _: config)
    monkeypatch.setattr(training, "validate_g2c_input_view", lambda **_: input_verification)
    monkeypatch.setattr(training, "_git_source_identity", lambda _: source_identity)
    monkeypatch.setattr(training, "G2CFrontTrainingDataset", FakeDataset)

    def reject_preflight(**_: object) -> object:
        raise training._G2CResumePreflightError("preflight drift")

    monkeypatch.setattr(training, "_train_one_formal_candidate", reject_preflight)

    with pytest.raises(RuntimeError, match="preflight drift"):
        run_g2c_formal_training(
            config_path=tmp_path / "config.json",
            train_input_root=tmp_path / "train-input",
            e016_config_path=tmp_path / "e016.json",
            e016_training_output=tmp_path / "e016-output",
            repository_root=tmp_path,
            output_root=output,
            decision_exit_go=True,
            resume=True,
        )
    assert file_sha256(budget_path) == budget_sha_before


def _minimal_prediction_row() -> dict[str, object]:
    return {
        "candidate_id": "W-KV0",
        "epoch": 5,
        "checkpoint_sha256": _sha("a"),
        "seed": 76501,
        "sample_index": 0,
        "viewpoint_id": G2C_VIEW_ORDER[0],
        "predicted_observable": True,
        "geometry_valid": True,
        "predicted_object_position_base_m": [0.1, 0.2, 0.02],
        "raw_covariance_base_m2": np.diag([1e-6, 1e-6, 0.0]).tolist(),
        "write_score": 0.8,
    }


def _minimal_scoring_row() -> dict[str, object]:
    return {
        "version": E018_P1_G2C_SELECTION_RESULT_VERSION,
        "phase": "privileged-score-after-complete-prediction-freeze/v1",
        "prediction_freeze_sha256": _sha("f"),
        "candidate_id": "W-KV0",
        "epoch": 5,
        "checkpoint_sha256": _sha("a"),
        "seed": 76501,
        "sample_index": 0,
        "viewpoint_id": G2C_VIEW_ORDER[0],
        "gt_observable": True,
        "predicted_observable": True,
        "geometry_valid": True,
        "world_xyz_error_m": 0.0,
        "gt_object_position_base_m": [0.1, 0.2, 0.02],
        "predicted_object_position_base_m": [0.1, 0.2, 0.02],
        "raw_covariance_base_m2": np.diag([1e-6, 1e-6, 0.0]).tolist(),
        "write_score": 0.8,
        "used_for_formal_selection": True,
        "test_data_read": False,
    }


@pytest.mark.parametrize(
    ("field", "tampered_value"),
    [
        ("predicted_object_normalized_uv", [0.3, 0.2]),
        ("object_visibility_probability", 0.7),
        ("projection_validity_probability", 0.6),
        ("object_sigma_xy_px", [2.0, 1.0]),
        ("object_mask_probability_at_prediction", 0.4),
        ("write_score", 0.3),
    ],
)
def test_prediction_row_rejects_shard_output_tampering(
    field: str, tampered_value: object
) -> None:
    frozen_output = {
        "object_uv": [0.2, 0.2],
        "goal_uv": [0.8, 0.8],
        "object_visibility_probability": 0.9,
        "goal_visibility_probability": 0.85,
        "projection_validity_probability": 0.95,
        "object_normalized_entropy": 0.1,
        "object_sigma_xy_px": [1.0, 1.0],
        "object_mask_probability_at_prediction": 0.8,
        "goal_mask_probability_at_prediction": 0.75,
        "predicted_observable": True,
        "write_score": 0.7,
    }
    row = {
        "predicted_object_normalized_uv": [0.2, 0.2],
        "predicted_goal_normalized_uv": [0.8, 0.8],
        "object_visibility_probability": 0.9,
        "goal_visibility_probability": 0.85,
        "projection_validity_probability": 0.95,
        "object_normalized_entropy": 0.1,
        "object_sigma_xy_px": [1.0, 1.0],
        "object_mask_probability_at_prediction": 0.8,
        "goal_mask_probability_at_prediction": 0.75,
        "predicted_observable": True,
        "write_score": 0.7,
    }
    model_val._assert_prediction_row_matches_frozen_output(row, frozen_output)
    row[field] = tampered_value

    with pytest.raises(RuntimeError, match="row/shard同源 output"):
        model_val._assert_prediction_row_matches_frozen_output(row, frozen_output)


def test_scoring_verifier_binds_each_row_to_frozen_prediction(tmp_path: Path) -> None:
    ledger = tmp_path / "prediction.jsonl"
    _atomic_jsonl(ledger, [_minimal_prediction_row()])
    inventory = [
        {
            "relative_path": ledger.name,
            "raw_sha256": file_sha256(ledger),
            "row_count": 1,
        }
    ]
    scoring = [_minimal_scoring_row()]

    model_val._verify_scoring_rows_against_frozen_predictions(
        freeze_root=tmp_path,
        prediction_inventory=inventory,
        scoring_rows=scoring,
        freeze_internal_sha256=_sha("f"),
        expected_used_for_selection=True,
    )
    scoring[0]["checkpoint_sha256"] = _sha("b")
    with pytest.raises(RuntimeError, match="逐行绑定"):
        model_val._verify_scoring_rows_against_frozen_predictions(
            freeze_root=tmp_path,
            prediction_inventory=inventory,
            scoring_rows=scoring,
            freeze_internal_sha256=_sha("f"),
            expected_used_for_selection=True,
        )


def test_phase_b_verifier_failure_after_label_consumption_is_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _formal_config()
    freeze_root = tmp_path / "freeze"
    freeze_root.mkdir()
    _atomic_json(
        freeze_root / "prediction_freeze.json",
        {
            "source_git_commit": "commit",
            "source_identity_sha256": _sha("d"),
            "frozen_at_unix_ns": 1,
        },
    )
    output = tmp_path / "selection"
    calls = {"label_load": 0}
    monkeypatch.setattr(
        model_val, "load_g2c_formal_training_config", lambda _: config
    )
    monkeypatch.setattr(
        model_val,
        "verify_g2c_prediction_freeze",
        lambda **_: {
            "freeze_internal_sha256": _sha("f"),
            "freeze_raw_sha256": _sha("e"),
        },
    )
    monkeypatch.setattr(
        model_val,
        "_git_source_identity",
        lambda _: {
            "git_commit": "commit",
            "source_tree_sha256": _sha("c"),
            "identity_sha256": _sha("d"),
        },
    )
    monkeypatch.setattr(model_val, "validate_g2c_input_view", lambda **_: {})

    def load_labels(_: Path) -> dict[tuple[int, int, str], dict[str, object]]:
        calls["label_load"] += 1
        return {}

    monkeypatch.setattr(model_val, "_load_model_val_labels", load_labels)
    monkeypatch.setattr(
        model_val,
        "_score_select_g2c_model_val_consumed",
        lambda **_: {"receipt": {"receipt_sha256": _sha("9")}},
    )
    monkeypatch.setattr(
        model_val,
        "_verify_g2c_model_val_selection",
        lambda **_: (_ for _ in ()).throw(RuntimeError("post-label verifier fail")),
    )

    with pytest.raises(RuntimeError, match="post-label verifier fail"):
        model_val.score_select_g2c_model_val(
            config_path=tmp_path / "config.json",
            prediction_freeze_root=freeze_root,
            model_val_label_input_root=tmp_path / "labels",
            repository_root=tmp_path,
            output_root=output,
            decision_exit_go=True,
        )
    assert calls["label_load"] == 1
    failure = json.loads(
        (output / "consumed_failure.json").read_text(encoding="utf-8")
    )
    phase = json.loads((output / "phase_state.json").read_text(encoding="utf-8"))
    assert failure["label_array_consumed"] is True
    assert failure["rerun_under_same_identity_allowed"] is False
    assert phase["status"] == "consumed-model-val-label-failed-no-rerun/v1"


def test_control_scoring_is_not_passed_to_formal_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _formal_config()
    freeze_root = tmp_path / "freeze"
    output = tmp_path / "output"
    freeze_root.mkdir()
    output.mkdir()
    for name in (
        "config_snapshot.json",
        "source_identity.json",
        "prediction_freeze_verification.json",
    ):
        _atomic_json(output / name, {})
    candidate_inventory = [{"kind": "candidate"}]
    control_inventory = [
        {
            "kind": "control",
            "checkpoint_sha256": _sha("5"),
            "checkpoint_parameter_sha256": _sha("6"),
            "checkpoint_provenance_sha256": _sha("7"),
            "checkpoint_model_config_sha256": _sha("8"),
        }
    ]
    checkpoint_inventory = []

    class DummyPrecisionLossConfig:
        def __init__(self, **values: object) -> None:
            self.values = values

    fake_losses = types.ModuleType("robot_vla.precision.losses")
    fake_losses.PrecisionLossConfig = DummyPrecisionLossConfig
    monkeypatch.setitem(sys.modules, "robot_vla.precision.losses", fake_losses)

    def read_array(path: Path, _: str) -> list[dict[str, object]]:
        if path.name == "prediction_inventory.json":
            return candidate_inventory
        if path.name == "diagnostic_control_prediction_inventory.json":
            return control_inventory
        if path.name == "loss_output_inventory.json":
            return []
        if path.name == "checkpoint_inventory.json":
            return checkpoint_inventory
        raise AssertionError(path)

    monkeypatch.setattr(model_val, "_read_json_array", read_array)
    losses = {
        (candidate, epoch): -0.25
        for candidate in G2C_CANDIDATE_IDS
        for epoch in G2C_CANDIDATE_EPOCHS
    }
    monkeypatch.setattr(
        model_val, "_score_frozen_loss_outputs", lambda **_: (losses, [])
    )

    candidate_rows = [{"row_kind": "candidate"}]
    control_rows = [
        {"row_kind": "control", "viewpoint_id": viewpoint}
        for viewpoint in G2C_VIEW_ORDER
    ]

    def score_rows(**kwargs: object) -> list[dict[str, object]]:
        return (
            candidate_rows
            if kwargs["used_for_formal_selection"] is True
            else control_rows
        )

    monkeypatch.setattr(model_val, "_score_prediction_rows", score_rows)
    monkeypatch.setattr(
        model_val,
        "summarize_g2c_model_val_view",
        lambda rows, *, viewpoint_id: {
            "viewpoint_id": viewpoint_id,
            "eligible": False,
        },
    )

    def select(rows: object, *, validation_losses: object) -> dict[str, object]:
        assert rows is candidate_rows
        assert validation_losses == losses
        return {
            "status": "complete-model-val-protocol-valid-negative",
            "protocol_valid": True,
            "selected": None,
            "candidates": [],
            "reason": "no_non_home_viewpoint_eligible",
        }

    monkeypatch.setattr(model_val, "select_g2c_checkpoint", select)
    result = model_val._score_select_g2c_model_val_consumed(
        config=config,
        freeze_verification={
            "freeze_raw_sha256": _sha("e"),
            "freeze_internal_sha256": _sha("f"),
        },
        freeze_root=freeze_root,
        freeze_marker={"freeze_sha256": _sha("f"), "frozen_at_unix_ns": 1},
        label_verification={},
        labels={},
        output=output,
        source_identity={
            "git_commit": "commit",
            "identity_sha256": _sha("d"),
        },
        label_open_started_at=2,
    )
    assert result["selected"] is None
    assert result["candidate_scoring_row_count"] == 1
    assert result["diagnostic_control_scoring_row_count"] == len(G2C_VIEW_ORDER)
    assert result["diagnostic_control_summary"]["eligible_for_selection"] is False


def test_frozen_decoded_uv_loss_is_exact_no_grad_replay_and_training_stays_live() -> None:
    torch = pytest.importorskip("torch")
    from robot_vla.precision.losses import (
        PrecisionSupervision,
        build_gaussian_heatmaps,
        precision_unet_loss,
    )
    from robot_vla.precision.model import PrecisionUNetOutput

    batch, keypoints, height, width = 2, 2, 8, 8
    heatmap = torch.randn(batch, keypoints, height, width, requires_grad=True)
    offsets = (torch.randn(batch, keypoints, 2, height, width) * 0.1).requires_grad_()
    output = PrecisionUNetOutput(
        heatmap_logits=heatmap,
        mask_logits=torch.randn(batch, 2, height, width, requires_grad=True),
        subpixel_offsets=offsets,
        motion_residual=torch.randn(batch, 4, requires_grad=True) * 1e-3,
        keypoint_log_variance=torch.randn(batch, keypoints, 2, requires_grad=True),
        motion_log_variance=torch.randn(batch, 4, requires_grad=True),
        visibility_logits=torch.randn(batch, keypoints, requires_grad=True),
        projection_validity_logit=torch.randn(batch, requires_grad=True),
    )
    target_uv = torch.tensor(
        [[[0.25, 0.25], [0.75, 0.75]], [[0.3, 0.4], [0.6, 0.7]]],
        dtype=torch.float32,
    )
    valid = torch.ones(batch, keypoints, dtype=torch.bool)
    supervision = PrecisionSupervision(
        heatmap_targets=build_gaussian_heatmaps(
            target_uv, valid, (height, width), sigma_px=1.5
        ),
        mask_targets=torch.zeros(batch, 2, height, width),
        normalized_uv_targets=target_uv,
        keypoint_valid=valid,
        keypoint_observable=valid.clone(),
        motion_residual_targets=torch.zeros(batch, 4),
        motion_valid=torch.ones(batch, 4, dtype=torch.bool),
        projection_valid=torch.ones(batch, dtype=torch.bool),
    )

    training_loss = precision_unet_loss(output, supervision)
    training_loss.loss.backward()
    assert heatmap.grad is not None
    assert offsets.grad is not None

    with torch.no_grad():
        frozen_uv = output.decode_keypoints().normalized_uv.float()
        original = precision_unet_loss(output, supervision)
        replay_output = PrecisionUNetOutput(
            heatmap_logits=output.heatmap_logits,
            mask_logits=output.mask_logits,
            subpixel_offsets=torch.empty(0, dtype=torch.float32),
            motion_residual=output.motion_residual,
            keypoint_log_variance=output.keypoint_log_variance,
            motion_log_variance=output.motion_log_variance,
            visibility_logits=output.visibility_logits,
            projection_validity_logit=output.projection_validity_logit,
        )
        replay = precision_unet_loss(
            replay_output,
            supervision,
            frozen_decoded_normalized_uv=frozen_uv,
        )
        for name in (
            "loss",
            "heatmap_loss",
            "mask_loss",
            "coordinate_loss",
            "motion_loss",
            "uncertainty_loss",
            "visibility_loss",
            "projection_loss",
        ):
            torch.testing.assert_close(
                getattr(original, name), getattr(replay, name), rtol=0.0, atol=0.0
            )

    with pytest.raises(RuntimeError, match="无梯度"):
        precision_unet_loss(
            output,
            supervision,
            frozen_decoded_normalized_uv=frozen_uv.detach(),
        )


@pytest.mark.parametrize(
    "value",
    [
        np.zeros((2, 2), dtype=np.float32),
        np.zeros((2, 2, 2), dtype=np.float64),
        np.full((2, 2, 2), np.nan, dtype=np.float32),
        np.full((2, 2, 2), 1.1, dtype=np.float32),
    ],
)
def test_frozen_decoded_uv_rejects_bad_contract(value: np.ndarray) -> None:
    torch = pytest.importorskip("torch")
    from robot_vla.precision.losses import PrecisionSupervision, precision_unet_loss
    from robot_vla.precision.model import PrecisionUNetOutput

    output = PrecisionUNetOutput(
        heatmap_logits=torch.zeros(2, 2, 4, 4),
        mask_logits=torch.zeros(2, 2, 4, 4),
        subpixel_offsets=torch.empty(0),
        motion_residual=torch.zeros(2, 4),
        keypoint_log_variance=torch.zeros(2, 2, 2),
        motion_log_variance=torch.zeros(2, 4),
        visibility_logits=torch.zeros(2, 2),
        projection_validity_logit=torch.zeros(2),
    )
    supervision = PrecisionSupervision(
        heatmap_targets=torch.ones(2, 2, 4, 4),
        mask_targets=torch.zeros(2, 2, 4, 4),
        normalized_uv_targets=torch.full((2, 2, 2), 0.5),
        keypoint_valid=torch.ones(2, 2, dtype=torch.bool),
        keypoint_observable=torch.ones(2, 2, dtype=torch.bool),
        motion_residual_targets=torch.zeros(2, 4),
        motion_valid=torch.ones(2, 4, dtype=torch.bool),
        projection_valid=torch.ones(2, dtype=torch.bool),
    )
    with torch.no_grad(), pytest.raises((TypeError, ValueError)):
        precision_unet_loss(
            output,
            supervision,
            frozen_decoded_normalized_uv=torch.from_numpy(value),
        )
