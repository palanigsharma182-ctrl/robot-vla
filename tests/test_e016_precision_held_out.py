from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_vla.precision.data import file_sha256
from robot_vla.precision.e016_evaluation import (
    _CALIBRATION_RECEIPT_FILES,
    E016_P1_TEST_CLAIM_VERSION,
    _EvaluationContext,
    _preflight_test_outputs,
    _rules_payload,
    _verify_calibration_package,
    _verify_rules,
    claim_e016_p1_test_once,
)
from robot_vla.precision.e016_training import load_e016_p1_config
from robot_vla.precision.memory_evaluation import GoalWriteCalibration


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_e016_test_once_claim_is_atomic_and_non_reusable(tmp_path: Path) -> None:
    path = tmp_path / "test_evaluation_claim.json"
    values = {
        "rules_sha256": "a" * 64,
        "registry_identity_sha256": "b" * 64,
        "checkpoint_sha256": "c" * 64,
        "source_tree_sha256": "d" * 64,
        "calibration_receipt_sha256": "e" * 64,
    }

    digest = claim_e016_p1_test_once(path, **values)

    assert digest == file_sha256(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {
        "version": E016_P1_TEST_CLAIM_VERSION,
        "status": "claimed-before-test-label-or-model-read",
        **values,
    }
    assert path.stat().st_mode & 0o777 == 0o400
    with pytest.raises(RuntimeError, match="禁止重复"):
        claim_e016_p1_test_once(path, **values)


def test_e016_test_once_claim_requires_existing_parent(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="parent"):
        claim_e016_p1_test_once(
            tmp_path / "missing" / "claim.json",
            rules_sha256="a" * 64,
            registry_identity_sha256="b" * 64,
            checkpoint_sha256="c" * 64,
            source_tree_sha256="d" * 64,
            calibration_receipt_sha256="e" * 64,
        )


def test_e016_output_preflight_rejects_overlap_before_claim(tmp_path: Path) -> None:
    protected = tmp_path / "dataset"
    protected.mkdir()

    with pytest.raises(ValueError, match="private/public"):
        _preflight_test_outputs(
            private=tmp_path / "result",
            public=tmp_path / "result" / "public",
            protected_roots=(protected,),
        )
    with pytest.raises(ValueError, match="只读输入"):
        _preflight_test_outputs(
            private=protected / "private",
            public=tmp_path / "public",
            protected_roots=(protected,),
        )


def test_e016_output_preflight_checks_writable_parents_without_reserving_targets(
    tmp_path: Path,
) -> None:
    private = tmp_path / "outputs" / "private"
    public = tmp_path / "outputs" / "public"

    _preflight_test_outputs(
        private=private,
        public=public,
        protected_roots=(tmp_path / "dataset",),
    )

    assert private.parent.is_dir()
    assert public.parent.is_dir()
    assert not private.exists()
    assert not public.exists()
    assert list(private.parent.glob(".e016-p1-output-preflight.*.tmp")) == []


def test_e016_calibration_package_is_verified_before_test_claim(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    config = load_e016_p1_config(repository / "configs" / "e016_p1_precision_observability_v1.json")
    registry = {
        "registry_identity_sha256": "1" * 64,
        "deployable_manifest_sha256": "2" * 64,
        "label_manifest_sha256": "3" * 64,
    }
    context = _EvaluationContext(
        config=config,
        source_tree_sha256="4" * 64,
        registry_audit=registry,
        checkpoint_sha256="5" * 64,
        checkpoint_parameter_sha256="6" * 64,
        checkpoint_provenance_sha256="7" * 64,
        training_source_tree_sha256="8" * 64,
        training_receipt_sha256="9" * 64,
        training_receipt={},
        model=None,
    )
    calibration = GoalWriteCalibration(
        enabled=True,
        threshold=0.75,
        validation_frame_count=20,
        structurally_eligible_count=10,
        oracle_safe_count=10,
        accepted_count=8,
        accepted_unsafe_count=0,
        safe_coverage=0.8,
    )
    rules = _rules_payload(
        context=context,
        validation_data_identity_sha256="a" * 64,
        calibration=calibration,
        max_age_s=0.25,
    )
    restored_calibration, restored_memory = _verify_rules(rules, context)
    assert restored_calibration == calibration
    assert restored_memory.max_unobserved_age_s == 0.25

    for name in _CALIBRATION_RECEIPT_FILES[:3]:
        (tmp_path / name).write_text("[]\n", encoding="utf-8")
    _write_json(tmp_path / "fresh_registry_audit.json", registry)
    _write_json(tmp_path / "frozen_rules.json", rules)
    _write_json(
        tmp_path / "summary.json",
        {
            "version": "e016-p1-fresh-held-out/v1",
            "phase": "calibrate",
            "status": "complete",
            "split": "val",
            "trajectory_count": 20,
            "registry_identity_sha256": registry["registry_identity_sha256"],
            "validation_data_identity_sha256": "a" * 64,
            "checkpoint_sha256": context.checkpoint_sha256,
            "config_sha256": config.sha256,
            "source_tree_sha256": context.source_tree_sha256,
            "write_calibration": calibration.to_dict(),
            "selected_max_unobserved_age_s": 0.25,
            "test_split_status": "unread",
            "test_privileged_label_file_read_count": 0,
            "test_model_forward_count": 0,
            "actuation_allowed": False,
        },
    )
    _write_json(tmp_path / "config_snapshot.json", config.to_dict())
    _write_json(
        tmp_path / "receipt.json",
        {
            "version": "e016-p1-fresh-held-out/v1",
            "phase": "calibrate",
            "status": "complete",
            "rules_sha256": rules["rules_sha256"],
            "files": {name: file_sha256(tmp_path / name) for name in _CALIBRATION_RECEIPT_FILES},
            "test_split_status": "unread",
        },
    )
    _write_json(
        tmp_path / "run_state.json",
        {
            "version": "e016-p1-fresh-held-out/v1",
            "phase": "calibrate",
            "status": "complete",
            "test_split_status": "unread",
        },
    )

    assert _verify_calibration_package(
        root=tmp_path,
        rules=rules,
        context=context,
    ) == file_sha256(tmp_path / "receipt.json")

    (tmp_path / "prediction_rows.jsonl").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="artifact SHA-256"):
        _verify_calibration_package(root=tmp_path, rules=rules, context=context)
