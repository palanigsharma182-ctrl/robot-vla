import hashlib
import json
import os
import sys
import tempfile
from argparse import Namespace
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import robot_vla.cli.collect_local_dagger_amended_pool as pool_module
from robot_vla.cli.collect_local_dagger_amended_pool import (
    AMENDED_ACTION_BUDGET_PROTOCOL,
    AMENDED_FORMAL_SEED_END_LIMIT,
    AMENDED_FORMAL_SEED_START,
    CANDIDATE_DATASET_STAGING_MARKER,
    CANONICAL_SELECTED_RECORD_FORMAT,
    ELIGIBLE_SELECTION_GATE,
    HIGH_RISK_SELECTION_COUNT,
    LOW_RISK_SELECTION_COUNT,
    PAIRED_CLEAN_EXPERT_PROTOCOL,
    _candidate_command,
    _expected_candidate_config,
    _load_existing_candidates,
    _load_record,
    _parse_args,
    _validate_resume_derived_artifacts,
    build_canonical_selected_records,
    build_pool_identity,
    compact_candidate_record,
)
from robot_vla.contracts import OUTCOME_PREDICATE_VERSION, RobotSpec
from robot_vla.data.trajectory import LocalDaggerProvenance, OutcomeEvidence
from robot_vla.data.writer import TrajectoryDatasetWriter
from robot_vla.local_dagger_protocol import (
    LOCAL_DAGGER_ACTION_BUDGET_PROTOCOL_FIELD,
    LOCAL_DAGGER_ACTION_BUDGET_USAGE_FIELD,
)
from robot_vla.sim.local_dagger_risk import compute_paired_risk_components


@pytest.fixture
def secure_tmp_path() -> Path:
    """DrvFs 不保留 chmod 位；lock 权限测试必须使用 Linux filesystem。"""

    with tempfile.TemporaryDirectory(prefix="e012-lock-", dir="/var/tmp") as value:
        yield Path(value)


def _args(
    tmp_path: Path,
    *,
    seed_start: int = 30_200,
    seed_end_exclusive: int = 30_440,
    resume: bool = False,
) -> Namespace:
    return Namespace(
        data=tmp_path / "data",
        model_cache=tmp_path / "cache",
        checkpoint=tmp_path / "checkpoint.pt",
        output=tmp_path / "out",
        boundary_type="grasp_lift",
        seed_start=seed_start,
        seed_end_exclusive=seed_end_exclusive,
        qwen_context_layer=12,
        sampling_seed=52_012,
        num_flow_steps=10,
        recency_decay=0.5,
        max_anomaly_replans=3,
        resume=resume,
    )


def _checkpoint_stats_payload() -> dict:
    return {
        "version": "franka-proprio-zscore/v1",
        "schema_version": "robot-vla-trajectory/v2",
        "embodiment": "maniskill-franka-panda-v1",
        "mean": [0.0] * 15,
        "std": [1.0] * 15,
        "count": 39_337,
    }


def _d0_compatibility_receipt() -> dict:
    return {
        "format": pool_module.D0_COMPATIBILITY_FORMAT,
        "status": "matched",
        "historical": {
            "projection": pool_module.D0_HISTORICAL_PROJECTION,
            "dataset_sha256": "c" * 64,
        },
        "current": {
            "projection": pool_module.D0_CURRENT_PROJECTION,
            "dataset_sha256": "d" * 64,
        },
        "translation": {
            "operation": "add-top-level-local_dagger-null",
            "translated_row_count": 220,
        },
        "manifest": {
            "sha256": "e" * 64,
            "trajectory_count": 220,
            "split_trajectory_counts": {"train": 176, "val": 22, "test": 22},
            "split_step_counts": {"train": 39_337, "val": 4_618, "test": 4_967},
        },
        "trajectory_files": {
            "hash_scheme": pool_module.D0_NPZ_SET_HASH_SCHEME,
            "count": 220,
            "aggregate_sha256": "f" * 64,
        },
        "step_count": 48_922,
        "audit_report": {"sha256": "1" * 64},
        "proprio_stats": {
            "sha256": "2" * 64,
            "semantic_sha256": _compact_sha256(_checkpoint_stats_payload()),
            "count": 39_337,
        },
    }


def _identity(
    tmp_path: Path,
    *,
    seed_start: int = 30_200,
    seed_end_exclusive: int = 30_440,
) -> dict:
    args = _args(
        tmp_path,
        seed_start=seed_start,
        seed_end_exclusive=seed_end_exclusive,
    )
    return build_pool_identity(
        args,
        source_revision="source-tree-sha256:" + "a" * 64,
        base_dataset_root=args.data.resolve(),
        checkpoint_path=args.checkpoint.resolve(),
        checkpoint_sha256="b" * 64,
        base_dataset_compatibility=_d0_compatibility_receipt(),
        runtime_identity={
            "python": "3.10.0",
            "platform": "test-platform",
            "packages": {
                "gymnasium": "1.0",
                "mani_skill": "3.0.1",
                "mplib": "0.1.1",
                "numpy": "1.26.4",
                "sapien": "3.0.3",
                "torch": "2.11.0+cu128",
                "transformers": "5.15.1",
            },
            "cuda": {
                "torch_cuda_version": "12.1",
                "cudnn_version": 8900,
                "cuda_visible_devices": "0",
                "device_index": 0,
                "device_name": "test-gpu",
                "device_uuid": "GPU-00000000-0000-0000-0000-000000000001",
                "compute_capability": [8, 0],
                "total_memory_bytes": 1,
                "nvidia_driver_version": "570.153.02",
            },
        },
    )


def _compact_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_compatibility_fixture(
    tmp_path: Path,
    meta_factory,
) -> tuple[Path, pool_module._FrozenD0Expectation, list[dict]]:
    root = tmp_path / "d0"
    (root / "trajectories").mkdir(parents=True)
    raw_rows: list[dict] = []
    current_files = []
    historical_files = []
    npz_receipts = []
    for index, split in enumerate(("train", "val", "test")):
        meta = replace(
            meta_factory(),
            trajectory_id=f"episode-{index:03d}",
            source_episode_id=f"source-{index:03d}",
            file=f"trajectories/episode-{index:03d}.npz",
            split=split,
            scene_id=f"scene-{index:03d}",
            randomization={"seed": index},
        )
        raw = meta.to_dict()
        assert raw.pop("local_dagger") is None
        raw_rows.append(raw)
        trajectory_path = root / meta.file
        trajectory_path.write_bytes(f"frozen-npz-{index}".encode())
        npz_sha256 = hashlib.sha256(trajectory_path.read_bytes()).hexdigest()
        parsed_current = pool_module.TrajectoryMeta.from_dict(raw).to_dict()
        parsed_historical = dict(parsed_current)
        assert parsed_historical.pop("local_dagger") is None
        current_files.append(
            {"meta": parsed_current, "npz_sha256": npz_sha256}
        )
        historical_files.append(
            {"meta": parsed_historical, "npz_sha256": npz_sha256}
        )
        npz_receipts.append({"file": meta.file, "sha256": npz_sha256})

    manifest_bytes = "".join(
        json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in raw_rows
    ).encode("utf-8")
    (root / "manifest.jsonl").write_bytes(manifest_bytes)
    historical_sha256 = _compact_sha256(
        sorted(historical_files, key=lambda item: str(item["meta"]))
    )
    current_sha256 = _compact_sha256(
        sorted(current_files, key=lambda item: str(item["meta"]))
    )
    stats_payload = {
        "version": "franka-proprio-zscore/v1",
        "schema_version": "robot-vla-trajectory/v2",
        "embodiment": "maniskill-franka-panda-v1",
        "mean": [0.0] * 15,
        "std": [1.0] * 15,
        "count": 5,
    }
    stats_bytes = json.dumps(stats_payload, indent=2).encode("utf-8")
    (root / "proprio_stats.json").write_bytes(stats_bytes)
    audit_payload = {
        "dataset_sha256": historical_sha256,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "trajectory_count": 3,
        "step_count": 15,
        "proprio_stats_count": 5,
        "split_trajectory_counts": {"train": 1, "val": 1, "test": 1},
        "split_step_counts": {"train": 5, "val": 5, "test": 5},
        "success_rate": 1.0,
    }
    audit_bytes = (
        json.dumps(audit_payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    (root / "audit_report.json").write_bytes(audit_bytes)
    expectation = pool_module._FrozenD0Expectation(
        historical_dataset_sha256=historical_sha256,
        current_dataset_sha256=current_sha256,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        audit_report_sha256=hashlib.sha256(audit_bytes).hexdigest(),
        proprio_stats_sha256=hashlib.sha256(stats_bytes).hexdigest(),
        proprio_stats_semantic_sha256=_compact_sha256(stats_payload),
        npz_set_sha256=_compact_sha256(
            sorted(npz_receipts, key=lambda item: item["file"])
        ),
        trajectory_count=3,
        step_count=15,
        proprio_stats_count=5,
        split_trajectory_counts=(("train", 1), ("val", 1), ("test", 1)),
        split_step_counts=(("train", 5), ("val", 5), ("test", 5)),
    )
    return root, expectation, raw_rows


def _common_record(identity: dict, *, seed: int, status: str) -> dict:
    return {
        "format": "robot-vla-local-dagger-collection/v1",
        "source_revision": identity["source_revision"],
        "base_dataset": identity["base_dataset"]["path"],
        "base_dataset_receipt": {
            "proprio_stats_sha256": identity["base_dataset"]["compatibility"][
                "proprio_stats"
            ]["sha256"],
            "proprio_stats_semantic_sha256": identity["base_dataset"][
                "compatibility"
            ]["proprio_stats"]["semantic_sha256"],
        },
        "checkpoint": {
            "path": identity["checkpoint"]["path"],
            "sha256": identity["checkpoint"]["sha256"],
            "metadata": {
                "epoch": 1,
                "proprio_stats": _checkpoint_stats_payload(),
            },
        },
        "config": _expected_candidate_config(identity, seed=seed),
        "status": status,
    }


def _rejected_record(identity: dict, *, seed: int) -> dict:
    record = _common_record(identity, seed=seed, status="rejected")
    usage = {"policy_actions": 300, "expert_actions": 0, "total_actions": 300}
    reason = "Policy 达到 300 Action 预算但未到达目标 boundary"
    record.update(
        {
            "failure": {"type": "EpisodeRejected", "reason": reason},
            "failure_diagnostics": {
                "format": "robot-vla-local-dagger-failure-diagnostics/v1",
                "environment_seed": seed,
                "boundary_type": "grasp_lift",
                "failure_reason": reason,
                "action_count": 300,
                "expert_takeover_step": None,
                "boundary_reached": False,
                "budget_exhaustion_phase": "policy",
                "final_transition": {"action_step": 300, "truncated": False},
                LOCAL_DAGGER_ACTION_BUDGET_PROTOCOL_FIELD: identity[
                    "action_budget_protocol"
                ],
                LOCAL_DAGGER_ACTION_BUDGET_USAGE_FIELD: usage,
            },
            LOCAL_DAGGER_ACTION_BUDGET_USAGE_FIELD: usage,
        }
    )
    return record


def _accepted_record(identity: dict, *, seed: int) -> dict:
    record = _common_record(identity, seed=seed, status="accepted")
    usage = {"policy_actions": 100, "expert_actions": 120, "total_actions": 220}
    policy_boundary = {
        "boundary_type": "grasp_lift",
        "control_step": 100,
        "tcp_object_relative_xyz_m": [0.01, 0.02, 0.03],
        "object_linear_speed_m_s": 0.04,
        "object_angular_speed_rad_s": 0.05,
        "joint_velocity_rms_rad_s": 0.06,
        "gripper_opening": 0.1,
        "is_grasped": True,
        "robot_object_contact_force_n": 1.0,
        "arm_mean_pairwise_disagreement": 0.07,
        "gripper_mean_pairwise_disagreement": 0.08,
    }
    paired_boundary = {
        "boundary_type": "grasp_lift",
        "control_step": 90,
        "tcp_object_relative_xyz_m": [0.0, 0.0, 0.0],
        "gripper_opening": 0.0,
    }
    trajectory = {
        "trajectory_id": f"local-dagger-grasp_lift-seed-{seed:06d}",
        "file": f"trajectories/local-dagger-grasp_lift-seed-{seed:06d}.npz",
        "num_steps": 220,
        "randomization": {
            "seed": seed,
            LOCAL_DAGGER_ACTION_BUDGET_PROTOCOL_FIELD: identity[
                "action_budget_protocol"
            ],
            LOCAL_DAGGER_ACTION_BUDGET_USAGE_FIELD: usage,
        },
        "local_dagger": {
            "source": "dagger_grasp_lift",
            "rollin_seed": seed,
            "rollin_policy_checkpoint_sha256": identity["checkpoint"]["sha256"],
            "boundary_type": "grasp_lift",
            "boundary_detection_step": 100,
            "expert_takeover_step": 100,
            "training_window_start": 100,
            "training_window_end": 164,
            "expert_recovery_success": True,
        },
        "outcome_evidence": {"task_completed": True},
    }
    record.update(
        {
            "result": {
                "trajectory": trajectory,
                "boundary": policy_boundary,
                "snapshot_round_trip": {"passed": True},
                LOCAL_DAGGER_ACTION_BUDGET_USAGE_FIELD: usage,
            },
            "paired_clean_expert": {
                "boundary": paired_boundary,
                "num_steps": 200,
                "task_completed": True,
            },
            "risk_components": compute_paired_risk_components(
                "grasp_lift",
                policy_boundary,
                paired_boundary,
            ),
            "eligible_for_risk_selection": True,
            "audit": {
                "trajectory_contract": "passed",
                "full_dataset_audit": "pending D0 union",
            },
            LOCAL_DAGGER_ACTION_BUDGET_USAGE_FIELD: usage,
        }
    )
    return record


def _write_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")


def _candidate_record_path(root: Path, *, seed: int = 30_200) -> Path:
    return root / "candidates" / f"seed-{seed:06d}" / "record.json"


def _write_experiment(path: Path, identity: dict) -> dict[str, str]:
    pool_module._atomic_write_json(path, identity)
    return pool_module._experiment_receipt(path)


def _compact_record(
    record: dict,
    record_path: Path,
    *,
    experiment_path: Path,
) -> dict:
    return compact_candidate_record(
        record,
        record_path,
        experiment_receipt=pool_module._experiment_receipt(experiment_path),
    )


def _write_accepted_artifact(record_path: Path, record: dict) -> None:
    trajectory = record["result"]["trajectory"]
    dataset_root = record_path.parent / "dataset"
    trajectory_path = dataset_root / trajectory["file"]
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory_path.write_bytes(b"frozen-npz-evidence")
    (dataset_root / "manifest.jsonl").write_text(
        json.dumps(trajectory, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _set_policy_boundary_step(record: dict, step: int) -> None:
    policy_boundary = record["result"]["boundary"]
    policy_boundary["control_step"] = step
    paired_boundary = record["paired_clean_expert"]["boundary"]
    record["risk_components"] = compute_paired_risk_components(
        "grasp_lift",
        policy_boundary,
        paired_boundary,
    )


def _rewrite_manifest(root: Path, rows: list[dict]) -> str:
    payload = "".join(
        json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows
    ).encode("utf-8")
    (root / "manifest.jsonl").write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_d0_compatibility_verifier_reproduces_both_projection_hashes(
    tmp_path: Path,
    meta_factory,
) -> None:
    root, expectation, _ = _write_compatibility_fixture(tmp_path, meta_factory)

    verification = pool_module._verify_d0_compatibility(
        root,
        expectation=expectation,
    )

    assert verification.receipt["status"] == "matched"
    assert verification.receipt["historical"] == {
        "projection": pool_module.D0_HISTORICAL_PROJECTION,
        "dataset_sha256": expectation.historical_dataset_sha256,
    }
    assert verification.receipt["current"] == {
        "projection": pool_module.D0_CURRENT_PROJECTION,
        "dataset_sha256": expectation.current_dataset_sha256,
    }
    assert verification.receipt["trajectory_files"]["aggregate_sha256"] == (
        expectation.npz_set_sha256
    )
    assert verification.receipt["proprio_stats"]["semantic_sha256"] == (
        expectation.proprio_stats_semantic_sha256
    )
    assert len(verification.entries) == 3


def test_d0_compatibility_rejects_any_raw_local_dagger_field(
    tmp_path: Path,
    meta_factory,
) -> None:
    root, expectation, rows = _write_compatibility_fixture(tmp_path, meta_factory)
    rows[0]["local_dagger"] = {"source": "must-not-be-downgraded"}
    expectation = replace(
        expectation,
        manifest_sha256=_rewrite_manifest(root, rows),
    )

    with pytest.raises(ValueError, match="禁止出现 local_dagger"):
        pool_module._verify_d0_compatibility(root, expectation=expectation)


def test_d0_compatibility_rejects_duplicate_json_keys(
    tmp_path: Path,
    meta_factory,
) -> None:
    root, expectation, _ = _write_compatibility_fixture(tmp_path, meta_factory)
    manifest_path = root / "manifest.jsonl"
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0].replace(
        '"trajectory_id":',
        '"trajectory_id":"shadowed","trajectory_id":',
        1,
    )
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    manifest_path.write_bytes(payload)
    expectation = replace(
        expectation,
        manifest_sha256=hashlib.sha256(payload).hexdigest(),
    )

    with pytest.raises(ValueError, match="重复 key"):
        pool_module._verify_d0_compatibility(root, expectation=expectation)


def test_d0_compatibility_rejects_npz_byte_drift_missing_and_extra(
    tmp_path: Path,
    meta_factory,
) -> None:
    byte_root, byte_expectation, _ = _write_compatibility_fixture(
        tmp_path / "byte",
        meta_factory,
    )
    byte_path = byte_root / "trajectories/episode-000.npz"
    byte_path.write_bytes(byte_path.read_bytes() + b"drift")
    with pytest.raises(ValueError, match="historical projection"):
        pool_module._verify_d0_compatibility(
            byte_root,
            expectation=byte_expectation,
        )

    missing_root, missing_expectation, _ = _write_compatibility_fixture(
        tmp_path / "missing",
        meta_factory,
    )
    (missing_root / "trajectories/episode-000.npz").unlink()
    with pytest.raises(FileNotFoundError, match="trajectory 路径"):
        pool_module._verify_d0_compatibility(
            missing_root,
            expectation=missing_expectation,
        )

    extra_root, extra_expectation, _ = _write_compatibility_fixture(
        tmp_path / "extra",
        meta_factory,
    )
    (extra_root / "trajectories/unreferenced.npz").write_bytes(b"extra")
    with pytest.raises(ValueError, match="manifest/NPZ 集合"):
        pool_module._verify_d0_compatibility(
            extra_root,
            expectation=extra_expectation,
        )


def test_d0_compatibility_rejects_path_escape_and_symlink(
    tmp_path: Path,
    meta_factory,
) -> None:
    escape_root, escape_expectation, rows = _write_compatibility_fixture(
        tmp_path / "escape",
        meta_factory,
    )
    rows[0]["file"] = "../outside.npz"
    (escape_root.parent / "outside.npz").write_bytes(b"outside")
    escape_expectation = replace(
        escape_expectation,
        manifest_sha256=_rewrite_manifest(escape_root, rows),
    )
    with pytest.raises(ValueError, match="路径非法"):
        pool_module._verify_d0_compatibility(
            escape_root,
            expectation=escape_expectation,
        )

    noncanonical_root, noncanonical_expectation, rows = (
        _write_compatibility_fixture(
            tmp_path / "noncanonical",
            meta_factory,
        )
    )
    rows[0]["file"] = "trajectories//episode-000.npz"
    noncanonical_expectation = replace(
        noncanonical_expectation,
        manifest_sha256=_rewrite_manifest(noncanonical_root, rows),
    )
    with pytest.raises(ValueError, match="路径非法"):
        pool_module._verify_d0_compatibility(
            noncanonical_root,
            expectation=noncanonical_expectation,
        )

    symlink_root, symlink_expectation, _ = _write_compatibility_fixture(
        tmp_path / "symlink",
        meta_factory,
    )
    trajectory = symlink_root / "trajectories/episode-000.npz"
    target = symlink_root.parent / "same-bytes.bin"
    target.write_bytes(trajectory.read_bytes())
    trajectory.unlink()
    trajectory.symlink_to(target)
    with pytest.raises(RuntimeError, match="symlink"):
        pool_module._verify_d0_compatibility(
            symlink_root,
            expectation=symlink_expectation,
        )


def test_d0_compatibility_rejects_count_step_and_stats_drift(
    tmp_path: Path,
    meta_factory,
) -> None:
    count_root, count_expectation, rows = _write_compatibility_fixture(
        tmp_path / "count",
        meta_factory,
    )
    count_expectation = replace(
        count_expectation,
        manifest_sha256=_rewrite_manifest(count_root, rows[:-1]),
    )
    with pytest.raises(ValueError, match="trajectory_count"):
        pool_module._verify_d0_compatibility(
            count_root,
            expectation=count_expectation,
        )

    step_root, step_expectation, rows = _write_compatibility_fixture(
        tmp_path / "step",
        meta_factory,
    )
    rows[0]["num_steps"] += 1
    step_expectation = replace(
        step_expectation,
        manifest_sha256=_rewrite_manifest(step_root, rows),
    )
    with pytest.raises(ValueError, match="step_count"):
        pool_module._verify_d0_compatibility(
            step_root,
            expectation=step_expectation,
        )

    stats_root, stats_expectation, _ = _write_compatibility_fixture(
        tmp_path / "stats",
        meta_factory,
    )
    stats_path = stats_root / "proprio_stats.json"
    stats_path.write_bytes(stats_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="proprio_stats SHA256"):
        pool_module._verify_d0_compatibility(
            stats_root,
            expectation=stats_expectation,
        )


def test_d0_compatibility_rejects_root_control_file_and_hardlink_aliases(
    tmp_path: Path,
    meta_factory,
) -> None:
    root, expectation, _ = _write_compatibility_fixture(
        tmp_path / "root-link",
        meta_factory,
    )
    root_alias = tmp_path / "d0-alias"
    root_alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(RuntimeError, match="dataset root symlink"):
        pool_module._verify_d0_compatibility(root_alias, expectation=expectation)

    control_root, control_expectation, _ = _write_compatibility_fixture(
        tmp_path / "control-link",
        meta_factory,
    )
    manifest = control_root / "manifest.jsonl"
    manifest_target = control_root.parent / "manifest-target.jsonl"
    manifest_target.write_bytes(manifest.read_bytes())
    manifest.unlink()
    manifest.symlink_to(manifest_target)
    with pytest.raises(RuntimeError, match="manifest.jsonl symlink"):
        pool_module._verify_d0_compatibility(
            control_root,
            expectation=control_expectation,
        )

    hardlink_root, hardlink_expectation, _ = _write_compatibility_fixture(
        tmp_path / "hardlink",
        meta_factory,
    )
    source = hardlink_root / "trajectories/episode-000.npz"
    os.link(source, hardlink_root.parent / "npz-hardlink")
    with pytest.raises(RuntimeError, match="只有一个硬链接"):
        pool_module._verify_d0_compatibility(
            hardlink_root,
            expectation=hardlink_expectation,
        )


def test_d0_verifier_failure_precedes_cuda_and_formal_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    args.checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(
        pool_module,
        "verify_frozen_d0_compatibility",
        lambda _: (_ for _ in ()).throw(ValueError("D0 preflight blocked")),
    )
    monkeypatch.setattr(
        pool_module,
        "_build_runtime_identity",
        lambda: pytest.fail("D0 gate 失败后不应初始化 CUDA"),
    )

    with pytest.raises(ValueError, match="D0 preflight blocked"):
        pool_module._run_locked(args)
    assert not args.output.exists()


def test_formal_checkpoint_mismatch_precedes_cuda_and_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    args.checkpoint.write_bytes(b"wrong-checkpoint")
    monkeypatch.setattr(
        pool_module,
        "verify_frozen_d0_compatibility",
        lambda _: SimpleNamespace(receipt={}, entries=()),
    )
    monkeypatch.setattr(
        pool_module,
        "_build_runtime_identity",
        lambda: pytest.fail("checkpoint gate 失败后不应初始化 CUDA"),
    )

    with pytest.raises(ValueError, match="checkpoint SHA256"):
        pool_module._run_locked(args)
    assert not args.output.exists()


def test_each_candidate_rechecks_frozen_stats_leaf(
    tmp_path: Path,
    meta_factory,
) -> None:
    root, expectation, _ = _write_compatibility_fixture(tmp_path, meta_factory)
    verification = pool_module._verify_d0_compatibility(
        root,
        expectation=expectation,
    )
    identity = {
        "base_dataset": {
            "path": str(root.resolve()),
            "compatibility": verification.receipt,
        }
    }
    pool_module._verify_candidate_stats_leaf(identity)

    stats_path = root / "proprio_stats.json"
    stats_path.write_bytes(stats_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="candidate 启动前"):
        pool_module._verify_candidate_stats_leaf(identity)


def test_each_candidate_rejects_replaced_d0_root_with_same_stats(
    tmp_path: Path,
    meta_factory,
) -> None:
    root, expectation, _ = _write_compatibility_fixture(tmp_path, meta_factory)
    verification = pool_module._verify_d0_compatibility(
        root,
        expectation=expectation,
    )
    identity = {
        "base_dataset": {
            "path": str(root.resolve()),
            "compatibility": verification.receipt,
        }
    }
    original = root.with_name("original-d0")
    root.rename(original)
    root.mkdir()
    (root / "proprio_stats.json").write_bytes(
        (original / "proprio_stats.json").read_bytes()
    )

    with pytest.raises(RuntimeError, match="root identity"):
        pool_module._verify_candidate_stats_leaf(
            identity,
            expected_root_identity=verification.root_stat_identity,
        )


def test_pool_identity_freezes_240_seed_example_without_hardcoding_pool_size(
    tmp_path: Path,
) -> None:
    identity = _identity(tmp_path)

    assert identity["environment_seeds"] == list(range(30_200, 30_440))
    assert identity["pool_size"] == 240
    assert identity["action_budget_protocol"]["name"] == (
        "segmented-300-180-480"
    )
    assert identity["paired_clean_expert_protocol"] == {
        "name": "legacy",
        "action_unit": "actual_environment_action",
        "environment_action_limit": 300,
    }
    assert identity["risk"]["selection"] == {
        "eligible_gate": ELIGIBLE_SELECTION_GATE,
        "high_count": HIGH_RISK_SELECTION_COUNT,
        "low_count": LOW_RISK_SELECTION_COUNT,
        "tie_break": "environment_seed ascending",
        "overlap_resolution": "select high first, then low from remaining",
    }
    assert identity["d1_input_contract"]["npz_directory_scan"] == "forbidden"
    assert identity["d1_input_contract"]["required_selection_receipt"] == {
        "source": "risk_selection.json",
        "sha256": True,
        "membership_fields": [
            "environment_seed",
            "selection_stratum",
            "risk_score",
            "index",
        ],
    }

    assert identity["qwen"] == {
        "model_id": "Qwen/Qwen3.5-2B",
        "revision": "15852e8c16360a2fea060d615a32b45270f8a8fc",
        "cache_path": str((tmp_path / "cache").resolve()),
    }
    assert identity["runtime"]["packages"]["mani_skill"] == "3.0.1"
    assert identity["runtime"]["packages"]["sapien"] == "3.0.3"
    assert identity["runtime"]["cuda"]["device_uuid"].startswith("GPU-")
    assert identity["runtime"]["cuda"]["nvidia_driver_version"] == "570.153.02"
    assert identity["base_dataset"]["audit"] == {
        "dataset_sha256": "c" * 64,
        "manifest_sha256": "e" * 64,
        "trajectory_count": 220,
        "step_count": 48_922,
    }
    assert identity["base_dataset"]["compatibility"] == (
        _d0_compatibility_receipt()
    )
    assert identity["seed_registry"] == {
        "reserved_amended_formal_range": {
            "start": AMENDED_FORMAL_SEED_START,
            "end_exclusive": AMENDED_FORMAL_SEED_END_LIMIT,
        },
        "legacy_e012a_end_exclusive": AMENDED_FORMAL_SEED_START,
        "checkpoint_validation_start": AMENDED_FORMAL_SEED_END_LIMIT,
    }

    smaller = _identity(tmp_path, seed_end_exclusive=30_220)
    assert smaller["pool_size"] == 20
    assert smaller["environment_seeds"] == list(range(30_200, 30_220))


def test_pool_identity_requires_complete_matched_d0_receipt() -> None:
    with pytest.raises(ValueError, match="非空 D0 compatibility"):
        pool_module._freeze_d0_compatibility({})

    unknown = _d0_compatibility_receipt()
    unknown["format"] = "unknown/v1"
    with pytest.raises(ValueError, match="format"):
        pool_module._freeze_d0_compatibility(unknown)

    incomplete = _d0_compatibility_receipt()
    del incomplete["proprio_stats"]["semantic_sha256"]
    with pytest.raises(ValueError, match="semantic"):
        pool_module._freeze_d0_compatibility(incomplete)


def test_runtime_identity_freezes_sapien_driver_and_physical_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    properties = SimpleNamespace(
        uuid="00000000-0000-0000-0000-000000000001",
        name="test-gpu",
        major=8,
        minor=9,
        total_memory=24,
    )
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            get_device_properties=lambda _index: properties,
        ),
        version=SimpleNamespace(cuda="12.8"),
        backends=SimpleNamespace(
            cudnn=SimpleNamespace(version=lambda: 91_900),
        ),
    )
    versions = {
        "gymnasium": "1.3.0",
        "mani-skill": "3.0.1",
        "mplib": "0.1.1",
        "numpy": "1.26.4",
        "sapien": "3.0.3",
        "torch": "2.11.0+cu128",
        "transformers": "5.15.1",
    }
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        pool_module.importlib.metadata,
        "version",
        lambda distribution: versions[distribution],
    )
    monkeypatch.setattr(
        pool_module,
        "_query_nvidia_driver_identity",
        lambda _properties: (
            "GPU-00000000-0000-0000-0000-000000000001",
            "570.153.02",
        ),
    )
    monkeypatch.setattr(pool_module.platform, "python_version", lambda: "3.10.12")
    monkeypatch.setattr(pool_module.platform, "platform", lambda: "test-platform")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    runtime = pool_module._build_runtime_identity()

    assert runtime["packages"]["sapien"] == "3.0.3"
    assert runtime["cuda"]["device_uuid"] == (
        "GPU-00000000-0000-0000-0000-000000000001"
    )
    assert runtime["cuda"]["nvidia_driver_version"] == "570.153.02"


def test_nvidia_driver_identity_uses_resolved_uuid_and_strict_single_line(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    executable = tmp_path / "nvidia-smi"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setattr(pool_module.shutil, "which", lambda _name: str(executable))
    captured: list[list[str]] = []

    def fake_run(command: list[str], **kwargs) -> SimpleNamespace:
        captured.append(command)
        assert kwargs["check"] is False
        assert kwargs["capture_output"] is True
        assert kwargs["timeout"] == 10
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "GPU-00000000-0000-0000-0000-000000000001, 570.153.02\n"
            ),
        )

    monkeypatch.setattr(pool_module.subprocess, "run", fake_run)
    properties = SimpleNamespace(uuid="00000000-0000-0000-0000-000000000001")

    assert pool_module._query_nvidia_driver_identity(properties) == (
        "GPU-00000000-0000-0000-0000-000000000001",
        "570.153.02",
    )
    assert captured == [
        [
            str(executable.resolve()),
            "--id=GPU-00000000-0000-0000-0000-000000000001",
            "--query-gpu=uuid,driver_version",
            "--format=csv,noheader,nounits",
        ]
    ]

    def ambiguous_run(_command: list[str], **_kwargs) -> SimpleNamespace:
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "GPU-00000000-0000-0000-0000-000000000001, 570.153.02\n"
                "GPU-00000000-0000-0000-0000-000000000002, 570.153.02\n"
            ),
        )

    monkeypatch.setattr(pool_module.subprocess, "run", ambiguous_run)
    with pytest.raises(RuntimeError, match="精确返回一行"):
        pool_module._query_nvidia_driver_identity(properties)

    monkeypatch.setattr(pool_module.shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="找不到可执行 nvidia-smi"):
        pool_module._query_nvidia_driver_identity(properties)

    with pytest.raises(RuntimeError, match="physical GPU UUID"):
        pool_module._query_nvidia_driver_identity(SimpleNamespace(uuid=None))


def test_formal_pool_lock_is_exclusive_persistent_and_reentrant_after_release(
    secure_tmp_path: Path,
) -> None:
    tmp_path = secure_tmp_path
    output = tmp_path / "formal-output"
    lock_path = pool_module._pool_lock_path(output)

    with pool_module._formal_pool_lock(output):
        first = lock_path.stat()
        assert first.st_nlink == 1
        with (
            pytest.raises(RuntimeError, match="已有 runner 持锁"),
            pool_module._formal_pool_lock(output),
        ):
            pass

    assert lock_path.is_file()
    assert lock_path.stat().st_ino == first.st_ino
    with pool_module._formal_pool_lock(output):
        assert lock_path.stat().st_ino == first.st_ino


def test_formal_pool_locks_for_different_outputs_do_not_conflict(
    secure_tmp_path: Path,
) -> None:
    tmp_path = secure_tmp_path
    first_output = tmp_path / "formal-a"
    second_output = tmp_path / "formal-b"

    with (
        pool_module._formal_pool_lock(first_output),
        pool_module._formal_pool_lock(second_output),
    ):
        assert pool_module._pool_lock_path(first_output).is_file()
        assert pool_module._pool_lock_path(second_output).is_file()


def test_formal_pool_lock_rejects_symlink_nonregular_and_multilink_paths(
    secure_tmp_path: Path,
) -> None:
    tmp_path = secure_tmp_path
    symlink_output = tmp_path / "symlink-output"
    symlink_lock = pool_module._pool_lock_path(symlink_output)
    target = tmp_path / "untrusted-target"
    target.write_text("do-not-lock", encoding="utf-8")
    symlink_lock.symlink_to(target)
    with (
        pytest.raises(RuntimeError, match="禁止 symlink"),
        pool_module._formal_pool_lock(symlink_output),
    ):
        pass

    nonregular_output = tmp_path / "nonregular-output"
    nonregular_lock = pool_module._pool_lock_path(nonregular_output)
    nonregular_lock.mkdir()
    with (
        pytest.raises(RuntimeError, match="不是普通文件"),
        pool_module._formal_pool_lock(nonregular_output),
    ):
        pass

    multilink_output = tmp_path / "multilink-output"
    multilink_lock = pool_module._pool_lock_path(multilink_output)
    multilink_lock.write_text("persistent", encoding="utf-8")
    os.link(multilink_lock, tmp_path / "second-link")
    with (
        pytest.raises(RuntimeError, match="只有一个硬链接"),
        pool_module._formal_pool_lock(multilink_output),
    ):
        pass


def test_formal_pool_lock_rejects_untrusted_parent_and_output_symlink(
    secure_tmp_path: Path,
) -> None:
    untrusted_parent = secure_tmp_path / "untrusted-parent"
    untrusted_parent.mkdir(mode=0o777)
    untrusted_parent.chmod(0o777)
    with (
        pytest.raises(PermissionError, match="group/other 不可写"),
        pool_module._formal_pool_lock(untrusted_parent / "formal-output"),
    ):
        pass

    trusted_parent = secure_tmp_path / "trusted-parent"
    trusted_parent.mkdir(mode=0o700)
    external = trusted_parent / "external"
    external.mkdir()
    output = trusted_parent / "formal-output"
    output.symlink_to(external, target_is_directory=True)
    with (
        pytest.raises(RuntimeError, match="formal output.*symlink"),
        pool_module._formal_pool_lock(output),
    ):
        pass


def test_seed_range_is_required_and_must_be_large_enough_for_gate(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="selection gate"):
        _identity(
            tmp_path,
            seed_start=30_200,
            seed_end_exclusive=30_219,
        )

    with pytest.raises(ValueError, match="预留起点"):
        _identity(
            tmp_path,
            seed_start=30_201,
            seed_end_exclusive=30_440,
        )

    with pytest.raises(ValueError, match="checkpoint validation"):
        _identity(
            tmp_path,
            seed_end_exclusive=31_001,
        )


def test_cli_requires_explicit_seed_range(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    required = [
        "collect_local_dagger_amended_pool",
        "--data",
        str(tmp_path / "data"),
        "--model-cache",
        str(tmp_path / "cache"),
        "--checkpoint",
        str(tmp_path / "checkpoint.pt"),
        "--output",
        str(tmp_path / "out"),
        "--boundary-type",
        "grasp_lift",
    ]
    monkeypatch.setattr(
        sys,
        "argv",
        required
        + [
            "--seed-start",
            "30200",
            "--seed-end-exclusive",
            "30440",
        ],
    )
    parsed = _parse_args()
    assert (parsed.seed_start, parsed.seed_end_exclusive) == (30_200, 30_440)

    monkeypatch.setattr(sys, "argv", required)
    with pytest.raises(SystemExit):
        _parse_args()


def test_candidate_command_always_uses_segmented_protocol(tmp_path: Path) -> None:
    identity = _identity(tmp_path)
    command = _candidate_command(
        identity,
        seed=30_200,
        candidate_dir=tmp_path / "candidates" / "seed-030200",
    )

    index = command.index("--action-budget-protocol")
    assert command[index + 1] == AMENDED_ACTION_BUDGET_PROTOCOL.value
    assert "--require-paired-clean-expert" in command
    assert command[command.index("--data") + 1] == identity["base_dataset"]["path"]
    assert command[command.index("--checkpoint") + 1] == identity["checkpoint"]["path"]


def test_record_requires_exact_config_and_blocks_error_resume(tmp_path: Path) -> None:
    identity = _identity(tmp_path)
    seed = 30_200
    record_path = _candidate_record_path(tmp_path, seed=seed)
    valid = _rejected_record(identity, seed=seed)
    _write_record(record_path, valid)
    assert _load_record(record_path, identity, seed)["status"] == "rejected"

    drifted = json.loads(json.dumps(valid))
    drifted["config"]["unfrozen_extra"] = True
    _write_record(record_path, drifted)
    with pytest.raises(ValueError, match="exact frozen config"):
        _load_record(record_path, identity, seed)


def test_record_rejects_duplicate_keys_and_nonfinite_constants(tmp_path: Path) -> None:
    identity = _identity(tmp_path)
    seed = 30_200
    record_path = _candidate_record_path(tmp_path, seed=seed)
    record_path.parent.mkdir(parents=True)
    payload = json.dumps(_rejected_record(identity, seed=seed), sort_keys=True)
    duplicate_status = payload.replace(
        '"status": "rejected"',
        '"status": "error", "status": "rejected"',
        1,
    )
    record_path.write_text(duplicate_status + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="重复 key"):
        _load_record(record_path, identity, seed)

    nonfinite = payload[:-1] + ', "ignored": NaN}'
    record_path.write_text(nonfinite + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="非有限 JSON constant"):
        _load_record(record_path, identity, seed)


@pytest.mark.parametrize("link_kind", ("symlink", "hardlink"))
def test_candidate_record_must_be_plain_contained_file(
    tmp_path: Path,
    link_kind: str,
) -> None:
    identity = _identity(tmp_path)
    seed = 30_200
    record_path = _candidate_record_path(tmp_path, seed=seed)
    record_path.parent.mkdir(parents=True)
    external = tmp_path / "external-record.json"
    external.write_text(
        json.dumps(_rejected_record(identity, seed=seed), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if link_kind == "symlink":
        record_path.symlink_to(external)
    else:
        os.link(external, record_path)

    with pytest.raises(RuntimeError, match="安全打开|硬链接|symlink"):
        _load_record(record_path, identity, seed)


def test_record_binds_actual_stats_file_and_checkpoint_metadata(
    tmp_path: Path,
) -> None:
    identity = _identity(tmp_path)
    seed = 30_200
    record_path = _candidate_record_path(tmp_path, seed=seed)
    valid = _rejected_record(identity, seed=seed)

    receipt_drift = json.loads(json.dumps(valid))
    receipt_drift["base_dataset_receipt"]["proprio_stats_sha256"] = "0" * 64
    _write_record(record_path, receipt_drift)
    with pytest.raises(ValueError, match="stats receipt"):
        _load_record(record_path, identity, seed)

    checkpoint_drift = json.loads(json.dumps(valid))
    checkpoint_drift["checkpoint"]["metadata"]["proprio_stats"]["mean"][0] = 1.0
    _write_record(record_path, checkpoint_drift)
    with pytest.raises(ValueError, match="checkpoint/proprio stats"):
        _load_record(record_path, identity, seed)


def test_accepted_record_requires_full_64_action_training_window(tmp_path: Path) -> None:
    identity = _identity(tmp_path)
    seed = 30_200
    record = _accepted_record(identity, seed=seed)
    short_usage = {"policy_actions": 100, "expert_actions": 50, "total_actions": 150}
    record[LOCAL_DAGGER_ACTION_BUDGET_USAGE_FIELD] = short_usage
    record["result"][LOCAL_DAGGER_ACTION_BUDGET_USAGE_FIELD] = short_usage
    trajectory = record["result"]["trajectory"]
    trajectory["num_steps"] = 150
    trajectory["randomization"][LOCAL_DAGGER_ACTION_BUDGET_USAGE_FIELD] = short_usage
    trajectory["local_dagger"]["training_window_end"] = 150
    record_path = _candidate_record_path(tmp_path, seed=seed)
    _write_record(record_path, record)

    with pytest.raises(ValueError, match="完整 64-action"):
        _load_record(record_path, identity, seed)


def test_accepted_policy_boundary_step_must_equal_takeover(tmp_path: Path) -> None:
    identity = _identity(tmp_path)
    seed = 30_200
    record = _accepted_record(identity, seed=seed)
    record["result"]["boundary"]["control_step"] = 99
    record_path = _candidate_record_path(tmp_path, seed=seed)
    _write_record(record_path, record)

    with pytest.raises(ValueError, match="Policy boundary/takeover"):
        _load_record(record_path, identity, seed)


@pytest.mark.parametrize("control_step", (0, 201))
def test_paired_boundary_step_must_be_within_clean_expert_rollout(
    tmp_path: Path,
    control_step: int,
) -> None:
    identity = _identity(tmp_path)
    seed = 30_200
    record = _accepted_record(identity, seed=seed)
    record["paired_clean_expert"]["boundary"]["control_step"] = control_step
    record_path = _candidate_record_path(tmp_path, seed=seed)
    _write_record(record_path, record)

    with pytest.raises(ValueError, match="paired boundary control_step"):
        _load_record(record_path, identity, seed)


def test_recorded_risk_components_must_exactly_match_frozen_recompute(
    tmp_path: Path,
) -> None:
    identity = _identity(tmp_path)
    seed = 30_200
    record = _accepted_record(identity, seed=seed)
    component = next(iter(record["risk_components"]))
    record["risk_components"][component] += 1.0
    record_path = _candidate_record_path(tmp_path, seed=seed)
    _write_record(record_path, record)

    with pytest.raises(ValueError, match="冻结公式精确重算"):
        _load_record(record_path, identity, seed)


@pytest.mark.parametrize("invalid_value", (-1.0, float("nan")))
def test_recorded_risk_components_must_be_finite_nonnegative(
    tmp_path: Path,
    invalid_value: float,
) -> None:
    identity = _identity(tmp_path)
    seed = 30_200
    record = _accepted_record(identity, seed=seed)
    component = next(iter(record["risk_components"]))
    record["risk_components"][component] = invalid_value
    record_path = _candidate_record_path(tmp_path, seed=seed)
    _write_record(record_path, record)

    with pytest.raises(ValueError, match="有限非负数|非有限 JSON constant"):
        _load_record(record_path, identity, seed)


def test_action_budget_usage_rejects_negative_counts(tmp_path: Path) -> None:
    identity = _identity(tmp_path)
    seed = 30_200
    record = _rejected_record(identity, seed=seed)
    invalid_usage = {"policy_actions": -1, "expert_actions": 0, "total_actions": -1}
    record[LOCAL_DAGGER_ACTION_BUDGET_USAGE_FIELD] = invalid_usage
    diagnostics = record["failure_diagnostics"]
    diagnostics[LOCAL_DAGGER_ACTION_BUDGET_USAGE_FIELD] = invalid_usage
    diagnostics["action_count"] = -1
    diagnostics["final_transition"]["action_step"] = -1
    record_path = _candidate_record_path(tmp_path, seed=seed)
    _write_record(record_path, record)

    with pytest.raises(ValueError, match="不得为负数"):
        _load_record(record_path, identity, seed)


def test_accepted_record_must_precede_environment_hard_deadline(tmp_path: Path) -> None:
    identity = _identity(tmp_path)
    seed = 30_200
    record = _accepted_record(identity, seed=seed)
    deadline_usage = {
        "policy_actions": 300,
        "expert_actions": 180,
        "total_actions": 480,
    }
    record[LOCAL_DAGGER_ACTION_BUDGET_USAGE_FIELD] = deadline_usage
    record["result"][LOCAL_DAGGER_ACTION_BUDGET_USAGE_FIELD] = deadline_usage
    trajectory = record["result"]["trajectory"]
    trajectory["num_steps"] = 480
    trajectory["randomization"][LOCAL_DAGGER_ACTION_BUDGET_USAGE_FIELD] = deadline_usage
    trajectory["local_dagger"].update(
        {
            "boundary_detection_step": 300,
            "expert_takeover_step": 300,
            "training_window_start": 300,
            "training_window_end": 364,
        }
    )
    _set_policy_boundary_step(record, 300)
    record_path = _candidate_record_path(tmp_path, seed=seed)
    _write_record(record_path, record)

    with pytest.raises(ValueError, match="严格早于 hard deadline"):
        _load_record(record_path, identity, seed)


def test_rejected_time_limit_requires_closed_deadline_signals(tmp_path: Path) -> None:
    identity = _identity(tmp_path)
    seed = 30_200
    record = _rejected_record(identity, seed=seed)
    deadline_usage = {
        "policy_actions": 300,
        "expert_actions": 180,
        "total_actions": 480,
    }
    record["failure"]["reason"] = "Episode 在可信成功前达到时间上限"
    record[LOCAL_DAGGER_ACTION_BUDGET_USAGE_FIELD] = deadline_usage
    diagnostics = record["failure_diagnostics"]
    diagnostics.update(
        {
            "failure_reason": record["failure"]["reason"],
            "action_count": 480,
            "expert_takeover_step": 300,
            "budget_exhaustion_phase": None,
            "final_transition": {"action_step": 480, "truncated": False},
            LOCAL_DAGGER_ACTION_BUDGET_USAGE_FIELD: deadline_usage,
        }
    )
    record_path = _candidate_record_path(tmp_path, seed=seed)
    _write_record(record_path, record)

    with pytest.raises(ValueError, match="hard-deadline.*不闭合"):
        _load_record(record_path, identity, seed)


def test_policy_action_300_boundary_terminal_keeps_boundary_priority(
    tmp_path: Path,
) -> None:
    identity = _identity(tmp_path)
    seed = 30_200
    record = _rejected_record(identity, seed=seed)
    reason = "目标 boundary 发生时环境已经结束"
    record["failure"]["reason"] = reason
    diagnostics = record["failure_diagnostics"]
    diagnostics.update(
        {
            "failure_reason": reason,
            "boundary_reached": True,
            "budget_exhaustion_phase": None,
            "final_transition": {
                "action_step": 300,
                "terminated": True,
                "truncated": False,
            },
        }
    )
    record_path = _candidate_record_path(tmp_path, seed=seed)
    _write_record(record_path, record)

    assert _load_record(record_path, identity, seed)["status"] == "rejected"

    diagnostics["boundary_reached"] = False
    _write_record(record_path, record)
    with pytest.raises(ValueError, match="boundary 优先证据"):
        _load_record(record_path, identity, seed)


def test_rejected_expert_cap_requires_nontruncated_closed_transition(
    tmp_path: Path,
) -> None:
    identity = _identity(tmp_path)
    seed = 30_200
    record = _rejected_record(identity, seed=seed)
    expert_usage = {
        "policy_actions": 100,
        "expert_actions": 180,
        "total_actions": 280,
    }
    record["failure"]["reason"] = (
        "Expert takeover 后达到 180 Action 恢复预算但任务未成功"
    )
    record[LOCAL_DAGGER_ACTION_BUDGET_USAGE_FIELD] = expert_usage
    diagnostics = record["failure_diagnostics"]
    diagnostics.update(
        {
            "failure_reason": record["failure"]["reason"],
            "action_count": 280,
            "expert_takeover_step": 100,
            "budget_exhaustion_phase": "expert",
            "final_transition": {"action_step": 280, "truncated": None},
            LOCAL_DAGGER_ACTION_BUDGET_USAGE_FIELD: expert_usage,
        }
    )
    record_path = _candidate_record_path(tmp_path, seed=seed)
    _write_record(record_path, record)

    with pytest.raises(ValueError, match="Expert-cap transition"):
        _load_record(record_path, identity, seed)

    errored = _common_record(identity, seed=seed, status="error")
    errored["failure"] = {"type": "RuntimeError", "reason": "CUDA failure"}
    _write_record(record_path, errored)
    with pytest.raises(RuntimeError, match="status=error"):
        _load_record(record_path, identity, seed)


def test_rejected_record_may_not_leave_canonical_dataset(tmp_path: Path) -> None:
    identity = _identity(tmp_path)
    seed = 30_200
    record_path = _candidate_record_path(tmp_path, seed=seed)
    _write_record(record_path, _rejected_record(identity, seed=seed))
    (record_path.parent / "dataset").mkdir()

    with pytest.raises(RuntimeError, match="残留 canonical dataset"):
        _load_record(record_path, identity, seed)


def test_rejected_record_blocks_dangling_dataset_symlink(tmp_path: Path) -> None:
    identity = _identity(tmp_path)
    seed = 30_200
    record_path = _candidate_record_path(tmp_path, seed=seed)
    _write_record(record_path, _rejected_record(identity, seed=seed))
    (record_path.parent / "dataset").symlink_to(
        tmp_path / "missing-dataset",
        target_is_directory=True,
    )

    with pytest.raises(RuntimeError, match="残留 canonical dataset"):
        _load_record(record_path, identity, seed)


def test_accepted_record_with_staging_marker_is_not_committed(tmp_path: Path) -> None:
    identity = _identity(tmp_path)
    seed = 30_200
    record = _accepted_record(identity, seed=seed)
    record_path = _candidate_record_path(tmp_path, seed=seed)
    _write_record(record_path, record)
    _write_accepted_artifact(record_path, record)
    (record_path.parent / "dataset" / CANDIDATE_DATASET_STAGING_MARKER).write_text(
        json.dumps({"format": "robot-vla-candidate-dataset-staging/v1"}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="uncommitted staging marker"):
        _load_record(record_path, identity, seed)


def test_accepted_record_blocks_dangling_staging_marker(tmp_path: Path) -> None:
    identity = _identity(tmp_path)
    seed = 30_200
    record = _accepted_record(identity, seed=seed)
    record_path = _candidate_record_path(tmp_path, seed=seed)
    _write_record(record_path, record)
    _write_accepted_artifact(record_path, record)
    (record_path.parent / "dataset" / CANDIDATE_DATASET_STAGING_MARKER).symlink_to(
        tmp_path / "missing-marker-target"
    )

    with pytest.raises(RuntimeError, match="uncommitted staging marker"):
        _load_record(record_path, identity, seed)


def test_accepted_manifest_rejects_duplicate_keys(tmp_path: Path) -> None:
    identity = _identity(tmp_path)
    seed = 30_200
    record = _accepted_record(identity, seed=seed)
    record_path = _candidate_record_path(tmp_path, seed=seed)
    _write_record(record_path, record)
    _write_accepted_artifact(record_path, record)
    manifest = record_path.parent / "dataset" / "manifest.jsonl"
    payload = manifest.read_text(encoding="utf-8")
    trajectory_id = record["result"]["trajectory"]["trajectory_id"]
    payload = payload.replace(
        f'"trajectory_id": "{trajectory_id}"',
        f'"trajectory_id": "shadow", "trajectory_id": "{trajectory_id}"',
        1,
    )
    manifest.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match="重复 key"):
        _load_record(record_path, identity, seed)


def test_accepted_dataset_root_must_not_be_external_symlink(tmp_path: Path) -> None:
    identity = _identity(tmp_path)
    seed = 30_200
    record = _accepted_record(identity, seed=seed)
    record_path = _candidate_record_path(tmp_path, seed=seed)
    _write_record(record_path, record)
    external_record = tmp_path / "external" / "record.json"
    _write_accepted_artifact(external_record, record)
    (record_path.parent / "dataset").symlink_to(
        external_record.parent / "dataset",
        target_is_directory=True,
    )

    with pytest.raises(RuntimeError, match="dataset root symlink"):
        _load_record(record_path, identity, seed)


@pytest.mark.parametrize("leaf", ("manifest", "trajectory"))
@pytest.mark.parametrize("link_kind", ("symlink", "hardlink"))
def test_accepted_artifact_leaves_must_be_plain_single_link_files(
    tmp_path: Path,
    leaf: str,
    link_kind: str,
) -> None:
    identity = _identity(tmp_path)
    seed = 30_200
    record = _accepted_record(identity, seed=seed)
    record_path = _candidate_record_path(tmp_path, seed=seed)
    _write_record(record_path, record)
    _write_accepted_artifact(record_path, record)
    dataset_root = record_path.parent / "dataset"
    if leaf == "manifest":
        artifact_path = dataset_root / "manifest.jsonl"
    else:
        artifact_path = dataset_root / record["result"]["trajectory"]["file"]
    external = tmp_path / f"external-{leaf}"
    artifact_path.replace(external)
    if link_kind == "symlink":
        artifact_path.symlink_to(external)
    else:
        os.link(external, artifact_path)

    with pytest.raises(RuntimeError, match="安全打开|硬链接|symlink"):
        _load_record(record_path, identity, seed)


def test_accepted_trajectory_file_must_use_canonical_relative_path(
    tmp_path: Path,
) -> None:
    identity = _identity(tmp_path)
    seed = 30_200
    record = _accepted_record(identity, seed=seed)
    trajectory = record["result"]["trajectory"]
    trajectory["file"] = "trajectories/../episode.npz"
    record_path = _candidate_record_path(tmp_path, seed=seed)
    _write_record(record_path, record)
    _write_accepted_artifact(record_path, record)

    with pytest.raises(ValueError, match="trajectory file 路径非法"):
        _load_record(record_path, identity, seed)


def test_resume_blocks_canonical_dataset_without_record(tmp_path: Path) -> None:
    identity = _identity(tmp_path)
    output = tmp_path / "out"
    _write_experiment(output / "experiment.json", identity)
    candidate_dir = output / "candidates" / "seed-030200"
    (candidate_dir / "dataset").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="partial candidate"):
        _load_existing_candidates(output, identity=identity)


@pytest.mark.parametrize("link_kind", ("symlink", "hardlink"))
def test_resume_experiment_must_be_plain_contained_file(
    tmp_path: Path,
    link_kind: str,
) -> None:
    identity = _identity(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    external = tmp_path / "external" / "experiment.json"
    _write_experiment(external, identity)
    experiment_path = output / "experiment.json"
    if link_kind == "symlink":
        experiment_path.symlink_to(external)
    else:
        os.link(external, experiment_path)

    with pytest.raises(RuntimeError, match="安全打开|硬链接"):
        _load_existing_candidates(output, identity=identity)


@pytest.mark.parametrize("level", ("candidates", "candidate"))
def test_resume_rejects_symlinked_candidate_hierarchy(
    tmp_path: Path,
    level: str,
) -> None:
    identity = _identity(tmp_path)
    seed = 30_200
    output = tmp_path / "out"
    _write_experiment(output / "experiment.json", identity)
    external = tmp_path / "external"
    external_record = _candidate_record_path(external, seed=seed)
    _write_record(external_record, _rejected_record(identity, seed=seed))
    if level == "candidates":
        (output / "candidates").symlink_to(
            external / "candidates",
            target_is_directory=True,
        )
    else:
        candidates_root = output / "candidates"
        candidates_root.mkdir()
        (candidates_root / f"seed-{seed:06d}").symlink_to(
            external_record.parent,
            target_is_directory=True,
        )

    with pytest.raises(RuntimeError, match="symlink"):
        _load_existing_candidates(output, identity=identity)


def test_compact_rejection_record_preserves_seed_reason_and_record_hash(
    tmp_path: Path,
) -> None:
    identity = _identity(tmp_path)
    record = _rejected_record(identity, seed=30_200)
    record_path = _candidate_record_path(tmp_path)
    _write_record(record_path, record)
    experiment_path = tmp_path / "experiment.json"
    _write_experiment(experiment_path, identity)

    row = _compact_record(
        record,
        record_path,
        experiment_path=experiment_path,
    )

    assert row["environment_seed"] == 30_200
    assert row["eligible_for_risk_selection"] is False
    assert row["failure"]["reason"] == record["failure"]["reason"]
    assert len(row["record_sha256"]) == 64
    assert row["pool"] == pool_module._experiment_receipt(experiment_path)


def test_canonical_manifest_contains_only_accepted_selected_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pool_module, "_audit_accepted_trajectory", lambda *_: None)
    identity = _identity(
        tmp_path,
        seed_start=30_200,
        seed_end_exclusive=30_220,
    )
    experiment_path = tmp_path / "out" / "experiment.json"
    experiment_path.parent.mkdir(parents=True)
    experiment_receipt = _write_experiment(experiment_path, identity)
    candidate_records = {}
    compact_candidates = []
    scored_candidates = []
    for index, seed in enumerate(identity["environment_seeds"]):
        record = _accepted_record(identity, seed=seed)
        record_path = (
            tmp_path
            / "out"
            / "candidates"
            / f"seed-{seed:06d}"
            / "record.json"
        )
        _write_record(record_path, record)
        _write_accepted_artifact(record_path, record)
        # 未被 record/manifest 引用的 NPZ 不应被目录扫描进 D1。
        (record_path.parent / "dataset" / "stray.npz").write_bytes(b"forbidden")
        candidate_records[seed] = (record, record_path)
        compact_candidates.append(
            _compact_record(
                record,
                record_path,
                experiment_path=experiment_path,
            )
        )
        scored_candidates.append(
            {
                "environment_seed": seed,
                "risk_score": float(index),
                "selection_stratum": (
                    "high" if index < HIGH_RISK_SELECTION_COUNT else "low"
                ),
            }
        )
    selection = SimpleNamespace(
        high_risk_seeds=tuple(identity["environment_seeds"][:14]),
        low_risk_seeds=tuple(identity["environment_seeds"][14:]),
        scored_candidates=tuple(scored_candidates),
    )
    risk_payload = {
        "high_risk_seeds": list(selection.high_risk_seeds),
        "low_risk_seeds": list(selection.low_risk_seeds),
        "selected_seeds": list(
            selection.high_risk_seeds + selection.low_risk_seeds
        ),
        "scored_candidates": list(selection.scored_candidates),
    }
    selection.to_dict = lambda: risk_payload
    candidate_manifest_path = experiment_path.parent / "collection_candidates.jsonl"
    pool_module._atomic_write_jsonl(candidate_manifest_path, compact_candidates)
    candidate_receipt = pool_module._candidate_manifest_receipt(
        candidate_manifest_path,
        rows=compact_candidates,
    )
    risk_selection_path = experiment_path.parent / "risk_selection.json"
    pool_module._atomic_write_json(
        risk_selection_path,
        pool_module._risk_selection_payload(
            selection,
            experiment_receipt=experiment_receipt,
            candidate_manifest_receipt=candidate_receipt,
        ),
    )
    risk_selection_sha256 = pool_module._sha256_file(risk_selection_path)

    rows = build_canonical_selected_records(
        identity,
        selection=selection,
        candidate_records=candidate_records,
        experiment_path=experiment_path,
        candidate_manifest_path=candidate_manifest_path,
        candidate_manifest_sha256=candidate_receipt[
            "collection_candidates_sha256"
        ],
        risk_selection_path=risk_selection_path,
        risk_selection_sha256=risk_selection_sha256,
    )

    assert len(rows) == 20
    assert {row["format"] for row in rows} == {CANONICAL_SELECTED_RECORD_FORMAT}
    assert all(row["candidate"]["status"] == "accepted" for row in rows)
    assert all(
        row["pool"]["experiment_sha256"]
        == experiment_receipt["experiment_sha256"]
        for row in rows
    )
    assert all(row["candidate_manifest"] == candidate_receipt for row in rows)
    assert all(row["selection"]["selected"] is True for row in rows)
    assert all(
        row["selection"]["risk_selection"] == str(risk_selection_path.resolve())
        and row["selection"]["risk_selection_sha256"]
        == risk_selection_sha256
        for row in rows
    )
    scored_by_seed = {
        row["environment_seed"]: row for row in risk_payload["scored_candidates"]
    }
    assert all(
        row["environment_seed"] in risk_payload["selected_seeds"]
        and row["selection"]["index"]
        == risk_payload["selected_seeds"].index(row["environment_seed"])
        and row["selection"]["stratum"]
        == scored_by_seed[row["environment_seed"]]["selection_stratum"]
        and row["selection"]["risk_score"]
        == scored_by_seed[row["environment_seed"]]["risk_score"]
        for row in rows
    )
    assert all(row["artifact"]["trajectory_file"].endswith(".npz") for row in rows)
    assert not any("stray.npz" in json.dumps(row) for row in rows)
    assert all(
        row["d1_input_contract"]["npz_directory_scan"] == "forbidden"
        for row in rows
    )

    risk_selection_path.write_text(
        risk_selection_path.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="risk selection SHA256 漂移"):
        build_canonical_selected_records(
            identity,
            selection=selection,
            candidate_records=candidate_records,
            experiment_path=experiment_path,
            candidate_manifest_path=candidate_manifest_path,
            candidate_manifest_sha256=candidate_receipt[
                "collection_candidates_sha256"
            ],
            risk_selection_path=risk_selection_path,
            risk_selection_sha256=risk_selection_sha256,
        )


def test_accepted_artifact_is_reloaded_and_mask_corruption_fails_closed(
    tmp_path: Path,
    meta_factory,
    arrays_factory,
) -> None:
    identity = _identity(tmp_path)
    seed = 30_200
    takeover = 16
    steps = 80
    usage = {
        "policy_actions": takeover,
        "expert_actions": steps - takeover,
        "total_actions": steps,
    }
    source = np.ones(steps, dtype=np.int8)
    source[:takeover] = 0
    skill_id = np.repeat(np.arange(5, dtype=np.int16), steps // 5)
    arrays = arrays_factory(
        steps=steps,
        skill_id=skill_id,
        action_source=source,
        expert_supervision_mask=source == 1,
    )
    outcome = OutcomeEvidence(
        predicate_version=OUTCOME_PREDICATE_VERSION,
        task_completed=True,
        final_is_released=True,
        stable_place_steps=30,
        external_goal_visible_steps=steps,
        wrist_goal_visible_steps=steps,
        both_goal_visible_steps=steps,
        final_object_to_goal_distance_m=0.0,
        final_object_linear_speed_m_s=0.0,
        final_object_angular_speed_rad_s=0.0,
    )
    meta = meta_factory(
        trajectory_id=f"local-dagger-grasp_lift-seed-{seed:06d}",
        source_episode_id=f"source-{seed}",
        file=f"trajectories/local-dagger-grasp_lift-seed-{seed:06d}.npz",
        scene_id=f"scene-{seed}",
        num_steps=steps,
        randomization={
            "seed": seed,
            LOCAL_DAGGER_ACTION_BUDGET_PROTOCOL_FIELD: identity[
                "action_budget_protocol"
            ],
            LOCAL_DAGGER_ACTION_BUDGET_USAGE_FIELD: usage,
        },
        outcome_evidence=outcome,
        local_dagger=LocalDaggerProvenance(
            source="dagger_grasp_lift",
            rollin_seed=seed,
            rollin_policy_checkpoint_sha256=identity["checkpoint"]["sha256"],
            boundary_type="grasp_lift",
            boundary_detection_step=takeover,
            expert_takeover_step=takeover,
            training_window_start=takeover,
            training_window_end=takeover + 64,
            expert_recovery_success=True,
        ),
    )
    record = _accepted_record(identity, seed=seed)
    record[LOCAL_DAGGER_ACTION_BUDGET_USAGE_FIELD] = usage
    record["result"][LOCAL_DAGGER_ACTION_BUDGET_USAGE_FIELD] = usage
    record["result"]["trajectory"] = meta.to_dict()
    _set_policy_boundary_step(record, takeover)
    record_path = _candidate_record_path(tmp_path, seed=seed)
    _write_record(record_path, record)
    TrajectoryDatasetWriter(record_path.parent / "dataset", RobotSpec()).write(
        meta,
        arrays,
    )

    assert _load_record(record_path, identity, seed)["status"] == "accepted"

    trajectory_path = record_path.parent / "dataset" / meta.file
    with np.load(trajectory_path, allow_pickle=False) as archive:
        payload = {name: np.asarray(archive[name]).copy() for name in archive.files}
    corrupted_mask = payload["expert_supervision_mask"]
    corrupted_mask[0] = True
    payload["expert_supervision_mask"] = corrupted_mask
    np.savez(trajectory_path, **payload)

    with pytest.raises(ValueError, match="supervision_mask"):
        _load_record(record_path, identity, seed)


def test_resume_detects_record_hash_drift_even_when_compact_fields_are_unchanged(
    tmp_path: Path,
) -> None:
    identity = _identity(tmp_path)
    output = tmp_path / "out"
    experiment_path = output / "experiment.json"
    output.mkdir(parents=True)
    experiment_receipt = _write_experiment(experiment_path, identity)
    seed = 30_200
    record_path = output / "candidates" / f"seed-{seed:06d}" / "record.json"
    record = _rejected_record(identity, seed=seed)
    _write_record(record_path, record)
    rows, candidate_records = _load_existing_candidates(output, identity=identity)
    candidate_path = output / "collection_candidates.jsonl"
    pool_module._atomic_write_jsonl(candidate_path, rows)
    candidate_receipt = pool_module._candidate_manifest_receipt(
        candidate_path,
        rows=rows,
    )
    pool_module._atomic_write_json(
        output / "collection_summary.json",
        pool_module._progress_summary(
            rows,
            expected=240,
            experiment_receipt=experiment_receipt,
            candidate_manifest_receipt=candidate_receipt,
        ),
    )
    _validate_resume_derived_artifacts(
        output,
        identity=identity,
        rows=rows,
        candidate_records=candidate_records,
        experiment_path=experiment_path,
    )

    record["checkpoint"]["metadata"]["harmless_but_mutated"] = True
    _write_record(record_path, record)
    mutated_rows, mutated_records = _load_existing_candidates(
        output,
        identity=identity,
    )
    with pytest.raises(ValueError, match="collection_candidates 内容漂移"):
        _validate_resume_derived_artifacts(
            output,
            identity=identity,
            rows=mutated_rows,
            candidate_records=mutated_records,
            experiment_path=experiment_path,
        )


def test_resume_accepts_candidates_one_step_ahead_of_summary_prefix(
    tmp_path: Path,
) -> None:
    identity = _identity(
        tmp_path,
        seed_start=30_200,
        seed_end_exclusive=30_220,
    )
    output = tmp_path / "out"
    experiment_path = output / "experiment.json"
    experiment_receipt = _write_experiment(experiment_path, identity)
    for seed in identity["environment_seeds"][:2]:
        record_path = output / "candidates" / f"seed-{seed:06d}" / "record.json"
        _write_record(record_path, _rejected_record(identity, seed=seed))
    rows, candidate_records = _load_existing_candidates(output, identity=identity)
    candidate_path = output / "collection_candidates.jsonl"
    pool_module._atomic_write_jsonl(candidate_path, rows)
    previous_receipt = pool_module._candidate_snapshot_receipt(
        candidate_path,
        rows=rows[:1],
    )
    pool_module._atomic_write_json(
        output / "collection_summary.json",
        pool_module._progress_summary(
            rows[:1],
            expected=20,
            experiment_receipt=experiment_receipt,
            candidate_manifest_receipt=previous_receipt,
        ),
    )

    _validate_resume_derived_artifacts(
        output,
        identity=identity,
        rows=rows,
        candidate_records=candidate_records,
        experiment_path=experiment_path,
    )
    pool_module._synchronize_resume_progress(
        output,
        rows,
        expected=20,
        experiment_receipt=experiment_receipt,
    )
    synchronized_summary = json.loads(
        (output / "collection_summary.json").read_text(encoding="utf-8")
    )
    assert synchronized_summary["completed_candidates"] == 2
    assert synchronized_summary["collection_candidates_row_count"] == 2

    # C 落后一条时，summary 只能绑定同一条前缀，不能再落后一步。
    pool_module._atomic_write_jsonl(candidate_path, rows[:1])
    stale_receipt = pool_module._candidate_snapshot_receipt(
        candidate_path,
        rows=[],
    )
    pool_module._atomic_write_json(
        output / "collection_summary.json",
        pool_module._progress_summary(
            [],
            expected=20,
            experiment_receipt=experiment_receipt,
            candidate_manifest_receipt=stale_receipt,
        ),
    )
    with pytest.raises(ValueError, match="progress/final 快照"):
        _validate_resume_derived_artifacts(
            output,
            identity=identity,
            rows=rows,
            candidate_records=candidate_records,
            experiment_path=experiment_path,
        )

    # Summary 也不得领先 candidate manifest。
    current_receipt = pool_module._candidate_snapshot_receipt(
        candidate_path,
        rows=rows,
    )
    pool_module._atomic_write_json(
        output / "collection_summary.json",
        pool_module._progress_summary(
            rows,
            expected=20,
            experiment_receipt=experiment_receipt,
            candidate_manifest_receipt=current_receipt,
        ),
    )
    with pytest.raises(ValueError, match="progress/final 快照"):
        _validate_resume_derived_artifacts(
            output,
            identity=identity,
            rows=rows,
            candidate_records=candidate_records,
            experiment_path=experiment_path,
        )


def test_gate_failure_summary_binds_experiment_and_full_candidates(
    tmp_path: Path,
) -> None:
    identity = _identity(
        tmp_path,
        seed_start=30_200,
        seed_end_exclusive=30_220,
    )
    output = tmp_path / "out"
    experiment_path = output / "experiment.json"
    experiment_receipt = _write_experiment(experiment_path, identity)
    for seed in identity["environment_seeds"]:
        record_path = output / "candidates" / f"seed-{seed:06d}" / "record.json"
        _write_record(record_path, _rejected_record(identity, seed=seed))
    rows, candidate_records = _load_existing_candidates(output, identity=identity)
    candidate_path = output / "collection_candidates.jsonl"
    pool_module._atomic_write_jsonl(candidate_path, rows)
    candidate_receipt = pool_module._candidate_manifest_receipt(
        candidate_path,
        rows=rows,
    )
    summary = pool_module._selection_summary(
        rows,
        expected=20,
        selection=None,
        experiment_receipt=experiment_receipt,
        candidate_manifest_receipt=candidate_receipt,
    )
    pool_module._atomic_write_json(output / "collection_summary.json", summary)

    _validate_resume_derived_artifacts(
        output,
        identity=identity,
        rows=rows,
        candidate_records=candidate_records,
        experiment_path=experiment_path,
    )
    assert summary["selection_gate_passed"] is False
    assert summary["experiment_sha256"] == experiment_receipt["experiment_sha256"]
    assert summary["collection_candidates_sha256"] == candidate_receipt[
        "collection_candidates_sha256"
    ]

    summary["experiment_sha256"] = "0" * 64
    pool_module._atomic_write_json(output / "collection_summary.json", summary)
    with pytest.raises(ValueError, match="progress/final 快照"):
        _validate_resume_derived_artifacts(
            output,
            identity=identity,
            rows=rows,
            candidate_records=candidate_records,
            experiment_path=experiment_path,
        )


def test_resume_rejects_noncanonical_experiment_bytes(tmp_path: Path) -> None:
    identity = _identity(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    experiment_path = output / "experiment.json"
    experiment_path.write_text(json.dumps(identity) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="canonical frozen bytes"):
        _validate_resume_derived_artifacts(
            output,
            identity=identity,
            rows=[],
            candidate_records={},
            experiment_path=experiment_path,
        )


def test_resume_accepts_each_selection_publish_snapshot_and_closes_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pool_module, "_audit_accepted_trajectory", lambda *_: None)
    identity = _identity(
        tmp_path,
        seed_start=30_200,
        seed_end_exclusive=30_220,
    )
    output = tmp_path / "out"
    output.mkdir()
    experiment_path = output / "experiment.json"
    experiment_receipt = _write_experiment(experiment_path, identity)
    rows = []
    candidate_records = {}
    for seed in identity["environment_seeds"]:
        record = _accepted_record(identity, seed=seed)
        record_path = output / "candidates" / f"seed-{seed:06d}" / "record.json"
        _write_record(record_path, record)
        _write_accepted_artifact(record_path, record)
        rows.append(
            _compact_record(
                record,
                record_path,
                experiment_path=experiment_path,
            )
        )
        candidate_records[seed] = (record, record_path)
    rows.sort(key=lambda row: row["environment_seed"])
    candidate_path = output / "collection_candidates.jsonl"
    candidate_receipt = pool_module._write_progress(
        output,
        rows,
        len(rows),
        experiment_receipt=experiment_receipt,
    )
    progress_summary = json.loads(
        (output / "collection_summary.json").read_text(encoding="utf-8")
    )

    def validate_snapshot() -> None:
        _validate_resume_derived_artifacts(
            output,
            identity=identity,
            rows=rows,
            candidate_records=candidate_records,
            experiment_path=experiment_path,
        )

    # selection 尚未发布、仅 risk、risk+canonical、最终 summary 四个原子快照。
    validate_snapshot()
    selection = pool_module.score_and_select_risk_candidates(
        "grasp_lift",
        rows,
        high_count=HIGH_RISK_SELECTION_COUNT,
        low_count=LOW_RISK_SELECTION_COUNT,
    )
    risk_path = output / "risk_selection.json"
    risk_payload = pool_module._risk_selection_payload(
        selection,
        experiment_receipt=experiment_receipt,
        candidate_manifest_receipt=candidate_receipt,
    )
    pool_module._atomic_write_json(risk_path, risk_payload)
    validate_snapshot()

    lagging_candidate_receipt = pool_module._candidate_snapshot_receipt(
        candidate_path,
        rows=rows[:-1],
    )
    pool_module._atomic_write_json(
        output / "collection_summary.json",
        pool_module._progress_summary(
            rows[:-1],
            expected=len(rows),
            experiment_receipt=experiment_receipt,
            candidate_manifest_receipt=lagging_candidate_receipt,
        ),
    )
    with pytest.raises(ValueError, match="progress/final 快照"):
        validate_snapshot()
    pool_module._atomic_write_json(
        output / "collection_summary.json",
        progress_summary,
    )

    risk_sha256 = pool_module._sha256_file(risk_path)
    canonical_rows = build_canonical_selected_records(
        identity,
        selection=selection,
        candidate_records=candidate_records,
        experiment_path=experiment_path,
        candidate_manifest_path=candidate_path,
        candidate_manifest_sha256=candidate_receipt[
            "collection_candidates_sha256"
        ],
        risk_selection_path=risk_path,
        risk_selection_sha256=risk_sha256,
    )
    canonical_path = output / "canonical_selected_records.jsonl"
    pool_module._atomic_write_jsonl(canonical_path, canonical_rows)
    validate_snapshot()

    canonical_sha256 = pool_module._sha256_file(canonical_path)
    final_summary = pool_module._selection_summary(
        rows,
        expected=len(rows),
        selection=selection,
        experiment_receipt=experiment_receipt,
        candidate_manifest_receipt=candidate_receipt,
        risk_selection_sha256=risk_sha256,
        canonical_selected_records_sha256=canonical_sha256,
    )
    pool_module._atomic_write_json(output / "collection_summary.json", final_summary)
    validate_snapshot()
    assert final_summary["risk_selection_sha256"] == risk_sha256
    assert final_summary["experiment_sha256"] == experiment_receipt[
        "experiment_sha256"
    ]
    assert final_summary["collection_candidates_sha256"] == candidate_receipt[
        "collection_candidates_sha256"
    ]
    assert (
        final_summary["canonical_selected_records_sha256"]
        == canonical_sha256
    )

    canonical_rows[0]["selection"]["risk_score"] += 0.01
    pool_module._atomic_write_jsonl(canonical_path, canonical_rows)
    with pytest.raises(ValueError, match="canonical selected records 漂移"):
        validate_snapshot()


def test_resume_cleanup_is_limited_to_known_root_atomic_temps(
    tmp_path: Path,
) -> None:
    output = tmp_path / "out"
    nested = output / "candidates" / "seed-030200"
    nested.mkdir(parents=True)
    known_names = (
        "collection_candidates.jsonl",
        "collection_summary.json",
        "risk_selection.json",
        "canonical_selected_records.jsonl",
    )
    stale = []
    for name in known_names:
        path = output / f".{name}.abcdefgh.tmp"
        path.write_text("interrupted", encoding="utf-8")
        stale.append(path)
    unknown = output / ".unknown.abcdefgh.tmp"
    unknown.write_text("preserve", encoding="utf-8")
    nested_temp = nested / ".record.json.abcdefgh.tmp"
    nested_temp.write_text("preserve", encoding="utf-8")

    pool_module._remove_stale_root_derived_temps(output)

    assert not any(path.exists() for path in stale)
    assert unknown.is_file()
    assert nested_temp.is_file()


def test_paired_clean_expert_protocol_is_legacy_300() -> None:
    assert PAIRED_CLEAN_EXPERT_PROTOCOL == {
        "name": "legacy",
        "action_unit": "actual_environment_action",
        "environment_action_limit": 300,
    }
