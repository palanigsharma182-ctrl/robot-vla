from __future__ import annotations

import json
import os

import pytest

from robot_vla.cli.build_e012_d1 import (
    _read_json,
    _validate_selection,
)
from robot_vla.sim.local_dagger_risk import score_and_select_risk_candidates


def _write_json(path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path, rows) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def test_strict_reader_rejects_duplicate_nonfinite_symlink_and_hardlink(tmp_path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"value": 1, "value": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="重复字段"):
        _read_json(duplicate, root=tmp_path)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value": NaN}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="非有限"):
        _read_json(nonfinite, root=tmp_path)

    valid = tmp_path / "valid.json"
    valid.write_text('{"value": 1}\n', encoding="utf-8")
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(valid)
    with pytest.raises(ValueError, match="普通文件"):
        _read_json(symlink, root=tmp_path)

    hardlink = tmp_path / "hardlink.json"
    os.link(valid, hardlink)
    with pytest.raises(ValueError, match="hardlink"):
        _read_json(valid, root=tmp_path)


def test_selection_must_recompute_from_complete_candidate_manifest(tmp_path) -> None:
    rows = []
    for seed in range(20):
        rows.append(
            {
                "environment_seed": seed,
                "boundary_type": "reach_grasp",
                "status": "accepted",
                "record": str(tmp_path / "candidates" / f"seed-{seed:06d}" / "record.json"),
                "episode_sampling_seed": 1_000 + seed,
                "eligible_for_risk_selection": True,
                "risk_components": {
                    "tcp_object_xy_error_m": float(seed),
                    "relative_z_deviation_m": float(seed),
                    "tcp_linear_speed_m_s": float(seed),
                    "joint_velocity_rms_rad_s": float(seed),
                    "gripper_opening_deviation": float(seed),
                    "arm_mean_pairwise_disagreement": float(seed),
                    "gripper_mean_pairwise_disagreement": float(seed),
                },
            }
        )
    selection = score_and_select_risk_candidates("reach_grasp", rows).to_dict()
    _write_jsonl(tmp_path / "collection_candidates.jsonl", rows)
    _write_json(tmp_path / "risk_selection.json", selection)
    _write_json(
        tmp_path / "collection_summary.json",
        {
            "format": "robot-vla-local-dagger-pool/v1",
            "scan_complete": True,
            "expected_candidates": 20,
            "completed_candidates": 20,
            "status_counts": {"accepted": 20},
            "selection_gate_passed": True,
            "high_risk_seeds": selection["high_risk_seeds"],
            "low_risk_seeds": selection["low_risk_seeds"],
            "selected_count": 20,
        },
    )
    experiment = {
        "boundary_type": "reach_grasp",
        "environment_seeds": list(range(20)),
    }

    _candidates, _risk, high, low = _validate_selection(
        tmp_path,
        experiment=experiment,
        expected_candidates=20,
    )

    assert high == tuple(selection["high_risk_seeds"])
    assert low == tuple(selection["low_risk_seeds"])
    drifted = json.loads((tmp_path / "risk_selection.json").read_text(encoding="utf-8"))
    drifted["scored_candidates"][0]["risk_score"] = 0.123
    _write_json(tmp_path / "risk_selection.json", drifted)
    with pytest.raises(ValueError, match="精确复算"):
        _validate_selection(
            tmp_path,
            experiment=experiment,
            expected_candidates=20,
        )
