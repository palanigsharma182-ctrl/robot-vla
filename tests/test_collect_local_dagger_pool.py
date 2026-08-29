from argparse import Namespace
from pathlib import Path

from robot_vla.cli.collect_local_dagger_pool import (
    FORMAL_SEED_RANGES,
    build_pool_identity,
    compact_candidate_record,
)


def _args(tmp_path: Path) -> Namespace:
    return Namespace(
        data=tmp_path / "data",
        model_cache=tmp_path / "cache",
        checkpoint=tmp_path / "checkpoint.pt",
        output=tmp_path / "out",
        boundary_type="reach_grasp",
        qwen_context_layer=12,
        sampling_seed=52_012,
        num_flow_steps=10,
        recency_decay=0.5,
        max_anomaly_replans=3,
        resume=False,
    )


def test_pool_identity_freezes_exact_formal_seed_range_and_risk_schema(tmp_path) -> None:
    identity = build_pool_identity(
        _args(tmp_path),
        source_revision="source-tree-sha256:" + "a" * 64,
        checkpoint_sha256="b" * 64,
    )

    start, end = FORMAL_SEED_RANGES["reach_grasp"]
    assert identity["environment_seeds"] == list(range(start, end))
    assert len(identity["environment_seeds"]) == 100
    assert identity["risk"]["selection"]["high_count"] == 14
    assert identity["risk"]["selection"]["low_count"] == 6
    assert identity["config"]["paired_clean_expert_required"] is True


def test_compact_rejection_record_preserves_seed_and_reason(tmp_path) -> None:
    record = {
        "status": "rejected",
        "config": {
            "environment_seed": 30_001,
            "boundary_type": "reach_grasp",
            "episode_sampling_seed": 7,
        },
        "failure": {"type": "EpisodeRejected", "reason": "boundary missing"},
    }

    row = compact_candidate_record(record, tmp_path / "record.json")

    assert row["environment_seed"] == 30_001
    assert row["eligible_for_risk_selection"] is False
    assert row["failure"]["reason"] == "boundary missing"
