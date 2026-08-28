from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_vla.evaluation.atomic import AtomicSkillEpisodeResult
from robot_vla.evaluation.checkpoint_sweep import (
    CheckpointSweepCandidate,
    assess_promotion_candidates,
    discover_sweep_candidates,
    predicate_residual,
    rank_candidate_summaries,
    select_confirmation_labels,
    summarize_candidate_results,
)


def _metric(epoch: int, loss: float) -> dict[str, object]:
    return {
        "event": "epoch",
        "epoch": epoch,
        "train": {"optimizer_steps": 64},
        "validation": {
            "loss": loss,
            "base_loss": loss * 0.8,
            "event_loss": loss * 0.4,
        },
    }


def _candidate(tmp_path: Path, label: str, epoch: int) -> CheckpointSweepCandidate:
    checkpoint = tmp_path / f"{label}.pt"
    checkpoint.touch()
    return CheckpointSweepCandidate(
        label=label,
        epoch=epoch,
        optimizer_steps=epoch * 64,
        checkpoint=checkpoint,
        validation_loss=0.1,
        validation_base_loss=0.08,
        validation_event_loss=0.04,
        is_validation_best=label.endswith("best"),
    )


def _result(
    *,
    skill: str,
    seed: int,
    success: bool,
    distance_m: float,
    steps: int = 100,
    saturation_count: int = 0,
) -> AtomicSkillEpisodeResult:
    target = 0 if skill == "reach" else 3
    return AtomicSkillEpisodeResult(
        seed=seed,
        skill_name=skill,
        instruction="Move the cube.",
        sampling_seed_base=seed + 100,
        success=success,
        failure_category=None if success else f"{skill}_failed",
        failure_stage=None,
        error=None,
        preparation_steps=0,
        initial_completed_skill_count=target,
        final_completed_skill_count=target + int(success),
        policy_environment_steps=steps,
        replans=max(1, steps // 4),
        sampling_seeds=(seed + 100,),
        action_chunks=1,
        tracking_correction_saturation_count=saturation_count,
        tracking_correction_requested_abs_max_rad=0.01,
        tracking_correction_applied_abs_max_rad=0.01,
        final_is_grasped=skill == "transport",
        final_tcp_to_object_distance_m=(distance_m if skill == "reach" else 0.001),
        final_object_height_above_support_m=0.1,
        final_object_to_goal_xy_distance_m=(
            distance_m if skill == "transport" else 0.2
        ),
        final_object_to_goal_distance_m=0.2,
        final_object_linear_speed_m_s=0.0,
        final_object_angular_speed_rad_s=0.0,
        wall_time_s=1.0,
    )


def test_discovers_periodic_and_nonperiodic_best_without_duplicate_latest(tmp_path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    rows = [_metric(epoch, 1.0 / epoch) for epoch in range(1, 21)]
    rows[18] = _metric(19, 0.01)
    (tmp_path / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (checkpoint_dir / "step-00000640.pt").touch()
    (checkpoint_dir / "step-00001280.pt").touch()
    (checkpoint_dir / "best.pt").touch()
    (checkpoint_dir / "latest.pt").touch()

    candidates, anchor = discover_sweep_candidates(tmp_path)

    assert [item.label for item in candidates] == ["e010", "e019-best", "e020"]
    assert anchor == "e019-best"
    assert sum(item.is_validation_best for item in candidates) == 1


def test_predicate_residual_uses_skill_specific_four_centimeter_margin() -> None:
    reach = _result(skill="reach", seed=1, success=False, distance_m=0.07)
    transport = _result(skill="transport", seed=1, success=True, distance_m=0.03)

    assert predicate_residual(reach) == pytest.approx(0.03)
    assert predicate_residual(transport) == 0.0


def test_ranking_prioritizes_comparable_behavior_then_success_and_residual(
    tmp_path,
) -> None:
    first = _candidate(tmp_path, "e010", 10)
    second = _candidate(tmp_path, "e020", 20)
    unsafe = _candidate(tmp_path, "e030", 30)
    summaries = [
        summarize_candidate_results(
            first,
            [
                _result(skill="reach", seed=1, success=True, distance_m=0.03),
                _result(skill="reach", seed=2, success=False, distance_m=0.06),
            ],
        ),
        summarize_candidate_results(
            second,
            [
                _result(skill="reach", seed=1, success=True, distance_m=0.03),
                _result(skill="reach", seed=2, success=True, distance_m=0.03),
            ],
        ),
        summarize_candidate_results(
            unsafe,
            [
                _result(
                    skill="reach",
                    seed=1,
                    success=True,
                    distance_m=0.03,
                    saturation_count=1,
                ),
                _result(skill="reach", seed=2, success=True, distance_m=0.03),
            ],
        ),
    ]

    ranked = rank_candidate_summaries(summaries, "reach")

    assert [row["candidate"]["label"] for row in ranked] == ["e020", "e010", "e030"]


def test_selects_union_of_per_skill_top_k_and_anchor(tmp_path) -> None:
    candidates = [
        _candidate(tmp_path, "e010", 10),
        _candidate(tmp_path, "e020", 20),
        _candidate(tmp_path, "e098-best", 98),
    ]
    summaries = []
    for index, candidate in enumerate(candidates):
        summaries.append(
            summarize_candidate_results(
                candidate,
                [
                    _result(
                        skill="reach",
                        seed=1,
                        success=index == 0,
                        distance_m=0.03 if index == 0 else 0.08 + index * 0.01,
                    ),
                    _result(
                        skill="transport",
                        seed=1,
                        success=index == 1,
                        distance_m=0.03 if index == 1 else 0.08 + index * 0.01,
                    ),
                ],
            )
        )

    labels, rankings = select_confirmation_labels(
        summaries, anchor_label="e098-best", top_k_per_skill=1
    )

    assert labels == ["e010", "e020", "e098-best"]
    assert rankings["reach"][0] == "e010"
    assert rankings["transport"][0] == "e020"


def test_promotion_requires_two_success_gain_no_regression_and_residual_gain(
    tmp_path,
) -> None:
    anchor = _candidate(tmp_path, "e098-best", 98)
    improved = _candidate(tmp_path, "e060", 60)
    anchor_rows = []
    improved_rows = []
    for seed in range(10):
        anchor_rows.extend(
            [
                _result(
                    skill="reach",
                    seed=seed,
                    success=seed < 2,
                    distance_m=0.03 if seed < 2 else 0.10,
                ),
                _result(
                    skill="transport",
                    seed=seed,
                    success=seed == 0,
                    distance_m=0.03 if seed == 0 else 0.08,
                ),
            ]
        )
        improved_rows.extend(
            [
                _result(
                    skill="reach",
                    seed=seed,
                    success=seed < 4,
                    distance_m=0.03 if seed < 4 else 0.06,
                ),
                _result(
                    skill="transport",
                    seed=seed,
                    success=seed == 0,
                    distance_m=0.03 if seed == 0 else 0.08,
                ),
            ]
        )
    summaries = [
        summarize_candidate_results(anchor, anchor_rows),
        summarize_candidate_results(improved, improved_rows),
    ]

    promoted, comparisons = assess_promotion_candidates(
        summaries, anchor_label="e098-best"
    )

    assert promoted == ["e060"]
    assert comparisons["e060"]["reach"]["success_delta"] == 2
    assert comparisons["e060"]["reach"]["paired_wins"] == 2
    assert comparisons["e060"]["transport"]["success_delta"] == 0


def test_zero_anchor_residual_uses_json_safe_undefined_ratio(tmp_path) -> None:
    anchor = _candidate(tmp_path, "e098-best", 98)
    candidate = _candidate(tmp_path, "e060", 60)
    anchor_rows = []
    candidate_rows = []
    for seed in range(3):
        for skill in ("reach", "transport"):
            anchor_rows.append(
                _result(skill=skill, seed=seed, success=seed == 0, distance_m=0.03)
            )
            candidate_rows.append(
                _result(skill=skill, seed=seed, success=True, distance_m=0.05)
            )
    summaries = [
        summarize_candidate_results(anchor, anchor_rows),
        summarize_candidate_results(candidate, candidate_rows),
    ]

    promoted, comparisons = assess_promotion_candidates(
        summaries, anchor_label="e098-best"
    )

    assert promoted == []
    assert comparisons["e060"]["reach"]["mean_predicate_residual_ratio"] is None
    json.dumps(comparisons, allow_nan=False)
