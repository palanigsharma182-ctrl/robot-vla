from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from robot_vla.cli.analyze_e012_paired_evaluation import run
from robot_vla.cli.evaluate_atomic_maniskill import (
    ATOMIC_EVALUATION_EXPERIMENT_FORMAT,
)
from robot_vla.cli.evaluate_maniskill import EVALUATION_EXPERIMENT_FORMAT
from robot_vla.cli.select_e012_checkpoint import run as run_checkpoint_selection
from robot_vla.contracts import PICK_AND_PLACE_SKILLS
from robot_vla.evaluation.atomic import (
    ATOMIC_ROLLOUT_FORMAT,
    AtomicSkillEpisodeResult,
    summarize_atomic_rollouts,
)
from robot_vla.evaluation.e012_checkpoint_selection import (
    E012CheckpointCandidate,
    select_e012_checkpoint,
)
from robot_vla.evaluation.e012_paired import (
    analyze_e012_pair,
    analyze_full_chain_pair,
    exact_paired_test,
)
from robot_vla.evaluation.rollout import (
    ROLLOUT_FORMAT,
    RolloutEpisodeResult,
    summarize_rollouts,
)


def _rollout(
    seed: int,
    *,
    completed: int,
    sampling_seed_base: int = 99,
    sampling_seeds: tuple[int, ...] = (10, 11, 12),
    tracking: int = 0,
) -> RolloutEpisodeResult:
    success = completed == len(PICK_AND_PLACE_SKILLS)
    failure_category = None if success else f"{PICK_AND_PLACE_SKILLS[completed]}_failed"
    completion_steps = (10, 20, 30, 40, 50)
    return RolloutEpisodeResult(
        seed_group="unseen",
        seed=seed,
        instruction=f"pick and place scene {seed % 3}",
        sampling_seed_base=sampling_seed_base,
        success=success,
        environment_success=success,
        predicate_success=success,
        failure_category=failure_category,
        failure_stage=None,
        error=None,
        environment_steps=60,
        replans=len(sampling_seeds),
        sampling_seeds=sampling_seeds,
        action_chunks=len(sampling_seeds),
        normalized_action_abs_max=0.5,
        physical_arm_delta_abs_max_rad=0.02,
        gripper_target_min=0.0,
        gripper_target_max=1.0,
        completed_skill_count=completed,
        skill_completed=tuple(index < completed for index in range(5)),
        terminated=success,
        truncated=not success,
        final_is_grasped=2 <= completed < 5,
        stable_grasp_steps=2 if completed >= 2 else 0,
        stable_place_steps=4 if success else 0,
        final_tcp_to_object_distance_m=0.01,
        final_object_height_above_support_m=0.05,
        final_object_to_goal_xy_distance_m=0.01,
        final_object_to_goal_distance_m=0.01,
        final_object_linear_speed_m_s=0.0,
        final_object_angular_speed_rad_s=0.0,
        wall_time_s=1.0,
        tracking_correction_saturation_count=tracking,
        skill_completion_environment_steps=tuple(
            step if index < completed else None
            for index, step in enumerate(completion_steps)
        ),
    )


def _atomic(seed: int, skill: str, *, success: bool = True) -> AtomicSkillEpisodeResult:
    target = PICK_AND_PLACE_SKILLS.index(skill)
    return AtomicSkillEpisodeResult(
        seed=seed,
        skill_name=skill,
        instruction=f"pick and place scene {seed % 3}",
        sampling_seed_base=99,
        success=success,
        failure_category=None if success else f"{skill}_failed",
        failure_stage=None,
        error=None,
        preparation_steps=target * 10,
        initial_completed_skill_count=target,
        final_completed_skill_count=target + int(success),
        policy_environment_steps=10,
        replans=2,
        sampling_seeds=(10, 11),
        action_chunks=2,
        tracking_correction_saturation_count=0,
        tracking_correction_requested_abs_max_rad=None,
        tracking_correction_applied_abs_max_rad=None,
        final_is_grasped=skill not in {"reach", "place"},
        final_tcp_to_object_distance_m=0.01,
        final_object_height_above_support_m=0.05,
        final_object_to_goal_xy_distance_m=0.01,
        final_object_to_goal_distance_m=0.01,
        final_object_linear_speed_m_s=0.0,
        final_object_angular_speed_rad_s=0.0,
        wall_time_s=0.5,
    )


def test_exact_paired_test_uses_two_sided_binomial_tail() -> None:
    assert exact_paired_test(3, 0) == {
        "method": "two-sided-exact-mcnemar-binomial",
        "discordant_pairs": 3,
        "p_value": 0.25,
    }
    assert exact_paired_test(0, 0)["p_value"] == 1.0


def test_full_chain_analysis_reports_common_support_and_predecessor_change() -> None:
    replay = [_rollout(1, completed=1), _rollout(2, completed=3)]
    dagger = [_rollout(1, completed=2), _rollout(2, completed=2)]

    result = analyze_full_chain_pair(replay, dagger)

    assert result["unconditional_paired"]["grasp"]["net_dagger_wins"] == 1
    assert result["unconditional_paired"]["lift"]["net_dagger_wins"] == -1
    assert result["handoffs"]["grasp"]["common_predecessor_support"] == 2
    assert result["handoffs"]["grasp"]["handoff_on_common_support"][
        "dagger_wins"
    ] == 1
    assert result["handoffs"]["lift"]["common_predecessor_support"] == 1
    assert result["handoffs"]["lift"]["handoff_on_common_support"][
        "replay_wins"
    ] == 1
    assert result["mean_completed_skill_count_delta"] == pytest.approx(0.0)


def test_full_chain_analysis_rejects_unpaired_flow_seed_base() -> None:
    with pytest.raises(ValueError, match="Flow sampling base"):
        analyze_full_chain_pair(
            [_rollout(1, completed=1)],
            [_rollout(1, completed=2, sampling_seed_base=100)],
        )


def test_stage_a_gate_requires_preregistered_seeds_and_passes_direction_signal() -> None:
    replay_full = [
        _rollout(seed, completed=1 if seed < 32_002 else 2)
        for seed in range(32_000, 32_020)
    ]
    dagger_full = [_rollout(seed, completed=2) for seed in range(32_000, 32_020)]
    replay_atomic = [
        _atomic(seed, skill)
        for skill in PICK_AND_PLACE_SKILLS
        for seed in range(32_020, 32_025)
    ]
    dagger_atomic = [
        _atomic(seed, skill)
        for skill in PICK_AND_PLACE_SKILLS
        for seed in range(32_020, 32_025)
    ]

    result = analyze_e012_pair(
        replay_full,
        dagger_full,
        replay_atomic=replay_atomic,
        dagger_atomic=dagger_atomic,
        protocol="stage-a",
    )

    assert result["full_chain"]["unconditional_paired"]["grasp"][
        "net_dagger_wins"
    ] == 2
    assert result["gate"]["passed"] is True
    assert all(result["gate"]["checks"].values())


def test_checkpoint_selection_applies_lexicographic_rule_after_exclusions() -> None:
    baseline_full = tuple(
        _rollout(seed, completed=2) for seed in range(31_000, 31_020)
    )
    baseline_atomic = tuple(
        _atomic(seed, skill)
        for skill in PICK_AND_PLACE_SKILLS
        for seed in range(31_020, 31_025)
    )
    equal_atomic = tuple(
        _atomic(seed, skill)
        for skill in PICK_AND_PLACE_SKILLS
        for seed in range(31_020, 31_025)
    )

    def full_with_lift_successes(count: int) -> tuple[RolloutEpisodeResult, ...]:
        return tuple(
            _rollout(seed, completed=3 if index < count else 2)
            for index, seed in enumerate(range(31_000, 31_020))
        )

    candidates = (
        E012CheckpointCandidate(
            label="e010",
            epoch=10,
            validation_total_loss=0.3,
            full_chain=full_with_lift_successes(0),
            atomic=equal_atomic,
        ),
        E012CheckpointCandidate(
            label="e020",
            epoch=20,
            validation_total_loss=0.2,
            full_chain=full_with_lift_successes(1),
            atomic=equal_atomic,
        ),
        E012CheckpointCandidate(
            label="e030",
            epoch=30,
            validation_total_loss=0.1,
            full_chain=full_with_lift_successes(1),
            atomic=equal_atomic,
        ),
    )

    result = select_e012_checkpoint(
        baseline_full_chain=baseline_full,
        baseline_atomic=baseline_atomic,
        candidates=candidates,
    )

    assert result["selection_gate_passed"] is True
    assert result["eligible_ranking"] == ["e030", "e020", "e010"]
    assert result["selected"]["label"] == "e030"


def test_checkpoint_selection_excludes_atomic_regression() -> None:
    baseline_full = tuple(
        _rollout(seed, completed=2) for seed in range(31_000, 31_020)
    )
    baseline_atomic = tuple(
        _atomic(seed, skill)
        for skill in PICK_AND_PLACE_SKILLS
        for seed in range(31_020, 31_025)
    )
    regressed_atomic = tuple(
        _atomic(seed, skill, success=not (skill == "grasp" and seed == 31_020))
        for skill in PICK_AND_PLACE_SKILLS
        for seed in range(31_020, 31_025)
    )
    clean_atomic = tuple(
        _atomic(seed, skill)
        for skill in PICK_AND_PLACE_SKILLS
        for seed in range(31_020, 31_025)
    )
    candidates = tuple(
        E012CheckpointCandidate(
            label=f"e{epoch:03d}",
            epoch=epoch,
            validation_total_loss=0.1 * epoch,
            full_chain=baseline_full,
            atomic=regressed_atomic if epoch == 30 else clean_atomic,
        )
        for epoch in (10, 20, 30)
    )

    result = select_e012_checkpoint(
        baseline_full_chain=baseline_full,
        baseline_atomic=baseline_atomic,
        candidates=candidates,
    )

    by_label = {row["label"]: row for row in result["candidates"]}
    assert by_label["e030"]["eligible"] is False
    assert by_label["e030"]["exclusion_checks"][
        "atomic_grasp_not_regressed"
    ] is False
    assert result["selected"]["label"] == "e010"


def _write_full_chain_run(
    root: Path,
    results: list[RolloutEpisodeResult],
    *,
    checkpoint_sha256: str,
) -> None:
    root.mkdir()
    experiment = {
        "format": EVALUATION_EXPERIMENT_FORMAT,
        "rollout_format": ROLLOUT_FORMAT,
        "dataset": {
            "dataset_sha256": "d" * 64,
            "manifest_sha256": "m" * 64,
            "trajectory_count": 1,
            "step_count": 10,
        },
        "checkpoint": {
            "sha256": checkpoint_sha256,
            "size_bytes": 1,
            "metadata": {"epoch": 30},
        },
        "evaluation_code_revision": "source:v1",
        "config": {
            "environment_id": "RobotVLAPickCubeToRegion-v1",
            "sampling_seed": 99,
            "inference_strategy": "temporal-ensemble",
        },
        "episodes": [
            {
                "seed_group": result.seed_group,
                "seed": result.seed,
                "instruction": result.instruction,
            }
            for result in results
        ],
    }
    summary = summarize_rollouts(results)
    summary.update(
        {
            "completed_episodes": len(results),
            "expected_episodes": len(results),
            "complete": True,
            "checkpoint_sha256": checkpoint_sha256,
            "dataset_sha256": "d" * 64,
        }
    )
    (root / "experiment.json").write_text(
        json.dumps(experiment, allow_nan=False), encoding="utf-8"
    )
    (root / "summary.json").write_text(
        json.dumps(summary, allow_nan=False), encoding="utf-8"
    )
    (root / "episodes.jsonl").write_text(
        "".join(
            json.dumps(result.to_dict(), allow_nan=False) + "\n" for result in results
        ),
        encoding="utf-8",
    )


def _write_atomic_run(
    root: Path,
    results: list[AtomicSkillEpisodeResult],
    *,
    checkpoint_sha256: str,
) -> None:
    root.mkdir()
    experiment = {
        "format": ATOMIC_EVALUATION_EXPERIMENT_FORMAT,
        "rollout_format": ATOMIC_ROLLOUT_FORMAT,
        "dataset": {
            "dataset_sha256": "d" * 64,
            "manifest_sha256": "m" * 64,
            "trajectory_count": 1,
            "step_count": 10,
        },
        "checkpoint": {
            "sha256": checkpoint_sha256,
            "size_bytes": 1,
            "metadata": {"epoch": 30},
        },
        "evaluation_code_revision": "source:v1",
        "config": {
            "environment_id": "RobotVLAPickCubeToRegion-v1",
            "sampling_seed": 99,
            "inference_strategy": "temporal-ensemble",
        },
        "episodes": [
            {
                "seed": result.seed,
                "skill_name": result.skill_name,
                "instruction": result.instruction,
            }
            for result in results
        ],
    }
    summary = summarize_atomic_rollouts(results)
    summary.update(
        {
            "completed_episodes": len(results),
            "expected_episodes": len(results),
            "complete": True,
            "checkpoint_sha256": checkpoint_sha256,
            "dataset_sha256": "d" * 64,
        }
    )
    (root / "experiment.json").write_text(
        json.dumps(experiment, allow_nan=False), encoding="utf-8"
    )
    (root / "summary.json").write_text(
        json.dumps(summary, allow_nan=False), encoding="utf-8"
    )
    (root / "episodes.jsonl").write_text(
        "".join(
            json.dumps(result.to_dict(), allow_nan=False) + "\n" for result in results
        ),
        encoding="utf-8",
    )


def test_cli_recomputes_complete_inputs_and_writes_receipted_analysis(
    tmp_path: Path,
) -> None:
    replay = tmp_path / "replay"
    dagger = tmp_path / "dagger"
    _write_full_chain_run(replay, [_rollout(40_000, completed=1)], checkpoint_sha256="r" * 64)
    _write_full_chain_run(dagger, [_rollout(40_000, completed=2)], checkpoint_sha256="g" * 64)
    output = tmp_path / "analysis.json"

    result = run(
        argparse.Namespace(
            replay_full_chain=[replay],
            dagger_full_chain=[dagger],
            replay_atomic=None,
            dagger_atomic=None,
            protocol="descriptive",
            output=output,
        )
    )

    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8")) == result
    assert result["full_chain"]["unconditional_paired"]["grasp"][
        "dagger_wins"
    ] == 1
    assert result["input_receipts"]["full_chain"]["replay"][0][
        "episodes_sha256"
    ]


def test_cli_rejects_summary_that_cannot_be_recomputed(tmp_path: Path) -> None:
    replay = tmp_path / "replay"
    dagger = tmp_path / "dagger"
    _write_full_chain_run(replay, [_rollout(40_000, completed=1)], checkpoint_sha256="r" * 64)
    _write_full_chain_run(dagger, [_rollout(40_000, completed=2)], checkpoint_sha256="g" * 64)
    summary = json.loads((dagger / "summary.json").read_text(encoding="utf-8"))
    summary["overall"]["skill_successes"]["grasp"] = 0
    (dagger / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="无法从 episodes 精确重算"):
        run(
            argparse.Namespace(
                replay_full_chain=[replay],
                dagger_full_chain=[dagger],
                replay_atomic=None,
                dagger_atomic=None,
                protocol="descriptive",
                output=tmp_path / "analysis.json",
            )
        )


def test_checkpoint_selection_cli_binds_metrics_checkpoint_hash_and_evaluation(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "training"
    checkpoint_root = run_root / "checkpoints"
    checkpoint_root.mkdir(parents=True)
    initial_sha256 = "i" * 64
    (run_root / "experiment.json").write_text(
        json.dumps(
            {
                "initialization": {
                    "mode": "init_checkpoint",
                    "checkpoint": {"sha256": initial_sha256},
                }
            }
        ),
        encoding="utf-8",
    )
    metrics = []
    checkpoint_hashes: dict[int, str] = {}
    for epoch in range(1, 31):
        metrics.append(
            json.dumps(
                {
                    "event": "epoch",
                    "epoch": epoch,
                    "train": {"optimizer_steps": 64},
                    "validation": {"loss": 1.0 / epoch},
                }
            )
        )
        if epoch in (10, 20, 30):
            checkpoint = checkpoint_root / f"step-{epoch * 64:08d}.pt"
            checkpoint.write_bytes(f"epoch-{epoch}".encode())
            checkpoint_hashes[epoch] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (run_root / "metrics.jsonl").write_text("\n".join(metrics) + "\n", encoding="utf-8")

    baseline_full_results = [
        _rollout(seed, completed=2) for seed in range(31_000, 31_020)
    ]
    baseline_atomic_results = [
        _atomic(seed, skill)
        for skill in PICK_AND_PLACE_SKILLS
        for seed in range(31_020, 31_025)
    ]
    baseline_full = tmp_path / "baseline-full"
    baseline_atomic = tmp_path / "baseline-atomic"
    _write_full_chain_run(
        baseline_full,
        baseline_full_results,
        checkpoint_sha256=initial_sha256,
    )
    _write_atomic_run(
        baseline_atomic,
        baseline_atomic_results,
        checkpoint_sha256=initial_sha256,
    )

    candidate_args: list[list[str]] = []
    for epoch, lift_successes in ((10, 0), (20, 1), (30, 1)):
        full = tmp_path / f"e{epoch}-full"
        atomic = tmp_path / f"e{epoch}-atomic"
        full_results = [
            _rollout(seed, completed=3 if index < lift_successes else 2)
            for index, seed in enumerate(range(31_000, 31_020))
        ]
        _write_full_chain_run(
            full,
            full_results,
            checkpoint_sha256=checkpoint_hashes[epoch],
        )
        _write_atomic_run(
            atomic,
            baseline_atomic_results,
            checkpoint_sha256=checkpoint_hashes[epoch],
        )
        candidate_args.append(
            [f"e{epoch:03d}", str(epoch), str(full), str(atomic)]
        )

    output = tmp_path / "selection.json"
    result = run_checkpoint_selection(
        argparse.Namespace(
            run=run_root,
            baseline_full_chain=baseline_full,
            baseline_atomic=baseline_atomic,
            candidate=candidate_args,
            output=output,
        )
    )

    assert output.is_file()
    assert result["selected"]["label"] == "e030"
    assert result["input_receipts"]["candidates"][2]["checkpoint"][
        "sha256"
    ] == checkpoint_hashes[30]
