import copy

import numpy as np
import pytest

from robot_vla.contracts import PICK_AND_PLACE_SKILLS
from robot_vla.diagnostics.gradient_conflict import (
    ALL_TRAINABLE_GROUP,
    CONFIRMATION_VECTOR_LABELS,
    GRADIENT_PROBE_FORMAT,
    PRIMITIVE_GRADIENT_GROUPS,
    add_all_trainable_gram,
    assess_conflict,
    build_episode_paired_plan,
    expand_confirmation_grams,
    gradient_group_for_parameter,
    gram_cosine,
    index_measurements,
    summarize_measurements,
    wilson_interval,
)


def _zero_primitive_grams(size: int) -> dict[str, list[list[float]]]:
    return {
        group: np.zeros((size, size), dtype=np.float64).tolist()
        for group in PRIMITIVE_GRADIENT_GROUPS
    }


def _measurement(
    *,
    checkpoint: str = "e100",
    stage: str = "discovery",
    repeat: int = 0,
    labels: tuple[str, ...] = ("reach", "transport"),
    cosine: float = -0.5,
) -> dict[str, object]:
    gram = np.eye(len(labels), dtype=np.float64)
    gram[0, 1] = cosine
    gram[1, 0] = cosine
    groups = _zero_primitive_grams(len(labels))
    groups["adapter"] = gram.tolist()
    return {
        "format": GRADIENT_PROBE_FORMAT,
        "checkpoint_label": checkpoint,
        "stage": stage,
        "repeat": repeat,
        "vector_labels": list(labels),
        "group_grams": groups,
    }


def _summary_row(
    checkpoint: str,
    stage: str,
    group: str,
    left: str,
    right: str,
    *,
    repeats: int,
    negative_count: int,
    median_cosine: float,
) -> dict[str, object]:
    return {
        "checkpoint_label": checkpoint,
        "stage": stage,
        "group": group,
        "left": left,
        "right": right,
        "available_repeats": repeats,
        "negative_count": negative_count,
        "median_cosine": median_cosine,
    }


@pytest.mark.parametrize(
    ("name", "expected"),
    (
        ("adapter.projection.weight", "adapter"),
        ("expert.state_encoder.norm.weight", "state_encoder"),
        ("expert.action_encoder.slot_embedding", "action_encoder"),
        ("expert.blocks.0.attention.q_projection.weight", "block_00"),
        ("expert.blocks.15.mlp.down_projection.weight", "block_15"),
        ("expert.final_norm.weight", "final_norm"),
        ("expert.velocity_head.bias", "velocity_head"),
    ),
)
def test_gradient_group_for_parameter(name: str, expected: str) -> None:
    assert gradient_group_for_parameter(name) == expected


@pytest.mark.parametrize(
    "name",
    (
        "context_encoder.qwen.weight",
        "expert.blocks.bad.weight",
        "expert.blocks.16.weight",
        "expert.unregistered.weight",
    ),
)
def test_gradient_group_rejects_unknown_or_out_of_range(name: str) -> None:
    with pytest.raises(ValueError):
        gradient_group_for_parameter(name)


def test_episode_paired_plan_is_deterministic_unique_and_skill_paired() -> None:
    samples = {
        f"trajectory-{episode:02d}": {
            skill: [episode * 100 + skill_index * 10 + offset for offset in range(2)]
            for skill_index, skill in enumerate(PICK_AND_PLACE_SKILLS)
        }
        for episode in range(8)
    }
    first = build_episode_paired_plan(
        samples,
        skills=PICK_AND_PLACE_SKILLS,
        repeats=2,
        batch_size=3,
        seed=19,
    )
    second = build_episode_paired_plan(
        samples,
        skills=PICK_AND_PLACE_SKILLS,
        repeats=2,
        batch_size=3,
        seed=19,
    )
    assert first == second
    selected = [episode for row in first for episode in row["episodes"]]
    assert len(selected) == len(set(selected)) == 6
    for row in first:
        assert set(row["batches"]) == set(PICK_AND_PLACE_SKILLS)
        assert all(len(indices) == 3 for indices in row["batches"].values())


def test_episode_paired_plan_requires_enough_fully_covered_episodes() -> None:
    samples = {
        "complete": {skill: [index] for index, skill in enumerate(PICK_AND_PLACE_SKILLS)},
        "missing-place": {
            skill: [100 + index]
            for index, skill in enumerate(PICK_AND_PLACE_SKILLS[:-1])
        },
    }
    with pytest.raises(ValueError, match="需要 2 个"):
        build_episode_paired_plan(
            samples,
            skills=PICK_AND_PLACE_SKILLS,
            repeats=1,
            batch_size=2,
            seed=0,
        )


def test_add_all_trainable_gram_sums_primitive_groups() -> None:
    groups = _zero_primitive_grams(2)
    groups["adapter"] = [[1.0, -0.25], [-0.25, 2.0]]
    groups["velocity_head"] = [[3.0, 0.5], [0.5, 4.0]]
    resolved = add_all_trainable_gram(groups, vector_count=2)
    assert resolved[ALL_TRAINABLE_GROUP] == [[4.0, 0.25], [0.25, 6.0]]


def test_add_all_trainable_gram_rejects_bad_or_inconsistent_gram() -> None:
    groups = _zero_primitive_grams(2)
    groups["adapter"] = [[1.0, 0.2], [0.1, 1.0]]
    with pytest.raises(ValueError, match="不是对称"):
        add_all_trainable_gram(groups, vector_count=2)

    groups = _zero_primitive_grams(2)
    groups["adapter"] = [[1.0, 0.0], [0.0, float("nan")]]
    with pytest.raises(ValueError, match="NaN/Inf"):
        add_all_trainable_gram(groups, vector_count=2)

    groups = _zero_primitive_grams(2)
    groups[ALL_TRAINABLE_GROUP] = [[1.0, 0.0], [0.0, 1.0]]
    with pytest.raises(ValueError, match="不等于"):
        add_all_trainable_gram(groups, vector_count=2)


def test_confirmation_expansion_constructs_total_vectors_without_extra_gradients() -> None:
    groups = _zero_primitive_grams(4)
    groups["adapter"] = np.eye(4, dtype=np.float64).tolist()
    expanded = expand_confirmation_grams(groups)
    adapter = np.asarray(expanded["adapter"])
    assert adapter.shape == (len(CONFIRMATION_VECTOR_LABELS),) * 2
    assert adapter[2, 2] == pytest.approx(2.0)
    assert adapter[5, 5] == pytest.approx(2.0)
    assert adapter[0, 2] == pytest.approx(1.0)
    assert adapter[2, 5] == pytest.approx(0.0)
    np.testing.assert_allclose(expanded[ALL_TRAINABLE_GROUP], adapter)


def test_gram_cosine_handles_exact_value_clamping_and_zero_norm() -> None:
    assert gram_cosine([[4.0, -3.0], [-3.0, 9.0]], 0, 1) == pytest.approx(-0.5)
    assert gram_cosine([[1.0, 1.000001], [1.000001, 1.0]], 0, 1) == 1.0
    assert gram_cosine([[0.0, 0.0], [0.0, 1.0]], 0, 1) is None


def test_measurement_index_rejects_duplicate_identity() -> None:
    record = _measurement()
    with pytest.raises(ValueError, match="重复"):
        index_measurements([record, copy.deepcopy(record)])


def test_summary_reports_median_iqr_negative_fraction_and_wilson() -> None:
    records = [
        _measurement(repeat=repeat, cosine=cosine)
        for repeat, cosine in enumerate((-0.8, -0.4, 0.2, 0.6))
    ]
    rows = summarize_measurements(records)
    row = next(
        value
        for value in rows
        if value["group"] == "adapter"
        and value["left"] == "reach"
        and value["right"] == "transport"
    )
    assert row["median_cosine"] == pytest.approx(-0.1)
    assert row["q25_cosine"] == pytest.approx(-0.5)
    assert row["q75_cosine"] == pytest.approx(0.3)
    assert row["negative_count"] == 2
    assert row["negative_fraction"] == pytest.approx(0.5)
    assert row["negative_fraction_wilson_95"] == pytest.approx(wilson_interval(2, 4))


def test_assessment_identifies_output_head_localized_conflict() -> None:
    rows = [
        _summary_row(
            "e100",
            "discovery",
            ALL_TRAINABLE_GROUP,
            "reach",
            "transport",
            repeats=8,
            negative_count=7,
            median_cosine=-0.3,
        ),
        _summary_row(
            "e100",
            "confirmation",
            ALL_TRAINABLE_GROUP,
            "reach_total",
            "transport_total",
            repeats=5,
            negative_count=4,
            median_cosine=-0.2,
        ),
        _summary_row(
            "e100",
            "confirmation",
            "velocity_head",
            "reach_total",
            "transport_total",
            repeats=5,
            negative_count=5,
            median_cosine=-0.4,
        ),
    ]
    result = assess_conflict(rows, confirmation_checkpoints=("e100",))
    assessment = result["checkpoint_assessments"]["e100"]
    assert assessment["confirmed_gradient_conflict"] is True
    assert assessment["diagnostic_labels"] == ["output_head_localized"]


def test_assessment_identifies_broad_expert_and_within_skill_conflict() -> None:
    rows = [
        _summary_row(
            "e100",
            "discovery",
            ALL_TRAINABLE_GROUP,
            "reach",
            "transport",
            repeats=8,
            negative_count=8,
            median_cosine=-0.5,
        ),
        _summary_row(
            "e100",
            "confirmation",
            ALL_TRAINABLE_GROUP,
            "reach_total",
            "transport_total",
            repeats=5,
            negative_count=5,
            median_cosine=-0.4,
        ),
        _summary_row(
            "e100",
            "confirmation",
            ALL_TRAINABLE_GROUP,
            "reach_base",
            "reach_event",
            repeats=5,
            negative_count=4,
            median_cosine=-0.2,
        ),
    ]
    rows.extend(
        _summary_row(
            "e100",
            "confirmation",
            f"block_{index:02d}",
            "reach_total",
            "transport_total",
            repeats=5,
            negative_count=4,
            median_cosine=-0.2,
        )
        for index in range(8)
    )
    result = assess_conflict(rows, confirmation_checkpoints=("e100",))
    labels = result["checkpoint_assessments"]["e100"]["diagnostic_labels"]
    assert "broad_expert_conflict" in labels
    assert "within_skill_base_event_conflict" in labels


def test_discovery_only_conflict_is_not_architecture_confirmation() -> None:
    rows = [
        _summary_row(
            "e098-best",
            "discovery",
            ALL_TRAINABLE_GROUP,
            "reach",
            "transport",
            repeats=8,
            negative_count=6,
            median_cosine=-0.1,
        ),
        _summary_row(
            "e098-best",
            "confirmation",
            ALL_TRAINABLE_GROUP,
            "reach_total",
            "transport_total",
            repeats=5,
            negative_count=3,
            median_cosine=-0.2,
        ),
    ]
    result = assess_conflict(rows, confirmation_checkpoints=("e098-best",))
    assessment = result["checkpoint_assessments"]["e098-best"]
    assert assessment["confirmed_gradient_conflict"] is False
    assert assessment["diagnostic_labels"] == ["train_only_unconfirmed"]
