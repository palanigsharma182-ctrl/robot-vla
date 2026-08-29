"""E010 五技能梯度 Gram 聚合、冲突门槛与样本计划。"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from statistics import median
from typing import Any

import numpy as np

from robot_vla.contracts import PICK_AND_PLACE_SKILLS

GRADIENT_PROBE_FORMAT = "robot-vla-skill-gradient-probe/v1"
EXPERT_BLOCK_COUNT = 16
PRIMITIVE_GRADIENT_GROUPS = (
    "adapter",
    "state_encoder",
    "action_encoder",
    *(f"block_{index:02d}" for index in range(EXPERT_BLOCK_COUNT)),
    "final_norm",
    "velocity_head",
)
ALL_TRAINABLE_GROUP = "all_trainable"
CONFIRMATION_PRIMITIVE_LABELS = (
    "reach_base",
    "reach_event",
    "transport_base",
    "transport_event",
)
CONFIRMATION_VECTOR_LABELS = (
    "reach_base",
    "reach_event",
    "reach_total",
    "transport_base",
    "transport_event",
    "transport_total",
)


def gradient_group_for_parameter(name: str) -> str:
    """把 policy trainable parameter name 映射到互斥诊断分组。"""

    if name.startswith("adapter."):
        return "adapter"
    if name.startswith("expert.state_encoder."):
        return "state_encoder"
    if name.startswith("expert.action_encoder."):
        return "action_encoder"
    if name.startswith("expert.final_norm."):
        return "final_norm"
    if name.startswith("expert.velocity_head."):
        return "velocity_head"
    prefix = "expert.blocks."
    if name.startswith(prefix):
        remainder = name[len(prefix) :]
        block_text, separator, _ = remainder.partition(".")
        if not separator or not block_text.isdigit():
            raise ValueError(f"无法解析 Expert block parameter: {name}")
        block = int(block_text)
        if not 0 <= block < EXPERT_BLOCK_COUNT:
            raise ValueError(f"Expert block 超出 E010 固定范围: {name}")
        return f"block_{block:02d}"
    raise ValueError(f"E010 发现未分组的 trainable parameter: {name}")


def build_episode_paired_plan(
    samples_by_episode: Mapping[str, Mapping[str, Sequence[int]]],
    *,
    skills: Sequence[str],
    repeats: int,
    batch_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    """用不重复 Episode 构造同场景、跨技能配对的 sample index 计划。"""

    resolved_skills = tuple(str(skill) for skill in skills)
    if (
        repeats <= 0
        or batch_size <= 0
        or seed < 0
        or not resolved_skills
        or len(set(resolved_skills)) != len(resolved_skills)
    ):
        raise ValueError("E010 sample plan 参数无效")
    unknown = sorted(set(resolved_skills) - set(PICK_AND_PLACE_SKILLS))
    if unknown:
        raise ValueError(f"E010 sample plan 包含未知技能: {unknown}")
    eligible = sorted(
        episode
        for episode, by_skill in samples_by_episode.items()
        if all(by_skill.get(skill) for skill in resolved_skills)
    )
    required = repeats * batch_size
    if len(eligible) < required:
        raise ValueError(
            f"E010 需要 {required} 个覆盖全部技能的不同 Episode，实际只有 {len(eligible)}"
        )
    generator = np.random.default_rng(seed)
    selected = generator.permutation(np.asarray(eligible, dtype=object))[:required].tolist()
    plan: list[dict[str, Any]] = []
    for repeat in range(repeats):
        episodes = selected[repeat * batch_size : (repeat + 1) * batch_size]
        batches: dict[str, list[int]] = {}
        for skill in resolved_skills:
            batches[skill] = [
                int(generator.choice(samples_by_episode[str(episode)][skill]))
                for episode in episodes
            ]
        plan.append(
            {
                "repeat": repeat,
                "episodes": [str(episode) for episode in episodes],
                "batches": batches,
            }
        )
    return plan


def _as_gram(value: Any, *, size: int, name: str) -> np.ndarray:
    gram = np.asarray(value, dtype=np.float64)
    if gram.shape != (size, size):
        raise ValueError(f"{name} Gram 应为 {(size, size)}，实际为 {gram.shape}")
    if not np.isfinite(gram).all():
        raise ValueError(f"{name} Gram 包含 NaN/Inf")
    if not np.allclose(gram, gram.T, rtol=1e-6, atol=1e-8):
        raise ValueError(f"{name} Gram 不是对称矩阵")
    if np.any(np.diag(gram) < -1e-8):
        raise ValueError(f"{name} Gram 对角线不能为负")
    return gram


def add_all_trainable_gram(
    group_grams: Mapping[str, Any], *, vector_count: int
) -> dict[str, list[list[float]]]:
    """验证互斥分组并增加由所有 primitive Gram 求和得到的总 Gram。"""

    actual = set(group_grams)
    primitive = set(PRIMITIVE_GRADIENT_GROUPS)
    if actual == primitive | {ALL_TRAINABLE_GROUP}:
        supplied_all = _as_gram(
            group_grams[ALL_TRAINABLE_GROUP],
            size=vector_count,
            name=ALL_TRAINABLE_GROUP,
        )
        selected = {key: value for key, value in group_grams.items() if key != ALL_TRAINABLE_GROUP}
    elif actual == primitive:
        supplied_all = None
        selected = dict(group_grams)
    else:
        missing = sorted(primitive - actual)
        extra = sorted(actual - primitive - {ALL_TRAINABLE_GROUP})
        raise ValueError(f"E010 gradient group 不完整：missing={missing}, extra={extra}")
    resolved: dict[str, list[list[float]]] = {}
    total = np.zeros((vector_count, vector_count), dtype=np.float64)
    for group in PRIMITIVE_GRADIENT_GROUPS:
        gram = _as_gram(selected[group], size=vector_count, name=group)
        total += gram
        resolved[group] = gram.tolist()
    if supplied_all is not None and not np.allclose(
        supplied_all, total, rtol=1e-5, atol=1e-6
    ):
        raise ValueError("all_trainable Gram 不等于 primitive group Gram 之和")
    resolved[ALL_TRAINABLE_GROUP] = total.tolist()
    return resolved


def expand_confirmation_grams(
    primitive_group_grams: Mapping[str, Any],
) -> dict[str, list[list[float]]]:
    """从 Reach/Transport 的 base/event primitive Gram 线性构造 total Gram。"""

    transform = np.asarray(
        (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 1.0, 1.0),
        ),
        dtype=np.float64,
    )
    actual = set(primitive_group_grams)
    expected = set(PRIMITIVE_GRADIENT_GROUPS)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"E010 confirmation primitive gradient group 不完整："
            f"missing={missing}, extra={extra}"
        )
    expanded = {
        group: (transform @ _as_gram(gram, size=4, name=group) @ transform.T).tolist()
        for group, gram in primitive_group_grams.items()
    }
    return add_all_trainable_gram(expanded, vector_count=len(CONFIRMATION_VECTOR_LABELS))


def gram_cosine(gram: Any, left: int, right: int) -> float | None:
    matrix = np.asarray(gram, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Gram 必须是方阵")
    if not 0 <= left < matrix.shape[0] or not 0 <= right < matrix.shape[0]:
        raise IndexError("Gram cosine index 超出范围")
    left_norm_sq = float(matrix[left, left])
    right_norm_sq = float(matrix[right, right])
    if left_norm_sq <= 1e-24 or right_norm_sq <= 1e-24:
        return None
    value = float(matrix[left, right]) / math.sqrt(left_norm_sq * right_norm_sq)
    return max(-1.0, min(1.0, value))


def wilson_interval(successes: int, total: int) -> list[float]:
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Wilson interval 计数无效")
    z = 1.959963984540054
    rate = successes / total
    scale = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / scale
    half = (
        z
        * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total))
        / scale
    )
    return [max(0.0, center - half), min(1.0, center + half)]


def measurement_identity(record: Mapping[str, Any]) -> tuple[str, str, int]:
    try:
        checkpoint = str(record["checkpoint_label"])
        stage = str(record["stage"])
        repeat = int(record["repeat"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("E010 measurement identity 无效") from error
    if not checkpoint or stage not in {"discovery", "confirmation"} or repeat < 0:
        raise ValueError("E010 measurement identity 无效")
    return checkpoint, stage, repeat


def index_measurements(
    records: Iterable[Mapping[str, Any]],
) -> dict[tuple[str, str, int], Mapping[str, Any]]:
    indexed: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for record in records:
        if record.get("format") != GRADIENT_PROBE_FORMAT:
            raise ValueError("E010 measurement format 不兼容")
        identity = measurement_identity(record)
        if identity in indexed:
            raise ValueError(f"E010 measurement identity 重复: {identity}")
        indexed[identity] = record
    return indexed


def _quantile(values: Sequence[float], fraction: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), fraction))


def summarize_measurements(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """从 raw Gram 生成可独立复算的逐 pair/group 统计。"""

    indexed = index_measurements(records)
    grouped: dict[tuple[str, str, str, str, str], list[tuple[int, float | None]]] = {}
    for identity, record in sorted(indexed.items()):
        checkpoint, stage, repeat = identity
        labels = [str(label) for label in record.get("vector_labels", [])]
        if len(labels) < 2 or len(set(labels)) != len(labels):
            raise ValueError(f"E010 {identity} vector_labels 无效")
        groups = add_all_trainable_gram(
            record.get("group_grams", {}), vector_count=len(labels)
        )
        for group, gram in groups.items():
            for left in range(len(labels)):
                for right in range(left + 1, len(labels)):
                    key = (checkpoint, stage, group, labels[left], labels[right])
                    grouped.setdefault(key, []).append(
                        (repeat, gram_cosine(gram, left, right))
                    )

    rows: list[dict[str, Any]] = []
    for key, values_by_repeat in sorted(grouped.items()):
        checkpoint, stage, group, left, right = key
        ordered = sorted(values_by_repeat)
        available = [value for _, value in ordered if value is not None]
        negative_count = sum(value < 0 for value in available)
        rows.append(
            {
                "checkpoint_label": checkpoint,
                "stage": stage,
                "group": group,
                "left": left,
                "right": right,
                "repeats": len(ordered),
                "available_repeats": len(available),
                "cosines_by_repeat": [
                    {"repeat": repeat, "cosine": value}
                    for repeat, value in ordered
                ],
                "median_cosine": median(available) if available else None,
                "q25_cosine": _quantile(available, 0.25) if available else None,
                "q75_cosine": _quantile(available, 0.75) if available else None,
                "minimum_cosine": min(available) if available else None,
                "maximum_cosine": max(available) if available else None,
                "negative_count": negative_count,
                "negative_fraction": (
                    negative_count / len(available) if available else None
                ),
                "negative_fraction_wilson_95": (
                    wilson_interval(negative_count, len(available))
                    if available
                    else None
                ),
            }
        )
    return rows


def _find_row(
    rows: Sequence[Mapping[str, Any]],
    *,
    checkpoint: str,
    stage: str,
    group: str,
    left: str,
    right: str,
) -> Mapping[str, Any] | None:
    target = {left, right}
    selected = [
        row
        for row in rows
        if row["checkpoint_label"] == checkpoint
        and row["stage"] == stage
        and row["group"] == group
        and {str(row["left"]), str(row["right"])} == target
    ]
    if len(selected) > 1:
        raise ValueError("E010 summary pair 重复")
    return selected[0] if selected else None


def _meets_threshold(
    row: Mapping[str, Any] | None, *, required_available: int, required_negative: int
) -> bool:
    if row is None:
        return False
    cosine = row.get("median_cosine")
    return (
        row.get("available_repeats") == required_available
        and isinstance(cosine, (int, float))
        and float(cosine) <= -0.10
        and int(row.get("negative_count", -1)) >= required_negative
    )


def assess_conflict(
    summary_rows: Sequence[Mapping[str, Any]],
    *,
    confirmation_checkpoints: Sequence[str],
) -> dict[str, Any]:
    """按 E010 预注册门槛判断冲突与模块位置。"""

    assessments: dict[str, Any] = {}
    confirmed_labels: list[str] = []
    for checkpoint in confirmation_checkpoints:
        discovery = _find_row(
            summary_rows,
            checkpoint=checkpoint,
            stage="discovery",
            group=ALL_TRAINABLE_GROUP,
            left="reach",
            right="transport",
        )
        confirmation = _find_row(
            summary_rows,
            checkpoint=checkpoint,
            stage="confirmation",
            group=ALL_TRAINABLE_GROUP,
            left="reach_total",
            right="transport_total",
        )
        discovery_met = _meets_threshold(
            discovery, required_available=8, required_negative=6
        )
        confirmation_met = _meets_threshold(
            confirmation, required_available=5, required_negative=4
        )
        confirmed = discovery_met and confirmation_met
        if confirmed:
            confirmed_labels.append(checkpoint)

        def confirmed_group(
            group: str, *, resolved_checkpoint: str = checkpoint
        ) -> bool:
            row = _find_row(
                summary_rows,
                checkpoint=resolved_checkpoint,
                stage="confirmation",
                group=group,
                left="reach_total",
                right="transport_total",
            )
            return _meets_threshold(row, required_available=5, required_negative=4)

        block_flags = {
            f"block_{index:02d}": confirmed_group(f"block_{index:02d}")
            for index in range(EXPERT_BLOCK_COUNT)
        }
        early_count = sum(block_flags[f"block_{index:02d}"] for index in range(12))
        late_count = sum(block_flags[f"block_{index:02d}"] for index in range(12, 16))
        total_block_count = early_count + late_count
        adapter_conflict = confirmed_group("adapter")
        velocity_conflict = confirmed_group("velocity_head")

        within_skill: dict[str, bool] = {}
        for skill in ("reach", "transport"):
            row = _find_row(
                summary_rows,
                checkpoint=checkpoint,
                stage="confirmation",
                group=ALL_TRAINABLE_GROUP,
                left=f"{skill}_base",
                right=f"{skill}_event",
            )
            within_skill[skill] = _meets_threshold(
                row, required_available=5, required_negative=4
            )

        labels: list[str] = []
        if confirmed:
            if adapter_conflict:
                labels.append("adapter_conflict")
            if total_block_count >= 8:
                labels.append("broad_expert_conflict")
            elif late_count >= 3 and early_count < 4:
                labels.append("late_expert_conflict")
            if velocity_conflict and not adapter_conflict and total_block_count < 4:
                labels.append("output_head_localized")
        if any(within_skill.values()):
            labels.append("within_skill_base_event_conflict")
        if discovery_met and not confirmation_met:
            labels.append("train_only_unconfirmed")
        if not confirmed and not labels:
            labels.append("no_confirmed_gradient_conflict")

        assessments[checkpoint] = {
            "discovery_threshold_met": discovery_met,
            "confirmation_threshold_met": confirmation_met,
            "confirmed_gradient_conflict": confirmed,
            "adapter_conflict": adapter_conflict if confirmed else False,
            "velocity_head_conflict": velocity_conflict if confirmed else False,
            "early_block_conflict_count": early_count if confirmed else 0,
            "late_block_conflict_count": late_count if confirmed else 0,
            "total_block_conflict_count": total_block_count if confirmed else 0,
            "within_skill_base_event_conflict": within_skill,
            "diagnostic_labels": labels,
        }
    return {
        "confirmed_checkpoint_labels": confirmed_labels,
        "checkpoint_assessments": assessments,
    }


__all__ = [
    "ALL_TRAINABLE_GROUP",
    "CONFIRMATION_PRIMITIVE_LABELS",
    "CONFIRMATION_VECTOR_LABELS",
    "EXPERT_BLOCK_COUNT",
    "GRADIENT_PROBE_FORMAT",
    "PRIMITIVE_GRADIENT_GROUPS",
    "add_all_trainable_gram",
    "assess_conflict",
    "build_episode_paired_plan",
    "expand_confirmation_grams",
    "gradient_group_for_parameter",
    "gram_cosine",
    "index_measurements",
    "measurement_identity",
    "summarize_measurements",
    "wilson_interval",
]
