"""独立验证 E010 raw Gram、聚合统计、预注册门槛和恢复身份。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

FORMAT = "robot-vla-skill-gradient-probe/v1"
VALIDATION_FORMAT = "robot-vla-skill-gradient-probe-validation/v1"
ALL_TRAINABLE = "all_trainable"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _wilson(successes: int, total: int) -> list[float]:
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


def _cosine(gram: Sequence[Sequence[float]], left: int, right: int) -> float | None:
    left_norm = float(gram[left][left])
    right_norm = float(gram[right][right])
    if left_norm <= 1e-24 or right_norm <= 1e-24:
        return None
    value = float(gram[left][right]) / math.sqrt(left_norm * right_norm)
    return max(-1.0, min(1.0, value))


def _assert_close(actual: Any, expected: Any, *, context: str, tolerance: float = 1e-12) -> float:
    if actual is None or expected is None:
        if actual is not expected:
            raise AssertionError(f"{context}: {actual!r} != {expected!r}")
        return 0.0
    difference = abs(float(actual) - float(expected))
    if difference > tolerance:
        raise AssertionError(
            f"{context}: {actual!r} != {expected!r}, abs diff={difference}"
        )
    return difference


def _validate_gram(
    gram: Any,
    *,
    size: int,
    context: str,
) -> list[list[float]]:
    if not isinstance(gram, list) or len(gram) != size:
        raise AssertionError(f"{context}: Gram 行数错误")
    resolved: list[list[float]] = []
    for row_index, row in enumerate(gram):
        if not isinstance(row, list) or len(row) != size:
            raise AssertionError(f"{context}: Gram 第 {row_index} 行长度错误")
        values = [float(value) for value in row]
        if not all(math.isfinite(value) for value in values):
            raise AssertionError(f"{context}: Gram 包含 NaN/Inf")
        resolved.append(values)
    for left in range(size):
        if resolved[left][left] < -1e-8:
            raise AssertionError(f"{context}: Gram 对角线为负")
        for right in range(size):
            _assert_close(
                resolved[left][right],
                resolved[right][left],
                context=f"{context} symmetry[{left},{right}]",
                tolerance=1e-8,
            )
    return resolved


def _identity(record: Mapping[str, Any]) -> tuple[str, str, int]:
    return (
        str(record["checkpoint_label"]),
        str(record["stage"]),
        int(record["repeat"]),
    )


def _row_key(row: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row["checkpoint_label"]),
        str(row["stage"]),
        str(row["group"]),
        str(row["left"]),
        str(row["right"]),
    )


def _meets(
    rows: Mapping[tuple[str, str, str, str, str], Mapping[str, Any]],
    checkpoint: str,
    stage: str,
    group: str,
    left: str,
    right: str,
    *,
    repeats: int,
    negatives: int,
) -> bool:
    row = rows.get((checkpoint, stage, group, left, right))
    if row is None:
        row = rows.get((checkpoint, stage, group, right, left))
    return bool(
        row
        and row["available_repeats"] == repeats
        and row["median_cosine"] is not None
        and float(row["median_cosine"]) <= -0.10
        and int(row["negative_count"]) >= negatives
    )


def _assess(
    rows: Mapping[tuple[str, str, str, str, str], Mapping[str, Any]],
    confirmation_labels: Sequence[str],
) -> dict[str, Any]:
    assessments: dict[str, Any] = {}
    confirmed_labels: list[str] = []
    for checkpoint in confirmation_labels:
        discovery = _meets(
            rows,
            checkpoint,
            "discovery",
            ALL_TRAINABLE,
            "reach",
            "transport",
            repeats=8,
            negatives=6,
        )
        confirmation = _meets(
            rows,
            checkpoint,
            "confirmation",
            ALL_TRAINABLE,
            "reach_total",
            "transport_total",
            repeats=5,
            negatives=4,
        )
        confirmed = discovery and confirmation
        if confirmed:
            confirmed_labels.append(checkpoint)

        def group_meets(
            group: str, *, resolved_checkpoint: str = checkpoint
        ) -> bool:
            return _meets(
                rows,
                resolved_checkpoint,
                "confirmation",
                group,
                "reach_total",
                "transport_total",
                repeats=5,
                negatives=4,
            )

        block_flags = [group_meets(f"block_{index:02d}") for index in range(16)]
        early_count = sum(block_flags[:12])
        late_count = sum(block_flags[12:])
        total_count = sum(block_flags)
        adapter = group_meets("adapter")
        velocity = group_meets("velocity_head")
        within = {
            skill: _meets(
                rows,
                checkpoint,
                "confirmation",
                ALL_TRAINABLE,
                f"{skill}_base",
                f"{skill}_event",
                repeats=5,
                negatives=4,
            )
            for skill in ("reach", "transport")
        }
        labels: list[str] = []
        if confirmed:
            if adapter:
                labels.append("adapter_conflict")
            if total_count >= 8:
                labels.append("broad_expert_conflict")
            elif late_count >= 3 and early_count < 4:
                labels.append("late_expert_conflict")
            if velocity and not adapter and total_count < 4:
                labels.append("output_head_localized")
        if any(within.values()):
            labels.append("within_skill_base_event_conflict")
        if discovery and not confirmation:
            labels.append("train_only_unconfirmed")
        if not confirmed and not labels:
            labels.append("no_confirmed_gradient_conflict")
        assessments[checkpoint] = {
            "discovery_threshold_met": discovery,
            "confirmation_threshold_met": confirmation,
            "confirmed_gradient_conflict": confirmed,
            "adapter_conflict": adapter if confirmed else False,
            "velocity_head_conflict": velocity if confirmed else False,
            "early_block_conflict_count": early_count if confirmed else 0,
            "late_block_conflict_count": late_count if confirmed else 0,
            "total_block_conflict_count": total_count if confirmed else 0,
            "within_skill_base_event_conflict": within,
            "diagnostic_labels": labels,
        }
    return {
        "confirmed_checkpoint_labels": confirmed_labels,
        "checkpoint_assessments": assessments,
    }


def validate(root: Path) -> dict[str, Any]:
    manifest_path = root / "probe-manifest.json"
    measurements_path = root / "measurements.jsonl"
    summary_path = root / "probe-summary.json"
    manifest = _load_json(manifest_path)
    summary = _load_json(summary_path)
    records = [
        json.loads(line)
        for line in measurements_path.read_text(encoding="utf-8").splitlines()
    ]
    if manifest["format"] != FORMAT or summary["format"] != FORMAT:
        raise AssertionError("E010 format 不兼容")
    if summary.get("complete") is not True:
        raise AssertionError("probe-summary.json 尚未完成")
    if (
        manifest.get("qwen_trainable") is not False
        or int(manifest.get("qwen_context_layer", -1)) != 12
        or float(manifest["objective"]["event_loss_weight"]) != 0.25
        or int(manifest["objective"]["executed_action_steps"]) != 4
        or manifest["objective"]["optimizer_step"] is not False
    ):
        raise AssertionError("E010 frozen model/objective 契约不一致")

    expected_values = [
        (
            str(value["checkpoint_label"]),
            str(value["stage"]),
            int(value["repeat"]),
        )
        for value in manifest["expected_measurement_identities"]
    ]
    expected = set(expected_values)
    if len(expected) != len(expected_values):
        raise AssertionError("manifest expected measurement identity 重复")
    indexed: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for record in records:
        if record.get("format") != FORMAT:
            raise AssertionError("measurement format 不兼容")
        identity = _identity(record)
        if identity in indexed:
            raise AssertionError(f"measurement identity 重复: {identity}")
        indexed[identity] = record
    if set(indexed) != expected:
        raise AssertionError(
            f"measurement identity 不完整: missing={sorted(expected - set(indexed))}, "
            f"extra={sorted(set(indexed) - expected)}"
        )
    if not (
        summary["completed_measurements"]
        == summary["expected_measurements"]
        == len(records)
    ):
        raise AssertionError("summary completed_measurements 错误")

    primitive_groups = [str(group) for group in manifest["gradient_groups"]]
    grouped_cosines: dict[
        tuple[str, str, str, str, str], list[tuple[int, float | None]]
    ] = defaultdict(list)
    max_all_trainable_abs_diff = 0.0
    for identity, record in sorted(indexed.items()):
        labels = [str(label) for label in record["vector_labels"]]
        size = len(labels)
        groups = record["group_grams"]
        if set(groups) != set(primitive_groups) | {ALL_TRAINABLE}:
            raise AssertionError(f"{identity}: gradient group 不完整")
        validated = {
            group: _validate_gram(
                groups[group], size=size, context=f"{identity}/{group}"
            )
            for group in groups
        }
        for left in range(size):
            for right in range(size):
                expected_total = sum(
                    validated[group][left][right] for group in primitive_groups
                )
                difference = _assert_close(
                    validated[ALL_TRAINABLE][left][right],
                    expected_total,
                    context=f"{identity}/all_trainable[{left},{right}]",
                    tolerance=1e-6,
                )
                max_all_trainable_abs_diff = max(
                    max_all_trainable_abs_diff, difference
                )
        for group, gram in validated.items():
            for left in range(size):
                for right in range(left + 1, size):
                    grouped_cosines[
                        (identity[0], identity[1], group, labels[left], labels[right])
                    ].append((identity[2], _cosine(gram, left, right)))

    independent_rows: dict[
        tuple[str, str, str, str, str], dict[str, Any]
    ] = {}
    for key, repeat_values in grouped_cosines.items():
        ordered = sorted(repeat_values)
        available = [value for _, value in ordered if value is not None]
        negative_count = sum(value < 0 for value in available)
        independent_rows[key] = {
            "checkpoint_label": key[0],
            "stage": key[1],
            "group": key[2],
            "left": key[3],
            "right": key[4],
            "repeats": len(ordered),
            "available_repeats": len(available),
            "cosines_by_repeat": [
                {"repeat": repeat, "cosine": cosine}
                for repeat, cosine in ordered
            ],
            "median_cosine": statistics.median(available) if available else None,
            "q25_cosine": _quantile(available, 0.25) if available else None,
            "q75_cosine": _quantile(available, 0.75) if available else None,
            "minimum_cosine": min(available) if available else None,
            "maximum_cosine": max(available) if available else None,
            "negative_count": negative_count,
            "negative_fraction": negative_count / len(available) if available else None,
            "negative_fraction_wilson_95": (
                _wilson(negative_count, len(available)) if available else None
            ),
        }

    summary_rows = {_row_key(row): row for row in summary["summary_rows"]}
    if len(summary_rows) != len(summary["summary_rows"]):
        raise AssertionError("summary_rows key 重复")
    if set(summary_rows) != set(independent_rows):
        raise AssertionError("summary_rows key 集合不一致")
    max_summary_abs_diff = 0.0
    numeric_fields = (
        "median_cosine",
        "q25_cosine",
        "q75_cosine",
        "minimum_cosine",
        "maximum_cosine",
        "negative_fraction",
    )
    for key, independent in independent_rows.items():
        observed = summary_rows[key]
        for field in ("repeats", "available_repeats", "negative_count"):
            if observed[field] != independent[field]:
                raise AssertionError(f"{key}/{field}: 聚合计数不一致")
        for field in numeric_fields:
            max_summary_abs_diff = max(
                max_summary_abs_diff,
                _assert_close(
                    observed[field], independent[field], context=f"{key}/{field}"
                ),
            )
        for index, (observed_item, independent_item) in enumerate(
            zip(
                observed["cosines_by_repeat"],
                independent["cosines_by_repeat"],
                strict=True,
            )
        ):
            if observed_item["repeat"] != independent_item["repeat"]:
                raise AssertionError(f"{key}/repeat[{index}] 不一致")
            max_summary_abs_diff = max(
                max_summary_abs_diff,
                _assert_close(
                    observed_item["cosine"],
                    independent_item["cosine"],
                    context=f"{key}/cosine[{index}]",
                ),
            )
        for index, (observed_value, independent_value) in enumerate(
            zip(
                observed["negative_fraction_wilson_95"] or (),
                independent["negative_fraction_wilson_95"] or (),
                strict=True,
            )
        ):
            max_summary_abs_diff = max(
                max_summary_abs_diff,
                _assert_close(
                    observed_value,
                    independent_value,
                    context=f"{key}/wilson[{index}]",
                ),
            )

    plan_by_stage = {
        "discovery": manifest["discovery"],
        "confirmation": manifest["confirmation"],
    }
    coverage: dict[str, int] = {}
    for stage, stage_manifest in plan_by_stage.items():
        plans = {int(plan["repeat"]): plan for plan in stage_manifest["sample_plan"]}
        episodes = [
            str(episode)
            for plan in plans.values()
            for episode in plan["episodes"]
        ]
        if len(episodes) != len(set(episodes)):
            raise AssertionError(f"{stage}: trajectory 跨 repeat 重复")
        coverage[stage] = len(episodes)
        stage_records = [
            record for identity, record in indexed.items() if identity[1] == stage
        ]
        for record in stage_records:
            plan = plans[int(record["repeat"])]
            if record["episodes"] != plan["episodes"]:
                raise AssertionError(f"{_identity(record)}: episodes 与 manifest 不一致")
            if record["sample_indices_by_skill"] != plan["batches"]:
                raise AssertionError(f"{_identity(record)}: sample indices 与 manifest 不一致")
            expected_flow_seed = int(stage_manifest["flow_seed_base"]) + int(record["repeat"])
            if int(record["flow_seed"]) != expected_flow_seed:
                raise AssertionError(f"{_identity(record)}: Flow seed 不一致")
    discovery_episodes = {
        episode
        for plan in manifest["discovery"]["sample_plan"]
        for episode in plan["episodes"]
    }
    confirmation_episodes = {
        episode
        for plan in manifest["confirmation"]["sample_plan"]
        for episode in plan["episodes"]
    }
    if discovery_episodes & confirmation_episodes:
        raise AssertionError("train discovery 与 val confirmation trajectory 重叠")

    integrity = summary["parameter_integrity"]
    for checkpoint, values in integrity.items():
        if values["parameter_sha256_before"] != values["parameter_sha256_after"]:
            raise AssertionError(f"{checkpoint}: parameter SHA256 前后不一致")
        if values["parameter_versions_unchanged"] is not True:
            raise AssertionError(f"{checkpoint}: parameter version 发生变化")
        for identity, record in indexed.items():
            if identity[0] == checkpoint and (
                record["parameter_sha256_before_checkpoint"]
                != values["parameter_sha256_before"]
            ):
                raise AssertionError(f"{identity}: measurement parameter hash 不一致")

    assessment = _assess(
        independent_rows,
        [str(value) for value in manifest["confirmation_checkpoint_labels"]],
    )
    if assessment != summary["assessment"]:
        raise AssertionError("预注册 assessment 独立复算不一致")

    return {
        "format": VALIDATION_FORMAT,
        "ready_to_share": True,
        "records_verified": len(records),
        "summary_rows_verified": len(independent_rows),
        "trajectory_coverage": coverage,
        "train_val_trajectory_overlap": 0,
        "max_all_trainable_abs_diff": max_all_trainable_abs_diff,
        "max_summary_abs_diff": max_summary_abs_diff,
        "parameter_integrity_verified": True,
        "assessment": assessment,
        "artifact_sha256": {
            manifest_path.name: _sha256(manifest_path),
            measurements_path.name: _sha256(measurements_path),
            summary_path.name: _sha256(summary_path),
        },
        "checks": [
            "measurement identities complete and unique",
            "all Gram matrices finite, symmetric, and non-negative on the diagonal",
            "all_trainable equals the sum of every primitive parameter group",
            "cosine, median, IQR, extrema, negative fraction, and Wilson 95% recomputed",
            "sample identities and Flow seeds match the manifest across checkpoints",
            "train and validation trajectories are disjoint",
            "Adapter/Expert parameter hashes and versions are unchanged",
            "pre-registered conflict labels independently reproduced",
        ],
    }


def main() -> None:
    args = _parse_args()
    result = validate(args.root)
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
