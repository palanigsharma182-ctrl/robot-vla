"""Layer 12 periodic checkpoint 的原子技能 sweep 聚合与选择。"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from robot_vla.evaluation.atomic import AtomicSkillEpisodeResult

CHECKPOINT_SWEEP_FORMAT = "robot-vla-checkpoint-sweep/v1"
SWEEP_SKILL_THRESHOLDS_M = {
    "reach": ("final_tcp_to_object_distance_m", 0.04),
    "transport": ("final_object_to_goal_xy_distance_m", 0.04),
}


@dataclass(frozen=True)
class CheckpointSweepCandidate:
    label: str
    epoch: int
    optimizer_steps: int
    checkpoint: Path
    validation_loss: float
    validation_base_loss: float
    validation_event_loss: float
    is_validation_best: bool = False

    def __post_init__(self) -> None:
        if not self.label or self.epoch <= 0 or self.optimizer_steps <= 0:
            raise ValueError("Checkpoint sweep 候选身份无效")
        losses = (
            self.validation_loss,
            self.validation_base_loss,
            self.validation_event_loss,
        )
        if any(not math.isfinite(value) or value < 0 for value in losses):
            raise ValueError("Checkpoint sweep validation loss 必须为有限非负数")
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"找不到 sweep Checkpoint: {self.checkpoint}")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["checkpoint"] = str(self.checkpoint)
        return payload


def _read_epoch_metrics(run_dir: Path) -> list[dict[str, Any]]:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"训练目录缺少 metrics.jsonl: {run_dir}")
    rows: list[dict[str, Any]] = []
    expected_epoch = 1
    cumulative_steps = 0
    for line_number, line in enumerate(
        metrics_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"metrics.jsonl 第 {line_number} 行无效") from error
        if row.get("event") != "epoch":
            continue
        epoch = int(row["epoch"])
        if epoch != expected_epoch:
            raise ValueError(f"训练 epoch 不连续：期望 {expected_epoch}，实际 {epoch}")
        expected_epoch += 1
        optimizer_steps = int(row["train"]["optimizer_steps"])
        if optimizer_steps <= 0:
            raise ValueError("每个 epoch 的 optimizer_steps 必须为正数")
        cumulative_steps += optimizer_steps
        validation = row["validation"]
        rows.append(
            {
                "epoch": epoch,
                "optimizer_steps": cumulative_steps,
                "validation_loss": float(validation["loss"]),
                "validation_base_loss": float(validation["base_loss"]),
                "validation_event_loss": float(validation["event_loss"]),
            }
        )
    if not rows:
        raise ValueError("metrics.jsonl 没有 epoch 指标")
    return rows


def discover_sweep_candidates(
    run_dir: str | Path,
) -> tuple[list[CheckpointSweepCandidate], str]:
    """发现所有实际 periodic 权重和非重复 validation best。"""

    directory = Path(run_dir).resolve()
    checkpoint_dir = directory / "checkpoints"
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"训练目录缺少 checkpoints: {directory}")
    rows = _read_epoch_metrics(directory)
    best_row = min(rows, key=lambda row: (row["validation_loss"], row["epoch"]))
    best_epoch = int(best_row["epoch"])
    candidates: list[CheckpointSweepCandidate] = []
    periodic_epochs: set[int] = set()
    for row in rows:
        checkpoint = checkpoint_dir / f"step-{int(row['optimizer_steps']):08d}.pt"
        if not checkpoint.is_file():
            continue
        epoch = int(row["epoch"])
        periodic_epochs.add(epoch)
        candidates.append(
            CheckpointSweepCandidate(
                label=f"e{epoch:03d}",
                epoch=epoch,
                optimizer_steps=int(row["optimizer_steps"]),
                checkpoint=checkpoint.resolve(),
                validation_loss=float(row["validation_loss"]),
                validation_base_loss=float(row["validation_base_loss"]),
                validation_event_loss=float(row["validation_event_loss"]),
                is_validation_best=epoch == best_epoch,
            )
        )
    if not candidates:
        raise ValueError("训练目录没有与 epoch 指标对应的 periodic Checkpoint")

    if best_epoch in periodic_epochs:
        anchor_label = f"e{best_epoch:03d}"
    else:
        best_checkpoint = checkpoint_dir / "best.pt"
        candidates.append(
            CheckpointSweepCandidate(
                label=f"e{best_epoch:03d}-best",
                epoch=best_epoch,
                optimizer_steps=int(best_row["optimizer_steps"]),
                checkpoint=best_checkpoint.resolve(),
                validation_loss=float(best_row["validation_loss"]),
                validation_base_loss=float(best_row["validation_base_loss"]),
                validation_event_loss=float(best_row["validation_event_loss"]),
                is_validation_best=True,
            )
        )
        anchor_label = f"e{best_epoch:03d}-best"
    candidates.sort(key=lambda item: (item.epoch, item.label))
    if len({item.label for item in candidates}) != len(candidates):
        raise ValueError("Checkpoint sweep 候选 label 重复")
    return candidates, anchor_label


def read_atomic_results(path: str | Path) -> list[AtomicSkillEpisodeResult]:
    rows: list[AtomicSkillEpisodeResult] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            rows.append(AtomicSkillEpisodeResult.from_dict(json.loads(line)))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"原子结果第 {line_number} 行无效: {error}") from error
    if not rows:
        raise ValueError("Checkpoint sweep 原子结果为空")
    identities = {(row.skill_name, row.seed) for row in rows}
    if len(identities) != len(rows):
        raise ValueError("Checkpoint sweep 原子结果包含重复 skill/seed")
    return rows


def _wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054
) -> list[float]:
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Wilson interval 计数无效")
    rate = successes / total
    scale = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / scale
    half = (
        z
        * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total))
        / scale
    )
    return [max(0.0, center - half), min(1.0, center + half)]


def predicate_residual(result: AtomicSkillEpisodeResult) -> float:
    try:
        field, threshold = SWEEP_SKILL_THRESHOLDS_M[result.skill_name]
    except KeyError as error:
        raise ValueError(f"Sweep 不支持技能 {result.skill_name}") from error
    return max(float(getattr(result, field)) - threshold, 0.0)


def summarize_candidate_results(
    candidate: CheckpointSweepCandidate,
    results: Iterable[AtomicSkillEpisodeResult],
) -> dict[str, Any]:
    selected = list(results)
    if not selected:
        raise ValueError("不能汇总空 checkpoint 结果")
    groups: dict[str, Any] = {}
    for skill_name in SWEEP_SKILL_THRESHOLDS_M:
        skill_results = [row for row in selected if row.skill_name == skill_name]
        if not skill_results:
            continue
        successes = sum(row.success for row in skill_results)
        system_issue_count = sum(
            row.error is not None or row.failure_stage is not None for row in skill_results
        )
        saturation_count = sum(
            row.tracking_correction_saturation_count for row in skill_results
        )
        anomaly_count = sum(row.anomaly_replan_count for row in skill_results)
        groups[skill_name] = {
            "episodes": len(skill_results),
            "successes": successes,
            "success_rate": successes / len(skill_results),
            "success_rate_wilson_95": _wilson_interval(successes, len(skill_results)),
            "mean_predicate_residual_m": sum(
                predicate_residual(row) for row in skill_results
            )
            / len(skill_results),
            "mean_policy_environment_steps": sum(
                row.policy_environment_steps for row in skill_results
            )
            / len(skill_results),
            "failure_counts": dict(
                sorted(
                    Counter(
                        row.failure_category for row in skill_results if not row.success
                    ).items()
                )
            ),
            "system_issue_count": system_issue_count,
            "tracking_correction_saturation_count": saturation_count,
            "anomaly_replan_count": anomaly_count,
            "behavior_comparable": (
                system_issue_count == 0 and saturation_count == 0 and anomaly_count == 0
            ),
            "per_seed": {
                str(row.seed): {
                    "success": row.success,
                    "predicate_residual_m": predicate_residual(row),
                    "policy_environment_steps": row.policy_environment_steps,
                    "failure_category": row.failure_category,
                }
                for row in sorted(skill_results, key=lambda item: item.seed)
            },
        }
    unknown = sorted({row.skill_name for row in selected} - set(SWEEP_SKILL_THRESHOLDS_M))
    if unknown:
        raise ValueError(f"Sweep 结果包含未配置技能: {unknown}")
    return {
        "candidate": candidate.to_dict(),
        "groups": groups,
    }


def rank_candidate_summaries(
    summaries: Iterable[dict[str, Any]], skill_name: str
) -> list[dict[str, Any]]:
    if skill_name not in SWEEP_SKILL_THRESHOLDS_M:
        raise ValueError(f"Sweep 不支持技能 {skill_name}")
    selected = [summary for summary in summaries if skill_name in summary["groups"]]
    if not selected:
        raise ValueError(f"没有 {skill_name} 候选结果")
    return sorted(
        selected,
        key=lambda summary: (
            not bool(summary["groups"][skill_name]["behavior_comparable"]),
            -int(summary["groups"][skill_name]["successes"]),
            float(summary["groups"][skill_name]["mean_predicate_residual_m"]),
            float(summary["groups"][skill_name]["mean_policy_environment_steps"]),
            int(summary["candidate"]["epoch"]),
            str(summary["candidate"]["label"]),
        ),
    )


def select_confirmation_labels(
    summaries: Iterable[dict[str, Any]],
    *,
    anchor_label: str,
    top_k_per_skill: int,
) -> tuple[list[str], dict[str, list[str]]]:
    if top_k_per_skill <= 0:
        raise ValueError("top_k_per_skill 必须为正数")
    materialized = list(summaries)
    labels = {str(summary["candidate"]["label"]) for summary in materialized}
    if anchor_label not in labels:
        raise ValueError("Sweep anchor 不在候选汇总中")
    selected = {anchor_label}
    rankings: dict[str, list[str]] = {}
    for skill_name in SWEEP_SKILL_THRESHOLDS_M:
        ranked = rank_candidate_summaries(materialized, skill_name)
        ranking = [str(summary["candidate"]["label"]) for summary in ranked]
        rankings[skill_name] = ranking
        selected.update(ranking[:top_k_per_skill])
    epochs = {
        str(summary["candidate"]["label"]): int(summary["candidate"]["epoch"])
        for summary in materialized
    }
    return sorted(selected, key=lambda label: (epochs[label], label)), rankings


def compare_to_anchor(
    candidate: dict[str, Any], anchor: dict[str, Any]
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for skill_name in SWEEP_SKILL_THRESHOLDS_M:
        candidate_group = candidate["groups"][skill_name]
        anchor_group = anchor["groups"][skill_name]
        candidate_seeds = candidate_group["per_seed"]
        anchor_seeds = anchor_group["per_seed"]
        if set(candidate_seeds) != set(anchor_seeds):
            raise ValueError("Confirmation 候选与 anchor seed 不一致")
        wins = losses = success_ties = failure_ties = 0
        for seed in sorted(candidate_seeds):
            candidate_success = bool(candidate_seeds[seed]["success"])
            anchor_success = bool(anchor_seeds[seed]["success"])
            if candidate_success and not anchor_success:
                wins += 1
            elif anchor_success and not candidate_success:
                losses += 1
            elif candidate_success:
                success_ties += 1
            else:
                failure_ties += 1
        anchor_residual = float(anchor_group["mean_predicate_residual_m"])
        candidate_residual = float(candidate_group["mean_predicate_residual_m"])
        if anchor_residual > 0:
            residual_ratio: float | None = candidate_residual / anchor_residual
        elif candidate_residual == 0:
            residual_ratio = 1.0
        else:
            residual_ratio = None
        comparisons[skill_name] = {
            "success_delta": int(candidate_group["successes"])
            - int(anchor_group["successes"]),
            "mean_predicate_residual_delta_m": candidate_residual - anchor_residual,
            "mean_predicate_residual_ratio": residual_ratio,
            "paired_wins": wins,
            "paired_losses": losses,
            "paired_success_ties": success_ties,
            "paired_failure_ties": failure_ties,
        }
    return comparisons


def assess_promotion_candidates(
    summaries: Iterable[dict[str, Any]], *, anchor_label: str
) -> tuple[list[str], dict[str, Any]]:
    materialized = list(summaries)
    by_label = {
        str(summary["candidate"]["label"]): summary for summary in materialized
    }
    try:
        anchor = by_label[anchor_label]
    except KeyError as error:
        raise ValueError("Confirmation 缺少 anchor") from error
    promoted: list[str] = []
    comparisons: dict[str, Any] = {}
    for label, summary in sorted(by_label.items()):
        if label == anchor_label:
            continue
        comparison = compare_to_anchor(summary, anchor)
        comparisons[label] = comparison
        groups = summary["groups"]
        no_success_regression = all(
            int(groups[skill]["successes"])
            >= int(anchor["groups"][skill]["successes"])
            for skill in SWEEP_SKILL_THRESHOLDS_M
        )
        behavior_comparable = all(
            bool(groups[skill]["behavior_comparable"])
            for skill in SWEEP_SKILL_THRESHOLDS_M
        )
        clearly_improved = any(
            comparison[skill]["success_delta"] >= 2
            and comparison[skill]["mean_predicate_residual_ratio"] is not None
            and comparison[skill]["mean_predicate_residual_ratio"] <= 0.8
            for skill in SWEEP_SKILL_THRESHOLDS_M
        )
        if no_success_regression and behavior_comparable and clearly_improved:
            promoted.append(label)
    return promoted, comparisons


__all__ = [
    "CHECKPOINT_SWEEP_FORMAT",
    "SWEEP_SKILL_THRESHOLDS_M",
    "CheckpointSweepCandidate",
    "assess_promotion_candidates",
    "compare_to_anchor",
    "discover_sweep_candidates",
    "predicate_residual",
    "rank_candidate_summaries",
    "read_atomic_results",
    "select_confirmation_labels",
    "summarize_candidate_results",
]
