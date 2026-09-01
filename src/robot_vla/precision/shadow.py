"""E013 100-seed paired 20 Hz budget shadow rollout；永不返回 Action。"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robot_vla.adapters import (
    FingerForceNormalizer,
    FingerForceStats,
    ProprioNormalizer,
    ProprioStats,
)
from robot_vla.contracts import RobotSpec
from robot_vla.data.trajectory import TrajectoryStore, load_manifest
from robot_vla.executive.contracts import PredicateSource
from robot_vla.observation import (
    OBSERVATION_MODALITIES,
    ObservationV2Frame,
    ObservationV2History,
)
from robot_vla.precision.checkpoint import (
    PrecisionCheckpointRole,
    load_torch_precision_frame_predictor,
)
from robot_vla.precision.data import canonical_sha256, file_sha256
from robot_vla.precision.provider import (
    PrecisionDetectionProvider,
    PrecisionDetectionProviderConfig,
    PrecisionGeometricMotionInput,
    TorchPrecisionFramePredictorConfig,
)
from robot_vla.precision.training import load_precision_experiment_config
from robot_vla.sim.collector import EpisodeRejected, TrustedPickPlaceCollector

PRECISION_SHADOW_VERSION = "e013-precision-paired-shadow/v1"


@dataclass(frozen=True)
class PrecisionShadowEpisodeReceipt:
    seed: int
    provider_call_count: int
    predicted_frame_count: int
    provider_failure_count: int
    deadline_miss_count: int
    provider_records_sha256: str
    observer_error_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PrecisionShadowSummary:
    version: str
    checkpoint_sha256: str
    calibration_sha256: str
    provider_identity_sha256: str
    seed_start: int
    requested_paired_seeds: int
    completed_baseline_episodes: int
    completed_shadow_episodes: int
    paired_episode_count: int
    baseline_failure_count: int
    shadow_collection_failure_count: int
    observer_error_count: int
    provider_failure_count: int
    action_parity_mismatch_count: int
    commanded_target_parity_mismatch_count: int
    episode_length_mismatch_count: int
    provider_call_count: int
    predicted_frame_count: int
    deadline_miss_count: int
    provider_latency_p50_s: float
    provider_latency_p95_s: float
    effective_rate_from_p95_hz: float
    actuation_allowed: bool
    expert_outcome_is_not_precision_treatment: bool
    gate_passed: bool
    episode_receipts_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_hold_geometry(window: Any, frame_index: int) -> PrecisionGeometricMotionInput:
    wrist_index = OBSERVATION_MODALITIES.index("rgb_wrist")
    return PrecisionGeometricMotionInput(
        timestamp_s=float(window.modality_timestamp_s[frame_index, wrist_index]),
        motion=(0.0, 0.0, 0.0, 0.0),
        source=PredicateSource.DEPLOYABLE_ESTIMATOR,
    )


class PrecisionShadowObserver:
    """只消费 Frame 并记录 Provider 结果；接口刻意没有 Action 返回值。"""

    def __init__(
        self,
        spec: RobotSpec,
        provider: PrecisionDetectionProvider,
        *,
        deadline_s: float,
    ) -> None:
        if deadline_s <= 0.0:
            raise ValueError("Precision shadow deadline_s 必须为正数")
        self.spec = spec
        self.provider = provider
        self.deadline_s = float(deadline_s)
        self.history = ObservationV2History(spec)
        self.wall_latency_s: list[float] = []

    def reset(self) -> None:
        self.history.reset()
        self.provider.reset()
        self.wall_latency_s.clear()

    def observe(
        self,
        frame: ObservationV2Frame,
        *,
        previous_command_q: np.ndarray,
        previous_action: np.ndarray | None,
    ) -> None:
        self.history.append(frame)
        window = self.history.snapshot(
            "precision shadow observation",
            previous_command_q=previous_command_q,
            previous_action=previous_action,
        )
        started = time.perf_counter()
        try:
            self.provider(window)
        finally:
            self.wall_latency_s.append(time.perf_counter() - started)

    def receipt(self, seed: int, observer_error_count: int) -> PrecisionShadowEpisodeReceipt:
        records = self.provider.records
        return PrecisionShadowEpisodeReceipt(
            seed=seed,
            provider_call_count=len(records),
            predicted_frame_count=sum(record.detections_count for record in records),
            provider_failure_count=sum(not record.success for record in records),
            deadline_miss_count=sum(record.total_latency_s > self.deadline_s for record in records),
            provider_records_sha256=self.provider.records_sha256,
            observer_error_count=observer_error_count,
        )


def _provider(
    *,
    deployable_root: Path,
    training_root: Path,
    checkpoint_sha256: str,
    temperature: float,
) -> PrecisionDetectionProvider:
    predictor = load_torch_precision_frame_predictor(
        training_root / "precision-formal.pt",
        expected_checkpoint_sha256=checkpoint_sha256,
        expected_role=PrecisionCheckpointRole.FORMAL_TRAINING,
        predictor_config=TorchPrecisionFramePredictorConfig(
            device="cuda",
            use_bf16=True,
            temperature=temperature,
            synchronize_cuda_for_latency=True,
        ),
    ).predictor
    spec = RobotSpec()
    proprio_path = deployable_root / "proprio_stats.json"
    force_path = deployable_root / "finger_force_stats.json"
    return PrecisionDetectionProvider(
        spec,
        predictor,
        ProprioNormalizer(ProprioStats.from_json(proprio_path), spec),
        FingerForceNormalizer(FingerForceStats.from_json(force_path), spec),
        _safe_hold_geometry,
        geometric_motion_provider_id="deployable-safe-hold/paired-shadow-only/v1",
        proprio_stats_sha256=file_sha256(proprio_path),
        finger_force_stats_sha256=file_sha256(force_path),
        config=PrecisionDetectionProviderConfig(enabled=True),
    )


def _collect_range(
    collector: TrustedPickPlaceCollector,
    seeds: range,
    *,
    observer: PrecisionShadowObserver | None = None,
) -> tuple[
    set[int],
    list[dict[str, Any]],
    list[PrecisionShadowEpisodeReceipt],
    list[float],
]:
    completed: set[int] = set()
    failures: list[dict[str, Any]] = []
    receipts: list[PrecisionShadowEpisodeReceipt] = []
    latencies: list[float] = []
    for seed in seeds:
        try:
            collector.collect(seed=seed, split="test")
        except EpisodeRejected as error:
            failures.append({"seed": seed, "type": type(error).__name__, "message": str(error)})
            continue
        completed.add(seed)
        if observer is not None:
            receipts.append(observer.receipt(seed, len(collector.shadow_observer_errors)))
            latencies.extend(record.total_latency_s for record in observer.provider.records)
    return completed, failures, receipts, latencies


def _entry_map(root: Path) -> dict[int, Any]:
    manifest = root / "manifest.jsonl"
    if not manifest.is_file():
        return {}
    return {int(entry.randomization["seed"]): entry for entry in load_manifest(root)}


def run_precision_paired_shadow(
    *,
    deployable_training_root: str | Path,
    config_path: str | Path,
    training_output: str | Path,
    held_out_output: str | Path,
    output_root: str | Path,
) -> PrecisionShadowSummary:
    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"Precision shadow output 已存在，拒绝覆盖: {output}")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    config = load_precision_experiment_config(config_path)
    held_out_receipt = json.loads(
        (Path(held_out_output) / "receipt.json").read_text(encoding="utf-8")
    )
    if not held_out_receipt["held_out"]["perception_gate_passed"]:
        raise RuntimeError("RGB-only perception gate 未通过，禁止 20 Hz shadow rollout")
    if not held_out_receipt["provider_latency"]["latency_gate_passed"]:
        raise RuntimeError("Provider latency gate 未通过，禁止 20 Hz shadow rollout")
    calibration = held_out_receipt["calibration"]
    checkpoint_sha256 = str(calibration["checkpoint_sha256"])
    provider = _provider(
        deployable_root=Path(deployable_training_root),
        training_root=Path(training_output),
        checkpoint_sha256=checkpoint_sha256,
        temperature=float(calibration["temperature"]),
    )
    spec = RobotSpec()
    deadline_s = min(
        config.shadow_rollout.p95_latency_max_s,
        1.0 / config.shadow_rollout.required_control_hz,
    )
    observer = PrecisionShadowObserver(spec, provider, deadline_s=deadline_s)
    start = config.shadow_rollout.start_seed
    seeds = range(start, start + config.shadow_rollout.episode_count)
    baseline_root = output / "baseline-deployable"
    shadow_root = output / "shadow-deployable"
    with TrustedPickPlaceCollector(baseline_root, spec) as baseline_collector:
        baseline_completed, baseline_failures, _, _ = _collect_range(
            baseline_collector,
            seeds,
        )
    with TrustedPickPlaceCollector(
        shadow_root,
        spec,
        shadow_observer=observer,
    ) as shadow_collector:
        (
            shadow_completed,
            shadow_failures,
            episode_receipts,
            latency_values,
        ) = _collect_range(
            shadow_collector,
            seeds,
            observer=observer,
        )

    baseline_entries = _entry_map(baseline_root)
    shadow_entries = _entry_map(shadow_root)
    baseline_store = TrajectoryStore(baseline_root, spec, cache_size=0)
    shadow_store = TrajectoryStore(shadow_root, spec, cache_size=0)
    action_mismatch = 0
    target_mismatch = 0
    length_mismatch = 0
    paired = sorted(baseline_completed & shadow_completed)
    for seed in paired:
        baseline = baseline_store.get(baseline_entries[seed])
        shadow = shadow_store.get(shadow_entries[seed])
        if baseline.num_steps != shadow.num_steps:
            length_mismatch += 1
            continue
        if not np.array_equal(baseline.action, shadow.action):
            action_mismatch += 1
        if not np.array_equal(
            baseline.commanded_joint_target_rad,
            shadow.commanded_joint_target_rad,
        ):
            target_mismatch += 1

    provider_latencies = np.asarray(latency_values, dtype=np.float64)
    if provider_latencies.size == 0:
        raise RuntimeError("Precision shadow 没有 Provider latency 样本")

    provider_failure_count = sum(receipt.provider_failure_count for receipt in episode_receipts)
    observer_error_count = sum(receipt.observer_error_count for receipt in episode_receipts)
    deadline_misses = sum(receipt.deadline_miss_count for receipt in episode_receipts)
    p50 = float(np.quantile(provider_latencies, 0.50))
    p95 = float(np.quantile(provider_latencies, 0.95))
    gate = (
        len(baseline_completed) == config.shadow_rollout.episode_count
        and len(shadow_completed) == config.shadow_rollout.episode_count
        and len(paired) == config.shadow_rollout.episode_count
        and not baseline_failures
        and not shadow_failures
        and observer_error_count == 0
        and provider_failure_count == 0
        and action_mismatch == 0
        and target_mismatch == 0
        and length_mismatch == 0
        and deadline_misses == 0
        and p95 <= deadline_s
    )
    receipt_payload = [receipt.to_dict() for receipt in episode_receipts]
    summary = PrecisionShadowSummary(
        version=PRECISION_SHADOW_VERSION,
        checkpoint_sha256=checkpoint_sha256,
        calibration_sha256=canonical_sha256(calibration),
        provider_identity_sha256=provider.identity.sha256,
        seed_start=start,
        requested_paired_seeds=config.shadow_rollout.episode_count,
        completed_baseline_episodes=len(baseline_completed),
        completed_shadow_episodes=len(shadow_completed),
        paired_episode_count=len(paired),
        baseline_failure_count=len(baseline_failures),
        shadow_collection_failure_count=len(shadow_failures),
        observer_error_count=observer_error_count,
        provider_failure_count=provider_failure_count,
        action_parity_mismatch_count=action_mismatch,
        commanded_target_parity_mismatch_count=target_mismatch,
        episode_length_mismatch_count=length_mismatch,
        provider_call_count=sum(receipt.provider_call_count for receipt in episode_receipts),
        predicted_frame_count=sum(receipt.predicted_frame_count for receipt in episode_receipts),
        deadline_miss_count=deadline_misses,
        provider_latency_p50_s=p50,
        provider_latency_p95_s=p95,
        effective_rate_from_p95_hz=float(1.0 / p95),
        actuation_allowed=False,
        expert_outcome_is_not_precision_treatment=True,
        gate_passed=gate,
        episode_receipts_sha256=canonical_sha256(receipt_payload),
    )
    _atomic_json(output / "summary.json", summary.to_dict())
    _atomic_json(
        output / "failures.json",
        {"baseline": baseline_failures, "shadow": shadow_failures},
    )
    _atomic_json(
        output / "episode_receipts.json",
        {"receipts": receipt_payload},
    )
    return summary


__all__ = [
    "PRECISION_SHADOW_VERSION",
    "PrecisionShadowEpisodeReceipt",
    "PrecisionShadowObserver",
    "PrecisionShadowSummary",
    "run_precision_paired_shadow",
]
