"""默认关闭、只记录不控制的 P1 Shadow Executive Observer。"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

from robot_vla.contracts import RobotSpec
from robot_vla.executive.contracts import (
    CompiledTaskPlan,
    ExecutiveDecision,
    ExecutiveSnapshot,
    ExecutiveState,
    PredicateEvidence,
)
from robot_vla.executive.estimation import (
    DeployableOutcomeEvidence,
    DeployablePredicateThresholds,
    DeployableStateEstimatorResult,
    FourFrameDeployableStateEstimator,
    TemporalTrackDiagnostics,
    WristKeypointDetection,
    build_deployable_snapshot,
)
from robot_vla.executive.executive import ExecutiveConfig, HierarchicalExecutive
from robot_vla.observation import ObservationV2Window

SHADOW_EXECUTIVE_OBSERVER_VERSION = "qwen-vla-shadow-executive-observer/v1"
SHADOW_EXECUTIVE_CADENCE = "replan-boundary/pre-execution-observation/v1"

DetectionProvider = Callable[
    [ObservationV2Window],
    tuple[WristKeypointDetection | None, ...],
]
OutcomeProvider = Callable[[ObservationV2Window], DeployableOutcomeEvidence]
AdditionalPredicateProvider = Callable[
    [ObservationV2Window, DeployableStateEstimatorResult],
    tuple[PredicateEvidence, ...],
]


@dataclass(frozen=True)
class ShadowExecutiveObserverConfig:
    """必须显式打开；P1 observer 永远不能获得 actuation authority。"""

    enabled: bool = False
    version: str = SHADOW_EXECUTIVE_OBSERVER_VERSION
    cadence: str = SHADOW_EXECUTIVE_CADENCE

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("Shadow Executive enabled 必须为 bool")
        if self.version != SHADOW_EXECUTIVE_OBSERVER_VERSION:
            raise ValueError(
                f"Shadow Executive version 必须为 {SHADOW_EXECUTIVE_OBSERVER_VERSION}"
            )
        if self.cadence != SHADOW_EXECUTIVE_CADENCE:
            raise ValueError(f"Shadow Executive cadence 必须为 {SHADOW_EXECUTIVE_CADENCE}")


@dataclass(frozen=True)
class ShadowExecutiveObservation:
    """单次 observer 调用的脱敏记录；成功快照本身保存在 transition ledger。"""

    record_index: int
    control_step: int
    executive_tick: int | None
    observation_timestamp_s: float | None
    success: bool
    decision: ExecutiveDecision | None
    object_diagnostics: TemporalTrackDiagnostics | None
    goal_diagnostics: TemporalTrackDiagnostics | None
    ledger_sha256: str
    latency_s: float
    error_type: str | None = None
    error_message: str | None = None
    version: str = SHADOW_EXECUTIVE_OBSERVER_VERSION
    cadence: str = SHADOW_EXECUTIVE_CADENCE

    def __post_init__(self) -> None:
        for value, name in (
            (self.record_index, "record_index"),
            (self.control_step, "control_step"),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} 必须是非负整数")
        if self.executive_tick is not None and (
            not isinstance(self.executive_tick, int)
            or isinstance(self.executive_tick, bool)
            or self.executive_tick < 0
        ):
            raise ValueError("executive_tick 必须是非负整数或 None")
        if self.observation_timestamp_s is not None and (
            not math.isfinite(self.observation_timestamp_s)
            or self.observation_timestamp_s < 0.0
        ):
            raise ValueError("observation_timestamp_s 必须是有限非负数或 None")
        if not isinstance(self.success, bool):
            raise TypeError("shadow observation success 必须为 bool")
        if not math.isfinite(self.latency_s) or self.latency_s < 0.0:
            raise ValueError("shadow observation latency_s 必须是有限非负数")
        if len(self.ledger_sha256) != 64:
            raise ValueError("ledger_sha256 必须是 64 位 SHA-256")
        if self.version != SHADOW_EXECUTIVE_OBSERVER_VERSION:
            raise ValueError(
                f"shadow observation version 必须为 {SHADOW_EXECUTIVE_OBSERVER_VERSION}"
            )
        if self.cadence != SHADOW_EXECUTIVE_CADENCE:
            raise ValueError(f"shadow observation cadence 必须为 {SHADOW_EXECUTIVE_CADENCE}")
        if self.success:
            if (
                self.executive_tick is None
                or self.decision is None
                or self.object_diagnostics is None
                or self.goal_diagnostics is None
                or self.error_type is not None
                or self.error_message is not None
            ):
                raise ValueError("成功 shadow observation 的 decision/diagnostics 字段不完整")
            if not self.decision.shadow_only or self.decision.actuation_allowed:
                raise ValueError("Shadow Executive decision 不能拥有 actuation authority")
        elif self.error_type is None:
            raise ValueError("失败 shadow observation 必须记录 error_type")

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "cadence": self.cadence,
            "record_index": self.record_index,
            "control_step": self.control_step,
            "executive_tick": self.executive_tick,
            "observation_timestamp_s": self.observation_timestamp_s,
            "success": self.success,
            "decision": None if self.decision is None else self.decision.to_dict(),
            "object_diagnostics": (
                None
                if self.object_diagnostics is None
                else self.object_diagnostics.to_dict()
            ),
            "goal_diagnostics": (
                None if self.goal_diagnostics is None else self.goal_diagnostics.to_dict()
            ),
            "ledger_sha256": self.ledger_sha256,
            "latency_s": self.latency_s,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


class ShadowExecutiveObserver:
    """在 Replan 边界重放 Executive；记录结果但永不修改 VLA Action。"""

    def __init__(
        self,
        spec: RobotSpec,
        plan: CompiledTaskPlan,
        estimator: FourFrameDeployableStateEstimator,
        predicate_thresholds: DeployablePredicateThresholds,
        detection_provider: DetectionProvider,
        *,
        outcome_provider: OutcomeProvider | None = None,
        additional_predicate_provider: AdditionalPredicateProvider | None = None,
        executive_config: ExecutiveConfig | None = None,
        config: ShadowExecutiveObserverConfig | None = None,
    ) -> None:
        selected_executive_config = executive_config or ExecutiveConfig()
        if not selected_executive_config.shadow_only:
            raise ValueError("ShadowExecutiveObserver 禁止使用 shadow_only=False")
        self.spec = spec
        self.plan = plan
        self.estimator = estimator
        self.predicate_thresholds = predicate_thresholds
        self.detection_provider = detection_provider
        self.outcome_provider = outcome_provider
        self.additional_predicate_provider = additional_predicate_provider
        self.executive_config = selected_executive_config
        self.config = config or ShadowExecutiveObserverConfig()
        self._records: list[ShadowExecutiveObservation] = []
        self._executive = HierarchicalExecutive(plan, selected_executive_config)
        self._next_executive_tick = 0
        self._last_control_step = -1

    @property
    def records(self) -> tuple[ShadowExecutiveObservation, ...]:
        return tuple(self._records)

    @property
    def state(self) -> ExecutiveState:
        return self._executive.state

    @property
    def ledger_jsonl(self) -> str:
        return self._executive.ledger.to_jsonl()

    @property
    def ledger_sha256(self) -> str:
        return self._executive.ledger.sha256()

    def reset(self) -> None:
        """开始新 Episode；调用方应在 reset 前保存上一 Episode 的 records/ledger。"""

        reset_provider = getattr(self.detection_provider, "reset", None)
        if callable(reset_provider):
            reset_provider()
        self._records.clear()
        self._executive = HierarchicalExecutive(self.plan, self.executive_config)
        self._next_executive_tick = 0
        self._last_control_step = -1

    def _restore_committed_executive(
        self,
        snapshots: tuple[ExecutiveSnapshot, ...],
    ) -> None:
        """从调用前 ledger 恢复，避免失败 step 留下部分状态。"""

        restored = HierarchicalExecutive(self.plan, self.executive_config)
        for snapshot in snapshots:
            restored.step(snapshot)
        self._executive = restored

    def observe(
        self,
        observation: object,
        *,
        control_step: int,
    ) -> ShadowExecutiveObservation | None:
        if not self.config.enabled:
            return None
        started = time.perf_counter()
        record_index = len(self._records)
        executive_tick: int | None = None
        timestamp: float | None = None
        decision: ExecutiveDecision | None = None
        estimate: DeployableStateEstimatorResult | None = None
        error_type: str | None = None
        error_message: str | None = None
        committed_snapshots = tuple(
            entry.snapshot for entry in self._executive.ledger.entries
        )
        try:
            if (
                not isinstance(control_step, int)
                or isinstance(control_step, bool)
                or control_step < 0
            ):
                raise ValueError("control_step 必须是非负整数")
            if control_step < self._last_control_step:
                raise ValueError("Shadow Executive control_step 不能回退")
            self._last_control_step = control_step
            if not isinstance(observation, ObservationV2Window):
                raise TypeError("Shadow Executive 只接受 ObservationV2Window")
            timestamp = observation.timestamp_s
            detections = self.detection_provider(observation)
            outcome = (
                None
                if self.outcome_provider is None
                else self.outcome_provider(observation)
            )
            estimate = self.estimator.estimate(
                observation,
                detections,
                outcome_evidence=outcome,
            )
            additional = (
                ()
                if self.additional_predicate_provider is None
                else self.additional_predicate_provider(observation, estimate)
            )
            executive_tick = self._next_executive_tick
            snapshot = build_deployable_snapshot(
                self.spec,
                observation,
                estimate.state_estimate,
                self.predicate_thresholds,
                tick=executive_tick,
                timestamp_s=timestamp,
                additional_predicates=additional,
            )
            decision = self._executive.step(snapshot)
            if not decision.shadow_only or decision.actuation_allowed:
                raise RuntimeError("Shadow Executive 意外获得 actuation authority")
            self._next_executive_tick += 1
            success = True
        except Exception as error:  # noqa: BLE001 - observer 失败必须记录且不得影响 Action
            success = False
            error_type = type(error).__name__
            error_message = str(error)
            self._restore_committed_executive(committed_snapshots)

        safe_control_step = (
            control_step
            if isinstance(control_step, int)
            and not isinstance(control_step, bool)
            and control_step >= 0
            else 0
        )
        record = ShadowExecutiveObservation(
            record_index=record_index,
            control_step=safe_control_step,
            executive_tick=executive_tick if success else None,
            observation_timestamp_s=timestamp,
            success=success,
            decision=decision if success else None,
            object_diagnostics=(
                estimate.object_diagnostics if success and estimate is not None else None
            ),
            goal_diagnostics=(
                estimate.goal_diagnostics if success and estimate is not None else None
            ),
            ledger_sha256=self._executive.ledger.sha256(),
            latency_s=time.perf_counter() - started,
            error_type=error_type,
            error_message=error_message,
        )
        self._records.append(record)
        return record


__all__ = [
    "SHADOW_EXECUTIVE_CADENCE",
    "SHADOW_EXECUTIVE_OBSERVER_VERSION",
    "AdditionalPredicateProvider",
    "DetectionProvider",
    "OutcomeProvider",
    "ShadowExecutiveObservation",
    "ShadowExecutiveObserver",
    "ShadowExecutiveObserverConfig",
]
