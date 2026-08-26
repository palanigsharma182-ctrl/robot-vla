"""一次 Replan 的推理失败安全与 Chunk 执行编排。"""

from __future__ import annotations

from dataclasses import dataclass, replace

from robot_vla.execution.chunk_executor import (
    ChunkExecutionResult,
    FrankaController,
    RecedingHorizonChunkExecutor,
)
from robot_vla.execution.temporal_ensemble import (
    TemporalChunkEnsembler,
    TemporalEnsembleConfig,
    TemporalEnsembleTrace,
)
from robot_vla.runtime.policy_runtime import (
    OnlineObservation,
    QwenVLARuntime,
    RuntimeActionChunk,
    SamplingTrace,
)


@dataclass(frozen=True)
class ReplanResult:
    action_chunk: RuntimeActionChunk | None
    execution: ChunkExecutionResult
    sampling: SamplingTrace | None
    ensemble_trace: TemporalEnsembleTrace | None = None
    anomaly_replan_count: int = 0


class QwenVLAReplanLoop:
    def __init__(
        self,
        runtime: QwenVLARuntime,
        executor: RecedingHorizonChunkExecutor,
        *,
        temporal_ensemble_enabled: bool = True,
        recency_decay: float = 0.5,
        max_anomaly_replans: int = 3,
    ) -> None:
        if max_anomaly_replans < 0:
            raise ValueError("max_anomaly_replans 不能为负数")
        self.runtime = runtime
        self.executor = executor
        self.temporal_ensemble_enabled = temporal_ensemble_enabled
        self.ensembler = TemporalChunkEnsembler(
            executor.spec,
            TemporalEnsembleConfig(recency_decay=recency_decay),
        )
        self.max_anomaly_replans = max_anomaly_replans
        self.control_step = 0
        self._consecutive_anomaly_replans = 0

    def _handle_anomaly(
        self,
        execution: ChunkExecutionResult,
    ) -> ChunkExecutionResult:
        self.ensembler.clear()
        if self.max_anomaly_replans == 0:
            anomaly_kind = (
                execution.anomaly_kind or execution.failure_stage or "unknown_anomaly"
            )
            return replace(
                execution,
                success=False,
                failure_stage=execution.failure_stage or "replan_anomaly_exhausted",
                error=execution.error or f"异常重规划已禁用；异常={anomaly_kind}",
                replan_required=False,
                anomaly_kind=anomaly_kind,
            )
        self._consecutive_anomaly_replans += 1
        anomaly_kind = execution.anomaly_kind or execution.failure_stage or "unknown_anomaly"
        if self._consecutive_anomaly_replans <= self.max_anomaly_replans:
            return replace(
                execution,
                success=True,
                replan_required=True,
                anomaly_kind=anomaly_kind,
            )
        return replace(
            execution,
            success=False,
            failure_stage="replan_anomaly_exhausted",
            error=(
                f"连续 {self._consecutive_anomaly_replans} 次异常重规划失败；"
                f"最后异常={anomaly_kind}; {execution.error or ''}"
            ),
            replan_required=False,
            anomaly_kind=anomaly_kind,
        )

    def replan_and_execute(
        self,
        observation: OnlineObservation,
        controller: FrankaController,
    ) -> ReplanResult:
        try:
            chunk = self.runtime.infer_action_chunk(observation)
        except Exception as error:  # noqa: BLE001 - 推理错误必须进入 fail-safe hold
            execution = self.executor.stop_for_failure(
                controller,
                stage="inference",
                error=error,
            )
            execution = self._handle_anomaly(execution)
            return ReplanResult(
                action_chunk=None,
                execution=execution,
                sampling=self.runtime.last_sampling_trace,
                anomaly_replan_count=self._consecutive_anomaly_replans,
            )
        ensemble = None
        ensembled_chunk = chunk
        if self.temporal_ensemble_enabled:
            ensemble = self.ensembler.add_and_compose(
                chunk.normalized_action,
                origin_control_step=self.control_step,
            )
            ensembled_chunk = replace(
                chunk,
                normalized_action=ensemble.normalized_action,
                physical_action=self.executor.action_adapter.denormalize(
                    ensemble.normalized_action
                ),
            )
        execution = self.executor.execute(ensembled_chunk.physical_action, controller)
        self.control_step += execution.executed_steps
        recoverable_failure = execution.failure_stage in {"chunk_safety", "step_safety"}
        if execution.replan_required or recoverable_failure:
            execution = self._handle_anomaly(execution)
        elif execution.success:
            self._consecutive_anomaly_replans = 0
        return ReplanResult(
            action_chunk=ensembled_chunk,
            execution=execution,
            sampling=ensembled_chunk.sampling,
            ensemble_trace=None if ensemble is None else ensemble.trace,
            anomaly_replan_count=self._consecutive_anomaly_replans,
        )
