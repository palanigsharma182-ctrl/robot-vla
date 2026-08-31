"""一次 Replan 的推理失败安全与 Chunk 执行编排。"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from robot_vla.execution.chunk_executor import (
    ChunkExecutionResult,
    FrankaController,
    RecedingHorizonChunkExecutor,
)
from robot_vla.execution.rtc import (
    ChunkInferenceStrategy,
    RTCConfig,
    resolve_inference_strategy,
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
    inference_strategy: ChunkInferenceStrategy = ChunkInferenceStrategy.TEMPORAL_ENSEMBLE


@dataclass(frozen=True)
class _RTCStoredChunk:
    origin_control_step: int
    normalized_action: np.ndarray


class QwenVLAReplanLoop:
    def __init__(
        self,
        runtime: QwenVLARuntime,
        executor: RecedingHorizonChunkExecutor,
        *,
        inference_strategy: str | ChunkInferenceStrategy | None = None,
        temporal_ensemble_enabled: bool | None = None,
        recency_decay: float = 0.5,
        rtc_config: RTCConfig | None = None,
        max_anomaly_replans: int = 3,
    ) -> None:
        if max_anomaly_replans < 0:
            raise ValueError("max_anomaly_replans 不能为负数")
        self.runtime = runtime
        self.executor = executor
        self.inference_strategy = resolve_inference_strategy(
            inference_strategy,
            legacy_temporal_ensemble_enabled=temporal_ensemble_enabled,
        )
        self.temporal_ensemble_enabled = (
            self.inference_strategy == ChunkInferenceStrategy.TEMPORAL_ENSEMBLE
        )
        self.rtc_config = rtc_config or RTCConfig(execution_horizon=executor.spec.execute_steps)
        if self.rtc_config.execution_horizon != executor.spec.execute_steps:
            raise ValueError("首版 RTC execution_horizon 必须等于 RobotSpec.execute_steps")
        self.ensembler = TemporalChunkEnsembler(
            executor.spec,
            TemporalEnsembleConfig(recency_decay=recency_decay),
        )
        self.max_anomaly_replans = max_anomaly_replans
        self.control_step = 0
        self._consecutive_anomaly_replans = 0
        self._rtc_previous_chunk: _RTCStoredChunk | None = None

    def reset(self) -> None:
        """清空跨 Chunk 历史；每个新 Episode 必须从普通 Flow 开始。"""

        self.ensembler.clear()
        self.executor.reset()
        self._rtc_previous_chunk = None
        self.control_step = 0
        self._consecutive_anomaly_replans = 0

    def _rtc_previous_overlap(self) -> np.ndarray | None:
        previous = self._rtc_previous_chunk
        if previous is None:
            return None
        elapsed = self.control_step - previous.origin_control_step
        if elapsed != self.rtc_config.execution_horizon:
            self._rtc_previous_chunk = None
            return None
        return previous.normalized_action[elapsed:].copy()

    def _handle_anomaly(
        self,
        execution: ChunkExecutionResult,
    ) -> ChunkExecutionResult:
        self.ensembler.clear()
        self.executor.reset()
        self._rtc_previous_chunk = None
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
        origin_control_step = self.control_step
        try:
            if self.inference_strategy == ChunkInferenceStrategy.RTC:
                chunk = self.runtime.infer_action_chunk(
                    observation,
                    rtc_previous_overlap=self._rtc_previous_overlap(),
                    rtc_config=self.rtc_config,
                )
            else:
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
                inference_strategy=self.inference_strategy,
            )
        ensemble = None
        ensembled_chunk = chunk
        if self.inference_strategy == ChunkInferenceStrategy.TEMPORAL_ENSEMBLE:
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
        anomaly_detected = execution.replan_required or recoverable_failure
        if anomaly_detected:
            execution = self._handle_anomaly(execution)
        elif execution.success:
            self._consecutive_anomaly_replans = 0
            if self.inference_strategy == ChunkInferenceStrategy.RTC:
                self._rtc_previous_chunk = _RTCStoredChunk(
                    origin_control_step=origin_control_step,
                    normalized_action=ensembled_chunk.normalized_action.copy(),
                )
        return ReplanResult(
            action_chunk=ensembled_chunk,
            execution=execution,
            sampling=ensembled_chunk.sampling,
            ensemble_trace=None if ensemble is None else ensemble.trace,
            anomaly_replan_count=self._consecutive_anomaly_replans,
            inference_strategy=self.inference_strategy,
        )
