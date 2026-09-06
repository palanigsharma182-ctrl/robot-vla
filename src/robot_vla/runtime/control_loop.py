"""一次 Replan 的推理失败安全与 Chunk 执行编排。"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from robot_vla.observation import ObservationV2Window

from robot_vla.executive.shadow import (
    ShadowExecutiveObservation,
    ShadowExecutiveObserver,
)
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
    QwenVLAObservationV2Runtime,
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
    shadow_observation: ShadowExecutiveObservation | None = None
    shadow_error: str | None = None


@dataclass(frozen=True)
class _RTCStoredChunk:
    origin_control_step: int
    normalized_action: np.ndarray


@dataclass(frozen=True)
class ObservationPause:
    """当前循环在 Chunk 边界暂停时的实际历史摘要；只对创建它的循环有效。"""

    generation: int
    control_step: int
    ensemble_chunks: int
    rtc_chunk_present: bool
    command_reference_present: bool


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
        shadow_observer: ShadowExecutiveObserver | None = None,
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
        self.shadow_observer = shadow_observer
        self._shadow_reset_error: str | None = None
        self._observation_pause: ObservationPause | None = None
        self._history_generation = 0

    def reset(self) -> None:
        """清空跨 Chunk 历史；每个新 Episode 必须从普通 Flow 开始。"""

        self.ensembler.clear()
        self.executor.reset()
        self._rtc_previous_chunk = None
        self.control_step = 0
        self._consecutive_anomaly_replans = 0
        self._shadow_reset_error = None
        self._observation_pause = None
        self._history_generation += 1
        if self.shadow_observer is not None:
            try:
                self.shadow_observer.reset()
            except Exception as error:  # noqa: BLE001 - shadow 失败不得阻断 Runtime reset
                self._shadow_reset_error = f"{type(error).__name__}: {error}"

    @property
    def observation_paused(self) -> bool:
        return self._observation_pause is not None

    def clear_action_history(self) -> None:
        """在同步动作边界撤销旧规划；保留时钟、采样序列和既有观察暂停。

        下一次执行必须重新推理。这里只清理动作上下文，不代表相机已返回 HOME，
        也不解除暂停或重置异常重规划预算。
        """
        self.ensembler.clear()
        self._rtc_previous_chunk = None
        self.executor.reset()

    def pause_for_observation(self) -> ObservationPause:
        """在同步 Chunk 调用之间撤销动作历史，保留 Episode 时钟和采样序列。

        调用方负责暂停期间的实际 hold、相机控制与 HOME 验证。此方法不执行
        env.step，也不允许从其他线程抢占正在执行的 Chunk。
        """

        if self.observation_paused:
            raise RuntimeError("当前已经暂停观测，不能重复创建暂停")
        self._history_generation += 1
        pause = ObservationPause(
            self._history_generation,
            self.control_step,
            self.ensembler.buffer_size,
            self._rtc_previous_chunk is not None,
            self.executor.previous_command_q is not None,
        )
        self._observation_pause = pause
        self.clear_action_history()
        self._consecutive_anomaly_replans = 0
        return pause

    def resume_after_observation(
        self,
        pause: ObservationPause,
        window: ObservationV2Window,
        controller: FrankaController,
    ) -> ReplanResult:
        """消费调用方已验证 HOME 的四个新同步帧，重新推理后执行新 Chunk。

        本层只验证时序、缓存与策略版本；相机 HOME 几何、重观察终态和
        Memory 提交资格由重观察消费者验证，不由布尔占位参数代替。
        V1 仍读取最新双图/15维状态，绝不把 V1 权重冒充八图 V2 策略。
        """

        if pause is not self._observation_pause or pause is None:
            raise RuntimeError("暂停身份已过期、已消费或不属于当前循环")
        window.validate(self.executor.spec, require_current_complete=True)
        if not window.history_valid.all() or not window.modality_valid.all():
            raise ValueError("恢复要求四个完整有效的新观测帧")
        hz = self.executor.spec.control_hz
        step = round(window.timestamp_s * hz)
        expected = (step - np.arange(3, -1, -1)) / hz
        if (step - 3 <= pause.control_step
                or not np.isclose(window.timestamp_s, step / hz, rtol=0, atol=1e-8)
                or not np.allclose(window.frame_timestamp_s, expected, rtol=0, atol=1e-8)
                or not np.allclose(window.modality_timestamp_s,
                                   expected[:, None], rtol=0, atol=1e-8)):
            raise ValueError("恢复帧必须位于暂停之后且是连续同步控制步")
        if (window.controller_valid.any() or self.ensembler.buffer_size
                or self._rtc_previous_chunk is not None
                or self.executor.previous_command_q is not None
                or self.executor.previous_action is not None):
            raise RuntimeError("恢复前不得残留旧 command reference 或动作历史")
        if isinstance(self.runtime, QwenVLAObservationV2Runtime):
            observation = window
        else:
            observation = OnlineObservation(
                window.rgb_external[-1].copy(), window.rgb_wrist[-1].copy(),
                window.physical_proprio[-1].copy(), window.instruction,
            )
        self.control_step = step
        self._observation_pause = None
        try:
            result = self.replan_and_execute(observation, controller)
        except Exception:
            self.ensembler.clear()
            self._rtc_previous_chunk = None
            self.executor.reset()
            self._observation_pause = replace(pause, control_step=self.control_step)
            raise
        if not result.execution.success or result.execution.replan_required:
            # 失败不会自动放行普通 VLA；由调用方保持 hold 并结束本次恢复。
            self.ensembler.clear()
            self._rtc_previous_chunk = None
            self.executor.reset()
            self._observation_pause = replace(pause, control_step=self.control_step)
        return result

    def _observe_shadow(
        self,
        observation: object,
        *,
        control_step: int,
    ) -> tuple[ShadowExecutiveObservation | None, str | None]:
        if self.shadow_observer is None:
            return None, None
        if self._shadow_reset_error is not None:
            return None, self._shadow_reset_error
        try:
            return (
                self.shadow_observer.observe(
                    observation,
                    control_step=control_step,
                ),
                None,
            )
        except Exception as error:  # noqa: BLE001 - 最后一道 action-parity 隔离
            return None, f"{type(error).__name__}: {error}"

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
        if self.observation_paused:
            raise RuntimeError("观测暂停期间禁止 VLA 推理和执行旧动作")
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
            shadow_observation, shadow_error = self._observe_shadow(
                observation,
                control_step=origin_control_step,
            )
            return ReplanResult(
                action_chunk=None,
                execution=execution,
                sampling=self.runtime.last_sampling_trace,
                anomaly_replan_count=self._consecutive_anomaly_replans,
                inference_strategy=self.inference_strategy,
                shadow_observation=shadow_observation,
                shadow_error=shadow_error,
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
        shadow_observation, shadow_error = self._observe_shadow(
            observation,
            control_step=origin_control_step,
        )
        return ReplanResult(
            action_chunk=ensembled_chunk,
            execution=execution,
            sampling=ensembled_chunk.sampling,
            ensemble_trace=None if ensemble is None else ensemble.trace,
            anomaly_replan_count=self._consecutive_anomaly_replans,
            inference_strategy=self.inference_strategy,
            shadow_observation=shadow_observation,
            shadow_error=shadow_error,
        )
