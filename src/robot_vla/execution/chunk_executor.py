"""D017：以每次 Replan 的真实 q 为基准执行 Action Chunk 前缀。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from robot_vla.adapters import ActionAdapter, ActionContractViolation
from robot_vla.contracts import RobotSpec


@dataclass(frozen=True)
class FrankaControlState:
    joint_positions: np.ndarray
    gripper_opening: float


class FrankaController(Protocol):
    def read_state(self) -> FrankaControlState: ...

    def send_action(self, controller_action: np.ndarray) -> None: ...

    def hold_current(self) -> None: ...


@dataclass(frozen=True)
class ChunkExecutionResult:
    success: bool
    executed_steps: int
    failure_stage: str | None = None
    error: str | None = None
    hold_succeeded: bool | None = None
    diagnostic: dict[str, Any] | None = None
    correction_saturation_steps: int = 0
    requested_correction_abs_max_rad: float | None = None
    applied_correction_abs_max_rad: float | None = None
    replan_required: bool = False
    anomaly_kind: str | None = None


class RecedingHorizonChunkExecutor:
    """不保存旧 Chunk；每次调用只执行新 Chunk 的前 K 步。"""

    def __init__(self, spec: RobotSpec) -> None:
        self.spec = spec
        self.action_adapter = ActionAdapter(spec)

    def _validated_state(self, controller: FrankaController) -> FrankaControlState:
        state = controller.read_state()
        q = np.asarray(state.joint_positions)
        if q.shape != (self.spec.arm_dof,) or q.dtype != np.float32 or not np.isfinite(q).all():
            raise ValueError("Controller joint_positions 必须是有限 float32 Franka arm 向量")
        if not np.isfinite(state.gripper_opening) or not 0.0 <= state.gripper_opening <= 1.0:
            raise ValueError("Controller gripper_opening 必须位于 [0,1]")
        return state

    @staticmethod
    def _error_text(error: Exception) -> str:
        return f"{type(error).__name__}: {error}"

    def stop_for_failure(
        self,
        controller: FrankaController,
        *,
        stage: str,
        error: Exception,
        executed_steps: int = 0,
        diagnostic: dict[str, Any] | None = None,
        correction_saturation_steps: int = 0,
        requested_correction_abs_max_rad: float | None = None,
        applied_correction_abs_max_rad: float | None = None,
    ) -> ChunkExecutionResult:
        if diagnostic is None and isinstance(error, ActionContractViolation):
            diagnostic = error.to_diagnostic()
        try:
            controller.hold_current()
        except Exception as hold_error:  # noqa: BLE001 - 安全边界必须报告 hold 失败
            return ChunkExecutionResult(
                success=False,
                executed_steps=executed_steps,
                failure_stage=stage,
                error=f"{self._error_text(error)}; hold 失败: {self._error_text(hold_error)}",
                hold_succeeded=False,
                diagnostic=diagnostic,
                correction_saturation_steps=correction_saturation_steps,
                requested_correction_abs_max_rad=requested_correction_abs_max_rad,
                applied_correction_abs_max_rad=applied_correction_abs_max_rad,
            )
        return ChunkExecutionResult(
            success=False,
            executed_steps=executed_steps,
            failure_stage=stage,
            error=self._error_text(error),
            hold_succeeded=True,
            diagnostic=diagnostic,
            correction_saturation_steps=correction_saturation_steps,
            requested_correction_abs_max_rad=requested_correction_abs_max_rad,
            applied_correction_abs_max_rad=applied_correction_abs_max_rad,
        )

    def execute(
        self,
        physical_chunk: np.ndarray,
        controller: FrankaController,
    ) -> ChunkExecutionResult:
        try:
            initial_state = self._validated_state(controller)
        except Exception as error:  # noqa: BLE001 - 控制器错误统一转成显式停止结果
            return self.stop_for_failure(
                controller,
                stage="initial_observation",
                error=error,
            )
        try:
            commands = self.action_adapter.build_receding_horizon_commands(
                initial_state.joint_positions,
                physical_chunk,
            )
        except Exception as error:  # noqa: BLE001 - Chunk 校验错误必须触发 hold
            return self.stop_for_failure(
                controller,
                stage="chunk_safety",
                error=error,
            )

        executed_steps = 0
        correction_saturation_steps = 0
        requested_correction_abs_max_rad = 0.0
        applied_correction_abs_max_rad = 0.0
        for chunk_step_index, (target_q, target_gripper) in enumerate(zip(
            commands.joint_position_targets,
            commands.gripper_opening_targets,
            strict=True,
        )):
            try:
                latest_state = self._validated_state(controller)
            except Exception as error:  # noqa: BLE001 - 在线状态错误必须停止当前 Chunk
                return self.stop_for_failure(
                    controller,
                    stage="step_observation",
                    error=error,
                    executed_steps=executed_steps,
                )
            requested_correction = target_q - latest_state.joint_positions
            correction = np.clip(
                requested_correction,
                -self.action_adapter.delta_limits,
                self.action_adapter.delta_limits,
            )
            if not np.allclose(correction, requested_correction, rtol=0.0, atol=1e-7):
                correction_saturation_steps += 1
            requested_correction_abs_max_rad = max(
                requested_correction_abs_max_rad,
                float(np.max(np.abs(requested_correction))),
            )
            applied_correction_abs_max_rad = max(
                applied_correction_abs_max_rad,
                float(np.max(np.abs(correction))),
            )
            physical_action = np.concatenate(
                (correction, np.asarray([target_gripper], dtype=np.float32))
            ).astype(np.float32, copy=False)
            try:
                controller_action = self.action_adapter.to_maniskill(physical_action)
            except Exception as error:  # noqa: BLE001 - 安全检查错误必须停止当前 Chunk
                diagnostic = (
                    error.to_diagnostic()
                    if isinstance(error, ActionContractViolation)
                    else {"kind": "unstructured_step_safety_error"}
                )
                diagnostic.update(
                    {
                        "chunk_step_index": chunk_step_index,
                        "latest_joint_positions_rad": latest_state.joint_positions.tolist(),
                        "target_joint_positions_rad": target_q.tolist(),
                        "requested_tracking_correction_rad": requested_correction.tolist(),
                        "applied_tracking_correction_rad": correction.tolist(),
                        "target_gripper_opening": float(target_gripper),
                    }
                )
                return self.stop_for_failure(
                    controller,
                    stage="step_safety",
                    error=error,
                    executed_steps=executed_steps,
                    diagnostic=diagnostic,
                    correction_saturation_steps=correction_saturation_steps,
                    requested_correction_abs_max_rad=requested_correction_abs_max_rad,
                    applied_correction_abs_max_rad=applied_correction_abs_max_rad,
                )
            try:
                controller.send_action(controller_action)
            except Exception as error:  # noqa: BLE001 - 控制器错误必须停止当前 Chunk
                return self.stop_for_failure(
                    controller,
                    stage="controller_step",
                    error=error,
                    executed_steps=executed_steps,
                    correction_saturation_steps=correction_saturation_steps,
                    requested_correction_abs_max_rad=requested_correction_abs_max_rad,
                    applied_correction_abs_max_rad=applied_correction_abs_max_rad,
                )
            executed_steps += 1
            if correction_saturation_steps > 0:
                return ChunkExecutionResult(
                    success=True,
                    executed_steps=executed_steps,
                    correction_saturation_steps=correction_saturation_steps,
                    requested_correction_abs_max_rad=requested_correction_abs_max_rad,
                    applied_correction_abs_max_rad=applied_correction_abs_max_rad,
                    replan_required=True,
                    anomaly_kind="tracking_correction_saturation",
                )

        return ChunkExecutionResult(
            success=True,
            executed_steps=executed_steps,
            correction_saturation_steps=correction_saturation_steps,
            requested_correction_abs_max_rad=requested_correction_abs_max_rad,
            applied_correction_abs_max_rad=applied_correction_abs_max_rad,
        )
