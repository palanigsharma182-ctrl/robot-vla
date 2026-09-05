"""World Model/Critic V0 的只读 shadow 候选合同。"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

import numpy as np

from robot_vla.contracts import RobotSpec

CANDIDATE_EVALUATION_CONTRACT_VERSION = "world-model-critic-shadow/v0"
NORMALIZED_PROPRIO_ABS_LIMIT = 5.0
NORMALIZED_PROPRIO_COMMAND_SCHEMA_ID = "franka-normalized-proprio-command/v0"
NORMALIZED_ACTION_SCHEMA_ID = "franka-normalized-delta-q-absolute-gripper/v1"


class CandidateEvaluationStatus(str, Enum):
    """候选评估状态；任何状态都不携带控制权限。"""

    SCORED = "scored"
    WARMUP = "warmup"
    ABSTAIN = "abstain"
    ERROR = "error"


def _nonempty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 不能为空")
    return value


def _readonly_float32(
    value: np.ndarray,
    *,
    expected_shape: tuple[int, ...],
    name: str,
    abs_limit: float,
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != expected_shape:
        raise ValueError(f"{name} 应为 {expected_shape}，实际为 {array.shape}")
    if array.dtype != np.float32:
        raise TypeError(f"{name} 必须是 float32")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} 包含 NaN 或 Inf")
    if np.any(np.abs(array) > abs_limit + 1e-5):
        raise ValueError(f"{name} 超出 [-{abs_limit},{abs_limit}]")
    copied = array.copy()
    copied.flags.writeable = False
    return copied


def action_digest(normalized_action: np.ndarray) -> str:
    """计算包含 shape/dtype 的稳定 Action digest。"""

    array = np.ascontiguousarray(normalized_action)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _validate_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} 必须是小写 SHA256")


@dataclass(frozen=True)
class CandidateEvaluationIdentity:
    """绑定一次评估所依赖的数据、策略和归一化身份。"""

    episode_id: str
    candidate_id: str
    task_id: str
    source_domain: str
    observation_schema_id: str
    action_schema_id: str
    policy_checkpoint_id: str
    proprio_stats_id: str

    def __post_init__(self) -> None:
        for name in (
            "episode_id",
            "candidate_id",
            "task_id",
            "observation_schema_id",
            "action_schema_id",
            "policy_checkpoint_id",
            "proprio_stats_id",
        ):
            _nonempty(getattr(self, name), name)
        if self.source_domain not in {"sim", "real", "offline"}:
            raise ValueError("source_domain 必须是 sim、real 或 offline")
        if self.observation_schema_id != NORMALIZED_PROPRIO_COMMAND_SCHEMA_ID:
            raise ValueError(
                "observation_schema_id 必须为 "
                f"{NORMALIZED_PROPRIO_COMMAND_SCHEMA_ID}"
            )
        if self.action_schema_id != NORMALIZED_ACTION_SCHEMA_ID:
            raise ValueError(f"action_schema_id 必须为 {NORMALIZED_ACTION_SCHEMA_ID}")


@dataclass(frozen=True)
class CandidateEvaluationRequest:
    """同一份 deployable 请求同时服务 sim 与 real shadow。

    请求有意不包含 object GT、contact GT、success label 或 simulator state。
    Action-only critic 消费 raw proposal；世界模型消费 effective prefix。

    ``normalized_arm_command_target_prefix`` 必须来自 Executor 当前 replan
    实际使用的同一次 controller q 读取：先由同一个 ``ActionAdapter`` 对
    effective Action Chunk 构造 commanded joint targets，再使用本请求绑定的
    ``ProprioStats`` 的 q[0:7] mean/std 归一化。禁止从可能陈旧的
    ``OnlineObservation.normalized_proprio`` 反推或累加该字段。该数组只表达
    commanded target，不表达 controller tracking correction。
    """

    identity: CandidateEvaluationIdentity
    origin_control_step: int
    observation_timestamp_ns: int
    normalized_proprio: np.ndarray
    observed_gripper_opening_ratio: float
    raw_normalized_action: np.ndarray
    effective_normalized_action: np.ndarray
    normalized_arm_command_target_prefix: np.ndarray
    previous_executed_steps: int | None = None
    contract_version: str = CANDIDATE_EVALUATION_CONTRACT_VERSION

    def __post_init__(self) -> None:
        spec = RobotSpec()
        if not isinstance(self.identity, CandidateEvaluationIdentity):
            raise TypeError("identity 必须是 CandidateEvaluationIdentity")
        if self.contract_version != CANDIDATE_EVALUATION_CONTRACT_VERSION:
            raise ValueError("candidate evaluation contract_version 不受支持")
        if (
            isinstance(self.origin_control_step, bool)
            or not isinstance(self.origin_control_step, int)
            or self.origin_control_step < 0
        ):
            raise ValueError("origin_control_step 必须是非负整数")
        if (
            isinstance(self.observation_timestamp_ns, bool)
            or not isinstance(self.observation_timestamp_ns, int)
            or self.observation_timestamp_ns < 0
        ):
            raise ValueError("observation_timestamp_ns 必须是非负整数")
        if self.previous_executed_steps is not None and (
            isinstance(self.previous_executed_steps, bool)
            or not isinstance(self.previous_executed_steps, int)
            or not 0 <= self.previous_executed_steps <= spec.execute_steps
        ):
            raise ValueError("previous_executed_steps 必须位于 [0,execute_steps]")
        if (
            isinstance(self.observed_gripper_opening_ratio, bool)
            or not isinstance(self.observed_gripper_opening_ratio, (int, float))
            or not math.isfinite(self.observed_gripper_opening_ratio)
            or not 0.0 <= self.observed_gripper_opening_ratio <= 1.0
        ):
            raise ValueError("observed_gripper_opening_ratio 必须位于 [0,1]")

        normalized_proprio = _readonly_float32(
            self.normalized_proprio,
            expected_shape=(spec.proprio_dim,),
            name="normalized_proprio",
            abs_limit=NORMALIZED_PROPRIO_ABS_LIMIT,
        )
        action_shape = (spec.action_horizon, spec.action_dim)
        raw_action = _readonly_float32(
            self.raw_normalized_action,
            expected_shape=action_shape,
            name="raw_normalized_action",
            abs_limit=1.0,
        )
        effective_action = _readonly_float32(
            self.effective_normalized_action,
            expected_shape=action_shape,
            name="effective_normalized_action",
            abs_limit=1.0,
        )
        normalized_command_target = _readonly_float32(
            self.normalized_arm_command_target_prefix,
            expected_shape=(spec.execute_steps, spec.arm_dof),
            name="normalized_arm_command_target_prefix",
            abs_limit=NORMALIZED_PROPRIO_ABS_LIMIT,
        )
        object.__setattr__(self, "normalized_proprio", normalized_proprio)
        object.__setattr__(self, "raw_normalized_action", raw_action)
        object.__setattr__(self, "effective_normalized_action", effective_action)
        object.__setattr__(
            self,
            "normalized_arm_command_target_prefix",
            normalized_command_target,
        )

    @property
    def effective_action_prefix(self) -> np.ndarray:
        prefix = self.effective_normalized_action[: RobotSpec().execute_steps].copy()
        prefix.flags.writeable = False
        return prefix

    @property
    def raw_action_digest(self) -> str:
        return action_digest(self.raw_normalized_action)

    @property
    def effective_action_digest(self) -> str:
        return action_digest(self.effective_normalized_action)

    @property
    def normalized_arm_command_target_prefix_digest(self) -> str:
        return action_digest(self.normalized_arm_command_target_prefix)

    @property
    def request_digest(self) -> str:
        """把 identity、时间和输入 Tensor digest 绑定为一次请求身份。"""

        payload = {
            "contract_version": self.contract_version,
            "episode_id": self.identity.episode_id,
            "candidate_id": self.identity.candidate_id,
            "task_id": self.identity.task_id,
            "source_domain": self.identity.source_domain,
            "observation_schema_id": self.identity.observation_schema_id,
            "action_schema_id": self.identity.action_schema_id,
            "policy_checkpoint_id": self.identity.policy_checkpoint_id,
            "proprio_stats_id": self.identity.proprio_stats_id,
            "origin_control_step": self.origin_control_step,
            "observation_timestamp_ns": self.observation_timestamp_ns,
            "previous_executed_steps": self.previous_executed_steps,
            "observed_gripper_opening_ratio": float(
                self.observed_gripper_opening_ratio
            ),
            "normalized_proprio_digest": action_digest(self.normalized_proprio),
            "raw_action_digest": self.raw_action_digest,
            "effective_action_digest": self.effective_action_digest,
            "normalized_arm_command_target_prefix_digest": (
                self.normalized_arm_command_target_prefix_digest
            ),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CandidateEvaluationReceipt:
    """可持久化 shadow 结果；类型本身不允许携带替代动作。"""

    identity: CandidateEvaluationIdentity
    status: CandidateEvaluationStatus
    reason_codes: tuple[str, ...]
    request_digest: str
    raw_action_digest: str
    effective_action_digest: str
    normalized_arm_command_target_prefix_digest: str
    evaluation_payload_digest: str | None
    world_model_architecture: str | None
    world_model_config_digest: str | None
    world_model_checkpoint_id: str | None
    critic_version: str | None
    critic_checkpoint_id: str | None
    calibration_id: str | None
    latency_ms: float
    action_parity_equal: bool = True
    actuation_allowed: bool = False
    contract_version: str = CANDIDATE_EVALUATION_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.identity, CandidateEvaluationIdentity):
            raise TypeError("identity 必须是 CandidateEvaluationIdentity")
        if self.contract_version != CANDIDATE_EVALUATION_CONTRACT_VERSION:
            raise ValueError("receipt contract_version 不受支持")
        if not isinstance(self.status, CandidateEvaluationStatus):
            raise TypeError("status 必须是 CandidateEvaluationStatus")
        if not isinstance(self.reason_codes, tuple):
            raise TypeError("reason_codes 必须是 tuple")
        if any(not isinstance(reason, str) or not reason.strip() for reason in self.reason_codes):
            raise ValueError("reason_codes 必须是非空字符串")
        for name in (
            "world_model_architecture",
            "world_model_checkpoint_id",
            "critic_version",
            "critic_checkpoint_id",
            "calibration_id",
        ):
            value = getattr(self, name)
            if value is not None:
                _nonempty(value, name)
        if (
            self.world_model_architecture is None
            and (
                self.world_model_config_digest is not None
                or self.world_model_checkpoint_id is not None
            )
        ):
            raise ValueError(
                "没有 world_model_architecture 时不能绑定 config digest 或 checkpoint"
            )
        if (
            self.world_model_architecture is not None
            and (
                self.world_model_config_digest is None
                or self.world_model_checkpoint_id is None
            )
        ):
            raise ValueError(
                "world model receipt 必须同时绑定 config digest 与 checkpoint identity"
            )
        if self.world_model_config_digest is not None:
            _validate_sha256(
                self.world_model_config_digest,
                "world_model_config_digest",
            )
        if self.critic_version is None and self.critic_checkpoint_id is not None:
            raise ValueError("没有 critic_version 时不能绑定 critic checkpoint")
        for name in (
            "request_digest",
            "raw_action_digest",
            "effective_action_digest",
            "normalized_arm_command_target_prefix_digest",
        ):
            _validate_sha256(getattr(self, name), name)
        if self.evaluation_payload_digest is not None:
            _validate_sha256(self.evaluation_payload_digest, "evaluation_payload_digest")
        if (
            self.status is CandidateEvaluationStatus.SCORED
            and self.evaluation_payload_digest is None
        ):
            raise ValueError("scored receipt 必须绑定 evaluation_payload_digest")
        if (
            isinstance(self.latency_ms, bool)
            or not isinstance(self.latency_ms, (int, float))
            or not math.isfinite(self.latency_ms)
        ):
            raise ValueError("latency_ms 必须是有限数值")
        if self.latency_ms < 0.0:
            raise ValueError("latency_ms 不能为负数")
        if not isinstance(self.action_parity_equal, bool):
            raise TypeError("action_parity_equal 必须为 bool")
        if not self.action_parity_equal and "action_parity_mismatch" not in self.reason_codes:
            raise ValueError("action parity 失败必须记录 action_parity_mismatch 原因码")
        if not isinstance(self.actuation_allowed, bool):
            raise TypeError("actuation_allowed 必须为 bool")
        if self.actuation_allowed:
            raise ValueError("V0 candidate 永不授予 actuator 权限")

    @classmethod
    def from_request(
        cls,
        request: CandidateEvaluationRequest,
        *,
        status: CandidateEvaluationStatus,
        reason_codes: tuple[str, ...],
        evaluation_payload_digest: str | None,
        world_model_architecture: str | None,
        world_model_config_digest: str | None,
        world_model_checkpoint_id: str | None,
        critic_version: str | None,
        critic_checkpoint_id: str | None,
        calibration_id: str | None,
        latency_ms: float,
        action_parity_equal: bool = True,
    ) -> CandidateEvaluationReceipt:
        """从已校验请求构造绑定 identity 与输入 digest 的 receipt。"""

        if not isinstance(request, CandidateEvaluationRequest):
            raise TypeError("request 必须是 CandidateEvaluationRequest")
        return cls(
            identity=request.identity,
            status=status,
            reason_codes=reason_codes,
            request_digest=request.request_digest,
            raw_action_digest=request.raw_action_digest,
            effective_action_digest=request.effective_action_digest,
            normalized_arm_command_target_prefix_digest=(
                request.normalized_arm_command_target_prefix_digest
            ),
            evaluation_payload_digest=evaluation_payload_digest,
            world_model_architecture=world_model_architecture,
            world_model_config_digest=world_model_config_digest,
            world_model_checkpoint_id=world_model_checkpoint_id,
            critic_version=critic_version,
            critic_checkpoint_id=critic_checkpoint_id,
            calibration_id=calibration_id,
            latency_ms=latency_ms,
            action_parity_equal=action_parity_equal,
        )

    def validate_against(self, request: CandidateEvaluationRequest) -> None:
        """拒绝把 receipt 与不同 Episode、时间或 Tensor 的请求拼接。"""

        if not isinstance(request, CandidateEvaluationRequest):
            raise TypeError("request 必须是 CandidateEvaluationRequest")
        mismatches: list[str] = []
        if self.identity != request.identity:
            mismatches.append("identity")
        if self.request_digest != request.request_digest:
            mismatches.append("request_digest")
        if self.raw_action_digest != request.raw_action_digest:
            mismatches.append("raw_action_digest")
        if self.effective_action_digest != request.effective_action_digest:
            mismatches.append("effective_action_digest")
        if (
            self.normalized_arm_command_target_prefix_digest
            != request.normalized_arm_command_target_prefix_digest
        ):
            mismatches.append("normalized_arm_command_target_prefix_digest")
        if mismatches:
            raise ValueError(f"receipt 与 request 不匹配: {','.join(mismatches)}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "identity": {
                "episode_id": self.identity.episode_id,
                "candidate_id": self.identity.candidate_id,
                "task_id": self.identity.task_id,
                "source_domain": self.identity.source_domain,
                "observation_schema_id": self.identity.observation_schema_id,
                "action_schema_id": self.identity.action_schema_id,
                "policy_checkpoint_id": self.identity.policy_checkpoint_id,
                "proprio_stats_id": self.identity.proprio_stats_id,
            },
            "status": self.status.value,
            "reason_codes": list(self.reason_codes),
            "request_digest": self.request_digest,
            "raw_action_digest": self.raw_action_digest,
            "effective_action_digest": self.effective_action_digest,
            "normalized_arm_command_target_prefix_digest": (
                self.normalized_arm_command_target_prefix_digest
            ),
            "evaluation_payload_digest": self.evaluation_payload_digest,
            "world_model_architecture": self.world_model_architecture,
            "world_model_config_digest": self.world_model_config_digest,
            "world_model_checkpoint_id": self.world_model_checkpoint_id,
            "critic_version": self.critic_version,
            "critic_checkpoint_id": self.critic_checkpoint_id,
            "calibration_id": self.calibration_id,
            "latency_ms": float(self.latency_ms),
            "action_parity_equal": self.action_parity_equal,
            "actuation_allowed": self.actuation_allowed,
        }


class ShadowCandidateObserver(Protocol):
    """外部 harness 可实现的只读 observer；接口不给 controller 或 executor。"""

    def evaluate(self, request: CandidateEvaluationRequest) -> CandidateEvaluationReceipt: ...


__all__ = [
    "CANDIDATE_EVALUATION_CONTRACT_VERSION",
    "CandidateEvaluationIdentity",
    "CandidateEvaluationReceipt",
    "CandidateEvaluationRequest",
    "CandidateEvaluationStatus",
    "NORMALIZED_ACTION_SCHEMA_ID",
    "NORMALIZED_PROPRIO_ABS_LIMIT",
    "NORMALIZED_PROPRIO_COMMAND_SCHEMA_ID",
    "ShadowCandidateObserver",
    "action_digest",
]
