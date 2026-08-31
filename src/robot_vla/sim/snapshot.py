"""E012a CollectionSession 的可恢复诊断 snapshot。

Snapshot 只用于 provenance 与 round-trip 诊断。正式 Local DAgger trajectory
仍必须来自原始 live session，禁止从 restore 分支拼接训练数据。
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import random
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

import numpy as np

SNAPSHOT_CONTRACT_VERSION = "robot-vla-collection-snapshot/v1"


class SnapshotContractError(RuntimeError):
    """Snapshot 缺少恢复完整性所需的状态。"""


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _clone_tree(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "clone"):
        return value.detach().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _clone_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tree(item) for item in value)
    return copy.deepcopy(value)


def _tree_sha256(value: Any) -> str:
    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, dict):
            digest.update(b"dict{")
            for key in sorted(item, key=lambda candidate: str(candidate)):
                digest.update(str(key).encode("utf-8"))
                digest.update(b"=")
                update(item[key])
            digest.update(b"}")
            return
        if isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode("ascii"))
            digest.update(str(len(item)).encode("ascii"))
            for child in item:
                update(child)
            return
        if is_dataclass(item):
            update(asdict(item))
            return
        if isinstance(item, (str, int, float, bool)) or item is None:
            digest.update(repr(item).encode("utf-8"))
            return
        array = _as_numpy(item)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(np.ascontiguousarray(array).tobytes())

    update(value)
    return digest.hexdigest()


def _json_sha256(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _rng_states(batched_rng: Any, name: str) -> tuple[Any, ...]:
    rngs = getattr(batched_rng, "rngs", None)
    if not isinstance(rngs, list) or not rngs:
        raise SnapshotContractError(f"{name} 没有可枚举的 per-env RNG")
    return tuple(_clone_tree(rng.get_state()) for rng in rngs)


def _restore_rng_states(batched_rng: Any, states: tuple[Any, ...], name: str) -> None:
    rngs = getattr(batched_rng, "rngs", None)
    if not isinstance(rngs, list) or len(rngs) != len(states):
        raise SnapshotContractError(f"{name} RNG batch size 与 snapshot 不一致")
    for rng, state in zip(rngs, states, strict=True):
        rng.set_state(_clone_tree(state))


def _set_cloned_attribute(owner: Any, name: str, value: Any) -> None:
    current = getattr(owner, name, None)
    if hasattr(current, "copy_") and hasattr(value, "to"):
        current.copy_(value.to(device=current.device, dtype=current.dtype))
    elif isinstance(current, np.ndarray) and isinstance(value, np.ndarray):
        if current.shape != value.shape or current.dtype != value.dtype:
            setattr(owner, name, value.copy())
        else:
            current[...] = value
    else:
        setattr(owner, name, _clone_tree(value))


@dataclass(frozen=True)
class ControllerTargetSnapshot:
    uid: str
    step: Any
    start_qpos: Any
    target_qpos: Any

    def to_summary(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "step": int(_as_numpy(self.step).reshape(-1)[0]),
            "start_qpos_sha256": _tree_sha256(self.start_qpos),
            "target_qpos_sha256": _tree_sha256(self.target_qpos),
            "target_qpos": _as_numpy(self.target_qpos).tolist(),
        }


@dataclass(frozen=True)
class TrackerSnapshot:
    active_skill_id: int
    stable_grasp_steps: int
    stable_place_steps: int
    task_completed: bool
    progress: Any


@dataclass(frozen=True)
class PolicyLoopSnapshot:
    control_step: int
    consecutive_anomaly_replans: int
    runtime_sample_index: int
    runtime_last_sampling_trace: Any
    temporal_chunks: tuple[Any, ...]
    temporal_next_sequence_id: int
    rtc_previous_chunk: Any


@dataclass(frozen=True)
class WrapperSnapshot:
    index: int
    class_name: str
    local_elapsed_steps: Any | None
    local_max_episode_steps: Any | None

    def to_summary(self) -> dict[str, Any]:
        def json_value(value: Any | None) -> Any | None:
            if value is None:
                return None
            array = _as_numpy(value)
            if array.ndim == 0:
                return array.item()
            return array.tolist()

        return {
            "index": self.index,
            "class_name": self.class_name,
            "local_elapsed_steps": json_value(self.local_elapsed_steps),
            "local_max_episode_steps": json_value(self.local_max_episode_steps),
        }


@dataclass(frozen=True)
class SnapshotEvidence:
    observation_hashes: dict[str, str]
    physical_proprio: np.ndarray
    predicate_state: dict[str, Any]
    contact_forces_n: tuple[float, float]
    camera_calibration: dict[str, Any]
    controller_targets_sha256: str

    def to_summary(self) -> dict[str, Any]:
        return {
            "observation_hashes": dict(self.observation_hashes),
            "physical_proprio": self.physical_proprio.tolist(),
            "predicate_state": dict(self.predicate_state),
            "contact_forces_n": list(self.contact_forces_n),
            "camera_calibration_sha256": _json_sha256(self.camera_calibration),
            "controller_targets_sha256": self.controller_targets_sha256,
        }


@dataclass(frozen=True)
class SnapshotRoundTripTolerances:
    """ManiSkill 3.0.1 / PhysX CPU restore 的预注册数值容差。"""

    immediate_proprio_atol: float = 0.0
    immediate_predicate_atol: float = 1e-7
    immediate_camera_atol: float = 1e-6
    immediate_rgb_max_abs_error: float = 1.0
    immediate_rgb_mean_abs_error: float = 2e-3
    next_state_atol: float = 2e-4
    next_controller_target_atol: float = 1e-7
    next_predicate_atol: float = 1e-7
    next_contact_force_atol_n: float = 1e-3

    def __post_init__(self) -> None:
        values = asdict(self)
        if any(not math.isfinite(value) or value < 0.0 for value in values.values()):
            raise ValueError("Snapshot round-trip tolerance 必须是有限非负数")


@dataclass(frozen=True)
class SnapshotRoundTripReport:
    passed: bool
    immediate_observation_hashes_equal: bool
    immediate_camera_calibration_equal: bool
    immediate_camera_max_abs_error: float
    immediate_controller_targets_equal: bool
    immediate_segmentation_equal: bool
    immediate_rgb_max_abs_error: float
    immediate_rgb_mean_abs_error: float
    immediate_proprio_max_abs_error: float
    immediate_predicate_max_abs_error: float
    immediate_is_grasped_equal: bool
    immediate_contact_force_max_abs_error_n: float
    main_rng_equal: bool
    episode_rng_equal: bool
    repeated_restore_observation_hashes_equal: bool
    next_state_max_abs_error: float
    next_controller_target_max_abs_error: float
    next_predicate_max_abs_error: float
    next_is_grasped_matches_source: bool
    next_contact_force_max_abs_error_n: float
    next_outcome_equal: bool
    tolerances: SnapshotRoundTripTolerances

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "immediate_observation_hashes_equal": (
                self.immediate_observation_hashes_equal
            ),
            "immediate_camera_calibration_equal": (
                self.immediate_camera_calibration_equal
            ),
            "immediate_camera_max_abs_error": self.immediate_camera_max_abs_error,
            "immediate_controller_targets_equal": (
                self.immediate_controller_targets_equal
            ),
            "immediate_segmentation_equal": self.immediate_segmentation_equal,
            "immediate_rgb_max_abs_error": self.immediate_rgb_max_abs_error,
            "immediate_rgb_mean_abs_error": self.immediate_rgb_mean_abs_error,
            "immediate_proprio_max_abs_error": (
                self.immediate_proprio_max_abs_error
            ),
            "immediate_predicate_max_abs_error": (
                self.immediate_predicate_max_abs_error
            ),
            "immediate_is_grasped_equal": self.immediate_is_grasped_equal,
            "immediate_contact_force_max_abs_error_n": (
                self.immediate_contact_force_max_abs_error_n
            ),
            "main_rng_equal": self.main_rng_equal,
            "episode_rng_equal": self.episode_rng_equal,
            "repeated_restore_observation_hashes_equal": (
                self.repeated_restore_observation_hashes_equal
            ),
            "next_state_max_abs_error": self.next_state_max_abs_error,
            "next_controller_target_max_abs_error": (
                self.next_controller_target_max_abs_error
            ),
            "next_predicate_max_abs_error": self.next_predicate_max_abs_error,
            "next_is_grasped_matches_source": (
                self.next_is_grasped_matches_source
            ),
            "next_contact_force_max_abs_error_n": (
                self.next_contact_force_max_abs_error_n
            ),
            "next_outcome_equal": self.next_outcome_equal,
            "tolerances": asdict(self.tolerances),
        }


@dataclass(frozen=True)
class CollectionSnapshotBundle:
    label: str
    replan_index: int
    environment_seed: int
    environment_state: dict[str, Any]
    controller_targets: tuple[ControllerTargetSnapshot, ...]
    main_rng_states: tuple[Any, ...]
    episode_rng_states: tuple[Any, ...]
    episode_seed: np.ndarray
    numpy_rng_state: Any
    python_rng_state: Any
    torch_cpu_rng_state: Any | None
    torch_cuda_rng_states: tuple[Any, ...]
    wrappers: tuple[WrapperSnapshot, ...]
    last_observation: dict[str, Any]
    tracker: TrackerSnapshot
    previous_command_q: np.ndarray
    session_done: bool
    policy_loop: PolicyLoopSnapshot
    evidence: SnapshotEvidence
    collection_policy_identity: dict[str, Any]
    version: str = SNAPSHOT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.version != SNAPSHOT_CONTRACT_VERSION:
            raise ValueError("Snapshot contract version 不兼容")
        if not self.label.strip() or self.replan_index < 0 or self.environment_seed < 0:
            raise ValueError("Snapshot label/replan/environment seed 无效")
        if not self.controller_targets:
            raise SnapshotContractError("Snapshot 缺少 controller target")
        previous = np.asarray(self.previous_command_q)
        if previous.ndim != 1 or not np.isfinite(previous).all():
            raise ValueError("Snapshot previous_command_q 必须是一维有限数组")
        _json_sha256(self.collection_policy_identity)

    def to_summary(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "label": self.label,
            "replan_index": self.replan_index,
            "environment_seed": self.environment_seed,
            "environment_state_sha256": _tree_sha256(self.environment_state),
            "controller_targets": [
                target.to_summary() for target in self.controller_targets
            ],
            "main_rng_sha256": _tree_sha256(self.main_rng_states),
            "episode_rng_sha256": _tree_sha256(self.episode_rng_states),
            "episode_seed": self.episode_seed.tolist(),
            "wrappers": [wrapper.to_summary() for wrapper in self.wrappers],
            "tracker": {
                "active_skill_id": self.tracker.active_skill_id,
                "stable_grasp_steps": self.tracker.stable_grasp_steps,
                "stable_place_steps": self.tracker.stable_place_steps,
                "task_completed": self.tracker.task_completed,
            },
            "policy": {
                "control_step": self.policy_loop.control_step,
                "runtime_sample_index": self.policy_loop.runtime_sample_index,
                "temporal_buffer_size": len(self.policy_loop.temporal_chunks),
                "temporal_next_sequence_id": (
                    self.policy_loop.temporal_next_sequence_id
                ),
            },
            "previous_command_q": self.previous_command_q.tolist(),
            "evidence": self.evidence.to_summary(),
            "collection_policy_identity": dict(self.collection_policy_identity),
            "collection_policy_identity_sha256": _json_sha256(
                self.collection_policy_identity
            ),
        }


class CollectionSnapshotRing:
    """只保留 boundary 前一个 Replan、crossing Replan 与 crossing state。"""

    def __init__(self, capacity: int = 3) -> None:
        if capacity < 3:
            raise ValueError("Snapshot ring 至少需要 3 个 slot")
        self._snapshots: deque[CollectionSnapshotBundle] = deque(maxlen=capacity)

    def append(self, snapshot: CollectionSnapshotBundle) -> None:
        if self._snapshots and snapshot.replan_index < self._snapshots[-1].replan_index:
            raise SnapshotContractError("Snapshot replan_index 不能倒退")
        self._snapshots.append(snapshot)

    @property
    def snapshots(self) -> tuple[CollectionSnapshotBundle, ...]:
        return tuple(self._snapshots)

    def summaries(self) -> tuple[dict[str, Any], ...]:
        return tuple(snapshot.to_summary() for snapshot in self._snapshots)


def _capture_controller_targets(base_env: Any) -> tuple[ControllerTargetSnapshot, ...]:
    controller = getattr(getattr(base_env, "agent", None), "controller", None)
    controllers = getattr(controller, "controllers", None)
    if not isinstance(controllers, dict) or not controllers:
        raise SnapshotContractError("Snapshot 只支持可枚举的 ManiSkill CombinedController")
    snapshots: list[ControllerTargetSnapshot] = []
    for uid, child in sorted(controllers.items()):
        missing = [
            name
            for name in ("_step", "_start_qpos", "_target_qpos")
            if not hasattr(child, name)
        ]
        if missing:
            raise SnapshotContractError(f"Controller {uid} 缺少 target state: {missing}")
        snapshots.append(
            ControllerTargetSnapshot(
                uid=str(uid),
                step=_clone_tree(child._step),
                start_qpos=_clone_tree(child._start_qpos),
                target_qpos=_clone_tree(child._target_qpos),
            )
        )
    return tuple(snapshots)


def _capture_wrappers(env: Any) -> tuple[WrapperSnapshot, ...]:
    snapshots: list[WrapperSnapshot] = []
    current = env
    index = 0
    while True:
        local = vars(current)
        snapshots.append(
            WrapperSnapshot(
                index=index,
                class_name=f"{type(current).__module__}.{type(current).__name__}",
                local_elapsed_steps=_clone_tree(local.get("_elapsed_steps")),
                local_max_episode_steps=_clone_tree(local.get("_max_episode_steps")),
            )
        )
        if "env" not in local:
            break
        current = local["env"]
        index += 1
    return tuple(snapshots)


def _capture_torch_rng() -> tuple[Any | None, tuple[Any, ...]]:
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return None, ()
    cpu = torch.random.get_rng_state().clone()
    cuda = tuple(state.clone() for state in torch.cuda.get_rng_state_all())
    return cpu, cuda


def _observation_hashes(observation: dict[str, Any], proprio: np.ndarray) -> dict[str, str]:
    sensor_data = observation["sensor_data"]
    values = {
        "rgb_external": sensor_data["base_camera"]["rgb"],
        "rgb_wrist": sensor_data["hand_camera"]["rgb"],
        "segmentation_external": sensor_data["base_camera"]["segmentation"],
        "segmentation_wrist": sensor_data["hand_camera"]["segmentation"],
        "proprio": proprio,
    }
    return {name: _tree_sha256(value) for name, value in values.items()}


def build_snapshot_evidence(
    observation: dict[str, Any],
    *,
    physical_proprio: np.ndarray,
    predicate_state: Any,
    contact_forces_n: tuple[float, float],
    camera_calibration: Any,
    controller_targets: tuple[ControllerTargetSnapshot, ...],
) -> SnapshotEvidence:
    proprio = np.asarray(physical_proprio, dtype=np.float32)
    if proprio.ndim != 1 or not np.isfinite(proprio).all():
        raise ValueError("Snapshot physical_proprio 必须是一维有限 float32")
    contacts = tuple(float(value) for value in contact_forces_n)
    if len(contacts) != 2 or not all(math.isfinite(value) for value in contacts):
        raise ValueError("Snapshot contact force 必须是两个有限值")
    predicate = asdict(predicate_state) if is_dataclass(predicate_state) else dict(predicate_state)
    calibration = (
        camera_calibration.to_dict()
        if hasattr(camera_calibration, "to_dict")
        else dict(camera_calibration)
    )
    return SnapshotEvidence(
        observation_hashes=_observation_hashes(observation, proprio),
        physical_proprio=proprio.copy(),
        predicate_state=predicate,
        contact_forces_n=(contacts[0], contacts[1]),
        camera_calibration=calibration,
        controller_targets_sha256=_tree_sha256(controller_targets),
    )


def capture_collection_snapshot(
    env: Any,
    session: Any,
    loop: Any,
    *,
    label: str,
    replan_index: int,
    environment_seed: int,
    physical_proprio: np.ndarray,
    predicate_state: Any,
    contact_forces_n: tuple[float, float],
    camera_calibration: Any,
    collection_policy_identity: dict[str, Any],
) -> CollectionSnapshotBundle:
    """在 Action 边界捕获完整、可恢复的 CollectionSession 状态。"""

    base_env = env.unwrapped
    if not hasattr(base_env, "get_state_dict"):
        raise SnapshotContractError("ManiSkill env 缺少 get_state_dict")
    controller_targets = _capture_controller_targets(base_env)
    tracker = session.tracker
    torch_cpu_rng_state, torch_cuda_rng_states = _capture_torch_rng()
    evidence = build_snapshot_evidence(
        session.observation,
        physical_proprio=physical_proprio,
        predicate_state=predicate_state,
        contact_forces_n=contact_forces_n,
        camera_calibration=camera_calibration,
        controller_targets=controller_targets,
    )
    return CollectionSnapshotBundle(
        label=label,
        replan_index=replan_index,
        environment_seed=environment_seed,
        environment_state=_clone_tree(base_env.get_state_dict()),
        controller_targets=controller_targets,
        main_rng_states=_rng_states(base_env._batched_main_rng, "main"),
        episode_rng_states=_rng_states(base_env._batched_episode_rng, "episode"),
        episode_seed=np.asarray(base_env._episode_seed, dtype=np.int64).copy(),
        numpy_rng_state=_clone_tree(np.random.get_state()),
        python_rng_state=_clone_tree(random.getstate()),
        torch_cpu_rng_state=torch_cpu_rng_state,
        torch_cuda_rng_states=torch_cuda_rng_states,
        wrappers=_capture_wrappers(env),
        last_observation=_clone_tree(session.observation),
        tracker=TrackerSnapshot(
            active_skill_id=int(tracker._active_skill_id),
            stable_grasp_steps=int(tracker._stable_grasp_steps),
            stable_place_steps=int(tracker._stable_place_steps),
            task_completed=bool(tracker._task_completed),
            progress=_clone_tree(session.progress),
        ),
        previous_command_q=np.asarray(session.previous_command_q, dtype=np.float32).copy(),
        session_done=bool(session.done),
        policy_loop=PolicyLoopSnapshot(
            control_step=int(loop.control_step),
            consecutive_anomaly_replans=int(loop._consecutive_anomaly_replans),
            runtime_sample_index=int(loop.runtime._sample_index),
            runtime_last_sampling_trace=_clone_tree(loop.runtime._last_sampling_trace),
            temporal_chunks=tuple(_clone_tree(loop.ensembler._chunks)),
            temporal_next_sequence_id=int(loop.ensembler._next_sequence_id),
            rtc_previous_chunk=_clone_tree(loop._rtc_previous_chunk),
        ),
        evidence=evidence,
        collection_policy_identity=_clone_tree(collection_policy_identity),
    )


def restore_collection_snapshot(
    snapshot: CollectionSnapshotBundle,
    env: Any,
    *,
    session: Any | None = None,
    loop: Any | None = None,
    restore_global_rng: bool = True,
) -> None:
    """恢复 snapshot；调用方必须确保 env 配置和 collection identity 相同。"""

    if snapshot.version != SNAPSHOT_CONTRACT_VERSION:
        raise SnapshotContractError("拒绝恢复未知 Snapshot contract")
    base_env = env.unwrapped
    base_env.set_state_dict(_clone_tree(snapshot.environment_state))

    controllers = getattr(base_env.agent.controller, "controllers", None)
    if not isinstance(controllers, dict):
        raise SnapshotContractError("恢复目标缺少 ManiSkill CombinedController")
    for target in snapshot.controller_targets:
        if target.uid not in controllers:
            raise SnapshotContractError(f"恢复目标缺少 Controller {target.uid}")
        controller = controllers[target.uid]
        _set_cloned_attribute(controller, "_step", target.step)
        _set_cloned_attribute(controller, "_start_qpos", target.start_qpos)
        _set_cloned_attribute(controller, "_target_qpos", target.target_qpos)

    current = env
    for expected in snapshot.wrappers:
        actual_name = f"{type(current).__module__}.{type(current).__name__}"
        if actual_name != expected.class_name:
            raise SnapshotContractError(
                f"Wrapper chain 不一致：期望 {expected.class_name}，实际 {actual_name}"
            )
        if expected.local_elapsed_steps is not None:
            _set_cloned_attribute(current, "_elapsed_steps", expected.local_elapsed_steps)
        if "env" not in vars(current):
            break
        current = vars(current)["env"]

    _restore_rng_states(base_env._batched_main_rng, snapshot.main_rng_states, "main")
    _restore_rng_states(
        base_env._batched_episode_rng,
        snapshot.episode_rng_states,
        "episode",
    )
    base_env._episode_seed = snapshot.episode_seed.copy()
    base_env._episode_rng = base_env._batched_episode_rng.rngs[0]
    base_env._last_obs = _clone_tree(snapshot.last_observation)

    if restore_global_rng:
        np.random.set_state(_clone_tree(snapshot.numpy_rng_state))
        random.setstate(_clone_tree(snapshot.python_rng_state))
        if snapshot.torch_cpu_rng_state is not None:
            torch = importlib.import_module("torch")
            torch.random.set_rng_state(snapshot.torch_cpu_rng_state.clone())
            if len(torch.cuda.get_rng_state_all()) != len(snapshot.torch_cuda_rng_states):
                raise SnapshotContractError("CUDA RNG device count 与 snapshot 不一致")
            torch.cuda.set_rng_state_all(
                [state.clone() for state in snapshot.torch_cuda_rng_states]
            )

    if session is not None:
        session.observation = _clone_tree(snapshot.last_observation)
        session.tracker._active_skill_id = snapshot.tracker.active_skill_id
        session.tracker._stable_grasp_steps = snapshot.tracker.stable_grasp_steps
        session.tracker._stable_place_steps = snapshot.tracker.stable_place_steps
        session.tracker._task_completed = snapshot.tracker.task_completed
        session.progress = _clone_tree(snapshot.tracker.progress)
        session.previous_command_q = snapshot.previous_command_q.copy()
        session.done = snapshot.session_done

    if loop is not None:
        loop.control_step = snapshot.policy_loop.control_step
        loop._consecutive_anomaly_replans = (
            snapshot.policy_loop.consecutive_anomaly_replans
        )
        loop.runtime._sample_index = snapshot.policy_loop.runtime_sample_index
        loop.runtime._last_sampling_trace = _clone_tree(
            snapshot.policy_loop.runtime_last_sampling_trace
        )
        loop.ensembler._chunks = list(
            _clone_tree(snapshot.policy_loop.temporal_chunks)
        )
        loop.ensembler._next_sequence_id = (
            snapshot.policy_loop.temporal_next_sequence_id
        )
        loop._rtc_previous_chunk = _clone_tree(snapshot.policy_loop.rtc_previous_chunk)


def _numeric_tree_max_abs_error(first: Any, second: Any) -> float:
    if isinstance(first, dict) and isinstance(second, dict):
        if set(first) != set(second):
            return math.inf
        return max(
            (_numeric_tree_max_abs_error(first[key], second[key]) for key in first),
            default=0.0,
        )
    if isinstance(first, (list, tuple)) and isinstance(second, (list, tuple)):
        if len(first) != len(second):
            return math.inf
        return max(
            (
                _numeric_tree_max_abs_error(left, right)
                for left, right in zip(first, second, strict=True)
            ),
            default=0.0,
        )
    if is_dataclass(first) and is_dataclass(second):
        return _numeric_tree_max_abs_error(asdict(first), asdict(second))
    try:
        left = _as_numpy(first)
        right = _as_numpy(second)
    except Exception:  # noqa: BLE001 - 非数值 leaf 必须精确比较
        return 0.0 if first == second else math.inf
    if left.shape != right.shape:
        return math.inf
    if left.dtype.kind in "OUS" or right.dtype.kind in "OUS":
        return 0.0 if np.array_equal(left, right) else math.inf
    if left.dtype.kind == "b" or right.dtype.kind == "b":
        return 0.0 if np.array_equal(left, right) else math.inf
    if left.size == 0:
        return 0.0
    return float(
        np.max(
            np.abs(
                left.astype(np.float64, copy=False)
                - right.astype(np.float64, copy=False)
            )
        )
    )


def _rng_probe(base_env: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
    main = tuple(
        int(rng.randint(2**31)) for rng in base_env._batched_main_rng.rngs
    )
    episode = tuple(
        int(rng.randint(2**31)) for rng in base_env._batched_episode_rng.rngs
    )
    return main, episode


def _step_outcome(step_output: Any) -> dict[str, Any]:
    if not isinstance(step_output, tuple) or len(step_output) != 5:
        raise SnapshotContractError("Gymnasium step output 必须是五元组")
    _, reward, terminated, truncated, info = step_output
    return {
        "reward": _clone_tree(reward),
        "terminated": _clone_tree(terminated),
        "truncated": _clone_tree(truncated),
        "success": _clone_tree(info.get("success")),
        "elapsed_steps": _clone_tree(info.get("elapsed_steps")),
    }


def _sensor_restore_errors(
    expected_observation: dict[str, Any],
    observations: tuple[dict[str, Any], ...],
) -> tuple[bool, float, float]:
    segmentation_equal = True
    rgb_max_error = 0.0
    rgb_sum_error = 0.0
    rgb_elements = 0
    expected_sensors = expected_observation["sensor_data"]
    for observation in observations:
        actual_sensors = observation["sensor_data"]
        for camera in ("base_camera", "hand_camera"):
            expected_segmentation = _as_numpy(
                expected_sensors[camera]["segmentation"]
            )
            actual_segmentation = _as_numpy(actual_sensors[camera]["segmentation"])
            segmentation_equal &= np.array_equal(
                expected_segmentation,
                actual_segmentation,
            )
            expected_rgb = _as_numpy(expected_sensors[camera]["rgb"]).astype(
                np.int16
            )
            actual_rgb = _as_numpy(actual_sensors[camera]["rgb"]).astype(np.int16)
            if expected_rgb.shape != actual_rgb.shape:
                return False, math.inf, math.inf
            difference = np.abs(expected_rgb - actual_rgb)
            rgb_max_error = max(rgb_max_error, float(np.max(difference, initial=0)))
            rgb_sum_error += float(np.sum(difference, dtype=np.float64))
            rgb_elements += difference.size
    rgb_mean_error = rgb_sum_error / rgb_elements if rgb_elements else 0.0
    return segmentation_equal, rgb_max_error, rgb_mean_error


def verify_snapshot_round_trip(
    snapshot: CollectionSnapshotBundle,
    env: Any,
    *,
    controller_action: np.ndarray,
    evidence_builder: Callable[
        [dict[str, Any], tuple[ControllerTargetSnapshot, ...]],
        SnapshotEvidence,
    ],
    tolerances: SnapshotRoundTripTolerances | None = None,
) -> SnapshotRoundTripReport:
    """在独立 env 上执行 restore→一步→restore→同一步的确定性诊断。"""

    resolved_tolerances = tolerances or SnapshotRoundTripTolerances()
    action = np.asarray(controller_action, dtype=np.float32)
    if action.ndim != 1 or not np.isfinite(action).all():
        raise ValueError("Snapshot round-trip controller_action 必须是一维有限数组")
    base_env = env.unwrapped

    restore_collection_snapshot(snapshot, env, restore_global_rng=False)
    first_observation = base_env.get_obs()
    first_observation_for_comparison = _clone_tree(first_observation)
    first_evidence = evidence_builder(
        first_observation,
        _capture_controller_targets(base_env),
    )
    first_rng = _rng_probe(base_env)
    first_step = env.step(action)
    first_next_evidence = evidence_builder(
        first_step[0],
        _capture_controller_targets(base_env),
    )
    first_next_state = _clone_tree(base_env.get_state_dict())
    first_next_controller = _capture_controller_targets(base_env)
    first_outcome = _step_outcome(first_step)

    restore_collection_snapshot(snapshot, env, restore_global_rng=False)
    second_observation = base_env.get_obs()
    second_observation_for_comparison = _clone_tree(second_observation)
    second_evidence = evidence_builder(
        second_observation,
        _capture_controller_targets(base_env),
    )
    second_rng = _rng_probe(base_env)
    second_step = env.step(action)
    second_next_evidence = evidence_builder(
        second_step[0],
        _capture_controller_targets(base_env),
    )
    second_next_state = _clone_tree(base_env.get_state_dict())
    second_next_controller = _capture_controller_targets(base_env)
    second_outcome = _step_outcome(second_step)

    observation_hashes_equal = (
        first_evidence.observation_hashes == snapshot.evidence.observation_hashes
    )
    repeated_hashes_equal = (
        second_evidence.observation_hashes == first_evidence.observation_hashes
    )
    camera_equal = (
        first_evidence.camera_calibration == snapshot.evidence.camera_calibration
        and second_evidence.camera_calibration == snapshot.evidence.camera_calibration
    )
    camera_error = max(
        _numeric_tree_max_abs_error(
            first_evidence.camera_calibration,
            snapshot.evidence.camera_calibration,
        ),
        _numeric_tree_max_abs_error(
            second_evidence.camera_calibration,
            snapshot.evidence.camera_calibration,
        ),
    )
    controller_equal = (
        first_evidence.controller_targets_sha256
        == snapshot.evidence.controller_targets_sha256
        and second_evidence.controller_targets_sha256
        == snapshot.evidence.controller_targets_sha256
    )
    proprio_error = max(
        _numeric_tree_max_abs_error(
            first_evidence.physical_proprio,
            snapshot.evidence.physical_proprio,
        ),
        _numeric_tree_max_abs_error(
            second_evidence.physical_proprio,
            snapshot.evidence.physical_proprio,
        ),
    )
    source_predicate_numeric = dict(snapshot.evidence.predicate_state)
    first_predicate_numeric = dict(first_evidence.predicate_state)
    second_predicate_numeric = dict(second_evidence.predicate_state)
    source_is_grasped = bool(source_predicate_numeric.pop("is_grasped"))
    first_is_grasped = bool(first_predicate_numeric.pop("is_grasped"))
    second_is_grasped = bool(second_predicate_numeric.pop("is_grasped"))
    predicate_error = max(
        _numeric_tree_max_abs_error(
            first_predicate_numeric,
            source_predicate_numeric,
        ),
        _numeric_tree_max_abs_error(
            second_predicate_numeric,
            source_predicate_numeric,
        ),
    )
    immediate_is_grasped_equal = (
        first_is_grasped == source_is_grasped
        and second_is_grasped == source_is_grasped
    )
    contact_error = max(
        _numeric_tree_max_abs_error(
            first_evidence.contact_forces_n,
            snapshot.evidence.contact_forces_n,
        ),
        _numeric_tree_max_abs_error(
            second_evidence.contact_forces_n,
            snapshot.evidence.contact_forces_n,
        ),
    )
    next_state_error = _numeric_tree_max_abs_error(
        first_next_state,
        second_next_state,
    )
    next_controller_error = _numeric_tree_max_abs_error(
        first_next_controller,
        second_next_controller,
    )
    next_predicate_error = _numeric_tree_max_abs_error(
        first_next_evidence.predicate_state,
        second_next_evidence.predicate_state,
    )
    next_is_grasped_matches_source = (
        bool(first_next_evidence.predicate_state["is_grasped"])
        == source_is_grasped
        and bool(second_next_evidence.predicate_state["is_grasped"])
        == source_is_grasped
    )
    next_contact_error = _numeric_tree_max_abs_error(
        first_next_evidence.contact_forces_n,
        second_next_evidence.contact_forces_n,
    )
    segmentation_equal, rgb_max_error, rgb_mean_error = _sensor_restore_errors(
        snapshot.last_observation,
        (first_observation_for_comparison, second_observation_for_comparison),
    )
    next_outcome_equal = _numeric_tree_max_abs_error(first_outcome, second_outcome) == 0.0
    passed = all(
        (
            repeated_hashes_equal,
            camera_error <= resolved_tolerances.immediate_camera_atol,
            controller_equal,
            segmentation_equal,
            rgb_max_error <= resolved_tolerances.immediate_rgb_max_abs_error,
            rgb_mean_error <= resolved_tolerances.immediate_rgb_mean_abs_error,
            proprio_error <= resolved_tolerances.immediate_proprio_atol,
            predicate_error <= resolved_tolerances.immediate_predicate_atol,
            first_rng[0] == second_rng[0],
            first_rng[1] == second_rng[1],
            next_state_error <= resolved_tolerances.next_state_atol,
            next_controller_error
            <= resolved_tolerances.next_controller_target_atol,
            next_predicate_error <= resolved_tolerances.next_predicate_atol,
            next_is_grasped_matches_source,
            next_contact_error <= resolved_tolerances.next_contact_force_atol_n,
            next_outcome_equal,
        )
    )
    return SnapshotRoundTripReport(
        passed=passed,
        immediate_observation_hashes_equal=observation_hashes_equal,
        immediate_camera_calibration_equal=camera_equal,
        immediate_camera_max_abs_error=camera_error,
        immediate_controller_targets_equal=controller_equal,
        immediate_segmentation_equal=segmentation_equal,
        immediate_rgb_max_abs_error=rgb_max_error,
        immediate_rgb_mean_abs_error=rgb_mean_error,
        immediate_proprio_max_abs_error=proprio_error,
        immediate_predicate_max_abs_error=predicate_error,
        immediate_is_grasped_equal=immediate_is_grasped_equal,
        immediate_contact_force_max_abs_error_n=contact_error,
        main_rng_equal=first_rng[0] == second_rng[0],
        episode_rng_equal=first_rng[1] == second_rng[1],
        repeated_restore_observation_hashes_equal=repeated_hashes_equal,
        next_state_max_abs_error=next_state_error,
        next_controller_target_max_abs_error=next_controller_error,
        next_predicate_max_abs_error=next_predicate_error,
        next_is_grasped_matches_source=next_is_grasped_matches_source,
        next_contact_force_max_abs_error_n=next_contact_error,
        next_outcome_equal=next_outcome_equal,
        tolerances=resolved_tolerances,
    )


__all__ = [
    "SNAPSHOT_CONTRACT_VERSION",
    "CollectionSnapshotBundle",
    "CollectionSnapshotRing",
    "ControllerTargetSnapshot",
    "SnapshotContractError",
    "SnapshotEvidence",
    "SnapshotRoundTripReport",
    "SnapshotRoundTripTolerances",
    "build_snapshot_evidence",
    "capture_collection_snapshot",
    "restore_collection_snapshot",
    "verify_snapshot_round_trip",
]
