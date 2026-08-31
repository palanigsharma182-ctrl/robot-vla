"""Oracle Geometry Reach A/B 的数据、模型与在线推理边界。"""

from __future__ import annotations

import hashlib
import math
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn

from robot_vla.adapters import ActionAdapter, ProprioNormalizer
from robot_vla.contracts import PICK_AND_PLACE_SKILLS, RobotSpec
from robot_vla.data.collator import QwenVLACollator
from robot_vla.data.dataset import ActionChunkDataset
from robot_vla.data.trajectory import TrajectoryMeta, resolve_trajectory_path
from robot_vla.model.expert import StandaloneActionExpert
from robot_vla.model.layers import FP32RMSNorm
from robot_vla.model.policy import QwenVLAPolicy
from robot_vla.model.qwen_context import QwenContext
from robot_vla.observation import validate_se3
from robot_vla.runtime.policy_runtime import (
    OnlineObservation,
    RuntimeActionChunk,
    RuntimeConfig,
    SamplingTrace,
)

OracleReachMode = Literal["control", "oracle"]

ORACLE_REACH_EXPERIMENT_FORMAT = "robot-vla-oracle-geometry-reach/v1"
ORACLE_REACH_CHECKPOINT_FORMAT = "robot-vla-oracle-geometry-reach-checkpoint/v1"
ORACLE_GEOMETRY_FORMAT = "tcp-to-object-world-delta-plus-distance-m/v1"
REACH_SKILL_ID = PICK_AND_PLACE_SKILLS.index("reach")
FRANKA_TCP_LINK_NAME = "panda_hand_tcp"
FRANKA_WORLD_BASE_POSITION_M = (-0.615, 0.0, 0.0)


class FrankaTCPForwardKinematics:
    """用 SAPIEN 的 Pinocchio FK 计算 Panda TCP 世界坐标；不暴露 IK/Jacobian。"""

    def __init__(
        self,
        urdf_path: str | Path,
        spec: RobotSpec,
        *,
        base_position_world_m: Sequence[float] = FRANKA_WORLD_BASE_POSITION_M,
    ) -> None:
        try:
            from sapien.wrapper.pinocchio_model import PinocchioModel
        except ImportError as error:
            raise ImportError("Oracle Reach 离线 FK 需要 sim extra 提供的 SAPIEN") from error

        path = Path(urdf_path)
        if not path.is_file():
            raise FileNotFoundError(f"找不到 Franka URDF: {path}")
        xml = path.read_text(encoding="utf-8")
        root = ET.fromstring(xml)
        joint_order = [
            node.attrib["name"]
            for node in root.findall("joint")
            if node.attrib.get("type") != "fixed"
        ]
        link_order = [node.attrib["name"] for node in root.findall("link")]
        if tuple(joint_order) != spec.active_joint_names:
            raise ValueError(
                "Franka URDF active joint 顺序不兼容："
                f"期望 {spec.active_joint_names}，实际 {tuple(joint_order)}"
            )
        if FRANKA_TCP_LINK_NAME not in link_order:
            raise ValueError(f"Franka URDF 缺少 TCP link: {FRANKA_TCP_LINK_NAME}")
        base = np.asarray(base_position_world_m, dtype=np.float64)
        if base.shape != (3,) or not np.isfinite(base).all():
            raise ValueError("Franka world base position 必须是有限 [3] 米坐标")

        self.spec = spec
        self.urdf_path = path
        self.base_position_world_m = base
        self.world_from_base = np.eye(4, dtype=np.float64)
        self.world_from_base[:3, 3] = base
        self._model = PinocchioModel(xml, [0.0, 0.0, -9.81])
        self._model.set_joint_order(joint_order)
        self._model.set_link_order(link_order)
        self._tcp_link_index = link_order.index(FRANKA_TCP_LINK_NAME)

    def _validated_arm_q(self, arm_q_rad: np.ndarray) -> np.ndarray:
        q = np.asarray(arm_q_rad, dtype=np.float64)
        if q.shape != (self.spec.arm_dof,) or not np.isfinite(q).all():
            raise ValueError(f"Franka arm_q 应为有限 [{self.spec.arm_dof}] rad")
        return q

    def pose_base(self, arm_q_rad: np.ndarray) -> np.ndarray:
        """返回 ``base_from_tcp`` 完整 SE(3)，而不是只保留平移。"""

        q = self._validated_arm_q(arm_q_rad)
        full_q = np.zeros(len(self.spec.active_joint_names), dtype=np.float64)
        full_q[: self.spec.arm_dof] = q
        self._model.compute_forward_kinematics(full_q)
        pose = self._model.get_link_pose(self._tcp_link_index)
        if not hasattr(pose, "to_transformation_matrix"):
            raise RuntimeError("SAPIEN Pinocchio link pose 缺少完整 SE(3) 接口")
        base_from_tcp = np.asarray(pose.to_transformation_matrix(), dtype=np.float64)
        if base_from_tcp.shape == (1, 4, 4):
            base_from_tcp = base_from_tcp[0]
        try:
            return validate_se3(base_from_tcp, "base_from_tcp").astype(np.float32)
        except ValueError as error:
            raise RuntimeError("Franka FK 返回了无效 TCP base pose") from error

    def pose_world(self, arm_q_rad: np.ndarray) -> np.ndarray:
        """返回当前固定 robot base 对应的 ``world_from_tcp`` 完整 SE(3)。"""

        world_from_tcp = self.world_from_base @ self.pose_base(arm_q_rad)
        return validate_se3(world_from_tcp, "world_from_tcp").astype(np.float32)

    def __call__(self, arm_q_rad: np.ndarray) -> np.ndarray:
        """兼容 Oracle Reach 旧接口：仍只返回 TCP 世界位置。"""

        return self.pose_world(arm_q_rad)[:3, 3].copy()


def find_maniskill_panda_urdf() -> Path:
    """从已安装 ManiSkill 获取与 panda_wristcam 完全相同的 URDF。"""

    try:
        from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
    except ImportError as error:
        raise ImportError("自动定位 Franka URDF 需要 sim extra") from error
    return Path(PandaWristCam.urdf_path)


def select_geometry_entries(
    root: str | Path,
    entries: Sequence[TrajectoryMeta],
) -> list[TrajectoryMeta]:
    """只保留真实保存当前物体位置的轨迹，不推断或回填旧数据。"""

    data_root = Path(root)
    selected: list[TrajectoryMeta] = []
    for entry in entries:
        path = resolve_trajectory_path(data_root, entry.file)
        with np.load(path, allow_pickle=False) as npz:
            if "object_position_m" in npz:
                selected.append(entry)
    if not selected:
        raise ValueError("所选 split 没有包含 object_position_m 的 Oracle 轨迹")
    return selected


class OracleReachDataset(ActionChunkDataset):
    """同一当前时刻构造 geometry，并只暴露 anchor skill=reach 的窗口。"""

    def __init__(
        self,
        root: str,
        entries: Sequence[TrajectoryMeta],
        spec: RobotSpec,
        proprio_normalizer: ProprioNormalizer,
        tcp_forward_kinematics: Callable[[np.ndarray], np.ndarray],
        *,
        cache_size: int = 2,
    ) -> None:
        geometry_entries = select_geometry_entries(root, entries)
        super().__init__(
            root,
            geometry_entries,
            spec,
            proprio_normalizer,
            cache_size=cache_size,
        )
        filtered_index: list[tuple[int, int]] = []
        relative_geometry: list[np.ndarray] = []
        window_ids: list[str] = []
        for entry_index, timestep in self.index:
            arrays = self.store.get(self.entries[entry_index])
            if int(arrays.skill_id[timestep]) != REACH_SKILL_ID:
                continue
            if arrays.object_position_m is None:
                raise RuntimeError("Oracle Dataset 内部混入缺少 object_position_m 的轨迹")
            arm_q = arrays.proprio[timestep, : spec.arm_dof]
            tcp_position = np.asarray(tcp_forward_kinematics(arm_q), dtype=np.float32)
            object_position = arrays.object_position_m[timestep]
            if tcp_position.shape != (3,) or not np.isfinite(tcp_position).all():
                raise ValueError("FK 必须返回有限 [3] TCP 世界坐标，单位为米")
            delta = (object_position - tcp_position).astype(np.float32)
            distance = np.float32(np.linalg.norm(delta.astype(np.float64)))
            geometry = np.concatenate((delta, np.asarray([distance], dtype=np.float32)))
            if not np.isfinite(geometry).all() or not 0.0 <= float(distance) <= 2.0:
                raise ValueError("TCP→object geometry 超出米制工作空间合理范围 [0,2]")
            filtered_index.append((entry_index, timestep))
            relative_geometry.append(geometry)
            window_ids.append(f"{self.entries[entry_index].trajectory_id}:{timestep}")
        if not filtered_index:
            raise ValueError("Oracle Dataset 没有 anchor skill_id=reach 的有效窗口")
        self.index = filtered_index
        self.relative_geometry_m = np.stack(relative_geometry).astype(np.float32)
        self.window_ids = tuple(window_ids)
        digest = hashlib.sha256()
        for identity in self.window_ids:
            digest.update(identity.encode("utf-8"))
            digest.update(b"\n")
        self.window_sha256 = digest.hexdigest()

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = super().__getitem__(index)
        if sample["skill_id"] != REACH_SKILL_ID:
            raise RuntimeError("Oracle Reach Dataset 暴露了非 Reach anchor")
        sample["relative_geometry"] = self.relative_geometry_m[index].copy()
        return sample


def _collate_action_fields(
    samples: list[dict[str, Any]],
    spec: RobotSpec,
) -> dict[str, Any]:
    if not samples:
        raise ValueError("不能 collate 空 Oracle Reach batch")
    proprio = np.stack([sample["proprio"] for sample in samples])
    action = np.stack([sample["action"] for sample in samples])
    action_mask = np.stack([sample["action_mask"] for sample in samples])
    event_mask = np.stack([sample["event_mask"] for sample in samples])
    geometry = np.stack([sample["relative_geometry"] for sample in samples])
    expected_action = (len(samples), spec.action_horizon, spec.action_dim)
    if proprio.shape != (len(samples), spec.proprio_dim):
        raise ValueError("Oracle Reach proprio batch shape 无效")
    if action.shape != expected_action:
        raise ValueError("Oracle Reach action batch shape 无效")
    if action_mask.shape != expected_action[:2] or event_mask.shape != expected_action[:2]:
        raise ValueError("Oracle Reach action/event mask shape 无效")
    if geometry.shape != (len(samples), 4):
        raise ValueError("Oracle Reach geometry batch 应为 [B,4]")
    if proprio.dtype != np.float32 or action.dtype != np.float32 or geometry.dtype != np.float32:
        raise ValueError("Oracle Reach proprio/action/geometry 必须为 float32")
    if action_mask.dtype != np.bool_ or event_mask.dtype != np.bool_:
        raise ValueError("Oracle Reach action/event mask 必须为 bool")
    return {
        "proprio": torch.from_numpy(proprio),
        "action": torch.from_numpy(action),
        "action_mask": torch.from_numpy(action_mask),
        "event_mask": torch.from_numpy(event_mask),
        "relative_geometry": torch.from_numpy(geometry),
        "trajectory_id": [str(sample["trajectory_id"]) for sample in samples],
        "timestep": torch.tensor([int(sample["timestep"]) for sample in samples]),
        "skill_id": torch.tensor([int(sample["skill_id"]) for sample in samples]),
    }


class OracleReachCollator:
    """A/B 共用动作字段；仅 context 输入组装方式随 mode 改变。"""

    def __init__(
        self,
        mode: OracleReachMode,
        spec: RobotSpec,
        qwen_collator: QwenVLACollator | None = None,
    ) -> None:
        if mode not in {"control", "oracle"}:
            raise ValueError(f"未知 Oracle Reach mode: {mode}")
        if (mode == "control") != (qwen_collator is not None):
            raise ValueError("Control 必须提供且 Oracle 禁止提供 Qwen collator")
        self.mode = mode
        self.spec = spec
        self.qwen_collator = qwen_collator

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        common = _collate_action_fields(samples, self.spec)
        if self.mode == "control":
            assert self.qwen_collator is not None
            control = self.qwen_collator(samples)
            for name in ("proprio", "action", "action_mask", "event_mask"):
                if not torch.equal(common[name], control[name]):
                    raise RuntimeError(f"Control Qwen collator 改变了 A/B 共享字段: {name}")
            common["qwen_inputs"] = control["qwen_inputs"]
            common["visual_tokens_per_image"] = control["visual_tokens_per_image"]
            common["context_lengths"] = control["context_lengths"]
        else:
            common["qwen_inputs"] = {"relative_geometry": common["relative_geometry"]}
            common["visual_tokens_per_image"] = [(0, 0)] * len(samples)
            common["context_lengths"] = torch.ones(len(samples), dtype=torch.long)
        return common


class OracleGeometryContextEncoder(nn.Module):
    """当前 TCP→object 的 4 维米制 geometry 到单个 720 维 context token。"""

    input_dim = 4
    output_dim = 720

    def __init__(self) -> None:
        super().__init__()
        self.input_projection = nn.Linear(self.input_dim, self.output_dim)
        self.activation = nn.SiLU()
        self.output_projection = nn.Linear(self.output_dim, self.output_dim)
        self.output_norm = FP32RMSNorm(self.output_dim, eps=1e-5)

    def forward(self, relative_geometry: torch.Tensor) -> QwenContext:
        if relative_geometry.ndim != 2 or relative_geometry.shape[-1] != self.input_dim:
            raise ValueError(
                f"relative_geometry 应为 [B,{self.input_dim}]，"
                f"实际为 {tuple(relative_geometry.shape)}"
            )
        if not torch.isfinite(relative_geometry).all():
            raise ValueError("relative_geometry 包含 NaN 或 Inf")
        delta = relative_geometry[:, :3]
        distance = relative_geometry[:, 3]
        recomputed = torch.linalg.vector_norm(delta.float(), dim=-1)
        if torch.any(distance < 0.0) or not torch.allclose(
            distance.float(), recomputed, atol=1e-5, rtol=1e-4
        ):
            raise ValueError("relative_geometry distance 必须等于 ||delta_position||")
        token = self.output_norm(
            self.output_projection(self.activation(self.input_projection(relative_geometry)))
        ).unsqueeze(1)
        mask = torch.ones(
            (relative_geometry.shape[0], 1),
            dtype=torch.bool,
            device=relative_geometry.device,
        )
        return QwenContext(tokens=token, mask=mask)


class _FrozenOracleInputBoundary(nn.Module):
    """占据正式 Policy 的冻结 context 边界；实际解析由子类 encode_context 完成。"""

    def train(self, mode: bool = True) -> _FrozenOracleInputBoundary:
        del mode
        return super().train(False)

    def forward(self, model_inputs: dict[str, Any]) -> QwenContext:
        del model_inputs
        raise RuntimeError("OracleGeometryPolicy 必须使用自身 encode_context")


class OracleGeometryPolicy(QwenVLAPolicy):
    """只替换 Context Encoder，完整复用 QwenVLAPolicy 的 Flow/Expert 实现。"""

    def __init__(
        self,
        expert: StandaloneActionExpert,
        encoder: OracleGeometryContextEncoder | None = None,
    ) -> None:
        super().__init__(
            _FrozenOracleInputBoundary(),
            expert,
            encoder or OracleGeometryContextEncoder(),
        )

    def encode_context(self, model_inputs: dict[str, Any]) -> QwenContext:
        if set(model_inputs) != {"relative_geometry"}:
            raise ValueError("Oracle Policy context 输入只能包含 relative_geometry")
        return self.adapter(model_inputs["relative_geometry"])


def parameter_state_sha256(module: nn.Module) -> str:
    """生成与设备无关的参数/Buffer 身份，用于证明 A/B Expert 初始化一致。"""

    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def current_relative_geometry(base_env: Any) -> np.ndarray:
    """在线 Oracle 只读取当前仿真时刻的 TCP 与 Cube 世界坐标。"""

    def numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    tcp = numpy(base_env.agent.tcp_pose.p)
    object_position = numpy(base_env.cube.pose.p)
    if tcp.shape != (1, 3) or object_position.shape != (1, 3):
        raise ValueError("Oracle Reach 在线几何只支持单环境 [1,3]")
    delta = (object_position[0] - tcp[0]).astype(np.float32)
    distance = np.float32(np.linalg.norm(delta.astype(np.float64)))
    geometry = np.concatenate((delta, np.asarray([distance], dtype=np.float32)))
    if not np.isfinite(geometry).all() or not 0.0 <= float(distance) <= 2.0:
        raise ValueError("Oracle Reach 在线几何不是有效米制工作空间坐标")
    return geometry


class OracleGeometryRuntime:
    """保持正式 ActionAdapter/Flow sampling，只用当前 GT geometry 构造 context。"""

    def __init__(
        self,
        policy: OracleGeometryPolicy,
        proprio_normalizer: ProprioNormalizer,
        spec: RobotSpec,
        device: str | torch.device,
        geometry_provider: Callable[[], np.ndarray],
        config: RuntimeConfig,
    ) -> None:
        self.policy = policy
        self.proprio_normalizer = proprio_normalizer
        self.spec = spec
        self.device = torch.device(device)
        self.geometry_provider = geometry_provider
        self.config = config
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("请求 CUDA Oracle Runtime，但当前 PyTorch 无可用 CUDA")
        if config.use_bf16 and self.device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("当前 CUDA 设备不支持 Oracle Runtime 所需的 BF16")
        self.action_adapter = ActionAdapter(spec)
        self.policy.to(self.device)
        self.policy.eval()
        self._sample_index = config.starting_sample_index
        self._last_sampling_trace: SamplingTrace | None = None

    @property
    def last_sampling_trace(self) -> SamplingTrace | None:
        return self._last_sampling_trace

    def _next_sampling_trace(self) -> SamplingTrace:
        sample_index = self._sample_index
        self._sample_index += 1
        seed = (self.config.sampling_seed + sample_index) % (2**63 - 1)
        trace = SamplingTrace(seed=seed, sample_index=sample_index)
        self._last_sampling_trace = trace
        return trace

    @torch.inference_mode()
    def infer_action_chunk(self, observation: OnlineObservation) -> RuntimeActionChunk:
        sampling = self._next_sampling_trace()
        physical_proprio = np.asarray(observation.physical_proprio)
        if physical_proprio.shape != (self.spec.proprio_dim,):
            raise ValueError(f"physical_proprio 应为 [{self.spec.proprio_dim}]")
        if physical_proprio.dtype != np.float32 or not np.isfinite(physical_proprio).all():
            raise ValueError("physical_proprio 必须是有限 float32")
        geometry = np.asarray(self.geometry_provider(), dtype=np.float32)
        if geometry.shape != (4,) or not np.isfinite(geometry).all():
            raise ValueError("Oracle geometry provider 必须返回有限 float32 [4]")
        normalized_proprio = self.proprio_normalizer.normalize(physical_proprio)
        proprio_tensor = torch.from_numpy(normalized_proprio).unsqueeze(0).to(self.device)
        geometry_tensor = torch.from_numpy(geometry).unsqueeze(0).to(self.device)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(sampling.seed)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.config.use_bf16,
        ):
            normalized_tensor = self.policy.sample_actions(
                {"relative_geometry": geometry_tensor},
                proprio_tensor,
                generator=generator,
                num_steps=self.config.num_flow_steps,
            )
        normalized_action = normalized_tensor[0].float().cpu().numpy()
        expected = (self.spec.action_horizon, self.spec.action_dim)
        if normalized_action.shape != expected or not np.isfinite(normalized_action).all():
            raise RuntimeError("Oracle Policy 返回无效 normalized Action Chunk")
        physical_action = self.action_adapter.denormalize(normalized_action)
        return RuntimeActionChunk(
            normalized_action=normalized_action.copy(),
            physical_action=physical_action.copy(),
            visual_tokens_per_image=(0, 0),
            context_length=1,
            sampling=sampling,
        )


def oracle_case(oracle_successes: int, episodes: int = 5) -> str:
    if episodes != 5 or not 0 <= oracle_successes <= episodes:
        raise ValueError("首版 Case 判断要求恰好 5 个有效 Oracle Episode")
    if oracle_successes >= 4:
        return "case_1"
    if oracle_successes <= 1:
        return "case_2"
    return "case_3"


def validate_reach_training_budget(
    *,
    epochs: int,
    samples_per_epoch: int,
    batch_size: int,
    learning_rate: float,
    event_loss_weight: float,
) -> None:
    if epochs <= 0 or samples_per_epoch <= 0 or batch_size <= 0:
        raise ValueError("Reach A/B epochs/samples/batch 必须为正数")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Reach A/B learning_rate 必须是有限正数")
    if event_loss_weight != 0.0:
        raise ValueError("Oracle Reach 第一版必须关闭 event loss，仅使用 L_base")


__all__ = [
    "FRANKA_WORLD_BASE_POSITION_M",
    "ORACLE_GEOMETRY_FORMAT",
    "ORACLE_REACH_CHECKPOINT_FORMAT",
    "ORACLE_REACH_EXPERIMENT_FORMAT",
    "FrankaTCPForwardKinematics",
    "OracleGeometryContextEncoder",
    "OracleGeometryPolicy",
    "OracleGeometryRuntime",
    "OracleReachCollator",
    "OracleReachDataset",
    "OracleReachMode",
    "current_relative_geometry",
    "find_maniskill_panda_urdf",
    "oracle_case",
    "parameter_state_sha256",
    "select_geometry_entries",
    "validate_reach_training_budget",
]
