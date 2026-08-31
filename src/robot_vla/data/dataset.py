"""把 trajectory/v2 转换成不依赖具体 Processor 的双图 Action Chunk 窗口。"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from typing import Any

import numpy as np

from robot_vla.adapters import ActionAdapter, FingerForceNormalizer, ProprioNormalizer
from robot_vla.contracts import OBSERVATION_HISTORY_LENGTH, RobotSpec
from robot_vla.data.events import EventDetectionConfig, detect_trajectory_events
from robot_vla.data.trajectory import TrajectoryMeta, TrajectoryStore
from robot_vla.observation import (
    OBSERVATION_V2_CONTROLLER_STATE_DIM,
    OBSERVATION_V2_FRAME_STATE_DIM,
)


class ActionChunkDataset:
    """返回原始双图、归一化状态和带 mask 的归一化 Action Chunk。"""

    def __init__(
        self,
        root: str,
        entries: Sequence[TrajectoryMeta],
        spec: RobotSpec,
        proprio_normalizer: ProprioNormalizer,
        *,
        cache_size: int = 2,
        event_detection_config: EventDetectionConfig | None = None,
    ) -> None:
        if not entries:
            raise ValueError("ActionChunkDataset 需要至少一条轨迹")
        splits = {entry.split for entry in entries}
        if len(splits) != 1:
            raise ValueError(f"一个 Dataset 只能包含一个 split，实际为 {sorted(splits)}")
        self.entries = list(entries)
        self.spec = spec
        self.proprio_normalizer = proprio_normalizer
        self.action_adapter = ActionAdapter(spec)
        self.event_detection_config = event_detection_config or EventDetectionConfig()
        self.store = TrajectoryStore(root, spec, cache_size=cache_size)
        self.index: list[tuple[int, int]] = []
        self.event_masks: list[np.ndarray] = []
        for entry_index, entry in enumerate(self.entries):
            arrays = self.store.get(entry)
            self.event_masks.append(
                detect_trajectory_events(
                    arrays,
                    self.event_detection_config,
                ).event_mask
            )
            valid_timesteps = np.flatnonzero(arrays.observation_valid).tolist()
            if entry.local_dagger is None:
                self.index.extend(
                    (entry_index, timestep) for timestep in valid_timesteps
                )
                continue
            provenance = entry.local_dagger
            eligible = [
                timestep
                for timestep in valid_timesteps
                if provenance.training_window_start <= timestep
                and timestep + self.spec.action_horizon <= provenance.training_window_end
                and bool(
                    arrays.expert_supervision_mask[
                        timestep : timestep + self.spec.action_horizon
                    ].all()
                )
            ]
            if not eligible:
                raise ValueError(
                    f"{entry.trajectory_id}: Local DAgger 没有完整 Expert/window Chunk"
                )
            self.index.extend((entry_index, timestep) for timestep in eligible)
        if not self.index:
            raise ValueError("Dataset 没有双相机和 proprio 同时有效的样本起点")

    def __len__(self) -> int:
        return len(self.index)

    def sampling_metadata(self, index: int) -> dict[str, Any]:
        """暴露 sampler 所需的最小身份，不泄漏 Dataset 内部存储结构。"""

        entry_index, timestep = self.index[index]
        meta = self.entries[entry_index]
        arrays = self.store.get(meta)
        provenance = meta.local_dagger
        return {
            "episode_key": meta.trajectory_id,
            "task_id": meta.task.task_id,
            "source": "base_d0" if provenance is None else provenance.source,
            "skill_id": int(arrays.skill_id[timestep]),
            "boundary_offset": (
                None
                if provenance is None
                else timestep - provenance.training_window_start
            ),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry_index, timestep = self.index[index]
        meta = self.entries[entry_index]
        arrays = self.store.get(meta)

        normalized_proprio = self.proprio_normalizer.normalize(arrays.proprio[timestep])
        action = np.zeros(
            (self.spec.action_horizon, self.spec.action_dim),
            dtype=np.float32,
        )
        action_mask = np.zeros(self.spec.action_horizon, dtype=np.bool_)
        supervision_mask = np.zeros(self.spec.action_horizon, dtype=np.bool_)
        event_mask = np.zeros(self.spec.action_horizon, dtype=np.bool_)
        valid_length = min(self.spec.action_horizon, arrays.num_steps - timestep)
        physical_actions = arrays.action[timestep : timestep + valid_length]
        action[:valid_length] = self.action_adapter.normalize(physical_actions, strict=True)
        action_mask[:valid_length] = True
        if meta.local_dagger is None:
            supervision_mask[:valid_length] = True
            source = "base_d0"
            boundary_offset = None
        else:
            if valid_length != self.spec.action_horizon:
                raise RuntimeError("Local DAgger Dataset index 包含不完整 Action Chunk")
            supervision_mask[:] = arrays.expert_supervision_mask[
                timestep : timestep + self.spec.action_horizon
            ]
            if not bool(supervision_mask.all()):
                raise RuntimeError("Local DAgger Dataset index 泄漏了 Policy Action")
            source = meta.local_dagger.source
            boundary_offset = timestep - meta.local_dagger.training_window_start
        action_mask &= supervision_mask
        event_mask[:valid_length] = self.event_masks[entry_index][
            timestep : timestep + valid_length
        ]
        event_mask &= action_mask

        return {
            "rgb_external": arrays.rgb_external[timestep].copy(),
            "rgb_wrist": arrays.rgb_wrist[timestep].copy(),
            "proprio": normalized_proprio,
            "action": action,
            "action_mask": action_mask,
            "supervision_mask": supervision_mask,
            "event_mask": event_mask,
            "instruction": meta.task.instruction,
            "trajectory_id": meta.trajectory_id,
            "timestep": timestep,
            "skill_id": int(arrays.skill_id[timestep]),
            "source": source,
            "boundary_offset": boundary_offset,
        }


class ObservationV2ActionChunkDataset(ActionChunkDataset):
    """只接受真实 Observation V2 数组，并构造 ``t-3..t`` 连续历史。"""

    def __init__(
        self,
        root: str,
        entries: Sequence[TrajectoryMeta],
        spec: RobotSpec,
        proprio_normalizer: ProprioNormalizer,
        *,
        finger_force_normalizer: FingerForceNormalizer,
        cache_size: int = 2,
        event_detection_config: EventDetectionConfig | None = None,
    ) -> None:
        super().__init__(
            root,
            entries,
            spec,
            proprio_normalizer,
            cache_size=cache_size,
            event_detection_config=event_detection_config,
        )
        self.finger_force_normalizer = finger_force_normalizer
        missing = [
            entry.trajectory_id
            for entry in self.entries
            if not self.store.get(entry).observation_v2_available
        ]
        if missing:
            raise ValueError(
                "Observation V2 Dataset 禁止伪造缺失状态；缺失轨迹="
                f"{missing[:5]}"
            )
        self.index = [
            (entry_index, timestep)
            for entry_index, timestep in self.index
            if bool(self.store.get(self.entries[entry_index]).observation_v2_valid[timestep])
        ]
        if not self.index:
            raise ValueError("Observation V2 Dataset 没有当前时刻完整有效的样本")

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = super().__getitem__(index)
        entry_index, timestep = self.index[index]
        arrays = self.store.get(self.entries[entry_index])
        history_length = OBSERVATION_HISTORY_LENGTH
        source_start = max(0, timestep - history_length + 1)
        source_indices = np.arange(source_start, timestep + 1, dtype=np.int64)
        destination_start = history_length - len(source_indices)
        if len(source_indices) > 1:
            expected_dt = 1.0 / self.spec.control_hz
            delta = np.diff(arrays.timestamp_action[source_indices])
            if np.any(np.abs(delta - expected_dt) > max(1e-6, expected_dt * 0.2)):
                raise ValueError("Observation V2 history 不是连续控制步")

        external = np.zeros(
            (history_length, *arrays.rgb_external.shape[1:]),
            dtype=np.uint8,
        )
        wrist = np.zeros(
            (history_length, *arrays.rgb_wrist.shape[1:]),
            dtype=np.uint8,
        )
        proprio = np.zeros((history_length, self.spec.proprio_dim), dtype=np.float32)
        tcp_position = np.zeros((history_length, 3), dtype=np.float32)
        tcp_rotation = np.zeros((history_length, 6), dtype=np.float32)
        wrist_position = np.zeros((history_length, 3), dtype=np.float32)
        wrist_rotation = np.zeros((history_length, 6), dtype=np.float32)
        finger_force = np.zeros((history_length, 2), dtype=np.float32)
        history_valid = np.zeros(history_length, dtype=np.bool_)
        modality_valid = np.zeros((history_length, 6), dtype=np.bool_)
        modality_age = np.zeros((history_length, 6), dtype=np.float32)
        frame_age = np.zeros(history_length, dtype=np.float32)
        current_time = float(arrays.timestamp_action[timestep])
        for destination, source in enumerate(source_indices, start=destination_start):
            history_valid[destination] = True
            valid = np.asarray(
                (
                    arrays.external_valid[source],
                    arrays.wrist_valid[source],
                    arrays.proprio_valid[source],
                    arrays.tcp_pose_valid[source],
                    arrays.camera_pose_valid[source],
                    arrays.finger_force_valid[source],
                ),
                dtype=np.bool_,
            )
            modality_valid[destination] = valid
            if valid[0]:
                external[destination] = arrays.rgb_external[source]
            if valid[1]:
                wrist[destination] = arrays.rgb_wrist[source]
            if valid[2]:
                proprio[destination] = arrays.proprio[source]
            if valid[3]:
                tcp_position[destination] = arrays.tcp_position_base_m[source]
                tcp_rotation[destination] = arrays.tcp_rotation_6d_base[source]
            if valid[4]:
                wrist_position[destination] = arrays.wrist_camera_position_base_m[source]
                wrist_rotation[destination] = arrays.wrist_camera_rotation_6d_base[source]
            if valid[5]:
                finger_force[destination] = (
                    arrays.left_finger_force_n[source],
                    arrays.right_finger_force_n[source],
                )
            frame_age[destination] = np.float32(
                current_time - arrays.timestamp_action[source]
            )
            timestamps = np.asarray(
                (
                    arrays.timestamp_external[source],
                    arrays.timestamp_wrist[source],
                    arrays.timestamp_proprio[source],
                    arrays.timestamp_tcp_pose[source],
                    arrays.timestamp_camera_pose[source],
                    arrays.timestamp_finger_force[source],
                ),
                dtype=np.float64,
            )
            modality_age[destination, valid] = (
                current_time - timestamps[valid]
            ).astype(np.float32)

        normalized_proprio = np.zeros_like(proprio)
        valid_proprio = modality_valid[:, 2]
        normalized_proprio[valid_proprio] = self.proprio_normalizer.normalize(
            proprio[valid_proprio]
        )
        normalized_finger_force = np.zeros_like(finger_force)
        valid_finger_force = modality_valid[:, 5]
        normalized_finger_force[valid_finger_force] = (
            self.finger_force_normalizer.normalize(
                finger_force[valid_finger_force]
            )
        )
        state_history = np.concatenate(
            (
                normalized_proprio,
                tcp_position,
                tcp_rotation,
                wrist_position,
                wrist_rotation,
                normalized_finger_force,
                frame_age[:, None],
                modality_valid.astype(np.float32),
            ),
            axis=-1,
        ).astype(np.float32, copy=False)
        if state_history.shape != (
            history_length,
            OBSERVATION_V2_FRAME_STATE_DIM,
        ):
            raise RuntimeError(f"Observation V2 state_history 维度漂移: {state_history.shape}")

        command_valid = bool(arrays.previous_command_valid[timestep])
        action_valid = bool(arrays.previous_action_valid[timestep])
        command = (
            arrays.previous_command_q_rad[timestep].copy()
            if command_valid
            else np.zeros(self.spec.arm_dof, dtype=np.float32)
        )
        tracking = (
            command - arrays.proprio[timestep, : self.spec.arm_dof]
            if command_valid
            else np.zeros(self.spec.arm_dof, dtype=np.float32)
        )
        previous_action = (
            arrays.previous_action[timestep].copy()
            if action_valid
            else np.zeros(self.spec.action_dim, dtype=np.float32)
        )
        controller_valid = np.asarray((command_valid, action_valid), dtype=np.bool_)
        controller_state = np.concatenate(
            (
                command,
                tracking,
                previous_action,
                controller_valid.astype(np.float32),
            )
        ).astype(np.float32, copy=False)
        if controller_state.shape != (OBSERVATION_V2_CONTROLLER_STATE_DIM,):
            raise RuntimeError(f"Observation V2 controller_state 维度漂移: {controller_state.shape}")

        sample.update(
            {
                "rgb_external_history": external,
                "rgb_wrist_history": wrist,
                "physical_proprio_history": proprio,
                "state_history": state_history,
                "state_history_mask": history_valid,
                "modality_valid": modality_valid,
                "frame_age_s": frame_age,
                "modality_age_s": modality_age,
                "controller_state": controller_state,
                "controller_valid": controller_valid,
            }
        )
        return sample


class CompositeActionChunkDataset:
    """把多个已审计数据根组合为一个逻辑 Dataset，不复制或别名化 NPZ。"""

    def __init__(self, components: Sequence[ActionChunkDataset]) -> None:
        if not components:
            raise ValueError("Composite Dataset 需要至少一个 component")
        self.components = tuple(components)
        self.offsets = [0]
        for component in self.components:
            self.offsets.append(self.offsets[-1] + len(component))
        if self.offsets[-1] <= 0:
            raise ValueError("Composite Dataset 不能是空 Dataset")

    def __len__(self) -> int:
        return self.offsets[-1]

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("Composite Dataset index 超出范围")
        component_index = bisect.bisect_right(self.offsets, index) - 1
        return component_index, index - self.offsets[component_index]

    def __getitem__(self, index: int) -> dict[str, Any]:
        component_index, local_index = self._locate(index)
        return self.components[component_index][local_index]

    def sampling_metadata(self, index: int) -> dict[str, Any]:
        component_index, local_index = self._locate(index)
        metadata = dict(
            self.components[component_index].sampling_metadata(local_index)
        )
        metadata["episode_key"] = (
            component_index,
            str(metadata["episode_key"]),
        )
        return metadata
