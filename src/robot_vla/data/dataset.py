"""把 trajectory/v2 转换成不依赖具体 Processor 的双图 Action Chunk 窗口。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from robot_vla.adapters import ActionAdapter, ProprioNormalizer
from robot_vla.contracts import RobotSpec
from robot_vla.data.events import EventDetectionConfig, detect_trajectory_events
from robot_vla.data.trajectory import TrajectoryMeta, TrajectoryStore


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
