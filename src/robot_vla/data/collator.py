"""使用固定 Qwen Processor 把 trajectory/v2 窗口组装成训练 batch。"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from robot_vla.contracts import RobotSpec
from robot_vla.model.qwen_processor import QwenVLAProcessorAdapter


class QwenVLACollator:
    def __init__(self, processor_adapter: QwenVLAProcessorAdapter, spec: RobotSpec) -> None:
        self.processor_adapter = processor_adapter
        self.spec = spec

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        if not samples:
            raise ValueError("不能 collate 空 batch")
        processed = self.processor_adapter.encode_batch(
            [sample["rgb_external"] for sample in samples],
            [sample["rgb_wrist"] for sample in samples],
            [sample["instruction"] for sample in samples],
        )
        proprio = np.stack([sample["proprio"] for sample in samples])
        action = np.stack([sample["action"] for sample in samples])
        action_mask = np.stack([sample["action_mask"] for sample in samples])
        supervision_mask = np.stack([sample["supervision_mask"] for sample in samples])
        event_mask = np.stack([sample["event_mask"] for sample in samples])
        expected_proprio = (len(samples), self.spec.proprio_dim)
        expected_action = (
            len(samples),
            self.spec.action_horizon,
            self.spec.action_dim,
        )
        if proprio.shape != expected_proprio:
            raise ValueError(f"proprio batch 应为 {expected_proprio}，实际为 {proprio.shape}")
        if action.shape != expected_action:
            raise ValueError(f"action batch 应为 {expected_action}，实际为 {action.shape}")
        if (
            action_mask.shape != expected_action[:2]
            or supervision_mask.shape != expected_action[:2]
            or event_mask.shape != expected_action[:2]
        ):
            raise ValueError("action_mask/supervision_mask/event_mask batch shape 无效")
        if proprio.dtype != np.float32 or action.dtype != np.float32:
            raise ValueError("proprio/action batch dtype 必须为 float32")
        if (
            action_mask.dtype != np.bool_
            or supervision_mask.dtype != np.bool_
            or event_mask.dtype != np.bool_
        ):
            raise ValueError("Action 相关 mask batch dtype 必须为 bool")
        if np.any(action_mask & ~supervision_mask):
            raise ValueError("action_mask 不能监督 supervision_mask 外的 Action Token")
        if np.any(event_mask & ~action_mask):
            raise ValueError("event_mask 不能标记无效 Action Token")
        return {
            "qwen_inputs": processed.model_inputs,
            "proprio": torch.from_numpy(proprio),
            "action": torch.from_numpy(action),
            "action_mask": torch.from_numpy(action_mask),
            "supervision_mask": torch.from_numpy(supervision_mask),
            "event_mask": torch.from_numpy(event_mask),
            "trajectory_id": [str(sample["trajectory_id"]) for sample in samples],
            "timestep": torch.tensor(
                [int(sample["timestep"]) for sample in samples],
                dtype=torch.long,
            ),
            "skill_id": torch.tensor(
                [int(sample["skill_id"]) for sample in samples],
                dtype=torch.long,
            ),
            "source": [str(sample["source"]) for sample in samples],
            "boundary_offset": [sample["boundary_offset"] for sample in samples],
            "visual_tokens_per_image": processed.visual_tokens_per_image,
            "context_lengths": torch.tensor(processed.context_lengths, dtype=torch.long),
        }
