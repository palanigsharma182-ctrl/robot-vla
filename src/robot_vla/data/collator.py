"""使用固定 Qwen Processor 把 trajectory/v2 窗口组装成训练 batch。"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from robot_vla.contracts import OBSERVATION_HISTORY_LENGTH, RobotSpec
from robot_vla.model.qwen_processor import QwenVLAProcessorAdapter
from robot_vla.observation import (
    OBSERVATION_V2_CONTROLLER_STATE_DIM,
    OBSERVATION_V2_FRAME_STATE_DIM,
)


def _collate_training_fields(
    samples: list[dict[str, Any]],
    spec: RobotSpec,
) -> dict[str, Any]:
    proprio = np.stack([sample["proprio"] for sample in samples])
    action = np.stack([sample["action"] for sample in samples])
    action_mask = np.stack([sample["action_mask"] for sample in samples])
    supervision_mask = np.stack([sample["supervision_mask"] for sample in samples])
    event_mask = np.stack([sample["event_mask"] for sample in samples])
    expected_proprio = (len(samples), spec.proprio_dim)
    expected_action = (
        len(samples),
        spec.action_horizon,
        spec.action_dim,
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
    }


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
        output = _collate_training_fields(samples, self.spec)
        output.update({
            "qwen_inputs": processed.model_inputs,
            "visual_tokens_per_image": processed.visual_tokens_per_image,
            "context_lengths": torch.tensor(processed.context_lengths, dtype=torch.long),
        })
        return output


class QwenVLAObservationV2Collator:
    """八图 history + 四个 temporal state token 的显式 V2 collator。"""

    def __init__(self, processor_adapter: QwenVLAProcessorAdapter, spec: RobotSpec) -> None:
        self.processor_adapter = processor_adapter
        self.spec = spec

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        if not samples:
            raise ValueError("不能 collate 空 batch")
        processed = self.processor_adapter.encode_history_batch(
            [sample["rgb_external_history"] for sample in samples],
            [sample["rgb_wrist_history"] for sample in samples],
            [sample["state_history_mask"] for sample in samples],
            [sample["instruction"] for sample in samples],
        )
        state_history = np.stack([sample["state_history"] for sample in samples])
        state_history_mask = np.stack(
            [sample["state_history_mask"] for sample in samples]
        )
        controller_state = np.stack([sample["controller_state"] for sample in samples])
        expected_state = (
            len(samples),
            OBSERVATION_HISTORY_LENGTH,
            OBSERVATION_V2_FRAME_STATE_DIM,
        )
        if state_history.shape != expected_state or state_history.dtype != np.float32:
            raise ValueError(f"state_history 必须是 float32 {expected_state}")
        if (
            state_history_mask.shape != expected_state[:2]
            or state_history_mask.dtype != np.bool_
        ):
            raise ValueError("state_history_mask 必须是 bool [B,4]")
        if not np.all(state_history_mask[:, -1]):
            raise ValueError("Observation V2 当前控制步必须有效")
        expected_controller = (len(samples), OBSERVATION_V2_CONTROLLER_STATE_DIM)
        if controller_state.shape != expected_controller or controller_state.dtype != np.float32:
            raise ValueError(f"controller_state 必须是 float32 {expected_controller}")
        output = _collate_training_fields(samples, self.spec)
        output.update(
            {
                "qwen_inputs": processed.model_inputs,
                "state_history": torch.from_numpy(state_history),
                "state_history_mask": torch.from_numpy(state_history_mask),
                "controller_state": torch.from_numpy(controller_state),
                "visual_tokens_per_image": processed.visual_tokens_per_image,
                "context_lengths": torch.tensor(
                    processed.context_lengths,
                    dtype=torch.long,
                ),
            }
        )
        return output
