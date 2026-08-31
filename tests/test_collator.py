import numpy as np
import pytest

torch = pytest.importorskip("torch")

from robot_vla.contracts import RobotSpec
from robot_vla.data.collator import QwenVLACollator, QwenVLAObservationV2Collator
from robot_vla.model.qwen_processor import VLA_CONTEXT_VALID_MASK, VLA_IMAGE_TIME_INDICES


def _sample(
    spec: RobotSpec,
    *,
    external_size: int,
    wrist_size: int,
    timestep: int,
) -> dict:
    return {
        "rgb_external": np.zeros((external_size, external_size, 3), dtype=np.uint8),
        "rgb_wrist": np.zeros((wrist_size, wrist_size, 3), dtype=np.uint8),
        "proprio": np.zeros(spec.proprio_dim, dtype=np.float32),
        "action": np.zeros((spec.action_horizon, spec.action_dim), dtype=np.float32),
        "action_mask": np.ones(spec.action_horizon, dtype=np.bool_),
        "supervision_mask": np.ones(spec.action_horizon, dtype=np.bool_),
        "event_mask": np.zeros(spec.action_horizon, dtype=np.bool_),
        "instruction": "Pick up the red cube.",
        "trajectory_id": f"episode-{timestep}",
        "timestep": timestep,
        "skill_id": 1,
        "source": "base_d0",
        "boundary_offset": None,
    }


def test_collator_preserves_qwen_inputs_and_training_tensor_contract(
    qwen_processor_adapter,
) -> None:
    spec = RobotSpec()
    collator = QwenVLACollator(qwen_processor_adapter, spec)

    batch = collator(
        [
            _sample(spec, external_size=512, wrist_size=256, timestep=0),
            _sample(spec, external_size=256, wrist_size=256, timestep=1),
        ]
    )

    assert batch["qwen_inputs"]["input_ids"].shape[0] == 2
    assert batch["qwen_inputs"]["attention_mask"].dtype == torch.bool
    assert batch["proprio"].shape == (2, 15)
    assert batch["proprio"].dtype == torch.float32
    assert batch["action"].shape == (2, 16, 8)
    assert batch["action_mask"].shape == (2, 16)
    assert batch["action_mask"].dtype == torch.bool
    assert batch["supervision_mask"].shape == (2, 16)
    assert batch["supervision_mask"].dtype == torch.bool
    assert batch["event_mask"].shape == (2, 16)
    assert batch["event_mask"].dtype == torch.bool
    assert batch["visual_tokens_per_image"] == ((256, 64), (64, 64))
    assert batch["trajectory_id"] == ["episode-0", "episode-1"]


def test_observation_v2_collator_emits_eight_images_and_temporal_state_tokens(
    qwen_processor_adapter,
) -> None:
    spec = RobotSpec()
    sample = _sample(spec, external_size=256, wrist_size=256, timestep=0)
    sample.update(
        {
            "rgb_external_history": np.zeros((4, 256, 256, 3), dtype=np.uint8),
            "rgb_wrist_history": np.zeros((4, 256, 256, 3), dtype=np.uint8),
            "state_history": np.zeros((4, 42), dtype=np.float32),
            "state_history_mask": np.asarray((False, False, True, True), dtype=np.bool_),
            "controller_state": np.zeros(24, dtype=np.float32),
        }
    )

    batch = QwenVLAObservationV2Collator(qwen_processor_adapter, spec)([sample])

    assert batch["state_history"].shape == (1, 4, 42)
    assert batch["state_history_mask"].tolist() == [[False, False, True, True]]
    assert batch["controller_state"].shape == (1, 24)
    assert batch["visual_tokens_per_image"] == ((64,) * 8,)
    assert VLA_IMAGE_TIME_INDICES in batch["qwen_inputs"]
    assert VLA_CONTEXT_VALID_MASK in batch["qwen_inputs"]
    time_indices = batch["qwen_inputs"][VLA_IMAGE_TIME_INDICES]
    context_valid = batch["qwen_inputs"][VLA_CONTEXT_VALID_MASK]
    assert time_indices.shape == context_valid.shape
    padded_visual = (time_indices >= 0) & (time_indices < 2)
    assert torch.count_nonzero(padded_visual & context_valid).item() == 0
    assert torch.count_nonzero(
        padded_visual & batch["qwen_inputs"]["attention_mask"]
    ).item() == 0
