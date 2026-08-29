import numpy as np
import pytest

torch = pytest.importorskip("torch")

from robot_vla.contracts import RobotSpec
from robot_vla.data.collator import QwenVLACollator


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
