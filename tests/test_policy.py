import subprocess
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from robot_vla.model.expert import (
    ExpertConfig,
    StandaloneActionExpert,
    TemporalExpertConfig,
    TemporalStandaloneActionExpert,
)
from robot_vla.model.policy import QwenVLAObservationV2Policy, QwenVLAPolicy
from robot_vla.model.qwen_context import FrozenQwenContextEncoder, QwenVLAAdapter
from robot_vla.model.qwen_processor import VLA_CONTEXT_VALID_MASK, VLA_IMAGE_TIME_INDICES


def test_policy_can_be_imported_in_a_fresh_python_process() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "from robot_vla.model.policy import QwenVLAPolicy"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


class CountingBaseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.calls = 0

    def forward(self, input_ids, **_kwargs):
        self.calls += 1
        tokens = input_ids.float().unsqueeze(-1).expand(-1, -1, 2048) * self.scale
        return SimpleNamespace(hidden_states=(tokens,))


class FakeQwen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=2048))
        self.model = CountingBaseModel()
        self.lm_head = nn.Linear(2048, 32, bias=False)


def _build_policy() -> tuple[QwenVLAPolicy, FakeQwen]:
    qwen = FakeQwen()
    config = ExpertConfig(
        proprio_dim=5,
        action_dim=3,
        action_horizon=4,
        context_dim=720,
        hidden_size=32,
        state_hidden_size=16,
        num_layers=4,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    policy = QwenVLAPolicy(
        FrozenQwenContextEncoder(qwen),
        StandaloneActionExpert(config),
    )
    return policy, qwen


def _inputs():
    model_inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.long),
        "attention_mask": torch.tensor(
            [[True, True, True, False], [True, True, False, False]]
        ),
        "mm_token_type_ids": torch.zeros(2, 4, dtype=torch.long),
        "pixel_values": torch.zeros(4, 1536),
        "image_grid_thw": torch.ones(4, 3, dtype=torch.long),
    }
    proprio = torch.randn(2, 5)
    action = torch.rand(2, 4, 3) * 2.0 - 1.0
    action_mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
    return model_inputs, proprio, action, action_mask


def test_flow_loss_runs_qwen_once_and_only_trains_adapter_expert() -> None:
    policy, qwen = _build_policy()
    policy.train()
    model_inputs, proprio, action, action_mask = _inputs()

    output = policy.flow_matching_loss(
        model_inputs,
        proprio,
        action,
        action_mask,
        generator=torch.Generator().manual_seed(11),
    )
    output.loss.backward()

    assert qwen.model.calls == 1
    assert qwen.training is False
    assert qwen.model.scale.grad is None
    assert policy.adapter.skip_projection.weight.grad is not None
    assert policy.expert.velocity_head.weight.grad is not None
    assert output.prediction.shape == action.shape
    assert output.loss.dtype == torch.float32
    assert output.base_loss == output.loss
    assert output.event_loss == 0


def test_event_loss_only_uses_valid_events_in_executed_prefix() -> None:
    policy, _qwen = _build_policy()
    model_inputs, proprio, action, action_mask = _inputs()
    event_mask = torch.tensor(
        [[True, False, True, False], [False, True, False, True]],
        dtype=torch.bool,
    )

    output = policy.flow_matching_loss(
        model_inputs,
        proprio,
        action,
        action_mask,
        event_mask=event_mask,
        event_loss_weight=2.0,
        executed_action_steps=2,
        generator=torch.Generator().manual_seed(17),
    )

    expected_critical = torch.tensor(
        [[True, False, False, False], [False, True, False, False]],
        dtype=torch.bool,
    )
    assert torch.equal(output.critical_mask, expected_critical)
    torch.testing.assert_close(output.loss, output.base_loss + 2.0 * output.event_loss)


def test_sampling_runs_qwen_once_for_all_flow_steps_and_is_reproducible() -> None:
    policy, qwen = _build_policy()
    policy.eval()
    model_inputs, proprio, _action, action_mask = _inputs()

    first = policy.sample_actions(
        model_inputs,
        proprio,
        action_mask=action_mask,
        generator=torch.Generator().manual_seed(23),
        num_steps=4,
    )
    first_call_count = qwen.model.calls
    second = policy.sample_actions(
        model_inputs,
        proprio,
        action_mask=action_mask,
        generator=torch.Generator().manual_seed(23),
        num_steps=4,
    )

    assert first_call_count == 1
    assert qwen.model.calls == 2
    assert first.shape == (2, 4, 3)
    assert first.dtype == torch.float32
    assert torch.count_nonzero(first[~action_mask]).item() == 0
    torch.testing.assert_close(first, second)


def _build_temporal_policy() -> tuple[QwenVLAObservationV2Policy, FakeQwen]:
    qwen = FakeQwen()
    config = TemporalExpertConfig(
        action_dim=3,
        action_horizon=4,
        context_dim=720,
        hidden_size=32,
        state_hidden_size=16,
        num_layers=4,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    policy = QwenVLAObservationV2Policy(
        FrozenQwenContextEncoder(qwen),
        TemporalStandaloneActionExpert(config),
        QwenVLAAdapter(history_length=4),
    )
    return policy, qwen


def test_observation_v2_policy_trains_and_samples_with_temporal_state() -> None:
    policy, qwen = _build_temporal_policy()
    model_inputs, _proprio, action, action_mask = _inputs()
    model_inputs[VLA_IMAGE_TIME_INDICES] = torch.tensor(
        ((-1, 0, 2, -1), (-1, 1, -1, -1)),
        dtype=torch.long,
    )
    model_inputs[VLA_CONTEXT_VALID_MASK] = model_inputs["attention_mask"].clone()
    state = torch.randn(2, 4, 42)
    state_mask = torch.tensor(
        ((False, False, True, True), (True, True, True, True)),
        dtype=torch.bool,
    )
    controller = torch.randn(2, 24)

    output = policy.flow_matching_loss(
        model_inputs,
        state,
        action,
        action_mask,
        state_history_mask=state_mask,
        controller_state=controller,
        generator=torch.Generator().manual_seed(31),
    )
    output.loss.backward()

    assert qwen.model.calls == 1
    assert output.prediction.shape == action.shape
    assert policy.adapter.visual_time_embedding.grad is not None
    assert policy.expert.state_encoder.frame_projection[0].weight.grad is not None

    sampled = policy.sample_actions(
        model_inputs,
        state,
        state_history_mask=state_mask,
        controller_state=controller,
        action_mask=action_mask,
        generator=torch.Generator().manual_seed(37),
        num_steps=2,
    )
    rtc = policy.sample_actions_rtc(
        model_inputs,
        state,
        torch.zeros_like(action),
        torch.ones_like(action_mask, dtype=torch.float32),
        state_history_mask=state_mask,
        controller_state=controller,
        generator=torch.Generator().manual_seed(41),
        num_steps=2,
        max_guidance_weight=1.0,
    )
    assert sampled.shape == action.shape
    assert rtc.guided_action.shape == action.shape
    assert rtc.raw_action.shape == action.shape
    assert len(rtc.guidance_coefficients) == 2
