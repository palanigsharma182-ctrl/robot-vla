"""只读采样包装与原Flow采样的精确一致性；只需小型CPU Expert。"""
import pytest

torch = pytest.importorskip('torch')

from experiments.memory_conditioning.conditioning import MEMORY_INPUT_KEY, MemoryBatch
from experiments.tcp_memory_control.policy import build_policy
from experiments.tcp_chunk_diagnostic.run import sample_with_unclamped
from robot_vla.model.expert import ExpertConfig, StandaloneActionExpert
from robot_vla.model.policy import QwenVLAPolicy
from robot_vla.model.qwen_context import QwenContext


class FakeContext(torch.nn.Module):
    def forward(self, inputs):
        return QwenContext(inputs['context'], inputs['mask'])


class IdentityAdapter(torch.nn.Module):
    output_dim = 720

    def forward(self, context):
        return context


@pytest.mark.parametrize('available', [False, True])
def test_sampling_wrapper_preserves_original_output(available):
    torch.manual_seed(31415)
    base = QwenVLAPolicy(FakeContext(), StandaloneActionExpert(ExpertConfig(
        hidden_size=32, state_hidden_size=16, num_layers=4, intermediate_size=64,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8)), IdentityAdapter())
    inputs = {'context': torch.randn(2, 5, 720), 'mask': torch.ones(2, 5, dtype=torch.bool)}
    proprio = torch.randn(2, 15)
    policy = build_policy(base, 'tcp-relative').eval()
    inputs = {**inputs, MEMORY_INPUT_KEY: MemoryBatch(
        torch.zeros(2, 12), torch.full((2, 1), available, dtype=torch.bool))}
    original = policy.sample_actions(inputs, proprio, num_steps=10,
                                    generator=torch.Generator().manual_seed(17))
    sampled, raw = sample_with_unclamped(policy, inputs, proprio, 17)
    assert torch.equal(original, sampled)
    assert torch.equal(raw.clamp(-1, 1), sampled)
