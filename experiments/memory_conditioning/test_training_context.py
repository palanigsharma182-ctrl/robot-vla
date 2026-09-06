"""验证离线训练缓存入口与候选 Policy 的真实插入点一致。"""
import torch
import pytest

from conditioning import MEMORY_INPUT_KEY, MemoryBatch
from probe import build_probe
from experiments.memory_conditioning.train import context_with_memory, validate_protocol


def test_cached_training_context_matches_policy_and_gradient():
    baseline, candidate, inputs, _ = build_probe()
    features = torch.randn(2, 12)
    # 实验训练为 batch=1；逐样本验证以避免缓存入口暗中广播。
    for index in range(2):
        selected = {key:value[index:index+1] for key,value in inputs.items()}
        memory = MemoryBatch(features[index:index+1], torch.ones(1,1,dtype=torch.bool))
        expected = candidate.encode_context({**selected, MEMORY_INPUT_KEY: memory})
        context = baseline.encode_context(selected)
        actual = context_with_memory(context, candidate.memory_encoder, memory.features)
        torch.testing.assert_close(actual.tokens, expected.tokens, rtol=0, atol=0)
        assert torch.equal(actual.mask, expected.mask)
        actual.tokens[:, -1].square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in candidate.memory_encoder.parameters())
    assert all(p.grad is None for p in candidate.context_encoder.parameters())


@pytest.mark.parametrize("replacement", [
    {"train_seeds":[76903]}, {"development_seeds":[1000100]},
    {"train_steps_per_arm":33}, {"learning_rate":1e-4}, {"sampling_seed":43},
])
def test_reject_protocol_substitution(replacement):
    protocol = dict(seeds=list(range(1000100,1000112)), train_seeds=list(range(1000100,1000108)),
        development_seeds=list(range(1000108,1000112)), train_steps_per_arm=32,
        batch_size=1, learning_rate=1e-5, sampling_seed=42)
    validate_protocol(protocol)
    with pytest.raises(ValueError):
        validate_protocol({**protocol, **replacement})
