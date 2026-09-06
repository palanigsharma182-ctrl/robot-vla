"""合成上下文、真实小型 Action Expert 的无动作接口探针；不作机器人收益结论。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch
from torch import nn

from conditioning import MEMORY_INPUT_KEY, MemoryBatch, MemoryConditionedPolicy
from robot_vla.model.expert import ExpertConfig, StandaloneActionExpert
from robot_vla.model.policy import QwenVLAPolicy
from robot_vla.model.qwen_context import QwenContext


class SyntheticFrozenContext(nn.Module):
    """仅替代昂贵的 Qwen 上游，用于明确标记的接口验证。"""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()), requires_grad=False)

    def forward(self, inputs):
        return QwenContext(inputs['context'] * self.scale, inputs['mask'])


class IdentityAdapter(nn.Module):
    output_dim = 720

    def forward(self, context):
        return context


def build_probe():
    torch.manual_seed(31415)
    encoder = SyntheticFrozenContext()
    expert = StandaloneActionExpert(ExpertConfig(
        hidden_size=32, state_hidden_size=16, num_layers=4, intermediate_size=64,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
    ))
    adapter = IdentityAdapter()
    baseline = QwenVLAPolicy(encoder, expert, adapter)
    candidate = MemoryConditionedPolicy(encoder, expert, adapter)
    inputs = {'context': torch.randn(2, 5, 720), 'mask': torch.ones(2, 5, dtype=torch.bool)}
    proprio = torch.randn(2, 15)
    return baseline, candidate, inputs, proprio


def run_probe():
    baseline, candidate, inputs, proprio = build_probe()
    candidate.eval()
    def sample(policy, payload):
        return policy.sample_actions(payload, proprio, num_steps=3,
                                     generator=torch.Generator().manual_seed(23))
    original = sample(baseline, inputs)
    absent = sample(candidate, inputs)
    empty = MemoryBatch(torch.zeros(2, 12), torch.zeros(2, 1, dtype=torch.bool))
    masked = sample(candidate, {**inputs, MEMORY_INPUT_KEY: empty})
    present = MemoryBatch(torch.ones(2, 12) * 0.1, torch.ones(2, 1, dtype=torch.bool))
    conditioned = sample(candidate, {**inputs, MEMORY_INPUT_KEY: present})
    candidate.train()
    target = torch.zeros(2, 16, 8)  # 合成梯度探针，不是 Expert 数据集。
    loss = candidate.flow_matching_loss(
        {**inputs, MEMORY_INPUT_KEY: present}, proprio, target,
        torch.ones(2, 16, dtype=torch.bool), generator=torch.Generator().manual_seed(71),
    ).loss
    loss.backward()
    gradients = [p.grad for p in candidate.memory_encoder.parameters()]
    result = dict(
        status='synthetic-interface-smoke', robot_effect='not-evaluated', training_steps=0,
        real_qwen_loaded=False, actuator_steps=0, action_shape=list(conditioned.shape),
        absent_parity=bool(torch.equal(original, absent)),
        all_masked_parity=bool(torch.equal(original, masked)),
        conditioned_finite=bool(torch.isfinite(conditioned).all()),
        memory_action_difference_l2=float(torch.linalg.vector_norm(conditioned-original)),
        memory_gradients_finite_nonzero=all(g is not None and bool(torch.isfinite(g).all())
                                           and float(g.abs().sum()) > 0 for g in gradients),
        frozen_context_no_gradient=candidate.context_encoder.scale.grad is None,
        flow_loss=float(loss.detach()), expert_config=asdict(candidate.expert.config),
    )
    assert all(result[k] for k in ['absent_parity', 'all_masked_parity', 'conditioned_finite',
                                   'memory_gradients_finite_nonzero', 'frozen_context_no_gradient'])
    assert result['memory_action_difference_l2'] > 0
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run_probe()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))
