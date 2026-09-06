"""联合候选与共同对照：共享上游trunk，双方重置动作投影，避免旧joint head优势。"""
from dataclasses import replace
import torch
from experiments.memory_conditioning.conditioning import MemoryConditionedPolicy
from robot_vla.model.expert import StandaloneActionExpert

ARMS = ('joint-world', 'tcp-relative')
RESET_PREFIXES = ('action_encoder.action_projection.', 'velocity_head.')


def build_policy(base, arm, seed=42):
    if arm not in ARMS:
        raise ValueError('未知实验arm')
    # 单独初始化Memory encoder，使维数不同的Expert构造不改变其初值。
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        expert = StandaloneActionExpert(replace(base.expert.config, action_dim=8 if arm=='joint-world' else 7))
        state = expert.state_dict()
        for name,value in base.expert.state_dict().items():
            if not name.startswith(RESET_PREFIXES):
                state[name] = value.detach().cpu().clone()
        expert.load_state_dict(state,strict=True)
        torch.manual_seed(seed+1)
        policy = MemoryConditionedPolicy(base.context_encoder,expert,base.adapter)
    policy.context_encoder.requires_grad_(False);policy.adapter.requires_grad_(False)
    return policy
