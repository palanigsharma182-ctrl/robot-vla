"""一次消费、错误后清除、Episode隔离和真实小 Expert 的 Runtime 回归。"""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.memory_reobserve.runtime import MemoryConditionedRuntime, restore_candidate
from experiments.memory_conditioning.conditioning import MemorySnapshot, MEMORY_SCHEMA, MemoryConditionedPolicy
from probe import build_probe
from robot_vla.adapters import ProprioNormalizer, ProprioStats
from robot_vla.contracts import RobotSpec
from robot_vla.runtime.policy_runtime import OnlineObservation, QwenVLARuntime, RuntimeConfig
from experiments.g2c_memory_integration.vla import CHECKPOINT_SHA256


def bind(runtime, observation, snapshot):
    runtime._bind_snapshot(observation, snapshot, frame_timestamp_s=snapshot.timestamp_s)


def setup_runtime():
    baseline, candidate, inputs, _ = build_probe()
    # 旧 probe 是脚本入口，短模块名和包名不能混作同一 Python 类身份。
    candidate = MemoryConditionedPolicy(candidate.context_encoder, candidate.expert, candidate.adapter)
    inputs = {k: v[:1] for k,v in inputs.items()}
    class Processor:
        def encode(self, *args):
            return SimpleNamespace(model_inputs=inputs, visual_tokens_per_image=((2, 3),), context_lengths=(5,))
    spec = RobotSpec()
    normalizer = ProprioNormalizer(ProprioStats(mean=(0.,)*15, std=(1.,)*15,
        count=100, embodiment=spec.embodiment), spec)
    config = RuntimeConfig(use_bf16=False, num_flow_steps=2, sampling_seed=42)
    runtime = MemoryConditionedRuntime(candidate, Processor(), normalizer, spec, 'cpu', config)
    original = QwenVLARuntime(baseline, Processor(), normalizer, spec, 'cpu', config)
    proprio = np.zeros(15, np.float32); proprio[3] = -1.; proprio[5] = 1.; proprio[-1] = 1.
    obs = OnlineObservation(np.zeros((8,8,3), np.uint8), np.zeros((8,8,3), np.uint8), proprio, 'pick')
    runtime.reset_memory_episode('episode')
    snapshot = MemorySnapshot('episode', 2.2, 0., (.1,)*12, True, (), 'base_camera', 'provider')
    return runtime, original, obs, snapshot


def test_masked_runtime_parity_and_single_consumption():
    runtime, original, obs, snapshot = setup_runtime()
    absent = replace(snapshot, features=(0.,)*12, available=False, reasons=('uninitialized',))
    bind(runtime, obs, absent)
    a = runtime.infer_action_chunk(obs)
    b = original.infer_action_chunk(obs)
    np.testing.assert_array_equal(a.normalized_action, b.normalized_action)
    assert a.context_length == b.context_length
    with pytest.raises(RuntimeError, match='显式'):
        runtime.infer_action_chunk(obs)
    with pytest.raises(ValueError, match='新'):
        bind(runtime, obs, absent)


def test_live_token_reaches_actual_flow_and_increases_context():
    runtime, original, obs, snapshot = setup_runtime()
    bind(runtime, obs, snapshot)
    a = runtime.infer_action_chunk(obs)
    b = original.infer_action_chunk(obs)
    assert a.context_length == b.context_length + 1
    assert not np.array_equal(a.normalized_action, b.normalized_action)
    assert runtime.memory_reads[0]['status'] == 'consumed'
    assert a.normalized_action.shape == (16,8)


def test_mismatched_observation_clears_pending_even_on_error():
    runtime, _, obs, snapshot = setup_runtime()
    bind(runtime, obs, snapshot)
    with pytest.raises(ValueError, match='实际观测'):
        runtime.infer_action_chunk(replace(obs, instruction='other'))
    with pytest.raises(RuntimeError, match='显式'):
        runtime.infer_action_chunk(obs)
    bind(runtime, obs, replace(snapshot, timestamp_s=2.25))
    runtime.infer_action_chunk(obs)


def test_episode_isolation_and_pending_overwrite_rejected():
    runtime, _, obs, snapshot = setup_runtime()
    with pytest.raises(ValueError, match='Episode'):
        bind(runtime, obs, replace(snapshot, episode_id='wrong'))
    bind(runtime, obs, snapshot)
    with pytest.raises(RuntimeError, match='尚未消费'):
        bind(runtime, obs, replace(snapshot, timestamp_s=3.))
    runtime.reset_memory_episode('next')
    with pytest.raises(RuntimeError, match='显式'):
        runtime.infer_action_chunk(obs)


def test_checkpoint_identity_and_missing_encoder_rejected():
    runtime, _, _, _ = setup_runtime()
    policy = runtime.policy
    payload = dict(format='memory-conditioning-m0/v1', upstream_sha256=CHECKPOINT_SHA256,
        memory_schema=MEMORY_SCHEMA, arm='memory', steps=32, expert=policy.expert.state_dict(),
        memory_encoder=policy.memory_encoder.state_dict())
    restore_candidate(policy, payload)
    for key, wrong in [('format','stage1'), ('upstream_sha256','wrong'), ('memory_schema','wrong'), ('arm','no-memory'), ('steps', 99)]:
        with pytest.raises(ValueError):
            restore_candidate(policy, {**payload, key:wrong})
    bad = dict(payload['memory_encoder']); bad.pop(next(iter(bad)))
    with pytest.raises(RuntimeError):
        restore_candidate(policy, {**payload, 'memory_encoder':bad})


def test_frame_time_mismatch_is_rejected():
    runtime, _, obs, snapshot = setup_runtime()
    with pytest.raises(ValueError, match='实际观测帧时间'):
        runtime._bind_snapshot(obs, snapshot, frame_timestamp_s=snapshot.timestamp_s + .05)
