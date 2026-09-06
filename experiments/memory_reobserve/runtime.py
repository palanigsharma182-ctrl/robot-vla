"""实验 V1 Runtime：严格加载 Memory 权重，每次推理只消费一份新快照。"""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.memory_conditioning.conditioning import (
    MEMORY_INPUT_KEY, MEMORY_SCHEMA, MemoryBatch, MemoryConditionedPolicy,
    MemorySnapshot, snapshot_memory,
)
from experiments.g2c_memory_integration.vla import CHECKPOINT_SHA256, load_runtime, sha256
from robot_vla.runtime.policy_runtime import OnlineObservation, QwenVLARuntime

M0_CANDIDATE_SHA256 = "171057668337bfbc127a5b4a6d31ed805b7f5a5cde70abc6318d3c8977cb5c5f"


def observation_digest(observation):
    """快照绑定实际双图、状态和指令，防止在别的规划输入上复用。"""
    h = hashlib.sha256(observation.instruction.encode())
    for value in (observation.rgb_external, observation.rgb_wrist, observation.physical_proprio):
        a = np.ascontiguousarray(value)
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def restore_candidate(policy, payload):
    """旧 M0 权重仅可作工程消费验证；不会被提升为新训练或正式 checkpoint。"""
    if payload.get('format') != 'memory-conditioning-m0/v1':
        raise ValueError('未支持的 Memory checkpoint 格式')
    if payload.get('upstream_sha256') != CHECKPOINT_SHA256 or payload.get('memory_schema') != MEMORY_SCHEMA:
        raise ValueError('Memory checkpoint 上游权重或特征身份不匹配')
    if payload.get('steps') != 32:
        raise ValueError('本工程仅消费固定32步旧M0候选')
    if payload.get('arm') != 'memory' or not isinstance(payload.get('memory_encoder'), dict):
        raise ValueError('需要包含 Encoder 的 Memory 候选权重')
    policy.expert.load_state_dict(payload['expert'], strict=True)
    policy.memory_encoder.load_state_dict(payload['memory_encoder'], strict=True)


class MemoryConditionedRuntime(QwenVLARuntime):
    """显式绑定、一次消费，失败后也清除，绝不沿用上次调用的 Memory。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.policy, MemoryConditionedPolicy):
            raise TypeError('Memory Runtime 必须使用 MemoryConditionedPolicy')
        self._pending = None
        self._episode_id = None
        self._last_timestamp = None
        self.memory_reads = []

    def reset_memory_episode(self, episode_id):
        if not episode_id or episode_id == self._episode_id:
            raise ValueError('新 Episode 必须有新的非空身份')
        self._episode_id = episode_id
        self._last_timestamp = None
        self._pending = None
        self.memory_reads = []

    def _bind_snapshot(self, observation, snapshot: MemorySnapshot, *, frame_timestamp_s):
        if snapshot.timestamp_s != frame_timestamp_s:
            raise ValueError("快照时间必须等于实际观测帧时间")
        if self._pending is not None:
            raise RuntimeError('上一份快照尚未消费')
        if snapshot.episode_id != self._episode_id or snapshot.schema != MEMORY_SCHEMA:
            raise ValueError('快照 Episode/schema 与 Runtime 不匹配')
        if self._last_timestamp is not None and snapshot.timestamp_s <= self._last_timestamp:
            raise ValueError('新规划必须绑定新的观测时间，不能复用旧快照')
        self._pending = (observation_digest(observation), snapshot)

    def bind_frame(self, frame, memory, safety, instruction):
        observation = OnlineObservation(frame.rgb_external, frame.rgb_wrist,
                                        frame.physical_proprio, instruction)
        snapshot = snapshot_memory(memory.state, memory.config, safety,
            episode_id=self._episode_id, timestamp_s=frame.timestamp_s)
        self._bind_snapshot(observation, snapshot, frame_timestamp_s=frame.timestamp_s)
        return snapshot

    def _prepare_model_inputs(self, observation):
        if self._pending is None:
            raise RuntimeError('缺少本次规划的显式 Memory 快照')
        digest, snapshot = self._pending
        if digest != observation_digest(observation):
            raise ValueError('Memory 快照与本次实际观测不匹配')
        processed, inputs = super()._prepare_model_inputs(observation)
        inputs[MEMORY_INPUT_KEY] = MemoryBatch.from_snapshots([snapshot], device=self.device)
        return processed, inputs

    def infer_action_chunk(self, observation, **kwargs):
        if self._pending is None:
            raise RuntimeError('缺少本次规划的显式 Memory 快照')
        _, snapshot = self._pending
        started = time.monotonic()
        record = dict(snapshot=asdict(snapshot), status='error')
        try:
            result = super().infer_action_chunk(observation, **kwargs)
            record.update(status='consumed', sampling=asdict(result.sampling))
            return replace(result, context_length=result.context_length + int(snapshot.available))
        finally:
            self._pending = None
            self._last_timestamp = snapshot.timestamp_s
            record['inference_wall_s'] = time.monotonic() - started
            self.memory_reads.append(record)


def load_memory_runtime(upstream: Path, model_cache: Path, candidate: Path, expected_sha256: str):
    if expected_sha256 != M0_CANDIDATE_SHA256 or sha256(candidate) != M0_CANDIDATE_SHA256:
        raise ValueError('Memory candidate 文件 SHA-256 不匹配')
    payload = torch.load(candidate, map_location='cpu', weights_only=True)
    # 在加载昂贵上游之前先拒绝不相容身份。
    if payload.get('upstream_sha256') != CHECKPOINT_SHA256 or payload.get('memory_schema') != MEMORY_SCHEMA:
        raise ValueError('候选上游或 schema 不匹配')
    base, identity = load_runtime(upstream, model_cache)
    policy = MemoryConditionedPolicy(base.policy.context_encoder, base.policy.expert, base.policy.adapter)
    restore_candidate(policy, payload)
    runtime = MemoryConditionedRuntime(policy, base.processor_adapter, base.proprio_normalizer,
        base.spec, base.device, base.config)
    identity.update(candidate_sha256=expected_sha256, candidate_format=payload['format'],
        memory_schema=MEMORY_SCHEMA, checkpoint_use='engineering-consumption-only',
        candidate_training_steps=payload['steps'], verified_training_protocol=payload['protocol'],
        verified_data_sha256=payload['data_sha256'],
        payload_identity_binding='whole-file SHA256 fixed in source; includes all protocol and data metadata')
    return runtime, identity


def load_trained_runtime(upstream: Path, model_cache: Path, candidate: Path, expected_sha256: str,
                         *, expected_training_identity_sha256: str, expected_arm: str):
    """新训练候选须完整通过身份核验；不放宽旧32步M0 loader。"""
    from experiments.memory_reobserve.protocol import PROTOCOL
    from robot_vla.precision.active_front_memory_provider import build_stage2_object_memory_config
    if len(expected_sha256)!=64 or sha256(candidate)!=expected_sha256:
        raise ValueError('新 checkpoint 文件 SHA256 不匹配')
    payload=torch.load(candidate,map_location='cpu',weights_only=True)
    training_identity=payload.get('training_identity')
    if not isinstance(training_identity,dict):
        raise ValueError('候选缺少可重算的训练身份')
    actual_identity=hashlib.sha256(json.dumps(training_identity,sort_keys=True,
        separators=(',',':'),allow_nan=False).encode()).hexdigest()
    if (actual_identity!=expected_training_identity_sha256
        or actual_identity!=payload.get('training_identity_sha256')
        or training_identity.get('protocol')!=PROTOCOL):
        raise ValueError('候选训练身份不匹配')
    if (payload.get('format')!='memory-reobserve-five-skills/v1'
        or payload.get('upstream_sha256')!=CHECKPOINT_SHA256
        or payload.get('memory_schema')!=MEMORY_SCHEMA or payload.get('protocol')!=PROTOCOL
        or payload.get('steps')!=PROTOCOL['train_steps_per_arm'] or payload.get('completed') is not True
        or payload.get('memory_config')!=asdict(build_stage2_object_memory_config())
        or expected_arm not in ('visual','memory') or payload.get('arm')!=expected_arm):
        raise ValueError('新候选训练/特征/上游协议不一致或尚未完成')
    for key in ('protocol_sha256','schedule_sha256'):
        if not isinstance(payload.get(key),str) or len(payload[key])!=64:
            raise ValueError('候选缺少数据/采样身份')
    for key,value in [('protocol_sha256',training_identity),('schedule_sha256',training_identity['schedule'])]:
        if hashlib.sha256((json.dumps(value,indent=2)+'\n').encode()).hexdigest()!=payload[key]:
            raise ValueError('候选训练/采样记录不能由内嵌身份重算')
    base,identity=load_runtime(upstream,model_cache)
    normalization=dict(mean=base.proprio_normalizer.mean.tolist(),
        std=base.proprio_normalizer.std.tolist(),clip=base.proprio_normalizer.clip)
    if payload.get('proprio_normalization')!=normalization:
        raise ValueError('候选 proprio 归一化与上游不一致')
    policy=MemoryConditionedPolicy(base.policy.context_encoder,base.policy.expert,base.policy.adapter)
    policy.expert.load_state_dict(payload['expert'],strict=True)
    policy.memory_encoder.load_state_dict(payload['memory_encoder'],strict=True)
    runtime=MemoryConditionedRuntime(policy,base.processor_adapter,base.proprio_normalizer,
        base.spec,base.device,base.config)
    identity.update(candidate_sha256=expected_sha256,arm=payload['arm'],training_steps=payload['steps'],
        checkpoint_use='new-development-training-only',protocol_sha256=payload['protocol_sha256'],
        schedule_sha256=payload['schedule_sha256'],memory_schema=MEMORY_SCHEMA)
    identity['training_identity_sha256']=actual_identity
    return runtime,identity
