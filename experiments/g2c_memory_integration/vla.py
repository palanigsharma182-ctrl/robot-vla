"""恢复冻结 V1 VLA 基线，并显式运行开发场景的暂停/恢复工程验收。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from robot_vla.adapters import FrankaObservationAdapter, ProprioNormalizer, ProprioStats
from robot_vla.contracts import RobotSpec, QWEN_MODEL_ID, QWEN_REVISION
from robot_vla.execution.chunk_executor import RecedingHorizonChunkExecutor
from robot_vla.model.factory import load_qwen_vla_policy
from robot_vla.model.qwen_processor import QwenVLAProcessorAdapter
from robot_vla.runtime.control_loop import QwenVLAReplanLoop
from robot_vla.runtime.policy_runtime import QwenVLARuntime, RuntimeConfig
from robot_vla.training.checkpoint import load_stage1_policy_checkpoint

CHECKPOINT_SHA256 = 'a542076f291e29b68e3d28930b15c40396d511a44eb358c2eaeb4e113c041ad6'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_runtime(checkpoint: Path, model_cache: Path):
    """文件身份固定于已有 E012 初始化基线；统计量使用同一已验证 checkpoint。"""
    if sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError('VLA checkpoint 不属于本次恢复的冻结基线')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    metadata = payload['metadata']
    stats = ProprioStats(**metadata['proprio_stats'])
    del payload
    spec = RobotSpec()
    stats.validate(spec)
    processor = QwenVLAProcessorAdapter.from_pretrained(cache_dir=str(model_cache), local_files_only=True)
    policy = load_qwen_vla_policy(cache_dir=str(model_cache), local_files_only=True,
                                  device='cuda', context_layer=12)
    verified = load_stage1_policy_checkpoint(checkpoint, policy, spec, processor.config, stats)
    runtime = QwenVLARuntime(policy, processor, ProprioNormalizer(stats, spec), spec,
                            'cuda', RuntimeConfig(sampling_seed=42))
    identity = dict(checkpoint_sha256=CHECKPOINT_SHA256, qwen_model=QWEN_MODEL_ID,
                    qwen_revision=QWEN_REVISION, checkpoint_metadata=verified,
                    runtime=asdict(runtime.config), torch=torch.__version__)
    return runtime, identity


def baseline(runtime, output: Path):
    """固定新开发场景，只验证真实推理与动作执行，不消费数据集评估 split。"""
    import gymnasium as gym
    import robot_vla.sim.pick_cube_to_region  # noqa: F401
    from robot_vla.evaluation.maniskill import _read_online_observation
    from run import DevelopmentController

    output.mkdir(parents=True, exist_ok=False)
    spec = RobotSpec()
    env = gym.make('RobotVLAPickCubeToRegion-v1', robot_uids='panda_wristcam', num_envs=1,
                   obs_mode='rgb', control_mode='pd_joint_delta_pos', sim_backend='cpu',
                   render_backend='gpu', sensor_configs={'width':128, 'height':128})
    records = []
    started = time.monotonic()
    try:
        observation, _ = env.reset(seed=1000001)
        controller = DevelopmentController(env, spec)
        loop = QwenVLAReplanLoop(runtime, RecedingHorizonChunkExecutor(spec))
        adapter = FrankaObservationAdapter(spec)
        for _ in range(2):
            online = _read_online_observation(observation, env.unwrapped, adapter,
                'pick the cube and place it in the target region')
            result = loop.replan_and_execute(online, controller)
            records.append(dict(execution=asdict(result.execution), sampling=None if result.sampling is None else asdict(result.sampling),
                action_shape=None if result.action_chunk is None else list(result.action_chunk.normalized_action.shape),
                finite=result.action_chunk is not None and bool(np.isfinite(result.action_chunk.normalized_action).all()),
                control_step=loop.control_step))
            if not result.execution.success or result.execution.replan_required:
                raise RuntimeError(f'基线执行未通过：{result.execution}')
            observation, _, terminated, truncated, _ = controller.last_step_output
            if bool(terminated.item()) or bool(truncated.item()):
                raise RuntimeError('基线场景提前结束')
        return dict(status='engineering-baseline-passed', observation_version='V1', seed=1000001,
                    task_success_claim=False, records=records, elapsed_s=time.monotonic()-started)
    finally:
        env.close()
        (output/'steps.json').write_text(json.dumps(records,indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--model-cache', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--mode', choices=['baseline','reobserve'], required=True)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--case', choices=['new-scene','historical-positive-76903'], default='new-scene')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        runtime, identity = load_runtime(args.checkpoint, args.model_cache)
        (args.output/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
        if args.mode == 'baseline':
            result = baseline(runtime, args.output/'baseline')
        else:
            if args.bundle is None:
                raise ValueError('reobserve 需要 --bundle')
            from run import run
            result = run(args.bundle, args.output/'route', case=args.case, vla_runtime=runtime)
        (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        if args.mode == 'reobserve' and not result['vla_resumed']:
            raise RuntimeError('重观察未满足恢复条件，VLA 保持暂停')
        print(json.dumps({'status':result['status'],'mode':args.mode}))
    except Exception as error:
        (args.output/'error.json').write_text(json.dumps({'type':type(error).__name__,'error':str(error)},indent=2)+'\n')
        raise


if __name__ == '__main__':
    main()
