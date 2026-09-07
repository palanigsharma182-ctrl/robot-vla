"""固定教师状态上的最小 TCP Chunk 诊断；只推理和运动学，不发送动作。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import time

import numpy as np
import torch

from experiments.g2c_memory_integration.vla import load_runtime, sha256
from experiments.memory_conditioning.conditioning import MemoryBatch, MEMORY_INPUT_KEY
from experiments.rgbd_memory_policy.stream import INSTRUCTION
from experiments.tcp_memory_control.data import prepare_examples
from experiments.tcp_memory_control.geometry import TCPActionSpec, apply_delta
from experiments.tcp_memory_control.kinematics import TCPKinematics
from experiments.tcp_memory_control.policy import build_policy
from experiments.tcp_memory_control.protocol import identity, sampling_seed
from experiments.tcp_memory_control.train import restore, verify_source
from experiments.tcp_chunk_diagnostic.metrics import summarize_chunk
from robot_vla.runtime.policy_runtime import _move_model_inputs
from robot_vla.training.flow_matching import euler_integrate_actions


FORMAT = 'tcp-teacher-state-chunk-diagnostic/v1'


def save(path, value):
    """独立运行目录内原子更新小型进度；历史记录不覆盖。"""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


@torch.no_grad()
def sample_with_unclamped(policy, inputs, proprio, seed, num_steps=10):
    """复用原 Euler 实现，在最后一次 velocity 调用中记录 clamp 前端点。"""
    generator = torch.Generator(device=proprio.device).manual_seed(seed)
    noise, mask, velocity_fn = policy._prepare_action_sampling(
        inputs, proprio, generator=generator,
    )
    final = None
    calls = 0

    def observe(state, flow_time):
        nonlocal final, calls
        velocity = velocity_fn(state, flow_time)
        calls += 1
        if calls == num_steps:
            valid = mask.unsqueeze(-1)
            delta = torch.where(valid, velocity.float(), torch.zeros_like(state))
            final = torch.where(valid, state + (-1.0 / num_steps) * delta,
                                torch.zeros_like(state)).detach()
        return velocity

    prediction = euler_integrate_actions(observe, noise, mask, num_steps=num_steps)
    if calls != num_steps or final is None:
        raise RuntimeError('原采样路径的去噪次数不符')
    if not torch.equal(prediction, final.clamp(-1, 1)):
        raise RuntimeError('诊断记录与原 Euler 输出不一致')
    return prediction, final


def label_roundtrip(raw, data_root, fk):
    """全部窗口的实际四步前缀与原始 commanded FK 对照。"""
    maxima = dict(translation_m=0.0, rotation_matrix_abs=0.0)
    for seed in sorted({x['seed'] for rows in raw.values() for x in rows}):
        with np.load(data_root / str(seed) / 'sequence.npz', allow_pickle=False) as z:
            targets = z['commanded_joint_target_rad'].copy()
        for x in [x for rows in raw.values() for x in rows if x['seed'] == seed]:
            target = x['base_from_tcp'].copy()
            physical = TCPActionSpec().denormalize(x['tcp_action'])
            for slot, action in enumerate(physical[:4]):
                target = apply_delta(target, action[:6], x['base_from_tcp'])
                expected = fk.pose_base(targets[x['anchor'] + slot])
                position_error = float(np.linalg.norm(target[:3, 3] - expected[:3, 3]))
                rotation_error = float(np.max(np.abs(target[:3, :3] - expected[:3, :3])))
                maxima['translation_m'] = max(maxima['translation_m'], position_error)
                maxima['rotation_matrix_abs'] = max(maxima['rotation_matrix_abs'], rotation_error)
                if position_error > 1e-6 or rotation_error > 1e-5:
                    raise ValueError(f'标签累计不一致: seed={seed}, anchor={x["anchor"]}, slot={slot}')
    return maxima


def execute(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + args.seconds
    records = []
    result = dict(schema=FORMAT, status='preflight', actuator_steps=0, optimizer_steps=0,
                  planned_windows=264, records=records, budget_s=args.seconds)

    def checkpoint():
        result['elapsed_s'] = time.monotonic() - started
        save(args.output / 'result.json', result)

    def stop(*_):
        raise TimeoutError('累计运行预算触发，保留已完成窗口与完整分母')

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGALRM, stop)
    signal.alarm(args.seconds)
    try:
        source_hash = verify_source(args.source_manifest)
        trained = json.loads((args.training / 'result.json').read_text())
        ident = json.loads((args.training / 'identity.json').read_text())
        if trained['status'] != 'completed' or identity(ident) != trained['identity_sha256']:
            raise ValueError('原训练身份未完成或不一致')
        fk = TCPKinematics()
        raw, hashes, denominator = prepare_examples(args.data, fk)
        if hashes != ident['data_sha256'] or denominator != ident['denominator']:
            raise ValueError('诊断数据与原训练快照不一致')
        if {k: len(v) for k, v in raw.items()} != {'train': 176, 'development': 88}:
            raise ValueError('必须包含原176 train / 88 development窗口')
        if sha256(fk.urdf_path) != ident['urdf_sha256']:
            raise ValueError('URDF与原训练不同')
        result['label_roundtrip'] = label_roundtrip(raw, args.data, fk)
        result['label_roundtrip']['steps_per_window'] = 4
        result['auxiliary_16step_geometry'] = 'deferred: original float32 FK rotations accumulate orthogonality drift; 22/264 teacher windows failed only at slots 12-16, none within first4'
        result['inputs'] = dict(training_identity_sha256=trained['identity_sha256'],
            training_source_sha256=source_hash, data_sha256=hashes,
            urdf_sha256=ident['urdf_sha256'], checkpoint_sha256=trained['results']['tcp-relative']['checkpoint_sha256'],
            diagnostic_source_sha256={name: sha256(Path(__file__).parent / name)
                                      for name in ('run.py', 'metrics.py')})
        checkpoint()
        base, upstream = load_runtime(args.checkpoint, args.model_cache)
        if json.loads(json.dumps(upstream)) != ident['upstream']:
            raise ValueError('上游权重、统计或运行依赖身份与训练不同')
        policy = build_policy(base.policy, 'tcp-relative').to(base.device)
        restore(policy, args.training / 'tcp-relative.pt',
                trained['results']['tcp-relative']['checkpoint_sha256'],
                trained['identity_sha256'], 'tcp-relative', source_hash)
        policy.eval()
        torch.cuda.reset_peak_memory_stats()
        result['environment'] = dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                                     bf16=True, cuda=torch.version.cuda)
        result['status'] = 'running'
        schedule = np.asarray(ident['schedule'])
        dropout = np.asarray(ident['dropout'], dtype=bool)
        train_indices = {(x['seed'], x['anchor']): i for i, x in enumerate(raw['train'])}
        historical = {(x['seed'], x['anchor']): x for x in trained['results']['tcp-relative']['development']}
        # 以轨迹交替处理两个split；不因错误大小选样或跳过。
        seeds = {'train': sorted({x['seed'] for x in raw['train']}),
                 'development': sorted({x['seed'] for x in raw['development']})}
        ordered = []
        for i in range(16):
            for split in ('train', 'development'):
                if i < len(seeds[split]):
                    ordered.extend((split, x) for x in raw[split] if x['seed'] == seeds[split][i])
        result['planned'] = [dict(split=s, seed=x['seed'], anchor=x['anchor']) for s, x in ordered]
        for index, (split, x) in enumerate(ordered):
            if time.monotonic() >= deadline:
                stop()
            seed_action = sampling_seed(x['seed'], x['anchor'])
            row = dict(split=split, seed=x['seed'], anchor=x['anchor'],
                       memory_available=x['snapshot']['available'], sampling_seed=seed_action,
                       status='running')
            records.append(row)
            encoded = base.processor_adapter.encode(x['rgb_external'], x['rgb_wrist'], INSTRUCTION)
            inputs = _move_model_inputs(encoded.model_inputs, base.device)
            inputs[MEMORY_INPUT_KEY] = MemoryBatch(
                torch.tensor(x['tcp_features'][None], device=base.device),
                torch.tensor([[x['snapshot']['available']]], device=base.device))
            proprio = torch.tensor(base.proprio_normalizer.normalize(x['physical_proprio'])[None],
                                   device=base.device)
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                prediction, unclamped = sample_with_unclamped(policy, inputs, proprio, seed_action)
                if index < 2:
                    original = policy.sample_actions(inputs, proprio, num_steps=10,
                        generator=torch.Generator(device=base.device).manual_seed(seed_action))
                    row['original_sampler_bitwise_equal'] = bool(torch.equal(prediction, original))
                    if not row['original_sampler_bitwise_equal']:
                        raise ValueError('诊断采样包装改变原预测')
            predicted = prediction[0].float().cpu().numpy()
            preclamp = unclamped[0].float().cpu().numpy()
            physical = TCPActionSpec().denormalize(predicted)
            teacher = TCPActionSpec().denormalize(x['tcp_action'])
            row['metrics'] = summarize_chunk(physical, teacher, x['base_from_tcp'],
                                             x['physical_proprio'][:7], fk)
            row.update(status='completed', predicted_normalized=predicted.tolist(),
                       predicted_physical=physical.tolist(), teacher_physical=teacher.tolist(),
                       unclamped_normalized=preclamp.tolist(),
                       clamp_count_by_channel=(np.abs(preclamp) > 1).sum(axis=0).tolist(),
                       sampled_first4_mae_normalized=float(np.abs(predicted[:4] - x['tcp_action'][:4]).mean()))
            if split == 'train':
                seen = schedule == train_indices[(x['seed'], x['anchor'])]
                row['training_exposures'] = int(seen.sum())
                row['memory_training_exposures'] = int((seen & ~dropout).sum()) if x['snapshot']['available'] else 0
            else:
                old = historical[(x['seed'], x['anchor'])]['sampled_first4_mae_normalized']
                row['historical_mae_abs_difference'] = abs(row['sampled_first4_mae_normalized'] - old)
            result['gpu_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            checkpoint()
            print(json.dumps(dict(index=index + 1, split=split, seed=x['seed'], anchor=x['anchor'],
                                  elapsed_s=round(time.monotonic() - started, 2))), flush=True)
        result['status'] = 'completed'
        result['completed_windows'] = len(records)
    except BaseException as error:
        result.update(status='incomplete', error_type=type(error).__name__, error=str(error))
        if records and records[-1]['status'] == 'running':
            records[-1]['status'] = 'error'
        raise
    finally:
        signal.alarm(0)
        checkpoint()


def main():
    parser = argparse.ArgumentParser()
    for name in ('data', 'training', 'checkpoint', 'model-cache', 'source-manifest', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=900)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 900:
        parser.error('本次诊断总进程预算必须在1–900秒')
    execute(args)


if __name__ == '__main__':
    main()
