"""续训 TCP 学生直到固定离线指标平台；保留教师合同和原训练目标。"""
from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path
import signal
import time

import numpy as np
import torch

from experiments.g2c_memory_integration.vla import load_runtime, sha256
from experiments.memory_conditioning.conditioning import MemoryBatch
from experiments.rgbd_memory_policy.stream import INSTRUCTION
from experiments.tcp_chunk_diagnostic.metrics import summarize_chunk
from experiments.tcp_chunk_diagnostic.run import label_roundtrip, save
from experiments.tcp_memory_control.data import prepare_examples
from experiments.tcp_memory_control.geometry import TCPActionSpec
from experiments.tcp_memory_control.kinematics import TCPKinematics
from experiments.tcp_memory_control.policy import build_policy
from experiments.tcp_memory_control.protocol import PROTOCOL, identity, sampling_seed
from experiments.tcp_memory_control.train import restore, verify_source
from experiments.tcp_teacher_continuation.convergence import CONFIG, decision
from robot_vla.runtime.policy_runtime import _move_model_inputs
from robot_vla.training.flow_matching import (
    sample_flow_training_target, masked_flow_mse, euler_integrate_actions,
)

FORMAT = 'tcp-teacher-continuation/v1'


def atomic_checkpoint(path, payload):
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    for name in ('data', 'training', 'checkpoint', 'model-cache', 'source-manifest', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.resume:
        if not (args.output/'latest.pt').is_file():
            parser.error('续跑要求本轮完整checkpoint')
    else:
        args.output.mkdir(parents=True, exist_ok=False)
    status = dict(status='preflight', optimizer_steps=0, actuator_steps=0,
                  max_steps=None, max_seconds=None, offline_only=True)
    started = time.monotonic()
    stopped = False

    def request_stop(*_):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def publish(**fields):
        status.update(fields, process_elapsed_s=time.monotonic()-started, updated_unix=time.time())
        save(args.output/'status.json', status)
        print(json.dumps(status, ensure_ascii=False), flush=True)

    try:
        source_hash = verify_source(args.source_manifest)
        ident = json.loads((args.training/'identity.json').read_text())
        prior = json.loads((args.training/'result.json').read_text())
        if prior['status'] != 'completed' or identity(ident) != prior['identity_sha256']:
            raise ValueError('原训练身份无效')
        fk = TCPKinematics()
        raw, hashes, denominator = prepare_examples(args.data, fk)
        if hashes != ident['data_sha256'] or denominator != ident['denominator']:
            raise ValueError('数据与教师训练身份不符')
        if {s: len(rows) for s, rows in raw.items()} != {'train': 176, 'development': 88}:
            raise ValueError('窗口分母发生改变')
        if sha256(fk.urdf_path) != ident['urdf_sha256']:
            raise ValueError('URDF改变')
        roundtrip = label_roundtrip(raw, args.data, fk)
        base, upstream = load_runtime(args.checkpoint, args.model_cache)
        if json.loads(json.dumps(upstream)) != ident['upstream']:
            raise ValueError('上游模型或统计量改变')
        policy = build_policy(base.policy, 'tcp-relative').to('cuda')
        parent_sha = prior['results']['tcp-relative']['checkpoint_sha256']
        restore(policy, args.training/'tcp-relative.pt', parent_sha,
                prior['identity_sha256'], 'tcp-relative', source_hash)
        policy.eval()
        config = dict(schema=FORMAT, parent_sha256=parent_sha,
                      parent_identity=prior['identity_sha256'], source_sha256=source_hash,
                      learning_rate=PROTOCOL['learning_rate'], accumulation=2, memory_dropout=.25,
                      eval_steps=10, dtype='bfloat16', sampling='train replacement; fixed evaluation seeds',
                      continuation_seed=20260908, stopping=CONFIG,
                      optimizer_restart='original checkpoint contains no AdamW state',
                      code={n: sha256(Path(__file__).parent/n) for n in ('run.py','convergence.py')},
                      diagnostic_code={n: sha256(Path(__file__).parent.parent/'tcp_chunk_diagnostic'/n)
                                       for n in ('run.py','metrics.py')})
        config = json.loads(json.dumps(config))
        config_id = identity(config)
        if args.resume:
            if json.loads((args.output/'config.json').read_text()) != config:
                raise ValueError('恢复配置或代码身份改变')
        else:
            save(args.output/'config.json', config)
            save(args.output/'input-verification.json', dict(parent_sha256=parent_sha,
                 identity_sha256=prior['identity_sha256'], data_sha256=hashes, label_roundtrip=roundtrip))
        publish(status='encoding_teacher_windows', configuration_identity=config_id)
        examples = {s: [] for s in raw}
        for split, rows in raw.items():
            for x in rows:
                if stopped:
                    raise InterruptedError('编码阶段收到停止请求，尚未更新权重')
                encoded = base.processor_adapter.encode(x['rgb_external'], x['rgb_wrist'], INSTRUCTION)
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    context = base.policy.encode_context(_move_model_inputs(encoded.model_inputs, base.device))
                examples[split].append(dict(seed=x['seed'], anchor=x['anchor'], context=context,
                    proprio=torch.tensor(base.proprio_normalizer.normalize(x['physical_proprio'])[None], device='cuda'),
                    action=torch.tensor(x['tcp_action'][None], device='cuda'),
                    features=torch.tensor(x['tcp_features'][None], device='cuda'),
                    available=x['snapshot']['available'], base_from_tcp=x['base_from_tcp'],
                    actual_q=x['physical_proprio'][:7], teacher=TCPActionSpec().denormalize(x['tcp_action'])))
        del raw
        gc.collect()
        params = list(policy.expert.parameters()) + list(policy.memory_encoder.parameters())
        optimizer = torch.optim.AdamW(params, lr=config['learning_rate'])
        mask = torch.ones((1,16), dtype=torch.bool, device='cuda')
        rng = np.random.default_rng(config['continuation_seed'])
        torch.manual_seed(config['continuation_seed'])
        torch.cuda.manual_seed_all(config['continuation_seed'])
        history, step, best_score, best_step = [], 0, float('inf'), 0
        window_losses = []
        exposures = [0]*len(examples['train'])
        if args.resume:
            p = torch.load(args.output/'latest.pt', map_location='cpu', weights_only=True)
            if p['format'] != FORMAT or p['configuration_identity'] != config_id:
                raise ValueError('续跑checkpoint身份无效')
            policy.expert.load_state_dict(p['expert'], strict=True)
            policy.memory_encoder.load_state_dict(p['memory_encoder'], strict=True)
            optimizer.load_state_dict(p['optimizer'])
            rng.bit_generator.state = p['numpy_rng']
            torch.set_rng_state(p['torch_rng'])
            torch.cuda.set_rng_state_all(p['cuda_rng'])
            history, step = p['history'], p['step']
            best_score, best_step = p['best_score'], p['best_step']
            window_losses, exposures = p['window_losses'], p['exposures']
            del p

        def condition(x, enabled=True):
            return policy.condition_context(x['context'], MemoryBatch(x['features'],
                torch.tensor([[enabled and x['available']]], device='cuda')))

        def loss(x, enabled, seed):
            generator = torch.Generator(device='cuda').manual_seed(seed)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                target = sample_flow_training_target(x['action'], mask, generator=generator)
                velocity = policy.expert(condition(x, enabled), x['proprio'], target.noisy_action,
                                         target.flow_time, mask)
                return masked_flow_mse(velocity, target.target_velocity, mask)

        @torch.no_grad()
        def assess():
            policy.eval()
            rows, groups = [], {}
            for split, batch in examples.items():
                selected = []
                for x in batch:
                    seed = sampling_seed(x['seed'], x['anchor'])
                    flow = float(loss(x, True, seed))
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        context = condition(x)
                        kv = policy.expert.prepare_context_kv(context)
                        noise = torch.randn(x['action'].shape, device='cuda',
                                            generator=torch.Generator(device='cuda').manual_seed(seed))
                        sampled = euler_integrate_actions(lambda a,t: policy.expert(
                            context, x['proprio'], a, t, mask, context_kv=kv), noise, mask, num_steps=10)
                    physical = TCPActionSpec().denormalize(sampled[0].float().cpu().numpy())
                    metrics = summarize_chunk(physical, x['teacher'], x['base_from_tcp'], x['actual_q'], fk)
                    ik = metrics['kinematics']['student']
                    row = dict(split=split, seed=x['seed'], anchor=x['anchor'], flow_mse=flow,
                        translation_mm=float(np.mean(metrics['action_error']['primary_translation_error_mm'])),
                        rotation_deg=float(np.mean(metrics['action_error']['primary_rotation_geodesic_error_deg'])),
                        endpoint_translation_mm=metrics['cumulative_target_error']['translation_error_mm'][3],
                        endpoint_rotation_deg=metrics['cumulative_target_error']['rotation_geodesic_error_deg'][3],
                        pass_005=ik['pass_005'], pass_01=ik['pass_01'], ik_solved=ik['successful_steps']==4,
                        sampled_first4_mae_normalized=float((sampled[:,:4]-x['action'][:,:4]).abs().mean()),
                        predicted_physical=physical.tolist())
                    selected.append(row)
                    rows.append(row)
                keys = ('flow_mse','translation_mm','rotation_deg','endpoint_translation_mm',
                        'endpoint_rotation_deg','sampled_first4_mae_normalized')
                groups[split] = {k: float(np.mean([r[k] for r in selected])) for k in keys}
                groups[split].update(windows=len(selected), trajectories=len({r['seed'] for r in selected}),
                    **{k:sum(r[k] for r in selected) for k in ('pass_005','pass_01','ik_solved')})
            save(args.output/f'evaluation-{step:08d}.json', dict(step=step, groups=groups, records=rows))
            return groups

        def persist(name):
            payload = dict(format=FORMAT, configuration_identity=config_id, parent_sha256=parent_sha,
                step=step, total_steps=256+step,
                expert={k:v.detach().cpu() for k,v in policy.expert.state_dict().items()},
                memory_encoder={k:v.detach().cpu() for k,v in policy.memory_encoder.state_dict().items()},
                optimizer=copy.deepcopy(optimizer.state_dict()), numpy_rng=rng.bit_generator.state,
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                history=history, best_score=best_score, best_step=best_step,
                window_losses=window_losses, exposures=exposures)
            atomic_checkpoint(args.output/name, payload)

        def evaluate_and_save():
            nonlocal best_score, best_step, window_losses
            publish(status='evaluating', optimizer_steps=step, total_optimizer_steps=256+step)
            groups = assess()
            baseline = history[0]['groups']['development'] if history else groups['development']
            score = float(np.mean([groups['development'][k]/max(baseline[k], 1e-8)
                                  for k in ('endpoint_translation_mm', 'endpoint_rotation_deg')]))
            entry = dict(step=step, groups=groups, selection_score=score,
                         preceding_training_loss=float(np.mean(window_losses)) if window_losses else None)
            if not history:
                old = {(x['seed'],x['anchor']):x for x in prior['results']['tcp-relative']['development']}
                rows = json.loads((args.output/f'evaluation-{step:08d}.json').read_text())['records']
                diff = max(abs(r['sampled_first4_mae_normalized']-old[(r['seed'],r['anchor'])]['sampled_first4_mae_normalized'])
                           for r in rows if r['split']=='development')
                if diff > 1e-5:
                    raise ValueError(f'续训前原采样复现失败: {diff}')
                entry['parent_mae_max_difference'] = diff
            history.append(entry)
            window_losses = []
            if score < best_score:
                best_score, best_step = score, step
                persist('best.pt')
            persist('latest.pt')
            save(args.output/'history.json', history)
            state = decision(history)
            publish(status=state, optimizer_steps=step, total_optimizer_steps=256+step,
                    groups=groups, best_score=best_score, best_step=best_step,
                    gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated())
            return state

        terminal = decision(history) if history else evaluate_and_save()
        while terminal == 'training' and not stopped:
            policy.eval()
            policy.expert.train()
            policy.memory_encoder.train()
            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            for micro in range(config['accumulation']):
                index = int(rng.integers(len(examples['train'])))
                enabled = bool(rng.random() >= config['memory_dropout'])
                exposures[index] += 1
                value = loss(examples['train'][index], enabled,
                             sampling_seed(config['continuation_seed'], step*2+micro))
                if not torch.isfinite(value):
                    raise ValueError('非有限训练loss；停止且保留已验证checkpoint')
                (value/config['accumulation']).backward()
                total += float(value.detach())/config['accumulation']
            torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            if any(p.grad is not None for p in policy.context_encoder.parameters()) or any(
                p.grad is not None for p in policy.adapter.parameters()):
                raise ValueError('冻结上游出现梯度')
            optimizer.step()
            step += 1
            window_losses.append(total)
            if step % 64 == 0:
                publish(status='training', optimizer_steps=step, total_optimizer_steps=256+step,
                        recent_training_loss=float(np.mean(window_losses[-64:])))
            if step % CONFIG['eval_every'] == 0:
                terminal = evaluate_and_save()
        if stopped:
            persist('latest.pt')
            publish(status='interrupted', optimizer_steps=step)
        else:
            publish(status=terminal, optimizer_steps=step, checkpoint_sha256=sha256(args.output/'latest.pt'),
                    best_checkpoint_sha256=sha256(args.output/'best.pt'))
    except BaseException as error:
        publish(status='failed', error_type=type(error).__name__, error=str(error))
        raise


if __name__ == '__main__':
    main()
