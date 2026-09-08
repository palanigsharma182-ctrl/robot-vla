"""一个作业训练一个独立Expert；历史冻结缓存只读，动作监督止于本技能出口。"""
import argparse
import json
from pathlib import Path
import signal
import time
import numpy as np
import torch

from experiments.seven_skill_dagger.data import skill_targets, independent_schedule, CorrectiveWindows, CorrectiveReplay
from experiments.seven_skill_dagger.run import load_student, PARENT_SHA
from experiments.skill_hierarchy.data import SevenSkillWindows
from experiments.skill_hierarchy.train import FeatureCache
from experiments.skill_hierarchy.contract import SKILLS
from experiments.tcp_atomic_skills.train import loss_for
from experiments.tcp_atomic_skills.protocol import save, sha, identity, verify_source
from experiments.tcp_memory_control.protocol import sampling_seed

FORMAT = 'independent-skill-checkpoint/v1'


def main():
    p = argparse.ArgumentParser()
    for name in ('checkpoint','model-cache','continued','parent-training','collection','metric-run',
                 'student','output','source-manifest'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--expected-sha', default=PARENT_SHA)
    p.add_argument('--corrective', type=Path)
    p.add_argument('--replay-manifest', type=Path)
    p.add_argument('--seed-offset', type=int, default=0)
    p.add_argument('--nominal-offset', type=int, default=0,
                   help='纯BC从确定性原示范序列跳过的窗口数，不改变训练噪声')
    p.add_argument('--skill', type=int, choices=range(7), required=True)
    p.add_argument('--steps', type=int, default=512)
    p.add_argument('--accumulation', type=int, default=4)
    p.add_argument('--wall-seconds', type=int, default=14400)
    args = p.parse_args()
    if args.nominal_offset < 0 or (args.nominal_offset and args.corrective):
        p.error('nominal-offset须非负；非零值当前仅支持纯BC覆盖对照')
    args.output.mkdir(parents=True, exist_ok=False)
    stopped = False
    started = time.monotonic()
    step = 0
    losses = []
    def stop(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    def check():
        if stopped or time.monotonic()-started >= args.wall_seconds:
            raise InterruptedError('作业停止或时间上限')
    def status(stage, **extra):
        row = dict(stage=stage, skill=SKILLS[args.skill], step=step,
                   elapsed_s=time.monotonic()-started, updated_unix=time.time(), **extra)
        save(args.output/'status.json', row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    try:
        status('preflight')
        source = verify_source(args.source_manifest)
        dataset = SevenSkillWindows(args.collection, 'train')
        base, policy, parent, old = load_student(args)
        if dataset.hashes != old['data']['train']:
            raise ValueError('输入数据与缓存来源不一致')
        correction = None
        replay_sources = []
        if args.replay_manifest and not args.corrective:
            raise ValueError('历史纠偏回放必须与当前轮纠偏一起使用')
        if args.corrective:
            if args.accumulation != 4:
                raise ValueError('首轮纠偏固定3个原示范+1个纠偏窗口')
            correction = CorrectiveWindows(args.corrective, args.skill, sha(args.student),
                                           {e.randomization['seed'] for e in dataset.entries})
        if args.replay_manifest:
            replay_sources = json.loads(args.replay_manifest.read_text())
            parts = [correction]
            for row in replay_sources:
                if sha(Path(row['root'])/'collection.json') != row['collection_sha256']:
                    raise ValueError('历史纠偏manifest身份不一致')
                parts.append(CorrectiveWindows(row['root'], args.skill, row['student_sha256'],
                             {e.randomization['seed'] for e in dataset.entries}))
            correction = CorrectiveReplay(parts)
        cache_root = Path(old['cache_root'])
        schedule = independent_schedule(dataset.buckets[args.skill], args.steps,
                                        args.accumulation, 1930042+args.skill+args.seed_offset,
                                        offset=args.nominal_offset)
        # 多进程只能读取已存在缓存，不允许隐式创建共享缓存条目。
        missing = [int(i) for i in np.unique(schedule) if not (cache_root/f'train-{i:06d}.pt').exists()]
        if missing:
            raise ValueError(f'冻结缓存缺少{len(missing)}个样本；需要先单独编码')
        config = dict(format=FORMAT, skill_id=args.skill, skill=SKILLS[args.skill],
                      source_sha256=source, parent_sha256=sha(args.student),
                      data_sha256=identity(dataset.hashes), cache_identity=old['cache_identity'],
                      steps=args.steps, accumulation=args.accumulation, learning_rate=1e-5,
                      nominal_offset=args.nominal_offset,
                      seed=1930042+args.skill+args.seed_offset, optimizer='independent AdamW reset',
                      labels='first contiguous skill segment; exit action included; later targets zero masked',
                      frozen='Qwen, adapter, memory encoder', selection='fixed final update; no acceptance claim',
                      wall_seconds=args.wall_seconds,
                      corrective_sha256=sha(args.corrective/'collection.json') if correction is not None else None,
                      corrective_windows=len(correction) if correction is not None else 0,
                      replay_manifest_sha256=sha(args.replay_manifest) if args.replay_manifest else None,
                      replay_sources=replay_sources,
                      mixture='3 nominal + 1 corrective' if correction is not None else '4 nominal')
        config_id = identity(config)
        save(args.output/'config.json', config)
        save(args.output/'schedule.json', schedule.tolist())
        cache = FeatureCache(cache_root, base, {'train':dataset}, old['cache_identity'], lambda:stopped)
        newcache = None
        if correction is not None:
            newcache = FeatureCache(args.output/'corrective-cache', base, {'corrective':correction}, config_id, lambda:stopped)
            corrective_schedule = independent_schedule(list(range(len(correction))), args.steps, 1, 1940042+args.skill+args.seed_offset)
            save(args.output/'corrective-schedule.json', corrective_schedule.tolist())
        params = list(policy.expert.parameters())
        optimizer = torch.optim.AdamW(params, lr=1e-5)
        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed_all(config['seed'])
        torch.use_deterministic_algorithms(True)
        def persist(complete=False):
            payload = dict(format=FORMAT, configuration_identity=config_id,
                           skill_id=args.skill, step=step, completed=complete,
                           parent_sha256=config['parent_sha256'],
                           expert={k:v.detach().cpu() for k,v in policy.expert.state_dict().items()},
                           memory_encoder={k:v.detach().cpu() for k,v in policy.memory_encoder.state_dict().items()},
                           optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state(),
                           cuda_rng=torch.cuda.get_rng_state_all(), losses=losses)
            temporary = args.output/'latest.tmp'
            torch.save(payload, temporary)
            temporary.replace(args.output/'latest.pt')
        try:
            for indices in schedule:
                check()
                policy.expert.train()
                optimizer.zero_grad(set_to_none=True)
                total = 0.
                for slot, index in enumerate(indices):
                    check()
                    index = int(index)
                    if correction is not None and slot == 3:
                        x = newcache.get('corrective', int(corrective_schedule[step, 0]))
                        if x['skill_id'] != args.skill:
                            raise ValueError('纠偏特征与目标技能不一致')
                    else:
                        x = cache.get('train', index)
                        e, anchor = dataset.index[index]
                        action, mask = skill_targets(dataset.labels[e], anchor, args.skill)
                        if x['anchor'] != anchor or x['skill_id'] != args.skill:
                            raise ValueError('冻结特征索引与技能不一致')
                        x['action'] = torch.tensor(action[None], device='cuda')
                        x['mask'] = torch.tensor(mask[None], device='cuda')
                    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                        loss = loss_for(policy, x, sampling_seed(config['seed'], step*args.accumulation+slot))
                    if not torch.isfinite(loss):
                        raise ValueError('训练loss非有限')
                    (loss/args.accumulation).backward()
                    total += float(loss.detach())/args.accumulation
                torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
                optimizer.step()
                step += 1
                losses.append(total)
                if step%16 == 0 or step==1:
                    status('training', loss_mean_last16=float(np.mean(losses[-16:])),
                           peak_gpu_mib=torch.cuda.max_memory_allocated()/1024**2)
                if step%128 == 0:
                    persist()
            persist(True)
        except BaseException:
            # 保存的step只计完成的optimizer更新；未完成梯度不进入恢复点。
            persist()
            raise
        loaded = torch.load(args.output/'latest.pt', map_location='cpu', weights_only=True)
        if loaded['skill_id'] != args.skill or loaded['configuration_identity'] != config_id or not loaded['completed']:
            raise ValueError('保存后的策略身份校验失败')
        policy.expert.load_state_dict(loaded['expert'], strict=True)
        save(args.output/'result.json', dict(status='completed', skill_id=args.skill, steps=step,
             checkpoint_sha256=sha(args.output/'latest.pt'), strict_reload=True, accepted=False,
             unique_windows=len(np.unique(schedule[:, :3] if correction is not None else schedule)),
             exposures=int(schedule.size), nominal_exposures=step*(3 if correction is not None else args.accumulation),
             corrective_exposures=step if correction is not None else 0))
        status('completed')
    except BaseException as exc:
        status('stopped' if isinstance(exc, InterruptedError) else 'error', error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()
