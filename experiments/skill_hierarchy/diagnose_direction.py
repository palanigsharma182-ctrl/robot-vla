"""固定早期教师状态的动作方向诊断；尾部 mask 仅比较有效前缀。"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.skill_hierarchy.data import SevenSkillWindows
from experiments.skill_hierarchy.full_task import load_student, verify_upstream
from experiments.tcp_atomic_skills.data import action_spec
from experiments.tcp_atomic_skills.protocol import save, verify_source
from experiments.tcp_atomic_skills.runtime import load_parent
from experiments.tcp_atomic_skills.train import encode, assess


def metrics(predicted, target):
    delta = predicted-target
    return dict(translation_error_mm=float(np.linalg.norm(delta[:, :3], axis=1).mean()*1000),
                rotation_error_deg=float(np.linalg.norm(delta[:, 3:6], axis=1).mean()*180/np.pi),
                gripper_max_abs=float(np.abs(delta[:, 6]).max()),
                target_translation_mm=float(np.linalg.norm(target[:, :3], axis=1).mean()*1000),
                target_rotation_deg=float(np.linalg.norm(target[:, 3:6], axis=1).mean()*180/np.pi))


def main():
    parser = argparse.ArgumentParser()
    for name in ('checkpoint', 'model-cache', 'continued', 'parent-training', 'training',
                 'collection', 'output', 'source-manifest'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(); source = verify_source(args.source_manifest)
    args.output.mkdir(exist_ok=False); started = time.monotonic()
    result = dict(status='running', source_sha256=source,
                  data_use='four existing tuning development scenes; post-hoc diagnosis', early=[], tails=[])

    def persist():
        result['elapsed_s'] = time.monotonic()-started
        save(args.output/'result.json', result)

    persist()
    try:
        config = json.loads((args.training/'config.json').read_text())
        trained = json.loads((args.training/'result.json').read_text())
        base, policy, upstream = load_parent(args)
        verify_upstream(upstream, config['upstream'])
        load_student(policy, args.training/'latest.pt', trained, config)
        result['checkpoint_sha256'] = trained['checkpoint_sha256']
        data = SevenSkillWindows(args.collection, 'val')
        if data.hashes != config['data']['val']:
            raise ValueError('数据身份不同')
        for i, (e, t) in enumerate(data.index):
            seed = data.entries[e].randomization['seed']
            if seed not in range(1811000, 1811004):
                continue
            early = t in (0, 8, 16, 24, 32, 48)
            tail = i in config['probes']['val'] and int(data.labels[e]['action_mask'][t].sum()) < 16
            if not early and not tail:
                continue
            raw = data[i]; x = encode(base, [raw])[0]
            native = np.array(assess(policy, [x])[0]['predicted_first4'])
            target = action_spec().denormalize(raw['action'][:4])
            if early:
                result['early'].append(dict(seed=seed, anchor=t, fine_skill_id=raw['skill_id'],
                                            metrics=metrics(native, target),
                                            predicted_first4=native.tolist(), target_first4=target.tolist()))
            if tail:
                full = np.array(assess(policy, [dict(x, mask=torch.ones_like(x['mask']))])[0]['predicted_first4'])
                valid = raw['action_mask'][:4]
                result['tails'].append(dict(seed=seed, anchor=t, valid_slots=int(x['mask'].sum()),
                    compared_valid_prefix_steps=int(valid.sum()),
                    native_vs_fullmask=metrics(native[valid], full[valid]),
                    native_vs_teacher=metrics(native[valid], target[valid]),
                    fullmask_vs_teacher=metrics(full[valid], target[valid])))
            persist()
        result['status'] = 'completed'
    except BaseException as exc:
        result.update(status='error', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        persist()


if __name__ == '__main__':
    main()
