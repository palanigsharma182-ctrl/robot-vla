"""只读权重／缓存诊断，并以原命令回放恢复失败时刻观测；不训练或选择策略。"""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from experiments.skill_hierarchy.data import SevenSkillWindows
from experiments.skill_hierarchy.full_task import load_student, verify_upstream
from experiments.skill_hierarchy.probe import HierarchyTeacher
from experiments.tcp_atomic_skills.protocol import identity, save, sha, verify_source
from experiments.tcp_atomic_skills.runtime import load_parent, predict
from experiments.tcp_atomic_skills.train import assess, encode
from experiments.tcp_memory_control.protocol import sampling_seed
from experiments.memory_reobserve.runtime import observation_digest
from robot_vla.adapters import FrankaObservationAdapter
from robot_vla.contracts import RobotSpec
from robot_vla.evaluation.maniskill import _read_online_observation
from robot_vla.model.qwen_context import QwenContext
from robot_vla.tasks.pick_place import build_pick_place_task

SEEDS = tuple(range(1811000, 1811004))


def max_difference(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f'比较 shape 不一致：{a.shape} / {b.shape}')
    return float(np.max(np.abs(a.astype(float)-b.astype(float))))


def tensor_difference(a, b):
    return max_difference(a.detach().float().cpu().numpy(), b.detach().float().cpu().numpy())


def load_cached(path, expected_identity):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload['configuration_identity'] != expected_identity:
        raise ValueError('缓存身份不匹配')
    result = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v
              for k, v in payload.items() if k not in ('configuration_identity', 'context')}
    result['context'] = QwenContext(**{k: None if v is None else v.to('cuda')
                                      for k, v in payload['context'].items()})
    return result


def compare_paths(base, policy, raw, cached=None):
    fresh = encode(base, [raw])[0]
    native = assess(policy, [fresh])[0]
    full = dict(fresh, mask=torch.ones_like(fresh['mask']))
    all_valid = native if bool(fresh['mask'].all()) else assess(policy, [full])[0]
    online = predict(base, policy, SimpleNamespace(**raw), sampling_seed(raw['seed'], raw['anchor']))
    record = dict(seed=raw['seed'], anchor=raw['anchor'], skill_id=raw['skill_id'],
                  valid_slots=int(fresh['mask'].sum()),
                  online_vs_fresh_fullmask_max_abs=max_difference(online[:4], all_valid['predicted_first4']),
                  native_vs_fullmask_max_abs=max_difference(native['predicted_first4'], all_valid['predicted_first4']),
                  native_teacher_metrics={k: native[k] for k in (
                      'translation_error_mm', 'rotation_delta_error_deg', 'gripper_mae')})
    if cached is not None:
        old = assess(policy, [cached])[0]
        record.update(cache_context_max_abs=tensor_difference(fresh['context'].tokens, cached['context'].tokens),
                      cache_context_mask_equal=torch.equal(fresh['context'].mask, cached['context'].mask),
                      cache_proprio_max_abs=tensor_difference(fresh['proprio'], cached['proprio']),
                      cache_action_max_abs=tensor_difference(fresh['action'], cached['action']),
                      cache_action_mask_equal=torch.equal(fresh['mask'], cached['mask']),
                      cached_vs_fresh_native_max_abs=max_difference(old['predicted_first4'], native['predicted_first4']))
    return record, online


def main():
    p = argparse.ArgumentParser()
    for name in ('checkpoint', 'model-cache', 'continued', 'parent-training', 'training',
                 'collection', 'prior-rollout', 'output', 'source-manifest'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    source = verify_source(args.source_manifest)
    args.output.mkdir(exist_ok=False)
    started = time.monotonic()
    result = dict(status='running', source_sha256=source, data_use='existing tuning development only',
                  training=False, cache_writes=False, scene_seeds=list(SEEDS),
                  cached_windows=[], initial_windows=[], replay=[],
                  prior_result_sha256=sha(args.prior_rollout/'result.json'))

    def persist():
        result['elapsed_s'] = time.monotonic()-started
        save(args.output/'result.json', result)

    persist()
    try:
        config = json.loads((args.training/'config.json').read_text())
        trained = json.loads((args.training/'result.json').read_text())
        base, policy, upstream = load_parent(args)
        result['upstream_identity'] = verify_upstream(upstream, config['upstream'])
        load_student(policy, args.training/'latest.pt', trained, config)
        result['checkpoint_sha256'] = trained['checkpoint_sha256']
        dataset = SevenSkillWindows(args.collection, 'val')
        if dataset.hashes != config['data']['val']:
            raise ValueError('development 数据与训练身份不同')
        selected = [i for i in config['probes']['val']
                    if dataset.entries[dataset.index[i][0]].randomization['seed'] in SEEDS]
        if len(selected) != 28:
            raise ValueError('预定四场景每技能一个缓存窗口应为 28 个')
        for index in selected:
            raw = dataset[index]
            path = Path(config['cache_root'])/f'val-{index:06d}.pt'
            cached = load_cached(path, config['cache_identity'])
            row, _ = compare_paths(base, policy, raw, cached)
            row.update(dataset_index=index, cache_sha256=sha(path))
            result['cached_windows'].append(row)
            persist()
        initial = {}
        for i, (e, t) in enumerate(dataset.index):
            seed = dataset.entries[e].randomization['seed']
            if t == 0 and seed in SEEDS:
                raw = dataset[i]
                row, prediction = compare_paths(base, policy, raw)
                old = json.loads((args.prior_rollout/str(seed)/'plans.json').read_text())[0]
                row['dataset_vs_prior_initial_prediction_max_abs'] = max_difference(prediction, old['physical'])
                row['observation_sha256'] = observation_digest(SimpleNamespace(**raw))
                result['initial_windows'].append(row)
                initial[seed] = raw
                persist()
        # 原控制命令回放，模型输出仅在 shadow 中比较，绝不替换回放动作。
        adapter = FrankaObservationAdapter(RobotSpec())
        for seed in SEEDS:
            folder = args.prior_rollout/str(seed)
            plans = json.loads((folder/'plans.json').read_text())
            control = [json.loads(l) for l in (folder/'control.jsonl').read_text().splitlines()]
            by_step = {x['step']: (i, x) for i, x in enumerate(plans)}
            selected_steps = sorted({min(by_step, key=lambda s: abs(s-target))
                                     for target in (0, 24, 48, 96, plans[-1]['step'])})
            record = dict(seed=seed, selected_steps=selected_steps, max_replay_joint_error_rad=0.,
                          replay_control_steps=0, samples=[])
            result['replay'].append(record)
            with HierarchyTeacher(None, max_episode_steps=400) as teacher:
                teacher.initialize(seed)
                observation = teacher.session.observation
                for step in range(max(selected_steps)+1):
                    if step in selected_steps:
                        plan_index, plan = by_step[step]
                        online = _read_online_observation(observation, teacher.env.unwrapped, adapter,
                                                         build_pick_place_task(seed % 3).instruction)
                        raw = dict(seed=seed, anchor=plan_index, skill_id=-1,
                                   rgb_external=online.rgb_external, rgb_wrist=online.rgb_wrist,
                                   physical_proprio=online.physical_proprio, instruction=online.instruction,
                                   action=np.zeros((16, 7), np.float32), action_mask=np.ones(16, bool),
                                   features=np.zeros(12, np.float32), available=False)
                        row, prediction = compare_paths(base, policy, raw)
                        row.pop('native_teacher_metrics')  # 此处没有该偏离状态的教师标签。
                        row.update(step=step, observation_sha256=observation_digest(online),
                                   regenerated_vs_prior_plan_max_abs=max_difference(prediction, plan['physical']))
                        if step == 0:
                            first = json.loads((folder/'initial.json').read_text())
                            row['prior_initial_observation_equal'] = row['observation_sha256'] == first['observation_sha256']
                            row['dataset_initial_observation_equal'] = row['observation_sha256'] == observation_digest(SimpleNamespace(**initial[seed]))
                        record['samples'].append(row)
                        persist()
                    if step == max(selected_steps):
                        break
                    old = control[step]
                    action = np.r_[(np.array(old['command_q'])-np.array(old['q_before']))/.1,
                                   2*old['gripper_target']-1].astype(np.float32)
                    observation, _, terminated, truncated, _ = teacher.env.step(torch.tensor(action)[None])
                    q = teacher.base_env.agent.robot.get_qpos()[0, :7].cpu().numpy()
                    difference = max_difference(q, old['q_after'])
                    record['max_replay_joint_error_rad'] = max(record['max_replay_joint_error_rad'], difference)
                    record['replay_control_steps'] += 1
                    if difference > 1e-5:
                        raise ValueError(f'原命令回放发生状态分叉：{seed}/{step+1}/{difference}')
                    if bool(terminated.item()) or bool(truncated.item()):
                        raise ValueError('回放在原停止点之前异常终止')
            persist()
        result['status'] = 'completed'
    except BaseException as exc:
        result.update(status='error', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        persist()


if __name__ == '__main__':
    main()
