"""按轨迹汇总 Chunk 诊断；窗口分母与失败记录保持完整。"""
import argparse
import json
from pathlib import Path

import numpy as np


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return dict(n=0, mean=None, median=None, p90=None, max=None)
    return dict(n=int(values.size), mean=float(values.mean()), median=float(np.median(values)),
                p90=float(np.quantile(values, .9)), max=float(values.max()))


def per_scene(rows):
    """每条轨迹先汇总，再比较不同轨迹；不是把窗口视为独立场景。"""
    return dict(
        seed=rows[0]['seed'], windows=len(rows),
        translation_error_mm=float(np.mean([r['metrics']['action_error']['primary_translation_error_mm'] for r in rows])),
        rotation_error_deg=float(np.mean([r['metrics']['action_error']['primary_rotation_geodesic_error_deg'] for r in rows])),
        endpoint_translation_error_mm=float(np.mean([r['metrics']['cumulative_target_error']['translation_error_mm'][3] for r in rows])),
        endpoint_rotation_error_deg=float(np.mean([r['metrics']['cumulative_target_error']['rotation_geodesic_error_deg'][3] for r in rows])),
        student_pass_005=float(np.mean([r['metrics']['kinematics']['student']['pass_005'] for r in rows])),
        student_pass_01=float(np.mean([r['metrics']['kinematics']['student']['pass_01'] for r in rows])),
    )


def group(rows):
    scenes = [per_scene([r for r in rows if r['seed'] == seed]) for seed in sorted({r['seed'] for r in rows})]
    output = dict(windows=len(rows), trajectories=len(scenes), per_scene=scenes,
                  scene_distribution={key: describe([s[key] for s in scenes]) for key in (
                      'translation_error_mm', 'rotation_error_deg', 'endpoint_translation_error_mm',
                      'endpoint_rotation_error_deg', 'student_pass_005', 'student_pass_01')})
    output['kinematics'] = {}
    for arm in ('teacher', 'student'):
        ik = [r['metrics']['kinematics'][arm] for r in rows]
        output['kinematics'][arm] = dict(
            windows=len(ik), solved=sum(r['successful_steps'] == 4 for r in ik),
            pass_005=sum(r['pass_005'] for r in ik), pass_01=sum(r['pass_01'] for r in ik),
            maximum_delta_rad=describe([r['max_joint_delta'] for r in ik if r['max_joint_delta'] is not None]),
            ik_failures=[dict(seed=r['seed'], anchor=r['anchor'], reason=r['metrics']['kinematics'][arm]['first_ik_failure_reason'])
                         for r in rows if r['metrics']['kinematics'][arm]['first_ik_failure_step'] is not None],
        )
    if rows:
        predicted = np.asarray([r['predicted_physical'] for r in rows])
        teacher = np.asarray([r['teacher_physical'] for r in rows])
        error = predicted - teacher
        output['first4_component_bias'] = error[:, :4].mean(axis=(0, 1)).tolist()
        output['first4_component_mae'] = np.abs(error[:, :4]).mean(axis=(0, 1)).tolist()
        output['first4_peak_rotation_rad'] = {
            'teacher': float(np.abs(teacher[:, :4, 3:6]).max()),
            'student': float(np.abs(predicted[:, :4, 3:6]).max())}
        output['slot_translation_error_mm'] = np.asarray([r['metrics']['action_error']['translation_error_mm'] for r in rows]).mean(axis=0).tolist()
        output['slot_rotation_error_deg'] = np.asarray([r['metrics']['action_error']['rotation_geodesic_error_deg'] for r in rows]).mean(axis=0).tolist()
        output['clamp_count_by_channel'] = np.asarray([r['clamp_count_by_channel'] for r in rows]).sum(axis=0).tolist()
        output['clamp_denominator_per_channel'] = 16 * len(rows)
    return output


def summarize(result):
    rows = result['records']
    completed = [r for r in rows if r['status'] == 'completed']
    keys = [(r['split'], r['seed'], r['anchor']) for r in completed]
    planned = {(r['split'], r['seed'], r['anchor']) for r in result.get('planned', [])}
    if len(keys) != len(set(keys)) or not set(keys).issubset(planned):
        raise ValueError('重复窗口或窗口不在预定计划')
    if result['status'] == 'completed' and (len(keys) != 264 or set(keys) != planned):
        raise ValueError('完成声明与完整分母不一致')
    groups = {}
    for split in ('train', 'development'):
        selected = [r for r in completed if r['split'] == split]
        groups[split] = group(selected)
        for available in (False, True):
            groups[f'{split}/memory-{available}'] = group([r for r in selected if r['memory_available'] == available])
    groups['train/exposed'] = group([r for r in completed if r['split'] == 'train' and r['training_exposures'] > 0])
    groups['train/not-exposed'] = group([r for r in completed if r['split'] == 'train' and r['training_exposures'] == 0])
    return dict(schema='tcp-chunk-diagnostic-summary/v1', status=result['status'], planned=264,
                completed=len(completed), not_completed=264-len(completed), groups=groups,
                label_roundtrip=result.get('label_roundtrip'),
                historical_development_mae_difference=describe([r['historical_mae_abs_difference'] for r in completed if r['split'] == 'development']),
                sampling_parity=[r['original_sampler_bitwise_equal'] for r in completed if 'original_sampler_bitwise_equal' in r],
                elapsed_s=result['elapsed_s'], gpu_peak_allocated_bytes=result.get('gpu_peak_allocated_bytes'),
                actuator_steps=0, optimizer_steps=0,
                limits='单个固定噪声；教师状态离线诊断；计划可执行性不代表PD跟踪、Reach或闭环成功；development已被使用，不是独立final test。')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('result', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = summarize(json.loads(args.result.read_text()))
    with args.output.open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({key: result[key] for key in ('status', 'planned', 'completed', 'elapsed_s')}))


if __name__ == '__main__':
    main()
