"""固定评估序列的平台判定；不把离线收敛解释为闭环成功。"""
import math

METRICS = ('flow_mse', 'translation_mm', 'rotation_deg',
           'endpoint_translation_mm', 'endpoint_rotation_deg')
CONFIG = dict(eval_every=512, window=8, mean_change=0.02, relative_span=0.10,
              confirmations=3, regression_ratio=1.20, regression_checks=4)


def plateau(history, split):
    """比较相邻四次评估的均值，同时约束八次评估的波动范围。"""
    window = CONFIG['window']
    if len(history) < window:
        return False
    for key in METRICS:
        values = [float(x['groups'][split][key]) for x in history[-window:]]
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError('收敛指标必须为非负有限值')
        before, after = sum(values[:4]) / 4, sum(values[4:]) / 4
        scale = max(before, after, 1e-8)
        if abs(after - before) / scale > CONFIG['mean_change']:
            return False
        if (max(values) - min(values)) / scale > CONFIG['relative_span']:
            return False
    return True


def decision(history):
    """三次连续平台才收敛；持续开发集退化单独标记，不冒充收敛。"""
    if len(history) < CONFIG['window'] + CONFIG['confirmations'] - 1:
        return 'training'
    stable = all(plateau(history[:len(history)-i], split)
                 for i in range(CONFIG['confirmations'])
                 for split in ('train', 'development'))
    # 只在训练指标已平台且开发集连续显著劣于历史最佳时停止。
    recent = history[-CONFIG['regression_checks']:]
    best = min(x['selection_score'] for x in history)
    if plateau(history, 'train') and all(
        x['selection_score'] > max(best, 1e-8) * CONFIG['regression_ratio'] for x in recent
    ):
        return 'development_regression'
    if stable:
        return 'offline_converged'
    return 'training'
