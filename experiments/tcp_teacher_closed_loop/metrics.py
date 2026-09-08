"""从原始command/actual日志计算指标，兼容原冻结trace格式。"""
import json
import numpy as np


def canonical_audit(value):
    """把内存与磁盘中的初态记录统一为相同JSON类型，不放宽数值比较。"""
    return json.loads(json.dumps(value,allow_nan=False))


def trajectory_metrics(trace, initial_distance, plans):
    steps=[x for x in trace if x['policy_step']>0 and not x['holding']]
    distances=[initial_distance]+[x['distance_m'] for x in steps]
    return dict(policy_steps=max((x['policy_step'] for x in trace),default=0),
        replans=len(plans),final_distance_m=trace[-1]['distance_m'] if trace else initial_distance,
        minimum_distance_m=min(distances),reached=bool(min(distances)<=.02),
        distance_by_policy_step={str(x['policy_step']):x['distance_m'] for x in steps},
        tracking_error_max_rad=max((float(np.max(np.abs(np.asarray(x['command_q'])-np.asarray(x['q_after']))))
                                    for x in steps),default=None),
        memory_actions=sum(x['action_used_memory'] for x in steps),
        occluded_memory_actions=sum(x['action_used_memory'] and x['occluded'] for x in steps))
