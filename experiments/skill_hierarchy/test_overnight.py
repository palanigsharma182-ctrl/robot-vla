"""最低时长、宏平均掩盖退化和阶段续接的反例。"""
import copy
from experiments.skill_hierarchy.overnight import plateau,may_finish,command_for


def history():
    row=dict(translation_error_mm=2.,rotation_delta_error_deg=.2,
             gripper_mae=.01,gripper_binary_accuracy=1.)
    return [dict(step=i*512,groups={'val':{'macro':dict(row),
            'per_skill':{str(s):dict(row) for s in range(7)}}}) for i in range(6)]


def test_plateau_cannot_stop_before_seven_hours():
    assessment=plateau(history())
    assert assessment['stable']
    assert not may_finish(25199,25200,assessment)
    assert may_finish(25200,25200,assessment)


def test_changing_skill_cannot_hide_in_stable_macro():
    h=history();h[-1]['groups']['val']['per_skill']['2']['gripper_mae']=.04
    assert not plateau(h)['stable']


def test_stable_regression_is_not_success():
    h=history()
    for e in h[1:]:e['groups']['val']['per_skill']['3']['translation_error_mm']=2.5
    r=plateau(h)
    assert r['stable'] and r['reason']=='plateau_with_regression'
    assert '3:translation_error_mm' in r['regressions']
    assert not r['closed_loop_verified']


def test_short_span_and_insufficient_evaluations_rejected():
    h=history()
    assert not plateau(h[:5])['stable']
    for i,e in enumerate(h):e['step']=i*100
    assert not plateau(h)['stable']


def test_command_preserves_training_inputs():
    spec={'command':['python','-m','train','--output','old','--source-manifest','src','--collection','data']}
    original=copy.deepcopy(spec)
    cmd=command_for(spec,'new','src2','old')
    assert cmd==['python','-m','train','--output','new','--source-manifest','src2','--collection','data','--continue-from','old']
    assert spec==original
