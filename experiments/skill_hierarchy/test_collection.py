"""教师侧七段标签与 BC 窗口的错位、截断反例。"""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest

from experiments.skill_hierarchy.collect import validate_sidecar, build_labels, TRAIN_SEEDS, DEV_SEEDS
from experiments.skill_hierarchy.contract import VERSION, BoundaryTracker, EXIT


def fixture():
    tracker = BoundaryTracker(); pose = np.eye(4).tolist()
    before = dict(held=0, opening=1, pregrasp_distance_m=.10, tcp_speed_m_s=0)
    before = tracker.observe(before, np.eye(4), 0.)
    rows = []
    for skill in ('approach', 'align', 'grasp', 'grasp', 'grasp', 'lift', 'transport', 'lower',
                  'release', 'release', 'release', 'release'):
        i = len(rows)
        row = dict(action_index=i, fine_skill_id=tracker.active, before=before,
                   tcp_from_object_before=pose, tcp_from_object_after=pose,
                   commanded_joint_target_rad=[0.]*7, gripper_command=1.)
        after = {k:(lo+hi)/2 for k,(lo,hi) in EXIT[skill].items()}
        before = tracker.observe(after, np.eye(4), (i+1)/20.)
        row['after'] = before; rows.append(row)
    arrays = SimpleNamespace(num_steps=len(rows), commanded_joint_target_rad=np.zeros((len(rows),7)),
                             action=np.tile([0.]*7+[1.], (len(rows),1)))
    side = dict(version=VERSION, source='privileged-teacher-evaluation', rows=rows, events=tracker.events)
    return arrays, side


def test_recompute_all_seven_boundaries():
    arrays, side = fixture()
    assert validate_sidecar(arrays, side).tolist() == [0,1,2,2,2,3,4,5,6,6,6,6]


@pytest.mark.parametrize('corruption', ['shift', 'command', 'drift', 'missing', 'continuity'])
def test_sidecar_corruption_rejected(corruption):
    arrays, original = fixture(); side = deepcopy(original)
    if corruption == 'shift': side['rows'][2]['fine_skill_id'] = 3
    if corruption == 'command': side['rows'][2]['commanded_joint_target_rad'][0] = .01
    if corruption == 'drift': side['rows'][4]['after']['relative_drift_m'] = .001
    if corruption == 'missing': side['rows'].pop()
    if corruption == 'continuity': side['rows'][1]['before'] = dict(side['rows'][1]['before'], opening=.85)
    with pytest.raises(ValueError): validate_sidecar(arrays, side)


def test_future_chunk_crosses_skill_boundary_and_only_masks_episode_tail():
    class FK:
        def pose_base(self, q):
            pose = np.eye(4); pose[:3,3] = q[:3]; return pose
    n=20; target=np.zeros((n,7),np.float32);target[:,0]=np.arange(1,n+1)*.001
    previous=np.vstack([np.zeros((1,7)),target[:-1]]).astype(np.float32)
    times=np.arange(n)/20.
    arrays=SimpleNamespace(num_steps=n, commanded_joint_target_rad=target, previous_command_q_rad=previous,
        observation_valid=np.ones(n,bool), previous_command_valid=np.ones(n,bool),
        action=np.column_stack([target-previous,np.ones(n)]), timestamp_action=times,
        timestamp_external=times,timestamp_wrist=times,timestamp_proprio=times,
        skill_id=np.zeros(n,np.int16),proprio=np.column_stack([previous,np.zeros((n,8))]))
    ids=np.array([0]*2+[1]*18,np.int16)
    labels,error=build_labels(arrays,ids,FK())
    assert labels['action_mask'][1].sum()==16
    assert labels['action_mask'][-1].sum()==1
    assert labels['action'].shape==(n,16,7)
    assert np.array_equal(labels['fine_skill_id'],ids)
    assert error < 1e-8


def test_scene_splits_disjoint():
    assert len(TRAIN_SEEDS)==128 and len(DEV_SEEDS)==32
    assert not set(TRAIN_SEEDS)&set(DEV_SEEDS)


def test_bc_example_does_not_expose_privileged_state():
    from experiments.skill_hierarchy.data import example
    arrays=SimpleNamespace(rgb_external=np.zeros((1,2,2,3),np.uint8),
        rgb_wrist=np.zeros((1,2,2,3),np.uint8),proprio=np.zeros((1,15),np.float32),
        object_position_m=np.array([[999.,999.,999.]]),is_grasped=np.array([True]))
    labels=dict(fine_skill_id=np.array([2]),action=np.zeros((1,16,7),np.float32),
                action_mask=np.ones((1,16),bool))
    item=example(arrays,labels,0,seed=1,trajectory_id='synthetic',instruction='pick and place')
    assert set(item)=={'seed','trajectory_id','anchor','skill_id','rgb_external','rgb_wrist',
                      'physical_proprio','instruction','action','action_mask','features','available'}
    assert not item['available'] and not item['features'].any()
    item['physical_proprio'][0]=1
    assert arrays.proprio[0,0]==0
