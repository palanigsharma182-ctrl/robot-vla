import numpy as np
import pytest
from experiments.seven_skill_dagger.data import skill_targets, independent_schedule


def test_boundary_keeps_exit_action_and_masks_next_skill():
    labels = dict(fine_skill_id=np.array([0, 0, 1, 1]),
                  action=np.ones((4, 16, 7), np.float32),
                  action_mask=np.ones((4, 16), bool))
    before = labels['action'].copy()
    action, mask = skill_targets(labels, 0, 0)
    assert mask.sum() == 2 and np.all(action[:2] == 1)
    assert np.all(action[2:] == 0)
    assert np.array_equal(labels['action'], before)
    assert skill_targets(labels, 1, 0)[1].sum() == 1
    with pytest.raises(ValueError):
        skill_targets(labels, 2, 0)


def test_stops_at_first_exit_even_if_skill_recurs():
    labels = dict(fine_skill_id=np.array([0, 1, 0]),
                  action=np.ones((3, 16, 7), np.float32),
                  action_mask=np.ones((3, 16), bool))
    assert skill_targets(labels, 0, 0)[1].sum() == 1


def test_schedule_is_exclusive_reproducible_and_covers_bucket():
    bucket = list(range(10, 29))
    schedule = independent_schedule(bucket, 20, 4, 123)
    assert set(schedule.flat) == set(bucket)
    assert len(set(schedule.flat[:19])) == 19
    assert np.array_equal(schedule, independent_schedule(bucket, 20, 4, 123))


def test_schedule_zero_offset_matches_legacy_permutation_stream():
    bucket = [2, 7, 13, 29, 41]
    rng = np.random.default_rng(19)
    legacy = np.concatenate([rng.permutation(bucket) for _ in range(3)])[:12]
    assert np.array_equal(independent_schedule(bucket, 3, 4, 19, offset=0).ravel(), legacy)


@pytest.mark.parametrize('offset', [3, 5, 17])
def test_schedule_continuation_preserves_stream_across_epoch_boundaries(offset):
    bucket = [2, 7, 13, 29, 41]
    full = independent_schedule(bucket, offset+8, 1, 19).ravel()
    continued = independent_schedule(bucket, 2, 4, 19, offset=offset)
    assert np.array_equal(continued.ravel(), full[offset:offset+8])
    assert set(continued.ravel()).issubset(bucket)


def test_schedule_rejects_negative_offset():
    with pytest.raises(ValueError):
        independent_schedule([1, 2], 1, 4, 19, offset=-1)


def corrective_fixture():
    row = dict(takeover=2, teacher_end=5, windows=3, recording_start_tick=10)
    mask = np.arange(16)[None,:] < np.array([3,2,1])[:,None]
    labels = dict(anchor=np.arange(2,5), action=np.zeros((3,16,7),np.float32),
                  action_mask=mask, fine_skill_id=np.full(3,2))
    rows = [dict(action_index=i, source='teacher' if i>=12 else 'student', fine_skill_id=2) for i in range(15)]
    return row, labels, rows


def test_corrective_keeps_all_teacher_anchors_including_one_action_tail():
    from experiments.seven_skill_dagger.data import validate_corrective_labels
    row, labels, rows = corrective_fixture()
    validate_corrective_labels(row, labels, rows, 2)


@pytest.mark.parametrize('fault', ['student', 'other_skill', 'tail', 'anchor', 'index'])
def test_corrective_rejects_contaminated_supervision(fault):
    from experiments.seven_skill_dagger.data import validate_corrective_labels
    row, labels, rows = corrective_fixture()
    if fault == 'student': rows[12]['source'] = 'student'
    if fault == 'other_skill': rows[14]['fine_skill_id'] = 3
    if fault == 'tail': labels['action_mask'][-1,1] = True
    if fault == 'anchor': labels['anchor'][0] = 1
    if fault == 'index': rows[14]['action_index'] = 13
    with pytest.raises(ValueError):
        validate_corrective_labels(row, labels, rows, 2)


class ReplayPart:
    def __init__(self, parent, count):
        self.student_sha = parent
        self.records = [dict(seed=1,skill=1,case='until-stop')]
        self.count = count
    def __len__(self): return self.count
    def __getitem__(self, index): return dict(trajectory_id='trajectory',anchor=index,skill_id=1)


def test_replay_keeps_new_and_old_student_states_and_index_boundaries():
    from experiments.seven_skill_dagger.data import CorrectiveReplay
    replay = CorrectiveReplay([ReplayPart('new',2), ReplayPart('old',3)])
    assert len(replay) == 5
    assert replay[1]['anchor'] == 1 and replay[2]['anchor'] == 0 and replay[4]['anchor'] == 2
    assert replay[0]['trajectory_id'] != replay[2]['trajectory_id']
    with pytest.raises(IndexError): replay[-1]
    with pytest.raises(IndexError): replay[5]


def test_replay_rejects_same_parent_same_collection_unit():
    from experiments.seven_skill_dagger.data import CorrectiveReplay
    with pytest.raises(ValueError):CorrectiveReplay([ReplayPart('same',2),ReplayPart('same',3)])
