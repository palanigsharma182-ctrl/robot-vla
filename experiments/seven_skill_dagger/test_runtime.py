import numpy as np
from experiments.seven_skill_dagger.runtime import SkillController


def controller(record, skill):
    c = SkillController.__new__(SkillController)
    c.record, c.start_skill = record, skill
    c.steps, c.limit, c.stop_reason, c.chunk_stop_requested = 12, 160, None, False
    return c


def test_collection_takes_over_before_close_without_advancing_time():
    c = controller(True, 0)
    assert c.should_interrupt_before_action(np.zeros(8))
    assert c.stop_reason == 'collection-before-close'
    assert c.steps == 12


def test_evaluation_and_grasp_are_not_guarded():
    for record, skill in ((False,0),(False,1),(True,2)):
        c = controller(record, skill)
        assert not c.should_interrupt_before_action(np.zeros(8))
        assert c.stop_reason is None


def test_collection_keeps_open_command():
    c = controller(True, 1)
    assert not c.should_interrupt_before_action(np.r_[np.zeros(7), 1.])
