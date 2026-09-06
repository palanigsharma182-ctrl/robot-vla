"""真实控制 tick、能力缺失语义与单次预算回归；不执行仿真。"""
from types import SimpleNamespace
import numpy as np
import pytest

from experiments.memory_reobserve.session import MemoryRouteSession
from robot_vla.precision.active_front_memory_provider import build_stage2_object_memory_config
from robot_vla.precision.object_memory import ExplicitObjectStateMemory


def setup_session():
    runtime=SimpleNamespace(reset_memory_episode=lambda e:None)
    session=MemoryRouteSession(1001200,runtime)
    session.reset('episode')
    memory=ExplicitObjectStateMemory(build_stage2_object_memory_config())
    memory.reset('episode')
    return session,memory


def frame(tick):
    proprio=np.zeros(15,np.float32); proprio[-1]=1.
    return SimpleNamespace(timestamp_s=tick/20,physical_proprio=proprio,finger_force_n=np.zeros(2))


def test_only_fresh_consecutive_ticks_form_single_capability_absent_request():
    session,memory=setup_session()
    for tick in (1,2,3):
        result=session.observe_trigger(frame(tick),tick,memory=memory,close_commanded=False,camera_home=True,tracking={'valid':True})
        assert result == (tick==3)
    assert session.request.attempt_index==1
    from robot_vla.precision.active_front_memory_provider import ACTIVE_FRONT_PRIMARY_PRIMITIVE_ID
    assert session.request.selected_primitive_id == ACTIVE_FRONT_PRIMARY_PRIMITIVE_ID
    assert session.request.trigger_reason.value=='no_qualified_wrist_provider_in_parent'
    assert not session.observe_trigger(frame(4),4,memory=memory,close_commanded=False,camera_home=True,tracking={'valid':True})
    with pytest.raises(ValueError,match='新鲜'):
        session.observe_trigger(frame(4),4,memory=memory,close_commanded=False,camera_home=True,tracking={'valid':True})


def test_gap_and_contact_latch_prevent_request():
    session,memory=setup_session()
    session.observe_trigger(frame(1),1,memory=memory,close_commanded=True,camera_home=True,tracking={'valid':True})
    for tick in (2,3,4):
        assert not session.observe_trigger(frame(tick),tick,memory=memory,close_commanded=False,camera_home=True,tracking={'valid':True})
    with pytest.raises(ValueError):
        session.observe_trigger(frame(6),6,memory=memory,close_commanded=False,camera_home=True,tracking={'valid':True})


def test_available_chunk_expiry_and_close_are_checked_before_send():
    from dataclasses import replace
    from test_conditioning import fixture_state
    from experiments.memory_conditioning.conditioning import snapshot_memory
    session,_=setup_session()
    state,config,safety=fixture_state()
    state=replace(state,episode_id='episode')
    memory=ExplicitObjectStateMemory(config)
    memory._state=state
    session.tracking_valid=True
    session.snapshot=snapshot_memory(state,config,safety,episode_id='episode',timestamp_s=5.9)
    action=np.zeros(8,np.float32); action[-1]=1.
    session.before_send(action,frame(118),memory)
    with pytest.raises(RuntimeError,match='memory_stale'):
        session.before_send(action,frame(125),memory)
    assert not memory.state.valid
    memory._state=state
    close=action.copy(); close[-1]=-1.
    with pytest.raises(RuntimeError,match='gripper_close_commanded'):
        session.before_send(close,frame(118),memory)
    memory._state=state
    assert session.after_send(frame(125),action,memory)
    assert not memory.state.valid


def test_invalid_tracking_does_not_trigger_even_with_finite_proprio():
    session,memory=setup_session()
    for tick in (1,2,3):
        assert not session.observe_trigger(frame(tick),tick,memory=memory,
            close_commanded=False,camera_home=True,tracking={'valid':False})


def test_cleanup_clears_ensemble_rtc_and_command_reference():
    from robot_vla.runtime.control_loop import QwenVLAReplanLoop
    from robot_vla.execution.chunk_executor import RecedingHorizonChunkExecutor
    from robot_vla.contracts import RobotSpec
    session,_=setup_session()
    loop=QwenVLAReplanLoop(SimpleNamespace(), RecedingHorizonChunkExecutor(RobotSpec()))
    loop.control_step=120
    loop._rtc_previous_chunk=np.ones((16,8),np.float32)
    loop.executor._previous_command_q=np.ones(7,np.float32)
    loop.executor._previous_action=np.ones(8,np.float32)
    session.interruption_reason='memory_stale'
    assert session.cleanup_after_execution(loop)
    assert loop.observation_paused and loop.control_step==120
    assert loop._rtc_previous_chunk is None and loop.executor.previous_command_q is None
    assert loop.executor.previous_action is None and loop.ensembler.buffer_size==0


@pytest.mark.parametrize('reason,paused', [
    ('memory_stale', False), ('gripper_close_commanded', False),
    ('memory_stale,controller_tracking_invalid', True), ('engineering-error', True),
])
def test_visual_fallback_only_clears_memory_context_loss(reason, paused):
    from robot_vla.runtime.control_loop import QwenVLAReplanLoop
    from robot_vla.execution.chunk_executor import RecedingHorizonChunkExecutor
    from robot_vla.contracts import RobotSpec
    session,_ = setup_session()
    session.visual_fallback = True
    loop = QwenVLAReplanLoop(SimpleNamespace(), RecedingHorizonChunkExecutor(RobotSpec()))
    loop.control_step = 127
    loop._rtc_previous_chunk = object()
    loop.executor._previous_command_q = np.ones(7, np.float32)
    session.interruption_reason = reason
    assert session.cleanup_after_execution(loop)
    assert loop.observation_paused == paused and loop.control_step == 127
    assert session.visual_only == (not paused)
    assert loop._rtc_previous_chunk is None and loop.executor.previous_command_q is None
    assert session.cleanup_records[-1]['all_history_empty']


def test_visual_fallback_cannot_unpause_an_observation_failure():
    from robot_vla.runtime.control_loop import QwenVLAReplanLoop
    from robot_vla.execution.chunk_executor import RecedingHorizonChunkExecutor
    from robot_vla.contracts import RobotSpec
    session,_ = setup_session(); session.visual_fallback = True
    loop = QwenVLAReplanLoop(SimpleNamespace(), RecedingHorizonChunkExecutor(RobotSpec()))
    pause = loop.pause_for_observation()
    session.interruption_reason = 'memory_stale'
    session.cleanup_after_execution(loop)
    assert loop._observation_pause is pause and not session.visual_only


@pytest.mark.parametrize('terminal', [False, True])
def test_zero_step_fallback_uses_real_hold_and_preserves_terminal(terminal):
    from experiments.memory_reobserve.controller import MemoryDevelopmentController
    session = SimpleNamespace(visual_only=True, runtime=SimpleNamespace(_last_timestamp=1.),
        cleanup_records=[], interruption_reason=None)
    controller = SimpleNamespace(session=session, frame=frame(20), tick=20,
        episode_done=False, chunk_stop_requested=True)
    loop = SimpleNamespace(observation_paused=False, control_step=20)
    def hold():
        controller.tick += 1
        controller.frame = frame(controller.tick)
        controller.episode_done = terminal
    controller.hold_current = hold
    if terminal:
        with pytest.raises(RuntimeError, match='禁止继续规划'):
            MemoryDevelopmentController.prepare_visual_replan(controller, loop)
        assert controller.chunk_stop_requested
        assert session.interruption_reason == 'episode-terminal-during-fresh-hold'
    else:
        MemoryDevelopmentController.prepare_visual_replan(controller, loop)
        assert not controller.chunk_stop_requested
    assert loop.control_step == controller.tick == 21
    assert controller.frame.timestamp_s == 1.05
