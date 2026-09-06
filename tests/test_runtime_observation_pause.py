"""暂停真实动作历史后的时序与重新推理回归，使用可检查的确定性策略。"""
from dataclasses import replace

import numpy as np
import pytest

from robot_vla.contracts import RobotSpec
from robot_vla.execution.chunk_executor import FrankaControlState, RecedingHorizonChunkExecutor
from robot_vla.observation import ObservationV2Frame, ObservationV2History
from robot_vla.runtime.control_loop import QwenVLAReplanLoop
from robot_vla.runtime.policy_runtime import (
    OnlineObservation, QwenVLAObservationV2Runtime, RuntimeActionChunk, SamplingTrace,
)


class Controller:
    def __init__(self):
        self.q = np.array([0, -.5, 0, -1.5, 0, 1.5, 0], np.float32)
        self.actions = []
        self.holds = 0

    def read_state(self):
        return FrankaControlState(self.q.copy(), 1.)

    def send_action(self, action):
        self.actions.append(action.copy())

    def hold_current(self):
        self.holds += 1


class Runtime:
    def __init__(self):
        self.calls = []
        self._last_sampling_trace = None
        self.fail = False

    @property
    def last_sampling_trace(self):
        return self._last_sampling_trace

    def infer_action_chunk(self, observation, **kwargs):
        self.calls.append((observation, kwargs))
        if self.fail:
            raise RuntimeError('inference fault')
        self._last_sampling_trace = SamplingTrace(42 + len(self.calls), len(self.calls) - 1)
        action = np.zeros((16, 8), np.float32)
        action[:, 0] = .001 * len(self.calls)
        action[:, -1] = 1
        return RuntimeActionChunk(action, action.copy(), (1, 1), 2, self.last_sampling_trace)


def window(start=20, count=4):
    history = ObservationV2History(RobotSpec())
    for step in range(start, start + count):
        t = step / 20
        history.append(ObservationV2Frame(
            np.full((8, 8, 3), step, np.uint8), np.full((8, 8, 3), step, np.uint8),
            np.r_[Controller().q, np.zeros(7), 1].astype(np.float32),
            np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32),
            np.zeros(2, np.float32), t, np.full(6, t, np.float64), np.ones(6, bool),
        ))
    return history.snapshot('pick the cube', previous_command_q=None, previous_action=None)


def setup(strategy='temporal-ensemble'):
    runtime, controller = Runtime(), Controller()
    loop = QwenVLAReplanLoop(runtime, RecedingHorizonChunkExecutor(RobotSpec()),
                            inference_strategy=strategy)
    result = loop.replan_and_execute(object(), controller)
    assert result.execution.success and result.execution.executed_steps == 4
    return loop, runtime, controller


@pytest.mark.parametrize('strategy', ['temporal-ensemble', 'rtc', 'newest-only'])
def test_nonempty_history_pause_blocks_inference_then_uses_fresh_observation(strategy):
    loop, runtime, controller = setup(strategy)
    pause = loop.pause_for_observation()
    assert pause.control_step == 4 and pause.command_reference_present
    assert pause.ensemble_chunks == (1 if strategy == 'temporal-ensemble' else 0)
    assert pause.rtc_chunk_present == (strategy == 'rtc')
    assert loop.control_step == 4 and loop.executor.previous_command_q is None
    with pytest.raises(RuntimeError, match='暂停期间'):
        loop.replan_and_execute(object(), controller)
    assert len(runtime.calls) == 1 and len(controller.actions) == 4
    result = loop.resume_after_observation(pause, window(), controller)
    assert result.execution.success and result.execution.executed_steps == 4
    assert loop.control_step == 27 and not loop.observation_paused
    assert result.sampling.sample_index == 1
    obs, kwargs = runtime.calls[-1]
    assert isinstance(obs, OnlineObservation) and (obs.rgb_external == 23).all()
    if strategy == 'rtc':
        assert kwargs['rtc_previous_overlap'] is None
    if strategy == 'temporal-ensemble':
        assert result.ensemble_trace.buffer_size == 1
    with pytest.raises(RuntimeError, match='身份'):
        loop.resume_after_observation(pause, window(), controller)


@pytest.mark.parametrize('invalid', ['padding', 'old', 'controller', 'skew'])
def test_invalid_home_window_cannot_unpause_or_send_actions(invalid):
    loop, runtime, controller = setup()
    pause = loop.pause_for_observation()
    w = window()
    if invalid == 'padding':
        w = window(count=3)
    elif invalid == 'old':
        w = window(start=1)
    elif invalid == 'controller':
        w = replace(w, controller_valid=np.ones(2, bool))
    else:
        times = w.modality_timestamp_s.copy(); times[:, 0] -= .001
        ages = w.modality_age_s.copy(); ages[:, 0] += .001
        w = replace(w, modality_timestamp_s=times, modality_age_s=ages)
    with pytest.raises((ValueError, RuntimeError)):
        loop.resume_after_observation(pause, w, controller)
    assert loop.observation_paused and len(runtime.calls) == 1 and len(controller.actions) == 4


def test_reset_invalidates_old_pause_identity():
    loop, _, controller = setup()
    pause = loop.pause_for_observation()
    with pytest.raises(RuntimeError, match='重复'):
        loop.pause_for_observation()
    loop.reset()
    with pytest.raises(RuntimeError, match='身份'):
        loop.resume_after_observation(pause, window(), controller)


def test_resume_inference_failure_stays_paused_without_old_actions():
    loop, runtime, controller = setup()
    pause = loop.pause_for_observation(); runtime.fail = True
    result = loop.resume_after_observation(pause, window(), controller)
    assert result.execution.replan_required
    assert loop.observation_paused and controller.holds == 1 and len(controller.actions) == 4
    assert loop.ensembler.buffer_size == 0 and loop.executor.previous_command_q is None


def test_v2_runtime_receives_full_home_window_without_v1_conversion():
    class V2(Runtime, QwenVLAObservationV2Runtime):
        pass
    loop, _, controller = setup()
    runtime = V2(); loop.runtime = runtime
    pause = loop.pause_for_observation(); w = window()
    result = loop.resume_after_observation(pause, w, controller)
    assert result.execution.success and runtime.calls[-1][0] is w


def test_mid_chunk_stop_reports_exact_boundary_before_history_is_cleared():
    class InterruptController(Controller):
        @property
        def chunk_stop_requested(self):
            return len(self.actions) == 2
    controller = InterruptController()
    runtime = Runtime()
    loop = QwenVLAReplanLoop(runtime, RecedingHorizonChunkExecutor(RobotSpec()))
    result = loop.replan_and_execute(object(), controller)
    assert result.execution.interrupted and result.execution.executed_steps == 2
    pause = loop.pause_for_observation()
    assert pause.control_step == 2 and pause.command_reference_present
    assert len(controller.actions) == 2


def test_resume_reanchors_new_delta_at_actual_q_instead_of_old_command():
    loop, _, controller = setup('newest-only')
    old_command = loop.executor.previous_command_q.copy()
    pause = loop.pause_for_observation()
    controller.q[0] += .02
    fresh = Controller(); fresh.q = controller.q.copy()
    result = loop.resume_after_observation(pause, window(), controller)
    expected = RecedingHorizonChunkExecutor(RobotSpec())
    expected.execute(result.action_chunk.physical_action, fresh)
    assert not np.allclose(controller.q, old_command)
    np.testing.assert_allclose(controller.actions[4], fresh.actions[0])


def test_failed_resume_token_cannot_retry_same_stale_home_frames():
    loop, runtime, controller = setup()
    pause = loop.pause_for_observation(); runtime.fail = True
    loop.resume_after_observation(pause, window(), controller)
    runtime.fail = False
    with pytest.raises(RuntimeError, match='身份'):
        loop.resume_after_observation(pause, window(), controller)
    assert len(runtime.calls) == 2 and len(controller.actions) == 4


@pytest.mark.parametrize('terminal_kind', ['terminated', 'truncated'])
@pytest.mark.parametrize('pause_after_steps', [None, 2])
def test_development_controllers_stop_on_first_terminal_step(terminal_kind, pause_after_steps):
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace
    path = Path(__file__).resolve().parents[1] / 'experiments/g2c_memory_integration/run.py'
    spec = importlib.util.spec_from_file_location('g2c_run_terminal_test', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    class Env:
        action_space = SimpleNamespace(low=np.full(8, -1.), high=np.full(8, 1.))
        calls = 0
        def step(self, action):
            self.calls += 1
            assert self.calls == 1, 'terminal 后不允许第二个动作'
            return {}, 0., np.array(terminal_kind == 'terminated'), np.array(terminal_kind == 'truncated'), {}
    env = Env()
    controller = module.DevelopmentController(env, RobotSpec(), pause_after_steps=pause_after_steps)
    controller.read_state = Controller().read_state
    loop = QwenVLAReplanLoop(Runtime(), RecedingHorizonChunkExecutor(RobotSpec()))
    result = loop.replan_and_execute(object(), controller)
    assert result.execution.interrupted and result.execution.executed_steps == 1
    assert controller.episode_done and env.calls == 1
    with pytest.raises(RuntimeError, match='已终止'):
        controller.send_action(np.zeros(8, np.float32))
    assert env.calls == 1
