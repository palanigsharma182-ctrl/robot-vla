from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("gymnasium")
pytest.importorskip("mani_skill")

import robot_vla.sim.collector as collector_module
from robot_vla.contracts import RobotSpec
from robot_vla.execution import ManiSkillFrankaController
from robot_vla.local_dagger_protocol import resolve_local_dagger_action_budget
from robot_vla.sim.collector import EpisodeRejected, TrustedPickPlaceCollector
from robot_vla.sim.local_dagger import (
    LocalDaggerPickPlaceCollector,
    _LocalDaggerPolicyController,
)


class _FakeActionAdapter:
    def normalize(self, action, *, strict):
        assert strict is True
        return action


class _FakeRecorder:
    def __init__(self, steps: int) -> None:
        self.action = [None] * steps

    def record_before_action(self, *args, **kwargs) -> None:
        self.action.append(args[1])

    def record_after_action(self, terminated, truncated, info) -> None:
        self.terminated = terminated
        self.truncated = truncated
        self.info = info


class _FakeCollector:
    def __init__(self, spec: RobotSpec) -> None:
        self.spec = spec
        self.action_adapter = _FakeActionAdapter()
        self.next_completed = 1

    def _actual_arm_q(self):
        return np.zeros(self.spec.arm_dof, dtype=np.float32)

    def _read_predicate_state(self):
        return SimpleNamespace(tcp_position=(0.0, 0.0, 0.0))

    def _read_contact_forces(self):
        return 0.0, 0.0


def _policy_controller_at_step_299(*, boundary_on_next_action: bool):
    spec = RobotSpec()
    collector = _FakeCollector(spec)
    before = SimpleNamespace(completed_skill_count=1, active_skill_id=1)
    after = SimpleNamespace(
        completed_skill_count=2 if boundary_on_next_action else 1,
        active_skill_id=1,
    )
    session = SimpleNamespace(
        observation={},
        progress=before,
        tracker=SimpleNamespace(update=lambda predicate: after),
        recorder=_FakeRecorder(299),
        previous_command_q=np.zeros(spec.arm_dof, dtype=np.float32),
        after_action_hook=None,
        done=False,
    )
    controller = object.__new__(_LocalDaggerPolicyController)
    controller.collector = collector
    controller.session = session
    controller.spec = spec
    controller.target_completed_skill_count = 2
    controller.action_budget = resolve_local_dagger_action_budget(
        "segmented-300-180-480"
    )
    controller.chunk_stop_requested = False
    controller.terminal_before_boundary = False
    controller.policy_budget_exhausted = False
    controller._last_tcp_position = np.zeros(3, dtype=np.float64)
    controller.last_tcp_linear_speed_m_s = 0.0
    return controller


def test_collector_only_overrides_environment_limit_for_amended_protocol(
    monkeypatch,
) -> None:
    calls = []

    def fake_init(self, dataset_root, spec=None, **kwargs) -> None:
        calls.append((dataset_root, spec, kwargs))

    monkeypatch.setattr(TrustedPickPlaceCollector, "__init__", fake_init)

    LocalDaggerPickPlaceCollector("/legacy")
    LocalDaggerPickPlaceCollector(
        "/amended",
        action_budget_protocol="segmented-300-180-480",
    )

    assert calls[0][2] == {}
    assert calls[1][2] == {"max_episode_steps": 480}


def test_gym_make_legacy_kwargs_remain_exact_and_amended_adds_480(
    monkeypatch,
) -> None:
    calls = []

    def fake_make(environment_id, **kwargs):
        calls.append((environment_id, kwargs))
        return SimpleNamespace(unwrapped=SimpleNamespace())

    monkeypatch.setattr(collector_module, "register_robot_vla_maniskill_envs", lambda: None)
    monkeypatch.setattr(collector_module.gym, "make", fake_make)

    TrustedPickPlaceCollector(None)
    TrustedPickPlaceCollector(None, max_episode_steps=480)

    assert calls[0][1] == {
        "obs_mode": "rgb+segmentation",
        "control_mode": "pd_joint_delta_pos",
        "num_envs": 1,
    }
    assert calls[1][1] == {
        **calls[0][1],
        "max_episode_steps": 480,
    }


def test_paired_clean_legacy_always_installs_boundary_capture_hook(
    monkeypatch,
) -> None:
    collector = object.__new__(LocalDaggerPickPlaceCollector)
    collector.spec = RobotSpec()
    recorder = SimpleNamespace(
        action=[],
        build=lambda: SimpleNamespace(num_steps=len(recorder.action)),
    )
    session = SimpleNamespace(
        progress=SimpleNamespace(completed_skill_count=0, task_completed=False),
        recorder=recorder,
        after_action_hook=None,
        done=False,
    )
    state = SimpleNamespace(tcp_position=(0.0, 0.0, 0.0))

    monkeypatch.setattr(
        LocalDaggerPickPlaceCollector,
        "_start_session",
        lambda self, seed: session,
    )
    monkeypatch.setattr(
        LocalDaggerPickPlaceCollector,
        "_read_predicate_state",
        lambda self: state,
    )
    monkeypatch.setattr(
        LocalDaggerPickPlaceCollector,
        "_phase_poses",
        lambda self: (object(), object(), object(), object(), object()),
    )
    monkeypatch.setattr(
        LocalDaggerPickPlaceCollector,
        "_clean_expert_boundary_diagnostics",
        lambda self, **kwargs: SimpleNamespace(control_step=kwargs["control_step"]),
    )

    move_completions = iter((1, 1, 3, 4, 4))

    def fake_move(self, current_session, pose, *, gripper_opening) -> None:
        assert current_session.after_action_hook is not None
        current_session.recorder.action.append(None)
        completed = next(move_completions)
        current_session.progress = SimpleNamespace(
            completed_skill_count=completed,
            task_completed=False,
        )
        current_session.after_action_hook(current_session, gripper_opening)

    def fake_hold(
        self,
        current_session,
        *,
        gripper_opening,
        steps,
        stop_on_success=False,
    ) -> None:
        assert current_session.after_action_hook is not None
        current_session.recorder.action.append(None)
        completed = 5 if stop_on_success else 2
        current_session.progress = SimpleNamespace(
            completed_skill_count=completed,
            task_completed=stop_on_success,
        )
        current_session.done = stop_on_success
        current_session.after_action_hook(current_session, gripper_opening)

    monkeypatch.setattr(LocalDaggerPickPlaceCollector, "_move_to_pose", fake_move)
    monkeypatch.setattr(LocalDaggerPickPlaceCollector, "_hold", fake_hold)

    result = collector.collect_clean_expert_boundary(
        seed=30_203,
        boundary_type="grasp_lift",
    )

    assert result.boundary.control_step == 3
    assert result.task_completed is True


@pytest.mark.parametrize(
    ("expert_actions", "terminated", "truncated", "rejected"),
    (
        (179, False, False, False),
        (180, False, False, True),
        (180, True, False, False),
        (180, False, True, False),
        (180, True, True, False),
    ),
)
def test_expert_budget_gate_counts_recorded_actions_and_preserves_terminal_priority(
    expert_actions: int,
    terminated: bool,
    truncated: bool,
    rejected: bool,
) -> None:
    collector = object.__new__(LocalDaggerPickPlaceCollector)
    collector.action_budget = resolve_local_dagger_action_budget(
        "segmented-300-180-480"
    )
    takeover = 100
    session = SimpleNamespace(
        recorder=SimpleNamespace(
            action=[None] * (takeover + expert_actions),
            terminated=[terminated],
            truncated=[truncated],
        )
    )

    if rejected:
        with pytest.raises(EpisodeRejected, match="180 Action"):
            collector._enforce_expert_action_budget_after_action(
                session,
                expert_takeover_step=takeover,
                failure_diagnostics=None,
            )
    else:
        collector._enforce_expert_action_budget_after_action(
            session,
            expert_takeover_step=takeover,
            failure_diagnostics=None,
        )


@pytest.mark.parametrize(
    ("boundary_on_next_action", "budget_exhausted"),
    ((False, True), (True, False)),
)
def test_policy_action_300_is_normal_chunk_stop_and_boundary_has_priority(
    monkeypatch,
    boundary_on_next_action: bool,
    budget_exhausted: bool,
) -> None:
    controller = _policy_controller_at_step_299(
        boundary_on_next_action=boundary_on_next_action
    )

    def fake_send_action(self, action) -> None:
        self.last_step_output = (
            {},
            0.0,
            np.asarray([False]),
            np.asarray([False]),
            {"success": np.asarray([False])},
        )

    monkeypatch.setattr(ManiSkillFrankaController, "send_action", fake_send_action)

    controller.send_action(np.zeros(controller.spec.action_dim, dtype=np.float32))

    assert len(controller.session.recorder.action) == 300
    assert controller.chunk_stop_requested is True
    assert controller.policy_budget_exhausted is budget_exhausted
