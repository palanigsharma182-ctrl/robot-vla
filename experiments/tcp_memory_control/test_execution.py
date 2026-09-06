"""检查0.1实际送达、越界拒绝、旧基线尺度及训练身份隔离。"""
import hashlib
import json
import numpy as np
import pytest

from robot_vla.adapters import ActionAdapter
from robot_vla.contracts import RobotSpec
from experiments.tcp_memory_control.geometry import TCPActionSpec
from experiments.tcp_memory_control.kinematics import TCPChunkExecutor
from experiments.tcp_memory_control.execution import (
    TCPExecutionCandidate, EXECUTION_SOURCE, compatible_training_source)
from experiments.tcp_memory_control.test_executor import Controller, Kinematics


class AmplifiedKinematics(Kinematics):
    def pose_base(self, q):
        p = np.eye(4); p[:3, 3] = q[:3] / 20
        return p

    def inverse(self, target, reference):
        q = reference.copy(); q[:3] = target[:3, 3] * 20
        return q


def chunk(joint_delta):
    value = np.zeros((16, 7), np.float32); value[:, 6] = 1
    value[:4, 0] = joint_delta / 20
    return value


@pytest.mark.parametrize('delta', [.08, .1])
def test_new_limit_reaches_controller_without_hidden_005_clip(delta):
    fk = AmplifiedKinematics(); c = Controller(); e = TCPExecutionCandidate(fk)
    result = e.execute(chunk(delta), c, fk.pose_base(c.q))
    assert result.success and result.executed_steps == 4
    assert result.correction_saturation_steps == 0 and not result.replan_required
    assert c.q[0] == pytest.approx(4 * delta)
    assert result.applied_correction_abs_max_rad == pytest.approx(delta)
    assert e.joint.spec.maniskill_arm_delta_range_rad == .1


def test_above_01_rejects_entire_prefix_without_actions():
    fk = AmplifiedKinematics(); c = Controller(); e = TCPExecutionCandidate(fk)
    with pytest.raises(ValueError, match='关节增量过大'):
        e.execute(chunk(.101), c, fk.pose_base(c.q))
    assert c.sent == 0


def test_old_executor_and_model_scales_unchanged():
    fk = AmplifiedKinematics(); c = Controller()
    with pytest.raises(ValueError, match='关节增量过大'):
        TCPChunkExecutor(fk).execute(chunk(.08), c, fk.pose_base(c.q))
    assert c.sent == 0
    joint = ActionAdapter(RobotSpec()).denormalize(np.ones((16, 8), np.float32))
    assert np.allclose(joint[:, :7], .05)
    tcp = TCPActionSpec().denormalize(np.ones((16, 7), np.float32))
    assert np.allclose(tcp[:, :3], .01) and np.allclose(tcp[:, 3:6], .1)


def manifests(tmp_path):
    old = {'experiments/tcp_memory_control/evaluate.py': 'old-eval',
           'experiments/tcp_memory_control/run.py': 'old-run',
           'experiments/tcp_memory_control/geometry.py': 'same-geometry'}
    new = {**old, 'experiments/tcp_memory_control/evaluate.py': 'new-eval',
           'experiments/tcp_memory_control/run.py': 'new-run',
           EXECUTION_SOURCE: 'new-execution'}
    a = tmp_path / 'old.json'; b = tmp_path / 'new.json'
    a.write_text(json.dumps(old)); b.write_text(json.dumps(new))
    return a, b, hashlib.sha256(a.read_bytes()).hexdigest()


def test_only_execution_changes_can_consume_original_training_identity(tmp_path):
    old, new, digest = manifests(tmp_path)
    assert compatible_training_source(new, old, digest) == digest
    value = json.loads(new.read_text()); value['experiments/tcp_memory_control/geometry.py'] = 'changed'
    new.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='执行入口以外'):
        compatible_training_source(new, old, digest)


@pytest.mark.parametrize('fault', ['missing-original', 'wrong-hash', 'missing-execution'])
def test_missing_or_mismatched_source_identity_rejected(tmp_path, fault):
    old, new, digest = manifests(tmp_path)
    if fault == 'missing-original': old = None
    if fault == 'wrong-hash': digest = '0' * 64
    if fault == 'missing-execution':
        value = json.loads(new.read_text()); del value[EXECUTION_SOURCE]
        new.write_text(json.dumps(value))
    with pytest.raises(ValueError): compatible_training_source(new, old, digest)


@pytest.mark.parametrize('fault', ['unexpected-file', 'removed-file', 'unchanged-entry'])
def test_source_difference_sets_are_bounded(tmp_path, fault):
    old, new, digest = manifests(tmp_path)
    value = json.loads(new.read_text())
    if fault == 'unexpected-file': value['unexpected.py'] = 'new'
    if fault == 'removed-file': del value['experiments/tcp_memory_control/geometry.py']
    if fault == 'unchanged-entry': value['experiments/tcp_memory_control/run.py'] = 'old-run'
    new.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='执行入口以外'):
        compatible_training_source(new, old, digest)


def test_expanded_comparison_keeps_joint_baseline_and_training_protocol():
    from experiments.tcp_memory_control.evaluate import evaluation_protocol
    from experiments.tcp_memory_control.protocol import PROTOCOL
    before = json.dumps(PROTOCOL, sort_keys=True)
    result = evaluation_protocol(True)
    assert result['arms'] == ['joint-world', 'tcp-relative']
    assert len(result['historical_seeds']) == 4
    assert len(result['new_development_seeds']) == 16
    assert not set(result['new_development_seeds']) & set(
        PROTOCOL['train_seeds'] + PROTOCOL['development_seeds'] + PROTOCOL['rollout_seeds'])
    assert json.dumps(PROTOCOL, sort_keys=True) == before
    assert evaluation_protocol(False)['new_development_seeds'] == []
