"""用户批准的0.1 rad执行候选；保留训练动作尺度和原0.05执行器。"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from robot_vla.contracts import RobotSpec
from robot_vla.execution.chunk_executor import RecedingHorizonChunkExecutor
from experiments.tcp_memory_control.kinematics import TCPChunkExecutor


EXECUTION_CONFIG = {
    'schema': 'tcp-memory-execution-limit/v2',
    'tcp_joint_delta_limit_rad': 0.1,
    'joint_baseline_delta_limit_rad': 0.05,
    'control_hz': 20,
    'model_action_scale_changed': False,
}
EXECUTION_SOURCE = 'experiments/tcp_memory_control/execution.py'
# 仅允许已审查的执行入口变化；FK/IK、训练、模型、标签及几何均必须相同。
EXECUTION_ONLY_CHANGES = frozenset({
    'experiments/tcp_memory_control/evaluate.py',
    'experiments/tcp_memory_control/run.py',
})


class TCPExecutionCandidate(TCPChunkExecutor):
    """规划前缀校验与actual跟踪修正共用0.1上限，控制器映射仍为±0.1。"""
    def __init__(self, kinematics):
        super().__init__(kinematics)
        spec = replace(RobotSpec(), joint_delta_limit_rad=(0.1,) * 7)
        self.joint = RecedingHorizonChunkExecutor(spec)


def require_execution_source(manifest_path):
    """现有verify_source校验全部hash；这里补充新执行模块的必需覆盖。"""
    entries = json.loads(Path(manifest_path).read_text())
    if EXECUTION_SOURCE not in entries:
        raise ValueError('源码manifest缺少0.1 rad执行模块')
    return entries


def compatible_training_source(current_manifest, training_manifest, expected_hash):
    """旧权重保持原训练身份；只允许显式记录的执行入口差异，不跳过权重校验。

    调用前必须由verify_source验证当前文件；旧manifest通过训练身份中的SHA核验。
    """
    current = require_execution_source(current_manifest)
    current_hash = hashlib.sha256(Path(current_manifest).read_bytes()).hexdigest()
    if current_hash == expected_hash:
        return expected_hash
    if training_manifest is None:
        raise ValueError('消费旧权重需提供原训练源码manifest')
    raw = Path(training_manifest).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ValueError('原训练源码manifest身份不符')
    previous = json.loads(raw)
    added = set(current) - set(previous)
    removed = set(previous) - set(current)
    changed = {name for name in set(current) & set(previous)
               if current[name] != previous[name]}
    allowed_added = {EXECUTION_SOURCE, 'experiments/tcp_memory_control/test_execution.py'}
    if (removed or changed != EXECUTION_ONLY_CHANGES or EXECUTION_SOURCE not in added
        or not added.issubset(allowed_added)):
        raise ValueError('存在执行入口以外的源码变化，不能直接复用旧权重')
    return expected_hash
