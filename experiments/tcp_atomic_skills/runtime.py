"""接触场景的 TCP 执行适配；GT 进度只用于评估，不进入学生输入。"""
from dataclasses import asdict
import json
import numpy as np
import torch

from experiments.tcp_atomic_skills.protocol import PROTOCOL, save, sha, identity
from experiments.tcp_memory_control.execution import TCPExecutionCandidate
from experiments.tcp_memory_control.geometry import TCPActionSpec
from experiments.tcp_memory_control.policy import build_policy
from experiments.tcp_memory_control.protocol import sampling_seed
from experiments.tcp_teacher_closed_loop.checkpoint import load_best, CONFIG_SHA, BEST_SHA
from experiments.g2c_memory_integration.vla import load_runtime
from experiments.memory_conditioning.conditioning import MEMORY_INPUT_KEY, MemoryBatch
from experiments.memory_reobserve.runtime import observation_digest
from robot_vla.adapters import FrankaObservationAdapter
from robot_vla.contracts import RobotSpec
from robot_vla.evaluation.maniskill import _read_online_observation, _read_predicate_state
from robot_vla.execution.maniskill_controller import ManiSkillFrankaController
from robot_vla.runtime.policy_runtime import _move_model_inputs
from experiments.tcp_atomic_skills.data import action_spec


class AtomicExecutor(TCPExecutionCandidate):
    """新任务只扩大TCP标签尺度；关节命令与IK物理检查保持原执行合同。"""
    def __init__(self, kinematics):
        super().__init__(kinematics)
        self.tcp_spec = action_spec()


def load_parent(args):
    base, upstream = load_runtime(args.checkpoint, args.model_cache)
    config_path = args.continued/'config.json'
    if sha(config_path) != CONFIG_SHA:
        raise ValueError('Reach续训配置身份改变')
    config = json.loads(config_path.read_text())
    prior = json.loads((args.parent_training/'result.json').read_text())
    ident = json.loads((args.parent_training/'identity.json').read_text())
    policy = build_policy(base.policy, 'tcp-relative').to('cuda')
    loaded = load_best(policy, args.continued/'best.pt', config, prior['identity_sha256'],
                       ident['source_manifest_sha256'])
    policy.eval()
    return base, policy, dict(parent=loaded, upstream=upstream)


class AtomicController(ManiSkillFrankaController):
    def __init__(self, env, preparation, target, limit, instruction, output):
        super().__init__(env, RobotSpec())
        self.observation = preparation.observation
        self.tracker, self.progress = preparation.tracker, preparation.progress
        self.initial_completed = self.progress.completed_skill_count
        self.preparation_steps = preparation.preparation_steps
        self.target, self.limit, self.instruction, self.output = target, limit, instruction, output
        self.steps = 0; self.chunk_stop_requested = False; self.stop_reason = None
        self.trace = []; self.completion_steps = {}; self.max_tracking = 0.

    def online(self):
        return _read_online_observation(self.observation, self.env.unwrapped,
                                        FrankaObservationAdapter(self.spec), self.instruction)

    def audit(self):
        return dict(observation_sha256=observation_digest(self.online()),
                    initial_completed=self.progress.completed_skill_count,
                    state=asdict(_read_predicate_state(self.env.unwrapped)),
                    preparation_steps=self.preparation_steps)

    def should_interrupt_before_action(self, value):
        return self.stop_reason is not None or self.steps >= self.limit

    def send_action(self, value):
        if self.should_interrupt_before_action(value):
            raise RuntimeError('终止后禁止继续动作')
        before = self.read_state().joint_positions.copy()
        command = before + np.asarray(value[:7])*self.spec.maniskill_arm_delta_range_rad
        super().send_action(value)
        self.observation, _, terminated, truncated, _ = self.last_step_output
        self.steps += 1
        after = self.read_state().joint_positions.copy()
        error = float(np.abs(command-after).max()); self.max_tracking = max(self.max_tracking, error)
        previous = self.progress.completed_skill_count
        # 接触、闭爪是任务正常动作；只在评估侧读取任务状态。
        state = _read_predicate_state(self.env.unwrapped)
        self.progress = self.tracker.update(state)
        for i in range(previous, self.progress.completed_skill_count):
            self.completion_steps[i] = self.steps
        if not np.isfinite(error) or error > PROTOCOL['tracking_limit_rad']:
            self.stop_reason = 'tracking-invalid'
        elif self.progress.completed_skill_count >= self.target:
            self.stop_reason = 'success'
        elif bool(terminated.item()) or bool(truncated.item()):
            self.stop_reason = 'environment-terminal'
        elif self.steps >= self.limit:
            self.stop_reason = 'step-budget-exhausted'
        self.chunk_stop_requested = self.stop_reason is not None
        row = dict(step=self.steps, q_before=before.tolist(), command_q=command.tolist(),
                   q_after=after.tolist(), gripper_target=float((value[-1]+1)/2),
                   gripper_actual=self.read_state().gripper_opening, tracking_error_rad=error,
                   completed=self.progress.completed_skill_count, state=asdict(state))
        self.trace.append(row)
        with (self.output/'control.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')

    def result(self):
        return dict(success=self.stop_reason == 'success', policy_steps=self.steps,
                    initial_completed=self.initial_completed, final_completed=self.progress.completed_skill_count,
                    target_completed=self.target, completion_steps=self.completion_steps,
                    stop_reason=self.stop_reason, tracking_error_max_rad=self.max_tracking,
                    preparation_steps=self.preparation_steps)


@torch.no_grad()
def predict(base, policy, online, seed, *, parent=False):
    encoded = base.processor_adapter.encode(online.rgb_external, online.rgb_wrist, online.instruction)
    inputs = _move_model_inputs(encoded.model_inputs, base.device)
    inputs[MEMORY_INPUT_KEY] = MemoryBatch(torch.zeros((1,12), device='cuda'),
                                          torch.zeros((1,1), dtype=torch.bool, device='cuda'))
    proprio = torch.tensor(base.proprio_normalizer.normalize(online.physical_proprio)[None], device='cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        normalized = policy.sample_actions(inputs, proprio, num_steps=10,
            generator=torch.Generator(device='cuda').manual_seed(seed))[0].float().cpu().numpy()
    return (TCPActionSpec() if parent else action_spec()).denormalize(normalized)


def execute_plan(executor, controller, physical, anchor):
    execution = executor.execute(physical, controller, anchor)
    if execution.correction_saturation_steps or execution.replan_required:
        controller.stop_reason = 'correction-saturation-or-anomaly'
    elif not execution.success or execution.executed_steps == 0:
        controller.stop_reason = controller.stop_reason or 'executor-failure-or-no-progress'
    return asdict(execution)
