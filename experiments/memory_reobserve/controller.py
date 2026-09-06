"""真实控制步中断；实际命令跟踪与同帧相机外参作为请求证据。"""
import numpy as np

from experiments.g2c_memory_integration.run import DevelopmentController
from robot_vla.evaluation.maniskill import _read_observation_v2_frame, _single_transform_matrix
from robot_vla.precision.active_external_observation import extract_active_external_observation
from robot_vla.precision.active_front_camera import ExternalCameraMotionState
from robot_vla.precision.active_front_memory_provider import ACTIVE_FRONT_HOME_PRIMITIVE_ID as HOME
from robot_vla.precision.qualified_front_provider import _check_camera_geometry
from experiments.memory_reobserve.session import MemoryContextInvalid


class MemoryDevelopmentController(DevelopmentController):
    def __init__(self, env, spec, *, session, memory, initial_frame, initial_tick,
                 home_sidecar, home_constraints):
        super().__init__(env, spec)
        self.session, self.memory = session, memory
        self.frame, self.tick = initial_frame, initial_tick
        self.home_sidecar, self.home_constraints = home_sidecar, home_constraints
        self.trigger_enabled = True
        self.holding = False
        self.last_target = None
        self.tracking_records, self.camera_records = [], []

    def resume_frame(self, frame, tick):
        self.frame, self.tick = frame, tick
        self.trigger_enabled = False
        self.chunk_stop_requested = False
        self.session.interruption_reason = None
        # 对应实际通过 HOME hold 的新参考；不恢复暂停前的 commanded target。
        self.last_target = frame.physical_proprio[:7].copy()
        self.session.tracking_valid = True

    def _camera_home(self):
        obs = self.last_step_output[0]
        sidecar = extract_active_external_observation(obs, camera_uid='base_camera',
            world_from_robot_base=_single_transform_matrix(self.env.unwrapped.agent.robot.pose, 'base'),
            commanded_world_from_external_camera_gl=self.home_sidecar.commanded_world_from_external_camera_gl,
            episode_id=self.session.episode, request_id=self.home_sidecar.request_id,
            observation_sequence_id=f'trigger-frame-{self.tick}', camera_command_sequence_id='held-home',
            control_tick=self.tick, control_timestamp_s=self.frame.timestamp_s,
            rgb_timestamp_s=self.frame.timestamp_s, camera_pose_timestamp_s=self.frame.timestamp_s,
            camera_motion_state=ExternalCameraMotionState.HOME_ANCHOR, viewpoint_primitive_id=HOME,
            settled=True, maximum_rotation_projection_error_frobenius=1e-6)
        _check_camera_geometry(sidecar.intrinsic_cv, sidecar.base_from_external_camera_cv, *self.home_constraints)
        self.camera_records.append(dict(tick=self.tick, timestamp_s=self.frame.timestamp_s,
            digest=sidecar.audit_digest(), pose=sidecar.base_from_external_camera_cv.tolist(),
            actual_pose_source=sidecar.actual_pose_source))
        return True

    def send_action(self, value):
        if not self.holding and not self.session.visual_fallback:
            self.session.before_send(value, self.frame, self.memory)
        q_before = self.read_state().joint_positions.copy()
        correction = np.asarray(value[:7]) * self.spec.maniskill_arm_delta_range_rad
        target = q_before + correction
        limits = np.asarray(self.spec.effective_joint_delta_limits_rad)
        # 使用实际发出的目标；触及限幅边缘保守判为可能饱和，不伪造原始未裁剪目标。
        saturation = bool(np.any(np.abs(correction) >= limits - 1e-7))
        super().send_action(value)
        self.last_target = target
        self.tick += 1
        self.frame = _read_observation_v2_frame(self.last_step_output[0], self.env.unwrapped,
            self.observation_adapter, self.spec, control_step=self.tick)
        error = np.abs(target - self.frame.physical_proprio[:7])
        tracking = dict(valid=bool(np.isfinite(error).all() and np.all(error <= limits) and not saturation),
            tick=self.tick, timestamp_s=self.frame.timestamp_s,
            actual_q=self.frame.physical_proprio[:7].tolist(), sent_command_target=target.tolist(),
            absolute_error_rad=error.tolist(), limits_rad=limits.tolist(), possible_saturation=saturation)
        self.tracking_records.append(tracking)
        self.session.tracking_valid = tracking['valid']
        if self.holding:
            return
        if self.trigger_enabled:
            self.chunk_stop_requested |= self.session.observe_trigger(self.frame, self.tick,
                memory=self.memory, close_commanded=bool(value[-1] < .95),
                camera_home=self._camera_home(), tracking=tracking)
        self.chunk_stop_requested |= self.session.after_send(self.frame, value, self.memory)

    def should_interrupt_before_action(self, value):
        """在 executor 计步之前撤销失效动作；不发送替代动作或伪造时间。"""
        if not self.session.visual_fallback or self.holding:
            return False
        try:
            self.session.before_send(value, self.frame, self.memory)
        except MemoryContextInvalid:
            self.chunk_stop_requested = True
            return True
        return False

    def prepare_visual_replan(self, loop):
        """零步中断时实际 hold 一步取得新画面；两套时钟按实际步数同步。"""
        if self.session.visual_only and not loop.observation_paused:
            if self.frame.timestamp_s <= self.session.runtime._last_timestamp:
                self.hold_current()
                loop.control_step = self.tick
                self.session.cleanup_records.append(dict(
                    reason='fresh-frame-after-zero-step-interruption',
                    control_step=self.tick, hold_steps=1, next_mode='visual-only'))
            if self.episode_done:
                self.session.interruption_reason = 'episode-terminal-during-fresh-hold'
                self.chunk_stop_requested = True
                raise RuntimeError('取得新画面的 hold 已终止场景，禁止继续规划')
            self.chunk_stop_requested = False

    def hold_current(self):
        self.holding = True
        try:
            super().hold_current()
        finally:
            self.holding = False
