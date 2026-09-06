"""RGB-D在实际动作后更新Memory；训练采集与在线执行共用同一个消费者。"""
from dataclasses import asdict, replace

import cv2
import numpy as np

from experiments.front_rgbd_memory.geometry import measure
from experiments.front_rgbd_memory.memory import MemoryReplay
from experiments.front_rgbd_memory.runner import ENV_ID, VIEWS, array
from experiments.g2c_memory_integration.run import DevelopmentController
from experiments.memory_conditioning.conditioning import snapshot_memory
from experiments.memory_reobserve.session import frame_safety
from experiments.memory_reobserve.runtime import observation_digest
from experiments.rgbd_memory_policy.protocol import PROTOCOL, occluded
from robot_vla.adapters import FrankaObservationAdapter
from robot_vla.contracts import RobotSpec
from robot_vla.evaluation.maniskill import _read_observation_v2_frame
from robot_vla.runtime.policy_runtime import OnlineObservation

INSTRUCTION = 'move the open gripper to a point 8 cm above the cube'


def make_env():
    import gymnasium as gym
    return gym.make(ENV_ID, num_envs=1, obs_mode='rgb+depth+segmentation',
        control_mode='pd_joint_delta_pos', sim_backend='cpu', render_backend='gpu',
        sensor_configs={'base_camera':dict(width=640,height=480,fov=np.pi/3)})


def setup_scene(env, seed):
    """场景生成参数不传给测量器或策略；训练/开发/执行种子分别预先固定。"""
    import sapien
    from scipy.spatial.transform import Rotation
    env.reset(seed=seed)
    rng=np.random.default_rng(seed)
    position=np.array([rng.uniform(-.09,.09),rng.uniform(-.07,.07),(.05,.12,.19,.26)[seed%4]])
    xyzw=Rotation.from_euler('xyz',[0.,rng.choice([-10.,0.,10.]),rng.choice([15.,45.,75.])],degrees=True).as_quat()
    base=env.unwrapped
    base.cube.set_pose(sapien.Pose(p=position,q=xyzw[[3,0,1,2]]))
    base.blocker.set_pose(sapien.Pose(p=[0,0,-10]))
    return position


class RGBDController(DevelopmentController):
    def __init__(self,env,episode,output):
        super().__init__(env,RobotSpec())
        self.episode,self.output=episode,output
        self.replay=MemoryReplay(episode)
        self.tick=0; self.policy_step=0; self.executing=False
        self.frame=None; self.snapshot=None; self.bound_snapshot=None
        self.closed=False; self.rows=[]; self.interruptions=[];self.holding=False
        self.view=0; self.fixture_position=None; self.current_occlusion=False
        self.last_target=self.read_state().joint_positions.copy()

    @property
    def stop_reason(self):
        if self.episode_done:return 'episode-terminal'
        if self.closed:return 'contact-or-gripper-close'
        if self.rows and not self.rows[-1]['tracking_valid']:return 'tracking-invalid'
        return None

    def set_view(self,index):
        import sapien
        from mani_skill.utils import sapien_utils
        self.view=index;self.replay.begin_view()
        pose=sapien_utils.look_at(VIEWS[index],[-.1,0.,.1])
        self.env.unwrapped._sensors['base_camera'].camera.set_local_pose(
            sapien.Pose(p=array(pose.p)[0],q=array(pose.q)[0]))

    def hold_current(self):
        self.holding=True
        try:super().hold_current()
        finally:self.holding=False

    def _set_occluder(self):
        import sapien
        enabled=occluded(self.policy_step) if self.executing else False
        # 遮挡位置仅属于事先确定的场景生成，不进入模型或观察策略。
        p=(np.asarray(VIEWS[self.view])+self.fixture_position)/2 if enabled else [0,0,-10]
        self.env.unwrapped.blocker.set_pose(sapien.Pose(p=p))
        self.current_occlusion=enabled

    def send_action(self,value):
        before=self.read_state().joint_positions.copy()
        target=before+np.asarray(value[:7])*self.spec.maniskill_arm_delta_range_rad
        self._set_occluder()
        super().send_action(value)
        self.tick+=1
        if self.executing:self.policy_step+=1
        self.last_target=target.copy()
        self.frame=_read_observation_v2_frame(self.last_step_output[0],self.env.unwrapped,
            FrankaObservationAdapter(self.spec),self.spec,control_step=self.tick)
        error=np.abs(target-self.frame.physical_proprio[:7])
        tracking=bool(np.isfinite(error).all() and np.all(error<=np.asarray(self.spec.effective_joint_delta_limits_rad)))
        self.closed |= bool(value[-1]<.95 or np.max(self.frame.finger_force_n)>.01)
        safe=frame_safety(self.frame,close_commanded=self.closed,tracking_valid=tracking)
        if self.episode_done:safe=replace(safe,pregrasp_window_open=False)
        if self.episode_done or safe.invalidation_reasons:
            reasons=tuple(safe.invalidation_reasons) or ('episode_terminal',)
            self.replay.memory.invalidate_for_safety(episode_id=self.episode,timestamp_s=self.frame.timestamp_s,reasons=reasons)
            self.chunk_stop_requested=True
        sensor=self.last_step_output[0]['sensor_data']['base_camera']
        params=self.last_step_output[0]['sensor_param']['base_camera']
        depth=array(sensor['depth'])[0,...,0]
        intrinsic=array(params['intrinsic_cv'])[0].astype(np.float64)
        world_from_base=array(self.env.unwrapped.agent.robot.pose.to_transformation_matrix())[0]
        transform=np.linalg.inv(world_from_base)@array(params['cam2world_gl'])[0]@np.diag([1.,-1.,-1.,1.])
        estimate,_=measure(self.frame.rgb_external,depth,intrinsic,transform)
        result=self.replay.update(estimate,timestamp=self.frame.timestamp_s,safety=safe)
        self.snapshot=snapshot_memory(self.replay.memory.state,self.replay.config,safe,
            episode_id=self.episode,timestamp_s=self.frame.timestamp_s)
        row=dict(tick=self.tick,policy_step=self.policy_step,view=self.view,occluded=self.current_occlusion,
            measurement_valid=estimate.valid,measurement_reason=estimate.reason,memory=result,
            snapshot=asdict(self.snapshot),tracking_valid=tracking,closed=self.closed,
            timestamp_s=self.frame.timestamp_s,
            input_digest=observation_digest(self.online()),
            action_used_memory=bool(self.bound_snapshot is not None and self.bound_snapshot.available and not self.holding),
            action_origin='hold' if self.holding else 'policy' if self.bound_snapshot is not None else 'teacher',
            action_snapshot_timestamp_s=None if self.bound_snapshot is None else self.bound_snapshot.timestamp_s)
        self.rows.append(row)
        if self.bound_snapshot is not None and self.bound_snapshot.available and not self.snapshot.available:
            self.chunk_stop_requested=True
            self.interruptions.append(dict(tick=self.tick,reasons=list(self.snapshot.reasons)))
        if self.bound_snapshot is not None and self.policy_step>=PROTOCOL['policy_steps']:
            self.chunk_stop_requested=True
        if self.output is not None:
            import json
            with (self.output/'observations.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            if (self.output/'frames').is_dir():
                np.savez_compressed(self.output/'frames'/f'{self.tick:04d}.npz',
                    rgb=self.frame.rgb_external,depth_mm=depth,intrinsic_cv=intrinsic,
                    base_from_camera_cv=transform,timestamp_s=np.float64(self.frame.timestamp_s))

    def warmup(self,fixture_position):
        self.fixture_position=fixture_position
        # 两臂使用相同预定观察：HOME→第二视角→HOME；无需GT挑选成功视角。
        for view in (0,1,0):
            self.set_view(view)
            for _ in range(3):
                self.hold_current()
                if self.stop_reason is not None:
                    raise RuntimeError('观察hold停止: '+self.stop_reason)
        self.executing=True;self.chunk_stop_requested=False

    def online(self):
        # Qwen仍输入128×128双图；高分辨率仅供几何，训练/执行使用相同resize。
        return OnlineObservation(cv2.resize(self.frame.rgb_external,(128,128),interpolation=cv2.INTER_AREA),
            self.frame.rgb_wrist.copy(),self.frame.physical_proprio.copy(),INSTRUCTION)

    def bind(self,enabled):
        snap=self.snapshot if enabled else replace(self.snapshot,features=(0.,)*12,available=False,
            reasons=('ablation_memory_masked',))
        self.bound_snapshot=snap
        self.chunk_stop_requested=False
        return snap

    def should_interrupt_before_action(self,value):
        if self.closed or self.episode_done:return True
        return bool(self.bound_snapshot is not None and self.bound_snapshot.available and not self.snapshot.available)
