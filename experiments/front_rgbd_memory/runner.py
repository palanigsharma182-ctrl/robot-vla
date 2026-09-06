"""高分辨率front RGB-D测量台架：多高度静止方块、相机位移、遮挡与Memory读取。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import signal
import sys
import time

import gymnasium as gym
import numpy as np
import sapien
from scipy.spatial.transform import Rotation

from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from robot_vla.sim.pick_cube_to_region import RobotVLAPickCubeToRegionEnv
from experiments.front_rgbd_memory.geometry import measure
from experiments.front_rgbd_memory.memory import MemoryReplay,tracking_safety

ENV_ID='FrontRGBDStaticCubeDiagnostic-v0'
HEIGHTS=(.05,.12,.19,.26)
VIEWS=((.3,0.,.6),(.3,-.08,.6))


def array(x):
    return x.detach().cpu().numpy() if hasattr(x,'detach') else np.asarray(x)


@register_env(ENV_ID,max_episode_steps=200)
class FixtureEnv(RobotVLAPickCubeToRegionEnv):
    """运动学、无碰撞目标是测量夹具，不冒充悬空动态操纵或原任务成功。"""
    def _load_scene(self,options):
        self.table_scene=TableSceneBuilder(self,robot_init_qpos_noise=0.)
        self.table_scene.build()
        self.cube=actors.build_cube(self.scene,half_size=.02,color=[1,0,0,1],name='cube',
                                    body_type='kinematic',add_collision=False,initial_pose=sapien.Pose(p=[0,0,.05]))
        self.goal_site=actors.build_sphere(self.scene,radius=self.goal_thresh,color=[0,1,0,1],name='goal_site',
                                          body_type='kinematic',add_collision=False,initial_pose=sapien.Pose(p=[0,0,-10]))
        self._hidden_objects.append(self.goal_site)
        self.blocker=actors.build_cube(self.scene,half_size=.12,color=[.05,.05,.05,1],name='occluder',
                                       body_type='kinematic',add_collision=False,initial_pose=sapien.Pose(p=[0,0,-10]))


def scenario(index):
    height=HEIGHTS[index//6]
    x=(-.08,.08)[(index//3)%2]
    yaw=(15.,45.,75.)[index%3]
    # 输入仅用于场景生成与独立评估；measure()不接收这些参数。
    position=np.array([x,(-.07,.07)[index%2],height])
    xyzw=Rotation.from_euler('xyz',[0.,(-10.,0.,10.)[index%3],yaw],degrees=True).as_quat()
    return position,xyzw[[3,0,1,2]]


def statistics(records):
    accepted=[r for r in records if r['measurement_valid']]
    values=np.asarray([r['error_xyz_m'] for r in accepted])
    errors=np.linalg.norm(values,axis=1) if len(values) else np.array([])
    return dict(frames=len(records),valid=len(accepted),coverage=len(accepted)/len(records),
                median_3d_m=float(np.median(errors)) if len(errors) else None,
                p90_3d_m=float(np.percentile(errors,90)) if len(errors) else None,
                p90_abs_z_m=float(np.percentile(np.abs(values[:,2]),90)) if len(values) else None,
                within_1cm_full_denominator=int((errors<.01).sum())/len(records),
                rejection_counts={reason:sum(r['reason']==reason for r in records) for reason in sorted({r['reason'] for r in records if not r['measurement_valid']})})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'frames').mkdir()
    started=time.monotonic();rows=[];episodes=[]
    scene_status=[dict(scene=1400000+i,status='not_run') for i in range(24)]
    def save_status():
        (args.output/'scene-status.json').write_text(json.dumps(dict(planned=24,scenes=scene_status),indent=2)+'\n')
    save_status()
    def stop_request(_signum,_frame):
        raise TimeoutError('收到停止请求，保存部分执行状态')
    signal.signal(signal.SIGTERM,stop_request)
    env=gym.make(ENV_ID,num_envs=1,obs_mode='rgb+depth+segmentation',control_mode='pd_joint_delta_pos',
                 sim_backend='cpu',render_backend='gpu',sensor_configs={'base_camera':dict(width=640,height=480,fov=np.pi/3)})
    # 固定hold仅推进真实仿真/传感器时钟；没有VLA、规划动作或操作任务评估。
    hold=np.zeros(8,np.float32);hold[-1]=1.
    try:
        for index in range(24):
            seed=1400000+index;episode=f'front-rgbd-{seed}'
            scene_status[index]['status']='running';save_status()
            env.reset(seed=seed)
            base=env.unwrapped
            object_position,orientation=scenario(index)
            base.cube.set_pose(sapien.Pose(p=object_position,q=orientation))
            base.blocker.set_pose(sapien.Pose(p=[0,0,-10]))
            replay=MemoryReplay(episode);tick=0;episode_rows=[]
            reference_q=array(base.agent.robot.get_qpos())[0,:7].copy()
            reference_tcp=array(base.agent.tcp_pose.p)[0].copy()

            def check_fixture(terminated,truncated):
                safety,tracking=tracking_safety(array(base.agent.robot.get_qpos())[0],array(base.agent.tcp_pose.p)[0],reference_q,reference_tcp)
                reasons=list(safety.invalidation_reasons)
                if bool(array(terminated).any()) or bool(array(truncated).any()):
                    reasons.append('pregrasp_window_closed')
                if reasons:
                    replay.memory.invalidate_for_safety(episode_id=episode,timestamp_s=tick/20.,reasons=tuple(reasons))
                    with (args.output/'fixture-errors.jsonl').open('a') as f:
                        f.write(json.dumps(dict(scene=seed,tick=tick,reasons=reasons,tracking=tracking))+'\n')
                    raise RuntimeError('测量台架终止或状态越界: '+','.join(reasons))
                return safety,tracking

            def capture(phase,view):
                nonlocal tick
                obs,_,terminated,truncated,_=env.step(hold);tick+=1
                safety,tracking=check_fixture(terminated,truncated)
                timestamp=tick/20.
                sensor=obs['sensor_data']['base_camera'];params=obs['sensor_param']['base_camera']
                rgb=array(sensor['rgb'])[0];depth=array(sensor['depth'])[0,...,0]
                assert rgb.shape==(480,640,3) and rgb.dtype==np.uint8
                assert array(obs['sensor_data']['hand_camera']['rgb']).shape[-3:]==(128,128,3)
                intrinsic=array(params['intrinsic_cv'])[0].astype(np.float64)
                world_from_camera_gl=array(params['cam2world_gl'])[0].astype(np.float64)
                world_from_base=array(base.agent.robot.pose.to_transformation_matrix())[0].astype(np.float64)
                transform=np.linalg.inv(world_from_base)@world_from_camera_gl@np.diag([1.,-1.,-1.,1.])
                estimate,mask=measure(rgb,depth,intrinsic,transform)
                memory=replay.update(estimate,timestamp=timestamp,safety=safety)
                # 直到测量和Memory更新完，才读取GT，严格留在审计侧。
                gt_world=array(base.cube.pose.p)[0].astype(np.float64)
                gt=(np.linalg.inv(world_from_base)@np.r_[gt_world,1.])[:3]
                segmentation=array(sensor['segmentation'])[0,...,0]
                actor_id=int(array(base.cube.per_scene_id).reshape(-1)[0])
                pixels=segmentation==actor_id
                sample_id=f'{seed}-{tick:03d}'
                np.savez_compressed(args.output/'frames'/(sample_id+'.npz'),rgb=rgb,depth_mm=depth,
                                    intrinsic_cv=intrinsic,base_from_camera_cv=transform)
                row=dict(sample_id=sample_id,scene=seed,episode=episode,tick=tick,timestamp_s=timestamp,
                         phase=phase,view=view,height_condition_m=float(object_position[2]),
                         measurement_valid=estimate.valid,reason=estimate.reason,diagnostics=estimate.diagnostics,
                         position_base_m=None if estimate.position is None else estimate.position.tolist(),
                         covariance_base_m2=None if estimate.covariance is None else estimate.covariance.tolist(),
                         error_xyz_m=None if estimate.position is None else (estimate.position-gt).tolist(),
                         gt_position_base_m=gt.tolist(),gt_visible_pixels=int(pixels.sum()),
                         mask_purity=float(pixels[mask].mean()) if mask.any() else None,memory=memory,
                         gripper_open=safety.gripper_open,depth_dtype=str(depth.dtype),depth_unit='mm',
                         **tracking,
                         rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),depth_sha256=hashlib.sha256(depth.tobytes()).hexdigest())
                rows.append(row);episode_rows.append(row)
                with (args.output/'records.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')

            for view,eye in enumerate(VIEWS):
                replay.begin_view()
                pose=sapien_utils.look_at(eye,[-.1,0.,.1])
                base._sensors['base_camera'].camera.set_local_pose(sapien.Pose(p=array(pose.p)[0],q=array(pose.q)[0]))
                for _ in range(3):capture('visible',view)
            # 遮挡夹具的位置属于场景生成，不作为估计器输入或候选视角选择。
            blocker_position=(np.asarray(VIEWS[-1])+object_position)/2
            base.blocker.set_pose(sapien.Pose(p=blocker_position))
            capture('occluded_recent',1)
            # 实际推进52个控制tick，不能只改Memory时间戳冒充物理等待。
            for _ in range(51):
                _,_,terminated,truncated,_=env.step(hold);tick+=1
                check_fixture(terminated,truncated)
            capture('occluded_expired',1)
            episodes.append(dict(scene=seed,commits=sum(r['memory']['accepted'] for r in episode_rows),
                                 snapshot_reads=sum(r['memory']['available'] for r in episode_rows),
                                 held_after_occlusion=episode_rows[-2]['memory']['available'],
                                 unavailable_after_expiry=not episode_rows[-1]['memory']['available'],
                                 elapsed_control_steps=tick))
            print(json.dumps(episodes[-1]),flush=True)
            scene_status[index].update(status='completed',commits=episodes[-1]['commits']);save_status()
    finally:
        error=sys.exc_info()[1]
        for item in scene_status:
            if item['status']=='running':item.update(status='error',error=str(error))
        save_status()
        env.close()
    assert len(rows)==192 and len(episodes)==24
    visible=[r for r in rows if r['phase']=='visible']
    summary=dict(status='completed-offline-measurement-and-memory-readout',scenes=24,frames=len(rows),
                 visible=statistics(visible),by_height={str(h):statistics([r for r in visible if r['height_condition_m']==h]) for h in HEIGHTS},
                 by_view={str(v):statistics([r for r in visible if r['view']==v]) for v in range(2)},
                 occluded=statistics([r for r in rows if r['phase']!='visible']),episodes=episodes,
                 initialized_episodes=sum(r['commits']>0 for r in episodes),
                 elapsed_s=time.monotonic()-started,geometry_scope='known 4cm cube; three visible orthogonal faces; arbitrary xyz and predefined rotations',
                 depth_scope='ideal rendered depth, perfect simulator calibration; not real RGB-D qualification',
                 control_scope='kinematic no-collision fixture, fixed robot hold only; no VLA/Action Expert',
                 memory_config=asdict(replay.config))
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
