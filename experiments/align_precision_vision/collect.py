"""现有ManiSkill场景中的感知数据采集；GT只写监督，不生成BC动作标签。"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from experiments.align_precision_vision.labels import corner_labels
from experiments.align_precision_vision.train import digest


def numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def matrix(value):
    return numpy(value.to_transformation_matrix())[0].astype(np.float64)


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n')


def collect(output, *, train_scenes=12, development_scenes=4, seed_start=73000,
            width=320, height=240, stride=8, wall_seconds=900):
    """只做开放夹爪接近/对齐与预定偏移；图像来自实际仿真步进。"""
    import gymnasium as gym
    import sapien
    from scipy.spatial.transform import Rotation
    from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
    from robot_vla.sim import register_robot_vla_maniskill_envs, PICK_CUBE_TO_REGION_ENV_ID
    from robot_vla.observation import opengl_camera_to_opencv
    if min(train_scenes, development_scenes, stride, width, height, wall_seconds) <= 0:
        raise ValueError('采集数量、尺寸和预算必须为正')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    (output/'samples').mkdir();(output/'audit').mkdir()
    started=time.monotonic()
    plan=[dict(seed=seed_start+i,split='train' if i<train_scenes else 'development',status='not_run')
          for i in range(train_scenes+development_scenes)]
    protocol=dict(schema='align-precision-collection/v1',environment=PICK_CUBE_TO_REGION_ENV_ID,
        resolution=[width,height],cameras=['base_camera','hand_camera'],stride=stride,
        label_visibility='facing surface + same-object depth 3x3 + 1mm face probe; tolerance 3mm',
        action_use='perception-only; pd_joint_pos motion-planner commands are not project BC labels',
        waypoints='approach z=80mm; x=+20mm,z=30mm,yaw=+10deg; x=-20mm,z=20mm,yaw=-10deg; nominal; retreat',
        wall_seconds=wall_seconds,records=plan)
    save_json(output/'protocol.json',protocol)
    manifest=dict(schema='align-precision-keypoints/v1',object_model='upright-cube-4cm',
                  pixel_convention='zero-based-pixel-centers',records=[])
    result=dict(status='running',records=plan,frames=0,rejected_frames=[])
    def persist():
        result.update(elapsed_s=time.monotonic()-started,frames=len(manifest['records']))
        states={f'scene-{r["seed"]}':r['status'] for r in plan}
        for frame in manifest['records']:frame['scene_collection_status']=states[frame['scene']]
        result['completed_scenes']=sum(r['status']=='completed' for r in plan)
        result['failed_scenes']=sum(r['status']=='failed' for r in plan)
        result['planned_scenes']=len(plan)
        result['status_meaning']='采集遍历完成不代表任务成功；失败前可信图像保留，按scene_collection_status区分'
        save_json(output/'manifest.json',manifest);save_json(output/'collection.json',result)
    def check():
        if time.monotonic()-started > wall_seconds:
            raise TimeoutError('采集达到墙钟上限')
    register_robot_vla_maniskill_envs()
    env=None;planner=None
    try:
        env=gym.make(PICK_CUBE_TO_REGION_ENV_ID,obs_mode='rgb+depth+segmentation',
            control_mode='pd_joint_pos',num_envs=1,sim_backend='physx_cpu',
            sensor_configs={'width':width,'height':height},max_episode_steps=1200)
        base=env.unwrapped
        for row in plan:
            check();row['status']='running';persist()
            obs,_=env.reset(seed=row['seed'])
            if planner is None:
                planner=PandaArmMotionPlanningSolver(env,debug=False,vis=False,
                    base_pose=base.agent.robot.pose,visualize_target_grasp_pose=False,
                    print_env_info=False,joint_vel_limits=.4,joint_acc_limits=.4)
            if not np.isclose(float(base.cube_half_size),.02):
                raise ValueError('当前监督仅支持边长4cm方块')
            frame_index=0;step=0
            scene=f'scene-{row["seed"]}'
            def capture(phase):
                nonlocal frame_index
                world_from_base=matrix(base.agent.robot.pose)
                world_from_object=matrix(base.cube.pose)
                tilt_deg=float(np.degrees(np.arccos(np.clip(world_from_object[2,2],-1,1))))
                height_error=abs(float(world_from_object[2,3])-.02)
                if tilt_deg>1. or height_error>.002:
                    rejected=dict(scene=scene,step=step,phase=phase,reason='upright_object_contract',
                                  tilt_deg=tilt_deg,height_error_m=height_error,files=[])
                    for camera_name in protocol['cameras']:
                        sensor=obs['sensor_data'][camera_name]
                        path=output/'audit'/f'rejected-{scene}-{step}-{camera_name}.npz'
                        np.savez_compressed(path,rgb=numpy(sensor['rgb'])[0],raw_depth=numpy(sensor['depth'])[0],
                                            world_from_object=world_from_object)
                        rejected['files'].append(dict(file=str(path.relative_to(output)),sha256=digest(path)))
                    result['rejected_frames'].append(rejected)
                    raise ValueError('方块不再满足直立桌面合同，违规观测已独立留档')
                base_from_object=np.linalg.inv(world_from_base)@world_from_object
                base_from_tcp=np.linalg.inv(world_from_base)@matrix(base.agent.tcp.pose)
                actor_id=int(numpy(base.cube.per_scene_id).reshape(-1)[0])
                for role in protocol['cameras']:
                    sensor=obs['sensor_data'][role];params=obs['sensor_param'][role]
                    rgb=numpy(sensor['rgb'])[0].astype(np.uint8)
                    raw_depth=numpy(sensor['depth'])[0,...,0]
                    # ManiSkill标准相机depth为int16毫米；禁止对未知单位猜测。
                    if raw_depth.dtype!=np.int16:
                        raise ValueError(f'未支持的ManiSkill深度dtype: {raw_depth.dtype}')
                    depth=raw_depth.astype(np.float32)/1000
                    mask=numpy(sensor['segmentation'])[0,...,0]==actor_id
                    k=numpy(params['intrinsic_cv'])[0].astype(np.float64)
                    camera=opengl_camera_to_opencv(numpy(params['cam2world_gl'])[0])
                    base_from_camera=np.linalg.inv(world_from_base)@camera
                    labels=corner_labels(k,np.linalg.inv(camera)@world_from_object,depth,mask)
                    name=f'{scene}-{step:05d}-{role}'
                    sample=output/'samples'/f'{name}.npz';audit=output/'audit'/f'{name}.npz'
                    np.savez_compressed(sample,rgb=rgb,pixel_uv=labels['pixel_uv'],visible=labels['visible'])
                    np.savez_compressed(audit,depth_m=depth,object_mask=mask,intrinsic=k,
                        base_from_camera_cv=base_from_camera,base_from_object=base_from_object,
                        base_from_tcp=base_from_tcp,timestamp_s=step/base.control_freq,
                        qpos=numpy(base.agent.robot.get_qpos())[0],object_tilt_deg=tilt_deg,
                        object_height_error_m=height_error,**labels)
                    manifest['records'].append(dict(id=name,scene=scene,split=row['split'],
                        file=str(sample.relative_to(output)),sha256=digest(sample),
                        audit_file=str(audit.relative_to(output)),audit_sha256=digest(audit),
                        camera=role,phase=phase,timestep=step))
                    frame_index+=1
            try:
                capture('reset')
                object_matrix=matrix(base.cube.pose)
                grasp=base.agent.build_grasp_pose(np.array([0.,0.,-1.]),object_matrix[:3,1],
                                                 numpy(base.cube.pose.p)[0])
                nominal=np.asarray(grasp.to_transformation_matrix())
                targets=[]
                for phase,offset,yaw in [('approach',[0,0,.08],0),('offset_right',[.02,0,.03],10),
                                         ('offset_left',[-.02,0,.02],-10),('align',[0,0,0],0),
                                         ('retreat',[0,0,.08],0)]:
                    target=nominal.copy();target[:3,3]+=offset
                    target[:3,:3]=Rotation.from_euler('z',yaw,degrees=True).as_matrix()@nominal[:3,:3]
                    targets.append((phase,sapien.Pose(target)))
                for phase,target in targets:
                    check()
                    path=planner.move_to_pose_with_screw(target,dry_run=True)
                    if isinstance(path,int) or path.get('status')!='Success':
                        raise RuntimeError(f'规划失败:{phase}')
                    for q in path['position']:
                        check()
                        obs,_,terminated,truncated,_=env.step(np.r_[q,1.].astype(np.float32))
                        step+=1
                        if step%stride==0:capture(phase)
                        if bool(numpy(terminated).any()) or bool(numpy(truncated).any()):
                            raise RuntimeError('感知采集环境提前终止')
                    if step%stride:capture(phase)
                row.update(status='completed',steps=step,frames=frame_index)
            except (RuntimeError,ValueError) as exc:
                # 失败前的可信观测也保留为感知样本；不能称作成功示范。
                row.update(status='failed',steps=step,frames=frame_index,error=f'{type(exc).__name__}: {exc}')
            persist();print(json.dumps(row),flush=True)
        result['status']='completed'
    except BaseException as exc:
        result.update(status='error',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        persist()
        if planner is not None:planner.close()
        if env is not None:env.close()
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    for name,default in [('train-scenes',12),('development-scenes',4),('seed-start',73000),
                         ('width',320),('height',240),('stride',8),('wall-seconds',900)]:
        p.add_argument('--'+name,type=int,default=default)
    args=p.parse_args();print(json.dumps(collect(**vars(args))))
