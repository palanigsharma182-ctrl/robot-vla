"""补采独立场景的四个固定阶段画面；GT单独保存，仅供监督与审计。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.oracle_reach_control.runner import PositionRuntime, numpy
from robot_vla.adapters import FrankaObservationAdapter
from robot_vla.contracts import RobotSpec
from robot_vla.diagnostics.oracle_reach import (
    FrankaTCPForwardKinematics, current_relative_geometry, find_maniskill_panda_urdf,
)
from robot_vla.diagnostics.oracle_reach_evaluation import _DistanceTraceController
from robot_vla.diagnostics.qwen_spatial_probe import project_world_point_to_gl_camera
from robot_vla.evaluation.maniskill import _read_online_observation
from robot_vla.execution import RecedingHorizonChunkExecutor
from robot_vla.runtime import QwenVLAReplanLoop
from robot_vla.sim.collector import TrustedPickPlaceCollector

SEEDS = tuple(range(1_300_000, 1_300_048))
STEPS = (0, 8, 16, 24)
PROMPT = "Pick up the red cube and place it in the green target region."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "images").mkdir()
    split_order = np.random.default_rng(20260907).permutation(SEEDS)
    train = set(map(int, split_order[:32]))
    records = []
    spec = RobotSpec()
    camera_reference = None
    with TrustedPickPlaceCollector(None, spec) as preparer:
        for seed in SEEDS:
            prep = preparer.prepare_atomic(seed=seed, skill_name="reach")
            base = preparer.base_env
            root = numpy(base.agent.robot.pose.to_transformation_matrix())[0]
            assert np.allclose(root[:3, :3], np.eye(3), atol=1e-6)
            fk = FrankaTCPForwardKinematics(find_maniskill_panda_urdf(), spec, base_position_world_m=root[:3, 3])
            q = numpy(base.agent.robot.get_qpos())[0, :7]
            assert np.linalg.norm(fk(q) - numpy(base.agent.tcp_pose.p)[0]) < 1e-5
            adapter = FrankaObservationAdapter(spec)
            controller = _DistanceTraceController(
                preparer.env, spec, prep.observation, prep.tracker, prep.progress,
                adapter, max_policy_steps=24,
            )
            executor = RecedingHorizonChunkExecutor(spec)
            runtime = PositionRuntime(fk, lambda: current_relative_geometry(base), spec, seed)
            runtime.command_reference = lambda: executor.previous_command_q
            loop = QwenVLAReplanLoop(runtime, executor, temporal_ensemble_enabled=True, recency_decay=.5)
            for phase, step in enumerate(STEPS):
                # 每个预定采样点也是块内停止点；随后提高上限继续同一Episode。
                controller.max_policy_steps = step
                while controller.environment_steps < step:
                    online = _read_online_observation(controller.observation, base, adapter, PROMPT)
                    before = controller.environment_steps
                    result = loop.replan_and_execute(online, controller)
                    if not result.execution.success or controller.environment_steps == before:
                        raise RuntimeError(f"数据采集执行失败 seed={seed}, step={before}: {result.execution}")
                    if controller.environment_steps > step:
                        raise RuntimeError("异常重规划使采样越过预定阶段，停止而非更换样本")
                observation = controller.observation
                sensor = observation['sensor_data']['base_camera']
                rgb = numpy(sensor['rgb'])[0]
                wrist = numpy(observation['sensor_data']['hand_camera']['rgb'])[0]
                params = observation['sensor_param']
                calibration = asdict(preparer._camera_calibration(observation))
                camera = np.r_[calibration['intrinsic_external'], calibration['world_from_external']]
                if camera_reference is None:
                    camera_reference = camera
                np.testing.assert_allclose(camera, camera_reference, rtol=0, atol=1e-6)
                object_position = numpy(base.cube.pose.p)[0].copy()
                height, width = rgb.shape[:2]
                projection = project_world_point_to_gl_camera(
                    object_position, calibration['intrinsic_external'],
                    calibration['world_from_external'], height, width,
                )
                uv = projection.normalized_uv
                projected = bool(((uv >= 0) & (uv <= 1)).all())
                segmentation = numpy(sensor['segmentation'])[0, ..., 0]
                actor_id = int(numpy(base.cube.per_scene_id).reshape(-1)[0])
                visible_pixels = int((segmentation == actor_id).sum())
                px = np.clip(np.round(projection.pixel_uv).astype(int), [0, 0], [width-1, height-1])
                center_visible = bool(projected and segmentation[px[1], px[0]] == actor_id)
                bucket = ('out_of_view' if not projected else 'occluded' if visible_pixels < 4 else
                          'visible_center' if center_visible else 'partly_visible')
                sample_id = f"{seed}-{step:02d}"
                # 图像文件不含GT；真实状态与标签在审计记录中单独保存。
                np.savez_compressed(args.output/'images'/f'{sample_id}.npz', rgb_external=rgb, rgb_wrist=wrist)
                record = dict(
                    sample_id=sample_id, scene=seed, split='train' if seed in train else 'development',
                    episode_id=f'qwen-spatial-{seed}',
                    rgb_external_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                    rgb_wrist_sha256=hashlib.sha256(wrist.tobytes()).hexdigest(),
                    actual_q=numpy(base.agent.robot.get_qpos())[0,:7].tolist(),
                    tcp_position_world_m=numpy(base.agent.tcp_pose.p)[0].tolist(),
                    phase=phase, step=step, timestamp_s=step/spec.control_hz, instruction=PROMPT,
                    uv=uv.tolist(), object_position_world_m=object_position.tolist(),
                    root_position_world_m=root[:3, 3].tolist(), image_size=[height, width],
                    calibration=calibration, projected=projected, visible_pixels=visible_pixels,
                    visibility=bucket, physical_proprio=_read_online_observation(observation, base, adapter, PROMPT).physical_proprio.tolist(),
                    world_from_tcp=numpy(base.agent.tcp_pose.to_transformation_matrix())[0].tolist(),
                    world_from_wrist_gl=numpy(params['hand_camera']['cam2world_gl'])[0].tolist(),
                )
                records.append(record)
                with (args.output/'samples.jsonl').open('a') as file:
                    file.write(json.dumps(record)+'\n')
            print(json.dumps(dict(scene=seed, samples=len(records))), flush=True)
    assert len(records) == 192
    assert all(r['projected'] for r in records), '保留全部采集，出视野时停止该sigmoid协议'
    (args.output/'summary.json').write_text(json.dumps(dict(
        scenes=48, samples=192, train_scenes=sorted(train),
        development_scenes=sorted(set(SEEDS)-train), steps=STEPS,
        visibility={v:sum(r['visibility']==v for r in records) for v in sorted({r['visibility'] for r in records})},
    ), indent=2)+'\n')


if __name__ == '__main__':
    main()
