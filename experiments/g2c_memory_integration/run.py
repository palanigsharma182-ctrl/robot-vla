"""正式 G2C → D049 Memory 的单 seed 工程 smoke；结束于 HOME commit/no-commit。

不加载 VLA，不执行操纵动作，不消费历史评估 seed。机械臂只沿用 G0 的 hold-open。
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from robot_vla.adapters import FrankaObservationAdapter
from robot_vla.contracts import RobotSpec
from robot_vla.execution.chunk_executor import RecedingHorizonChunkExecutor
from robot_vla.execution.temporal_ensemble import TemporalChunkEnsembler
from robot_vla.executive.contracts import PhaseId
from robot_vla.observation import ObservationV2History
from robot_vla.precision.active_external_observation import extract_active_external_observation
from robot_vla.precision.active_front_camera import (
    ExternalCameraMotionState as Motion, FrontCameraOrientationMode,
    compose_camera_orientation_wxyz, sample_translation_path, smootherstep,
    rotation_angular_distance_rad,
)
from robot_vla.precision.active_front_memory import (
    ActiveFrontStage2MemoryOrchestrator, ActiveFrontSourceRecheckEvidence, PendingActiveViewState,
)
from robot_vla.precision.active_front_memory_provider import (
    ACTIVE_FRONT_HOME_PRIMITIVE_ID as HOME, ACTIVE_FRONT_PRIMARY_PRIMITIVE_ID as PRIMARY,
    ActiveFrontStage2Config, PassiveBaselineEvidence, build_stage2_object_memory_config,
)
from robot_vla.precision.active_front_reobserve import (
    ActionHistoryResetReceipt, ActiveFrontReobserveRequest, ActiveFrontTriggerReason,
    ActiveFrontSafetyEvidence, HomeV2BarrierFrame,
)
from robot_vla.precision.calibrated_front_provider import canonical_sha256, array_sha256
from robot_vla.precision.object_memory import ExplicitObjectStateMemory, ObjectMemorySafetyContext
from robot_vla.precision.qualified_front_provider import D049FrontProvider, _check_camera_geometry


def run(bundle: Path, output: Path, *, case: str = "new-scene") -> dict:
    import gymnasium as gym
    import sapien
    from mani_skill.utils import sapien_utils
    import robot_vla.sim.pick_cube_to_region  # noqa: F401
    from robot_vla.evaluation.maniskill import _read_observation_v2_frame, _single_transform_matrix

    cases = {"new-scene": 1000001, "historical-positive-76903": 76903}
    if case not in cases:
        raise ValueError("只允许本轮固定开发样例，不开放 selection/final-test seed")
    seed = cases[case]
    output.mkdir(parents=True, exist_ok=False)
    provider = D049FrontProvider(bundle)
    spec = RobotSpec()
    adapter = FrankaObservationAdapter(spec)
    memory = ExplicitObjectStateMemory(build_stage2_object_memory_config())
    transaction = ActiveFrontStage2MemoryOrchestrator(memory, config=ActiveFrontStage2Config.development(min_information_gain=.10))
    episode = f"g2c-main-engineering-{seed}"
    transaction.reset_episode(episode, episode_generation=1)
    request_id = episode+"-request"
    # 本 smoke 从未加载 VLA；这里验证实际为空的缓存失效，不冒充非空策略缓存验收。
    executor = RecedingHorizonChunkExecutor(spec)
    ensembler = TemporalChunkEnsembler(spec)
    pending_action = rtc_overlap = None
    home_history = ObservationV2History(spec)
    env = gym.make("RobotVLAPickCubeToRegion-v1", robot_uids="panda_wristcam", num_envs=1,
        obs_mode="rgb", control_mode="pd_joint_delta_pos", sim_backend="cpu", render_backend="gpu",
        sensor_configs={"width":128, "height":128})
    rows = []
    frames = []
    tick = 0
    baseline_q = baseline_tcp = None
    phase = PhaseId.ACQUIRE_TRACK
    home_position = np.array([.3, 0., .6])
    alternate_position = np.array([.3, -.16, .48])
    target = np.array([-.1, 0., .1])
    # G0C 冻结配置：yaw offset 12°，本 PRIMARY 的 pitch offset 为 8°。
    pitch = np.deg2rad(8.)
    action = np.zeros(8, np.float32); action[-1] = 1.

    def sample(position, pitch_angle, motion, primitive, *, settled=False):
        nonlocal tick
        base = env.unwrapped
        nominal = sapien_utils.look_at(position, target)
        nominal_q = nominal.q.detach().cpu().numpy()
        if nominal_q.shape != (1, 4):
            raise ValueError("单环境 look_at quaternion 必须是 [1,4]")
        q = compose_camera_orientation_wxyz(nominal_q[0], FrontCameraOrientationMode("route", 0., pitch_angle))
        command = sapien.Pose(p=position, q=q)
        base._sensors["base_camera"].camera.set_local_pose(command)
        commanded_gl = (command * sapien.Pose(q=[-.5,-.5,.5,.5])).to_transformation_matrix()
        observation, _, terminated, truncated, _ = env.step(action)
        tick += 1
        if bool(terminated.item()) or bool(truncated.item()):
            raise RuntimeError("工程 route 提前终止")
        frame = _read_observation_v2_frame(observation, base, adapter, spec, control_step=tick)
        sidecar = extract_active_external_observation(observation, camera_uid="base_camera",
            world_from_robot_base=_single_transform_matrix(base.agent.robot.pose, "base"),
            commanded_world_from_external_camera_gl=commanded_gl, episode_id=episode, request_id=request_id,
            observation_sequence_id=f"frame-{tick}", camera_command_sequence_id=f"command-{tick}",
            control_tick=tick, control_timestamp_s=frame.timestamp_s, rgb_timestamp_s=frame.timestamp_s,
            camera_pose_timestamp_s=frame.timestamp_s, camera_motion_state=motion,
            viewpoint_primitive_id=primitive, settled=settled, maximum_rotation_projection_error_frobenius=1e-6)
        qpos = base.agent.robot.get_qpos().detach().cpu().numpy()[0]
        arm_drift = 0. if baseline_q is None else float(np.max(np.abs(qpos[:7]-baseline_q)))
        tcp_drift = 0. if baseline_tcp is None else float(np.linalg.norm(frame.base_from_tcp[:3,3]-baseline_tcp[:3,3]))
        tcp_angle = 0. if baseline_tcp is None else rotation_angular_distance_rad(frame.base_from_tcp[:3,:3], baseline_tcp[:3,:3])
        active = ActiveFrontSafetyEvidence(arm_hold_pass=arm_drift<=1e-5+1e-12,
            tcp_hold_pass=bool(tcp_drift<=1e-5+1e-12 and tcp_angle<=1e-4+1e-12),
            gripper_open_hold_pass=bool(np.min(qpos[7:])>=.039-1e-12),
            contact_absent=bool(np.max(frame.finger_force_n)<=.01+1e-12), active_window_open=True)
        safety = ObjectMemorySafetyContext(True, active.gripper_open_hold_pass,
            active.arm_hold_pass and active.tcp_hold_pass, not active.contact_absent, False, False, False, False)
        rows.append(dict(tick=tick, motion=motion.value, sidecar_sha256=sidecar.audit_digest(),
            primitive=primitive, settled=settled, control_timestamp_s=sidecar.control_timestamp_s,
            rgb_timestamp_s=sidecar.rgb_timestamp_s, camera_pose_timestamp_s=sidecar.camera_pose_timestamp_s,
            base_from_external_camera_cv=sidecar.base_from_external_camera_cv.tolist(),
            arm_drift_rad=arm_drift, tcp_drift_m=tcp_drift, tcp_drift_rad=tcp_angle,
            safety=asdict(active), rgb_sha256=array_sha256(frame.rgb_external)))
        if active.failure() is not None:
            raise RuntimeError(f"hold-open 失败: {active.failure().value}")
        return frame, sidecar, safety, active

    def infer(frame, sidecar, safety):
        result = provider.predict(sidecar, physical_proprio=frame.physical_proprio,
            base_from_tcp=frame.base_from_tcp, finger_force_n=frame.finger_force_n,
            tcp_timestamp_s=frame.timestamp_s, episode_generation=1, source_phase=phase,
            safety=safety, static_plane_scope_verified=bool(env.unwrapped.cube_half_size==.02))
        frames.append(dict(tick=tick, input_digest=result.model_input_digest,
            output_digest=result.provider_output_digest, score_components=asdict(result.score_components)))
        if hasattr(result, "position_base_m"):
            frames[-1].update(position_base_m=None if result.position_base_m is None else np.asarray(result.position_base_m).tolist(),
                covariance_base_m2=None if result.covariance_base_m2 is None else np.asarray(result.covariance_base_m2).tolist())
        return result

    try:
        env.reset(seed=seed)
        for _ in range(5):
            frame, sidecar, safety, active = sample(home_position, 0., Motion.HOME_ANCHOR, HOME, settled=True)
        baseline_q = env.unwrapped.agent.robot.get_qpos().detach().cpu().numpy()[0,:7].copy()
        baseline_tcp = frame.base_from_tcp.copy()
        home = infer(frame, sidecar, safety)
        wrist_absent_sha = canonical_sha256({"scope":"engineering-no-wrist-provider", "episode":episode})
        request = ActiveFrontReobserveRequest(episode, 1, request_id, phase, phase, tick,
            frame.timestamp_s, ActiveFrontTriggerReason.NO_QUALIFIED_WRIST_PROVIDER_IN_PARENT, 1, PRIMARY, f"command-{tick}")
        baseline = PassiveBaselineEvidence(episode,1,request_id,frame.timestamp_s,False,wrist_absent_sha,
            home,False,None,None)
        pending_action = rtc_overlap = None
        ensembler.clear(); executor.reset()
        reset = ActionHistoryResetReceipt(episode, request_id, tick, 0, 1,
            pending_action is None, ensembler.buffer_size==0, rtc_overlap is None,
            executor.previous_command_q is None and executor.previous_action is None)
        transaction.begin_collection(request, reset_receipt=reset, baseline=baseline)
        for i, position in enumerate(sample_translation_path(home_position, alternate_position, steps=40)):
            sample(position, pitch*smootherstep((i+1)/40), Motion.MOVE_TO_VIEW, PRIMARY)
        for _ in range(4): sample(alternate_position, pitch, Motion.SETTLE_AT_VIEW, PRIMARY)
        for _ in range(3):
            frame, sidecar, safety, active = sample(alternate_position, pitch, Motion.COLLECT, PRIMARY, settled=True)
            evidence = infer(frame, sidecar, safety)
            if transaction.state is PendingActiveViewState.COLLECTING:
                adaptation = transaction.observe_collect_frame(evidence, safety=safety)
                frames[-1].update(eligible=adaptation.eligible, reasons=adaptation.rejection_reasons)
        candidate = transaction.pending_candidate
        transaction.mark_returning_home(timestamp_s=frame.timestamp_s,
            candidate_digest=None if candidate is None else candidate.digest)
        for i, position in enumerate(sample_translation_path(alternate_position, home_position, steps=40)):
            sample(position, pitch*(1-smootherstep((i+1)/40)), Motion.RETURN_HOME, HOME)
        home_history.reset()
        for _ in range(4):
            frame, sidecar, safety, active = sample(home_position, 0., Motion.VERIFY_HOME_AND_ARM_HOLD, HOME, settled=True)
            _check_camera_geometry(sidecar.intrinsic_cv, sidecar.base_from_external_camera_cv,
                                  *provider.bundle_verification["camera_constraints"][HOME])
            home_history.append(frame)
            transaction.accept_home_v2_barrier_frame(HomeV2BarrierFrame(sidecar.observation_sequence_id,
                True, bool(frame.modality_valid.all()), True, False), timestamp_s=frame.timestamp_s, safety=active)
        window = home_history.snapshot("hold open; return external camera HOME",
            previous_command_q=executor.previous_command_q, previous_action=executor.previous_action)
        if not window.history_valid.all() or not window.modality_valid.all() or window.controller_valid.any():
            raise RuntimeError("HOME V2 完整性或失效后的 command reference 不匹配")
        if transaction.state is PendingActiveViewState.HOME_BARRIER_PASSED:
            checked = transaction.recheck_source(ActiveFrontSourceRecheckEvidence(episode,1,request_id,
                candidate.digest, frame.timestamp_s,phase,True,active.failure() is None,True,False,wrist_absent_sha), safety=active)
            if checked:
                try:
                    transaction.commit(candidate_digest=candidate.digest, commit_timestamp_s=frame.timestamp_s, safety=safety)
                except RuntimeError:
                    if (transaction.no_commit_receipt is None
                            or transaction.state is not PendingActiveViewState.HOME_VERIFIED_FAILED_SAFE_HOLD
                            or transaction.memory_write_count != 0):
                        raise
        result = dict(status="engineering-smoke", case=case, seed=seed,
            independent_effect_evidence=False, state=transaction.state.value,
            memory_write_count=transaction.memory_write_count, provider_forward_count=provider.forward_count,
            action_reset=asdict(reset), action_history_initially_empty=True, vla_inference_executed=False,
            home_v2_frames=int(window.history_valid.sum()), candidate=None if candidate is None else {
                "digest":candidate.digest,"eligible":candidate.commit_eligible,"reasons":candidate.rejection_reasons,
                "minimum_score":candidate.minimum_candidate_score,"information_gain":candidate.information_gain},
            terminal_reasons=transaction.terminal_reasons,
            no_commit=None if transaction.no_commit_receipt is None else asdict(transaction.no_commit_receipt),
            commit=None if transaction.commit_receipt is None else asdict(transaction.commit_receipt))
        (output/"result.json").write_text(json.dumps(result, indent=2)+"\n")
        return result
    except Exception as error:
        (output/"result.json").write_text(json.dumps(dict(status="engineering-error", tick=tick,
            state=transaction.state.value, error_type=type(error).__name__, error=str(error),
            terminal_reasons=transaction.terminal_reasons, memory_write_count=transaction.memory_write_count,
            provider_forward_count=provider.forward_count), indent=2)+"\n")
        raise
    finally:
        env.close()
        (output/"camera.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
        (output/"provider.jsonl").write_text("".join(json.dumps(r)+"\n" for r in frames))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--case", choices=("new-scene", "historical-positive-76903"), default="new-scene")
    args = parser.parse_args()
    print(json.dumps(run(args.bundle, args.output, case=args.case), indent=2))
