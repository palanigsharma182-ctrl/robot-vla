"""正式 G2C → D049 Memory 的单 seed 工程 smoke；结束于 HOME commit/no-commit。

默认不加载 VLA；显式传入 Runtime 时验证执行、暂停及 HOME 后重新规划。
不消费历史评估 seed，不把工程路线当成任务效果实验。
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
from robot_vla.execution.maniskill_controller import ManiSkillFrankaController
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
from robot_vla.runtime.control_loop import QwenVLAReplanLoop
from robot_vla.runtime.policy_runtime import OnlineObservation, QwenVLARuntime


class DevelopmentController(ManiSkillFrankaController):
    """开发 runner 的逐步终止检测，并可在指定控制步停止当前 Chunk。"""

    def __init__(self, env, spec, *, pause_after_steps=None):
        super().__init__(env, spec)
        self.sent = 0
        self.pause_after_steps = pause_after_steps
        self.chunk_stop_requested = False
        self.episode_done = False

    def send_action(self, value):
        if self.episode_done:
            raise RuntimeError("已终止的开发 Episode 禁止继续发送动作")
        super().send_action(value)
        self.sent += 1
        _, _, terminated, truncated, _ = self.last_step_output
        self.episode_done = bool(terminated.item()) or bool(truncated.item())
        self.chunk_stop_requested = self.sent == self.pause_after_steps or self.episode_done


def run(bundle: Path, output: Path, *, case: str = "new-scene",
        vla_runtime: QwenVLARuntime | None = None,
        development_seed: int | None = None, after_commit=None, memory_session=None,
        acquisition=None, environment=None) -> dict:
    import gymnasium as gym
    import sapien
    from mani_skill.utils import sapien_utils
    import robot_vla.sim.pick_cube_to_region  # noqa: F401
    from robot_vla.evaluation.maniskill import _read_observation_v2_frame, _single_transform_matrix

    cases = {"new-scene": 1000001, "historical-positive-76903": 76903}
    if case not in cases:
        raise ValueError("只允许本轮固定开发样例，不开放 selection/final-test seed")
    seed = cases[case]
    if environment is not None and acquisition is None:
        raise ValueError('借用环境仅用于显式隔离采集/开发消费者')
    if acquisition is not None:
        from experiments.memory_reobserve.protocol import Acquisition
        if not isinstance(acquisition, Acquisition) or any(x is not None for x in
                (development_seed, after_commit, memory_session, vla_runtime)):
            raise ValueError('新采集协议不能与旧采集或策略执行入口混用')
        seed = acquisition.seed
    if memory_session is not None:
        if development_seed is not None or after_commit is not None or vla_runtime is not None:
            raise ValueError('Memory Runtime 工程验收不能混用旧采集/执行入口')
        seed, vla_runtime = memory_session.seed, memory_session.runtime
    if development_seed is not None:
        if development_seed not in range(1000100, 1000112):
            raise ValueError("Memory 小试仅允许冻结的十二个 development seeds")
        if vla_runtime is not None:
            raise ValueError("Memory 数据采集 seed 禁止模型接入 actuator")
        seed = development_seed
    if after_commit is not None and (vla_runtime is not None or development_seed is None):
        raise ValueError("采集回调仅用于隔离开发 seed，不能与 VLA 争用执行器")
    output.mkdir(parents=True, exist_ok=False)
    provider = D049FrontProvider(bundle)
    spec = RobotSpec()
    adapter = FrankaObservationAdapter(spec)
    memory = ExplicitObjectStateMemory(build_stage2_object_memory_config())
    transaction = ActiveFrontStage2MemoryOrchestrator(memory, config=ActiveFrontStage2Config.development(min_information_gain=.10))
    episode = f"g2c-main-engineering-{seed}"
    transaction.reset_episode(episode, episode_generation=1)
    request_id = episode+"-request"
    if acquisition is not None and hasattr(acquisition, 'decide_request'):
        request_id = episode+'-active-front-01'
    if memory_session is not None:
        memory_session.reset(episode)
        request_id = episode+'-active-front-01'
    # 本 smoke 从未加载 VLA；这里验证实际为空的缓存失效，不冒充非空策略缓存验收。
    executor = RecedingHorizonChunkExecutor(spec)
    ensembler = TemporalChunkEnsembler(spec)
    loop = None if vla_runtime is None else QwenVLAReplanLoop(vla_runtime, executor)
    if loop is not None:
        ensembler = loop.ensembler
    vla_records = []
    pause = None
    pending_action = rtc_overlap = None
    home_history = ObservationV2History(spec)
    env = environment if environment is not None else gym.make("RobotVLAPickCubeToRegion-v1", robot_uids="panda_wristcam", num_envs=1,
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

    def record_vla(result, label):
        vla_records.append(dict(stage=label, execution=asdict(result.execution),
            conditioning=('memory' if memory_session is not None and memory_session.snapshot.available
                          else 'visual-only'),
            sampling=None if result.sampling is None else asdict(result.sampling),
            action_sha256=None if result.action_chunk is None else array_sha256(result.action_chunk.normalized_action),
            ensemble_buffer=None if result.ensemble_trace is None else result.ensemble_trace.buffer_size,
            control_step=loop.control_step))
        if not result.execution.success or result.execution.replan_required:
            raise RuntimeError(f"VLA {label} 未完成: {result.execution.failure_stage}")
        if controller.last_step_output is not None:
            _, _, terminated, truncated, _ = controller.last_step_output
            if bool(terminated.item()) or bool(truncated.item()):
                raise RuntimeError(f"VLA {label} 时场景已经终止")

    def sample(position, pitch_angle, motion, primitive, *, settled=False, stabilizing=False):
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
        # 暂停后的制动阶段单独记录有界漂移；只有稳定后才开始冻结相机路线。
        arm_limit, tcp_limit, angle_limit = (.05, .02, .1) if stabilizing else (1e-5, 1e-5, 1e-4)
        active = ActiveFrontSafetyEvidence(arm_hold_pass=arm_drift<=arm_limit+1e-12,
            tcp_hold_pass=bool(tcp_drift<=tcp_limit+1e-12 and tcp_angle<=angle_limit+1e-12),
            gripper_open_hold_pass=bool(np.min(qpos[7:])>=.039-1e-12),
            contact_absent=bool(np.max(frame.finger_force_n)<=.01+1e-12), active_window_open=True)
        safety = ObjectMemorySafetyContext(True, active.gripper_open_hold_pass,
            active.arm_hold_pass and active.tcp_hold_pass, not active.contact_absent, False, False, False, False)
        rows.append(dict(tick=tick, motion=motion.value, sidecar_sha256=sidecar.audit_digest(),
            primitive=primitive, settled=settled, control_timestamp_s=sidecar.control_timestamp_s,
            rgb_timestamp_s=sidecar.rgb_timestamp_s, camera_pose_timestamp_s=sidecar.camera_pose_timestamp_s,
            base_from_external_camera_cv=sidecar.base_from_external_camera_cv.tolist(),
            hold_reference_available=baseline_q is not None and baseline_tcp is not None,
            stabilizing=stabilizing, drift_limits=(arm_limit, tcp_limit, angle_limit),
            arm_drift_rad=None if baseline_q is None else arm_drift,
            tcp_drift_m=None if baseline_tcp is None else tcp_drift,
            tcp_drift_rad=None if baseline_tcp is None else tcp_angle,
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
        if environment is None:
            env.reset(seed=seed)
        home_evidence = []
        for initial_step in range(5):
            frame, sidecar, safety, active = sample(home_position, 0., Motion.HOME_ANCHOR, HOME, settled=True)
            if acquisition is not None and hasattr(acquisition, 'decide_request') and initial_step==1:
                baseline_q=frame.physical_proprio[:7].copy()
                baseline_tcp=frame.base_from_tcp.copy()
            if acquisition is not None and hasattr(acquisition, 'decide_request') and initial_step>=2:
                _check_camera_geometry(sidecar.intrinsic_cv,sidecar.base_from_external_camera_cv,
                    *provider.bundle_verification['camera_constraints'][HOME])
                home_evidence.append((infer(frame, sidecar, safety),active.failure() is None))
        baseline_q = env.unwrapped.agent.robot.get_qpos().detach().cpu().numpy()[0,:7].copy()
        baseline_tcp = frame.base_from_tcp.copy()
        home = home_evidence[-1][0] if home_evidence else infer(frame, sidecar, safety)
        baseline_timestamp = frame.timestamp_s
        if home_evidence and not acquisition.decide_request(home_evidence):
            capture = acquisition.capture(env=env, frame=frame, memory=memory,
                safety=safety, episode=episode, tick=tick, output=output)
            result = dict(status='development-no-observation-request',seed=seed,capture=capture,
                memory_write_count=0,provider_forward_count=provider.forward_count,
                request_evidence=acquisition.request_records,independent_effect_evidence=False)
            (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
            return result
        if loop is not None:
            if memory_session is not None:
                from experiments.memory_reobserve.controller import MemoryDevelopmentController
                controller = MemoryDevelopmentController(env, spec, session=memory_session,
                    memory=memory, initial_frame=frame, initial_tick=tick,
                    home_sidecar=sidecar, home_constraints=provider.bundle_verification["camera_constraints"][HOME])
                memory_session.bind(frame, memory, 'pick the cube and place it in the target region', safety=safety)
            else:
                controller = DevelopmentController(env, spec, pause_after_steps=2)
            # 执行真实策略的前两步后，在现有 executor 的控制步边界中断。
            loop.control_step = tick
            before = loop.replan_and_execute(OnlineObservation(frame.rgb_external,
                frame.rgb_wrist, frame.physical_proprio, "pick the cube and place it in the target region"), controller)
            record_vla(before, "before-pause")
            if memory_session is not None and memory_session.request is None:
                raise RuntimeError('真实控制帧未形成观察请求；停止本工程场景，不伪造触发原因')
            if not before.execution.interrupted or (memory_session is None and before.execution.executed_steps != 2):
                raise RuntimeError("未在指定控制步边界中断")
            tick = loop.control_step
            pause = loop.pause_for_observation()
            controller.chunk_stop_requested = False
            action[-1] = 2 * controller.read_state().gripper_opening - 1
            # 暂停初始状态必须允许 hold-open；不通过重新开夹爪伪造资格。
            if controller.read_state().gripper_opening < .975:
                raise RuntimeError("暂停时夹爪不满足既有 hold-open 条件")
            trigger_tick, trigger_timestamp = tick, tick / spec.control_hz
            if memory_session is None:
                baseline_q = baseline_tcp = None
            else:
                baseline_q = controller.frame.physical_proprio[:7].copy()
                baseline_tcp = controller.frame.base_from_tcp.copy()
            for _ in range(20):
                frame, sidecar, safety, active = sample(home_position, 0., Motion.HOME_ANCHOR, HOME,
                    settled=True, stabilizing=memory_session is not None)
            baseline_q = env.unwrapped.agent.robot.get_qpos().detach().cpu().numpy()[0,:7].copy()
            baseline_tcp = frame.base_from_tcp.copy()
        else:
            trigger_tick, trigger_timestamp = tick, frame.timestamp_s
        wrist_absent_sha = canonical_sha256({"scope":"engineering-no-wrist-provider", "episode":episode})
        request = ActiveFrontReobserveRequest(episode, 1, request_id, phase, phase, trigger_tick,
            trigger_timestamp, ActiveFrontTriggerReason.NO_QUALIFIED_WRIST_PROVIDER_IN_PARENT, 1, PRIMARY, f"command-{trigger_tick}")
        if memory_session is not None:
            request = memory_session.request
        if home_evidence:
            request = acquisition.request
        baseline = PassiveBaselineEvidence(episode,1,request_id,baseline_timestamp,False,wrist_absent_sha,
            home,False,None,None)
        pending_action = rtc_overlap = None
        if loop is None:
            ensembler.clear(); executor.reset()
        generation = 1 if pause is None else pause.generation
        reset = ActionHistoryResetReceipt(episode, request_id, trigger_tick, generation-1, generation,
            pending_action is None, ensembler.buffer_size==0, rtc_overlap is None,
            executor.previous_command_q is None and executor.previous_action is None)
        transaction.begin_collection(request, reset_receipt=reset, baseline=baseline)
        if memory_session is not None:
            memory_session.supervisor.begin(reset)
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
        return_steps = (10 if acquisition is not None else
                        40 if memory_session is None else memory_session.return_steps)
        for i, position in enumerate(sample_translation_path(alternate_position, home_position, steps=return_steps)):
            sample(position, pitch*(1-smootherstep((i+1)/return_steps)), Motion.RETURN_HOME, HOME)
        home_history.reset()
        for _ in range(4):
            frame, sidecar, safety, active = sample(home_position, 0., Motion.VERIFY_HOME_AND_ARM_HOLD, HOME, settled=True)
            _check_camera_geometry(sidecar.intrinsic_cv, sidecar.base_from_external_camera_cv,
                                  *provider.bundle_verification["camera_constraints"][HOME])
            home_history.append(frame)
            transaction.accept_home_v2_barrier_frame(HomeV2BarrierFrame(sidecar.observation_sequence_id,
                True, bool(frame.modality_valid.all()), True, False), timestamp_s=frame.timestamp_s, safety=active)
        window = home_history.snapshot("pick the cube and place it in the target region",
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
        capture = None
        if acquisition is not None:
            capture = acquisition.capture(env=env, frame=frame, memory=memory,
                safety=safety, episode=episode, tick=tick, output=output)
        if after_commit is not None and transaction.commit_receipt is not None:
            capture = after_commit(env=env, frame=frame, memory=memory,
                safety=safety, episode=episode, tick=tick, output=output)
        if loop is not None and transaction.commit_receipt is not None:
            if memory_session is not None:
                controller.resume_frame(frame, tick)
                memory_session.bind(frame, memory, window.instruction, safety=safety)
            # HOME 的实际几何、四帧和 source recheck 已在上方通过；Memory 不进入 VLA 输入。
            resumed = loop.resume_after_observation(pause, window, controller)
            record_vla(resumed, "after-home-fresh-replan")
            tick = loop.control_step
            if resumed.ensemble_trace.buffer_size != 1:
                raise RuntimeError("恢复推理混入了暂停前的旧动作")
            if memory_session is not None:
                memory_session.cleanup_after_execution(loop)
            post_replans = 2 if memory_session is None else memory_session.post_replans
            for index in range(post_replans - 1):
                if loop.observation_paused:
                    break
                if memory_session is not None:
                    controller.prepare_visual_replan(loop)
                    tick = loop.control_step
                observation = controller.last_step_output[0]
                latest = _read_observation_v2_frame(observation, env.unwrapped, adapter, spec, control_step=tick)
                if memory_session is not None:
                    memory_session.bind(latest, memory, window.instruction)
                continued = loop.replan_and_execute(OnlineObservation(latest.rgb_external,
                    latest.rgb_wrist, latest.physical_proprio, window.instruction), controller)
                record_vla(continued, "continued-execution")
                tick = loop.control_step
                if memory_session is not None:
                    memory_session.cleanup_after_execution(loop)
        result = dict(status="engineering-smoke", case=case, seed=seed, capture=capture,
            independent_effect_evidence=False, state=transaction.state.value,
            memory_write_count=transaction.memory_write_count, provider_forward_count=provider.forward_count,
            action_reset=asdict(reset), action_history_initially_empty=loop is None,
            vla_inference_executed=bool(vla_records), vla_execution=vla_records,
            pause_history=None if pause is None else asdict(pause),
            vla_resumed=any(r["stage"] == "continued-execution" for r in vla_records),
            source_scope="isolated-development-fixed-phase/no-external-executive/no-qualified-wrist-owner",
            home_v2_frames=int(window.history_valid.sum()), candidate=None if candidate is None else {
                "digest":candidate.digest,"eligible":candidate.commit_eligible,"reasons":candidate.rejection_reasons,
                "minimum_score":candidate.minimum_candidate_score,"information_gain":candidate.information_gain},
            terminal_reasons=transaction.terminal_reasons,
            no_commit=None if transaction.no_commit_receipt is None else asdict(transaction.no_commit_receipt),
            commit=None if transaction.commit_receipt is None else asdict(transaction.commit_receipt))
        (output/"result.json").write_text(json.dumps(result, indent=2)+"\n")
        return result
    except Exception as error:
        if memory_session is not None and loop is not None:
            memory_session.interruption_reason = memory_session.interruption_reason or "engineering-error"
            if memory_session.interruption_reason == "capability-trigger":
                memory_session.interruption_reason = "engineering-error-after-trigger"
            memory_session.cleanup_after_execution(loop)
        (output/"result.json").write_text(json.dumps(dict(status="engineering-error", tick=tick,
            state=transaction.state.value, error_type=type(error).__name__, error=str(error),
            terminal_reasons=transaction.terminal_reasons, memory_write_count=transaction.memory_write_count,
            provider_forward_count=provider.forward_count), indent=2)+"\n")
        raise
    finally:
        if environment is None:
            env.close()
        (output/"camera.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
        (output/"provider.jsonl").write_text("".join(json.dumps(r)+"\n" for r in frames))
        (output/"vla.jsonl").write_text("".join(json.dumps(r)+"\n" for r in vla_records))
        if memory_session is not None:
            (output/'memory-runtime.json').write_text(json.dumps(dict(
                trigger=memory_session.trigger_records, reads=memory_session.runtime.memory_reads,
                sends=memory_session.send_records, cleanup=memory_session.cleanup_records,
                tracking=controller.tracking_records if "controller" in locals() else [],
                camera_trigger=controller.camera_records if "controller" in locals() else []), indent=2)+'\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--case", choices=("new-scene", "historical-positive-76903"), default="new-scene")
    args = parser.parse_args()
    print(json.dumps(run(args.bundle, args.output, case=args.case), indent=2))
