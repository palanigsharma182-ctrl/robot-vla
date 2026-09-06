"""实验测量到现有显式Memory的适配，不继承D049资格或actuator权限。"""
from __future__ import annotations

import numpy as np

from experiments.memory_conditioning.conditioning import snapshot_memory
from robot_vla.precision.object_memory import (
    ExplicitObjectStateMemory, ObjectCandidateWindowVerifier, ObjectMeasurement,
    ObjectMemoryConfig, ObjectMemorySafetyContext,
)
from experiments.front_rgbd_memory.geometry import PROVIDER_ID


def candidate_config():
    return ObjectMemoryConfig(
        max_unobserved_age_s=2.5,max_innovation_m=.010,max_position_std_m=.020,
        min_candidate_frames=3,max_candidate_gap_s=.075,max_candidate_position_spread_m=.005,
        max_sensor_skew_s=.010,expected_source_camera='base_camera',
        expected_source_model_identity=PROVIDER_ID,require_covariance=True,
        covariance_growth_m2_per_s=1e-6,
    )


class MemoryReplay:
    """只读出快照，不加载Qwen、不生成策略动作；拒绝/过期同样记账。"""
    def __init__(self,episode):
        self.episode=episode
        self.config=candidate_config()
        self.verifier=ObjectCandidateWindowVerifier(self.config)
        self.memory=ExplicitObjectStateMemory(self.config)
        self.verifier.reset(episode)
        self.memory.reset(episode,timestamp_s=0.)

    def update(self,estimate,*,timestamp,safety):
        valid=bool(estimate.valid)
        measurement=ObjectMeasurement(
            timestamp_s=timestamp,rgb_timestamp_s=timestamp,camera_pose_timestamp_s=timestamp,
            tcp_pose_timestamp_s=timestamp,position_base_m=estimate.position,
            covariance_base_m2=estimate.covariance,confidence=.5 if valid else 0.,
            projection_valid=valid,in_fov=valid,observable=valid,geometry_valid=valid,
            write_gate_passed=valid,source_camera='base_camera',source_model_identity=PROVIDER_ID,
        )
        decision=self.verifier.observe(measurement,episode_id=self.episode,safety=safety)
        update=self.memory.update(decision,episode_id=self.episode,safety=safety)
        snap=snapshot_memory(self.memory.state,self.config,safety,episode_id=self.episode,timestamp_s=timestamp)
        return dict(accepted=update.measurement_accepted,reasons=update.rejection_reasons,
                    candidate_frames=decision.frame_count,available=snap.available,
                    snapshot_features=list(snap.features),snapshot_reasons=snap.reasons,
                    last_observed_timestamp_s=snap.last_observed_timestamp_s,
                    position=None if self.memory.state.position_base_m is None else list(self.memory.state.position_base_m))

    def begin_view(self):
        """新视角重新收集三帧候选；既有Memory保留自身来源和年龄。"""
        self.verifier.reset(self.episode)


def fixture_safety(gripper_open,tracking_valid):
    """静止无碰撞测量台架的显式范围；不能复用为真实操纵中的安全判定。"""
    return ObjectMemorySafetyContext(
        pregrasp_window_open=True,gripper_open=bool(gripper_open),controller_tracking_valid=bool(tracking_valid),
        object_contact_detected=False,gripper_close_commanded=False,grasp_candidate=False,
        grasp_verified=False,object_maybe_moved=False,
    )


def tracking_safety(q,tcp,reference_q,reference_tcp):
    q=np.asarray(q);tcp=np.asarray(tcp)
    q_drift=float(np.max(np.abs(q[:7]-reference_q)))
    tcp_drift=float(np.linalg.norm(tcp-reference_tcp))
    tracking=bool(np.isfinite(q).all() and np.isfinite(tcp).all() and q_drift<.03 and tcp_drift<.01)
    return fixture_safety(q[-2:].sum()>.06,tracking),dict(
        q_drift_rad=q_drift if np.isfinite(q_drift) else None,
        tcp_drift_m=tcp_drift if np.isfinite(tcp_drift) else None,
        tracking_valid=tracking,
    )
