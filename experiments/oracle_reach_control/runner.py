"""隔离比较 Oracle 学习策略与显式位置伺服；不改变 canonical 控制路径。"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from robot_vla.adapters import ActionAdapter, FrankaObservationAdapter, ProprioNormalizer, ProprioStats
from robot_vla.cli.diagnose_oracle_reach import _build_policy, _load_policy_checkpoint
from robot_vla.cli.evaluate_maniskill import _load_audit_identity
from robot_vla.cli.train_stage1 import compute_source_revision
from robot_vla.contracts import RobotSpec
from robot_vla.data.trajectory import load_manifest
from robot_vla.diagnostics.oracle_reach import (
    FrankaTCPForwardKinematics, OracleGeometryRuntime, current_relative_geometry,
    find_maniskill_panda_urdf,
)
from robot_vla.diagnostics.oracle_reach_evaluation import _DistanceTraceController
from robot_vla.evaluation.atomic import derive_atomic_sampling_seed
from robot_vla.evaluation.maniskill import _read_online_observation, _reset_atomic_time_limit
from robot_vla.execution import RecedingHorizonChunkExecutor
from robot_vla.runtime import QwenVLAReplanLoop
from robot_vla.runtime.policy_runtime import RuntimeActionChunk, RuntimeConfig, SamplingTrace
from robot_vla.sim.collector import TrustedPickPlaceCollector
from robot_vla.tasks.pick_place import build_pick_place_task

CHECKPOINT_SHA = "294b555fd16e2c6ceb2528b167289cdb94edb93d36844d16715cae58f89c9f05"
SEEDS = tuple(range(1_200_000, 1_200_010))
SETTINGS = dict(damping=0.02, gain=0.25, max_cartesian_step_m=0.005, epsilon_rad=1e-4)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def position_step(fk, q, target, *, limits, damping, gain, max_cartesian_step_m, epsilon_rad):
    """已知运动学的阻尼最小二乘；只约束位置，不额外使用 GT 姿态。"""
    q, target = np.asarray(q, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if q.shape != (7,) or target.shape != (3,) or not np.isfinite(np.r_[q, target]).all():
        raise ValueError("位置伺服输入必须为有限 q[7] 与 target[3]")
    jacobian = np.empty((3, 7))
    for i in range(7):
        offset = np.zeros(7)
        offset[i] = epsilon_rad
        jacobian[:, i] = (fk(q + offset) - fk(q - offset)) / (2 * epsilon_rad)
    error = gain * (target - fk(q))
    error *= min(1.0, max_cartesian_step_m / max(float(np.linalg.norm(error)), 1e-12))
    dq = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping**2 * np.eye(3), error)
    dq /= max(1.0, float(np.max(np.abs(dq) / limits)))
    if not np.isfinite(dq).all():
        raise ValueError("位置伺服输出非有限值")
    return dq


class PositionRuntime:
    def __init__(self, fk, geometry_provider, spec, sampling_seed):
        self.fk, self.geometry_provider, self.spec = fk, geometry_provider, spec
        self.adapter = ActionAdapter(spec)
        self.sampling_seed = sampling_seed
        self.index = 0
        self.last_sampling_trace = None
        self.command_reference = lambda: None

    def infer_action_chunk(self, observation):
        q = np.asarray(observation.physical_proprio[:7], dtype=np.float64)
        target = self.fk(q) + np.asarray(self.geometry_provider()[:3])
        reference = self.command_reference()
        reference = q.copy() if reference is None else np.asarray(reference, dtype=np.float64).copy()
        joint_limits = np.asarray(self.spec.joint_position_limits_rad)
        physical = np.zeros((self.spec.action_horizon, self.spec.action_dim), dtype=np.float32)
        physical[:, -1] = 1.0
        for i in range(self.spec.action_horizon):
            desired = q + position_step(
                self.fk, q, target, limits=self.adapter.delta_limits, **SETTINGS,
            )
            desired = np.clip(desired, joint_limits[:, 0] + 1e-5, joint_limits[:, 1] - 1e-5)
            # 标签以此前 commanded target 为基准；不能直接把 actual-q correction 当标签。
            delta = np.clip(desired - reference, -self.adapter.delta_limits, self.adapter.delta_limits)
            physical[i, :7] = delta
            reference += delta
            q = reference.copy()
        trace = SamplingTrace(self.sampling_seed + self.index, self.index)
        self.index += 1
        self.last_sampling_trace = trace
        return RuntimeActionChunk(
            self.adapter.normalize(physical), physical, (0, 0), 1, trace,
        )


class OpenGripperRuntime:
    """双方共同固定夹爪张开，避免把抓取行为混入位置控制对照。"""

    def __init__(self, inner, spec):
        self.inner, self.adapter = inner, ActionAdapter(spec)

    @property
    def last_sampling_trace(self):
        return self.inner.last_sampling_trace

    def infer_action_chunk(self, observation):
        chunk = self.inner.infer_action_chunk(observation)
        physical = chunk.physical_action.copy()
        physical[:, -1] = 1.0
        return replace(chunk, physical_action=physical, normalized_action=self.adapter.normalize(physical))


def numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def episode(preparer, inner, spec, seed, preparation):
    adapter = FrankaObservationAdapter(spec)
    _reset_atomic_time_limit(preparer.env)
    controller = _DistanceTraceController(
        preparer.env, spec, preparation.observation, preparation.tracker,
        preparation.progress, adapter, max_policy_steps=100,
    )
    executor = RecedingHorizonChunkExecutor(spec)
    if isinstance(inner, PositionRuntime):
        inner.command_reference = lambda: executor.previous_command_q
    loop = QwenVLAReplanLoop(
        OpenGripperRuntime(inner, spec), executor,
        temporal_ensemble_enabled=True, recency_decay=0.5, max_anomaly_replans=3,
    )
    errors, inference_ms = [], []
    anomalies = saturations = 0
    instruction = build_pick_place_task(seed % 3).instruction
    while not controller.done and controller.progress.completed_skill_count == 0:
        observation = _read_online_observation(controller.observation, preparer.base_env, adapter, instruction)
        started = time.monotonic()
        result = loop.replan_and_execute(observation, controller)
        inference_ms.append((time.monotonic() - started) * 1000)
        anomalies += int(result.execution.replan_required)
        saturations += result.execution.correction_saturation_steps
        if loop.control_step != controller.environment_steps:
            raise RuntimeError("loop 时钟与实际环境步数分叉")
        if not result.execution.success:
            errors.append(dict(stage=result.execution.failure_stage, error=result.execution.error))
            break
    return dict(
        seed=seed, success=controller.progress.completed_skill_count >= 1,
        steps=controller.environment_steps, distance_trace_m=controller.distance_trace_m,
        errors=errors, anomaly_replans=anomalies, saturation_steps=saturations,
        mean_replan_and_execution_ms=float(np.mean(inference_ms)),
        final_object_position_m=numpy(preparer.base_env.cube.pose.p)[0].tolist(),
        final_tcp_position_m=numpy(preparer.base_env.agent.tcp_pose.p)[0].tolist(),
        final_is_grasped=controller.progress.outcome.grasped,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    assert digest(args.checkpoint) == CHECKPOINT_SHA
    assert not set(SEEDS) & {int(e.randomization["seed"]) for e in load_manifest(args.data)}
    spec = RobotSpec()
    policy = _build_policy("oracle", seed=42, model_cache=None, device="cuda")
    metadata = _load_policy_checkpoint(args.checkpoint, "oracle", policy)
    data = _load_audit_identity(args.data)
    assert metadata["dataset_sha256"] == data["dataset_sha256"]
    normalizer = ProprioNormalizer(ProprioStats.from_json(args.data / "proprio_stats.json"), spec)
    protocol = dict(
        seeds=SEEDS, data_use="development", checkpoint_sha256=CHECKPOINT_SHA,
        settings=SETTINGS, max_steps=100, threshold_m=0.04, gripper="open_both_arms",
        orientation="unconstrained_both_arms", ensemble_decay=0.5, flow_steps=10,
        source_revision=compute_source_revision(Path(__file__).resolve().parents[2]),
        runner_sha256=digest(__file__), dataset=data,
    )
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    records = []
    with TrustedPickPlaceCollector(None, spec) as preparer:
        for seed in SEEDS:
            pair_initial = None
            # 交替顺序降低启动/热身对耗时描述的偏倚，耗时不作为主指标。
            arms = ("learned", "analytic") if seed % 2 == 0 else ("analytic", "learned")
            for arm in arms:
                prep = preparer.prepare_atomic(seed=seed, skill_name="reach")
                root_pose = numpy(preparer.base_env.agent.robot.pose.to_transformation_matrix())[0]
                assert np.allclose(root_pose[:3, :3], np.eye(3), atol=1e-6)
                fk = FrankaTCPForwardKinematics(find_maniskill_panda_urdf(), spec, base_position_world_m=root_pose[:3, 3])
                q = numpy(preparer.base_env.agent.robot.get_qpos())[0, :7]
                tcp = numpy(preparer.base_env.agent.tcp_pose.p)[0]
                obj = numpy(preparer.base_env.cube.pose.p)[0]
                initial = np.r_[q, tcp, obj]
                assert np.linalg.norm(fk(q) - tcp) < 1e-5
                if pair_initial is None:
                    pair_initial = initial
                else:
                    np.testing.assert_allclose(initial, pair_initial, rtol=0, atol=1e-7)
                sampling = derive_atomic_sampling_seed(42424, seed, "reach")
                provider = lambda: current_relative_geometry(preparer.base_env)
                runtime = (
                    PositionRuntime(fk, provider, spec, sampling) if arm == "analytic" else
                    OracleGeometryRuntime(policy, normalizer, spec, "cuda", provider, RuntimeConfig(num_flow_steps=10, sampling_seed=sampling))
                )
                record = episode(preparer, runtime, spec, seed, prep)
                record.update(arm=arm, initial_state=initial.tolist())
                record["final_object_displacement_m"] = float(np.linalg.norm(
                    np.asarray(record["final_object_position_m"]) - obj,
                ))
                records.append(record)
                with (args.output / "episodes.jsonl").open("a") as file:
                    file.write(json.dumps(record) + "\n")
                print(json.dumps({k: record[k] for k in ["seed", "arm", "success", "steps", "errors"]}), flush=True)
    assert len(records) == 20
    summary = {}
    for arm in ("learned", "analytic"):
        rows = [r for r in records if r["arm"] == arm]
        summary[arm] = dict(
            episodes=len(rows), successes=sum(r["success"] for r in rows),
            mean_final_distance_m=float(np.mean([r["distance_trace_m"][-1] for r in rows])),
            mean_steps=float(np.mean([r["steps"] for r in rows])),
            errors=sum(bool(r["errors"]) for r in rows),
        )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
