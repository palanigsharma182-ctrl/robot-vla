"""从专家验证的前置状态独立评估五个 pick-place 原子技能。"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from robot_vla.cli.evaluate_maniskill import (
    _atomic_write_json,
    _load_audit_identity,
    _seed_from_meta,
    _sha256_file,
)
from robot_vla.contracts import PICK_AND_PLACE_SKILLS
from robot_vla.data.trajectory import load_manifest
from robot_vla.evaluation.atomic import (
    ATOMIC_ROLLOUT_FORMAT,
    AtomicSkillEpisodeResult,
    summarize_atomic_rollouts,
)
from robot_vla.execution.rtc import (
    ChunkInferenceStrategy,
    RTCConfig,
    resolve_inference_strategy,
)
from robot_vla.tasks.pick_place import build_pick_place_task

ATOMIC_EVALUATION_EXPERIMENT_FORMAT = "robot-vla-maniskill-atomic-evaluation/v1"


@dataclass(frozen=True)
class AtomicEvaluationSpec:
    seed: int
    skill_name: str
    instruction: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument(
        "--qwen-context-layer", type=int, choices=(12, 24), default=24
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=10_000)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--skills",
        nargs="+",
        choices=PICK_AND_PLACE_SKILLS,
        default=list(PICK_AND_PLACE_SKILLS),
    )
    parser.add_argument("--max-policy-steps", type=int, default=100)
    parser.add_argument("--sampling-seed", type=int, default=42_424)
    parser.add_argument("--num-flow-steps", type=int, default=10)
    parser.add_argument(
        "--inference-strategy",
        choices=tuple(strategy.value for strategy in ChunkInferenceStrategy),
        default=None,
    )
    parser.add_argument(
        "--temporal-ensemble",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="旧兼容开关；新实验请使用 --inference-strategy",
    )
    parser.add_argument("--recency-decay", type=float, default=0.5)
    parser.add_argument("--rtc-execution-horizon", type=int, default=4)
    parser.add_argument("--rtc-max-guidance-weight", type=float, default=10.0)
    parser.add_argument(
        "--max-anomaly-replans",
        type=int,
        default=3,
        help="连续安全/推理异常可重规划次数；0 表示异常后立即失败",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def build_atomic_specs(
    data_root: Path,
    *,
    seed_start: int,
    episodes: int,
    skills: tuple[str, ...] | list[str],
) -> list[AtomicEvaluationSpec]:
    if seed_start < 0 or episodes <= 0:
        raise ValueError("原子评估 seed_start 必须非负，episodes 必须为正数")
    selected_skills = tuple(skills)
    if not selected_skills or len(selected_skills) != len(set(selected_skills)):
        raise ValueError("原子评估 skills 不能为空或重复")
    if any(skill not in PICK_AND_PLACE_SKILLS for skill in selected_skills):
        raise ValueError("原子评估包含未知技能")
    dataset_seeds = {_seed_from_meta(entry) for entry in load_manifest(data_root)}
    selected_seeds: list[int] = []
    candidate = seed_start
    while len(selected_seeds) < episodes:
        if candidate not in dataset_seeds:
            selected_seeds.append(candidate)
        candidate += 1
    return [
        AtomicEvaluationSpec(
            seed=seed,
            skill_name=skill_name,
            instruction=build_pick_place_task(seed % 3).instruction,
        )
        for seed in selected_seeds
        for skill_name in selected_skills
    ]


def _read_existing_results(path: Path) -> list[AtomicSkillEpisodeResult]:
    if not path.is_file():
        return []
    results: list[AtomicSkillEpisodeResult] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            results.append(AtomicSkillEpisodeResult.from_dict(json.loads(line)))
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
            raise ValueError(f"原子 episodes.jsonl 第 {line_number} 行无效: {error}") from error
    return results


def run(args: argparse.Namespace) -> None:
    import torch

    from robot_vla.adapters import ProprioNormalizer, ProprioStats
    from robot_vla.cli.train_stage1 import compute_source_revision
    from robot_vla.contracts import RobotSpec
    from robot_vla.evaluation.maniskill import ManiSkillAtomicPickPlaceEvaluator
    from robot_vla.model.factory import load_qwen_vla_policy
    from robot_vla.model.qwen_processor import QwenVLAProcessorAdapter
    from robot_vla.training.checkpoint import load_stage1_policy_checkpoint

    strategy = resolve_inference_strategy(
        args.inference_strategy,
        legacy_temporal_ensemble_enabled=args.temporal_ensemble,
    )
    rtc_config = RTCConfig(
        execution_horizon=args.rtc_execution_horizon,
        max_guidance_weight=args.rtc_max_guidance_weight,
    )
    if (
        args.num_flow_steps <= 0
        or args.sampling_seed < 0
        or args.max_policy_steps <= 0
        or not 0.0 < args.recency_decay < 1.0
        or args.max_anomaly_replans < 0
    ):
        raise ValueError("原子评估 Flow steps、sampling seed 或 policy steps 无效")
    if not torch.cuda.is_available():
        raise RuntimeError("ManiSkill 正式原子评估需要 CUDA")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"找不到 Checkpoint: {args.checkpoint}")
    specs = build_atomic_specs(
        args.data,
        seed_start=args.seed_start,
        episodes=args.episodes,
        skills=args.skills,
    )
    spec = RobotSpec()
    stats = ProprioStats.from_json(args.data / "proprio_stats.json")
    stats.validate(spec)
    processor = QwenVLAProcessorAdapter.from_pretrained(
        cache_dir=str(args.model_cache),
        local_files_only=True,
    )
    policy = load_qwen_vla_policy(
        cache_dir=str(args.model_cache),
        local_files_only=True,
        device="cuda",
        context_layer=args.qwen_context_layer,
    )
    checkpoint_metadata = load_stage1_policy_checkpoint(
        args.checkpoint,
        policy,
        spec,
        processor.config,
        stats,
    )
    project_root = Path(__file__).resolve().parents[3]
    experiment = {
        "format": ATOMIC_EVALUATION_EXPERIMENT_FORMAT,
        "rollout_format": ATOMIC_ROLLOUT_FORMAT,
        "dataset": _load_audit_identity(args.data),
        "checkpoint": {
            "sha256": _sha256_file(args.checkpoint),
            "size_bytes": args.checkpoint.stat().st_size,
            "metadata": checkpoint_metadata,
        },
        "evaluation_code_revision": compute_source_revision(project_root),
        "config": {
            "environment_id": "RobotVLAPickCubeToRegion-v1",
            "control_mode": "pd_joint_delta_pos",
            "num_flow_steps": args.num_flow_steps,
            "sampling_seed": args.sampling_seed,
            "max_policy_steps": args.max_policy_steps,
            "preparation": "trusted-mplib-prerequisites/v1",
            "inference_strategy": strategy.value,
            "temporal_ensemble_enabled": (
                strategy == ChunkInferenceStrategy.TEMPORAL_ENSEMBLE
            ),
            "recency_decay": args.recency_decay,
            "rtc_execution_horizon": rtc_config.execution_horizon,
            "rtc_max_guidance_weight": rtc_config.max_guidance_weight,
            "rtc_schedule": rtc_config.schedule,
            "max_anomaly_replans": args.max_anomaly_replans,
            "qwen_context_layer": args.qwen_context_layer,
        },
        "episodes": [asdict(item) for item in specs],
    }
    experiment = json.loads(json.dumps(experiment, sort_keys=True, allow_nan=False))
    args.output.mkdir(parents=True, exist_ok=True)
    experiment_path = args.output / "experiment.json"
    episodes_path = args.output / "episodes.jsonl"
    summary_path = args.output / "summary.json"
    if args.resume:
        if not experiment_path.is_file():
            raise FileNotFoundError("--resume 要求原子评估输出目录存在 experiment.json")
        if json.loads(experiment_path.read_text(encoding="utf-8")) != experiment:
            raise ValueError("恢复原子评估时实验身份或 Episode 列表发生变化")
    else:
        if any(args.output.iterdir()):
            raise FileExistsError("原子评估输出目录非空；拒绝覆盖")
        _atomic_write_json(experiment_path, experiment)

    results = _read_existing_results(episodes_path)
    expected = {(item.skill_name, item.seed) for item in specs}
    completed = {(item.skill_name, item.seed) for item in results}
    if len(completed) != len(results) or not completed <= expected:
        raise ValueError("已有原子评估结果重复或不属于当前实验")
    normalizer = ProprioNormalizer(stats, spec)
    with ManiSkillAtomicPickPlaceEvaluator(
        policy,
        processor,
        normalizer,
        spec,
        num_flow_steps=args.num_flow_steps,
        sampling_seed=args.sampling_seed,
        inference_strategy=strategy,
        recency_decay=args.recency_decay,
        rtc_config=rtc_config,
        max_anomaly_replans=args.max_anomaly_replans,
    ) as evaluator:
        for item in specs:
            identity = (item.skill_name, item.seed)
            if identity in completed:
                continue
            result = evaluator.evaluate(
                seed=item.seed,
                skill_name=item.skill_name,
                instruction=item.instruction,
                max_policy_steps=args.max_policy_steps,
            )
            with episodes_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result.to_dict(), sort_keys=True, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            results.append(result)
            completed.add(identity)
            partial = summarize_atomic_rollouts(results)
            partial["completed_episodes"] = len(results)
            partial["expected_episodes"] = len(specs)
            _atomic_write_json(summary_path, partial)
            print(json.dumps(result.to_dict(), sort_keys=True), flush=True)
    if completed != expected:
        raise RuntimeError("原子评估未产生全部预期 Episode")
    summary = summarize_atomic_rollouts(results)
    summary.update(
        {
            "completed_episodes": len(results),
            "expected_episodes": len(specs),
            "complete": True,
            "checkpoint_sha256": experiment["checkpoint"]["sha256"],
            "dataset_sha256": experiment["dataset"]["dataset_sha256"],
        }
    )
    _atomic_write_json(summary_path, summary)
    print(json.dumps(summary, sort_keys=True), flush=True)


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
