"""从正式 Stage 1 Checkpoint 运行可恢复的 test/unseen ManiSkill 闭环评估。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from robot_vla.data.trajectory import TrajectoryMeta, load_manifest
from robot_vla.evaluation.rollout import (
    ROLLOUT_FORMAT,
    RolloutEpisodeResult,
    RolloutEpisodeSpec,
    summarize_rollouts,
)
from robot_vla.tasks.pick_place import build_pick_place_task

EVALUATION_EXPERIMENT_FORMAT = "robot-vla-maniskill-evaluation/v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument(
        "--qwen-context-layer", type=int, choices=(12, 24), default=24
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--test-episodes",
        type=int,
        default=None,
        help="默认评估全部 test 轨迹；0 可用于只运行全新 seed 的 smoke test",
    )
    parser.add_argument("--unseen-seed-start", type=int, default=10_000)
    parser.add_argument("--unseen-episodes", type=int, default=20)
    parser.add_argument("--sampling-seed", type=int, default=42_424)
    parser.add_argument("--num-flow-steps", type=int, default=10)
    parser.add_argument(
        "--temporal-ensemble",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="启用重叠 Action Chunk 的 temporal ensemble；用 --no-temporal-ensemble 关闭",
    )
    parser.add_argument("--recency-decay", type=float, default=0.5)
    parser.add_argument(
        "--max-anomaly-replans",
        type=int,
        default=3,
        help="连续安全/推理异常可重规划次数；0 表示异常后立即失败",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _seed_from_meta(entry: TrajectoryMeta) -> int:
    seed = entry.randomization.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError(f"{entry.trajectory_id} 的 randomization.seed 无效")
    return seed


def build_episode_specs(
    data_root: Path,
    *,
    test_episodes: int | None,
    unseen_seed_start: int,
    unseen_episodes: int,
) -> list[RolloutEpisodeSpec]:
    if test_episodes is not None and test_episodes < 0:
        raise ValueError("test_episodes 必须非负或省略")
    if unseen_seed_start < 0 or unseen_episodes < 0:
        raise ValueError("unseen seed 范围无效")
    entries = load_manifest(data_root)
    dataset_seeds = [_seed_from_meta(entry) for entry in entries]
    if len(dataset_seeds) != len(set(dataset_seeds)):
        raise ValueError("Dataset manifest 存在重复环境 seed")
    test_entries = [entry for entry in entries if entry.split == "test"]
    if test_episodes is not None:
        test_entries = test_entries[:test_episodes]
    specs = [
        RolloutEpisodeSpec("test", _seed_from_meta(entry), entry.task.instruction)
        for entry in test_entries
    ]

    used_seeds = set(dataset_seeds)
    candidate = unseen_seed_start
    unseen_added = 0
    while unseen_added < unseen_episodes:
        if candidate not in used_seeds:
            specs.append(
                RolloutEpisodeSpec(
                    "unseen",
                    candidate,
                    build_pick_place_task(candidate % 3).instruction,
                )
            )
            unseen_added += 1
        candidate += 1
    if not specs:
        raise ValueError("至少需要一个 test 或 unseen Rollout Episode")
    return specs


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_audit_identity(data_root: Path) -> dict[str, Any]:
    path = data_root / "audit_report.json"
    if not path.is_file():
        raise FileNotFoundError("闭环评估要求数据集存在 audit_report.json")
    report = json.loads(path.read_text(encoding="utf-8"))
    manifest_path = data_root / "manifest.jsonl"
    if not manifest_path.is_file() or _sha256_file(manifest_path) != report.get(
        "manifest_sha256"
    ):
        raise ValueError("audit_report.json 已过期：manifest SHA256 不一致")
    if len(load_manifest(data_root)) != int(report["trajectory_count"]):
        raise ValueError("audit_report.json 已过期：trajectory_count 不一致")
    return {
        "dataset_sha256": report["dataset_sha256"],
        "manifest_sha256": report["manifest_sha256"],
        "trajectory_count": int(report["trajectory_count"]),
        "step_count": int(report["step_count"]),
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _read_existing_results(path: Path) -> list[RolloutEpisodeResult]:
    if not path.is_file():
        return []
    results: list[RolloutEpisodeResult] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            results.append(RolloutEpisodeResult.from_dict(json.loads(line)))
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
            raise ValueError(f"episodes.jsonl 第 {line_number} 行无效: {error}") from error
    return results


def run(args: argparse.Namespace) -> None:
    import torch

    from robot_vla.adapters import ProprioNormalizer, ProprioStats
    from robot_vla.cli.train_stage1 import compute_source_revision
    from robot_vla.contracts import RobotSpec
    from robot_vla.model.factory import load_qwen_vla_policy
    from robot_vla.model.qwen_processor import QwenVLAProcessorAdapter
    from robot_vla.training.checkpoint import load_stage1_policy_checkpoint

    if (
        args.num_flow_steps <= 0
        or args.sampling_seed < 0
        or not 0.0 < args.recency_decay < 1.0
        or args.max_anomaly_replans < 0
    ):
        raise ValueError("Flow、sampling seed 或控制层配置无效")
    if not torch.cuda.is_available():
        raise RuntimeError("ManiSkill 正式闭环评估需要 CUDA")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"找不到 Checkpoint: {args.checkpoint}")

    specs = build_episode_specs(
        args.data,
        test_episodes=args.test_episodes,
        unseen_seed_start=args.unseen_seed_start,
        unseen_episodes=args.unseen_episodes,
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
        "format": EVALUATION_EXPERIMENT_FORMAT,
        "rollout_format": ROLLOUT_FORMAT,
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
            "temporal_ensemble_enabled": args.temporal_ensemble,
            "recency_decay": args.recency_decay,
            "max_anomaly_replans": args.max_anomaly_replans,
            "qwen_context_layer": args.qwen_context_layer,
        },
        "episodes": [asdict(episode) for episode in specs],
    }
    # JSON 是磁盘上的实验身份；先标准化 tuple 等容器，保证 --resume 精确比较。
    experiment = json.loads(json.dumps(experiment, sort_keys=True, allow_nan=False))

    args.output.mkdir(parents=True, exist_ok=True)
    experiment_path = args.output / "experiment.json"
    episodes_path = args.output / "episodes.jsonl"
    summary_path = args.output / "summary.json"
    if args.resume:
        if not experiment_path.is_file():
            raise FileNotFoundError("--resume 要求输出目录存在 experiment.json")
        existing_experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
        if existing_experiment != experiment:
            raise ValueError("恢复评估时实验身份或 Episode 列表发生变化")
    else:
        if any(args.output.iterdir()):
            raise FileExistsError("评估输出目录非空；拒绝覆盖，请换目录或使用 --resume")
        _atomic_write_json(experiment_path, experiment)

    results = _read_existing_results(episodes_path)
    expected_identities = {(episode.seed_group, episode.seed) for episode in specs}
    completed = {(result.seed_group, result.seed) for result in results}
    if len(completed) != len(results) or not completed <= expected_identities:
        raise ValueError("已有 Rollout 结果重复或不属于当前实验")

    normalizer = ProprioNormalizer(stats, spec)
    from robot_vla.evaluation.maniskill import ManiSkillPickPlaceEvaluator

    with ManiSkillPickPlaceEvaluator(
        policy,
        processor,
        normalizer,
        spec,
        num_flow_steps=args.num_flow_steps,
        sampling_seed=args.sampling_seed,
        temporal_ensemble_enabled=args.temporal_ensemble,
        recency_decay=args.recency_decay,
        max_anomaly_replans=args.max_anomaly_replans,
    ) as evaluator:
        for episode in specs:
            identity = (episode.seed_group, episode.seed)
            if identity in completed:
                continue
            result = evaluator.evaluate(episode)
            with episodes_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result.to_dict(), sort_keys=True, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            results.append(result)
            completed.add(identity)
            partial_summary = summarize_rollouts(results)
            partial_summary["completed_episodes"] = len(results)
            partial_summary["expected_episodes"] = len(specs)
            _atomic_write_json(summary_path, partial_summary)
            print(json.dumps(result.to_dict(), sort_keys=True), flush=True)

    if completed != expected_identities:
        raise RuntimeError("闭环评估未产生全部预期 Episode")
    summary = summarize_rollouts(results)
    summary["completed_episodes"] = len(results)
    summary["expected_episodes"] = len(specs)
    summary["complete"] = True
    summary["checkpoint_sha256"] = experiment["checkpoint"]["sha256"]
    summary["dataset_sha256"] = experiment["dataset"]["dataset_sha256"]
    _atomic_write_json(summary_path, summary)
    print(json.dumps(summary, sort_keys=True), flush=True)


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
