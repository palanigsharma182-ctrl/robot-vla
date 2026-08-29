"""E012a 单 Episode Local DAgger live-takeover 采集入口。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="冻结 D0 与 ProprioStats")
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--boundary-type",
        choices=("reach_grasp", "grasp_lift"),
        required=True,
    )
    parser.add_argument("--qwen-context-layer", type=int, choices=(12, 24), default=12)
    parser.add_argument("--sampling-seed", type=int, default=52_012)
    parser.add_argument("--num-flow-steps", type=int, default=10)
    parser.add_argument("--recency-decay", type=float, default=0.5)
    parser.add_argument("--max-anomaly-replans", type=int, default=3)
    parser.add_argument("--trajectory-id")
    parser.add_argument(
        "--skip-snapshot-round-trip",
        action="store_true",
        help="仅用于工程调试；正式 smoke/collection 不应跳过 snapshot gate",
    )
    parser.add_argument(
        "--require-paired-clean-expert",
        action="store_true",
        help="正式候选必须运行相同 seed 的完整 clean Expert boundary diagnosis",
    )
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def derive_collection_sampling_seed(
    base_seed: int,
    *,
    environment_seed: int,
    boundary_type: str,
) -> int:
    if base_seed < 0 or environment_seed < 0:
        raise ValueError("sampling/environment seed 不能为负数")
    if boundary_type not in {"reach_grasp", "grasp_lift"}:
        raise ValueError(f"未知 boundary_type: {boundary_type}")
    identity = f"{base_seed}:{boundary_type}:{environment_seed}".encode()
    return int.from_bytes(hashlib.sha256(identity).digest()[:8], "big") % (2**63 - 1)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def run(args: argparse.Namespace) -> None:
    import torch

    from robot_vla.adapters import ProprioNormalizer, ProprioStats
    from robot_vla.cli.train_stage1 import compute_source_revision
    from robot_vla.contracts import RobotSpec
    from robot_vla.data.audit import audit_trajectory
    from robot_vla.data.trajectory import TrajectoryStore
    from robot_vla.model.factory import load_qwen_vla_policy
    from robot_vla.model.qwen_processor import QwenVLAProcessorAdapter
    from robot_vla.runtime import QwenVLARuntime, RuntimeConfig
    from robot_vla.sim.collector import EpisodeRejected
    from robot_vla.sim.local_dagger import LocalDaggerPickPlaceCollector
    from robot_vla.sim.local_dagger_risk import compute_paired_risk_components
    from robot_vla.training.checkpoint import load_stage1_policy_checkpoint

    if args.seed < 0 or args.sampling_seed < 0:
        raise ValueError("seed 不能为负数")
    if args.num_flow_steps <= 0 or not 0.0 < args.recency_decay < 1.0:
        raise ValueError("Flow/temporal 配置无效")
    if args.max_anomaly_replans < 0:
        raise ValueError("max_anomaly_replans 不能为负数")
    if not torch.cuda.is_available():
        raise RuntimeError("E012a Local DAgger collection 需要 CUDA")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"找不到 Checkpoint: {args.checkpoint}")
    if args.record.exists():
        raise FileExistsError(f"拒绝覆盖已有 collection record: {args.record}")

    spec = RobotSpec()
    stats = ProprioStats.from_json(args.data / "proprio_stats.json")
    stats.validate(spec)
    normalizer = ProprioNormalizer(stats, spec)
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
    checkpoint_sha256 = _sha256_file(args.checkpoint)
    checkpoint_metadata = load_stage1_policy_checkpoint(
        args.checkpoint,
        policy,
        spec,
        processor.config,
        stats,
    )
    episode_sampling_seed = derive_collection_sampling_seed(
        args.sampling_seed,
        environment_seed=args.seed,
        boundary_type=args.boundary_type,
    )
    runtime = QwenVLARuntime(
        policy,
        processor,
        normalizer,
        spec,
        "cuda",
        RuntimeConfig(
            num_flow_steps=args.num_flow_steps,
            use_bf16=True,
            sampling_seed=episode_sampling_seed,
        ),
    )

    project_root = Path(__file__).resolve().parents[3]
    source_revision = compute_source_revision(project_root)
    collection_config = {
        "environment_seed": args.seed,
        "boundary_type": args.boundary_type,
        "sampling_seed_base": args.sampling_seed,
        "episode_sampling_seed": episode_sampling_seed,
        "num_flow_steps": args.num_flow_steps,
        "recency_decay": args.recency_decay,
        "max_anomaly_replans": args.max_anomaly_replans,
        "qwen_context_layer": args.qwen_context_layer,
        "snapshot_round_trip_required": not args.skip_snapshot_round_trip,
        "paired_clean_expert_required": args.require_paired_clean_expert,
    }
    common_record = {
        "format": "robot-vla-local-dagger-collection/v1",
        "source_revision": source_revision,
        "base_dataset": str(args.data.resolve()),
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": checkpoint_sha256,
            "metadata": checkpoint_metadata,
        },
        "config": collection_config,
    }
    try:
        with LocalDaggerPickPlaceCollector(args.output, spec) as collector:
            result = collector.collect_local_dagger(
                runtime,
                seed=args.seed,
                boundary_type=args.boundary_type,
                policy_checkpoint_sha256=checkpoint_sha256,
                trajectory_id=args.trajectory_id,
                recency_decay=args.recency_decay,
                max_anomaly_replans=args.max_anomaly_replans,
                verify_snapshot_round_trip=not args.skip_snapshot_round_trip,
            )
        paired_clean_expert = None
        risk_components = None
        if args.require_paired_clean_expert:
            with LocalDaggerPickPlaceCollector(None, spec) as expert_collector:
                paired_clean_expert = expert_collector.collect_clean_expert_boundary(
                    seed=args.seed,
                    boundary_type=args.boundary_type,
                )
            risk_components = compute_paired_risk_components(
                args.boundary_type,
                result.boundary.to_dict(),
                paired_clean_expert.boundary.to_dict(),
            )
    except Exception as error:
        failed_record = {
            **common_record,
            "status": "rejected" if isinstance(error, EpisodeRejected) else "error",
            "failure": {
                "type": type(error).__name__,
                "reason": str(error),
            },
        }
        _atomic_write_json(args.record, failed_record)
        print(json.dumps(failed_record, sort_keys=True, allow_nan=False), flush=True)
        raise

    arrays = TrajectoryStore(args.output, spec, cache_size=0).get(result.meta)
    audit_trajectory(arrays, result.meta, spec)
    record = {
        **common_record,
        "status": "accepted",
        "result": result.to_dict(),
        "paired_clean_expert": (
            None
            if paired_clean_expert is None
            else paired_clean_expert.to_dict()
        ),
        "risk_components": risk_components,
        "eligible_for_risk_selection": (
            paired_clean_expert is not None and risk_components is not None
        ),
        "audit": {
            "trajectory_contract": "passed",
            "full_dataset_audit": "pending D0 union",
        },
    }
    _atomic_write_json(args.record, record)
    print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
