"""E012a 单 Episode Local DAgger live-takeover 采集入口。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from typing_extensions import Self

from robot_vla.local_dagger_protocol import (
    LOCAL_DAGGER_ACTION_BUDGET_PROTOCOL_FIELD,
    LOCAL_DAGGER_ACTION_BUDGET_USAGE_FIELD,
    LocalDaggerActionBudgetProtocol,
    resolve_local_dagger_action_budget,
)

_CANDIDATE_STAGING_FORMAT = "robot-vla-candidate-dataset-staging/v1"
_CANDIDATE_STAGING_MARKER = ".candidate-dataset-staging.json"


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
    parser.add_argument(
        "--action-budget-protocol",
        choices=tuple(item.value for item in LocalDaggerActionBudgetProtocol),
        default=LocalDaggerActionBudgetProtocol.LEGACY.value,
        help="默认 legacy；E012 amendment 使用 segmented-300-180-480",
    )
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


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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


class _CandidateDatasetPublisher:
    """把单个 candidate 数据集隔离到 sibling staging，再原子发布。"""

    def __init__(self, canonical_root: Path) -> None:
        self.canonical_root = canonical_root.resolve()
        if not self.canonical_root.name:
            raise ValueError("candidate dataset canonical root 无效")
        self.staging_root: Path | None = None
        self._published = False
        self._commit_prepared = False
        self._committed = False
        self._published_identity: tuple[int, int] | None = None

    @property
    def root(self) -> Path:
        if self.staging_root is None:
            raise RuntimeError("candidate staging 尚未创建")
        return self.staging_root

    def __enter__(self) -> Self:
        parent = self.canonical_root.parent
        parent.mkdir(parents=True, exist_ok=True)
        if self.canonical_root.exists():
            raise FileExistsError(
                f"拒绝覆盖已有 canonical candidate dataset: {self.canonical_root}"
            )
        prefix = f".{self.canonical_root.name}.candidate-staging-"
        stale = sorted(
            (item for item in parent.iterdir() if item.name.startswith(prefix)),
            key=lambda item: item.name,
        )
        if stale:
            names = ", ".join(item.name for item in stale)
            raise RuntimeError(
                "检测到未完成 candidate staging；拒绝静默复用或覆盖："
                f"{names}"
            )
        self.staging_root = Path(tempfile.mkdtemp(prefix=prefix, dir=parent))
        _atomic_write_json(
            self.staging_root / _CANDIDATE_STAGING_MARKER,
            {
                "format": _CANDIDATE_STAGING_FORMAT,
                "canonical_root": str(self.canonical_root),
                "staging_directory": self.staging_root.name,
                "state": "uncommitted",
            },
        )
        return self

    def publish(self) -> None:
        if self.staging_root is None:
            raise RuntimeError("candidate staging 尚未创建")
        if self._published or self._committed:
            raise RuntimeError("candidate dataset 已进入发布流程")
        if self.canonical_root.exists():
            raise FileExistsError(
                f"拒绝覆盖并发创建的 canonical candidate dataset: {self.canonical_root}"
            )
        if not (self.staging_root / "manifest.jsonl").is_file():
            raise RuntimeError("candidate staging 缺少 manifest.jsonl")
        os.replace(self.staging_root, self.canonical_root)
        self._published = True
        stat = self.canonical_root.stat()
        self._published_identity = (stat.st_dev, stat.st_ino)

    def prepare_commit(self) -> None:
        """先移除未提交标记，使 accepted record 成为最后一个原子写。"""

        if not self._published or self._commit_prepared or self._committed:
            raise RuntimeError("candidate dataset 尚未发布或已进入提交阶段")
        marker = self.canonical_root / _CANDIDATE_STAGING_MARKER
        marker.unlink()
        self._commit_prepared = True

    def commit(self) -> None:
        if not self._commit_prepared or self._committed:
            raise RuntimeError("candidate dataset 尚未准备提交或已经提交")
        self._committed = True

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._committed:
            return
        root = self.canonical_root if self._published else self.staging_root
        if root is None or not root.exists():
            return
        if self._published:
            stat = root.stat()
            if (stat.st_dev, stat.st_ino) != self._published_identity:
                raise RuntimeError(
                    f"拒绝清理 identity 漂移的 canonical candidate dataset: {root}"
                )
        if not self._commit_prepared:
            marker = root / _CANDIDATE_STAGING_MARKER
            if not marker.is_file():
                raise RuntimeError(
                    f"拒绝清理缺少 ownership marker 的 candidate dataset: {root}"
                )
            payload = json.loads(marker.read_text(encoding="utf-8"))
            expected = {
                "format": _CANDIDATE_STAGING_FORMAT,
                "canonical_root": str(self.canonical_root),
                "staging_directory": self.root.name,
                "state": "uncommitted",
            }
            if payload != expected:
                raise RuntimeError(
                    f"拒绝清理 ownership marker 漂移的 candidate dataset: {root}"
                )
        shutil.rmtree(root)


def _validate_candidate_publish_gates(
    *,
    snapshot_round_trip: Any,
    snapshot_required: bool,
    paired_clean_expert: Any,
    paired_required: bool,
    risk_components: dict[str, float] | None,
) -> bool:
    """发布前显式闭合 snapshot、paired Expert 与 risk eligibility gate。"""

    if snapshot_required and (
        snapshot_round_trip is None or snapshot_round_trip.passed is not True
    ):
        raise RuntimeError("正式 candidate 缺少通过的 boundary snapshot round-trip")
    eligible = (
        paired_clean_expert is not None
        and paired_clean_expert.task_completed is True
        and risk_components is not None
    )
    if paired_required and not eligible:
        raise RuntimeError("正式 candidate 未通过 paired clean Expert / risk eligibility gate")
    return eligible


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
    from robot_vla.sim.local_dagger_diagnostics import LocalDaggerFailureDiagnostics
    from robot_vla.sim.local_dagger_risk import compute_paired_risk_components
    from robot_vla.training.checkpoint import load_stage1_policy_checkpoint

    if args.seed < 0 or args.sampling_seed < 0:
        raise ValueError("seed 不能为负数")
    if args.num_flow_steps <= 0 or not 0.0 < args.recency_decay < 1.0:
        raise ValueError("Flow/temporal 配置无效")
    if args.max_anomaly_replans < 0:
        raise ValueError("max_anomaly_replans 不能为负数")
    action_budget = resolve_local_dagger_action_budget(
        getattr(
            args,
            "action_budget_protocol",
            LocalDaggerActionBudgetProtocol.LEGACY.value,
        )
    )
    if not torch.cuda.is_available():
        raise RuntimeError("E012a Local DAgger collection 需要 CUDA")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"找不到 Checkpoint: {args.checkpoint}")
    if args.record.exists():
        raise FileExistsError(f"拒绝覆盖已有 collection record: {args.record}")

    spec = RobotSpec()
    stats_path = args.data / "proprio_stats.json"
    stats_sha256_before = _sha256_file(stats_path)
    stats = ProprioStats.from_json(stats_path)
    stats.validate(spec)
    stats_sha256_after = _sha256_file(stats_path)
    if stats_sha256_after != stats_sha256_before:
        raise RuntimeError("proprio_stats.json 在读取期间发生变化")
    stats_receipt = {
        "proprio_stats_sha256": stats_sha256_after,
        "proprio_stats_semantic_sha256": _canonical_json_sha256(asdict(stats)),
    }
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
    action_budget_plan_metadata = action_budget.planned_metadata()
    if action_budget_plan_metadata is not None:
        collection_config[LOCAL_DAGGER_ACTION_BUDGET_PROTOCOL_FIELD] = dict(
            action_budget_plan_metadata
        )
    common_record = {
        "format": "robot-vla-local-dagger-collection/v1",
        "source_revision": source_revision,
        "base_dataset": str(args.data.resolve()),
        "base_dataset_receipt": stats_receipt,
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": checkpoint_sha256,
            "metadata": checkpoint_metadata,
        },
        "config": collection_config,
    }
    failure_diagnostics = LocalDaggerFailureDiagnostics(
        environment_seed=args.seed,
        boundary_type=args.boundary_type,
        action_budget_protocol=action_budget.protocol,
    )
    try:
        with _CandidateDatasetPublisher(args.output) as publisher:
            collector_options: dict[str, Any] = {}
            if action_budget.amended:
                collector_options["action_budget_protocol"] = action_budget.protocol
            with LocalDaggerPickPlaceCollector(
                publisher.root,
                spec,
                **collector_options,
            ) as collector:
                result = collector.collect_local_dagger(
                    runtime,
                    seed=args.seed,
                    boundary_type=args.boundary_type,
                    policy_checkpoint_sha256=checkpoint_sha256,
                    trajectory_id=args.trajectory_id,
                    recency_decay=args.recency_decay,
                    max_anomaly_replans=args.max_anomaly_replans,
                    verify_snapshot_round_trip=not args.skip_snapshot_round_trip,
                    failure_diagnostics=failure_diagnostics,
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

            arrays = TrajectoryStore(publisher.root, spec, cache_size=0).get(result.meta)
            audit_trajectory(arrays, result.meta, spec)
            eligible_for_risk_selection = _validate_candidate_publish_gates(
                snapshot_round_trip=result.snapshot_round_trip,
                snapshot_required=not args.skip_snapshot_round_trip,
                paired_clean_expert=paired_clean_expert,
                paired_required=args.require_paired_clean_expert,
                risk_components=risk_components,
            )
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
                "eligible_for_risk_selection": eligible_for_risk_selection,
                "audit": {
                    "trajectory_contract": "passed",
                    "full_dataset_audit": "pending D0 union",
                },
            }
            if result.action_budget_usage is not None:
                record[LOCAL_DAGGER_ACTION_BUDGET_USAGE_FIELD] = dict(
                    result.action_budget_usage
                )

            # canonical dataset 只在全部 gate 闭合后出现；record 写入或 marker
            # 提交失败时，context manager 会回滚本次 canonical dataset。
            publisher.publish()
            publisher.prepare_commit()
            _atomic_write_json(args.record, record)
            publisher.commit()
    except Exception as error:
        failed_record = {
            **common_record,
            "status": "rejected" if isinstance(error, EpisodeRejected) else "error",
            "failure": {
                "type": type(error).__name__,
                "reason": str(error),
            },
            "failure_diagnostics": failure_diagnostics.to_dict(
                failure_reason=str(error),
            ),
        }
        action_budget_usage = action_budget.usage_metadata(
            total_actions=failure_diagnostics.action_count,
            expert_takeover_step=failure_diagnostics.expert_takeover_step,
        )
        if action_budget_usage is not None:
            failed_record[LOCAL_DAGGER_ACTION_BUDGET_USAGE_FIELD] = action_budget_usage
        _atomic_write_json(args.record, failed_record)
        print(json.dumps(failed_record, sort_keys=True, allow_nan=False), flush=True)
        raise
    print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
