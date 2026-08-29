"""E012a 正式 100-seed Local DAgger 候选池 runner。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from robot_vla.cli.collect_local_dagger import _sha256_file
from robot_vla.cli.evaluate_maniskill import _load_audit_identity
from robot_vla.cli.train_stage1 import compute_source_revision
from robot_vla.data.trajectory import load_manifest
from robot_vla.sim.local_dagger_risk import (
    RISK_COMPONENT_UNITS,
    RISK_CONTRACT_VERSION,
    score_and_select_risk_candidates,
)

POOL_FORMAT = "robot-vla-local-dagger-pool/v1"
FORMAL_SEED_RANGES = {
    "reach_grasp": (30_000, 30_100),
    "grasp_lift": (30_100, 30_200),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--boundary-type",
        choices=tuple(FORMAL_SEED_RANGES),
        required=True,
    )
    parser.add_argument("--qwen-context-layer", type=int, choices=(12, 24), default=12)
    parser.add_argument("--sampling-seed", type=int, default=52_012)
    parser.add_argument("--num-flow-steps", type=int, default=10)
    parser.add_argument("--recency-decay", type=float, default=0.5)
    parser.add_argument("--max-anomaly-replans", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build_pool_identity(
    args: argparse.Namespace,
    *,
    source_revision: str,
    checkpoint_sha256: str,
    base_dataset_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    start, end = FORMAL_SEED_RANGES[args.boundary_type]
    return {
        "format": POOL_FORMAT,
        "source_revision": source_revision,
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": checkpoint_sha256,
        },
        "base_dataset": {
            "path": str(args.data.resolve()),
            "audit": base_dataset_identity,
        },
        "model_cache": str(args.model_cache.resolve()),
        "boundary_type": args.boundary_type,
        "environment_seeds": list(range(start, end)),
        "config": {
            "inference_strategy": "temporal-ensemble",
            "qwen_context_layer": args.qwen_context_layer,
            "sampling_seed_base": args.sampling_seed,
            "num_flow_steps": args.num_flow_steps,
            "recency_decay": args.recency_decay,
            "max_anomaly_replans": args.max_anomaly_replans,
            "snapshot_round_trip_required": True,
            "paired_clean_expert_required": True,
        },
        "risk": {
            "version": RISK_CONTRACT_VERSION,
            "component_units": dict(RISK_COMPONENT_UNITS[args.boundary_type]),
            "percentile": "zero-based mid-rank / (eligible_count - 1)",
            "score": "unweighted arithmetic mean of component percentiles",
            "selection": {
                "high_count": 14,
                "low_count": 6,
                "tie_break": "environment_seed ascending",
                "overlap_resolution": "select high first, then low from remaining",
            },
        },
    }


def _load_record(path: Path, identity: dict[str, Any], seed: int) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("format") != "robot-vla-local-dagger-collection/v1":
        raise ValueError(f"seed {seed}: candidate record format 不兼容")
    config = record.get("config", {})
    expected_config = identity["config"]
    checks = {
        "environment_seed": seed,
        "boundary_type": identity["boundary_type"],
        "sampling_seed_base": expected_config["sampling_seed_base"],
        "num_flow_steps": expected_config["num_flow_steps"],
        "recency_decay": expected_config["recency_decay"],
        "max_anomaly_replans": expected_config["max_anomaly_replans"],
        "qwen_context_layer": expected_config["qwen_context_layer"],
        "snapshot_round_trip_required": True,
        "paired_clean_expert_required": True,
    }
    for key, expected in checks.items():
        if config.get(key) != expected:
            raise ValueError(f"seed {seed}: candidate config {key} 漂移")
    if record.get("source_revision") != identity["source_revision"]:
        raise ValueError(f"seed {seed}: source revision 漂移")
    if record.get("checkpoint", {}).get("sha256") != identity["checkpoint"]["sha256"]:
        raise ValueError(f"seed {seed}: checkpoint identity 漂移")
    return record


def compact_candidate_record(record: dict[str, Any], record_path: Path) -> dict[str, Any]:
    config = record["config"]
    row: dict[str, Any] = {
        "environment_seed": int(config["environment_seed"]),
        "boundary_type": str(config["boundary_type"]),
        "status": str(record["status"]),
        "record": str(record_path.resolve()),
        "episode_sampling_seed": int(config["episode_sampling_seed"]),
    }
    if record["status"] == "accepted":
        result = record["result"]
        row.update(
            {
                "trajectory_id": result["trajectory"]["trajectory_id"],
                "dataset_root": str((record_path.parent / "dataset").resolve()),
                "expert_takeover_step": result["trajectory"]["local_dagger"][
                    "expert_takeover_step"
                ],
                "policy_boundary": result["boundary"],
                "paired_clean_expert_boundary": record["paired_clean_expert"][
                    "boundary"
                ],
                "risk_components": record["risk_components"],
                "snapshot_round_trip_passed": result["snapshot_round_trip"][
                    "passed"
                ],
                "trajectory_audit": record["audit"]["trajectory_contract"],
                "eligible_for_risk_selection": record[
                    "eligible_for_risk_selection"
                ],
            }
        )
    else:
        row["failure"] = record.get("failure")
        row["eligible_for_risk_selection"] = False
    return row


def _candidate_command(
    args: argparse.Namespace,
    *,
    seed: int,
    candidate_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "robot_vla.cli.collect_local_dagger",
        "--data",
        str(args.data.resolve()),
        "--model-cache",
        str(args.model_cache.resolve()),
        "--checkpoint",
        str(args.checkpoint.resolve()),
        "--output",
        str((candidate_dir / "dataset").resolve()),
        "--record",
        str((candidate_dir / "record.json").resolve()),
        "--seed",
        str(seed),
        "--boundary-type",
        args.boundary_type,
        "--qwen-context-layer",
        str(args.qwen_context_layer),
        "--sampling-seed",
        str(args.sampling_seed),
        "--num-flow-steps",
        str(args.num_flow_steps),
        "--recency-decay",
        str(args.recency_decay),
        "--max-anomaly-replans",
        str(args.max_anomaly_replans),
        "--require-paired-clean-expert",
    ]


def _write_progress(output: Path, rows: list[dict[str, Any]], expected: int) -> None:
    ordered = sorted(rows, key=lambda row: row["environment_seed"])
    _atomic_write_jsonl(output / "collection_candidates.jsonl", ordered)
    rejection_reasons = Counter(
        row.get("failure", {}).get("reason", "unknown")
        for row in ordered
        if row["status"] == "rejected"
    )
    statuses = Counter(row["status"] for row in ordered)
    _atomic_write_json(
        output / "collection_summary.json",
        {
            "format": POOL_FORMAT,
            "scan_complete": len(ordered) == expected,
            "expected_candidates": expected,
            "completed_candidates": len(ordered),
            "status_counts": dict(sorted(statuses.items())),
            "rejection_reasons": dict(sorted(rejection_reasons.items())),
            "eligible_for_risk_selection": sum(
                bool(row["eligible_for_risk_selection"]) for row in ordered
            ),
        },
    )


def run(args: argparse.Namespace) -> None:
    if args.sampling_seed < 0 or args.num_flow_steps <= 0:
        raise ValueError("sampling/Flow 配置无效")
    if not 0.0 < args.recency_decay < 1.0 or args.max_anomaly_replans < 0:
        raise ValueError("temporal/anomaly 配置无效")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"找不到 checkpoint: {args.checkpoint}")
    project_root = Path(__file__).resolve().parents[3]
    identity = build_pool_identity(
        args,
        source_revision=compute_source_revision(project_root),
        checkpoint_sha256=_sha256_file(args.checkpoint),
        base_dataset_identity=_load_audit_identity(args.data),
    )
    base_seeds = {
        int(entry.randomization["seed"])
        for entry in load_manifest(args.data)
        if "seed" in entry.randomization
    }
    overlap = base_seeds.intersection(identity["environment_seeds"])
    if overlap:
        raise ValueError(f"正式 collection seeds 与 D0 重叠: {sorted(overlap)}")
    experiment_path = args.output / "experiment.json"
    if args.resume:
        if not experiment_path.is_file():
            raise FileNotFoundError("--resume 要求既有 experiment.json")
        existing = json.loads(experiment_path.read_text(encoding="utf-8"))
        if existing != identity:
            raise ValueError("--resume experiment identity 漂移")
    else:
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError("正式 pool 输出目录非空；拒绝覆盖")
        args.output.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(experiment_path, identity)

    rows: list[dict[str, Any]] = []
    seeds = identity["environment_seeds"]
    for seed in seeds:
        candidate_dir = args.output / "candidates" / f"seed-{seed:06d}"
        record_path = candidate_dir / "record.json"
        if record_path.is_file():
            if not args.resume:
                raise FileExistsError(f"seed {seed}: record 已存在")
            record = _load_record(record_path, identity, seed)
        else:
            if candidate_dir.exists() and any(candidate_dir.iterdir()):
                raise RuntimeError(f"seed {seed}: 存在无 record 的 partial candidate")
            candidate_dir.mkdir(parents=True, exist_ok=True)
            command = _candidate_command(
                args,
                seed=seed,
                candidate_dir=candidate_dir,
            )
            environment = os.environ.copy()
            environment.setdefault("HF_HUB_OFFLINE", "1")
            environment.setdefault("TRANSFORMERS_OFFLINE", "1")
            with (
                (candidate_dir / "stdout.log").open("w", encoding="utf-8") as stdout,
                (candidate_dir / "stderr.log").open("w", encoding="utf-8") as stderr,
            ):
                completed = subprocess.run(
                    command,
                    cwd=project_root,
                    env=environment,
                    stdout=stdout,
                    stderr=stderr,
                    check=False,
                )
            if not record_path.is_file():
                raise RuntimeError(
                    f"seed {seed}: subprocess exit={completed.returncode} 且未写 record"
                )
            record = _load_record(record_path, identity, seed)
            if record["status"] == "accepted" and completed.returncode != 0:
                raise RuntimeError(f"seed {seed}: accepted record 与 subprocess exit 冲突")
            if record["status"] == "error":
                raise RuntimeError(f"seed {seed}: 工程错误：{record['failure']}")
        row = compact_candidate_record(record, record_path)
        rows.append(row)
        _write_progress(args.output, rows, len(seeds))
        print(
            json.dumps(
                {
                    "event": "candidate_complete",
                    "seed": seed,
                    "status": row["status"],
                    "completed": len(rows),
                    "expected": len(seeds),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    eligible = [row for row in rows if row["eligible_for_risk_selection"]]
    summary_path = args.output / "collection_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if len(eligible) < 20:
        summary["selection_gate_passed"] = False
        summary["selection_failure"] = "eligible candidates 少于 20"
        _atomic_write_json(summary_path, summary)
        return
    selection = score_and_select_risk_candidates(args.boundary_type, eligible)
    _atomic_write_json(args.output / "risk_selection.json", selection.to_dict())
    summary.update(
        {
            "selection_gate_passed": True,
            "high_risk_seeds": list(selection.high_risk_seeds),
            "low_risk_seeds": list(selection.low_risk_seeds),
            "selected_count": 20,
        }
    )
    _atomic_write_json(summary_path, summary)


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
