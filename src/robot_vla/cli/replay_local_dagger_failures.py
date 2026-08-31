"""E012 Grasp→Lift 正式失败的可恢复、无干预 forensic replay runner。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from robot_vla.sim.local_dagger_diagnostics import (
    EPISODE_TIME_LIMIT_REASON,
    LOCAL_DAGGER_DIAGNOSTIC_FORMAT,
    POLICY_BEFORE_BOUNDARY_REASON,
)

REPLAY_FORMAT = "robot-vla-local-dagger-failure-replay/v1"
REPLAY_CANDIDATE_FORMAT = "robot-vla-local-dagger-failure-replay-candidate/v1"
SUBPROCESS_FORMAT = "robot-vla-local-dagger-failure-replay-subprocess/v1"
COLLECTION_FORMAT = "robot-vla-local-dagger-collection/v1"
POOL_FORMAT = "robot-vla-local-dagger-pool/v1"
TARGET_BOUNDARY_TYPE = "grasp_lift"
TARGET_FAILURE_REASONS = (
    POLICY_BEFORE_BOUNDARY_REASON,
    EPISODE_TIME_LIMIT_REASON,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formal",
        type=Path,
        required=True,
        help="原 E012a Grasp→Lift formal pool 根目录",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="独立 forensic replay 输出根目录",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
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


def _atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
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


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"缺少 {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} 不是有效 JSON: {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{label} 顶层必须是 JSON object: {path}")
    return value


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"缺少 {label}: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{label} 第 {line_number} 行为空")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{label} 第 {line_number} 行不是有效 JSON"
                ) from error
            if not isinstance(row, dict):
                raise TypeError(f"{label} 第 {line_number} 行不是 JSON object")
            rows.append(row)
    return rows


def _nonnegative_int(value: Any, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} 必须是非负整数")
    return value


def select_replay_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_seeds: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """验证 formal index 并选出两类可识别的 rejected failures。"""

    by_seed: dict[int, dict[str, Any]] = {}
    for index, value in enumerate(rows):
        row = dict(value)
        seed = _nonnegative_int(
            row.get("environment_seed"),
            label=f"formal row {index} environment_seed",
        )
        if seed in by_seed:
            raise ValueError(f"formal collection_candidates 存在重复 seed: {seed}")
        if row.get("boundary_type") != TARGET_BOUNDARY_TYPE:
            raise ValueError(f"seed {seed}: formal boundary_type 不是 grasp_lift")
        status = row.get("status")
        if status not in {"accepted", "rejected"}:
            raise RuntimeError(f"seed {seed}: formal pool 含工程 status={status!r}")
        failure = row.get("failure")
        if status == "rejected" and not isinstance(failure, dict):
            raise ValueError(f"seed {seed}: rejected formal row 缺少 failure")
        reason = failure.get("reason") if isinstance(failure, dict) else None
        if reason in TARGET_FAILURE_REASONS and status != "rejected":
            raise ValueError(f"seed {seed}: 目标 failure reason 与 status 冲突")
        by_seed[seed] = row

    if expected_seeds is not None:
        normalized_expected = [
            _nonnegative_int(value, label="formal experiment environment_seed")
            for value in expected_seeds
        ]
        if len(normalized_expected) != len(set(normalized_expected)):
            raise ValueError("formal experiment environment_seeds 存在重复")
        if set(by_seed) != set(normalized_expected):
            missing = sorted(set(normalized_expected) - set(by_seed))
            extra = sorted(set(by_seed) - set(normalized_expected))
            raise ValueError(
                "formal collection_candidates 与 experiment seed 集合不一致: "
                f"missing={missing}, extra={extra}"
            )

    selected = [
        row
        for seed, row in sorted(by_seed.items())
        if row["status"] == "rejected"
        and row["failure"]["reason"] in TARGET_FAILURE_REASONS
    ]
    if not selected:
        raise ValueError("formal pool 中没有可重放的目标 failure")
    return selected


def validate_original_record(
    row: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    formal_experiment: Mapping[str, Any],
    expected_record_path: Path,
) -> None:
    """Fail closed 地对齐 formal compact row、record 与 pool identity。"""

    seed = _nonnegative_int(row.get("environment_seed"), label="environment_seed")
    row_record = row.get("record")
    if not isinstance(row_record, str):
        raise TypeError(f"seed {seed}: formal row 缺少 record path")
    if Path(row_record).resolve() != expected_record_path.resolve():
        raise ValueError(f"seed {seed}: formal row record path 不属于指定 formal pool")
    if record.get("format") != COLLECTION_FORMAT:
        raise ValueError(f"seed {seed}: original collection record format 不兼容")
    if record.get("status") != "rejected" or row.get("status") != "rejected":
        raise ValueError(f"seed {seed}: original status 不是 rejected")
    original_failure = record.get("failure")
    if not isinstance(original_failure, dict):
        raise TypeError(f"seed {seed}: original record 缺少 failure")
    if row.get("failure") != original_failure:
        raise ValueError(f"seed {seed}: formal row 与 original record failure 漂移")
    if original_failure.get("reason") not in TARGET_FAILURE_REASONS:
        raise ValueError(f"seed {seed}: original reason 不在 forensic replay 范围")

    config = record.get("config")
    if not isinstance(config, dict):
        raise TypeError(f"seed {seed}: original record 缺少 config")
    if config.get("environment_seed") != seed:
        raise ValueError(f"seed {seed}: original config environment_seed 漂移")
    if config.get("boundary_type") != TARGET_BOUNDARY_TYPE:
        raise ValueError(f"seed {seed}: original config boundary_type 漂移")
    if row.get("boundary_type") != config["boundary_type"]:
        raise ValueError(f"seed {seed}: formal row boundary_type 漂移")
    if row.get("episode_sampling_seed") != config.get("episode_sampling_seed"):
        raise ValueError(f"seed {seed}: episode_sampling_seed 漂移")

    pool_config = formal_experiment.get("config")
    if not isinstance(pool_config, dict):
        raise TypeError("formal experiment 缺少 config")
    config_mapping = {
        "qwen_context_layer": "qwen_context_layer",
        "sampling_seed_base": "sampling_seed_base",
        "num_flow_steps": "num_flow_steps",
        "recency_decay": "recency_decay",
        "max_anomaly_replans": "max_anomaly_replans",
        "snapshot_round_trip_required": "snapshot_round_trip_required",
        "paired_clean_expert_required": "paired_clean_expert_required",
    }
    for record_key, pool_key in config_mapping.items():
        if config.get(record_key) != pool_config.get(pool_key):
            raise ValueError(f"seed {seed}: original config {record_key} 与 pool 漂移")

    checkpoint = record.get("checkpoint")
    pool_checkpoint = formal_experiment.get("checkpoint")
    if not isinstance(checkpoint, dict) or not isinstance(pool_checkpoint, dict):
        raise TypeError(f"seed {seed}: checkpoint identity 缺失")
    for key in ("path", "sha256"):
        if checkpoint.get(key) != pool_checkpoint.get(key):
            raise ValueError(f"seed {seed}: original checkpoint {key} 与 pool 漂移")

    base_dataset = record.get("base_dataset")
    pool_dataset = formal_experiment.get("base_dataset")
    pool_dataset_path = (
        pool_dataset.get("path") if isinstance(pool_dataset, dict) else None
    )
    if base_dataset != pool_dataset_path:
        raise ValueError(f"seed {seed}: original base_dataset 与 pool 漂移")


def build_replay_experiment(
    formal_root: Path,
    *,
    replay_source_revision: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """仅读取 JSON/JSONL 构建完整可审计的 replay identity 与 targets。"""

    formal_root = formal_root.resolve()
    formal_experiment_path = formal_root / "experiment.json"
    formal_candidates_path = formal_root / "collection_candidates.jsonl"
    formal_experiment = _read_json(
        formal_experiment_path,
        label="formal experiment.json",
    )
    if formal_experiment.get("format") != POOL_FORMAT:
        raise ValueError("formal experiment format 不兼容")
    if formal_experiment.get("boundary_type") != TARGET_BOUNDARY_TYPE:
        raise ValueError("forensic replay 首版只接受 grasp_lift formal pool")
    expected_seeds = formal_experiment.get("environment_seeds")
    if not isinstance(expected_seeds, list):
        raise TypeError("formal experiment 缺少 environment_seeds")
    model_cache = formal_experiment.get("model_cache")
    if not isinstance(model_cache, str) or not model_cache:
        raise ValueError("formal experiment 缺少 model_cache")

    rows = _read_jsonl(
        formal_candidates_path,
        label="formal collection_candidates.jsonl",
    )
    selected_rows = select_replay_rows(rows, expected_seeds=expected_seeds)
    targets: list[dict[str, Any]] = []
    frozen_targets: list[dict[str, Any]] = []
    for row in selected_rows:
        seed = int(row["environment_seed"])
        record_path = formal_root / "candidates" / f"seed-{seed:06d}" / "record.json"
        record = _read_json(record_path, label=f"seed {seed} original record")
        validate_original_record(
            row,
            record,
            formal_experiment=formal_experiment,
            expected_record_path=record_path,
        )
        original_record_sha256 = _sha256_file(record_path)
        target = {
            "environment_seed": seed,
            "boundary_type": TARGET_BOUNDARY_TYPE,
            "model_cache": model_cache,
            "replay_source_revision": replay_source_revision,
            "original_row": dict(row),
            "original_record": dict(record),
            "original_record_path": str(record_path.resolve()),
            "original_record_sha256": original_record_sha256,
        }
        targets.append(target)
        frozen_targets.append(
            {
                "environment_seed": seed,
                "boundary_type": TARGET_BOUNDARY_TYPE,
                "status": record["status"],
                "failure": record["failure"],
                "source_revision": record.get("source_revision"),
                "base_dataset": record["base_dataset"],
                "checkpoint": record["checkpoint"],
                "config": record["config"],
                "record": str(record_path.resolve()),
                "record_sha256": original_record_sha256,
            }
        )

    experiment = {
        "format": REPLAY_FORMAT,
        "purpose": "exploratory GL failure decomposition; not training data",
        "source_revision": replay_source_revision,
        "replay_source_revision": replay_source_revision,
        "formal_source_revision": formal_experiment.get("source_revision"),
        "checkpoint": formal_experiment["checkpoint"],
        "base_dataset": formal_experiment["base_dataset"],
        "model_cache": model_cache,
        "config": formal_experiment["config"],
        "environment_seeds": [
            target["environment_seed"] for target in frozen_targets
        ],
        "formal_pool": {
            "path": str(formal_root),
            "experiment": formal_experiment,
            "experiment_sha256": _sha256_file(formal_experiment_path),
            "collection_candidates": str(formal_candidates_path.resolve()),
            "collection_candidates_sha256": _sha256_file(formal_candidates_path),
        },
        "target_boundary_type": TARGET_BOUNDARY_TYPE,
        "target_failure_reasons": list(TARGET_FAILURE_REASONS),
        "selected_count": len(frozen_targets),
        "selected_candidates": frozen_targets,
        "execution": {
            "entrypoint": "robot_vla.cli.collect_local_dagger",
            "offline_environment": True,
            "resume_contract": "completed candidate directories are immutable",
            "trajectory_usage": "diagnostic replay only; forbidden as training data",
        },
    }
    # 标准化 tuple 等容器，保证磁盘身份与 --resume 比较一致。
    normalized = json.loads(json.dumps(experiment, sort_keys=True, allow_nan=False))
    return normalized, targets


def candidate_command(
    target: Mapping[str, Any],
    *,
    candidate_dir: Path,
    python_executable: str = sys.executable,
) -> list[str]:
    """从原 record 生成重放命令，不接受新的推理配置。"""

    record = target["original_record"]
    config = record["config"]
    command = [
        python_executable,
        "-m",
        "robot_vla.cli.collect_local_dagger",
        "--data",
        str(Path(record["base_dataset"]).resolve()),
        "--model-cache",
        str(Path(target["model_cache"]).resolve()),
        "--checkpoint",
        str(Path(record["checkpoint"]["path"]).resolve()),
        "--output",
        str((candidate_dir / "dataset").resolve()),
        "--record",
        str((candidate_dir / "record.json").resolve()),
        "--seed",
        str(config["environment_seed"]),
        "--boundary-type",
        str(config["boundary_type"]),
        "--qwen-context-layer",
        str(config["qwen_context_layer"]),
        "--sampling-seed",
        str(config["sampling_seed_base"]),
        "--num-flow-steps",
        str(config["num_flow_steps"]),
        "--recency-decay",
        str(config["recency_decay"]),
        "--max-anomaly-replans",
        str(config["max_anomaly_replans"]),
    ]
    if not config["snapshot_round_trip_required"]:
        command.append("--skip-snapshot-round-trip")
    if config["paired_clean_expert_required"]:
        command.append("--require-paired-clean-expert")
    return command


def reconcile_replay_record(
    target: Mapping[str, Any],
    replay_record: Mapping[str, Any],
    *,
    replay_record_path: Path,
    subprocess_returncode: int,
) -> dict[str, Any]:
    """对原 formal 失败与重放做逐 seed 严格对齐。"""

    original = target["original_record"]
    original_failure = original["failure"]
    replay_failure = replay_record.get("failure")
    replay_reason = (
        replay_failure.get("reason") if isinstance(replay_failure, dict) else None
    )
    diagnostics = replay_record.get("failure_diagnostics")
    diagnostics_match = (
        isinstance(diagnostics, dict)
        and diagnostics.get("format") == LOCAL_DAGGER_DIAGNOSTIC_FORMAT
        and diagnostics.get("environment_seed") == target["environment_seed"]
        and diagnostics.get("boundary_type") == target["boundary_type"]
        and diagnostics.get("failure_reason") == replay_reason
    )
    status_matches = replay_record.get("status") == original["status"]
    reason_matches = replay_reason == original_failure["reason"]
    failure_type_matches = (
        isinstance(replay_failure, dict)
        and replay_failure.get("type") == original_failure.get("type")
    )
    config_matches = replay_record.get("config") == original["config"]
    checkpoint_matches = replay_record.get("checkpoint") == original["checkpoint"]
    base_dataset_matches = replay_record.get("base_dataset") == original["base_dataset"]
    source_revision_matches = (
        replay_record.get("source_revision") == target["replay_source_revision"]
    )
    replay_status = replay_record.get("status")
    subprocess_contract_matches = (
        (replay_status == "accepted" and subprocess_returncode == 0)
        or (replay_status == "rejected" and subprocess_returncode != 0)
    )
    identity_contract_matches = all(
        (
            config_matches,
            checkpoint_matches,
            base_dataset_matches,
            source_revision_matches,
        )
    )
    failure_record_contract_matches = (
        (replay_status == "accepted" and replay_failure is None)
        or (
            replay_status == "rejected"
            and isinstance(replay_failure, dict)
            and diagnostics_match
        )
    )
    engineering_error = (
        replay_record.get("format") != COLLECTION_FORMAT
        or replay_status == "error"
        or replay_status not in {"accepted", "rejected"}
        or not subprocess_contract_matches
        or not identity_contract_matches
        or not failure_record_contract_matches
    )
    reconciled = all(
        (
            status_matches,
            reason_matches,
            failure_type_matches,
        )
    ) and not engineering_error
    classification = (
        "engineering_error"
        if engineering_error
        else "matched"
        if reconciled
        else "outcome_mismatch"
    )
    return {
        "format": REPLAY_CANDIDATE_FORMAT,
        "environment_seed": target["environment_seed"],
        "boundary_type": target["boundary_type"],
        "status": replay_status,
        "failure": replay_failure,
        "record": str(replay_record_path.resolve()),
        "failure_diagnostics": diagnostics,
        "source_revision": replay_record.get("source_revision"),
        "config": replay_record.get("config"),
        "checkpoint": replay_record.get("checkpoint"),
        "base_dataset": replay_record.get("base_dataset"),
        "reconciled": reconciled,
        "original": {
            "status": original["status"],
            "failure": original_failure,
            "record": target["original_record_path"],
            "record_sha256": target["original_record_sha256"],
            "source_revision": original.get("source_revision"),
        },
        "reconciliation": {
            "classification": classification,
            "reconciled": reconciled,
            "exact_match": reconciled,
            "status_matches": status_matches,
            "reason_matches": reason_matches,
            "failure_type_matches": failure_type_matches,
            "config_matches": config_matches,
            "checkpoint_matches": checkpoint_matches,
            "base_dataset_matches": base_dataset_matches,
            "source_revision_matches": source_revision_matches,
            "identity_contract_matches": identity_contract_matches,
            "failure_record_contract_matches": failure_record_contract_matches,
            "failure_diagnostics_matches": diagnostics_match,
            "subprocess_contract_matches": subprocess_contract_matches,
            "subprocess_returncode": subprocess_returncode,
        },
    }


def summarize_replay(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_candidates: int,
) -> dict[str, Any]:
    statuses = Counter(str(row.get("status")) for row in rows)
    original_reasons = Counter(
        str(row["original"]["failure"]["reason"]) for row in rows
    )
    replay_reasons = Counter()
    for row in rows:
        failure = row.get("failure")
        reason = failure.get("reason", "none") if isinstance(failure, dict) else "none"
        replay_reasons[str(reason)] += 1
    classifications = Counter(
        str(row["reconciliation"]["classification"]) for row in rows
    )
    scan_complete = len(rows) == expected_candidates
    all_reconciled = (
        scan_complete and classifications.get("matched", 0) == expected_candidates
    )
    return {
        "format": REPLAY_FORMAT,
        "complete": scan_complete,
        "scan_complete": scan_complete,
        "all_reconciled": all_reconciled,
        "blocked": classifications.get("engineering_error", 0) > 0,
        "expected_candidates": expected_candidates,
        "completed_candidates": len(rows),
        "status_counts": dict(sorted(statuses.items())),
        "original_reason_counts": dict(sorted(original_reasons.items())),
        "replay_reason_counts": dict(sorted(replay_reasons.items())),
        "reconciliation_counts": dict(sorted(classifications.items())),
        "matched_seeds": [
            int(row["environment_seed"])
            for row in rows
            if row["reconciliation"]["reconciled"]
        ],
        "mismatched_seeds": [
            int(row["environment_seed"])
            for row in rows
            if not row["reconciliation"]["reconciled"]
            and row["reconciliation"]["classification"] == "outcome_mismatch"
        ],
        "engineering_error_seeds": [
            int(row["environment_seed"])
            for row in rows
            if row["reconciliation"]["classification"] == "engineering_error"
        ],
    }


def _write_progress(
    output: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_candidates: int,
) -> None:
    ordered = sorted(rows, key=lambda row: int(row["environment_seed"]))
    _atomic_write_jsonl(output / "replay_candidates.jsonl", ordered)
    _atomic_write_json(
        output / "summary.json",
        summarize_replay(ordered, expected_candidates=expected_candidates),
    )


def _validate_runtime_inputs(targets: Sequence[Mapping[str, Any]]) -> None:
    checkpoint_hashes: dict[Path, str] = {}
    for target in targets:
        seed = target["environment_seed"]
        record = target["original_record"]
        data = Path(record["base_dataset"])
        model_cache = Path(target["model_cache"])
        checkpoint = Path(record["checkpoint"]["path"])
        if not data.is_dir():
            raise FileNotFoundError(f"seed {seed}: 原 base dataset 不存在: {data}")
        if not model_cache.is_dir():
            raise FileNotFoundError(f"seed {seed}: 原 model cache 不存在: {model_cache}")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"seed {seed}: 原 checkpoint 不存在: {checkpoint}")
        resolved_checkpoint = checkpoint.resolve()
        if resolved_checkpoint not in checkpoint_hashes:
            checkpoint_hashes[resolved_checkpoint] = _sha256_file(resolved_checkpoint)
        actual_sha256 = checkpoint_hashes[resolved_checkpoint]
        if actual_sha256 != record["checkpoint"]["sha256"]:
            raise ValueError(f"seed {seed}: checkpoint SHA256 与 formal identity 不一致")


def _completed_attempt(output: Path, seed: int) -> Path | None:
    attempts_root = output / ".attempts"
    if not attempts_root.is_dir():
        return None
    completed = [
        path
        for path in sorted(attempts_root.glob(f"seed-{seed:06d}-*"))
        if (path / "record.json").is_file() and (path / "subprocess.json").is_file()
    ]
    if len(completed) > 1:
        raise RuntimeError(f"seed {seed}: 存在多个完整 replay attempt，拒绝自动选择")
    return completed[0] if completed else None


def _materialize_candidate(
    args: argparse.Namespace,
    target: Mapping[str, Any],
    *,
    project_root: Path,
) -> Path:
    seed = int(target["environment_seed"])
    candidate_dir = args.output / "candidates" / f"seed-{seed:06d}"
    if candidate_dir.exists():
        if not candidate_dir.is_dir():
            raise RuntimeError(f"seed {seed}: candidate path 不是目录")
        if not (candidate_dir / "record.json").is_file():
            raise RuntimeError(f"seed {seed}: completed candidate 缺少 record.json")
        if not (candidate_dir / "subprocess.json").is_file():
            raise RuntimeError(f"seed {seed}: completed candidate 缺少 subprocess.json")
        return candidate_dir

    recovered = _completed_attempt(args.output, seed)
    if recovered is not None:
        candidate_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(recovered, candidate_dir)
        return candidate_dir

    attempts_root = args.output / ".attempts"
    attempts_root.mkdir(parents=True, exist_ok=True)
    attempt_dir = Path(
        tempfile.mkdtemp(prefix=f"seed-{seed:06d}-", dir=attempts_root)
    )
    command = candidate_command(target, candidate_dir=attempt_dir)
    environment = os.environ.copy()
    environment.setdefault("HF_HUB_OFFLINE", "1")
    environment.setdefault("TRANSFORMERS_OFFLINE", "1")
    with (
        (attempt_dir / "stdout.log").open("w", encoding="utf-8") as stdout,
        (attempt_dir / "stderr.log").open("w", encoding="utf-8") as stderr,
    ):
        completed = subprocess.run(
            command,
            cwd=project_root,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    record_path = attempt_dir / "record.json"
    if not record_path.is_file():
        raise RuntimeError(
            f"seed {seed}: replay subprocess exit={completed.returncode} 但未写 record"
        )
    _atomic_write_json(
        attempt_dir / "subprocess.json",
        {
            "format": SUBPROCESS_FORMAT,
            "returncode": completed.returncode,
            "command": command,
        },
    )
    candidate_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(attempt_dir, candidate_dir)
    return candidate_dir


def run(args: argparse.Namespace) -> None:
    # 保持纯 JSON 辅助函数可在无 Torch/ManiSkill 的审计环境中导入。
    from robot_vla.cli.train_stage1 import compute_source_revision

    formal_root = args.formal.resolve()
    output = args.output.resolve()
    if output == formal_root or output.is_relative_to(formal_root):
        raise ValueError("forensic replay output 必须独立于 formal pool")

    project_root = Path(__file__).resolve().parents[3]
    experiment, targets = build_replay_experiment(
        formal_root,
        replay_source_revision=compute_source_revision(project_root),
    )
    _validate_runtime_inputs(targets)

    experiment_path = output / "experiment.json"
    if args.resume:
        existing = _read_json(experiment_path, label="replay experiment.json")
        if existing != experiment:
            raise ValueError("--resume replay experiment identity 漂移")
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("replay output 目录非空；拒绝覆盖")
        output.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(experiment_path, experiment)

    rows: list[dict[str, Any]] = []
    for target in targets:
        seed = int(target["environment_seed"])
        candidate_dir = _materialize_candidate(
            args,
            target,
            project_root=project_root,
        )
        replay_record_path = candidate_dir / "record.json"
        replay_record = _read_json(
            replay_record_path,
            label=f"seed {seed} replay record",
        )
        subprocess_record = _read_json(
            candidate_dir / "subprocess.json",
            label=f"seed {seed} subprocess record",
        )
        if subprocess_record.get("format") != SUBPROCESS_FORMAT:
            raise RuntimeError(f"seed {seed}: subprocess record format 不兼容")
        returncode = subprocess_record.get("returncode")
        if not isinstance(returncode, int) or isinstance(returncode, bool):
            raise TypeError(f"seed {seed}: subprocess returncode 无效")
        row = reconcile_replay_record(
            target,
            replay_record,
            replay_record_path=replay_record_path,
            subprocess_returncode=returncode,
        )
        _atomic_write_json(candidate_dir / "reconciliation.json", row)
        rows.append(row)
        _write_progress(output, rows, expected_candidates=len(targets))
        print(
            json.dumps(
                {
                    "event": "replay_complete",
                    "seed": seed,
                    "reconciliation": row["reconciliation"]["classification"],
                    "completed": len(rows),
                    "expected": len(targets),
                },
                sort_keys=True,
                allow_nan=False,
            ),
            flush=True,
        )
        classification = row["reconciliation"]["classification"]
        if classification == "engineering_error":
            raise RuntimeError(f"seed {seed}: replay 工程 error，已写 reconciliation")

    summary = summarize_replay(rows, expected_candidates=len(targets))
    if not summary["complete"]:
        raise RuntimeError("forensic replay 未产生全部重放结果")


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
