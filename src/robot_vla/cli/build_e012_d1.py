"""从正式 RG/GL canonical selection 构建 E012 D1 additions 与 union audit。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from robot_vla.adapters import ProprioStats
from robot_vla.contracts import RobotSpec
from robot_vla.data.audit import audit_trajectory
from robot_vla.data.trajectory import (
    TrajectoryMeta,
    TrajectoryStore,
    load_manifest,
)
from robot_vla.sim.local_dagger_risk import score_and_select_risk_candidates

D1_BUILD_FORMAT = "robot-vla-e012-d1-build/v1"
D1_AUDIT_FORMAT = "robot-vla-e012-d1-additions-audit/v1"
D1_UNION_AUDIT_FORMAT = "robot-vla-e012-d0-d1-union-audit/v1"
LEGACY_POOL_FORMAT = "robot-vla-local-dagger-pool/v1"
AMENDED_POOL_FORMAT = "robot-vla-local-dagger-pool/v2"
CANONICAL_SELECTED_RECORD_FORMAT = (
    "robot-vla-local-dagger-canonical-selected-record/v1"
)
BOUNDARY_SOURCES = {
    "reach_grasp": "dagger_reach_grasp",
    "grasp_lift": "dagger_grasp_lift",
}
EXPECTED_SELECTED = 20
EXPECTED_HIGH = 14
EXPECTED_LOW = 6


@dataclass(frozen=True)
class SelectedArtifact:
    boundary_type: str
    environment_seed: int
    stratum: str
    selection_index: int
    risk_score: float
    trajectory: TrajectoryMeta
    dataset_root: Path
    npz_path: Path
    npz_sha256: str
    record_path: Path
    record_sha256: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d0", type=Path, required=True)
    parser.add_argument("--rg-pool", type=Path, required=True)
    parser.add_argument("--gl-pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON 包含重复字段: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"JSON 禁止非有限常量: {value}")


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _assert_plain_file(path: Path, *, root: Path | None = None) -> os.stat_result:
    absolute = _lexical_absolute(path)
    if root is not None:
        absolute_root = _lexical_absolute(root)
        root_stat = absolute_root.lstat()
        if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
            raise ValueError(f"可信根目录必须是普通目录: {absolute_root}")
        if not absolute.is_relative_to(absolute_root):
            raise ValueError(f"文件路径逃逸: {absolute}")
        relative = absolute.relative_to(absolute_root)
        current = absolute_root
        for part in relative.parts[:-1]:
            current /= part
            current_stat = current.lstat()
            if stat.S_ISLNK(current_stat.st_mode):
                raise ValueError(f"父目录禁止 symlink: {current}")
            if not stat.S_ISDIR(current_stat.st_mode):
                raise ValueError(f"父路径不是目录: {current}")
    file_stat = absolute.lstat()
    if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
        raise ValueError(f"只接受普通文件: {absolute}")
    if file_stat.st_nlink != 1:
        raise ValueError(f"普通文件禁止 hardlink: {absolute}")
    return file_stat


def _read_plain_bytes(path: Path, *, root: Path | None = None) -> bytes:
    absolute = _lexical_absolute(path)
    before = _assert_plain_file(absolute, root=root)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(absolute, flags)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino, opened.st_size) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
        ):
            raise RuntimeError(f"文件在 open 前发生变化: {absolute}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
        ):
            raise RuntimeError(f"文件在读取期间发生变化: {absolute}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _strict_json_bytes(raw: bytes, *, label: str) -> Any:
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} 不是 strict JSON: {exc}") from exc


def _read_json(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    value = _strict_json_bytes(_read_plain_bytes(path, root=root), label=str(path))
    if not isinstance(value, dict):
        raise TypeError(f"JSON 顶层必须是 object: {path}")
    return value


def _read_jsonl(path: Path, *, root: Path | None = None) -> list[dict[str, Any]]:
    raw = _read_plain_bytes(path, root=root)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"JSONL 禁止空行: {path}:{line_number}")
        value = _strict_json_bytes(line, label=f"{path}:{line_number}")
        if not isinstance(value, dict):
            raise TypeError(f"JSONL row 必须是 object: {path}:{line_number}")
        rows.append(value)
    if not rows:
        raise ValueError(f"JSONL 不能为空: {path}")
    return rows


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path, *, root: Path | None = None) -> str:
    absolute = _lexical_absolute(path)
    before = _assert_plain_file(absolute, root=root)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(absolute, flags)
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ):
            raise RuntimeError(f"文件在 hash open 前发生变化: {absolute}")
        while chunk := os.read(descriptor, 8 * 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
        ):
            raise RuntimeError(f"文件在 hash 期间发生变化: {absolute}")
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _atomic_write(path: Path, raw: bytes) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    raw = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    _atomic_write(path, raw)


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    raw = b"".join(
        (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        for row in rows
    )
    _atomic_write(path, raw)


def _atomic_copy_and_hash(source: Path, target: Path, *, root: Path) -> str:
    absolute = _lexical_absolute(source)
    before = _assert_plain_file(absolute, root=root)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    source_descriptor = os.open(absolute, flags)
    temporary: Path | None = None
    digest = hashlib.sha256()
    try:
        opened = os.fstat(source_descriptor)
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ):
            raise RuntimeError(f"NPZ 在 copy open 前发生变化: {absolute}")
        with tempfile.NamedTemporaryFile(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            while chunk := os.read(source_descriptor, 8 * 1024 * 1024):
                digest.update(chunk)
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        after = os.fstat(source_descriptor)
        if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
        ):
            raise RuntimeError(f"NPZ 在 copy 期间发生变化: {absolute}")
        os.chmod(temporary, 0o600)
        os.replace(temporary, target)
        temporary = None
        return digest.hexdigest()
    finally:
        os.close(source_descriptor)
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _d0_identity(root: Path) -> dict[str, Any]:
    audit_path = root / "audit_report.json"
    manifest_path = root / "manifest.jsonl"
    stats_path = root / "proprio_stats.json"
    audit = _read_json(audit_path, root=root)
    if float(audit.get("success_rate", 0.0)) != 1.0:
        raise ValueError("D0 success_rate 必须为 1.0")
    if _sha256_file(manifest_path, root=root) != audit.get("manifest_sha256"):
        raise ValueError("D0 manifest SHA256 与 audit 不一致")
    entries = load_manifest(root)
    if len(entries) != int(audit["trajectory_count"]):
        raise ValueError("D0 trajectory_count 与 manifest 不一致")
    seeds = sorted(
        int(entry.randomization["seed"])
        for entry in entries
        if "seed" in entry.randomization
    )
    return {
        "path": str(_lexical_absolute(root)),
        "audit": audit,
        "audit_sha256": _sha256_file(audit_path, root=root),
        "manifest_sha256": _sha256_file(manifest_path, root=root),
        "proprio_stats_sha256": _sha256_file(stats_path, root=root),
        "environment_seeds": seeds,
    }


def _validate_pool_identity(
    pool_root: Path,
    *,
    boundary_type: str,
    expected_format: str,
    d0: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    experiment_path = pool_root / "experiment.json"
    experiment = _read_json(experiment_path, root=pool_root)
    if experiment.get("format") != expected_format:
        raise ValueError(f"{boundary_type} pool format 不兼容")
    if experiment.get("boundary_type") != boundary_type:
        raise ValueError(f"{boundary_type} pool boundary identity 漂移")
    base = experiment.get("base_dataset", {})
    if _lexical_absolute(Path(str(base.get("path", "")))) != _lexical_absolute(
        Path(d0["path"])
    ):
        raise ValueError(f"{boundary_type} pool D0 path 漂移")
    for key in ("dataset_sha256", "manifest_sha256", "trajectory_count", "step_count"):
        if base.get("audit", {}).get(key) != d0["audit"].get(key):
            raise ValueError(f"{boundary_type} pool D0 {key} 漂移")
    checkpoint = experiment.get("checkpoint", {})
    checkpoint_sha256 = str(checkpoint.get("sha256", ""))
    if len(checkpoint_sha256) != 64:
        raise ValueError(f"{boundary_type} pool checkpoint identity 无效")
    return experiment, _sha256_file(experiment_path, root=pool_root)


def _validate_selection(
    pool_root: Path,
    *,
    experiment: dict[str, Any],
    expected_candidates: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], tuple[int, ...], tuple[int, ...]]:
    candidates_path = pool_root / "collection_candidates.jsonl"
    summary_path = pool_root / "collection_summary.json"
    risk_path = pool_root / "risk_selection.json"
    candidates = _read_jsonl(candidates_path, root=pool_root)
    summary = _read_json(summary_path, root=pool_root)
    risk = _read_json(risk_path, root=pool_root)
    expected_seeds = tuple(int(seed) for seed in experiment["environment_seeds"])
    observed_seeds = tuple(int(row["environment_seed"]) for row in candidates)
    if len(candidates) != expected_candidates or observed_seeds != expected_seeds:
        raise ValueError("pool candidate manifest 不是完整冻结 seed 序列")
    statuses = Counter(str(row.get("status")) for row in candidates)
    if set(statuses) - {"accepted", "rejected"}:
        raise ValueError(f"pool 包含工程 error/status: {dict(statuses)}")
    if (
        summary.get("scan_complete") is not True
        or int(summary.get("completed_candidates", -1)) != expected_candidates
        or int(summary.get("expected_candidates", -1)) != expected_candidates
        or summary.get("status_counts") != dict(sorted(statuses.items()))
    ):
        raise ValueError("pool final summary 与 candidates 不一致")
    if summary.get("selection_gate_passed") is not True:
        raise ValueError("pool 未通过 20 条 selection gate")
    eligible = [row for row in candidates if row.get("eligible_for_risk_selection") is True]
    recomputed = score_and_select_risk_candidates(
        str(experiment["boundary_type"]),
        eligible,
    ).to_dict()
    mathematical_risk = {key: risk.get(key) for key in recomputed}
    if mathematical_risk != recomputed:
        raise ValueError("risk_selection 无法从完整 candidate pool 精确复算")
    high = tuple(int(seed) for seed in recomputed["high_risk_seeds"])
    low = tuple(int(seed) for seed in recomputed["low_risk_seeds"])
    if len(high) != EXPECTED_HIGH or len(low) != EXPECTED_LOW:
        raise ValueError("risk selection 不是冻结的 14 high + 6 low")
    if len(set(high + low)) != EXPECTED_SELECTED:
        raise ValueError("risk selection seed 必须是 20 个互异值")
    if summary.get("high_risk_seeds") != list(high) or summary.get(
        "low_risk_seeds"
    ) != list(low):
        raise ValueError("summary 与 risk selection 漂移")
    return candidates, risk, high, low


def _selected_artifact(
    pool_root: Path,
    *,
    experiment: dict[str, Any],
    candidate_row: dict[str, Any],
    score_row: dict[str, Any],
    seed: int,
    stratum: str,
    selection_index: int,
    canonical_row: dict[str, Any] | None,
) -> SelectedArtifact:
    boundary_type = str(experiment["boundary_type"])
    candidate_dir = pool_root / "candidates" / f"seed-{seed:06d}"
    record_path = candidate_dir / "record.json"
    expected_record = _lexical_absolute(record_path)
    if _lexical_absolute(Path(str(candidate_row.get("record", "")))) != expected_record:
        raise ValueError(f"seed {seed}: candidate record path 不 canonical")
    record = _read_json(record_path, root=pool_root)
    record_sha256 = _sha256_file(record_path, root=pool_root)
    config = record.get("config", {})
    if (
        record.get("status") != "accepted"
        or record.get("eligible_for_risk_selection") is not True
        or int(config.get("environment_seed", -1)) != seed
        or config.get("boundary_type") != boundary_type
        or record.get("source_revision") != experiment.get("source_revision")
        or record.get("checkpoint", {}).get("sha256")
        != experiment.get("checkpoint", {}).get("sha256")
    ):
        raise ValueError(f"seed {seed}: selected candidate identity/status 漂移")
    if record.get("audit", {}).get("trajectory_contract") != "passed":
        raise ValueError(f"seed {seed}: candidate trajectory audit 未通过")
    snapshot = record.get("result", {}).get("snapshot_round_trip", {})
    if snapshot.get("passed") is not True:
        raise ValueError(f"seed {seed}: snapshot round-trip 未通过")
    trajectory_value = record.get("result", {}).get("trajectory")
    if not isinstance(trajectory_value, dict):
        raise TypeError(f"seed {seed}: trajectory metadata 缺失")
    trajectory = TrajectoryMeta.from_dict(trajectory_value)
    provenance = trajectory.local_dagger
    if (
        provenance is None
        or provenance.source != BOUNDARY_SOURCES[boundary_type]
        or provenance.boundary_type != boundary_type
        or provenance.rollin_seed != seed
        or provenance.rollin_policy_checkpoint_sha256
        != experiment["checkpoint"]["sha256"]
    ):
        raise ValueError(f"seed {seed}: Local DAgger provenance 漂移")
    dataset_root = candidate_dir / "dataset"
    manifest_path = dataset_root / "manifest.jsonl"
    manifest_rows = _read_jsonl(manifest_path, root=pool_root)
    if manifest_rows != [trajectory_value]:
        raise ValueError(f"seed {seed}: candidate manifest 与 record trajectory 不一致")
    npz_path = dataset_root / trajectory.file
    npz_sha256 = _sha256_file(npz_path, root=pool_root)
    arrays = TrajectoryStore(dataset_root, RobotSpec(), cache_size=0).get(trajectory)
    audit_trajectory(arrays, trajectory, RobotSpec())
    valid_anchors = [
        timestep
        for timestep in np.flatnonzero(arrays.observation_valid).tolist()
        if provenance.training_window_start <= timestep
        and timestep + RobotSpec().action_horizon <= provenance.training_window_end
        and bool(
            arrays.expert_supervision_mask[
                timestep : timestep + RobotSpec().action_horizon
            ].all()
        )
    ]
    if len(valid_anchors) != 49:
        raise ValueError(
            f"seed {seed}: Expert-only training anchor 应为 49，实际为 {len(valid_anchors)}"
        )
    if canonical_row is not None and (
        canonical_row.get("format") != CANONICAL_SELECTED_RECORD_FORMAT
        or int(canonical_row.get("environment_seed", -1)) != seed
        or canonical_row.get("boundary_type") != boundary_type
        or canonical_row.get("trajectory") != trajectory_value
        or canonical_row.get("candidate", {}).get("status") != "accepted"
        or canonical_row.get("candidate", {}).get("eligible_for_risk_selection")
        is not True
        or canonical_row.get("candidate", {}).get("record_sha256") != record_sha256
        or canonical_row.get("selection", {}).get("selected") is not True
        or canonical_row.get("selection", {}).get("stratum") != stratum
        or int(canonical_row.get("selection", {}).get("index", -1))
        != selection_index
        or float(canonical_row.get("selection", {}).get("risk_score", math.nan))
        != float(score_row["risk_score"])
    ):
        raise ValueError(f"seed {seed}: canonical selected record 漂移")
    return SelectedArtifact(
        boundary_type=boundary_type,
        environment_seed=seed,
        stratum=stratum,
        selection_index=selection_index,
        risk_score=float(score_row["risk_score"]),
        trajectory=trajectory,
        dataset_root=dataset_root,
        npz_path=npz_path,
        npz_sha256=npz_sha256,
        record_path=record_path,
        record_sha256=record_sha256,
    )


def _load_selected_pool(
    pool_root: Path,
    *,
    experiment: dict[str, Any],
    expected_candidates: int,
    require_canonical: bool,
) -> tuple[list[SelectedArtifact], dict[str, str]]:
    candidates, _risk, high, low = _validate_selection(
        pool_root,
        experiment=experiment,
        expected_candidates=expected_candidates,
    )
    candidates_by_seed = {int(row["environment_seed"]): row for row in candidates}
    score_rows = {
        int(row["environment_seed"]): row
        for row in _read_json(pool_root / "risk_selection.json", root=pool_root)[
            "scored_candidates"
        ]
    }
    canonical_by_seed: dict[int, dict[str, Any]] = {}
    canonical_path = pool_root / "canonical_selected_records.jsonl"
    if require_canonical:
        canonical_rows = _read_jsonl(canonical_path, root=pool_root)
        if len(canonical_rows) != EXPECTED_SELECTED:
            raise ValueError("amended GL canonical manifest 必须恰好包含 20 条")
        canonical_by_seed = {
            int(row["environment_seed"]): row for row in canonical_rows
        }
        if len(canonical_by_seed) != EXPECTED_SELECTED:
            raise ValueError("amended GL canonical manifest seed 重复")
        summary = _read_json(pool_root / "collection_summary.json", root=pool_root)
        canonical_sha256 = _sha256_file(canonical_path, root=pool_root)
        if (
            summary.get("publication_state") != "complete"
            or summary.get("canonical_selected_records_sha256") != canonical_sha256
        ):
            raise ValueError("amended GL canonical manifest 未被 final summary 绑定")
    selected: list[SelectedArtifact] = []
    for selection_index, seed in enumerate(high + low):
        stratum = "high" if seed in high else "low"
        selected.append(
            _selected_artifact(
                pool_root,
                experiment=experiment,
                candidate_row=candidates_by_seed[seed],
                score_row=score_rows[seed],
                seed=seed,
                stratum=stratum,
                selection_index=selection_index,
                canonical_row=canonical_by_seed.get(seed),
            )
        )
    receipts = {
        "experiment_sha256": _sha256_file(pool_root / "experiment.json", root=pool_root),
        "collection_candidates_sha256": _sha256_file(
            pool_root / "collection_candidates.jsonl", root=pool_root
        ),
        "collection_summary_sha256": _sha256_file(
            pool_root / "collection_summary.json", root=pool_root
        ),
        "risk_selection_sha256": _sha256_file(
            pool_root / "risk_selection.json", root=pool_root
        ),
    }
    if require_canonical:
        receipts["canonical_selected_records_sha256"] = _sha256_file(
            canonical_path,
            root=pool_root,
        )
    return selected, receipts


def _additions_audit(
    output: Path,
    entries: list[TrajectoryMeta],
    *,
    copied_hashes: dict[str, str],
) -> tuple[dict[str, Any], ProprioStats]:
    spec = RobotSpec()
    store = TrajectoryStore(output, spec, cache_size=0)
    source_counts: Counter[str] = Counter()
    anchor_counts: Counter[str] = Counter()
    steps = 0
    proprio: list[np.ndarray] = []
    canonical_files: list[dict[str, Any]] = []
    for entry in entries:
        arrays = store.get(entry)
        audit_trajectory(arrays, entry, spec)
        provenance = entry.local_dagger
        if provenance is None:
            raise RuntimeError("D1 additions 混入 clean trajectory")
        source_counts[provenance.source] += 1
        anchors = sum(
            1
            for timestep in np.flatnonzero(arrays.observation_valid).tolist()
            if provenance.training_window_start <= timestep
            and timestep + spec.action_horizon <= provenance.training_window_end
            and bool(
                arrays.expert_supervision_mask[
                    timestep : timestep + spec.action_horizon
                ].all()
            )
        )
        if anchors != 49:
            raise RuntimeError(
                f"{entry.trajectory_id}: D1 Expert-only anchor 应为 49，实际为 {anchors}"
            )
        anchor_counts[provenance.source] += anchors
        steps += arrays.num_steps
        proprio.append(arrays.proprio)
        canonical_files.append(
            {
                "meta": entry.to_dict(),
                "npz_sha256": copied_hashes[entry.trajectory_id],
            }
        )
    stats = ProprioStats.fit(proprio, spec)
    canonical_payload = json.dumps(
        sorted(canonical_files, key=lambda item: str(item["meta"])),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    report = {
        "format": D1_AUDIT_FORMAT,
        "dataset_sha256": _sha256_bytes(canonical_payload),
        "manifest_sha256": _sha256_file(output / "manifest.jsonl", root=output),
        "trajectory_count": len(entries),
        "step_count": steps,
        "success_rate": 1.0,
        "split_trajectory_counts": {"train": len(entries), "val": 0, "test": 0},
        "source_trajectory_counts": dict(sorted(source_counts.items())),
        "expert_only_anchor_counts": dict(sorted(anchor_counts.items())),
        "observed_proprio_stats_count": stats.count,
        "proprio_stats_usage": "diagnostic_only; training must use frozen D0 stats",
    }
    return report, stats


def run(args: argparse.Namespace) -> None:
    d0_root = _lexical_absolute(args.d0)
    rg_root = _lexical_absolute(args.rg_pool)
    gl_root = _lexical_absolute(args.gl_pool)
    output = _lexical_absolute(args.output)
    if output.exists():
        raise FileExistsError("D1 output 必须不存在，拒绝覆盖或恢复 partial build")
    from robot_vla.cli.collect_local_dagger_amended_pool import (
        _read_formal_checkpoint,
        verify_frozen_d0_compatibility,
    )
    from robot_vla.cli.train_stage1 import compute_source_revision

    project_root = Path(__file__).resolve().parents[3]
    code_revision = compute_source_revision(project_root)
    full_d0_verification = verify_frozen_d0_compatibility(d0_root)
    d0 = _d0_identity(d0_root)
    rg_experiment, _ = _validate_pool_identity(
        rg_root,
        boundary_type="reach_grasp",
        expected_format=LEGACY_POOL_FORMAT,
        d0=d0,
    )
    gl_experiment, _ = _validate_pool_identity(
        gl_root,
        boundary_type="grasp_lift",
        expected_format=AMENDED_POOL_FORMAT,
        d0=d0,
    )
    if (
        gl_experiment.get("base_dataset", {}).get("compatibility")
        != full_d0_verification.receipt
    ):
        raise ValueError("D1 pre-build D0 full verifier receipt 与 GL frozen identity 漂移")
    if rg_experiment["checkpoint"]["sha256"] != gl_experiment["checkpoint"]["sha256"]:
        raise ValueError("RG/GL collection checkpoint identity 不一致")
    checkpoint_path, checkpoint_read = _read_formal_checkpoint(
        Path(str(gl_experiment["checkpoint"]["path"]))
    )
    if checkpoint_read.sha256 != gl_experiment["checkpoint"]["sha256"]:
        raise ValueError("D1 pre-build checkpoint SHA256 漂移")
    rg_selected, rg_receipts = _load_selected_pool(
        rg_root,
        experiment=rg_experiment,
        expected_candidates=100,
        require_canonical=False,
    )
    gl_selected, gl_receipts = _load_selected_pool(
        gl_root,
        experiment=gl_experiment,
        expected_candidates=240,
        require_canonical=True,
    )
    selected = rg_selected + gl_selected
    selected_seeds = [item.environment_seed for item in selected]
    if len(selected) != 40 or len(set(selected_seeds)) != 40:
        raise RuntimeError("D1 必须由 40 个互异正式 seed 构成")
    trajectory_ids = [item.trajectory.trajectory_id for item in selected]
    if len(set(trajectory_ids)) != len(trajectory_ids):
        raise RuntimeError("D1 selected trajectory_id 重复")
    d0_overlap = set(selected_seeds).intersection(d0["environment_seeds"])
    if d0_overlap:
        raise ValueError(f"D1 collection seed 与 D0 重叠: {sorted(d0_overlap)}")

    output.mkdir(mode=0o700, parents=True)
    os.chmod(output, 0o700)
    trajectories_dir = output / "trajectories"
    trajectories_dir.mkdir(mode=0o700)
    marker = output / ".building"
    _atomic_write(marker, b"incomplete\n")
    entries: list[TrajectoryMeta] = []
    provenance_rows: list[dict[str, Any]] = []
    copied_hashes: dict[str, str] = {}
    for item in selected:
        target_file = f"trajectories/{item.trajectory.trajectory_id}.npz"
        target = output / target_file
        copied_sha256 = _atomic_copy_and_hash(
            item.npz_path,
            target,
            root=item.dataset_root.parent.parent.parent,
        )
        if copied_sha256 != item.npz_sha256:
            raise RuntimeError(f"seed {item.environment_seed}: NPZ copy SHA256 漂移")
        entry = replace(item.trajectory, file=target_file)
        entries.append(entry)
        copied_hashes[entry.trajectory_id] = copied_sha256
        provenance_rows.append(
            {
                "format": "robot-vla-e012-d1-selected-artifact/v1",
                "boundary_type": item.boundary_type,
                "environment_seed": item.environment_seed,
                "selection": {
                    "stratum": item.stratum,
                    "index": item.selection_index,
                    "risk_score": item.risk_score,
                },
                "record": {
                    "path": str(_lexical_absolute(item.record_path)),
                    "sha256": item.record_sha256,
                },
                "source_npz": {
                    "path": str(_lexical_absolute(item.npz_path)),
                    "sha256": item.npz_sha256,
                },
                "d1_npz": {"file": target_file, "sha256": copied_sha256},
            }
        )
    manifest_rows = [entry.to_dict() for entry in entries]
    _atomic_write_jsonl(output / "manifest.jsonl", manifest_rows)
    _atomic_write_jsonl(
        output / "rg_canonical_selected_records.jsonl",
        provenance_rows[:EXPECTED_SELECTED],
    )
    _atomic_write_jsonl(
        output / "gl_canonical_selected_records.jsonl",
        provenance_rows[EXPECTED_SELECTED:],
    )
    _atomic_write_jsonl(output / "selected_artifacts.jsonl", provenance_rows)
    additions_audit, observed_stats = _additions_audit(
        output,
        entries,
        copied_hashes=copied_hashes,
    )
    _atomic_write_json(output / "audit_report.json", additions_audit)
    _atomic_write_json(
        output / "observed_proprio_stats.json",
        asdict(observed_stats),
    )
    union_audit = {
        "format": D1_UNION_AUDIT_FORMAT,
        "passed": True,
        "base_d0": {
            "path": d0["path"],
            "dataset_sha256": d0["audit"]["dataset_sha256"],
            "manifest_sha256": d0["manifest_sha256"],
            "trajectory_count": d0["audit"]["trajectory_count"],
            "step_count": d0["audit"]["step_count"],
        },
        "dagger_additions": additions_audit,
        "logical_union": {
            "trajectory_count": int(d0["audit"]["trajectory_count"]) + len(entries),
            "step_count": int(d0["audit"]["step_count"])
            + int(additions_audit["step_count"]),
            "training_sources": {
                "base_d0": int(d0["audit"]["split_trajectory_counts"]["train"]),
                **additions_audit["source_trajectory_counts"],
            },
            "validation_source": "base_d0/val only",
            "test_source": "base_d0/test only",
        },
        "proprio_stats": {
            "source": "frozen D0",
            "sha256": d0["proprio_stats_sha256"],
            "d1_observed_stats_usage": "diagnostic_only",
        },
        "seed_sets": {
            "d0_count": len(d0["environment_seeds"]),
            "rg_selected": [item.environment_seed for item in rg_selected],
            "gl_selected": [item.environment_seed for item in gl_selected],
            "pairwise_disjoint": True,
        },
    }
    _atomic_write_json(output / "union_audit.json", union_audit)
    for item in selected:
        if _sha256_file(item.record_path, root=item.dataset_root.parent.parent.parent) != (
            item.record_sha256
        ):
            raise RuntimeError(
                f"seed {item.environment_seed}: record 在 D1 build 期间发生变化"
            )
    current_rg_receipts = {
        "experiment_sha256": _sha256_file(rg_root / "experiment.json", root=rg_root),
        "collection_candidates_sha256": _sha256_file(
            rg_root / "collection_candidates.jsonl", root=rg_root
        ),
        "collection_summary_sha256": _sha256_file(
            rg_root / "collection_summary.json", root=rg_root
        ),
        "risk_selection_sha256": _sha256_file(
            rg_root / "risk_selection.json", root=rg_root
        ),
    }
    current_gl_receipts = {
        "experiment_sha256": _sha256_file(gl_root / "experiment.json", root=gl_root),
        "collection_candidates_sha256": _sha256_file(
            gl_root / "collection_candidates.jsonl", root=gl_root
        ),
        "collection_summary_sha256": _sha256_file(
            gl_root / "collection_summary.json", root=gl_root
        ),
        "risk_selection_sha256": _sha256_file(
            gl_root / "risk_selection.json", root=gl_root
        ),
        "canonical_selected_records_sha256": _sha256_file(
            gl_root / "canonical_selected_records.jsonl", root=gl_root
        ),
    }
    if current_rg_receipts != rg_receipts or current_gl_receipts != gl_receipts:
        raise RuntimeError("RG/GL upstream receipts 在 D1 build 期间发生变化")
    build_receipt = {
        "format": D1_BUILD_FORMAT,
        "passed": True,
        "code_revision": code_revision,
        "checkpoint_sha256": rg_experiment["checkpoint"]["sha256"],
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_read.sha256,
            "size_bytes": checkpoint_read.stat_identity[4],
        },
        "d0": {
            "audit_sha256": d0["audit_sha256"],
            "manifest_sha256": d0["manifest_sha256"],
            "proprio_stats_sha256": d0["proprio_stats_sha256"],
        },
        "rg_pool": rg_receipts,
        "gl_pool": gl_receipts,
        "outputs": {
            name: _sha256_file(output / name, root=output)
            for name in (
                "manifest.jsonl",
                "rg_canonical_selected_records.jsonl",
                "gl_canonical_selected_records.jsonl",
                "selected_artifacts.jsonl",
                "audit_report.json",
                "observed_proprio_stats.json",
                "union_audit.json",
            )
        },
    }
    _atomic_write_json(output / "build_receipt.json", build_receipt)
    marker.unlink()
    print(json.dumps(build_receipt, sort_keys=True, allow_nan=False), flush=True)


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
