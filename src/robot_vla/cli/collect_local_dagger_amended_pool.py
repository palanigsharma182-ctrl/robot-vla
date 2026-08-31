"""E012 amended 正式 Grasp→Lift Local DAgger v2 候选池 runner。"""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from robot_vla.cli.collect_local_dagger import (
    _CANDIDATE_STAGING_MARKER,
    _sha256_file,
    derive_collection_sampling_seed,
)
from robot_vla.contracts import QWEN_MODEL_ID, QWEN_REVISION
from robot_vla.data.trajectory import TrajectoryMeta
from robot_vla.local_dagger_protocol import (
    EXPERT_ACTION_BUDGET_EXHAUSTED_REASON,
    LOCAL_DAGGER_ACTION_BUDGET_PROTOCOL_FIELD,
    LOCAL_DAGGER_ACTION_BUDGET_UNIT,
    LOCAL_DAGGER_ACTION_BUDGET_USAGE_FIELD,
    POLICY_ACTION_BUDGET_EXHAUSTED_REASON,
    LocalDaggerActionBudgetProtocol,
    resolve_local_dagger_action_budget,
)
from robot_vla.sim.local_dagger_diagnostics import (
    EPISODE_TIME_LIMIT_REASON,
    LOCAL_DAGGER_DIAGNOSTIC_FORMAT,
)
from robot_vla.sim.local_dagger_risk import (
    RISK_COMPONENT_UNITS,
    RISK_CONTRACT_VERSION,
    compute_paired_risk_components,
    score_and_select_risk_candidates,
)

POOL_FORMAT = "robot-vla-local-dagger-pool/v2"
COLLECTION_FORMAT = "robot-vla-local-dagger-collection/v1"
CANONICAL_SELECTED_RECORD_FORMAT = (
    "robot-vla-local-dagger-canonical-selected-record/v1"
)
CANDIDATE_DATASET_STAGING_MARKER = _CANDIDATE_STAGING_MARKER
AMENDED_FORMAL_BOUNDARY_TYPE = "grasp_lift"
AMENDED_FORMAL_SEED_START = 30_200
AMENDED_FORMAL_SEED_END_LIMIT = 31_000
AMENDED_FORMAL_CHECKPOINT_SHA256 = (
    "a542076f291e29b68e3d28930b15c40396d511a44eb358c2eaeb4e113c041ad6"
)
AMENDED_ACTION_BUDGET_PROTOCOL = (
    LocalDaggerActionBudgetProtocol.SEGMENTED_300_180_480
)
ELIGIBLE_SELECTION_GATE = 20
HIGH_RISK_SELECTION_COUNT = 14
LOW_RISK_SELECTION_COUNT = 6
PAIRED_CLEAN_EXPERT_PROTOCOL = {
    "name": LocalDaggerActionBudgetProtocol.LEGACY.value,
    "action_unit": LOCAL_DAGGER_ACTION_BUDGET_UNIT,
    "environment_action_limit": 300,
}
RISK_SELECTION_FILENAME = "risk_selection.json"
CANONICAL_SELECTED_RECORDS_FILENAME = "canonical_selected_records.jsonl"
_ROOT_DERIVED_ATOMIC_FILENAMES = (
    "collection_candidates.jsonl",
    "collection_summary.json",
    RISK_SELECTION_FILENAME,
    CANONICAL_SELECTED_RECORDS_FILENAME,
)
_BOUNDARY_ENDED_REASON = "目标 boundary 发生时环境已经结束"
_RUNTIME_DISTRIBUTIONS = {
    "gymnasium": "gymnasium",
    "mani_skill": "mani-skill",
    "mplib": "mplib",
    "numpy": "numpy",
    "sapien": "sapien",
    "torch": "torch",
    "transformers": "transformers",
}
_GPU_UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
_NVIDIA_DRIVER_VERSION_PATTERN = re.compile(r"^[0-9]+(?:\.[0-9]+){1,3}$")
_POOL_LOCK_SUFFIX = ".e012-amended-pool.lock"
D0_COMPATIBILITY_FORMAT = "robot-vla-dataset-hash-compatibility/v1"
D0_HISTORICAL_PROJECTION = "trajectory-meta-pre-local-dagger-null/v1"
D0_CURRENT_PROJECTION = "trajectory-meta-with-local-dagger-null/v2"
D0_NPZ_SET_HASH_SCHEME = "sorted-relative-file-sha256-pairs/v1"


@dataclass(frozen=True)
class _FrozenD0Expectation:
    historical_dataset_sha256: str
    current_dataset_sha256: str
    manifest_sha256: str
    audit_report_sha256: str
    proprio_stats_sha256: str
    proprio_stats_semantic_sha256: str
    npz_set_sha256: str
    trajectory_count: int
    step_count: int
    proprio_stats_count: int
    split_trajectory_counts: tuple[tuple[str, int], ...]
    split_step_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class _D0CompatibilityVerification:
    receipt: dict[str, Any]
    entries: tuple[TrajectoryMeta, ...]
    resolved_root: Path
    root_stat_identity: tuple[int, ...]


@dataclass(frozen=True)
class _StableFileRead:
    sha256: str
    payload: bytes | None
    stat_identity: tuple[int, ...]


_FROZEN_D0_EXPECTATION = _FrozenD0Expectation(
    historical_dataset_sha256=(
        "bc024b6b9c566ca9500945fb6ac262bf657bee713d8a5816229bdc8478139407"
    ),
    current_dataset_sha256=(
        "bb06628a01b1c55a388aecc18c57f1a773717175ca0e6d71f14268274b92c6cf"
    ),
    manifest_sha256=(
        "43f131cc1b79b93cf6e38f3f5e476d7a03fe29410daa720a672989c95afc477f"
    ),
    audit_report_sha256=(
        "b7ab50f0240d0795c49ebad7d8ed7695753b30c97cb72a7724a66162da81a24f"
    ),
    proprio_stats_sha256=(
        "fdad9119e3c86a26cc8c63ac2d38d6575a3f79dba475daff6bd290a0967ef64d"
    ),
    proprio_stats_semantic_sha256=(
        "e0638aa48d52d80270460c518d6423002a1be87e56d3b47a19d860de984d6e3a"
    ),
    npz_set_sha256=(
        "b6ea7f054bf9e93cab34273c0f6e6ebc852ab10a2a16568506d8b8eb44531e46"
    ),
    trajectory_count=220,
    step_count=48_922,
    proprio_stats_count=39_337,
    split_trajectory_counts=(("train", 176), ("val", 22), ("test", 22)),
    split_step_counts=(("train", 39_337), ("val", 4_618), ("test", 4_967)),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--boundary-type",
        choices=(AMENDED_FORMAL_BOUNDARY_TYPE,),
        required=True,
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        required=True,
        help="amended formal pool 的独立 seed 起点（含）",
    )
    parser.add_argument(
        "--seed-end-exclusive",
        type=int,
        required=True,
        help="amended formal pool 的 seed 终点（不含）；pool size 由两者显式冻结",
    )
    parser.add_argument("--qwen-context-layer", type=int, choices=(12, 24), default=12)
    parser.add_argument("--sampling-seed", type=int, default=52_012)
    parser.add_argument("--num-flow-steps", type=int, default=10)
    parser.add_argument("--recency-decay", type=float, default=0.5)
    parser.add_argument("--max-anomaly-replans", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _lexical_absolute(path: Path) -> Path:
    """返回不跟随 symlink 的绝对 lexical path。"""

    return Path(os.path.abspath(os.fspath(path)))


def _path_entry_exists(path: Path) -> bool:
    """使用 lstat 语义判断目录项，dangling symlink 也算存在。"""

    try:
        os.lstat(path)
    except FileNotFoundError:
        return False
    return True


def _pool_lock_path(output: Path) -> Path:
    absolute_output = _lexical_absolute(output)
    if not absolute_output.name:
        raise ValueError("formal output 不能是 filesystem root")
    parent = absolute_output.parent
    try:
        parent_stat = os.lstat(parent)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "formal output 的父目录必须在取得 single-writer lock 前存在"
        ) from exc
    if stat.S_ISLNK(parent_stat.st_mode) or not stat.S_ISDIR(parent_stat.st_mode):
        raise RuntimeError("formal output 父目录必须是非 symlink 普通目录")
    if parent.resolve(strict=True) != parent:
        raise RuntimeError("formal output 父目录路径禁止包含 symlink")
    if parent_stat.st_uid != os.geteuid() or parent_stat.st_mode & 0o022:
        raise PermissionError(
            "formal output 父目录必须由当前用户拥有，且 group/other 不可写"
        )
    if _path_entry_exists(absolute_output):
        output_stat = os.lstat(absolute_output)
        if stat.S_ISLNK(output_stat.st_mode) or not stat.S_ISDIR(output_stat.st_mode):
            raise RuntimeError("formal output 必须是非 symlink 普通目录")
        if absolute_output.resolve(strict=True) != absolute_output:
            raise RuntimeError("formal output 路径禁止包含 symlink")
        if output_stat.st_uid != os.geteuid() or output_stat.st_mode & 0o022:
            raise PermissionError(
                "formal output 必须由当前用户拥有，且 group/other 不可写"
            )
    return parent / f".{absolute_output.name}{_POOL_LOCK_SUFFIX}"


def _validate_lock_stat(value: os.stat_result, *, label: str) -> None:
    if not stat.S_ISREG(value.st_mode):
        raise RuntimeError(f"formal pool lock {label} 不是普通文件")
    if value.st_nlink != 1:
        raise RuntimeError(f"formal pool lock {label} 必须只有一个硬链接")


@contextmanager
def _formal_pool_lock(output: Path) -> Iterator[None]:
    """用持久 sibling inode 排除同一 output 的并发 new/resume runner。"""

    lock_path = _pool_lock_path(output)
    try:
        before = os.lstat(lock_path)
    except FileNotFoundError:
        before = None
    if before is not None:
        if stat.S_ISLNK(before.st_mode):
            raise RuntimeError("formal pool lock 禁止 symlink")
        _validate_lock_stat(before, label="path")

    flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_NONBLOCK
        | os.O_NOFOLLOW
        | os.O_CLOEXEC
    )
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise RuntimeError(f"无法安全打开 formal pool lock: {lock_path}") from exc
    locked = False
    try:
        opened = os.fstat(descriptor)
        _validate_lock_stat(opened, label="inode")
        linked = os.lstat(lock_path)
        if (
            linked.st_dev != opened.st_dev
            or linked.st_ino != opened.st_ino
        ):
            raise RuntimeError("formal pool lock path/inode 在打开时发生替换")
        _validate_lock_stat(linked, label="linked inode")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise RuntimeError(
                    f"同一 formal output 已有 runner 持锁: {output.resolve(strict=False)}"
                ) from exc
            raise RuntimeError("formal pool single-writer lock 获取失败") from exc
        locked = True
        linked_after_lock = os.lstat(lock_path)
        if (
            linked_after_lock.st_dev != opened.st_dev
            or linked_after_lock.st_ino != opened.st_ino
        ):
            raise RuntimeError("formal pool lock path/inode 在加锁时发生替换")
        _validate_lock_stat(linked_after_lock, label="locked inode")
        yield
    finally:
        if locked:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


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
        _fsync_directory(path.parent)
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
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    """让 atomic rename 本身也进入父目录的持久化边界。"""

    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _jsonl_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    """按 `_atomic_write_jsonl` 的精确字节格式计算 receipt。"""

    digest = hashlib.sha256()
    for row in rows:
        line = json.dumps(dict(row), sort_keys=True, allow_nan=False) + "\n"
        digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def _json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = (
        json.dumps(
            dict(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _compact_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_plain_path(
    path: Path,
    *,
    kind: str,
    label: str,
    context: str = "D0 compatibility",
) -> os.stat_result:
    try:
        value = os.lstat(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{context} 缺少 {label}: {path}") from exc
    if stat.S_ISLNK(value.st_mode):
        raise RuntimeError(f"{context} 禁止 {label} symlink: {path}")
    if kind == "file" and not stat.S_ISREG(value.st_mode):
        raise RuntimeError(f"{context} {label} 不是普通文件: {path}")
    if kind == "directory" and not stat.S_ISDIR(value.st_mode):
        raise RuntimeError(f"{context} {label} 不是目录: {path}")
    return value


def _stable_stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _read_plain_file_stable(
    path: Path,
    *,
    label: str,
    capture_bytes: bool,
    context: str = "D0 compatibility",
) -> _StableFileRead:
    """用同一 file descriptor 哈希，并拒绝 hardlink 与读中替换/改写。"""

    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError(f"{context} 无法安全打开 {label}: {path}") from exc
    chunks: list[bytes] | None = [] if capture_bytes else None
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise RuntimeError(f"{context} {label} 不是普通文件")
        if opened.st_nlink != 1:
            raise RuntimeError(f"{context} {label} 必须只有一个硬链接")
        linked = os.lstat(path)
        if stat.S_ISLNK(linked.st_mode):
            raise RuntimeError(f"{context} 禁止 {label} symlink")
        if _stable_stat_identity(linked) != _stable_stat_identity(opened):
            raise RuntimeError(f"{context} {label} path/inode 不一致")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 8 * 1024 * 1024):
            digest.update(chunk)
            if chunks is not None:
                chunks.append(chunk)
        after_read = os.fstat(descriptor)
        linked_after_read = os.lstat(path)
        expected_identity = _stable_stat_identity(opened)
        if (
            _stable_stat_identity(after_read) != expected_identity
            or _stable_stat_identity(linked_after_read) != expected_identity
        ):
            raise RuntimeError(f"{context} {label} 在哈希期间发生变化")
        payload = None if chunks is None else b"".join(chunks)
        return _StableFileRead(
            sha256=digest.hexdigest(),
            payload=payload,
            stat_identity=expected_identity,
        )
    finally:
        os.close(descriptor)


def _assert_plain_file_identity(
    path: Path,
    *,
    expected: tuple[int, ...],
    label: str,
    context: str = "D0 compatibility",
) -> None:
    value = _require_plain_path(
        path,
        kind="file",
        label=label,
        context=context,
    )
    if value.st_nlink != 1:
        raise RuntimeError(f"{context} {label} 必须只有一个硬链接")
    if _stable_stat_identity(value) != expected:
        raise RuntimeError(f"{context} {label} 在全局快照期间发生变化")


def _assert_plain_directory_identity(
    path: Path,
    *,
    expected: tuple[int, ...],
    label: str,
    context: str,
) -> None:
    value = _require_plain_path(
        path,
        kind="directory",
        label=label,
        context=context,
    )
    if _stable_stat_identity(value) != expected:
        raise RuntimeError(f"{context} {label} identity 发生变化")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"JSON object 存在重复 key: {key!r}")
        value[key] = item
    return value


def _strict_json_loads(payload: str, *, label: str) -> Any:
    def reject_non_finite_constant(value: str) -> Any:
        raise ValueError(f"{label} 禁止非有限 JSON constant: {value}")

    try:
        return json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=reject_non_finite_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} 不是有效 JSON") from exc


def _read_manifest_objects(payload: bytes) -> list[dict[str, Any]]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("D0 manifest 不是有效 UTF-8") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        row = _strict_json_loads(line, label=f"D0 manifest 第 {line_number} 行")
        if not isinstance(row, dict):
            raise TypeError(f"D0 manifest 第 {line_number} 行必须是 JSON object")
        rows.append(row)
    if not rows:
        raise ValueError("D0 manifest 不能为空")
    return rows


def _scan_plain_npz_files(root: Path) -> tuple[set[str], dict[str, tuple[int, ...]]]:
    """拒绝 D0 树内 symlink/特殊文件，并返回实际 NPZ 相对路径集合。"""

    found: set[str] = set()
    directories: dict[str, tuple[int, ...]] = {}
    for current, directory_names, file_names in os.walk(
        root,
        topdown=True,
        followlinks=False,
    ):
        current_path = Path(current)
        current_stat = _require_plain_path(
            current_path,
            kind="directory",
            label="dataset directory",
        )
        directories[current_path.relative_to(root).as_posix()] = (
            _stable_stat_identity(current_stat)
        )
        for name in directory_names:
            _require_plain_path(
                current_path / name,
                kind="directory",
                label="子目录",
            )
        for name in file_names:
            path = current_path / name
            _require_plain_path(path, kind="file", label="文件")
            if path.suffix == ".npz":
                found.add(path.relative_to(root).as_posix())
    return found, directories


def _resolve_plain_dataset_file(root: Path, relative_value: str) -> Path:
    relative = Path(relative_value)
    if (
        relative.is_absolute()
        or relative == Path(".")
        or ".." in relative.parts
        or "\\" in relative_value
        or relative.as_posix() != relative_value
        or relative.suffix != ".npz"
    ):
        raise ValueError(f"D0 trajectory file 路径非法: {relative_value!r}")
    cursor = root
    for index, component in enumerate(relative.parts):
        cursor /= component
        _require_plain_path(
            cursor,
            kind="file" if index == len(relative.parts) - 1 else "directory",
            label="trajectory 路径",
        )
    resolved = cursor.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError(f"D0 trajectory file 逃逸 dataset root: {relative_value!r}")
    return resolved


def _validate_expected_d0(expectation: _FrozenD0Expectation) -> None:
    for name in (
        "historical_dataset_sha256",
        "current_dataset_sha256",
        "manifest_sha256",
        "audit_report_sha256",
        "proprio_stats_sha256",
        "proprio_stats_semantic_sha256",
        "npz_set_sha256",
    ):
        if re.fullmatch(r"[0-9a-f]{64}", getattr(expectation, name)) is None:
            raise ValueError(f"D0 expectation {name} 不是小写 SHA256")
    for name in ("trajectory_count", "step_count", "proprio_stats_count"):
        value = getattr(expectation, name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"D0 expectation {name} 必须是正整数")
    split_trajectories = dict(expectation.split_trajectory_counts)
    split_steps = dict(expectation.split_step_counts)
    if set(split_trajectories) != {"train", "val", "test"} or any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in split_trajectories.values()
    ):
        raise ValueError("D0 expectation split_trajectory_counts 非法")
    if set(split_steps) != {"train", "val", "test"} or any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in split_steps.values()
    ):
        raise ValueError("D0 expectation split_step_counts 非法")
    if sum(split_trajectories.values()) != expectation.trajectory_count:
        raise ValueError("D0 expectation split trajectory totals 不闭合")
    if sum(split_steps.values()) != expectation.step_count:
        raise ValueError("D0 expectation split step totals 不闭合")
    if split_steps["train"] != expectation.proprio_stats_count:
        raise ValueError("D0 expectation train steps/proprio stats count 不闭合")


def _verify_d0_compatibility(
    data_root: Path,
    *,
    expectation: _FrozenD0Expectation,
) -> _D0CompatibilityVerification:
    """只读验证 frozen clean D0 在历史与当前 metadata 投影下的身份。"""

    from robot_vla.adapters import ProprioStats
    from robot_vla.contracts import RobotSpec

    _validate_expected_d0(expectation)
    input_root_stat = _require_plain_path(
        data_root,
        kind="directory",
        label="dataset root",
    )
    input_root_identity = _stable_stat_identity(input_root_stat)
    root = data_root.resolve(strict=True)
    resolved_root_stat = _require_plain_path(
        root,
        kind="directory",
        label="resolved dataset root",
    )
    if _stable_stat_identity(resolved_root_stat) != input_root_identity:
        raise RuntimeError("D0 dataset root resolve identity 不一致")
    manifest_path = root / "manifest.jsonl"
    audit_path = root / "audit_report.json"
    stats_path = root / "proprio_stats.json"
    for path, label in (
        (manifest_path, "manifest.jsonl"),
        (audit_path, "audit_report.json"),
        (stats_path, "proprio_stats.json"),
    ):
        _require_plain_path(path, kind="file", label=label)

    manifest_read = _read_plain_file_stable(
        manifest_path,
        label="manifest.jsonl",
        capture_bytes=True,
    )
    manifest_sha256 = manifest_read.sha256
    manifest_bytes = manifest_read.payload
    if manifest_sha256 != expectation.manifest_sha256:
        raise ValueError("D0 manifest SHA256 与 frozen identity 不一致")
    if manifest_bytes is None:  # pragma: no cover - capture_bytes=True contract
        raise RuntimeError("D0 manifest stable reader 未返回 bytes")
    raw_rows = _read_manifest_objects(manifest_bytes)
    if any("local_dagger" in row for row in raw_rows):
        raise ValueError(
            "legacy D0 原始 manifest 禁止出现 local_dagger；"
            "不得把 current-schema 数据降级投影"
        )
    entries: list[TrajectoryMeta] = []
    seen_trajectory_ids: set[str] = set()
    seen_seeds: set[int] = set()
    source_splits: dict[str, str] = {}
    scene_splits: dict[str, str] = {}
    for row_index, row in enumerate(raw_rows, start=1):
        try:
            entry = TrajectoryMeta.from_dict(row)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"D0 manifest 第 {row_index} 行 metadata 无效") from exc
        if entry.trajectory_id in seen_trajectory_ids:
            raise ValueError(f"D0 trajectory_id 重复: {entry.trajectory_id}")
        seen_trajectory_ids.add(entry.trajectory_id)
        for label, mapping, key in (
            ("source_episode_id", source_splits, entry.source_episode_id),
            ("scene_id", scene_splits, entry.scene_id),
        ):
            previous = mapping.setdefault(key, entry.split)
            if previous != entry.split:
                raise ValueError(f"D0 {label}={key!r} 跨越 split")
        seed = entry.randomization.get("seed")
        if (
            not isinstance(seed, int)
            or isinstance(seed, bool)
            or seed < 0
            or seed in seen_seeds
        ):
            raise ValueError(f"D0 randomization.seed 缺失、非法或重复: {seed!r}")
        seen_seeds.add(seed)
        entries.append(entry)
    if len(entries) != expectation.trajectory_count:
        raise ValueError("D0 trajectory_count 与 frozen identity 不一致")
    step_count = sum(entry.num_steps for entry in entries)
    if step_count != expectation.step_count:
        raise ValueError("D0 step_count 与 frozen identity 不一致")
    split_trajectory_counts = dict.fromkeys(("train", "val", "test"), 0)
    split_step_counts = dict.fromkeys(("train", "val", "test"), 0)
    for entry in entries:
        split_trajectory_counts[entry.split] += 1
        split_step_counts[entry.split] += entry.num_steps
    if split_trajectory_counts != dict(expectation.split_trajectory_counts):
        raise ValueError("D0 split trajectory counts 与 frozen identity 不一致")
    if split_step_counts != dict(expectation.split_step_counts):
        raise ValueError("D0 split step counts 与 frozen identity 不一致")

    actual_npz_files, initial_directory_identities = _scan_plain_npz_files(root)
    file_identities: dict[Path, tuple[int, ...]] = {
        manifest_path: manifest_read.stat_identity,
    }
    referenced_files: set[str] = set()
    current_files: list[dict[str, Any]] = []
    historical_files: list[dict[str, Any]] = []
    npz_receipts: list[dict[str, str]] = []
    for raw_row, entry in zip(raw_rows, entries, strict=True):
        if str(raw_row.get("trajectory_id", "")) != entry.trajectory_id:
            raise ValueError("D0 raw/parsed trajectory 顺序或 identity 漂移")
        if entry.local_dagger is not None:
            raise ValueError("legacy D0 parsed metadata 禁止包含 Local DAgger provenance")
        relative_file = Path(entry.file).as_posix()
        if relative_file in referenced_files:
            raise ValueError(f"D0 manifest 重复引用 trajectory file: {relative_file}")
        trajectory_path = _resolve_plain_dataset_file(root, entry.file)
        referenced_files.add(relative_file)
        npz_read = _read_plain_file_stable(
            trajectory_path,
            label=f"trajectory NPZ {relative_file}",
            capture_bytes=False,
        )
        npz_sha256 = npz_read.sha256
        file_identities[trajectory_path] = npz_read.stat_identity
        current_meta = entry.to_dict()
        if current_meta.get("local_dagger", object()) is not None:
            raise ValueError("legacy D0 current projection 必须只注入 local_dagger:null")
        historical_meta = dict(current_meta)
        del historical_meta["local_dagger"]
        current_files.append({"meta": current_meta, "npz_sha256": npz_sha256})
        historical_files.append(
            {"meta": historical_meta, "npz_sha256": npz_sha256}
        )
        npz_receipts.append({"file": relative_file, "sha256": npz_sha256})

    if referenced_files != actual_npz_files:
        missing = sorted(referenced_files - actual_npz_files)
        extra = sorted(actual_npz_files - referenced_files)
        raise ValueError(
            "D0 manifest/NPZ 集合不一致: "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    historical_sha256 = _compact_json_sha256(
        sorted(historical_files, key=lambda item: str(item["meta"]))
    )
    current_sha256 = _compact_json_sha256(
        sorted(current_files, key=lambda item: str(item["meta"]))
    )
    npz_set_sha256 = _compact_json_sha256(
        sorted(npz_receipts, key=lambda item: item["file"])
    )
    if historical_sha256 != expectation.historical_dataset_sha256:
        raise ValueError("D0 historical projection SHA256 与 frozen identity 不一致")
    if current_sha256 != expectation.current_dataset_sha256:
        raise ValueError("D0 current projection SHA256 与 frozen translation 不一致")
    if npz_set_sha256 != expectation.npz_set_sha256:
        raise ValueError("D0 NPZ set SHA256 与 frozen content receipt 不一致")

    audit_read = _read_plain_file_stable(
        audit_path,
        label="audit_report.json",
        capture_bytes=True,
    )
    audit_sha256 = audit_read.sha256
    audit_bytes = audit_read.payload
    file_identities[audit_path] = audit_read.stat_identity
    if audit_sha256 != expectation.audit_report_sha256:
        raise ValueError("D0 audit_report SHA256 与 frozen identity 不一致")
    if audit_bytes is None:  # pragma: no cover - capture_bytes=True contract
        raise RuntimeError("D0 audit stable reader 未返回 bytes")
    try:
        audit_text = audit_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("D0 audit_report 不是有效 UTF-8") from exc
    audit_report = _strict_json_loads(audit_text, label="D0 audit_report")
    if not isinstance(audit_report, dict):
        raise TypeError("D0 audit_report 必须是 JSON object")
    expected_audit_fields = {
        "dataset_sha256": historical_sha256,
        "manifest_sha256": manifest_sha256,
        "trajectory_count": expectation.trajectory_count,
        "step_count": expectation.step_count,
        "proprio_stats_count": expectation.proprio_stats_count,
        "split_trajectory_counts": dict(expectation.split_trajectory_counts),
        "split_step_counts": dict(expectation.split_step_counts),
        "success_rate": 1.0,
    }
    if any(audit_report.get(key) != value for key, value in expected_audit_fields.items()):
        raise ValueError("D0 audit_report identity/count 字段与 frozen receipt 不一致")

    stats_read = _read_plain_file_stable(
        stats_path,
        label="proprio_stats.json",
        capture_bytes=True,
    )
    stats_sha256 = stats_read.sha256
    stats_bytes = stats_read.payload
    file_identities[stats_path] = stats_read.stat_identity
    if stats_sha256 != expectation.proprio_stats_sha256:
        raise ValueError("D0 proprio_stats SHA256 与 frozen identity 不一致")
    if stats_bytes is None:  # pragma: no cover - capture_bytes=True contract
        raise RuntimeError("D0 proprio stats stable reader 未返回 bytes")
    try:
        stats_text = stats_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("D0 proprio_stats 不是有效 UTF-8") from exc
    stats_payload = _strict_json_loads(stats_text, label="D0 proprio_stats")
    expected_stats_keys = {
        "version",
        "schema_version",
        "embodiment",
        "mean",
        "std",
        "count",
    }
    if not isinstance(stats_payload, dict) or set(stats_payload) != expected_stats_keys:
        raise ValueError("D0 proprio_stats 顶层 schema 漂移")
    stats = ProprioStats(
        mean=tuple(float(value) for value in stats_payload["mean"]),
        std=tuple(float(value) for value in stats_payload["std"]),
        count=int(stats_payload["count"]),
        version=str(stats_payload["version"]),
        schema_version=str(stats_payload["schema_version"]),
        embodiment=str(stats_payload["embodiment"]),
    )
    stats.validate(RobotSpec())
    if stats.count != expectation.proprio_stats_count:
        raise ValueError("D0 proprio_stats count 与 frozen identity 不一致")
    stats_semantic_sha256 = _compact_json_sha256(asdict(stats))
    if stats_semantic_sha256 != expectation.proprio_stats_semantic_sha256:
        raise ValueError("D0 proprio_stats semantic SHA256 与 frozen identity 不一致")

    final_npz_files, final_directory_identities = _scan_plain_npz_files(root)
    if final_npz_files != actual_npz_files:
        raise RuntimeError("D0 NPZ 集合在全局快照期间发生变化")
    if final_directory_identities != initial_directory_identities:
        raise RuntimeError("D0 directory identity 在全局快照期间发生变化")
    for path, expected_identity in file_identities.items():
        _assert_plain_file_identity(
            path,
            expected=expected_identity,
            label=path.relative_to(root).as_posix(),
        )
    try:
        resolved_after_verification = data_root.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError("D0 dataset root 在全局快照期间不可解析") from exc
    if resolved_after_verification != root:
        raise RuntimeError("D0 dataset root 在全局快照期间发生替换")
    final_input_root_stat = _require_plain_path(
        data_root,
        kind="directory",
        label="dataset root",
    )
    if _stable_stat_identity(final_input_root_stat) != input_root_identity:
        raise RuntimeError("D0 dataset root identity 在全局快照期间发生变化")

    receipt = {
        "format": D0_COMPATIBILITY_FORMAT,
        "status": "matched",
        "historical": {
            "projection": D0_HISTORICAL_PROJECTION,
            "dataset_sha256": historical_sha256,
        },
        "current": {
            "projection": D0_CURRENT_PROJECTION,
            "dataset_sha256": current_sha256,
        },
        "translation": {
            "operation": "add-top-level-local_dagger-null",
            "translated_row_count": len(entries),
        },
        "manifest": {
            "sha256": manifest_sha256,
            "trajectory_count": len(entries),
            "split_trajectory_counts": split_trajectory_counts,
            "split_step_counts": split_step_counts,
        },
        "trajectory_files": {
            "hash_scheme": D0_NPZ_SET_HASH_SCHEME,
            "count": len(npz_receipts),
            "aggregate_sha256": npz_set_sha256,
        },
        "step_count": step_count,
        "audit_report": {"sha256": audit_sha256},
        "proprio_stats": {
            "sha256": stats_sha256,
            "semantic_sha256": stats_semantic_sha256,
            "count": stats.count,
        },
    }
    return _D0CompatibilityVerification(
        receipt=json.loads(json.dumps(receipt, sort_keys=True, allow_nan=False)),
        entries=tuple(entries),
        resolved_root=root,
        root_stat_identity=input_root_identity,
    )


def verify_frozen_d0_compatibility(
    data_root: Path,
) -> _D0CompatibilityVerification:
    """正式 E012 runner 的固定 D0 pre-rollout compatibility gate。"""

    return _verify_d0_compatibility(
        data_root,
        expectation=_FROZEN_D0_EXPECTATION,
    )


def _verify_candidate_stats_leaf(
    identity: Mapping[str, Any],
    *,
    expected_root_identity: tuple[int, ...] | None = None,
) -> None:
    data_root = Path(identity["base_dataset"]["path"])
    if expected_root_identity is not None:
        _assert_plain_directory_identity(
            data_root,
            expected=expected_root_identity,
            label="D0 dataset root",
            context="candidate 启动前 input gate",
        )
    stats_path = data_root / "proprio_stats.json"
    observed = _read_plain_file_stable(
        stats_path,
        label="candidate proprio_stats.json",
        capture_bytes=False,
        context="candidate 启动前 input gate",
    )
    expected_sha256 = identity["base_dataset"]["compatibility"]["proprio_stats"][
        "sha256"
    ]
    if observed.sha256 != expected_sha256:
        raise ValueError("candidate 启动前 D0 proprio_stats SHA256 漂移")


def _read_formal_checkpoint(path: Path) -> tuple[Path, _StableFileRead]:
    absolute = _lexical_absolute(path)
    _require_plain_path(
        absolute,
        kind="file",
        label="checkpoint",
        context="amended formal input gate",
    )
    if absolute.resolve(strict=True) != absolute:
        raise RuntimeError("amended formal checkpoint 路径禁止包含 symlink")
    stable = _read_plain_file_stable(
        absolute,
        label="checkpoint",
        capture_bytes=False,
        context="amended formal input gate",
    )
    return absolute, stable


def _verify_candidate_checkpoint_leaf(
    identity: Mapping[str, Any],
    *,
    expected_stat_identity: tuple[int, ...],
) -> None:
    checkpoint_path = Path(identity["checkpoint"]["path"])
    _assert_plain_file_identity(
        checkpoint_path,
        expected=expected_stat_identity,
        label="checkpoint",
        context="candidate 启动前 input gate",
    )


def _read_experiment_identity(
    experiment_path: Path,
) -> tuple[dict[str, Any], _StableFileRead]:
    absolute = _lexical_absolute(experiment_path)
    output_root = absolute.parent
    if absolute.name != "experiment.json":
        raise ValueError("pool experiment receipt 路径必须是 experiment.json")
    _require_plain_path(
        output_root,
        kind="directory",
        label="formal output root",
        context="formal pool experiment",
    )
    if output_root.resolve(strict=True) != output_root:
        raise RuntimeError("formal pool experiment output 路径禁止包含 symlink")
    stable = _read_plain_file_stable(
        absolute,
        label="experiment.json",
        capture_bytes=True,
        context="formal pool experiment",
    )
    if stable.payload is None:  # pragma: no cover - capture_bytes=True contract
        raise RuntimeError("formal pool experiment stable reader 未返回 bytes")
    try:
        text = stable.payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("formal pool experiment 不是有效 UTF-8") from exc
    value = _strict_json_loads(text, label="formal pool experiment.json")
    if not isinstance(value, dict):
        raise TypeError("formal pool experiment.json 必须是 JSON object")
    return value, stable


def _experiment_receipt(experiment_path: Path) -> dict[str, str]:
    _, stable = _read_experiment_identity(experiment_path)
    return {
        "experiment": str(_lexical_absolute(experiment_path)),
        "experiment_sha256": stable.sha256,
    }


def _candidate_manifest_receipt(
    path: Path,
    *,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved.name != "collection_candidates.jsonl" or not resolved.is_file():
        raise FileNotFoundError("collection candidates receipt 不存在")
    observed_sha256 = _sha256_file(resolved)
    if observed_sha256 != _jsonl_sha256(rows):
        raise ValueError("collection_candidates SHA256 与 rows 不一致")
    return _candidate_snapshot_receipt(resolved, rows=rows)


def _candidate_snapshot_receipt(
    path: Path,
    *,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """描述 canonical JSONL 的已发布前缀；不声称路径当前仍等于该前缀。"""

    return {
        "collection_candidates": str(path.resolve()),
        "collection_candidates_row_count": len(rows),
        "collection_candidates_sha256_scope": "jsonl_prefix",
        "collection_candidates_sha256": _jsonl_sha256(rows),
    }


def _remove_stale_root_derived_temps(output: Path) -> None:
    """清理已确认 experiment 身份后遗留的本 runner 原子写临时文件。"""

    patterns = tuple(
        re.compile(rf"^\.{re.escape(name)}\.[a-z0-9_]{{8}}\.tmp$")
        for name in _ROOT_DERIVED_ATOMIC_FILENAMES
    )
    for path in output.iterdir():
        if not any(pattern.fullmatch(path.name) for pattern in patterns):
            continue
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"--resume 原子写临时路径类型非法: {path.name}")
        path.unlink()
    _fsync_directory(output)


def _query_nvidia_driver_identity(properties: Any) -> tuple[str, str]:
    """用 resolved GPU UUID 精确查询单卡 driver；不经 shell。"""

    raw_uuid = str(getattr(properties, "uuid", "")).strip().lower()
    if _GPU_UUID_PATTERN.fullmatch(raw_uuid) is None:
        raise RuntimeError(
            "CUDA device properties 缺少合法 physical GPU UUID；"
            "amended formal runner 要求 PyTorch >= 2.5 的 UUID 支持"
        )
    expected_uuid = f"GPU-{raw_uuid}"

    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise RuntimeError("正式 runtime 找不到可执行 nvidia-smi")
    try:
        command = Path(executable).resolve(strict=True)
    except OSError as exc:
        raise RuntimeError("正式 runtime 无法解析 nvidia-smi") from exc
    if not command.is_file() or not os.access(command, os.X_OK):
        raise RuntimeError("正式 runtime 的 nvidia-smi 不是可执行普通文件")

    try:
        completed = subprocess.run(
            [
                str(command),
                f"--id={expected_uuid}",
                "--query-gpu=uuid,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("nvidia-smi runtime identity 查询失败") from exc
    if completed.returncode != 0:
        raise RuntimeError(
            "nvidia-smi runtime identity 查询返回非零状态: "
            f"{completed.returncode}"
        )
    lines = completed.stdout.splitlines()
    if len(lines) != 1:
        raise RuntimeError("nvidia-smi runtime identity 必须精确返回一行")
    fields = [field.strip() for field in lines[0].split(",")]
    if len(fields) != 2:
        raise RuntimeError("nvidia-smi runtime identity schema 漂移")
    observed_uuid, driver_version = fields
    if observed_uuid.lower() != expected_uuid.lower():
        raise RuntimeError("nvidia-smi GPU UUID 与 CUDA device properties 不一致")
    if _NVIDIA_DRIVER_VERSION_PATTERN.fullmatch(driver_version) is None:
        raise RuntimeError("nvidia-smi driver version schema 漂移")
    return expected_uuid, driver_version


def _build_runtime_identity() -> dict[str, Any]:
    """冻结会影响正式 rollout / resume 的关键 runtime 身份。"""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - formal GPU 环境前置条件
        raise RuntimeError("amended formal runner 需要 PyTorch runtime") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("amended formal runner 需要可用 CUDA runtime")

    packages: dict[str, str] = {}
    missing: list[str] = []
    for key, distribution in _RUNTIME_DISTRIBUTIONS.items():
        try:
            packages[key] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            missing.append(distribution)
    if missing:
        raise RuntimeError(f"正式 runtime 缺少版本身份: {sorted(missing)}")

    device_index = int(torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device_index)
    device_uuid, driver_version = _query_nvidia_driver_identity(properties)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "cuda": {
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device_index": device_index,
            "device_name": properties.name,
            "device_uuid": device_uuid,
            "compute_capability": [int(properties.major), int(properties.minor)],
            "total_memory_bytes": int(properties.total_memory),
            "nvidia_driver_version": driver_version,
        },
    }


def _freeze_d0_compatibility(
    compatibility: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(compatibility, Mapping) or not compatibility:
        raise ValueError("formal pool 必须冻结非空 D0 compatibility receipt")
    frozen = json.loads(json.dumps(compatibility, sort_keys=True, allow_nan=False))
    historical = frozen.get("historical")
    current = frozen.get("current")
    translation = frozen.get("translation")
    manifest = frozen.get("manifest")
    trajectory_files = frozen.get("trajectory_files")
    audit_report = frozen.get("audit_report")
    stats = frozen.get("proprio_stats")
    if frozen.get("format") != D0_COMPATIBILITY_FORMAT:
        raise ValueError("D0 compatibility format 不兼容")
    if frozen.get("status") != "matched":
        raise ValueError("D0 compatibility 必须是 matched receipt")
    for name, value in (
        ("historical", historical),
        ("current", current),
        ("translation", translation),
        ("manifest", manifest),
        ("trajectory_files", trajectory_files),
        ("audit_report", audit_report),
        ("proprio_stats", stats),
    ):
        if not isinstance(value, dict):
            raise TypeError(f"D0 compatibility {name} 必须是 JSON object")
    if historical.get("projection") != D0_HISTORICAL_PROJECTION:
        raise ValueError("D0 historical projection version 漂移")
    if current.get("projection") != D0_CURRENT_PROJECTION:
        raise ValueError("D0 current projection version 漂移")
    if translation != {
        "operation": "add-top-level-local_dagger-null",
        "translated_row_count": manifest.get("trajectory_count"),
    }:
        raise ValueError("D0 compatibility translation receipt 漂移")
    trajectory_count = manifest.get("trajectory_count")
    step_count = frozen.get("step_count")
    stats_count = stats.get("count")
    for name, value in (
        ("trajectory_count", trajectory_count),
        ("step_count", step_count),
        ("proprio_stats_count", stats_count),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise TypeError(f"D0 compatibility {name} 必须是正整数")
    if trajectory_files.get("count") != trajectory_count:
        raise ValueError("D0 compatibility manifest/NPZ count 不一致")
    if trajectory_files.get("hash_scheme") != D0_NPZ_SET_HASH_SCHEME:
        raise ValueError("D0 compatibility NPZ set hash scheme 漂移")
    for label, value in (
        ("historical dataset", historical.get("dataset_sha256")),
        ("current dataset", current.get("dataset_sha256")),
        ("manifest", manifest.get("sha256")),
        ("NPZ set", trajectory_files.get("aggregate_sha256")),
        ("audit report", audit_report.get("sha256")),
        ("proprio stats", stats.get("sha256")),
        ("proprio stats semantic", stats.get("semantic_sha256")),
    ):
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"D0 compatibility {label} 缺少小写 SHA256")
    split_trajectory_counts = manifest.get("split_trajectory_counts")
    split_step_counts = manifest.get("split_step_counts")
    if (
        not isinstance(split_trajectory_counts, dict)
        or set(split_trajectory_counts) != {"train", "val", "test"}
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in split_trajectory_counts.values()
        )
        or sum(split_trajectory_counts.values()) != trajectory_count
    ):
        raise ValueError("D0 compatibility split trajectory counts 不闭合")
    if (
        not isinstance(split_step_counts, dict)
        or set(split_step_counts) != {"train", "val", "test"}
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in split_step_counts.values()
        )
        or sum(split_step_counts.values()) != step_count
        or split_step_counts["train"] != stats_count
    ):
        raise ValueError("D0 compatibility split step/stats counts 不闭合")
    return frozen, {
        "dataset_sha256": historical["dataset_sha256"],
        "manifest_sha256": manifest["sha256"],
        "trajectory_count": trajectory_count,
        "step_count": step_count,
    }


def build_pool_identity(
    args: argparse.Namespace,
    *,
    source_revision: str,
    base_dataset_root: Path,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    base_dataset_compatibility: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
) -> dict[str, Any]:
    start = int(args.seed_start)
    end = int(args.seed_end_exclusive)
    if not base_dataset_root.is_absolute() or not checkpoint_path.is_absolute():
        raise ValueError("formal D0/checkpoint identity 必须使用冻结绝对路径")
    if args.boundary_type != AMENDED_FORMAL_BOUNDARY_TYPE:
        raise ValueError("E012 amended formal runner 只允许 grasp_lift boundary")
    if start < 0 or end <= start:
        raise ValueError("formal seed range 必须是非负且非空的半开区间")
    if start != AMENDED_FORMAL_SEED_START:
        raise ValueError(
            f"amended formal seed 必须从预留起点 {AMENDED_FORMAL_SEED_START} 开始"
        )
    if end > AMENDED_FORMAL_SEED_END_LIMIT:
        raise ValueError(
            "amended formal seed range 不得进入 checkpoint validation 保留区 "
            f"[{AMENDED_FORMAL_SEED_END_LIMIT}, ...)"
        )
    if end - start < ELIGIBLE_SELECTION_GATE:
        raise ValueError("formal seed pool 小于 20，不可能通过冻结 selection gate")
    if not isinstance(runtime_identity, Mapping) or not runtime_identity:
        raise ValueError("formal pool 必须冻结非空 runtime identity")
    frozen_runtime = json.loads(
        json.dumps(runtime_identity, sort_keys=True, allow_nan=False)
    )
    frozen_compatibility, base_dataset_identity = _freeze_d0_compatibility(
        base_dataset_compatibility
    )
    action_budget_protocol = resolve_local_dagger_action_budget(
        AMENDED_ACTION_BUDGET_PROTOCOL
    ).planned_metadata()
    if action_budget_protocol is None:
        raise RuntimeError("amended formal pool 缺少 segmented action-budget metadata")
    return {
        "format": POOL_FORMAT,
        "source_revision": source_revision,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
        },
        "base_dataset": {
            "path": str(base_dataset_root),
            "audit": base_dataset_identity,
            "compatibility": frozen_compatibility,
        },
        "qwen": {
            "model_id": QWEN_MODEL_ID,
            "revision": QWEN_REVISION,
            "cache_path": str(args.model_cache.resolve()),
        },
        "runtime": frozen_runtime,
        "boundary_type": args.boundary_type,
        "seed_registry": {
            "reserved_amended_formal_range": {
                "start": AMENDED_FORMAL_SEED_START,
                "end_exclusive": AMENDED_FORMAL_SEED_END_LIMIT,
            },
            "legacy_e012a_end_exclusive": AMENDED_FORMAL_SEED_START,
            "checkpoint_validation_start": AMENDED_FORMAL_SEED_END_LIMIT,
        },
        "seed_start": start,
        "seed_end_exclusive": end,
        "pool_size": end - start,
        "environment_seeds": list(range(start, end)),
        "action_budget_protocol": action_budget_protocol,
        "paired_clean_expert_protocol": dict(PAIRED_CLEAN_EXPERT_PROTOCOL),
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
                "eligible_gate": ELIGIBLE_SELECTION_GATE,
                "high_count": HIGH_RISK_SELECTION_COUNT,
                "low_count": LOW_RISK_SELECTION_COUNT,
                "tie_break": "environment_seed ascending",
                "overlap_resolution": "select high first, then low from remaining",
            },
        },
        "d1_input_contract": {
            "mode": "canonical accepted+selected records only",
            "canonical_manifest": CANONICAL_SELECTED_RECORDS_FILENAME,
            "required_candidate_status": "accepted",
            "required_selection": True,
            "required_selection_receipt": {
                "source": RISK_SELECTION_FILENAME,
                "sha256": True,
                "membership_fields": [
                    "environment_seed",
                    "selection_stratum",
                    "risk_score",
                    "index",
                ],
            },
            "directory_scan": "forbidden",
            "npz_directory_scan": "forbidden",
        },
    }


def _expected_candidate_config(
    identity: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    pool_config = identity["config"]
    return {
        "environment_seed": seed,
        "boundary_type": identity["boundary_type"],
        "sampling_seed_base": pool_config["sampling_seed_base"],
        "episode_sampling_seed": derive_collection_sampling_seed(
            int(pool_config["sampling_seed_base"]),
            environment_seed=seed,
            boundary_type=str(identity["boundary_type"]),
        ),
        "num_flow_steps": pool_config["num_flow_steps"],
        "recency_decay": pool_config["recency_decay"],
        "max_anomaly_replans": pool_config["max_anomaly_replans"],
        "qwen_context_layer": pool_config["qwen_context_layer"],
        "snapshot_round_trip_required": True,
        "paired_clean_expert_required": True,
        LOCAL_DAGGER_ACTION_BUDGET_PROTOCOL_FIELD: identity[
            "action_budget_protocol"
        ],
    }


def _validate_action_budget_usage(
    record: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
    seed: int,
) -> dict[str, int]:
    usage = record.get(LOCAL_DAGGER_ACTION_BUDGET_USAGE_FIELD)
    required_keys = {"policy_actions", "expert_actions", "total_actions"}
    if not isinstance(usage, dict) or set(usage) != required_keys:
        raise ValueError(f"seed {seed}: action-budget usage schema 漂移")
    if any(
        not isinstance(value, int) or isinstance(value, bool)
        for value in usage.values()
    ):
        raise TypeError(f"seed {seed}: action-budget usage 必须是精确整数")
    if any(value < 0 for value in usage.values()):
        raise ValueError(f"seed {seed}: action-budget usage 不得为负数")
    if usage["policy_actions"] + usage["expert_actions"] != usage["total_actions"]:
        raise ValueError(f"seed {seed}: Policy/Expert/total action 计数不闭合")

    planned = identity["action_budget_protocol"]
    if usage["policy_actions"] > planned["policy_action_limit"]:
        raise ValueError(f"seed {seed}: Policy actions 超出冻结上限")
    if usage["expert_actions"] > planned["expert_action_limit"]:
        raise ValueError(f"seed {seed}: Expert actions 超出冻结上限")
    if usage["total_actions"] > planned["environment_action_limit"]:
        raise ValueError(f"seed {seed}: total actions 超出环境 hard limit")
    return {key: int(usage[key]) for key in sorted(required_keys)}


def _validate_accepted_record(
    record: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
    seed: int,
    usage: Mapping[str, int],
) -> None:
    if record.get("eligible_for_risk_selection") is not True:
        raise ValueError(f"seed {seed}: accepted candidate 未通过 eligible gate")
    if record.get("audit") != {
        "trajectory_contract": "passed",
        "full_dataset_audit": "pending D0 union",
    }:
        raise ValueError(f"seed {seed}: accepted candidate audit contract 漂移")
    result = record.get("result")
    if not isinstance(result, dict):
        raise TypeError(f"seed {seed}: accepted candidate 缺少 result")
    if result.get(LOCAL_DAGGER_ACTION_BUDGET_USAGE_FIELD) != usage:
        raise ValueError(f"seed {seed}: result/action-budget usage 不一致")
    snapshot = result.get("snapshot_round_trip")
    if not isinstance(snapshot, dict) or snapshot.get("passed") is not True:
        raise ValueError(f"seed {seed}: snapshot round-trip 未通过")

    trajectory = result.get("trajectory")
    if not isinstance(trajectory, dict):
        raise TypeError(f"seed {seed}: accepted candidate 缺少 trajectory")
    if trajectory.get("num_steps") != usage["total_actions"]:
        raise ValueError(f"seed {seed}: trajectory/action-budget total 不一致")
    if usage["total_actions"] >= identity["action_budget_protocol"][
        "environment_action_limit"
    ]:
        raise ValueError(f"seed {seed}: accepted success 必须严格早于 hard deadline")
    randomization = trajectory.get("randomization")
    if not isinstance(randomization, dict):
        raise TypeError(f"seed {seed}: trajectory randomization 缺失")
    if randomization.get("seed") != seed:
        raise ValueError(f"seed {seed}: trajectory environment seed 漂移")
    if (
        randomization.get(LOCAL_DAGGER_ACTION_BUDGET_PROTOCOL_FIELD)
        != identity["action_budget_protocol"]
    ):
        raise ValueError(f"seed {seed}: trajectory action-budget protocol 漂移")
    if randomization.get(LOCAL_DAGGER_ACTION_BUDGET_USAGE_FIELD) != usage:
        raise ValueError(f"seed {seed}: trajectory action-budget usage 漂移")

    provenance = trajectory.get("local_dagger")
    if not isinstance(provenance, dict):
        raise TypeError(f"seed {seed}: trajectory 缺少 Local DAgger provenance")
    takeover = provenance.get("expert_takeover_step")
    if not isinstance(takeover, int) or isinstance(takeover, bool):
        raise TypeError(f"seed {seed}: Expert takeover step 必须是整数")
    if (
        provenance.get("source") != "dagger_grasp_lift"
        or provenance.get("rollin_seed") != seed
        or provenance.get("rollin_policy_checkpoint_sha256")
        != identity["checkpoint"]["sha256"]
        or provenance.get("boundary_type") != identity["boundary_type"]
        or provenance.get("boundary_detection_step") != takeover
        or provenance.get("training_window_start") != takeover
        or provenance.get("expert_recovery_success") is not True
    ):
        raise ValueError(f"seed {seed}: Local DAgger provenance 漂移")
    if takeover != usage["policy_actions"]:
        raise ValueError(f"seed {seed}: takeover/Policy action 计数不一致")
    expected_window_end = takeover + 64
    if (
        expected_window_end > usage["total_actions"]
        or provenance.get("training_window_end") != expected_window_end
    ):
        raise ValueError(f"seed {seed}: 缺少完整 64-action training window")
    outcome = trajectory.get("outcome_evidence")
    if not isinstance(outcome, dict) or outcome.get("task_completed") is not True:
        raise ValueError(f"seed {seed}: accepted trajectory 缺少完整成功证据")

    policy_boundary = result.get("boundary")
    paired = record.get("paired_clean_expert")
    if (
        not isinstance(policy_boundary, dict)
        or policy_boundary.get("boundary_type") != identity["boundary_type"]
        or policy_boundary.get("control_step") != takeover
    ):
        raise ValueError(f"seed {seed}: Policy boundary/takeover identity 漂移")
    if not isinstance(paired, dict) or paired.get("task_completed") is not True:
        raise ValueError(f"seed {seed}: paired clean Expert 未完成任务")
    paired_steps = paired.get("num_steps")
    if (
        not isinstance(paired_steps, int)
        or isinstance(paired_steps, bool)
        or not 0 < paired_steps <= PAIRED_CLEAN_EXPERT_PROTOCOL["environment_action_limit"]
    ):
        raise ValueError(f"seed {seed}: paired clean Expert 不满足 legacy-300")
    paired_boundary = paired.get("boundary")
    if (
        not isinstance(paired_boundary, dict)
        or paired_boundary.get("boundary_type") != identity["boundary_type"]
    ):
        raise ValueError(f"seed {seed}: paired clean Expert boundary identity 漂移")
    paired_control_step = paired_boundary.get("control_step")
    if (
        not isinstance(paired_control_step, int)
        or isinstance(paired_control_step, bool)
        or not 0 < paired_control_step <= paired_steps
    ):
        raise ValueError(f"seed {seed}: paired boundary control_step 超出 rollout")
    risk_components = record.get("risk_components")
    if (
        not isinstance(risk_components, dict)
        or set(risk_components) != set(RISK_COMPONENT_UNITS[identity["boundary_type"]])
    ):
        raise ValueError(f"seed {seed}: paired risk component schema 漂移")
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in risk_components.values()
    ):
        raise ValueError(f"seed {seed}: paired risk components 必须是有限非负数")
    expected_risk_components = compute_paired_risk_components(
        str(identity["boundary_type"]),
        policy_boundary,
        paired_boundary,
    )
    if risk_components != expected_risk_components:
        raise ValueError(f"seed {seed}: paired risk components 未按冻结公式精确重算")


def _validate_rejected_record(
    record: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
    seed: int,
    usage: Mapping[str, int],
) -> None:
    failure = record.get("failure")
    if (
        not isinstance(failure, dict)
        or failure.get("type") != "EpisodeRejected"
        or not isinstance(failure.get("reason"), str)
        or not failure["reason"]
    ):
        raise ValueError(f"seed {seed}: rejected candidate failure contract 漂移")
    diagnostics = record.get("failure_diagnostics")
    if not isinstance(diagnostics, dict):
        raise TypeError(f"seed {seed}: rejected candidate 缺少 failure diagnostics")
    expert_takeover_step = diagnostics.get("expert_takeover_step")
    if (
        diagnostics.get("format") != LOCAL_DAGGER_DIAGNOSTIC_FORMAT
        or diagnostics.get("environment_seed") != seed
        or diagnostics.get("boundary_type") != identity["boundary_type"]
        or diagnostics.get("failure_reason") != failure["reason"]
        or diagnostics.get(LOCAL_DAGGER_ACTION_BUDGET_PROTOCOL_FIELD)
        != identity["action_budget_protocol"]
        or diagnostics.get(LOCAL_DAGGER_ACTION_BUDGET_USAGE_FIELD) != usage
        or diagnostics.get("action_count") != usage["total_actions"]
        or (
            usage["expert_actions"] > 0
            and expert_takeover_step != usage["policy_actions"]
        )
    ):
        raise ValueError(f"seed {seed}: rejected diagnostics/protocol identity 漂移")
    if usage["expert_actions"] == 0 and expert_takeover_step is not None:
        raise ValueError(f"seed {seed}: rejection takeover/usage 不一致")

    reason = str(failure["reason"])
    final_transition = diagnostics.get("final_transition")
    is_final_transition = isinstance(final_transition, dict)
    is_time_limit_reason = reason == EPISODE_TIME_LIMIT_REASON
    is_hard_limit_usage = (
        usage["total_actions"]
        == identity["action_budget_protocol"]["environment_action_limit"]
    )
    has_truncated_transition = (
        is_final_transition and final_transition.get("truncated") is True
    )
    hard_deadline_signals = (
        is_time_limit_reason,
        is_hard_limit_usage,
        has_truncated_transition,
    )
    if any(hard_deadline_signals) and not all(hard_deadline_signals):
        raise ValueError(
            f"seed {seed}: hard-deadline reason/usage/truncation 三信号不闭合"
        )
    if all(hard_deadline_signals):
        if final_transition.get("action_step") != usage["total_actions"]:
            raise ValueError(f"seed {seed}: hard-deadline transition step 漂移")
        return

    budget_phase = diagnostics.get("budget_exhaustion_phase")
    is_expert_reason = reason == EXPERT_ACTION_BUDGET_EXHAUSTED_REASON
    is_expert_cap_usage = (
        usage["expert_actions"]
        == identity["action_budget_protocol"]["expert_action_limit"]
    )
    is_expert_phase = budget_phase == "expert"
    if is_expert_reason and not (is_expert_cap_usage and is_expert_phase):
        raise ValueError(f"seed {seed}: Expert-cap failure 缺少 cap usage/phase")
    if is_expert_phase and not is_expert_reason:
        raise ValueError(f"seed {seed}: Expert-cap phase 缺少对应 failure reason")
    if is_expert_reason and (
        not is_final_transition
        or final_transition.get("truncated") is not False
        or final_transition.get("action_step") != usage["total_actions"]
    ):
        raise ValueError(f"seed {seed}: Expert-cap transition 必须非 truncated 且 step 闭合")
    final_progress = diagnostics.get("final_progress")
    if (
        is_expert_cap_usage
        and not is_expert_reason
        and (
            not isinstance(final_progress, dict)
            or final_progress.get("task_completed") is not True
        )
    ):
        raise ValueError(
            f"seed {seed}: Expert=180 但非 cap failure 时必须有已完成行为证据"
        )

    is_policy_reason = reason == POLICY_ACTION_BUDGET_EXHAUSTED_REASON
    is_policy_cap_usage = (
        usage["policy_actions"]
        == identity["action_budget_protocol"]["policy_action_limit"]
        and usage["expert_actions"] == 0
    )
    is_policy_phase = budget_phase == "policy"
    if is_policy_reason != is_policy_phase:
        raise ValueError(f"seed {seed}: Policy-cap reason/phase 双向不闭合")
    if is_policy_reason:
        if not is_policy_cap_usage:
            raise ValueError(f"seed {seed}: Policy-cap failure 缺少 cap usage")
        if (
            diagnostics.get("boundary_reached") is not False
            or not is_final_transition
            or final_transition.get("truncated") is not False
            or final_transition.get("action_step") != usage["total_actions"]
        ):
            raise ValueError(
                f"seed {seed}: Policy-cap transition/boundary 必须非 truncated 且闭合"
            )
    elif is_policy_cap_usage:
        boundary_ended_at_cap = (
            reason == _BOUNDARY_ENDED_REASON
            and diagnostics.get("boundary_reached") is True
            and is_final_transition
            and final_transition.get("action_step") == usage["total_actions"]
            and (
                final_transition.get("terminated") is True
                or final_transition.get("truncated") is True
            )
        )
        if not boundary_ended_at_cap:
            raise ValueError(
                f"seed {seed}: Policy=300 的非 cap rejection 缺少 boundary 优先证据"
            )


def _candidate_layout(record_path: Path, *, seed: int) -> tuple[Path, Path]:
    """验证 formal candidate 的 lexical 层级，禁止经 symlink 逃逸。"""

    absolute_record = _lexical_absolute(record_path)
    candidate_dir = absolute_record.parent
    candidates_root = candidate_dir.parent
    output_root = candidates_root.parent
    expected_name = f"seed-{seed:06d}"
    if (
        absolute_record.name != "record.json"
        or candidate_dir.name != expected_name
        or candidates_root.name != "candidates"
    ):
        raise ValueError(
            f"seed {seed}: record 必须位于 output/candidates/{expected_name}/record.json"
        )
    context = f"seed {seed} candidate evidence"
    for directory, label in (
        (output_root, "formal output root"),
        (candidates_root, "candidates root"),
        (candidate_dir, "candidate directory"),
    ):
        _require_plain_path(
            directory,
            kind="directory",
            label=label,
            context=context,
        )
        if directory.resolve(strict=True) != directory:
            raise RuntimeError(f"{context} {label} 路径禁止包含 symlink")
    return candidate_dir, candidate_dir / "dataset"


def _read_candidate_record(
    path: Path,
    *,
    seed: int,
) -> tuple[dict[str, Any], _StableFileRead]:
    _candidate_layout(path, seed=seed)
    stable = _read_plain_file_stable(
        _lexical_absolute(path),
        label="record.json",
        capture_bytes=True,
        context=f"seed {seed} candidate evidence",
    )
    if stable.payload is None:  # pragma: no cover - capture_bytes=True contract
        raise RuntimeError(f"seed {seed}: record stable reader 未返回 bytes")
    try:
        text = stable.payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"seed {seed}: candidate record 不是有效 UTF-8") from exc
    value = _strict_json_loads(text, label=f"seed {seed} candidate record")
    if not isinstance(value, dict):
        raise TypeError(f"seed {seed}: candidate record 必须是 JSON object")
    return value, stable


def _load_record(path: Path, identity: dict[str, Any], seed: int) -> dict[str, Any]:
    record, stable_record = _read_candidate_record(path, seed=seed)
    if record.get("format") != COLLECTION_FORMAT:
        raise ValueError(f"seed {seed}: candidate record format 不兼容")
    expected_config = _expected_candidate_config(identity, seed=seed)
    if record.get("config") != expected_config:
        raise ValueError(f"seed {seed}: candidate config 不是 exact frozen config")
    if record.get("source_revision") != identity["source_revision"]:
        raise ValueError(f"seed {seed}: source revision 漂移")
    if record.get("base_dataset") != identity["base_dataset"]["path"]:
        raise ValueError(f"seed {seed}: base dataset path 漂移")
    expected_stats = identity["base_dataset"]["compatibility"]["proprio_stats"]
    if record.get("base_dataset_receipt") != {
        "proprio_stats_sha256": expected_stats["sha256"],
        "proprio_stats_semantic_sha256": expected_stats["semantic_sha256"],
    }:
        raise ValueError(f"seed {seed}: base dataset stats receipt 漂移")
    checkpoint = record.get("checkpoint")
    if not isinstance(checkpoint, dict) or set(checkpoint) != {
        "path",
        "sha256",
        "metadata",
    }:
        raise ValueError(f"seed {seed}: checkpoint record schema 漂移")
    if (
        checkpoint["path"] != identity["checkpoint"]["path"]
        or checkpoint["sha256"] != identity["checkpoint"]["sha256"]
        or not isinstance(checkpoint["metadata"], dict)
    ):
        raise ValueError(f"seed {seed}: checkpoint identity 漂移")
    checkpoint_stats = checkpoint["metadata"].get("proprio_stats")
    if (
        not isinstance(checkpoint_stats, dict)
        or _compact_json_sha256(checkpoint_stats)
        != expected_stats["semantic_sha256"]
    ):
        raise ValueError(f"seed {seed}: checkpoint/proprio stats identity 漂移")

    status = record.get("status")
    if status == "error":
        raise RuntimeError(f"seed {seed}: 既有 candidate 是 status=error，拒绝 resume")
    if status not in {"accepted", "rejected"}:
        raise ValueError(f"seed {seed}: candidate status 不兼容: {status!r}")
    usage = _validate_action_budget_usage(record, identity=identity, seed=seed)
    if status == "accepted":
        _validate_accepted_record(
            record,
            identity=identity,
            seed=seed,
            usage=usage,
        )
        _accepted_artifact_receipt(record, record_path=path, seed=seed)
    else:
        dataset_root = path.parent / "dataset"
        if _path_entry_exists(dataset_root):
            raise RuntimeError(
                f"seed {seed}: rejected/error candidate 残留 canonical dataset"
            )
        _validate_rejected_record(
            record,
            identity=identity,
            seed=seed,
            usage=usage,
        )
    _assert_plain_file_identity(
        _lexical_absolute(path),
        expected=stable_record.stat_identity,
        label="record.json",
        context=f"seed {seed} candidate evidence",
    )
    return record


def compact_candidate_record(
    record: dict[str, Any],
    record_path: Path,
    *,
    experiment_receipt: Mapping[str, str],
) -> dict[str, Any]:
    config = record["config"]
    seed = int(config["environment_seed"])
    observed_record, stable_record = _read_candidate_record(
        record_path,
        seed=seed,
    )
    if observed_record != record:
        raise ValueError(f"seed {seed}: compact 前 record 内容发生变化")
    if set(experiment_receipt) != {"experiment", "experiment_sha256"}:
        raise ValueError("candidate compact row 缺少 experiment receipt")
    row: dict[str, Any] = {
        "pool": dict(experiment_receipt),
        "environment_seed": seed,
        "boundary_type": str(config["boundary_type"]),
        "status": str(record["status"]),
        "record": str(_lexical_absolute(record_path)),
        "record_sha256": stable_record.sha256,
        "episode_sampling_seed": int(config["episode_sampling_seed"]),
        LOCAL_DAGGER_ACTION_BUDGET_USAGE_FIELD: record[
            LOCAL_DAGGER_ACTION_BUDGET_USAGE_FIELD
        ],
    }
    if record["status"] == "accepted":
        result = record["result"]
        artifact = _accepted_artifact_receipt(
            record,
            record_path=record_path,
            seed=seed,
        )
        row.update(
            {
                "trajectory_id": result["trajectory"]["trajectory_id"],
                "dataset_root": artifact["dataset_root"],
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
                "artifact": artifact,
            }
        )
    else:
        row["failure"] = record.get("failure")
        row["eligible_for_risk_selection"] = False
    _assert_plain_file_identity(
        _lexical_absolute(record_path),
        expected=stable_record.stat_identity,
        label="record.json",
        context=f"seed {seed} candidate evidence",
    )
    return row


def _candidate_command(
    identity: Mapping[str, Any],
    *,
    seed: int,
    candidate_dir: Path,
) -> list[str]:
    config = identity["config"]
    return [
        sys.executable,
        "-m",
        "robot_vla.cli.collect_local_dagger",
        "--data",
        str(identity["base_dataset"]["path"]),
        "--model-cache",
        str(identity["qwen"]["cache_path"]),
        "--checkpoint",
        str(identity["checkpoint"]["path"]),
        "--output",
        str((candidate_dir / "dataset").resolve()),
        "--record",
        str((candidate_dir / "record.json").resolve()),
        "--seed",
        str(seed),
        "--boundary-type",
        str(identity["boundary_type"]),
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
        "--action-budget-protocol",
        AMENDED_ACTION_BUDGET_PROTOCOL.value,
        "--require-paired-clean-expert",
    ]


def _progress_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected: int,
    experiment_receipt: Mapping[str, str],
    candidate_manifest_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: int(row["environment_seed"]))
    rejection_reasons = Counter(
        row.get("failure", {}).get("reason", "unknown")
        for row in ordered
        if row["status"] == "rejected"
    )
    statuses = Counter(str(row["status"]) for row in ordered)
    return {
        "format": POOL_FORMAT,
        **dict(experiment_receipt),
        **dict(candidate_manifest_receipt),
        "publication_state": "progress",
        "scan_complete": len(ordered) == expected,
        "expected_candidates": expected,
        "completed_candidates": len(ordered),
        "status_counts": dict(sorted(statuses.items())),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "eligible_for_risk_selection": sum(
            bool(row["eligible_for_risk_selection"]) for row in ordered
        ),
    }


def _write_progress(
    output: Path,
    rows: list[dict[str, Any]],
    expected: int,
    *,
    experiment_receipt: Mapping[str, str],
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: row["environment_seed"])
    candidate_path = output / "collection_candidates.jsonl"
    _atomic_write_jsonl(candidate_path, ordered)
    candidate_receipt = _candidate_manifest_receipt(candidate_path, rows=ordered)
    _atomic_write_json(
        output / "collection_summary.json",
        _progress_summary(
            ordered,
            expected=expected,
            experiment_receipt=experiment_receipt,
            candidate_manifest_receipt=candidate_receipt,
        ),
    )
    return candidate_receipt


def _synchronize_resume_progress(
    output: Path,
    rows: list[dict[str, Any]],
    expected: int,
    *,
    experiment_receipt: Mapping[str, str],
) -> None:
    """验证完成后，将 record→C→S 的合法一步落后快照推进到当前前缀。"""

    candidate_path = output / "collection_candidates.jsonl"
    observed_rows = (
        _read_jsonl(candidate_path, label="resume collection_candidates.jsonl")
        if candidate_path.is_file()
        else []
    )
    ordered_rows = sorted(rows, key=lambda row: int(row["environment_seed"]))
    summary_path = output / "collection_summary.json"
    observed_summary_count = None
    if summary_path.is_file():
        observed_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if isinstance(observed_summary, dict):
            observed_summary_count = observed_summary.get("completed_candidates")
    if observed_rows != ordered_rows or observed_summary_count != len(ordered_rows):
        _write_progress(
            output,
            rows,
            expected,
            experiment_receipt=experiment_receipt,
        )


def _parse_jsonl_text(payload: str, *, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        payload.splitlines(),
        start=1,
    ):
        if not line.strip():
            raise ValueError(f"{label}:{line_number}: 不允许空行")
        value = _strict_json_loads(line, label=f"{label}:{line_number}")
        if not isinstance(value, dict):
            raise TypeError(f"{label}:{line_number}: 必须是 JSON object")
        rows.append(value)
    return rows


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    stable = _read_plain_file_stable(
        path,
        label=label,
        capture_bytes=True,
        context="formal pool JSONL",
    )
    if stable.payload is None:  # pragma: no cover - capture_bytes=True contract
        raise RuntimeError(f"{label}: stable reader 未返回 bytes")
    try:
        payload = stable.payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label}: 不是有效 UTF-8") from exc
    return _parse_jsonl_text(payload, label=label)


def _resolve_candidate_trajectory_file(
    dataset_root: Path,
    relative_value: str,
    *,
    seed: int,
) -> Path:
    relative = Path(relative_value)
    if (
        relative.is_absolute()
        or relative == Path(".")
        or ".." in relative.parts
        or "\\" in relative_value
        or relative.as_posix() != relative_value
        or relative.suffix != ".npz"
    ):
        raise ValueError(f"seed {seed}: trajectory file 路径非法: {relative_value!r}")
    cursor = dataset_root
    for index, component in enumerate(relative.parts):
        cursor /= component
        _require_plain_path(
            cursor,
            kind="file" if index == len(relative.parts) - 1 else "directory",
            label="trajectory 路径",
            context=f"seed {seed} candidate evidence",
        )
    resolved = cursor.resolve(strict=True)
    if resolved != cursor or not resolved.is_relative_to(dataset_root):
        raise ValueError(f"seed {seed}: trajectory file 逃逸 dataset root")
    return cursor


def _accepted_artifact_receipt(
    record: Mapping[str, Any],
    *,
    record_path: Path,
    seed: int,
) -> dict[str, Any]:
    """从 accepted record 指定的单条 manifest 发行 receipt，不扫描 NPZ 目录。"""

    result = record.get("result")
    trajectory = result.get("trajectory") if isinstance(result, dict) else None
    if not isinstance(trajectory, dict):
        raise TypeError(f"seed {seed}: accepted record 缺少 trajectory")
    candidate_dir, dataset_root = _candidate_layout(record_path, seed=seed)
    dataset_stat = _require_plain_path(
        dataset_root,
        kind="directory",
        label="accepted dataset root",
        context=f"seed {seed} candidate evidence",
    )
    dataset_identity = _stable_stat_identity(dataset_stat)
    if (
        dataset_root.resolve(strict=True) != dataset_root
        or dataset_root.parent != candidate_dir
    ):
        raise RuntimeError(f"seed {seed}: accepted dataset 逃逸 candidate directory")
    staging_marker = dataset_root / CANDIDATE_DATASET_STAGING_MARKER
    if _path_entry_exists(staging_marker):
        raise RuntimeError(
            f"seed {seed}: accepted dataset 仍带 uncommitted staging marker"
        )
    manifest_path = dataset_root / "manifest.jsonl"
    manifest_read = _read_plain_file_stable(
        manifest_path,
        label="accepted canonical manifest",
        capture_bytes=True,
        context=f"seed {seed} candidate evidence",
    )
    if manifest_read.payload is None:  # pragma: no cover
        raise RuntimeError(f"seed {seed}: manifest stable reader 未返回 bytes")
    try:
        manifest_text = manifest_read.payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"seed {seed}: accepted manifest 不是有效 UTF-8") from exc
    manifest_rows = _parse_jsonl_text(
        manifest_text,
        label=f"seed {seed} accepted manifest",
    )
    if manifest_rows != [trajectory]:
        raise ValueError(f"seed {seed}: manifest 不是 record 指定的唯一 trajectory")

    file_value = trajectory.get("file")
    if not isinstance(file_value, str) or not file_value:
        raise TypeError(f"seed {seed}: trajectory file 非法")
    trajectory_path = _resolve_candidate_trajectory_file(
        dataset_root,
        file_value,
        seed=seed,
    )
    trajectory_read = _read_plain_file_stable(
        trajectory_path,
        label="record 指定的 trajectory NPZ",
        capture_bytes=False,
        context=f"seed {seed} candidate evidence",
    )
    _audit_accepted_trajectory(dataset_root, trajectory)
    _assert_plain_file_identity(
        manifest_path,
        expected=manifest_read.stat_identity,
        label="accepted canonical manifest",
        context=f"seed {seed} candidate evidence",
    )
    _assert_plain_file_identity(
        trajectory_path,
        expected=trajectory_read.stat_identity,
        label="record 指定的 trajectory NPZ",
        context=f"seed {seed} candidate evidence",
    )
    _assert_plain_directory_identity(
        dataset_root,
        expected=dataset_identity,
        label="accepted dataset root",
        context=f"seed {seed} candidate evidence",
    )
    return {
        "dataset_root": str(dataset_root),
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_read.sha256,
        "trajectory_file": str(trajectory_path),
        "trajectory_sha256": trajectory_read.sha256,
        "trajectory_audit": "passed",
    }


def _audit_accepted_trajectory(
    dataset_root: Path,
    trajectory: Mapping[str, Any],
) -> None:
    """重新加载并审计 selected candidate，不信任 record 中的 passed 字符串。"""

    from robot_vla.contracts import RobotSpec
    from robot_vla.data.audit import audit_trajectory
    from robot_vla.data.trajectory import TrajectoryMeta, TrajectoryStore

    meta = TrajectoryMeta.from_dict(dict(trajectory))
    spec = RobotSpec()
    arrays = TrajectoryStore(dataset_root, spec, cache_size=0).get(meta)
    audit_trajectory(arrays, meta, spec)


def _risk_selection_payload(
    selection: Any,
    *,
    experiment_receipt: Mapping[str, str],
    candidate_manifest_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    payload = selection.to_dict()
    if not isinstance(payload, dict):
        raise TypeError("risk selection payload 必须是 JSON object")
    if "upstream" in payload:
        raise ValueError("risk selection mathematical payload 不得预占 upstream")
    return {
        **payload,
        "upstream": {
            **dict(experiment_receipt),
            **dict(candidate_manifest_receipt),
        },
    }


def build_canonical_selected_records(
    identity: Mapping[str, Any],
    *,
    selection: Any,
    candidate_records: Mapping[int, tuple[Mapping[str, Any], Path]],
    experiment_path: Path,
    candidate_manifest_path: Path,
    candidate_manifest_sha256: str,
    risk_selection_path: Path,
    risk_selection_sha256: str,
) -> list[dict[str, Any]]:
    """只将 risk selector 明确选中的 accepted records 发行为 D1 候选输入。"""

    resolved_experiment = experiment_path.resolve()
    resolved_candidates = candidate_manifest_path.resolve()
    resolved_risk_selection = risk_selection_path.resolve()
    experiment_receipt = _experiment_receipt(resolved_experiment)
    if experiment_receipt["experiment_sha256"] != _json_sha256(identity):
        raise ValueError("canonical selection 的 experiment bytes 漂移")
    if (
        resolved_risk_selection.parent != resolved_experiment.parent
        or resolved_risk_selection.name != RISK_SELECTION_FILENAME
        or not resolved_risk_selection.is_file()
    ):
        raise ValueError("canonical selection 必须绑定同一 pool 的 risk_selection.json")
    if (
        resolved_candidates.parent != resolved_experiment.parent
        or resolved_candidates.name != "collection_candidates.jsonl"
        or not resolved_candidates.is_file()
    ):
        raise ValueError("canonical selection 必须绑定同一 pool 的 candidates")
    candidate_rows = _read_jsonl(
        resolved_candidates,
        label="canonical collection_candidates.jsonl",
    )
    candidate_receipt = _candidate_manifest_receipt(
        resolved_candidates,
        rows=candidate_rows,
    )
    if candidate_receipt["collection_candidates_sha256"] != candidate_manifest_sha256:
        raise ValueError("canonical selection 的 collection candidates SHA256 漂移")
    if [int(row.get("environment_seed", -1)) for row in candidate_rows] != [
        int(seed) for seed in identity["environment_seeds"]
    ]:
        raise ValueError("canonical selection 必须绑定完整冻结 candidate pool")
    if any(row.get("pool") != experiment_receipt for row in candidate_rows):
        raise ValueError("canonical selection 的 candidate experiment receipt 漂移")
    if _sha256_file(resolved_risk_selection) != risk_selection_sha256:
        raise ValueError("canonical selection 的 risk selection SHA256 漂移")
    observed_risk_selection = json.loads(
        resolved_risk_selection.read_text(encoding="utf-8")
    )
    if (
        not isinstance(observed_risk_selection, dict)
        or _json_sha256(observed_risk_selection) != risk_selection_sha256
    ):
        raise ValueError("canonical selection 的 risk selection 非 canonical JSON")
    if observed_risk_selection != _risk_selection_payload(
        selection,
        experiment_receipt=experiment_receipt,
        candidate_manifest_receipt=candidate_receipt,
    ):
        raise ValueError("canonical selection 与 risk_selection.json 内容不一致")

    high_seeds = tuple(int(seed) for seed in selection.high_risk_seeds)
    low_seeds = tuple(int(seed) for seed in selection.low_risk_seeds)
    if len(high_seeds) != HIGH_RISK_SELECTION_COUNT:
        raise RuntimeError("canonical selection 的 high-risk 数量漂移")
    if len(low_seeds) != LOW_RISK_SELECTION_COUNT:
        raise RuntimeError("canonical selection 的 low-risk 数量漂移")
    ordered_seeds = high_seeds + low_seeds
    if len(set(ordered_seeds)) != ELIGIBLE_SELECTION_GATE:
        raise RuntimeError("canonical selection 必须是 20 个互异 seed")
    scored_by_seed = {
        int(row["environment_seed"]): row for row in selection.scored_candidates
    }
    rows: list[dict[str, Any]] = []
    for selection_index, seed in enumerate(ordered_seeds):
        if seed not in candidate_records or seed not in scored_by_seed:
            raise RuntimeError(f"seed {seed}: selected candidate 缺少 canonical record")
        record, record_path = candidate_records[seed]
        if (
            record.get("status") != "accepted"
            or record.get("eligible_for_risk_selection") is not True
        ):
            raise RuntimeError(f"seed {seed}: 禁止发行非 accepted+eligible record")
        score_row = scored_by_seed[seed]
        stratum = "high" if seed in high_seeds else "low"
        if score_row.get("selection_stratum") != stratum:
            raise RuntimeError(f"seed {seed}: risk selection stratum 漂移")
        observed_record, stable_record = _read_candidate_record(
            record_path,
            seed=seed,
        )
        if observed_record != record:
            raise ValueError(f"seed {seed}: canonical 发行前 record 内容发生变化")
        receipt = _accepted_artifact_receipt(
            record,
            record_path=record_path,
            seed=seed,
        )
        trajectory = record["result"]["trajectory"]
        rows.append(
            {
                "format": CANONICAL_SELECTED_RECORD_FORMAT,
                "pool": {
                    "format": identity["format"],
                    **experiment_receipt,
                    "source_revision": identity["source_revision"],
                },
                "candidate_manifest": candidate_receipt,
                "environment_seed": seed,
                "boundary_type": identity["boundary_type"],
                "candidate": {
                    "status": "accepted",
                    "eligible_for_risk_selection": True,
                    "record": str(_lexical_absolute(record_path)),
                    "record_sha256": stable_record.sha256,
                },
                "selection": {
                    "selected": True,
                    "stratum": stratum,
                    "index": selection_index,
                    "risk_score": score_row["risk_score"],
                    "risk_selection": str(resolved_risk_selection),
                    "risk_selection_sha256": risk_selection_sha256,
                },
                "trajectory": trajectory,
                "artifact": receipt,
                "d1_input_contract": {
                    "source": "canonical_selected_records.jsonl",
                    "directory_scan": "forbidden",
                    "npz_directory_scan": "forbidden",
                },
            }
        )
        _assert_plain_file_identity(
            _lexical_absolute(record_path),
            expected=stable_record.stat_identity,
            label="record.json",
            context=f"seed {seed} candidate evidence",
        )
    return rows


def _selection_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected: int,
    selection: Any | None,
    experiment_receipt: Mapping[str, str],
    candidate_manifest_receipt: Mapping[str, Any],
    risk_selection_sha256: str | None = None,
    canonical_selected_records_sha256: str | None = None,
) -> dict[str, Any]:
    summary = _progress_summary(
        rows,
        expected=expected,
        experiment_receipt=experiment_receipt,
        candidate_manifest_receipt=candidate_manifest_receipt,
    )
    if selection is None:
        summary.update(
            {
                "publication_state": "gate_failed",
                "selection_gate_passed": False,
                "selection_failure": (
                    f"eligible candidates 少于 {ELIGIBLE_SELECTION_GATE}"
                ),
            }
        )
    else:
        if (
            risk_selection_sha256 is None
            or canonical_selected_records_sha256 is None
        ):
            raise ValueError("final selection summary 缺少 risk/canonical SHA256 receipt")
        summary.update(
            {
                "publication_state": "complete",
                "selection_gate_passed": True,
                "high_risk_seeds": list(selection.high_risk_seeds),
                "low_risk_seeds": list(selection.low_risk_seeds),
                "selected_count": ELIGIBLE_SELECTION_GATE,
                "risk_selection": RISK_SELECTION_FILENAME,
                "risk_selection_sha256": risk_selection_sha256,
                "canonical_selected_records": (
                    CANONICAL_SELECTED_RECORDS_FILENAME
                ),
                "canonical_selected_records_sha256": (
                    canonical_selected_records_sha256
                ),
            }
        )
    return summary


def _load_existing_candidates(
    output: Path,
    *,
    identity: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    dict[int, tuple[dict[str, Any], Path]],
]:
    seeds = [int(seed) for seed in identity["environment_seeds"]]
    experiment_receipt = _experiment_receipt(output / "experiment.json")
    absolute_output = _lexical_absolute(output)
    _require_plain_path(
        absolute_output,
        kind="directory",
        label="formal output root",
        context="--resume candidate scan",
    )
    candidates_root = absolute_output / "candidates"
    if not _path_entry_exists(candidates_root):
        return [], {}
    _require_plain_path(
        candidates_root,
        kind="directory",
        label="candidates root",
        context="--resume candidate scan",
    )
    if candidates_root.resolve(strict=True) != candidates_root:
        raise RuntimeError("--resume candidates root 路径禁止包含 symlink")
    expected_names = {f"seed-{seed:06d}" for seed in seeds}
    unexpected = sorted(
        path.name
        for path in candidates_root.iterdir()
        if path.name not in expected_names
    )
    if unexpected:
        raise RuntimeError(f"--resume 存在未冻结 candidate 目录: {unexpected}")

    rows: list[dict[str, Any]] = []
    records: dict[int, tuple[dict[str, Any], Path]] = {}
    missing_seen = False
    for seed in seeds:
        candidate_dir = candidates_root / f"seed-{seed:06d}"
        record_path = candidate_dir / "record.json"
        if not _path_entry_exists(record_path):
            if _path_entry_exists(candidate_dir):
                _require_plain_path(
                    candidate_dir,
                    kind="directory",
                    label=f"seed {seed} candidate directory",
                    context="--resume candidate scan",
                )
            if _path_entry_exists(candidate_dir) and any(candidate_dir.iterdir()):
                raise RuntimeError(f"seed {seed}: 存在无 record 的 partial candidate")
            missing_seen = True
            continue
        if missing_seen:
            raise RuntimeError("--resume candidate records 不是冻结 seed 的连续前缀")
        record = _load_record(record_path, identity, seed)
        rows.append(
            compact_candidate_record(
                record,
                record_path,
                experiment_receipt=experiment_receipt,
            )
        )
        records[seed] = (record, record_path)
    return rows, records


def _validate_resume_derived_artifacts(
    output: Path,
    *,
    identity: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    candidate_records: Mapping[int, tuple[Mapping[str, Any], Path]],
    experiment_path: Path,
) -> None:
    """已有 derived files 只允许是最后一次原子写前/后的可证明快照。"""

    allowed_names = {
        "experiment.json",
        "candidates",
        "collection_candidates.jsonl",
        "collection_summary.json",
        "risk_selection.json",
        "canonical_selected_records.jsonl",
    }
    unexpected = sorted(path.name for path in output.iterdir() if path.name not in allowed_names)
    if unexpected:
        raise RuntimeError(f"--resume 输出根目录存在未冻结产物: {unexpected}")

    ordered_rows = sorted(rows, key=lambda row: int(row["environment_seed"]))
    completed = len(ordered_rows)
    expected = len(identity["environment_seeds"])
    experiment_receipt = _experiment_receipt(experiment_path)
    if experiment_receipt["experiment_sha256"] != _json_sha256(identity):
        raise ValueError("--resume experiment.json 不是 canonical frozen bytes")
    candidate_summary_path = output / "collection_candidates.jsonl"
    summary_path = output / "collection_summary.json"
    observed_rows: list[dict[str, Any]] = []
    if candidate_summary_path.is_file():
        observed_rows = _read_jsonl(
            candidate_summary_path,
            label="resume collection_candidates.jsonl",
        )
        if len(observed_rows) not in {completed, completed - 1}:
            raise ValueError("--resume collection_candidates progress 长度漂移")
        if observed_rows != ordered_rows[: len(observed_rows)]:
            raise ValueError("--resume collection_candidates 内容漂移")
        if _sha256_file(candidate_summary_path) != _jsonl_sha256(observed_rows):
            raise ValueError("--resume collection_candidates 非 canonical JSONL 字节")
    elif completed > 1:
        raise FileNotFoundError("--resume 缺少 collection_candidates.jsonl")
    candidate_count = len(observed_rows)

    def candidate_receipt_for_count(count: int) -> dict[str, Any]:
        return _candidate_snapshot_receipt(
            candidate_summary_path,
            rows=ordered_rows[:count],
        )

    eligible = [row for row in ordered_rows if row["eligible_for_risk_selection"]]
    selection = None
    final_summary = None
    expected_risk: dict[str, Any] | None = None
    expected_canonical: list[dict[str, Any]] | None = None
    risk_path = output / RISK_SELECTION_FILENAME
    canonical_path = output / CANONICAL_SELECTED_RECORDS_FILENAME
    if completed == expected:
        if len(eligible) >= ELIGIBLE_SELECTION_GATE:
            selection = score_and_select_risk_candidates(
                identity["boundary_type"],
                eligible,
                high_count=HIGH_RISK_SELECTION_COUNT,
                low_count=LOW_RISK_SELECTION_COUNT,
            )
            if candidate_count == expected:
                expected_risk = _risk_selection_payload(
                    selection,
                    experiment_receipt=experiment_receipt,
                    candidate_manifest_receipt=candidate_receipt_for_count(expected),
                )
        elif candidate_count == expected:
            final_summary = _selection_summary(
                ordered_rows,
                expected=expected,
                selection=None,
                experiment_receipt=experiment_receipt,
                candidate_manifest_receipt=candidate_receipt_for_count(expected),
            )

    if selection is None:
        if risk_path.exists() or canonical_path.exists():
            raise RuntimeError("--resume 未通过 gate 却存在 selection/canonical 产物")
    else:
        if risk_path.exists() and not risk_path.is_file():
            raise RuntimeError("--resume risk_selection.json 不是普通文件")
        if canonical_path.exists() and not canonical_path.is_file():
            raise RuntimeError("--resume canonical selected records 不是普通文件")
        if canonical_path.is_file() and not risk_path.is_file():
            raise RuntimeError("--resume canonical selected records 缺少前置 risk selection")
        if (risk_path.exists() or canonical_path.exists()) and expected_risk is None:
            raise RuntimeError(
                "--resume selection/canonical 缺少完整 collection_candidates 上游"
            )

        risk_sha256: str | None = None
        canonical_sha256: str | None = None
        if risk_path.is_file():
            observed_risk = json.loads(risk_path.read_text(encoding="utf-8"))
            if observed_risk != expected_risk:
                raise ValueError("--resume risk_selection.json 漂移")
            risk_sha256 = _sha256_file(risk_path)
            if (
                not isinstance(observed_risk, dict)
                or _json_sha256(observed_risk) != risk_sha256
            ):
                raise ValueError("--resume risk_selection.json 非 canonical JSON")
            expected_canonical = build_canonical_selected_records(
                identity,
                selection=selection,
                candidate_records=candidate_records,
                experiment_path=experiment_path,
                candidate_manifest_path=candidate_summary_path,
                candidate_manifest_sha256=candidate_receipt_for_count(expected)[
                    "collection_candidates_sha256"
                ],
                risk_selection_path=risk_path,
                risk_selection_sha256=risk_sha256,
            )
        if canonical_path.is_file():
            observed_canonical = _read_jsonl(
                canonical_path,
                label="resume canonical_selected_records.jsonl",
            )
            if observed_canonical != expected_canonical:
                raise ValueError("--resume canonical selected records 漂移")
            canonical_sha256 = _sha256_file(canonical_path)
            if _jsonl_sha256(observed_canonical) != canonical_sha256:
                raise ValueError(
                    "--resume canonical selected records 非 canonical JSONL"
                )
        if risk_sha256 is not None and canonical_sha256 is not None:
            final_summary = _selection_summary(
                ordered_rows,
                expected=expected,
                selection=selection,
                experiment_receipt=experiment_receipt,
                candidate_manifest_receipt=candidate_receipt_for_count(expected),
                risk_selection_sha256=risk_sha256,
                canonical_selected_records_sha256=canonical_sha256,
            )

    if summary_path.is_file():
        observed_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(observed_summary, dict):
            raise TypeError("--resume collection_summary 必须是 JSON object")
        if _json_sha256(observed_summary) != _sha256_file(summary_path):
            raise ValueError("--resume collection_summary 非 canonical JSON")
        observed_count = observed_summary.get("completed_candidates")
        allowed_summaries: list[dict[str, Any]] = []
        if risk_path.is_file() or candidate_count == completed - 1:
            progress_counts = {candidate_count}
        else:
            progress_counts = {candidate_count, candidate_count - 1}
        for count in progress_counts:
            if 0 <= count <= candidate_count:
                allowed_summaries.append(
                    _progress_summary(
                        ordered_rows[:count],
                        expected=expected,
                        experiment_receipt=experiment_receipt,
                        candidate_manifest_receipt=candidate_receipt_for_count(count),
                    )
                )
        if final_summary is not None:
            allowed_summaries.append(final_summary)
        if observed_summary not in allowed_summaries:
            raise ValueError(
                "--resume collection_summary 不是可证明的 progress/final 快照"
            )
        if not isinstance(observed_count, int) or isinstance(observed_count, bool):
            raise TypeError("--resume collection_summary completed_candidates 非法")
    elif completed > 1:
        raise FileNotFoundError("--resume 缺少 collection_summary.json")


def run(args: argparse.Namespace) -> None:
    with _formal_pool_lock(args.output):
        _run_locked(args)


def _run_locked(args: argparse.Namespace) -> None:
    if args.sampling_seed < 0 or args.num_flow_steps <= 0:
        raise ValueError("sampling/Flow 配置无效")
    if not 0.0 < args.recency_decay < 1.0 or args.max_anomaly_replans < 0:
        raise ValueError("temporal/anomaly 配置无效")
    project_root = Path(__file__).resolve().parents[3]
    base_dataset_verification = verify_frozen_d0_compatibility(args.data)
    checkpoint_path, checkpoint_read = _read_formal_checkpoint(args.checkpoint)
    checkpoint_sha256 = checkpoint_read.sha256
    if checkpoint_sha256 != AMENDED_FORMAL_CHECKPOINT_SHA256:
        raise ValueError("amended formal checkpoint SHA256 与冻结 E011 identity 不一致")
    from robot_vla.cli.train_stage1 import compute_source_revision

    identity = build_pool_identity(
        args,
        source_revision=compute_source_revision(project_root),
        base_dataset_root=base_dataset_verification.resolved_root,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        base_dataset_compatibility=base_dataset_verification.receipt,
        runtime_identity=_build_runtime_identity(),
    )
    base_seeds = {
        int(entry.randomization["seed"])
        for entry in base_dataset_verification.entries
    }
    overlap = base_seeds.intersection(identity["environment_seeds"])
    if overlap:
        raise ValueError(f"正式 collection seeds 与 D0 重叠: {sorted(overlap)}")
    experiment_path = args.output / "experiment.json"
    if args.resume:
        existing, _ = _read_experiment_identity(experiment_path)
        if existing != identity:
            raise ValueError("--resume experiment identity 漂移")
        _remove_stale_root_derived_temps(args.output)
        rows, candidate_records = _load_existing_candidates(
            args.output,
            identity=identity,
        )
        _validate_resume_derived_artifacts(
            args.output,
            identity=identity,
            rows=rows,
            candidate_records=candidate_records,
            experiment_path=experiment_path,
        )
    else:
        if _path_entry_exists(args.output) and any(args.output.iterdir()):
            raise FileExistsError("正式 pool 输出目录非空；拒绝覆盖")
        args.output.mkdir(parents=True, exist_ok=True)
        output_stat = _require_plain_path(
            _lexical_absolute(args.output),
            kind="directory",
            label="formal output root",
            context="amended formal output gate",
        )
        if output_stat.st_uid != os.geteuid() or output_stat.st_mode & 0o022:
            raise PermissionError(
                "formal output 必须由当前用户拥有，且 group/other 不可写"
            )
        _atomic_write_json(experiment_path, identity)
        rows = []
        candidate_records = {}

    experiment_receipt = _experiment_receipt(experiment_path)
    seeds = [int(seed) for seed in identity["environment_seeds"]]
    candidate_summary_path = args.output / "collection_candidates.jsonl"
    if args.resume and rows:
        _synchronize_resume_progress(
            args.output,
            rows,
            len(seeds),
            experiment_receipt=experiment_receipt,
        )

    for seed in seeds[len(rows) :]:
        _verify_candidate_stats_leaf(
            identity,
            expected_root_identity=base_dataset_verification.root_stat_identity,
        )
        _verify_candidate_checkpoint_leaf(
            identity,
            expected_stat_identity=checkpoint_read.stat_identity,
        )
        candidate_dir = args.output / "candidates" / f"seed-{seed:06d}"
        record_path = candidate_dir / "record.json"
        if _path_entry_exists(record_path):
            raise FileExistsError(f"seed {seed}: 非前缀 record 已存在")
        if _path_entry_exists(candidate_dir) and any(candidate_dir.iterdir()):
            raise RuntimeError(f"seed {seed}: 存在无 record 的 partial candidate")
        candidate_dir.mkdir(parents=True, exist_ok=True)
        _candidate_layout(record_path, seed=seed)
        command = _candidate_command(
            identity,
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
        _verify_candidate_stats_leaf(
            identity,
            expected_root_identity=base_dataset_verification.root_stat_identity,
        )
        _verify_candidate_checkpoint_leaf(
            identity,
            expected_stat_identity=checkpoint_read.stat_identity,
        )
        if not _path_entry_exists(record_path):
            raise RuntimeError(
                f"seed {seed}: subprocess exit={completed.returncode} 且未写 record"
            )
        record = _load_record(record_path, identity, seed)
        expected_returncode_is_zero = record["status"] == "accepted"
        if (completed.returncode == 0) is not expected_returncode_is_zero:
            raise RuntimeError(
                f"seed {seed}: {record['status']} record 与 subprocess exit "
                f"{completed.returncode} 冲突"
            )
        row = compact_candidate_record(
            record,
            record_path,
            experiment_receipt=experiment_receipt,
        )
        rows.append(row)
        candidate_records[seed] = (record, record_path)
        _write_progress(
            args.output,
            rows,
            len(seeds),
            experiment_receipt=experiment_receipt,
        )
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

    final_d0_verification = verify_frozen_d0_compatibility(
        base_dataset_verification.resolved_root
    )
    if (
        final_d0_verification.receipt != base_dataset_verification.receipt
        or final_d0_verification.resolved_root
        != base_dataset_verification.resolved_root
        or final_d0_verification.root_stat_identity
        != base_dataset_verification.root_stat_identity
    ):
        raise RuntimeError("正式 selection 前 D0 compatibility receipt 漂移")
    final_checkpoint_path, final_checkpoint_read = _read_formal_checkpoint(
        checkpoint_path
    )
    if (
        final_checkpoint_path != checkpoint_path
        or final_checkpoint_read.sha256 != checkpoint_sha256
        or final_checkpoint_read.stat_identity != checkpoint_read.stat_identity
    ):
        raise RuntimeError("正式 selection 前 checkpoint identity 漂移")

    eligible = [row for row in rows if row["eligible_for_risk_selection"]]
    ordered_rows = sorted(rows, key=lambda row: int(row["environment_seed"]))
    candidate_receipt = _candidate_manifest_receipt(
        candidate_summary_path,
        rows=ordered_rows,
    )
    summary_path = args.output / "collection_summary.json"
    if len(eligible) < ELIGIBLE_SELECTION_GATE:
        _atomic_write_json(
            summary_path,
            _selection_summary(
                rows,
                expected=len(seeds),
                selection=None,
                experiment_receipt=experiment_receipt,
                candidate_manifest_receipt=candidate_receipt,
            ),
        )
        return
    selection = score_and_select_risk_candidates(
        args.boundary_type,
        eligible,
        high_count=HIGH_RISK_SELECTION_COUNT,
        low_count=LOW_RISK_SELECTION_COUNT,
    )
    risk_path = args.output / RISK_SELECTION_FILENAME
    _atomic_write_json(
        risk_path,
        _risk_selection_payload(
            selection,
            experiment_receipt=experiment_receipt,
            candidate_manifest_receipt=candidate_receipt,
        ),
    )
    risk_sha256 = _sha256_file(risk_path)
    canonical_rows = build_canonical_selected_records(
        identity,
        selection=selection,
        candidate_records=candidate_records,
        experiment_path=experiment_path,
        candidate_manifest_path=candidate_summary_path,
        candidate_manifest_sha256=candidate_receipt[
            "collection_candidates_sha256"
        ],
        risk_selection_path=risk_path,
        risk_selection_sha256=risk_sha256,
    )
    canonical_path = args.output / CANONICAL_SELECTED_RECORDS_FILENAME
    _atomic_write_jsonl(canonical_path, canonical_rows)
    canonical_sha256 = _sha256_file(canonical_path)
    _atomic_write_json(
        summary_path,
        _selection_summary(
            rows,
            expected=len(seeds),
            selection=selection,
            experiment_receipt=experiment_receipt,
            candidate_manifest_receipt=candidate_receipt,
            risk_selection_sha256=risk_sha256,
            canonical_selected_records_sha256=canonical_sha256,
        ),
    )


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
