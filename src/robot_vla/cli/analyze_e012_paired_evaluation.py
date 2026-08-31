"""从正式 rollout 目录生成 E012 replay/DAgger 配对评估证据。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from robot_vla.cli.evaluate_atomic_maniskill import (
    ATOMIC_EVALUATION_EXPERIMENT_FORMAT,
)
from robot_vla.cli.evaluate_maniskill import EVALUATION_EXPERIMENT_FORMAT
from robot_vla.evaluation.atomic import (
    ATOMIC_ROLLOUT_FORMAT,
    AtomicSkillEpisodeResult,
    summarize_atomic_rollouts,
)
from robot_vla.evaluation.e012_paired import (
    PAIRED_PROTOCOL_SEEDS,
    analyze_e012_pair,
)
from robot_vla.evaluation.rollout import (
    ROLLOUT_FORMAT,
    RolloutEpisodeResult,
    summarize_rollouts,
)

_ResultT = TypeVar("_ResultT", RolloutEpisodeResult, AtomicSkillEpisodeResult)


@dataclass(frozen=True)
class LoadedEvaluation:
    root: Path
    experiment: dict[str, Any]
    summary: dict[str, Any]
    results: list[RolloutEpisodeResult] | list[AtomicSkillEpisodeResult]
    receipt: dict[str, Any]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replay-full-chain",
        type=Path,
        action="append",
        required=True,
        help="pi_replay full-chain 目录；Stage A+B 时按同一顺序重复传入",
    )
    parser.add_argument(
        "--dagger-full-chain",
        type=Path,
        action="append",
        required=True,
        help="pi_dagger full-chain 目录；数量和顺序必须与 replay 对齐",
    )
    parser.add_argument("--replay-atomic", type=Path)
    parser.add_argument("--dagger-atomic", type=Path)
    parser.add_argument(
        "--protocol",
        choices=("descriptive", *PAIRED_PROTOCOL_SEEDS),
        default="descriptive",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _reject_constant(value: str) -> Any:
    raise ValueError(f"JSON 禁止非有限常量: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON 包含重复键: {key}")
        result[key] = value
    return result


def _strict_json_loads(text: str, *, label: str) -> Any:
    try:
        value = json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} 不是 strict JSON: {error}") from error
    return value


def _regular_file(path: Path, *, label: str) -> Path:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{label} 不存在: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} 必须是非 symlink regular file: {path}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    _regular_file(path, label=label)
    value = _strict_json_loads(path.read_text(encoding="utf-8"), label=label)
    if not isinstance(value, dict):
        raise TypeError(f"{label} 顶层必须是 object")
    return value


def _read_jsonl(
    path: Path,
    *,
    label: str,
    factory: Callable[[dict[str, Any]], _ResultT],
) -> list[_ResultT]:
    _regular_file(path, label=label)
    rows: list[_ResultT] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{label} 第 {line_number} 行为空")
            value = _strict_json_loads(
                line,
                label=f"{label} 第 {line_number} 行",
            )
            if not isinstance(value, dict):
                raise TypeError(f"{label} 第 {line_number} 行顶层必须是 object")
            try:
                rows.append(factory(value))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"{label} 第 {line_number} 行不符合结果契约: {error}"
                ) from error
    if not rows:
        raise ValueError(f"{label} 不能为空")
    return rows


def _validate_summary(
    summary: dict[str, Any],
    computed: dict[str, Any],
    experiment: dict[str, Any],
    *,
    expected_episodes: int,
    label: str,
) -> None:
    if summary.get("complete") is not True:
        raise ValueError(f"{label} summary 尚未 complete")
    if summary.get("completed_episodes") != expected_episodes or summary.get(
        "expected_episodes"
    ) != expected_episodes:
        raise ValueError(f"{label} summary Episode 计数不完整")
    for key, value in computed.items():
        if summary.get(key) != value:
            raise ValueError(f"{label} summary 无法从 episodes 精确重算: {key}")
    if summary.get("checkpoint_sha256") != experiment["checkpoint"]["sha256"]:
        raise ValueError(f"{label} summary checkpoint identity 漂移")
    if summary.get("dataset_sha256") != experiment["dataset"]["dataset_sha256"]:
        raise ValueError(f"{label} summary dataset identity 漂移")


def _load_full_chain(root: Path) -> LoadedEvaluation:
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"full-chain root 不是目录: {root}")
    experiment_path = root / "experiment.json"
    summary_path = root / "summary.json"
    episodes_path = root / "episodes.jsonl"
    experiment = _read_json(experiment_path, label="full-chain experiment.json")
    summary = _read_json(summary_path, label="full-chain summary.json")
    if experiment.get("format") != EVALUATION_EXPERIMENT_FORMAT:
        raise ValueError("full-chain experiment format 不一致")
    if experiment.get("rollout_format") != ROLLOUT_FORMAT:
        raise ValueError("full-chain rollout format 不一致")
    results = _read_jsonl(
        episodes_path,
        label="full-chain episodes.jsonl",
        factory=RolloutEpisodeResult.from_dict,
    )
    expected_rows = experiment.get("episodes")
    if not isinstance(expected_rows, list):
        raise TypeError("full-chain experiment 缺少 episodes")
    expected = {(row["seed_group"], row["seed"]) for row in expected_rows}
    observed = {(row.seed_group, row.seed) for row in results}
    if len(expected) != len(expected_rows) or observed != expected:
        raise ValueError("full-chain episodes 与 experiment identity 不一致")
    _validate_summary(
        summary,
        summarize_rollouts(results),
        experiment,
        expected_episodes=len(expected_rows),
        label="full-chain",
    )
    return LoadedEvaluation(
        root=root,
        experiment=experiment,
        summary=summary,
        results=results,
        receipt={
            "directory_name": root.name,
            "experiment_sha256": _sha256(experiment_path),
            "summary_sha256": _sha256(summary_path),
            "episodes_sha256": _sha256(episodes_path),
            "checkpoint_sha256": experiment["checkpoint"]["sha256"],
            "dataset_sha256": experiment["dataset"]["dataset_sha256"],
            "episodes": len(results),
        },
    )


def _load_atomic(root: Path) -> LoadedEvaluation:
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"atomic root 不是目录: {root}")
    experiment_path = root / "experiment.json"
    summary_path = root / "summary.json"
    episodes_path = root / "episodes.jsonl"
    experiment = _read_json(experiment_path, label="atomic experiment.json")
    summary = _read_json(summary_path, label="atomic summary.json")
    if experiment.get("format") != ATOMIC_EVALUATION_EXPERIMENT_FORMAT:
        raise ValueError("atomic experiment format 不一致")
    if experiment.get("rollout_format") != ATOMIC_ROLLOUT_FORMAT:
        raise ValueError("atomic rollout format 不一致")
    results = _read_jsonl(
        episodes_path,
        label="atomic episodes.jsonl",
        factory=AtomicSkillEpisodeResult.from_dict,
    )
    expected_rows = experiment.get("episodes")
    if not isinstance(expected_rows, list):
        raise TypeError("atomic experiment 缺少 episodes")
    expected = {(row["skill_name"], row["seed"]) for row in expected_rows}
    observed = {(row.skill_name, row.seed) for row in results}
    if len(expected) != len(expected_rows) or observed != expected:
        raise ValueError("atomic episodes 与 experiment identity 不一致")
    _validate_summary(
        summary,
        summarize_atomic_rollouts(results),
        experiment,
        expected_episodes=len(expected_rows),
        label="atomic",
    )
    return LoadedEvaluation(
        root=root,
        experiment=experiment,
        summary=summary,
        results=results,
        receipt={
            "directory_name": root.name,
            "experiment_sha256": _sha256(experiment_path),
            "summary_sha256": _sha256(summary_path),
            "episodes_sha256": _sha256(episodes_path),
            "checkpoint_sha256": experiment["checkpoint"]["sha256"],
            "dataset_sha256": experiment["dataset"]["dataset_sha256"],
            "episodes": len(results),
        },
    )


def _without_checkpoint(experiment: dict[str, Any]) -> dict[str, Any]:
    result = dict(experiment)
    result.pop("checkpoint", None)
    return result


def _run_identity(experiment: dict[str, Any]) -> dict[str, Any]:
    return {
        "format": experiment.get("format"),
        "rollout_format": experiment.get("rollout_format"),
        "dataset": experiment.get("dataset"),
        "evaluation_code_revision": experiment.get("evaluation_code_revision"),
        "config": experiment.get("config"),
    }


def _validate_paired_directories(
    replay: LoadedEvaluation,
    dagger: LoadedEvaluation,
    *,
    label: str,
) -> None:
    if _without_checkpoint(replay.experiment) != _without_checkpoint(dagger.experiment):
        raise ValueError(f"{label} replay/DAgger experiment 除 checkpoint 外不一致")


def _merge_full_chain(
    replay_paths: list[Path], dagger_paths: list[Path]
) -> tuple[list[RolloutEpisodeResult], list[RolloutEpisodeResult], dict[str, Any]]:
    if len(replay_paths) != len(dagger_paths):
        raise ValueError("replay/DAgger full-chain 目录数量不一致")
    replay_runs = [_load_full_chain(path) for path in replay_paths]
    dagger_runs = [_load_full_chain(path) for path in dagger_paths]
    for index, (replay, dagger) in enumerate(
        zip(replay_runs, dagger_runs, strict=True), start=1
    ):
        _validate_paired_directories(replay, dagger, label=f"full-chain pair {index}")

    for label, runs in (("replay", replay_runs), ("dagger", dagger_runs)):
        checkpoint_sha = runs[0].experiment["checkpoint"]["sha256"]
        identity = _run_identity(runs[0].experiment)
        for run in runs[1:]:
            if run.experiment["checkpoint"]["sha256"] != checkpoint_sha:
                raise ValueError(f"{label} Stage A/B 使用了不同 checkpoint")
            if _run_identity(run.experiment) != identity:
                raise ValueError(f"{label} Stage A/B 评估配置或数据身份漂移")

    replay_results = [
        result
        for run in replay_runs
        for result in run.results
        if isinstance(result, RolloutEpisodeResult)
    ]
    dagger_results = [
        result
        for run in dagger_runs
        for result in run.results
        if isinstance(result, RolloutEpisodeResult)
    ]
    for label, rows in (("replay", replay_results), ("dagger", dagger_results)):
        identities = {(row.seed_group, row.seed) for row in rows}
        if len(identities) != len(rows):
            raise ValueError(f"{label} 合并 full-chain 目录后 seed 重复")
    return replay_results, dagger_results, {
        "replay": [run.receipt for run in replay_runs],
        "dagger": [run.receipt for run in dagger_runs],
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"拒绝覆盖 paired evaluation 输出: {path}")
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
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def run(args: argparse.Namespace) -> dict[str, Any]:
    replay_full, dagger_full, full_receipts = _merge_full_chain(
        args.replay_full_chain,
        args.dagger_full_chain,
    )
    if (args.replay_atomic is None) != (args.dagger_atomic is None):
        raise ValueError("--replay-atomic 与 --dagger-atomic 必须成对提供")
    replay_atomic_results: list[AtomicSkillEpisodeResult] | None = None
    dagger_atomic_results: list[AtomicSkillEpisodeResult] | None = None
    atomic_receipts: dict[str, Any] | None = None
    if args.replay_atomic is not None and args.dagger_atomic is not None:
        replay_atomic = _load_atomic(args.replay_atomic)
        dagger_atomic = _load_atomic(args.dagger_atomic)
        _validate_paired_directories(replay_atomic, dagger_atomic, label="atomic pair")
        replay_atomic_results = [
            row
            for row in replay_atomic.results
            if isinstance(row, AtomicSkillEpisodeResult)
        ]
        dagger_atomic_results = [
            row
            for row in dagger_atomic.results
            if isinstance(row, AtomicSkillEpisodeResult)
        ]
        if (
            replay_atomic.experiment["checkpoint"]["sha256"]
            != full_receipts["replay"][0]["checkpoint_sha256"]
            or dagger_atomic.experiment["checkpoint"]["sha256"]
            != full_receipts["dagger"][0]["checkpoint_sha256"]
        ):
            raise ValueError("full-chain 与 atomic 使用了不同 checkpoint")
        atomic_receipts = {
            "replay": replay_atomic.receipt,
            "dagger": dagger_atomic.receipt,
        }

    analysis = analyze_e012_pair(
        replay_full,
        dagger_full,
        replay_atomic=replay_atomic_results,
        dagger_atomic=dagger_atomic_results,
        protocol=args.protocol,
    )
    analysis["input_receipts"] = {
        "full_chain": full_receipts,
        "atomic": atomic_receipts,
    }
    if not math.isfinite(analysis["full_chain"]["mean_completed_skill_count_delta"]):
        raise ValueError("paired evaluation 产生非有限统计")
    _atomic_write_json(args.output, analysis)
    return analysis


def main() -> None:
    analysis = run(_parse_args())
    print(json.dumps(analysis, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
