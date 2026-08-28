"""顺序运行并聚合 Layer 12 periodic checkpoint 的 Reach/Transport sweep。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from robot_vla.cli.evaluate_atomic_maniskill import build_atomic_specs
from robot_vla.cli.train_stage1 import compute_source_revision
from robot_vla.evaluation.checkpoint_sweep import (
    CHECKPOINT_SWEEP_FORMAT,
    CheckpointSweepCandidate,
    assess_promotion_candidates,
    discover_sweep_candidates,
    read_atomic_results,
    select_confirmation_labels,
    summarize_candidate_results,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qwen-context-layer", type=int, choices=(12, 24), default=12)
    parser.add_argument("--screen-seed-start", type=int, default=10_020)
    parser.add_argument("--screen-episodes", type=int, default=3)
    parser.add_argument("--confirm-seed-start", type=int, default=10_023)
    parser.add_argument("--confirm-episodes", type=int, default=10)
    parser.add_argument("--top-k-per-skill", type=int, default=2)
    parser.add_argument("--max-policy-steps", type=int, default=100)
    parser.add_argument("--sampling-seed", type=int, default=42_424)
    parser.add_argument("--num-flow-steps", type=int, default=10)
    parser.add_argument(
        "--temporal-ensemble",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--recency-decay", type=float, default=0.5)
    parser.add_argument("--max-anomaly-replans", type=int, default=3)
    parser.add_argument(
        "--screen-only",
        action="store_true",
        help="只完成 Stage A；重新运行且不带此参数会复用结果并继续 Stage B",
    )
    return parser.parse_args()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _selected_seeds(data: Path, *, seed_start: int, episodes: int) -> list[int]:
    specs = build_atomic_specs(
        data,
        seed_start=seed_start,
        episodes=episodes,
        skills=["reach", "transport"],
    )
    return list(dict.fromkeys(item.seed for item in specs))


def _manifest(
    args: argparse.Namespace,
    *,
    candidates: list[CheckpointSweepCandidate],
    anchor_label: str,
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[3]
    return {
        "format": CHECKPOINT_SWEEP_FORMAT,
        "sweep_code_revision": compute_source_revision(project_root),
        "run": str(args.run.resolve()),
        "data": str(args.data.resolve()),
        "model_cache": str(args.model_cache.resolve()),
        "anchor_label": anchor_label,
        "candidates": [item.to_dict() for item in candidates],
        "skills": ["reach", "transport"],
        "screen": {
            "seed_start": args.screen_seed_start,
            "episodes_per_skill": args.screen_episodes,
            "seeds": _selected_seeds(
                args.data,
                seed_start=args.screen_seed_start,
                episodes=args.screen_episodes,
            ),
        },
        "confirmation": {
            "seed_start": args.confirm_seed_start,
            "episodes_per_skill": args.confirm_episodes,
            "seeds": _selected_seeds(
                args.data,
                seed_start=args.confirm_seed_start,
                episodes=args.confirm_episodes,
            ),
            "top_k_per_skill": args.top_k_per_skill,
        },
        "evaluation": {
            "qwen_context_layer": args.qwen_context_layer,
            "max_policy_steps": args.max_policy_steps,
            "sampling_seed": args.sampling_seed,
            "num_flow_steps": args.num_flow_steps,
            "temporal_ensemble_enabled": args.temporal_ensemble,
            "recency_decay": args.recency_decay,
            "max_anomaly_replans": args.max_anomaly_replans,
        },
    }


def _write_or_validate_manifest(path: Path, manifest: dict[str, Any]) -> None:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError("已有 sweep manifest 与当前实验身份不一致")
        return
    if path.parent.exists() and any(path.parent.iterdir()):
        raise FileExistsError("Sweep 输出目录非空但缺少 manifest，拒绝覆盖")
    _atomic_write_json(path, manifest)


def _is_complete(path: Path, expected_episodes: int) -> bool:
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return bool(payload.get("complete")) and int(payload.get("completed_episodes", -1)) == int(
        expected_episodes
    )


def _atomic_command(
    args: argparse.Namespace,
    *,
    candidate: CheckpointSweepCandidate,
    output: Path,
    seed_start: int,
    episodes: int,
    resume: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "robot_vla.cli.evaluate_atomic_maniskill",
        "--data",
        str(args.data),
        "--model-cache",
        str(args.model_cache),
        "--qwen-context-layer",
        str(args.qwen_context_layer),
        "--checkpoint",
        str(candidate.checkpoint),
        "--output",
        str(output),
        "--seed-start",
        str(seed_start),
        "--episodes",
        str(episodes),
        "--skills",
        "reach",
        "transport",
        "--max-policy-steps",
        str(args.max_policy_steps),
        "--sampling-seed",
        str(args.sampling_seed),
        "--num-flow-steps",
        str(args.num_flow_steps),
        "--recency-decay",
        str(args.recency_decay),
        "--max-anomaly-replans",
        str(args.max_anomaly_replans),
        "--temporal-ensemble" if args.temporal_ensemble else "--no-temporal-ensemble",
    ]
    if resume:
        command.append("--resume")
    return command


def _run_candidate(
    args: argparse.Namespace,
    *,
    stage: str,
    candidate: CheckpointSweepCandidate,
    seed_start: int,
    episodes: int,
) -> None:
    output = args.output / stage / candidate.label
    expected = episodes * 2
    if _is_complete(output / "summary.json", expected):
        print(
            json.dumps(
                {"event": "candidate_skip_complete", "stage": stage, "label": candidate.label},
                sort_keys=True,
            ),
            flush=True,
        )
        return
    resume = output.is_dir() and any(output.iterdir())
    print(
        json.dumps(
            {
                "event": "candidate_start",
                "stage": stage,
                "label": candidate.label,
                "epoch": candidate.epoch,
                "resume": resume,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    subprocess.run(
        _atomic_command(
            args,
            candidate=candidate,
            output=output,
            seed_start=seed_start,
            episodes=episodes,
            resume=resume,
        ),
        check=True,
    )


def _summarize_stage(
    args: argparse.Namespace,
    *,
    stage: str,
    candidates: Iterable[CheckpointSweepCandidate],
    seed_start: int,
    episodes: int,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for candidate in candidates:
        output = args.output / stage / candidate.label
        if not _is_complete(output / "summary.json", episodes * 2):
            raise RuntimeError(f"{stage}/{candidate.label} 未完成")
        results = read_atomic_results(output / "episodes.jsonl")
        expected_seeds = set(
            _selected_seeds(args.data, seed_start=seed_start, episodes=episodes)
        )
        identities = {(row.skill_name, row.seed) for row in results}
        expected = {
            (skill_name, seed)
            for skill_name in ("reach", "transport")
            for seed in expected_seeds
        }
        if identities != expected:
            raise ValueError(f"{stage}/{candidate.label} Episode 身份不完整")
        summaries.append(summarize_candidate_results(candidate, results))
    return summaries


def run(args: argparse.Namespace) -> None:
    if (
        args.screen_episodes <= 0
        or args.confirm_episodes <= 0
        or args.top_k_per_skill <= 0
        or args.screen_seed_start < 0
        or args.confirm_seed_start < 0
    ):
        raise ValueError("Sweep seed、episodes 和 top-k 必须有效")
    candidates, anchor_label = discover_sweep_candidates(args.run)
    by_label = {item.label: item for item in candidates}
    manifest = _manifest(args, candidates=candidates, anchor_label=anchor_label)
    _write_or_validate_manifest(args.output / "sweep-manifest.json", manifest)

    for candidate in candidates:
        _run_candidate(
            args,
            stage="screen",
            candidate=candidate,
            seed_start=args.screen_seed_start,
            episodes=args.screen_episodes,
        )
    screen_summaries = _summarize_stage(
        args,
        stage="screen",
        candidates=candidates,
        seed_start=args.screen_seed_start,
        episodes=args.screen_episodes,
    )
    confirmation_labels, rankings = select_confirmation_labels(
        screen_summaries,
        anchor_label=anchor_label,
        top_k_per_skill=args.top_k_per_skill,
    )
    screen_summary = {
        "format": CHECKPOINT_SWEEP_FORMAT,
        "stage": "screen",
        "complete": True,
        "candidate_summaries": screen_summaries,
        "rankings": rankings,
        "confirmation_labels": confirmation_labels,
    }
    _atomic_write_json(args.output / "screen-summary.json", screen_summary)
    print(json.dumps({"event": "screen_complete", **screen_summary}, sort_keys=True), flush=True)
    if args.screen_only:
        return

    confirmation_candidates = [by_label[label] for label in confirmation_labels]
    for candidate in confirmation_candidates:
        _run_candidate(
            args,
            stage="confirm",
            candidate=candidate,
            seed_start=args.confirm_seed_start,
            episodes=args.confirm_episodes,
        )
    confirmation_summaries = _summarize_stage(
        args,
        stage="confirm",
        candidates=confirmation_candidates,
        seed_start=args.confirm_seed_start,
        episodes=args.confirm_episodes,
    )
    promotion_labels, comparisons = assess_promotion_candidates(
        confirmation_summaries, anchor_label=anchor_label
    )
    sweep_summary = {
        "format": CHECKPOINT_SWEEP_FORMAT,
        "complete": True,
        "anchor_label": anchor_label,
        "screen_confirmation_labels": confirmation_labels,
        "confirmation_candidate_summaries": confirmation_summaries,
        "comparisons_to_anchor": comparisons,
        "promotion_candidate_labels": promotion_labels,
    }
    _atomic_write_json(args.output / "sweep-summary.json", sweep_summary)
    print(json.dumps({"event": "sweep_complete", **sweep_summary}, sort_keys=True), flush=True)


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
