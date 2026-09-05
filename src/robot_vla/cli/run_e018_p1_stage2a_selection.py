"""E018-P1 Stage 2A 信息增益选择的隔离 Pass A/Pass B 入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from robot_vla.precision.e018_p1_stage2a_selection import (
    STAGE2A_SELECTION_GO,
    STAGE2A_SELECTION_PREFLIGHT_GO,
    load_e018_p1_stage2a_selection_config,
    verify_selection_parent_gate,
)
from robot_vla.precision.e018_p1_stage2a_selection_runtime import (
    run_e018_p1_stage2a_selection_capture,
    run_e018_p1_stage2a_selection_preflight_one_route,
    run_e018_p1_stage2a_selection_score_private,
    verify_e018_p1_stage2a_selection_preflight,
    verify_e018_p1_stage2a_selection_public,
    verify_e018_p1_stage2a_selection_result,
)


def _add_base_configs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--selection-config", type=Path, required=True)
    parser.add_argument("--stage2a-config", type=Path, required=True)
    parser.add_argument("--qualification-config", type=Path, required=True)


def _add_source_identity(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--expected-source-git-commit", required=True)
    parser.add_argument("--expected-source-identity-sha256", required=True)


def _add_parent_artifacts(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stage2a-artifact-root", type=Path, required=True)
    parser.add_argument(
        "--stage2a-control-evidence-root", type=Path, required=True
    )
    parser.add_argument("--artifact-inventory", type=Path, required=True)
    parser.add_argument(
        "--parent-replay-artifact-root", type=Path, required=True
    )
    parser.add_argument(
        "--parent-replay-control-evidence-root", type=Path, required=True
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify_config = subparsers.add_parser("verify-config")
    verify_config.add_argument("--selection-config", type=Path, required=True)

    verify_parents = subparsers.add_parser("verify-parents")
    _add_base_configs(verify_parents)
    _add_parent_artifacts(verify_parents)

    capture = subparsers.add_parser("capture-public")
    _add_base_configs(capture)
    _add_parent_artifacts(capture)
    _add_source_identity(capture)
    capture.add_argument("--g0c-config", type=Path, required=True)
    capture.add_argument("--data-config", type=Path, required=True)
    capture.add_argument("--stats-root", type=Path, required=True)
    capture.add_argument("--selected-checkpoint", type=Path, required=True)
    capture.add_argument("--repository-root", type=Path, required=True)
    capture.add_argument("--artifact-root", type=Path, required=True)
    capture.add_argument("--expected-config-raw-sha256", required=True)
    capture.add_argument("--expected-config-canonical-sha256", required=True)
    capture.add_argument(
        "--selection-go",
        required=True,
        help=f"必须精确为 {STAGE2A_SELECTION_GO}",
    )

    preflight = subparsers.add_parser("preflight-one-route")
    _add_base_configs(preflight)
    _add_parent_artifacts(preflight)
    _add_source_identity(preflight)
    preflight.add_argument("--g0c-config", type=Path, required=True)
    preflight.add_argument("--data-config", type=Path, required=True)
    preflight.add_argument("--stats-root", type=Path, required=True)
    preflight.add_argument("--selected-checkpoint", type=Path, required=True)
    preflight.add_argument("--repository-root", type=Path, required=True)
    preflight.add_argument("--artifact-root", type=Path, required=True)
    preflight.add_argument("--expected-config-raw-sha256", required=True)
    preflight.add_argument("--expected-config-canonical-sha256", required=True)
    preflight.add_argument(
        "--preflight-go",
        required=True,
        help=f"必须精确为 {STAGE2A_SELECTION_PREFLIGHT_GO}",
    )

    verify_preflight = subparsers.add_parser("verify-preflight")
    verify_preflight.add_argument("--selection-config", type=Path, required=True)
    verify_preflight.add_argument("--artifact-root", type=Path, required=True)
    _add_source_identity(verify_preflight)

    verify_public = subparsers.add_parser("verify-public")
    _add_base_configs(verify_public)
    _add_source_identity(verify_public)
    verify_public.add_argument("--public-root", type=Path, required=True)

    score = subparsers.add_parser("score-private")
    _add_base_configs(score)
    _add_source_identity(score)
    score.add_argument("--public-root", type=Path, required=True)
    score.add_argument("--private-root", type=Path, required=True)
    score.add_argument("--result-root", type=Path, required=True)
    score.add_argument(
        "--selection-go",
        required=True,
        help=f"必须精确为 {STAGE2A_SELECTION_GO}",
    )

    verify_result = subparsers.add_parser("verify-result")
    _add_base_configs(verify_result)
    _add_source_identity(verify_result)
    verify_result.add_argument("--public-root", type=Path, required=True)
    verify_result.add_argument("--result-root", type=Path, required=True)
    return parser


def _base_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "selection_config_path": args.selection_config,
        "stage2a_config_path": args.stage2a_config,
        "qualification_config_path": args.qualification_config,
    }


def _source_kwargs(args: argparse.Namespace) -> dict[str, str]:
    return {
        "expected_source_git_commit": args.expected_source_git_commit,
        "expected_source_identity_sha256": (
            args.expected_source_identity_sha256
        ),
    }


def _parent_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        **_base_kwargs(args),
        "stage2a_artifact_root": args.stage2a_artifact_root,
        "stage2a_control_evidence_root": args.stage2a_control_evidence_root,
        "artifact_inventory_path": args.artifact_inventory,
        "parent_replay_artifact_root": args.parent_replay_artifact_root,
        "parent_replay_control_evidence_root": (
            args.parent_replay_control_evidence_root
        ),
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result: dict[str, Any]
    if args.command == "verify-config":
        loaded = load_e018_p1_stage2a_selection_config(args.selection_config)
        result = {
            "verified": True,
            "version": loaded.payload["version"],
            "status": loaded.payload["status"],
            "raw_sha256": loaded.raw_sha256,
            "canonical_sha256": loaded.canonical_sha256,
            "selection_seeds": [77001, 77025],
            "fresh_test_status": "prohibited-unread",
        }
    elif args.command == "verify-parents":
        result = verify_selection_parent_gate(**_parent_kwargs(args))
    elif args.command == "capture-public":
        if args.selection_go != STAGE2A_SELECTION_GO:
            raise PermissionError("Stage 2A selection capture 缺 exact GO token")
        result = run_e018_p1_stage2a_selection_capture(
            **_parent_kwargs(args),
            **_source_kwargs(args),
            g0c_config_path=args.g0c_config,
            data_config_path=args.data_config,
            stats_root=args.stats_root,
            selected_checkpoint_path=args.selected_checkpoint,
            repository_root=args.repository_root,
            artifact_root=args.artifact_root,
            expected_config_raw_sha256=args.expected_config_raw_sha256,
            expected_config_canonical_sha256=(
                args.expected_config_canonical_sha256
            ),
            exact_go_token=args.selection_go,
        )
    elif args.command == "preflight-one-route":
        if args.preflight_go != STAGE2A_SELECTION_PREFLIGHT_GO:
            raise PermissionError("Stage 2A preflight 缺 exact preflight token")
        result = run_e018_p1_stage2a_selection_preflight_one_route(
            **_parent_kwargs(args),
            **_source_kwargs(args),
            g0c_config_path=args.g0c_config,
            data_config_path=args.data_config,
            stats_root=args.stats_root,
            selected_checkpoint_path=args.selected_checkpoint,
            repository_root=args.repository_root,
            artifact_root=args.artifact_root,
            expected_config_raw_sha256=args.expected_config_raw_sha256,
            expected_config_canonical_sha256=(
                args.expected_config_canonical_sha256
            ),
            exact_preflight_token=args.preflight_go,
        )
    elif args.command == "verify-preflight":
        result = verify_e018_p1_stage2a_selection_preflight(
            selection_config_path=args.selection_config,
            artifact_root=args.artifact_root,
            **_source_kwargs(args),
        )
    elif args.command == "verify-public":
        result = verify_e018_p1_stage2a_selection_public(
            **_base_kwargs(args),
            **_source_kwargs(args),
            public_root=args.public_root,
        )
    elif args.command == "score-private":
        result = run_e018_p1_stage2a_selection_score_private(
            **_base_kwargs(args),
            **_source_kwargs(args),
            public_root=args.public_root,
            private_root=args.private_root,
            result_root=args.result_root,
            exact_go_token=args.selection_go,
        )
    elif args.command == "verify-result":
        result = verify_e018_p1_stage2a_selection_result(
            **_base_kwargs(args),
            **_source_kwargs(args),
            public_root=args.public_root,
            result_root=args.result_root,
        )
    else:  # pragma: no cover - argparse 已限定 choices
        raise RuntimeError(f"未知 command: {args.command}")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
