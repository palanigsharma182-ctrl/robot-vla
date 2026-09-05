"""运行 E018-P1 Stage 2A fixed-gain conditional evaluation。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from robot_vla.precision.e018_p1_stage2a_evaluation import (
    STAGE2A_EVALUATION_GO,
    STAGE2A_EVALUATION_PREFLIGHT_GO,
    load_e018_p1_stage2a_evaluation_config,
)
from robot_vla.precision.e018_p1_stage2a_evaluation_runtime import (
    run_e018_p1_stage2a_evaluation_capture,
    run_e018_p1_stage2a_evaluation_score_private,
    verify_e018_p1_stage2a_evaluation_parent_gate,
    verify_e018_p1_stage2a_evaluation_public,
    verify_e018_p1_stage2a_evaluation_result,
)


def _add_base_configs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--evaluation-config", type=Path, required=True)
    parser.add_argument("--stage2a-config", type=Path, required=True)
    parser.add_argument("--qualification-config", type=Path, required=True)


def _add_source_identity(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--expected-source-git-commit", required=True)
    parser.add_argument("--expected-source-identity-sha256", required=True)


def _add_replay_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--g0c-config", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--stats-root", type=Path, required=True)


def _add_parent_inputs(parser: argparse.ArgumentParser) -> None:
    _add_replay_inputs(parser)
    parser.add_argument("--selected-checkpoint", type=Path, required=True)
    parser.add_argument("--selection-config", type=Path, required=True)
    parser.add_argument("--selection-public-root", type=Path, required=True)
    parser.add_argument("--selection-result-root", type=Path, required=True)
    parser.add_argument("--decision-gate", type=Path, required=True)
    parser.add_argument("--artifact-inventory", type=Path, required=True)


def _add_formal_go_identity(
    parser: argparse.ArgumentParser,
    *,
    include_preflight_roots: bool,
) -> None:
    parser.add_argument("--formal-go-receipt", type=Path, required=True)
    parser.add_argument("--expected-formal-go-raw-sha256", required=True)
    parser.add_argument("--expected-formal-go-internal-sha256", required=True)
    if include_preflight_roots:
        parser.add_argument("--preflight-public-root", type=Path, required=True)
        parser.add_argument("--preflight-result-root", type=Path, required=True)


def _add_capture(
    parser: argparse.ArgumentParser,
    *,
    go_token: str,
    require_formal_go: bool,
) -> None:
    _add_base_configs(parser)
    _add_parent_inputs(parser)
    _add_source_identity(parser)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--expected-config-raw-sha256", required=True)
    parser.add_argument("--expected-config-canonical-sha256", required=True)
    parser.add_argument(
        "--evaluation-go",
        required=True,
        help=f"必须精确为 {go_token}",
    )
    if require_formal_go:
        _add_formal_go_identity(parser, include_preflight_roots=True)


def _add_public_verifier(parser: argparse.ArgumentParser) -> None:
    _add_base_configs(parser)
    _add_source_identity(parser)
    parser.add_argument("--public-root", type=Path, required=True)


def _add_score(
    parser: argparse.ArgumentParser,
    *,
    go_token: str,
    require_formal_go: bool,
) -> None:
    _add_base_configs(parser)
    _add_replay_inputs(parser)
    _add_source_identity(parser)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument(
        "--evaluation-go",
        required=True,
        help=f"必须精确为 {go_token}",
    )
    if require_formal_go:
        _add_formal_go_identity(parser, include_preflight_roots=False)


def _add_result_verifier(parser: argparse.ArgumentParser) -> None:
    _add_public_verifier(parser)
    parser.add_argument("--result-root", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    verify_config = commands.add_parser("verify-config")
    verify_config.add_argument("--evaluation-config", type=Path, required=True)

    verify_parents = commands.add_parser("verify-parents")
    _add_base_configs(verify_parents)
    _add_parent_inputs(verify_parents)

    capture = commands.add_parser("capture-public")
    _add_capture(
        capture,
        go_token=STAGE2A_EVALUATION_GO,
        require_formal_go=True,
    )
    verify_public = commands.add_parser("verify-public")
    _add_public_verifier(verify_public)
    score = commands.add_parser("score-private")
    _add_score(
        score,
        go_token=STAGE2A_EVALUATION_GO,
        require_formal_go=True,
    )
    verify_result = commands.add_parser("verify-result")
    _add_result_verifier(verify_result)

    preflight_capture = commands.add_parser("preflight-capture-public")
    _add_capture(
        preflight_capture,
        go_token=STAGE2A_EVALUATION_PREFLIGHT_GO,
        require_formal_go=False,
    )
    preflight_public = commands.add_parser("preflight-verify-public")
    _add_public_verifier(preflight_public)
    preflight_score = commands.add_parser("preflight-score-private")
    _add_score(
        preflight_score,
        go_token=STAGE2A_EVALUATION_PREFLIGHT_GO,
        require_formal_go=False,
    )
    preflight_result = commands.add_parser("preflight-verify-result")
    _add_result_verifier(preflight_result)
    return parser


def _base_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "evaluation_config_path": args.evaluation_config,
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


def _replay_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "g0c_config_path": args.g0c_config,
        "data_config_path": args.data_config,
        "stats_root": args.stats_root,
    }


def _parent_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        **_base_kwargs(args),
        **_replay_kwargs(args),
        "selected_checkpoint_path": args.selected_checkpoint,
        "selection_config_path": args.selection_config,
        "selection_public_root": args.selection_public_root,
        "selection_result_root": args.selection_result_root,
        "decision_gate_path": args.decision_gate,
        "artifact_inventory_path": args.artifact_inventory,
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    preflight = args.command.startswith("preflight-")
    result: dict[str, Any]
    if args.command == "verify-config":
        loaded = load_e018_p1_stage2a_evaluation_config(args.evaluation_config)
        result = {
            "verified": True,
            "version": loaded.payload["version"],
            "status": loaded.payload["status"],
            "raw_sha256": loaded.raw_sha256,
            "canonical_sha256": loaded.canonical_sha256,
            "evaluation_seeds": loaded.payload["split"]["seeds"],
            "selected_gain": loaded.payload["fixed_rule"][
                "selected_min_information_gain"
            ],
            "fresh_test_status": "prohibited-unread",
        }
    elif args.command == "verify-parents":
        result = verify_e018_p1_stage2a_evaluation_parent_gate(
            **_parent_kwargs(args)
        )
    elif args.command in {"capture-public", "preflight-capture-public"}:
        result = run_e018_p1_stage2a_evaluation_capture(
            **_parent_kwargs(args),
            **_source_kwargs(args),
            repository_root=args.repository_root,
            artifact_root=args.artifact_root,
            expected_config_raw_sha256=args.expected_config_raw_sha256,
            expected_config_canonical_sha256=(
                args.expected_config_canonical_sha256
            ),
            exact_go_token=args.evaluation_go,
            formal_go_receipt_path=getattr(args, "formal_go_receipt", None),
            expected_formal_go_raw_sha256=getattr(
                args, "expected_formal_go_raw_sha256", None
            ),
            expected_formal_go_internal_sha256=getattr(
                args, "expected_formal_go_internal_sha256", None
            ),
            preflight_public_root=getattr(args, "preflight_public_root", None),
            preflight_result_root=getattr(args, "preflight_result_root", None),
            preflight=preflight,
        )
    elif args.command in {"verify-public", "preflight-verify-public"}:
        result = verify_e018_p1_stage2a_evaluation_public(
            **_base_kwargs(args),
            **_source_kwargs(args),
            public_root=args.public_root,
            preflight=preflight,
        )
    elif args.command in {"score-private", "preflight-score-private"}:
        result = run_e018_p1_stage2a_evaluation_score_private(
            **_base_kwargs(args),
            **_replay_kwargs(args),
            **_source_kwargs(args),
            public_root=args.public_root,
            private_root=args.private_root,
            result_root=args.result_root,
            repository_root=args.repository_root,
            exact_go_token=args.evaluation_go,
            formal_go_receipt_path=getattr(args, "formal_go_receipt", None),
            expected_formal_go_raw_sha256=getattr(
                args, "expected_formal_go_raw_sha256", None
            ),
            expected_formal_go_internal_sha256=getattr(
                args, "expected_formal_go_internal_sha256", None
            ),
            preflight=preflight,
        )
    elif args.command in {"verify-result", "preflight-verify-result"}:
        result = verify_e018_p1_stage2a_evaluation_result(
            **_base_kwargs(args),
            **_source_kwargs(args),
            public_root=args.public_root,
            result_root=args.result_root,
            preflight=preflight,
        )
    else:  # pragma: no cover - argparse 已限定 choices
        raise RuntimeError(f"未知 command: {args.command}")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
