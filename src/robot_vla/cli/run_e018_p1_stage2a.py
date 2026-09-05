"""E018-P1 Stage 2A PRIMARY+Memory 隔离 integration smoke 入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from robot_vla.precision.e018_p1_stage2a import (
    STAGE2A_INTEGRATION_SMOKE_GO,
    load_e018_p1_stage2a_config,
    run_e018_p1_stage2a_integration_smoke,
    verify_e018_p1_stage2a_integration_smoke,
    verify_stage2a_parent_gate,
)


def _add_parent_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--qualification-config", type=Path, required=True)
    parser.add_argument(
        "--qualification-public-execution-root", type=Path, required=True
    )
    parser.add_argument("--qualification-result-root", type=Path, required=True)
    parser.add_argument("--g0c-config", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify_config = subparsers.add_parser("verify-config")
    verify_config.add_argument("--config", type=Path, required=True)

    verify_parents = subparsers.add_parser("verify-parents")
    _add_parent_inputs(verify_parents)

    smoke = subparsers.add_parser("smoke")
    _add_parent_inputs(smoke)
    smoke.add_argument("--stats-root", type=Path, required=True)
    smoke.add_argument("--selected-checkpoint", type=Path, required=True)
    smoke.add_argument("--repository-root", type=Path, required=True)
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--expected-config-raw-sha256", required=True)
    smoke.add_argument("--expected-config-canonical-sha256", required=True)
    smoke.add_argument("--expected-source-git-commit", required=True)
    smoke.add_argument("--expected-source-identity-sha256", required=True)
    smoke.add_argument(
        "--integration-smoke-go",
        required=True,
        help=f"必须精确为 {STAGE2A_INTEGRATION_SMOKE_GO}",
    )

    verify = subparsers.add_parser("verify")
    verify.add_argument("--config", type=Path, required=True)
    verify.add_argument("--qualification-config", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    verify.add_argument("--expected-source-git-commit", required=True)
    verify.add_argument("--expected-source-identity-sha256", required=True)
    return parser


def _parent_kwargs(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "stage2_config_path": args.config,
        "qualification_config_path": args.qualification_config,
        "qualification_public_execution_root": (
            args.qualification_public_execution_root
        ),
        "qualification_result_root": args.qualification_result_root,
        "g0c_config_path": args.g0c_config,
        "data_config_path": args.data_config,
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result: dict[str, Any]
    if args.command == "verify-config":
        loaded = load_e018_p1_stage2a_config(args.config)
        result = {
            "verified": True,
            "version": loaded.payload["version"],
            "status": loaded.payload["status"],
            "raw_sha256": loaded.raw_sha256,
            "canonical_sha256": loaded.canonical_sha256,
            "integration_smoke_seeds": [76901, 76910],
            "fresh_test_status": "prohibited-unread",
        }
    elif args.command == "verify-parents":
        result = verify_stage2a_parent_gate(**_parent_kwargs(args))
    elif args.command == "smoke":
        if args.integration_smoke_go != STAGE2A_INTEGRATION_SMOKE_GO:
            raise PermissionError("Stage 2A integration smoke 缺少 exact GO token")
        result = run_e018_p1_stage2a_integration_smoke(
            **_parent_kwargs(args),
            stats_root=args.stats_root,
            selected_checkpoint_path=args.selected_checkpoint,
            repository_root=args.repository_root,
            output_root=args.output,
            expected_stage2_config_raw_sha256=(
                args.expected_config_raw_sha256
            ),
            expected_stage2_config_canonical_sha256=(
                args.expected_config_canonical_sha256
            ),
            expected_source_git_commit=args.expected_source_git_commit,
            expected_source_identity_sha256=(
                args.expected_source_identity_sha256
            ),
            integration_smoke_go=args.integration_smoke_go,
        )
    elif args.command == "verify":
        result = verify_e018_p1_stage2a_integration_smoke(
            stage2_config_path=args.config,
            qualification_config_path=args.qualification_config,
            output_root=args.output,
            expected_source_git_commit=args.expected_source_git_commit,
            expected_source_identity_sha256=(
                args.expected_source_identity_sha256
            ),
        )
    else:  # pragma: no cover - argparse 已限定 choices
        raise RuntimeError(f"未知 command: {args.command}")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
