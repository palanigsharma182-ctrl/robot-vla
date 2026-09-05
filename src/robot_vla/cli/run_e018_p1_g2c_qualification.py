"""运行/评分/公开验证 E018-P1 G2C dynamic qualification。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from robot_vla.precision.e018_p1_g2c_qualification import (
    load_g2c_dynamic_qualification_config,
    run_e018_p1_g2c_qualification_capture,
    run_e018_p1_g2c_qualification_smoke,
    score_e018_p1_g2c_qualification,
    verify_g2c_qualification_combined_artifacts,
    verify_g2c_qualification_execution,
    verify_g2c_qualification_failure,
    verify_g2c_qualification_parents,
    verify_g2c_qualification_result,
)


def _add_parent_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--g0c-config", type=Path, required=True)
    parser.add_argument("--g0c-receipt", type=Path, required=True)
    parser.add_argument("--calibration-config", type=Path, required=True)
    parser.add_argument("--calibration-prediction-freeze-root", type=Path, required=True)
    parser.add_argument("--calibration-result-root", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, required=True)


def _add_capture_inputs(parser: argparse.ArgumentParser) -> None:
    _add_parent_inputs(parser)
    parser.add_argument("--stats-root", type=Path, required=True)
    parser.add_argument("--selected-checkpoint", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--private-label-output", type=Path, required=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify_config = subparsers.add_parser("verify-config")
    verify_config.add_argument("--config", type=Path, required=True)

    verify_parents = subparsers.add_parser("verify-parents")
    _add_parent_inputs(verify_parents)

    smoke = subparsers.add_parser("smoke")
    _add_capture_inputs(smoke)
    smoke.add_argument("--seed", type=int, required=True)
    smoke.add_argument("--alternate-viewpoint-id", default="LEFT_LOW__CENTER")

    capture = subparsers.add_parser("capture")
    _add_capture_inputs(capture)
    capture.add_argument("--expected-source-git-commit", required=True)
    capture.add_argument("--expected-source-identity-sha256", required=True)
    capture.add_argument("--decision-receipt", type=Path, required=True)
    capture.add_argument("--expected-decision-receipt-raw-sha256", required=True)
    capture.add_argument("--expected-decision-receipt-internal-sha256", required=True)
    capture.add_argument("--decision-execution-go", action="store_true")

    verify_execution = subparsers.add_parser("verify-execution")
    verify_execution.add_argument("--config", type=Path, required=True)
    verify_execution.add_argument("--public-execution-root", type=Path, required=True)

    verify_failure = subparsers.add_parser("verify-failure")
    verify_failure.add_argument("--config", type=Path, required=True)
    verify_failure.add_argument("--public-execution-root", type=Path, required=True)

    score = subparsers.add_parser("score")
    score.add_argument("--config", type=Path, required=True)
    score.add_argument("--public-execution-root", type=Path, required=True)
    score.add_argument("--private-label-root", type=Path, required=True)
    score.add_argument("--result-output", type=Path, required=True)
    score.add_argument("--decision-scoring-go", action="store_true")

    verify = subparsers.add_parser("verify")
    verify.add_argument("--config", type=Path, required=True)
    verify.add_argument("--public-execution-root", type=Path, required=True)
    verify.add_argument("--result-root", type=Path, required=True)

    verify_artifacts = subparsers.add_parser("verify-artifacts")
    verify_artifacts.add_argument("--config", type=Path, required=True)
    verify_artifacts.add_argument("--public-execution-root", type=Path, required=True)
    verify_artifacts.add_argument("--private-label-root", type=Path, required=True)
    verify_artifacts.add_argument("--result-root", type=Path, required=True)
    return parser.parse_args()


def _parent_kwargs(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "qualification_config_path": args.config,
        "g0c_config_path": args.g0c_config,
        "g0c_receipt_path": args.g0c_receipt,
        "calibration_config_path": args.calibration_config,
        "calibration_prediction_freeze_root": (args.calibration_prediction_freeze_root),
        "calibration_result_root": args.calibration_result_root,
        "data_config_path": args.data_config,
    }


def _capture_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        **_parent_kwargs(args),
        "stats_root": args.stats_root,
        "selected_checkpoint_path": args.selected_checkpoint,
        "repository_root": args.repository_root,
        "public_output_root": args.public_output,
        "private_label_output_root": args.private_label_output,
    }


def main() -> None:
    args = _parse_args()
    if args.command == "verify-config":
        config = load_g2c_dynamic_qualification_config(args.config)
        result: dict[str, Any] = {
            "verified": True,
            "version": config["version"],
            "status": config["status"],
            "config_sha256": config["config_sha256"],
        }
    elif args.command == "verify-parents":
        parent = verify_g2c_qualification_parents(**_parent_kwargs(args))
        result = {
            "verified": True,
            "config_sha256": parent["config_sha256"],
            "g0c_config_sha256": parent["g0c_config_sha256"],
            "calibration_verification": parent["calibration_verification"],
            "calibration_identities": parent["calibration_identities"],
        }
    elif args.command == "smoke":
        result = run_e018_p1_g2c_qualification_smoke(
            **_capture_kwargs(args),
            seed=args.seed,
            alternate_viewpoint_id=args.alternate_viewpoint_id,
        )
    elif args.command == "capture":
        result = run_e018_p1_g2c_qualification_capture(
            **_capture_kwargs(args),
            expected_source_git_commit=args.expected_source_git_commit,
            expected_source_identity_sha256=(args.expected_source_identity_sha256),
            decision_execution_go=args.decision_execution_go,
            decision_receipt_path=args.decision_receipt,
            expected_decision_receipt_raw_sha256=(args.expected_decision_receipt_raw_sha256),
            expected_decision_receipt_internal_sha256=(
                args.expected_decision_receipt_internal_sha256
            ),
        )
    elif args.command == "verify-execution":
        result = verify_g2c_qualification_execution(
            qualification_config_path=args.config,
            public_execution_root=args.public_execution_root,
        )
    elif args.command == "verify-failure":
        result = verify_g2c_qualification_failure(
            qualification_config_path=args.config,
            public_execution_root=args.public_execution_root,
        )
    elif args.command == "score":
        result = score_e018_p1_g2c_qualification(
            qualification_config_path=args.config,
            public_execution_root=args.public_execution_root,
            private_label_root=args.private_label_root,
            result_output_root=args.result_output,
            decision_scoring_go=args.decision_scoring_go,
        )
    elif args.command == "verify":
        result = verify_g2c_qualification_result(
            qualification_config_path=args.config,
            public_execution_root=args.public_execution_root,
            result_root=args.result_root,
        )
    elif args.command == "verify-artifacts":
        result = verify_g2c_qualification_combined_artifacts(
            qualification_config_path=args.config,
            public_execution_root=args.public_execution_root,
            private_label_root=args.private_label_root,
            result_root=args.result_root,
        )
    else:  # pragma: no cover - argparse 已限制 choices
        raise RuntimeError(f"未知 command: {args.command}")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
