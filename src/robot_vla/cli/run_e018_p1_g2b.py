"""运行 E018-P1 G2B covariance calibration / provider requalification。"""

from __future__ import annotations

import argparse
import json

from robot_vla.precision.e018_p1_g2b import (
    run_e018_p1_g2b_calibration,
    run_e018_p1_g2b_qualification,
)


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)
    parser.add_argument("--parent-g2a-config", required=True)
    parser.add_argument("--parent-g2a-receipt", required=True)
    parser.add_argument("--e016-config", required=True)
    parser.add_argument("--e013-deployable-root", required=True)
    parser.add_argument("--e016-fresh-deployable-root", required=True)
    parser.add_argument("--training-output", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--output-root", required=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)
    calibration = subparsers.add_parser("calibrate")
    _common(calibration)
    calibration.add_argument("--e016-fresh-label-root", required=True)

    qualification = subparsers.add_parser("qualify")
    _common(qualification)
    qualification.add_argument("--parent-g0c-config", required=True)
    qualification.add_argument("--parent-g0c-receipt", required=True)
    qualification.add_argument("--calibration-output", required=True)
    qualification.add_argument("--preflight-only", action="store_true")
    qualification.add_argument(
        "--decision-exit-go",
        action="store_true",
        help="仅在 decision Agent 明确批准完整 50-seed 出口后使用",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    common = {
        "config_path": args.config,
        "parent_g2a_config_path": args.parent_g2a_config,
        "parent_g2a_receipt_path": args.parent_g2a_receipt,
        "e016_config_path": args.e016_config,
        "e013_deployable_root": args.e013_deployable_root,
        "e016_fresh_deployable_root": args.e016_fresh_deployable_root,
        "training_output": args.training_output,
        "repository_root": args.repository_root,
        "output_root": args.output_root,
    }
    if args.phase == "calibrate":
        result = run_e018_p1_g2b_calibration(
            **common,
            e016_fresh_label_root=args.e016_fresh_label_root,
        )
    else:
        result = run_e018_p1_g2b_qualification(
            **common,
            parent_g0c_config_path=args.parent_g0c_config,
            parent_g0c_receipt_path=args.parent_g0c_receipt,
            calibration_output=args.calibration_output,
            preflight_only=args.preflight_only,
            decision_exit_go=args.decision_exit_go,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
