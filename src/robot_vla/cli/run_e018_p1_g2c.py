"""运行 E018-P1 G2C front-provider DATA / engineering smoke。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_vla.precision.e018_p1_g2c import (
    run_e018_p1_g2c_smoke,
    verify_g2c_smoke_receipt,
)
from robot_vla.precision.e018_p1_g2c_data import (
    run_e018_p1_g2c_data,
    verify_g2c_data_receipt,
)


def _data_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parent-g0c-config", type=Path, required=True)
    parser.add_argument("--parent-g0c-receipt", type=Path, required=True)
    parser.add_argument("--e013-deployable-root", type=Path, required=True)
    parser.add_argument("--e016-fresh-deployable-root", type=Path, required=True)
    parser.add_argument("--stats-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)

    smoke = subparsers.add_parser("smoke")
    _data_inputs(smoke)
    smoke.add_argument("--e016-config", type=Path, required=True)
    smoke.add_argument("--training-output", type=Path, required=True)

    data = subparsers.add_parser("collect-data")
    _data_inputs(data)
    data.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    data.add_argument(
        "--decision-exit-go",
        action="store_true",
        help="只有 full-data 前独立 R2 明确批准后才可使用",
    )

    verify_data = subparsers.add_parser("verify-data")
    verify_data.add_argument("--output", type=Path, required=True)
    verify_data.add_argument("--config", type=Path)
    verify_data.add_argument("--parent-g0c-config", type=Path)

    verify_smoke = subparsers.add_parser("verify-smoke")
    verify_smoke.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.phase == "verify-data":
        result = verify_g2c_data_receipt(
            args.output,
            config_path=args.config,
            parent_g0c_config_path=args.parent_g0c_config,
        )
    elif args.phase == "verify-smoke":
        result = verify_g2c_smoke_receipt(args.output)
    else:
        common = {
            "config_path": args.config,
            "parent_g0c_config_path": args.parent_g0c_config,
            "parent_g0c_receipt_path": args.parent_g0c_receipt,
            "e013_deployable_root": args.e013_deployable_root,
            "e016_fresh_deployable_root": args.e016_fresh_deployable_root,
            "stats_root": args.stats_root,
            "inventory_path": args.inventory,
            "repository_root": args.repository_root,
            "output_root": args.output,
        }
        if args.phase == "collect-data":
            result = run_e018_p1_g2c_data(
                **common,
                mode=args.mode,
                decision_exit_go=args.decision_exit_go,
            )
        else:
            result = run_e018_p1_g2c_smoke(
                **common,
                e016_config_path=args.e016_config,
                training_output=args.training_output,
            )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
