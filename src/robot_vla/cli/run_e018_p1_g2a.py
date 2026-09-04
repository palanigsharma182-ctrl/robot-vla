"""运行 E018-P1 G2A front-provider development qualification。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_vla.precision.e018_p1_g2a import run_e018_p1_g2a


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parent-g0c-config", type=Path, required=True)
    parser.add_argument("--parent-g0c-receipt", type=Path, required=True)
    parser.add_argument("--e016-config", type=Path, required=True)
    parser.add_argument("--e013-deployable-root", type=Path, required=True)
    parser.add_argument("--e016-fresh-deployable-root", type=Path, required=True)
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="只跑一个冻结 seed 的两阶段接口 smoke，不计算 qualification gate",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_e018_p1_g2a(
        config_path=args.config,
        parent_g0c_config_path=args.parent_g0c_config,
        parent_g0c_receipt_path=args.parent_g0c_receipt,
        e016_config_path=args.e016_config,
        e013_deployable_root=args.e013_deployable_root,
        e016_fresh_deployable_root=args.e016_fresh_deployable_root,
        training_output=args.training_output,
        repository_root=args.repository_root,
        output_root=args.output,
        preflight_only=args.preflight_only,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
