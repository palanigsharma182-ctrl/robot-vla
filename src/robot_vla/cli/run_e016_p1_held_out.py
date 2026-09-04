"""运行 E016-P1 fresh validation calibration 或 test-once goal-memory replay。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_vla.precision.e016_evaluation import (
    run_e016_p1_calibration,
    run_e016_p1_test_once,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("calibrate", "test-once"), required=True)
    parser.add_argument("--deployable-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--calibration-root", type=Path)
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--public-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.phase == "calibrate":
        if args.output is None:
            raise ValueError("calibrate phase 必须提供 --output")
        result = run_e016_p1_calibration(
            deployable_root=args.deployable_root,
            label_root=args.label_root,
            config_path=args.config,
            training_output=args.training_output,
            repository_root=args.repository_root,
            output_root=args.output,
        )
    else:
        if (
            args.calibration_root is None
            or args.private_output is None
            or args.public_output is None
        ):
            raise ValueError(
                "test-once phase 必须提供 --calibration-root/--private-output/--public-output"
            )
        result = run_e016_p1_test_once(
            deployable_root=args.deployable_root,
            label_root=args.label_root,
            config_path=args.config,
            training_output=args.training_output,
            repository_root=args.repository_root,
            calibration_root=args.calibration_root,
            private_output_root=args.private_output,
            public_output_root=args.public_output,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
