"""运行 E018-P0 recorded-validation Object Memory 开发实验。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_vla.precision.e018_evaluation import run_e018_p0_development


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployable-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--parent-config", type=Path, required=True)
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--goal-calibration-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_e018_p0_development(
        deployable_root=args.deployable_root,
        label_root=args.label_root,
        parent_config_path=args.parent_config,
        training_output=args.training_output,
        goal_calibration_root=args.goal_calibration_root,
        config_path=args.config,
        repository_root=args.repository_root,
        output_root=args.output,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
