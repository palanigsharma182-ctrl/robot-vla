"""冻结 E018-P1 G2C 的 deployable-only model-validation 输出。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_vla.precision.e018_p1_g2c_model_val import (
    run_g2c_model_val_prediction_freeze,
    verify_g2c_prediction_freeze,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    run = phases.add_parser("freeze")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--training-output", type=Path, required=True)
    run.add_argument("--e016-training-output", type=Path, required=True)
    run.add_argument("--model-val-deployable-input", type=Path, required=True)
    run.add_argument("--repository-root", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--decision-exit-go", action="store_true")
    verify = phases.add_parser("verify")
    verify.add_argument("--config", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.phase == "freeze":
        result = run_g2c_model_val_prediction_freeze(
            config_path=args.config,
            training_output_root=args.training_output,
            e016_training_output=args.e016_training_output,
            model_val_deployable_input_root=args.model_val_deployable_input,
            repository_root=args.repository_root,
            output_root=args.output,
            decision_exit_go=args.decision_exit_go,
        )
    else:
        result = verify_g2c_prediction_freeze(
            config_path=args.config, output_root=args.output
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
