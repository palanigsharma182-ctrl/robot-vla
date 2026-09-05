"""在完整 prediction freeze 后评分并选择 E018-P1 G2C 候选。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_vla.precision.e018_p1_g2c_model_val import (
    score_select_g2c_model_val,
    verify_g2c_model_val_selection,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    run = phases.add_parser("score-select")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--prediction-freeze", type=Path, required=True)
    run.add_argument("--model-val-label-input", type=Path, required=True)
    run.add_argument("--repository-root", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--decision-exit-go", action="store_true")
    verify = phases.add_parser("verify")
    verify.add_argument("--config", type=Path, required=True)
    verify.add_argument("--prediction-freeze", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.phase == "score-select":
        result = score_select_g2c_model_val(
            config_path=args.config,
            prediction_freeze_root=args.prediction_freeze,
            model_val_label_input_root=args.model_val_label_input,
            repository_root=args.repository_root,
            output_root=args.output,
            decision_exit_go=args.decision_exit_go,
        )
    else:
        result = verify_g2c_model_val_selection(
            config_path=args.config,
            prediction_freeze_root=args.prediction_freeze,
            output_root=args.output,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
