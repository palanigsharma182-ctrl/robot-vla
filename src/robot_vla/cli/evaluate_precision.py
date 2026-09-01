"""运行 E013 held-out perception、confidence calibration 与 Provider latency。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_vla.precision.held_out import evaluate_precision_checkpoint


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployable-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    receipt = evaluate_precision_checkpoint(
        deployable_root=args.deployable_root,
        label_root=args.label_root,
        config_path=args.config,
        training_output=args.training_output,
        output_root=args.output,
    )
    print(json.dumps(receipt.to_dict(), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
