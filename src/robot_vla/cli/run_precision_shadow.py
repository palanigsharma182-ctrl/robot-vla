"""执行 E013 100-seed paired、no-actuation、20 Hz budget shadow rollout。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_vla.precision.shadow import run_precision_paired_shadow


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployable-training-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--held-out-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_precision_paired_shadow(
        deployable_training_root=args.deployable_training_root,
        config_path=args.config,
        training_output=args.training_output,
        held_out_output=args.held_out_output,
        output_root=args.output,
        repository_root=args.repository_root,
    )
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
