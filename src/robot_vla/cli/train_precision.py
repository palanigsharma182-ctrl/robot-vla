"""执行 E013 real-sample overfit gate、正式训练和 checkpoint 冻结。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_vla.precision.training import train_precision_formal


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployable-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    receipt = train_precision_formal(
        deployable_root=args.deployable_root,
        label_root=args.label_root,
        config_path=args.config,
        output_root=args.output,
        repository_root=args.repository_root,
    )
    print(json.dumps(receipt.to_dict(), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
