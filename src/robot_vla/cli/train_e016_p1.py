"""在云端 RTX 4090 运行 E016-P1 corrected-observability 正式训练。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_vla.precision.e016_training import run_e016_p1_formal_training


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployable-root", type=Path, required=True)
    parser.add_argument("--source-label-root", type=Path, required=True)
    parser.add_argument("--p0-output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    receipt = run_e016_p1_formal_training(
        deployable_root=args.deployable_root,
        source_label_root=args.source_label_root,
        p0_output_root=args.p0_output,
        config_path=args.config,
        output_root=args.output,
        repository_root=args.repository_root,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    if not receipt["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
