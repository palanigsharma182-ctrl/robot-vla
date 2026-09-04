"""在云端 RTX 5080 运行 E017-P0 保守可观察性分类行微调。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_vla.precision.e017_training import run_e017_p0_training


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployable-root", type=Path, required=True)
    parser.add_argument("--source-label-root", type=Path, required=True)
    parser.add_argument("--p0-output", type=Path, required=True)
    parser.add_argument("--e016-config", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--parent-receipt", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    receipt = run_e017_p0_training(
        deployable_root=args.deployable_root,
        source_label_root=args.source_label_root,
        p0_output_root=args.p0_output,
        e016_config_path=args.e016_config,
        parent_checkpoint_path=args.parent_checkpoint,
        parent_receipt_path=args.parent_receipt,
        config_path=args.config,
        output_root=args.output,
        repository_root=args.repository_root,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    if not receipt["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
