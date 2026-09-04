"""运行 E018-P1 G0 独立 front-camera 仿真可行性实验。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_vla.precision.e018_p1_g0 import run_e018_p1_g0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_e018_p1_g0(
        config_path=args.config,
        repository_root=args.repository_root,
        output_root=args.output,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
