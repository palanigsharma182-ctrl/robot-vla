"""输出 Stage 1 运行目录中验证 loss 最低的实际 periodic Checkpoint。"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from robot_vla.evaluation.checkpoint_selection import select_best_periodic_checkpoint


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    selection = select_best_periodic_checkpoint(args.run).to_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{args.output.name}.",
        suffix=".tmp",
        dir=args.output.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(selection, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps(selection, sort_keys=True), flush=True)


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
