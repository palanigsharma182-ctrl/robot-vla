"""构建/准备/运行/验证 E018-P1 G2C-TRAIN/v1。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_vla.precision.e018_p1_g2c_data import _atomic_json
from robot_vla.precision.e018_p1_g2c_training import (
    build_g2c_formal_training_config,
    prepare_g2c_model_val_deployable_view,
    prepare_g2c_model_val_label_view,
    prepare_g2c_train_input_view,
    run_g2c_formal_training,
    verify_g2c_formal_training,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)

    build = phases.add_parser("build-config")
    build.add_argument("--data-root", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)

    prepare = phases.add_parser("prepare-input-views")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--data-root", type=Path, required=True)
    prepare.add_argument("--train-output", type=Path, required=True)
    prepare.add_argument("--model-val-output", type=Path, required=True)
    prepare.add_argument("--label-output", type=Path, required=True)

    train = phases.add_parser("formal-train")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--train-input", type=Path, required=True)
    train.add_argument("--e016-config", type=Path, required=True)
    train.add_argument("--e016-training-output", type=Path, required=True)
    train.add_argument("--repository-root", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--decision-exit-go", action="store_true")

    verify = phases.add_parser("verify")
    verify.add_argument("--config", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.phase == "build-config":
        if args.output.exists():
            raise FileExistsError(f"G2C TRAIN config 已存在: {args.output}")
        result = build_g2c_formal_training_config(args.data_root)
        _atomic_json(args.output, result)
    elif args.phase == "prepare-input-views":
        result = {
            "train": prepare_g2c_train_input_view(
                config_path=args.config,
                data_root=args.data_root,
                output_root=args.train_output,
            ),
            "model_val_deployable": prepare_g2c_model_val_deployable_view(
                config_path=args.config,
                data_root=args.data_root,
                output_root=args.model_val_output,
            ),
            "model_val_privileged": prepare_g2c_model_val_label_view(
                config_path=args.config,
                data_root=args.data_root,
                output_root=args.label_output,
            ),
        }
    elif args.phase == "formal-train":
        result = run_g2c_formal_training(
            config_path=args.config,
            train_input_root=args.train_input,
            e016_config_path=args.e016_config,
            e016_training_output=args.e016_training_output,
            repository_root=args.repository_root,
            output_root=args.output,
            decision_exit_go=args.decision_exit_go,
            resume=args.resume,
        )
    else:
        result = verify_g2c_formal_training(
            config_path=args.config, output_root=args.output
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
