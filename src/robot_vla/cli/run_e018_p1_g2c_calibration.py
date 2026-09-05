"""准备、冻结、校准并验证 E018-P1 G2C selected front provider。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_vla.precision.e018_p1_g2c_calibration import (
    build_g2c_calibration_config,
    build_g2c_calibration_phase_a_completion_marker,
    finalize_g2c_calibration_phase_a_persistence,
    prepare_g2c_calibration_deployable_view,
    prepare_g2c_calibration_privileged_view,
    record_g2c_calibration_phase_a_check_evidence,
    run_g2c_calibration_prediction_freeze,
    run_g2c_calibration_synthetic_gpu_smoke,
    score_calibrate_g2c,
    verify_g2c_calibration_phase_a_persistence,
    verify_g2c_calibration_prediction_freeze,
    verify_g2c_calibration_result,
)
from robot_vla.precision.e018_p1_g2c_data import _atomic_json


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)

    build = phases.add_parser("build-config")
    build.add_argument("--output", type=Path, required=True)

    deployable = phases.add_parser("prepare-deployable-input")
    deployable.add_argument("--config", type=Path, required=True)
    deployable.add_argument("--training-config", type=Path, required=True)
    deployable.add_argument("--data-root", type=Path, required=True)
    deployable.add_argument("--output", type=Path, required=True)
    deployable.add_argument("--decision-exit-go", action="store_true")

    freeze = phases.add_parser("freeze-predictions")
    freeze.add_argument("--config", type=Path, required=True)
    freeze.add_argument("--training-config", type=Path, required=True)
    freeze.add_argument("--training-output", type=Path, required=True)
    freeze.add_argument("--model-val-prediction-freeze", type=Path, required=True)
    freeze.add_argument("--model-val-selection", type=Path, required=True)
    freeze.add_argument("--calibration-deployable-input", type=Path, required=True)
    freeze.add_argument("--repository-root", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--decision-exit-go", action="store_true")

    verify_freeze = phases.add_parser("verify-freeze")
    verify_freeze.add_argument("--config", type=Path, required=True)
    verify_freeze.add_argument("--output", type=Path, required=True)

    persistence = phases.add_parser("verify-phase-a-persistence")
    persistence.add_argument("--config", type=Path, required=True)
    persistence.add_argument("--prediction-freeze", type=Path, required=True)
    persistence.add_argument("--persistence-receipt", type=Path, required=True)
    persistence.add_argument(
        "--expected-persistence-receipt-raw-sha256", required=True
    )

    record_check = phases.add_parser("record-phase-a-check")
    record_check.add_argument("--config", type=Path, required=True)
    record_check.add_argument("--prediction-freeze", type=Path, required=True)
    record_check.add_argument(
        "--check-phase",
        choices=(
            "pre-marker-artifact-check",
            "post-marker-artifact-check",
            "post-marker-completion-marker-check",
        ),
        required=True,
    )
    record_check.add_argument("--artifact-id", required=True)
    record_check.add_argument("--worker-id", required=True)
    record_check.add_argument("--remote-path", required=True)
    record_check.add_argument("--combined-report", type=Path, required=True)
    record_check.add_argument("--rclone-exit-code", type=int, required=True)
    record_check.add_argument("--completion-marker", type=Path)
    record_check.add_argument("--output", type=Path, required=True)
    record_check.add_argument("--decision-exit-go", action="store_true")

    marker = phases.add_parser("build-phase-a-completion-marker")
    marker.add_argument("--config", type=Path, required=True)
    marker.add_argument("--prediction-freeze", type=Path, required=True)
    marker.add_argument("--pre-marker-check-evidence", type=Path, required=True)
    marker.add_argument("--artifact-id", required=True)
    marker.add_argument("--worker-id", required=True)
    marker.add_argument("--remote-path", required=True)
    marker.add_argument("--output", type=Path, required=True)
    marker.add_argument("--decision-exit-go", action="store_true")

    finalize = phases.add_parser("finalize-phase-a-persistence")
    finalize.add_argument("--config", type=Path, required=True)
    finalize.add_argument("--prediction-freeze", type=Path, required=True)
    finalize.add_argument("--control-root", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    finalize.add_argument("--decision-exit-go", action="store_true")

    privileged = phases.add_parser("prepare-privileged-input")
    privileged.add_argument("--config", type=Path, required=True)
    privileged.add_argument("--training-config", type=Path, required=True)
    privileged.add_argument("--data-root", type=Path, required=True)
    privileged.add_argument("--prediction-freeze", type=Path, required=True)
    privileged.add_argument("--repository-root", type=Path, required=True)
    privileged.add_argument("--phase-a-persistence-receipt", type=Path, required=True)
    privileged.add_argument(
        "--expected-phase-a-persistence-receipt-raw-sha256", required=True
    )
    privileged.add_argument("--output", type=Path, required=True)
    privileged.add_argument("--decision-exit-go", action="store_true")

    score = phases.add_parser("score-calibrate")
    score.add_argument("--config", type=Path, required=True)
    score.add_argument("--training-config", type=Path, required=True)
    score.add_argument("--prediction-freeze", type=Path, required=True)
    score.add_argument("--calibration-privileged-input", type=Path, required=True)
    score.add_argument("--repository-root", type=Path, required=True)
    score.add_argument("--phase-a-persistence-receipt", type=Path, required=True)
    score.add_argument(
        "--expected-phase-a-persistence-receipt-raw-sha256", required=True
    )
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--decision-exit-go", action="store_true")

    verify = phases.add_parser("verify")
    verify.add_argument("--config", type=Path, required=True)
    verify.add_argument("--prediction-freeze", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)

    smoke = phases.add_parser("smoke")
    smoke.add_argument("--config", type=Path, required=True)
    smoke.add_argument("--selected-checkpoint", type=Path, required=True)
    smoke.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.phase == "build-config":
        if args.output.exists():
            raise FileExistsError(f"G2C calibration config 已存在: {args.output}")
        result = build_g2c_calibration_config()
        _atomic_json(args.output, result)
    elif args.phase == "prepare-deployable-input":
        result = prepare_g2c_calibration_deployable_view(
            calibration_config_path=args.config,
            training_config_path=args.training_config,
            data_root=args.data_root,
            output_root=args.output,
            decision_exit_go=args.decision_exit_go,
        )
    elif args.phase == "freeze-predictions":
        result = run_g2c_calibration_prediction_freeze(
            calibration_config_path=args.config,
            training_config_path=args.training_config,
            training_output_root=args.training_output,
            model_val_prediction_freeze_root=args.model_val_prediction_freeze,
            model_val_selection_root=args.model_val_selection,
            calibration_deployable_input_root=args.calibration_deployable_input,
            repository_root=args.repository_root,
            output_root=args.output,
            decision_exit_go=args.decision_exit_go,
        )
    elif args.phase == "verify-freeze":
        result = verify_g2c_calibration_prediction_freeze(
            calibration_config_path=args.config,
            output_root=args.output,
        )
    elif args.phase == "verify-phase-a-persistence":
        result = verify_g2c_calibration_phase_a_persistence(
            calibration_config_path=args.config,
            prediction_freeze_root=args.prediction_freeze,
            persistence_receipt_path=args.persistence_receipt,
            expected_receipt_raw_sha256=(
                args.expected_persistence_receipt_raw_sha256
            ),
        )
    elif args.phase == "record-phase-a-check":
        result = record_g2c_calibration_phase_a_check_evidence(
            calibration_config_path=args.config,
            prediction_freeze_root=args.prediction_freeze,
            phase=args.check_phase,
            artifact_id=args.artifact_id,
            worker_id=args.worker_id,
            remote_path=args.remote_path,
            combined_report_path=args.combined_report,
            rclone_exit_code=args.rclone_exit_code,
            output_path=args.output,
            completion_marker_path=args.completion_marker,
            decision_exit_go=args.decision_exit_go,
        )
    elif args.phase == "build-phase-a-completion-marker":
        result = build_g2c_calibration_phase_a_completion_marker(
            calibration_config_path=args.config,
            prediction_freeze_root=args.prediction_freeze,
            pre_marker_check_evidence_path=args.pre_marker_check_evidence,
            artifact_id=args.artifact_id,
            worker_id=args.worker_id,
            remote_path=args.remote_path,
            output_path=args.output,
            decision_exit_go=args.decision_exit_go,
        )
    elif args.phase == "finalize-phase-a-persistence":
        result = finalize_g2c_calibration_phase_a_persistence(
            calibration_config_path=args.config,
            prediction_freeze_root=args.prediction_freeze,
            control_root=args.control_root,
            output_path=args.output,
            decision_exit_go=args.decision_exit_go,
        )
    elif args.phase == "prepare-privileged-input":
        result = prepare_g2c_calibration_privileged_view(
            calibration_config_path=args.config,
            training_config_path=args.training_config,
            data_root=args.data_root,
            prediction_freeze_root=args.prediction_freeze,
            repository_root=args.repository_root,
            phase_a_persistence_receipt_path=(
                args.phase_a_persistence_receipt
            ),
            expected_phase_a_persistence_receipt_raw_sha256=(
                args.expected_phase_a_persistence_receipt_raw_sha256
            ),
            output_root=args.output,
            decision_exit_go=args.decision_exit_go,
        )
    elif args.phase == "score-calibrate":
        result = score_calibrate_g2c(
            calibration_config_path=args.config,
            training_config_path=args.training_config,
            prediction_freeze_root=args.prediction_freeze,
            calibration_privileged_input_root=(
                args.calibration_privileged_input
            ),
            repository_root=args.repository_root,
            phase_a_persistence_receipt_path=(
                args.phase_a_persistence_receipt
            ),
            expected_phase_a_persistence_receipt_raw_sha256=(
                args.expected_phase_a_persistence_receipt_raw_sha256
            ),
            output_root=args.output,
            decision_exit_go=args.decision_exit_go,
        )
    elif args.phase == "verify":
        result = verify_g2c_calibration_result(
            calibration_config_path=args.config,
            prediction_freeze_root=args.prediction_freeze,
            output_root=args.output,
        )
    else:
        result = run_g2c_calibration_synthetic_gpu_smoke(
            calibration_config_path=args.config,
            selected_checkpoint_path=args.selected_checkpoint,
            output_root=args.output,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
