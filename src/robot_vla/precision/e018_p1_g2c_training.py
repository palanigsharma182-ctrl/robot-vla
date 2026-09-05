"""E018-P1 G2C-TRAIN/v1 的隔离训练、预测冻结与标签后评分。

本模块把三个有意不可合并的进程边界编码为独立入口：

``formal train -> deployable-only prediction freeze -> label-only score/select``

正式执行均要求显式 Decision GO。配置构建与 input view 准备是机械性步骤，
不会训练、读取 test、访问 Memory 或产生任何 actuator command。
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from robot_vla.precision.e018_p1_g2a import canonical_sha256, file_sha256
from robot_vla.precision.e018_p1_g2c import (
    G2C_CANDIDATE_EPOCHS,
    G2C_CANDIDATE_IDS,
    G2C_CANDIDATE_INITIALIZATION_SEEDS,
    G2C_SHARED_SAMPLER_SEED,
    _build_supervision,
    _collate_training,
    _zero_warm_start_keypoint_log_variance_rows,
)
from robot_vla.precision.e018_p1_g2c_data import (
    E018_P1_G2C_DATA_RESULT_VERSION,
    G2C_DEPLOYABLE_SCHEMA_VERSION,
    G2C_LABEL_SCHEMA_VERSION,
    G2C_MANIFEST_SCHEMA_VERSION,
    G2C_STATIC_SPLITS,
    G2C_VIEW_ORDER,
    G2CFrontTrainingDataset,
    _atomic_json,
    _atomic_jsonl,
    _read_jsonl,
    _resolve_artifact_file,
)

E018_P1_G2C_FORMAL_TRAIN_CONFIG_VERSION = "e018-p1-g2c-train-development/v1"
E018_P1_G2C_FORMAL_TRAIN_RESULT_VERSION = "e018-p1-g2c-train-result/v1"
E018_P1_G2C_INPUT_VIEW_VERSION = "e018-p1-g2c-input-view/v1"
E018_P1_G2C_PREDICTION_FREEZE_VERSION = "e018-p1-g2c-model-val-freeze/v1"
E018_P1_G2C_SELECTION_RESULT_VERSION = "e018-p1-g2c-model-val-selection/v1"
G2C_DIAGNOSTIC_CONTROL_ID = "CONTROL-E016-EPOCH12"
_FORMAL_PERMISSION_KEYS = {
    "test_array_reads",
    "memory_reads",
    "memory_writes",
    "runtime_camera_actuation",
    "physical_camera_actuation",
    "arm_motion_commands",
    "gripper_close_commands",
    "manipulation_progression",
}
_MODEL_PARENT_KEYS = {
    "e016_config_sha256",
    "e016_checkpoint_sha256",
    "e016_checkpoint_parameter_sha256",
    "e016_checkpoint_provenance_sha256",
    "e016_checkpoint_model_config_sha256",
    "source_training_camera",
    "target_training_camera",
}
_INPUT_INVENTORY_KEYS = {
    "all_inventory_sha256",
    "total_seed_count",
    "total_sample_count",
    "splits",
}
_SPLIT_INVENTORY_KEYS = {
    "split",
    "seed_count",
    "sample_count",
    "deployable_inventory_sha256",
    "privileged_inventory_sha256",
    "paired_inventory_sha256",
}

D038_ACCEPTED_DATA = {
    "source_git_commit": "b84536279fc751e65b9f685d951c4f77043f675c",
    "source_identity_sha256": (
        "f226b1f66c775ae8ff86a2111f0ff9b0f15aaab97155b2aa9f9096752253a39c"
    ),
    "data_config_sha256": (
        "56718c0611fc620ccfb767141d8d0867ea5d03806348396d0a2e201fbff3d5de"
    ),
    "data_identity_sha256": (
        "07919f413224fba797d4c12df25e2d5aec8ded8213e3283a07feed282701cfa3"
    ),
    "data_receipt_raw_sha256": (
        "0bd4c2c6dd008889f9c02bb09e050d65b98d97620acbc8bfa5d225f1ed16e99d"
    ),
    "data_receipt_internal_sha256": (
        "0b52c3f1463087ad04275237c4567e656e698ab1043991b11d6c41d6711aa383"
    ),
    "deployable_manifest_raw_sha256": (
        "5f99d3bc56381926061d61e2f1a07aea6c4655dcdd60b7b743b323c24697dff7"
    ),
    "privileged_manifest_raw_sha256": (
        "08a7e126a176936366f17688eacc6574412901ed171b5af405e8a91f7e93036c"
    ),
}

_EXPECTED_SPLIT_SEEDS = {
    "train": tuple(range(76001, 76401)),
    "model_val": tuple(range(76501, 76601)),
    "calibration": tuple(range(76601, 76651)),
}
_EXPECTED_SPLIT_SAMPLES = {"train": 4400, "model_val": 1100, "calibration": 550}
_SHA256_PATTERN = __import__("re").compile(r"[0-9a-f]{64}")
_LOSS_COMPONENT_NAMES = (
    "loss",
    "heatmap_loss",
    "mask_loss",
    "coordinate_loss",
    "motion_loss",
    "uncertainty_loss",
    "visibility_loss",
    "projection_loss",
)
_LOSS_SHARD_ARRAYS = (
    "seed",
    "sample_index",
    "viewpoint_id",
    "input_sha256",
    "heatmap_logits",
    "mask_logits",
    "decoded_normalized_uv",
    "motion_residual",
    "keypoint_log_variance",
    "motion_log_variance",
    "visibility_logits",
    "projection_validity_logit",
)
_CHECKPOINT_INVENTORY_KEYS = {
    "candidate_id",
    "epoch",
    "relative_path",
    "format_version",
    "checkpoint_sha256",
    "parameter_state_sha256",
    "model_config_sha256",
    "provenance_sha256",
    "provenance",
    "examples_seen",
    "optimizer_steps",
    "training_state_relative_path",
    "training_state_raw_sha256",
    "training_state_identity_sha256",
    "optimizer_state_identity_sha256",
    "scheduler_state_identity_sha256",
    "rng_state_identity_sha256",
    "sampler_state_identity_sha256",
    "active_gpu_elapsed_s_at_checkpoint",
    "budget_timing_identity_sha256",
}
_EPOCH_TRACE_KEYS = {
    "candidate_id",
    "epoch",
    "sample_count",
    "batch_count",
    "sample_order_sha256",
    "sampler_generator_state_before_sha256",
    "sampler_generator_state_after_sha256",
    "loss",
    "maximum_gradient_norm_pre_clip",
    "maximum_gradient_norm_post_clip",
    "learning_rate_after_scheduler_step",
    "examples_seen_total",
    "optimizer_steps_total",
    "parameter_state_sha256",
    "motion_head_parameter_sha256",
    "optimizer_state_identity_sha256",
    "scheduler_state_identity_sha256",
    "rng_state_identity_sha256",
}
_RESUME_STATE_KEYS = {
    "version",
    "candidate_id",
    "config_sha256",
    "source_identity_sha256",
    "completed_epoch",
    "examples_seen",
    "optimizer_steps",
    "model_state",
    "optimizer_state",
    "scheduler_state",
    "rng_state",
    "initialization",
    "epoch_trace",
    "checkpoint_inventory",
    "active_gpu_elapsed_s",
}
_TRAINING_STATE_KEYS = {
    "version",
    "kind",
    "candidate_id",
    "epoch",
    "config_sha256",
    "source_identity_sha256",
    "checkpoint_sha256",
    "checkpoint_parameter_sha256",
    "checkpoint_provenance_sha256",
    "examples_seen",
    "optimizer_steps",
    "active_gpu_elapsed_s_at_checkpoint",
    "gpu_budget_hours",
    "optimizer_state",
    "scheduler_state",
    "rng_state",
    "state_identity_sha256",
}
_BUDGET_STATE_KEYS = {
    "version",
    "candidate_id",
    "config_sha256",
    "source_identity_sha256",
    "last_fully_resumable_epoch",
    "active_attempt_epoch",
    "active_attempt_batch_count",
    "active_gpu_elapsed_s",
    "gpu_budget_hours",
}


class _G2CResumePreflightError(RuntimeError):
    """恢复证据尚未通过时阻止 outer runner 改写任何旧 artifact。"""


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} 必须是 64 位小写 SHA-256")
    return value


def _require_exact_keys(value: Any, expected: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} 必须是 JSON object")
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{name} keys 漂移: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _read_json(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{name} 不存在或是 symlink: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{name} 必须是 JSON object")
    return value


def _read_json_array(path: Path, name: str) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{name} 不存在或是 symlink: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise TypeError(f"{name} 必须是 object array")
    return value


def _verify_exact_regular_file_tree(
    root: Path, *, expected_files: set[str], name: str
) -> int:
    """拒绝额外文件、额外目录、symlink 与特殊文件，并返回全树字节数。"""

    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError(f"{name} root 不存在、不是目录或是 symlink: {root}")
    normalized: set[str] = set()
    for relative in expected_files:
        candidate = Path(relative)
        if (
            not relative
            or candidate.is_absolute()
            or ".." in candidate.parts
            or candidate.as_posix() != relative
        ):
            raise ValueError(f"{name} 白名单含非法 relative path: {relative!r}")
        normalized.add(relative)
    if normalized != expected_files:
        raise AssertionError("G2C exact-tree 白名单归一化漂移")
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise RuntimeError(f"{name} 禁止 symlink: {relative}")
        if path.is_file():
            actual_files.add(relative)
        elif path.is_dir():
            actual_directories.add(relative)
        else:
            raise RuntimeError(f"{name} 禁止特殊文件: {relative}")
    expected_directories = {
        parent.as_posix()
        for relative in expected_files
        for parent in Path(relative).parents
        if parent.as_posix() != "."
    }
    if actual_files != expected_files or actual_directories != expected_directories:
        raise RuntimeError(
            f"{name} 完整文件树不在白名单: "
            f"missing_files={sorted(expected_files - actual_files)}, "
            f"extra_files={sorted(actual_files - expected_files)}, "
            f"missing_dirs={sorted(expected_directories - actual_directories)}, "
            f"extra_dirs={sorted(actual_directories - expected_directories)}"
        )
    return sum((root / relative).stat().st_size for relative in actual_files)


def _assert_unlinked_regular_file_tree(root: Path, *, name: str) -> None:
    """在读取隔离 view 内容前拒绝 symlink、hardlink 与特殊文件。"""

    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError(f"{name} root 不存在、不是目录或是 symlink: {root}")
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise RuntimeError(f"{name} 禁止 symlink: {relative}")
        if path.is_file():
            if path.stat().st_nlink != 1:
                raise RuntimeError(f"{name} 禁止 hardlink: {relative}")
        elif not path.is_dir():
            raise RuntimeError(f"{name} 禁止特殊文件: {relative}")


def _g2c_training_artifact_names(
    checkpoint_inventory: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    """从冻结候选/epoch机械生成 receipt 内必须精确覆盖的 artifact 文件。"""

    expected_pairs = [
        (candidate_id, epoch)
        for candidate_id in G2C_CANDIDATE_IDS
        for epoch in G2C_CANDIDATE_EPOCHS
    ]
    if [
        (item.get("candidate_id"), item.get("epoch"))
        for item in checkpoint_inventory
    ] != expected_pairs:
        raise RuntimeError("G2C training checkpoint inventory candidate/epoch 漂移")
    checkpoint_names: list[str] = []
    companion_names: list[str] = []
    for item in checkpoint_inventory:
        _require_exact_keys(dict(item), _CHECKPOINT_INVENTORY_KEYS, "G2C checkpoint item")
        candidate_id = str(item["candidate_id"])
        epoch = int(item["epoch"])
        expected_checkpoint = f"precision-{candidate_id.lower()}-epoch-{epoch:02d}.pt"
        expected_companion = (
            f"training-state-{candidate_id.lower()}-epoch-{epoch:02d}.pt"
        )
        if (
            item.get("relative_path") != expected_checkpoint
            or item.get("training_state_relative_path") != expected_companion
        ):
            raise RuntimeError("G2C training checkpoint/companion filename 漂移")
        checkpoint_names.append(f"candidates/{candidate_id}/{expected_checkpoint}")
        companion_names.append(f"candidates/{candidate_id}/{expected_companion}")
    return (
        "config_snapshot.json",
        "source_identity.json",
        "train_input_verification.json",
        "training_summary.json",
        *(
            f"candidates/{candidate_id}/initialization.json"
            for candidate_id in G2C_CANDIDATE_IDS
        ),
        *(
            f"candidates/{candidate_id}/epoch_trace.json"
            for candidate_id in G2C_CANDIDATE_IDS
        ),
        *(
            f"candidates/{candidate_id}/checkpoint_inventory.json"
            for candidate_id in G2C_CANDIDATE_IDS
        ),
        *(
            f"candidates/{candidate_id}/resume_state.pt"
            for candidate_id in G2C_CANDIDATE_IDS
        ),
        *(
            f"candidates/{candidate_id}/budget_state.json"
            for candidate_id in G2C_CANDIDATE_IDS
        ),
        *checkpoint_names,
        *companion_names,
    )


def _fsync_path(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path.parent, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_torch_save(value: Mapping[str, Any], path: Path) -> None:
    import torch

    if path.exists():
        raise FileExistsError(f"拒绝覆盖 Torch artifact: {path}")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
        torch.save(dict(value), temporary)
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        temporary = None
        _fsync_path(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _replace_torch_resume_state(value: Mapping[str, Any], path: Path) -> None:
    """只允许原子替换单个明确的 crash-resume 状态，不改写冻结 checkpoint。"""

    import torch

    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
        torch.save(dict(value), temporary)
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        temporary = None
        _fsync_path(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _manifest_inventory(
    deployable_rows: Sequence[Mapping[str, Any]] | None,
    label_rows: Sequence[Mapping[str, Any]] | None,
    *,
    split: str,
    deployable_root: Path | None = None,
    label_root: Path | None = None,
) -> dict[str, Any]:
    """把 ordered manifest 行规范化为能绑定每个输入文件 SHA 的 inventory。"""

    if split not in G2C_STATIC_SPLITS:
        raise ValueError("G2C inventory split 未冻结")
    expected_seeds = _EXPECTED_SPLIT_SEEDS[split]

    def normalize(
        rows: Sequence[Mapping[str, Any]] | None,
        *,
        privileged: bool,
        root: Path | None,
    ) -> list[dict[str, Any]] | None:
        if rows is None:
            return None
        selected = [dict(row) for row in rows if row.get("split") == split]
        if tuple(row.get("seed") for row in selected) != expected_seeds:
            raise RuntimeError(f"G2C {split} manifest seed/order 漂移")
        result: list[dict[str, Any]] = []
        for row in selected:
            expected_keys = {
                "manifest_schema_version",
                "split",
                "seed",
                "sample_count",
                "view_order",
                "schema_version",
                "file",
                "sha256",
                "contains_model_input_rgb"
                if privileged
                else "contains_privileged_labels",
            }
            if privileged:
                expected_keys.update({"source_deployable_file", "source_deployable_sha256"})
            _require_exact_keys(row, expected_keys, f"G2C {split} manifest row")
            expected_schema = (
                G2C_LABEL_SCHEMA_VERSION if privileged else G2C_DEPLOYABLE_SCHEMA_VERSION
            )
            if (
                row["manifest_schema_version"] != G2C_MANIFEST_SCHEMA_VERSION
                or row["schema_version"] != expected_schema
                or row["sample_count"] != len(G2C_VIEW_ORDER)
                or tuple(row["view_order"]) != G2C_VIEW_ORDER
                or (privileged and row["contains_model_input_rgb"] is not False)
                or (not privileged and row["contains_privileged_labels"] is not False)
            ):
                raise RuntimeError(f"G2C {split} manifest schema/role 漂移")
            sha = _require_sha256(row["sha256"], f"G2C {split} bundle sha")
            if root is not None:
                bundle = _resolve_artifact_file(root, str(row["file"]))
                if bundle.is_symlink() or file_sha256(bundle) != sha:
                    raise RuntimeError(f"G2C {split} bundle 文件 SHA 漂移")
            item = {
                "split": split,
                "seed": int(row["seed"]),
                "sample_count": int(row["sample_count"]),
                "view_order": list(row["view_order"]),
                "file": str(row["file"]),
                "sha256": sha,
            }
            if privileged:
                item.update(
                    {
                        "source_deployable_file": str(row["source_deployable_file"]),
                        "source_deployable_sha256": _require_sha256(
                            row["source_deployable_sha256"],
                            f"G2C {split} label source sha",
                        ),
                    }
                )
            result.append(item)
        return result

    deployable = normalize(deployable_rows, privileged=False, root=deployable_root)
    labels = normalize(label_rows, privileged=True, root=label_root)
    if deployable is not None and labels is not None:
        for source, target in zip(deployable, labels, strict=True):
            if (
                source["seed"] != target["seed"]
                or source["file"] != target["source_deployable_file"]
                or source["sha256"] != target["source_deployable_sha256"]
            ):
                raise RuntimeError(f"G2C {split} deployable/label lineage 漂移")
    return {
        "split": split,
        "seed_count": len(expected_seeds),
        "sample_count": _EXPECTED_SPLIT_SAMPLES[split],
        "deployable_inventory_sha256": (
            None if deployable is None else canonical_sha256(deployable)
        ),
        "privileged_inventory_sha256": (
            None if labels is None else canonical_sha256(labels)
        ),
        "paired_inventory_sha256": (
            None
            if deployable is None or labels is None
            else canonical_sha256(
                [
                    {"deployable": source, "privileged": target}
                    for source, target in zip(deployable, labels, strict=True)
                ]
            )
        ),
    }


def g2c_formal_training_protocol() -> dict[str, Any]:
    """D038 后独立 TRAIN/v1 协议；不修改旧 smoke protocol identity。"""

    return {
        "version": E018_P1_G2C_FORMAL_TRAIN_CONFIG_VERSION,
        "candidate_ids": list(G2C_CANDIDATE_IDS),
        "candidate_initialization_seeds": dict(G2C_CANDIDATE_INITIALIZATION_SEEDS),
        "shared_sampler_seed": G2C_SHARED_SAMPLER_SEED,
        "epochs_per_candidate": 20,
        "checkpoint_epochs": list(G2C_CANDIDATE_EPOCHS),
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
            "gradient_clip_norm": 1.0,
        },
        "scheduler": {
            "name": "CosineAnnealingLR",
            "t_max_epochs": 20,
            "eta_min": 1.5e-5,
            "step_unit": "completed-epoch",
        },
        "loader": {
            "batch_size": 32,
            "shuffle": True,
            "num_workers": 0,
            "drop_last": False,
            "spatial_augmentation": False,
        },
        "loss": {
            "heatmap_weight": 1.0,
            "mask_weight": 0.5,
            "mask_dice_weight": 1.0,
            "coordinate_weight": 2.0,
            "motion_weight": 1.0,
            "uncertainty_weight": 0.1,
            "visibility_weight": 1.0,
            "projection_weight": 1.0,
            "keypoint_temperature": 1.0,
            "heatmap_sigma_px": 1.5,
        },
        "precision": "BF16-autocast-with-float32-loss/v1",
        "motion_head_policy": "frozen-zero-shadow-only",
        "model_validation": {
            "batch_size": 32,
            "shuffle": False,
            "sample_count_per_checkpoint": 1100,
            "selection_checkpoint_count": 8,
            "candidate_prediction_ledger_count": 8,
            "candidate_prediction_row_count": 8800,
            "candidate_loss_output_shard_count": 280,
            "diagnostic_control": {
                "control_id": G2C_DIAGNOSTIC_CONTROL_ID,
                "source": "exact-e016-selected-epoch12-role-substitution/v1",
                "prediction_ledger_count": 1,
                "prediction_row_count": 1100,
                "loss_output_shard_count": 0,
                "validation_loss_count": 0,
                "eligible_for_selection": False,
            },
            "total_prediction_ledger_count": 9,
            "total_prediction_row_count": 9900,
            "model_val_unique_deployable_bundle_count": 100,
            "model_val_deployable_bundle_open_count": 900,
            "loss_aggregation": "sum(batch_loss*actual_batch_size)/1100",
            "prediction_before_privileged_label": True,
            "frozen_arrays": list(_LOSS_SHARD_ARRAYS),
        },
        "budgets": {
            "model_epochs": 40,
            "gpu_hours_max": 10.0,
            "artifact_bytes_max": 20 * 1024**3,
        },
        "zero_eligible_policy": "selected-null-protocol-valid-negative/v1",
    }


def build_g2c_formal_training_config(data_root: str | Path) -> dict[str, Any]:
    """从 D038 accepted DATA receipt 机械生成 TRAIN/v1 config。"""

    root = Path(data_root)
    receipt_path = root / "data_receipt.json"
    receipt = _read_json(receipt_path, "G2C DATA receipt")
    unsigned_receipt = dict(receipt)
    receipt_internal = unsigned_receipt.pop("receipt_sha256", None)
    if (
        receipt.get("version") != E018_P1_G2C_DATA_RESULT_VERSION
        or receipt.get("status") != "complete-data-pass"
        or receipt.get("mode") != "full"
        or receipt.get("gate_passed") is not True
        or receipt.get("canonical_data_receipt") is not True
        or receipt.get("checkpoint_write_count") != 0
        or receipt_internal != canonical_sha256(unsigned_receipt)
    ):
        raise RuntimeError("G2C TRAIN/v1 只能绑定通过的 canonical full DATA receipt")
    actual_accepted = {
        "source_identity_sha256": receipt.get("source_identity_sha256"),
        "data_config_sha256": receipt.get("config_sha256"),
        "data_identity_sha256": receipt.get("data_identity_sha256"),
        "data_receipt_raw_sha256": file_sha256(receipt_path),
        "data_receipt_internal_sha256": receipt_internal,
        "deployable_manifest_raw_sha256": file_sha256(
            root / "deployable" / "manifest.jsonl"
        ),
        "privileged_manifest_raw_sha256": file_sha256(
            root / "privileged_labels" / "manifest.jsonl"
        ),
    }
    source_identity = _read_json(root / "source_identity.json", "G2C source identity")
    actual_accepted["source_git_commit"] = source_identity.get("git_commit")
    if actual_accepted != D038_ACCEPTED_DATA:
        raise RuntimeError("G2C DATA identity 不是 D038 accepted canonical parent")
    artifacts = receipt.get("artifact_sha256")
    if not isinstance(artifacts, dict) or any(
        artifacts.get(name) != actual_accepted[expected]
        for name, expected in (
            ("deployable/manifest.jsonl", "deployable_manifest_raw_sha256"),
            ("privileged_labels/manifest.jsonl", "privileged_manifest_raw_sha256"),
        )
    ):
        raise RuntimeError("G2C DATA receipt 未绑定 canonical manifests")
    config_snapshot = _read_json(root / "config_snapshot.json", "G2C DATA config snapshot")
    if canonical_sha256(config_snapshot) != receipt["config_sha256"]:
        raise RuntimeError("G2C DATA config snapshot identity 漂移")
    deployable_rows = _read_jsonl(
        root / "deployable" / "manifest.jsonl", "G2C deployable manifest"
    )
    label_rows = _read_jsonl(
        root / "privileged_labels" / "manifest.jsonl", "G2C privileged manifest"
    )
    inventories = {
        split: _manifest_inventory(
            deployable_rows,
            label_rows,
            split=split,
            deployable_root=root / "deployable",
            label_root=root / "privileged_labels",
        )
        for split in G2C_STATIC_SPLITS
    }
    all_inventory_sha256 = canonical_sha256(
        [{"split": split, **inventories[split]} for split in G2C_STATIC_SPLITS]
    )
    parents = config_snapshot.get("parents")
    if not isinstance(parents, dict):
        raise TypeError("G2C DATA config 缺少 E016 parent")
    model_parent = {
        name: parents[name]
        for name in (
            "e016_config_sha256",
            "e016_checkpoint_sha256",
            "e016_checkpoint_parameter_sha256",
            "e016_checkpoint_provenance_sha256",
            "e016_checkpoint_model_config_sha256",
            "source_training_camera",
            "target_training_camera",
        )
    }
    payload = {
        "version": E018_P1_G2C_FORMAL_TRAIN_CONFIG_VERSION,
        "status": "frozen-pre-formal-train-awaiting-source-r2-go/v1",
        "decision": {
            "data_acceptance": "D038",
            "formal_train_execution": "HOLD-until-new-source-r2-go",
            "model_val_execution": "HOLD-until-separate-go",
        },
        "data_parent": dict(D038_ACCEPTED_DATA),
        "input_inventories": {
            "all_inventory_sha256": all_inventory_sha256,
            "total_seed_count": 550,
            "total_sample_count": 6050,
            "splits": inventories,
        },
        "model_parent": model_parent,
        "protocol": g2c_formal_training_protocol(),
        "permissions": {
            "test_array_reads": 0,
            "memory_reads": 0,
            "memory_writes": 0,
            "runtime_camera_actuation": 0,
            "physical_camera_actuation": 0,
            "arm_motion_commands": 0,
            "gripper_close_commands": 0,
            "manipulation_progression": 0,
        },
    }
    payload["config_sha256"] = canonical_sha256(payload)
    return payload


def load_g2c_formal_training_config(path: str | Path) -> dict[str, Any]:
    config = _read_json(Path(path), "G2C TRAIN config")
    internal = config.get("config_sha256")
    unsigned = dict(config)
    unsigned.pop("config_sha256", None)
    if internal != canonical_sha256(unsigned):
        raise RuntimeError("G2C TRAIN config internal SHA-256 漂移")
    _require_exact_keys(
        config,
        {
            "version",
            "status",
            "decision",
            "data_parent",
            "input_inventories",
            "model_parent",
            "protocol",
            "permissions",
            "config_sha256",
        },
        "G2C TRAIN config",
    )
    permissions = _require_exact_keys(
        config["permissions"], _FORMAL_PERMISSION_KEYS, "G2C TRAIN permissions"
    )
    model_parent = _require_exact_keys(
        config["model_parent"], _MODEL_PARENT_KEYS, "G2C TRAIN model parent"
    )
    inventories = _require_exact_keys(
        config["input_inventories"],
        _INPUT_INVENTORY_KEYS,
        "G2C TRAIN input inventories",
    )
    if (
        config["version"] != E018_P1_G2C_FORMAL_TRAIN_CONFIG_VERSION
        or config["status"] != "frozen-pre-formal-train-awaiting-source-r2-go/v1"
        or config["data_parent"] != D038_ACCEPTED_DATA
        or config["protocol"] != g2c_formal_training_protocol()
        or config["decision"]
        != {
            "data_acceptance": "D038",
            "formal_train_execution": "HOLD-until-new-source-r2-go",
            "model_val_execution": "HOLD-until-separate-go",
        }
        or any(type(value) is not int or value != 0 for value in permissions.values())
    ):
        raise RuntimeError("G2C TRAIN config protocol/permission 漂移")
    for name in (
        "e016_config_sha256",
        "e016_checkpoint_sha256",
        "e016_checkpoint_parameter_sha256",
        "e016_checkpoint_provenance_sha256",
        "e016_checkpoint_model_config_sha256",
    ):
        _require_sha256(model_parent[name], f"G2C model_parent.{name}")
    if (
        model_parent["source_training_camera"] != "hand_camera"
        or model_parent["target_training_camera"] != "base_camera"
    ):
        raise RuntimeError("G2C TRAIN model parent camera UID 漂移")
    splits = _require_exact_keys(
        inventories["splits"], set(G2C_STATIC_SPLITS), "G2C TRAIN inventory splits"
    )
    if (
        inventories.get("total_seed_count") != 550
        or inventories.get("total_sample_count") != 6050
    ):
        raise RuntimeError("G2C TRAIN config inventory count 漂移")
    _require_sha256(
        inventories["all_inventory_sha256"], "G2C all_inventory_sha256"
    )
    for split in G2C_STATIC_SPLITS:
        item = _require_exact_keys(
            splits[split], _SPLIT_INVENTORY_KEYS, f"G2C TRAIN {split} inventory"
        )
        if (
            item.get("split") != split
            or item.get("seed_count") != len(_EXPECTED_SPLIT_SEEDS[split])
            or item.get("sample_count") != _EXPECTED_SPLIT_SAMPLES[split]
        ):
            raise RuntimeError(f"G2C TRAIN {split} inventory 漂移")
        for name in (
            "deployable_inventory_sha256",
            "privileged_inventory_sha256",
            "paired_inventory_sha256",
        ):
            _require_sha256(item.get(name), f"G2C {split}.{name}")
    expected_all = canonical_sha256(
        [
            {"split": split, **splits[split]}
            for split in G2C_STATIC_SPLITS
        ]
    )
    if inventories.get("all_inventory_sha256") != expected_all:
        raise RuntimeError("G2C TRAIN all inventory SHA 漂移")
    return config


def _copy_input_bundle(source: Path, target: Path) -> None:
    """复制到独立 inode；禁止 hardlink/symlink 污染 canonical DATA。"""

    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"G2C input bundle 不存在或是 symlink: {source}")
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
        shutil.copyfile(source, temporary)
        if source.stat().st_ino == temporary.stat().st_ino:
            raise RuntimeError("G2C input view copy 意外共享 canonical DATA inode")
        os.chmod(temporary, 0o400)
        _fsync_path(temporary)
        os.replace(temporary, target)
        temporary = None
        _fsync_path(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _prepare_g2c_input_view(
    *,
    config_path: str | Path,
    data_root: str | Path,
    output_root: str | Path,
    role: str,
) -> dict[str, Any]:
    config = load_g2c_formal_training_config(config_path)
    role_spec = {
        "train-paired": ("train", True, True),
        "model-val-deployable": ("model_val", True, False),
        "model-val-privileged": ("model_val", False, True),
    }
    if role not in role_spec:
        raise ValueError("G2C input view role 未冻结")
    split, include_deployable, include_labels = role_spec[role]
    source = Path(data_root)
    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"G2C input view 已存在，拒绝覆盖: {output}")
    receipt_path = source / "data_receipt.json"
    if file_sha256(receipt_path) != config["data_parent"]["data_receipt_raw_sha256"]:
        raise RuntimeError("G2C input view DATA receipt 漂移")
    deployable_manifest = source / "deployable" / "manifest.jsonl"
    label_manifest = source / "privileged_labels" / "manifest.jsonl"
    if (
        file_sha256(deployable_manifest)
        != config["data_parent"]["deployable_manifest_raw_sha256"]
        or file_sha256(label_manifest)
        != config["data_parent"]["privileged_manifest_raw_sha256"]
    ):
        raise RuntimeError("G2C input view source manifest 漂移")
    deployable_all = _read_jsonl(deployable_manifest, "G2C deployable manifest")
    label_all = _read_jsonl(label_manifest, "G2C privileged manifest")
    deployable_rows = [row for row in deployable_all if row.get("split") == split]
    label_rows = [row for row in label_all if row.get("split") == split]
    inventory = _manifest_inventory(
        deployable_rows if include_deployable else None,
        label_rows if include_labels else None,
        split=split,
        deployable_root=source / "deployable" if include_deployable else None,
        label_root=source / "privileged_labels" if include_labels else None,
    )
    expected = config["input_inventories"]["splits"][split]
    if include_deployable and (
        inventory["deployable_inventory_sha256"]
        != expected["deployable_inventory_sha256"]
    ):
        raise RuntimeError("G2C deployable input inventory 漂移")
    if include_labels and (
        inventory["privileged_inventory_sha256"]
        != expected["privileged_inventory_sha256"]
    ):
        raise RuntimeError("G2C privileged input inventory 漂移")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    copied: list[dict[str, Any]] = []
    if include_deployable:
        _atomic_jsonl(output / "deployable" / "manifest.jsonl", deployable_rows)
        for row in deployable_rows:
            source_path = _resolve_artifact_file(source / "deployable", str(row["file"]))
            target = output / "deployable" / str(row["file"])
            _copy_input_bundle(source_path, target)
            copied.append({"role": "deployable", "file": str(row["file"]), "sha256": row["sha256"]})
    if include_labels:
        _atomic_jsonl(output / "privileged_labels" / "manifest.jsonl", label_rows)
        for row in label_rows:
            source_path = _resolve_artifact_file(
                source / "privileged_labels", str(row["file"])
            )
            target = output / "privileged_labels" / str(row["file"])
            _copy_input_bundle(source_path, target)
            copied.append({"role": "privileged", "file": str(row["file"]), "sha256": row["sha256"]})
    receipt = {
        "version": E018_P1_G2C_INPUT_VIEW_VERSION,
        "status": "complete-input-view-pass",
        "role": role,
        "split": split,
        "config_sha256": config["config_sha256"],
        "data_identity_sha256": config["data_parent"]["data_identity_sha256"],
        "seed_count": len(_EXPECTED_SPLIT_SEEDS[split]),
        "sample_count": _EXPECTED_SPLIT_SAMPLES[split],
        "deployable_included": include_deployable,
        "privileged_included": include_labels,
        "inventory": inventory,
        "copied_file_inventory_sha256": canonical_sha256(copied),
        "forbidden_split_names": [name for name in G2C_STATIC_SPLITS if name != split],
        "test_array_read_count": 0,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    _atomic_json(output / "input_view_receipt.json", receipt)
    return validate_g2c_input_view(
        config_path=config_path, input_root=output, expected_role=role
    )


def prepare_g2c_train_input_view(
    *, config_path: str | Path, data_root: str | Path, output_root: str | Path
) -> dict[str, Any]:
    return _prepare_g2c_input_view(
        config_path=config_path,
        data_root=data_root,
        output_root=output_root,
        role="train-paired",
    )


def prepare_g2c_model_val_deployable_view(
    *, config_path: str | Path, data_root: str | Path, output_root: str | Path
) -> dict[str, Any]:
    return _prepare_g2c_input_view(
        config_path=config_path,
        data_root=data_root,
        output_root=output_root,
        role="model-val-deployable",
    )


def prepare_g2c_model_val_label_view(
    *, config_path: str | Path, data_root: str | Path, output_root: str | Path
) -> dict[str, Any]:
    return _prepare_g2c_input_view(
        config_path=config_path,
        data_root=data_root,
        output_root=output_root,
        role="model-val-privileged",
    )


def validate_g2c_input_view(
    *,
    config_path: str | Path,
    input_root: str | Path,
    expected_role: str,
    verify_bundle_bytes: bool = True,
) -> dict[str, Any]:
    if not isinstance(verify_bundle_bytes, bool):
        raise TypeError("verify_bundle_bytes 必须是 bool")
    config = load_g2c_formal_training_config(config_path)
    root = Path(input_root)
    _assert_unlinked_regular_file_tree(root, name="G2C input view")
    receipt_path = root / "input_view_receipt.json"
    receipt = _read_json(receipt_path, "G2C input view receipt")
    internal = receipt.get("receipt_sha256")
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256", None)
    role_spec = {
        "train-paired": ("train", True, True),
        "model-val-deployable": ("model_val", True, False),
        "model-val-privileged": ("model_val", False, True),
    }
    if expected_role not in role_spec:
        raise ValueError("G2C expected input role 未冻结")
    split, include_deployable, include_labels = role_spec[expected_role]
    if (
        internal != canonical_sha256(unsigned)
        or receipt.get("version") != E018_P1_G2C_INPUT_VIEW_VERSION
        or receipt.get("status") != "complete-input-view-pass"
        or receipt.get("role") != expected_role
        or receipt.get("split") != split
        or receipt.get("config_sha256") != config["config_sha256"]
        or receipt.get("data_identity_sha256")
        != config["data_parent"]["data_identity_sha256"]
        or receipt.get("deployable_included") is not include_deployable
        or receipt.get("privileged_included") is not include_labels
        or receipt.get("test_array_read_count") != 0
    ):
        raise RuntimeError("G2C input view receipt identity/role 漂移")
    if include_deployable != (root / "deployable").is_dir() or include_labels != (
        root / "privileged_labels"
    ).is_dir():
        raise RuntimeError("G2C input view 可达目录与角色不一致")
    deployable_rows = (
        _read_jsonl(root / "deployable" / "manifest.jsonl", "G2C input deployable manifest")
        if include_deployable
        else None
    )
    label_rows = (
        _read_jsonl(
            root / "privileged_labels" / "manifest.jsonl", "G2C input label manifest"
        )
        if include_labels
        else None
    )
    inventory = _manifest_inventory(
        deployable_rows,
        label_rows,
        split=split,
        deployable_root=(
            root / "deployable" if include_deployable and verify_bundle_bytes else None
        ),
        label_root=(
            root / "privileged_labels" if include_labels and verify_bundle_bytes else None
        ),
    )
    if inventory != receipt.get("inventory"):
        raise RuntimeError("G2C input view inventory/receipt 漂移")
    expected = config["input_inventories"]["splits"][split]
    for name, included in (
        ("deployable_inventory_sha256", include_deployable),
        ("privileged_inventory_sha256", include_labels),
    ):
        if included and inventory[name] != expected[name]:
            raise RuntimeError(f"G2C input view {name} 与 config 漂移")
    expected_files = {"input_view_receipt.json"}
    copied = []
    if deployable_rows is not None:
        expected_files.add("deployable/manifest.jsonl")
        expected_files.update(f"deployable/{row['file']}" for row in deployable_rows)
        copied.extend(
            {
                "role": "deployable",
                "file": str(row["file"]),
                "sha256": str(row["sha256"]),
            }
            for row in deployable_rows
        )
    if label_rows is not None:
        expected_files.add("privileged_labels/manifest.jsonl")
        expected_files.update(f"privileged_labels/{row['file']}" for row in label_rows)
        copied.extend(
            {
                "role": "privileged",
                "file": str(row["file"]),
                "sha256": str(row["sha256"]),
            }
            for row in label_rows
        )
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise RuntimeError(f"G2C input view 禁止 symlink: {relative}")
        if path.is_file():
            actual_files.add(relative)
        elif path.is_dir():
            actual_directories.add(relative)
        else:
            raise RuntimeError(f"G2C input view 禁止特殊文件: {relative}")
    expected_directories = {
        parent.as_posix()
        for name in expected_files
        for parent in Path(name).parents
        if parent.as_posix() != "."
    }
    if actual_files != expected_files or actual_directories != expected_directories:
        raise RuntimeError("G2C input view 完整文件树不在角色白名单")
    if receipt.get("copied_file_inventory_sha256") != canonical_sha256(copied):
        raise RuntimeError("G2C input view copied inventory 漂移")
    result = {
        "version": E018_P1_G2C_INPUT_VIEW_VERSION,
        "verified": True,
        "role": expected_role,
        "split": split,
        "seed_count": len(_EXPECTED_SPLIT_SEEDS[split]),
        "sample_count": _EXPECTED_SPLIT_SAMPLES[split],
        "bundle_bytes_verified": verify_bundle_bytes,
        "receipt_raw_sha256": file_sha256(receipt_path),
        "receipt_internal_sha256": internal,
        "inventory": inventory,
    }
    result["verification_sha256"] = canonical_sha256(result)
    return result


def _git_source_identity(repository_root: Path) -> dict[str, Any]:
    from robot_vla.precision.training import source_tree_sha256

    safe = str(repository_root.resolve())
    git = ("git", "-c", f"safe.directory={safe}")
    status = subprocess.run(
        [*git, "status", "--porcelain", "--untracked-files=all"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if status:
        raise RuntimeError("G2C formal TRAIN 要求 exact-clean Git worktree")
    commit = subprocess.run(
        [*git, "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    identity = {
        "git_commit": commit,
        "source_tree_sha256": source_tree_sha256(repository_root),
    }
    identity["identity_sha256"] = canonical_sha256(identity)
    return identity


def _tensor_sha256(value: Any) -> str:
    import torch

    if not isinstance(value, torch.Tensor):
        raise TypeError("G2C tensor identity 只接受 Tensor")
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _capture_training_rng_state(sampler_generator: Any) -> dict[str, Any]:
    import torch

    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].copy()),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
        "sampler_generator": sampler_generator.get_state(),
    }


def _restore_training_rng_state(state: Mapping[str, Any], sampler_generator: Any) -> None:
    import torch

    required = {"python", "numpy", "torch_cpu", "torch_cuda", "sampler_generator"}
    if set(state) != required:
        raise ValueError("G2C resume RNG state keys 漂移")
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            numpy_state["keys"].cpu().numpy(),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_states = state["torch_cuda"]
    if len(cuda_states) != torch.cuda.device_count():
        raise RuntimeError("G2C resume CUDA RNG state 数量漂移")
    torch.cuda.set_rng_state_all([item.cpu() for item in cuda_states])
    sampler_generator.set_state(state["sampler_generator"].cpu())


def _rng_state_sha256(state: Mapping[str, Any]) -> str:
    numpy_state = state["numpy"]
    return canonical_sha256(
        {
            "python": state["python"],
            "numpy": {
                "bit_generator": numpy_state["bit_generator"],
                "keys_sha256": _tensor_sha256(numpy_state["keys"]),
                "position": numpy_state["position"],
                "has_gauss": numpy_state["has_gauss"],
                "cached_gaussian": numpy_state["cached_gaussian"],
            },
            "torch_cpu_sha256": _tensor_sha256(state["torch_cpu"]),
            "torch_cuda_sha256": [_tensor_sha256(item) for item in state["torch_cuda"]],
            "sampler_generator_sha256": _tensor_sha256(state["sampler_generator"]),
        }
    )


def _state_identity_value(value: Any) -> Any:
    """把 optimizer/scheduler/RNG state 变成稳定、无原始 tensor 的身份树。"""

    try:
        import torch
    except ModuleNotFoundError:
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        return {
            "kind": "tensor",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": _tensor_sha256(value),
        }
    if isinstance(value, Mapping):
        # Optimizer state 通常以 int parameter id 为 key。不能把 key 直接 str()，
        # 否则整数 1 与字符串 "1" 会在身份树中碰撞并被静默覆盖。
        items = []
        for key, child in value.items():
            if isinstance(key, bool):
                key_identity = {"type": "bool", "value": key}
            elif isinstance(key, int):
                key_identity = {"type": "int", "value": key}
            elif isinstance(key, str):
                key_identity = {"type": "str", "value": key}
            else:
                raise TypeError(
                    "G2C state identity mapping key 只支持 str/int/bool: "
                    f"{type(key).__name__}"
                )
            items.append(
                {
                    "key": key_identity,
                    "value": _state_identity_value(child),
                }
            )
        items.sort(key=lambda item: canonical_sha256(item["key"]))
        return {"kind": "mapping", "items": items}
    if isinstance(value, (tuple, list)):
        return [_state_identity_value(child) for child in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"G2C state identity 不支持类型: {type(value).__name__}")


def _state_identity_sha256(value: Any) -> str:
    return canonical_sha256(_state_identity_value(value))


def _ensure_checkpoint_training_state(
    *,
    path: Path,
    candidate_id: str,
    epoch: int,
    checkpoint: Mapping[str, Any],
    config: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    examples_seen: int,
    optimizer_steps: int,
    optimizer_state: Mapping[str, Any],
    scheduler_state: Mapping[str, Any],
    rng_state: Mapping[str, Any],
    active_gpu_elapsed_s: float,
) -> dict[str, Any]:
    import torch

    unsigned = {
        "version": E018_P1_G2C_FORMAL_TRAIN_RESULT_VERSION,
        "kind": "immutable-checkpoint-training-state/v1",
        "candidate_id": candidate_id,
        "epoch": epoch,
        "config_sha256": config["config_sha256"],
        "source_identity_sha256": source_identity["identity_sha256"],
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "checkpoint_parameter_sha256": checkpoint["parameter_state_sha256"],
        "checkpoint_provenance_sha256": checkpoint["provenance_sha256"],
        "examples_seen": examples_seen,
        "optimizer_steps": optimizer_steps,
        "active_gpu_elapsed_s_at_checkpoint": active_gpu_elapsed_s,
        "gpu_budget_hours": config["protocol"]["budgets"]["gpu_hours_max"],
        "optimizer_state": dict(optimizer_state),
        "scheduler_state": dict(scheduler_state),
        "rng_state": dict(rng_state),
    }
    identity = _state_identity_sha256(unsigned)
    payload = {**unsigned, "state_identity_sha256": identity}
    if path.exists():
        existing = torch.load(path, map_location="cpu", weights_only=True)
        existing_unsigned = {
            name: value
            for name, value in existing.items()
            if name != "state_identity_sha256"
        }
        deterministic_names = set(unsigned) - {"active_gpu_elapsed_s_at_checkpoint"}
        if (
            not isinstance(existing, dict)
            or _state_identity_sha256(existing_unsigned)
            != existing.get("state_identity_sha256")
            or any(
                _state_identity_sha256(existing_unsigned[name])
                != _state_identity_sha256(unsigned[name])
                for name in deterministic_names
            )
            or not 0.0
            <= float(existing.get("active_gpu_elapsed_s_at_checkpoint", -1.0))
            <= active_gpu_elapsed_s
        ):
            raise RuntimeError("G2C immutable checkpoint training-state 漂移")
        payload = existing
        identity = existing["state_identity_sha256"]
    else:
        _atomic_torch_save(payload, path)
    return {
        "training_state_relative_path": path.name,
        "training_state_raw_sha256": file_sha256(path),
        "training_state_identity_sha256": identity,
        "optimizer_state_identity_sha256": _state_identity_sha256(
            payload["optimizer_state"]
        ),
        "scheduler_state_identity_sha256": _state_identity_sha256(
            payload["scheduler_state"]
        ),
        "rng_state_identity_sha256": _rng_state_sha256(payload["rng_state"]),
        "sampler_state_identity_sha256": _tensor_sha256(
            payload["rng_state"]["sampler_generator"]
        ),
        "active_gpu_elapsed_s_at_checkpoint": payload[
            "active_gpu_elapsed_s_at_checkpoint"
        ],
        "budget_timing_identity_sha256": canonical_sha256(
            {
                "active_gpu_elapsed_s_at_checkpoint": payload[
                    "active_gpu_elapsed_s_at_checkpoint"
                ],
                "gpu_budget_hours": payload["gpu_budget_hours"],
            }
        ),
    }


def _seed_training(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def remaining_g2c_active_gpu_budget_seconds(
    *, budget_hours: float, persisted_active_elapsed_s: Sequence[float]
) -> tuple[float, float]:
    """用所有持久化计时高水位计算剩余额度；resume 不会重置预算。"""

    if not math.isfinite(budget_hours) or budget_hours <= 0.0:
        raise ValueError("G2C GPU budget hours 必须是有限正数")
    values = [float(value) for value in persisted_active_elapsed_s]
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("G2C persisted active GPU elapsed 必须有限非负")
    consumed = max(values, default=0.0)
    remaining = budget_hours * 3600.0 - consumed
    if remaining <= 0.0:
        raise TimeoutError("G2C formal TRAIN 累计 active GPU budget 已耗尽")
    return consumed, remaining


def _load_formal_candidate_model(
    *,
    candidate_id: str,
    config: Mapping[str, Any],
    e016_config_path: Path,
    e016_training_output: Path,
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    from robot_vla.precision.checkpoint import (
        PrecisionCheckpointRole,
        load_precision_checkpoint,
        precision_parameter_state_sha256,
    )
    from robot_vla.precision.e016_pretraining import new_e016_frozen_motion_model
    from robot_vla.precision.e016_training import load_e016_p1_config

    if candidate_id not in G2C_CANDIDATE_IDS:
        raise ValueError("G2C formal candidate 必须是 W-KV0/S")
    seed = G2C_CANDIDATE_INITIALIZATION_SEEDS[candidate_id]
    _seed_training(seed)
    parent = config["model_parent"]
    e016 = load_e016_p1_config(e016_config_path)
    if e016.sha256 != parent["e016_config_sha256"]:
        raise RuntimeError("G2C formal E016 config identity 漂移")
    if candidate_id == "W-KV0":
        loaded = load_precision_checkpoint(
            e016_training_output / "precision-formal.pt",
            expected_checkpoint_sha256=parent["e016_checkpoint_sha256"],
            expected_provenance_sha256=parent["e016_checkpoint_provenance_sha256"],
            expected_role=PrecisionCheckpointRole.FORMAL_TRAINING,
        )
        if (
            loaded.receipt.parameter_state_sha256
            != parent["e016_checkpoint_parameter_sha256"]
            or loaded.receipt.model_config_sha256
            != parent["e016_checkpoint_model_config_sha256"]
        ):
            raise RuntimeError("G2C W-KV0 E016 parameter/model-config identity 漂移")
        model = loaded.model.to(device)
        reset = _zero_warm_start_keypoint_log_variance_rows(model)
        if reset["parameter_sha256_before"] != loaded.receipt.parameter_state_sha256:
            raise RuntimeError("G2C W-KV0 reset parent identity 漂移")
        initialization = {
            "kind": "e016-selected-epoch12-warm-start-keypoint-variance-zero",
            "source_checkpoint_sha256": loaded.receipt.checkpoint_sha256,
            "source_parameter_sha256": loaded.receipt.parameter_state_sha256,
            "source_provenance_sha256": loaded.receipt.provenance_sha256,
            "model_config_sha256": loaded.receipt.model_config_sha256,
            "keypoint_logvariance_reset": reset,
        }
    else:
        model, _ = new_e016_frozen_motion_model(device)
        initialization = {
            "kind": "random",
            "source_checkpoint_sha256": None,
            "source_parameter_sha256": None,
            "source_provenance_sha256": None,
            "model_config_sha256": canonical_sha256(model.config),
            "keypoint_logvariance_reset": None,
        }
        if initialization["model_config_sha256"] != parent[
            "e016_checkpoint_model_config_sha256"
        ]:
            raise RuntimeError("G2C S model config 与 E016 architecture 漂移")
    model.motion_head.requires_grad_(False)
    initialization.update(
        {
            "candidate_id": candidate_id,
            "initialization_seed": seed,
            "shared_sampler_seed": G2C_SHARED_SAMPLER_SEED,
            "initial_parameter_sha256": precision_parameter_state_sha256(
                model.state_dict()
            ),
            "initial_motion_head_parameter_sha256": (
                precision_parameter_state_sha256(model.motion_head.state_dict())
            ),
        }
    )
    return model, initialization


def _sample_identity(sample: Mapping[str, Any]) -> dict[str, Any]:
    audit = sample["audit"]
    return {
        "seed": int(audit["seed"]),
        "sample_index": int(audit["sample_index"]),
        "viewpoint_id": str(audit["viewpoint_id"]),
        "input_sha256": str(audit["input_sha256"]),
    }


def _ensure_formal_epoch_checkpoint(
    *,
    path: Path,
    model: Any,
    candidate_id: str,
    epoch: int,
    examples_seen: int,
    optimizer_steps: int,
    config: Mapping[str, Any],
    source_identity: Mapping[str, Any],
) -> dict[str, Any]:
    from robot_vla.precision.checkpoint import (
        PrecisionCheckpointProvenance,
        PrecisionCheckpointRole,
        load_precision_checkpoint,
        precision_parameter_state_sha256,
        save_precision_checkpoint,
    )

    provenance = PrecisionCheckpointProvenance(
        role=PrecisionCheckpointRole.FORMAL_TRAINING,
        data_identity_sha256=config["data_parent"]["data_identity_sha256"],
        training_config_sha256=config["config_sha256"],
        source_tree_sha256=source_identity["source_tree_sha256"],
        seed=G2C_CANDIDATE_INITIALIZATION_SEEDS[candidate_id],
        examples_seen=examples_seen,
        optimizer_steps=optimizer_steps,
    )
    current_parameter_sha = precision_parameter_state_sha256(model.state_dict())
    if path.exists():
        checkpoint_sha = file_sha256(path)
        loaded = load_precision_checkpoint(
            path,
            expected_checkpoint_sha256=checkpoint_sha,
            expected_provenance_sha256=provenance.sha256,
            expected_role=PrecisionCheckpointRole.FORMAL_TRAINING,
        )
        receipt = loaded.receipt
        if receipt.parameter_state_sha256 != current_parameter_sha:
            raise RuntimeError("G2C resumed checkpoint 与重算 epoch parameter 漂移")
    else:
        receipt = save_precision_checkpoint(path, model, provenance)
        _fsync_path(path)
    return {
        "candidate_id": candidate_id,
        "epoch": epoch,
        "relative_path": path.name,
        **receipt.to_dict(),
        "provenance": provenance.to_dict(),
        "provenance_sha256": provenance.sha256,
        "examples_seen": examples_seen,
        "optimizer_steps": optimizer_steps,
    }


def _active_gpu_elapsed_s(
    *, active_before_process_s: float, process_started_monotonic_s: float
) -> float:
    elapsed = active_before_process_s + (
        time.monotonic() - process_started_monotonic_s
    )
    if not math.isfinite(elapsed) or elapsed < active_before_process_s:
        raise RuntimeError("G2C active GPU monotonic 计时漂移")
    return elapsed


def _write_candidate_budget_state(
    path: Path,
    *,
    candidate_id: str,
    config: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    last_fully_resumable_epoch: int,
    active_attempt_epoch: int | None,
    active_attempt_batch_count: int,
    active_gpu_elapsed_s: float,
) -> None:
    _atomic_json(
        path,
        {
            "version": E018_P1_G2C_FORMAL_TRAIN_RESULT_VERSION,
            "candidate_id": candidate_id,
            "config_sha256": config["config_sha256"],
            "source_identity_sha256": source_identity["identity_sha256"],
            "last_fully_resumable_epoch": last_fully_resumable_epoch,
            "active_attempt_epoch": active_attempt_epoch,
            "active_attempt_batch_count": active_attempt_batch_count,
            "active_gpu_elapsed_s": active_gpu_elapsed_s,
            "gpu_budget_hours": config["protocol"]["budgets"]["gpu_hours_max"],
        },
    )


def _validate_zero_epoch_restart_candidate(
    *,
    candidate_root: Path,
    candidate_id: str,
    config: Mapping[str, Any],
    source_identity: Mapping[str, Any],
) -> dict[str, Any] | None:
    """验证没有首个 epoch resume state 时可安全从相同初始化重跑。"""

    budget_path = candidate_root / "budget_state.json"
    initialization_path = candidate_root / "initialization.json"
    expected_files = {"budget_state.json"}
    if initialization_path.is_file() and not initialization_path.is_symlink():
        expected_files.add("initialization.json")
    _verify_exact_regular_file_tree(
        candidate_root,
        expected_files=expected_files,
        name=f"G2C {candidate_id} zero-epoch restart",
    )
    budget_state = _read_json(
        budget_path, f"G2C {candidate_id} zero-epoch restart budget state"
    )
    _require_exact_keys(
        budget_state,
        _BUDGET_STATE_KEYS,
        f"G2C {candidate_id} zero-epoch restart budget state",
    )
    batch_count = budget_state.get("active_attempt_batch_count")
    elapsed = float(budget_state.get("active_gpu_elapsed_s", math.nan))
    budget_hours = config["protocol"]["budgets"]["gpu_hours_max"]
    if (
        budget_state.get("version")
        != E018_P1_G2C_FORMAL_TRAIN_RESULT_VERSION
        or budget_state.get("candidate_id") != candidate_id
        or budget_state.get("config_sha256") != config["config_sha256"]
        or budget_state.get("source_identity_sha256")
        != source_identity["identity_sha256"]
        or budget_state.get("last_fully_resumable_epoch") != 0
        or budget_state.get("active_attempt_epoch") != 1
        or not isinstance(batch_count, int)
        or isinstance(batch_count, bool)
        or not 0 <= batch_count <= 138
        or not math.isfinite(elapsed)
        or not 0.0 <= elapsed <= float(budget_hours) * 3600.0
        or budget_state.get("gpu_budget_hours") != budget_hours
    ):
        raise RuntimeError(
            f"G2C {candidate_id} zero-epoch restart state/identity 漂移"
        )
    if "initialization.json" not in expected_files:
        return None
    initialization = _read_json(
        initialization_path, f"G2C {candidate_id} zero-epoch initialization"
    )
    if not isinstance(initialization, dict):
        raise TypeError(f"G2C {candidate_id} zero-epoch initialization 类型漂移")
    return initialization


def _validate_resume_progress_semantics(
    *,
    state: Mapping[str, Any],
    candidate_id: str,
    config: Mapping[str, Any],
    source_identity: Mapping[str, Any],
) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]]]:
    """在加载 GPU model 前验证 resume 的计数、trace 与 checkpoint 进度。"""

    _require_exact_keys(state, _RESUME_STATE_KEYS, "G2C candidate resume payload")
    completed_epoch = state.get("completed_epoch")
    if (
        state.get("version") != E018_P1_G2C_FORMAL_TRAIN_RESULT_VERSION
        or state.get("candidate_id") != candidate_id
        or state.get("config_sha256") != config["config_sha256"]
        or state.get("source_identity_sha256")
        != source_identity["identity_sha256"]
        or not isinstance(completed_epoch, int)
        or isinstance(completed_epoch, bool)
        or not 1 <= completed_epoch <= config["protocol"]["epochs_per_candidate"]
        or state.get("examples_seen") != completed_epoch * 4400
        or state.get("optimizer_steps") != completed_epoch * 138
        or not isinstance(state.get("initialization"), dict)
        or not isinstance(state.get("model_state"), dict)
        or not isinstance(state.get("optimizer_state"), dict)
        or not isinstance(state.get("scheduler_state"), dict)
        or not isinstance(state.get("rng_state"), dict)
    ):
        raise RuntimeError("G2C candidate resume identity/count/type 漂移")
    active_elapsed = float(state.get("active_gpu_elapsed_s", math.nan))
    if (
        not math.isfinite(active_elapsed)
        or not 0.0
        <= active_elapsed
        <= float(config["protocol"]["budgets"]["gpu_hours_max"]) * 3600.0
    ):
        raise RuntimeError("G2C candidate resume active GPU budget 漂移")
    traces = state.get("epoch_trace")
    checkpoints = state.get("checkpoint_inventory")
    if (
        not isinstance(traces, list)
        or len(traces) != completed_epoch
        or any(not isinstance(row, dict) for row in traces)
        or not isinstance(checkpoints, list)
        or any(not isinstance(item, dict) for item in checkpoints)
    ):
        raise RuntimeError("G2C candidate resume trace/checkpoint 类型或长度漂移")
    for index, row in enumerate(traces, start=1):
        _require_exact_keys(row, _EPOCH_TRACE_KEYS, "G2C resume epoch trace row")
        losses = row.get("loss")
        if (
            row.get("candidate_id") != candidate_id
            or row.get("epoch") != index
            or row.get("sample_count") != 4400
            or row.get("batch_count") != 138
            or row.get("examples_seen_total") != index * 4400
            or row.get("optimizer_steps_total") != index * 138
            or not isinstance(losses, dict)
            or set(losses) != set(_LOSS_COMPONENT_NAMES)
            or any(not math.isfinite(float(value)) for value in losses.values())
            or not math.isfinite(float(row.get("maximum_gradient_norm_pre_clip")))
            or float(row["maximum_gradient_norm_pre_clip"]) < 0.0
            or not math.isfinite(float(row.get("maximum_gradient_norm_post_clip")))
            or not 0.0
            <= float(row["maximum_gradient_norm_post_clip"])
            <= config["protocol"]["optimizer"]["gradient_clip_norm"] + 1e-5
            or not math.isfinite(float(row.get("learning_rate_after_scheduler_step")))
            or float(row["learning_rate_after_scheduler_step"]) < 0.0
        ):
            raise RuntimeError("G2C candidate resume epoch trace 语义漂移")
        for name in (
            "sample_order_sha256",
            "sampler_generator_state_before_sha256",
            "sampler_generator_state_after_sha256",
            "parameter_state_sha256",
            "motion_head_parameter_sha256",
            "optimizer_state_identity_sha256",
            "scheduler_state_identity_sha256",
            "rng_state_identity_sha256",
        ):
            _require_sha256(row.get(name), f"G2C resume trace {name}")
        if index > 1 and (
            row["sampler_generator_state_before_sha256"]
            != traces[index - 2]["sampler_generator_state_after_sha256"]
        ):
            raise RuntimeError("G2C candidate resume sampler epoch chain 漂移")
    if any(
        row["motion_head_parameter_sha256"]
        != traces[0]["motion_head_parameter_sha256"]
        for row in traces
    ):
        raise RuntimeError("G2C candidate resume frozen Motion Head 漂移")
    expected_epochs = [
        epoch for epoch in G2C_CANDIDATE_EPOCHS if epoch <= completed_epoch
    ]
    if [item.get("epoch") for item in checkpoints] != expected_epochs:
        raise RuntimeError("G2C candidate resume checkpoint epoch inventory 漂移")
    for item in checkpoints:
        _require_exact_keys(
            item, _CHECKPOINT_INVENTORY_KEYS, "G2C resume checkpoint item"
        )
        epoch = int(item["epoch"])
        expected_checkpoint = f"precision-{candidate_id.lower()}-epoch-{epoch:02d}.pt"
        expected_companion = (
            f"training-state-{candidate_id.lower()}-epoch-{epoch:02d}.pt"
        )
        if (
            item.get("candidate_id") != candidate_id
            or item.get("relative_path") != expected_checkpoint
            or item.get("training_state_relative_path") != expected_companion
            or item.get("examples_seen") != epoch * 4400
            or item.get("optimizer_steps") != epoch * 138
            or item.get("parameter_state_sha256")
            != traces[epoch - 1]["parameter_state_sha256"]
        ):
            raise RuntimeError("G2C candidate resume checkpoint metadata 漂移")
        for name in (
            "checkpoint_sha256",
            "parameter_state_sha256",
            "model_config_sha256",
            "provenance_sha256",
            "training_state_raw_sha256",
            "training_state_identity_sha256",
            "optimizer_state_identity_sha256",
            "scheduler_state_identity_sha256",
            "rng_state_identity_sha256",
            "sampler_state_identity_sha256",
            "budget_timing_identity_sha256",
        ):
            _require_sha256(item.get(name), f"G2C resume checkpoint {name}")
    return completed_epoch, traces, checkpoints


def _validate_candidate_resume_state(
    *,
    state: Mapping[str, Any],
    candidate_root: Path,
    candidate_id: str,
    config: Mapping[str, Any],
    source_identity: Mapping[str, Any],
) -> int:
    """在继续昂贵训练前交叉验证 resume、trace、checkpoint 与 companion。"""

    import torch

    from robot_vla.precision.checkpoint import (
        PrecisionCheckpointRole,
        load_precision_checkpoint,
        precision_parameter_state_sha256,
    )

    completed_epoch, traces, checkpoints = _validate_resume_progress_semantics(
        state=state,
        candidate_id=candidate_id,
        config=config,
        source_identity=source_identity,
    )
    expected_files = {
        "initialization.json",
        "epoch_trace.json",
        "checkpoint_inventory.json",
        "resume_state.pt",
        "budget_state.json",
        *(str(item["relative_path"]) for item in checkpoints),
        *(str(item["training_state_relative_path"]) for item in checkpoints),
    }
    _assert_unlinked_regular_file_tree(
        candidate_root, name=f"G2C {candidate_id} resume tree"
    )
    _verify_exact_regular_file_tree(
        candidate_root,
        expected_files=expected_files,
        name=f"G2C {candidate_id} resume tree",
    )
    persisted_traces = _read_json_array(
        candidate_root / "epoch_trace.json",
        f"G2C {candidate_id} persisted resume trace",
    )
    persisted_checkpoints = _read_json_array(
        candidate_root / "checkpoint_inventory.json",
        f"G2C {candidate_id} persisted checkpoint inventory",
    )
    budget_state = _read_json(
        candidate_root / "budget_state.json",
        f"G2C {candidate_id} persisted resume budget",
    )
    _require_exact_keys(
        budget_state,
        _BUDGET_STATE_KEYS,
        f"G2C {candidate_id} persisted resume budget",
    )
    active_attempt_epoch = budget_state.get("active_attempt_epoch")
    active_attempt_batch_count = budget_state.get("active_attempt_batch_count")
    budget_elapsed = float(budget_state.get("active_gpu_elapsed_s", math.nan))
    budget_hours = config["protocol"]["budgets"]["gpu_hours_max"]
    if (
        persisted_traces != traces
        or persisted_checkpoints != checkpoints
        or budget_state.get("version")
        != E018_P1_G2C_FORMAL_TRAIN_RESULT_VERSION
        or budget_state.get("candidate_id") != candidate_id
        or budget_state.get("config_sha256") != config["config_sha256"]
        or budget_state.get("source_identity_sha256")
        != source_identity["identity_sha256"]
        or budget_state.get("last_fully_resumable_epoch") != completed_epoch
        or active_attempt_epoch
        not in ({None} if completed_epoch == 20 else {None, completed_epoch + 1})
        or not isinstance(active_attempt_batch_count, int)
        or isinstance(active_attempt_batch_count, bool)
        or not 0 <= active_attempt_batch_count <= 138
        or (active_attempt_epoch is None and active_attempt_batch_count != 0)
        or not math.isfinite(budget_elapsed)
        or not float(state["active_gpu_elapsed_s"]) <= budget_elapsed
        or budget_elapsed > float(budget_hours) * 3600.0
        or budget_state.get("gpu_budget_hours") != budget_hours
    ):
        raise RuntimeError("G2C candidate resume persisted evidence/budget 漂移")
    initialization = _read_json(
        candidate_root / "initialization.json",
        f"G2C {candidate_id} resume initialization",
    )
    final_trace = traces[-1]
    if (
        initialization != state["initialization"]
        or precision_parameter_state_sha256(state["model_state"])
        != final_trace["parameter_state_sha256"]
        or _state_identity_sha256(state["optimizer_state"])
        != final_trace["optimizer_state_identity_sha256"]
        or _state_identity_sha256(state["scheduler_state"])
        != final_trace["scheduler_state_identity_sha256"]
        or _rng_state_sha256(state["rng_state"])
        != final_trace["rng_state_identity_sha256"]
        or _tensor_sha256(state["rng_state"]["sampler_generator"])
        != final_trace["sampler_generator_state_after_sha256"]
    ):
        raise RuntimeError("G2C candidate resume state/trace identity 漂移")
    for item in checkpoints:
        epoch = int(item["epoch"])
        checkpoint_path = candidate_root / str(item["relative_path"])
        companion_path = candidate_root / str(item["training_state_relative_path"])
        if (
            checkpoint_path.name != item["relative_path"]
            or checkpoint_path.is_symlink()
            or checkpoint_path.stat().st_nlink != 1
            or file_sha256(checkpoint_path) != item["checkpoint_sha256"]
            or companion_path.name != item["training_state_relative_path"]
            or companion_path.is_symlink()
            or companion_path.stat().st_nlink != 1
            or file_sha256(companion_path) != item["training_state_raw_sha256"]
        ):
            raise RuntimeError("G2C candidate resume checkpoint/companion 文件漂移")
        loaded = load_precision_checkpoint(
            checkpoint_path,
            expected_checkpoint_sha256=item["checkpoint_sha256"],
            expected_provenance_sha256=item["provenance_sha256"],
            expected_role=PrecisionCheckpointRole.FORMAL_TRAINING,
        )
        if (
            loaded.receipt.parameter_state_sha256
            != item["parameter_state_sha256"]
            or loaded.receipt.model_config_sha256 != item["model_config_sha256"]
            or loaded.provenance.to_dict() != item["provenance"]
            or loaded.provenance.training_config_sha256 != config["config_sha256"]
            or loaded.provenance.data_identity_sha256
            != config["data_parent"]["data_identity_sha256"]
            or loaded.provenance.source_tree_sha256
            != source_identity["source_tree_sha256"]
            or loaded.provenance.seed
            != G2C_CANDIDATE_INITIALIZATION_SEEDS[candidate_id]
        ):
            raise RuntimeError("G2C candidate resume checkpoint provenance 漂移")
        companion = torch.load(companion_path, map_location="cpu", weights_only=True)
        _require_exact_keys(
            companion, _TRAINING_STATE_KEYS, "G2C resume checkpoint companion"
        )
        unsigned_companion = {
            name: value
            for name, value in companion.items()
            if name != "state_identity_sha256"
        }
        if (
            _state_identity_sha256(unsigned_companion)
            != companion["state_identity_sha256"]
            or companion["state_identity_sha256"]
            != item["training_state_identity_sha256"]
            or companion.get("version")
            != E018_P1_G2C_FORMAL_TRAIN_RESULT_VERSION
            or companion.get("kind") != "immutable-checkpoint-training-state/v1"
            or companion.get("candidate_id") != candidate_id
            or companion.get("epoch") != epoch
            or companion.get("config_sha256") != config["config_sha256"]
            or companion.get("source_identity_sha256")
            != source_identity["identity_sha256"]
            or companion.get("checkpoint_sha256") != item["checkpoint_sha256"]
            or companion.get("checkpoint_parameter_sha256")
            != item["parameter_state_sha256"]
            or companion.get("checkpoint_provenance_sha256")
            != item["provenance_sha256"]
            or companion.get("examples_seen") != epoch * 4400
            or companion.get("optimizer_steps") != epoch * 138
            or companion.get("gpu_budget_hours")
            != config["protocol"]["budgets"]["gpu_hours_max"]
            or not math.isfinite(
                float(companion.get("active_gpu_elapsed_s_at_checkpoint", math.nan))
            )
            or not 0.0
            <= float(companion["active_gpu_elapsed_s_at_checkpoint"])
            <= float(state["active_gpu_elapsed_s"])
            or companion["active_gpu_elapsed_s_at_checkpoint"]
            != item["active_gpu_elapsed_s_at_checkpoint"]
            or canonical_sha256(
                {
                    "active_gpu_elapsed_s_at_checkpoint": companion[
                        "active_gpu_elapsed_s_at_checkpoint"
                    ],
                    "gpu_budget_hours": companion["gpu_budget_hours"],
                }
            )
            != item["budget_timing_identity_sha256"]
            or _state_identity_sha256(companion["optimizer_state"])
            != item["optimizer_state_identity_sha256"]
            or _state_identity_sha256(companion["scheduler_state"])
            != item["scheduler_state_identity_sha256"]
            or _rng_state_sha256(companion["rng_state"])
            != item["rng_state_identity_sha256"]
            or _tensor_sha256(companion["rng_state"]["sampler_generator"])
            != item["sampler_state_identity_sha256"]
            or item["optimizer_state_identity_sha256"]
            != traces[epoch - 1]["optimizer_state_identity_sha256"]
            or item["scheduler_state_identity_sha256"]
            != traces[epoch - 1]["scheduler_state_identity_sha256"]
            or item["rng_state_identity_sha256"]
            != traces[epoch - 1]["rng_state_identity_sha256"]
            or item["sampler_state_identity_sha256"]
            != traces[epoch - 1]["sampler_generator_state_after_sha256"]
        ):
            raise RuntimeError("G2C candidate resume companion identity/state 漂移")
        del loaded, companion
    if checkpoints:
        latest = checkpoints[-1]
        if float(state["active_gpu_elapsed_s"]) < float(
            latest["active_gpu_elapsed_s_at_checkpoint"]
        ):
            raise RuntimeError("G2C candidate resume active GPU time 倒退")
        if completed_epoch == int(latest["epoch"]) and (
            precision_parameter_state_sha256(state["model_state"])
            != latest["parameter_state_sha256"]
            or _state_identity_sha256(state["optimizer_state"])
            != latest["optimizer_state_identity_sha256"]
            or _state_identity_sha256(state["scheduler_state"])
            != latest["scheduler_state_identity_sha256"]
            or _rng_state_sha256(state["rng_state"])
            != latest["rng_state_identity_sha256"]
        ):
            raise RuntimeError("G2C candidate resume 与最后 companion 漂移")
    return completed_epoch


def _train_one_formal_candidate(
    *,
    candidate_id: str,
    samples: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    e016_config_path: Path,
    e016_training_output: Path,
    candidate_root: Path,
    resume: bool,
    deadline_monotonic_s: float,
    active_gpu_elapsed_s_before_process: float,
    active_process_started_monotonic_s: float,
) -> dict[str, Any]:
    import torch

    from robot_vla.precision.checkpoint import precision_parameter_state_sha256
    from robot_vla.precision.losses import PrecisionLossConfig, precision_unet_loss

    protocol = config["protocol"]
    if len(samples) != _EXPECTED_SPLIT_SAMPLES["train"]:
        raise RuntimeError("G2C formal train sample count 必须是 4400")
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("G2C formal TRAIN 要求 CUDA")
    resume_path = candidate_root / "resume_state.pt"
    initialization_path = candidate_root / "initialization.json"
    trace_path = candidate_root / "epoch_trace.json"
    checkpoint_inventory_path = candidate_root / "checkpoint_inventory.json"
    budget_state_path = candidate_root / "budget_state.json"
    sampler = torch.Generator(device="cpu")
    sampler.manual_seed(G2C_SHARED_SAMPLER_SEED)
    if resume and resume_path.is_symlink():
        raise _G2CResumePreflightError(
            "G2C resume preflight: candidate resume_state.pt 禁止 symlink"
        )
    if resume and resume_path.is_file() and resume_path.stat().st_nlink != 1:
        raise _G2CResumePreflightError(
            "G2C resume preflight: candidate resume_state.pt 禁止 hardlink"
        )
    zero_epoch_restart = resume and candidate_root.exists() and not resume_path.is_file()
    try:
        persisted_initialization = (
            _validate_zero_epoch_restart_candidate(
                candidate_root=candidate_root,
                candidate_id=candidate_id,
                config=config,
                source_identity=source_identity,
            )
            if zero_epoch_restart
            else None
        )
    except Exception as error:
        raise _G2CResumePreflightError(
            f"G2C zero-epoch resume preflight failed: {error}"
        ) from error
    if resume and resume_path.is_file():
        try:
            state = torch.load(resume_path, map_location="cpu", weights_only=True)
            resume_completed_epoch = _validate_candidate_resume_state(
                state=state,
                candidate_root=candidate_root,
                candidate_id=candidate_id,
                config=config,
                source_identity=source_identity,
            )
        except Exception as error:
            raise _G2CResumePreflightError(
                f"G2C candidate resume preflight failed: {error}"
            ) from error
        _write_candidate_budget_state(
            budget_state_path,
            candidate_id=candidate_id,
            config=config,
            source_identity=source_identity,
            last_fully_resumable_epoch=resume_completed_epoch,
            active_attempt_epoch=(
                None
                if resume_completed_epoch == protocol["epochs_per_candidate"]
                else resume_completed_epoch + 1
            ),
            active_attempt_batch_count=0,
            active_gpu_elapsed_s=_active_gpu_elapsed_s(
                active_before_process_s=active_gpu_elapsed_s_before_process,
                process_started_monotonic_s=active_process_started_monotonic_s,
            ),
        )
        model, fresh_initialization = _load_formal_candidate_model(
            candidate_id=candidate_id,
            config=config,
            e016_config_path=e016_config_path,
            e016_training_output=e016_training_output,
            device=device,
        )
        if fresh_initialization != state["initialization"]:
            raise RuntimeError("G2C candidate resume initialization 漂移")
        model.load_state_dict(state["model_state"], strict=True)
        parameters = [item for item in model.parameters() if item.requires_grad]
        optimizer = torch.optim.AdamW(
            parameters,
            lr=protocol["optimizer"]["learning_rate"],
            weight_decay=protocol["optimizer"]["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=protocol["scheduler"]["t_max_epochs"],
            eta_min=protocol["scheduler"]["eta_min"],
        )
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        _restore_training_rng_state(state["rng_state"], sampler)
        initialization = state["initialization"]
        traces = list(state["epoch_trace"])
        checkpoints = list(state["checkpoint_inventory"])
        completed_epoch = resume_completed_epoch
        examples_seen = int(state["examples_seen"])
        optimizer_steps = int(state["optimizer_steps"])
    else:
        candidate_root.mkdir(mode=0o700, parents=True, exist_ok=resume)
        _write_candidate_budget_state(
            budget_state_path,
            candidate_id=candidate_id,
            config=config,
            source_identity=source_identity,
            last_fully_resumable_epoch=0,
            active_attempt_epoch=1,
            active_attempt_batch_count=0,
            active_gpu_elapsed_s=_active_gpu_elapsed_s(
                active_before_process_s=active_gpu_elapsed_s_before_process,
                process_started_monotonic_s=active_process_started_monotonic_s,
            ),
        )
        model, initialization = _load_formal_candidate_model(
            candidate_id=candidate_id,
            config=config,
            e016_config_path=e016_config_path,
            e016_training_output=e016_training_output,
            device=device,
        )
        if (
            zero_epoch_restart
            and persisted_initialization is not None
            and persisted_initialization != initialization
        ):
            raise RuntimeError("G2C zero-epoch restart initialization identity 漂移")
        parameters = [item for item in model.parameters() if item.requires_grad]
        optimizer = torch.optim.AdamW(
            parameters,
            lr=protocol["optimizer"]["learning_rate"],
            weight_decay=protocol["optimizer"]["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=protocol["scheduler"]["t_max_epochs"],
            eta_min=protocol["scheduler"]["eta_min"],
        )
        traces: list[dict[str, Any]] = []
        checkpoints: list[dict[str, Any]] = []
        completed_epoch = 0
        examples_seen = 0
        optimizer_steps = 0
        if not initialization_path.exists():
            _atomic_json(initialization_path, initialization)
    loss_values = dict(protocol["loss"])
    loss_values.pop("heatmap_sigma_px")
    loss_config = PrecisionLossConfig(**loss_values)
    batches_per_epoch = math.ceil(len(samples) / protocol["loader"]["batch_size"])
    model.train()
    initial_motion_sha = initialization["initial_motion_head_parameter_sha256"]
    for epoch in range(completed_epoch + 1, protocol["epochs_per_candidate"] + 1):
        generator_before = sampler.get_state().clone()
        order = torch.randperm(len(samples), generator=sampler).tolist()
        generator_after = sampler.get_state().clone()
        order_identity = [_sample_identity(samples[index]) for index in order]
        order_sha = canonical_sha256(order_identity)
        weighted_losses = {name: 0.0 for name in _LOSS_COMPONENT_NAMES}
        maximum_pre_clip = 0.0
        maximum_post_clip = 0.0
        for start in range(0, len(order), protocol["loader"]["batch_size"]):
            if time.monotonic() >= deadline_monotonic_s:
                raise TimeoutError("G2C formal TRAIN 达到冻结的 10 GPU-hour wall deadline")
            indices = order[start : start + protocol["loader"]["batch_size"]]
            batch = _collate_training([samples[index] for index in indices])
            moved = {
                name: value.to(device) if isinstance(value, torch.Tensor) else value
                for name, value in batch.items()
            }
            optimizer.zero_grad(set_to_none=True)
            supervision = _build_supervision(moved)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                output = model(
                    moved["image"], moved["structured_state"], moved["geometric_motion"]
                )
                loss = precision_unet_loss(output, supervision, loss_config)
            if not bool(torch.isfinite(loss.loss)):
                raise RuntimeError(f"G2C {candidate_id} epoch {epoch} loss NaN/Inf")
            loss.loss.backward()
            pre_clip = torch.nn.utils.clip_grad_norm_(
                parameters, protocol["optimizer"]["gradient_clip_norm"]
            )
            if not bool(torch.isfinite(pre_clip)):
                raise RuntimeError(f"G2C {candidate_id} epoch {epoch} gradient NaN/Inf")
            post_clip_squared = sum(
                float(torch.sum(parameter.grad.detach().float().square()).item())
                for parameter in parameters
                if parameter.grad is not None
            )
            post_clip = math.sqrt(post_clip_squared)
            if post_clip > protocol["optimizer"]["gradient_clip_norm"] + 1e-5:
                raise RuntimeError("G2C formal post-clip gradient norm 漂移")
            optimizer.step()
            batch_size = int(moved["image"].shape[0])
            for name in _LOSS_COMPONENT_NAMES:
                value = float(getattr(loss, name).detach().float().item())
                if not math.isfinite(value):
                    raise RuntimeError("G2C formal loss component NaN/Inf")
                weighted_losses[name] += value * batch_size
            maximum_pre_clip = max(maximum_pre_clip, float(pre_clip.detach().item()))
            maximum_post_clip = max(maximum_post_clip, post_clip)
            examples_seen += batch_size
            optimizer_steps += 1
            active_elapsed = _active_gpu_elapsed_s(
                active_before_process_s=active_gpu_elapsed_s_before_process,
                process_started_monotonic_s=active_process_started_monotonic_s,
            )
            _write_candidate_budget_state(
                budget_state_path,
                candidate_id=candidate_id,
                config=config,
                source_identity=source_identity,
                last_fully_resumable_epoch=completed_epoch,
                active_attempt_epoch=epoch,
                active_attempt_batch_count=(
                    start // protocol["loader"]["batch_size"] + 1
                ),
                active_gpu_elapsed_s=active_elapsed,
            )
            if time.monotonic() >= deadline_monotonic_s:
                raise TimeoutError("G2C formal TRAIN 达到冻结的 10 GPU-hour wall deadline")
        scheduler.step()
        completed_epoch = epoch
        optimizer_state = optimizer.state_dict()
        scheduler_state = scheduler.state_dict()
        rng_state = _capture_training_rng_state(sampler)
        epoch_trace = {
            "candidate_id": candidate_id,
            "epoch": epoch,
            "sample_count": len(order),
            "batch_count": batches_per_epoch,
            "sample_order_sha256": order_sha,
            "sampler_generator_state_before_sha256": _tensor_sha256(generator_before),
            "sampler_generator_state_after_sha256": _tensor_sha256(generator_after),
            "loss": {
                name: weighted_losses[name] / len(order) for name in _LOSS_COMPONENT_NAMES
            },
            "maximum_gradient_norm_pre_clip": maximum_pre_clip,
            "maximum_gradient_norm_post_clip": maximum_post_clip,
            "learning_rate_after_scheduler_step": float(optimizer.param_groups[0]["lr"]),
            "examples_seen_total": examples_seen,
            "optimizer_steps_total": optimizer_steps,
            "parameter_state_sha256": precision_parameter_state_sha256(model.state_dict()),
            "motion_head_parameter_sha256": precision_parameter_state_sha256(
                model.motion_head.state_dict()
            ),
            "optimizer_state_identity_sha256": _state_identity_sha256(
                optimizer_state
            ),
            "scheduler_state_identity_sha256": _state_identity_sha256(
                scheduler_state
            ),
            "rng_state_identity_sha256": _rng_state_sha256(rng_state),
        }
        if epoch_trace["motion_head_parameter_sha256"] != initial_motion_sha:
            raise RuntimeError("G2C formal frozen Motion Head 漂移")
        traces.append(epoch_trace)
        if epoch in G2C_CANDIDATE_EPOCHS:
            checkpoint_active_elapsed = _active_gpu_elapsed_s(
                active_before_process_s=active_gpu_elapsed_s_before_process,
                process_started_monotonic_s=active_process_started_monotonic_s,
            )
            checkpoint = _ensure_formal_epoch_checkpoint(
                path=candidate_root / f"precision-{candidate_id.lower()}-epoch-{epoch:02d}.pt",
                model=model,
                candidate_id=candidate_id,
                epoch=epoch,
                examples_seen=examples_seen,
                optimizer_steps=optimizer_steps,
                config=config,
                source_identity=source_identity,
            )
            checkpoint.update(
                _ensure_checkpoint_training_state(
                    path=candidate_root
                    / f"training-state-{candidate_id.lower()}-epoch-{epoch:02d}.pt",
                    candidate_id=candidate_id,
                    epoch=epoch,
                    checkpoint=checkpoint,
                    config=config,
                    source_identity=source_identity,
                    examples_seen=examples_seen,
                    optimizer_steps=optimizer_steps,
                    optimizer_state=optimizer_state,
                    scheduler_state=scheduler_state,
                    rng_state=rng_state,
                    active_gpu_elapsed_s=checkpoint_active_elapsed,
                )
            )
            existing = [item for item in checkpoints if item["epoch"] == epoch]
            if existing and existing != [checkpoint]:
                raise RuntimeError("G2C resume checkpoint inventory 漂移")
            if not existing:
                checkpoints.append(checkpoint)
        resume_active_elapsed = _active_gpu_elapsed_s(
            active_before_process_s=active_gpu_elapsed_s_before_process,
            process_started_monotonic_s=active_process_started_monotonic_s,
        )
        resume_state = {
            "version": E018_P1_G2C_FORMAL_TRAIN_RESULT_VERSION,
            "candidate_id": candidate_id,
            "config_sha256": config["config_sha256"],
            "source_identity_sha256": source_identity["identity_sha256"],
            "completed_epoch": completed_epoch,
            "examples_seen": examples_seen,
            "optimizer_steps": optimizer_steps,
            "model_state": {
                name: value.detach().cpu().contiguous()
                for name, value in model.state_dict().items()
            },
            "optimizer_state": optimizer_state,
            "scheduler_state": scheduler_state,
            "rng_state": rng_state,
            "initialization": initialization,
            "epoch_trace": traces,
            "checkpoint_inventory": checkpoints,
            "active_gpu_elapsed_s": resume_active_elapsed,
        }
        _replace_torch_resume_state(resume_state, resume_path)
        _atomic_json(trace_path, traces)
        _atomic_json(checkpoint_inventory_path, checkpoints)
        _write_candidate_budget_state(
            budget_state_path,
            candidate_id=candidate_id,
            config=config,
            source_identity=source_identity,
            last_fully_resumable_epoch=completed_epoch,
            active_attempt_epoch=None,
            active_attempt_batch_count=0,
            active_gpu_elapsed_s=_active_gpu_elapsed_s(
                active_before_process_s=active_gpu_elapsed_s_before_process,
                process_started_monotonic_s=active_process_started_monotonic_s,
            ),
        )
    final_rng = _capture_training_rng_state(sampler)
    final_active_elapsed = _active_gpu_elapsed_s(
        active_before_process_s=active_gpu_elapsed_s_before_process,
        process_started_monotonic_s=active_process_started_monotonic_s,
    )
    _write_candidate_budget_state(
        budget_state_path,
        candidate_id=candidate_id,
        config=config,
        source_identity=source_identity,
        last_fully_resumable_epoch=completed_epoch,
        active_attempt_epoch=None,
        active_attempt_batch_count=0,
        active_gpu_elapsed_s=final_active_elapsed,
    )
    result = {
        "candidate_id": candidate_id,
        "status": "complete-formal-training-candidate-pass",
        "initialization": initialization,
        "completed_epochs": completed_epoch,
        "examples_seen": examples_seen,
        "optimizer_steps": optimizer_steps,
        "batches_per_epoch": batches_per_epoch,
        "checkpoint_count": len(checkpoints),
        "checkpoint_inventory": checkpoints,
        "epoch_trace_raw_sha256": file_sha256(trace_path),
        "checkpoint_inventory_raw_sha256": file_sha256(checkpoint_inventory_path),
        "resume_state_raw_sha256": file_sha256(resume_path),
        "resume_rng_state_sha256": _rng_state_sha256(final_rng),
        "active_gpu_elapsed_s": final_active_elapsed,
        "final_parameter_state_sha256": precision_parameter_state_sha256(
            model.state_dict()
        ),
        "initial_motion_head_parameter_sha256": initial_motion_sha,
        "final_motion_head_parameter_sha256": precision_parameter_state_sha256(
            model.motion_head.state_dict()
        ),
    }
    if result["checkpoint_count"] != len(G2C_CANDIDATE_EPOCHS):
        raise RuntimeError("G2C formal candidate checkpoint count 漂移")
    return result


def run_g2c_formal_training(
    *,
    config_path: str | Path,
    train_input_root: str | Path,
    e016_config_path: str | Path,
    e016_training_output: str | Path,
    repository_root: str | Path,
    output_root: str | Path,
    decision_exit_go: bool,
    resume: bool = False,
) -> dict[str, Any]:
    """训练 W-KV0/S；第一条语句即执行 Gate，HOLD 时不打开任何输入。"""

    if decision_exit_go is not True:
        raise PermissionError("G2C formal TRAIN 仍为 HOLD；未收到 source R2 GO")
    output = Path(output_root)
    if output.exists() and not resume:
        raise FileExistsError(f"G2C formal TRAIN output 已存在: {output}")
    if not output.exists() and resume:
        raise FileNotFoundError("G2C formal TRAIN resume output 不存在")
    config = load_g2c_formal_training_config(config_path)
    input_verification = validate_g2c_input_view(
        config_path=config_path,
        input_root=train_input_root,
        expected_role="train-paired",
    )
    source_identity = _git_source_identity(Path(repository_root))
    if not resume:
        output.mkdir(mode=0o700, parents=True, exist_ok=False)
        _atomic_json(output / "config_snapshot.json", config)
        _atomic_json(output / "source_identity.json", source_identity)
        _atomic_json(output / "train_input_verification.json", input_verification)
        _atomic_json(
            output / "run_state.json",
            {
                "version": E018_P1_G2C_FORMAL_TRAIN_RESULT_VERSION,
                "status": "formal-training-in-progress",
                "config_sha256": config["config_sha256"],
                "source_identity_sha256": source_identity["identity_sha256"],
                "gpu_budget_hours": config["protocol"]["budgets"]["gpu_hours_max"],
                "active_gpu_elapsed_s_before_process": 0.0,
            },
        )
    else:
        if (
            _read_json(output / "config_snapshot.json", "G2C train config snapshot")
            != config
            or _read_json(output / "source_identity.json", "G2C train source identity")
            != source_identity
            or _read_json(
                output / "train_input_verification.json", "G2C train input verification"
            )
            != input_verification
        ):
            raise RuntimeError("G2C formal TRAIN resume parent identity 漂移")
        run_state = _read_json(output / "run_state.json", "G2C train run state")
        if run_state.get("status") != "formal-training-in-progress":
            raise RuntimeError("G2C formal TRAIN 只允许恢复未完成运行")
    dataset = G2CFrontTrainingDataset(train_input_root, "train")
    samples = [dataset[index] for index in range(len(dataset))]
    del dataset
    if len(samples) != _EXPECTED_SPLIT_SAMPLES["train"]:
        raise RuntimeError("G2C train-only view sample count 漂移")
    candidate_results: dict[str, Any] = {}
    started = time.perf_counter()
    training_started_at_unix_ns = time.time_ns()
    budget_seconds = float(config["protocol"]["budgets"]["gpu_hours_max"]) * 3600.0
    prior_active_gpu_elapsed_s = 0.0
    if resume:
        for candidate_id in G2C_CANDIDATE_IDS:
            budget_path = output / "candidates" / candidate_id / "budget_state.json"
            if budget_path.is_file():
                budget_state = _read_json(budget_path, f"G2C {candidate_id} budget state")
                if (
                    budget_state.get("candidate_id") != candidate_id
                    or budget_state.get("config_sha256") != config["config_sha256"]
                    or budget_state.get("source_identity_sha256")
                    != source_identity["identity_sha256"]
                ):
                    raise RuntimeError("G2C resume active GPU budget identity 漂移")
                prior_active_gpu_elapsed_s = max(
                    prior_active_gpu_elapsed_s,
                    float(budget_state.get("active_gpu_elapsed_s", -1.0)),
                )
    active_gpu_elapsed_s_before_process, remaining_budget_seconds = (
        remaining_g2c_active_gpu_budget_seconds(
            budget_hours=config["protocol"]["budgets"]["gpu_hours_max"],
            persisted_active_elapsed_s=[prior_active_gpu_elapsed_s],
        )
    )
    active_process_started_monotonic_s = time.monotonic()
    prior_active_gpu_elapsed_s = active_gpu_elapsed_s_before_process
    deadline_monotonic_s = (
        active_process_started_monotonic_s + remaining_budget_seconds
    )
    deadline_unix_ns = training_started_at_unix_ns + int(
        remaining_budget_seconds * 1e9
    )
    for candidate_id in G2C_CANDIDATE_IDS:
        try:
            if time.monotonic() >= deadline_monotonic_s:
                raise TimeoutError("G2C formal TRAIN 累计 active GPU wall 已耗尽")
            candidate_results[candidate_id] = _train_one_formal_candidate(
                candidate_id=candidate_id,
                samples=samples,
                config=config,
                source_identity=source_identity,
                e016_config_path=Path(e016_config_path),
                e016_training_output=Path(e016_training_output),
                candidate_root=output / "candidates" / candidate_id,
                resume=resume,
                deadline_monotonic_s=deadline_monotonic_s,
                active_gpu_elapsed_s_before_process=(
                    active_gpu_elapsed_s_before_process
                ),
                active_process_started_monotonic_s=(
                    active_process_started_monotonic_s
                ),
            )
            if time.monotonic() >= deadline_monotonic_s:
                raise TimeoutError("G2C formal TRAIN 累计 active GPU wall 已耗尽")
        except Exception as error:
            if isinstance(error, _G2CResumePreflightError):
                raise
            failed_budget_path = (
                output / "candidates" / candidate_id / "budget_state.json"
            )
            if failed_budget_path.is_file():
                failed_budget = _read_json(
                    failed_budget_path, f"G2C {candidate_id} failed budget state"
                )
                last_epoch = int(
                    failed_budget.get("last_fully_resumable_epoch", 0)
                )
                active_epoch = failed_budget.get("active_attempt_epoch")
                active_batch_count = int(
                    failed_budget.get("active_attempt_batch_count", 0)
                )
            else:
                last_epoch = 0
                active_epoch = 1
                active_batch_count = 0
            _write_candidate_budget_state(
                failed_budget_path,
                candidate_id=candidate_id,
                config=config,
                source_identity=source_identity,
                last_fully_resumable_epoch=last_epoch,
                active_attempt_epoch=active_epoch,
                active_attempt_batch_count=active_batch_count,
                active_gpu_elapsed_s=_active_gpu_elapsed_s(
                    active_before_process_s=active_gpu_elapsed_s_before_process,
                    process_started_monotonic_s=active_process_started_monotonic_s,
                ),
            )
            if not isinstance(error, TimeoutError):
                raise
            active_values = [prior_active_gpu_elapsed_s]
            for name in G2C_CANDIDATE_IDS:
                path = output / "candidates" / name / "budget_state.json"
                if path.is_file():
                    active_values.append(
                        float(_read_json(path, f"G2C {name} budget state")["active_gpu_elapsed_s"])
                    )
            _atomic_json(
                output / "run_state.json",
                {
                    "version": E018_P1_G2C_FORMAL_TRAIN_RESULT_VERSION,
                    "status": "formal-training-budget-exhausted-no-further-optimization",
                    "config_sha256": config["config_sha256"],
                    "source_identity_sha256": source_identity["identity_sha256"],
                    "gpu_budget_hours": config["protocol"]["budgets"]["gpu_hours_max"],
                    "active_gpu_elapsed_s": max(active_values),
                },
            )
            raise
        prior_active_gpu_elapsed_s = candidate_results[candidate_id][
            "active_gpu_elapsed_s"
        ]
        gc.collect()
        __import__("torch").cuda.empty_cache()
    traces = {
        candidate_id: _read_json_array(
            output / "candidates" / candidate_id / "epoch_trace.json",
            f"G2C {candidate_id} trace",
        )
        for candidate_id in G2C_CANDIDATE_IDS
    }
    paired_orders_equal = all(
        traces["W-KV0"][epoch]["sample_order_sha256"]
        == traces["S"][epoch]["sample_order_sha256"]
        for epoch in range(20)
    )
    if not paired_orders_equal:
        raise RuntimeError("G2C W-KV0/S 逐 epoch sampler order 不一致")
    checkpoint_inventory = [
        item
        for candidate_id in G2C_CANDIDATE_IDS
        for item in candidate_results[candidate_id]["checkpoint_inventory"]
    ]
    elapsed = time.perf_counter() - started
    summary = {
        "version": E018_P1_G2C_FORMAL_TRAIN_RESULT_VERSION,
        "status": "complete-formal-training-pass",
        "classification": "development-only-no-test-no-actuation",
        "config_sha256": config["config_sha256"],
        "data_identity_sha256": config["data_parent"]["data_identity_sha256"],
        "source_git_commit": source_identity["git_commit"],
        "source_identity_sha256": source_identity["identity_sha256"],
        "train_input_receipt_raw_sha256": input_verification["receipt_raw_sha256"],
        "train_input_receipt_internal_sha256": input_verification[
            "receipt_internal_sha256"
        ],
        "candidate_results": candidate_results,
        "checkpoint_inventory": checkpoint_inventory,
        "checkpoint_count": len(checkpoint_inventory),
        "paired_epoch_sample_orders_equal": paired_orders_equal,
        "model_epoch_count": 40,
        "training_started_at_unix_ns": training_started_at_unix_ns,
        "deadline_unix_ns": deadline_unix_ns,
        "gpu_wall_budget_hours": config["protocol"]["budgets"]["gpu_hours_max"],
        "active_gpu_elapsed_s": prior_active_gpu_elapsed_s,
        "elapsed_s": elapsed,
        "train_label_sample_materialization_count": 4400,
        "optimizer_example_consumption_per_candidate": 88000,
        "optimizer_example_consumption_total": 176000,
        "optimizer_steps_per_candidate": 2760,
        "optimizer_steps_total": 5520,
        "training_privileged_label_bundle_open_count": 400,
        "model_val_label_bundle_open_count": 0,
        "calibration_label_bundle_open_count": 0,
        "qualification_label_bundle_open_count": 0,
        "test_array_read_count": 0,
        "memory_read_count": 0,
        "memory_write_count": 0,
        "runtime_camera_actuation_count": 0,
        "physical_camera_actuation_count": 0,
        "arm_motion_command_count": 0,
        "gripper_close_command_count": 0,
        "manipulation_progression_count": 0,
    }
    if summary["checkpoint_count"] != 8:
        raise RuntimeError("G2C formal TRAIN 必须冻结 8 checkpoint")
    if prior_active_gpu_elapsed_s > budget_seconds:
        raise TimeoutError("G2C formal TRAIN 累计 active GPU wall 超过 10 小时")
    _atomic_json(output / "training_summary.json", summary)
    artifact_names = _g2c_training_artifact_names(checkpoint_inventory)
    artifact_sha = {name: file_sha256(output / name) for name in artifact_names}
    artifact_bytes_before_receipt = sum(
        (output / name).stat().st_size for name in artifact_names
    )
    if (
        artifact_bytes_before_receipt
        > config["protocol"]["budgets"]["artifact_bytes_max"]
    ):
        raise RuntimeError("G2C formal TRAIN artifact 超过 20 GiB")
    receipt = {
        "version": E018_P1_G2C_FORMAL_TRAIN_RESULT_VERSION,
        "status": summary["status"],
        "classification": summary["classification"],
        "config_sha256": config["config_sha256"],
        "data_identity_sha256": summary["data_identity_sha256"],
        "source_git_commit": summary["source_git_commit"],
        "source_identity_sha256": summary["source_identity_sha256"],
        "checkpoint_inventory": checkpoint_inventory,
        "checkpoint_count": 8,
        "paired_epoch_sample_orders_equal": True,
        "model_epoch_count": 40,
        "optimizer_example_consumption_total": 176000,
        "optimizer_steps_total": 5520,
        "artifact_bytes_before_receipt": artifact_bytes_before_receipt,
        "artifact_sha256": artifact_sha,
        "training_privileged_label_bundle_open_count": 400,
        "model_val_label_bundle_open_count": 0,
        "calibration_label_bundle_open_count": 0,
        "qualification_label_bundle_open_count": 0,
        "test_array_read_count": 0,
        "memory_read_count": 0,
        "memory_write_count": 0,
        "runtime_camera_actuation_count": 0,
        "physical_camera_actuation_count": 0,
        "arm_motion_command_count": 0,
        "gripper_close_command_count": 0,
        "manipulation_progression_count": 0,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    _atomic_json(output / "training_receipt.json", receipt)
    _atomic_json(
        output / "run_state.json",
        {
            "version": E018_P1_G2C_FORMAL_TRAIN_RESULT_VERSION,
            "status": "complete-formal-training-pass",
            "config_sha256": config["config_sha256"],
            "receipt_sha256": receipt["receipt_sha256"],
        },
    )
    total_artifact_bytes = _verify_exact_regular_file_tree(
        output,
        expected_files=set(artifact_names)
        | {"training_receipt.json", "run_state.json"},
        name="G2C formal TRAIN",
    )
    if total_artifact_bytes > config["protocol"]["budgets"]["artifact_bytes_max"]:
        raise RuntimeError("G2C formal TRAIN 完整 artifact 超过 20 GiB")
    return {
        **summary,
        "receipt": receipt,
        "total_artifact_bytes": total_artifact_bytes,
    }


def verify_g2c_formal_training(
    *, config_path: str | Path, output_root: str | Path
) -> dict[str, Any]:
    """验证 formal TRAIN artifact；不读取任何 Dataset 或 validation label。"""

    import torch

    from robot_vla.precision.checkpoint import (
        PrecisionCheckpointRole,
        load_precision_checkpoint,
        precision_parameter_state_sha256,
    )

    config = load_g2c_formal_training_config(config_path)
    root = Path(output_root)
    receipt_path = root / "training_receipt.json"
    receipt = _read_json(receipt_path, "G2C training receipt")
    _require_exact_keys(
        receipt,
        {
            "version",
            "status",
            "classification",
            "config_sha256",
            "data_identity_sha256",
            "source_git_commit",
            "source_identity_sha256",
            "checkpoint_inventory",
            "checkpoint_count",
            "paired_epoch_sample_orders_equal",
            "model_epoch_count",
            "optimizer_example_consumption_total",
            "optimizer_steps_total",
            "artifact_bytes_before_receipt",
            "artifact_sha256",
            "training_privileged_label_bundle_open_count",
            "model_val_label_bundle_open_count",
            "calibration_label_bundle_open_count",
            "qualification_label_bundle_open_count",
            "test_array_read_count",
            "memory_read_count",
            "memory_write_count",
            "runtime_camera_actuation_count",
            "physical_camera_actuation_count",
            "arm_motion_command_count",
            "gripper_close_command_count",
            "manipulation_progression_count",
            "receipt_sha256",
        },
        "G2C training receipt",
    )
    internal = receipt.get("receipt_sha256")
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256", None)
    if (
        internal != canonical_sha256(unsigned)
        or receipt.get("version") != E018_P1_G2C_FORMAL_TRAIN_RESULT_VERSION
        or receipt.get("status") != "complete-formal-training-pass"
        or receipt.get("classification") != "development-only-no-test-no-actuation"
        or receipt.get("config_sha256") != config["config_sha256"]
        or receipt.get("data_identity_sha256")
        != config["data_parent"]["data_identity_sha256"]
        or receipt.get("checkpoint_count") != 8
        or receipt.get("paired_epoch_sample_orders_equal") is not True
        or receipt.get("model_epoch_count") != 40
        or receipt.get("optimizer_example_consumption_total") != 176000
        or receipt.get("optimizer_steps_total") != 5520
    ):
        raise RuntimeError("G2C training receipt identity/status 漂移")
    zero_fields = (
        "model_val_label_bundle_open_count",
        "calibration_label_bundle_open_count",
        "qualification_label_bundle_open_count",
        "test_array_read_count",
        "memory_read_count",
        "memory_write_count",
        "runtime_camera_actuation_count",
        "physical_camera_actuation_count",
        "arm_motion_command_count",
        "gripper_close_command_count",
        "manipulation_progression_count",
    )
    if any(receipt.get(name) != 0 for name in zero_fields):
        raise RuntimeError("G2C training forbidden read/actuation counter 漂移")
    if receipt.get("training_privileged_label_bundle_open_count") != 400:
        raise RuntimeError("G2C training train-only label bundle count 漂移")
    checkpoint_inventory = receipt.get("checkpoint_inventory")
    if not isinstance(checkpoint_inventory, list) or len(checkpoint_inventory) != 8:
        raise RuntimeError("G2C training checkpoint inventory 必须有 8 项")
    artifact_names = _g2c_training_artifact_names(checkpoint_inventory)
    artifacts = receipt.get("artifact_sha256")
    if not isinstance(artifacts, dict) or set(artifacts) != set(artifact_names):
        raise RuntimeError("G2C training artifact inventory 必须等于冻结文件白名单")
    for name, expected_sha in artifacts.items():
        _require_sha256(expected_sha, f"G2C training artifact {name}")
        path = root / name
        if path.is_symlink() or file_sha256(path) != expected_sha:
            raise RuntimeError(f"G2C training artifact SHA 漂移: {name}")
    if _read_json(root / "config_snapshot.json", "G2C training config snapshot") != config:
        raise RuntimeError("G2C training config snapshot 漂移")
    source = _read_json(root / "source_identity.json", "G2C training source identity")
    if (
        source.get("identity_sha256") != canonical_sha256(
            {
                "git_commit": source.get("git_commit"),
                "source_tree_sha256": source.get("source_tree_sha256"),
            }
        )
        or receipt.get("source_git_commit") != source.get("git_commit")
        or receipt.get("source_identity_sha256") != source.get("identity_sha256")
    ):
        raise RuntimeError("G2C training source identity 漂移")
    expected_pairs = [
        (candidate_id, epoch)
        for candidate_id in G2C_CANDIDATE_IDS
        for epoch in G2C_CANDIDATE_EPOCHS
    ]
    if [
        (item.get("candidate_id"), item.get("epoch")) for item in checkpoint_inventory
    ] != expected_pairs:
        raise RuntimeError("G2C training checkpoint candidate/epoch order 漂移")
    verified_checkpoints: list[dict[str, Any]] = []
    companion_states: dict[tuple[str, int], dict[str, Any]] = {}
    for item in checkpoint_inventory:
        candidate_id = str(item["candidate_id"])
        checkpoint = root / "candidates" / candidate_id / str(item["relative_path"])
        if checkpoint.name != item["relative_path"] or checkpoint.is_symlink():
            raise RuntimeError("G2C checkpoint relative path 漂移")
        loaded = load_precision_checkpoint(
            checkpoint,
            expected_checkpoint_sha256=item["checkpoint_sha256"],
            expected_provenance_sha256=item["provenance_sha256"],
            expected_role=PrecisionCheckpointRole.FORMAL_TRAINING,
        )
        if (
            loaded.receipt.to_dict()
            != {
                name: item[name]
                for name in (
                    "format_version",
                    "checkpoint_sha256",
                    "parameter_state_sha256",
                    "model_config_sha256",
                    "provenance_sha256",
                )
            }
            or loaded.provenance.to_dict() != item["provenance"]
            or loaded.provenance.training_config_sha256 != config["config_sha256"]
            or loaded.provenance.data_identity_sha256
            != config["data_parent"]["data_identity_sha256"]
            or loaded.provenance.source_tree_sha256 != source["source_tree_sha256"]
            or loaded.provenance.seed
            != G2C_CANDIDATE_INITIALIZATION_SEEDS[candidate_id]
        ):
            raise RuntimeError("G2C checkpoint receipt/provenance 漂移")
        training_state_path = (
            root
            / "candidates"
            / candidate_id
            / str(item.get("training_state_relative_path"))
        )
        if (
            training_state_path.name != item.get("training_state_relative_path")
            or training_state_path.is_symlink()
            or file_sha256(training_state_path)
            != item.get("training_state_raw_sha256")
        ):
            raise RuntimeError("G2C checkpoint companion training-state 文件漂移")
        training_state = torch.load(
            training_state_path, map_location="cpu", weights_only=True
        )
        if (
            not isinstance(training_state, dict)
            or set(training_state) != _TRAINING_STATE_KEYS
        ):
            raise RuntimeError("G2C checkpoint companion training-state keys 漂移")
        training_state_identity = training_state.get("state_identity_sha256")
        training_state_unsigned = {
            name: value
            for name, value in training_state.items()
            if name != "state_identity_sha256"
        }
        if (
            training_state_identity != item.get("training_state_identity_sha256")
            or _state_identity_sha256(training_state_unsigned)
            != training_state_identity
            or training_state.get("candidate_id") != candidate_id
            or training_state.get("epoch") != item["epoch"]
            or training_state.get("config_sha256") != config["config_sha256"]
            or training_state.get("source_identity_sha256")
            != source["identity_sha256"]
            or training_state.get("checkpoint_sha256")
            != item["checkpoint_sha256"]
            or training_state.get("checkpoint_parameter_sha256")
            != item["parameter_state_sha256"]
            or training_state.get("checkpoint_provenance_sha256")
            != item["provenance_sha256"]
            or training_state.get("examples_seen") != int(item["epoch"]) * 4400
            or training_state.get("optimizer_steps") != int(item["epoch"]) * 138
            or training_state.get("version")
            != E018_P1_G2C_FORMAL_TRAIN_RESULT_VERSION
            or training_state.get("kind")
            != "immutable-checkpoint-training-state/v1"
            or training_state.get("gpu_budget_hours")
            != config["protocol"]["budgets"]["gpu_hours_max"]
            or not math.isfinite(
                float(training_state.get("active_gpu_elapsed_s_at_checkpoint"))
            )
            or float(training_state["active_gpu_elapsed_s_at_checkpoint"]) < 0.0
            or float(training_state["active_gpu_elapsed_s_at_checkpoint"])
            > config["protocol"]["budgets"]["gpu_hours_max"] * 3600.0
            or _state_identity_sha256(training_state["optimizer_state"])
            != item.get("optimizer_state_identity_sha256")
            or _state_identity_sha256(training_state["scheduler_state"])
            != item.get("scheduler_state_identity_sha256")
            or _rng_state_sha256(training_state["rng_state"])
            != item.get("rng_state_identity_sha256")
            or _tensor_sha256(training_state["rng_state"]["sampler_generator"])
            != item.get("sampler_state_identity_sha256")
            or training_state.get("active_gpu_elapsed_s_at_checkpoint")
            != item.get("active_gpu_elapsed_s_at_checkpoint")
            or canonical_sha256(
                {
                    "active_gpu_elapsed_s_at_checkpoint": training_state[
                        "active_gpu_elapsed_s_at_checkpoint"
                    ],
                    "gpu_budget_hours": training_state["gpu_budget_hours"],
                }
            )
            != item.get("budget_timing_identity_sha256")
        ):
            raise RuntimeError("G2C checkpoint companion optimizer/scheduler/RNG/sampler 漂移")
        companion_states[(candidate_id, int(item["epoch"]))] = training_state
        verified_checkpoints.append(
            {
                "candidate_id": candidate_id,
                "epoch": int(item["epoch"]),
                "checkpoint_sha256": loaded.receipt.checkpoint_sha256,
                "parameter_state_sha256": loaded.receipt.parameter_state_sha256,
                "provenance_sha256": loaded.receipt.provenance_sha256,
                "model_config_sha256": loaded.receipt.model_config_sha256,
                "training_state_raw_sha256": item["training_state_raw_sha256"],
                "training_state_identity_sha256": training_state_identity,
                "optimizer_state_identity_sha256": item[
                    "optimizer_state_identity_sha256"
                ],
                "scheduler_state_identity_sha256": item[
                    "scheduler_state_identity_sha256"
                ],
                "rng_state_identity_sha256": item["rng_state_identity_sha256"],
                "sampler_state_identity_sha256": item[
                    "sampler_state_identity_sha256"
                ],
            }
        )
    traces: dict[str, list[dict[str, Any]]] = {}
    for candidate_id in G2C_CANDIDATE_IDS:
        trace = _read_json_array(
            root / "candidates" / candidate_id / "epoch_trace.json",
            f"G2C {candidate_id} epoch trace",
        )
        if (
            len(trace) != 20
            or [row.get("epoch") for row in trace] != list(range(1, 21))
            or any(row.get("candidate_id") != candidate_id for row in trace)
            or any(row.get("sample_count") != 4400 for row in trace)
            or any(row.get("batch_count") != 138 for row in trace)
            or any(row.get("motion_head_parameter_sha256") != trace[0]["motion_head_parameter_sha256"] for row in trace)
        ):
            raise RuntimeError(f"G2C {candidate_id} epoch trace 漂移")
        for row in trace:
            _require_exact_keys(
                row,
                _EPOCH_TRACE_KEYS,
                "G2C epoch trace row",
            )
            for name in (
                "sample_order_sha256",
                "sampler_generator_state_before_sha256",
                "sampler_generator_state_after_sha256",
                "parameter_state_sha256",
                "motion_head_parameter_sha256",
                "optimizer_state_identity_sha256",
                "scheduler_state_identity_sha256",
                "rng_state_identity_sha256",
            ):
                _require_sha256(row.get(name), f"G2C trace {name}")
            losses = row.get("loss")
            if (
                not isinstance(losses, dict)
                or set(losses) != set(_LOSS_COMPONENT_NAMES)
                or any(not math.isfinite(float(value)) for value in losses.values())
            ):
                raise RuntimeError("G2C epoch trace loss 漂移")
            epoch = int(row["epoch"])
            if (
                row.get("examples_seen_total") != epoch * 4400
                or row.get("optimizer_steps_total") != epoch * 138
                or not math.isfinite(float(row.get("maximum_gradient_norm_pre_clip")))
                or float(row["maximum_gradient_norm_pre_clip"]) < 0.0
                or not math.isfinite(float(row.get("maximum_gradient_norm_post_clip")))
                or not 0.0
                <= float(row["maximum_gradient_norm_post_clip"])
                <= config["protocol"]["optimizer"]["gradient_clip_norm"] + 1e-5
                or not math.isfinite(
                    float(row.get("learning_rate_after_scheduler_step"))
                )
                or float(row["learning_rate_after_scheduler_step"]) < 0.0
            ):
                raise RuntimeError("G2C epoch trace count/gradient/scheduler 漂移")
        traces[candidate_id] = trace
        resume_path = root / "candidates" / candidate_id / "resume_state.pt"
        resume_state = torch.load(resume_path, map_location="cpu", weights_only=True)
        terminal_item = next(
            item
            for item in checkpoint_inventory
            if item["candidate_id"] == candidate_id and item["epoch"] == 20
        )
        terminal_companion = companion_states[(candidate_id, 20)]
        initialization = _read_json(
            root / "candidates" / candidate_id / "initialization.json",
            f"G2C {candidate_id} initialization",
        )
        budget_state = _read_json(
            root / "candidates" / candidate_id / "budget_state.json",
            f"G2C {candidate_id} budget state",
        )
        if isinstance(resume_state, dict):
            _validate_resume_progress_semantics(
                state=resume_state,
                candidate_id=candidate_id,
                config=config,
                source_identity=source,
            )
        if (
            not isinstance(resume_state, dict)
            or set(resume_state) != _RESUME_STATE_KEYS
            or resume_state.get("version")
            != E018_P1_G2C_FORMAL_TRAIN_RESULT_VERSION
            or resume_state.get("completed_epoch") != 20
            or resume_state.get("candidate_id") != candidate_id
            or resume_state.get("config_sha256") != config["config_sha256"]
            or resume_state.get("source_identity_sha256") != source["identity_sha256"]
            or resume_state.get("examples_seen") != 88000
            or resume_state.get("optimizer_steps") != 2760
            or resume_state.get("epoch_trace") != trace
            or resume_state.get("checkpoint_inventory")
            != [item for item in checkpoint_inventory if item["candidate_id"] == candidate_id]
            or resume_state.get("initialization") != initialization
            or precision_parameter_state_sha256(resume_state["model_state"])
            != terminal_item["parameter_state_sha256"]
            or _state_identity_sha256(resume_state["optimizer_state"])
            != terminal_item["optimizer_state_identity_sha256"]
            or _state_identity_sha256(resume_state["optimizer_state"])
            != _state_identity_sha256(terminal_companion["optimizer_state"])
            or _state_identity_sha256(resume_state["scheduler_state"])
            != terminal_item["scheduler_state_identity_sha256"]
            or _state_identity_sha256(resume_state["scheduler_state"])
            != _state_identity_sha256(terminal_companion["scheduler_state"])
            or _rng_state_sha256(resume_state["rng_state"])
            != terminal_item["rng_state_identity_sha256"]
            or _rng_state_sha256(resume_state["rng_state"])
            != _rng_state_sha256(terminal_companion["rng_state"])
            or _tensor_sha256(resume_state["rng_state"]["sampler_generator"])
            != terminal_item["sampler_state_identity_sha256"]
            or trace[-1]["optimizer_state_identity_sha256"]
            != terminal_item["optimizer_state_identity_sha256"]
            or trace[-1]["scheduler_state_identity_sha256"]
            != terminal_item["scheduler_state_identity_sha256"]
            or trace[-1]["rng_state_identity_sha256"]
            != terminal_item["rng_state_identity_sha256"]
            or not math.isfinite(float(resume_state.get("active_gpu_elapsed_s", math.nan)))
            or float(resume_state["active_gpu_elapsed_s"])
            < float(terminal_item["active_gpu_elapsed_s_at_checkpoint"])
            or float(resume_state["active_gpu_elapsed_s"])
            > float(budget_state.get("active_gpu_elapsed_s", -1.0))
            or float(resume_state["active_gpu_elapsed_s"])
            > config["protocol"]["budgets"]["gpu_hours_max"] * 3600.0
        ):
            raise RuntimeError("G2C final resume state identity 漂移")
        _require_exact_keys(
            budget_state,
            _BUDGET_STATE_KEYS,
            "G2C final budget state",
        )
        if (
            budget_state.get("version")
            != E018_P1_G2C_FORMAL_TRAIN_RESULT_VERSION
            or budget_state.get("candidate_id") != candidate_id
            or budget_state.get("config_sha256") != config["config_sha256"]
            or budget_state.get("source_identity_sha256") != source["identity_sha256"]
            or budget_state.get("last_fully_resumable_epoch") != 20
            or budget_state.get("active_attempt_epoch") is not None
            or budget_state.get("active_attempt_batch_count") != 0
            or budget_state.get("gpu_budget_hours")
            != config["protocol"]["budgets"]["gpu_hours_max"]
            or float(budget_state.get("active_gpu_elapsed_s", -1.0)) < 0.0
            or float(budget_state.get("active_gpu_elapsed_s", math.inf))
            > config["protocol"]["budgets"]["gpu_hours_max"] * 3600.0
        ):
            raise RuntimeError("G2C cumulative active GPU budget state 漂移")
    if any(
        traces["W-KV0"][index]["sample_order_sha256"]
        != traces["S"][index]["sample_order_sha256"]
        for index in range(20)
    ):
        raise RuntimeError("G2C paired sample order 漂移")
    artifact_bytes_before_receipt = sum(
        (root / name).stat().st_size for name in artifact_names
    )
    if (
        receipt.get("artifact_bytes_before_receipt")
        != artifact_bytes_before_receipt
        or artifact_bytes_before_receipt
        > config["protocol"]["budgets"]["artifact_bytes_max"]
    ):
        raise RuntimeError("G2C training artifact byte budget 漂移")
    run_state = _read_json(root / "run_state.json", "G2C final training run state")
    _require_exact_keys(
        run_state,
        {"version", "status", "config_sha256", "receipt_sha256"},
        "G2C final training run state",
    )
    if run_state != {
        "version": E018_P1_G2C_FORMAL_TRAIN_RESULT_VERSION,
        "status": "complete-formal-training-pass",
        "config_sha256": config["config_sha256"],
        "receipt_sha256": internal,
    }:
        raise RuntimeError("G2C final training run state 漂移")
    total_artifact_bytes = _verify_exact_regular_file_tree(
        root,
        expected_files=set(artifact_names)
        | {"training_receipt.json", "run_state.json"},
        name="G2C formal TRAIN",
    )
    if total_artifact_bytes > config["protocol"]["budgets"]["artifact_bytes_max"]:
        raise RuntimeError("G2C training 完整 artifact byte budget 漂移")
    result = {
        "version": E018_P1_G2C_FORMAL_TRAIN_RESULT_VERSION,
        "status": receipt["status"],
        "verified": True,
        "config_sha256": config["config_sha256"],
        "data_identity_sha256": receipt["data_identity_sha256"],
        "source_git_commit": receipt["source_git_commit"],
        "source_identity_sha256": receipt["source_identity_sha256"],
        "receipt_raw_sha256": file_sha256(receipt_path),
        "receipt_internal_sha256": internal,
        "checkpoint_count": 8,
        "checkpoint_inventory_sha256": canonical_sha256(verified_checkpoints),
        "paired_epoch_sample_orders_equal": True,
        "total_artifact_bytes": total_artifact_bytes,
        "model_val_label_bundle_open_count": 0,
        "test_array_read_count": 0,
    }
    result["verification_sha256"] = canonical_sha256(result)
    return result
