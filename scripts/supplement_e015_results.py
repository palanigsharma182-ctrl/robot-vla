"""确定性修正 E015 memory availability 聚合并生成 V1.0 瓶颈报告。

本脚本只读取已冻结的逐帧 JSONL，不加载 checkpoint、不执行模型 forward、不改变
validation 规则，也不重新消费 fresh test。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

E015_EXPERIMENT_VERSION = "e015-precision-goal-memory/v1"
E015_PUBLIC_VERSION = "e015-precision-goal-memory-public/v1"
SUPPLEMENT_VERSION = "e015-precision-memory-supplement/v1"
SUPPLEMENT_POLICY = "deterministic-postprocessing-no-model-forward/v1"
EVALUATION_PROTOCOL_SCOPE = {
    "all_split_integrity_audit_before_calibration": True,
    "integrity_audit_used_test_predictions": False,
    "test_used_for_rule_selection": False,
    "test_model_forward_evaluation_count": 1,
    "test_once_claim_scope": "model-forward-and-shadow-replay/v1",
}
SUPPLEMENT_SOURCE_FILES = (
    "scripts/supplement_e015_results.py",
    "scripts/verify_e015_public_results.py",
    "src/robot_vla/precision/memory_evaluation.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} 必须是 JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} 必须是 JSON object")
            rows.append(value)
    return rows


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_receipt(root: Path, *, version: str) -> dict[str, Any]:
    receipt = _read_json(root / "receipt.json")
    if receipt.get("version") != version or receipt.get("status") != "complete":
        raise RuntimeError(f"原始 E015 receipt 状态或版本漂移: {root}")
    for name, expected in receipt.get("files", {}).items():
        if _sha256(root / name) != expected:
            raise RuntimeError(f"原始 E015 文件 SHA-256 漂移: {name}")
    return receipt


def _supplement_source_sha256(repository_root: Path) -> str:
    files: dict[str, str] = {}
    for relative in SUPPLEMENT_SOURCE_FILES:
        path = repository_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"E015 supplement source 不存在: {path}")
        files[relative] = _sha256(path)
    return _canonical_sha256({"version": SUPPLEMENT_VERSION, "files": files})


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile values 不能为空")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _availability_audit(
    rows: list[dict[str, Any]],
    *,
    expected_summary: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, int]]:
    expected_frames = int(expected_summary["frame_count"])
    if len(rows) != expected_frames:
        raise RuntimeError("E015 memory replay frame count 与原始 summary 不一致")
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_episode[str(row["episode_id"])].append(row)
    initialized = {
        episode_id
        for episode_id, episode_rows in by_episode.items()
        if any(bool(row["memory_measurement_accepted"]) for row in episode_rows)
    }
    unobservable = [row for row in rows if not bool(row["gt_observable"])]
    valid_unobservable = [row for row in unobservable if bool(row["memory_valid"])]
    unavailable_unobservable = [
        row for row in unobservable if not bool(row["memory_valid"])
    ]
    uninitialized_unobservable = [
        row for row in unavailable_unobservable if row["memory_age_s"] is None
    ]
    previously_initialized_invalid = [
        row for row in unavailable_unobservable if row["memory_age_s"] is not None
    ]
    initialized_rows = [row for row in rows if str(row["episode_id"]) in initialized]
    initialized_unobservable = [
        row for row in initialized_rows if not bool(row["gt_observable"])
    ]
    initialized_memory_valid = [
        row for row in initialized_rows if bool(row["memory_valid"])
    ]
    initialized_memory_valid_unobservable = [
        row
        for row in initialized_unobservable
        if bool(row["memory_valid"])
    ]
    expected_unobservable = int(expected_summary["gt_unobservable_count"])
    expected_valid = int(
        expected_summary["memory_valid_while_gt_unobservable_count"]
    )
    if len(unobservable) != expected_unobservable or len(valid_unobservable) != expected_valid:
        raise RuntimeError("E015 availability supplement 与 canonical 聚合不一致")
    audit = {
        "episode_count": len(by_episode),
        "initialized_episode_count": len(initialized),
        "never_initialized_episode_count": len(by_episode) - len(initialized),
        "initialized_episode_frame_count": len(initialized_rows),
        "initialized_episode_unobservable_frame_count": len(initialized_unobservable),
        "initialized_episode_memory_valid_frame_count": len(initialized_memory_valid),
        "initialized_episode_memory_valid_while_gt_unobservable_count": len(
            initialized_memory_valid_unobservable
        ),
        "memory_unavailable_while_gt_unobservable_count": len(
            unavailable_unobservable
        ),
        "memory_uninitialized_while_gt_unobservable_count": len(
            uninitialized_unobservable
        ),
        "memory_previously_initialized_but_invalid_while_gt_unobservable_count": len(
            previously_initialized_invalid
        ),
        "memory_coverage_within_initialized_episodes": (
            len(initialized_memory_valid) / len(initialized_rows)
            if initialized_rows
            else 0.0
        ),
        "memory_unobservable_coverage_within_initialized_episodes": (
            len(initialized_memory_valid_unobservable) / len(initialized_unobservable)
            if initialized_unobservable
            else 0.0
        ),
    }
    corrected_fields = {
        "stale_or_uninitialized_occluded_count": len(unavailable_unobservable),
        "memory_unavailable_while_gt_unobservable_count": len(
            unavailable_unobservable
        ),
        "memory_uninitialized_while_gt_unobservable_count": len(
            uninitialized_unobservable
        ),
        "memory_previously_initialized_but_invalid_while_gt_unobservable_count": len(
            previously_initialized_invalid
        ),
    }
    return audit, corrected_fields


def _unsafe_write_audit(
    rows: list[dict[str, Any]],
    *,
    threshold: float,
    expected_count: int,
) -> dict[str, Any]:
    accepted = [
        row
        for row in rows
        if bool(row["write_evidence"]["structurally_eligible"])
        and float(row["write_evidence"]["score"]) >= threshold
    ]
    unsafe = [row for row in accepted if not bool(row["oracle_safe_measurement"])]
    if len(unsafe) != expected_count:
        raise RuntimeError("E015 unsafe-write supplement 与 canonical 聚合不一致")
    if not unsafe:
        return {
            "count": 0,
            "rate_among_accepted": 0.0,
            "rate_overall": 0.0,
        }
    errors_mm = [float(row["world_xy_error_m"]) * 1000.0 for row in unsafe]
    margins = [float(row["write_evidence"]["score"]) - threshold for row in unsafe]
    visibility = [
        float(row["write_evidence"]["visibility_probability"]) for row in unsafe
    ]
    local_visible = [
        float(row["observability"]["local_goal_visible_fraction"])
        for row in unsafe
    ]
    occlusion = Counter(str(row["observability"]["occlusion_type"]) for row in unsafe)
    return {
        "count": len(unsafe),
        "rate_among_accepted": len(unsafe) / len(accepted),
        "rate_overall": len(unsafe) / len(rows),
        "catastrophic_count": sum(error > 20.0 for error in errors_mm),
        "world_xy_error_p50_mm": _quantile(errors_mm, 0.50),
        "world_xy_error_p90_mm": _quantile(errors_mm, 0.90),
        "world_xy_error_max_mm": max(errors_mm),
        "score_margin_over_threshold_min": min(margins),
        "score_margin_over_threshold_max": max(margins),
        "visibility_probability_min": min(visibility),
        "visibility_probability_max": max(visibility),
        "local_goal_visible_fraction_min": min(local_visible),
        "local_goal_visible_fraction_max": max(local_visible),
        "legacy_visible_count": sum(
            bool(row["observability"]["legacy_visible"]) for row in unsafe
        ),
        "center_inside_goal_mask_count": sum(
            bool(row["observability"]["center_inside_goal_mask"]) for row in unsafe
        ),
        "center_inside_object_mask_count": sum(
            bool(row["observability"]["center_inside_object_mask"]) for row in unsafe
        ),
        "occlusion_type_counts": dict(sorted(occlusion.items())),
    }


def _readme(summary: dict[str, Any]) -> str:
    audit = summary["e015_a_observability_audit"]
    write = summary["e015_b_write_measurements"]
    memory = summary["e015_b_memory_replay"]
    availability = summary["memory_availability_audit"]
    unsafe = summary["unsafe_write_audit"]
    calibration = summary["write_calibration"]
    mismatch_rate = (
        audit["legacy_contract_mismatch_count"] / audit["legacy_goal_visible_count"]
    )
    return f"""# E015 — Explicit geometric goal state memory

E015-A 与 E015-B 使用 frozen E013 checkpoint 和全新 seeds 完成；没有训练、没有修改
checkpoint、没有 Action 输出，也没有 actuator。fresh validation 只用于冻结 write threshold 与
memory age，fresh test 只评估一次。

术语边界：数据生成后、calibration 前会对所有 split 做 schema、文件 identity 和 oracle
round-trip 完整性 audit；该 audit 不产生 test prediction，也不参与 threshold/age 选择。test-once
claim 的精确范围是 **U-Net model forward 与 shadow replay**，这部分只执行一次。

## V1.0 瓶颈判断

**显式 base-frame memory 的状态保持机制成立，但 E015 工程 gate 未通过。** memory 能在当前
RGB 不可观察时安全保留历史 goal，未产生 catastrophic state；真正限制 V1.0 的是可靠
measurement 的准入和 Episode 初始化，而不是 memory 的 hold 逻辑，也不是主体分布的毫米级定位。

- 100 个 test Episode 中只有 `{availability["initialized_episode_count"]}` 个得到过可靠 write，
  `{availability["never_initialized_episode_count"]}` 个始终没有初始化 memory。
- 当前帧 measurement coverage 为 `{memory["current_measurement_coverage"]:.4%}`，memory coverage
  提升到 `{memory["memory_coverage"]:.4%}`。
- GT 不可观察时，当前 measurement 有效 `{memory["current_valid_while_gt_unobservable_count"]}` 帧，
  memory 有效 `{memory["memory_valid_while_gt_unobservable_count"]}` 帧；在已初始化 Episode 内的
  遮挡覆盖为 `{availability["memory_unobservable_coverage_within_initialized_episodes"]:.4%}`。
- memory world-XY p90/max 为 `{memory["memory_error"]["p90_mm"]:.3f}` /
  `{memory["memory_error"]["max_mm"]:.3f} mm`，catastrophic=`{memory["memory_catastrophic_count"]}`，
  Episode reset leakage=`{memory["episode_reset_leakage_count"]}`。

## E015-A：observability contract

- test frames：`{audit["frame_count"]}`；goal exists/projected：
  `{audit["goal_exists_count"]}` / `{audit["goal_projection_valid_count"]}`；真正 observable：
  `{audit["goal_observable_count"]}`。
- legacy visible 为 `{audit["legacy_goal_visible_count"]}`，其中 `{audit["legacy_contract_mismatch_count"]}`
  帧（`{mismatch_rate:.2%}`）满足“mask 还有像素”，但 projected center 已无直接视觉证据。
- 不可观察主要由 out-of-frame `{audit["occlusion_type_counts"].get("out_of_frame", 0)}`、
  object occlusion `{audit["occlusion_type_counts"].get("object_occlusion", 0)}` 和其他遮挡/背景
  `{audit["occlusion_type_counts"].get("other_occlusion_or_background", 0)}` 构成。

这证实 `exists / projected / observable` 不能继续混用；旧 keypoint-visible 标签会把一部分遮挡帧
当作普通 goal-center supervision。

## E015-B：write gate 与 memory

validation 上冻结的 threshold 为 `{calibration["threshold"]:.9f}`，接受
`{calibration["accepted_count"]}` 帧、unsafe=`{calibration["accepted_unsafe_count"]}`，安全
measurement coverage 仅 `{calibration["safe_coverage"]:.4%}`。test 上接受 `{write["accepted_count"]}`
帧，其中 unsafe=`{write["accepted_unsafe_count"]}`、catastrophic=`{write["accepted_catastrophic_count"]}`。

两次 unsafe write 都是 strict center-ray contract 下的 `other_occlusion_or_background`：定位误差仍小，
但当前 RGB 没有足够的 goal-center 视觉证据。它们的 score 只比冻结阈值高
`{unsafe["score_margin_over_threshold_min"]:.7f}–{unsafe["score_margin_over_threshold_max"]:.7f}`，
world-XY 最大误差 `{unsafe["world_xy_error_max_mm"]:.3f} mm`。因此失败点不是 20 cm hallucination，
而是单帧 scalar gate 在阈值边界无法保证跨 split 的零 false acceptance。

## 聚合修正

原始 generator 的 `stale_or_uninitialized_occluded_count` 错误地排除了 `memory_age_s=null` 的
未初始化帧，因此从 `0` 确定性修正为
`{memory["stale_or_uninitialized_occluded_count"]}`。该修正只重算既有 JSONL，未执行模型 forward、
未改规则、未重新读取 test 模型输出；canonical private receipt 与 test-once claim 均保留。

## 下一步边界

1. 先修训练/评估 label contract：不可观察 goal 不再作为普通 keypoint supervision，并单独训练、校准
   observability；必须使用新的 validation/test seeds 验证。
2. 提高安全初始化覆盖：优先加入部署可得的多帧一致性与明确初始化阶段；若 wrist 在任务早期仍看不到
   goal，再通过统一 base-frame 接口接入 external-camera measurement。
3. write authorization 继续 fail closed：使用时间一致性、innovation、workspace 和 mask-support margin 的
   联合门禁，而不是只依赖一个单帧 scalar score；新规则仍需预注册并在 fresh test 上验证。
4. 在 unsafe write=0、catastrophic=0、初始化覆盖达到新门槛之前，不接 controller/actuator；本次结果
   `safe_for_actuator_promotion=false`。

公开目录只含脱敏聚合与 SHA-256。逐帧身份、位置、mask/RGB 和私有路径未发布。
"""


def supplement(args: argparse.Namespace) -> dict[str, Any]:
    original_private = args.original_private_root.resolve()
    original_public = args.original_public_root.resolve()
    output_private = args.output_private_root.resolve()
    output_public = args.output_public_root.resolve()
    repository_root = args.repository_root.resolve()
    if output_private.exists() or output_public.exists():
        raise FileExistsError("E015 supplemented 输出已存在，拒绝覆盖")

    _verify_receipt(
        original_private,
        version=E015_EXPERIMENT_VERSION,
    )
    _verify_receipt(
        original_public,
        version=E015_PUBLIC_VERSION,
    )
    subprocess.run(
        [
            sys.executable,
            str(original_public / "verify_summary.py"),
            str(original_public),
        ],
        cwd=repository_root,
        check=True,
    )
    original_private_summary = _read_json(original_private / "summary_private.json")
    original_public_summary = _read_json(original_public / "sanitized_summary.json")
    if original_private_summary.get("test_split_status_after_e015") != (
        "consumed-for-evaluation"
    ):
        raise RuntimeError("E015 private test 状态漂移")

    memory_rows = _read_jsonl(original_private / "memory_replay.jsonl")
    prediction_rows = _read_jsonl(original_private / "prediction_rows.jsonl")
    canonical_memory = original_public_summary["e015_b_memory_replay"]
    availability, corrected_fields = _availability_audit(
        memory_rows,
        expected_summary=canonical_memory,
    )
    threshold = float(original_public_summary["write_calibration"]["threshold"])
    expected_unsafe = int(
        original_public_summary["e015_b_write_measurements"]["accepted_unsafe_count"]
    )
    unsafe = _unsafe_write_audit(
        prediction_rows,
        threshold=threshold,
        expected_count=expected_unsafe,
    )
    supplement_source = _supplement_source_sha256(repository_root)
    original_public_receipt_sha256 = _sha256(original_public / "receipt.json")
    original_private_receipt_sha256 = _sha256(original_private / "receipt.json")

    public_summary = copy.deepcopy(original_public_summary)
    original_aggregate_value = int(
        public_summary["e015_b_memory_replay"][
            "stale_or_uninitialized_occluded_count"
        ]
    )
    public_summary["e015_b_memory_replay"].update(corrected_fields)
    public_summary.update(
        {
            "status": "complete-supplemented",
            "supplement_version": SUPPLEMENT_VERSION,
            "supplement_policy": SUPPLEMENT_POLICY,
            "supplement_source_tree_sha256": supplement_source,
            "original_public_receipt_sha256": original_public_receipt_sha256,
            "evaluation_protocol_scope": EVALUATION_PROTOCOL_SCOPE,
            "memory_availability_audit": availability,
            "unsafe_write_audit": unsafe,
            "aggregation_correction": {
                "field": "stale_or_uninitialized_occluded_count",
                "original_value": original_aggregate_value,
                "corrected_value": corrected_fields[
                    "stale_or_uninitialized_occluded_count"
                ],
                "cause": "age-none-uninitialized-rows-were-excluded/v1",
                "model_forward_repeated": False,
                "test_rules_changed": False,
            },
        }
    )

    private_summary = {
        "version": E015_EXPERIMENT_VERSION,
        "status": "complete-supplemented",
        "supplement_version": SUPPLEMENT_VERSION,
        "supplement_policy": SUPPLEMENT_POLICY,
        "supplement_source_tree_sha256": supplement_source,
        "original_private_receipt_sha256": original_private_receipt_sha256,
        "original_public_receipt_sha256": original_public_receipt_sha256,
        "rules_sha256": original_private_summary["rules_sha256"],
        "test_once_claim_sha256": original_private_summary["test_once_claim_sha256"],
        "test_split_status_after_e015": "consumed-for-evaluation",
        "evaluation_protocol_scope": EVALUATION_PROTOCOL_SCOPE,
        "memory_availability_audit": availability,
        "unsafe_write_audit": unsafe,
        "aggregation_correction": public_summary["aggregation_correction"],
        "model_forward_repeated": False,
        "actuation_allowed": False,
    }
    output_private.mkdir(mode=0o700, parents=True, exist_ok=False)
    _atomic_json(output_private / "summary_private_supplement.json", private_summary)
    private_files = {
        "summary_private_supplement.json": _sha256(
            output_private / "summary_private_supplement.json"
        )
    }
    supplement_private_receipt = {
        "version": E015_EXPERIMENT_VERSION,
        "status": "complete-supplemented",
        "supplement_version": SUPPLEMENT_VERSION,
        "original_private_receipt_sha256": original_private_receipt_sha256,
        "files": private_files,
        "contains_trajectory_identity": False,
        "contains_raw_rgb": False,
        "contains_raw_heatmaps": False,
    }
    _atomic_json(output_private / "receipt.json", supplement_private_receipt)

    output_public.mkdir(mode=0o755, parents=True, exist_ok=False)
    _atomic_json(output_public / "sanitized_summary.json", public_summary)
    _atomic_text(output_public / "README.md", _readme(public_summary))
    shutil.copyfile(
        repository_root / "scripts" / "verify_e015_public_results.py",
        output_public / "verify_summary.py",
    )
    public_files = {
        name: _sha256(output_public / name)
        for name in ("README.md", "sanitized_summary.json", "verify_summary.py")
    }
    supplemented_receipt = {
        "version": E015_PUBLIC_VERSION,
        "status": "complete-supplemented",
        "supplement_version": SUPPLEMENT_VERSION,
        "supplement_policy": SUPPLEMENT_POLICY,
        "supplement_source_tree_sha256": supplement_source,
        "original_private_receipt_sha256": original_private_receipt_sha256,
        "original_public_receipt_sha256": original_public_receipt_sha256,
        "supplement_private_receipt_sha256": _sha256(output_private / "receipt.json"),
        "rules_sha256": public_summary["rules_sha256"],
        "files": public_files,
        "contains_raw_rgb": False,
        "contains_raw_heatmaps": False,
        "contains_trajectory_identity": False,
        "contains_model_weights": False,
        "contains_sensitive_paths": False,
    }
    _atomic_json(output_public / "receipt.json", supplemented_receipt)
    subprocess.run(
        [sys.executable, str(output_public / "verify_summary.py"), str(output_public)],
        cwd=repository_root,
        check=True,
    )
    return public_summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-private-root", type=Path, required=True)
    parser.add_argument("--original-public-root", type=Path, required=True)
    parser.add_argument("--output-private-root", type=Path, required=True)
    parser.add_argument("--output-public-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    result = supplement(_parse_args())
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
