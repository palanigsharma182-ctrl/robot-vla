"""Observation V2 的在线目标选择与相对几何 Layer-12 smoke probe。"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from robot_vla.contracts import (
    OBSERVATION_HISTORY_LENGTH,
    OBSERVATION_V2_VERSION,
    PICK_AND_PLACE_SKILLS,
    RobotSpec,
)
from robot_vla.data.trajectory import TrajectoryMeta, TrajectoryStore
from robot_vla.diagnostics.qwen_spatial_probe import (
    project_world_point_to_gl_camera,
    summarize_spatial_predictions,
    unproject_gl_camera_to_world_plane,
)
from robot_vla.model.qwen_processor import QwenVLAProcessorAdapter

V2_ONLINE_GEOMETRY_PROBE_FORMAT = "robot-vla-v2-online-geometry-probe/v1"
ONLINE_GEOMETRY_TARGETS = ("object", "goal")
REACH_SKILL_ID = PICK_AND_PLACE_SKILLS.index("reach")
V2_IMAGES_PER_SAMPLE = OBSERVATION_HISTORY_LENGTH * 2
# 历史顺序固定为 t-3 external/wrist ... t external/wrist。
CURRENT_EXTERNAL_IMAGE_INDEX = (OBSERVATION_HISTORY_LENGTH - 1) * 2


def _finite_xyz(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} 必须是有限 [3] 米制坐标")
    return result


def _history_source_indices(
    arrays: Any,
    timestep: int,
    spec: RobotSpec,
) -> tuple[np.ndarray, int]:
    source_start = max(0, timestep - OBSERVATION_HISTORY_LENGTH + 1)
    source_indices = np.arange(source_start, timestep + 1, dtype=np.int64)
    destination_start = OBSERVATION_HISTORY_LENGTH - len(source_indices)
    if len(source_indices) > 1:
        expected_dt = 1.0 / spec.control_hz
        delta = np.diff(arrays.timestamp_action[source_indices])
        tolerance = max(1e-6, expected_dt * 0.2)
        if np.any(np.abs(delta - expected_dt) > tolerance):
            raise ValueError("Online geometry probe 的 V2 history 不是连续控制步")
    return source_indices, destination_start


def _complete_v2_history_source_indices(
    arrays: Any,
    timestep: int,
    spec: RobotSpec,
) -> np.ndarray | None:
    """只接受四个连续且所有 V2 模态均有效的历史控制步。"""

    source_indices, destination_start = _history_source_indices(
        arrays,
        timestep,
        spec,
    )
    if (
        destination_start != 0
        or len(source_indices) != OBSERVATION_HISTORY_LENGTH
        or not bool(np.all(arrays.observation_v2_valid[source_indices]))
    ):
        return None
    return source_indices


class QwenV2OnlineGeometryProbeDataset:
    """构造八图 Reach window；GT 只作为 probe 监督和测试指标。"""

    def __init__(
        self,
        root: str | Path,
        entries: Sequence[TrajectoryMeta],
        spec: RobotSpec,
        *,
        cache_size: int = 2,
    ) -> None:
        if not entries:
            raise ValueError("V2 online geometry probe Dataset 需要至少一条轨迹")
        splits = {entry.split for entry in entries}
        if len(splits) != 1:
            raise ValueError("V2 online geometry probe 轨迹必须属于同一个 split")
        self.root = Path(root)
        self.entries = list(entries)
        self.spec = spec
        self.store = TrajectoryStore(self.root, spec, cache_size=cache_size)
        self.index: list[tuple[int, int]] = []
        self.target_uv_external: list[np.ndarray] = []
        self.target_position_world_m: list[np.ndarray] = []
        self.image_size_external: list[tuple[int, int]] = []
        self.intrinsic_external: list[np.ndarray] = []
        self.world_from_external: list[np.ndarray] = []
        self.rejected_missing_v2 = 0
        self.rejected_missing_geometry = 0
        self.rejected_incomplete_history = 0
        self.rejected_out_of_view = 0
        window_ids: list[str] = []

        for entry_index, entry in enumerate(self.entries):
            arrays = self.store.get(entry)
            if (
                entry.randomization.get("observation_contract_version") != OBSERVATION_V2_VERSION
                or not arrays.observation_v2_available
            ):
                self.rejected_missing_v2 += 1
                continue
            if arrays.object_position_m is None:
                self.rejected_missing_geometry += 1
                continue
            try:
                goal_position = _finite_xyz(
                    entry.randomization["goal_position_m"],
                    "goal_position_m",
                )
            except (KeyError, TypeError, ValueError):
                self.rejected_missing_geometry += 1
                continue

            height, width = map(int, arrays.rgb_external.shape[1:3])
            calibration = entry.camera_calibration
            intrinsic = np.asarray(
                calibration.intrinsic_external,
                dtype=np.float32,
            ).reshape(3, 3)
            world_from_camera = np.asarray(
                calibration.world_from_external,
                dtype=np.float32,
            ).reshape(4, 4)
            candidates = np.flatnonzero(
                arrays.observation_v2_valid & (arrays.skill_id == REACH_SKILL_ID)
            )
            for timestep in candidates.tolist():
                if _complete_v2_history_source_indices(arrays, timestep, spec) is None:
                    self.rejected_incomplete_history += 1
                    continue
                targets = np.stack(
                    (
                        _finite_xyz(
                            arrays.object_position_m[timestep],
                            "object_position_m",
                        ),
                        goal_position,
                    )
                ).astype(np.float32)
                projections: list[np.ndarray] = []
                visible = True
                for target in targets:
                    try:
                        projection = project_world_point_to_gl_camera(
                            target,
                            intrinsic.ravel(),
                            world_from_camera.ravel(),
                            height,
                            width,
                        )
                    except ValueError:
                        visible = False
                        break
                    u, v = projection.normalized_uv.tolist()
                    if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
                        visible = False
                        break
                    projections.append(projection.normalized_uv)
                if not visible:
                    self.rejected_out_of_view += 1
                    continue
                self.index.append((entry_index, timestep))
                self.target_uv_external.append(np.stack(projections).astype(np.float32))
                self.target_position_world_m.append(targets)
                self.image_size_external.append((height, width))
                self.intrinsic_external.append(intrinsic.copy())
                self.world_from_external.append(world_from_camera.copy())
                window_ids.append(f"{entry.trajectory_id}:{timestep}")

        if not self.index:
            raise ValueError("V2 online geometry probe 没有 object/goal 同时可见的 Reach window")
        digest = hashlib.sha256()
        for identity in window_ids:
            digest.update(identity.encode("utf-8"))
            digest.update(b"\n")
        self.window_ids = tuple(window_ids)
        self.window_sha256 = digest.hexdigest()

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry_index, timestep = self.index[index]
        entry = self.entries[entry_index]
        arrays = self.store.get(entry)
        source_indices = _complete_v2_history_source_indices(
            arrays,
            timestep,
            self.spec,
        )
        if source_indices is None:
            raise RuntimeError("已索引的 online geometry window 不再具有完整 V2 历史")
        destination_start = 0
        external = np.zeros(
            (OBSERVATION_HISTORY_LENGTH, *arrays.rgb_external.shape[1:]),
            dtype=np.uint8,
        )
        wrist = np.zeros(
            (OBSERVATION_HISTORY_LENGTH, *arrays.rgb_wrist.shape[1:]),
            dtype=np.uint8,
        )
        history_valid = np.zeros(OBSERVATION_HISTORY_LENGTH, dtype=np.bool_)
        for destination, source in enumerate(source_indices, start=destination_start):
            history_valid[destination] = True
            if arrays.external_valid[source]:
                external[destination] = arrays.rgb_external[source]
            if arrays.wrist_valid[source]:
                wrist[destination] = arrays.rgb_wrist[source]
        if not history_valid[-1]:
            raise RuntimeError("V2 online geometry probe 当前控制步必须有效")
        return {
            "rgb_external_history": external,
            "rgb_wrist_history": wrist,
            "history_valid": history_valid,
            "instruction": entry.task.instruction,
            "target_uv_external": self.target_uv_external[index].copy(),
            "target_position_world_m": self.target_position_world_m[index].copy(),
            "image_size_external": np.asarray(
                self.image_size_external[index],
                dtype=np.int64,
            ),
            "intrinsic_external": self.intrinsic_external[index].copy(),
            "world_from_external": self.world_from_external[index].copy(),
            "trajectory_id": entry.trajectory_id,
            "timestep": timestep,
        }


class QwenV2OnlineGeometryProbeCollator:
    def __init__(self, processor: QwenVLAProcessorAdapter) -> None:
        self.processor = processor

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        if not samples:
            raise ValueError("不能 collate 空 V2 online geometry probe batch")
        processed = self.processor.encode_history_batch(
            [sample["rgb_external_history"] for sample in samples],
            [sample["rgb_wrist_history"] for sample in samples],
            [sample["history_valid"] for sample in samples],
            [sample["instruction"] for sample in samples],
        )
        target_uv = np.stack([sample["target_uv_external"] for sample in samples])
        target_position = np.stack([sample["target_position_world_m"] for sample in samples])
        expected_uv = (len(samples), len(ONLINE_GEOMETRY_TARGETS), 2)
        expected_position = (len(samples), len(ONLINE_GEOMETRY_TARGETS), 3)
        if target_uv.shape != expected_uv or target_uv.dtype != np.float32:
            raise ValueError(f"target_uv_external 必须是 float32 {expected_uv}")
        if target_position.shape != expected_position or target_position.dtype != np.float32:
            raise ValueError(f"target_position_world_m 必须是 float32 {expected_position}")
        return {
            "qwen_inputs": processed.model_inputs,
            "target_uv_external": torch.from_numpy(target_uv),
            "target_position_world_m": torch.from_numpy(target_position),
            "image_size_external": torch.from_numpy(
                np.stack([sample["image_size_external"] for sample in samples])
            ),
            "intrinsic_external": torch.from_numpy(
                np.stack([sample["intrinsic_external"] for sample in samples])
            ),
            "world_from_external": torch.from_numpy(
                np.stack([sample["world_from_external"] for sample in samples])
            ),
            "trajectory_id": [str(sample["trajectory_id"]) for sample in samples],
            "timestep": torch.tensor(
                [int(sample["timestep"]) for sample in samples],
                dtype=torch.long,
            ),
            "visual_tokens_per_image": processed.visual_tokens_per_image,
        }


@dataclass(frozen=True)
class SelectedVisualTokenLayout:
    mask: torch.Tensor
    normalized_centers: torch.Tensor
    grid_shapes: torch.Tensor


def _contiguous_runs(positions: torch.Tensor) -> list[torch.Tensor]:
    if positions.numel() == 0:
        return []
    breaks = torch.nonzero(positions[1:] != positions[:-1] + 1, as_tuple=False).flatten() + 1
    return list(torch.tensor_split(positions, breaks.tolist()))


def build_selected_visual_token_layout(
    input_ids: torch.Tensor,
    image_grid_thw: torch.Tensor,
    *,
    image_token_id: int,
    merge_size: int,
    images_per_sample: int = V2_IMAGES_PER_SAMPLE,
    selected_image_index: int = CURRENT_EXTERNAL_IMAGE_INDEX,
) -> SelectedVisualTokenLayout:
    """定位八图 prompt 中一个指定图像的合并后视觉 token 网格。"""

    if input_ids.ndim != 2:
        raise ValueError("input_ids 必须是 [B,N]")
    batch_size, sequence_length = input_ids.shape
    if images_per_sample <= 0:
        raise ValueError("images_per_sample 必须为正整数")
    if not 0 <= selected_image_index < images_per_sample:
        raise ValueError("selected_image_index 超出图像顺序")
    if image_grid_thw.shape != (batch_size * images_per_sample, 3):
        raise ValueError("image_grid_thw 必须与 batch 和每样本图像数严格对齐")
    if merge_size <= 0:
        raise ValueError("merge_size 必须为正整数")

    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    centers = torch.zeros(
        batch_size,
        sequence_length,
        2,
        dtype=torch.float32,
        device=input_ids.device,
    )
    grid_shapes = torch.empty(
        batch_size,
        2,
        dtype=torch.long,
        device=input_ids.device,
    )
    for batch_index in range(batch_size):
        positions = torch.nonzero(
            input_ids[batch_index] == int(image_token_id),
            as_tuple=False,
        ).flatten()
        runs = _contiguous_runs(positions)
        if len(runs) != images_per_sample:
            raise ValueError(f"样本 {batch_index} 应包含 {images_per_sample} 个 image token span")
        selected_shape: tuple[int, int] | None = None
        for image_index, run in enumerate(runs):
            temporal, grid_h, grid_w = (
                int(value)
                for value in image_grid_thw[batch_index * images_per_sample + image_index].tolist()
            )
            if temporal != 1 or grid_h <= 0 or grid_w <= 0:
                raise ValueError("每张图必须对应正尺寸、单时间片视觉网格")
            if grid_h % merge_size != 0 or grid_w % merge_size != 0:
                raise ValueError("视觉网格不能被 merge_size 整除")
            merged_shape = (grid_h // merge_size, grid_w // merge_size)
            if run.numel() != merged_shape[0] * merged_shape[1]:
                raise ValueError("image token span 与合并后视觉网格不一致")
            if image_index == selected_image_index:
                selected_shape = merged_shape
        if selected_shape is None:
            raise RuntimeError("没有解析到 selected image span")
        merged_h, merged_w = selected_shape
        selected_positions = runs[selected_image_index]
        rows = torch.arange(merged_h, device=input_ids.device, dtype=torch.float32)
        cols = torch.arange(merged_w, device=input_ids.device, dtype=torch.float32)
        row_grid, col_grid = torch.meshgrid(rows, cols, indexing="ij")
        selected_centers = torch.stack(
            (
                (col_grid.reshape(-1) + 0.5) / merged_w,
                (row_grid.reshape(-1) + 0.5) / merged_h,
            ),
            dim=-1,
        )
        mask[batch_index, selected_positions] = True
        centers[batch_index, selected_positions] = selected_centers
        grid_shapes[batch_index] = torch.tensor(
            (merged_h, merged_w),
            dtype=torch.long,
            device=input_ids.device,
        )
    return SelectedVisualTokenLayout(mask, centers, grid_shapes)


def compact_selected_visual_tokens(
    tokens: torch.Tensor,
    layout: SelectedVisualTokenLayout,
) -> tuple[torch.Tensor, torch.Tensor]:
    """把 selected image span 压成固定的 [B,M,H] token 与中心坐标。"""

    if tokens.ndim != 3 or layout.mask.shape != tokens.shape[:2]:
        raise ValueError("tokens/layout shape 不对齐")
    counts = layout.mask.sum(dim=1)
    if counts.numel() == 0 or torch.any(counts <= 0) or not torch.all(counts == counts[0]):
        raise ValueError("selected image 的每样本视觉 token 数必须相同且非零")
    batch_size, _, hidden_size = tokens.shape
    token_count = int(counts[0].item())
    compact_tokens = tokens[layout.mask].reshape(batch_size, token_count, hidden_size)
    compact_centers = layout.normalized_centers[layout.mask].reshape(
        batch_size,
        token_count,
        2,
    )
    return compact_tokens, compact_centers


@dataclass(frozen=True)
class OnlineGeometryProbeOutput:
    predicted_uv: torch.Tensor
    selector_logits: torch.Tensor
    selected_token_indices: torch.Tensor


class OnlineVisualTargetProbe(nn.Module):
    """从所有当前 external tokens 自主选择 object/goal 并回归连续 UV。"""

    def __init__(
        self,
        hidden_size: int = 2048,
        target_count: int = len(ONLINE_GEOMETRY_TARGETS),
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or target_count <= 0:
            raise ValueError("hidden_size/target_count 必须为正整数")
        self.hidden_size = int(hidden_size)
        self.target_count = int(target_count)
        self.selector = nn.Linear(self.hidden_size, self.target_count)
        self.position_decoders = nn.ModuleList(
            nn.Linear(self.hidden_size, 2) for _ in range(self.target_count)
        )

    def forward(
        self,
        visual_tokens: torch.Tensor,
        visual_mask: torch.Tensor | None = None,
    ) -> OnlineGeometryProbeOutput:
        if visual_tokens.ndim != 3 or visual_tokens.shape[-1] != self.hidden_size:
            raise ValueError(f"visual_tokens 必须是 [B,M,{self.hidden_size}]")
        batch_size, token_count, _ = visual_tokens.shape
        if visual_mask is None:
            visual_mask = torch.ones(
                batch_size,
                token_count,
                dtype=torch.bool,
                device=visual_tokens.device,
            )
        if visual_mask.shape != visual_tokens.shape[:2] or visual_mask.dtype != torch.bool:
            raise ValueError("visual_mask 必须是对齐 visual_tokens 的 bool [B,M]")
        if not torch.all(visual_mask.any(dim=1)):
            raise ValueError("每个样本必须至少包含一个有效 visual token")
        probe_tokens = visual_tokens.to(dtype=self.selector.weight.dtype)
        selector_logits = self.selector(probe_tokens).transpose(1, 2)
        selector_logits = selector_logits.masked_fill(
            ~visual_mask.unsqueeze(1),
            -torch.inf,
        )
        weights = torch.softmax(selector_logits, dim=-1)
        pooled = torch.einsum("bkm,bmh->bkh", weights, probe_tokens)
        predicted = torch.stack(
            [
                torch.sigmoid(decoder(pooled[:, target_index]))
                for target_index, decoder in enumerate(self.position_decoders)
            ],
            dim=1,
        ).float()
        return OnlineGeometryProbeOutput(
            predicted_uv=predicted,
            selector_logits=selector_logits,
            selected_token_indices=selector_logits.argmax(dim=-1),
        )


def nearest_visual_token_indices(
    visual_token_centers: torch.Tensor,
    target_uv: torch.Tensor,
    visual_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if visual_token_centers.ndim != 3 or visual_token_centers.shape[-1] != 2:
        raise ValueError("visual_token_centers 必须是 [B,M,2]")
    batch_size, token_count, _ = visual_token_centers.shape
    if target_uv.ndim != 3 or target_uv.shape[0] != batch_size or target_uv.shape[-1] != 2:
        raise ValueError("target_uv 必须是 [B,K,2]")
    if not torch.isfinite(target_uv).all():
        raise ValueError("target_uv 必须是有限坐标")
    if visual_mask is None:
        visual_mask = torch.ones(
            batch_size,
            token_count,
            dtype=torch.bool,
            device=visual_token_centers.device,
        )
    if visual_mask.shape != (batch_size, token_count) or visual_mask.dtype != torch.bool:
        raise ValueError("visual_mask 必须是 bool [B,M]")
    distances = torch.sum(
        (visual_token_centers.float().unsqueeze(1) - target_uv.float().unsqueeze(2)).square(),
        dim=-1,
    ).masked_fill(~visual_mask.unsqueeze(1), torch.inf)
    return distances.argmin(dim=-1)


@dataclass(frozen=True)
class OnlineGeometryProbeLoss:
    loss: torch.Tensor
    coordinate_loss: torch.Tensor
    selector_loss: torch.Tensor
    target_token_indices: torch.Tensor


def online_geometry_probe_loss(
    output: OnlineGeometryProbeOutput,
    target_uv: torch.Tensor,
    visual_token_centers: torch.Tensor,
    *,
    selector_loss_weight: float,
    visual_mask: torch.Tensor | None = None,
) -> OnlineGeometryProbeLoss:
    if output.predicted_uv.shape != target_uv.shape:
        raise ValueError("predicted_uv 与 target_uv shape 必须一致")
    if not math.isfinite(selector_loss_weight) or selector_loss_weight < 0.0:
        raise ValueError("selector_loss_weight 必须是有限非负数")
    target_indices = nearest_visual_token_indices(
        visual_token_centers,
        target_uv,
        visual_mask,
    )
    coordinate_loss = F.mse_loss(output.predicted_uv.float(), target_uv.float())
    selector_loss = F.cross_entropy(
        output.selector_logits.reshape(-1, output.selector_logits.shape[-1]),
        target_indices.reshape(-1),
    )
    return OnlineGeometryProbeLoss(
        loss=coordinate_loss + float(selector_loss_weight) * selector_loss,
        coordinate_loss=coordinate_loss,
        selector_loss=selector_loss,
        target_token_indices=target_indices,
    )


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return None
    return float(np.percentile(array, percentile))


def summarize_online_geometry_predictions(
    predicted_uv: np.ndarray,
    target_uv: np.ndarray,
    selected_token_indices: np.ndarray,
    nearest_token_indices: np.ndarray,
    image_sizes_hw: np.ndarray,
    grid_shapes_hw: np.ndarray,
    intrinsics: np.ndarray,
    world_from_cameras: np.ndarray,
    target_positions_world_m: np.ndarray,
) -> dict[str, Any]:
    """汇总 online object/goal 定位与 object→goal 相对 XY 误差。"""

    predicted = np.asarray(predicted_uv, dtype=np.float64)
    targets = np.asarray(target_uv, dtype=np.float64)
    positions = np.asarray(target_positions_world_m, dtype=np.float64)
    count = predicted.shape[0]
    target_count = len(ONLINE_GEOMETRY_TARGETS)
    if predicted.shape != (count, target_count, 2) or targets.shape != predicted.shape:
        raise ValueError("predicted_uv/target_uv 必须是非空 [B,2,2]")
    if count <= 0 or positions.shape != (count, target_count, 3):
        raise ValueError("target_positions_world_m 必须是非空 [B,2,3]")
    selected = np.asarray(selected_token_indices, dtype=np.int64)
    nearest = np.asarray(nearest_token_indices, dtype=np.int64)
    if selected.shape != (count, target_count) or nearest.shape != selected.shape:
        raise ValueError("selector indices 必须是 [B,2]")

    image_sizes = np.asarray(image_sizes_hw, dtype=np.int64)
    grids = np.asarray(grid_shapes_hw, dtype=np.int64)
    intrinsic_array = np.asarray(intrinsics, dtype=np.float64)
    transforms = np.asarray(world_from_cameras, dtype=np.float64)
    if image_sizes.shape != (count, 2) or grids.shape != (count, 2):
        raise ValueError("image_sizes/grid_shapes 必须是 [B,2]")
    if intrinsic_array.shape != (count, 3, 3) or transforms.shape != (count, 4, 4):
        raise ValueError("相机矩阵 batch shape 无效")

    by_target: dict[str, Any] = {}
    predicted_world = np.full((count, target_count, 3), np.nan, dtype=np.float64)
    for target_index, target_name in enumerate(ONLINE_GEOMETRY_TARGETS):
        by_target[target_name] = summarize_spatial_predictions(
            predicted[:, target_index],
            targets[:, target_index],
            image_sizes,
            grids,
            intrinsic_array,
            transforms,
            positions[:, target_index],
        )
        for sample_index in range(count):
            height, width = (int(value) for value in image_sizes[sample_index])
            try:
                point = unproject_gl_camera_to_world_plane(
                    predicted[sample_index, target_index],
                    float(positions[sample_index, target_index, 2]),
                    intrinsic_array[sample_index].ravel(),
                    transforms[sample_index].ravel(),
                    height,
                    width,
                )
            except ValueError:
                continue
            predicted_world[sample_index, target_index] = point

    valid_relative = np.isfinite(predicted_world).all(axis=(1, 2))
    relative_errors: list[float] = []
    for sample_index in np.flatnonzero(valid_relative).tolist():
        predicted_delta = (
            predicted_world[sample_index, 1, :2] - predicted_world[sample_index, 0, :2]
        )
        target_delta = positions[sample_index, 1, :2] - positions[sample_index, 0, :2]
        relative_errors.append(float(np.linalg.norm(predicted_delta - target_delta)))

    selector_accuracy = {
        target_name: float(np.mean(selected[:, target_index] == nearest[:, target_index]))
        for target_index, target_name in enumerate(ONLINE_GEOMETRY_TARGETS)
    }
    relative_summary = {
        "samples": count,
        "valid_samples": len(relative_errors),
        "invalid_samples": count - len(relative_errors),
        "median_error_m": _percentile(relative_errors, 50),
        "p90_error_m": _percentile(relative_errors, 90),
        "max_error_m": None if not relative_errors else float(max(relative_errors)),
    }
    object_metrics = by_target["object"]
    goal_metrics = by_target["goal"]
    complete = (
        int(object_metrics["invalid_world_unprojections"]) == 0
        and int(goal_metrics["invalid_world_unprojections"]) == 0
        and relative_summary["invalid_samples"] == 0
    )
    coarse_reach_screen = (
        complete
        and float(object_metrics["median_world_xy_error_m"]) <= 0.02
        and float(object_metrics["p90_world_xy_error_m"]) <= 0.04
    )
    deployable_precision_candidate = (
        complete
        and float(object_metrics["p90_world_xy_error_m"]) <= 0.01
        and float(goal_metrics["p90_world_xy_error_m"]) <= 0.015
        and float(relative_summary["p90_error_m"]) <= 0.015
    )
    return {
        "screening_only": True,
        "samples": count,
        "by_target": by_target,
        "selector_exact_nearest_token_accuracy": selector_accuracy,
        "object_to_goal_relative_xy": relative_summary,
        "thresholds": {
            "coarse_reach_object_median_m_max": 0.02,
            "coarse_reach_object_p90_m_max": 0.04,
            "deployable_object_p90_m_max": 0.01,
            "deployable_goal_p90_m_max": 0.015,
            "deployable_object_goal_relative_p90_m_max": 0.015,
        },
        "coarse_reach_screen_passed": coarse_reach_screen,
        "deployable_precision_candidate": deployable_precision_candidate,
        "limitations": [
            "GT object/goal 只用于训练监督与测试指标，测试选择器不读取 GT token index",
            "只在 Reach window 的已知桌面高度平面上测 object/goal world XY",
            "不测 Z、TCP orientation、Action 解码、接触动力学或完整闭环成功",
            "通过只允许继续 E013，不等价于机器人可部署",
        ],
    }


__all__ = [
    "CURRENT_EXTERNAL_IMAGE_INDEX",
    "ONLINE_GEOMETRY_TARGETS",
    "V2_IMAGES_PER_SAMPLE",
    "V2_ONLINE_GEOMETRY_PROBE_FORMAT",
    "OnlineGeometryProbeLoss",
    "OnlineGeometryProbeOutput",
    "OnlineVisualTargetProbe",
    "QwenV2OnlineGeometryProbeCollator",
    "QwenV2OnlineGeometryProbeDataset",
    "SelectedVisualTokenLayout",
    "build_selected_visual_token_layout",
    "compact_selected_visual_tokens",
    "nearest_visual_token_indices",
    "online_geometry_probe_loss",
    "summarize_online_geometry_predictions",
]
