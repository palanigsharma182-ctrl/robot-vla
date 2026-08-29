"""冻结 Qwen Layer 12/24 后，从 GT 方块 token 线性读出连续空间位置。"""

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

from robot_vla.contracts import PICK_AND_PLACE_SKILLS, RobotSpec
from robot_vla.data.trajectory import TrajectoryMeta, TrajectoryStore
from robot_vla.model.qwen_processor import QwenVLAProcessorAdapter

QWEN_SPATIAL_PROBE_FORMAT = "robot-vla-qwen-spatial-probe/v1"
REACH_SKILL_ID = PICK_AND_PLACE_SKILLS.index("reach")


@dataclass(frozen=True)
class ImageProjection:
    pixel_uv: np.ndarray
    normalized_uv: np.ndarray
    depth_m: float


def _matrix(value: Sequence[float], shape: tuple[int, int], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.size != shape[0] * shape[1] or not np.isfinite(array).all():
        raise ValueError(f"{name} 必须是有限 {shape} 矩阵")
    return array.reshape(shape)


def project_world_point_to_gl_camera(
    world_point_m: np.ndarray,
    intrinsic_cv: Sequence[float],
    world_from_camera_gl: Sequence[float],
    image_height: int,
    image_width: int,
) -> ImageProjection:
    """把世界点投影到 ManiSkill `cam2world_gl + intrinsic_cv` 图像。"""

    point = np.asarray(world_point_m, dtype=np.float64)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError("world_point_m 必须是有限 [3] 世界坐标")
    if image_height <= 0 or image_width <= 0:
        raise ValueError("图像 H/W 必须为正整数")
    intrinsic = _matrix(intrinsic_cv, (3, 3), "intrinsic_cv")
    world_from_camera = _matrix(
        world_from_camera_gl,
        (4, 4),
        "world_from_camera_gl",
    )
    camera_from_world = np.linalg.inv(world_from_camera)
    point_gl = camera_from_world @ np.concatenate((point, np.ones(1, dtype=np.float64)))
    # OpenGL camera: +x right, +y up, -z forward；intrinsic_cv 使用 OpenCV 坐标。
    point_cv = np.asarray((point_gl[0], -point_gl[1], -point_gl[2]), dtype=np.float64)
    depth = float(point_cv[2])
    if not math.isfinite(depth) or depth <= 0.0:
        raise ValueError("世界点位于相机后方或相机平面上")
    projected = intrinsic @ (point_cv / depth)
    pixel_uv = projected[:2]
    if not np.isfinite(pixel_uv).all():
        raise ValueError("世界点投影产生 NaN 或 Inf")
    normalized_uv = np.asarray(
        (
            (pixel_uv[0] + 0.5) / image_width,
            (pixel_uv[1] + 0.5) / image_height,
        ),
        dtype=np.float64,
    )
    return ImageProjection(
        pixel_uv=pixel_uv.astype(np.float32),
        normalized_uv=normalized_uv.astype(np.float32),
        depth_m=depth,
    )


def unproject_gl_camera_to_world_plane(
    normalized_uv: np.ndarray,
    plane_world_z_m: float,
    intrinsic_cv: Sequence[float],
    world_from_camera_gl: Sequence[float],
    image_height: int,
    image_width: int,
) -> np.ndarray:
    """把归一化图像坐标射线与给定世界 Z 平面求交。"""

    uv = np.asarray(normalized_uv, dtype=np.float64)
    if uv.shape != (2,) or not np.isfinite(uv).all():
        raise ValueError("normalized_uv 必须是有限 [2]")
    if not math.isfinite(plane_world_z_m):
        raise ValueError("plane_world_z_m 必须是有限数值")
    if image_height <= 0 or image_width <= 0:
        raise ValueError("图像 H/W 必须为正整数")
    intrinsic = _matrix(intrinsic_cv, (3, 3), "intrinsic_cv")
    world_from_camera = _matrix(
        world_from_camera_gl,
        (4, 4),
        "world_from_camera_gl",
    )
    pixel = np.asarray(
        (uv[0] * image_width - 0.5, uv[1] * image_height - 0.5, 1.0),
        dtype=np.float64,
    )
    direction_cv = np.linalg.inv(intrinsic) @ pixel
    direction_gl = np.asarray(
        (direction_cv[0], -direction_cv[1], -direction_cv[2]),
        dtype=np.float64,
    )
    origin_world = world_from_camera[:3, 3]
    direction_world = world_from_camera[:3, :3] @ direction_gl
    if abs(float(direction_world[2])) <= 1e-12:
        raise ValueError("相机射线与目标世界 Z 平面平行")
    scale = (float(plane_world_z_m) - float(origin_world[2])) / float(direction_world[2])
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("目标世界 Z 平面位于相机射线后方")
    point_world = origin_world + scale * direction_world
    if not np.isfinite(point_world).all():
        raise ValueError("反投影世界点包含 NaN 或 Inf")
    return point_world.astype(np.float32)


class QwenExternalSpatialProbeDataset:
    """只暴露 Reach anchor，并提供方块中心在 external 图像中的 GT 投影。"""

    def __init__(
        self,
        root: str | Path,
        entries: Sequence[TrajectoryMeta],
        spec: RobotSpec,
        *,
        cache_size: int = 2,
    ) -> None:
        if not entries:
            raise ValueError("Spatial Probe Dataset 需要至少一条轨迹")
        splits = {entry.split for entry in entries}
        if len(splits) != 1:
            raise ValueError("Spatial Probe Dataset 的轨迹必须属于同一个 split")
        self.root = Path(root)
        self.entries = list(entries)
        self.store = TrajectoryStore(self.root, spec, cache_size=cache_size)
        self.index: list[tuple[int, int]] = []
        self.target_uv_external: list[np.ndarray] = []
        self.object_position_m: list[np.ndarray] = []
        self.image_size_external: list[tuple[int, int]] = []
        self.intrinsic_external: list[np.ndarray] = []
        self.world_from_external: list[np.ndarray] = []
        self.rejected_missing_geometry = 0
        self.rejected_out_of_view = 0
        window_ids: list[str] = []

        for entry_index, entry in enumerate(self.entries):
            arrays = self.store.get(entry)
            if arrays.object_position_m is None:
                self.rejected_missing_geometry += int(
                    np.count_nonzero(arrays.observation_valid & (arrays.skill_id == REACH_SKILL_ID))
                )
                continue
            height, width = map(int, arrays.rgb_external.shape[1:3])
            calibration = entry.camera_calibration
            intrinsic = np.asarray(calibration.intrinsic_external, dtype=np.float32).reshape(3, 3)
            world_from_camera = np.asarray(
                calibration.world_from_external,
                dtype=np.float32,
            ).reshape(4, 4)
            candidates = np.flatnonzero(
                arrays.observation_valid & (arrays.skill_id == REACH_SKILL_ID)
            )
            for timestep in candidates.tolist():
                object_position = arrays.object_position_m[timestep]
                try:
                    projection = project_world_point_to_gl_camera(
                        object_position,
                        intrinsic.ravel(),
                        world_from_camera.ravel(),
                        height,
                        width,
                    )
                except ValueError:
                    self.rejected_out_of_view += 1
                    continue
                u, v = projection.normalized_uv.tolist()
                if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
                    self.rejected_out_of_view += 1
                    continue
                self.index.append((entry_index, timestep))
                self.target_uv_external.append(projection.normalized_uv)
                self.object_position_m.append(object_position.astype(np.float32, copy=True))
                self.image_size_external.append((height, width))
                self.intrinsic_external.append(intrinsic.copy())
                self.world_from_external.append(world_from_camera.copy())
                window_ids.append(f"{entry.trajectory_id}:{timestep}")

        if not self.index:
            raise ValueError("Spatial Probe Dataset 没有可见方块的 Reach 窗口")
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
        return {
            "rgb_external": arrays.rgb_external[timestep].copy(),
            "rgb_wrist": arrays.rgb_wrist[timestep].copy(),
            "instruction": entry.task.instruction,
            "target_uv_external": self.target_uv_external[index].copy(),
            "object_position_m": self.object_position_m[index].copy(),
            "image_size_external": np.asarray(
                self.image_size_external[index],
                dtype=np.int64,
            ),
            "intrinsic_external": self.intrinsic_external[index].copy(),
            "world_from_external": self.world_from_external[index].copy(),
            "trajectory_id": entry.trajectory_id,
            "timestep": timestep,
            "skill_id": REACH_SKILL_ID,
        }


class QwenSpatialProbeCollator:
    def __init__(self, processor: QwenVLAProcessorAdapter) -> None:
        self.processor = processor

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        if not samples:
            raise ValueError("不能 collate 空 Spatial Probe batch")
        processed = self.processor.encode_batch(
            [sample["rgb_external"] for sample in samples],
            [sample["rgb_wrist"] for sample in samples],
            [sample["instruction"] for sample in samples],
        )
        target_uv = np.stack([sample["target_uv_external"] for sample in samples])
        object_position = np.stack([sample["object_position_m"] for sample in samples])
        image_size = np.stack([sample["image_size_external"] for sample in samples])
        intrinsic = np.stack([sample["intrinsic_external"] for sample in samples])
        world_from_camera = np.stack([sample["world_from_external"] for sample in samples])
        if target_uv.shape != (len(samples), 2) or target_uv.dtype != np.float32:
            raise ValueError("target_uv_external batch 必须是 float32 [B,2]")
        if object_position.shape != (len(samples), 3) or object_position.dtype != np.float32:
            raise ValueError("object_position_m batch 必须是 float32 [B,3]")
        return {
            "qwen_inputs": processed.model_inputs,
            "target_uv_external": torch.from_numpy(target_uv),
            "object_position_m": torch.from_numpy(object_position),
            "image_size_external": torch.from_numpy(image_size),
            "intrinsic_external": torch.from_numpy(intrinsic),
            "world_from_external": torch.from_numpy(world_from_camera),
            "trajectory_id": [str(sample["trajectory_id"]) for sample in samples],
            "timestep": torch.tensor(
                [int(sample["timestep"]) for sample in samples],
                dtype=torch.long,
            ),
            "visual_tokens_per_image": processed.visual_tokens_per_image,
        }


@dataclass(frozen=True)
class ExternalVisualTokenLayout:
    mask: torch.Tensor
    normalized_centers: torch.Tensor
    grid_shapes: torch.Tensor


def _contiguous_runs(positions: torch.Tensor) -> list[torch.Tensor]:
    if positions.numel() == 0:
        return []
    breaks = torch.nonzero(positions[1:] != positions[:-1] + 1, as_tuple=False).flatten() + 1
    return list(torch.tensor_split(positions, breaks.tolist()))


def build_external_visual_token_layout(
    input_ids: torch.Tensor,
    image_grid_thw: torch.Tensor,
    *,
    image_token_id: int,
    merge_size: int,
) -> ExternalVisualTokenLayout:
    """把每个样本第一个 image span 映射到合并后 external 视觉网格中心。"""

    if input_ids.ndim != 2:
        raise ValueError("input_ids 必须是 [B,N]")
    batch_size, sequence_length = input_ids.shape
    if image_grid_thw.shape != (batch_size * 2, 3):
        raise ValueError("双图 image_grid_thw 必须是 [B*2,3]")
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
        if len(runs) != 2:
            raise ValueError(
                f"样本 {batch_index} 应包含 external/wrist 两个 image token span"
            )
        merged_shapes: list[tuple[int, int]] = []
        for view_offset, view_name in enumerate(("external", "wrist")):
            temporal, grid_h, grid_w = (
                int(value)
                for value in image_grid_thw[batch_index * 2 + view_offset].tolist()
            )
            if temporal != 1:
                raise ValueError(f"首版 Spatial Probe 只支持单帧 {view_name} 图像")
            if grid_h <= 0 or grid_w <= 0:
                raise ValueError(f"{view_name} image grid 必须为正数")
            if grid_h % merge_size != 0 or grid_w % merge_size != 0:
                raise ValueError(
                    f"{view_name} image grid 不能被 merge_size 整除"
                )
            merged_shape = (grid_h // merge_size, grid_w // merge_size)
            if runs[view_offset].numel() != merged_shape[0] * merged_shape[1]:
                raise ValueError(
                    f"{view_name} image token span 与合并后视觉网格不一致"
                )
            merged_shapes.append(merged_shape)
        merged_h, merged_w = merged_shapes[0]
        external_positions = runs[0]
        rows = torch.arange(merged_h, device=input_ids.device, dtype=torch.float32)
        cols = torch.arange(merged_w, device=input_ids.device, dtype=torch.float32)
        row_grid, col_grid = torch.meshgrid(rows, cols, indexing="ij")
        external_centers = torch.stack(
            (
                (col_grid.reshape(-1) + 0.5) / merged_w,
                (row_grid.reshape(-1) + 0.5) / merged_h,
            ),
            dim=-1,
        )
        mask[batch_index, external_positions] = True
        centers[batch_index, external_positions] = external_centers
        grid_shapes[batch_index] = torch.tensor(
            (merged_h, merged_w),
            dtype=torch.long,
            device=input_ids.device,
        )
    return ExternalVisualTokenLayout(
        mask=mask,
        normalized_centers=centers,
        grid_shapes=grid_shapes,
    )


@dataclass(frozen=True)
class SpatialProbeOutput:
    predicted_uv: torch.Tensor
    target_token_index: torch.Tensor


class LinearVisualTokenPositionProbe(nn.Module):
    """从 GT 选中的方块 visual token 直接线性回归连续图像坐标。"""

    def __init__(self, hidden_size: int = 2048) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size 必须为正整数")
        self.hidden_size = int(hidden_size)
        self.position_decoder = nn.Linear(self.hidden_size, 2)

    def forward(
        self,
        tokens: torch.Tensor,
        layout: ExternalVisualTokenLayout,
        target_uv: torch.Tensor,
    ) -> SpatialProbeOutput:
        if tokens.ndim != 3 or tokens.shape[-1] != self.hidden_size:
            raise ValueError(f"tokens 必须是 [B,N,{self.hidden_size}]")
        if layout.mask.shape != tokens.shape[:2] or layout.mask.dtype != torch.bool:
            raise ValueError("visual token mask 必须与 [B,N] 对齐")
        if layout.normalized_centers.shape != (*tokens.shape[:2], 2):
            raise ValueError("visual token centers 必须是 [B,N,2]")
        if target_uv.shape != (tokens.shape[0], 2):
            raise ValueError("target_uv 必须是 [B,2]")
        target_token_index = nearest_external_visual_token_indices(
            layout,
            target_uv,
        )
        batch_indices = torch.arange(tokens.shape[0], device=tokens.device)
        selected_tokens = tokens[batch_indices, target_token_index]
        predicted_uv = self.decode_selected_tokens(selected_tokens)
        return SpatialProbeOutput(
            predicted_uv=predicted_uv,
            target_token_index=target_token_index,
        )

    def decode_selected_tokens(self, selected_tokens: torch.Tensor) -> torch.Tensor:
        if selected_tokens.ndim != 2 or selected_tokens.shape[-1] != self.hidden_size:
            raise ValueError(
                f"selected_tokens 必须是 [B,{self.hidden_size}]"
            )
        # Qwen 固定以 BF16 加载；显式对齐到 probe 参数 dtype，保证关闭
        # autocast 时仍只改变读出精度，不改变冻结表征。
        probe_tokens = selected_tokens.to(dtype=self.position_decoder.weight.dtype)
        signed_uv = self.position_decoder(probe_tokens).float()
        predicted_uv = 0.5 * (signed_uv + 1.0)
        if predicted_uv.shape != (selected_tokens.shape[0], 2):
            raise RuntimeError("position probe 输出必须是 [B,2]")
        return predicted_uv


def nearest_external_visual_token_indices(
    layout: ExternalVisualTokenLayout,
    target_uv: torch.Tensor,
) -> torch.Tensor:
    """只用 GT UV 选择最近的 external 粗 token，不返回其坐标给 probe。"""

    if layout.mask.ndim != 2 or layout.mask.dtype != torch.bool:
        raise ValueError("visual token mask 必须是 bool [B,N]")
    if layout.normalized_centers.shape != (*layout.mask.shape, 2):
        raise ValueError("visual token centers 必须是 [B,N,2]")
    if target_uv.shape != (layout.mask.shape[0], 2):
        raise ValueError("target_uv 必须是 [B,2]")
    if not torch.isfinite(target_uv).all():
        raise ValueError("target_uv 必须是有限坐标")
    if not torch.all(layout.mask.any(dim=1)):
        raise ValueError("每个样本必须至少有一个 external visual token")
    distances = torch.sum(
        (layout.normalized_centers - target_uv.float().unsqueeze(1)).square(),
        dim=-1,
    ).masked_fill(~layout.mask, torch.inf)
    return distances.argmin(dim=1)


def spatial_probe_loss(
    predicted_uv: torch.Tensor,
    target_uv: torch.Tensor,
) -> torch.Tensor:
    if target_uv.shape != predicted_uv.shape:
        raise ValueError("target_uv 必须与 position probe 输出 shape 相同")
    return F.mse_loss(predicted_uv.float(), target_uv.float())


def build_matched_linear_probes(
    *,
    seed: int,
    hidden_size: int = 2048,
) -> nn.ModuleDict:
    """Layer 12/24 使用逐参数相同的 probe 初始化。"""

    if seed < 0:
        raise ValueError("seed 不能为负数")
    torch.manual_seed(seed)
    layer12 = LinearVisualTokenPositionProbe(hidden_size)
    torch.manual_seed(seed)
    layer24 = LinearVisualTokenPositionProbe(hidden_size)
    return nn.ModuleDict({"layer12": layer12, "layer24": layer24})


def _percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values.astype(np.float64), percentile))


def summarize_spatial_predictions(
    predicted_uv: np.ndarray,
    target_uv: np.ndarray,
    image_sizes_hw: np.ndarray,
    grid_shapes_hw: np.ndarray,
    intrinsics: np.ndarray,
    world_from_cameras: np.ndarray,
    object_positions_m: np.ndarray,
) -> dict[str, float | int | None]:
    """汇总图像、视觉 token 和已知桌面平面上的世界 XY 定位误差。"""

    predicted = np.asarray(predicted_uv, dtype=np.float64)
    target = np.asarray(target_uv, dtype=np.float64)
    count = predicted.shape[0]
    if predicted.shape != (count, 2) or target.shape != (count, 2) or count <= 0:
        raise ValueError("predicted_uv/target_uv 必须是非空 [B,2]")
    image_sizes = np.asarray(image_sizes_hw, dtype=np.int64)
    grids = np.asarray(grid_shapes_hw, dtype=np.int64)
    intrinsic_array = np.asarray(intrinsics, dtype=np.float64)
    transforms = np.asarray(world_from_cameras, dtype=np.float64)
    objects = np.asarray(object_positions_m, dtype=np.float64)
    if image_sizes.shape != (count, 2) or grids.shape != (count, 2):
        raise ValueError("image_sizes/grid_shapes 必须是 [B,2]")
    if intrinsic_array.shape != (count, 3, 3) or transforms.shape != (count, 4, 4):
        raise ValueError("相机矩阵 batch shape 无效")
    if objects.shape != (count, 3):
        raise ValueError("object_positions_m 必须是 [B,3]")
    if not all(
        np.isfinite(value).all()
        for value in (predicted, target, intrinsic_array, transforms, objects)
    ):
        raise ValueError("Spatial Probe 指标输入包含 NaN 或 Inf")

    height_width = image_sizes.astype(np.float64)
    width_height = height_width[:, ::-1]
    predicted_pixels = predicted * width_height - 0.5
    target_pixels = target * width_height - 0.5
    pixel_error = np.linalg.norm(predicted_pixels - target_pixels, axis=1)
    grid_width_height = grids[:, ::-1].astype(np.float64)
    token_error = np.linalg.norm((predicted - target) * grid_width_height, axis=1)
    normalized_error = np.linalg.norm(predicted - target, axis=1)

    world_xy_errors: list[float] = []
    invalid_unprojections = 0
    for index in range(count):
        height, width = (int(value) for value in image_sizes[index])
        try:
            point = unproject_gl_camera_to_world_plane(
                predicted[index],
                float(objects[index, 2]),
                intrinsic_array[index].ravel(),
                transforms[index].ravel(),
                height,
                width,
            )
        except ValueError:
            invalid_unprojections += 1
            continue
        world_xy_errors.append(
            float(np.linalg.norm(point[:2].astype(np.float64) - objects[index, :2]))
        )
    valid_world_samples = len(world_xy_errors)
    result: dict[str, float | int | None] = {
        "samples": count,
        "mean_normalized_error": float(normalized_error.mean()),
        "median_normalized_error": _percentile(normalized_error, 50),
        "mean_pixel_error": float(pixel_error.mean()),
        "median_pixel_error": _percentile(pixel_error, 50),
        "p90_pixel_error": _percentile(pixel_error, 90),
        "mean_visual_token_error": float(token_error.mean()),
        "median_visual_token_error": _percentile(token_error, 50),
        "p90_visual_token_error": _percentile(token_error, 90),
        "within_1_visual_token_rate": float(np.mean(token_error <= 1.0)),
        "within_2_visual_tokens_rate": float(np.mean(token_error <= 2.0)),
        "invalid_world_unprojections": invalid_unprojections,
        "valid_world_unprojection_samples": valid_world_samples,
        "valid_world_unprojection_rate": valid_world_samples / count,
        "mean_world_xy_error_m": None,
        "median_world_xy_error_m": None,
        "p90_world_xy_error_m": None,
    }
    if world_xy_errors:
        world = np.asarray(world_xy_errors, dtype=np.float64)
        result.update(
            {
                "mean_world_xy_error_m": float(world.mean()),
                "median_world_xy_error_m": _percentile(world, 50),
                "p90_world_xy_error_m": _percentile(world, 90),
            }
        )
    return result


def interpret_layer12_probe(
    layer12: dict[str, Any],
    layer24: dict[str, Any],
    token_center_reference: dict[str, Any],
) -> dict[str, Any]:
    """以 Reach 4 cm 门槛为背景，给出预注册式 screening 判定。"""

    l12 = layer12["position"]
    l24 = layer24["position"]
    l12_median_value = l12.get("median_world_xy_error_m")
    l12_p90_value = l12.get("p90_world_xy_error_m")
    l24_median_value = l24.get("median_world_xy_error_m")
    token_center_median_value = token_center_reference.get(
        "median_world_xy_error_m"
    )
    l12_world_complete = int(l12.get("invalid_world_unprojections", 0)) == 0
    l24_world_complete = int(l24.get("invalid_world_unprojections", 0)) == 0
    token_center_world_complete = (
        int(token_center_reference.get("invalid_world_unprojections", 0)) == 0
    )
    l12_median = None if l12_median_value is None else float(l12_median_value)
    l12_p90 = None if l12_p90_value is None else float(l12_p90_value)
    l24_median = None if l24_median_value is None else float(l24_median_value)
    token_center_median = (
        None
        if token_center_median_value is None
        else float(token_center_median_value)
    )
    usable = (
        l12_world_complete
        and l12_median is not None
        and l12_p90 is not None
        and l12_median <= 0.02
        and l12_p90 <= 0.04
    )
    clearly_better = (
        l12_world_complete
        and l24_world_complete
        and l12_median is not None
        and l24_median is not None
        and l12_median <= 0.8 * l24_median
    )
    layer_ratio = (
        None
        if l12_median is None or l24_median in {None, 0.0}
        else l12_median / l24_median
    )
    token_center_ratio = (
        None
        if l12_median is None or token_center_median in {None, 0.0}
        else l12_median / token_center_median
    )
    beats_token_center = (
        l12_world_complete
        and token_center_world_complete
        and token_center_ratio is not None
        and token_center_ratio <= 0.8
    )
    subtoken_evidence = usable and beats_token_center
    return {
        "screening_only": True,
        "thresholds": {
            "median_world_xy_error_m_max": 0.02,
            "p90_world_xy_error_m_max": 0.04,
            "layer12_vs_layer24_median_ratio_max": 0.8,
            "layer12_vs_token_center_median_ratio_max": 0.8,
        },
        "requires_complete_world_unprojection": True,
        "layer12_has_reach_usable_position": usable,
        "layer12_clearly_better_than_layer24": clearly_better,
        "layer12_beats_token_center_quantization": beats_token_center,
        "layer12_has_subtoken_position_evidence": subtoken_evidence,
        "layer12_to_layer24_median_world_xy_error_ratio": layer_ratio,
        "layer12_to_token_center_median_world_xy_error_ratio": token_center_ratio,
        "interpretation": (
            "Layer12 的方块 token 可线性解码 Reach 尺度且优于粗 token 中心的位置"
            if subtoken_evidence
            else (
                "Layer12 达到 Reach 尺度，但没有优于 GT 粗 token 中心的子 token 精度证据"
                if usable
                else "Layer12 尚未达到 Reach 尺度的位置线性解码门槛"
            )
        ),
    }


__all__ = [
    "QWEN_SPATIAL_PROBE_FORMAT",
    "ExternalVisualTokenLayout",
    "ImageProjection",
    "LinearVisualTokenPositionProbe",
    "QwenExternalSpatialProbeDataset",
    "QwenSpatialProbeCollator",
    "SpatialProbeOutput",
    "build_external_visual_token_layout",
    "build_matched_linear_probes",
    "interpret_layer12_probe",
    "nearest_external_visual_token_indices",
    "project_world_point_to_gl_camera",
    "spatial_probe_loss",
    "summarize_spatial_predictions",
    "unproject_gl_camera_to_world_plane",
]
