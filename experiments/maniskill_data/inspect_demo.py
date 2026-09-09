"""只读检查官方 ManiSkill HDF5/JSON；不加载数组，也不执行转换。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


UNKNOWN = "unknown"
REPORT_SCHEMA = "robot-vla-maniskill-demo-preflight/v1"
CUSTOM_ENV_ID = "RobotVLAPickCubeToRegion-v1"


def _unknown_field() -> dict[str, Any]:
    return {"status": UNKNOWN, "value": UNKNOWN, "source": UNKNOWN}


def _nested_value(payload: dict[str, Any], path: str) -> Any:
    value: Any = payload
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _first_field(payload: dict[str, Any], paths: tuple[str, ...]) -> dict[str, Any]:
    for path in paths:
        value = _nested_value(payload, path)
        if value is not None:
            return {"status": "available", "value": value, "source": path}
    return _unknown_field()


def _load_metadata(
    path: Path | None, warnings: list[str]
) -> tuple[dict[str, Any], str]:
    if path is None or not path.exists():
        warnings.append("未找到同名 JSON metadata；环境、机器人和 episode 级语义可能为 unknown")
        return {}, "missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"无法读取 JSON metadata {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("JSON metadata 根节点必须是对象")
    return payload, "available"


def _dataset_inventory(node: Any, h5py: Any) -> list[dict[str, Any]]:
    """只访问 HDF5 元数据属性；禁止 dataset[...] 或 dataset.read_direct。"""
    rows: list[dict[str, Any]] = []

    def record(name: str, value: Any) -> None:
        if isinstance(value, h5py.Dataset):
            rows.append(
                {
                    "path": name,
                    "shape": [int(size) for size in value.shape],
                    "dtype": str(value.dtype),
                }
            )

    if isinstance(node, h5py.Dataset):
        record("", node)
    else:
        node.visititems(record)
    return sorted(rows, key=lambda row: row["path"])


def _node_lengths(
    group: Any, name: str, inventory: list[dict[str, Any]]
) -> dict[str, Any]:
    if name not in group:
        return {
            "present": False,
            "status": UNKNOWN,
            "lengths": UNKNOWN,
            "leaf_count": 0,
            "scalar_leaf_count": 0,
        }
    leaves = [
        row
        for row in inventory
        if row["path"] == name or row["path"].startswith(f"{name}/")
    ]
    lengths = sorted({row["shape"][0] for row in leaves if row["shape"]})
    scalar_count = sum(not row["shape"] for row in leaves)
    status = UNKNOWN if not lengths else ("available" if len(lengths) == 1 else "inconsistent")
    return {
        "present": True,
        "status": status,
        "lengths": lengths if lengths else UNKNOWN,
        "leaf_count": len(leaves),
        "scalar_leaf_count": scalar_count,
    }


def _single_length(field: dict[str, Any]) -> int | None:
    lengths = field["lengths"]
    return lengths[0] if isinstance(lengths, list) and len(lengths) == 1 else None


def _episode_success(
    episode: dict[str, Any], warnings: list[str]
) -> tuple[bool | None, str]:
    direct = episode.get("success")
    info = episode.get("info")
    nested = info.get("success") if isinstance(info, dict) else None
    if isinstance(direct, bool) and isinstance(nested, bool) and direct != nested:
        warnings.append(
            f"episode {episode.get('episode_id', UNKNOWN)} "
            "的 success 与 info.success 冲突"
        )
        return None, UNKNOWN
    if isinstance(direct, bool):
        return direct, "episodes[].success"
    if isinstance(nested, bool):
        return nested, "episodes[].info.success"
    return None, UNKNOWN


def _metadata_report(
    metadata: dict[str, Any], warnings: list[str]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    episodes_value = metadata.get("episodes")
    if episodes_value is None:
        episodes: list[dict[str, Any]] = []
    elif not isinstance(episodes_value, list) or not all(
        isinstance(row, dict) for row in episodes_value
    ):
        warnings.append("metadata.episodes 不是对象列表")
        episodes = []
    else:
        episodes = episodes_value

    by_group: dict[str, dict[str, Any]] = {}
    duplicate_ids: set[str] = set()
    control_modes: set[str] = set()
    unknown_control = 0
    elapsed_steps: list[int] = []
    success_values: list[bool] = []
    success_sources: set[str] = set()
    for episode in episodes:
        episode_id = episode.get("episode_id")
        if isinstance(episode_id, (str, int)) and not isinstance(episode_id, bool):
            group_name = f"traj_{episode_id}"
            if group_name in by_group:
                duplicate_ids.add(str(episode_id))
            by_group[group_name] = episode
        else:
            warnings.append("存在缺失或非法 episode_id 的 metadata episode")
        mode = episode.get("control_mode")
        if isinstance(mode, str) and mode:
            control_modes.add(mode)
        else:
            unknown_control += 1
        elapsed = episode.get("elapsed_steps")
        if isinstance(elapsed, int) and not isinstance(elapsed, bool) and elapsed >= 0:
            elapsed_steps.append(elapsed)
        value, source = _episode_success(episode, warnings)
        if value is not None:
            success_values.append(value)
            success_sources.add(source)
    if duplicate_ids:
        warnings.append(f"metadata episode_id 重复: {sorted(duplicate_ids)}")
    if len(control_modes) > 1:
        warnings.append(
            f"metadata 包含多个 episode control_mode: {sorted(control_modes)}；"
            "后续不可按单一动作语义合并"
        )

    control_source = "episodes[].control_mode" if control_modes else UNKNOWN
    if not control_modes:
        fallback_mode = _first_field(
            metadata,
            ("env_info.env_kwargs.control_mode", "control_mode"),
        )
        if (
            fallback_mode["status"] == "available"
            and isinstance(fallback_mode["value"], str)
            and fallback_mode["value"]
        ):
            control_modes.add(fallback_mode["value"])
            control_source = fallback_mode["source"]

    known_success = len(success_values)
    total = len(episodes)
    success_report = {
        "status": (
            UNKNOWN
            if known_success == 0
            else ("available" if known_success == total else "partial")
        ),
        "metadata_episode_total": total,
        "known_episode_denominator": known_success,
        "unknown_episode_count": total - known_success,
        "successful_episode_count": sum(success_values),
        "success_rate_over_known": (
            sum(success_values) / known_success if known_success else UNKNOWN
        ),
        "sources": sorted(success_sources) if success_sources else UNKNOWN,
    }
    env_info = metadata.get("env_info") if isinstance(metadata.get("env_info"), dict) else {}
    if metadata and not env_info:
        warnings.append("metadata.env_info 缺失或不是对象")
    report = {
        "environment_id": _first_field(metadata, ("env_info.env_id", "env_id")),
        "robot": _first_field(
            metadata,
            (
                "env_info.env_kwargs.robot_uids",
                "env_info.env_kwargs.robot_uid",
                "env_info.env_kwargs.robot_id",
                "robot_uids",
                "robot_uid",
            ),
        ),
        "control_modes": {
            "status": "available" if control_modes else UNKNOWN,
            "value": sorted(control_modes) if control_modes else UNKNOWN,
            "known_episode_count": total - unknown_control,
            "unknown_episode_count": unknown_control,
            "source": control_source,
        },
        "timing": {
            "control_hz": _first_field(
                metadata,
                (
                    "env_info.env_kwargs.control_hz",
                    "env_info.env_kwargs.control_freq",
                    "env_info.env_kwargs.sim_config.control_freq",
                    "control_hz",
                    "control_freq",
                ),
            ),
            "sim_hz": _first_field(
                metadata,
                (
                    "env_info.env_kwargs.sim_hz",
                    "env_info.env_kwargs.sim_freq",
                    "env_info.env_kwargs.sim_config.sim_freq",
                    "sim_hz",
                    "sim_freq",
                ),
            ),
            "max_episode_steps": _first_field(
                metadata, ("env_info.max_episode_steps", "max_episode_steps")
            ),
            "elapsed_steps": {
                "status": "available" if elapsed_steps else UNKNOWN,
                "known_episode_count": len(elapsed_steps),
                "unknown_episode_count": total - len(elapsed_steps),
                "values": sorted(set(elapsed_steps)) if elapsed_steps else UNKNOWN,
                "source": "episodes[].elapsed_steps" if elapsed_steps else UNKNOWN,
            },
            "note": "elapsed_steps/max_episode_steps 是步数，不据此推断秒或控制频率",
        },
        "success": success_report,
        "source_type": _first_field(metadata, ("source_type",)),
    }
    return report, by_group


def _check_expected_length(
    trajectory: str,
    field_name: str,
    field: dict[str, Any],
    expected: int,
    warnings: list[str],
) -> None:
    if not field["present"]:
        return
    lengths = field["lengths"]
    if not isinstance(lengths, list) or lengths != [expected]:
        warnings.append(f"{trajectory}/{field_name} 长度 {lengths}，期望 {expected}")


def _trajectory_report(
    name: str,
    group: Any,
    h5py: Any,
    metadata_episode: dict[str, Any] | None,
    warnings: list[str],
) -> dict[str, Any]:
    inventory = _dataset_inventory(group, h5py)
    fields = {
        key: _node_lengths(group, key, inventory)
        for key in ("actions", "obs", "env_states", "terminated", "truncated", "success", "fail")
    }
    action_length = _single_length(fields["actions"])
    action_dataset = next((row for row in inventory if row["path"] == "actions"), None)
    action_dim: int | str = UNKNOWN
    if action_dataset is not None and len(action_dataset["shape"]) >= 2:
        action_dim = action_dataset["shape"][-1]
    if action_dataset is None:
        warnings.append(f"{name} 缺少 actions dataset")
    else:
        if action_dataset["dtype"] != "float32":
            warnings.append(
                f"{name}/actions dtype 为 {action_dataset['dtype']}，"
                "官方常规格式期望 float32"
            )
        if action_length is not None:
            for field_name in ("terminated", "truncated", "success", "fail"):
                _check_expected_length(
                    name, field_name, fields[field_name], action_length, warnings
                )
            for field_name in ("obs", "env_states"):
                _check_expected_length(
                    name, field_name, fields[field_name], action_length + 1, warnings
                )
    if metadata_episode is not None and action_length is not None:
        elapsed = metadata_episode.get("elapsed_steps")
        if isinstance(elapsed, int) and not isinstance(elapsed, bool) and elapsed != action_length:
            warnings.append(
                f"{name} actions 长度 {action_length} 与 metadata "
                f"elapsed_steps {elapsed} 不一致"
            )
    timestamp_paths = [
        row["path"]
        for row in inventory
        if row["path"].split("/")[-1].lower() in {"time", "times", "timestamp", "timestamps"}
        or "timestamp" in row["path"].split("/")[-1].lower()
    ]
    return {
        "name": name,
        "metadata_episode_present": metadata_episode is not None,
        "action_length": action_length if action_length is not None else UNKNOWN,
        "action_dim": action_dim,
        "observation_length": fields["obs"],
        "env_states_length": fields["env_states"],
        "success_length": fields["success"],
        "terminated_length": fields["terminated"],
        "truncated_length": fields["truncated"],
        "timestamp_dataset_paths": timestamp_paths if timestamp_paths else UNKNOWN,
        "datasets": inventory,
    }


def _adapter_preflight(
    metadata_report: dict[str, Any], trajectories: list[dict[str, Any]]
) -> dict[str, Any]:
    env_id = metadata_report["environment_id"]["value"]
    qpos_paths: list[str] = []
    qvel_paths: list[str] = []
    action_dims: set[int] = set()
    for trajectory in trajectories:
        for row in trajectory.get("datasets", []):
            leaf = row["path"].split("/")[-1].lower()
            qualified = f"{trajectory['name']}/{row['path']}"
            if leaf == "qpos":
                qpos_paths.append(qualified)
            elif leaf == "qvel":
                qvel_paths.append(qualified)
        if isinstance(trajectory.get("action_dim"), int):
            action_dims.add(trajectory["action_dim"])
    if env_id == CUSTOM_ENV_ID:
        task_status = "matching_environment_id_but_semantics_still_require_validation"
    elif env_id == "PickCube-v1":
        task_status = "requires_task_mapping"
    else:
        task_status = UNKNOWN if env_id == UNKNOWN else "requires_task_mapping"
    return {
        "overall_status": "preflight_only_not_converted",
        "conversion_performed": False,
        "target_contract": {
            "proprio": "float32 [T+1,15] = arm q[7] + arm dq[7] + calibrated gripper opening[1]",
            "tcp_action_chunk": (
                "float32 [N,16,7] = fixed-anchor commanded TCP delta "
                "[translation3 + rotvec3] + gripper target[1]"
            ),
            "chunk_label_rule": "首步 actual TCP→commanded TCP；后续 commanded TCP→commanded TCP",
            "control_hz": 20.0,
        },
        "task_identity": {
            "status": task_status,
            "source_environment_id": env_id,
            "project_environment_id": CUSTOM_ENV_ID,
            "reason": (
                "官方 PickCube-v1 与项目环境的采样、目标可见性、释放稳定判据不同；"
                "不得直接合并任务标签或成功率"
            ),
        },
        "robot_identity": {
            "status": "requires_joint_name_and_calibration_validation",
            "metadata": metadata_report["robot"],
            "reason": "即使 metadata 指向 Panda，也仍须核验 9 个 active joint 的顺序和双指开口标定",
        },
        "proprio_15d": {
            "status": (
                "requires_replay_or_field_validation"
                if qpos_paths and qvel_paths
                else "blocked_missing_direct_qpos_qvel"
            ),
            "qpos_paths": qpos_paths if qpos_paths else UNKNOWN,
            "qvel_paths": qvel_paths if qvel_paths else UNKNOWN,
            "reason": "字段名/shape 清单不能证明 joint 顺序、单位、数值有效性或 gripper 标定",
        },
        "tcp_chunk_16x7": {
            "status": "blocked_requires_command_provenance_and_fk_replay",
            "source_action_dims": sorted(action_dims) if action_dims else UNKNOWN,
            "reason": (
                "原始 action 为 7D 也不等于项目 TCP 7D；需按 JSON control_mode 重放，"
                "取得 actual TCP、每步 commanded target、gripper target、时间与 frame 后再构造标签"
            ),
        },
        "timing": {
            "status": (
                "metadata_available"
                if metadata_report["timing"]["control_hz"]["status"] == "available"
                else "blocked_control_hz_unknown"
            ),
            "metadata_control_hz": metadata_report["timing"]["control_hz"],
            "reason": "20 Hz 目标合同不能从 episode 步数或数组长度推断",
        },
        "required_next_steps": [
            "在固定 mani-skill==3.0.1 环境核验 JSON env_id/env_kwargs/reset_kwargs/control_mode",
            "按原 control_mode 重放成功 episode，显式采集 joint names、qpos/qvel、"
            "actual TCP 和 commanded targets",
            "为 PickCube-v1→RobotVLAPickCubeToRegion-v1 定义隔离的任务/成功语义映射，保留来源身份",
            "用现有 command_chunk 规则生成候选标签后，再做 shape、frame、时间和 command provenance 审计",
        ],
    }


def inspect_demo(
    h5_path: str | Path, metadata_path: str | Path | None = None
) -> dict[str, Any]:
    """生成只读 preflight report；不会读取任何 HDF5 dataset payload。"""
    source = Path(h5_path)
    if not source.is_file():
        raise FileNotFoundError(f"HDF5 文件不存在: {source}")
    resolved_metadata = (
        Path(metadata_path)
        if metadata_path is not None
        else source.with_suffix(".json")
    )
    warnings: list[str] = []
    metadata, metadata_status = _load_metadata(resolved_metadata, warnings)
    metadata_report, metadata_by_group = _metadata_report(metadata, warnings)
    try:
        import h5py  # 惰性导入：模块与 --help 不要求预先安装 h5py。
    except ModuleNotFoundError as error:
        raise RuntimeError("检查 HDF5 需要 h5py；当前环境未安装") from error

    trajectories: list[dict[str, Any]] = []
    root_non_trajectory: list[str] = []
    with h5py.File(source, "r") as handle:
        for name in sorted(handle.keys()):
            node = handle[name]
            if name.startswith("traj_") and isinstance(node, h5py.Group):
                trajectories.append(
                    _trajectory_report(name, node, h5py, metadata_by_group.get(name), warnings)
                )
            else:
                root_non_trajectory.append(name)
                if name.startswith("traj_"):
                    warnings.append(f"根节点 {name} 不是 HDF5 group")
    h5_names = {row["name"] for row in trajectories}
    missing_h5 = sorted(set(metadata_by_group) - h5_names)
    unlisted_h5 = (
        sorted(h5_names - set(metadata_by_group))
        if metadata_status == "available"
        else []
    )
    if missing_h5:
        warnings.append(f"metadata 中存在但 HDF5 缺失的轨迹: {missing_h5}")
    if unlisted_h5:
        warnings.append(f"HDF5 中存在但 metadata 未列出的轨迹: {unlisted_h5}")

    env_presence = [row["env_states_length"]["present"] for row in trajectories]
    hdf5_report = {
        "trajectory_count": len(trajectories),
        "root_non_trajectory_keys": root_non_trajectory,
        "env_states": {
            "status": (
                UNKNOWN
                if not trajectories
                else (
                    "all"
                    if all(env_presence)
                    else ("none" if not any(env_presence) else "partial")
                )
            ),
            "present_trajectory_count": sum(env_presence),
            "trajectory_denominator": len(trajectories),
        },
        "success_dataset": {
            "present_trajectory_count": sum(
                row["success_length"]["present"] for row in trajectories
            ),
            "trajectory_denominator": len(trajectories),
            "values_read": False,
        },
        "missing_hdf5_for_metadata": missing_h5,
        "unlisted_hdf5_trajectories": unlisted_h5,
        "trajectories": trajectories,
    }
    return {
        "schema": REPORT_SCHEMA,
        "mode": "read_only_shape_dtype_inventory",
        "hdf5_path": str(source),
        "metadata_path": str(resolved_metadata),
        "metadata_status": metadata_status,
        "metadata": metadata_report,
        "hdf5": hdf5_report,
        "adapter_preflight": _adapter_preflight(metadata_report, trajectories),
        "warnings": warnings,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="只读检查官方 ManiSkill demo 的 HDF5 层级及 JSON metadata；不转换数据。"
    )
    parser.add_argument("hdf5", help="官方 trajectory*.h5 路径")
    parser.add_argument("--metadata", help="JSON metadata 路径；默认使用 HDF5 同名 .json")
    parser.add_argument("--compact", action="store_true", help="输出单行 JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = inspect_demo(args.hdf5, args.metadata)
    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=None if args.compact else 2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
