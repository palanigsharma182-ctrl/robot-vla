"""供父 Agent 在云端运行的合成 HDF5 targeted tests。"""

from __future__ import annotations

import json

import h5py
import numpy as np

from experiments.maniskill_data.inspect_demo import inspect_demo, main


def _write_metadata(path, episodes):
    path.write_text(
        json.dumps(
            {
                "env_info": {
                    "env_id": "PickCube-v1",
                    "max_episode_steps": 200,
                    "env_kwargs": {"robot_uids": "panda_wristcam", "control_freq": 20},
                },
                "episodes": episodes,
                "source_type": "motionplanning",
            }
        ),
        encoding="utf-8",
    )


def test_inventory_reports_lengths_success_denominator_and_adapter_blocks(tmp_path):
    h5_path = tmp_path / "trajectory.h5"
    with h5py.File(h5_path, "w") as handle:
        trajectory = handle.create_group("traj_0")
        trajectory.create_dataset("actions", shape=(3, 7), dtype=np.float32)
        trajectory.create_dataset("terminated", shape=(3,), dtype=np.bool_)
        trajectory.create_dataset("truncated", shape=(3,), dtype=np.bool_)
        trajectory.create_dataset("success", shape=(3,), dtype=np.bool_)
        obs = trajectory.create_group("obs").create_group("agent")
        obs.create_dataset("qpos", shape=(4, 9), dtype=np.float32)
        obs.create_dataset("qvel", shape=(4, 9), dtype=np.float32)
        states = trajectory.create_group("env_states").create_group("articulations")
        states.create_dataset("panda", shape=(4, 18), dtype=np.float32)
    _write_metadata(
        h5_path.with_suffix(".json"),
        [
            {
                "episode_id": 0,
                "control_mode": "pd_joint_pos",
                "elapsed_steps": 3,
                "info": {"success": True},
            }
        ],
    )

    report = inspect_demo(h5_path)

    assert report["metadata"]["success"] == {
        "status": "available",
        "metadata_episode_total": 1,
        "known_episode_denominator": 1,
        "unknown_episode_count": 0,
        "successful_episode_count": 1,
        "success_rate_over_known": 1.0,
        "sources": ["episodes[].info.success"],
    }
    assert report["hdf5"]["env_states"]["status"] == "all"
    trajectory = report["hdf5"]["trajectories"][0]
    assert trajectory["action_length"] == 3
    assert trajectory["action_dim"] == 7
    assert trajectory["observation_length"]["lengths"] == [4]
    assert {row["path"] for row in trajectory["datasets"]} >= {
        "actions",
        "obs/agent/qpos",
        "obs/agent/qvel",
        "env_states/articulations/panda",
    }
    preflight = report["adapter_preflight"]
    assert preflight["overall_status"] == "preflight_only_not_converted"
    assert preflight["task_identity"]["status"] == "requires_task_mapping"
    assert preflight["proprio_15d"]["status"] == "requires_replay_or_field_validation"
    assert (
        preflight["tcp_chunk_16x7"]["status"]
        == "blocked_requires_command_provenance_and_fk_replay"
    )


def test_missing_metadata_and_observation_are_explicit_unknown(tmp_path):
    h5_path = tmp_path / "compressed.h5"
    with h5py.File(h5_path, "w") as handle:
        trajectory = handle.create_group("traj_7")
        trajectory.create_dataset("actions", shape=(2, 7), dtype=np.float32)
        trajectory.create_dataset("env_states", shape=(3, 12), dtype=np.float32)

    report = inspect_demo(h5_path)

    assert report["metadata_status"] == "missing"
    assert report["metadata"]["environment_id"]["value"] == "unknown"
    assert report["metadata"]["control_modes"]["value"] == "unknown"
    trajectory = report["hdf5"]["trajectories"][0]
    assert trajectory["observation_length"]["status"] == "unknown"
    assert trajectory["env_states_length"]["lengths"] == [3]
    assert (
        report["adapter_preflight"]["timing"]["status"]
        == "blocked_control_hz_unknown"
    )


def test_partial_success_denominator_and_length_mismatches_are_preserved(tmp_path):
    h5_path = tmp_path / "trajectory.h5"
    with h5py.File(h5_path, "w") as handle:
        trajectory = handle.create_group("traj_0")
        trajectory.create_dataset("actions", shape=(3, 6), dtype=np.float64)
        trajectory.create_dataset("terminated", shape=(2,), dtype=np.bool_)
        obs = trajectory.create_group("obs")
        obs.create_dataset("agent", shape=(3, 15), dtype=np.float32)
    _write_metadata(
        h5_path.with_suffix(".json"),
        [
            {
                "episode_id": 0,
                "control_mode": "pd_joint_pos",
                "elapsed_steps": 4,
                "success": False,
            },
            {"episode_id": 1, "elapsed_steps": 2},
        ],
    )

    report = inspect_demo(h5_path)

    success = report["metadata"]["success"]
    assert success["status"] == "partial"
    assert success["known_episode_denominator"] == 1
    assert success["unknown_episode_count"] == 1
    assert success["success_rate_over_known"] == 0.0
    assert report["hdf5"]["missing_hdf5_for_metadata"] == ["traj_1"]
    assert any("actions dtype" in warning for warning in report["warnings"])
    assert any("obs 长度" in warning for warning in report["warnings"])
    assert any("elapsed_steps" in warning for warning in report["warnings"])


def test_cli_outputs_json(tmp_path, capsys):
    h5_path = tmp_path / "trajectory.h5"
    with h5py.File(h5_path, "w") as handle:
        trajectory = handle.create_group("traj_0")
        trajectory.create_dataset("actions", shape=(1, 7), dtype=np.float32)

    assert main([str(h5_path), "--compact"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["mode"] == "read_only_shape_dtype_inventory"
    assert output["adapter_preflight"]["conversion_performed"] is False
