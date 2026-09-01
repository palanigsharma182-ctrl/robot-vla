"""批量采集并审计可信 ManiSkill pick-place 数据。"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from robot_vla.contracts import RobotSpec
from robot_vla.data.audit import audit_dataset
from robot_vla.data.recovery import RECOVERY_PROFILES
from robot_vla.data.trajectory import load_manifest
from robot_vla.data.writer import plan_scene_splits
from robot_vla.precision.data import audit_precision_dataset
from robot_vla.sim import PICK_CUBE_TO_REGION_ENV_ID
from robot_vla.sim.collector import EpisodeRejected, TrustedPickPlaceCollector


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--precision-label-output",
        type=Path,
        default=None,
        help="可选的 E013 privileged label sibling root；不得位于 deployable output 内",
    )
    parser.add_argument("--train", type=int, default=24)
    parser.add_argument("--val", type=int, default=3)
    parser.add_argument("--test", type=int, default=3)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--max-candidates", type=int, default=300)
    parser.add_argument(
        "--extend-existing",
        action="store_true",
        help="只允许在环境、seed 范围和 split 比例不变时提高已有数据集目标数",
    )
    parser.add_argument(
        "--recovery-profiles",
        nargs="+",
        choices=RECOVERY_PROFILES,
        default=(),
        help="为新增轨迹循环插入指定的可信失败恢复扰动",
    )
    parser.add_argument(
        "--recovery-profile-targets",
        type=int,
        nargs="+",
        default=(),
        help="与 recovery-profiles 对齐的 Dataset 最终各 profile 轨迹总数",
    )
    return parser.parse_args()


def _collection_config(args: argparse.Namespace) -> dict[str, object]:
    config: dict[str, object] = {
        "environment_id": PICK_CUBE_TO_REGION_ENV_ID,
        "train": args.train,
        "val": args.val,
        "test": args.test,
        "start_seed": args.start_seed,
        "max_candidates": args.max_candidates,
        "precision_label_sidecar": args.precision_label_output is not None,
    }
    if args.recovery_profiles:
        config["recovery_profiles"] = list(args.recovery_profiles)
    if args.recovery_profile_targets:
        config["recovery_profile_targets"] = list(args.recovery_profile_targets)
    return config


def _is_compatible_extension(
    existing: dict[str, object],
    requested: dict[str, object],
) -> bool:
    fixed_keys = ("environment_id", "start_seed", "max_candidates")
    if any(existing.get(key) != requested.get(key) for key in fixed_keys):
        return False
    old_profiles = tuple(existing.get("recovery_profiles", ()))
    new_profiles = tuple(requested.get("recovery_profiles", ()))
    if old_profiles and old_profiles != new_profiles:
        return False
    splits = ("train", "val", "test")
    try:
        old_counts = {split: int(existing[split]) for split in splits}
        new_counts = {split: int(requested[split]) for split in splits}
    except (KeyError, TypeError, ValueError):
        return False
    if any(new_counts[split] < old_counts[split] for split in splits):
        return False
    old_total = sum(old_counts.values())
    new_total = sum(new_counts.values())
    return old_total > 0 and new_total > 0 and all(
        old_counts[split] * new_total == new_counts[split] * old_total
        for split in splits
    )


def _check_or_write_config(
    output: Path,
    config: dict[str, object],
    *,
    allow_extension: bool = False,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "collection_config.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing == config:
            return
        if not allow_extension or not _is_compatible_extension(existing, config):
            raise ValueError("已有数据集的 collection_config 与本次参数不兼容")
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=".collection_config.",
        suffix=".tmp",
        dir=output,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(config, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _existing_state(output: Path) -> tuple[dict[str, int], set[int]]:
    counts = {split: 0 for split in ("train", "val", "test")}
    seeds: set[int] = set()
    manifest = output / "manifest.jsonl"
    if not manifest.is_file() or not manifest.read_text(encoding="utf-8").strip():
        return counts, seeds
    for entry in load_manifest(output):
        counts[entry.split] += 1
        seeds.add(int(entry.randomization["seed"]))
    return counts, seeds


def _existing_recovery_counts(
    output: Path,
    profiles: tuple[str, ...],
) -> dict[str, int]:
    counts = {profile: 0 for profile in profiles}
    manifest = output / "manifest.jsonl"
    if not manifest.is_file() or not manifest.read_text(encoding="utf-8").strip():
        return counts
    for entry in load_manifest(output):
        profile = entry.randomization.get("recovery_profile")
        if profile in counts:
            counts[profile] += 1
    return counts


def _next_recovery_profile(
    profiles: tuple[str, ...],
    counts: dict[str, int],
    targets: dict[str, int] | None = None,
) -> str | None:
    if not profiles:
        return None
    order = {profile: index for index, profile in enumerate(profiles)}
    if targets is not None:
        available = [profile for profile in profiles if counts[profile] < targets[profile]]
        if not available:
            return None
        return min(
            available,
            key=lambda profile: (
                counts[profile] / targets[profile],
                order[profile],
            ),
        )
    return min(profiles, key=lambda profile: (counts[profile], order[profile]))


def _validate_existing_split_plan(output: Path, split_map: dict[str, str]) -> None:
    manifest = output / "manifest.jsonl"
    if not manifest.is_file() or not manifest.read_text(encoding="utf-8").strip():
        return
    for entry in load_manifest(output):
        expected = split_map.get(entry.scene_id)
        if expected is None or expected != entry.split:
            raise ValueError(
                f"已有 scene split 与当前确定性计划不一致: "
                f"{entry.scene_id} actual={entry.split} expected={expected}"
            )


def collect_dataset(args: argparse.Namespace) -> None:
    target = {"train": args.train, "val": args.val, "test": args.test}
    recovery_profiles = tuple(args.recovery_profiles)
    recovery_profile_targets = tuple(args.recovery_profile_targets)
    if any(value <= 0 for value in target.values()):
        raise ValueError("train/val/test 目标轨迹数必须都是正数")
    if args.start_seed < 0 or args.max_candidates < sum(target.values()):
        raise ValueError("seed 范围或 max_candidates 无效")
    if len(set(recovery_profiles)) != len(recovery_profiles):
        raise ValueError("recovery_profiles 不能重复")
    if recovery_profile_targets:
        if len(recovery_profile_targets) != len(recovery_profiles):
            raise ValueError("recovery-profile-targets 必须与 recovery-profiles 等长")
        if any(target <= 0 for target in recovery_profile_targets):
            raise ValueError("recovery-profile-targets 必须全部为正整数")
    profile_targets = (
        dict(zip(recovery_profiles, recovery_profile_targets, strict=True))
        if recovery_profile_targets
        else None
    )

    config = _collection_config(args)
    _check_or_write_config(
        args.output,
        config,
        allow_extension=args.extend_existing,
    )
    counts, existing_seeds = _existing_state(args.output)
    recovery_counts = _existing_recovery_counts(args.output, recovery_profiles)
    if any(counts[split] > target[split] for split in target):
        raise ValueError(f"已有 split 数量超过目标: existing={counts}, target={target}")
    if profile_targets is not None:
        if any(recovery_counts[name] > profile_targets[name] for name in recovery_profiles):
            raise ValueError("已有 recovery profile 数量超过请求目标")
        remaining_trajectories = sum(target.values()) - sum(counts.values())
        remaining_recovery = sum(
            profile_targets[name] - recovery_counts[name]
            for name in recovery_profiles
        )
        if remaining_recovery != remaining_trajectories:
            raise ValueError(
                "recovery profile 剩余目标必须等于待采集轨迹数："
                f"profiles={remaining_recovery}, trajectories={remaining_trajectories}"
            )

    seeds = list(range(args.start_seed, args.start_seed + args.max_candidates))
    scene_ids = [f"{PICK_CUBE_TO_REGION_ENV_ID}:seed={seed}" for seed in seeds]
    total = sum(target.values())
    split_map = plan_scene_splits(
        scene_ids,
        train_fraction=target["train"] / total,
        val_fraction=target["val"] / total,
    )
    _validate_existing_split_plan(args.output, split_map)
    failure_path = args.output / "collection_failures.jsonl"

    with TrustedPickPlaceCollector(
        args.output,
        RobotSpec(),
        precision_label_root=args.precision_label_output,
    ) as collector:
        for seed, scene_id in zip(seeds, scene_ids, strict=True):
            split = split_map[scene_id]
            if seed in existing_seeds or counts[split] >= target[split]:
                continue
            recovery_profile = _next_recovery_profile(
                recovery_profiles,
                recovery_counts,
                profile_targets,
            )
            if recovery_profiles and recovery_profile is None:
                raise RuntimeError("recovery profile 目标已耗尽，但 split 轨迹目标尚未完成")
            try:
                meta = collector.collect(
                    seed=seed,
                    split=split,
                    recovery_profile=recovery_profile,
                )
            except EpisodeRejected as exc:
                with failure_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "seed": seed,
                                "split": split,
                                "recovery_profile": recovery_profile,
                                "reason": str(exc),
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                print(
                    f"REJECT seed={seed} split={split} "
                    f"recovery={recovery_profile}: {exc}",
                    flush=True,
                )
                continue
            counts[split] += 1
            existing_seeds.add(seed)
            if recovery_profile is not None:
                recovery_counts[recovery_profile] += 1
            print(
                f"ACCEPT {meta.trajectory_id} steps={meta.num_steps} "
                f"counts={counts} recovery_counts={recovery_counts}",
                flush=True,
            )
            if counts == target:
                break

    if counts != target:
        raise RuntimeError(f"候选 seed 耗尽：只采集到 {counts}，目标为 {target}")
    report = audit_dataset(args.output, RobotSpec())
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True), flush=True)
    if args.precision_label_output is not None:
        precision_report = audit_precision_dataset(
            args.output,
            args.precision_label_output,
            RobotSpec(),
        )
        if not precision_report.passed:
            raise RuntimeError("E013 Precision Dataset audit gate 未通过")
        print(json.dumps(precision_report.to_dict(), indent=2, sort_keys=True), flush=True)


def main() -> None:
    collect_dataset(_parse_args())


if __name__ == "__main__":
    main()
