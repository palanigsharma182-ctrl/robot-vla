"""验证大陆 GPU 环境是否满足 qwen-vla-v0.1 的运行边界。"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

QWEN_MODEL_ID = "Qwen/Qwen3.5-2B"
QWEN_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
EXPECTED_VERSIONS = {
    "accelerate": "1.14.0",
    "mani_skill": "3.0.1",
    "transformers": "5.15.1",
}
RUNTIME_PROFILES = {
    "container-cu132": {
        "torch": "2.13.0+cu132",
        "torchvision": "0.28.0+cu132",
        "cuda": "13.2",
    },
    "vm-cu128": {
        "torch": "2.11.0+cu128",
        "torchvision": "0.26.0+cu128",
        "cuda": "12.8",
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qwen-config",
        action="store_true",
        help="通过固定 revision 下载并检查 Qwen 配置，不下载模型权重。",
    )
    parser.add_argument(
        "--maniskill",
        action="store_true",
        help="创建 PickCube 双相机环境；要求宿主机完整透传 NVIDIA Vulkan。",
    )
    parser.add_argument(
        "--hf-endpoint",
        default="https://hf-mirror.com",
        help="Hugging Face 兼容端点，默认使用大陆可访问的 HF Mirror。",
    )
    parser.add_argument(
        "--runtime-profile",
        choices=sorted(RUNTIME_PROFILES),
        default=os.environ.get("ROBOT_VLA_RUNTIME_PROFILE", "container-cu132"),
        help="选择需要严格验证的 PyTorch/CUDA 基础环境。",
    )
    return parser.parse_args()


def _version(module: object) -> str:
    return str(getattr(module, "__version__", "unknown"))


def _check_packages(runtime_profile: str) -> list[str]:
    import accelerate
    import mani_skill
    import torch
    import torchvision
    import transformers

    modules = {
        "accelerate": accelerate,
        "mani_skill": mani_skill,
        "torch": torch,
        "torchvision": torchvision,
        "transformers": transformers,
    }
    expected_versions = {
        **EXPECTED_VERSIONS,
        "torch": RUNTIME_PROFILES[runtime_profile]["torch"],
        "torchvision": RUNTIME_PROFILES[runtime_profile]["torchvision"],
    }
    errors: list[str] = []
    print(f"runtime_profile={runtime_profile}")
    for name, expected in expected_versions.items():
        actual = _version(modules[name])
        print(f"{name}={actual}")
        if actual != expected:
            errors.append(f"{name} 版本应为 {expected}，实际为 {actual}")

    print(f"cuda_build={torch.version.cuda}")
    expected_cuda = RUNTIME_PROFILES[runtime_profile]["cuda"]
    if torch.version.cuda != expected_cuda:
        errors.append(f"PyTorch CUDA 应为 {expected_cuda}，实际为 {torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        errors.append("PyTorch 无法访问 CUDA")
        return errors

    print(f"cuda_device={torch.cuda.get_device_name(0)}")
    print(f"bf16_supported={torch.cuda.is_bf16_supported()}")
    total_memory = torch.cuda.get_device_properties(0).total_memory
    print(f"vram_gib={total_memory / 1024**3:.2f}")
    if total_memory < 23 * 1024**3:
        errors.append("可见 GPU 显存不足 23 GiB")
    if not torch.cuda.is_bf16_supported():
        errors.append("当前 GPU/PyTorch 不支持 BF16")
    if not hasattr(transformers, "Qwen3_5ForConditionalGeneration"):
        errors.append("Transformers 不包含 Qwen3_5ForConditionalGeneration")
    return errors


def _check_qwen_config() -> list[str]:
    from transformers import AutoConfig

    errors: list[str] = []
    with tempfile.TemporaryDirectory(prefix="robot-vla-qwen-config-") as cache_dir:
        config = AutoConfig.from_pretrained(
            QWEN_MODEL_ID,
            revision=QWEN_REVISION,
            cache_dir=cache_dir,
            trust_remote_code=False,
        )

    facts = {
        "model_type": config.model_type,
        "text_hidden_size": config.text_config.hidden_size,
        "text_layers": config.text_config.num_hidden_layers,
        "vision_hidden_size": config.vision_config.hidden_size,
        "vision_depth": config.vision_config.depth,
        "vision_out_hidden_size": config.vision_config.out_hidden_size,
    }
    for name, value in facts.items():
        print(f"qwen_{name}={value}")

    expected = {
        "model_type": "qwen3_5",
        "text_hidden_size": 2048,
        "text_layers": 24,
        "vision_hidden_size": 1024,
        "vision_depth": 24,
        "vision_out_hidden_size": 2048,
    }
    for name, expected_value in expected.items():
        if facts[name] != expected_value:
            errors.append(f"Qwen {name} 应为 {expected_value}，实际为 {facts[name]}")
    return errors


def _check_vulkan() -> list[str]:
    result = subprocess.run(
        ["vulkaninfo", "--summary"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    output = f"{result.stdout}\n{result.stderr}".strip()
    if result.returncode != 0:
        return [f"Vulkan 初始化失败：{output}"]
    print(output)
    return []


def _print_shapes(value: object, prefix: str = "obs") -> None:
    if hasattr(value, "shape"):
        print(f"{prefix}_shape={tuple(value.shape)}")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            _print_shapes(child, f"{prefix}.{key}")


def _check_maniskill() -> list[str]:
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    errors = _check_vulkan()
    if errors:
        return errors

    env = gym.make(
        "PickCube-v1",
        num_envs=1,
        obs_mode="rgb",
        control_mode="pd_joint_delta_pos",
        render_mode="rgb_array",
    )
    try:
        obs, _ = env.reset(seed=7)
        print(f"maniskill_device={env.unwrapped.device}")
        _print_shapes(obs)
        frame = env.render()
        print(f"maniskill_render_shape={tuple(frame.shape)}")
    finally:
        env.close()
    return []


def main() -> int:
    args = _parse_args()
    os.environ.setdefault("HF_ENDPOINT", args.hf_endpoint)
    errors = _check_packages(args.runtime_profile)

    if args.qwen_config:
        try:
            errors.extend(_check_qwen_config())
        except Exception as exc:  # noqa: BLE001 - 自检应汇总所有失败原因
            errors.append(f"Qwen 配置检查失败：{type(exc).__name__}: {exc}")

    if args.maniskill:
        try:
            errors.extend(_check_maniskill())
        except Exception as exc:  # noqa: BLE001 - 自检应汇总所有失败原因
            errors.append(f"ManiSkill 检查失败：{type(exc).__name__}: {exc}")

    if errors:
        print("\n运行环境检查失败：", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("\n运行环境检查通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
