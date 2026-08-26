# 大陆 RTX 4090 完整虚拟机环境

本目录用于在 Ubuntu 22.04 KVM/裸金属 GPU 实例上构建完整的
`qwen-vla-v0.1` 开发环境。与 `infra/gpu_container` 不同，本 profile 不依赖云厂商
预装 PyTorch，而是在独立 venv 中固定安装 CUDA 12.8 wheel。

## 已验证基础边界

```text
Ubuntu       22.04
GPU          RTX 4090 24GB
Driver       570.153.02
Driver CUDA  12.8
Python       3.10
PyTorch      2.11.0+cu128
Torchvision  0.26.0+cu128
```

脚本不会安装、升级或卸载 NVIDIA 驱动。如果 `nvidia-smi`、NVIDIA ICD 或
`vulkaninfo --summary` 失败，脚本会停止，而不是用 Mesa 软件渲染伪装成通过。

## 构建

```bash
bash infra/gpu_vm/bootstrap.sh
```

默认项目环境为 `/opt/robot-vla/env`。PyPI 使用清华镜像，PyTorch 使用官方 CUDA 12.8
wheel 源；均可通过环境变量覆盖。

## 验证

安装脚本会把当前用户加入 `render` 组。重新登录 SSH 后执行：

```bash
/opt/robot-vla/env/bin/python \
  infra/gpu_container/verify_runtime.py \
  --runtime-profile vm-cu128 \
  --qwen-config \
  --maniskill
```

该检查同时验证：

- CUDA、BF16 和 24GB 显存；
- 固定 Qwen3.5-2B revision 的结构字段；
- NVIDIA Vulkan 实例；
- ManiSkill `PickCube-v1` RGB 环境和一帧渲染。

只有整条命令退出码为 0，才可以制作完整系统镜像。

## 镜像清单

验证通过后，在清理软件包缓存之前记录环境和构建输入：

```bash
sudo bash infra/gpu_container/capture_manifest.sh
```

清单默认写入 `/opt/robot-vla/image-manifest`，只保存系统、GPU、包版本和构建脚本哈希，
不采集环境变量、Token、SSH 私钥或主机网络信息。
