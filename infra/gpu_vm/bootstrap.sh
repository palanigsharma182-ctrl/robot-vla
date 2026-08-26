#!/usr/bin/env bash
set -euo pipefail

# 为已经安装 NVIDIA 570 驱动的 Ubuntu 22.04 RTX 4090 完整虚拟机创建
# Python 3.10 + CUDA 12.8 项目环境。此脚本不会安装或升级 NVIDIA 驱动。

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
target_env="${ROBOT_VLA_ENV_DIR:-/opt/robot-vla/env}"
pypi_index="${ROBOT_VLA_PYPI_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
pytorch_index="${ROBOT_VLA_PYTORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
target_user="${ROBOT_VLA_USER:-${USER}}"

if [[ "$(id -u)" -eq 0 ]]; then
  sudo_cmd=()
else
  if ! sudo -n true 2>/dev/null; then
    echo "错误：需要 root 或免密码 sudo 来安装系统依赖。" >&2
    exit 1
  fi
  sudo_cmd=(sudo -n)
fi

if [[ ! -r /etc/os-release ]]; then
  echo "错误：无法识别操作系统。" >&2
  exit 1
fi

# shellcheck disable=SC1091
source /etc/os-release
if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "22.04" ]]; then
  echo "错误：该镜像脚本只验证过 Ubuntu 22.04，当前为 ${PRETTY_NAME:-未知系统}。" >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "错误：没有检测到 nvidia-smi；请先使用云厂商 NVIDIA GPU 基础镜像。" >&2
  exit 1
fi

driver_cuda="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -n 1)"
if [[ "${driver_cuda}" != "12.8" ]]; then
  echo "错误：该 profile 要求驱动支持 CUDA 12.8，实际报告 ${driver_cuda:-未知}。" >&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive
"${sudo_cmd[@]}" apt-get -o Acquire::Retries=3 update
"${sudo_cmd[@]}" apt-get install --yes --no-install-recommends \
  git \
  libgl1 \
  libglib2.0-0 \
  libvulkan1 \
  python3-venv \
  rsync \
  vulkan-tools

if getent group render >/dev/null 2>&1; then
  "${sudo_cmd[@]}" usermod -aG render "${target_user}"
fi

if ! "${sudo_cmd[@]}" vulkaninfo --summary >/dev/null 2>&1; then
  echo "错误：NVIDIA Vulkan 初始化失败，停止构建环境。" >&2
  exit 1
fi

if [[ ! -x "${target_env}/bin/python" ]]; then
  "${sudo_cmd[@]}" install -d -m 755 "$(dirname -- "${target_env}")"
  "${sudo_cmd[@]}" python3 -m venv "${target_env}"
fi

"${sudo_cmd[@]}" env PIP_DISABLE_PIP_VERSION_CHECK=1 \
  "${target_env}/bin/python" -m pip install \
  --index-url "${pypi_index}" \
  --upgrade \
  pip setuptools wheel

"${sudo_cmd[@]}" env PIP_DISABLE_PIP_VERSION_CHECK=1 \
  "${target_env}/bin/python" -m pip install \
  --index-url "${pytorch_index}" \
  'torch==2.11.0+cu128' \
  'torchvision==0.26.0+cu128'

"${sudo_cmd[@]}" env PIP_DISABLE_PIP_VERSION_CHECK=1 \
  "${target_env}/bin/python" -m pip install \
  --index-url "${pypi_index}" \
  --upgrade-strategy only-if-needed \
  --requirement "${script_dir}/requirements.direct.txt"

"${target_env}/bin/python" -m pip check
echo "环境已准备完成：${target_env}"
echo "重新登录后运行完整检查："
echo "${target_env}/bin/python infra/gpu_container/verify_runtime.py --runtime-profile vm-cu128 --qwen-config --maniskill"
