#!/usr/bin/env bash
set -euo pipefail

# 为云厂商已经注入 NVIDIA 驱动和 PyTorch 的 Ubuntu 22.04 GPU 容器创建
# 独立项目环境。此脚本不会安装 NVIDIA 驱动、CUDA apt 包或 Docker。

if [[ "$(id -u)" -ne 0 ]]; then
  echo "错误：系统依赖和 /opt 环境创建需要 root。" >&2
  exit 1
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source_env="${ROBOT_VLA_SOURCE_ENV:-/usr/local/miniconda3/envs/py312}"
target_env="${ROBOT_VLA_ENV_DIR:-/opt/robot-vla/env}"
conda_bin="${ROBOT_VLA_CONDA_BIN:-/usr/local/miniconda3/bin/conda}"
pypi_index="${ROBOT_VLA_PYPI_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"

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
  echo "错误：没有检测到 nvidia-smi；应先让云厂商为容器透传 GPU。" >&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get -o Acquire::Retries=3 update
apt-get install --yes --no-install-recommends \
  git \
  libgl1 \
  libglib2.0-0 \
  libvulkan1 \
  rsync \
  vulkan-tools

if [[ ! -x "${target_env}/bin/python" ]]; then
  if [[ ! -x "${conda_bin}" || ! -x "${source_env}/bin/python" ]]; then
    echo "错误：找不到可克隆的厂商 Python 环境：${source_env}" >&2
    exit 1
  fi
  install -d -m 755 "$(dirname -- "${target_env}")"
  "${conda_bin}" create --yes --prefix "${target_env}" --clone "${source_env}"
fi

PIP_DISABLE_PIP_VERSION_CHECK=1 \
PIP_ROOT_USER_ACTION=ignore \
"${target_env}/bin/python" -m pip install \
  --index-url "${pypi_index}" \
  --upgrade-strategy only-if-needed \
  --requirement "${script_dir}/requirements.direct.txt"

"${target_env}/bin/python" -m pip check
echo "环境已准备完成：${target_env}"
echo "下一步：${target_env}/bin/python ${script_dir}/verify_runtime.py --qwen-config"
