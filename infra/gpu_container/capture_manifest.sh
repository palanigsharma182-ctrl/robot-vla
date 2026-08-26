#!/usr/bin/env bash
set -euo pipefail

# 生成不包含环境变量、Token 或主机网络信息的镜像清单。

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/../.." && pwd)"
vm_script_dir="${project_root}/infra/gpu_vm"
target_env="${ROBOT_VLA_ENV_DIR:-/opt/robot-vla/env}"
conda_bin="${ROBOT_VLA_CONDA_BIN:-/usr/local/miniconda3/bin/conda}"
output_dir="${1:-/opt/robot-vla/image-manifest}"

if [[ ! -x "${target_env}/bin/python" ]]; then
  echo "错误：项目环境不存在：${target_env}" >&2
  exit 1
fi

install -d -m 755 "${output_dir}"

{
  printf 'captured_at_utc='
  date -u +'%Y-%m-%dT%H:%M:%SZ'
  printf 'os='
  . /etc/os-release
  printf '%s\n' "${PRETTY_NAME}"
  printf 'kernel='
  uname -r
  printf 'python='
  "${target_env}/bin/python" --version 2>&1
  printf 'gpu='
  nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap \
    --format=csv,noheader
  printf 'root_filesystem='
  df -hT / | tail -n 1
} > "${output_dir}/runtime.txt"

"${target_env}/bin/python" -m pip list --format=freeze \
  > "${output_dir}/pip-freeze.txt"

if [[ -x "${conda_bin}" ]]; then
  "${conda_bin}" list --prefix "${target_env}" --explicit \
    > "${output_dir}/conda-explicit.txt"
fi

dpkg-query -W -f='${binary:Package}=${Version}\n' \
  git libgl1 libglib2.0-0 libvulkan1 rsync vulkan-tools \
  > "${output_dir}/system-packages.txt"

sha256sum \
  "${project_root}/pyproject.toml" \
  "${script_dir}/bootstrap.sh" \
  "${script_dir}/requirements.direct.txt" \
  "${script_dir}/verify_runtime.py" \
  "${vm_script_dir}/bootstrap.sh" \
  "${vm_script_dir}/requirements.direct.txt" \
  > "${output_dir}/input-sha256.txt"

vulkaninfo --summary > "${output_dir}/vulkan-summary.txt" 2>&1 || true

echo "镜像清单已写入：${output_dir}"
