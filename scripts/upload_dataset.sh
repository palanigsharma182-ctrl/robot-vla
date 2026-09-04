#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "用法: $0 <本地数据集目录> [Google Drive 子目录名]" >&2
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage
    exit 2
fi
if ! command -v rclone >/dev/null 2>&1; then
    echo "错误: WSL 中未找到 rclone" >&2
    exit 127
fi
rclone_remote=${RCLONE_REMOTE:-gdrive}
if [[ -z "$rclone_remote" || "$rclone_remote" == *:* ]]; then
    echo "错误: RCLONE_REMOTE 应为不带冒号的 remote 名称" >&2
    exit 2
fi

source_dir=${1%/}
if [[ ! -d "$source_dir" ]]; then
    echo "错误: 数据集目录不存在: $source_dir" >&2
    exit 2
fi
remote_name=${2:-$(basename "$source_dir")}
if [[ -z "$remote_name" || "$remote_name" == /* || "/$remote_name/" == *"/../"* ]]; then
    echo "错误: Google Drive 子目录必须是 VLA/datasets 下的相对路径" >&2
    exit 2
fi

destination="${rclone_remote}:VLA/datasets/$remote_name"
echo "上传数据集: $source_dir -> $destination"
exec rclone copy "$source_dir" "$destination" \
    --progress \
    --stats=10s \
    --stats-one-line \
    --retries=10 \
    --low-level-retries=20 \
    --retries-sleep=10s \
    --contimeout=30s \
    --timeout=10m \
    --transfers=4 \
    --checkers=8 \
    --drive-chunk-size=64M \
    --partial-suffix=.partial \
    --create-empty-src-dirs
