#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "用法: $0 <本地 checkpoint 文件或目录> [VLA/checkpoints 下的远端子目录]" >&2
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

source_path=${1%/}
if [[ ! -e "$source_path" ]]; then
    echo "错误: checkpoint 文件或目录不存在: $source_path" >&2
    exit 2
fi
if [[ $# -eq 2 ]]; then
    remote_name=${2%/}
elif [[ -d "$source_path" ]]; then
    remote_name=$(basename "$source_path")
else
    remote_name=""
fi
if [[ "$remote_name" == /* || "/$remote_name/" == *"/../"* ]]; then
    echo "错误: 远端子目录必须是 VLA/checkpoints 下的相对路径" >&2
    exit 2
fi

destination="${rclone_remote}:VLA/checkpoints"
if [[ -n "$remote_name" ]]; then
    destination="$destination/$remote_name"
fi
echo "上传 checkpoint: $source_path -> $destination"
exec rclone copy "$source_path" "$destination" \
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
    --partial-suffix=.partial
