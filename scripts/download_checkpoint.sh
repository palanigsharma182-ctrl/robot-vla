#!/usr/bin/env bash
set -euo pipefail
umask 077

usage() {
    echo "用法: $0 <VLA/checkpoints 下的远端文件或目录> <本地目标目录>" >&2
}

if [[ $# -ne 2 ]]; then
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

remote_name=${1%/}
destination=${2%/}
if [[ -z "$remote_name" || "$remote_name" == /* || "/$remote_name/" == *"/../"* ]]; then
    echo "错误: 远端路径必须是 VLA/checkpoints 下的相对路径" >&2
    exit 2
fi
if [[ -z "$destination" ]]; then
    echo "错误: 本地目标目录不能为空" >&2
    exit 2
fi
install -d -m 700 "$destination"

source_path="${rclone_remote}:VLA/checkpoints/$remote_name"
echo "下载 checkpoint: $source_path -> $destination"
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
