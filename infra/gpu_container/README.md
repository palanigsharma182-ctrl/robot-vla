# 大陆 GPU 容器环境

本目录用于构建 `qwen-vla-v0.1` 的云端开发环境。目标基础实例是单张 RTX 4090 24GB，
Ubuntu 22.04，并由云厂商预先注入 NVIDIA 驱动、CUDA 用户态运行时和 PyTorch。

## 环境边界

镜像包含：

- Git、rsync、OpenGL/Vulkan Loader 等小型系统依赖；
- 独立的 `/opt/robot-vla/env` Python 环境；
- 固定版本的 Transformers、ManiSkill、SAPIEN 及开发工具；
- 项目代码或一个稳定 revision（可选）。

镜像不包含：

- NVIDIA 驱动或由 apt 安装的 CUDA；
- SSH 私钥、Hugging Face Token、代理凭据和 `.env`；
- Qwen 权重、数据集、Checkpoint、日志或 Hugging Face 下载缓存。

驱动由宿主机提供，因此每次换实例都必须重新执行运行环境检查。CUDA 可用不代表
Vulkan 可用；ManiSkill 双相机环境要求云厂商同时透传 NVIDIA graphics/Vulkan 能力。

## 构建

在项目根目录执行：

```bash
bash infra/gpu_container/bootstrap.sh
```

默认配置：

```text
源环境       /usr/local/miniconda3/envs/py312
项目环境     /opt/robot-vla/env
PyPI         https://pypi.tuna.tsinghua.edu.cn/simple
HF Endpoint  https://hf-mirror.com
```

这些路径可以通过 `ROBOT_VLA_SOURCE_ENV`、`ROBOT_VLA_ENV_DIR`、
`ROBOT_VLA_CONDA_BIN` 和 `ROBOT_VLA_PYPI_INDEX` 覆盖。

## 验证

只检查 Python、CUDA、BF16 和依赖版本：

```bash
/opt/robot-vla/env/bin/python \
  infra/gpu_container/verify_runtime.py
```

额外检查固定 Qwen revision 的配置，不下载权重：

```bash
/opt/robot-vla/env/bin/python \
  infra/gpu_container/verify_runtime.py \
  --qwen-config
```

创建 ManiSkill `PickCube-v1` RGB 环境并渲染一帧：

```bash
/opt/robot-vla/env/bin/python \
  infra/gpu_container/verify_runtime.py \
  --qwen-config \
  --maniskill
```

最后一条命令必须成功，才能把实例标记为“完整 qwen-vla-v0.1 开发环境”。如果出现
`ERROR_INCOMPATIBLE_DRIVER` 或 `vkCreateInstance`，应让云厂商为容器启用 Vulkan/graphics
透传，不能通过重装容器内 NVIDIA 驱动解决。

生成不包含环境变量、Token 或网络地址的镜像清单：

```bash
bash infra/gpu_container/capture_manifest.sh
```

## 大陆网络策略

- PyPI 默认使用清华镜像，阿里云镜像可作为备用；
- Qwen 固定为 `Qwen/Qwen3.5-2B` revision
  `15852e8c16360a2fea060d615a32b45270f8a8fc`；
- 权重优先从云厂商公共模型盘或 ModelScope 获取，HF Mirror 作为兼容端点；
- 只有正式包不包含必需功能时才使用 GitHub 源码依赖，并锁定 commit；
- 镜像地址属于部署配置，不进入模型与 Dataset 契约。

## 存储布局

云厂商未明确声明持久化路径前，不假设 `/workspace` 或 `/root` 会在容器释放后保留。
推荐将下列目录挂载到云端持久化卷：

```text
<persistent-root>/huggingface   模型缓存
<persistent-root>/data          轨迹数据
<persistent-root>/checkpoints   Checkpoint
<persistent-root>/artifacts     实验记录
```

本地 SSD 用作第二份备份，而不是通过 SSHFS 直接承担训练期间的随机读取。

## 制作云厂商镜像前

1. 基础检查和 Qwen 配置检查通过；
2. 完整环境还要求 ManiSkill/Vulkan 检查通过；
3. 停止训练、下载和写 Checkpoint 的进程；
4. 删除临时下载、pip/conda/Hugging Face 缓存；
5. 检查镜像中不存在 SSH 私钥、Token、`.env`、数据或 Checkpoint；
6. 记录基础镜像 ID、GPU 驱动、Python、PyTorch/CUDA 和顶层依赖版本；
7. 使用云厂商“保存容器镜像/制作自定义镜像”，不要在容器内启动 Docker。

镜像创建后，应启动一个新容器并重新运行全部检查；原容器通过不代表保存后的镜像可以
正确获得宿主机 GPU 和 Vulkan 设备。
