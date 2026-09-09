# 官方 ManiSkill demos → BC：导入前检查

本目录只准备路径二：检查官方 ManiSkill 3.0.1 HDF5 与同名 JSON 是否具备后续 BC 导入所需证据。
当前工具不会下载、重放、转换、训练或读取 HDF5 dataset payload，也不会把官方 demo 写入项目
`robot-vla-trajectory/v2`。

## 只读检查

```bash
python experiments/maniskill_data/inspect_demo.py /path/to/trajectory.h5
# metadata 不与 HDF5 同名时：
python experiments/maniskill_data/inspect_demo.py /path/to/trajectory.h5 \
  --metadata /path/to/trajectory.json
```

输出是 JSON preflight report。检查器惰性导入 `h5py`，递归列出每个 dataset 的 path、shape、dtype，
但不执行 `dataset[...]` 或 `read_direct`。报告包含：

- JSON 中 env、robot、episode control mode、步数/可能的频率字段；缺失项明确为 `unknown`；
- metadata 已知成功 episode 数、成功数、未知数及**已知分母**，不会从未读取的 HDF5 `success` 数组猜结果；
- 每条轨迹 actions、obs、env_states、terminated/truncated/success 的首维长度；
- metadata episode 与 `traj_<episode_id>`、`elapsed_steps` 及官方 T/T+1 约定的不一致；
- 适配 15D proprio 与 16×7 commanded-target TCP chunk 前仍缺哪些证据。

父 Agent 可在具有 `h5py` 的云端运行独立合成测试：

```bash
python -m pytest -q experiments/maniskill_data/test_inspect_demo.py
```

本机按任务边界不执行测试。合成测试不需要 ManiSkill、渲染、GPU、官方数据或网络。

## 为什么 7D action 仍不能直接训练当前 TCP BC

ManiSkill 3.0.1 官方格式规定 `actions` 是 `[T,A]`，但动作含义由 JSON 的 `control_mode` 决定。
官方压缩 demo 通常是 `obs_mode=none`、`pd_joint_pos`，只保留 env_states。项目 TCP 标签则是
`tcp-anchor-command-delta-rotvec-gripper/v1`：每个窗口为 16×7，前三维是固定 chunk anchor 轴向的
平移，后三维是同一轴向的 rotvec，最后一维是 gripper target；第一步为 actual TCP→commanded TCP，
其后为 commanded TCP→commanded TCP。两者恰好都是七维不构成语义等价。

当前 15D proprio 还要求按项目 `FrankaObservationAdapter` 形成 arm q[7] + arm dq[7] + calibrated
gripper opening[1]。仅看到 qpos/qvel 的字段名和 shape 仍不能证明 active joint 顺序、单位、finger 标定
或数值有效。若官方压缩数据没有 obs，必须按原 JSON 环境、reset kwargs 与 control mode 重放后采集，
不能从 env_states 的任意切片猜造 deployable proprio。

官方 `PickCube-v1` 与项目 `RobotVLAPickCubeToRegion-v1` 也不能当成同一任务。项目环境改变了 cube/goal
采样，使 goal 对相机可见，并要求释放、连续稳定放置和 robot static；因此官方 success 只能保留为来源任务
证据，不能直接作为项目任务成功标签或与项目成功率共用分母。

## 后续操作计划与缺口

1. 对实际文件先保存本工具报告，核对 env_id、env_kwargs、全部 episode/reset kwargs/control mode、动作维数、
   env_states/obs 是否存在及成功分母。报告只是结构证据，不表示已转换。
2. 在固定 `mani-skill==3.0.1` 环境按 metadata 重放选定的**成功训练 episode**。需要显式记录 robot active
   joint names、qpos/qvel、actual TCP、每步 controller commanded target、gripper target、frame 与控制频率。
3. 为官方 PickCube 单独定义来源 identity、train/development split 和到项目任务的映射策略。官方 success、
   项目稳定释放 success、失败 episode 与缺失 success 的 denominator 分开保存；不消费受保护 test。
4. 复用 `experiments/tcp_atomic_skills/data.py::command_chunk` 的 actual-first /
   commanded-successive 规则生成候选 `[N,16,7]` 标签，再验证 20 Hz 连续性、frame、FK round trip、范围、
   尾部 mask 与 command provenance。
5. 只有上述审计通过后，才另行实现 `robot-vla-trajectory/v2` 写入和 BC 训练。当前没有官方文件、实际报告、
   replay 结果、转换产物或训练结果；云端依赖/真实文件兼容性仍由父 Agent验证。

格式依据：ManiSkill v3.0.1 官方 `docs/source/user_guide/datasets/demos.md`。其中规定 trajectory group、
JSON metadata、`actions/terminated/truncated` 的 T 长度以及 `env_states/obs` 的 T+1 长度，并明确要求从
associated JSON 核对 controller。
