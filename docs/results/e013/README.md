# E013 Precision v1 results

E013 的 1–8 步已经按冻结协议完成并通过；第 9 步 `20 Hz`、paired、no-actuation shadow 已完整执行，
但正式 gate **未通过**。因此当前可公开结论是“最小可部署状态、RGB-only Precision 训练、held-out
感知与四帧 Provider 工程链路成立”，不是“机器人已经达到毫米或厘米级放置精度”，也不允许把
Precision Observer 接入 actuator。

完整技术报告见 [report/report.html](report/report.html)，机器可读的脱敏汇总见
[summary.json](summary.json)，报告 packaging/QA 状态见
[report/delivery_receipt.json](report/delivery_receipt.json)。

## 1–9 步最终状态

| 步骤 | 结果 | 可支持的结论 |
|---|---|---|
| 1. 冻结 deployable/privileged schema | 通过 | GT label 与部署输入物理隔离 |
| 2. 采集 Observation V2 | 通过 | TCP、动态 wrist-camera pose、四时刻历史、F_L/F_R 完整 |
| 3. 生成独立 privileged labels | 通过 | 模型输入与 privileged array overlap 为空 |
| 4. 时间、坐标、投影和 split audit | 通过 | 7,987/7,987 Action semantic parity，split 无 scene 泄漏 |
| 5. 64 真实样本 overfit | 通过 | 训练/损失链可记忆真实样本，Motion Head 未改变 |
| 6. 正式训练 Precision U-Net | 通过 | 20 epochs、95,520 examples、3,000 optimizer steps 完整 |
| 7. 冻结 formal checkpoint | 通过 | 预注册规则选择 epoch 4，strict reload 通过 |
| 8. held-out、calibration、四帧 latency | 通过 | 条件 XY p90 1.52 mm；完整四帧 Provider p95 18.84 ms |
| 9. 100-seed paired no-actuation shadow | **失败** | 95 个干净 pair；5 个 Expert rejection；7 次 deadline miss |

第 9 步是预注册停止边界。没有更换失败 seed、删除失败 Episode、放宽 `50 ms` deadline、降低四帧输入，
也没有启动新的 actuation/闭环实验。

## 架构与数据流

```text
deployable root
  wrist RGB + TCP pose + dynamic wrist-camera pose + F_L/F_R
  + proprio/controller state + exact timestamps/validity
           |
           +-- current wrist frame --> single-frame Precision U-Net
           |                              object/goal heatmap + mask
           |                              visibility/projection/uncertainty
           |
           +-- t-3..t frames ----------> same frozen U-Net, oldest-to-newest
                                          + matching pose/time geometry
                                          + no-actuation observer ledger

privileged sibling root
  segmentation + GT base-frame object/goal + projected UV
           |
           +-- training labels / offline evaluation only
               never enters model input or shadow forward
```

第一版不是四帧堆成 12 通道，也不是双相机八图。U-Net 每次只处理一张 wrist RGB；Provider 对最近四个
时刻顺序调用同一个冻结模型，并把每一帧绑定到自己的 TCP、相机位姿和 timestamp。Episode 前三次调用
使用显式零 padding validity，绝不复制首帧。`F_L/F_R` 已进入最小状态和 identity，但本轮没有实现或启用
force-contact controller。

## 数据与训练证据

- Dataset：40 条完整成功 trajectory、7,987 ticks；train/val/test 为 `24/6/10` 条和
  `4,776/1,167/2,044` samples。
- Observation V2 六个模型模态 coverage 为 `1.0`，invalid count 为 `0`；Action 标签与执行语义
  `7,987/7,987` 对齐。
- RGB-only forward 只读取 `rgb_wrist`、`structured_state`、`geometric_motion`；privileged overlap 为 `[]`。
- Oracle projection/backprojection round-trip p90 为 `7.45e-9 m`，invalid 为 `0`。它只证明坐标公式自洽，
  不是视觉或机器人毫米精度。
- 64 真实样本 overfit 的 normalized-UV MAE 从 `0.2190` 降到 `0.000478`，mask IoU 为 `0.815`；
  Motion Head 逐参数保持冻结身份。
- 正式训练完成全部 20 epochs。按预注册的最小 validation normalized-UV MAE、同分取较早 epoch，
  选择 epoch 4：UV MAE `0.002387`、pixel p90 `0.706 px`、mask IoU `0.854`。

## Held-out 感知通过，但不能解释成 2 mm 放置

在 2,044 个 test samples、3,307 个有效 keypoints 上：

- world-XY p50/p90 为 `0.370/1.517 mm`；object p90 `0.997 mm`，goal p90 `2.312 mm`；
- pixel p90 为 `0.683 px`，mask IoU 为 `0.857`，invalid backprojection 为 `0`；
- confidence coverage/precision 为 `93.20%/92.64%`；接受样本 world-XY p90 为 `1.575 mm`；
- 最大 world-XY error 为 `208.5 mm`，说明仍存在很长的 outlier tail。

这些毫米数字是在模拟器 held-out RGB 上，使用离线 GT `z` 平面完成 backprojection 后的**条件 XY 感知误差**。
它不是 TCP 跟踪误差、接触误差、最终 placement error 或真实硬件精度。训练 split 只有 24 条 trajectory，
物体、纹理、光照和相机分布也较窄。

第一版 latency evaluation 曾把 trajectory prefix padding 混入 200 次测量，只产生 `794/800` 个预测帧；
该结果被保留但标记 superseded。修正后的 v3 只取完整历史，得到 `200/200` calls、`800/800` predictions、
Provider failure `0`，p95 `18.84 ms`，约 `53.08 Hz`。

## 第 9 步为什么失败

第二次 2-seed prewarmed smoke 通过：20 次预热后，396 calls 的 p95 为 `19.09 ms`，deadline miss 为 `0`，
Action/commanded-target/episode-length mismatch 都为 `0`。这证明 cold-start preflight 对小 smoke 有效，但
不足以代表 19,100 次正式调用的极端尾部。

正式 seeds `132000..132099` 的结果为：

- baseline 与 shadow 都接受 `95/100`，且失败 seed 集完全相同；
- 每臂有 4 个 MPlib trusted screw path 规划失败、1 个 controller correction safety-limit rejection；
- 95 个成功 pair 的 Action 与 commanded target 数组 bitwise 相等，episode length mismatch 为 `0`；
- Provider/observer failure 都为 `0`；19,100 calls 的 p50/p95 为 `18.34/20.77 ms`；
- 仍有 7 次单 call 超过 `50 ms`，deadline-miss rate 约 `0.0366%`；
- 75,830 个预测帧精确满足 `4*calls - 6*episodes`，即每个 Episode 只在前三个窗口使用规定 padding；
- `actuation_allowed=false`，所有 Action 与 outcome 都来自 Expert，不能称为 Precision treatment 效果。

正式 gate 同时要求 `100/100` 干净 pairs 和 deadline miss `0`，所以 p95 通过不能覆盖这两个失败条件。
当前 receipt 只保留逐 Episode miss count 与 Provider-record digest，没有保存逐 call latency，因此无法仅凭本轮
聚合证据把 7 次超时唯一归因于 CUDA、Python GC、CPU scheduling、I/O 或 simulator 抢占。

## 当前瓶颈与下一步

1. **实时尾部而非平均吞吐。** 新实验应先加入脱敏逐 call latency bucket/phase telemetry，再比较 frame cache、
   异步 GPU worker、固定内存、CUDA graph 或 CPU 调度隔离；不能只继续看 p95。
2. **Expert/collector 仍不是 100% 可用。** 应在新实验 ID 中修复 MPlib path robustness 与 controller
   correction safety rejection，并用全新预注册 seeds 重跑；不能事后筛掉当前 5 个失败 seed。
3. **感知 outlier 仍需 fail-closed。** 208.5 mm 最大误差说明 calibration confidence 不能单独作为执行授权；
   后续至少要加入几何创新、轨迹一致性、workspace 和 contact guard。
4. **尚缺完整控制误差预算。** 下一门槛应独立测量 camera/TCP calibration、Cartesian IK、tracking、接触与
   final placement，而不是把离线 GT-plane XY p90 直接当作系统精度。
5. **保持 Motion Head 冻结。** 在新的 no-actuation shadow 全部通过前，不启用 learned residual，不接入
   actuator，也不把 Expert success 当作 Precision 成功。

## 发布与可重复性边界

GitHub 只包含源码、测试、配置、脱敏聚合、SHA-256 和自包含报告；不包含 NPZ、RGB、视频、模型权重、
stdout/stderr、物理 GPU UUID、凭据、敏感绝对路径或 E012 原始中断证据。私有 Dataset、labels、checkpoint
和 receipts 已单独保存到本地归档，并通过逐文件 SHA 校验。

Portable report 的 canonical artifact validation、HTML packaging、payload equality 和 semantic fallback
结构检查已通过。当前本机没有兼容的 Chromium headless-shell，因此 enhanced reader 的 desktop/narrow
browser smoke 与 source-dialog interaction 未执行；这是报告交付 QA 限制，不影响实验 gate 或聚合数值。

E012 的既有限制继续保留：从不可变 epoch-24 full-state checkpoint 严格恢复后，trainer 结构、examples、
optimizer steps、source exposure、boundary offsets 与冻结 identity 一致，但 CUDA loss/gradient 数值轨迹
不再 bitwise reproducible。

公开 compact evidence 可独立检查：

```bash
python3 docs/results/e013/verify_summary.py
```
