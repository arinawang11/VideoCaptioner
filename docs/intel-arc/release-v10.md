# VideoCaptioner Intel Arc / AI Boost NPU v10

v10 汇总了设备选择、NPU 稳定性、启动流程和可观测性方面的一批改动，并首次把
**NPU / iGPU / CPU 的实测性能与已知问题**写入发布说明。

- 版本：`v1.4.3-intel.10`（安装包 v10）
- 日期：2026-11-27
- 安装包：`VideoCaptioner-IntelArc-Setup-v10.exe`，259.9 MB
- SHA-256：`b7ec7613f26485ec2e625a71e3f9d0ea96ea5faad2587c960e370acff76060ad`

> **测试样品说明**：本页性能与问题记录来自一台 **Intel Core Ultra 7 155H QS
> （Quality Sample，质量验证样品）** 笔记本，并非零售正式版。QS 样品的 iGPU/NPU
> 硅片步进、固件和驱动兼容性可能与量产机不同。因此文中关于 NPU 长音频
> `DEVICE_LOST`、GPU int8 路径崩溃、回退 CPU 等结论 **仅代表该样品**，不应直接
> 推广为正式版硬件的普遍结论。欢迎使用正式版硬件的用户通过 Issue / Release
> 评论反馈实际表现。

## 本版变化

### 1. 设备选择

- 转录「运行设备」把设备类型与具体 Intel GPU **合并为一个下拉框**：
  `自动选择 / Intel GPU（自动）/ Intel GPU · <设备名> · 核显/独显 / Intel NPU / Intel CPU / CUDA / CPU`
- GPU 按**设备名称**保存与解析，不再依赖会变化的 D3D11 序号；
- `Intel(R) Arc(TM) Graphics` 正确识别为**核显**（此前误标为独显），选中核显会显示冻屏风险提示；
- 合成侧编码设备同样按名称选择。

### 2. 稳定性和回退策略

- **NPU 失败不再回退核显**：没有独立 Arc 时回退 CPU；有独立 Arc 才回退 GPU；
- `resolve_openvino_device("intel_npu")` 不再静默返回 GPU；NPU 被占用/不可用时明确报错；
- 只有核显 + NPU 的机器上，`自动` 优先选 **NPU**；有独立 Arc 时仍优先独显；
- NPU 使用 **60 秒 chunk**（240 秒会触发 `DEVICE_LOST`），并设置 `max_new_tokens=448`；
- GUI 启动不再用 ffmpeg 探测 QSV（该探测会在部分机器上初始化共享 iGPU 媒体驱动，导致启动冻屏）；
- 转录结果缓存已移除：同一文件每次都会重新转录。

### 3. 可观测性

- 每块记录并显示真实耗时与倍速（`x realtime`）；
- NPU 首次编译期间每 10 秒输出心跳，避免看起来像卡死；
- GUI 启动阶段写 `UI startup: ...` 日志，便于定位启动问题；
- 「关于 / 检查更新」标注 Intel 兼容分支，更新检查指向本分支 GitHub Releases。

## 实测性能（本机）

测试机：Core Ultra 7 155H ES / Arc iGPU 7D55 / AI Boost NPU 3720，
Windows `10.0.26300`，OpenVINO `2026.4.1`，模型 `large-v3-turbo-int8-ov`。

| 设备 | 配置 | 23:18 音频耗时 | 倍速 | 结果 |
| --- | --- | --- | --- | --- |
| Intel CPU | int8，240 s chunk | 199.2 s | **7.0x** | 稳定完成 |
| Intel Arc iGPU | int8 + **f32**，240 s chunk | 199.1 s | **7.0x** | 完成（手动选择） |
| Intel NPU | int8，60 s chunk | 前 14 块约 9.2–10.6 s/60 s | **约 6.0x** | 第 15/24 块 `DEVICE_LOST`，回退 CPU |

NPU 回退后 CPU 从零重跑完整音频：`210.4 s`（`6.6x`），本次**没有冻屏/死机**，但整体任务约 6 分钟。
详细过程见 [`test-log-2026-11-27.md`](test-log-2026-11-27.md)。

## 已知问题

> 以下结论基于 **QS 工程样品**（155H QS）；正式版硬件的 NPU/GPU 稳定性可能不同，
> 欢迎正式版用户反馈。若正式版上 NPU 长音频稳定，可优先使用 NPU。

### NPU（Meteor Lake / 3720）

- `large-v3-turbo` 长音频仍可能间歇 `DEVICE_LOST`（Level Zero 设备复位）；
  WER 中可见 `LKD_0x141_Tdr:6_IMAGE_npu_kmd.sys` 记录。
- 失败后应用会回退 **CPU**（不回退核显），并从 0 重新转录；安全但耗时约翻倍。
- 建议：
  - 长音频优先用 **Intel CPU**（实测 7.0x，稳定）；
  - 想用 NPU 时改用 `small` / `base`；
  - 出现 `DEVICE_LOST` 后**重启一次**再试，不要在同一会话反复重试。

### iGPU（Arc Graphics / 7D55）

- **GPU int8 / 无 f32 hint 路径不可用**：会出现 `CL_OUT_OF_HOST_MEMORY`，或推理时
  `python.exe` 以 `0xc0000409`（`ucrtbase.dll`）fail-fast；系统内存充足，属于
  OpenCL/IGC 编译器/驱动路径问题。
- 当前核显强制 `INFERENCE_PRECISION_HINT=f32`，可稳定运行（7.0x），但速度与 CPU 基本持平。
- 老版本（v5）能用 int8，是因为当时机器上已有编译好的 `.cl_cache` 内核；11-27 排查
  `CL_OUT_OF_HOST_MEMORY` 时清理了缓存，之后 int8 内核无法重新编译。
- 老 v7 安装包内的 int8 模型与现有模型 **SHA-256 完全一致**；安装包不含 `.cl_cache`，
  因此重装老版本也无法恢复 int8。
- **不要删除 `models\openvino\.cache`**（尤其 `.cl_cache`）。核显与显示共享硬件，
  手动选择核显跑长任务仍有冻屏风险，应用不会自动回退到核显。

### 其他

- NPU 首次编译（新缓存键）约 **3–4 分钟**，之后（含重启）缓存命中约 **3 秒**；
  缓存位于 `%LOCALAPPDATA%\VideoCaptioner\models\openvino\.cache`。
- AV1 QSV + medium 合成在 iGPU 上较慢；追求速度可改用 H.264/HEVC 或更快的 AV1 预设。
- 本机为 ES 工程样机：请勿更新 BIOS。

## 安装 / 升级

1. 运行 `VideoCaptioner-IntelArc-Setup-v10.exe`，覆盖安装即可；
2. 首次 NPU 使用会编译模型，耐心等待 3–4 分钟（界面有心跳提示）；
3. 保留 `%LOCALAPPDATA%\VideoCaptioner\models\openvino\.cache`，可避免重复编译；
4. 如需恢复默认配置，再删除该缓存即可（会重新下载 / 编译）。

## 验证

- 单元测试：61 passed（OpenVINO、设备选择、版本检查、无缓存转录等）；
- CI：Ubuntu + Windows 通过；
- 实机：CPU / GPU(f32) / NPU(部分) 转录、GUI 启动、QSV 合成均已跑通。
