# VideoCaptioner · Intel Arc / AI Boost 本地转录

本仓库是 [WEIFENG2333/VideoCaptioner](https://github.com/WEIFENG2333/VideoCaptioner)
的支线项目，分支 `feat/intel-arc-openvino`，目标是在 Intel 平台上离线完成
Whisper 语音转写（不依赖 NVIDIA CUDA，也不上传音频）。

- 后端：Intel OpenVINO GenAI `WhisperPipeline`
- 设备：Intel GPU（Arc 独显 / Core Ultra 核显）、Intel AI Boost NPU、Intel CPU
- 模型：`large-v3-turbo`（按需自动拉取）及其它 Whisper 型号
- 架构：OpenVINO 跑在独立子进程，UI 进程不加载 OpenVINO

> 归档状态：CPU 与 GPU(f32) 完整通过；NPU 长音频在 QS 样品上会 `DEVICE_LOST`
> 并回退 CPU（无冻屏）。详见 2.2 版本对照与 v10 说明。
> 自 v8 起安装包不再内置模型；v10 安装包约 259.9 MB，首次使用时按需自动拉取。

---

## 1. 快速开始

### 使用安装包（推荐）

运行 `VideoCaptioner-IntelArc-Setup-v10.exe`（Windows 10/11 x64，无需管理员权限）。

安装后：设置 → 转录 → Faster Whisper：

| 选项 | 说明 |
| --- | --- |
| 运行设备 | 合并下拉：自动 / Intel GPU（按名称）/ Intel NPU / Intel CPU / CUDA / CPU |
| 模型 / 精度 | `large-v3-turbo` 等；INT8 / FP16 可选；首次转录自动拉取 |

模型与配置放在 `%LOCALAPPDATA%\VideoCaptioner`，程序放在
`%LOCALAPPDATA%\VideoCaptioner-IntelArc`。

### 从源码运行

```bash
pip install -e ".[intel]"
videocaptioner-gui
```

首次使用会自动下载对应精度的 OpenVINO 模型：

- 选择 INT8 → `whisper-*-int8-ov`；选择 FP16 → `whisper-*-fp16-ov`
- iGPU 会自动加 `INFERENCE_PRECISION_HINT=f32`（仅 f32 稳定；int8/无 hint 路径会崩）
- NPU 仅支持 int8

---

## 2. 设备选择与实测性能

> 测试机为 **Core Ultra 7 155H QS（Quality Sample，质量验证样品）**，
> 32 GB，Arc iGPU 7D55，Intel AI Boost NPU 3720，Windows 11 26300。
> QS 样品的硅片/固件可能与零售正式版不同，下文的 NPU/GPU 结论仅代表该样品；
> 欢迎正式版硬件用户反馈实际表现。

### 2.1 v10 实测（2026-11-27）

| 路径 | 测试音频 | 耗时 | 倍速 | 结果 |
| --- | --- | --- | --- | --- |
| **CPU int8** | 整集 1398.9 s | **199.2 s** | **7.0x** | 稳定完成 |
| **GPU iGPU + int8 + f32** | 整集 1398.9 s | **199.1 s** | **7.0x** | 完成（手动选择） |
| **NPU 3720 + int8** | 60 s × 14 块 | 约 9.2–10.6 s/块 | **约 6.0x** | 第 15/24 块 `DEVICE_LOST` |
| NPU 失败后 CPU 重跑 | 整集 | 210.4 s | 6.6x | 安全完成，未冻屏 |

选择建议（v10）：

- **稳定优先**：`intel_cpu`，实测 7.0x，不占用核显/NPU，无崩溃风险；
- **想用 NPU**：优先 `small` / `base`；`large-v3-turbo` 长音频仍可能
  `DEVICE_LOST`，会回退 CPU 并从 0 重跑；
- **核显**：只能用 f32（应用默认），速度约 7.0x，与 CPU 持平；核显与显示共用硬件，
  手动跑长任务仍有冻屏风险；
- NPU 首次编译（新缓存键）约 **3–4 分钟**，界面每 10 秒有心跳；之后（含重启）
  缓存命中约 **3 秒**。

> 历史版本（v6–v9）的 NPU 使用 240 s chunk，整集约 189 s / 7.4x，但
> `DEVICE_LOST` 风险更高；不同版本的 GPU/CPU/NPU 行为差异见下表。

### 2.2 版本兼容性对照（v6–v10）

| 版本 | 安装包 / 模型 | 核显（iGPU） | NPU | 失败回退 | 设备选择 | 其他 |
| --- | --- | --- | --- | --- | --- | --- |
| **v10（当前）** | 259.9 MB，模型按需下载 | **仅 f32**；QS 上 int8/无 hint 会 `CL_OUT_OF_HOST_MEMORY` / `0xc0000409`；约 7.0x | 60 s chunk；约 6.0x；large 长音频可能 `DEVICE_LOST` | **NPU → CPU** | 设备类型与具体 GPU **合并按名称选择**，`自动` 在纯核显机器优先 NPU | 无 ASR 结果缓存；QSV 启动探测已移除；编译心跳 |
| v9（本地构建，未发布） | 245 MB，模型按需下载 | f32；约 4–7x | 240 s chunk；整集约 7.4x | NPU → GPU → CPU | 枚举设备 + 单独 GPU 名称卡 | 活跃设备标签；QSV 启动探测仍在，部分机器启动可能冻屏 |
| v8（v10 前 Latest） | 245 MB，模型按需下载 | f32；约 4–7x | 240 s chunk | NPU → GPU → CPU | 枚举设备 | 无活跃设备标签；QSV 启动探测仍在 |
| v7 | 902.5 MB，**内置模型** | f32（v6 起） | 240 s chunk | NPU → GPU → CPU | 枚举设备 | 安装即带模型，无需首次下载；设备选择较旧 |
| v6 | 902.5 MB，**内置模型** | **首次强制 f32** | 240 s chunk | NPU → GPU → CPU | 枚举设备 | 用于规避 iGPU fp16 编译器 abort |
| v5 及更早 | 902.5 MB，**内置模型** | **不强制 f32**；默认/int8 路径依赖已编译的 `.cl_cache` | 240 s chunk | NPU → GPU → CPU | 枚举设备 | 无按名称选择；有 ASR 结果缓存 |

版本差异要点：

- **模型分发**：v7 及更早安装包内置模型（902.5 MB）；v8+ 为瘦安装包，首次使用按需下载。
- **核显精度**：v6 起对 iGPU 强制 `f32`；v5 及更早走默认/int8 路径。老版本能跑 int8
  依赖当时机器上已存在的 OpenCL 内核缓存，**不代表在全新机器上也能复现**。
- **NPU 分块**：v10 为 60 s；v6–v9 为 240 s。240 s 在 QS 样品上更容易触发
  `DEVICE_LOST`。
- **失败回退**：v10 起 NPU 失败只回退 **CPU**（不回退核显）；v6–v9 会回退 GPU。
- **设备选择**：v10 才把设备类型与具体 GPU 名称合并；旧版需分别选择。
- **结果缓存**：v10 起转录结果不缓存，每次重新识别；旧版会命中 ASR 缓存。
- **QS 说明**：以上 v10 的 NPU/GPU 结论基于 QS 样品，正式版硬件可能不同。

### 2.3 台式机参考性能（Arc Pro B50 / UHD 770）

本机（台式机）的 **B50 / UHD 770 + int8 / fp16 人工测试均通过**；以下为 agent 实测。
台式机与 QS 笔记本的 CPU/GPU/驱动不同，**不能与 2.1 的数字直接对比**，仅作参考。

测试条件：OpenVINO `2026.4.1`，模型 `large-v3-turbo`，chunk 240 s；
B50 驱动 `32.0.101.8976`，UHD 770 驱动 `32.0.101.7092`。

| 设备 | 模型 / 计算精度 | 音频 | 耗时 | 倍速 | 加载 |
| --- | --- | --- | --- | --- | --- |
| **Arc Pro B50（dGPU）** | int8 模型，默认 fp16 计算 | 整集 1340.1 s | **19.74 s** | **67.9x** | 1.16 s |
| **Arc Pro B50（dGPU）** | fp16 模型，fp16 计算 | 整集 1340.1 s | **26.16 s** | **51.2x** | 2.14 s |
| UHD 770（iGPU） | int8 模型，**f32** 计算 | 90 s 短片 | 42.44 s | **2.12x** | 1.27 s |
| UHD 770（iGPU） | fp16 模型，**f32** 计算 | 90 s 短片 | 46.65 s | **1.93x** | 2.23 s |

结论：

- **dGPU（B50）非常快**：int8 模型 + 默认 fp16 计算约 **68x**，fp16 模型约 **51x**；
  在本机 1340 s 音频上一次跑完，均无崩溃。
- **iGPU（UHD 770）明显更慢**：f32 计算约 **2x**；全段 1340 s 实测约 533 s（2.5x），
  因此本次 UHD 只做 90 s 短片测试，不再跑全段。
- 同样使用 f32 时，int8 模型仍略快于 fp16 模型；dGPU 上优先 int8 模型。
- B50 的 dGPU 路径使用默认 fp16 计算；UHD 770（iGPU）按应用策略强制 f32。

---

## 3. 关键修复

### 3.1 核显 fp16 编译器崩溃 → 自动改用 f32

工程样品 Meteor Lake 的 IGC 编译器在为 Whisper encoder 生成 fp16 kernel 时
会直接 `abort()`，表现为 worker 在 `Transcribing chunk 1/N` 后静默消失：

```
WER: pythonw.exe  BEX64  0xc0000409  (fast-fail 7 = abort)
     ucrtbase.dll + openvino_intel_gpu_plugin.dll + igdrcl64.dll + igc-default64.dll
```

最小复现：encoder 单次 `[1,128,3000]` 推理 1 秒内 `Fatal Python error: Aborted`。
加上 `INFERENCE_PRECISION_HINT=f32` 后同一推理 3.4 秒成功。

实现（`videocaptioner/core/asr/openvino_asr.py`）：

- OpenVINO 设备名含 `(iGPU)` → 自动 `INFERENCE_PRECISION_HINT=f32`
- 独显 `(dGPU)` → 保持 fp16
- 可用 `VIDEOCAPTIONER_OPENVINO_GPU_PRECISION=f32|fp16` 覆盖；但 v10 的 QS 核显上
  fp16/int8/无 hint 路径会崩溃，**只建议保持 f32**
- 失败重试链：v10 `GPU` → `GPU+f32` → `CPU int8`；`NPU` 失败 → `CPU`
- v6–v9：`NPU` 失败 → `GPU` → `CPU int8`；v5 及更早没有 f32 强制

### 3.2 NPU 原生时间戳 + 分块/解码限制

旧实现为规避 `openvino.genai#4806`，在 NPU 上做“停顿分句 + 逐句无时间戳”。
OpenVINO 2026.4.1 中该问题已修复，原生时间戳可用。分句会让每句重复跑一次
固定 30 秒 encoder，是最慢的根源。

v10 起 NPU 使用 **60 秒分块**（v6–v9 为 240 秒；240 秒在 QS 样品上更容易触发
`DEVICE_LOST`），并额外传 `max_new_tokens=448`：不限制 token 数时解码会越过窗口，
触发 NPU 驱动看门狗（`ZE_RESULT_ERROR_DEVICE_LOST`）。分句方案保留为
“时间戳返回空”时的兜底。

### 3.3 NPU 设备丢失不能自愈

`DEVICE_LOST` 之后在进程内重建 NPU pipeline 会挂住，必须复位设备或重启：

```powershell
Disable-PnpDevice -InstanceId 'PCI\VEN_8086&DEV_7D1D&SUBSYS_380C17AA&REV_04\3&11583659&0&58' -Confirm:$false
Enable-PnpDevice  -InstanceId 'PCI\VEN_8086&DEV_7D1D&SUBSYS_380C17AA&REV_04\3&11583659&0&58' -Confirm:$false
```

因此应用层遇到 NPU 失败时：

- **v10 起**：回退 **CPU**（不回退核显），从 0 重新转录；安全但耗时约翻倍；
- **v6–v9**：回退 GPU（f32）→ CPU；
- 持续 `DEVICE_LOST` 仍需 PnP 复位或重启；不要在同一会话反复重建 NPU。

---

## 4. 测试证据

v10 实测见 [`docs/intel-arc/test-log-2026-11-27.md`](docs/intel-arc/test-log-2026-11-27.md)。
以下早期数据来自 v6/v7（NPU 240 s chunk），仅作历史参考，不能直接与 v10 对比：

- 整集 NPU（v6/v7）：1398.9 s 音频 → 189.1 s，6 块，542 段，覆盖到 `00:23:14`
- 300 次连续 NPU 生成：全部成功，内存稳定在 ~3.56 GB
- 早期分句路径累计 ~460 次调用：1 次 `DEVICE_LOST`（约 0.2%）
- 60 秒音频（旧版）：NPU 10.5 s / GPU 12.8 s / CPU 14.2 s
- 旧版单元测试：13 passed

v10 实测：

- CPU int8：199.2 s / 1398.9 s → **7.0x**；
- GPU iGPU + int8 + f32：199.1 s → **7.0x**；
- NPU 60 s × 14 块约 **6.0x**，第 15/24 块 `DEVICE_LOST` → CPU 全量重跑
  210.4 s（6.6x），**未冻屏**；
- 单元测试：**61 passed**（OpenVINO、设备选择、版本检查、无缓存转录等）。

---

## 5. 已知限制

1. **NPU 首次编译慢**：缓存键首次出现时约 3–4 分钟（主机侧解释模式，单/少核），
   界面会每 10 秒输出心跳；之后（含重启）通常 3 秒左右命中缓存。
2. **不要清理 `models\openvino\.cache`（尤其 `.cl_cache`）**：2026-11-27 排查 GPU 编译
   OOM 时清理过一次，导致旧版可用的 int8 OpenCL 内核缓存丢失；此后 iGPU int8/无 hint
   路径会以 `CL_OUT_OF_HOST_MEMORY` 或 `0xc0000409` 崩溃，只能使用 f32。
3. **NPU 长音频仍可能 `DEVICE_LOST`**：`large-v3-turbo` 实测在第 15/24 块触发；
   应用会回退 **CPU**（不再回退核显），从 0 重新转录，安全但耗时约翻倍。
   出现后建议重启再试，长音频优先用 CPU 或 `small` 模型。
4. **核显与显示共享硬件**：手动让 iGPU 跑 OpenVINO 长任务仍有冻屏风险，应用不会
   自动回退到核显；默认 f32 也只能保证稳定，不能保证比 CPU 快。
5. **QS 质量验证样品**：本机为 Core Ultra 7 155H **QS（Quality Sample）**，iGPU 为
   `DEV_7D55`、NPU 为 `DEV_7D1D`；硅片/固件可能与零售正式版不同。本文中 NPU
   `DEVICE_LOST`、GPU int8 崩溃、CPU 回退等结论 **仅代表该样品**，不能直接推广到
   正式版硬件；欢迎正式版用户反馈实际表现。不得更新 BIOS。
6. **词级时间戳**：Intel GPU 不支持 encoder attention 分解，会自动降级为分段
   时间戳。

---

## 6. 故障排查

| 现象 | 原因 / 处理 |
| --- | --- |
| 转写开始后立刻报错，日志无 ERROR | GPU/IGC 原生崩溃；确认是否 iGPU 且已启用 f32 |
| 停在“加载模型”很久 | NPU 首次编译 3–4 分钟（v10；旧版曾约 10 分钟），之后约 3 秒；保留 `models/openvino/.cache` |
| `NPU inference failed ... falling back to Intel CPU` | v10 起 NPU 看门狗复位后回退 CPU（旧版回退 GPU）；任务会继续，建议重启后再试 |
| NPU 一直 `DEVICE_LOST` | PnP 复位 NPU 或重启；不要在同一进程内重建 |
| iGPU 转录报 `CL_OUT_OF_HOST_MEMORY` / `0xc0000409` | 核显 int8/无 f32 路径不可用；改 CPU 或 f32（应用默认） |
| `No Intel NPU detected` | 安装 Intel NPU 驱动（≥ 32.0.100.3104） |
| `No Intel GPU detected` | 安装 Intel 显卡驱动；或改用 `intel_cpu` |
| 数据/模型路径不对 | 统一使用 `%LOCALAPPDATA%\VideoCaptioner` |

---

## 7. 代码结构（相关部分）

```
videocaptioner/
  config.py                              # 数据路径、dev 模式判定
  core/asr/openvino_asr.py               # OpenVINO 实现、f32 策略、NPU 路径与回退
  core/asr/openvino_process.py           # 子进程调度、重试链
  core/asr/openvino_worker.py            # worker 入口（--device/--precision/...）
  core/utils/hardware.py                 # GPU/NPU 探测、设备解析
  ui/components/FasterWhisperSettingWidget.py
docs/
  config/intel-arc-gpu.md                # 使用与原理说明
  intel-arc/test-report.md               # 测试与诊断归档
  intel-arc/test-log-2026-11-27.md       # v10 实测记录
  intel-arc/release-v10.md               # v10 发布说明与版本兼容性
build/
  installer.iss                          # Inno Setup 打包定义
  portable/                              # 便携 Python 运行时 + 模型载荷
tests/test_asr/test_openvino_backend.py  # 后端单元测试
```

---

## 8. 构建与打包

```bash
# 1) 构建 wheel 并装入便携运行时
python -m pip wheel . --no-deps -w dist
build/portable/runtime/python.exe -m pip install --no-deps --force-reinstall dist/videocaptioner-*.whl

# 2) 编译安装包（需 Inno Setup 6）
ISCC.exe build/installer.iss

# 3) 产物（不再封装模型，首次运行按需自动拉取）
dist/VideoCaptioner-IntelArc-Setup-v10.exe
```

v10 安装包：272,553,687 字节（259.9 MB），SHA-256
`b7ec7613f26485ec2e625a71e3f9d0ea96ea5faad2587c960e370acff76060ad`。

---

## 9. 相关文档

- [`docs/config/intel-arc-gpu.md`](docs/config/intel-arc-gpu.md)：架构、模型、排障
- [`docs/intel-arc/test-report.md`](docs/intel-arc/test-report.md)：崩溃证据与实测数据
- [`docs/intel-arc/test-log-2026-11-27.md`](docs/intel-arc/test-log-2026-11-27.md)：v10 实测记录（CPU/GPU/NPU、int8 bug）
- [`docs/intel-arc/release-v10.md`](docs/intel-arc/release-v10.md)：v10 发布说明与已知问题
- [`BRANCH.md`](BRANCH.md)：分支变更摘要
