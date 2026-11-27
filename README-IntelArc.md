# VideoCaptioner · Intel Arc / AI Boost 本地转录

本仓库是 [WEIFENG2333/VideoCaptioner](https://github.com/WEIFENG2333/VideoCaptioner)
的支线项目，分支 `feat/intel-arc-openvino`，目标是在 Intel 平台上离线完成
Whisper 语音转写（不依赖 NVIDIA CUDA，也不上传音频）。

- 后端：Intel OpenVINO GenAI `WhisperPipeline`
- 设备：Intel GPU（Arc 独显 / Core Ultra 核显）、Intel AI Boost NPU、Intel CPU
- 模型：`large-v3-turbo`（按需自动拉取）及其它 Whisper 型号
- 架构：OpenVINO 跑在独立子进程，UI 进程不加载 OpenVINO

> 归档状态：GPU / NPU / CPU 三条本地路径均已在本机实测通过。
> 自 v8 起安装包不再内置模型（仅内置 OpenVINO 运行时，约 245 MB），首次使用时按需自动拉取。

---

## 1. 快速开始

### 使用安装包（推荐）

运行 `VideoCaptioner-IntelArc-Setup-v8.exe`（Windows 10/11 x64，无需管理员权限）。

安装后：设置 → 转录 → Faster Whisper：

| 选项 | 说明 |
| --- | --- |
| 运行设备 | `intel_npu` / `intel_gpu` / `intel_cpu` / `cpu` |
| 模型 | `large-v3-turbo`（首次转录自动拉取，支持断点续传与重试） |

模型与配置放在 `%LOCALAPPDATA%\VideoCaptioner`，程序放在
`%LOCALAPPDATA%\VideoCaptioner-IntelArc`。

### 从源码运行

```bash
pip install -e ".[intel]"
videocaptioner-gui
```

首次使用会自动下载对应精度的 OpenVINO 模型：

- `intel_gpu` → `whisper-*-fp16-ov`（iGPU 会自动改用 f32 精度提示）
- `intel_npu` / `intel_cpu` → `whisper-*-int8-ov`

---

## 2. 设备选择与实测性能

测试机：Lenovo ThinkBook 13x G4 IMH（21KR），Core Ultra 7 155H，
32 GB，Arc iGPU，Intel AI Boost NPU 3720，Windows 11 26300。

| 路径 | 测试音频 | 耗时 | 倍速 |
| --- | --- | --- | --- |
| **NPU 3720 + int8** | 整集 1398.9 s | **189.1 s** | **7.4x** |
| NPU 3720 + int8 | 240 s | 35.0 s | 6.9x |
| NPU 3720 + int8 | 60 s | 10.5 s | 5.7x |
| GPU iGPU + int8 + f32 | 60 s | 12.8 s | 4.7x |
| CPU + int8 | 60 s | 14.2 s | 4.2x |

选择建议：

- **追求速度**：`intel_npu`（本机最快）
- **追求稳定/兼容**：`intel_gpu`；驱动异常时自动切 `intel_cpu`
- **低功耗/后台**：`intel_npu`
- NPU 首次使用（缓存为空）会有一次 **约 10 分钟**的模型编译，界面停在
  “加载模型”，属正常现象；之后命中缓存约 3 秒

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
- 可用 `VIDEOCAPTIONER_OPENVINO_GPU_PRECISION=f32|fp16` 覆盖
- 失败重试链：`GPU` → `GPU+f32` → `CPU int8`；`NPU` 失败 → `GPU` → `CPU int8`

### 3.2 NPU 改用原生时间戳 + 限制解码长度

旧实现为规避 `openvino.genai#4806`，在 NPU 上做“停顿分句 + 逐句无时间戳”。
OpenVINO 2026.4.1 中该问题已修复，原生时间戳可用。分句会让每句重复跑一次
固定 30 秒 encoder，是最慢的根源。

现在 NPU 与 GPU 走同一条 240 秒分块 + 原生时间戳路径，并额外传
`max_new_tokens=448`：不限制 token 数时解码会越过窗口，触发 NPU 驱动看门狗
（`ZE_RESULT_ERROR_DEVICE_LOST`）。分句方案保留为“时间戳返回空”时的兜底。

### 3.3 NPU 设备丢失不能自愈

`DEVICE_LOST` 之后在进程内重建 NPU pipeline 会挂住，必须复位设备或重启：

```powershell
Disable-PnpDevice -InstanceId 'PCI\VEN_8086&DEV_7D1D&SUBSYS_380C17AA&REV_04\3&11583659&0&58' -Confirm:$false
Enable-PnpDevice  -InstanceId 'PCI\VEN_8086&DEV_7D1D&SUBSYS_380C17AA&REV_04\3&11583659&0&58' -Confirm:$false
```

因此应用层遇到 NPU 失败会整体切到 GPU（f32）继续，不会中断任务。

---

## 4. 测试证据

完整记录见 [`docs/intel-arc/test-report.md`](docs/intel-arc/test-report.md)。

摘要：

- 整集 NPU：1398.9 s 音频 → 189.1 s，6 块，542 段，覆盖到 `00:23:14`
- 300 次连续 NPU 生成：全部成功，内存稳定在 ~3.56 GB
- 早期分句路径累计 ~460 次调用：1 次 `DEVICE_LOST`（约 0.2%）
- 60 秒音频：NPU 10.5 s / GPU 12.8 s / CPU 14.2 s
- 单元测试：`pytest tests/test_asr/test_openvino_backend.py` → 13 passed

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
5. **工程样品硬件**：本机 iGPU 为 `DEV_7D55`、NPU 为 `DEV_7D1D`，零售机行为可能不同；
   不得更新 BIOS。
6. **词级时间戳**：Intel GPU 不支持 encoder attention 分解，会自动降级为分段
   时间戳。

---

## 6. 故障排查

| 现象 | 原因 / 处理 |
| --- | --- |
| 转写开始后立刻报错，日志无 ERROR | GPU/IGC 原生崩溃；确认是否 iGPU 且已启用 f32 |
| 停在“加载模型”很久 | NPU/GPU 冷编译；等待，或保留 `models/openvino/.cache` |
| `NPU inference failed ... falling back to Intel CPU` | NPU 看门狗复位；任务会由 CPU 继续，建议重启后再试 |
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
dist/VideoCaptioner-IntelArc-Setup-v8.exe
```

v8 安装包：245.1 MB，SHA-256
`a86db8c5e0b3eb45aae8aa8646f42fe3485e199d0044fa5b6780ef14ab3716d6`。

---

## 9. 相关文档

- [`docs/config/intel-arc-gpu.md`](docs/config/intel-arc-gpu.md)：架构、模型、排障
- [`docs/intel-arc/test-report.md`](docs/intel-arc/test-report.md)：崩溃证据与实测数据
- [`docs/intel-arc/test-log-2026-11-27.md`](docs/intel-arc/test-log-2026-11-27.md)：v10 实测记录（CPU/GPU/NPU、int8 bug）
- [`docs/intel-arc/release-v10.md`](docs/intel-arc/release-v10.md)：v10 发布说明与已知问题
- [`BRANCH.md`](BRANCH.md)：分支变更摘要
