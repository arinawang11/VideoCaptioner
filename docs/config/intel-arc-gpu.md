# Intel Arc GPU / AI Boost NPU 本地转录（OpenVINO）

本分支（`feat/intel-arc-openvino`）为 Intel 硬件增加本地 Whisper 语音转录能力：

- **首要目标**：Arc A 系列（Alchemist：A310 / A380 / A580 / A750 / A770）
- **同时支持**：Arc B 系列（Battlemage）、Core Ultra 核显（如酷睿 Ultra 7 155H 的 Arc iGPU）
- **NPU 支持**：Intel AI Boost NPU（酷睿 Ultra 7 155H 的 Meteor Lake NPU 3720、
  Lunar Lake NPU 4000、Arrow Lake / Panther Lake NPU）

## 为什么需要单独适配

Faster-Whisper 使用的 CTranslate2 只支持 NVIDIA CUDA，
在 Intel 显卡 / NPU 上选择 `cuda` 会直接失败（或静默回退 CPU）。

本分支改用 Intel 官方的 **OpenVINO GenAI WhisperPipeline**：

- GPU：fp16 预转换模型，原生支持时间戳
- NPU：int8 预转换模型，配合分句策略与三项官方 bug 规避（见下文）
- 长音频自动分片，时间戳按分片偏移拼接

> 重要：OpenVINO 2026 需要较新的 MSVC 运行库（MSVCP140 ≥ 14.4），
> 而 PyQt5 自带的是 14.26。二者在同一进程加载会导致 `0xc0000005` 闪退。
> 因此 OpenVINO 始终运行在独立子进程中（`openvino_worker.py`），
> 主界面进程不加载任何 OpenVINO DLL。

## 安装

```bash
# 源码方式（推荐开发）
pip install -e ".[intel]"

# 或已发布包
pip install "videocaptioner[intel]"
```

Windows 安装包（`VideoCaptioner-IntelArc-Setup-v3`）额外内置：

- OpenVINO 2026.4.1 运行时：Intel GPU 插件、Intel NPU 插件 + NPU 编译器、CPU 插件
- `whisper-large-v3-turbo-int8-ov` 模型（约 790MB）

安装后 GPU/CPU 在没有 fp16 模型时会直接复用该 int8 模型，首次转录不需要联网下载。
NPU 驱动属于系统驱动（Intel NPU Driver ≥ 32.0.100.3104），仍需单独安装。

`intel` 可选依赖固定为相互兼容的版本：

```
openvino==2026.4.1
openvino-tokenizers==2026.4.1.0
openvino-genai==2026.4.1.0
```

驱动要求：

- GPU：最新 Intel 显卡驱动（Arc A 系列建议 31.0.101 及以上）
- NPU：Intel NPU 驱动 **32.0.100.3104 或更新**

确认 OpenVINO 能看到设备：

```bash
python -c "import openvino as ov; c=ov.Core(); print(c.available_devices); [print(d, c.get_property(d,'FULL_DEVICE_NAME')) for d in c.available_devices]"
```

Core Ultra 7 155H 正常应看到：

```
CPU
GPU.0  Intel(R) Arc(TM) Graphics (iGPU)
NPU    Intel(R) AI Boost
```

## 使用

1. 打开软件 → 设置 → 转录 → Faster Whisper
2. **运行设备**：
   - 独显 / 核显 → `intel_gpu`
   - Intel AI Boost NPU → `intel_npu`
3. **模型**选择 `large-v3-turbo`（默认推荐），首次转录会自动下载对应精度的模型
4. 直接开始转录

模型下载到 `AppData/models/openvino/`（源码运行时为仓库下的 `AppData/models/openvino/`），
编译缓存位于 `AppData/models/openvino/.cache`，第二次启动不再重新编译。

- `intel_gpu` → 自动使用 `OpenVINO/whisper-*-fp16-ov`
- `intel_npu` → 自动使用 `OpenVINO/whisper-*-int8-ov`（体积约减半，匹配 NPU INT8 算力）

### 模型与硬件建议

| 模型 | fp16（GPU） | int8（NPU） | 说明 |
| --- | --- | --- | --- |
| tiny | 46MB | 46MB | 速度最快，准确率低，仅测试用 |
| base | 81MB | 81MB | NPU 上性价比高 |
| small | 245MB | 245MB | **NPU 推荐**，准确率/速度平衡 |
| medium | 748MB | 748MB | NPU 上偏重 |
| large-v3-turbo | 1.6GB | 790MB | GPU 推荐；NPU 也可运行 |
| large-v3 | 3.0GB | 1.5GB | 准确率最高，速度较慢 |

- **Arc A 系列**：A770 16GB 建议 `large-v3-turbo`；A380 6GB 建议 `small` / `medium`
- **Core Ultra 7 155H**：
  - 优先 `intel_gpu`（Arc iGPU，8 Xe 核心）：`large-v3-turbo` fp16，速度快且原生时间戳
  - 低功耗场景可切到 `intel_npu`：建议 `small` / `base` INT8；
    NPU 采用分句转录，时间戳由语音停顿切分得到

## NPU 适配细节（酷睿 Ultra 7 155H）

Intel NPU（AI Boost，11 TOPS INT8，Meteor Lake NPU 3720）在 OpenVINO 2026.x 上
跑 Whisper 有三个已知问题，本分支已全部处理：

1. **时间戳导致输出损坏**（[openvino.genai#4806](https://github.com/openvinotoolkit/openvino.genai/issues/4806)）
   `return_timestamps=True` 在 NPU 上会返回空文本 / 重复碎片 / KV-Cache 满。
   → NPU 路径改为：先用 20ms RMS 能量做语音活动检测，按自然停顿切分
   （每段 < 14 秒，远小于 NPU 30 秒静态编码窗口），每段用
   `return_timestamps=False` 识别，字幕时间戳由切分区间精确得到。

2. **第二次调用复用上一段音频**（[openvino#37937](https://github.com/openvinotoolkit/openvino/issues/37937)）
   NPU 在 1-token 语言检测后 `num_stored_tokens` 未同步，后续调用可能沿用旧
   编码器状态（OpenVINO ≤ 2026.4）。
   → 自动识别语言时：第 1 段执行语言检测（此时 `num_stored_tokens == 0` 一定安全），
   之后把检测到的语言 token 固定下来，保证每段都是多 token prefill。

3. **Level0 内存不足 / 退出清理**（[openvino.genai#1460](https://github.com/openvinotoolkit/openvino.genai/issues/1460)）
   → NPU 初始化失败时自动设置 `DISABLE_OPENVINO_GENAI_NPU_L0=1` 重试；
   仍失败则回退到 Intel GPU；worker 退出前显式销毁 pipeline（`gc.collect()`）。

4. **大模型导致 NPU device lost**
   Meteor Lake NPU 3720 跑 `large-v3-turbo` 时，首次加载约 5 分钟，推理中可能报
   `ZE_RESULT_ERROR_DEVICE_LOST`（device hung / reset）。
   → 捕获该错误后自动切换到 Intel GPU，复用同一 int8 模型继续转录，不再中断任务；
   NPU 上建议使用 `small` / `base`，大模型优先 `intel_gpu`。

NPU 不可用时（未装驱动 / 非 Core Ultra），选择 `intel_npu` 会自动回退到 Intel GPU，
并在日志中给出提示，不会中断转录。

## 实现

| 文件 | 作用 |
| --- | --- |
| `videocaptioner/core/asr/openvino_asr.py` | OpenVINO Whisper 推理（GPU 分片 + NPU 分句/语言固定/回退） |
| `videocaptioner/core/asr/openvino_worker.py` | 独立子进程入口，输出 `PROGRESS`/`ERROR`/`DONE` 协议 |
| `videocaptioner/core/asr/openvino_process.py` | 主进程侧封装：启动子进程、解析进度、读取 SRT |
| `videocaptioner/core/utils/model_downloader.py` | hf-mirror + aria2c 断点续传、文件大小校验、fp16/int8 仓库选择 |
| `videocaptioner/core/utils/hardware.py` | 显卡 / NPU 枚举（不加载 OpenVINO）与设备映射 |
| `videocaptioner/ui/thread/hf_model_download_thread.py` | 模型下载线程 |
| `videocaptioner/core/asr/transcribe.py` | `intel_gpu` / `intel_npu` 分发到 OpenVINO |
| `videocaptioner/core/asr/faster_whisper.py` | 无 NVIDIA 时自动回退 CPU；模型缺失/损坏自动补齐 |
| `videocaptioner/ui/common/config.py` | 设备选项新增 `intel_gpu` / `intel_npu` |
| `videocaptioner/ui/components/FasterWhisperSettingWidget.py` | 设备/模型选择、模型状态校验、hf-mirror 下载 |

### 处理流程

```
GUI 进程
  └─ OpenVINOProcessASR（不 import openvino）
       └─ runtime/python.exe openvino_worker.py
            ├─ 下载/校验 OpenVINO 模型（NPU 自动 int8）
            ├─ GPU/CPU：WhisperPipeline(large-v3-turbo-fp16)，按 240s 分片
            ├─ NPU：WhisperPipeline(small-int8)，按停顿分句 (<14s)，语言固定
            └─ 输出 SRT → 主进程读取
```

NPU 检测在 UI 进程里通过 Windows 注册表完成（`VEN_8086&DEV_7D1D/AD1D/643E/B03E`），
不加载 OpenVINO，耗时 < 2ms；Linux 下读取 `/sys/class/accel/accel*`。

## 故障排查

| 现象 | 原因 / 处理 |
| --- | --- |
| 软件闪退，事件日志指向 `MSVCP140.dll` + `0xc0000005` | OpenVINO 被加载进了 Qt 主进程。确认 `openvino_process.py` 走子进程，且 UI 侧只调用 `hardware.describe_hardware()` |
| 未检测到 Intel GPU | 更新显卡驱动；运行上面的 `available_devices` 检查命令 |
| 选择 `intel_npu` 却回退到 GPU | 未安装 NPU 驱动或 NPU 被禁用；安装 Intel NPU 驱动 ≥ 32.0.100.3104，确认 `available_devices` 包含 `NPU` |
| NPU 初始化报 Level0 / out of memory | 已自动重试 `DISABLE_OPENVINO_GENAI_NPU_L0=1`；仍失败会回退 GPU |
| 显存不足 (Out of Memory) | 换小模型（`small`/`medium`）或关闭占用显存的程序 |
| 模型下载失败 | 默认走 `hf-mirror.com` 并自动回退 `huggingface.co`；也可手动下载到 `AppData/models/openvino/<repo>` |
| 词级时间戳无效 | Intel GPU 插件暂不支持 `word_timestamps`，会自动降级为分段级时间戳 |
| 路径包含中文导致 pip `.pth` 解码错误 | 将项目放到纯英文路径，或设置环境变量 `PYTHONUTF8=1` |

## 验证情况

- GPU 实测：Intel Core 7 251TE + Arc Pro B50（16GB），驱动 32.0.101.8976，
  OpenVINO 2026.4.1；22 分 20 秒视频 `large-v3-turbo` fp16 约 31 秒完成（约 43x 实时）
- NPU 路径：本机无 NPU 硬件，NPU 分支（分句、语言固定、无时间戳生成）
  已在 GPU 上用等价代码路径验证通过：
  90 秒音频切出 27 个语音段（最长 13.98s），自动识别语言 `<|en|>`，
  28 条字幕覆盖 0–90s
- `intel_npu` 在无 NPU 机器上自动回退 GPU，90 秒音频 5.6 秒完成

Arc A 系列使用同一套 OpenVINO GPU 插件与 fp16 模型，代码路径完全一致；
Core Ultra 7 155H 的 NPU 3720 使用 int8 模型与 NPU 专用分支，
实际速度受型号、驱动与功耗策略影响，建议先用 `small` 验证。


## 集成显卡 fp16 kernel 崩溃（工程样品机器）

现象：

- `intel_gpu` 转写时 worker 在 `Transcribing chunk 1/N` 之后立刻消失，日志里没有 ERROR
- WER 记录 `pythonw.exe`、`BEX64`、`0xc0000409`（fast-fail 7 = `FAST_FAIL_FATAL_APP_EXIT`，即 `abort()`）、
  故障模块 `ucrtbase.dll`，同时加载 `openvino_intel_gpu_plugin.dll`、`igdrcl64.dll`、`igc-default64.dll`
- 最小复现：对 `whisper-large-v3-turbo` 的 encoder 做一次 `[1, 128, 3000]` 推理，
  1 秒内 `Fatal Python error: Aborted`；换成 `INFERENCE_PRECISION_HINT=f32` 后同一推理 3.4 秒成功

结论：IGC 编译器在这台工程样品 iGPU（`PCI\VEN_8086&DEV_7D03`）上为 Whisper encoder
生成 fp16 kernel 时直接 `abort()`。这是原生进程退出，Python 的 `try/except` 完全捕获不到，
所以界面只能看到 worker 最后一行 stdout。

策略：

- OpenVINO 设备名包含 `(iGPU)` 时，自动对 GPU 使用 `INFERENCE_PRECISION_HINT=f32`
- 独立显卡（`(dGPU)`）继续保持 fp16
- 可用 `VIDEOCAPTIONER_OPENVINO_GPU_PRECISION=f32|fp16` 覆盖
- 失败重试顺序：`GPU` → `GPU + f32` → `CPU int8`；`NPU` 失败 → `CPU int8`

实测（Core Ultra 7 155H ES / Arc iGPU / 驱动 32.0.101.9175 / OpenVINO 2026.4.1）：

| 路径 | 60 秒音频耗时 | 相对实时 |
| --- | --- | --- |
| GPU int8 + f32 | 12.8 s | 约 4.7x |
| GPU fp16 + f32 | 15.9 s | 约 3.8x |
| CPU int8 | 14.2 s | 约 4.2x |
| NPU large-v3-turbo | 不可用（`npu_kmd.sys` 看门狗 TDR） | — |

在这台机器上 GPU 与 CPU 速度接近，GPU 不再是唯一选择；但 `intel_gpu` 现在可以稳定完成转写。


## Intel AI Boost NPU（3720，Meteor Lake）

### 失败现象与根因

之前的失败表现为 `npu_kmd.sys` 看门狗复位：

```
L0 zeFenceHostSynchronize result: ZE_RESULT_ERROR_DEVICE_LOST, code 0x70000001
WER LiveKernelEvent  Kernel_141  LKD_0x141_Tdr:6_IMAGE_npu_kmd.sys
```

实测定位到三个独立问题：

1. **冷启动编译很慢，但不是卡死**
   清空 `models/openvino/.cache` 后，第一次创建 NPU pipeline 需要 **~10 分钟**，
   单核 100% CPU（`PIPE_OK 584.9s`）。第二次命中缓存只要 2.7 秒。
   UI 上看起来像"卡住"，实际在编译。

2. **不传 `max_new_tokens` 会跑飞**
   NPU + `return_timestamps=True` 且不限制 token 数时，解码超过约 18 秒就会触发
   驱动看门狗，报 `DEVICE_LOST`。显式传 `max_new_tokens=448`（Whisper 单窗口上限）
   后稳定，30 秒音频 6.1 秒完成。

3. **`DEVICE_LOST` 后设备不会自愈**
   进程内重建 NPU pipeline 会挂住；必须 `Disable-PnpDevice` / `Enable-PnpDevice`
   复位 NPU（或重启）。因此应用层遇到 NPU 失败只能整体切到 GPU。

### 关键结论：`#4806` 在 OpenVINO 2026.4.1 已修复

之前为规避 `return_timestamps=True` 在 NPU 上返回空文本，实现了"停顿分句 +
逐句 `return_timestamps=False`"。实测 2026.4.1 上原生时间戳已经正常：

| 方案 | 60 秒音频 | 倍速 |
| --- | --- | --- |
| 旧：分句（16 次 encoder） | ~41 s | 1.5x |
| 新：原生时间戳（2 个 30s 窗口） | 10.5 s | 5.7x |

原因是每句都要跑一次固定 30 秒的 encoder，分句把算力浪费在重复编码上。

### 当前实现

- NPU 走与 GPU 相同的 `_transcribe_gpu()` 路径（240 秒块 + 原生时间戳）
- NPU 额外传 `max_new_tokens=448`
- 如果 NPU 返回空文本，自动回退到分句方案（`_transcribe_npu`）
- 如果 NPU 抛 `DEVICE_LOST`，自动切 Intel GPU（f32）+ int8 继续
- 分句方案保留为兜底，不再默认使用

### 性能实测（Core Ultra 7 155H ES / NPU 3720 / 驱动 32.0.100.5600）

| 路径 | 整集 1398.9 s | 倍速 |
| --- | --- | --- |
| **NPU large-v3-turbo int8（新）** | **189.1 s** | **7.4x** |
| GPU int8 + f32 | ~300 s | 4.7x |
| CPU int8 | ~330 s | 4.2x |

整集输出 542 个分段，覆盖到 `00:23:14`。

### 稳定性

- 3 秒片段连续 300 次生成：全部成功，内存稳定在 ~3.56 GB
- 早期分句路径累计 ~460 次调用中出现 1 次 `DEVICE_LOST`（约 0.2%）
- 注意：**不要在推理中途强杀进程**，会让 NPU 进入 `DEVICE_LOST` 状态，
  只能通过 PnP 复位或重启恢复
