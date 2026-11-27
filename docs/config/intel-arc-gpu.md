# Intel Arc GPU 本地转录（OpenVINO）

本分支（`feat/intel-arc-openvino`）为 **Intel Arc 显卡** 增加本地 Whisper 语音转录能力，
首要适配目标是 **Arc A 系列（Alchemist：A310 / A380 / A580 / A750 / A770）**，
同时兼容 Arc B 系列（Battlemage）、Intel 核显和 NPU。

## 为什么需要单独适配

Faster-Whisper 使用的 CTranslate2 只支持 NVIDIA CUDA，
在 Intel 显卡上选择 `cuda` 会直接失败（或静默回退 CPU）。

本分支改用 Intel 官方的 **OpenVINO GenAI WhisperPipeline**：

- 支持 Arc A 系列 / B 系列 / 核显 / NPU
- 使用 OpenVINO 官方预转换的 fp16 模型，无需自行转换
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

`intel` 可选依赖固定为相互兼容的版本：

```
openvino==2026.4.1
openvino-tokenizers==2026.4.1.0
openvino-genai==2026.4.1.0
```

驱动要求：安装最新版 Intel 显卡驱动（Arc A 系列建议 31.0.101 及以上）。
可用以下命令确认 OpenVINO 能看到显卡：

```bash
python -c "import openvino as ov; c=ov.Core(); print(c.available_devices); print(c.get_property('GPU','FULL_DEVICE_NAME'))"
```

## 使用

1. 打开软件 → 设置 → 转录 → Faster Whisper
2. **运行设备**选择 `intel_gpu`（Intel NPU 选择 `intel_npu`）
3. **模型**选择 `large-v3-turbo`（默认推荐），首次转录会自动下载 OpenVINO 模型
4. 直接开始转录

模型下载到 `AppData/models/openvino/`（源码运行时为仓库下的 `AppData/models/openvino/`），
编译缓存位于 `AppData/models/openvino/.cache`，第二次启动不再重新编译。

### 模型与显存建议

| 模型 | fp16 大小（约） | 建议显存 | 说明 |
| --- | --- | --- | --- |
| tiny / base | 75MB / 145MB | 2GB+ | 速度最快，准确率低，仅测试用 |
| small | 484MB | 3GB+ | 低显存 A380 可用 |
| medium | 1.5GB | 5GB+ | 平衡 |
| large-v3-turbo | 1.6GB | 6GB+ | **推荐**，速度与准确率平衡 |
| large-v3 | 3.0GB | 10GB+ | 准确率最高，速度较慢 |

A 系列（如 A770 16GB）建议 `large-v3-turbo`；
A380（6GB）建议 `small` 或 `medium`。

## 实现

| 文件 | 作用 |
| --- | --- |
| `videocaptioner/core/asr/openvino_asr.py` | OpenVINO Whisper 推理实现（子进程内使用） |
| `videocaptioner/core/asr/openvino_worker.py` | 独立子进程入口，输出 `PROGRESS`/`ERROR`/`DONE` 协议 |
| `videocaptioner/core/asr/openvino_process.py` | 主进程侧封装：启动子进程、解析进度、读取 SRT |
| `videocaptioner/core/utils/model_downloader.py` | hf-mirror + aria2c 断点续传、文件大小校验 |
| `videocaptioner/core/utils/hardware.py` | 显卡枚举（不加载 OpenVINO）与设备映射 |
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
            ├─ 下载/校验 OpenVINO 模型（如缺失）
            ├─ WhisperPipeline(model, GPU.1/GPU.0/NPU)
            ├─ 按 240s 分片转录
            └─ 输出 SRT → 主进程读取
```

## 故障排查

| 现象 | 原因 / 处理 |
| --- | --- |
| 软件闪退，事件日志指向 `MSVCP140.dll` + `0xc0000005` | OpenVINO 被加载进了 Qt 主进程。确认 `openvino_process.py` 走子进程，且 UI 侧只调用 `hardware.describe_hardware()` |
| 未检测到 Intel GPU | 更新显卡驱动；运行上面的 `available_devices` 检查命令 |
| 显存不足 (Out of Memory) | 换小模型（`small`/`medium`）或关闭占用显存的程序 |
| 模型下载失败 | 默认走 `hf-mirror.com` 并自动回退 `huggingface.co`；也可手动下载到 `AppData/models/openvino/<repo>` |
| 词级时间戳无效 | Intel GPU 插件暂不支持 `word_timestamps`，会自动降级为分段级时间戳 |
| 路径包含中文导致 pip `.pth` 解码错误 | 将项目放到纯英文路径，或设置环境变量 `PYTHONUTF8=1` |

## 已验证环境

- CPU：Intel Core 7 251TE
- GPU：Intel Arc Pro B50（Battlemage，16GB）
- 驱动：32.0.101.8976
- OpenVINO：2026.4.1
- 结果：22 分 20 秒视频，`large-v3-turbo` fp16，约 31 秒完成（约 43x 实时）

Arc A 系列使用同一套 OpenVINO GPU 插件与 fp16 模型，代码路径完全一致；
实际速度受型号、显存和驱动影响，建议先用 `small` 验证，再按显存选择更大模型。
