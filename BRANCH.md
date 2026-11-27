# 分支说明：Intel Arc / AI Boost NPU 本地转录适配

本目录是 [WEIFENG2333/VideoCaptioner](https://github.com/WEIFENG2333/VideoCaptioner)
的支线项目，基于 `master`（v1.4.x）创建：

- 分支：`feat/intel-arc-openvino`
- 目标：让 Intel 硬件完成本地 Whisper 转录
  - Arc A 系列（Alchemist：A310/A380/A580/A750/A770）——首要目标
  - Arc B 系列（Battlemage）、Core Ultra 核显（如 155H 的 Arc iGPU）
  - **Intel AI Boost NPU**（酷睿 Ultra 7 155H 的 Meteor Lake NPU 3720、
    Lunar Lake NPU 4000、Arrow Lake / Panther Lake NPU）
- 后端：Intel OpenVINO GenAI（`WhisperPipeline`）
  - GPU：fp16 预转换模型
  - NPU：int8 预转换模型 + 语音分句 + 语言固定 + Level0 重试
- 架构：OpenVINO 运行在独立子进程，避免与 PyQt5 的 MSVCP140 冲突导致闪退

## 快速开始

```bash
pip install -e ".[intel]"
videocaptioner-gui
```

设置 → 转录 → Faster Whisper：

- 运行设备：`intel_gpu`（Arc / 核显）或 `intel_npu`（AI Boost NPU）
- 模型：`large-v3-turbo`（GPU 推荐）/ `small`（NPU 推荐）
- 首次转录会自动下载 OpenVINO 模型到 `AppData/models/openvino/`
  - `intel_gpu` → `whisper-*-fp16-ov`
  - `intel_npu` → `whisper-*-int8-ov`

NPU 驱动要求：Intel NPU 驱动 ≥ 32.0.100.3104（Windows）。

## 验证

```bash
# 轻量单元测试（无需 GPU / NPU）
pytest tests/test_asr/test_openvino_backend.py -v

# 确认 OpenVINO 能看到设备
python -c "import openvino as ov; c=ov.Core(); print(c.available_devices)"
```

## 已实现内容

- `intel_gpu` / `intel_npu` 设备选项与分发
- OpenVINO Whisper 子进程后端（进度上报、SRT 输出、词级时间戳自动降级）
- **NPU 专项适配**
  - 自动使用 INT8 模型（体积约减半）
  - 规避 `return_timestamps=True` 在 NPU 上输出损坏的问题（openvino.genai#4806）：
    20ms RMS 语音活动检测 + 自然停顿分句（每段 < 14s）+ 精确区间时间戳
  - 规避 NPU 第二次调用复用旧音频状态的问题（openvino#37937）：
    首段自动检测语言后固定语言 token
  - Level0 OOM 自动重试（`DISABLE_OPENVINO_GENAI_NPU_L0=1`）→ 失败回退 GPU
  - 推理中 NPU `DEVICE_LOST` 自动回退 Intel GPU，并复用同一 int8 模型继续转录
  - 安装包内置 OpenVINO GPU/NPU 插件、NPU 编译器与 `large-v3-turbo-int8` 模型
  - worker 退出前显式销毁 pipeline（openvino.genai#1460）
  - UI 进程通过注册表枚举 NPU（不加载 OpenVINO，< 2ms）
- 修复打包 dev 模式误判：安装后数据/模型统一放到 `%LOCALAPPDATA%\VideoCaptioner`
- hf-mirror + aria2c / 并行分段下载，下载后校验大小
- Faster-Whisper 无 NVIDIA 时自动回退 CPU、模型缺失/损坏自动补齐
- 设置页模型列表按设备过滤、模型下载状态校验
- 文档：`docs/config/intel-arc-gpu.md`

## 说明

- 当前分支只做“能跑通、可复现”的适配，不修改上游原有行为（默认仍是 `cuda`/`auto`）。
- `auto` 会按 NVIDIA → Intel GPU → Intel NPU → CPU 的顺序自动选择；
  Core Ultra 155H 上默认走 iGPU（吞吐高于 NPU），低功耗场景可手动切 `intel_npu`。
- 本机（Arc Pro B50）无 NPU 硬件，NPU 分支已在 GPU 上做等价路径验证；
  155H NPU 3720 按 Intel 官方支持与上述 workaround 适配。
- 更详细的原理、模型建议和故障排查见 `docs/config/intel-arc-gpu.md`。
- **iGPU fp16 编译器崩溃修复**：OpenVINO 设备名含 `(iGPU)` 时自动使用 `INFERENCE_PRECISION_HINT=f32`（工程样品 MTL 的 IGC 在编译 fp16 kernel 时会 `abort()`）；独显保持 fp16；失败重试 GPU -> GPU+f32 -> CPU int8
- `intel_cpu` 设备选项（OpenVINO CPU + int8），作为不稳定 GPU 的稳定回退
