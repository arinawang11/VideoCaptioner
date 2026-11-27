# 分支说明：Intel Arc / AI Boost NPU 本地转录适配

本目录是 [WEIFENG2333/VideoCaptioner](https://github.com/WEIFENG2333/VideoCaptioner)
的支线项目，基于 `master`（v1.4.x）创建：

- 分支：`feat/intel-arc-openvino`
- 目标：让 Intel 硬件完成离线 Whisper 转写
  - Intel Arc 独显（A/B 系列）
  - Core Ultra 核显（如 155H 的 Arc iGPU）
  - Intel AI Boost NPU（Meteor Lake 3720 等）
  - Intel CPU（OpenVINO，兜底）
- 后端：Intel OpenVINO GenAI `WhisperPipeline`
- 架构：OpenVINO 运行在独立子进程，UI 进程不加载 OpenVINO

> 状态：**已完成**。GPU / NPU / CPU 三条路径均在
> Core Ultra 7 155H 工程样品上实测通过；自 v8 起安装包仅内置运行时（~245 MB），模型按需自动拉取。

## 快速开始

```bash
pip install -e ".[intel]"
videocaptioner-gui
```

设置 → 转录 → Faster Whisper → 运行设备：

| 设备 | 说明 |
| --- | --- |
| `intel_npu` | Intel AI Boost；本机最快（整集 7.4x） |
| `intel_gpu` | Arc / 核显；核显自动使用 f32 精度提示 |
| `intel_cpu` | OpenVINO CPU + int8；稳定兜底 |
| `cpu` / `cuda` | 上游原有路径，本分支不改行为 |

模型推荐 `large-v3-turbo`。首次使用会自动按需下载
对应精度的模型到 `%LOCALAPPDATA%\VideoCaptioner\models\openvino\`（支持断点续传与 3 次重试）。

## 已实现内容

- `intel_gpu` / `intel_npu` / `intel_cpu` 设备选项与分发
- OpenVINO Whisper 子进程后端（进度上报、SRT 输出、词级时间戳自动降级）
- **核显 fp16 编译器崩溃修复**
  - OpenVINO 设备名含 `(iGPU)` → 自动 `INFERENCE_PRECISION_HINT=f32`
  - 独显保持 fp16；可用 `VIDEOCAPTIONER_OPENVINO_GPU_PRECISION` 覆盖
  - 失败重试链：`GPU` → `GPU+f32` → `CPU int8`
- **NPU 专项适配**
  - 改用原生时间戳（240 秒分块），不再逐句重复编码
  - 显式 `max_new_tokens=448`，避免解码越界触发 NPU 看门狗
  - 时间戳返回空时自动回退到停顿分句方案
  - `DEVICE_LOST` 自动切 Intel GPU（f32）+ int8 继续
  - Level0 OOM 自动重试（`DISABLE_OPENVINO_GENAI_NPU_L0=1`）
  - worker 退出前显式销毁 pipeline（openvino.genai#1460）
  - UI 进程通过注册表枚举 NPU（不加载 OpenVINO，< 2ms）
- 修复打包 dev 模式误判：数据/模型统一放到 `%LOCALAPPDATA%\VideoCaptioner`
- hf-mirror + aria2c / 并行分段下载，下载后校验大小
- Faster-Whisper 无 NVIDIA 时自动回退 CPU、模型缺失/损坏自动补齐
- 设置页模型列表按设备过滤、模型下载状态校验

## 实测性能

测试机：Core Ultra 7 155H / 32 GB / Arc iGPU / NPU 3720 / Windows 11 26300。

| 路径 | 音频 | 耗时 | 倍速 |
| --- | --- | --- | --- |
| **NPU int8** | 整集 1398.9 s | **189.1 s** | **7.4x** |
| NPU int8 | 60 s | 10.5 s | 5.7x |
| GPU int8 + f32 | 60 s | 12.8 s | 4.7x |
| CPU int8 | 60 s | 14.2 s | 4.2x |

整集 NPU 输出 542 段，覆盖到 `00:23:14`。

## 验证

```bash
# 轻量单元测试（无需 GPU / NPU）
pytest tests/test_asr/test_openvino_backend.py -v

# 确认 OpenVINO 能看到设备
python -c "import openvino as ov; c=ov.Core(); print(c.available_devices)"
```

## 说明

- 当前分支只做“能跑通、可复现”的适配，不修改上游默认行为。
- `auto` 会按 NVIDIA → Intel GPU → Intel NPU → CPU 顺序自动选择。
- BIOS 未更新（工程样品机器，禁止更新）。
- NPU 缓存为空时首次编译约 10 分钟，属正常现象。
- 不要在推理过程中强杀进程：会让 NPU 进入 `DEVICE_LOST`，需 PnP 复位或重启。
- 详细说明见：
  - [`README-IntelArc.md`](README-IntelArc.md)
  - [`docs/config/intel-arc-gpu.md`](docs/config/intel-arc-gpu.md)
  - [`docs/intel-arc/test-report.md`](docs/intel-arc/test-report.md)

## 安装包

| 版本 | 说明 |
| --- | --- |
| v3 | 首次内置 int8 模型；GPU 走 fp16，崩溃 |
| v4 | 新增 `intel_cpu`、GPU 崩溃回退 CPU |
| v5 | iGPU 自动 f32，GPU 可用 |
| **v6** | NPU 原生时间戳 + 解码上限，三条路径全部可用 |
| **v7** | 合并社区 PR #1 / #2（下载自愈、词级时间戳回退），修正 fp16 完整性误判 |
| **v8** | 合并社区 PR #3（4 项稳定性修复）；不再封装模型改为按需拉取，安装包降至 245.1 MB |

v8：`VideoCaptioner-IntelArc-Setup-v8.exe`，245.1 MB，
SHA-256 `a86db8c5e0b3eb45aae8aa8646f42fe3485e199d0044fa5b6780ef14ab3716d6`。

v7：`VideoCaptioner-IntelArc-Setup-v7.exe`，902.5 MB，
SHA-256 `697b41334c8f775a7d94459de1ed0e11ef5972a17db1f9555d499ef3f7811ecb`。

v6：`VideoCaptioner-IntelArc-Setup-v6.exe`，902.5 MB，
SHA-256 `6f7780b836ea172c82e6db0f2c855323aab137b527fbe3b7f4ae00a7832b9d6c`。
