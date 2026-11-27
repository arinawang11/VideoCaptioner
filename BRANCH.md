# 分支说明：Intel Arc（A 系列优先）本地转录适配

本目录是 [WEIFENG2333/VideoCaptioner](https://github.com/WEIFENG2333/VideoCaptioner)
的支线项目，基于 `master`（v1.4.x）创建：

- 分支：`feat/intel-arc-openvino`
- 目标：让 Intel Arc 显卡（优先 A 系列 Alchemist：A310/A380/A580/A750/A770，
  同时支持 B 系列 Battlemage、核显和 NPU）可以完成本地 Whisper 转录
- 后端：Intel OpenVINO GenAI（`WhisperPipeline`），fp16 预转换模型
- 架构：OpenVINO 运行在独立子进程，避免与 PyQt5 的 MSVCP140 冲突导致闪退

## 快速开始

```bash
pip install -e ".[intel]"
videocaptioner-gui
```

设置 → 转录 → Faster Whisper：

- 运行设备：`intel_gpu`（或 `intel_npu`）
- 模型：`large-v3-turbo`（推荐）
- 首次转录会自动下载 OpenVINO 模型到 `AppData/models/openvino/`

## 验证

```bash
# 轻量单元测试（无需 GPU）
pytest tests/test_asr/test_openvino_backend.py -v

# 确认 OpenVINO 能看到 Arc 显卡
python -c "import openvino as ov; c=ov.Core(); print(c.available_devices)"
```

## 已实现内容

- `intel_gpu` / `intel_npu` 设备选项与分发
- OpenVINO Whisper 子进程后端（进度上报、SRT 输出、词级时间戳自动降级）
- hf-mirror + aria2c 模型下载，下载后校验大小（也修复了 Faster-Whisper 模型下载）
- Faster-Whisper 无 NVIDIA 时自动回退 CPU、模型缺失/损坏自动补齐
- 设置页模型列表按设备过滤、模型下载状态校验
- 文档：`docs/config/intel-arc-gpu.md`

## 说明

- 当前分支只做“能跑通、可复现”的最小适配，不修改上游原有行为（默认仍是 `cuda`）。
- 如果你希望默认设备自动选择 `intel_gpu`，可以在首次启动时手动切换一次，配置会保存。
- 更详细的原理、模型/显存建议和故障排查见 `docs/config/intel-arc-gpu.md`。
