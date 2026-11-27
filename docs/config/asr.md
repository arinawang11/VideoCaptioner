# ASR 配置指南

语音识别（ASR）配置详解。

## 支持的 ASR 引擎

| 引擎 | 特点 | 推荐场景 |
|------|------|---------|
| **FasterWhisper** | 准确度高，支持GPU | 推荐使用 |
| **WhisperCpp** | 轻量级 | CPU环境 |
| **Whisper API** | 云端服务 | 无需本地模型 |
| **B接口/J接口** | 免费在线 | 快速测试 |

## Intel Arc 显卡（OpenVINO）

Intel Arc（A 系列 / B 系列）、核显或 NPU 请把运行设备设为 `intel_gpu` 或 `intel_npu`，
详见 [Intel Arc GPU 本地转录](/config/intel-arc-gpu)。

## 模型下载

待补充...

## 配置参数

待补充...

---

相关文档：
- [快速开始](/guide/getting-started)
- [LLM 配置](/config/llm)
