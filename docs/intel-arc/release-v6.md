# VideoCaptioner Intel Arc / AI Boost NPU v6

本版本把 Intel 平台的三条本地转录路径全部打通：**NPU、GPU、CPU**。
安装包内置 OpenVINO 运行时与 `whisper-large-v3-turbo-int8` 模型，安装后无需再下载模型。

## 亮点

- **核显 fp16 崩溃修复**：OpenVINO 设备名含 `(iGPU)` 时自动使用
  `INFERENCE_PRECISION_HINT=f32`，避开 IGC 编译器在部分核显上生成 fp16 kernel 时的
  `abort()`；独显仍保持 fp16。
- **NPU 原生时间戳**：不再逐句分句，改为 240 秒分块 + 原生时间戳，
  整集速度从 1.5x 提升到 **7.4x**。
- **NPU 解码上限**：显式 `max_new_tokens=448`，避免解码越界触发
  `npu_kmd.sys` 看门狗（`ZE_RESULT_ERROR_DEVICE_LOST`）。
- **自动回退**：`GPU → GPU+f32 → CPU int8`；`NPU → GPU → CPU`，任务不会中断。
- **新增 `intel_cpu`**：OpenVINO CPU + int8，作为不稳定驱动的稳定兜底。

## 实测性能

测试机：Lenovo ThinkBook 13x G4 IMH，Core Ultra 7 155H，32 GB，
Arc iGPU，Intel AI Boost NPU 3720，Windows 11 26300。

| 路径 | 音频 | 耗时 | 倍速 |
| --- | --- | --- | --- |
| **NPU + int8** | 整集 1398.9 s | **189.1 s** | **7.4x** |
| NPU + int8 | 240 s | 35.0 s | 6.9x |
| GPU + int8 + f32 | 60 s | 12.8 s | 4.7x |
| CPU + int8 | 60 s | 14.2 s | 4.2x |

整集 NPU 输出 542 个分段，覆盖到 `00:23:14`，无中断。

## 安装

1. 下载 `VideoCaptioner-IntelArc-Setup-v6.exe`
2. 运行安装（无需管理员权限），覆盖安装即可
3. 设置 → 转录 → Faster Whisper：
   - 运行设备：`intel_npu`（最快）/ `intel_gpu` / `intel_cpu`
   - 模型：`large-v3-turbo`（已内置 int8）

## 安装包校验

| 项 | 值 |
| --- | --- |
| 文件 | `VideoCaptioner-IntelArc-Setup-v6.exe` |
| 大小 | 946,378,859 字节（902.5 MB） |
| SHA-256 | `6f7780b836ea172c82e6db0f2c855323aab137b527fbe3b7f4ae00a7832b9d6c` |

## 环境要求

- Windows 10/11 x64
- Intel 显卡驱动（OpenVINO GPU 插件）
- Intel NPU 驱动 ≥ 32.0.100.3104（仅使用 NPU 时需要）
- BIOS 无需更新

## 已知问题

- NPU 缓存为空时首次编译约 **10 分钟**，界面停在“加载模型”属正常现象，之后约 3 秒
- 不要在推理过程中强杀进程：会让 NPU 进入 `DEVICE_LOST`，需要 PnP 复位或重启
- NPU 仍存在低概率驱动看门狗（约 0.2% 调用），触发后会自动切到 GPU 继续
- 词级时间戳在 Intel GPU 上不支持，会自动降级为分段级时间戳

## 验证

- CI：Ubuntu / Windows 双平台（ruff + OpenVINO 单元测试）
- 单元测试：13 passed
- 文档：[README-IntelArc.md](https://github.com/liu-nexotic/VideoCaptioner/blob/feat/intel-arc-openvino/README-IntelArc.md)、
  [测试报告](https://github.com/liu-nexotic/VideoCaptioner/blob/feat/intel-arc-openvino/docs/intel-arc/test-report.md)
