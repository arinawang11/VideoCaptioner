# VideoCaptioner Intel Arc / AI Boost NPU v7

v7 在 v6 的基础上合并了两个社区修复，并修正了一个 fp16 模型完整性判断的回归。
安装包仍然内置 OpenVINO 运行时与 `whisper-large-v3-turbo-int8` 模型，安装后无需下载模型。

## 本版变更

### 1. 模型下载中断后可以自愈（#1，作者 @arinawang11）

- 完整性检查从“两个 XML 存在”扩展为 **18 个必需文件**（IR、tokenizer IR、
  preprocessor / generation config、HF tokenizer 文件）
- 下载增加 3 次指数退避重试，aria2c / 并行分片 / requests 三条路径统一受益，
  `.part` 断点续传保证不重复下载
- 修复打开时缺 `preprocessor_config.json` 导致 GenAI 按 80 mel 处理、
  与 large-v3 的 128 mel 冲突的问题

### 2. GPU 词级时间戳回退后输出错乱（#2，作者 @arinawang11）

- 词级时间戳在 GPU 上不被支持时，失败的那次 `generate()` 会污染 pipeline 内部状态；
  现在回退前重建 pipeline，避免第一个窗口的段落错位/丢失
- 额外加了 NPU 保护：`DEVICE_LOST` 直接上抛，交给 NPU → GPU 回退，
  避免在已失效的 NPU 上重建 pipeline 卡死

### 3. fp16 模型不再被误判为不完整

`openvino_config.json` 只存在于 int8 仓库，fp16 仓库没有这个文件。
本版把它从必需列表移除，避免 fp16 用户每次启动都触发重复下载。

## 实测（与 v6 相同）

测试机：Lenovo ThinkBook 13x G4 IMH，Core Ultra 7 155H，32 GB，
Arc iGPU，Intel AI Boost NPU 3720，Windows 11 26300。

| 路径 | 音频 | 耗时 | 倍速 |
| --- | --- | --- | --- |
| **NPU + int8** | 整集 1398.9 s | **189.1 s** | **7.4x** |
| GPU + int8 + f32 | 60 s | 12.8 s | 4.7x |
| CPU + int8 | 60 s | 14.2 s | 4.2x |

## 安装

1. 下载 `VideoCaptioner-IntelArc-Setup-v7.exe`
2. 运行安装（无需管理员权限），覆盖安装即可
3. 设置 → 转录 → Faster Whisper：
   - 运行设备：`intel_npu`（最快）/ `intel_gpu` / `intel_cpu`
   - 模型：`large-v3-turbo`（已内置 int8）

## 安装包校验

| 项 | 值 |
| --- | --- |
| 文件 | `VideoCaptioner-IntelArc-Setup-v7.exe` |
| 大小 | 946,381,484 字节（902.5 MB） |
| SHA-256 | `697b41334c8f775a7d94459de1ed0e11ef5972a17db1f9555d499ef3f7811ecb` |

## 已知问题

- NPU 缓存为空时首次编译约 10 分钟，界面停在“加载模型”属正常现象
- 不要在推理过程中强杀进程：会让 NPU 进入 `DEVICE_LOST`，需要 PnP 复位或重启
- NPU 仍存在低概率驱动看门狗（约 0.2% 调用），触发后会自动切到 GPU 继续
- 词级时间戳在 Intel GPU 上不支持，会自动降级为分段级时间戳

## 验证

- CI：Ubuntu / Windows 双平台（ruff + OpenVINO 单元测试），14 passed
- 本地冒烟：90 秒音频，GPU fp16 + 词级时间戳，8.3 秒完成，69 行 SRT
- 模型完整性：内置 fp16 / int8 模型均判定 complete，无重复下载
