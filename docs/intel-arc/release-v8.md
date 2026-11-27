# VideoCaptioner Intel Arc / AI Boost NPU v8

v8 合并了 PR #3 的四项稳定性修复，并从本版起**不再将 Whisper 模型封装进安装包**——
安装包体积从 **902.5 MB 缩小到 245.1 MB（减少 73%）**，首次转录时由程序自动按需拉取对应模型。

## 本版变更

### 1. 安装包瘦身：不再内置模型

- 安装包仅保留便携 Python 运行时、OpenVINO GPU/NPU/CPU 插件与工具链
- 首次转录时程序自动从 `hf-mirror.com`（失败回退 `huggingface.co`）按需下载所选模型到
  `%LOCALAPPDATA%\VideoCaptioner\models\openvino\`
- 配合 v7 的断点续传 + 3 次指数退避重试 + 18 文件完整性自愈，下载中断可自动补齐
- 若本地已有完整的 int8 模型且未装 fp16，GPU 路径会自动复用该 int8 模型，避免重复下载；
  残缺目录不会误触发复用

### 2. 合并社区修复（PR #3，作者 @arinawang11）

1. **模型管理对话框关闭递归爆栈**：移除 `_on_dialog_reject()` 中多余的 `self.reject()`，
   解决关闭对话框时触发 `RecursionError` 的问题
2. **消除 OpenVINO worker 死缓存写入**：`BaseASR.run()` 的缓存写入与读侧保持一致，
   `use_cache=False` 的 worker 不再向共享 diskcache 写入永不被读的条目
3. **修复 `optimize_timing` 重叠段倒挂**：相邻字幕存在重叠（负 gap）时保持原边界，
   避免中点公式算出 `start > end` 的倒挂时间轴；正向小间隙收紧行为不变
4. **分离 UI 与 worker 日志**：worker 通过 `VIDEOCAPTIONER_LOG_FILE=openvino_worker.log`
   写独立日志文件，避免 Windows 下两进程共享 `app.log` 轮转时因句柄占用导致 `PermissionError` 丢日志

## 安装与使用

1. 下载 `VideoCaptioner-IntelArc-Setup-v8.exe`（245.1 MB）
2. 运行安装（无需管理员权限），覆盖旧版即可保留已有模型与设置
3. 设置 → 转录 → Faster Whisper：
   - 运行设备：`intel_npu`（本机最快）/ `intel_gpu` / `intel_cpu`
   - 模型：`large-v3-turbo`（或按需选择 `small` / `medium` 等）
4. 首次开始转录时程序会自动下载所选 OpenVINO 模型（支持断点续传与重试）

## 安装包校验

| 项 | 值 |
| --- | --- |
| 文件 | `VideoCaptioner-IntelArc-Setup-v8.exe` |
| 大小 | 257,019,366 字节（245.1 MB） |
| SHA-256 | `a86db8c5e0b3eb45aae8aa8646f42fe3485e199d0044fa5b6780ef14ab3716d6` |

## 环境要求

- Windows 10/11 x64
- Intel 显卡驱动（OpenVINO GPU 插件）
- Intel NPU 驱动 ≥ 32.0.100.3104（仅使用 NPU 时需要）
- 首次下载模型需联网（默认走 `hf-mirror.com`，失败自动回退 `huggingface.co`）

## 已知问题

- NPU 缓存为空时首次编译约 10 分钟，界面停在“加载模型”属正常现象，之后约 3 秒
- 不要在推理过程中强杀进程：会让 NPU 进入 `DEVICE_LOST`，需要 PnP 复位或重启
- NPU 仍存在低概率驱动看门狗（约 0.2% 调用），触发后会自动切到 GPU 继续
- 词级时间戳在 Intel GPU 上不支持，会自动降级为分段级时间戳

## 验证

- PR #3 CI：Ubuntu / Windows（Intel backend）+ 上游 Python CI 全部通过
- 本地验证：重叠段边界保持、正向间隙缝合、`use_cache=False` 零写入、`openvino_worker.log` 独立写入
- 打包验证：portable 运行时安装干净 wheel（`1.4.4.dev2+gb1cfd1a53`），Inno Setup 编译无模型安装包 245.1 MB
