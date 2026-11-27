# Intel Arc / NPU 本地转录：测试与诊断归档

记录时间：2026-11-26
分支：`feat/intel-arc-openvino`
相关提交：`ab668df`（GPU f32）、`75b8a2f`（NPU 原生时间戳 + 解码上限）

---

## 1. 测试环境

| 项 | 值 |
| --- | --- |
| 机器 | Lenovo ThinkBook 13x G4 IMH (21KR)，工程样品 |
| CPU | Intel Core Ultra 7 155H |
| 内存 | 32 GB |
| iGPU | Intel Arc Graphics，`PCI\VEN_8086&DEV_7D03` |
| iGPU 驱动 | `32.0.101.9175`（2026-11-04，`oem69.inf`） |
| NPU | Intel AI Boost 3720，`PCI\VEN_8086&DEV_7D1D` |
| NPU 驱动 | `32.0.100.5600`（2026-11-02，`oem31.inf`） |
| OS | Windows 11 26300 |
| BIOS | `LENOVO MTCN22WW.T05.SITBox8400`（2023-09-05，**未更新**） |
| OpenVINO | `2026.4.1-22982` |
| OpenVINO GenAI | `2026.4.1.0` |
| Python | 3.11.9（便携运行时） |

电源：AC 供电、电池 99%、平衡模式、处理器上限 100%、无降频、热区 ~27 °C。
上述条件排除了电源/温度导致的性能限制。

---

## 2. 测试方法

- 所有探测都运行在**独立子进程**，避免原生崩溃带走主进程
- 每个用例单独进程、单独日志，失败不影响后续
- 缓存目录：`%LOCALAPPDATA%\VideoCaptioner\models\openvino\.cache`
- 冷/热缓存分别测试（清空缓存 vs 复用编译产物）
- 长任务使用 WMI 创建进程 + 文件重定向，避免阻塞远程通道
- 探针内置看门狗，超时 `os._exit(9)` 并留下 `PROBE_TIMEOUT` 标记

关键探针：`gpu_probe.py` / `npu_probe.py`（临时工具，不随仓库发布）。

---

## 3. GPU：iGPU fp16 编译器崩溃

### 3.1 症状

GPU 转写时 worker 在 `Transcribing chunk 1/N` 后直接消失，应用日志没有 ERROR，
只能看到 worker 最后几行 stdout。

### 3.2 崩溃证据

```
EventType   = BEX64
Exception   = 0xc0000409  (fast-fail 7 = FAST_FAIL_FATAL_APP_EXIT / abort)
FaultModule = ucrtbase.dll  (+0xa527e)
LoadedModule:
  openvino_intel_gpu_plugin.dll
  igdrcl64.dll        (Intel GPU OpenCL)
  igc-default64.dll   (Intel Graphics Compiler)
```

无 `Display 4101`、无 WHEA、无 minidump —— 不是 TDR，是进程内 `abort()`。

### 3.3 最小复现

对 `whisper-large-v3-turbo-int8-ov` 的 encoder 单独做一次推理：

```python
model = core.read_model("openvino_encoder_model.xml")
model.reshape({model.input(0): [1, 128, 3000]})
compiled = core.compile_model(model, "GPU")
compiled({0: np.zeros((1, 128, 3000), np.float32)})
```

| 配置 | 结果 |
| --- | --- |
| 默认（fp16 计算） | `Fatal Python error: Aborted`（1 秒内） |
| `INFERENCE_PRECISION_HINT=f32` | 成功，3.4 s |

同机其它验证：

| 用例 | fp16 | f32 |
| --- | --- | --- |
| 最小卷积 `[1,3,64,64]` | ✅ 2.4 s | — |
| encoder 编译 | ✅ 4.1 s | ✅ 5.8 s |
| encoder 推理 | ❌ abort | ✅ 3.4 s |
| decoder 编译 | ✅ 3.7 s | — |

结论：GPU 插件本身可用，只有 IGC 生成的 **fp16 kernel** 在这台工程样品 iGPU 上
会让进程 `abort()`。int8 与 fp16 模型都会崩，因此与模型精度无关。

### 3.4 修复与验证

- iGPU 自动 `INFERENCE_PRECISION_HINT=f32`，独显保持 fp16
- 60 秒音频完整转写：12.8 s（4.7x），SRT 正常
- fp16 模型 + f32 提示：15.9 s（3.8x），同样正常

---

## 4. NPU：驱动看门狗与三个独立问题

### 4.1 失败时间线（修复前）

```
20:28:49  NPU 加载 large-v3-turbo int8
          CPU 分句：353 utterances / 1398.9 s
          第 1 句语言检测成功：<|en|>
20:32:42  L0 zeFenceHostSynchronize → ZE_RESULT_ERROR_DEVICE_LOST (0x70000001)
          OpenVINO: zero_wrappers.cpp:382
20:32:45  WER Kernel_141  LKD_0x141_Tdr:6_IMAGE_npu_kmd.sys
```

历史同类记录：`19:47:40`、`19:48:57`、`19:49:30` 各一次 `Kernel_141`；
`19:51:41` 出现过整机硬挂（`Kernel-Power 41`，bugcheck 字段全 0，无 minidump）。

### 4.2 问题一：冷启动编译被误认为卡死

```
T=2.5     PIPE_START
PIPE_OK   584.9 s      # 单核 100% CPU
GEN1_OK   2.6 s
```

- 清空缓存后首次创建 NPU pipeline：**~585 s（约 10 分钟）**
- 命中缓存：**2.7 s**
- 期间进程 CPU ≈ 94% × 1 核，内存约 1.1–2.1 GB

不是死锁，是 NPU 图编译。UI 上表现为长时间停在“加载模型”。

### 4.3 问题二：不限制 `max_new_tokens` 会跑飞

| 配置 | 结果 |
| --- | --- |
| `return_timestamps=True`，不传 `max_new_tokens` | 17.9 s 后 `DEVICE_LOST` |
| `return_timestamps=True`，`max_new_tokens=448` | 6.1 s 成功 |
| `return_timestamps=False`（旧分句路径） | 逐句 2.2 s，可用 |

加显式上限后连续 3 次 30 秒用例均成功；240 秒用例 35.0 s 完成、106 段、无截断。

### 4.4 问题三：`DEVICE_LOST` 后不能自愈

- 失败后重建 NPU pipeline：> 60 s 无响应（挂住）
- `Disable-PnpDevice` / `Enable-PnpDevice` 复位后：立即恢复，OpenVINO 重新枚举
- 结论：应用层只能切到 GPU，不能原地恢复 NPU

### 4.5 `#4806` 在新版本已修复

旧实现为规避 `return_timestamps=True` 返回空文本，做了“停顿分句 + 逐句无时间戳”。
实测 2026.4.1 上原生时间戳正常（30 s → 15 段，60 s → 29 段，时间轴正确）。

分句方案的真实代价：

| 方案 | 60 秒音频 | 倍速 |
| --- | --- | --- |
| 分句（16 次 encoder） | ~41 s | 1.5x |
| 原生时间戳（2 个窗口） | 10.5 s | 5.7x |

原因：每句都要重跑一次固定 30 秒 encoder。

---

## 5. 最终性能

| 路径 | 音频 | 耗时 | 倍速 |
| --- | --- | --- | --- |
| NPU int8（原生时间戳） | 30 s | 6.1 s | 4.9x |
| NPU int8 | 60 s | 10.5 s | 5.7x |
| NPU int8 | 240 s | 35.0 s | 6.9x |
| **NPU int8（整集）** | **1398.9 s** | **189.1 s** | **7.4x** |
| GPU int8 + f32 | 60 s | 12.8 s | 4.7x |
| GPU fp16 + f32 | 60 s | 15.9 s | 3.8x |
| CPU int8 | 60 s | 14.2 s | 4.2x |

整集输出：6 个 240 秒块、542 个分段，最后时间轴 `00:23:14,619`（覆盖到结尾）。

---

## 6. 稳定性数据

| 项目 | 结果 |
| --- | --- |
| 300 次连续 NPU 生成（3 s 片段） | 300/300 成功 |
| 80 次连续生成 | 80/80 成功 |
| 40 次连续生成 | 40/40 成功 |
| 早期分句路径累计 ~460 次 | 1 次 `DEVICE_LOST`（~0.2%） |
| 进程内存 | 稳定在 ~3.56 GB，无持续增长 |
| 整集 NPU 转写 | 成功，无中断 |

已知可复现的触发条件：

1. 推理中途强杀进程 → 设备进入 `DEVICE_LOST`，后续推理全部失败
2. `return_timestamps=True` 且不限制 `max_new_tokens` → 看门狗复位

---

## 7. 复现命令

```powershell
# 查看设备
runtime\python.exe -c "import openvino as ov; c=ov.Core(); print(c.available_devices)"

# 查看 NPU 属性
runtime\python.exe -c "import openvino as ov; c=ov.Core(); print(c.get_property('NPU','DEVICE_ARCHITECTURE'))"

# 复位 NPU（需要管理员）
Disable-PnpDevice -InstanceId 'PCI\VEN_8086&DEV_7D1D&SUBSYS_380C17AA&REV_04\3&11583659&0&58' -Confirm:$false
Enable-PnpDevice  -InstanceId 'PCI\VEN_8086&DEV_7D1D&SUBSYS_380C17AA&REV_04\3&11583659&0&58' -Confirm:$false

# 单元测试
pytest tests/test_asr/test_openvino_backend.py -q
```

---

## 8. 安装包记录

| 版本 | 说明 |
| --- | --- |
| v3 | 首次内置 int8 模型 + OpenVINO 运行时；GPU 仍走 fp16，失败 |
| v4 | 新增 `intel_cpu`、GPU 崩溃后回退 CPU |
| v5 | iGPU 自动 f32，GPU 路径可用（整集验证通过） |
| **v6** | NPU 原生时间戳 + `max_new_tokens=448`，三条路径全部可用 |

v6：`VideoCaptioner-IntelArc-Setup-v6.exe`，902.5 MB，
SHA-256 `6f7780b836ea172c82e6db0f2c855323aab137b527fbe3b7f4ae00a7832b9d6c`。

---

## 9. 遗留与后续

- NPU 低概率 `DEVICE_LOST` 仍存在，应用层已通过 GPU 回退兜底
- 能否在应用内自动复位 NPU（需要管理员权限）值得评估
- 零售版 Meteor Lake / Arrow Lake / Lunar Lake 的 f32 策略需要更多样本验证
- NPU 冷编译 10 分钟可以通过随包分发编译缓存进一步优化
