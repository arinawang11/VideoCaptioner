# coding:utf-8
"""Standalone OpenVINO Whisper worker process.

OpenVINO 2026 requires a newer MSVC runtime (MSVCP140 >= 14.4) than the one
bundled with PyQt5 (14.26). Loading both DLL sets in one process causes an
access violation (0xc0000005) and the whole GUI crashes. The UI process only
starts this script; all OpenVINO DLLs are loaded here.

stdout protocol:
    PROGRESS <0-100>|<message>
    ERROR <message>
    DONE
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

# 本进程的日志全部写独立文件，避免与 UI 进程共写 app.log 导致轮转时
# 另一进程 rename 失败（见 core/utils/logger.py）。
os.environ.setdefault("VIDEOCAPTIONER_LOG_FILE", "openvino_worker.log")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Support running directly from a source checkout (repo root on sys.path).
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def emit_progress(percent, message):
    message = str(message).replace("\n", " ").replace("\r", " ")
    print(f"PROGRESS {int(percent)}|{message}", flush=True)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="OpenVINO Whisper worker")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--device", default="intel_gpu")
    parser.add_argument("--precision", default="")
    parser.add_argument("--precision-hint", default="")
    parser.add_argument("--gpu-name", default="")
    parser.add_argument("--language", default="en")
    parser.add_argument("--output", required=True)
    parser.add_argument("--words", action="store_true")
    args = parser.parse_args()

    from videocaptioner.core.asr.openvino_asr import OpenVINOASR

    asr = OpenVINOASR(
        args.audio,
        whisper_model=args.model,
        model_dir=args.model_dir,
        language=args.language,
        device=args.device,
        need_word_time_stamp=args.words,
        precision=args.precision,
        precision_hint=args.precision_hint,
        gpu_name=args.gpu_name,
    )
    data = asr.run(callback=emit_progress)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_srt(save_path=str(output_path))
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    # OpenVINO/TBB teardown can hang while unloading DLLs at interpreter
    # shutdown, leaving the parent waiting for EOF forever. The result file is
    # already closed here, so leave via os._exit() after flushing the protocol.
    exit_code = 0
    try:
        exit_code = main()
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        message = str(e).replace("\n", " ").replace("\r", " ")
        print(f"ERROR {message}", flush=True)
        exit_code = 1
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(exit_code)
