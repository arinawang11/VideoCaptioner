# coding:utf-8
"""Run OpenVINO transcription in an isolated child process.

The UI process must never import OpenVINO: OpenVINO 2026 needs a newer MSVC
runtime than the one bundled with PyQt5 and mixing them crashes the GUI with
0xc0000005. Isolating the backend also keeps the UI responsive during long
transcriptions.

If the Intel GPU driver/IGC compiler crashes the worker natively (for example
BEX64 / 0xc0000409 in ucrtbase.dll on engineering-sample machines), the parent
retries once on the OpenVINO CPU plugin with the int8 model.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Union

from ...config import MODEL_PATH, ROOT_PATH
from ..utils.logger import setup_logger
from .asr_data import ASRData, ASRDataSeg
from .base import BaseASR

logger = setup_logger("openvino_process")

WORKER_SCRIPT = Path(__file__).with_name("openvino_worker.py")


def _find_python() -> str:
    """Locate the Python interpreter that has the OpenVINO packages installed."""
    if sys.executable.lower().endswith(("python.exe", "pythonw.exe", "python")):
        return sys.executable

    candidates = [
        Path(ROOT_PATH) / "runtime" / "python.exe",
        Path(sys.prefix) / "python.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    found = shutil.which("python")
    if found:
        return found
    raise RuntimeError("Cannot find a Python interpreter for the OpenVINO worker")


class OpenVINOProcessASR(BaseASR):
    """OpenVINO Whisper backend, executed in a subprocess."""

    def __init__(
        self,
        audio_input: Union[str, bytes],
        whisper_model: str = "large-v3-turbo",
        model_dir: Optional[str] = None,
        language: str = "en",
        device: str = "intel_gpu",
        use_cache: bool = False,
        need_word_time_stamp: bool = False,
    ):
        super().__init__(audio_input, use_cache, need_word_time_stamp)
        self.model_name = whisper_model
        self.model_dir = Path(model_dir) if model_dir else Path(MODEL_PATH)
        self.language = language
        self.device = device
        self.need_word_time_stamp = need_word_time_stamp

    def _build_command(
        self,
        audio_path: Path,
        output_path: Path,
        device: Optional[str] = None,
        precision: Optional[str] = None,
        precision_hint: Optional[str] = None,
    ) -> List[str]:
        cmd = [
            _find_python(),
            str(WORKER_SCRIPT),
            "--audio",
            str(audio_path),
            "--model",
            str(self.model_name),
            "--model-dir",
            str(self.model_dir),
            "--device",
            str(device or self.device),
            "--language",
            str(self.language or ""),
            "--output",
            str(output_path),
        ]
        if precision:
            cmd.extend(["--precision", str(precision)])
        if precision_hint:
            cmd.extend(["--precision-hint", str(precision_hint)])
        if self.need_word_time_stamp:
            cmd.append("--words")
        return cmd

    def _run_worker(
        self, cmd: List[str], output_path: Path, callback
    ) -> Tuple[bool, str]:
        logger.debug("OpenVINO command: %s", " ".join(cmd))

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        # Avoid locale-dependent .pth decoding when the install path has
        # non-ASCII characters (e.g. a Chinese user profile folder).
        env["PYTHONUTF8"] = "1"
        # Do not let the child inherit PyQt5's Qt folder (old MSVCP140.dll).
        env["PATH"] = os.pathsep.join(
            part
            for part in env.get("PATH", "").split(os.pathsep)
            if part and "PyQt5" not in part and "Qt5" not in part
        )

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
            env=env,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )

        error_messages: List[str] = []
        recent_lines: List[str] = []
        assert process.stdout is not None
        while True:
            line = process.stdout.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            if line.startswith("PROGRESS "):
                try:
                    value, message = line[len("PROGRESS ") :].split("|", 1)
                    callback(int(float(value)), message)
                except Exception:
                    pass
            elif line.startswith("ERROR "):
                error_messages.append(line[len("ERROR ") :])
            elif line == "DONE":
                callback(100, "completed")
            else:
                recent_lines.append(line)
                if len(recent_lines) > 40:
                    recent_lines.pop(0)

        process.wait()
        if process.returncode == 0 and output_path.exists():
            return True, ""

        detail = "\n".join(error_messages) or "\n".join(recent_lines[-15:])
        if not detail:
            detail = (
                f"OpenVINO worker exited unexpectedly "
                f"(return code {process.returncode})"
            )
        return False, detail

    def _run(self, callback=None, **kwargs) -> str:
        if callback is None:

            def callback(x, y):
                return None

        if not isinstance(self.audio_input, str):
            raise RuntimeError(
                "OpenVINO backend requires an audio file path, not raw bytes"
            )

        temp_dir = Path(tempfile.gettempdir()) / "videocaptioner_asr"
        temp_dir.mkdir(parents=True, exist_ok=True)

        attempts: List[Tuple[str, Optional[str], Optional[str]]] = [
            (self.device, None, None)
        ]
        device_lower = str(self.device).lower()
        if device_lower.startswith(("intel_gpu", "gpu")):
            # IGC may abort natively while compiling fp16 kernels on some
            # integrated GPUs; f32 kernels are stable there.
            attempts.append((self.device, None, "f32"))
            attempts.append(("cpu", "int8", None))
        elif device_lower.startswith(("intel_npu", "npu")):
            attempts.append(("cpu", "int8", None))

        with tempfile.TemporaryDirectory(dir=temp_dir) as temp_path:
            output_path = Path(temp_path) / "openvino_output.srt"
            last_detail = ""

            for index, (device, precision, precision_hint) in enumerate(attempts):
                if output_path.exists():
                    output_path.unlink()
                cmd = self._build_command(
                    Path(self.audio_input),
                    output_path,
                    device,
                    precision,
                    precision_hint,
                )
                if index == 0:
                    callback(1, "starting OpenVINO worker")
                else:
                    label = f"{device} ({precision_hint or precision or 'auto'})"
                    callback(21, f"retrying on {label}")
                logger.info(
                    "OpenVINO worker attempt %d/%d (device=%s, precision=%s, hint=%s)",
                    index + 1,
                    len(attempts),
                    device,
                    precision or "auto",
                    precision_hint or "auto",
                )

                ok, detail = self._run_worker(cmd, output_path, callback)
                if ok:
                    logger.debug("OpenVINO transcription finished")
                    return output_path.read_text(encoding="utf-8")

                last_detail = detail
                logger.error(
                    "OpenVINO worker attempt %d/%d failed: %s",
                    index + 1,
                    len(attempts),
                    detail,
                )

            raise RuntimeError(last_detail or "OpenVINO transcription failed")

    def _make_segments(self, resp_data: str) -> List[ASRDataSeg]:
        asr_data = ASRData.from_srt(resp_data)
        filtered = []
        for seg in asr_data.segments:
            text = seg.text.strip()
            if not text:
                continue
            if text.startswith(("【", "[", "(", "（")):
                continue
            filtered.append(seg)
        return filtered

    def _get_key(self):
        return (
            f"{self.crc32_hex}-openvino-process-{self.model_name}-{self.device}"
            f"-{self.language}-{int(self.need_word_time_stamp)}"
        )
