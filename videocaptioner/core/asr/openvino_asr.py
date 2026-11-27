# coding:utf-8
"""Intel Arc / iGPU / NPU speech recognition via OpenVINO GenAI.

CTranslate2 (used by faster-whisper) only supports NVIDIA CUDA and cannot run
on Intel GPUs. This backend uses Intel's official OpenVINO GenAI Whisper
pipeline, which runs on Intel Arc A-series (Alchemist), Arc B-series
(Battlemage), integrated GPUs and NPUs.

The pipeline itself is executed in a separate process (see
``openvino_worker.py`` / ``openvino_process.py``) because loading OpenVINO and
PyQt5 in the same process crashes with 0xc0000005 (conflicting MSVC runtimes).
"""

from __future__ import annotations

import re
import wave
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Optional

if TYPE_CHECKING:
    import numpy as np

from ...config import MODEL_PATH
from ..entities import SubtitleLayoutEnum
from ..utils.hardware import resolve_openvino_device
from ..utils.logger import setup_logger
from ..utils.model_downloader import (
    OPENVINO_WHISPER_REPOS,
    ensure_openvino_whisper_model,
)
from .asr_data import ASRData, ASRDataSeg
from .base import BaseASR

logger = setup_logger("openvino_asr")

# Whisper internally works on 30s windows; aggregate a few minutes per generate
# call to keep progress reporting responsive without excessive memory usage.
CHUNK_SECONDS = 240


def _read_wav_mono_16k(path: str) -> "np.ndarray":
    """Read a 16 kHz mono WAV file as float32 in [-1, 1]."""
    import numpy as np

    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.getnframes()
        raw = wav.readframes(frames)

    if sample_width == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sample_width == 1:
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise RuntimeError(f"Unsupported WAV sample width: {sample_width * 8} bit")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if sample_rate != 16000:
        raise RuntimeError(f"Audio must be 16 kHz, got {sample_rate} Hz")
    return audio


def _clean_text(text: str) -> str:
    text = re.sub(r"<\|[^|]*\|>", "", text or "")
    return text.strip()


class OpenVINOASR(BaseASR):
    """OpenVINO GenAI Whisper implementation (runs in the worker process)."""

    def __init__(
        self,
        audio_input,
        whisper_model: str = "large-v3-turbo",
        model_dir: Optional[str] = None,
        language: str = "en",
        device: str = "intel_gpu",
        use_cache: bool = False,
        need_word_time_stamp: bool = False,
    ):
        super().__init__(audio_input, use_cache, need_word_time_stamp)
        if whisper_model not in OPENVINO_WHISPER_REPOS:
            raise RuntimeError(f"Unsupported OpenVINO model: {whisper_model}")
        self.model_name = whisper_model
        self.model_dir = Path(model_dir) if model_dir else Path(MODEL_PATH)
        self.language = language
        self.device = device
        self.need_word_time_stamp = need_word_time_stamp
        self.pipeline = None
        self.used_device = ""
        self._word_ts_supported = True
        self._callback: Optional[Callable[[int, str], None]] = None

    # ------------------------------------------------------------------
    def _build_config(self) -> dict:
        config = {"task": "transcribe", "return_timestamps": True}
        if self.language:
            config["language"] = f"<|{self.language}|>"
        if self.need_word_time_stamp:
            config["word_timestamps"] = True
        return config

    def _create_pipeline(self):
        import openvino_genai as ov_genai

        device = resolve_openvino_device(self.device)
        model_path = ensure_openvino_whisper_model(
            self.model_name, self.model_dir, progress=self._download_progress
        )
        # Persist compiled kernels so later runs start quickly.
        cache_dir = Path(self.model_dir) / "openvino" / ".cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        properties = {"CACHE_DIR": str(cache_dir)}

        logger.info("Loading OpenVINO model: %s (device: %s)", model_path, device)
        try:
            pipeline = ov_genai.WhisperPipeline(str(model_path), device, **properties)
        except TypeError:
            pipeline = ov_genai.WhisperPipeline(str(model_path), device)
        except Exception as e:
            if device.upper().startswith("NPU"):
                fallback = resolve_openvino_device("intel_gpu")
                logger.warning("NPU failed (%s), falling back to %s", e, fallback)
                device = fallback
                try:
                    pipeline = ov_genai.WhisperPipeline(
                        str(model_path), device, **properties
                    )
                except TypeError:
                    pipeline = ov_genai.WhisperPipeline(str(model_path), device)
            else:
                raise
        self.pipeline = pipeline
        self.used_device = device
        return pipeline

    def _download_progress(self, percent: int, message: str):
        callback = getattr(self, "_callback", None)
        if callback:
            callback(2 + int(percent * 0.16), message)

    # ------------------------------------------------------------------
    def _transcribe_chunk(self, chunk: "np.ndarray", offset: float, config: dict) -> List[ASRDataSeg]:
        try:
            result = self.pipeline.generate(chunk.tolist(), **config)
        except Exception as e:
            if config.get("word_timestamps") and self._word_ts_supported:
                # Intel GPU plugin does not support encoder attention decomposition
                # required by word_timestamps; fall back to segment timestamps.
                logger.warning(
                    "Word-level timestamps are not supported on this device, "
                    "falling back to segment timestamps: %s",
                    e,
                )
                self._word_ts_supported = False
                config.pop("word_timestamps", None)
                result = self.pipeline.generate(chunk.tolist(), **config)
            else:
                raise

        segments: List[ASRDataSeg] = []
        words = getattr(result, "words", None)
        chunks = getattr(result, "chunks", None)

        if self.need_word_time_stamp and words:
            for word in words:
                text = _clean_text(word.word)
                if not text:
                    continue
                start = int((offset + word.start_ts) * 1000)
                end = int((offset + word.end_ts) * 1000)
                if end <= start:
                    end = start + 80
                segments.append(ASRDataSeg(text, start, end))
            if segments:
                return segments

        if chunks:
            limit_ms = int((offset + len(chunk) / 16000) * 1000)
            for item in chunks:
                text = _clean_text(item.text)
                if not text:
                    continue
                start = max(int(offset * 1000), int((offset + item.start_ts) * 1000))
                end = min(limit_ms, int((offset + item.end_ts) * 1000))
                if end <= start:
                    end = start + 200
                segments.append(ASRDataSeg(text, start, end))
            if segments:
                return segments

        texts = getattr(result, "texts", None) or []
        text = _clean_text(texts[0]) if texts else ""
        if text:
            start = int(offset * 1000)
            end = int((offset + len(chunk) / 16000) * 1000)
            segments.append(ASRDataSeg(text, start, end))
        return segments

    def _run(self, callback=None, **kwargs) -> str:
        if callback is None:

            def callback(x, y):
                return None

        self._callback = callback

        try:
            import openvino_genai  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "OpenVINO GenAI is not installed. Install the Intel extras:\n"
                '  pip install -e ".[intel]"\n'
                f"Details: {e}"
            )

        callback(1, "initializing OpenVINO")
        self._create_pipeline()
        callback(20, f"transcribing on {self.used_device}")

        audio = _read_wav_mono_16k(self.audio_input)
        total_samples = len(audio)
        if total_samples == 0:
            raise RuntimeError("Empty audio input")

        chunk_samples = CHUNK_SECONDS * 16000
        chunk_count = max(1, (total_samples + chunk_samples - 1) // chunk_samples)
        config = self._build_config()

        all_segments: List[ASRDataSeg] = []
        for index in range(chunk_count):
            start_sample = index * chunk_samples
            end_sample = min(total_samples, start_sample + chunk_samples)
            chunk = audio[start_sample:end_sample]
            offset = start_sample / 16000.0

            logger.info(
                "Transcribing chunk %d/%d (%.1fs - %.1fs)",
                index + 1,
                chunk_count,
                offset,
                end_sample / 16000.0,
            )
            all_segments.extend(self._transcribe_chunk(chunk, offset, config))

            progress = 20 + int(78 * (index + 1) / chunk_count)
            callback(progress, f"transcribing {index + 1}/{chunk_count}")

        if not all_segments:
            raise RuntimeError("No speech recognized; check the audio or the model")

        all_segments.sort(key=lambda seg: seg.start_time)
        srt_text = ASRData(all_segments).to_srt(layout=SubtitleLayoutEnum.ORIGINAL_ON_TOP)
        callback(100, "completed")
        return srt_text

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
            f"{self.crc32_hex}-openvino-{self.model_name}-{self.device}"
            f"-{self.language}-{int(self.need_word_time_stamp)}"
        )
