# coding:utf-8
"""Intel Arc / iGPU / NPU speech recognition via OpenVINO GenAI.

CTranslate2 (used by faster-whisper) only supports NVIDIA CUDA and cannot run
on Intel GPUs or NPUs. This backend uses Intel's official OpenVINO GenAI Whisper
pipeline, supporting:
- Intel Arc A-series (Alchemist) & B-series (Battlemage) discrete GPUs
- Intel Core Ultra integrated Arc GPUs (e.g. Core Ultra 7 155H 8-Xe iGPU)
- Intel AI Boost NPUs (e.g. Core Ultra 7 155H Meteor Lake NPU 3720, Lunar Lake,
  Arrow Lake, Panther Lake)

NPU-specific engineering workarounds included here:
1. Weight format: NPU automatically uses INT8 IR models (``whisper-*-int8-ov``),
   halving memory footprint and matching the NPU's native INT8 MAC array.
2. Timestamp bug (#4806): ``WhisperPipeline.generate(..., return_timestamps=True)``
   on NPU returns empty/corrupted text in OpenVINO 2026.x. On NPU we segment
   16 kHz audio at natural speech pauses (< 14s, well within NPU's 30s static
   encoder window) and run ``return_timestamps=False`` per utterance.
3. State-reset bug (openvino#37937): in OpenVINO <= 2026.4, ``reset_state()``
   on NPU does not sync ``num_stored_tokens`` for 1-token language-detection
   prompts on call #2+. When language is auto-detect, we detect on utterance #1
   (when ``num_stored_tokens == 0``) and pin that language token for #2..N so
   every call runs a full multi-token prefill.
4. Level0 OOM retry & teardown (#1460): retries with
   ``DISABLE_OPENVINO_GENAI_NPU_L0=1`` on NPU init errors, falls back to Intel
   GPU if needed, and explicitly destroys the pipeline before worker exit.

The pipeline itself is executed in a separate process (see
``openvino_worker.py`` / ``openvino_process.py``) because loading OpenVINO and
PyQt5 in the same process crashes with 0xc0000005 (conflicting MSVC runtimes).
"""

from __future__ import annotations

import gc
import os
import re
import wave
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

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

# Whisper internally works on 30s windows; on GPU/CPU we aggregate 4 minutes per
# generate call to keep progress reporting responsive without excessive memory.
CHUNK_SECONDS = 240

# On NPU, encoder input is statically reshaped to a single 30s window and
# return_timestamps=False is required (#4806), so utterances are capped at 14s.
NPU_MAX_SEGMENT_SEC = 14.0
NPU_MIN_SILENCE_SEC = 0.30


def _is_npu_device(device: str) -> bool:
    return (device or "").strip().upper().startswith("NPU")


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


def segment_speech_for_npu(
    audio: "np.ndarray",
    sample_rate: int = 16000,
    max_segment_sec: float = NPU_MAX_SEGMENT_SEC,
    min_silence_sec: float = NPU_MIN_SILENCE_SEC,
) -> List[Tuple[int, int]]:
    """Split 16 kHz audio into pause-aligned speech regions ``[(start, end)]``.

    Every returned span is strictly shorter than Whisper's 30s NPU static
    encoder window (capped around ``max_segment_sec``), non-overlapping, and
    aligned to natural speech pauses so accurate SRT timestamps are preserved
    even with ``return_timestamps=False`` on NPU.
    """
    import numpy as np

    total = int(len(audio))
    if total == 0:
        return []

    frame_ms = 20
    frame_len = sample_rate * frame_ms // 1000  # 320 samples @ 16 kHz
    n_frames = (total + frame_len - 1) // frame_len
    padded = np.zeros(n_frames * frame_len, dtype=np.float32)
    padded[:total] = audio

    frames = padded.reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    if n_frames >= 3:
        kernel = np.ones(3, dtype=np.float32) / 3.0
        smooth = np.convolve(rms, kernel, mode="same")
    else:
        smooth = rms

    p20 = float(np.percentile(smooth, 20))
    p95 = float(np.percentile(smooth, 95))
    if p95 < 1e-4:
        return []

    threshold = max(0.006, min(0.035, p20 * 2.5 + (p95 - p20) * 0.08))
    active = smooth >= threshold

    # Bridge short pauses inside a phrase (< min_silence_sec).
    min_silence_frames = max(1, int(round(min_silence_sec * 1000 / frame_ms)))
    idx = 0
    while idx < n_frames:
        if active[idx]:
            idx += 1
            continue
        gap_start = idx
        while idx < n_frames and not active[idx]:
            idx += 1
        if gap_start > 0 and idx < n_frames and (idx - gap_start) < min_silence_frames:
            active[gap_start:idx] = True

    # Collect contiguous speech runs and drop tiny noise blips (< 200ms).
    min_speech_frames = max(1, int(round(200 / frame_ms)))
    raw_runs: List[Tuple[int, int]] = []
    idx = 0
    while idx < n_frames:
        if not active[idx]:
            idx += 1
            continue
        run_start = idx
        while idx < n_frames and active[idx]:
            idx += 1
        if (idx - run_start) >= min_speech_frames:
            raw_runs.append((run_start, idx))

    max_frames = max(20, int(round(max_segment_sec * 1000 / frame_ms)))
    if not raw_runs:
        # Constant-level audio fallback: fixed windows <= max_segment_sec.
        step = max_frames * frame_len
        return [(s, min(total, s + step)) for s in range(0, total, step)]

    # Split long continuous monologues at the quietest local dip in [50%, 100%] of max_frames.
    bounded_runs: List[Tuple[int, int]] = []
    half_frames = max(10, max_frames // 2)
    for run_start, run_end in raw_runs:
        cur = run_start
        while (run_end - cur) > max_frames:
            win_lo = cur + half_frames
            win_hi = cur + max_frames
            split_at = win_lo + int(np.argmin(smooth[win_lo:win_hi]))
            if split_at <= cur:
                split_at = cur + max_frames
            bounded_runs.append((cur, split_at))
            cur = split_at
        if cur < run_end:
            bounded_runs.append((cur, run_end))

    # Merge very short adjacent fragments (< 1.2s) if separated by < 0.5s and combined <= 10s.
    short_frames = int(round(1200 / frame_ms))
    merge_gap_frames = int(round(500 / frame_ms))
    merge_cap_frames = int(round(10000 / frame_ms))
    merged_runs: List[Tuple[int, int]] = []
    for run_start, run_end in bounded_runs:
        if merged_runs:
            prev_start, prev_end = merged_runs[-1]
            if (
                (prev_end - prev_start) < short_frames
                and (run_start - prev_end) <= merge_gap_frames
                and (run_end - prev_start) <= merge_cap_frames
            ):
                merged_runs[-1] = (prev_start, run_end)
                continue
        merged_runs.append((run_start, run_end))

    # Convert frames to samples with gentle pre/post padding, never overlapping neighbors.
    pad_pre = int(0.12 * sample_rate)
    pad_post = int(0.16 * sample_rate)
    raw_samples = [(f0 * frame_len, min(total, f1 * frame_len)) for f0, f1 in merged_runs]
    spans: List[Tuple[int, int]] = []
    for i, (s0, s1) in enumerate(raw_samples):
        lo_bound = 0 if i == 0 else (raw_samples[i - 1][1] + s0) // 2
        hi_bound = total if i == len(raw_samples) - 1 else (s1 + raw_samples[i + 1][0]) // 2
        padded_s0 = max(lo_bound, s0 - pad_pre)
        padded_s1 = min(hi_bound, s1 + pad_post)
        if padded_s1 > padded_s0:
            spans.append((padded_s0, padded_s1))
    return spans


def split_clauses_proportionally(
    text: str,
    start_ms: int,
    end_ms: int,
    max_clause_ms: int = 6000,
) -> List[ASRDataSeg]:
    """Split a longer multi-sentence NPU utterance proportionally across clauses."""
    cleaned = _clean_text(text)
    if not cleaned:
        return []

    duration = max(200, end_ms - start_ms)
    if duration <= max_clause_ms:
        return [ASRDataSeg(cleaned, start_ms, start_ms + duration)]

    parts = [p.strip() for p in re.split(r"(?<=[。！？；.!?;])\s*", cleaned) if p.strip()]
    if len(parts) <= 1 and duration > 9000:
        comma_parts = [p.strip() for p in re.split(r"(?<=[，,、])\s*", cleaned) if p.strip()]
        if len(comma_parts) > 1:
            parts = []
            buf = ""
            for piece in comma_parts:
                buf = f"{buf} {piece}".strip() if buf else piece
                if len(buf) >= 18:
                    parts.append(buf)
                    buf = ""
            if buf:
                if parts:
                    parts[-1] = f"{parts[-1]} {buf}".strip()
                else:
                    parts.append(buf)

    if len(parts) <= 1:
        return [ASRDataSeg(cleaned, start_ms, start_ms + duration)]

    weights = [max(1, len(p)) for p in parts]
    total_weight = sum(weights)
    segments: List[ASRDataSeg] = []
    cursor = start_ms
    for idx, (part, weight) in enumerate(zip(parts, weights)):
        if idx == len(parts) - 1:
            seg_end = start_ms + duration
        else:
            seg_end = cursor + max(200, int(round(duration * weight / total_weight)))
            seg_end = min(seg_end, start_ms + duration - 100 * (len(parts) - idx - 1))
        if seg_end <= cursor:
            seg_end = cursor + 200
        segments.append(ASRDataSeg(part, cursor, seg_end))
        cursor = seg_end
    return segments


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
        self.used_precision = "fp16"
        self._word_ts_supported = True
        self._npu_detected_language = ""
        self._callback: Optional[Callable[[int, str], None]] = None

    # ------------------------------------------------------------------
    def _build_config(self) -> dict:
        config = {"task": "transcribe", "return_timestamps": True}
        if self.language:
            config["language"] = f"<|{self.language}|>"
        if self.need_word_time_stamp:
            config["word_timestamps"] = True
        return config

    @staticmethod
    def _construct_pipeline(ov_genai, model_path: Path, device: str, properties: dict):
        try:
            return ov_genai.WhisperPipeline(str(model_path), device, **properties)
        except TypeError:
            return ov_genai.WhisperPipeline(str(model_path), device)

    def _create_pipeline(self):
        import openvino_genai as ov_genai

        requested_npu = (self.device or "").lower().startswith(("intel_npu", "npu"))
        device = resolve_openvino_device(self.device)
        if requested_npu and not _is_npu_device(device):
            logger.warning(
                "Intel NPU was requested (%s) but is unavailable in OpenVINO; "
                "using %s instead (install Intel NPU driver >= 32.0.100.3104 for NPU)",
                self.device,
                device,
            )

        precision = "int8" if _is_npu_device(device) else "fp16"
        model_path = ensure_openvino_whisper_model(
            self.model_name,
            self.model_dir,
            progress=self._download_progress,
            precision=precision,
        )
        # Persist compiled kernels/blobs so later runs start quickly.
        cache_dir = Path(self.model_dir) / "openvino" / ".cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        properties = {"CACHE_DIR": str(cache_dir)}

        logger.info(
            "Loading OpenVINO model: %s (device: %s, precision: %s)",
            model_path,
            device,
            precision,
        )
        try:
            pipeline = self._construct_pipeline(ov_genai, model_path, device, properties)
        except Exception as first_err:
            if _is_npu_device(device):
                # Official OpenVINO NPU troubleshooting workaround for Level0 OOM
                # on Meteor Lake / Core Ultra 155H drivers:
                if os.environ.get("DISABLE_OPENVINO_GENAI_NPU_L0") != "1":
                    logger.warning(
                        "NPU pipeline init failed (%s); retrying with "
                        "DISABLE_OPENVINO_GENAI_NPU_L0=1",
                        first_err,
                    )
                    os.environ["DISABLE_OPENVINO_GENAI_NPU_L0"] = "1"
                    try:
                        pipeline = self._construct_pipeline(
                            ov_genai, model_path, device, properties
                        )
                    except Exception as second_err:
                        first_err = second_err
                    else:
                        self.pipeline = pipeline
                        self.used_device = device
                        self.used_precision = precision
                        return pipeline

                fallback = resolve_openvino_device("intel_gpu")
                logger.warning(
                    "NPU failed (%s), falling back to %s", first_err, fallback
                )
                device = fallback
                pipeline = self._construct_pipeline(
                    ov_genai, model_path, device, properties
                )
            else:
                raise

        self.pipeline = pipeline
        self.used_device = device
        self.used_precision = precision
        return pipeline

    def _download_progress(self, percent: int, message: str):
        callback = getattr(self, "_callback", None)
        if callback:
            callback(2 + int(percent * 0.16), message)

    # ------------------------------------------------------------------
    def _transcribe_npu(
        self, audio: "np.ndarray", callback: Callable[[int, str], None]
    ) -> List[ASRDataSeg]:
        """Transcribe on Intel AI Boost NPU with pause segmentation (#4806, #37937)."""
        spans = segment_speech_for_npu(audio, sample_rate=16000)
        if not spans:
            return []

        total_spans = len(spans)
        logger.info(
            "NPU speech segmentation produced %d utterance(s) across %.1fs audio",
            total_spans,
            len(audio) / 16000.0,
        )
        all_segments: List[ASRDataSeg] = []

        for idx, (s0, s1) in enumerate(spans):
            clip = audio[s0:s1]
            start_ms = int(round(s0 * 1000.0 / 16000.0))
            end_ms = int(round(s1 * 1000.0 / 16000.0))

            gen_kwargs = {
                "task": "transcribe",
                # Workaround for openvino.genai#4806: return_timestamps=True
                # corrupts/empties decoded text on NPU.
                "return_timestamps": False,
            }
            if self.language:
                gen_kwargs["language"] = f"<|{self.language}|>"
            elif self._npu_detected_language:
                # Workaround for openvino#37937: pin language detected on the
                # first utterance (when num_stored_tokens == 0) so call #2+
                # always runs multi-token prefill instead of 1-token detect.
                gen_kwargs["language"] = self._npu_detected_language

            result = self.pipeline.generate(clip.tolist(), **gen_kwargs)

            if not self.language and not self._npu_detected_language:
                detected = (getattr(result, "language", "") or "").strip()
                if detected:
                    lang_code = detected.strip("<|>")
                    if lang_code:
                        self._npu_detected_language = f"<|{lang_code}|>"
                        logger.info(
                            "NPU auto-detected language on first utterance: %s",
                            self._npu_detected_language,
                        )

            texts = getattr(result, "texts", None) or []
            text = _clean_text(texts[0]) if texts else ""
            if text:
                all_segments.extend(
                    split_clauses_proportionally(text, start_ms, end_ms)
                )

            progress = 20 + int(78 * (idx + 1) / total_spans)
            callback(progress, f"NPU transcribing {idx + 1}/{total_spans}")

        return all_segments

    # ------------------------------------------------------------------
    def _transcribe_chunk(
        self, chunk: "np.ndarray", offset: float, config: dict
    ) -> List[ASRDataSeg]:
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

        try:
            callback(1, "initializing OpenVINO")
            self._create_pipeline()
            callback(20, f"transcribing on {self.used_device} ({self.used_precision})")

            audio = _read_wav_mono_16k(self.audio_input)
            total_samples = len(audio)
            if total_samples == 0:
                raise RuntimeError("Empty audio input")

            if _is_npu_device(self.used_device):
                all_segments = self._transcribe_npu(audio, callback)
            else:
                chunk_samples = CHUNK_SECONDS * 16000
                chunk_count = max(1, (total_samples + chunk_samples - 1) // chunk_samples)
                config = self._build_config()

                all_segments = []
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
            srt_text = ASRData(all_segments).to_srt(
                layout=SubtitleLayoutEnum.ORIGINAL_ON_TOP
            )
            callback(100, "completed")
            return srt_text
        finally:
            # Workaround for openvino.genai#1460: destroy NPU pipeline explicitly
            # before worker process teardown so Level0 cleans up cleanly.
            self.pipeline = None
            gc.collect()

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
