# coding:utf-8
"""Video codec and encoder selection for FFmpeg video synthesis.

Supports H.264, H.265 (HEVC), AV1 and VP9 across:
- Intel QSV hardware encoders (``h264_qsv``, ``hevc_qsv``, ``av1_qsv``, ``vp9_qsv``)
- SVT-AV1 (``libsvtav1``)
- CPU software encoders (``libx264``, ``libx265``, ``libaom-av1``, ``libvpx-vp9``)

Each encoder has different FFmpeg quality/preset flags (``-crf`` vs
``-global_quality``, ``-preset`` vs ``-cpu-used``, ``-b:v 0``, ``-low_power 1``,
``-tag:v hvc1``). Centralising resolution and flag generation here keeps
``video_utils.py``, ``ass_renderer.py`` and ``rounded_renderer.py`` consistent.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Set, Union

from ..entities import VideoCodecEnum, VideoEncoderEnum
from .logger import setup_logger

logger = setup_logger("video_encoder")

_CREATIONFLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0

_QSV_BY_CODEC = {
    VideoCodecEnum.H264: "h264_qsv",
    VideoCodecEnum.H265: "hevc_qsv",
    VideoCodecEnum.AV1: "av1_qsv",
    VideoCodecEnum.VP9: "vp9_qsv",
}

_SOFTWARE_BY_CODEC = {
    VideoCodecEnum.H264: "libx264",
    VideoCodecEnum.H265: "libx265",
    VideoCodecEnum.AV1: "libaom-av1",
    VideoCodecEnum.VP9: "libvpx-vp9",
}

# Friendly labels shown in the UI status bar and logs.
_ENCODER_DISPLAY = {
    "libx264": "软件 H.264 (libx264)",
    "libx265": "软件 H.265 (libx265)",
    "libsvtav1": "SVT-AV1 (libsvtav1)",
    "libaom-av1": "软件 AV1 (libaom-av1)",
    "libvpx-vp9": "软件 VP9 (libvpx-vp9)",
    "h264_qsv": "Intel QSV H.264 (h264_qsv)",
    "hevc_qsv": "Intel QSV H.265 (hevc_qsv)",
    "av1_qsv": "Intel QSV AV1 (av1_qsv)",
    "vp9_qsv": "Intel QSV VP9 (vp9_qsv)",
}

_ENCODER_LINE_RE = re.compile(r"^\s*V[A-Z\.]{5}\s+([A-Za-z0-9_\-]+)\s+")


@dataclass(frozen=True)
class ResolvedVideoEncoder:
    """Concrete FFmpeg video encoder selected for a synthesis task."""

    vcodec: str
    codec: VideoCodecEnum
    requested_encoder: VideoEncoderEnum
    fallback_reason: str = ""

    @property
    def is_hardware(self) -> bool:
        return self.vcodec.endswith(("_qsv", "_nvenc", "_amf", "_videotoolbox"))

    @property
    def display_name(self) -> str:
        return _ENCODER_DISPLAY.get(self.vcodec, self.vcodec)


def coerce_video_codec(
    value: Optional[Union[VideoCodecEnum, str]],
) -> VideoCodecEnum:
    if isinstance(value, VideoCodecEnum):
        return value
    raw = str(value or "").strip().lower()
    mapping = {
        "h264": VideoCodecEnum.H264,
        "h.264": VideoCodecEnum.H264,
        "avc": VideoCodecEnum.H264,
        "libx264": VideoCodecEnum.H264,
        "h265": VideoCodecEnum.H265,
        "h.265": VideoCodecEnum.H265,
        "h.265 (hevc)": VideoCodecEnum.H265,
        "hevc": VideoCodecEnum.H265,
        "libx265": VideoCodecEnum.H265,
        "av1": VideoCodecEnum.AV1,
        "libsvtav1": VideoCodecEnum.AV1,
        "libaom-av1": VideoCodecEnum.AV1,
        "vp9": VideoCodecEnum.VP9,
        "libvpx-vp9": VideoCodecEnum.VP9,
    }
    return mapping.get(raw, VideoCodecEnum.H264)


def coerce_video_encoder(
    value: Optional[Union[VideoEncoderEnum, str]],
) -> VideoEncoderEnum:
    if isinstance(value, VideoEncoderEnum):
        return value
    raw = str(value or "").strip().lower()
    mapping = {
        "auto": VideoEncoderEnum.AUTO,
        "自动": VideoEncoderEnum.AUTO,
        "qsv": VideoEncoderEnum.QSV,
        "intel qsv": VideoEncoderEnum.QSV,
        "intel_qsv": VideoEncoderEnum.QSV,
        "svt": VideoEncoderEnum.SVT,
        "svt-av1": VideoEncoderEnum.SVT,
        "libsvtav1": VideoEncoderEnum.SVT,
        "software": VideoEncoderEnum.SOFTWARE,
        "cpu": VideoEncoderEnum.SOFTWARE,
        "软件": VideoEncoderEnum.SOFTWARE,
        "软件编码": VideoEncoderEnum.SOFTWARE,
    }
    return mapping.get(raw, VideoEncoderEnum.AUTO)


@lru_cache(maxsize=1)
def list_ffmpeg_encoders() -> Set[str]:
    """Return the set of video encoder names compiled into ``ffmpeg``."""
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            creationflags=_CREATIONFLAGS,
        )
    except Exception as exc:
        logger.debug("Failed to query ffmpeg encoders: %s", exc)
        return set()

    encoders: Set[str] = set()
    for line in (proc.stdout or "").splitlines():
        match = _ENCODER_LINE_RE.match(line)
        if match:
            encoders.add(match.group(1))
    return encoders


@lru_cache(maxsize=16)
def probe_encoder_runtime(vcodec: str) -> bool:
    """Verify that a hardware encoder actually initialises on this machine."""
    encoders = list_ffmpeg_encoders()
    if encoders and vcodec not in encoders:
        return False

    # Software encoders compiled into ffmpeg do not need a hardware probe.
    if not vcodec.endswith(("_qsv", "_nvenc", "_amf")):
        return True

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=128x128:r=30:d=0.1",
        "-frames:v",
        "1",
        "-c:v",
        vcodec,
    ]
    if vcodec == "vp9_qsv":
        cmd.extend(["-low_power", "1"])
    cmd.extend(["-pix_fmt", "nv12", "-f", "null", "-"])

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            creationflags=_CREATIONFLAGS,
        )
        ok = proc.returncode == 0
        if not ok:
            logger.debug("Encoder probe failed for %s: %s", vcodec, proc.stderr.strip()[:200])
        return ok
    except Exception as exc:
        logger.debug("Encoder probe error for %s: %s", vcodec, exc)
        return False


def _pick_software_encoder(codec: VideoCodecEnum, available: Set[str]) -> str:
    if codec == VideoCodecEnum.AV1:
        # Prefer SVT-AV1 over libaom-av1 when available for software AV1 too?
        # Keep explicit separation: software AV1 picks libaom-av1 if present,
        # falling back to libsvtav1 when libaom is missing.
        if not available or "libaom-av1" in available:
            return "libaom-av1"
        if "libsvtav1" in available:
            return "libsvtav1"
        return "libaom-av1"
    return _SOFTWARE_BY_CODEC[codec]


def software_fallback_for(
    vcodec: str, codec: Optional[Union[VideoCodecEnum, str]] = None
) -> str:
    """Return a safe CPU software encoder for ``codec`` if hardware fails."""
    resolved_codec = coerce_video_codec(codec) if codec is not None else VideoCodecEnum.H264
    if codec is None:
        if "hevc" in vcodec or "265" in vcodec:
            resolved_codec = VideoCodecEnum.H265
        elif "av1" in vcodec:
            resolved_codec = VideoCodecEnum.AV1
        elif "vp9" in vcodec:
            resolved_codec = VideoCodecEnum.VP9
    available = list_ffmpeg_encoders()
    if resolved_codec == VideoCodecEnum.AV1 and (not available or "libsvtav1" in available):
        return "libsvtav1"
    return _pick_software_encoder(resolved_codec, available)


def resolve_video_encoder(
    video_codec: Optional[Union[VideoCodecEnum, str]] = None,
    video_encoder: Optional[Union[VideoEncoderEnum, str]] = None,
    output_path: str = "",
    *,
    available_encoders: Optional[Set[str]] = None,
    probe_hardware: bool = True,
) -> ResolvedVideoEncoder:
    """Pick the concrete FFmpeg ``-c:v`` encoder for ``(codec, encoder)``."""
    codec = coerce_video_codec(video_codec)
    encoder = coerce_video_encoder(video_encoder)
    available = available_encoders if available_encoders is not None else list_ffmpeg_encoders()

    # WebM only accepts VP8/VP9/AV1.
    if output_path and Path(output_path).suffix.lower() == ".webm":
        if codec not in (VideoCodecEnum.VP9, VideoCodecEnum.AV1):
            codec = VideoCodecEnum.VP9

    def _has(name: str) -> bool:
        return (not available) or (name in available)

    def _hw_ok(name: str) -> bool:
        if not _has(name):
            return False
        return probe_encoder_runtime(name) if probe_hardware else True

    # 1) Explicit SVT-AV1
    if encoder == VideoEncoderEnum.SVT:
        if codec == VideoCodecEnum.AV1:
            if _has("libsvtav1"):
                return ResolvedVideoEncoder("libsvtav1", codec, encoder)
            fb = _pick_software_encoder(codec, available)
            return ResolvedVideoEncoder(
                fb,
                codec,
                encoder,
                fallback_reason="当前 FFmpeg 未内置 libsvtav1，已回退到软件编码器",
            )
        # SVT only supports AV1 in upstream FFmpeg.
        fb = _pick_software_encoder(codec, available)
        return ResolvedVideoEncoder(
            fb,
            codec,
            encoder,
            fallback_reason=f"SVT-AV1 仅支持 AV1 格式，{codec.value} 已自动改用 {fb}",
        )

    # 2) Explicit Intel QSV
    if encoder == VideoEncoderEnum.QSV:
        qsv_name = _QSV_BY_CODEC[codec]
        if _hw_ok(qsv_name):
            return ResolvedVideoEncoder(qsv_name, codec, encoder)
        fb = software_fallback_for(qsv_name, codec)
        return ResolvedVideoEncoder(
            fb,
            codec,
            encoder,
            fallback_reason=f"本机不可用 {qsv_name}，已自动回退到 {fb}",
        )

    # 3) Explicit CPU software encoder
    if encoder == VideoEncoderEnum.SOFTWARE:
        sw = _pick_software_encoder(codec, available)
        return ResolvedVideoEncoder(sw, codec, encoder)

    # 4) AUTO: prefer QSV if usable -> SVT-AV1 for AV1 -> CPU software
    qsv_name = _QSV_BY_CODEC[codec]
    if _hw_ok(qsv_name):
        return ResolvedVideoEncoder(qsv_name, codec, encoder)
    if codec == VideoCodecEnum.AV1 and _has("libsvtav1"):
        return ResolvedVideoEncoder("libsvtav1", codec, encoder)
    sw = _pick_software_encoder(codec, available)
    return ResolvedVideoEncoder(sw, codec, encoder)


def _map_qsv_preset(preset: str) -> str:
    # QSV does not accept ultrafast/superfast.
    mapping = {
        "ultrafast": "veryfast",
        "superfast": "veryfast",
        "veryfast": "veryfast",
        "faster": "faster",
        "fast": "fast",
        "medium": "medium",
        "slow": "slow",
        "slower": "slower",
        "veryslow": "veryslow",
    }
    return mapping.get((preset or "medium").lower(), "medium")


def _map_svt_preset(preset: str) -> str:
    # SVT-AV1 numeric preset (0=slowest/best .. 13=fastest). 5-8 is the sweet spot.
    mapping = {
        "ultrafast": "10",
        "superfast": "9",
        "veryfast": "8",
        "faster": "8",
        "fast": "8",
        "medium": "6",
        "slow": "5",
        "slower": "4",
        "veryslow": "3",
    }
    return mapping.get((preset or "medium").lower(), "6")


def _map_vp9_cpu_used(preset: str) -> str:
    # libvpx-vp9 / libaom-av1 -cpu-used (higher is faster).
    mapping = {
        "ultrafast": "6",
        "superfast": "6",
        "veryfast": "5",
        "faster": "5",
        "fast": "5",
        "medium": "4",
        "slow": "2",
        "slower": "1",
        "veryslow": "0",
    }
    return mapping.get((preset or "medium").lower(), "4")


def _map_aom_cpu_used(preset: str) -> str:
    mapping = {
        "ultrafast": "8",
        "superfast": "8",
        "veryfast": "7",
        "faster": "7",
        "fast": "7",
        "medium": "6",
        "slow": "4",
        "slower": "3",
        "veryslow": "2",
    }
    return mapping.get((preset or "medium").lower(), "6")


def _scale_crf_for_av1_vp9(crf: int) -> int:
    """Translate an x264 CRF (18..32) to a comparable AV1/VP9 CRF scale."""
    return max(16, min(55, int(round(crf + 6))))


def build_video_encode_args(
    vcodec: str,
    crf: int = 23,
    preset: str = "medium",
    output_path: str = "",
) -> List[str]:
    """Build encoder-specific FFmpeg video arguments for ``vcodec``."""
    ext = Path(output_path).suffix.lower() if output_path else ".mp4"
    is_mp4_like = ext in (".mp4", ".mov", ".m4v", "")

    if vcodec == "libx264":
        return [
            "-c:v",
            "libx264",
            "-crf",
            str(crf),
            "-preset",
            preset,
            "-pix_fmt",
            "yuv420p",
        ]

    if vcodec == "libx265":
        args = [
            "-c:v",
            "libx265",
            "-crf",
            str(crf),
            "-preset",
            preset,
            "-pix_fmt",
            "yuv420p",
        ]
        if is_mp4_like:
            args.extend(["-tag:v", "hvc1"])
        return args

    if vcodec == "libsvtav1":
        return [
            "-c:v",
            "libsvtav1",
            "-crf",
            str(_scale_crf_for_av1_vp9(crf)),
            "-preset",
            _map_svt_preset(preset),
            "-pix_fmt",
            "yuv420p",
        ]

    if vcodec == "libaom-av1":
        return [
            "-c:v",
            "libaom-av1",
            "-crf",
            str(_scale_crf_for_av1_vp9(crf)),
            "-b:v",
            "0",
            "-cpu-used",
            _map_aom_cpu_used(preset),
            "-row-mt",
            "1",
            "-pix_fmt",
            "yuv420p",
        ]

    if vcodec == "libvpx-vp9":
        return [
            "-c:v",
            "libvpx-vp9",
            "-crf",
            str(_scale_crf_for_av1_vp9(crf)),
            "-b:v",
            "0",
            "-cpu-used",
            _map_vp9_cpu_used(preset),
            "-row-mt",
            "1",
            "-pix_fmt",
            "yuv420p",
        ]

    if vcodec == "h264_qsv":
        return [
            "-c:v",
            "h264_qsv",
            "-global_quality",
            str(crf),
            "-preset",
            _map_qsv_preset(preset),
            "-pix_fmt",
            "nv12",
        ]

    if vcodec == "hevc_qsv":
        args = [
            "-c:v",
            "hevc_qsv",
            "-global_quality",
            str(crf),
            "-preset",
            _map_qsv_preset(preset),
            "-pix_fmt",
            "nv12",
        ]
        if is_mp4_like:
            args.extend(["-tag:v", "hvc1"])
        return args

    if vcodec == "av1_qsv":
        return [
            "-c:v",
            "av1_qsv",
            "-global_quality",
            str(_scale_crf_for_av1_vp9(crf)),
            "-preset",
            _map_qsv_preset(preset),
            "-pix_fmt",
            "nv12",
        ]

    if vcodec == "vp9_qsv":
        return [
            "-c:v",
            "vp9_qsv",
            "-global_quality",
            str(_scale_crf_for_av1_vp9(crf)),
            "-low_power",
            "1",
            "-pix_fmt",
            "nv12",
        ]

    # Fallback for any custom/unknown vcodec passed directly by callers.
    return ["-c:v", vcodec, "-crf", str(crf), "-preset", preset]


def build_audio_encode_args(output_path: str = "") -> List[str]:
    """Use stream copy for MP4/MKV and Opus for WebM containers."""
    ext = Path(output_path).suffix.lower() if output_path else ".mp4"
    if ext == ".webm":
        return ["-c:a", "libopus", "-b:a", "128k"]
    return ["-c:a", "copy"]
