"""Unit tests for video codec & encoder selection and FFmpeg flag generation."""

from videocaptioner.core.entities import VideoCodecEnum, VideoEncoderEnum
from videocaptioner.core.utils.video_encoder import (
    build_audio_encode_args,
    build_video_encode_args,
    coerce_video_codec,
    coerce_video_encode_device,
    coerce_video_encoder,
    qsv_decode_args,
    qsv_download_filter,
    qsv_hw_args,
    resolve_video_encoder,
)

ALL_ENCODERS = {
    "libx264",
    "libx265",
    "libsvtav1",
    "libaom-av1",
    "libvpx-vp9",
    "h264_qsv",
    "hevc_qsv",
    "av1_qsv",
    "vp9_qsv",
}

SW_ONLY_ENCODERS = {
    "libx264",
    "libx265",
    "libsvtav1",
    "libaom-av1",
    "libvpx-vp9",
}


def test_coerce_codec_and_encoder_aliases():
    assert coerce_video_codec("h264") == VideoCodecEnum.H264
    assert coerce_video_codec("H.265 (HEVC)") == VideoCodecEnum.H265
    assert coerce_video_codec("hevc") == VideoCodecEnum.H265
    assert coerce_video_codec("av1") == VideoCodecEnum.AV1
    assert coerce_video_codec("vp9") == VideoCodecEnum.VP9

    assert coerce_video_encoder("auto") == VideoEncoderEnum.AUTO
    assert coerce_video_encoder("qsv") == VideoEncoderEnum.QSV
    assert coerce_video_encoder("Intel QSV") == VideoEncoderEnum.QSV
    assert coerce_video_encoder("svt") == VideoEncoderEnum.SVT
    assert coerce_video_encoder("SVT-AV1") == VideoEncoderEnum.SVT
    assert coerce_video_encoder("software") == VideoEncoderEnum.SOFTWARE
    assert coerce_video_encoder("软件编码") == VideoEncoderEnum.SOFTWARE


def test_resolve_qsv_across_all_four_codecs():
    cases = [
        (VideoCodecEnum.H264, "h264_qsv"),
        (VideoCodecEnum.H265, "hevc_qsv"),
        (VideoCodecEnum.AV1, "av1_qsv"),
        (VideoCodecEnum.VP9, "vp9_qsv"),
    ]
    for codec, expected in cases:
        res = resolve_video_encoder(
            codec,
            VideoEncoderEnum.QSV,
            available_encoders=ALL_ENCODERS,
            probe_hardware=False,
        )
        assert res.vcodec == expected
        assert res.is_hardware is True
        assert res.fallback_reason == ""


def test_qsv_hw_args_bind_specific_adapter():
    assert qsv_hw_args(None) == []
    args = qsv_hw_args(1)
    assert args == [
        "-init_hw_device",
        "d3d11va=d3d:1",
        "-init_hw_device",
        "qsv=hw@d3d",
        "-filter_hw_device",
        "hw",
    ]


def test_qsv_decode_args_match_encoder_adapter():
    assert qsv_decode_args(None) == []
    assert qsv_decode_args(1) == [
        "-hwaccel",
        "qsv",
        "-hwaccel_device",
        "hw",
        "-hwaccel_output_format",
        "qsv",
    ]
    assert qsv_download_filter() == "hwdownload,format=nv12"


def test_coerce_encode_device_aliases():
    from videocaptioner.core.entities import VideoEncodeDeviceEnum

    assert coerce_video_encode_device("auto") == VideoEncodeDeviceEnum.AUTO
    assert coerce_video_encode_device("dgpu") == VideoEncodeDeviceEnum.DISCRETE
    assert coerce_video_encode_device("独显") == VideoEncodeDeviceEnum.DISCRETE
    assert coerce_video_encode_device("igpu") == VideoEncodeDeviceEnum.INTEGRATED
    assert coerce_video_encode_device("核显") == VideoEncodeDeviceEnum.INTEGRATED


def test_resolve_qsv_falls_back_when_unavailable():
    res_h264 = resolve_video_encoder(
        VideoCodecEnum.H264,
        VideoEncoderEnum.QSV,
        available_encoders=SW_ONLY_ENCODERS,
        probe_hardware=False,
    )
    assert res_h264.vcodec == "libx264"
    assert res_h264.fallback_reason

    res_av1 = resolve_video_encoder(
        VideoCodecEnum.AV1,
        VideoEncoderEnum.QSV,
        available_encoders=SW_ONLY_ENCODERS,
        probe_hardware=False,
    )
    assert res_av1.vcodec == "libsvtav1"
    assert res_av1.fallback_reason


def test_resolve_svt_and_software_encoders():
    svt = resolve_video_encoder(
        VideoCodecEnum.AV1,
        VideoEncoderEnum.SVT,
        available_encoders=ALL_ENCODERS,
        probe_hardware=False,
    )
    assert svt.vcodec == "libsvtav1"
    assert svt.fallback_reason == ""

    # SVT requested with H.264 gracefully falls back to libx264
    svt_h264 = resolve_video_encoder(
        VideoCodecEnum.H264,
        VideoEncoderEnum.SVT,
        available_encoders=ALL_ENCODERS,
        probe_hardware=False,
    )
    assert svt_h264.vcodec == "libx264"
    assert svt_h264.fallback_reason

    # Explicit software encoders across all 4 codecs
    assert (
        resolve_video_encoder(
            VideoCodecEnum.H264,
            VideoEncoderEnum.SOFTWARE,
            available_encoders=ALL_ENCODERS,
            probe_hardware=False,
        ).vcodec
        == "libx264"
    )
    assert (
        resolve_video_encoder(
            VideoCodecEnum.H265,
            VideoEncoderEnum.SOFTWARE,
            available_encoders=ALL_ENCODERS,
            probe_hardware=False,
        ).vcodec
        == "libx265"
    )
    assert (
        resolve_video_encoder(
            VideoCodecEnum.AV1,
            VideoEncoderEnum.SOFTWARE,
            available_encoders=ALL_ENCODERS,
            probe_hardware=False,
        ).vcodec
        == "libaom-av1"
    )
    assert (
        resolve_video_encoder(
            VideoCodecEnum.VP9,
            VideoEncoderEnum.SOFTWARE,
            available_encoders=ALL_ENCODERS,
            probe_hardware=False,
        ).vcodec
        == "libvpx-vp9"
    )


def test_webm_container_enforces_webm_compatible_codec():
    res = resolve_video_encoder(
        VideoCodecEnum.H264,
        VideoEncoderEnum.SOFTWARE,
        output_path="out.webm",
        available_encoders=ALL_ENCODERS,
        probe_hardware=False,
    )
    assert res.codec == VideoCodecEnum.VP9
    assert res.vcodec == "libvpx-vp9"
    assert build_audio_encode_args("out.webm") == ["-c:a", "libopus", "-b:a", "128k"]
    assert build_audio_encode_args("out.mp4") == ["-c:a", "copy"]


def test_build_video_encode_args_per_encoder():
    h265_mp4 = build_video_encode_args("hevc_qsv", crf=23, preset="ultrafast", output_path="a.mp4")
    assert "-global_quality" in h265_mp4
    assert "veryfast" in h265_mp4  # ultrafast mapped to valid QSV preset
    assert "hvc1" in h265_mp4

    vp9_qsv = build_video_encode_args("vp9_qsv", crf=28, preset="medium", output_path="a.mp4")
    assert "-low_power" in vp9_qsv and "1" in vp9_qsv

    svt = build_video_encode_args("libsvtav1", crf=28, preset="medium", output_path="a.mp4")
    assert svt[:2] == ["-c:v", "libsvtav1"]
    assert "-preset" in svt and "6" in svt

    vp9_sw = build_video_encode_args("libvpx-vp9", crf=28, preset="fast", output_path="a.webm")
    assert "-b:v" in vp9_sw and "-cpu-used" in vp9_sw
