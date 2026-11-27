"""Tests for the Intel Arc / OpenVINO backend (no GPU required)."""

from videocaptioner.core.entities import FasterWhisperModelEnum
from videocaptioner.core.utils.hardware import (
    describe_hardware,
    is_intel_arc_a_series,
    is_intel_arc_b_series,
    resolve_faster_whisper_device,
)
from videocaptioner.core.utils.model_downloader import (
    FASTER_WHISPER_REPOS,
    OPENVINO_WHISPER_REPOS,
    faster_whisper_model_status,
    openvino_whisper_model_dir,
)


def test_all_ui_models_have_backend_mappings():
    """Every model exposed in the UI must have both CT2 and OpenVINO mappings."""
    for model in FasterWhisperModelEnum:
        assert model.value in FASTER_WHISPER_REPOS
        assert model.value in OPENVINO_WHISPER_REPOS
        assert OPENVINO_WHISPER_REPOS[model.value].endswith("-ov")


def test_arc_series_name_detection():
    assert is_intel_arc_a_series("Intel(R) Arc(TM) A770 Graphics")
    assert is_intel_arc_a_series("Intel(R) Arc(TM) A380 Graphics")
    assert is_intel_arc_a_series("Intel(R) Arc(TM) A750 Graphics")
    assert not is_intel_arc_a_series("Intel(R) Arc(TM) B580 Graphics")

    assert is_intel_arc_b_series("Intel(R) Arc(TM) B580 Graphics")
    assert is_intel_arc_b_series("Intel(R) Arc(TM) Pro B50 Graphics")
    assert not is_intel_arc_b_series("Intel(R) Arc(TM) A770 Graphics")


def test_faster_whisper_never_uses_cuda_on_intel_options():
    assert resolve_faster_whisper_device("intel_gpu") == "cpu"
    assert resolve_faster_whisper_device("intel_npu") == "cpu"
    assert resolve_faster_whisper_device("cpu") == "cpu"


def test_truncated_model_is_reported_as_not_ready(tmp_path):
    model_dir = tmp_path / "faster-whisper-tiny"
    model_dir.mkdir()
    (model_dir / "model.bin").write_bytes(b"0" * 1024)

    ready, reason = faster_whisper_model_status(tmp_path, "tiny")
    assert ready is False
    assert "size" in reason.lower()


def test_openvino_model_directory_layout(tmp_path):
    model_path = openvino_whisper_model_dir(tmp_path, "large-v3-turbo")
    assert model_path.name == "whisper-large-v3-turbo-fp16-ov"
    assert model_path.parent.name == "openvino"


def test_hardware_description_is_not_empty():
    assert describe_hardware().strip()
