"""Tests for the Intel Arc / OpenVINO backend (no GPU required)."""

import numpy as np

from videocaptioner.core.asr.openvino_asr import (
    segment_speech_for_npu,
    split_clauses_proportionally,
)
from videocaptioner.core.entities import FasterWhisperModelEnum
from videocaptioner.core.utils.hardware import (
    describe_hardware,
    intel_npu_adapters,
    is_intel_arc_a_series,
    is_intel_arc_b_series,
    resolve_faster_whisper_device,
)
from videocaptioner.core.utils.model_downloader import (
    FASTER_WHISPER_REPOS,
    OPENVINO_WHISPER_INT8_REPOS,
    OPENVINO_WHISPER_REPOS,
    faster_whisper_model_status,
    openvino_whisper_model_dir,
)


def test_all_ui_models_have_backend_mappings():
    """Every model exposed in the UI must have CT2, fp16 and int8 mappings."""
    for model in FasterWhisperModelEnum:
        assert model.value in FASTER_WHISPER_REPOS
        assert model.value in OPENVINO_WHISPER_REPOS
        assert model.value in OPENVINO_WHISPER_INT8_REPOS
        assert OPENVINO_WHISPER_REPOS[model.value].endswith("-fp16-ov")
        assert OPENVINO_WHISPER_INT8_REPOS[model.value].endswith("-int8-ov")


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


def test_openvino_int8_model_directory_layout(tmp_path):
    model_path = openvino_whisper_model_dir(tmp_path, "large-v3-turbo", precision="int8")
    assert model_path.name == "whisper-large-v3-turbo-int8-ov"
    assert model_path.parent.name == "openvino"


def test_intel_npu_adapters_list_without_openvino():
    # Must never raise and must not import OpenVINO in the UI process.
    assert isinstance(intel_npu_adapters(), list)


def test_npu_speech_segmentation_splits_on_pauses():
    sr = 16000
    silence = np.zeros(int(1.0 * sr), dtype=np.float32)

    def tone(seconds: float) -> np.ndarray:
        t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
        return (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    audio = np.concatenate([silence, tone(2.0), silence, tone(3.0), silence])
    spans = segment_speech_for_npu(audio)

    assert 2 <= len(spans) <= 3
    for start, end in spans:
        assert 0 <= start < end <= len(audio)
        assert (end - start) / sr <= 14.5
    assert [start for start, _ in spans] == sorted(start for start, _ in spans)


def test_npu_speech_segmentation_caps_long_speech():
    sr = 16000
    t = np.arange(25 * sr) / sr
    audio = (0.2 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    spans = segment_speech_for_npu(audio)

    assert spans
    assert all((end - start) / sr <= 14.5 for start, end in spans)
    # Long monologues must still be fully covered.
    assert sum(end - start for start, end in spans) >= len(audio) * 0.9


def test_clause_splitting_keeps_total_time():
    segments = split_clauses_proportionally(
        "First sentence. Second sentence! Third sentence?",
        1000,
        13000,
        max_clause_ms=6000,
    )
    assert len(segments) == 3
    assert segments[0].start_time == 1000
    assert segments[-1].end_time == 13000
    assert [seg.text for seg in segments] == [
        "First sentence.",
        "Second sentence!",
        "Third sentence?",
    ]

def test_gpu_precision_hint_explicit_and_env(monkeypatch):
    from videocaptioner.core.asr.openvino_asr import _gpu_precision_hint

    assert _gpu_precision_hint("GPU", "f32") == "f32"
    assert _gpu_precision_hint("GPU", "fp16") == "fp16"

    monkeypatch.setenv("VIDEOCAPTIONER_OPENVINO_GPU_PRECISION", "fp16")
    assert _gpu_precision_hint("GPU") == "fp16"
    assert _gpu_precision_hint("CPU") == ""


def test_gpu_precision_hint_uses_f32_on_integrated_gpu(monkeypatch):
    from videocaptioner.core.asr.openvino_asr import _gpu_precision_hint
    from videocaptioner.core.utils import hardware

    monkeypatch.delenv("VIDEOCAPTIONER_OPENVINO_GPU_PRECISION", raising=False)

    monkeypatch.setattr(
        hardware,
        "intel_gpu_devices",
        lambda: [("GPU", "Intel(R) Arc(TM) Graphics (iGPU)")],
    )
    assert _gpu_precision_hint("GPU") == "f32"

    monkeypatch.setattr(
        hardware,
        "intel_gpu_devices",
        lambda: [("GPU.1", "Intel(R) Arc(TM) Pro B50 Graphics (dGPU)")],
    )
    assert _gpu_precision_hint("GPU.1") == ""


def test_openvino_required_files_cover_tokenizer_but_not_int8_only_config():
    """A partial download must not be reported complete.

    The tokenizer files are needed to load the pipeline, but
    ``openvino_config.json`` only exists in the int8 repos: requiring it would
    make every fp16 model look incomplete and re-download on each run.
    """
    from videocaptioner.core.utils.model_downloader import (
        OPENVINO_MODEL_REQUIRED_FILES,
    )

    required = set(OPENVINO_MODEL_REQUIRED_FILES)
    assert "openvino_tokenizer.xml" in required
    assert "openvino_tokenizer.bin" in required
    assert "preprocessor_config.json" in required
    assert "tokenizer.json" in required
    assert "vocab.json" in required
    assert "openvino_config.json" not in required


def test_format_runtime_device_includes_device_name_and_precision():
    from videocaptioner.core.asr.openvino_asr import _format_runtime_device

    assert (
        _format_runtime_device(
            "GPU.1", "Intel(R) Arc(TM) Pro B50 Graphics (dGPU)", "fp16", ""
        )
        == "GPU.1 · Intel(R) Arc(TM) Pro B50 Graphics (dGPU) (fp16)"
    )
    assert (
        _format_runtime_device(
            "GPU", "Intel(R) Arc(TM) Graphics (iGPU)", "int8", "f32"
        )
        == "GPU · Intel(R) Arc(TM) Graphics (iGPU) (int8, f32)"
    )
    assert (
        _format_runtime_device("NPU", "Intel(R) AI Boost", "int8", "")
        == "NPU · Intel(R) AI Boost (int8)"
    )
    assert _format_runtime_device("CPU", "", "int8", "") == "CPU (int8)"
