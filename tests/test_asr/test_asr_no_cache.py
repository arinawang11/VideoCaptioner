"""Transcription must always run again; results are never cached."""

import pytest

from videocaptioner.core.asr.asr_data import ASRDataSeg
from videocaptioner.core.asr.base import BaseASR


class CountingASR(BaseASR):
    """Minimal ASR that counts how often the real recognition runs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_count = 0

    def _run(self, callback=None, **kwargs):
        self.run_count += 1
        return {"text": "hello"}

    def _make_segments(self, resp_data):
        return [ASRDataSeg(resp_data["text"], 0, 1000)]


def _make_asr(monkeypatch) -> CountingASR:
    # Avoid decoding audio with ffmpeg; the CRC of the fake bytes is enough.
    monkeypatch.setattr(BaseASR, "_get_audio_duration", lambda self: 1.0)
    return CountingASR(b"fake-audio-bytes")


def test_run_transcribes_every_time(monkeypatch):
    asr = _make_asr(monkeypatch)

    first = asr.run()
    second = asr.run()

    assert asr.run_count == 2
    assert [seg.text for seg in first.segments] == ["hello"]
    assert [seg.text for seg in second.segments] == ["hello"]


def test_base_asr_has_no_cache_switch(monkeypatch):
    monkeypatch.setattr(BaseASR, "_get_audio_duration", lambda self: 1.0)
    with pytest.raises(TypeError):
        CountingASR(b"fake-audio-bytes", use_cache=True)


def test_rate_limit_cache_is_still_available(monkeypatch):
    """Only result caching was removed; Bcut/JianYing rate limiting needs it."""
    asr = _make_asr(monkeypatch)
    assert asr._cache is not None
