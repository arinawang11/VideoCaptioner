"""Regression tests for FFmpeg text decoding in subtitle rendering.

Chinese Windows decodes child output as GBK by default; ffmpeg writes UTF-8.
A non-ASCII title in the media aborts the reader thread and leaves
``result.stderr`` as ``None``, which used to crash with
"expected string or bytes-like object, got 'NoneType'".
"""

from videocaptioner.core.subtitle import ass_renderer


class _Result:
    def __init__(self, stderr):
        self.stderr = stderr
        self.stdout = ""


def test_get_video_resolution_survives_none_stderr(monkeypatch):
    monkeypatch.setattr(
        ass_renderer.subprocess, "run", lambda *a, **k: _Result(None)
    )
    assert ass_renderer._get_video_resolution("any.mp4") == (1920, 1080)


def test_get_video_resolution_requests_utf8(monkeypatch):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured.update(kwargs)
        return _Result("Stream #0:0: Video: h264, yuv420p, 1280x720, 30 fps")

    monkeypatch.setattr(ass_renderer.subprocess, "run", fake_run)
    assert ass_renderer._get_video_resolution("any.mp4") == (1280, 720)
    assert captured.get("encoding") == "utf-8"
    assert captured.get("errors") == "replace"


def test_check_cuda_available_survives_none_stderr(monkeypatch):
    from videocaptioner.core.utils import video_utils

    def fake_run(cmd, **kwargs):
        if "-hwaccels" in cmd:
            return _Result("Hardware acceleration methods:\ncuda\n")
        return _Result(None)

    monkeypatch.setattr(video_utils.subprocess, "run", fake_run)
    assert video_utils.check_cuda_available() is False
