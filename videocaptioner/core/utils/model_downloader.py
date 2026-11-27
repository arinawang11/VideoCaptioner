# coding:utf-8
"""Model download helpers.

Downloads HuggingFace models through the hf-mirror.com mirror (fast in mainland
China), preferring aria2c multi-connection downloads with resume support, and
verifying file sizes after every download so half-written model folders are
never treated as "installed".

Used by:
- Faster-Whisper model downloads (CTranslate2)
- OpenVINO Whisper model downloads (Intel Arc / iGPU / NPU)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from videocaptioner.config import BIN_PATH, BUNDLED_BIN_PATH

try:
    import requests
except Exception:  # pragma: no cover - requests is a hard dependency
    requests = None

ProgressCallback = Optional[Callable[[int, str], None]]

# hf-mirror first, huggingface.co as fallback
HF_ENDPOINTS = ("https://hf-mirror.com", "https://huggingface.co")

# CTranslate2 (faster-whisper) repos and the exact size of model.bin.
# Systran does not publish large-v3-turbo, so a community CT2 conversion is used.
FASTER_WHISPER_REPOS: Dict[str, Tuple[str, int]] = {
    "tiny": ("Systran/faster-whisper-tiny", 75538270),
    "base": ("Systran/faster-whisper-base", 145217532),
    "small": ("Systran/faster-whisper-small", 483546902),
    "medium": ("Systran/faster-whisper-medium", 1527906378),
    "large-v1": ("Systran/faster-whisper-large-v1", 3086912962),
    "large-v2": ("Systran/faster-whisper-large-v2", 3086912962),
    "large-v3": ("Systran/faster-whisper-large-v3", 3087284237),
    "large-v3-turbo": ("deepdml/faster-whisper-large-v3-turbo-ct2", 1617884929),
}

# Pre-converted OpenVINO FP16 IR models (recommended for Intel Arc dGPU / iGPU).
OPENVINO_WHISPER_REPOS: Dict[str, str] = {
    "tiny": "OpenVINO/whisper-tiny-fp16-ov",
    "base": "OpenVINO/whisper-base-fp16-ov",
    "small": "OpenVINO/whisper-small-fp16-ov",
    "medium": "OpenVINO/whisper-medium-fp16-ov",
    # OpenVINO does not publish large-v1; large-v2 is used instead
    "large-v1": "OpenVINO/whisper-large-v2-fp16-ov",
    "large-v2": "OpenVINO/whisper-large-v2-fp16-ov",
    "large-v3": "OpenVINO/whisper-large-v3-fp16-ov",
    "large-v3-turbo": "OpenVINO/whisper-large-v3-turbo-fp16-ov",
}

# Pre-converted OpenVINO INT8 IR models (recommended for Intel AI Boost NPU,
# such as Core Ultra 7 155H Meteor Lake NPU 3720: ~50% smaller footprint and
# native INT8 execution on the NPU MAC array).
OPENVINO_WHISPER_INT8_REPOS: Dict[str, str] = {
    "tiny": "OpenVINO/whisper-tiny-int8-ov",
    "base": "OpenVINO/whisper-base-int8-ov",
    "small": "OpenVINO/whisper-small-int8-ov",
    "medium": "OpenVINO/whisper-medium-int8-ov",
    "large-v1": "OpenVINO/whisper-large-v2-int8-ov",
    "large-v2": "OpenVINO/whisper-large-v2-int8-ov",
    "large-v3": "OpenVINO/whisper-large-v3-int8-ov",
    "large-v3-turbo": "OpenVINO/whisper-large-v3-turbo-int8-ov",
}

_IGNORED_FILES = {".gitattributes", ".gitignore", "README.md", ".msc"}


def _log(message: str) -> None:
    try:
        from videocaptioner.core.utils.logger import setup_logger

        setup_logger("model_downloader").info(message)
    except Exception:  # pragma: no cover
        print(message)


def _get_json(url: str, timeout: int = 30):
    if requests is None:
        raise RuntimeError("requests is not installed")
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as first_error:
        # Mirror sites occasionally serve an incomplete certificate chain
        try:
            import urllib3

            urllib3.disable_warnings()
        except Exception:
            pass
        try:
            resp = requests.get(url, timeout=timeout, verify=False)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            raise first_error


def list_hf_files(repo_id: str, revision: str = "main") -> List[Dict]:
    """List files of a HuggingFace repo as ``[{"path": str, "size": int}]``."""
    last_error: Optional[Exception] = None
    for base in HF_ENDPOINTS:
        url = f"{base}/api/models/{repo_id}/tree/{revision}?recursive=true"
        try:
            data = _get_json(url)
            files: List[Dict] = []
            for item in data:
                if item.get("type") != "file":
                    continue
                path = item.get("path", "")
                if not path or path in _IGNORED_FILES:
                    continue
                size = item.get("size") or 0
                lfs = item.get("lfs") or {}
                if lfs.get("size"):
                    size = lfs["size"]
                files.append({"path": path, "size": int(size)})
            if files:
                return files
        except Exception as e:  # try next endpoint
            last_error = e
            _log(f"Failed to list {repo_id} via {base}: {e}")
    raise RuntimeError(f"Cannot list files of {repo_id}: {last_error}")


def _aria2c_path() -> str:
    for folder in (BIN_PATH, BUNDLED_BIN_PATH):
        candidate = Path(folder) / "aria2c.exe"
        if candidate.exists():
            return str(candidate)
    return shutil.which("aria2c") or ""


def _format_size(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024:
            return f"{num:.1f}{unit}"
        num /= 1024
    return f"{num:.1f}TB"


def _probe_download(url: str) -> tuple[int, bool]:
    """Return ``(content_length, supports_ranges)`` for a URL."""
    if requests is None:
        return 0, False

    def _do(verify: bool):
        resp = requests.head(url, allow_redirects=True, timeout=30, verify=verify)
        resp.raise_for_status()
        size = int(resp.headers.get("Content-Length") or 0)
        ranges = resp.headers.get("Accept-Ranges", "").lower() == "bytes"
        return size, ranges

    try:
        return _do(True)
    except Exception:
        try:
            import urllib3

            urllib3.disable_warnings()
        except Exception:
            pass
        try:
            return _do(False)
        except Exception:
            return 0, False


def _download_parallel(
    url: str,
    dest: Path,
    on_percent: Optional[Callable[[int], None]] = None,
    connections: int = 8,
) -> None:
    """Multi-connection download fallback when aria2c is not available."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    total, supports_ranges = _probe_download(url)
    if not total or not supports_ranges:
        raise RuntimeError("server does not support range requests")

    connections = max(1, min(connections, total // (4 * 1024 * 1024) or 1))
    chunk_size = total // connections
    parts = []
    for index in range(connections):
        start = index * chunk_size
        end = total - 1 if index == connections - 1 else start + chunk_size - 1
        parts.append((index, start, end))

    part_paths = [dest.with_name(dest.name + f".part{index}") for index, _, _ in parts]
    done_bytes = [p.stat().st_size if p.exists() else 0 for p in part_paths]
    lock = threading.Lock()

    def _range_len(start: int, end: int) -> int:
        return end - start + 1

    def _report() -> None:
        if on_percent:
            on_percent(min(99, int(sum(done_bytes) * 100 / total)))

    def _fetch(index: int, start: int, end: int) -> None:
        part = part_paths[index]
        expected = _range_len(start, end)
        if part.exists() and part.stat().st_size == expected:
            return
        if part.exists():
            part.unlink()

        import urllib3

        urllib3.disable_warnings()
        headers = {"Range": f"bytes={start}-{end}"}
        try:
            resp = requests.get(url, headers=headers, stream=True, timeout=60)
            resp.raise_for_status()
        except Exception:
            resp = requests.get(
                url, headers=headers, stream=True, timeout=60, verify=False
            )
            resp.raise_for_status()

        with resp, open(part, "wb") as f:
            for chunk in resp.iter_content(chunk_size=512 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                with lock:
                    done_bytes[index] += len(chunk)
                    _report()

    with ThreadPoolExecutor(max_workers=connections) as executor:
        futures = [executor.submit(_fetch, index, start, end) for index, start, end in parts]
        for future in futures:
            future.result()

    merged = dest.with_name(dest.name + ".merging")
    with open(merged, "wb") as out:
        for part in part_paths:
            with open(part, "rb") as src:
                while True:
                    data = src.read(1024 * 1024)
                    if not data:
                        break
                    out.write(data)
    for part in part_paths:
        part.unlink(missing_ok=True)
    os.replace(merged, dest)


def _download_with_requests(
    url: str,
    dest: Path,
    expected_size: int = 0,
    on_percent: Optional[Callable[[int], None]] = None,
) -> None:
    if requests is None:
        raise RuntimeError("requests is not installed")

    tmp = dest.with_suffix(dest.suffix + ".part")

    def _do(verify: bool) -> None:
        headers = {}
        if tmp.exists():
            headers["Range"] = f"bytes={tmp.stat().st_size}-"
        with requests.get(url, stream=True, timeout=30, verify=verify, headers=headers) as r:
            if r.status_code == 416:  # already complete
                return
            r.raise_for_status()
            mode = "ab" if tmp.exists() and r.status_code == 206 else "wb"
            total = expected_size or int(r.headers.get("Content-Length") or 0)
            done = tmp.stat().st_size if mode == "ab" else 0
            with open(tmp, mode) as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    done += len(chunk)
                    if on_percent and total:
                        on_percent(min(99, int(done * 100 / total)))

    try:
        _do(verify=True)
    except Exception as e:
        _log(f"HTTPS verification failed, retrying without verification: {e}")
        try:
            import urllib3

            urllib3.disable_warnings()
        except Exception:
            pass
        _do(verify=False)

    if dest.exists():
        dest.unlink()
    os.replace(tmp, dest)


def _download_file_once(
    url: str,
    dest: Path,
    expected_size: int = 0,
    on_percent: Optional[Callable[[int], None]] = None,
) -> None:
    """One download attempt: aria2c, then parallel ranges, then requests."""
    control_file = dest.with_name(dest.name + ".aria2")
    if dest.exists() and not control_file.exists():
        if expected_size and dest.stat().st_size != expected_size:
            _log(f"Removing corrupt file before re-download: {dest}")
            try:
                dest.unlink()
            except Exception:
                pass

    aria = _aria2c_path()
    if aria:
        cmd = [
            aria,
            "--check-certificate=false",
            "-x16",
            "-s16",
            "-k4M",
            "--file-allocation=none",
            "--continue=true",
            "--auto-file-renaming=false",
            "--allow-overwrite=true",
            "--show-console-readout=false",
            "--summary-interval=2",
            f"--dir={dest.parent}",
            f"--out={dest.name}",
            url,
        ]
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        last_percent = -1
        assert process.stdout is not None
        for line in process.stdout:
            if not line or "%" not in line or on_percent is None:
                continue
            try:
                marker = line[line.index("(") + 1 : line.index("%")]
                percent = int("".join(ch for ch in marker if ch.isdigit()) or 0)
                if percent != last_percent:
                    last_percent = percent
                    on_percent(min(99, percent))
            except Exception:
                pass
        process.wait()
        if process.returncode == 0 and dest.exists():
            return
        _log(f"aria2c failed (code={process.returncode}), falling back to requests")

    if expected_size > 4 * 1024 * 1024:
        try:
            _download_parallel(url, dest, on_percent)
            if not expected_size or dest.stat().st_size == expected_size:
                return
            _log("parallel download size mismatch, retrying single connection")
            dest.unlink(missing_ok=True)
        except Exception as e:
            _log(f"parallel download failed ({e}), falling back to single connection")
            for part in dest.parent.glob(dest.name + ".part*"):
                part.unlink(missing_ok=True)

    _download_with_requests(url, dest, expected_size, on_percent)


def download_file(
    url: str,
    dest: Path,
    expected_size: int = 0,
    on_percent: Optional[Callable[[int], None]] = None,
    retries: int = 3,
) -> None:
    """Download one file with resume support, progress reporting and retries.

    Flaky networks (SSL resets, dropped connections) abort single requests,
    which used to fail the whole worker. All backends keep partial data in
    resumable files, so retrying continues where the previous attempt stopped.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    last_error: Optional[Exception] = None
    for attempt in range(max(1, retries)):
        try:
            _download_file_once(url, dest, expected_size, on_percent)
            return
        except Exception as e:  # noqa: BLE001 - retry any transport error
            last_error = e
            if attempt < retries - 1:
                delay = min(30, 2**attempt)
                _log(
                    f"Download of {dest.name} failed ({e}), "
                    f"retrying in {delay}s ({attempt + 1}/{retries})"
                )
                time.sleep(delay)
    raise last_error if last_error else RuntimeError(f"Download failed: {url}")


def is_file_ready(path: Path, expected_size: int = 0) -> bool:
    path = Path(path)
    if not path.exists() or not path.is_file():
        return False
    if expected_size and path.stat().st_size != expected_size:
        return False
    return path.stat().st_size > 0


def download_hf_model(
    repo_id: str,
    dest_dir: Path,
    progress: ProgressCallback = None,
    label: str = "",
) -> Path:
    """Download a whole HuggingFace repo, skipping already valid files."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    files = list_hf_files(repo_id)
    total = sum(f["size"] for f in files) or 1
    finished = 0
    label = label or repo_id.split("/")[-1]

    if progress:
        progress(0, f"Preparing download: {label}")

    for item in files:
        rel = item["path"]
        size = item["size"]
        target = dest_dir / rel
        if is_file_ready(target, size):
            finished += size
            if progress:
                progress(min(99, int(finished * 100 / total)), f"{label} ready")
            continue

        url = f"{HF_ENDPOINTS[0]}/{repo_id}/resolve/main/{rel}"

        def _on_percent(percent: int, _size=size, _rel=rel, _base=finished):
            if progress:
                current = _base + _size * percent / 100
                progress(
                    min(99, int(current * 100 / total)),
                    f"Downloading {label}: {_rel} ({percent}%)",
                )

        _log(f"Downloading model file: {repo_id}/{rel} ({_format_size(size)})")
        download_file(url, target, size, _on_percent)

        if size and (not target.exists() or target.stat().st_size != size):
            actual = target.stat().st_size if target.exists() else 0
            raise RuntimeError(
                f"Size verification failed for {rel}: expected {size}, got {actual}"
            )
        finished += size
        if progress:
            progress(min(99, int(finished * 100 / total)), f"Finished {label}: {rel}")

    (dest_dir / ".download_complete").write_text(str(int(time.time())), encoding="utf-8")
    if progress:
        progress(100, f"{label} download complete")
    return dest_dir


def faster_whisper_model_dir(model_dir: Path, model_name: str) -> Path:
    return Path(model_dir) / f"faster-whisper-{model_name}"


def faster_whisper_model_status(model_dir: Path, model_name: str) -> Tuple[bool, str]:
    """Check whether a CTranslate2 model folder is complete."""
    model_path = faster_whisper_model_dir(model_dir, model_name)
    if not model_path.exists():
        return False, "model directory does not exist"
    repo, expected = FASTER_WHISPER_REPOS.get(model_name, ("", 0))
    bin_path = model_path / "model.bin"
    if not bin_path.exists():
        return False, "model.bin is missing"
    if expected and bin_path.stat().st_size != expected:
        return (
            False,
            f"model.bin has an unexpected size "
            f"({_format_size(bin_path.stat().st_size)}, expected {_format_size(expected)})",
        )
    return True, ""


def ensure_faster_whisper_model(
    model_name: str, model_dir: Path, progress: ProgressCallback = None
) -> Path:
    """Make sure a CTranslate2 model is fully downloaded."""
    repo, _expected = FASTER_WHISPER_REPOS.get(model_name, ("", 0))
    if not repo:
        raise RuntimeError(f"Unknown Faster-Whisper model: {model_name}")

    model_path = faster_whisper_model_dir(model_dir, model_name)
    ready, reason = faster_whisper_model_status(model_dir, model_name)
    if ready:
        return model_path

    if model_path.exists():
        for junk in model_path.glob("model.bin_*"):
            try:
                junk.unlink()
            except Exception:
                pass
        for junk in model_path.glob("._____temp"):
            shutil.rmtree(junk, ignore_errors=True)
        _log(f"Model {model_name} is incomplete ({reason}), downloading again")

    return download_hf_model(repo, model_path, progress, label=f"Faster-Whisper {model_name}")


def _select_openvino_repo(model_name: str, precision: str = "fp16") -> str:
    table = (
        OPENVINO_WHISPER_INT8_REPOS
        if (precision or "").lower() == "int8"
        else OPENVINO_WHISPER_REPOS
    )
    repo = table.get(model_name)
    if not repo:
        raise RuntimeError(f"Unknown OpenVINO Whisper model: {model_name}")
    return repo


def openvino_whisper_model_dir(
    model_dir: Path, model_name: str, precision: str = "fp16"
) -> Path:
    repo = _select_openvino_repo(model_name, precision)
    return Path(model_dir) / "openvino" / repo.split("/")[-1]


# Files every OpenVINO Whisper repo ships. A previous download can be
# interrupted after the two XML files land, leaving the rest missing; checking
# only the XMLs then quietly breaks inference at pipeline build time (e.g. a
# missing preprocessor_config.json makes GenAI default to 80 mel bins while
# large-v3 weights expect 128). Existence checks work offline, so bundled
# models never trigger a re-download.
OPENVINO_MODEL_REQUIRED_FILES = (
    "openvino_encoder_model.xml",
    "openvino_encoder_model.bin",
    "openvino_decoder_model.xml",
    "openvino_decoder_model.bin",
    "openvino_detokenizer.xml",
    "openvino_detokenizer.bin",
    "preprocessor_config.json",
)


def openvino_model_files_complete(model_path: Path) -> bool:
    """Whether every required OpenVINO model file exists (no size check)."""
    return all((model_path / name).is_file() for name in OPENVINO_MODEL_REQUIRED_FILES)


def ensure_openvino_whisper_model(
    model_name: str,
    model_dir: Path,
    progress: ProgressCallback = None,
    precision: str = "fp16",
) -> Path:
    """Make sure a pre-converted OpenVINO Whisper model is available."""
    repo = _select_openvino_repo(model_name, precision)
    model_path = openvino_whisper_model_dir(model_dir, model_name, precision=precision)
    tag = f"{model_name} ({precision})" if precision != "fp16" else model_name
    if openvino_model_files_complete(model_path):
        if progress:
            progress(100, f"OpenVINO model {tag} is ready")
        return model_path
    return download_hf_model(repo, model_path, progress, label=f"OpenVINO {tag}")
