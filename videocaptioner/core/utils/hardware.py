# coding:utf-8
"""Hardware detection helpers.

Important: the OpenVINO runtime must never be imported in the Qt (UI) process.
OpenVINO 2026 requires a newer MSVC runtime than the one bundled with PyQt5
(14.26), and loading both in one process crashes with 0xc0000005.

Therefore all UI-facing helpers below avoid importing OpenVINO; the
OpenVINO-specific helpers are only used inside ``openvino_worker.py``.
"""

from __future__ import annotations

import ctypes
import os
import re
import shutil
import subprocess
from typing import List, Tuple


def _run(cmd: List[str], timeout: int = 8) -> str:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=timeout,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        return (result.stdout or "") + (result.stderr or "")
    except Exception:
        return ""


def display_adapters() -> List[str]:
    """Enumerate display adapter names without WMI or OpenVINO."""
    if os.name != "nt":
        return []
    try:
        from ctypes import wintypes

        class DISPLAY_DEVICEW(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("DeviceName", wintypes.WCHAR * 32),
                ("DeviceString", wintypes.WCHAR * 128),
                ("StateFlags", wintypes.DWORD),
                ("DeviceID", wintypes.WCHAR * 128),
                ("DeviceKey", wintypes.WCHAR * 128),
            ]

        names: List[str] = []
        user32 = ctypes.windll.user32
        index = 0
        while True:
            device = DISPLAY_DEVICEW()
            device.cb = ctypes.sizeof(DISPLAY_DEVICEW)
            if not user32.EnumDisplayDevicesW(None, index, ctypes.byref(device), 0):
                break
            name = (device.DeviceString or "").strip()
            if name and "microsoft" not in name.lower():
                names.append(name)
            index += 1
        # One physical adapter can be reported once per output; keep it unique.
        return list(dict.fromkeys(names))
    except Exception:
        return []


def intel_gpu_adapters() -> List[str]:
    return [name for name in display_adapters() if "intel" in name.lower()]


def is_intel_arc(name: str) -> bool:
    return "arc" in (name or "").lower()


def is_intel_arc_a_series(name: str) -> bool:
    """Arc A-series (Alchemist): A310 / A380 / A580 / A750 / A770 ..."""
    return bool(re.search(r"arc.*\ba\d{3}\b", (name or "").lower()))


def is_intel_arc_b_series(name: str) -> bool:
    """Arc B-series (Battlemage): B570 / B580 / Pro B50 ..."""
    return bool(re.search(r"arc.*\bb\d{2,3}\b", (name or "").lower()))


def has_nvidia_gpu() -> bool:
    if any("nvidia" in name.lower() for name in display_adapters()):
        return True
    if shutil.which("nvidia-smi"):
        return "GPU" in _run(["nvidia-smi", "-L"])
    return False


def has_intel_gpu() -> bool:
    return bool(intel_gpu_adapters())


def openvino_packages_available() -> bool:
    """Check whether the Intel extras are installed without loading them."""
    import importlib.util

    try:
        return (
            importlib.util.find_spec("openvino") is not None
            and importlib.util.find_spec("openvino_genai") is not None
        )
    except Exception:
        return False


def preferred_backend_device() -> str:
    """Pick a sensible default for the ``auto`` device option."""
    if has_nvidia_gpu():
        return "cuda"
    if has_intel_gpu() and openvino_packages_available():
        return "intel_gpu"
    return "cpu"


def describe_hardware() -> str:
    """Human readable hardware summary for the settings page (no OpenVINO)."""
    adapters = display_adapters()
    parts: List[str] = []
    intel = [name for name in adapters if "intel" in name.lower()]
    nvidia = [name for name in adapters if "nvidia" in name.lower()]
    if intel:
        parts.append("Intel GPU: " + " / ".join(intel))
    if nvidia:
        parts.append("NVIDIA GPU: " + " / ".join(nvidia))
    if not parts:
        if adapters:
            parts.append("GPU: " + " / ".join(adapters))
        else:
            parts.append("No acceleration hardware detected")
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# The helpers below are only used inside the OpenVINO worker subprocess.
# ---------------------------------------------------------------------------
def openvino_available() -> bool:
    try:
        import openvino  # noqa: F401

        return True
    except Exception:
        return False


def list_openvino_devices() -> List[Tuple[str, str]]:
    """Return ``[(device_id, full_name)]`` such as ``[("GPU.1", "Intel(R) Arc(TM) A770")]``."""
    if not openvino_available():
        return []
    try:
        import openvino as ov

        core = ov.Core()
        devices: List[Tuple[str, str]] = []
        for device in core.available_devices:
            try:
                name = core.get_property(device, "FULL_DEVICE_NAME")
            except Exception:
                name = device
            devices.append((device, name))
        return devices
    except Exception:
        return []


def intel_gpu_devices() -> List[Tuple[str, str]]:
    return [(device, name) for device, name in list_openvino_devices() if device.startswith("GPU")]


def intel_npu_available() -> bool:
    return any(device.startswith("NPU") for device, _ in list_openvino_devices())


def resolve_openvino_device(preference: str) -> str:
    """Map the UI device option to an OpenVINO device string.

    Prefer a discrete Arc GPU over the integrated one, so ``intel_gpu`` does not
    accidentally run on the iGPU when both are present.
    """
    preference = (preference or "").lower()

    if preference.startswith("intel_npu") or preference.startswith("npu"):
        if intel_npu_available():
            return "NPU"
        return resolve_openvino_device("intel_gpu")

    if preference.startswith("intel_gpu") or preference.startswith("gpu"):
        gpus = intel_gpu_devices()
        if not gpus:
            raise RuntimeError(
                "No Intel GPU detected. Please install the Intel graphics driver "
                "(OpenVINO GPU plugin)."
            )
        arc = [(device, name) for device, name in gpus if is_intel_arc(name)]
        return (arc or gpus)[0][0]

    if preference.startswith("cpu"):
        return "CPU"

    return "AUTO"


def resolve_faster_whisper_device(preference: str) -> str:
    """CTranslate2 only supports cuda/cpu; never return cuda without an NVIDIA GPU."""
    preference = (preference or "").lower()
    if preference.startswith("intel"):
        return "cpu"
    if preference.startswith("cuda"):
        return "cuda" if has_nvidia_gpu() else "cpu"
    return "cpu"
