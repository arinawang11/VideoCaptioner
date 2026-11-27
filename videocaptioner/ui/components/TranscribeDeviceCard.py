# coding: utf-8
"""Merged transcription device selector: device class + concrete Intel GPU."""

from typing import List, Optional, Tuple

from PyQt5.QtCore import Qt, pyqtSignal
from qfluentwidgets import ComboBox, SettingCard
from qfluentwidgets import FluentIcon as FIF

from videocaptioner.core.utils.hardware import openvino_gpu_choices


def build_device_choices(
    gpu_choices: Optional[List[Tuple[str, str]]] = None,
) -> List[Tuple[str, str, str]]:
    """Build ``[(display, device, gpu_name)]`` for the merged selector.

    ``gpu_choices`` is ``[(display, adapter_name)]`` from
    :func:`openvino_gpu_choices`; it defaults to the live enumeration.
    """
    if gpu_choices is None:
        try:
            gpu_choices = openvino_gpu_choices()
        except Exception:
            gpu_choices = []

    choices: List[Tuple[str, str, str]] = [
        ("自动选择（推荐）", "auto", "")
    ]
    # Always keep the automatic Intel GPU entry: OpenVINO may still expose a
    # GPU even when display enumeration fails.
    choices.append(("Intel GPU（自动，优先独显）", "intel_gpu", ""))
    for display, adapter_name in gpu_choices or []:
        choices.append((f"Intel GPU · {display}", "intel_gpu", adapter_name))
    choices.extend(
        [
            ("Intel NPU（AI Boost）", "intel_npu", ""),
            ("Intel CPU（OpenVINO）", "intel_cpu", ""),
            ("NVIDIA CUDA", "cuda", ""),
            ("CPU（Faster-Whisper）", "cpu", ""),
        ]
    )
    return choices


class TranscribeDeviceCard(SettingCard):
    """运行设备下拉框，把 Intel GPU 设备名直接并入同一项。

    Emits ``(device, gpu_adapter_name)`` where ``gpu_adapter_name`` is only set
    for concrete Intel GPU entries.
    """

    deviceChanged = pyqtSignal(str, str)

    def __init__(
        self,
        title: str,
        content: Optional[str] = None,
        parent=None,
        icon=FIF.IOT,
    ):
        super().__init__(icon, title, content, parent)
        self.comboBox = ComboBox(self)
        self.comboBox.setMinimumWidth(320)
        self.hBoxLayout.addWidget(self.comboBox, 1, Qt.AlignRight)  # type: ignore
        self.hBoxLayout.addSpacing(16)

        self._choices: List[Tuple[str, str, str]] = []
        self._refresh_choices()
        self.comboBox.currentIndexChanged.connect(self._on_index_changed)

    def _refresh_choices(self) -> None:
        choices = build_device_choices()
        self._choices = choices
        self.comboBox.blockSignals(True)
        self.comboBox.clear()
        for display, _device, _gpu in choices:
            self.comboBox.addItem(display)
        self.comboBox.blockSignals(False)

    def _index_for(self, device: str, gpu: str) -> int:
        device = (device or "auto").strip().lower()
        gpu = (gpu or "").strip()
        if device in ("", "auto"):
            return 0
        if device == "intel_gpu":
            if gpu:
                for i, (_d, dev, name) in enumerate(self._choices):
                    if dev == "intel_gpu" and name == gpu:
                        return i
            for i, (_d, dev, name) in enumerate(self._choices):
                if dev == "intel_gpu" and not name:
                    return i
            return 0
        for i, (_d, dev, _name) in enumerate(self._choices):
            if dev == device:
                return i
        return 0

    def set_choice(self, device: str, gpu: str = "") -> None:
        """Select the matching entry without emitting ``deviceChanged``."""
        index = self._index_for(device, gpu)
        self.comboBox.blockSignals(True)
        self.comboBox.setCurrentIndex(index)
        self.comboBox.blockSignals(False)

    def refresh(self, device: str = "", gpu: str = "") -> None:
        """Re-enumerate adapters (GPU hot-plug) and keep the stored choice."""
        self._refresh_choices()
        self.set_choice(device, gpu)

    def current_choice(self) -> Tuple[str, str]:
        if 0 <= self.comboBox.currentIndex() < len(self._choices):
            _display, device, gpu = self._choices[self.comboBox.currentIndex()]
            return device, gpu
        return "auto", ""

    def current_display(self) -> str:
        return self.comboBox.currentText()

    def _on_index_changed(self, _index: int) -> None:
        device, gpu = self.current_choice()
        self.deviceChanged.emit(device, gpu)
