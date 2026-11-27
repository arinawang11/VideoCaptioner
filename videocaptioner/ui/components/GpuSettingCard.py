# coding: utf-8
"""GPU adapter selector card: choose a concrete Intel GPU by its device name."""

from typing import Callable, List, Optional, Tuple

from PyQt5.QtCore import Qt, pyqtSignal
from qfluentwidgets import ComboBox, SettingCard
from qfluentwidgets import FluentIcon as FIF

from videocaptioner.core.utils.video_encoder import qsv_adapter_choices

AUTO_LABEL = "自动（优先独显）"


class GpuComboBoxSettingCard(SettingCard):
    """Setting card that lists detected Intel GPU adapters by name."""

    valueChanged = pyqtSignal(str)  # emits the adapter name, "" for auto

    def __init__(
        self,
        title: str,
        content: Optional[str] = None,
        parent=None,
        icon=FIF.IOT,
        choices_provider: Optional[Callable[[], List[Tuple[str, str]]]] = None,
        auto_label: str = AUTO_LABEL,
    ):
        super().__init__(icon, title, content, parent)
        self._provider = choices_provider or qsv_adapter_choices
        self._auto_label = auto_label
        self.comboBox = ComboBox(self)
        self.comboBox.setMinimumWidth(300)
        self.hBoxLayout.addWidget(self.comboBox, 1, Qt.AlignRight)  # type: ignore
        self.hBoxLayout.addSpacing(16)

        self._choices: List[Tuple[str, str]] = []
        self.refresh()
        self.comboBox.currentIndexChanged.connect(self._on_index_changed)

    def refresh(self, selected: str = "") -> None:
        """Rebuild the adapter list from the live probe/provider."""
        try:
            choices = [(self._auto_label, "")] + list(self._provider())
        except Exception:
            choices = [(self._auto_label, "")]

        current = selected or self.value()
        self._choices = choices
        self.comboBox.blockSignals(True)
        self.comboBox.clear()
        for display, _name in choices:
            self.comboBox.addItem(display)
        index = 0
        for i, (_display, name) in enumerate(choices):
            if name and name == current:
                index = i
                break
        self.comboBox.setCurrentIndex(index)
        self.comboBox.blockSignals(False)

    def _on_index_changed(self, index: int) -> None:
        self.valueChanged.emit(self.value())

    def value(self) -> str:
        if 0 <= self.comboBox.currentIndex() < len(self._choices):
            return self._choices[self.comboBox.currentIndex()][1]
        return ""

    def setValue(self, name: str) -> None:
        for i, (_display, adapter_name) in enumerate(self._choices):
            if adapter_name == (name or ""):
                self.comboBox.blockSignals(True)
                self.comboBox.setCurrentIndex(i)
                self.comboBox.blockSignals(False)
                return
        self.refresh(name)

    def display_text(self) -> str:
        return self.comboBox.currentText()
