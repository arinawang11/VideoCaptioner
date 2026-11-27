# coding:utf-8
"""HuggingFace (hf-mirror) model download thread."""

from pathlib import Path

from PyQt5.QtCore import QThread, pyqtSignal

from videocaptioner.core.utils.logger import setup_logger
from videocaptioner.core.utils.model_downloader import download_hf_model

logger = setup_logger("hf_model_download_thread")


class HfModelDownloadThread(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, repo_id: str, save_path, label: str = ""):
        super().__init__()
        self.repo_id = repo_id
        self.save_path = Path(save_path)
        self.label = label

    def run(self):
        try:
            self.progress.emit(0, "starting download...")
            download_hf_model(
                self.repo_id,
                self.save_path,
                progress=lambda value, message: self.progress.emit(value, message),
                label=self.label,
            )
            self.progress.emit(100, "download complete")
            self.finished.emit()
        except Exception as e:
            logger.exception("Model download failed: %s", e)
            self.error.emit(str(e))
