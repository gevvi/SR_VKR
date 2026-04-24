from PyQt6.QtCore import QThread, pyqtSignal

from config.settings import ProcessingSettings
from controllers.app_controller import AppController


class ProcessingWorker(QThread):
    progress_changed = pyqtSignal(int, int, str)
    finished_success = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, settings: ProcessingSettings):
        super().__init__()
        self.settings = settings
        self.controller = AppController()

    def run(self):
        try:
            results = self.controller.process_batch(self.settings, self._emit_progress)
            self.finished_success.emit(results)
        except Exception as exc:
            self.failed.emit(str(exc))

    def _emit_progress(self, current: int, total: int, filename: str):
        self.progress_changed.emit(current, total, filename)
