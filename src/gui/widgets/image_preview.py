from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QLabel, QSizePolicy


class ImagePreview(QLabel):
    def __init__(self, title: str):
        super().__init__(title)
        self._title = title
        self._source_pixmap: QPixmap | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet('border: 1px solid #bcbcbc; background: #fafafa; color: #666; padding: 8px;')
        self._show_placeholder()

    def _show_placeholder(self):
        self._source_pixmap = None
        super().setPixmap(QPixmap())
        self.setText(self._title)

    def set_preview(self, pixmap):
        if pixmap is None or pixmap.isNull():
            self._show_placeholder()
            return
        self._source_pixmap = pixmap
        scaled = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setText('')
        super().setPixmap(scaled)

    def resizeEvent(self, event):
        if self._source_pixmap is not None and not self._source_pixmap.isNull():
            self.set_preview(self._source_pixmap)
        super().resizeEvent(event)