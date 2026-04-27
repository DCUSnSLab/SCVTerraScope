"""Single-image input tab — just an Open button + the loaded path."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

_EXTS = "Images (*.jpg *.jpeg *.png *.bmp);;All files (*)"


class ImageTab(QWidget):
    """Tab for single-image input. Emits `image_selected(path)`."""

    image_selected = pyqtSignal(str)  # absolute path

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()
        self._loaded: Path | None = None

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        box = QGroupBox("Single image")
        lay = QVBoxLayout(box)
        btn = QPushButton("Open Image…")
        btn.clicked.connect(self._open_dialog)
        lay.addWidget(btn)
        self.lbl = QLabel("(no image loaded)")
        self.lbl.setStyleSheet("color: gray;")
        self.lbl.setWordWrap(True)
        lay.addWidget(self.lbl)
        root.addWidget(box)
        root.addStretch(1)

    def _open_dialog(self) -> None:
        # Default to ~/data when present (matches Phase 1-1 UX).
        start = str(Path.home() / "data") if (Path.home() / "data").is_dir() else str(Path.home())
        path, _ = QFileDialog.getOpenFileName(self, "Open image", start, _EXTS)
        if path:
            self._loaded = Path(path)
            self.lbl.setText(str(path))
            self.lbl.setStyleSheet("")
            self.image_selected.emit(path)

    def loaded_path(self) -> Path | None:
        return self._loaded
