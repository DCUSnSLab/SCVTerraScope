"""Folder input tab — Open Folder button + recursive file list."""

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

from scvterrascope.gui.widgets.file_list import FileList

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class FolderTab(QWidget):
    """Tab for folder/batch input.

    Emits:
      - `folder_selected(paths)` once after an Open Folder dialog.
      - `image_selected(path)` whenever the user clicks a thumbnail row.
    """

    folder_selected = pyqtSignal(list)   # list[str]
    image_selected = pyqtSignal(str)     # absolute path

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        box = QGroupBox("Folder")
        lay = QVBoxLayout(box)
        btn = QPushButton("Open Folder…")
        btn.clicked.connect(self._open_dialog)
        lay.addWidget(btn)
        self.lbl = QLabel("(no folder loaded)")
        self.lbl.setStyleSheet("color: gray;")
        self.lbl.setWordWrap(True)
        lay.addWidget(self.lbl)
        root.addWidget(box)

        self.file_list = FileList(self)
        # Surface the click as an image_selected signal at this tab's level.
        self.file_list.image_selected.connect(self.image_selected.emit)
        root.addWidget(self.file_list, 1)

    def _open_dialog(self) -> None:
        from PyQt6.QtWidgets import QMessageBox

        start = str(Path.home() / "data") if (Path.home() / "data").is_dir() else str(Path.home())
        directory = QFileDialog.getExistingDirectory(self, "Open folder", start)
        if not directory:
            return
        d = Path(directory)
        files = sorted(p for p in d.rglob("*") if p.suffix.lower() in _IMG_EXTS)
        if not files:
            QMessageBox.information(self, "No images", f"No images under {d}.")
            return
        self.file_list.set_paths(files)
        self.lbl.setText(f"{d}  ({len(files)} images)")
        self.lbl.setStyleSheet("")
        self.folder_selected.emit([str(p) for p in files])
