"""Bottom-left file list for folder mode."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QListWidget, QListWidgetItem


class FileList(QListWidget):
    """A flat list of image paths. Click → emit `image_selected`."""

    image_selected = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.itemClicked.connect(self._on_clicked)

    def set_paths(self, paths: Sequence[Path]) -> None:
        self.clear()
        for p in paths:
            item = QListWidgetItem(p.name)
            item.setData(0x100, str(p))  # Qt.ItemDataRole.UserRole == 0x100
            item.setToolTip(str(p))
            self.addItem(item)

    def _on_clicked(self, item: QListWidgetItem) -> None:
        self.image_selected.emit(item.data(0x100))
