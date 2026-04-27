"""Table view of detections (class, score, bbox)."""

from __future__ import annotations

from typing import Iterable

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHeaderView, QTableWidget, QTableWidgetItem

from scvterrascope.inference.engine import Detection


class ResultsTable(QTableWidget):
    """Columns: class, score, x1, y1, x2, y2.

    Emits `row_selected(index)` so the main window can highlight the
    matching bbox on the canvas.
    """

    row_selected = pyqtSignal(int)

    HEADERS = ("class", "score", "x1", "y1", "x2", "y2")

    def __init__(self, parent=None) -> None:
        super().__init__(0, len(self.HEADERS), parent)
        self.setHorizontalHeaderLabels(self.HEADERS)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.itemSelectionChanged.connect(self._on_selection)

    def set_detections(
        self,
        detections: Iterable[Detection],
        palette: list[tuple[int, int, int]] | None = None,
    ) -> None:
        rows = list(detections)
        self.setRowCount(len(rows))
        for r, det in enumerate(rows):
            x1, y1, x2, y2 = det.bbox_xyxy
            cells = (
                det.class_name,
                f"{det.score:.3f}",
                f"{x1:.1f}",
                f"{y1:.1f}",
                f"{x2:.1f}",
                f"{y2:.1f}",
            )
            for c, value in enumerate(cells):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if c == 0 and palette is not None:
                    rgb = palette[(det.class_id - 1) % len(palette)]
                    item.setForeground(QColor(*rgb))
                self.setItem(r, c, item)

    def _on_selection(self) -> None:
        items = self.selectedItems()
        if not items:
            self.row_selected.emit(-1)
            return
        self.row_selected.emit(items[0].row())
