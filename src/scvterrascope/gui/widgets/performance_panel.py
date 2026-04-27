"""Performance panel — last-frame timing, rolling FPS, GPU memory.

The panel updates from `MainWindow._on_inference_finished` after every
prediction. Rolling stats use a fixed-size deque so the very first
warmup-bound inference doesn't permanently skew the average.
"""

from __future__ import annotations

import statistics
from collections import deque
from typing import Iterable

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from scvterrascope.inference.engine import InferenceResult


class PerformancePanel(QWidget):
    """Right-dock metrics for the last frame + rolling averages."""

    DEFAULT_HISTORY = 30  # frames of rolling stats

    def __init__(self, history: int = DEFAULT_HISTORY, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._history: deque[InferenceResult] = deque(maxlen=history)
        self._history_size = history

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        layout.addWidget(self._build_last_frame_box())
        layout.addWidget(self._build_rolling_box())
        layout.addWidget(self._build_gpu_box())
        layout.addWidget(self._build_model_box())
        layout.addStretch(1)

    # ---- builders ------------------------------------------------
    def _build_last_frame_box(self) -> QGroupBox:
        box = QGroupBox("Last frame")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.lbl_pre = self._mk_metric()
        self.lbl_fwd = self._mk_metric()
        self.lbl_post = self._mk_metric()
        self.lbl_total = self._mk_metric(bold=True)
        self.lbl_fps = self._mk_metric(bold=True)
        self.lbl_dets = self._mk_metric()
        form.addRow("preprocess:", self.lbl_pre)
        form.addRow("forward:", self.lbl_fwd)
        form.addRow("postprocess:", self.lbl_post)
        form.addRow("total:", self.lbl_total)
        form.addRow("FPS:", self.lbl_fps)
        form.addRow("detections:", self.lbl_dets)
        return box

    def _build_rolling_box(self) -> QGroupBox:
        box = QGroupBox(f"Rolling avg (last {self._history_size})")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.lbl_avg_total = self._mk_metric()
        self.lbl_avg_fps = self._mk_metric(bold=True)
        self.lbl_range = self._mk_metric()
        self.lbl_count = self._mk_metric()
        form.addRow("total:", self.lbl_avg_total)
        form.addRow("FPS:", self.lbl_avg_fps)
        form.addRow("range:", self.lbl_range)
        form.addRow("samples:", self.lbl_count)
        return box

    def _build_gpu_box(self) -> QGroupBox:
        box = QGroupBox("GPU memory")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.lbl_mem_alloc = self._mk_metric()
        self.lbl_mem_peak = self._mk_metric()
        self.lbl_mem_total = self._mk_metric()
        self.bar_mem = QProgressBar()
        self.bar_mem.setRange(0, 100)
        self.bar_mem.setFormat("%p%  (peak)")
        self.bar_mem.setTextVisible(True)
        form.addRow("allocated:", self.lbl_mem_alloc)
        form.addRow("peak:", self.lbl_mem_peak)
        form.addRow("total:", self.lbl_mem_total)
        form.addRow("usage:", self.bar_mem)
        return box

    def _build_model_box(self) -> QGroupBox:
        box = QGroupBox("Model / device")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.lbl_device = self._mk_metric()
        self.lbl_params = self._mk_metric()
        self.lbl_image = self._mk_metric()
        form.addRow("device:", self.lbl_device)
        form.addRow("params:", self.lbl_params)
        form.addRow("image (H×W):", self.lbl_image)
        return box

    @staticmethod
    def _mk_metric(*, bold: bool = False) -> QLabel:
        lbl = QLabel("—")
        lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if bold:
            f = QFont()
            f.setBold(True)
            lbl.setFont(f)
        return lbl

    # ---- public api ----------------------------------------------
    def reset(self) -> None:
        """Clear history. Call when the engine reloads (different model)."""
        self._history.clear()
        for lbl in (
            self.lbl_pre, self.lbl_fwd, self.lbl_post, self.lbl_total,
            self.lbl_fps, self.lbl_dets, self.lbl_avg_total, self.lbl_avg_fps,
            self.lbl_range, self.lbl_count, self.lbl_mem_alloc, self.lbl_mem_peak,
        ):
            lbl.setText("—")
        self.bar_mem.setValue(0)

    def set_static_info(
        self,
        *,
        device: str,
        device_name: str,
        param_count: int,
        gpu_total_mb: float,
        image_size: int,
    ) -> None:
        """Called once after engine.load() succeeds. Static across predictions."""
        if device_name:
            self.lbl_device.setText(f"{device}  ({device_name})")
        else:
            self.lbl_device.setText(device)
        self.lbl_params.setText(f"{param_count / 1e6:.2f} M")
        self.lbl_image.setText(f"{image_size} × {image_size} (letterbox)")
        if gpu_total_mb > 0:
            self.lbl_mem_total.setText(f"{gpu_total_mb:,.0f} MB")
        else:
            self.lbl_mem_total.setText("— (CPU)")

    def update_from_result(self, result: InferenceResult) -> None:
        self._history.append(result)
        # Last-frame fields.
        self.lbl_pre.setText(f"{result.preprocess_ms:>6.1f} ms")
        self.lbl_fwd.setText(f"{result.inference_ms:>6.1f} ms")
        self.lbl_post.setText(f"{result.postprocess_ms:>6.1f} ms")
        self.lbl_total.setText(f"{result.total_ms:>6.1f} ms")
        self.lbl_fps.setText(f"{result.fps:>6.2f}")
        self.lbl_dets.setText(f"{len(result.detections)}")

        # Rolling stats.
        totals = [r.total_ms for r in self._history]
        self.lbl_avg_total.setText(f"{statistics.fmean(totals):>6.1f} ms")
        avg_fps = 1000.0 / statistics.fmean(totals) if totals else 0.0
        self.lbl_avg_fps.setText(f"{avg_fps:>6.2f}")
        self.lbl_range.setText(f"{min(totals):.0f}–{max(totals):.0f} ms")
        self.lbl_count.setText(f"{len(self._history)} / {self._history_size}")

        # GPU memory.
        if result.gpu_mem_total_mb > 0:
            self.lbl_mem_alloc.setText(f"{result.gpu_mem_alloc_mb:,.0f} MB")
            self.lbl_mem_peak.setText(f"{result.gpu_mem_peak_mb:,.0f} MB")
            pct = int(round(100 * result.gpu_mem_peak_mb / result.gpu_mem_total_mb))
            self.bar_mem.setValue(min(100, max(0, pct)))


def format_history_summary(results: Iterable[InferenceResult]) -> str:
    """Standalone helper — useful for logging or future export."""
    items = list(results)
    if not items:
        return "no inferences yet"
    totals = [r.total_ms for r in items]
    return (
        f"n={len(items)}  "
        f"mean={statistics.fmean(totals):.1f}ms  "
        f"min={min(totals):.1f}  max={max(totals):.1f}  "
        f"FPS_mean={1000.0 / statistics.fmean(totals):.2f}"
    )
