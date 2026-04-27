"""Left dock — checkpoint selector, threshold slider, class filter."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class ControlPanel(QWidget):
    """All inference parameters live here.

    Threshold and class-filter changes emit `display_filters_changed` so
    the main window can re-draw without re-running inference. Checkpoint
    and image-size changes emit `engine_config_changed` and require a
    model reload.
    """

    run_inference_requested = pyqtSignal()
    display_filters_changed = pyqtSignal()  # threshold / class filter
    engine_config_changed = pyqtSignal()    # checkpoint / top_k
    preprocess_changed = pyqtSignal()       # aspect-crop mode flipped
    model_changed = pyqtSignal(str)         # new model_kind selected

    def __init__(self, class_names: Sequence[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._class_names = list(class_names)
        self._class_checkboxes: list[QCheckBox] = []

        layout = QVBoxLayout(self)

        # Input controls (Open Image / Open Folder / Open Bag) live in
        # the top-tabbed Input dock now. This panel only owns engine,
        # preprocessing, display filters, and class toggles.
        layout.addWidget(self._build_engine_box())
        layout.addWidget(self._build_preprocess_box())
        layout.addWidget(self._build_display_box())
        layout.addWidget(self._build_class_box())
        layout.addWidget(self._build_progress_box())
        layout.addStretch(1)

    # ---- builders ------------------------------------------------
    def _build_engine_box(self) -> QGroupBox:
        from scvterrascope.inference import model_choices

        box = QGroupBox("Engine")
        lay = QGridLayout(box)

        lay.addWidget(QLabel("Model:"), 0, 0)
        self.model_combo = QComboBox()
        for key, label in model_choices():
            self.model_combo.addItem(label, key)
        self.model_combo.setToolTip(
            "Inference backend.\n"
            "  DINOv3+DETR: SCVTerraVision-trained, 16 CODa classes.\n"
            "  YOLO11/12:   Ultralytics pretrained, 80 COCO classes.\n"
            "Switching reloads the model — first YOLO use auto-downloads weights."
        )
        self.model_combo.currentIndexChanged.connect(self._on_model_combo_changed)
        lay.addWidget(self.model_combo, 0, 1)

        lay.addWidget(QLabel("Checkpoint:"), 1, 0)
        self.ckpt_edit = QLineEdit()
        self.ckpt_edit.setPlaceholderText("DINOv3+DETR: path/to/epoch_050.pt    YOLO: optional override")
        self.ckpt_edit.editingFinished.connect(self.engine_config_changed.emit)
        browse = QPushButton("Browse")
        browse.clicked.connect(self._browse_ckpt)
        row = QHBoxLayout()
        row.addWidget(self.ckpt_edit, 1)
        row.addWidget(browse)
        lay.addLayout(row, 1, 1)

        lay.addWidget(QLabel("Top-K:"), 2, 0)
        self.topk_spin = QSpinBox()
        self.topk_spin.setRange(10, 300)
        self.topk_spin.setValue(100)
        self.topk_spin.valueChanged.connect(self.engine_config_changed.emit)
        lay.addWidget(self.topk_spin, 2, 1)

        run_btn = QPushButton("Run Inference")
        run_btn.clicked.connect(self.run_inference_requested.emit)
        lay.addWidget(run_btn, 3, 0, 1, 2)
        return box

    def _on_model_combo_changed(self, _idx: int) -> None:
        kind = self.model_combo.currentData()
        if kind:
            self.model_changed.emit(kind)

    def _build_preprocess_box(self) -> QGroupBox:
        box = QGroupBox("Preprocess")
        lay = QGridLayout(box)

        lay.addWidget(QLabel("Letterbox pad:"), 0, 0)
        self.pad_combo = QComboBox()
        for label, value in (
            ("Auto (recommended)", "auto"),
            ("Symmetric (center)", "symmetric"),
            ("Bottom-right (training)", "bottom_right"),
        ):
            self.pad_combo.addItem(label, value)
        self.pad_combo.setToolTip(
            "Where letterbox padding goes inside the 1024×1024 input.\n"
            "  Auto:        symmetric for OOD aspects, bottom-right for ≈training.\n"
            "  Symmetric:   centers content vertically — best for 16:9 dashcam.\n"
            "  Bottom-right: byte-equivalent to training preprocess."
        )
        self.pad_combo.currentIndexChanged.connect(self.preprocess_changed.emit)
        lay.addWidget(self.pad_combo, 0, 1)

        lay.addWidget(QLabel("Aspect crop:"), 1, 0)
        self.crop_combo = QComboBox()
        for label, value in (
            ("None (recommended)", "none"),
            ("Auto", "auto"),
            ("Always", "always"),
        ):
            self.crop_combo.addItem(label, value)
        self.crop_combo.setToolTip(
            "(Optional) Center-crop input to training aspect (~1.20:1) BEFORE\n"
            "letterboxing. Crop loses ~33% pixels for 16:9 sources, so the\n"
            "default is None — letterbox-symmetric is preferred."
        )
        self.crop_combo.currentIndexChanged.connect(self.preprocess_changed.emit)
        lay.addWidget(self.crop_combo, 1, 1)

        return box

    def _build_display_box(self) -> QGroupBox:
        box = QGroupBox("Display")
        lay = QGridLayout(box)
        lay.addWidget(QLabel("Score ≥"), 0, 0)
        self.score_slider = QSlider(Qt.Orientation.Horizontal)
        self.score_slider.setRange(0, 100)
        self.score_slider.setValue(30)
        self.score_value = QLabel("0.30")
        self.score_slider.valueChanged.connect(self._on_score_changed)
        lay.addWidget(self.score_slider, 0, 1)
        lay.addWidget(self.score_value, 0, 2)
        return box

    def _build_class_box(self) -> QGroupBox:
        from PyQt6.QtWidgets import QScrollArea

        box = QGroupBox("Classes (uncheck to hide)")
        outer = QVBoxLayout(box)
        self._class_box_inner = QWidget()
        self._class_grid = QGridLayout(self._class_box_inner)
        self._populate_class_grid()
        # 80 COCO classes is too tall for the dock — wrap in a scroll area.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._class_box_inner)
        scroll.setFixedHeight(220)
        outer.addWidget(scroll)
        return box

    def _populate_class_grid(self) -> None:
        # Clear any prior checkboxes (during set_class_names rebuild).
        for cb in self._class_checkboxes:
            cb.setParent(None)
            cb.deleteLater()
        self._class_checkboxes = []
        cols = 2
        for idx, name in enumerate(self._class_names):
            cb = QCheckBox(name)
            cb.setChecked(True)
            cb.stateChanged.connect(self.display_filters_changed.emit)
            self._class_checkboxes.append(cb)
            self._class_grid.addWidget(cb, idx // cols, idx % cols)

    def set_class_names(self, names: Sequence[str]) -> None:
        """Rebuild the class-filter grid (called when the engine changes)."""
        self._class_names = list(names)
        self._populate_class_grid()

    def _build_progress_box(self) -> QGroupBox:
        box = QGroupBox("Progress")
        lay = QVBoxLayout(box)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        lay.addWidget(self.progress)
        return box

    # ---- accessors -----------------------------------------------
    def score_threshold(self) -> float:
        return self.score_slider.value() / 100.0

    def class_filter(self) -> list[str] | None:
        enabled = [cb.text() for cb in self._class_checkboxes if cb.isChecked()]
        # `None` means "no filter at all" — keep separate from "all 16 enabled"
        # so the GUI is explicit when nothing is filtered.
        if len(enabled) == len(self._class_checkboxes):
            return None
        return enabled

    def checkpoint_path(self) -> Path | None:
        text = self.ckpt_edit.text().strip()
        return Path(text).expanduser() if text else None

    def set_checkpoint_path(self, path: Path | None) -> None:
        self.ckpt_edit.setText(str(path) if path else "")

    def top_k(self) -> int:
        return self.topk_spin.value()

    def set_top_k(self, value: int) -> None:
        self.topk_spin.setValue(value)

    def aspect_crop_mode(self) -> str:
        return self.crop_combo.currentData()

    def set_aspect_crop_mode(self, mode: str) -> None:
        for i in range(self.crop_combo.count()):
            if self.crop_combo.itemData(i) == mode:
                self.crop_combo.setCurrentIndex(i)
                return

    def pad_position(self) -> str:
        return self.pad_combo.currentData()

    def set_pad_position(self, mode: str) -> None:
        for i in range(self.pad_combo.count()):
            if self.pad_combo.itemData(i) == mode:
                self.pad_combo.setCurrentIndex(i)
                return

    def model_kind(self) -> str:
        return self.model_combo.currentData()

    def set_model_kind(self, kind: str) -> None:
        for i in range(self.model_combo.count()):
            if self.model_combo.itemData(i) == kind:
                # Block signals so programmatic init doesn't fire model_changed.
                self.model_combo.blockSignals(True)
                self.model_combo.setCurrentIndex(i)
                self.model_combo.blockSignals(False)
                return

    def set_progress(self, current: int, total: int) -> None:
        if total <= 0:
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
            return
        self.progress.setRange(0, total)
        self.progress.setValue(current)

    # ---- internal ------------------------------------------------
    def _on_score_changed(self, value: int) -> None:
        self.score_value.setText(f"{value / 100.0:.2f}")
        self.display_filters_changed.emit()

    def _browse_ckpt(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select checkpoint", "", "PyTorch checkpoints (*.pt *.pth);;All files (*)"
        )
        if path:
            self.ckpt_edit.setText(path)
            self.engine_config_changed.emit()
