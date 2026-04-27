"""Bag-input tab widget — open dialog, topic combo, transport bar.

The widget owns a BagReader and PlaybackController internally. It emits:
  - `frame_ready(BagFrame)` whenever a frame should be inferred + shown
  - `engine_config_changed()` if a fresh engine load is needed (currently
    not used by this tab — bag input doesn't change the engine — but we
    keep the signal for symmetry with the other input tabs).

The MainWindow connects `frame_ready` to InferenceWorker.submit_image,
and the worker's `is_busy()` is wired into PlaybackController as the
drop-frame predicate.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from scvterrascope.gui.playback import PlaybackController

LOG = logging.getLogger(__name__)

_SPEEDS = ("0.25x", "0.5x", "1.0x", "2.0x", "4.0x")
_SPEED_VALUES = {label: float(label.rstrip("x")) for label in _SPEEDS}


class RosBagTab(QWidget):
    """Self-contained ROS bag input tab.

    Lifecycle:
        - `open_bag(path)`        — load a bag dir, populate topic combo
        - on topic combo change   — attach reader+topic to PlaybackController
        - transport buttons       — drive PlaybackController
        - frame_ready signal      — out to MainWindow → InferenceWorker
    """

    frame_ready = pyqtSignal(object)  # BagFrame

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._reader: Any = None
        self.controller = PlaybackController(self)
        self.controller.frame_ready.connect(self.frame_ready.emit)
        self.controller.seek_changed.connect(self._on_seek_changed)
        self.controller.state_changed.connect(self._on_state_changed)
        # Internal flag to avoid emitting seek slider valueChanged in a feedback loop.
        self._suppress_seek = False
        self._build_ui()
        self._update_enabled(active=False)

    # ----- public API --------------------------------------------------
    def attach_busy_check(self, fn) -> None:
        """Wire InferenceWorker.is_busy as the drop-frame predicate."""
        self.controller.set_busy_check(fn)

    def close_reader(self) -> None:
        """Detach controller + close current bag (called on app exit)."""
        self.controller.detach()
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    # ----- UI construction --------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        # --- Bag selector + metadata
        meta = QGroupBox("Bag")
        meta_lay = QGridLayout(meta)
        self.btn_open = QPushButton("Open Bag…")
        self.btn_open.clicked.connect(self._open_dialog)
        self.lbl_bag = QLabel("(no bag loaded)")
        self.lbl_bag.setStyleSheet("color: gray;")
        self.lbl_bag.setWordWrap(True)
        self.lbl_meta = QLabel("")
        self.lbl_meta.setStyleSheet("color: gray;")
        meta_lay.addWidget(self.btn_open, 0, 0)
        meta_lay.addWidget(self.lbl_bag, 0, 1, 1, 2)
        meta_lay.addWidget(QLabel("Topic:"), 1, 0)
        self.topic_combo = QComboBox()
        self.topic_combo.currentIndexChanged.connect(self._on_topic_changed)
        meta_lay.addWidget(self.topic_combo, 1, 1, 1, 2)
        meta_lay.addWidget(self.lbl_meta, 2, 0, 1, 3)
        root.addWidget(meta)

        # --- Transport bar
        transport = QGroupBox("Playback")
        t_lay = QGridLayout(transport)

        self.btn_back = QPushButton("⏮ -1")
        self.btn_back.clicked.connect(lambda: self.controller.step(-1))
        self.btn_play = QPushButton("⏯ Play")
        self.btn_play.clicked.connect(self.controller.toggle)
        self.btn_fwd = QPushButton("⏭ +1")
        self.btn_fwd.clicked.connect(lambda: self.controller.step(+1))
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.btn_back)
        btn_row.addWidget(self.btn_play, 1)
        btn_row.addWidget(self.btn_fwd)
        t_lay.addLayout(btn_row, 0, 0, 1, 3)

        t_lay.addWidget(QLabel("Speed:"), 1, 0)
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(_SPEEDS)
        self.speed_combo.setCurrentText("1.0x")
        self.speed_combo.currentTextChanged.connect(
            lambda s: self.controller.set_speed(_SPEED_VALUES.get(s, 1.0))
        )
        t_lay.addWidget(self.speed_combo, 1, 1)

        # Seek bar + frame/time labels.
        self.seek = QSlider(Qt.Orientation.Horizontal)
        self.seek.setRange(0, 0)
        # Seek by mouse drag — only seek when the user releases (avoids
        # spamming inference while the slider is being dragged). Click
        # also ends in a release.
        self.seek.sliderReleased.connect(self._on_seek_released)
        # Live label update during drag without triggering inference.
        self.seek.valueChanged.connect(self._on_seek_value_changed)
        t_lay.addWidget(self.seek, 2, 0, 1, 3)

        self.lbl_position = QLabel("Frame: — / —    ROS time: —")
        t_lay.addWidget(self.lbl_position, 3, 0, 1, 3)

        root.addWidget(transport)
        root.addStretch(1)

    # ----- bag loading ------------------------------------------------
    def _open_dialog(self) -> None:
        # Default to ~/data — both SCV bag dirs live there in this env.
        start = str(Path.home() / "data") if (Path.home() / "data").is_dir() else str(Path.home())
        directory = QFileDialog.getExistingDirectory(self, "Open ROS 2 bag (directory)", start)
        if directory:
            self.open_bag(Path(directory))

    def open_bag(self, path: Path) -> None:
        from PyQt6.QtWidgets import QMessageBox

        from scvterrascope.rosbag import BagReader

        # Tear down previous bag (if any) cleanly.
        self.close_reader()

        try:
            reader = BagReader(path).open()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Open bag failed", f"{path}\n{exc}")
            return
        topics = reader.image_topics()
        if not topics:
            QMessageBox.warning(
                self, "No image topics",
                f"No sensor_msgs/Image topics found in:\n{path}",
            )
            reader.close()
            return

        self._reader = reader
        self.lbl_bag.setText(str(path))
        self.lbl_bag.setStyleSheet("")
        # Populate topic combo. Block signals while we build to avoid
        # firing _on_topic_changed multiple times during construction.
        self.topic_combo.blockSignals(True)
        self.topic_combo.clear()
        for t in topics:
            self.topic_combo.addItem(f"{t.topic}  ({t.count} frames)", t.topic)
        self.topic_combo.setCurrentIndex(0)
        self.topic_combo.blockSignals(False)

        self._update_meta_label()
        self._update_enabled(active=True)
        # Trigger initial topic attach.
        self._on_topic_changed(0)

    def _on_topic_changed(self, _idx: int) -> None:
        if self._reader is None or self.topic_combo.count() == 0:
            return
        topic = self.topic_combo.currentData()
        if not topic:
            return
        self.controller.attach(self._reader, topic)
        self._update_meta_label()

    # ----- transport feedback -----------------------------------------
    def _on_seek_changed(self, current: int, total: int) -> None:
        self._suppress_seek = True
        self.seek.setRange(0, max(0, total - 1))
        self.seek.setValue(current)
        self._suppress_seek = False
        self._update_position_label(current, total)

    def _on_state_changed(self, state: str) -> None:
        self.btn_play.setText("⏸ Pause" if state == "playing" else "⏯ Play")

    def _on_seek_value_changed(self, value: int) -> None:
        if self._suppress_seek:
            return
        # Update label live during drag — but don't seek until release.
        self._update_position_label(value, max(0, self.seek.maximum() + 1))

    def _on_seek_released(self) -> None:
        self.controller.seek_to(self.seek.value())

    # ----- helpers ----------------------------------------------------
    def _update_meta_label(self) -> None:
        if self._reader is None:
            self.lbl_meta.setText("")
            return
        topic = self.topic_combo.currentData()
        info = next((t for t in self._reader.image_topics() if t.topic == topic), None)
        if info is None:
            self.lbl_meta.setText(f"duration: {self._reader.duration_seconds():.1f}s")
            return
        self.lbl_meta.setText(
            f"{info.msgtype}  ·  {info.count} frames  ·  "
            f"{self._reader.duration_seconds():.1f}s total"
        )

    def _update_position_label(self, current: int, total: int) -> None:
        if self._reader is None or total == 0:
            self.lbl_position.setText("Frame: — / —    ROS time: —")
            return
        # Approximate ROS time using the bag's metadata duration (seek
        # via timestamp index is exact but we'd need to query it again —
        # this is just for the label).
        per = self._reader.duration_seconds() / max(1, total - 1) if total > 1 else 0.0
        self.lbl_position.setText(
            f"Frame: {current} / {total - 1}    "
            f"ROS time: {current * per:.2f}s / {self._reader.duration_seconds():.2f}s"
        )

    def _update_enabled(self, *, active: bool) -> None:
        for w in (
            self.btn_back, self.btn_play, self.btn_fwd,
            self.speed_combo, self.seek, self.topic_combo,
        ):
            w.setEnabled(active)
