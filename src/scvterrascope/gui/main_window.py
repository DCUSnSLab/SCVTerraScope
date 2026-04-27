"""SCVTerraScope main window — wires widgets, engine, and worker together."""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QStatusBar,
)

from scvterrascope.gui.config import AppConfig
from scvterrascope.gui.widgets import (
    ControlPanel,
    FileList,
    ImageCanvas,
    PerformancePanel,
    ResultsTable,
)
from scvterrascope.gui.worker import InferenceWorker
from scvterrascope.inference import InferenceEngine, InferenceResult
from scvterrascope.labels import Taxonomy, load_taxonomy
from scvterrascope.visualization import draw_detections, palette_for

LOG = logging.getLogger(__name__)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class MainWindow(QMainWindow):
    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.setWindowTitle("SCVTerraScope — Object Detection Monitor")
        self.resize(1400, 900)
        self.cfg = config

        # Taxonomy is needed before the control panel can list classes;
        # if it can't load (no SCVTerraVision sibling), fall back to a
        # numbered placeholder so the GUI still launches for layout demos.
        self.taxonomy = self._safe_load_taxonomy()
        self.palette = list(palette_for(max(16, len(self.taxonomy))))

        self.engine: InferenceEngine | None = None
        self.worker: InferenceWorker | None = None

        # Results state — kept so the threshold slider can re-filter
        # without re-running inference.
        self._last_image: Image.Image | None = None
        self._last_result: InferenceResult | None = None
        self._last_image_path: Path | None = None

        self._build_widgets()
        self._build_menu()
        self._build_status_bar()
        self._wire_signals()

        if self.cfg.checkpoint_path:
            self.controls.set_checkpoint_path(self.cfg.checkpoint_path)
        self.controls.set_top_k(self.cfg.top_k)
        self.controls.set_aspect_crop_mode(self.cfg.aspect_crop_mode)
        self.controls.set_pad_position(self.cfg.pad_position)

    # ----------------------------------------------------------------
    # Construction
    # ----------------------------------------------------------------
    def _safe_load_taxonomy(self) -> Taxonomy:
        try:
            return load_taxonomy(path=self.cfg.taxonomy_path)
        except FileNotFoundError as exc:
            LOG.warning("could not load taxonomy: %s", exc)
            # The bundled YAML is the default; this branch only triggers
            # when the user passed an explicit `taxonomy_path` that doesn't
            # exist. Fall back to numbered placeholders so the GUI still
            # builds for layout demos.
            from scvterrascope.labels import OperationalClass, Taxonomy

            return Taxonomy(
                classes=tuple(
                    OperationalClass(id=i + 1, name=f"class_{i + 1}") for i in range(16)
                )
            )

    def _build_widgets(self) -> None:
        self.canvas = ImageCanvas(self)
        self.setCentralWidget(self.canvas)

        self.controls = ControlPanel(self.taxonomy.names(), self)
        left = QDockWidget("Controls", self)
        left.setWidget(self.controls)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, left)

        self.file_list = FileList(self)
        files_dock = QDockWidget("Files", self)
        files_dock.setWidget(self.file_list)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, files_dock)
        self.tabifyDockWidget(left, files_dock)
        left.raise_()

        self.results_table = ResultsTable(self)
        bottom = QDockWidget("Detections", self)
        bottom.setWidget(self.results_table)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, bottom)

        self.perf_panel = PerformancePanel(parent=self)
        perf_dock = QDockWidget("Performance", self)
        perf_dock.setWidget(self.perf_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, perf_dock)

    def _build_menu(self) -> None:
        bar = self.menuBar()
        file_menu = bar.addMenu("&File")

        open_img = QAction("Open Image…", self)
        open_img.triggered.connect(self._open_image_dialog)
        file_menu.addAction(open_img)

        open_dir = QAction("Open Folder…", self)
        open_dir.triggered.connect(self._open_folder_dialog)
        file_menu.addAction(open_dir)

        export = QAction("Export Result…", self)
        export.triggered.connect(self._export_result)
        file_menu.addAction(export)

        file_menu.addSeparator()
        quit_a = QAction("Quit", self)
        quit_a.triggered.connect(self.close)
        file_menu.addAction(quit_a)

    def _build_status_bar(self) -> None:
        bar = QStatusBar(self)
        self.setStatusBar(bar)
        self.status_device = QLabel("device: —")
        self.status_ckpt = QLabel("ckpt: —")
        self.status_time = QLabel("infer: —")
        bar.addWidget(self.status_device)
        bar.addPermanentWidget(self.status_ckpt)
        bar.addPermanentWidget(self.status_time)

    def _wire_signals(self) -> None:
        c = self.controls
        c.open_image_requested.connect(self._open_image_dialog)
        c.open_folder_requested.connect(self._open_folder_dialog)
        c.run_inference_requested.connect(self._run_current_image)
        c.display_filters_changed.connect(self._redraw_overlays)
        c.engine_config_changed.connect(self._invalidate_engine)
        c.preprocess_changed.connect(self._on_preprocess_changed)
        self.file_list.image_selected.connect(self._on_file_selected)
        self.results_table.row_selected.connect(self._on_row_selected)

    # ----------------------------------------------------------------
    # Engine lifecycle
    # ----------------------------------------------------------------
    def _ensure_engine(self) -> bool:
        ckpt = self.controls.checkpoint_path()
        if ckpt is None:
            QMessageBox.warning(self, "Checkpoint", "Set a checkpoint path first.")
            return False
        if (
            self.engine is not None
            and self.engine.is_loaded()
            and self.engine.checkpoint_path == ckpt
            and self.engine.top_k == self.controls.top_k()
        ):
            return True
        try:
            self.engine = InferenceEngine(
                checkpoint_path=ckpt,
                device=self.cfg.device,
                image_size=self.cfg.image_size,
                top_k=self.controls.top_k(),
                taxonomy=self.taxonomy,
                aspect_crop_mode=self.controls.aspect_crop_mode(),
                pad_position=self.controls.pad_position(),
            )
            self.statusBar().showMessage("Loading checkpoint…", 0)
            self.engine.load()
        except Exception as exc:
            QMessageBox.critical(self, "Engine load failed", str(exc))
            return False
        self.status_device.setText(f"device: {self.engine.device}")
        self.status_ckpt.setText(f"ckpt: {ckpt.name} (epoch={self.engine.epoch})")
        self.statusBar().showMessage("Ready.", 3000)
        # Push static info into the performance panel — it stays the same
        # across predictions, only timings/memory change per-frame.
        self.perf_panel.reset()
        self.perf_panel.set_static_info(
            device=self.engine.device,
            device_name=self.engine.device_name,
            param_count=self.engine.param_count,
            gpu_total_mb=self.engine.gpu_total_mb,
            image_size=self.cfg.image_size,
        )
        # (Re)build worker for the new engine.
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(2000)
        self.worker = InferenceWorker(self.engine, self)
        self.worker.signals.finished.connect(self._on_inference_finished)
        self.worker.signals.failed.connect(self._on_inference_failed)
        self.worker.signals.progress.connect(self.controls.set_progress)
        return True

    def _invalidate_engine(self) -> None:
        # Don't reload eagerly — wait until the next Run Inference click,
        # so a typo in the path doesn't cause a 30s GPU load every keystroke.
        if self.engine is not None:
            self.engine = None

    def _on_preprocess_changed(self) -> None:
        """Pre/letterbox toggles are cheap to flip — no model reload needed."""
        if self.engine is not None and self.engine.is_loaded():
            self.engine.aspect_crop_mode = self.controls.aspect_crop_mode()
            self.engine.pad_position = self.controls.pad_position()
            if self._last_image_path is not None:
                self._run_current_image()

    # ----------------------------------------------------------------
    # File I/O
    # ----------------------------------------------------------------
    def _open_image_dialog(self) -> None:
        start = str(self.cfg.samples_dir if self.cfg.samples_dir.is_dir() else Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self, "Open image", start, "Images (*.jpg *.jpeg *.png *.bmp);;All files (*)"
        )
        if path:
            self._load_single(Path(path))

    def _open_folder_dialog(self) -> None:
        start = str(self.cfg.samples_dir if self.cfg.samples_dir.is_dir() else Path.home())
        directory = QFileDialog.getExistingDirectory(self, "Open folder", start)
        if directory:
            self._load_folder(Path(directory))

    def _load_single(self, path: Path) -> None:
        self.file_list.set_paths([path])
        self._show_image(path, run=True)

    def _load_folder(self, directory: Path) -> None:
        # Recursive walk catches CODa's `2d_rect/cam0/<SEQ>/*.jpg` layout.
        files = sorted(
            p
            for p in directory.rglob("*")
            if p.suffix.lower() in IMAGE_EXTS
        )
        if not files:
            QMessageBox.information(self, "No images", f"No images under {directory}.")
            return
        self.file_list.set_paths(files)
        self._show_image(files[0], run=True)

    def _on_file_selected(self, path: str) -> None:
        self._show_image(Path(path), run=True)

    def _show_image(self, path: Path, *, run: bool) -> None:
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            QMessageBox.critical(self, "Open image failed", f"{path}\n{exc}")
            return
        self._last_image = image
        self._last_image_path = path
        self._last_result = None
        self.canvas.set_image(image)
        self.results_table.set_detections([], palette=self.palette)
        self.statusBar().showMessage(f"Loaded {path.name}", 3000)
        if run:
            self._run_current_image()

    # ----------------------------------------------------------------
    # Inference
    # ----------------------------------------------------------------
    def _run_current_image(self) -> None:
        if self._last_image_path is None:
            return
        if not self._ensure_engine():
            return
        assert self.worker is not None
        self.worker.submit([self._last_image_path])

    def _on_inference_finished(self, image_path: str, result: InferenceResult) -> None:
        # Always feed the performance panel — folder mode runs many frames
        # and we want rolling stats even when the user has switched canvas.
        self.perf_panel.update_from_result(result)
        if Path(image_path) != self._last_image_path:
            return  # user moved on; stale callback for canvas/table only
        self._last_result = result
        self.status_time.setText(
            f"total: {result.total_ms:.0f} ms ({result.fps:.1f} FPS)"
        )
        self._redraw_overlays()

    def _on_inference_failed(self, image_path: str, message: str) -> None:
        if Path(image_path) != self._last_image_path:
            return
        QMessageBox.critical(self, "Inference failed", f"{image_path}\n{message}")

    def _redraw_overlays(self, *, highlight_index: int | None = None) -> None:
        if self._last_image is None:
            return
        if self._last_result is None:
            self.canvas.set_image(self._last_image, fit=False)
            self.results_table.set_detections([], palette=self.palette)
            return
        threshold = self.controls.score_threshold()
        class_filter = self.controls.class_filter()
        rendered = draw_detections(
            self._last_image,
            self._last_result.detections,
            palette=self.palette,
            score_threshold=threshold,
            class_filter=class_filter,
            highlight_index=highlight_index,
        )
        self.canvas.set_image(rendered, fit=False)
        # Table shows the unfiltered list so the user sees "what would be
        # there at threshold=0.0" — the slider hides them in the canvas
        # only.
        self.results_table.set_detections(
            self._last_result.detections, palette=self.palette
        )

    def _on_row_selected(self, idx: int) -> None:
        if idx < 0 or self._last_result is None:
            self._redraw_overlays()
            return
        self._redraw_overlays(highlight_index=idx)

    # ----------------------------------------------------------------
    # Export
    # ----------------------------------------------------------------
    def _export_result(self) -> None:
        if self._last_result is None or self._last_image is None:
            QMessageBox.information(self, "Export", "Nothing to export yet.")
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Export to")
        if not out_dir:
            return
        out = Path(out_dir)
        stem = self._last_image_path.stem if self._last_image_path else "result"
        rendered = draw_detections(
            self._last_image,
            self._last_result.detections,
            palette=self.palette,
            score_threshold=self.controls.score_threshold(),
            class_filter=self.controls.class_filter(),
        )
        rendered.save(out / f"{stem}_overlay.png")
        import json

        preds = [
            {
                "class_id": d.class_id,
                "class_name": d.class_name,
                "score": d.score,
                "bbox_xyxy": list(d.bbox_xyxy),
            }
            for d in self._last_result.detections
        ]
        (out / f"{stem}_detections.json").write_text(
            json.dumps(preds, indent=2), encoding="utf-8"
        )
        self.statusBar().showMessage(f"Exported to {out}", 5000)

    # ----------------------------------------------------------------
    # Cleanup
    # ----------------------------------------------------------------
    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(2000)
        super().closeEvent(event)
