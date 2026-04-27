"""SCVTerraScope main window — wires widgets, engine, and worker together.

Layout (Phase 2):
    ┌─ Menu: File ─────────────────────────────────────────┐
    │ ┌ Input dock ┐ ┌─ Image canvas (central) ─┐ ┌ Perf ┐ │
    │ │ [Image]    │ │                           │ │       │ │
    │ │ [Folder]   │ │  PIL image + bbox overlay │ │ rolling│
    │ │ [ROS Bag]  │ │                           │ │  + GPU │
    │ └────────────┘ └───────────────────────────┘ └───────┘ │
    │ ┌ Controls dock ┐                                       │
    │ │ Engine        │                                       │
    │ │ Preprocess    │                                       │
    │ │ Display       │                                       │
    │ │ Classes       │                                       │
    │ │ Progress      │                                       │
    │ └───────────────┘                                       │
    │ ┌── Detections dock ───────────────────────────────────┐│
    │ │ class | score | bbox                                 ││
    │ └─────────────────────────────────────────────────────┘ │
    └─────────────────────────────────────────────────────────┘

The Input dock is a QTabWidget with one tab per input source. ROS Bag
tab carries its own playback transport; switching to it does NOT
re-route the InferenceWorker — the worker is shared across all input
modes via `submit(paths)` (folder mode) and `submit_image(pil, tag)`
(single image / bag mode).
"""

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
    QTabWidget,
)

from scvterrascope.gui.config import AppConfig
from scvterrascope.gui.widgets import (
    ControlPanel,
    FolderTab,
    ImageCanvas,
    ImageTab,
    PerformancePanel,
    ResultsTable,
    RosBagTab,
)
from scvterrascope.gui.worker import InferenceWorker
from scvterrascope.inference import (
    BaseInferenceEngine,
    InferenceResult,
    build_engine,
    family_for,
)
from scvterrascope.labels import Taxonomy, load_taxonomy
from scvterrascope.visualization import draw_detections, palette_for

LOG = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.setWindowTitle("SCVTerraScope — Object Detection Monitor")
        self.resize(1500, 950)
        self.cfg = config

        self.taxonomy = self._safe_load_taxonomy()
        self.palette = list(palette_for(max(16, len(self.taxonomy))))

        self.engine: BaseInferenceEngine | None = None
        self.worker: InferenceWorker | None = None
        # `_engine_signature` lets _ensure_engine know whether the in-memory
        # engine still matches the user's controls (model_kind + checkpoint
        # + top_k) — switching any of them triggers a reload.
        self._engine_signature: tuple = ()

        # Inference state — single slot, shared across input modes.
        self._last_image: Image.Image | None = None
        self._last_result: InferenceResult | None = None
        # `_last_tag` is a string identifier for the most recent submission:
        # a Path string for image/folder mode, "bag:<topic>:<idx>" for bag
        # mode. We compare it on inference_finished to drop stale callbacks.
        self._last_tag: str | None = None

        self._build_widgets()
        self._build_menu()
        self._build_status_bar()
        self._wire_signals()

        if self.cfg.checkpoint_path:
            self.controls.set_checkpoint_path(self.cfg.checkpoint_path)
        self.controls.set_top_k(self.cfg.top_k)
        self.controls.set_aspect_crop_mode(self.cfg.aspect_crop_mode)
        self.controls.set_pad_position(self.cfg.pad_position)
        self.controls.set_model_kind(self.cfg.model_kind)

    # ----------------------------------------------------------------
    # Construction
    # ----------------------------------------------------------------
    def _safe_load_taxonomy(self) -> Taxonomy:
        try:
            return load_taxonomy(path=self.cfg.taxonomy_path)
        except FileNotFoundError as exc:
            LOG.warning("could not load taxonomy: %s", exc)
            from scvterrascope.labels import OperationalClass, Taxonomy

            return Taxonomy(
                classes=tuple(
                    OperationalClass(id=i + 1, name=f"class_{i + 1}") for i in range(16)
                )
            )

    def _build_widgets(self) -> None:
        self.canvas = ImageCanvas(self)
        self.setCentralWidget(self.canvas)

        # Input QTabWidget — one tab per input source. Sits in the upper
        # left dock area.
        self.tab_image = ImageTab(self)
        self.tab_folder = FolderTab(self)
        self.tab_bag = RosBagTab(self)

        self.input_tabs = QTabWidget(self)
        self.input_tabs.addTab(self.tab_image, "Image")
        self.input_tabs.addTab(self.tab_folder, "Folder")
        self.input_tabs.addTab(self.tab_bag, "ROS Bag")

        input_dock = QDockWidget("Input", self)
        input_dock.setWidget(self.input_tabs)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, input_dock)

        # Controls dock (engine / preprocess / display / classes).
        self.controls = ControlPanel(self.taxonomy.names(), self)
        controls_dock = QDockWidget("Controls", self)
        controls_dock.setWidget(self.controls)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, controls_dock)
        # Stack vertically: Input on top, Controls below.
        self.splitDockWidget(input_dock, controls_dock, Qt.Orientation.Vertical)

        # Bottom: detections table.
        self.results_table = ResultsTable(self)
        bottom = QDockWidget("Detections", self)
        bottom.setWidget(self.results_table)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, bottom)

        # Right: performance panel (rolling avg / GPU mem / model info).
        self.perf_panel = PerformancePanel(parent=self)
        perf_dock = QDockWidget("Performance", self)
        perf_dock.setWidget(self.perf_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, perf_dock)

    def _build_menu(self) -> None:
        bar = self.menuBar()
        file_menu = bar.addMenu("&File")

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
        self.status_source = QLabel("source: —")
        bar.addWidget(self.status_device)
        bar.addPermanentWidget(self.status_source)
        bar.addPermanentWidget(self.status_ckpt)
        bar.addPermanentWidget(self.status_time)

    def _wire_signals(self) -> None:
        c = self.controls
        c.run_inference_requested.connect(self._run_current_image)
        c.display_filters_changed.connect(self._redraw_overlays)
        c.engine_config_changed.connect(self._invalidate_engine)
        c.preprocess_changed.connect(self._on_preprocess_changed)
        c.model_changed.connect(self._on_model_changed)

        # Input tabs.
        self.tab_image.image_selected.connect(self._on_image_tab_selected)
        self.tab_folder.image_selected.connect(self._on_image_tab_selected)
        self.tab_bag.frame_ready.connect(self._on_bag_frame)
        self.input_tabs.currentChanged.connect(self._on_tab_changed)

        # Detection row → highlight box.
        self.results_table.row_selected.connect(self._on_row_selected)

    # ----------------------------------------------------------------
    # Engine lifecycle
    # ----------------------------------------------------------------
    def _ensure_engine(self) -> bool:
        model_kind = self.controls.model_kind()
        family = family_for(model_kind)
        ckpt = self.controls.checkpoint_path()

        # DINOv3+DETR needs a user checkpoint; YOLO can fall back to the
        # auto-downloaded Ultralytics weight matching `model_kind`.
        if family == "dinov3_detr" and ckpt is None:
            QMessageBox.warning(
                self, "Checkpoint",
                "DINOv3+DETR requires a checkpoint path (set in 'Checkpoint:').",
            )
            return False

        signature = (model_kind, str(ckpt) if ckpt else None, self.controls.top_k())
        if (
            self.engine is not None
            and self.engine.is_loaded()
            and self._engine_signature == signature
        ):
            return True

        try:
            engine = build_engine(
                model_kind,
                checkpoint_path=ckpt,
                device=self.cfg.device,
                image_size=self.cfg.image_size,
                top_k=self.controls.top_k(),
                aspect_crop_mode=self.controls.aspect_crop_mode(),
                pad_position=self.controls.pad_position(),
                taxonomy=self.taxonomy if family == "dinov3_detr" else None,
            )
            self.statusBar().showMessage(f"Loading {model_kind}…", 0)
            engine.load()
        except Exception as exc:
            QMessageBox.critical(self, "Engine load failed", str(exc))
            return False

        self.engine = engine
        self._engine_signature = signature
        # Rebuild taxonomy + class-filter grid + palette for the new engine.
        self.taxonomy = engine.taxonomy
        self.palette = list(palette_for(max(16, len(self.taxonomy))))
        self.controls.set_class_names(self.taxonomy.names())

        self.status_device.setText(f"device: {engine.device}")
        ckpt_label = ckpt.name if ckpt else f"{model_kind}.pt"
        epoch_str = f" (epoch={engine.epoch})" if engine.epoch >= 0 else ""
        self.status_ckpt.setText(f"model: {model_kind}  ckpt: {ckpt_label}{epoch_str}")
        self.statusBar().showMessage("Ready.", 3000)
        self.perf_panel.reset()
        self.perf_panel.set_static_info(
            device=engine.device,
            device_name=engine.device_name,
            param_count=engine.param_count,
            gpu_total_mb=engine.gpu_total_mb,
            image_size=self.cfg.image_size,
        )
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(2000)
        self.worker = InferenceWorker(engine, self)
        self.worker.signals.finished.connect(self._on_inference_finished)
        self.worker.signals.failed.connect(self._on_inference_failed)
        self.worker.signals.progress.connect(self.controls.set_progress)
        self.tab_bag.attach_busy_check(self.worker.is_busy)
        return True

    def _invalidate_engine(self) -> None:
        if self.engine is not None:
            self.engine = None
        self._engine_signature = ()

    def _on_model_changed(self, _model_kind: str) -> None:
        """User picked a different backend in the Model dropdown."""
        self._invalidate_engine()
        # Don't eagerly reload — wait until next inference trigger so the
        # user can also flip checkpoint/top-k before paying the load cost.
        self.statusBar().showMessage(
            "Model changed — will reload on next inference run.", 4000
        )

    def _on_preprocess_changed(self) -> None:
        if self.engine is not None and self.engine.is_loaded():
            self.engine.aspect_crop_mode = self.controls.aspect_crop_mode()
            self.engine.pad_position = self.controls.pad_position()
            # Re-run inference for the most recent input so the user sees
            # the effect immediately.
            if self._last_image is not None:
                self._submit_current_image(rerun=True)

    # ----------------------------------------------------------------
    # Tab switching
    # ----------------------------------------------------------------
    def _on_tab_changed(self, _idx: int) -> None:
        # Stash any in-flight inference so we don't render stale results
        # for the new mode. The user can re-trigger from the tab.
        self._last_tag = None
        # When leaving the bag tab while it's playing, pause it.
        if self.input_tabs.currentWidget() is not self.tab_bag:
            self.tab_bag.controller.pause()

    # ----------------------------------------------------------------
    # Image / folder tab → inference
    # ----------------------------------------------------------------
    def _on_image_tab_selected(self, path_str: str) -> None:
        path = Path(path_str)
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            QMessageBox.critical(self, "Open image failed", f"{path}\n{exc}")
            return
        self._last_image = image
        self.canvas.set_image(image)
        self.results_table.set_detections([], palette=self.palette)
        self.status_source.setText(f"source: {path.name}")
        self.statusBar().showMessage(f"Loaded {path.name}", 3000)
        self._submit_current_image(tag=str(path))

    def _submit_current_image(self, *, tag: str | None = None, rerun: bool = False) -> None:
        """Send the current PIL frame to the worker.

        Used by both image/folder tabs and the preprocess-changed re-run.
        For bag mode, see `_on_bag_frame`.
        """
        if self._last_image is None:
            return
        if not self._ensure_engine():
            return
        assert self.worker is not None
        if tag is None:
            tag = self._last_tag or "rerun"
        self._last_tag = tag
        self._last_result = None
        self.worker.submit_image(self._last_image, tag)

    def _run_current_image(self) -> None:
        """Bound to the ControlPanel 'Run Inference' button."""
        if self._last_image is None:
            return
        self._submit_current_image(rerun=True)

    # ----------------------------------------------------------------
    # ROS Bag → inference
    # ----------------------------------------------------------------
    def _on_bag_frame(self, frame: object) -> None:
        """Handler for RosBagTab.frame_ready (BagFrame).

        Skips the raw-image canvas update on purpose — `_redraw_overlays`
        will set_image once when inference finishes. With autoplay running
        at 5+ FPS the raw → overlay double-flash is wasted work (the user
        only ever sees the overlay frame anyway), and even with the fast
        numpy/QImage path it still doubles the GUI-thread cost per frame.
        See bench in `docs/progress/phase2-1_rosbag_monitor.md`.
        """
        if not self._ensure_engine():
            return
        assert self.worker is not None
        # frame.image is a PIL.Image.Image (RGB).
        self._last_image = frame.image  # type: ignore[attr-defined]
        tag = f"bag:{frame.topic}:{frame.idx:06d}"  # type: ignore[attr-defined]
        self._last_tag = tag
        self._last_result = None
        # Show "inferring…" feedback instead of swapping to the raw image.
        self.status_source.setText(
            f"source: bag idx={frame.idx}  (inferring…)"  # type: ignore[attr-defined]
        )
        self.worker.submit_image(self._last_image, tag)

    # ----------------------------------------------------------------
    # Inference result handling
    # ----------------------------------------------------------------
    def _on_inference_finished(self, tag: str, result: InferenceResult) -> None:
        # Always feed the perf panel — rolling stats matter even when the
        # user has switched canvas mid-inference.
        self.perf_panel.update_from_result(result)
        if tag != self._last_tag:
            return
        self._last_result = result
        self.status_time.setText(
            f"total: {result.total_ms:.0f} ms ({result.fps:.1f} FPS)"
        )
        self._redraw_overlays()

    def _on_inference_failed(self, tag: str, message: str) -> None:
        if tag != self._last_tag:
            return
        QMessageBox.critical(self, "Inference failed", f"{tag}\n{message}")

    def _redraw_overlays(self, *, highlight_index: int | None = None) -> None:
        if self._last_image is None:
            return
        if self._last_result is None:
            self.canvas.set_image(self._last_image, fit=False)
            self.results_table.set_detections([], palette=self.palette)
            return
        rendered = draw_detections(
            self._last_image,
            self._last_result.detections,
            palette=self.palette,
            score_threshold=self.controls.score_threshold(),
            class_filter=self.controls.class_filter(),
            highlight_index=highlight_index,
        )
        self.canvas.set_image(rendered, fit=False)
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
        # Build a stem from the last tag so bag exports get unique names.
        stem = (self._last_tag or "result").replace("/", "_").replace(":", "_")
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
        self.tab_bag.close_reader()
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(2000)
        super().closeEvent(event)
