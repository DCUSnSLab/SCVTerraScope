"""QThread-based inference worker.

Keeping the heavy PyTorch forward off the GUI thread is what stops the
window from freezing while a 1024×1024 letterboxed forward pass runs.

Two submission modes:
  - `submit(paths)`        — Phase 1-1 path-based queue (Image / Folder tab).
  - `submit_image(image, tag)` — Phase 2 single in-memory PIL frame
    (ROS Bag tab; tag is typically a synthetic id like "bag_idx_0123").

Both modes emit the same `started/finished/failed/progress` signals so the
main window doesn't care which input source produced a frame. `is_busy()`
lets callers (e.g. PlaybackController) drop frames they can't keep up with.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6.QtCore import QObject, QThread, QMutex, pyqtSignal


class InferenceSignals(QObject):
    started = pyqtSignal(str)               # tag (path string or synthetic)
    finished = pyqtSignal(str, object)      # tag, InferenceResult
    failed = pyqtSignal(str, str)           # tag, error message
    progress = pyqtSignal(int, int)         # current, total (folder/playback)


class InferenceWorker(QThread):
    """Worker that consumes either a path queue or a single PIL frame.

    Folder mode: call `submit(paths)`. The worker walks the list and
    emits per-path signals.

    Bag/Playback mode: call `submit_image(pil, tag)` repeatedly. Each
    call replaces any pending in-flight frame (so playback that runs
    faster than inference automatically drops frames). The currently-
    inferring frame still completes before the next one starts.
    """

    def __init__(self, engine: Any, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.engine = engine
        self.signals = InferenceSignals()
        # Path queue (folder mode). Mutually exclusive with single-frame mode.
        self._queue: list[Path] = []
        # Single-frame slot (bag/playback). Replaced on each submit_image.
        self._pending: tuple[Any, str] | None = None
        self._mutex = QMutex()
        self._busy = False
        self._stop = False
        self._mode: str = "idle"  # "paths" | "image" | "idle"

    # ----- public API ----------------------------------------------
    def submit(self, paths: list[Path]) -> None:
        """Replace the queue with `paths` and wake the worker (folder mode)."""
        self._mutex.lock()
        try:
            self._queue = list(paths)
            self._pending = None
            self._mode = "paths"
            self._stop = False
        finally:
            self._mutex.unlock()
        if not self.isRunning():
            self.start()

    def submit_image(self, image: Any, tag: str) -> None:
        """Submit a single PIL frame (bag/playback mode).

        If the worker is mid-inference on a previous frame, the new
        frame waits in the slot. If another frame arrives before that
        runs, the older one is dropped — most-recent-only semantics.
        """
        self._mutex.lock()
        try:
            self._pending = (image, tag)
            self._queue = []
            self._mode = "image"
            self._stop = False
        finally:
            self._mutex.unlock()
        if not self.isRunning():
            self.start()

    def is_busy(self) -> bool:
        """True while a forward pass is actively running.

        PlaybackController checks this to decide whether to skip the
        next frame instead of submitting a backlog.
        """
        return self._busy

    def stop(self) -> None:
        self._mutex.lock()
        try:
            self._stop = True
            self._queue = []
            self._pending = None
        finally:
            self._mutex.unlock()
        self.requestInterruption()

    # ----- thread loop ---------------------------------------------
    def run(self) -> None:  # noqa: D401 - QThread API
        from PIL import Image

        if self._mode == "paths":
            queue = list(self._queue)
            total = len(queue)
            for idx, path in enumerate(queue):
                if self._should_stop():
                    return
                tag = str(path)
                self.signals.started.emit(tag)
                self.signals.progress.emit(idx, total)
                try:
                    img = Image.open(path).convert("RGB")
                    self._busy = True
                    result = self.engine.predict(img)
                finally:
                    self._busy = False
                self.signals.finished.emit(tag, result)
            self.signals.progress.emit(total, total)
            return

        # image / playback mode — drain the single slot until it stays empty.
        # We loop because callers can submit more frames while we're working.
        while not self._should_stop():
            self._mutex.lock()
            pending = self._pending
            self._pending = None
            self._mutex.unlock()
            if pending is None:
                # No more pending frames — exit the run() so the thread can
                # be re-started by the next submit_image() call.
                return
            image, tag = pending
            self.signals.started.emit(tag)
            try:
                self._busy = True
                result = self.engine.predict(image)
            except Exception as exc:
                self._busy = False
                self.signals.failed.emit(tag, f"{type(exc).__name__}: {exc}")
                continue
            self._busy = False
            self.signals.finished.emit(tag, result)

    # ----- helpers -------------------------------------------------
    def _should_stop(self) -> bool:
        return self._stop or self.isInterruptionRequested()
