"""QThread-based inference worker.

Keeping the heavy PyTorch forward off the GUI thread is what stops the
window from freezing while a 1024×1024 letterboxed forward pass runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6.QtCore import QObject, QThread, pyqtSignal


class InferenceSignals(QObject):
    started = pyqtSignal(str)               # image path
    finished = pyqtSignal(str, object)      # image path, InferenceResult
    failed = pyqtSignal(str, str)           # image path, error message
    progress = pyqtSignal(int, int)         # current, total (folder mode)


class InferenceWorker(QThread):
    """Worker that consumes a queue of image paths and emits results.

    Construct one worker per GUI session; queue paths via `submit()`. The
    worker self-loops on its internal queue and exits when `stop()` is
    called.
    """

    def __init__(self, engine: Any, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.engine = engine
        self.signals = InferenceSignals()
        self._queue: list[Path] = []
        self._stop = False

    def submit(self, paths: list[Path]) -> None:
        """Replace the current queue with `paths` and wake the worker."""
        self._queue = list(paths)
        self._stop = False
        if not self.isRunning():
            self.start()

    def stop(self) -> None:
        self._stop = True
        self._queue = []
        self.requestInterruption()

    def run(self) -> None:  # noqa: D401 - QThread API
        from PIL import Image

        total = len(self._queue)
        for idx, path in enumerate(list(self._queue)):
            if self._stop or self.isInterruptionRequested():
                return
            spath = str(path)
            self.signals.started.emit(spath)
            self.signals.progress.emit(idx, total)
            try:
                image = Image.open(path).convert("RGB")
                result = self.engine.predict(image)
            except Exception as exc:  # surface every error to the GUI status bar
                self.signals.failed.emit(spath, f"{type(exc).__name__}: {exc}")
                continue
            self.signals.finished.emit(spath, result)
        self.signals.progress.emit(total, total)
