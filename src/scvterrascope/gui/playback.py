"""Playback state machine for ROS bag tab.

Owns the current frame index + autoplay timer; emits a `frame_ready`
signal whenever a new frame should be displayed and inferred. The main
window connects this to InferenceWorker.submit_image, which in turn
applies most-recent-only drop semantics if inference can't keep up.

Why a state machine: play/pause/step/seek interact (e.g., a manual seek
mid-play should pause until user resumes). Centralizing the logic here
keeps the bag widget purely presentational.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

LOG = logging.getLogger(__name__)


class PlaybackController(QObject):
    """Frame-index state machine + autoplay timer."""

    # Emitted whenever a new BagFrame should be shown / inferred.
    frame_ready = pyqtSignal(object)         # BagFrame
    # Emitted when current_index changes — for seek bar / counters.
    seek_changed = pyqtSignal(int, int)      # current_idx, total
    # "stopped" (no bag) | "paused" | "playing"
    state_changed = pyqtSignal(str)

    BASE_FPS_DEFAULT = 30.0  # most camera bags are ~30 fps

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._reader: Any = None
        self._topic: str = ""
        self._frame_count: int = 0
        self._index: int = 0
        self._speed: float = 1.0
        self._base_fps: float = self.BASE_FPS_DEFAULT
        self._is_busy: Callable[[], bool] | None = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        self._state: str = "stopped"

    # ----- public state ------------------------------------------------
    @property
    def state(self) -> str:
        return self._state

    @property
    def current_index(self) -> int:
        return self._index

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def speed(self) -> float:
        return self._speed

    # ----- attach / detach --------------------------------------------
    def attach(self, reader: Any, topic: str) -> None:
        """Bind to a (BagReader, topic) and seek to frame 0."""
        self.pause()
        self._reader = reader
        self._topic = topic
        self._frame_count = reader.frame_count(topic)
        self._index = 0
        self._set_state("paused")
        self._emit_seek()
        self.seek_to(0)  # show first frame immediately

    def detach(self) -> None:
        self.pause()
        self._reader = None
        self._topic = ""
        self._frame_count = 0
        self._index = 0
        self._set_state("stopped")
        self._emit_seek()

    def set_busy_check(self, fn: Callable[[], bool] | None) -> None:
        """Inject a callable that returns True if the worker is mid-inference.

        When True during autoplay, the timer skips emitting the next
        frame so the queue doesn't pile up.
        """
        self._is_busy = fn

    # ----- transport --------------------------------------------------
    def play(self) -> None:
        if self._reader is None or self._state == "playing":
            return
        if self._index >= self._frame_count - 1:
            self.seek_to(0)
        self._set_state("playing")
        self._update_timer_interval()

    def pause(self) -> None:
        if self._timer.isActive():
            self._timer.stop()
        if self._state == "playing":
            self._set_state("paused")

    def toggle(self) -> None:
        (self.pause if self._state == "playing" else self.play)()

    def step(self, delta: int = 1) -> None:
        """Move by `delta` frames (typically ±1) and pause."""
        if self._reader is None:
            return
        self.pause()
        new_idx = max(0, min(self._frame_count - 1, self._index + delta))
        if new_idx == self._index and delta != 0:
            return  # at edge, no-op
        self.seek_to(new_idx)

    def seek_to(self, index: int) -> None:
        """Jump to absolute frame index, emit frame_ready, stay paused."""
        if self._reader is None:
            return
        idx = max(0, min(self._frame_count - 1, int(index)))
        try:
            frame = self._reader.frame_at(self._topic, idx)
        except Exception as exc:  # noqa: BLE001 — defensive
            LOG.warning("seek_to(%d) failed: %s", idx, exc)
            return
        self._index = idx
        self.frame_ready.emit(frame)
        self._emit_seek()

    def set_speed(self, speed: float) -> None:
        """1.0 = native frame rate. Min 0.1 to avoid pathological intervals."""
        self._speed = max(0.1, float(speed))
        if self._state == "playing":
            self._update_timer_interval()

    def set_base_fps(self, fps: float) -> None:
        """Override the assumed source frame rate (default 30 FPS)."""
        self._base_fps = max(1.0, float(fps))
        if self._state == "playing":
            self._update_timer_interval()

    # ----- internals --------------------------------------------------
    def _on_tick(self) -> None:
        if self._reader is None or self._state != "playing":
            return
        # Drop frame if worker is still inferring the previous one. This
        # is what makes autoplay smooth at any inference speed.
        if self._is_busy is not None and self._is_busy():
            return
        next_idx = self._index + 1
        if next_idx >= self._frame_count:
            self.pause()
            return
        try:
            frame = self._reader.step_forward(self._topic)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("step_forward failed: %s", exc)
            self.pause()
            return
        if frame is None:
            self.pause()
            return
        self._index = frame.idx
        self.frame_ready.emit(frame)
        self._emit_seek()

    def _update_timer_interval(self) -> None:
        # 1.0x → 1000/30 ≈ 33 ms, 4.0x → 8 ms, 0.5x → 67 ms.
        interval = max(5, int(round(1000.0 / (self._base_fps * self._speed))))
        if self._timer.isActive():
            self._timer.setInterval(interval)
        else:
            self._timer.start(interval)

    def _set_state(self, new_state: str) -> None:
        if new_state != self._state:
            self._state = new_state
            self.state_changed.emit(new_state)

    def _emit_seek(self) -> None:
        self.seek_changed.emit(self._index, self._frame_count)
