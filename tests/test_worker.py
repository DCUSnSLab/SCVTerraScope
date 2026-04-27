"""InferenceWorker — submit_image / drop-frame behavior.

We mock the engine so these tests don't need torch on the path. The
goal is to exercise the slot-and-drop semantics of the new bag/playback
submission mode without sourcing a real model.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

import pytest

# Headless Qt — works in CI / when DISPLAY is unset.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QCoreApplication, QEventLoop, QTimer  # noqa: E402

from scvterrascope.gui.worker import InferenceWorker  # noqa: E402


class _FakeResult:
    pass


class _FakeEngine:
    """Stub engine: predict() blocks `delay_s` and returns a sentinel."""

    def __init__(self, delay_s: float = 0.05) -> None:
        self.delay_s = delay_s
        self.calls: list[Any] = []

    def predict(self, image: Any) -> _FakeResult:
        self.calls.append(image)
        time.sleep(self.delay_s)
        return _FakeResult()


@pytest.fixture
def app() -> QCoreApplication:
    a = QCoreApplication.instance() or QCoreApplication(sys.argv)
    return a


def _wait_for(predicate, timeout_ms: int = 2000) -> bool:
    """Spin the Qt event loop until `predicate()` is true or timeout."""
    loop = QEventLoop()
    deadline = time.monotonic() + timeout_ms / 1000.0
    timer = QTimer()
    timer.setInterval(20)
    timer.timeout.connect(lambda: (predicate() or time.monotonic() > deadline) and loop.quit())
    timer.start()
    loop.exec()
    timer.stop()
    return predicate()


def test_submit_image_emits_finished(app: QCoreApplication) -> None:
    worker = InferenceWorker(_FakeEngine(delay_s=0.01))
    received: list[tuple[str, object]] = []
    worker.signals.finished.connect(lambda tag, res: received.append((tag, res)))

    worker.submit_image("FAKE_PIL", "frame-0")
    assert _wait_for(lambda: len(received) >= 1)
    assert received[0][0] == "frame-0"
    worker.stop()
    worker.wait(2000)


def test_submit_image_drops_intermediate_frames(app: QCoreApplication) -> None:
    """Rapid submits while busy should keep only the LATEST pending frame."""
    eng = _FakeEngine(delay_s=0.20)  # slow predict to force pending overlap
    worker = InferenceWorker(eng)
    received: list[str] = []
    worker.signals.finished.connect(lambda tag, _r: received.append(tag))

    # Submit frame 0 (becomes in-flight after thread starts).
    worker.submit_image("FAKE", "f-0")
    # Spam frames while #0 is still running.
    for i in range(1, 6):
        time.sleep(0.02)
        worker.submit_image("FAKE", f"f-{i}")

    assert _wait_for(lambda: len(received) >= 2, timeout_ms=4000)
    worker.stop()
    worker.wait(2000)

    # We must have processed f-0 first, then ONE of the later frames
    # (drop semantics — not all six). Last received must be the most
    # recently submitted that survived the slot.
    assert received[0] == "f-0"
    assert len(received) <= 3, f"expected ≤3 frames (drop), got {received}"
    assert received[-1] == "f-5"


def test_submit_paths_still_works(app: QCoreApplication, tmp_path) -> None:
    """Folder-mode path queue must keep its old per-path emission."""
    from PIL import Image

    paths = []
    for i in range(3):
        p = tmp_path / f"img_{i}.png"
        Image.new("RGB", (8, 8), color=(i * 50, 0, 0)).save(p)
        paths.append(p)

    worker = InferenceWorker(_FakeEngine(delay_s=0.01))
    received: list[str] = []
    worker.signals.finished.connect(lambda tag, _r: received.append(tag))

    worker.submit(paths)
    assert _wait_for(lambda: len(received) >= 3, timeout_ms=4000)
    worker.stop()
    worker.wait(2000)

    assert [p.name for p in paths] == [s.split("/")[-1] for s in received]
