"""PlaybackController state machine — uses a fake reader so torch/ros aren't needed."""

from __future__ import annotations

import os
import sys
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")

from PyQt6.QtCore import QCoreApplication, QEventLoop, QTimer  # noqa: E402

from scvterrascope.gui.playback import PlaybackController  # noqa: E402


class _FakeBagFrame:
    def __init__(self, idx: int) -> None:
        self.idx = idx
        self.ros_time_ns = 1_000_000 * idx
        self.image = None
        self.topic = "/fake/topic"
        self.encoding = "rgb8"


class _FakeReader:
    def __init__(self, count: int = 100) -> None:
        self._count = count

    def frame_count(self, topic: str) -> int:
        return self._count

    def frame_at(self, topic: str, idx: int) -> _FakeBagFrame:
        return _FakeBagFrame(idx)

    def step_forward(self, topic: str) -> _FakeBagFrame | None:
        return _FakeBagFrame(self._step_state.get(topic, 0))

    # PlaybackController only calls step_forward via its tick. We track
    # the next-to-yield index manually so successive calls advance.
    _step_state: dict[str, int] = {}

    def reset(self) -> None:
        self._step_state.clear()

    def __init_subclass__(cls):  # noqa: D401 - unused
        super().__init_subclass__()


class _AdvancingReader(_FakeReader):
    def step_forward(self, topic):  # type: ignore[override]
        cur = self._step_state.get(topic, 0)
        if cur >= self._count:
            return None
        self._step_state[topic] = cur + 1
        return _FakeBagFrame(cur)


@pytest.fixture
def app() -> QCoreApplication:
    return QCoreApplication.instance() or QCoreApplication(sys.argv)


def _drain(ms: int = 30) -> None:
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def test_attach_emits_first_frame_and_seek(app: QCoreApplication) -> None:
    pb = PlaybackController()
    frames: list[int] = []
    seeks: list[tuple[int, int]] = []
    states: list[str] = []
    pb.frame_ready.connect(lambda f: frames.append(f.idx))
    pb.seek_changed.connect(lambda c, t: seeks.append((c, t)))
    pb.state_changed.connect(states.append)

    pb.attach(_FakeReader(count=50), "/fake/topic")

    assert pb.state == "paused"
    assert pb.frame_count == 50
    assert pb.current_index == 0
    assert frames == [0]
    assert seeks[-1] == (0, 50)
    assert "paused" in states


def test_step_forward_and_back(app: QCoreApplication) -> None:
    pb = PlaybackController()
    frames: list[int] = []
    pb.frame_ready.connect(lambda f: frames.append(f.idx))
    pb.attach(_FakeReader(count=10), "/t")
    frames.clear()

    pb.step(+1)
    pb.step(+1)
    pb.step(-1)
    assert frames == [1, 2, 1]


def test_seek_to_clamps(app: QCoreApplication) -> None:
    pb = PlaybackController()
    frames: list[int] = []
    pb.frame_ready.connect(lambda f: frames.append(f.idx))
    pb.attach(_FakeReader(count=10), "/t")
    frames.clear()

    pb.seek_to(99)   # past end
    pb.seek_to(-5)   # before start
    pb.seek_to(7)
    assert frames == [9, 0, 7]


def test_play_advances_and_pauses_at_end(app: QCoreApplication) -> None:
    reader = _AdvancingReader(count=4)
    pb = PlaybackController()
    pb.set_base_fps(200.0)  # 5 ms interval at speed 1 — fast for tests
    pb.set_speed(1.0)
    frames: list[int] = []
    states: list[str] = []
    pb.frame_ready.connect(lambda f: frames.append(f.idx))
    pb.state_changed.connect(states.append)
    pb.attach(reader, "/t")
    frames.clear()

    pb.play()
    _drain(200)  # let the timer tick a few times
    assert pb.state == "paused"  # auto-pauses at end
    # Should have advanced through frames 1..3 (frame 0 was on attach).
    assert frames[-1] == 3
    assert "playing" in states and states[-1] == "paused"


def test_busy_check_drops_frames(app: QCoreApplication) -> None:
    """If is_busy() returns True, the timer tick should NOT emit a new frame."""
    reader = _AdvancingReader(count=20)
    pb = PlaybackController()
    pb.set_base_fps(1000.0)  # 1 ms interval — many ticks
    busy = [True]
    pb.set_busy_check(lambda: busy[0])
    frames: list[int] = []
    pb.frame_ready.connect(lambda f: frames.append(f.idx))
    pb.attach(reader, "/t")
    initial = len(frames)

    pb.play()
    _drain(50)
    # While busy, no new frames should have been emitted.
    assert len(frames) == initial
    busy[0] = False
    _drain(50)
    pb.pause()
    # Once not busy, frames advance.
    assert len(frames) > initial
