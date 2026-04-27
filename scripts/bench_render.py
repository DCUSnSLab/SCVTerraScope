"""Micro-benchmark for the GUI render path.

Measures the dominant cost between:
  - pil_to_pixmap (PNG round-trip vs numpy/QImage direct)
  - draw_detections (full path)
  - one simulated end-to-end "frame received → overlay set" cycle

Run offscreen so it doesn't pop a window:
  QT_QPA_PLATFORM=offscreen .venv/bin/python scripts/bench_render.py
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from PyQt6.QtGui import QImage, QPixmap  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from scvterrascope.inference.engine import Detection  # noqa: E402
from scvterrascope.visualization.draw import draw_detections, palette_for  # noqa: E402


# Need a QApplication for QPixmap construction.
_app = QApplication.instance() or QApplication(sys.argv)


def png_roundtrip_pixmap(image: Image.Image) -> QPixmap:
    """Current implementation — PNG encode + decode."""
    buf = BytesIO()
    image.save(buf, format="PNG")
    pix = QPixmap()
    pix.loadFromData(buf.getvalue(), "PNG")
    return pix


def numpy_qimage_pixmap(image: Image.Image) -> QPixmap:
    """Proposed Step B — numpy buffer → QImage → QPixmap."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    arr = np.asarray(image)
    h, w, ch = arr.shape
    qimg = QImage(arr.tobytes(), w, h, w * ch, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


def time_call(fn, *args, n: int = 30, warmup: int = 3) -> tuple[float, float]:
    for _ in range(warmup):
        fn(*args)
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn(*args)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples), statistics.fmean(samples)


def make_dummy_detections(n: int = 100) -> list[Detection]:
    rng = np.random.default_rng(42)
    out = []
    for i in range(n):
        x1 = rng.integers(0, 1800)
        y1 = rng.integers(0, 900)
        w = rng.integers(40, 120)
        h = rng.integers(40, 120)
        out.append(Detection(
            class_id=int((i % 16) + 1),
            class_name=f"class_{(i % 16) + 1}",
            score=float(rng.uniform(0.1, 0.95)),
            bbox_xyxy=(float(x1), float(y1), float(x1 + w), float(y1 + h)),
        ))
    return out


def main() -> None:
    sizes = [(1920, 1080), (1224, 1024)]
    for W, H in sizes:
        # Synthesize a vivid RGB image (uniform color is too compressible).
        arr = np.random.default_rng(0).integers(0, 256, size=(H, W, 3), dtype=np.uint8)
        img = Image.fromarray(arr, "RGB")
        dets = make_dummy_detections(100)
        palette = list(palette_for(16))

        print(f"\n=== {W}x{H} ===")
        med, avg = time_call(png_roundtrip_pixmap, img)
        print(f"  pil_to_pixmap (PNG roundtrip):   median={med:6.1f} ms  avg={avg:6.1f} ms")
        med, avg = time_call(numpy_qimage_pixmap, img)
        print(f"  pil_to_pixmap (numpy/QImage):    median={med:6.1f} ms  avg={avg:6.1f} ms")
        med, avg = time_call(draw_detections, img, dets, n=30, warmup=2)
        print(f"  draw_detections (100 boxes):     median={med:6.1f} ms  avg={avg:6.1f} ms")

        # Simulated full cycle as in current bag mode (raw set + draw + overlay set).
        def cycle_current():
            png_roundtrip_pixmap(img)                   # raw set_image
            rendered = draw_detections(img, dets, palette=palette, score_threshold=0.3)
            png_roundtrip_pixmap(rendered)              # overlay set_image
        med, avg = time_call(cycle_current, n=10, warmup=2)
        print(f"  CURRENT cycle (raw+draw+overlay): median={med:6.1f} ms  avg={avg:6.1f} ms")

        # Step C only (no raw set, still PNG):
        def cycle_step_c_only():
            rendered = draw_detections(img, dets, palette=palette, score_threshold=0.3)
            png_roundtrip_pixmap(rendered)
        med, avg = time_call(cycle_step_c_only, n=15, warmup=2)
        print(f"  + Step C only (no raw set):      median={med:6.1f} ms  avg={avg:6.1f} ms")

        # Step B+C combined:
        def cycle_step_bc():
            rendered = draw_detections(img, dets, palette=palette, score_threshold=0.3)
            numpy_qimage_pixmap(rendered)
        med, avg = time_call(cycle_step_bc, n=30, warmup=2)
        print(f"  + Step B+C (numpy + no raw):     median={med:6.1f} ms  avg={avg:6.1f} ms")


if __name__ == "__main__":
    main()
