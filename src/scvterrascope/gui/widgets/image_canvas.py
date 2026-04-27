"""Zoomable / pannable image viewer with detection overlays."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPainter, QPixmap, QWheelEvent
from PyQt6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView


def pil_to_pixmap(image: Image.Image) -> QPixmap:
    """Convert a PIL image to QPixmap via numpy → QImage (no PNG round-trip).

    Bench (1920×1080, RGB):
      PNG round-trip path  —  ~212 ms per call
      numpy/QImage path    —  ~1.1 ms per call (190× faster)

    The PNG path made bag autoplay a ~2 s/frame walk; this is what
    makes 5 FPS feasible on the GUI side. See
    `docs/progress/phase2-1_rosbag_monitor.md` for the bench table.

    QImage references the underlying buffer, so we must hand it owned
    bytes (`.tobytes()`) — the array would otherwise be GCed before
    QPixmap.fromImage finishes the deep copy.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    arr = np.asarray(image)
    h, w, ch = arr.shape  # ch=3 for RGB
    qimg = QImage(arr.tobytes(), w, h, w * ch, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


class ImageCanvas(QGraphicsView):
    """QGraphicsView with mouse-wheel zoom and middle-button pan.

    The canvas holds exactly one pixmap item; updating it replaces the
    pixmap in place rather than tearing down the scene, so the user's
    current zoom / pan persists across image switches when scenes match.
    """

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._pix_item: QGraphicsPixmapItem | None = None
        self._zoom = 1.0

    def set_image(self, image: Image.Image, *, fit: bool = True) -> None:
        pix = pil_to_pixmap(image)
        scene = self.scene()
        if self._pix_item is None:
            self._pix_item = scene.addPixmap(pix)
        else:
            self._pix_item.setPixmap(pix)
        scene.setSceneRect(0, 0, pix.width(), pix.height())
        if fit:
            self.reset_zoom()

    def reset_zoom(self) -> None:
        if self._pix_item is None:
            return
        self.resetTransform()
        self.fitInView(self._pix_item, Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom = 1.0

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 - Qt API
        if self._pix_item is None:
            return
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        new_zoom = self._zoom * factor
        # Cap zoom to a sensible range so wheel-spam can't lose the image.
        if not 0.05 <= new_zoom <= 40.0:
            return
        self.scale(factor, factor)
        self._zoom = new_zoom
