"""Zoomable / pannable image viewer with detection overlays."""

from __future__ import annotations

from io import BytesIO
from typing import Any

from PIL import Image
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QPixmap, QWheelEvent
from PyQt6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView


def pil_to_pixmap(image: Image.Image) -> QPixmap:
    """Round-trip a PIL image to QPixmap via in-memory PNG.

    PNG roundtrip is slower than QImage(numpy_buffer, ...) but spares us
    from worrying about row strides and channel order for an MVP.
    """
    buf = BytesIO()
    image.save(buf, format="PNG")
    pix = QPixmap()
    pix.loadFromData(buf.getvalue(), "PNG")
    return pix


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
