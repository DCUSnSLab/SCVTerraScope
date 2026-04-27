"""Detection overlay rendering (bbox, class labels, confidence)."""

from scvterrascope.visualization.draw import (
    DrawStyle,
    draw_detections,
    palette_for,
)

__all__ = ["DrawStyle", "draw_detections", "palette_for"]
