"""Draw detection overlays on a PIL image.

Pure-Pillow rendering so this module can be unit-tested without a torch /
PyQt environment. The GUI layer is free to convert the resulting PIL image
to a QPixmap with a simple bytes-roundtrip.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass
from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont

from scvterrascope.inference.engine import Detection


@dataclass(frozen=True)
class DrawStyle:
    line_width: int = 3
    label_padding: int = 3
    show_score: bool = True
    score_decimals: int = 2


def palette_for(n: int) -> tuple[tuple[int, int, int], ...]:
    """Generate `n` visually distinct RGB colors in HSV equal-spaced order.

    Stable across runs (no random shuffle) so a class always maps to the
    same color in the GUI.
    """
    out: list[tuple[int, int, int]] = []
    for i in range(n):
        h = i / max(n, 1)
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
        out.append((int(r * 255), int(g * 255), int(b * 255)))
    return tuple(out)


def _default_font() -> ImageFont.ImageFont:
    """Try to locate a readable TrueType font; fall back to PIL's default."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=16)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_detections(
    image: Image.Image,
    detections: Iterable[Detection],
    *,
    palette: Sequence[tuple[int, int, int]] | None = None,
    score_threshold: float = 0.0,
    class_filter: Sequence[str] | None = None,
    style: DrawStyle | None = None,
    highlight_index: int | None = None,
) -> Image.Image:
    """Return a copy of `image` with bbox overlays.

    Filters apply at draw time, not at predict time, so the caller can
    redraw with new thresholds without re-running inference.

    Parameters
    ----------
    palette : per-class color palette, indexed by `class_id - 1`. If None,
        a default 16-class HSV palette is used.
    score_threshold : detections below this score are skipped.
    class_filter : optional whitelist of class names. None means no filter.
    highlight_index : if given, that detection (in input order) gets a
        thicker box for the GUI's "click row → highlight box" UX.
    """
    style = style or DrawStyle()
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = _default_font()

    detections = list(detections)
    if not detections:
        return canvas

    max_class_id = max(d.class_id for d in detections)
    if palette is None:
        palette = palette_for(max(16, max_class_id))
    elif len(palette) < max_class_id:
        # Caller passed a palette but the detection set has a higher class id —
        # extend deterministically so we never index OOB.
        palette = tuple(palette) + palette_for(max_class_id)[len(palette):]

    allowed: set[str] | None = set(class_filter) if class_filter is not None else None

    for idx, det in enumerate(detections):
        if det.score < score_threshold:
            continue
        if allowed is not None and det.class_name not in allowed:
            continue

        color = palette[det.class_id - 1]
        x1, y1, x2, y2 = det.bbox_xyxy
        thickness = style.line_width + (2 if idx == highlight_index else 0)
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=thickness)

        if style.show_score:
            label = f"{det.class_name} {det.score:.{style.score_decimals}f}"
        else:
            label = det.class_name

        # Label background — a filled rectangle that contrasts with the bbox.
        text_bbox = draw.textbbox((0, 0), label, font=font)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]
        pad = style.label_padding
        # Place above the box if there's room; otherwise inside top-left.
        ly1 = y1 - th - 2 * pad
        if ly1 < 0:
            ly1 = y1
        ly2 = ly1 + th + 2 * pad
        lx1 = x1
        lx2 = x1 + tw + 2 * pad
        draw.rectangle([(lx1, ly1), (lx2, ly2)], fill=color)
        # Pick black or white text depending on background brightness.
        luma = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        text_color = (0, 0, 0) if luma > 160 else (255, 255, 255)
        draw.text((lx1 + pad, ly1 + pad), label, fill=text_color, font=font)

    return canvas
