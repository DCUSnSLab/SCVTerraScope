"""Detection drawing — palette stability + filter behavior + bbox geometry."""

from __future__ import annotations

from PIL import Image

from scvterrascope.inference.engine import Detection
from scvterrascope.visualization.draw import (
    DrawStyle,
    draw_detections,
    palette_for,
)


def test_palette_is_stable() -> None:
    p1 = palette_for(16)
    p2 = palette_for(16)
    assert p1 == p2
    assert len(p1) == 16
    # Every entry should be a 3-tuple of bytes.
    for r, g, b in p1:
        assert 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255


def _det(class_id: int, score: float, name: str = "") -> Detection:
    return Detection(
        class_id=class_id,
        class_name=name or f"class_{class_id}",
        score=score,
        bbox_xyxy=(10.0, 20.0, 100.0, 80.0),
    )


def test_threshold_filters_drawing() -> None:
    img = Image.new("RGB", (200, 200), color=(255, 255, 255))
    dets = [_det(1, 0.9), _det(2, 0.1)]

    drawn_low = draw_detections(img, dets, score_threshold=0.05)
    drawn_high = draw_detections(img, dets, score_threshold=0.5)

    # With high threshold, fewer non-white pixels (only one box drawn).
    low_modified = sum(1 for px in drawn_low.getdata() if px != (255, 255, 255))
    high_modified = sum(1 for px in drawn_high.getdata() if px != (255, 255, 255))
    assert low_modified > high_modified > 0


def test_class_filter_hides_classes() -> None:
    img = Image.new("RGB", (200, 200), color=(255, 255, 255))
    dets = [_det(1, 0.9, "pedestrian"), _det(2, 0.9, "bicycle")]
    none = draw_detections(img, dets, class_filter=[])
    one = draw_detections(img, dets, class_filter=["pedestrian"])
    none_mod = sum(1 for px in none.getdata() if px != (255, 255, 255))
    one_mod = sum(1 for px in one.getdata() if px != (255, 255, 255))
    assert none_mod == 0  # nothing drawn when filter is empty
    assert one_mod > 0


def test_empty_detections_returns_unmodified_copy() -> None:
    img = Image.new("RGB", (50, 50), color=(0, 0, 0))
    out = draw_detections(img, [])
    # Same dimensions, all-black still.
    assert out.size == img.size
    assert all(px == (0, 0, 0) for px in out.getdata())


def test_highlight_thickens_box() -> None:
    img = Image.new("RGB", (200, 200), color=(255, 255, 255))
    dets = [_det(1, 0.9)]
    plain = draw_detections(img, dets, style=DrawStyle(line_width=2))
    bold = draw_detections(img, dets, style=DrawStyle(line_width=2), highlight_index=0)
    plain_pixels = sum(1 for px in plain.getdata() if px != (255, 255, 255))
    bold_pixels = sum(1 for px in bold.getdata() if px != (255, 255, 255))
    assert bold_pixels > plain_pixels
