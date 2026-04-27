"""Preprocess module — letterbox round-trip + ImageNet normalization."""

from __future__ import annotations

from PIL import Image

from scvterrascope.inference.preprocess import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    letterbox_resize,
    preprocess_pil,
    unproject_letterbox_xyxy,
)


def test_unproject_round_trip_full_image() -> None:
    # A box at origin in letterbox frame at scale 0.5 → (0,0,200,400) in original.
    x1, y1, x2, y2 = unproject_letterbox_xyxy(
        (0.0, 0.0, 100.0, 200.0), scale=0.5, original_size=(400, 200)
    )
    assert (x1, y1, x2, y2) == (0.0, 0.0, 200.0, 400.0)


def test_unproject_clamps_to_image_bounds() -> None:
    # Box exits the right/bottom of an 800x600 image; should clamp.
    x1, y1, x2, y2 = unproject_letterbox_xyxy(
        (700.0, 500.0, 1000.0, 800.0), scale=1.0, original_size=(600, 800)
    )
    assert x1 == 700.0
    assert y1 == 500.0
    assert x2 == 800.0  # clamped to image width
    assert y2 == 600.0  # clamped to image height


def test_letterbox_resize_landscape_bottom_right() -> None:
    """Default (bottom_right) — byte-equivalent to training preprocess."""
    import numpy as np

    arr = np.full((480, 640, 3), 100, dtype=np.uint8)
    padded, scale, (pad_top, pad_bottom, pad_left, pad_right) = letterbox_resize(
        arr, target_size=1024
    )
    assert padded.shape == (1024, 1024, 3)
    assert abs(scale - 1.6) < 1e-6
    # 1024 / 640 = 1.6; new_h = 480 * 1.6 = 768; pad on bottom only.
    assert pad_top == 0
    assert pad_bottom == 1024 - 768
    assert pad_left == 0
    assert pad_right == 0
    # Content area filled, padded area is the default pad value (114).
    assert padded[0, 0, 0] == 100
    assert padded[800, 0, 0] == 114


def test_letterbox_resize_symmetric() -> None:
    """Symmetric padding splits vertical pad equally on top/bottom."""
    import numpy as np

    arr = np.full((480, 640, 3), 100, dtype=np.uint8)
    padded, _scale, (pad_top, pad_bottom, pad_left, pad_right) = letterbox_resize(
        arr, target_size=1024, pad_position="symmetric"
    )
    # 1024 - 768 = 256 total vertical pad → 128 top + 128 bottom.
    assert pad_top == 128
    assert pad_bottom == 128
    assert pad_left == 0
    assert pad_right == 0
    # Top rows are padding (114), middle rows have content, bottom is padding.
    assert padded[0, 0, 0] == 114        # top pad
    assert padded[500, 0, 0] == 100      # content
    assert padded[1000, 0, 0] == 114     # bottom pad


def test_letterbox_resize_auto_picks_symmetric_for_wide_input() -> None:
    """Auto mode: 16:9 (1.78) is far from training 1.20 → use symmetric."""
    import numpy as np

    arr = np.full((1080, 1920, 3), 100, dtype=np.uint8)
    _padded, _scale, (pad_top, pad_bottom, pad_left, pad_right) = letterbox_resize(
        arr, target_size=1024, pad_position="auto"
    )
    # Total vertical pad = 1024 - 576 = 448. Symmetric → 224 / 224.
    assert pad_top == 224
    assert pad_bottom == 224


def test_letterbox_resize_auto_picks_bottom_right_for_training_aspect() -> None:
    """Auto mode: CODa 1224×1024 (1.195) is on target → keep bottom_right."""
    import numpy as np

    arr = np.full((1024, 1224, 3), 100, dtype=np.uint8)
    _padded, _scale, (pad_top, pad_bottom, pad_left, pad_right) = letterbox_resize(
        arr, target_size=1024, pad_position="auto"
    )
    # Within tolerance → bottom_right convention preserved.
    assert pad_top == 0
    assert pad_bottom == 1024 - int(round(1024 * (1024 / 1224)))


def test_preprocess_pil_round_trip() -> None:
    """Verify shape + dtype + scale invariants of preprocess_pil."""
    img = Image.new("RGB", (640, 480), color=(128, 64, 32))
    pre = preprocess_pil(img, target_size=1024)

    assert pre.pixel_values.shape == (3, 1024, 1024)
    assert pre.pixel_mask.shape == (1024, 1024)
    assert pre.original_size == (480, 640)
    assert abs(pre.scale - 1.6) < 1e-6
    assert pre.pad_h == 1024 - 768
    assert pre.pad_w == 0


def test_imagenet_constants_are_canonical() -> None:
    # These are part of the trained model's input contract — pinning them
    # in a test makes a silent change show up in CI rather than at AP-time.
    assert IMAGENET_MEAN == (0.485, 0.456, 0.406)
    assert IMAGENET_STD == (0.229, 0.224, 0.225)


# ---- aspect-crop tests -------------------------------------------------

def test_maybe_center_crop_none_is_no_op() -> None:
    from scvterrascope.inference.preprocess import maybe_center_crop

    img = Image.new("RGB", (1920, 1080), color=(0, 0, 0))
    out, off = maybe_center_crop(img, mode="none")
    assert out.size == (1920, 1080)
    assert off == (0, 0)


def test_maybe_center_crop_auto_crops_wide_input() -> None:
    """1920x1080 (1.78:1) is well outside CODa's 1.20:1 — auto should crop."""
    from scvterrascope.inference.preprocess import maybe_center_crop

    img = Image.new("RGB", (1920, 1080), color=(0, 0, 0))
    out, off = maybe_center_crop(img, mode="auto", target_aspect=1.20, tolerance=0.10)
    new_w, new_h = out.size
    assert new_h == 1080  # height preserved
    assert new_w == round(1080 * 1.20) == 1296
    # symmetric crop
    assert off == ((1920 - 1296) // 2, 0)


def test_maybe_center_crop_auto_skips_when_within_tolerance() -> None:
    """CODa 1224x1024 is right on target — auto should NOT crop."""
    from scvterrascope.inference.preprocess import maybe_center_crop

    img = Image.new("RGB", (1224, 1024), color=(0, 0, 0))
    out, off = maybe_center_crop(img, mode="auto", target_aspect=1.20, tolerance=0.10)
    assert out.size == (1224, 1024)
    assert off == (0, 0)


def test_maybe_center_crop_auto_crops_tall_input() -> None:
    """Portrait input (too tall) should be cropped vertically."""
    from scvterrascope.inference.preprocess import maybe_center_crop

    img = Image.new("RGB", (1024, 1920), color=(0, 0, 0))
    out, off = maybe_center_crop(img, mode="auto", target_aspect=1.20, tolerance=0.10)
    new_w, new_h = out.size
    assert new_w == 1024
    assert new_h == round(1024 / 1.20) == 853
    assert off == (0, (1920 - 853) // 2)


def test_unproject_with_crop_offset() -> None:
    """Crop offset must be added to bring boxes back to true-original coords."""
    from scvterrascope.inference.preprocess import unproject_letterbox_xyxy

    scale = 1024 / 1296
    x1, y1, x2, y2 = unproject_letterbox_xyxy(
        (200.0, 100.0, 400.0, 200.0),
        scale=scale,
        original_size=(1080, 1296),
        crop_offset=(312, 0),
        true_original_size=(1080, 1920),
    )
    assert abs(x1 - (200 / scale + 312)) < 1e-3
    assert abs(y1 - (100 / scale)) < 1e-3
    assert abs(x2 - (400 / scale + 312)) < 1e-3
    assert abs(y2 - (200 / scale)) < 1e-3


def test_unproject_with_symmetric_pad_offset() -> None:
    """pad_top/pad_left must be subtracted before dividing by scale."""
    from scvterrascope.inference.preprocess import unproject_letterbox_xyxy

    # SCV 1920x1080 with symmetric letterbox: scale = 1024/1920 = 0.5333,
    # new_h = 576, pad_h_total = 448 → pad_top = 224, pad_bottom = 224.
    scale = 1024 / 1920
    pad_top = 224
    # A model prediction at letterbox y = pad_top + 50 should land at y_orig = 50/scale ≈ 94.
    _, y1, _, _ = unproject_letterbox_xyxy(
        (0.0, float(pad_top + 50), 100.0, float(pad_top + 100)),
        scale=scale,
        original_size=(1080, 1920),
        pad_top=pad_top,
        pad_left=0,
    )
    assert abs(y1 - 50.0 / scale) < 1e-3

    # And a box centered in the letterbox (y = 512) should map to roughly
    # the vertical centre of the original (~540).
    _, y_mid, _, _ = unproject_letterbox_xyxy(
        (0.0, 512.0, 1.0, 513.0),
        scale=scale,
        original_size=(1080, 1920),
        pad_top=pad_top,
        pad_left=0,
    )
    assert abs(y_mid - (512 - 224) / scale) < 1e-3


def test_preprocess_pil_records_crop_offset() -> None:
    """preprocess_pil with auto crop should set crop_offset and true_original_size."""
    from scvterrascope.inference.preprocess import DEFAULT_TARGET_ASPECT

    img = Image.new("RGB", (1920, 1080), color=(128, 64, 32))
    pre = preprocess_pil(img, target_size=1024, aspect_crop_mode="auto")
    # 1920x1080 (1.78:1) → cropped to round(1080 * DEFAULT_TARGET_ASPECT) wide.
    expected_w = round(1080 * DEFAULT_TARGET_ASPECT)
    assert pre.original_size == (1080, expected_w)
    assert pre.crop_offset == ((1920 - expected_w) // 2, 0)
    assert pre.true_original_size == (1080, 1920)
    assert abs(pre.scale - 1024 / max(1080, expected_w)) < 1e-6


def test_preprocess_pil_no_crop_when_mode_none() -> None:
    """Backwards compat: aspect_crop_mode='none' must reproduce old behavior."""
    img = Image.new("RGB", (1920, 1080), color=(128, 64, 32))
    pre = preprocess_pil(img, target_size=1024, aspect_crop_mode="none")
    assert pre.original_size == (1080, 1920)
    assert pre.crop_offset == (0, 0)
    assert pre.true_original_size is None
    assert abs(pre.scale - 1024 / 1920) < 1e-6
