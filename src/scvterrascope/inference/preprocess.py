"""Letterbox + ImageNet normalization for SCVTerraScope inference.

`letterbox_resize` + `IMAGENET_MEAN` / `IMAGENET_STD` are vendored from
`SCVTerraVision/training/train_detection.py` @ 2026-04-26. They MUST stay
byte-equivalent — the inverse mapping in `unproject_letterbox_xyxy` (and
the model itself) assumes the same scale-and-pad rule used at training
time. If the training side ever changes the letterbox policy (different
pad value, top-left vs centered padding, etc.), update this file in
lockstep. See `docs/decisions/20260426_vendor-model-code.md`.

Optional aspect-ratio center-crop is layered ON TOP of the canonical
letterbox path (not vendored — SCVTerraScope-specific OOD mitigation).
When enabled, the input image is center-cropped to a target aspect ratio
(default 1.20, matching CODa training) BEFORE letterboxing. This puts the
content distribution closer to what the trained model expects, which
sharply improves prediction accuracy on inputs whose aspect differs
significantly from training (e.g., 16:9 dashcam at 1.78:1). The cropped
region's offset travels through the pipeline so detection bboxes are
returned in TRUE-ORIGINAL pixel coordinates — the GUI can keep showing
the full untouched image with boxes in the right place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

# ImageNet normalization constants — these are part of the trained model's
# input contract. Changing them silently degrades AP without any error.
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

# CODa training aspect ratio (1224 / 1024). Used as the center-crop target
# whenever `aspect_crop_mode != "none"`.
DEFAULT_TARGET_ASPECT: float = 1224 / 1024  # 1.1953...
DEFAULT_ASPECT_TOLERANCE: float = 0.10      # within ±10% of target → no crop in "auto"

AspectCropMode = Literal["none", "auto", "always"]

# letterbox_resize padding-position knob.
#   "bottom_right":  pad on bottom and right only. Byte-equivalent to the
#                    SCVTerraVision training preprocess; safe default.
#   "symmetric":     split padding equally on top/bottom (or left/right).
#                    Useful when the input aspect differs sharply from the
#                    training distribution — content lands in the middle of
#                    the letterbox where the model learned to look for it.
#   "auto":          'bottom_right' if input aspect is within
#                    DEFAULT_ASPECT_TOLERANCE of training, else 'symmetric'.
PadPosition = Literal["bottom_right", "symmetric", "auto"]


@dataclass(frozen=True)
class Preprocessed:
    pixel_values: Any  # numpy (3, H, W) float32, ImageNet-normalized
    pixel_mask: Any    # numpy (H, W) uint8, 1 = real image area
    scale: float       # multiply original-pixel coords by this to get letterbox coords
    pad_h: int         # total vertical padding (pad_top + pad_bottom)
    pad_w: int         # total horizontal padding (pad_left + pad_right)
    target_size: int
    # Size (H, W) of the image *as fed to letterbox* — equals the cropped
    # size if center-cropping was applied, else the true input size.
    original_size: tuple[int, int]
    # Crop offset (x, y) inside the true-original image. (0, 0) means no
    # crop was applied. unproject adds this to bring boxes into true-original
    # coordinates.
    crop_offset: tuple[int, int] = (0, 0)
    # Size (H, W) of the TRUE original image (pre-crop). Used by unproject
    # to clamp box edges to the full image bounds. If None, equals
    # `original_size`.
    true_original_size: tuple[int, int] | None = None
    # Where the content sits inside the (target_size × target_size) letterbox.
    # `pad_top + pad_bottom = pad_h`, `pad_left + pad_right = pad_w`.
    # Default 0/0 reproduces the legacy bottom-right layout (pad_top=0,
    # pad_left=0, all padding on bottom-right).
    pad_top: int = 0
    pad_left: int = 0

    def true_size_or_original(self) -> tuple[int, int]:
        return self.true_original_size if self.true_original_size is not None else self.original_size


def letterbox_resize(
    image: Any,
    target_size: int,
    pad_value: int = 114,
    *,
    pad_position: PadPosition = "bottom_right",
    target_aspect: float = DEFAULT_TARGET_ASPECT,
    aspect_tolerance: float = DEFAULT_ASPECT_TOLERANCE,
) -> tuple[Any, float, tuple[int, int, int, int]]:
    """Aspect-preserving resize + pad to (target_size, target_size).

    The 'bottom_right' default is byte-equivalent to the vendored
    SCVTerraVision training preprocess. 'symmetric' centers the content
    inside the letterbox — same content/padding ratio, but the padding
    is split top/bottom (or left/right) so the model's learned spatial
    priors land on real pixels rather than gray bars when the input
    aspect differs sharply from training.

    Returns:
        padded image HxWx3 uint8,
        scale factor (target_size / max(orig_h, orig_w)),
        (pad_top, pad_bottom, pad_left, pad_right).
    """
    import numpy as np

    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))

    from PIL import Image

    resized = np.asarray(
        Image.fromarray(image).resize((new_w, new_h), Image.BILINEAR),
        dtype=np.uint8,
    )

    pad_h_total = target_size - new_h
    pad_w_total = target_size - new_w

    resolved = pad_position
    if resolved == "auto":
        cur_aspect = w / h
        # If close to training aspect, the bottom-right convention matches
        # what the model was trained on — keep that for fidelity.
        if abs(cur_aspect / target_aspect - 1.0) <= aspect_tolerance:
            resolved = "bottom_right"
        else:
            resolved = "symmetric"

    if resolved == "symmetric":
        pad_top = pad_h_total // 2
        pad_left = pad_w_total // 2
    else:  # 'bottom_right'
        pad_top = 0
        pad_left = 0
    pad_bottom = pad_h_total - pad_top
    pad_right = pad_w_total - pad_left

    padded = np.full((target_size, target_size, 3), pad_value, dtype=np.uint8)
    padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    return padded, scale, (pad_top, pad_bottom, pad_left, pad_right)


def maybe_center_crop(
    image: Any,
    *,
    mode: AspectCropMode = "none",
    target_aspect: float = DEFAULT_TARGET_ASPECT,
    tolerance: float = DEFAULT_ASPECT_TOLERANCE,
) -> tuple[Any, tuple[int, int]]:
    """Optionally center-crop a PIL.Image to `target_aspect` (W/H).

    Returns:
        (cropped_image, (x_offset, y_offset)) in true-original pixel coords.

    Modes:
        - "none":   never crop. Returns image unchanged, offset (0, 0).
        - "auto":   crop only if abs(aspect/target - 1) > tolerance.
        - "always": always crop to target aspect (a no-op if already exact).

    The crop is centered: equal pixels removed from left/right (when too
    wide) or top/bottom (when too tall). 1-pixel rounding errors fall on
    the right/bottom side.
    """
    if mode == "none":
        return image, (0, 0)

    w, h = image.size
    cur_aspect = w / h

    if mode == "auto":
        # Within tolerance band → no crop, model is fine without it.
        if abs(cur_aspect / target_aspect - 1.0) <= tolerance:
            return image, (0, 0)

    if cur_aspect > target_aspect:
        # Too wide — crop horizontally.
        new_w = int(round(h * target_aspect))
        if new_w >= w:
            return image, (0, 0)
        x_offset = (w - new_w) // 2
        cropped = image.crop((x_offset, 0, x_offset + new_w, h))
        return cropped, (x_offset, 0)
    else:
        # Too tall — crop vertically.
        new_h = int(round(w / target_aspect))
        if new_h >= h:
            return image, (0, 0)
        y_offset = (h - new_h) // 2
        cropped = image.crop((0, y_offset, w, y_offset + new_h))
        return cropped, (0, y_offset)


def preprocess_pil(
    image: Any,
    target_size: int = 1024,
    *,
    aspect_crop_mode: AspectCropMode = "none",
    pad_position: PadPosition = "bottom_right",
    target_aspect: float = DEFAULT_TARGET_ASPECT,
    aspect_tolerance: float = DEFAULT_ASPECT_TOLERANCE,
) -> Preprocessed:
    """Letterbox + ImageNet-normalize a PIL.Image, with optional pre-crop.

    Knobs (each with a training-byte-equivalent default):
      - `aspect_crop_mode="none"`: no center crop.
      - `pad_position="bottom_right"`: padding goes on bottom/right only.

    Recommended for OOD aspects (e.g., 16:9 dashcam frames):
      - `pad_position="auto"` (or "symmetric"): centers the content vertically
        in the letterbox so the model's spatial priors align with real pixels
        rather than the gray padding strip.
    """
    import numpy as np

    if image.mode != "RGB":
        image = image.convert("RGB")

    true_w, true_h = image.size  # PIL convention: (W, H)
    cropped, (cx, cy) = maybe_center_crop(
        image,
        mode=aspect_crop_mode,
        target_aspect=target_aspect,
        tolerance=aspect_tolerance,
    )
    arr = np.asarray(cropped, dtype=np.uint8)  # (H, W, 3)
    orig_h, orig_w = arr.shape[:2]

    padded, scale, (pad_top, pad_bottom, pad_left, pad_right) = letterbox_resize(
        arr,
        target_size,
        pad_position=pad_position,
        target_aspect=target_aspect,
        aspect_tolerance=aspect_tolerance,
    )

    norm = padded.astype(np.float32) / 255.0
    norm = (norm - np.asarray(IMAGENET_MEAN, dtype=np.float32)) / np.asarray(
        IMAGENET_STD, dtype=np.float32
    )
    pixel_values = norm.transpose(2, 0, 1)  # (3, H, W)

    pixel_mask = np.zeros((target_size, target_size), dtype=np.uint8)
    new_h = target_size - pad_top - pad_bottom
    new_w = target_size - pad_left - pad_right
    pixel_mask[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = 1

    cropped_size = (int(orig_h), int(orig_w))
    true_size = (int(true_h), int(true_w))

    return Preprocessed(
        pixel_values=pixel_values,
        pixel_mask=pixel_mask,
        scale=float(scale),
        pad_h=int(pad_top + pad_bottom),
        pad_w=int(pad_left + pad_right),
        target_size=int(target_size),
        original_size=cropped_size,
        crop_offset=(int(cx), int(cy)),
        true_original_size=true_size if (cx, cy) != (0, 0) else None,
        pad_top=int(pad_top),
        pad_left=int(pad_left),
    )


def unproject_letterbox_xyxy(
    box_xyxy: tuple[float, float, float, float],
    *,
    scale: float,
    original_size: tuple[int, int],
    crop_offset: tuple[int, int] = (0, 0),
    true_original_size: tuple[int, int] | None = None,
    pad_top: int = 0,
    pad_left: int = 0,
) -> tuple[float, float, float, float]:
    """Map an xyxy box from the letterbox frame back to true-original pixels.

    Pipeline:
        letterbox xyxy   --(- pad_top/left, ÷ scale)-->   cropped-image xyxy
        cropped-image xyxy   + crop_offset -->   true-original xyxy

    Boxes that touch the padding edges may yield slightly out-of-range
    coords; we clamp to the true-original bounds rather than drop so the
    GUI can still render a partial box.
    """
    cx, cy = crop_offset
    if true_original_size is None:
        true_original_size = original_size
    true_h, true_w = true_original_size

    x1, y1, x2, y2 = box_xyxy
    # Subtract padding offset so the box is in the resized-content frame,
    # then divide by scale to land in cropped-image coords.
    x1c = (x1 - pad_left) / scale
    y1c = (y1 - pad_top) / scale
    x2c = (x2 - pad_left) / scale
    y2c = (y2 - pad_top) / scale

    # Translate from cropped-image to true-original coordinates.
    x1o = x1c + cx
    y1o = y1c + cy
    x2o = x2c + cx
    y2o = y2c + cy

    # Clamp to true-original bounds.
    x1o = max(0.0, min(float(true_w), x1o))
    x2o = max(0.0, min(float(true_w), x2o))
    y1o = max(0.0, min(float(true_h), y1o))
    y2o = max(0.0, min(float(true_h), y2o))
    return x1o, y1o, x2o, y2o
