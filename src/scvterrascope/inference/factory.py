"""Build an inference engine by name (string used in AppConfig + GUI).

Names follow this convention:
  - "dinov3_detr"     → SCVTerraVision-trained DINOv3 + DeformableDETR
  - "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"
  - "yolo12n", "yolo12s", "yolo12m", "yolo12l", "yolo12x"

For the YOLO names the factory passes "<name>.pt" to Ultralytics, which
auto-downloads on first use. For DETR the caller supplies the trained
checkpoint via `checkpoint_path`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scvterrascope.inference.engine import InferenceEngine
from scvterrascope.inference.yolo_engine import SUPPORTED_YOLO_NAMES, YoloEngine

# (key, label, family). The label is what the GUI dropdown shows; family
# determines which engine class to instantiate. Order is preserved.
MODEL_REGISTRY: tuple[tuple[str, str, str], ...] = (
    ("dinov3_detr", "DINOv3 + Deformable DETR (CODa 16-class)", "dinov3_detr"),
    ("yolo12n",     "YOLO12 nano  (COCO 80-class, ~5 MB)",     "yolo"),
    ("yolo12s",     "YOLO12 small (COCO 80-class)",            "yolo"),
    ("yolo12m",     "YOLO12 medium (COCO 80-class)",           "yolo"),
    ("yolo12l",     "YOLO12 large (COCO 80-class)",            "yolo"),
    ("yolo12x",     "YOLO12 x-large (COCO 80-class)",          "yolo"),
    ("yolo11n",     "YOLO11 nano  (COCO 80-class, ~5 MB)",     "yolo"),
    ("yolo11x",     "YOLO11 x-large (COCO 80-class)",          "yolo"),
)


def model_choices() -> tuple[tuple[str, str], ...]:
    """`(key, label)` pairs the GUI dropdown displays."""
    return tuple((k, label) for k, label, _ in MODEL_REGISTRY)


def family_for(model_kind: str) -> str:
    """Look up the family ('dinov3_detr' | 'yolo') of a registered model."""
    for k, _label, family in MODEL_REGISTRY:
        if k == model_kind:
            return family
    if model_kind in SUPPORTED_YOLO_NAMES:
        return "yolo"
    return "unknown"


def build_engine(
    model_kind: str,
    *,
    checkpoint_path: Path | None,
    device: str,
    image_size: int,
    top_k: int,
    aspect_crop_mode: str = "none",
    pad_position: str = "bottom_right",
    taxonomy: Any = None,
) -> Any:
    """Instantiate the engine class that matches `model_kind`.

    For DINOv3+DETR the `checkpoint_path` must point at a SCVTerraVision
    `.pt` file. For YOLO the checkpoint is the Ultralytics weight name
    (auto-downloaded) — `checkpoint_path` from the GUI is ignored unless
    the user explicitly typed an Ultralytics-compatible .pt path.
    """
    family = family_for(model_kind)
    if family == "dinov3_detr":
        if checkpoint_path is None:
            raise ValueError("DINOv3+DETR requires checkpoint_path")
        return InferenceEngine(
            checkpoint_path=checkpoint_path,
            device=device,
            image_size=image_size,
            top_k=top_k,
            taxonomy=taxonomy,
            aspect_crop_mode=aspect_crop_mode,
            pad_position=pad_position,
        )

    if family == "yolo":
        # Map "yolo12n" → "yolo12n.pt" unless user passed an explicit path
        # ending in .pt that exists on disk.
        if checkpoint_path is not None and Path(checkpoint_path).suffix == ".pt" \
           and Path(checkpoint_path).is_file():
            weight = str(checkpoint_path)
        else:
            weight = f"{model_kind}.pt"
        return YoloEngine(
            checkpoint_path=weight,
            device=device,
            image_size=image_size,
            top_k=top_k,
            aspect_crop_mode=aspect_crop_mode,
            pad_position=pad_position,
        )

    raise ValueError(f"unknown model_kind {model_kind!r}; see MODEL_REGISTRY")
