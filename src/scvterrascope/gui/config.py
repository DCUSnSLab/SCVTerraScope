"""Load/merge SCVTerraScope GUI configuration from YAML + CLI overrides."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AppConfig:
    checkpoint_path: Path | None = None
    device: str = "auto"
    image_size: int = 1024
    score_threshold: float = 0.30
    top_k: int = 100
    taxonomy_path: Path | None = None
    samples_dir: Path = field(default_factory=lambda: Path("data/coda_samples"))
    # Aspect-crop strategy when input image's W:H differs from the model's
    # training aspect (~1.20:1 for CODa). "none" preserves prior behavior;
    # "auto" crops only when the input is far from the training aspect;
    # "always" crops every input. See preprocess.maybe_center_crop.
    aspect_crop_mode: str = "auto"
    # Letterbox padding placement: "bottom_right" (training default),
    # "symmetric" (centered), or "auto" (auto-pick by aspect).
    pad_position: str = "bottom_right"
    # Inference backend. See scvterrascope.inference.factory.MODEL_REGISTRY:
    # "dinov3_detr" | "yolo12n" | "yolo12s" | "yolo12m" | "yolo12l" |
    # "yolo12x" | "yolo11n" | "yolo11x".
    model_kind: str = "dinov3_detr"


def _coerce_path(value: Any) -> Path | None:
    if value is None or value == "":
        return None
    return Path(value).expanduser()


def load_config(path: Path | None = None) -> AppConfig:
    """Load YAML config; missing keys keep their dataclass defaults."""
    if path is None:
        path = Path("configs/default.yaml")
    if not path.is_file():
        return AppConfig()
    import yaml

    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg = AppConfig()
    cfg.checkpoint_path = _coerce_path(raw.get("checkpoint_path"))
    cfg.device = str(raw.get("device", cfg.device))
    cfg.image_size = int(raw.get("image_size", cfg.image_size))
    cfg.score_threshold = float(raw.get("score_threshold", cfg.score_threshold))
    cfg.top_k = int(raw.get("top_k", cfg.top_k))
    cfg.taxonomy_path = _coerce_path(raw.get("taxonomy_path"))
    samples = _coerce_path(raw.get("samples_dir"))
    if samples is not None:
        cfg.samples_dir = samples
    crop = raw.get("aspect_crop_mode")
    if crop in ("none", "auto", "always"):
        cfg.aspect_crop_mode = crop
    pad = raw.get("pad_position")
    if pad in ("bottom_right", "symmetric", "auto"):
        cfg.pad_position = pad
    model = raw.get("model_kind")
    if isinstance(model, str) and model:
        cfg.model_kind = model
    return cfg
