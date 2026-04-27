"""Engine factory + Protocol shape tests (no model load)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scvterrascope.inference import (
    BaseInferenceEngine,
    InferenceEngine,
    YoloEngine,
    build_engine,
    family_for,
    model_choices,
)


def test_model_registry_has_dinov3_and_yolo() -> None:
    keys = [k for k, _ in model_choices()]
    assert "dinov3_detr" in keys
    assert "yolo12n" in keys
    assert "yolo12x" in keys
    assert "yolo11n" in keys


def test_family_dispatch() -> None:
    assert family_for("dinov3_detr") == "dinov3_detr"
    assert family_for("yolo12n") == "yolo"
    assert family_for("yolo11x") == "yolo"
    assert family_for("madeup") == "unknown"


def test_build_engine_yolo_uses_name_pt(tmp_path: Path) -> None:
    """YOLO build should default to '<name>.pt' even when checkpoint_path is None."""
    eng = build_engine(
        "yolo12n",
        checkpoint_path=None,
        device="cpu",
        image_size=640,
        top_k=50,
    )
    assert isinstance(eng, YoloEngine)
    assert str(eng.checkpoint_path).endswith("yolo12n.pt")


def test_build_engine_dinov3_requires_checkpoint() -> None:
    with pytest.raises(ValueError):
        build_engine(
            "dinov3_detr",
            checkpoint_path=None,
            device="cpu",
            image_size=1024,
            top_k=100,
        )


def test_engines_share_required_methods() -> None:
    """Both engines must expose the same method surface for the GUI.

    isinstance against the runtime_checkable Protocol would trip the
    @property guards (`device` raises before load), so we verify the
    callable surface directly.
    """
    detr = InferenceEngine(checkpoint_path="/dev/null")
    yolo = YoloEngine(checkpoint_path="yolo12n.pt")
    for attr in ("is_loaded", "load", "predict",
                 "checkpoint_path", "image_size", "top_k",
                 "aspect_crop_mode", "pad_position"):
        assert hasattr(detr, attr), f"DINOv3+DETR missing {attr}"
        assert hasattr(yolo, attr), f"YOLO missing {attr}"
    # Mutable attrs the GUI flips after construction.
    yolo.aspect_crop_mode = "auto"
    yolo.pad_position = "symmetric"
    assert yolo.aspect_crop_mode == "auto"


def test_yolo_engine_carries_letterbox_knobs() -> None:
    """Sanity: YOLO engine accepts (and ignores) the DETR-style preprocess knobs
    so the GUI can flip them without special-casing."""
    eng = YoloEngine(
        checkpoint_path="yolo12n.pt",
        aspect_crop_mode="auto",
        pad_position="symmetric",
        top_k=42,
        image_size=640,
    )
    assert eng.aspect_crop_mode == "auto"
    assert eng.pad_position == "symmetric"
    assert eng.top_k == 42
    assert eng.image_size == 640
