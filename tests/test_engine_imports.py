"""Engine module imports cheaply; vendored model is reachable."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_engine_module_imports_without_model_load() -> None:
    """Module-level import must not eagerly load torch / build the model."""
    from scvterrascope.inference import (
        Detection,
        InferenceEngine,
        InferenceResult,
    )

    eng = InferenceEngine(checkpoint_path="/nonexistent.pt")
    assert not eng.is_loaded()
    assert eng.image_size == 1024
    assert eng.top_k == 100
    det = Detection(class_id=1, class_name="x", score=0.5, bbox_xyxy=(0.0, 0.0, 1.0, 1.0))
    assert det.class_id == 1


def test_load_raises_when_checkpoint_missing(tmp_path: Path) -> None:
    from scvterrascope.inference import InferenceEngine

    eng = InferenceEngine(checkpoint_path=tmp_path / "definitely_not_here.pt")
    with pytest.raises(FileNotFoundError):
        eng.load()


def test_vendored_model_is_importable() -> None:
    """`scvterrascope.model` must be self-contained — no SCVTerraVision needed."""
    from scvterrascope.model import (
        DetrHeadConfig,
        DinoV3Backbone,
        DinoV3BackboneConfig,
        DinoV3DeformableDetr,
    )

    # Construction without torch should succeed (lazy imports).
    head_cfg = DetrHeadConfig()
    assert head_cfg.num_labels == 16
    assert head_cfg.d_model == 256
    bb_cfg = DinoV3BackboneConfig()
    assert bb_cfg.patch_size == 16
    assert bb_cfg.hidden_dim == 768
    bb = DinoV3Backbone(bb_cfg)
    # Patch math is pure-Python — runs without torch.
    assert bb.patch_grid_shape(1024, 1024) == (64, 64)
    # Wrapper instantiation is also lazy.
    wrapper = DinoV3DeformableDetr(head_cfg, bb_cfg)
    assert wrapper.head_config is head_cfg
    assert wrapper.backbone is not None


def test_no_terravision_path_helper_remaining() -> None:
    """The old sys.path injection helper must be gone."""
    import importlib

    with pytest.raises(ImportError):
        importlib.import_module("scvterrascope._terravision_path")


def test_inference_result_has_perf_fields() -> None:
    """PerformancePanel relies on these fields; pin them in a test."""
    from scvterrascope.inference.engine import InferenceResult
    from scvterrascope.inference.preprocess import Preprocessed
    import numpy as np

    pre = Preprocessed(
        pixel_values=np.zeros((3, 8, 8), dtype=np.float32),
        pixel_mask=np.ones((8, 8), dtype=np.uint8),
        scale=1.0, pad_h=0, pad_w=0,
        target_size=8, original_size=(8, 8),
    )
    r = InferenceResult(
        detections=(),
        inference_ms=10.0,
        image_size=(8, 8),
        preprocess=pre,
        preprocess_ms=2.0,
        postprocess_ms=1.0,
        gpu_mem_alloc_mb=100.0,
        gpu_mem_peak_mb=150.0,
        gpu_mem_total_mb=8000.0,
    )
    assert r.total_ms == 13.0
    assert abs(r.fps - 1000.0 / 13.0) < 1e-6
    # Defaults still work — backwards compat for any older tests.
    r0 = InferenceResult(detections=(), inference_ms=10.0, image_size=(8, 8), preprocess=pre)
    assert r0.preprocess_ms == 0.0
    assert r0.postprocess_ms == 0.0
    assert r0.gpu_mem_alloc_mb == 0.0
