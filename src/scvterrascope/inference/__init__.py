"""Model loading, preprocessing, and inference orchestration."""

from scvterrascope.inference.engine import (
    Detection,
    InferenceEngine,
    InferenceResult,
)
from scvterrascope.inference.preprocess import (
    DEFAULT_TARGET_ASPECT,
    AspectCropMode,
    PadPosition,
    Preprocessed,
    maybe_center_crop,
    preprocess_pil,
    unproject_letterbox_xyxy,
)

__all__ = [
    "AspectCropMode",
    "DEFAULT_TARGET_ASPECT",
    "Detection",
    "InferenceEngine",
    "InferenceResult",
    "PadPosition",
    "Preprocessed",
    "maybe_center_crop",
    "preprocess_pil",
    "unproject_letterbox_xyxy",
]
