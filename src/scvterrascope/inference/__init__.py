"""Model loading, preprocessing, and inference orchestration."""

from scvterrascope.inference.engine import (
    BaseInferenceEngine,
    Detection,
    InferenceEngine,
    InferenceResult,
)
from scvterrascope.inference.factory import (
    MODEL_REGISTRY,
    build_engine,
    family_for,
    model_choices,
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
from scvterrascope.inference.yolo_engine import SUPPORTED_YOLO_NAMES, YoloEngine

__all__ = [
    "AspectCropMode",
    "BaseInferenceEngine",
    "DEFAULT_TARGET_ASPECT",
    "Detection",
    "InferenceEngine",
    "InferenceResult",
    "MODEL_REGISTRY",
    "PadPosition",
    "Preprocessed",
    "SUPPORTED_YOLO_NAMES",
    "YoloEngine",
    "build_engine",
    "family_for",
    "maybe_center_crop",
    "model_choices",
    "preprocess_pil",
    "unproject_letterbox_xyxy",
]
