"""Ultralytics YOLO11/YOLO12 inference engine.

Conforms to `BaseInferenceEngine` so the GUI can swap it in for the
DINOv3+DETR engine without changes downstream of `predict()`. The user-
visible difference is the class set (COCO 80 vs CODa 16) and the
inference speed.

We do NOT vendor any YOLO model code — Ultralytics is a stable pip
package that handles weight download, letterbox, NMS, postprocess for
us. Our wrapper just maps Ultralytics' Results to our `Detection` /
`InferenceResult` types.

Aspect-crop / pad-position knobs are accepted but ignored — YOLO does
its own internal letterbox at the requested `imgsz`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scvterrascope.inference.engine import (
    Detection,
    InferenceResult,
    _resolve_device,
)
from scvterrascope.inference.preprocess import (
    DEFAULT_TARGET_ASPECT,
    AspectCropMode,
    PadPosition,
    Preprocessed,
)
from scvterrascope.labels import OperationalClass, Taxonomy

LOG = logging.getLogger(__name__)


# Acceptable Ultralytics weight names that we expose to the GUI dropdown.
# Other custom .pt paths can be passed through `checkpoint_path` directly.
SUPPORTED_YOLO_NAMES = (
    "yolo12n",
    "yolo12s",
    "yolo12m",
    "yolo12l",
    "yolo12x",
    "yolo11n",
    "yolo11s",
    "yolo11m",
    "yolo11l",
    "yolo11x",
)


@dataclass
class YoloEngineConfig:
    """Light wrapper so the factory can build either by name or by path."""

    weight: str | Path = "yolo12n.pt"  # name (auto-download) or local .pt path
    image_size: int = 1024


class YoloEngine:
    """Ultralytics-backed inference. Same public shape as InferenceEngine."""

    def __init__(
        self,
        checkpoint_path: str | Path = "yolo12n.pt",
        *,
        device: str = "auto",
        image_size: int = 1024,
        top_k: int = 100,
        # Accepted for interface parity; ignored — Ultralytics letterboxes.
        aspect_crop_mode: AspectCropMode = "none",
        pad_position: PadPosition = "bottom_right",
        target_aspect: float = DEFAULT_TARGET_ASPECT,
        taxonomy: Taxonomy | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self._device_spec = device
        self.image_size = int(image_size)
        self.top_k = int(top_k)
        self.aspect_crop_mode: AspectCropMode = aspect_crop_mode
        self.pad_position: PadPosition = pad_position
        self.target_aspect: float = float(target_aspect)

        self._device: str | None = None
        self._model: Any = None
        self._taxonomy = taxonomy
        self._param_count: int | None = None
        self._device_name: str = ""
        self._gpu_total_mb: float = 0.0

    # ----- lifecycle -----------------------------------------------
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self.is_loaded():
            return

        try:
            import torch
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "YoloEngine requires ultralytics. `pip install ultralytics` "
                "or rerun `pip install -e .[dev]`."
            ) from exc

        self._device = _resolve_device(self._device_spec)
        LOG.info("loading YOLO from %s on device=%s", self.checkpoint_path, self._device)

        # Ultralytics resolves bare names like 'yolo12n.pt' against its
        # own assets server and caches in cwd / ~/.config/Ultralytics.
        # If the user gave an explicit path we pass it through verbatim.
        weight = str(self.checkpoint_path)
        self._model = YOLO(weight)
        # Force one warmup forward on the right device + sync VRAM stats.
        self._model.to(self._device)

        # Ultralytics' .names is {idx: name}; build a Taxonomy that mirrors
        # our 1-indexed `class_id` convention so draw_detections / palette
        # don't need a separate code path.
        if self._taxonomy is None:
            classes = tuple(
                OperationalClass(id=i + 1, name=str(self._model.names[i]))
                for i in sorted(self._model.names.keys())
            )
            self._taxonomy = Taxonomy(classes=classes)

        # Param count — sum across the underlying torch module.
        try:
            self._param_count = sum(p.numel() for p in self._model.model.parameters())
        except Exception:  # noqa: BLE001 — defensive: Ultralytics layout may shift
            self._param_count = 0

        if self._device.startswith("cuda"):
            idx = torch.cuda.current_device()
            self._device_name = torch.cuda.get_device_name(idx)
            self._gpu_total_mb = torch.cuda.get_device_properties(idx).total_memory / (1024 * 1024)

        LOG.info(
            "loaded YOLO %s — %d-class taxonomy, %.1fM params",
            self.checkpoint_path.name,
            len(self._taxonomy),
            (self._param_count or 0) / 1e6,
        )

    # ----- properties (BaseInferenceEngine parity) -----------------
    @property
    def device(self) -> str:
        if self._device is None:
            raise RuntimeError("call load() before reading device")
        return self._device

    @property
    def device_name(self) -> str:
        return self._device_name

    @property
    def epoch(self) -> int:
        # YOLO pretrained weights have no notion of "training epoch the
        # user is at". -1 is the same sentinel DETR uses for "unknown".
        return -1

    @property
    def taxonomy(self) -> Taxonomy:
        if self._taxonomy is None:
            raise RuntimeError("call load() before reading taxonomy")
        return self._taxonomy

    @property
    def param_count(self) -> int:
        if self._param_count is None:
            raise RuntimeError("call load() before reading param_count")
        return self._param_count

    @property
    def gpu_total_mb(self) -> float:
        return self._gpu_total_mb

    # ----- inference -----------------------------------------------
    def predict(self, image: Any) -> InferenceResult:
        """Run YOLO predict on a PIL.Image and adapt to InferenceResult."""
        if not self.is_loaded():
            raise RuntimeError("call load() before predict()")
        import torch

        on_cuda = self._device.startswith("cuda") if self._device else False
        if on_cuda:
            torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()

        # Ultralytics accepts PIL.Image directly; threshold=0.0 so the GUI
        # slider can re-filter without re-running the model. `verbose=False`
        # mutes the per-frame stdout spam.
        results = self._model.predict(
            source=image,
            imgsz=self.image_size,
            conf=0.0,
            max_det=self.top_k,
            device=self._device,
            verbose=False,
        )
        if on_cuda:
            torch.cuda.synchronize()
        wall_total_ms = (time.perf_counter() - t0) * 1000.0

        if not results:
            raise RuntimeError("Ultralytics returned an empty results list")
        r0 = results[0]

        # Ultralytics records its own preprocess/inference/postprocess
        # times in milliseconds (without cuda sync — close enough for the
        # PerformancePanel display).
        speed = dict(getattr(r0, "speed", {}) or {})
        pre_ms = float(speed.get("preprocess", 0.0))
        fwd_ms = float(speed.get("inference", 0.0))
        post_ms = float(speed.get("postprocess", 0.0))
        # If the speed dict is empty (older builds), fall back to wall time.
        if pre_ms == fwd_ms == post_ms == 0.0:
            fwd_ms = wall_total_ms

        # Detections — boxes is None when nothing fired (rare with conf=0).
        detections: list[Detection] = []
        boxes = getattr(r0, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.detach().cpu().tolist()
            confs = boxes.conf.detach().cpu().tolist()
            cls_idx = boxes.cls.detach().cpu().tolist()
            assert self._taxonomy is not None
            for box, conf, ci in zip(xyxy, confs, cls_idx):
                ci_int = int(ci)
                try:
                    name = self._taxonomy.name_for(ci_int)
                except IndexError:
                    LOG.warning("YOLO label %d outside taxonomy — skipping", ci_int)
                    continue
                x1, y1, x2, y2 = (float(v) for v in box)
                detections.append(
                    Detection(
                        class_id=ci_int + 1,
                        class_name=name,
                        score=float(conf),
                        bbox_xyxy=(x1, y1, x2, y2),
                    )
                )

        gpu_alloc_mb = 0.0
        gpu_peak_mb = 0.0
        if on_cuda:
            gpu_alloc_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            gpu_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        # Build a Preprocessed stub — InferenceResult requires it for
        # downstream code (e.g., the diagnose script). Ultralytics handles
        # letterbox internally so we report sensible scalar metadata
        # without numpy buffers.
        import numpy as np

        w, h = image.size
        pre = Preprocessed(
            pixel_values=np.zeros((3, self.image_size, self.image_size), dtype=np.float32),
            pixel_mask=np.ones((self.image_size, self.image_size), dtype=np.uint8),
            scale=self.image_size / max(h, w),
            pad_h=0,
            pad_w=0,
            target_size=self.image_size,
            original_size=(int(h), int(w)),
            crop_offset=(0, 0),
            true_original_size=None,
            pad_top=0,
            pad_left=0,
        )

        return InferenceResult(
            detections=tuple(detections),
            inference_ms=fwd_ms,
            image_size=(int(h), int(w)),
            preprocess=pre,
            preprocess_ms=pre_ms,
            postprocess_ms=post_ms,
            gpu_mem_alloc_mb=gpu_alloc_mb,
            gpu_mem_peak_mb=gpu_peak_mb,
            gpu_mem_total_mb=self._gpu_total_mb,
        )
