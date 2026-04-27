"""High-level inference orchestration: checkpoint → PIL → list[Detection].

This module mirrors the inference loop pattern from SCVTerraVision's
`scripts/eval_detection.py` so that boxes shown in the GUI match what
training-side metrics see. The model architecture itself is vendored —
see `scvterrascope.model.detr_head.DinoV3DeformableDetr`.

Heavy imports (torch / transformers) are deferred until `load()` is
called, so unit tests that don't need a model can still import this
module on machines without the ML stack provisioned.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scvterrascope.inference.preprocess import (
    DEFAULT_TARGET_ASPECT,
    AspectCropMode,
    PadPosition,
    Preprocessed,
    preprocess_pil,
    unproject_letterbox_xyxy,
)
from scvterrascope.labels import Taxonomy, load_taxonomy

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    """One predicted box in original-image pixel coordinates."""

    class_id: int                                    # 1-indexed CODa category id
    class_name: str
    score: float
    bbox_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class InferenceResult:
    detections: tuple[Detection, ...]
    inference_ms: float                              # forward pass alone (cuda-synced)
    image_size: tuple[int, int]                      # (H, W) of the input image
    preprocess: Preprocessed
    # Stage-level timings — useful to spot whether the GUI is preprocess- or
    # forward-bound at a glance.
    preprocess_ms: float = 0.0
    postprocess_ms: float = 0.0
    # GPU memory snapshot during this prediction. Both 0.0 on CPU.
    gpu_mem_alloc_mb: float = 0.0   # currently allocated at end of inference
    gpu_mem_peak_mb: float = 0.0    # max allocated during inference
    gpu_mem_total_mb: float = 0.0   # device total VRAM

    @property
    def total_ms(self) -> float:
        return self.preprocess_ms + self.inference_ms + self.postprocess_ms

    @property
    def fps(self) -> float:
        return 1000.0 / self.total_ms if self.total_ms > 0 else 0.0


def _resolve_device(spec: str) -> str:
    """Map an `auto` / `cuda` / `cpu` spec to a concrete torch device string."""
    import torch

    if spec == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if spec.startswith("cuda") and not torch.cuda.is_available():
        LOG.warning("requested %s but CUDA unavailable; falling back to cpu", spec)
        return "cpu"
    return spec


class InferenceEngine:
    """Loads a DINOv3 + Deformable DETR checkpoint and runs inference.

    The score threshold is applied **on the postprocessor output**, not in
    HF's internal `top_k`. This way the GUI can drop the slider without
    re-running forward.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "auto",
        image_size: int = 1024,
        top_k: int = 100,
        taxonomy: Taxonomy | None = None,
        aspect_crop_mode: AspectCropMode = "none",
        pad_position: PadPosition = "auto",
        target_aspect: float = DEFAULT_TARGET_ASPECT,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self._device_spec = device
        self.image_size = int(image_size)
        self.top_k = int(top_k)
        self._taxonomy = taxonomy
        # Aspect handling is mutable at runtime so the GUI can flip it
        # without reloading the checkpoint.
        self.aspect_crop_mode: AspectCropMode = aspect_crop_mode
        self.pad_position: PadPosition = pad_position
        self.target_aspect: float = float(target_aspect)
        self._device: str | None = None
        self._model: Any = None
        self._processor: Any = None
        self._epoch: int | None = None
        self._param_count: int | None = None
        self._gpu_total_mb: float = 0.0
        self._device_name: str = ""

    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Materialize the model + checkpoint. Idempotent."""
        if self.is_loaded():
            return
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {self.checkpoint_path}")

        try:
            import torch
            from transformers import DeformableDetrImageProcessor

            from scvterrascope.model import DinoV3DeformableDetr
        except ImportError as exc:
            raise ImportError(
                "torch / transformers must be installed. "
                "See docs/runbooks/phase1-1_launch.md."
            ) from exc

        self._device = _resolve_device(self._device_spec)
        LOG.info("device=%s", self._device)

        wrapper = DinoV3DeformableDetr()
        model = wrapper.load()
        model.to(self._device).eval()

        ckpt = torch.load(self.checkpoint_path, map_location=self._device)
        state = ckpt.get("model_state", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            LOG.warning("missing keys: %d (e.g., %s)", len(missing), missing[:3])
        if unexpected:
            LOG.warning("unexpected keys: %d (e.g., %s)", len(unexpected), unexpected[:3])
        self._epoch = int(ckpt.get("epoch", -1))

        self._model = model
        self._processor = DeformableDetrImageProcessor()
        self._param_count = sum(p.numel() for p in model.parameters())

        if self._device.startswith("cuda"):
            idx = torch.cuda.current_device()
            self._device_name = torch.cuda.get_device_name(idx)
            self._gpu_total_mb = torch.cuda.get_device_properties(idx).total_memory / (1024 * 1024)

        if self._taxonomy is None:
            self._taxonomy = load_taxonomy()
        LOG.info(
            "loaded %s (epoch=%s) — %d-class taxonomy, %s params",
            self.checkpoint_path.name,
            self._epoch,
            len(self._taxonomy),
            f"{self._param_count / 1e6:.1f}M",
        )

    @property
    def device(self) -> str:
        if self._device is None:
            raise RuntimeError("call load() before reading device")
        return self._device

    @property
    def epoch(self) -> int:
        if self._epoch is None:
            raise RuntimeError("call load() before reading epoch")
        return self._epoch

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
    def device_name(self) -> str:
        """Human-readable device name (e.g., 'NVIDIA GeForce RTX 4060 Ti'). Empty on CPU."""
        return self._device_name

    @property
    def gpu_total_mb(self) -> float:
        return self._gpu_total_mb

    def predict(self, image: Any) -> InferenceResult:
        """Run a single forward + postprocess on a PIL.Image.

        Returns ALL detections above HF's internal threshold (0.0). The GUI
        is responsible for filtering by score_threshold. Times each stage
        (preprocess / forward / postprocess) and snapshots GPU memory so
        the GUI's PerformancePanel can show throughput at a glance.
        """
        if not self.is_loaded():
            raise RuntimeError("call load() before predict()")
        import torch

        on_cuda = self._device.startswith("cuda")
        if on_cuda:
            # Reset peak counter so this prediction's number isn't polluted
            # by earlier inferences.
            torch.cuda.reset_peak_memory_stats()

        # ---- preprocess ----
        t0 = time.perf_counter()
        pre = preprocess_pil(
            image,
            target_size=self.image_size,
            aspect_crop_mode=self.aspect_crop_mode,
            pad_position=self.pad_position,
            target_aspect=self.target_aspect,
        )
        pixel_values = torch.from_numpy(pre.pixel_values).unsqueeze(0).to(self._device)
        pixel_mask = torch.from_numpy(pre.pixel_mask).unsqueeze(0).long().to(self._device)
        if on_cuda:
            torch.cuda.synchronize()
        preprocess_ms = (time.perf_counter() - t0) * 1000.0

        # ---- forward ----
        t0 = time.perf_counter()
        with torch.inference_mode():
            outputs = self._model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        if on_cuda:
            torch.cuda.synchronize()
        forward_ms = (time.perf_counter() - t0) * 1000.0

        # ---- postprocess ----
        t0 = time.perf_counter()
        target_sizes = torch.tensor(
            [[self.image_size, self.image_size]],
            dtype=torch.long,
            device=self._device,
        )
        results = self._processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.0, top_k=self.top_k
        )[0]
        detections = self._results_to_detections(results, pre)
        if on_cuda:
            torch.cuda.synchronize()
        postprocess_ms = (time.perf_counter() - t0) * 1000.0

        gpu_alloc_mb = 0.0
        gpu_peak_mb = 0.0
        if on_cuda:
            gpu_alloc_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            gpu_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        return InferenceResult(
            detections=detections,
            inference_ms=forward_ms,
            image_size=pre.original_size,
            preprocess=pre,
            preprocess_ms=preprocess_ms,
            postprocess_ms=postprocess_ms,
            gpu_mem_alloc_mb=gpu_alloc_mb,
            gpu_mem_peak_mb=gpu_peak_mb,
            gpu_mem_total_mb=self._gpu_total_mb,
        )

    def _results_to_detections(
        self, per_image: dict[str, Any], pre: Preprocessed
    ) -> tuple[Detection, ...]:
        scores = per_image["scores"].cpu().tolist()
        labels = per_image["labels"].cpu().tolist()
        boxes = per_image["boxes"].cpu().tolist()  # xyxy in letterbox frame

        out: list[Detection] = []
        assert self._taxonomy is not None
        for score, label, box in zip(scores, labels, boxes):
            x1, y1, x2, y2 = unproject_letterbox_xyxy(
                tuple(box),
                scale=pre.scale,
                original_size=pre.original_size,
                crop_offset=pre.crop_offset,
                true_original_size=pre.true_original_size,
                pad_top=pre.pad_top,
                pad_left=pre.pad_left,
            )
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                continue
            try:
                name = self._taxonomy.name_for(int(label))
            except IndexError:
                LOG.warning("label %d out of taxonomy range — skipping box", label)
                continue
            out.append(
                Detection(
                    class_id=int(label) + 1,        # back to 1-indexed COCO
                    class_name=name,
                    score=float(score),
                    bbox_xyxy=(x1, y1, x2, y2),
                )
            )
        return tuple(out)
