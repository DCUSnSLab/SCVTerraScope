"""Vendored DINOv3 + Deformable DETR model definitions.

This subpackage contains code copied verbatim from SCVTerraVision so that
SCVTerraScope can run independently — no `pip install -e ../SCVTerraVision`,
no sys.path injection, no fragility around the training repo's pyproject
shape. See `docs/decisions/20260426_vendor-model-code.md` for the rationale
and the sync policy that keeps these files aligned with the training side.
"""

from scvterrascope.model.backbone import (
    DEFAULT_HIDDEN_DIM,
    DEFAULT_MODEL_ID,
    DEFAULT_PATCH_SIZE,
    DinoV3Backbone,
    DinoV3BackboneConfig,
)
from scvterrascope.model.detr_head import (
    DetrHeadConfig,
    DinoV3DeformableDetr,
)

__all__ = [
    "DEFAULT_HIDDEN_DIM",
    "DEFAULT_MODEL_ID",
    "DEFAULT_PATCH_SIZE",
    "DetrHeadConfig",
    "DinoV3Backbone",
    "DinoV3BackboneConfig",
    "DinoV3DeformableDetr",
]
