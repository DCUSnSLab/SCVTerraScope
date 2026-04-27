"""DINOv3 ViT-B/16 backbone wrapper for detection heads.

Vendored from SCVTerraVision/models/backbone/dinov3_backbone.py @ 2026-04-26
and **modified to skip the HuggingFace Hub entirely** at inference time.
The training-side code did `AutoModel.from_pretrained(model_id)` which
downloads ~327MB of pretrained DINOv3 weights — but those weights get
overwritten immediately when SCVTerraScope loads `epoch_050.pt`, so the
download is pure waste. We build the architecture directly from a vendored
config dict + `AutoModel.from_config(cfg)`, leaving the parameters at
random init. The detection checkpoint then fills in the trained weights
via `load_state_dict`.

Practical consequences:
  - **No HF_TOKEN required** — no Hub call ever happens.
  - **No 327MB pretrained download** — saves bandwidth + disk.
  - **No network at inference** — the GUI works fully offline once
    transformers (the Python package) is installed.
  - Sync responsibility: if SCVTerraVision updates the backbone config
    (different hidden_size, new register tokens, etc.), update
    `_VITB16_DINOV3_CONFIG` below in lockstep with the training side.

Returns a dense (B, C, H_patch, W_patch) feature map — the layout
DETR-style heads (Phase 1-2b) expect, not the raw (B, N_tokens, C)
HF output.

Design decisions:
  - torch and transformers are imported **inside** `load()` / `forward()` so
    this module imports cleanly in environments without them. Config
    construction and `patch_grid_shape(...)` work without any ML deps.
  - DINOv3 prepends 1 CLS + K register tokens (K=4 in the current HF
    implementation). Rather than hardcoding K, we slice the last
    `H_patch * W_patch` tokens from `last_hidden_state`. That's robust to
    register-count changes across checkpoints.
  - `freeze=True` puts the backbone in eval() and disables grads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Torch / transformers are intentionally NOT imported at module scope. See
# docstring above. All methods that need them do a local import and fail
# with a helpful message if they're absent.

DEFAULT_MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DEFAULT_PATCH_SIZE = 16
DEFAULT_HIDDEN_DIM = 768  # ViT-B
_SUPPORTED_DTYPES = ("float32", "float16", "bfloat16")

# Architecture config copied from
# `~/.cache/huggingface/hub/models--facebook--dinov3-vitb16-pretrain-lvd1689m/.../config.json`
# @ 2026-04-26 (transformers 4.56.0.dev0). This is the spec — NOT the
# weights. `AutoModel.from_config(...)` instantiates with random params,
# then SCVTerraScope's checkpoint load fills in the trained values.
# Sync if SCVTerraVision changes the backbone architecture.
_VITB16_DINOV3_CONFIG = {
    "model_type": "dinov3_vit",
    "hidden_size": 768,
    "image_size": 224,
    "intermediate_size": 3072,
    "layer_norm_eps": 1e-05,
    "layerscale_value": 1.0,
    "num_attention_heads": 12,
    "num_channels": 3,
    "num_hidden_layers": 12,
    "num_register_tokens": 4,
    "patch_size": 16,
    "pos_embed_rescale": 2.0,
    "rope_theta": 100.0,
    "hidden_act": "gelu",
    "initializer_range": 0.02,
    "attention_dropout": 0.0,
    "drop_path_rate": 0.0,
    "mlp_bias": True,
    "proj_bias": True,
    "query_bias": True,
    "key_bias": False,
    "value_bias": True,
    "use_gated_mlp": False,
}


@dataclass(frozen=True)
class DinoV3BackboneConfig:
    """Serializable config for the DINOv3 backbone wrapper."""

    model_id: str = DEFAULT_MODEL_ID
    patch_size: int = DEFAULT_PATCH_SIZE
    hidden_dim: int = DEFAULT_HIDDEN_DIM
    # First-stage training freezes the backbone; the training loop unfreezes
    # after N epochs per the plan's "2-epoch freeze then low-LR fine-tune"
    # policy. For inference, leave True.
    freeze: bool = True
    # "float32" | "float16" | "bfloat16".
    dtype: str = "float32"
    # -1 = final hidden state. Integer ≥ 0 taps a specific transformer block.
    output_layer: int = -1

    def __post_init__(self) -> None:
        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {self.patch_size}")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.dtype not in _SUPPORTED_DTYPES:
            raise ValueError(
                f"dtype must be one of {_SUPPORTED_DTYPES}, got {self.dtype!r}"
            )


class DinoV3Backbone:
    """Emit (B, C, H_patch, W_patch) features from a DINOv3 ViT."""

    def __init__(self, config: DinoV3BackboneConfig | None = None) -> None:
        self.config = config or DinoV3BackboneConfig()
        self._model: Any = None  # torch.nn.Module once loaded

    # -- Shape math: callable without torch --------------------------------

    def patch_grid_shape(self, image_height: int, image_width: int) -> tuple[int, int]:
        """Expected (H_patch, W_patch) for an image of the given pixel size."""
        p = self.config.patch_size
        if image_height <= 0 or image_width <= 0:
            raise ValueError(
                f"image size must be positive, got ({image_height}, {image_width})"
            )
        if image_height % p != 0 or image_width % p != 0:
            raise ValueError(
                f"image size ({image_height}, {image_width}) not divisible by "
                f"patch_size={p}. DINOv3 ViT-B/16 requires patch-multiple inputs."
            )
        return (image_height // p, image_width // p)

    def num_patches(self, image_height: int, image_width: int) -> int:
        h, w = self.patch_grid_shape(image_height, image_width)
        return h * w

    # -- Actual model loading / forward: needs torch + transformers --------

    def load(self) -> Any:
        """Build the DINOv3 architecture with random weights — no HF Hub call.

        The detection checkpoint (`epoch_050.pt`) overwrites every parameter
        immediately after this returns, so downloading the pretrained
        weights would just be discarded work. Idempotent: second calls
        return the same instance.
        """
        if self._model is not None:
            return self._model

        try:
            import torch  # noqa: F401
            from transformers import CONFIG_MAPPING, AutoModel
        except ImportError as e:
            raise ImportError(
                "DinoV3Backbone.load() requires torch and transformers. "
                "Install via `pip install -e .[dev]`."
            ) from e

        # `model_type` is registered in transformers ≥4.56 (DINOv3 landed
        # there). We instantiate the config class directly to avoid any
        # `from_pretrained` codepath that would hit the Hub.
        config_class = CONFIG_MAPPING[_VITB16_DINOV3_CONFIG["model_type"]]
        cfg_kwargs = {k: v for k, v in _VITB16_DINOV3_CONFIG.items() if k != "model_type"}
        if self.config.dtype != "float32":
            import torch

            cfg_kwargs["torch_dtype"] = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }[self.config.dtype]
        cfg = config_class(**cfg_kwargs)

        model = AutoModel.from_config(cfg)
        if self.config.freeze:
            for param in model.parameters():
                param.requires_grad_(False)
            model.eval()
        self._model = model
        return model

    def forward(self, pixel_values: Any) -> Any:
        """Run backbone and reshape patch tokens into (B, C, Hp, Wp).

        pixel_values: torch.Tensor of shape (B, 3, H, W), dtype matching
        `config.dtype`. Caller is responsible for the ImageNet-style
        normalization DINOv3 expects.
        """
        model = self.load()

        if pixel_values.ndim != 4 or pixel_values.shape[1] != 3:
            raise ValueError(
                f"pixel_values must be (B, 3, H, W); got shape {tuple(pixel_values.shape)}"
            )
        B, _, H, W = pixel_values.shape
        Hp, Wp = self.patch_grid_shape(int(H), int(W))
        num_patch = Hp * Wp

        need_hidden = self.config.output_layer != -1
        output = model(
            pixel_values=pixel_values,
            output_hidden_states=need_hidden,
        )
        if need_hidden:
            # hidden_states[0] is embeddings; blocks are indexed from 1.
            hidden = output.hidden_states[self.config.output_layer]
        else:
            hidden = output.last_hidden_state

        # DINOv3 layout: [CLS, register_1..K, patch_1..Hp*Wp]. Slicing the
        # tail is robust to register-count changes across checkpoints.
        if hidden.shape[1] < num_patch:
            raise RuntimeError(
                f"backbone returned {hidden.shape[1]} tokens but expected at "
                f"least {num_patch} patches for input {H}x{W} (patch={self.config.patch_size})."
            )
        patch_tokens = hidden[:, -num_patch:, :]  # (B, Hp*Wp, C)
        features = (
            patch_tokens.transpose(1, 2).reshape(B, -1, Hp, Wp).contiguous()
        )
        return features

    __call__ = forward
