# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Nunchaku SVDQuant quantization config for diffusion transformers.

Nunchaku checkpoints use diffusers-style FeedForward naming (net.0.proj / net.2)
while vLLM-Omni models may use different conventions (e.g. w13 / w2 for Z-Image).
Additionally, the SwiGLU activation order differs between diffusers and vLLM:
diffusers computes ``x[:d] * silu(x[d:])`` while vLLM's SiluAndMul computes
``silu(x[:d]) * x[d:]``, requiring a shard swap during weight loading.

This config maintains per-model key mapping tables so that model code stays clean.
"""

import torch

from .base import DiffusionQuantizationConfig

# Per-model weight key mapping tables.
# Each entry: "source_key_fragment": ("target_key_fragment", swap_swiglu)
#   - swap_swiglu=True: swap the two halves of the merged gate+up weight
#     to account for SwiGLU activation order difference.
_MODEL_KEY_MAPPING: dict[str, dict[str, tuple[str, bool]]] = {
    "ZImageTransformer2DModel": {
        ".net.0.proj.": (".w13.", True),
        ".net.2.": (".w2.", False),
    },
    # Pipeline wrapper delegates to the transformer above, but the loader
    # sees the pipeline class name. Use the same mapping.
    "ZImagePipeline": {
        ".net.0.proj.": (".w13.", True),
        ".net.2.": (".w2.", False),
    },
    # Models whose internal naming already matches Nunchaku/diffusers style
    # (e.g. Flux uses net[0]/net[2]) need no mapping — just omit them here.
    #
    # To add a new model, add an entry mapping Nunchaku's diffusers-style keys
    # to the model's internal parameter names.
}


class DiffusionNunchakuConfig(DiffusionQuantizationConfig):
    """Nunchaku SVDQuant W4A4 quantization config.

    Uses Nunchaku's custom CUDA kernels for W4A4 GEMM with low-rank
    correction.  The underlying ``NunchakuConfig`` (a vLLM
    ``QuantizationConfig``) is passed to linear layers so they create
    the required quantized parameters (qweight, wscales, proj_down,
    proj_up, smooth_factor, etc.).

    Args:
        rank: Low-rank approximation dimension for SVDQuant.
        precision: Quantization precision ("int4" or "nvfp4").
        act_unsigned: Whether to use unsigned activation quantization.
    """

    def __init__(
        self,
        rank: int = 32,
        precision: str = "int4",
        act_unsigned: bool = False,
    ):
        from vllm_omni.diffusion.layers.quantization.svdq_nunchaku import (
            NunchakuConfig,
        )

        self.rank = rank
        self.precision = precision
        self.act_unsigned = act_unsigned

        # NunchakuConfig implements vLLM's QuantizationConfig interface,
        # providing NunchakuLinearMethod that creates quantized parameters
        # (qweight, wscales, proj_down, proj_up, smooth_factor, etc.)
        # and runs the SVDQuant CUDA kernels in forward().
        self._vllm_config = NunchakuConfig(
            rank=rank,
            precision=precision,
            act_unsigned=act_unsigned,
        )

    def get_name(self) -> str:
        return "nunchaku"

    @classmethod
    def get_min_capability(cls) -> int:
        return 80  # Ampere or newer

    def transform_weight(
        self, name: str, loaded_weight: torch.Tensor, model_class_name: str = ""
    ) -> list[tuple[str, torch.Tensor]]:
        """Remap Nunchaku (diffusers-style) checkpoint keys to model conventions.

        Uses the per-model mapping table in ``_MODEL_KEY_MAPPING``.  Models not
        listed in the table are assumed to already use diffusers-compatible
        naming and are passed through unchanged.
        """
        mapping = _MODEL_KEY_MAPPING.get(model_class_name, {})
        for src, (dst, swap) in mapping.items():
            if src in name:
                new_name = name.replace(src, dst)
                if swap:
                    # Swap halves to fix SwiGLU activation order.
                    # Diffusers: [gate, up], vLLM SiluAndMul expects: [up, gate]
                    mid = loaded_weight.shape[0] // 2
                    loaded_weight = torch.cat([loaded_weight[mid:], loaded_weight[:mid]], dim=0)
                return [(new_name, loaded_weight)]
        return [(name, loaded_weight)]
