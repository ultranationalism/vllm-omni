# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Base class for diffusion model quantization configurations."""

from abc import ABC
from typing import TYPE_CHECKING, ClassVar

import torch

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig,
    )


class DiffusionQuantizationConfig(ABC):
    """Base class for diffusion model quantization configurations.

    This provides a thin wrapper over vLLM's quantization configs,
    allowing diffusion-model-specific defaults and future extensibility.

    Subclasses should:
        - Set quant_config_cls to the vLLM QuantizationConfig class
        - Call super().__init__() after creating self._vllm_config
        - Optionally override get_name() and get_min_capability() if needed
    """

    # Subclasses should set this to the vLLM QuantizationConfig class
    quant_config_cls: ClassVar[type["QuantizationConfig"] | None] = None

    # The underlying vLLM config instance
    _vllm_config: "QuantizationConfig | None" = None

    def get_name(self) -> str:
        """Return the quantization method name (e.g., 'fp8', 'int8').

        By default, delegates to the underlying vLLM config instance.
        """
        if self._vllm_config is not None:
            return self._vllm_config.get_name()
        raise NotImplementedError("Subclass must initialize _vllm_config or override get_name().")

    def get_vllm_quant_config(self) -> "QuantizationConfig | None":
        """Return the underlying vLLM QuantizationConfig for linear layers."""
        return self._vllm_config

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        """Return supported activation dtypes."""
        return [torch.bfloat16, torch.float16]

    def transform_weight(
        self, name: str, loaded_weight: "torch.Tensor", model_class_name: str = ""
    ) -> list[tuple[str, "torch.Tensor"]]:
        """Transform a checkpoint weight entry before it reaches load_weights.

        Override this in subclasses to remap checkpoint key names and/or
        manipulate tensors (e.g. split, swap shards) so the model's
        load_weights sees the canonical naming convention.

        Args:
            name: Original weight name from checkpoint.
            loaded_weight: The weight tensor.
            model_class_name: The model class name (e.g. "ZImageTransformer2DModel"),
                used by quantization methods that need per-model key mapping.

        Returns:
            A list of (new_name, tensor) pairs.  Most implementations return
            a single pair; returning multiple pairs is useful when a single
            checkpoint tensor must be split into several model parameters.
            Returning an empty list drops the weight silently.
        """
        return [(name, loaded_weight)]

    @classmethod
    def get_min_capability(cls) -> int:
        """Minimum GPU compute capability required.

        By default, delegates to the underlying vLLM config class.
        """
        if cls.quant_config_cls is not None:
            return cls.quant_config_cls.get_min_capability()
        return 80  # Ampere default
