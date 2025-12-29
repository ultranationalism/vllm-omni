"""Nunchaku SVDQuant quantization plugin for vLLM.

This module implements a vLLM-compatible quantization plugin for Nunchaku's
SVDQuant algorithm, supporting Tensor Parallel (TP) distribution.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

try:
    from nunchaku.ops.gemm import svdq_gemm_w4a4_cuda
    from nunchaku.ops.gemv import awq_gemv_w4a16_cuda
    from nunchaku.ops.quantize import svdq_quantize_w4a4_act_fuse_lora_cuda
    _NUNCHAKU_OPS_AVAILABLE = True
except ImportError:
    svdq_gemm_w4a4_cuda = None
    awq_gemv_w4a16_cuda = None
    svdq_quantize_w4a4_act_fuse_lora_cuda = None
    _NUNCHAKU_OPS_AVAILABLE = False


class NunchakuConfig(QuantizationConfig):
    """Configuration for Nunchaku SVDQuant quantization.
    
    Args:
        rank: Low-rank approximation dimension for SVDQ
        precision: Quantization precision ("int4" or "nvfp4")
        group_size: Group size for quantization (auto-determined by precision)
        act_unsigned: Whether to use unsigned quantization for activations
    """

    def __init__(
        self,
        rank: int = 32,
        precision: str = "int4",
        act_unsigned: bool = False,
    ) -> None:
        if not _NUNCHAKU_OPS_AVAILABLE:
            raise ImportError(
                "Nunchaku is not installed. Please install it to use "
                "Nunchaku quantization."
            )
        
        self.rank = rank
        self.precision = precision
        self.act_unsigned = act_unsigned
        
        # Determine group size based on precision
        if precision == "nvfp4":
            self.group_size = 16
        elif precision == "int4":
            self.group_size = 64
        else:
            raise ValueError(f"Invalid precision: {precision}")

    def __repr__(self) -> str:
        return (f"NunchakuConfig(rank={self.rank}, "
                f"precision={self.precision}, "
                f"act_unsigned={self.act_unsigned})")

    @classmethod
    def get_name(cls) -> str:
        """Return the name of the quantization method."""
        return "nunchaku"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        """Return supported activation dtypes."""
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        """Return the minimum CUDA compute capability required."""
        return 80  # Requires Ampere or newer

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        """Return config filenames to search for quantization config."""
        return ["quantization_config.json", "config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "NunchakuConfig":
        """Create config from a dictionary."""
        rank = config.get("rank", 32)
        precision = config.get("precision", "int4")
        act_unsigned = config.get("act_unsigned", False)
        return cls(
            rank=rank,
            precision=precision,
            act_unsigned=act_unsigned,
        )

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional["NunchakuLinearMethod"]:
        """Get the quantization method for a layer."""
        if isinstance(layer, LinearBase):
            return NunchakuLinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        """Return names of activation scales (for smooth quantization)."""
        return ["smooth_factor"]


class NunchakuLinearMethod(LinearMethodBase):
    """Linear method for Nunchaku SVDQuant quantization.
    
    This implements the vLLM LinearMethodBase interface for SVDQ quantization,
    handling weight creation with proper TP distribution and forward computation.
    """

    def __init__(self, quant_config: NunchakuConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Create quantized weights for the layer.
        
        This method determines the TP direction (ColumnParallel or RowParallel)
        and creates appropriately sharded parameters.
        
        Args:
            layer: The layer module to add parameters to
            input_size_per_partition: Input dimension after TP partitioning
            output_partition_sizes: List of output dimensions after TP partitioning
            input_size: Original input dimension (before TP)
            output_size: Original output dimension (before TP)
            params_dtype: Data type for parameters
        """
        output_size_per_partition = sum(output_partition_sizes)
        
        # Determine TP parallelization direction
        is_row_parallel = (input_size_per_partition != input_size)
        is_col_parallel = (output_size_per_partition != output_size)
        
        config = self.quant_config
        rank = config.rank
        precision = config.precision
        group_size = config.group_size
        
        # Helper function to set weight attributes
        def set_weight_attrs(param: nn.Parameter, attrs: Dict[str, Any]):
            """Set weight attributes for TP sharding."""
            for key, value in attrs.items():
                setattr(param, key, value)
        
        # Quantized weight tensor (packed int4 -> int8)
        # Shape: (out_features, in_features // 2)
        qweight = nn.Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(qweight, {
            "input_dim": 1,
            "output_dim": 0,
            "weight_loader": default_weight_loader,
        })
        
        # Weight scales
        # Shape: (in_features // group_size, out_features)
        wscales_dtype = (torch.float8_e4m3fn 
                        if precision == "nvfp4" 
                        else params_dtype)
        wscales = nn.Parameter(
            torch.empty(
                input_size_per_partition // group_size,
                output_size_per_partition,
                dtype=wscales_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(wscales, {
            "input_dim": 0,
            "output_dim": 1,
            "weight_loader": default_weight_loader,
        })
        
        # Low-rank projection matrices
        # proj_down: (in_features, rank)
        proj_down = nn.Parameter(
            torch.empty(
                input_size_per_partition,
                rank,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(proj_down, {
            "input_dim": 0,
            "output_dim": 1,
            "weight_loader": default_weight_loader,
        })
        
        # proj_up: (out_features, rank)
        proj_up = nn.Parameter(
            torch.empty(
                output_size_per_partition,
                rank,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(proj_up, {
            "input_dim": 0,
            "output_dim": 1,
            "weight_loader": default_weight_loader,
        })
        
        # Smooth factors for activation quantization
        # Shape: (in_features,)
        smooth_factor = nn.Parameter(
            torch.empty(
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(smooth_factor, {
            "input_dim": 0,
            "weight_loader": default_weight_loader,
        })
        
        smooth_factor_orig = nn.Parameter(
            torch.empty(
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(smooth_factor_orig, {
            "input_dim": 0,
            "weight_loader": default_weight_loader,
        })
        
        # Override TP sharding attributes based on parallelization direction
        if is_col_parallel:
            # Column parallel: split output dimension (N)
            # Mark which dimension should be sharded for TP
            qweight.output_dim = 0
            wscales.output_dim = 1
            proj_up.output_dim = 0
        
        if is_row_parallel:
            # Row parallel: split input dimension (K)
            # Mark which dimension should be sharded for TP
            qweight.input_dim = 1
            wscales.input_dim = 0
            proj_down.input_dim = 0
            smooth_factor.input_dim = 0
            smooth_factor_orig.input_dim = 0
        
        # Register all parameters
        layer.register_parameter("qweight", qweight)
        layer.register_parameter("wscales", wscales)
        layer.register_parameter("proj_down", proj_down)
        layer.register_parameter("proj_up", proj_up)
        layer.register_parameter("smooth_factor", smooth_factor)
        layer.register_parameter("smooth_factor_orig", smooth_factor_orig)
        
        # Optional parameters for nvfp4 precision
        if precision == "nvfp4":
            wcscales = nn.Parameter(
                torch.ones(
                    output_size_per_partition,
                    dtype=params_dtype,
                ),
                requires_grad=False,
            )
            set_weight_attrs(wcscales, {
                "output_dim": 0,
                "weight_loader": default_weight_loader,
            })
            if is_col_parallel:
                wcscales.output_dim = 0
            layer.register_parameter("wcscales", wcscales)
            # wtscale is a scalar, stored as attribute
            layer.wtscale = 1.0
        else:
            layer.wcscales = None
            layer.wtscale = None
        
        # Store config attributes
        layer.in_features = input_size
        layer.out_features = output_size
        layer.rank = rank
        layer.precision = precision
        layer.group_size = group_size
        layer.act_unsigned = config.act_unsigned

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply SVDQ quantized linear transformation.
        
        Args:
            layer: The layer with quantized weights
            x: Input tensor, shape (N, in_features) or (B, S, in_features)
            bias: Optional bias tensor
            
        Returns:
            Output tensor, shape matches input except last dimension
        """
        # Handle 3D input (batch, seq, hidden)
        orig_shape = x.shape
        if x.ndim == 3:
            batch_size, seq_len, channels = x.shape
            x = x.reshape(batch_size * seq_len, channels)
        
        # Quantize activations and compute low-rank hidden states
        quantized_x, ascales, lora_act_out = svdq_quantize_w4a4_act_fuse_lora_cuda(
            x,
            lora_down=layer.proj_down,
            smooth=layer.smooth_factor,
            fp4=layer.precision == "nvfp4",
            pad_size=256,
        )
        
        # The quantize kernel may pad the batch dimension
        padded_batch = quantized_x.shape[0]
        real_batch = x.shape[0]
        out_features = layer.out_features
        
        # Allocate output buffer
        output = torch.empty(
            padded_batch,
            out_features,
            dtype=layer.proj_up.dtype,
            device=x.device,
        )
        
        # Perform quantized GEMM with low-rank correction
        svdq_gemm_w4a4_cuda(
            act=quantized_x,
            wgt=layer.qweight,
            out=output,
            ascales=ascales,
            wscales=layer.wscales,
            lora_act_in=lora_act_out,
            lora_up=layer.proj_up,
            bias=bias,
            fp4=layer.precision == "nvfp4",
            alpha=layer.wtscale,
            wcscales=layer.wcscales,
            act_unsigned=layer.act_unsigned,
        )
        
        # Trim padded rows
        output = output[:real_batch]
        
        # Restore original shape if input was 3D
        if len(orig_shape) == 3:
            output = output.reshape(orig_shape[0], orig_shape[1], -1)
        
        return output
