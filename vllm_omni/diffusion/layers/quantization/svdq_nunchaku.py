"""Nunchaku SVDQuant quantization plugin for vLLM.

This module implements a vLLM-compatible quantization plugin for Nunchaku's
SVDQuant algorithm, supporting Tensor Parallel (TP) distribution.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.logger import init_logger

logger = init_logger(__name__)

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


# ============================================================================
# Weight Packing and Padding Utilities
# ============================================================================


def pad_to_multiple(tensor: torch.Tensor, dim: int, multiple: int) -> torch.Tensor:
    """Pad a tensor along a specific dimension to be a multiple of a value.

    Args:
        tensor: Input tensor to pad
        dim: Dimension to pad along
        multiple: Target multiple for dimension size

    Returns:
        Padded tensor
    """
    current_size = tensor.shape[dim]
    if current_size % multiple == 0:
        return tensor

    target_size = ((current_size + multiple - 1) // multiple) * multiple
    pad_size = target_size - current_size

    # Create padding specification (right pad on the specified dimension)
    pad_spec = [0, 0] * tensor.ndim
    pad_spec[-(2 * dim + 1)] = pad_size

    return torch.nn.functional.pad(tensor, pad_spec, value=0)


def pack_lowrank_weight(weight: torch.Tensor, down: bool = True) -> torch.Tensor:
    """Pack low-rank projection weights for Nunchaku SVDQuant CUDA kernels.

    Nunchaku's CUDA kernels require low-rank matrices to follow a specific
    memory layout for optimal performance. This function packs FP16/BF16
    weights according to these requirements.

    Args:
        weight: Low-rank weight tensor
            - For down-projection: shape (in_features, rank)
            - For up-projection: shape (out_features, rank)
        down: True for down-projection, False for up-projection

    Returns:
        Packed weight tensor with optimized memory layout
    """
    if down:
        # Down-projection: (in_features, rank)
        # Nunchaku kernel expects shape (in_features, rank) with assertion: lora_down.shape[0] == N
        M, K = weight.shape

        # Ensure rank (K) is aligned to 16 for efficient memory access
        if K % 16 != 0:
            weight = pad_to_multiple(weight, dim=1, multiple=16)
            K = weight.shape[1]

        # Keep original shape (in_features, rank) - no transpose needed
        packed = weight.contiguous()

    else:
        # Up-projection: (out_features, rank)
        # Pack in row-major order but ensure proper alignment
        N, K = weight.shape

        # Ensure rank (K) is aligned to 16
        if K % 16 != 0:
            weight = pad_to_multiple(weight, dim=1, multiple=16)
            K = weight.shape[1]

        # Keep row-major but ensure contiguous memory
        packed = weight.contiguous()

    return packed


@register_quantization_config("nunchaku")
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

    # Layers that Nunchaku quantizes: attention (to_qkv, to_out) and
    # feed_forward (net.0.proj / w13, net.2 / w2) in transformer blocks
    # (layers, noise_refiner, context_refiner).
    # Everything else (t_embedder, cap_embedder, all_x_embedder,
    # all_final_layer, adaLN_modulation) stays unquantized.
    _QUANTIZED_LAYER_PATTERNS = (
        ".attention.to_qkv",
        ".attention.to_out.",
        ".feed_forward.w13",
        ".feed_forward.w2",
        ".feed_forward.net.0.proj",
        ".feed_forward.net.2",
    )

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional["NunchakuLinearMethod"]:
        """Get the quantization method for a layer.

        Only quantizes attention projections and FFN layers in transformer
        blocks, matching Nunchaku's PTQ scope.
        """
        if isinstance(layer, LinearBase):
            # Nunchaku quantizes attention (QKVParallelLinear, RowParallelLinear)
            # and FFN (MergedColumnParallelLinear, RowParallelLinear) layers.
            # ReplicatedLinear layers (adaLN, embedders, final_layer) stay
            # unquantized — they use standard weight/bias from checkpoint.
            if isinstance(layer, (QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear)):
                return NunchakuLinearMethod(self)
            return UnquantizedLinearMethod()
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

    @staticmethod
    def _swap_swiglu_halves(layer: nn.Module) -> None:
        """Swap gate/up halves of quantized MergedColumnParallelLinear weights.

        Diffusers FeedForward (GEGLU) convention:
            output = gate * silu(up)
            weight layout: [gate_rows ; up_rows]  (gate = first half)

        vLLM SiluAndMul convention:
            output = silu(first_half) * second_half
            weight layout: [silu_rows ; linear_rows]

        To match, we swap: put up (silu'd) first, gate (linear) second.
        """
        half = layer.qweight.shape[0] // 2
        # qweight: (out, in//2) — swap output halves
        layer.qweight.data = torch.cat(
            [layer.qweight.data[half:], layer.qweight.data[:half]], dim=0
        )

        # wscales: (in//gs, out) — swap output halves (dim 1)
        half_s = layer.wscales.shape[1] // 2
        layer.wscales.data = torch.cat(
            [layer.wscales.data[:, half_s:], layer.wscales.data[:, :half_s]],
            dim=1,
        )

        # proj_up: (out, rank) — swap output halves
        half_p = layer.proj_up.shape[0] // 2
        layer.proj_up.data = torch.cat(
            [layer.proj_up.data[half_p:], layer.proj_up.data[:half_p]], dim=0
        )

        # wcscales: (out,) — swap halves [nvfp4 only]
        if hasattr(layer, "wcscales") and layer.wcscales is not None:
            half_c = layer.wcscales.shape[0] // 2
            layer.wcscales.data = torch.cat(
                [layer.wcscales.data[half_c:], layer.wcscales.data[:half_c]]
            )

        logger.debug("Swapped SwiGLU gate/up halves for %s", type(layer).__name__)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """Post-process quantized weights after checkpoint loading.

        Two key steps:
        1. SwiGLU shard swap for MergedColumnParallelLinear (w13):
           Diffusers computes gate * silu(up) (gate=first half, up=second),
           but vLLM's SiluAndMul computes silu(first) * second.
           We swap the two output halves so the activation order matches.
        2. Extract alpha from wtscale for the CUDA kernel.
        """
        # Materialize any meta-device parameters that weren't loaded from
        # the checkpoint (e.g. wtscale, wcscales for nvfp4).  Find the
        # device of an already-loaded parameter to use as the target.
        target_device = None
        for p in layer.parameters():
            if not p.is_meta:
                target_device = p.device
                break
        if target_device is not None:
            for name, p in list(layer.named_parameters()):
                if p.is_meta:
                    new_p = Parameter(
                        torch.ones_like(p, device=target_device),
                        requires_grad=False,
                    )
                    setattr(layer, name, new_p)

        alpha: Optional[float] = None

        # Extract wtscale from checkpoint if available
        if hasattr(layer, "wtscale") and layer.wtscale is not None:
            wtscale = layer.wtscale
            if isinstance(wtscale, Parameter):
                wtscale_val = float(wtscale.data.detach().cpu().item())
            elif isinstance(wtscale, torch.Tensor):
                wtscale_val = float(wtscale.detach().cpu().item())
            else:
                wtscale_val = float(wtscale)

            # Check if actually loaded from checkpoint (not default 1.0)
            if abs(wtscale_val - 1.0) > 1e-6:
                alpha = wtscale_val
                wtscale_from_checkpoint = True
                logger.debug(f"Using wtscale from checkpoint: {alpha}")
            else:
                logger.debug(f"wtscale is default value 1.0, not from checkpoint")

        # 2. If wtscale not loaded, use alpha=1.0 and let wcscales handle scaling
        # IMPORTANT: Do NOT calculate alpha from wcscales.mean() because:
        # - Nunchaku kernel uses: Output = (Accumulator × alpha) × wcscales
        # - If we set alpha=wcscales.mean(), we get: Output = Acc × mean(wcscales) × wcscales (WRONG!)
        # - But Nunchaku expects: Output = Acc × 1.0 × wcscales = Acc × wcscales (CORRECT)
        if alpha is None:
            alpha = 1.0

        layer._nunchaku_alpha = alpha


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

        # For W4A4 quantization, LoRA projections and smooth factors must use bfloat16
        # for correct activation quantization (matching Nunchaku's implementation)
        lora_dtype = torch.bfloat16 if precision == "nvfp4" else params_dtype

        # Helper function to set weight attributes
        def set_weight_attrs(param: nn.Parameter, attrs: Dict[str, Any]):
            """Set weight attributes for TP sharding."""
            for key, value in attrs.items():
                setattr(param, key, value)

        # Create custom weight loaders for proj_down and proj_up
        # These will handle name mapping (lora_down/lora_up -> proj_down/proj_up)
        # and weight packing for CUDA kernel requirements
        def proj_down_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
            """Custom loader for proj_down that packs weights."""
            # Pack for down-projection (transpose + pad)
            packed_weight = pack_lowrank_weight(loaded_weight, down=True)
            logger.debug(
                "Packed proj_down weight: %s -> %s",
                loaded_weight.shape,
                packed_weight.shape,
            )
            # Use default loader for the packed weight
            default_weight_loader(param, packed_weight)

        def proj_up_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
            """Custom loader for proj_up that packs weights."""
            # Pack for up-projection (pad only)
            packed_weight = pack_lowrank_weight(loaded_weight, down=False)
            logger.debug(
                "Packed proj_up weight: %s -> %s",
                loaded_weight.shape,
                packed_weight.shape,
            )
            # Use default loader for the packed weight
            default_weight_loader(param, packed_weight)

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
        # proj_down: (in_features, rank) - will be padded by weight loader if needed
        proj_down = nn.Parameter(
            torch.empty(
                input_size_per_partition,
                rank,
                dtype=lora_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(proj_down, {
            "input_dim": 0,  # Input dimension is first dimension
            "output_dim": 1,
            "weight_loader": proj_down_loader,  # Use custom loader
        })

        # proj_up: (out_features, rank) - will be padded by weight loader if needed
        proj_up = nn.Parameter(
            torch.empty(
                output_size_per_partition,
                rank,
                dtype=lora_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(proj_up, {
            "input_dim": 0,
            "output_dim": 1,
            "weight_loader": proj_up_loader,  # Use custom loader
        })

        # Smooth factors for activation quantization
        # Shape: (in_features,)
        smooth_factor = nn.Parameter(
            torch.empty(
                input_size_per_partition,
                dtype=lora_dtype,
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
                dtype=lora_dtype,
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
            proj_down.input_dim = 0  # Input dimension is first dimension (in_features)
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
            # wcscales: per-channel scales, must match output dtype (bfloat16)
            # NOT float8_e4m3fn like wscales!
            wcscales = nn.Parameter(
                torch.ones(
                    output_size_per_partition,
                    dtype=lora_dtype,  # Use lora_dtype (bfloat16) for consistency with output
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

            # wtscale: Global scale parameter, must match output dtype (bfloat16)
            # Create as Parameter so it can be loaded from checkpoint
            wtscale = nn.Parameter(
                torch.ones(1, dtype=lora_dtype),  # Shape (1,) to match checkpoint
                requires_grad=False,
            )
            set_weight_attrs(wtscale, {
                "weight_loader": default_weight_loader,
            })
            layer.register_parameter("wtscale", wtscale)
        else:
            layer.wcscales = None
            layer.wtscale = None

        # Store config attributes
        layer.in_features = input_size
        layer.out_features = output_size
        layer.out_features_per_partition = output_size_per_partition  # For TP
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
        # Reshape to 2D for kernel processing
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1])

        # Quantize activations and compute low-rank hidden states
        quantized_x, ascales, lora_act_out = svdq_quantize_w4a4_act_fuse_lora_cuda(
            x_2d,
            lora_down=layer.proj_down,
            smooth=layer.smooth_factor,
            fp4=layer.precision == "nvfp4",
            pad_size=256,
        )

        # Use per-partition output features (critical for TP correctness)
        out_features = layer.out_features_per_partition

        # IMPORTANT: Allocate output buffer using quantized_x.shape[0] (padded batch size)
        # The quantization kernel may pad the batch dimension, so we must match that
        out_2d = torch.empty(
            quantized_x.shape[0],  # Use padded batch size from quantized_x
            out_features,
            dtype=layer.proj_up.dtype,
            device=x_2d.device,
        )

        # Use cached alpha value (extracted in process_weights_after_loading)
        alpha = getattr(layer, "_nunchaku_alpha", None)
        wcscales = layer.wcscales if hasattr(layer, "wcscales") else None

        # Perform quantized GEMM with low-rank correction
        svdq_gemm_w4a4_cuda(
            act=quantized_x,
            wgt=layer.qweight,
            out=out_2d,
            ascales=ascales,
            wscales=layer.wscales,
            lora_act_in=lora_act_out,
            lora_up=layer.proj_up,
            bias=bias,
            fp4=layer.precision == "nvfp4",
            alpha=alpha,
            wcscales=wcscales,
            act_unsigned=layer.act_unsigned,
        )

        # Trim padding if batch was padded by quantization kernel
        # out_2d might have shape (padded_batch, out_features)
        # We need to trim it back to (actual_batch, out_features)
        actual_batch = x_2d.shape[0]
        if out_2d.shape[0] > actual_batch:
            out_2d = out_2d[:actual_batch]

        # Reshape back to original shape (except last dimension)
        output = out_2d.reshape(*orig_shape[:-1], out_features)

        # Gated-activation output swap for MergedColumnParallelLinear.
        #
        # Nunchaku checkpoints are quantized from diffusers models.
        # diffusers' gated activations (GEGLU, SwiGLU) use the convention:
        #   hidden, gate = proj(x).chunk(2)  =>  [linear ; activation]
        # vLLM's gated activations (SiluAndMul, GeluAndMul) use:
        #   act(x[:d]) * x[d:]              =>  [activation ; linear]
        #
        # Since the qweight is stored in a tiled/interleaved MMA layout
        # (see nunchaku/lora/flux/packer.py:pack_weight), we cannot swap
        # the weight rows directly. Instead we swap the output halves here.
        #
        # Assumption: MergedColumnParallelLinear is only used for gated FFN
        # (SwiGLU / GeGLU) whose downstream activation follows the vLLM
        # convention. This holds for all current diffusion models (Z-Image,
        # Flux2, HunyuanImage3, etc.). If a future model uses
        # MergedColumnParallelLinear with a diffusers-convention activation,
        # this swap must be skipped for that model.
        if isinstance(layer, MergedColumnParallelLinear):
            d = output.shape[-1] // 2
            output = torch.cat([output[..., d:], output[..., :d]], dim=-1)

        return output
