# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Nunchaku SVDQuant W4A4 Quantized Inference Example.

This script demonstrates how to run text-to-image generation using
Nunchaku SVDQuant W4A4 quantization for faster inference.

Requirements:
    - Nunchaku library installed: pip install nunchaku
    - Quantized checkpoint with SVDQuant weights

Usage:
    python text_to_image_quant.py \\
        --model /path/to/nunchaku-checkpoint \\
        --prompt "a cup of coffee on the table" \\
        --output zimage_quant_output.png \\
        --rank 128 \\
        --precision nvfp4
"""

import argparse
import time
from pathlib import Path

import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig, logger
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an image with Nunchaku SVDQuant quantization."
    )
    parser.add_argument(
        "--model",
        default="ultranationalism/nunchaku-z-image-turbo",
        help="Diffusion model name or local path. Must be a quantized checkpoint with Nunchaku SVDQuant weights.",
    )
    parser.add_argument(
        "--prompt",
        default="a cup of coffee on the table",
        help="Text prompt for image generation.",
    )
    parser.add_argument(
        "--negative-prompt",
        default=None,
        help="Negative prompt for classifier-free conditional guidance.",
    )
    parser.add_argument("--seed", type=int, default=142, help="Random seed for deterministic results.")
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=4.0,
        help="True classifier-free guidance scale.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=1.0,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument("--height", type=int, default=1024, help="Height of generated image.")
    parser.add_argument("--width", type=int, default=1024, help="Width of generated image.")
    parser.add_argument(
        "--output",
        type=str,
        default="zimage_quant_output.png",
        help="Path to save the generated image (PNG).",
    )
    parser.add_argument(
        "--num-images-per-prompt",
        type=int,
        default=1,
        help="Number of images to generate for the given prompt.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Number of denoising steps for the diffusion sampler.",
    )

    # Nunchaku quantization arguments
    parser.add_argument(
        "--rank",
        type=int,
        default=32,
        help="Low-rank dimension for SVDQuant. Common values: 32, 64, 128.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="int4",
        choices=["int4", "nvfp4"],
        help=(
            "Quantization precision. "
            "'int4': Standard 4-bit integer quantization (group_size=64). "
            "'nvfp4': NVIDIA FP4 format (group_size=16, requires Ampere+ GPU)."
        ),
    )
    parser.add_argument(
        "--act-unsigned",
        action="store_true",
        help="Use unsigned quantization for activations (may improve quality in some cases).",
    )

    # Parallelism arguments
    parser.add_argument(
        "--ulysses-degree",
        type=int,
        default=1,
        help="Number of GPUs used for ulysses sequence parallelism.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs used for tensor parallelism (TP) inside the DiT.",
    )

    # Offloading arguments
    parser.add_argument(
        "--enable-cpu-offload",
        action="store_true",
        help="Enable CPU offloading for diffusion models.",
    )
    parser.add_argument(
        "--enable-layerwise-offload",
        action="store_true",
        help="Enable layerwise (blockwise) offloading on DiT modules.",
    )

    # Other arguments
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable torch.compile and force eager execution.",
    )
    parser.add_argument(
        "--vae-use-slicing",
        action="store_true",
        help="Enable VAE slicing for memory optimization.",
    )
    parser.add_argument(
        "--vae-use-tiling",
        action="store_true",
        help="Enable VAE tiling for memory optimization.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(args.seed)

    parallel_config = DiffusionParallelConfig(
        ulysses_degree=args.ulysses_degree,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    omni = Omni(
        model=args.model,
        vae_use_slicing=args.vae_use_slicing,
        vae_use_tiling=args.vae_use_tiling,
        quantization="nunchaku",
        quantization_config={
            "rank": args.rank,
            "precision": args.precision,
            "act_unsigned": args.act_unsigned,
        },
        parallel_config=parallel_config,
        enforce_eager=args.enforce_eager,
        enable_cpu_offload=args.enable_cpu_offload,
        enable_layerwise_offload=args.enable_layerwise_offload,
    )

    # Print configuration
    print(f"\n{'=' * 60}")
    print("Generation Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Inference steps: {args.num_inference_steps}")
    print(f"  Quantization: nunchaku (rank={args.rank}, precision={args.precision})")
    print(
        f"  Parallel: ulysses_degree={args.ulysses_degree}, "
        f"tensor_parallel_size={args.tensor_parallel_size}"
    )
    print(f"  CPU offload: {args.enable_cpu_offload}")
    print(f"  Layerwise offload: {args.enable_layerwise_offload}")
    print(f"  Image size: {args.width}x{args.height}")
    print(f"{'=' * 60}\n")

    generation_start = time.perf_counter()
    outputs = omni.generate(
        {
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
        },
        OmniDiffusionSamplingParams(
            height=args.height,
            width=args.width,
            generator=generator,
            true_cfg_scale=args.cfg_scale,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            num_outputs_per_prompt=args.num_images_per_prompt,
        ),
    )
    generation_time = time.perf_counter() - generation_start

    print(f"Total generation time: {generation_time:.4f} seconds ({generation_time * 1000:.2f} ms)")

    # Extract images
    if not outputs or len(outputs) == 0:
        raise ValueError("No output generated from omni.generate()")

    first_output = outputs[0]
    if not hasattr(first_output, "request_output") or not first_output.request_output:
        raise ValueError("No request_output found in OmniRequestOutput")

    req_out = first_output.request_output[0]
    if not isinstance(req_out, OmniRequestOutput) or not hasattr(req_out, "images"):
        raise ValueError("Invalid request_output structure or missing 'images' key")

    images = req_out.images
    if not images:
        raise ValueError("No images found in request_output")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix or ".png"
    stem = output_path.stem or "zimage_quant_output"
    if len(images) <= 1:
        images[0].save(output_path)
        print(f"Saved generated image to {output_path}")
    else:
        for idx, img in enumerate(images):
            save_path = output_path.parent / f"{stem}_{idx}{suffix}"
            img.save(save_path)
            print(f"Saved generated image to {save_path}")

    omni.close()


if __name__ == "__main__":
    main()
