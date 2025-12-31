# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Z-Image Quantized (Nunchaku SVDQuant W4A4) Inference Example.

This script demonstrates how to run text-to-image generation using
Z-Image with Nunchaku SVDQuant W4A4 quantization for faster inference.

Requirements:
    - Nunchaku library installed: pip install nunchaku
    - Quantized Z-Image checkpoint with SVDQuant weights

Usage:
    python text_to_image_quant.py \\
        --model /path/to/zimage-svdquant-checkpoint \\
        --prompt "a cup of coffee on the table" \\
        --output zimage_quant_output.png \\
        --quantization nunchaku \\
        --rank 32 \\
        --precision nvfp4
"""

import argparse
import time
from pathlib import Path

import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig, logger
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.utils.platform_utils import detect_device_type, is_npu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an image with Z-Image using Nunchaku SVDQuant quantization."
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
        "--seed",
        type=int,
        default=142,
        help="Random seed for deterministic results.",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=4.0,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Height of generated image.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Width of generated image.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="zimage_quant_output.png",
        help="Path to save the generated image (PNG).",
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="Number of images to generate for the given prompt.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps for the diffusion sampler.",
    )
    
    # Quantization-specific arguments
    parser.add_argument(
        "--quantization",
        type=str,
        default="nunchaku",
        choices=["nunchaku"],
        help="Quantization method to use. Currently only 'nunchaku' (SVDQuant) is supported.",
    )
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
        "--act_unsigned",
        action="store_true",
        help="Use unsigned quantization for activations (may improve quality in some cases).",
    )
    parser.add_argument(
        "--ulysses_degree",
        type=int,
        default=1,
        help="Number of GPUs used for ulysses sequence parallelism.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    device = detect_device_type()
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # Enable VAE memory optimizations on NPU
    vae_use_slicing = is_npu()
    vae_use_tiling = is_npu()

    # Build quantization configuration
    quantization_config = {
        "rank": args.rank,
        "precision": args.precision,
        "act_unsigned": args.act_unsigned,
    }

    parallel_config = DiffusionParallelConfig(ulysses_degree=args.ulysses_degree)
    
    # Initialize Omni with quantization
    omni = Omni(
        model=args.model,
        vae_use_slicing=vae_use_slicing,
        vae_use_tiling=vae_use_tiling,
        quantization=args.quantization,
        quantization_config=quantization_config,
        parallel_config=parallel_config,
    )

    # Print configuration
    print(f"\n{'=' * 60}")
    print("Generation Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Inference steps: {args.num_inference_steps}")
    print(f"  Quantization: {args.quantization}")
    print(f"    - Rank: {args.rank}")
    print(f"    - Precision: {args.precision}")
    print(f"    - Act Unsigned: {args.act_unsigned}")
    print(f"  Parallel configuration: ulysses_degree={args.ulysses_degree}")
    print(f"  Image size: {args.width}x{args.height}")
    print(f"{'=' * 60}\n")

    # Time profiling for generation
    generation_start = time.perf_counter()
    outputs = omni.generate(
        args.prompt,
        height=args.height,
        width=args.width,
        generator=generator,
        true_cfg_scale=args.cfg_scale,
        num_inference_steps=args.num_inference_steps,
        num_outputs_per_prompt=args.num_images_per_prompt,
    )
    generation_end = time.perf_counter()
    generation_time = generation_end - generation_start

    # Print profiling results
    print(f"\n{'=' * 60}")
    print(f"Total generation time: {generation_time:.4f} seconds ({generation_time * 1000:.2f} ms)")
    print(f"{'=' * 60}\n")

    # Extract images from OmniRequestOutput
    if not outputs or len(outputs) == 0:
        raise ValueError("No output generated from omni.generate()")
    logger.info(f"Outputs: {outputs}")

    # Extract images from request_output[0]['images']
    first_output = outputs[0]
    if not hasattr(first_output, "request_output") or not first_output.request_output:
        raise ValueError("No request_output found in OmniRequestOutput")

    req_out = first_output.request_output[0]
    if not isinstance(req_out, dict) or "images" not in req_out:
        raise ValueError("Invalid request_output structure or missing 'images' key")

    images = req_out["images"]
    if not images:
        raise ValueError("No images found in request_output")

    # Save images
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
