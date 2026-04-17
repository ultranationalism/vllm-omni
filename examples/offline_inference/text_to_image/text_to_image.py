# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import os
import time
from pathlib import Path
from typing import Any

import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig, logger
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.lora.request import LoRARequest
from vllm_omni.lora.utils import stable_lora_int_id
from vllm_omni.platforms import current_omni_platform


def is_nextstep_model(model_name: str) -> bool:
    """Check if the model is a NextStep model by reading its config."""
    from vllm.transformers_utils.config import get_hf_file_to_dict

    try:
        cfg = get_hf_file_to_dict("config.json", model_name)
        if cfg and cfg.get("model_type") == "nextstep":
            return True
    except Exception:
        pass
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an image with supported diffusion models.")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen-Image",
        help="Diffusion model name or local path. Supported models: "
        "Qwen/Qwen-Image, Tongyi-MAI/Z-Image-Turbo, Qwen/Qwen-Image-2512, stepfun-ai/NextStep-1.1, "
        "black-forest-labs/FLUX.1-dev, black-forest-labs/FLUX.2-klein-9B, "
        "black-forest-labs/FLUX.2-dev, tencent/HunyuanImage-3.0-Instruct, "
        "meituan-longcat/LongCat-Image, OvisAI/Ovis-Image, "
        "stabilityai/stable-diffusion-3.5-medium, Tongyi-MAI/Z-Image-Turbo and etc.",
    )
    parser.add_argument(
        "--stage-configs-path",
        type=str,
        default=None,
        help="Path to a YAML file containing stage configurations for Omni.",
    )
    parser.add_argument("--prompt", default="a cup of coffee on the table", help="Text prompt for image generation.")
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=None,
        help="Multiple prompts for batched generation. Overrides --prompt when set. "
        "Each prompt is dispatched as part of a single omni.generate() batch call.",
    )
    parser.add_argument(
        "--negative-prompt",
        default=None,
        help="negative prompt for classifier-free conditional guidance.",
    )
    parser.add_argument("--seed", type=int, default=142, help="Random seed for deterministic results.")
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=4.0,
        help="True classifier-free guidance scale specific to Qwen-Image.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=4.0,
        help="Classifier-free guidance scale. HunyuanImage3 recommends 4.0-5.0.",
    )
    parser.add_argument("--height", type=int, default=1024, help="Height of generated image.")
    parser.add_argument("--width", type=int, default=1024, help="Width of generated image.")
    parser.add_argument(
        "--output",
        type=str,
        default="qwen_image_output.png",
        help="Path to save the generated image (PNG). Used only in single-output mode "
        "(one prompt, one LoRA combo, one image). Ignored when --output-dir is set.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for batch/XYZ output. Required when there are multiple prompts, "
        "multiple LoRA combos, or --xyz is set. Files are saved as "
        "p{prompt_idx}_c{combo_idx}_n{img_idx}.png; XYZ mode additionally writes a grid.png.",
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
    parser.add_argument(
        "--cache-backend",
        type=str,
        default=None,
        choices=["cache_dit", "tea_cache"],
        help=(
            "Cache backend to use for acceleration. "
            "Options: 'cache_dit' (DBCache + SCM + TaylorSeer), 'tea_cache' (Timestep Embedding Aware Cache). "
            "Default: None (no cache acceleration)."
        ),
    )
    parser.add_argument(
        "--enable-cache-dit-summary",
        action="store_true",
        help="Enable cache-dit summary logging after diffusion forward passes.",
    )
    parser.add_argument(
        "--ulysses-degree",
        type=int,
        default=1,
        help="Number of GPUs used for ulysses sequence parallelism.",
    )
    parser.add_argument(
        "--ulysses-mode",
        type=str,
        default="strict",
        choices=["strict", "advanced_uaa"],
        help="Ulysses sequence-parallel mode: 'strict' (divisibility required) or 'advanced_uaa' (UAA).",
    )
    parser.add_argument(
        "--ring-degree",
        type=int,
        default=1,
        help="Number of GPUs used for ring sequence parallelism.",
    )
    parser.add_argument(
        "--cfg-parallel-size",
        type=int,
        default=1,
        choices=[1, 2],
        help="Number of GPUs used for classifier free guidance parallel size.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable torch.compile and force eager execution.",
    )
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
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=["fp8", "int8", "gguf"],
        help="Quantization method for the transformer. "
        "Options: 'fp8' (FP8 W8A8 on Ada/Hopper, weight-only on older GPUs), 'int8' (Int8 W8A8), 'gguf' (GGUF quantized weights). "
        "Default: None (no quantization, uses BF16).",
    )
    parser.add_argument(
        "--gguf-model",
        type=str,
        default=None,
        help=("GGUF file path or HF reference for transformer weights. Required when --quantization gguf is set."),
    )
    parser.add_argument(
        "--ignored-layers",
        type=str,
        default=None,
        help="Comma-separated list of layer name patterns to skip quantization. "
        "Only used when --quantization is set. "
        "Available layers: to_qkv, to_out, add_kv_proj, to_add_out, img_mlp, txt_mlp, proj_out. "
        "Example: --ignored-layers 'add_kv_proj,to_add_out'",
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
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs used for tensor parallelism (TP) inside the DiT.",
    )
    parser.add_argument(
        "--enable-expert-parallel",
        action="store_true",
        help="Enable expert parallelism for MoE layers.",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Path to LoRA adapter folder (PEFT format). Init-time static load: the adapter is "
        "pre-loaded into the engine cache and applied to every request. Mutually exclusive with --lora-paths.",
    )
    parser.add_argument(
        "--lora-scale",
        type=float,
        default=1.0,
        help="Scale factor for --lora-path (default: 1.0).",
    )
    parser.add_argument(
        "--lora-paths",
        nargs="+",
        default=None,
        help="Multiple LoRA adapter folders (PEFT format) for per-request composition. "
        "Each request applies all listed adapters with the matching --lora-scales. "
        "Mutually exclusive with --lora-path.",
    )
    parser.add_argument(
        "--lora-scales",
        nargs="+",
        type=float,
        default=None,
        help="Per-adapter scales for --lora-paths. Length must match --lora-paths; "
        "defaults to 1.0 per adapter when omitted.",
    )
    parser.add_argument(
        "--max-loras",
        type=int,
        default=None,
        help="Maximum number of LoRA slots active simultaneously. Defaults to max(len(--lora-paths), 1).",
    )
    parser.add_argument(
        "--xyz",
        action="store_true",
        help="XYZ plot mode: for each prompt, render a matrix of LoRA combos "
        "(baseline / each adapter alone / all composed) and stitch them into grid.png. "
        "The composed column is only added when --lora-paths has >=2 entries. "
        "Requires --output-dir and at least one --lora-paths entry.",
    )
    parser.add_argument(
        "--vae-patch-parallel-size",
        type=int,
        default=1,
        help="Number of ranks used for VAE patch/tile parallelism (decode/encode).",
    )
    # NextStep-1.1 specific arguments
    parser.add_argument(
        "--guidance-scale-2",
        type=float,
        default=1.0,
        help="Secondary guidance scale (e.g. image-level CFG for NextStep-1.1).",
    )
    parser.add_argument(
        "--timesteps-shift",
        type=float,
        default=1.0,
        help="[NextStep-1.1 only] Timesteps shift parameter for sampling.",
    )
    parser.add_argument(
        "--cfg-schedule",
        type=str,
        default="constant",
        choices=["constant", "linear"],
        help="[NextStep-1.1 only] CFG schedule type.",
    )
    parser.add_argument(
        "--use-norm",
        action="store_true",
        help="[NextStep-1.1 only] Apply layer normalization to sampled tokens.",
    )
    parser.add_argument(
        "--enable-diffusion-pipeline-profiler",
        action="store_true",
        help="Enable diffusion pipeline profiler to display stage durations.",
    )
    parser.add_argument(
        "--log-stats",
        action="store_true",
        help="Enable logging of diffusion pipeline stats.",
    )
    parser.add_argument(
        "--init-timeout",
        type=int,
        default=600,
        help="Timeout for initializing a single stage in seconds (default: 600s)",
    )
    parser.add_argument(
        "--stage-init-timeout",
        type=int,
        default=600,
        help="Timeout for initializing a single stage in seconds (default: 600s)",
    )
    parser.add_argument(
        "--use-system-prompt",
        type=str,
        default=None,
        choices=["None", "dynamic", "en_vanilla", "en_recaption", "en_think_recaption", "en_unified", "custom"],
        help="System prompt preset for generation. Recommended: en_unified.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help=("Custom system prompt. Used when --use-system-prompt is custom. "),
    )
    return parser.parse_args()


def _resolve_prompts(args: argparse.Namespace) -> list[str]:
    """Return the list of prompts to run. Prefers --prompts; falls back to --prompt."""
    if args.prompts:
        return list(args.prompts)
    return [args.prompt]


def _build_lora_request(path: str) -> LoRARequest:
    return LoRARequest(
        lora_name=Path(path).stem,
        lora_int_id=stable_lora_int_id(path),
        lora_path=path,
    )


def _resolve_lora_combos(args: argparse.Namespace) -> tuple[list[tuple[list[LoRARequest], list[float], str]], bool]:
    """Resolve the list of LoRA combos to run for per-request composition.

    Returns:
        (combos, is_per_request) where combos is a list of
        (lora_requests, lora_scales, label) tuples. ``is_per_request`` is True
        when per-request LoRA (via --lora-paths / --xyz) is active; False when
        the script runs in init-time LoRA mode (or no LoRA at all).

    In XYZ mode the combos are: baseline (no LoRA) + each adapter alone + full
    composition. Otherwise a single combo applies all --lora-paths at once.
    """
    if args.lora_path and args.lora_paths:
        raise ValueError("--lora-path and --lora-paths are mutually exclusive.")

    if not args.lora_paths:
        # Init-time --lora-path (or no LoRA) — a single implicit combo per
        # request, and the adapter is injected by Omni at init.
        return [([], [], "default")], False

    lora_paths = list(args.lora_paths)
    if args.lora_scales is None:
        lora_scales = [1.0] * len(lora_paths)
    else:
        lora_scales = list(args.lora_scales)
    if len(lora_paths) != len(lora_scales):
        raise ValueError(
            f"--lora-paths ({len(lora_paths)}) and --lora-scales ({len(lora_scales)}) must have the same length."
        )

    requests = [_build_lora_request(p) for p in lora_paths]
    names = [req.lora_name for req in requests]

    if not args.xyz:
        label = "+".join(f"{n}({s:.2f})" for n, s in zip(names, lora_scales))
        return [(requests, lora_scales, label)], True

    # XYZ mode: baseline + each adapter alone + composed
    combos: list[tuple[list[LoRARequest], list[float], str]] = [([], [], "baseline")]
    for req, scale in zip(requests, lora_scales):
        combos.append(([req], [scale], f"{req.lora_name}({scale:.2f})"))
    if len(requests) > 1:
        composed_label = "+".join(f"{n}({s:.2f})" for n, s in zip(names, lora_scales))
        combos.append((requests, lora_scales, composed_label))
    return combos, True


def _compose_grid(
    results: dict[tuple[int, int], list[Any]],
    num_rows: int,
    num_cols: int,
) -> Any:
    """Stitch the first image of each (row, col) cell into a single grid PIL image."""
    from PIL import Image

    sample = next(iter(results.values()))[0]
    cell_w, cell_h = sample.width, sample.height
    grid = Image.new("RGB", (cell_w * num_cols, cell_h * num_rows), color="white")
    for (r, c), imgs in results.items():
        grid.paste(imgs[0], (c * cell_w, r * cell_h))
    return grid


def main():
    args = parse_args()
    use_nextstep = is_nextstep_model(args.model)

    prompts = _resolve_prompts(args)
    lora_combos, lora_is_per_request = _resolve_lora_combos(args)

    # --output-dir is required whenever the script produces more than one image
    # (multiple prompts, multiple LoRA combos, or num_images_per_prompt > 1).
    requires_output_dir = len(prompts) > 1 or len(lora_combos) > 1 or args.num_images_per_prompt > 1 or args.xyz
    if requires_output_dir and not args.output_dir:
        raise ValueError(
            "--output-dir is required when running multiple prompts, multiple LoRA combos, "
            "multiple images per prompt, or --xyz. Single --output is only valid for one image."
        )
    if args.xyz and not lora_is_per_request:
        raise ValueError("--xyz requires --lora-paths to define the adapters to plot.")
    if args.max_loras is not None and args.lora_paths and args.max_loras < len(args.lora_paths):
        raise ValueError(
            f"--max-loras ({args.max_loras}) is smaller than len(--lora-paths) ({len(args.lora_paths)}). "
            "The composed combo needs one slot per adapter — raise --max-loras or remove it to auto-size."
        )

    cache_config = None
    cache_backend = args.cache_backend

    if cache_backend == "cache_dit":
        # cache-dit configuration: Hybrid DBCache + SCM + TaylorSeer
        # All parameters marked with [cache-dit only] in DiffusionCacheConfig
        cache_config = {
            # DBCache parameters [cache-dit only]
            "Fn_compute_blocks": 1,  # Optimized for single-transformer models
            "Bn_compute_blocks": 0,  # Number of backward compute blocks
            "max_warmup_steps": 4,  # Maximum warmup steps (works for few-step models)
            "residual_diff_threshold": 0.24,  # Higher threshold for more aggressive caching
            "max_continuous_cached_steps": 3,  # Limit to prevent precision degradation
            # TaylorSeer parameters [cache-dit only]
            "enable_taylorseer": False,  # Disabled by default (not suitable for few-step models)
            "taylorseer_order": 1,  # TaylorSeer polynomial order
            # SCM (Step Computation Masking) parameters [cache-dit only]
            "scm_steps_mask_policy": None,  # SCM mask policy: None (disabled), "slow", "medium", "fast", "ultra"
            "scm_steps_policy": "dynamic",  # SCM steps policy: "dynamic" or "static"
        }
    elif cache_backend == "tea_cache":
        # TeaCache configuration
        # All parameters marked with [tea_cache only] in DiffusionCacheConfig
        cache_config = {
            # TeaCache parameters [tea_cache only]
            "rel_l1_thresh": 0.2,  # Threshold for accumulated relative L1 distance
            # Note: coefficients will use model-specific defaults based on model_type
            #       (e.g., QwenImagePipeline or FluxPipeline)
        }

    # assert args.ring_degree == 1, "Ring attention is not supported yet"
    parallel_config = DiffusionParallelConfig(
        ulysses_degree=args.ulysses_degree,
        ring_degree=args.ring_degree,
        ulysses_mode=args.ulysses_mode,
        cfg_parallel_size=args.cfg_parallel_size,
        tensor_parallel_size=args.tensor_parallel_size,
        vae_patch_parallel_size=args.vae_patch_parallel_size,
        enable_expert_parallel=args.enable_expert_parallel,
    )

    # Check if profiling is requested via environment variable
    profiler_enabled = bool(os.getenv("VLLM_TORCH_PROFILER_DIR"))

    # Prepare LoRA kwargs for Omni initialization
    lora_args: dict[str, Any] = {}
    if args.lora_path:
        lora_args["lora_path"] = args.lora_path
        print(f"Using init-time LoRA from: {args.lora_path}")

    # max_loras sizes the adapter cache at init. Per-request combos may load
    # several adapters simultaneously, so default to max(len(lora_paths), 1).
    if args.max_loras is not None:
        lora_args["max_loras"] = args.max_loras
    elif args.lora_paths:
        lora_args["max_loras"] = max(len(args.lora_paths), 1)

    # Build quantization kwargs: use quantization_config dict when
    # ignored_layers is specified so the list flows through OmniDiffusionConfig
    quant_kwargs: dict[str, Any] = {}
    ignored_layers = [s.strip() for s in args.ignored_layers.split(",") if s.strip()] if args.ignored_layers else None
    if args.quantization == "gguf":
        if not args.gguf_model:
            raise ValueError("--gguf-model is required when --quantization gguf is set.")
        quant_kwargs["quantization_config"] = {
            "method": "gguf",
            "gguf_model": args.gguf_model,
        }
    elif args.quantization and ignored_layers:
        quant_kwargs["quantization_config"] = {
            "method": args.quantization,
            "ignored_layers": ignored_layers,
        }
    elif args.quantization:
        quant_kwargs["quantization"] = args.quantization

    omni_kwargs = {
        "model": args.model,
        "enable_layerwise_offload": args.enable_layerwise_offload,
        "vae_use_slicing": args.vae_use_slicing,
        "vae_use_tiling": args.vae_use_tiling,
        "cache_backend": args.cache_backend,
        "cache_config": cache_config,
        "enable_cache_dit_summary": args.enable_cache_dit_summary,
        "parallel_config": parallel_config,
        "enforce_eager": args.enforce_eager,
        "enable_cpu_offload": args.enable_cpu_offload,
        "mode": "text-to-image",
        "log_stats": args.log_stats,
        "enable_diffusion_pipeline_profiler": args.enable_diffusion_pipeline_profiler,
        "init_timeout": args.init_timeout,
        "stage_init_timeout": args.stage_init_timeout,
        **lora_args,
        **quant_kwargs,
    }
    if args.stage_configs_path:
        omni_kwargs["stage_configs_path"] = args.stage_configs_path
    if use_nextstep:
        # NextStep-1.1 requires explicit pipeline class
        omni_kwargs["model_class_name"] = "NextStep11Pipeline"
    omni = Omni(**omni_kwargs)

    if profiler_enabled:
        print("[Profiler] Starting profiling...")
        omni.start_profile()

    # Time profiling for generation
    print(f"\n{'=' * 60}")
    print("Generation Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Inference steps: {args.num_inference_steps}")
    print(f"  Cache backend: {cache_backend if cache_backend else 'None (no acceleration)'}")
    print(f"  Quantization: {args.quantization if args.quantization else 'None (BF16)'}")
    if ignored_layers:
        print(f"  Ignored layers: {ignored_layers}")
    print(
        f"  Parallel configuration: tensor_parallel_size={args.tensor_parallel_size}, "
        f"ulysses_degree={args.ulysses_degree}, ulysses_mode={args.ulysses_mode}, "
        f"ring_degree={args.ring_degree}, cfg_parallel_size={args.cfg_parallel_size}, "
        f"vae_patch_parallel_size={args.vae_patch_parallel_size}, "
        f"enable_expert_parallel={args.enable_expert_parallel}."
    )
    print(f"  CPU offload: {args.enable_cpu_offload}; CPU Layerwise Offload: {args.enable_layerwise_offload}")
    print(f"  Image size: {args.width}x{args.height}")
    if args.lora_path:
        print(f"  Init-time LoRA: scale={args.lora_scale}")
    if lora_is_per_request:
        print(f"  Per-request LoRA combos ({len(lora_combos)}):")
        for idx, combo in enumerate(lora_combos):
            print(f"    [{idx}] {combo[2]}")
    print(f"  Prompts: {len(prompts)}")
    if args.stage_configs_path:
        print(f"  stage-configs-path: {args.stage_configs_path}")
    print(f"{'=' * 60}\n")

    extra_args = {
        "timesteps_shift": args.timesteps_shift,
        "cfg_schedule": args.cfg_schedule,
        "use_norm": args.use_norm,
        "use_system_prompt": args.use_system_prompt,
        "system_prompt": args.system_prompt,
    }

    prompt_batch = [{"prompt": p, "negative_prompt": args.negative_prompt} for p in prompts]
    # (combo_idx, prompt_idx) -> list of PIL images (length == num_images_per_prompt)
    cell_images: dict[tuple[int, int], list[Any]] = {}

    generation_start = time.perf_counter()
    for combo_idx, (lora_reqs, lora_scales, combo_label) in enumerate(lora_combos):
        # Fresh generator per combo so the seed is the only source of variance
        # within a cell; across combos the seed is reused so differences isolate
        # to the LoRA weights (matches the test in tests/e2e/.../test_diffusion_lora.py).
        combo_generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(args.seed)
        sp = OmniDiffusionSamplingParams(
            height=args.height,
            width=args.width,
            generator=combo_generator,
            true_cfg_scale=args.cfg_scale,
            guidance_scale=args.guidance_scale,
            guidance_scale_2=args.guidance_scale_2,
            num_inference_steps=args.num_inference_steps,
            num_outputs_per_prompt=args.num_images_per_prompt,
            lora_requests=lora_reqs,
            lora_scales=lora_scales,
            extra_args=extra_args,
        )
        print(f"[combo {combo_idx}/{len(lora_combos) - 1}] {combo_label} ...")
        combo_outputs = omni.generate(prompt_batch, sp)
        if len(combo_outputs) != len(prompts):
            raise ValueError(f"Expected {len(prompts)} outputs for combo {combo_idx}, got {len(combo_outputs)}.")
        for p_idx, out in enumerate(combo_outputs):
            if not getattr(out, "request_output", None) or not hasattr(out.request_output, "images"):
                raise ValueError(f"Combo {combo_idx} prompt {p_idx}: missing request_output.images")
            imgs = out.request_output.images
            if not imgs:
                raise ValueError(f"Combo {combo_idx} prompt {p_idx}: empty image list")
            cell_images[(p_idx, combo_idx)] = imgs

    generation_end = time.perf_counter()
    generation_time = generation_end - generation_start

    # Print profiling results
    print(f"Total generation time: {generation_time:.4f} seconds ({generation_time * 1000:.2f} ms)")

    if profiler_enabled:
        print("\n[Profiler] Stopping profiler and collecting results...")
        profile_results = omni.stop_profile()
        if profile_results and isinstance(profile_results, dict):
            traces = profile_results.get("traces", [])
            print("\n" + "=" * 60)
            print("PROFILING RESULTS:")
            for rank, trace in enumerate(traces):
                print(f"\nRank {rank}:")
                if trace:
                    print(f"  • Trace: {trace}")
            if not traces:
                print("  No traces collected.")
            print("=" * 60)
        else:
            print("[Profiler] No valid profiling data returned.")

    logger.info("Produced %d cells (combos=%d, prompts=%d)", len(cell_images), len(lora_combos), len(prompts))

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for (p_idx, c_idx), imgs in cell_images.items():
            for n_idx, img in enumerate(imgs):
                save_path = out_dir / f"p{p_idx:02d}_c{c_idx:02d}_n{n_idx:02d}.png"
                img.save(save_path)
                print(f"Saved {save_path}")
        if args.xyz:
            grid = _compose_grid(cell_images, num_rows=len(prompts), num_cols=len(lora_combos))
            grid_path = out_dir / "grid.png"
            grid.save(grid_path)
            print(f"Saved XYZ grid to {grid_path}")
    else:
        # Single-output mode: exactly one cell, one image.
        only_images = next(iter(cell_images.values()))
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        only_images[0].save(output_path)
        print(f"Saved generated image to {output_path}")


if __name__ == "__main__":
    main()
