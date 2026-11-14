"""Generate candidate food photographs with Z-Image-Turbo (Tongyi-MAI, Apache-2.0 open weights) on an 8 GB GPU.

    python render/generate_zimage.py --base <Z-Image-Turbo diffusers folder> --gguf <z-image-turbo-Q8_0.gguf>
                                     --out <dir> --dish ramen,burger --style window --seeds 1-8

Why this model: the judge kept marking FLUX.1-schnell food down as "plasticky", "AI sheen", "pale and uniform";
Z-Image-Turbo is known for photographs that read as real. Same 8 GB recipe as generate_flux.py: the Qwen3 text
encoder is streamed through the GPU layer by layer to encode the prompts once, then dropped; the 8-bit GGUF
transformer runs with block-level group offload. Local only: no network, no paid service.
"""
import argparse
import gc
import os

import torch
from accelerate import cpu_offload
from diffusers import GGUFQuantizationConfig, ZImagePipeline, ZImageTransformer2DModel

from prompts import DISHES, STYLES


def seed_list(spec: str) -> list[int]:
    seeds: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return seeds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dish", default="ramen", help="comma-separated: " + ",".join(sorted(DISHES)))
    ap.add_argument("--style", default="window", choices=sorted(STYLES))
    ap.add_argument("--seeds", default="1-4")
    ap.add_argument("--steps", type=int, default=9)
    ap.add_argument("--size", type=int, default=1024)
    args = ap.parse_args()
    dishes = args.dish.split(",")

    pipe = ZImagePipeline.from_pretrained(args.base, transformer=None, torch_dtype=torch.bfloat16)
    cpu_offload(pipe.text_encoder, execution_device=torch.device("cuda"))
    embeds = {}
    with torch.no_grad():
        for dish in dishes:
            prompt = f"{DISHES[dish]} {STYLES[args.style]}"
            pe, _ = pipe.encode_prompt(prompt=prompt, device="cuda", do_classifier_free_guidance=False)
            embeds[dish] = [e.cpu() for e in pe]
            print("encoded", dish, flush=True)
    pipe.text_encoder = None
    gc.collect()
    torch.cuda.empty_cache()

    pipe.transformer = ZImageTransformer2DModel.from_single_file(
        args.gguf,
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        config=args.base,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )
    pipe.transformer.enable_group_offload(
        onload_device=torch.device("cuda"), offload_device=torch.device("cpu"),
        offload_type="block_level", num_blocks_per_group=4, use_stream=True, low_cpu_mem_usage=True,
    )
    pipe.vae.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    os.makedirs(args.out, exist_ok=True)
    for dish in dishes:
        for seed in seed_list(args.seeds):
            image = pipe(
                prompt_embeds=[e.to("cuda") for e in embeds[dish]],
                width=args.size,
                height=args.size,
                num_inference_steps=args.steps,
                guidance_scale=0.0,
                generator=torch.Generator("cuda").manual_seed(seed),
            ).images[0]
            path = os.path.join(args.out, f"zi_{args.style}_{dish}_{seed}.png")
            image.save(path)
            print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
