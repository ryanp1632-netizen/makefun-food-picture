"""Generate candidate food photographs with FLUX.1-schnell (Apache-2.0 open weights) on an 8 GB laptop GPU.

    python render/generate_flux.py --base <flux-schnell diffusers folder> --gguf <flux1-schnell-Q4_K_S.gguf>
                                   --out <dir> --dish ramen,burger --seeds 1-8

How it fits in 8 GB: the prompts are encoded once with the CLIP and T5 text encoders on the CPU, the encoders are
dropped, and only the 4-bit GGUF transformer and the VAE go to the GPU. Local only: no network, no paid service.
"""
import argparse
import gc
import os

import torch
from accelerate import cpu_offload
from diffusers import FluxPipeline, FluxTransformer2DModel, GGUFQuantizationConfig

from prompts import DISHES, STYLE


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
    ap.add_argument("--seeds", default="1-4")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--offload", default="model", choices=["none", "model", "group"])
    args = ap.parse_args()
    dishes = args.dish.split(",")

    pipe = FluxPipeline.from_pretrained(args.base, transformer=None, torch_dtype=torch.bfloat16)
    # bf16 T5-XXL on a laptop CPU takes many minutes, and it is too big for the GPU: stream it through the GPU
    # one layer at a time instead (a few seconds a prompt).
    cpu_offload(pipe.text_encoder, execution_device=torch.device("cuda"))
    cpu_offload(pipe.text_encoder_2, execution_device=torch.device("cuda"))
    embeds = {}
    with torch.no_grad():
        for dish in dishes:
            prompt = f"{DISHES[dish]} {STYLE}"
            pe, ppe, _ = pipe.encode_prompt(prompt=prompt, prompt_2=prompt, device="cuda", max_sequence_length=256)
            embeds[dish] = (pe.cpu(), ppe.cpu())
            print("encoded", dish, flush=True)
    pipe.text_encoder = pipe.text_encoder_2 = None
    gc.collect()
    torch.cuda.empty_cache()

    pipe.transformer = FluxTransformer2DModel.from_single_file(
        args.gguf,
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        config=args.base,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )
    if args.offload == "model":
        pipe.enable_model_cpu_offload()
    elif args.offload == "group":
        pipe.transformer.enable_group_offload(
            onload_device=torch.device("cuda"), offload_device=torch.device("cpu"),
            offload_type="block_level", num_blocks_per_group=4, use_stream=True,
        )
        pipe.vae.to("cuda")
    else:
        pipe.to("cuda")
    pipe.vae.enable_tiling()

    os.makedirs(args.out, exist_ok=True)
    for dish in dishes:
        pe, ppe = embeds[dish]
        for seed in seed_list(args.seeds):
            image = pipe(
                prompt_embeds=pe.to("cuda"),
                pooled_prompt_embeds=ppe.to("cuda"),
                width=args.size,
                height=args.size,
                num_inference_steps=args.steps,
                guidance_scale=0.0,
                generator=torch.Generator("cpu").manual_seed(seed),
            ).images[0]
            path = os.path.join(args.out, f"flux_{dish}_{seed}.png")
            image.save(path)
            print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
