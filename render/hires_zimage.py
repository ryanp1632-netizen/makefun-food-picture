"""Hi-res detail pass: upscale a chosen Z-Image candidate to 2048 px and re-denoise it lightly with Z-Image-Turbo
img2img, so fine texture (noodle strands, fat, sesame, steam) is re-drawn at a higher resolution; finish.py then
downsamples it to 1024. The judge kept docking "slightly uniform, plasticky" surfaces.

    python render/hires_zimage.py --base <Z-Image-Turbo folder> --gguf <q8 gguf> --dish ramen4 --style izakaya
                                  --strength 0.3 --seed 1 in.png out.png
"""
import argparse
import gc

import torch
from accelerate import cpu_offload
from diffusers import GGUFQuantizationConfig, ZImageImg2ImgPipeline, ZImageTransformer2DModel
from PIL import Image

from prompts import DISHES, STYLES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--dish", required=True)
    ap.add_argument("--style", default="izakaya", choices=sorted(STYLES))
    ap.add_argument("--strength", type=float, default=0.3)
    ap.add_argument("--steps", type=int, default=9)
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("src")
    ap.add_argument("dst")
    args = ap.parse_args()

    pipe = ZImageImg2ImgPipeline.from_pretrained(args.base, transformer=None, torch_dtype=torch.bfloat16)
    cpu_offload(pipe.text_encoder, execution_device=torch.device("cuda"))
    with torch.no_grad():
        pe, _ = pipe.encode_prompt(
            prompt=f"{DISHES[args.dish]} {STYLES[args.style]}", device="cuda", do_classifier_free_guidance=False
        )
    pe = [e.cpu() for e in pe]
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
    pipe.vae.enable_tiling()

    image = Image.open(args.src).convert("RGB").resize((args.size, args.size), Image.LANCZOS)
    out = pipe(
        prompt_embeds=[e.to("cuda") for e in pe],
        image=image,
        strength=args.strength,
        width=args.size,
        height=args.size,
        num_inference_steps=args.steps,
        guidance_scale=0.0,
        generator=torch.Generator("cuda").manual_seed(args.seed),
    ).images[0]
    out.save(args.dst)
    print("wrote", args.dst, flush=True)


if __name__ == "__main__":
    main()
