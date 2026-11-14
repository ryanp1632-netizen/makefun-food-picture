"""Photo-realism refine pass: a low-strength SDXL (RealVisXL V5.0) img2img over a FLUX candidate.

The judge marked our first FLUX burger down for a "plasticky", "AI sheen" look. RealVisXL is tuned on photographs;
at strength ~0.2-0.3 it keeps the FLUX composition and re-renders surface texture (crumb, char, fat, steam).

    python render/refine.py --model <RealVisXL_V5.0_fp16.safetensors> --strength 0.25 --seed 7 in.png out.png "<short prompt>"
"""
import argparse

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLImg2ImgPipeline
from PIL import Image

NEGATIVE = (
    "plastic, cgi, 3d render, illustration, cartoon, waxy, oversaturated, smooth skin, blurry, text, watermark, "
    "deformed, artifacts"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--strength", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("pairs", nargs="+", help="in.png out.png [in.png out.png ...] prompt")
    args = ap.parse_args()
    *paths, prompt = args.pairs
    prompt = f"{prompt}, raw photo, real food photography, natural texture, detailed, sharp"

    pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(args.model, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)

    for src, dst in zip(paths[0::2], paths[1::2]):
        image = Image.open(src).convert("RGB")
        out = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE,
            image=image,
            strength=args.strength,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            generator=torch.Generator("cuda").manual_seed(args.seed),
        ).images[0]
        out.save(dst)
        print("wrote", dst, flush=True)


if __name__ == "__main__":
    main()
