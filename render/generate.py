"""Generate candidate food photographs with a local, open-weight SDXL model.

Runs on a local GPU; no network at generation time once the weights are on disk, no paid service.

    python render/generate.py --model <RealVisXL_V5.0_fp16.safetensors> --out <dir> --dish ramen --seeds 1-8

Each candidate is written as <dish>_<seed>.png at 1024x1024. render/finish.py turns the chosen one into entry.png.
"""
import argparse
import os

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline

STYLE = (
    "close-up hero shot from a low 30-degree angle, the dish fills the frame, professional editorial food "
    "photography, 100mm lens, shallow depth of field, dark moody background with soft bokeh, warm golden "
    "backlight making the steam glow, rich glossy textures, appetizing, mouth-watering, ultra detailed, "
    "sharp focus on the food, award-winning food magazine cover"
)

DISHES = {
    "ramen": (
        "a steaming bowl of tonkotsu ramen, chopsticks lifting glossy noodles, a halved soft-boiled ramen egg "
        "with a jammy orange yolk, seared chashu pork slices with charred edges, chili oil swirls, sliced "
        "scallions, sesame seeds, nori sheet, creamy golden broth, wisps of steam rising, dark stoneware bowl "
        "on a rustic wooden table"
    ),
    "burger": (
        "a towering double cheeseburger, glossy toasted brioche bun with sesame seeds, two seared smash patties "
        "with crispy edges, melted cheddar cheese dripping down the side, crispy bacon, fresh lettuce, tomato, "
        "pickles, golden french fries beside it, on a dark slate board, wisps of steam"
    ),
    "pizza": (
        "a neapolitan margherita pizza fresh from a wood-fired oven, a slice being lifted with long stretchy "
        "melted mozzarella cheese pull, leopard-spotted charred puffy crust, bright san marzano tomato sauce, "
        "fresh basil leaves, drizzle of olive oil, on a wooden peel, fire glowing in the background"
    ),
    "lava": (
        "a warm chocolate lava cake on a dark plate, cut open with molten glossy chocolate flowing out, a scoop "
        "of vanilla bean ice cream melting beside it, fresh raspberries, dusting of cocoa powder, flakes of sea "
        "salt, wisps of steam"
    ),
    "pancakes": (
        "a tall stack of fluffy golden buttermilk pancakes, maple syrup pouring and dripping down the sides, a "
        "melting pat of butter on top, fresh blueberries and raspberries, dusting of powdered sugar, on a "
        "white plate on a rustic table, morning window light"
    ),
}

NEGATIVE = (
    "text, watermark, logo, signature, letters, cartoon, illustration, painting, 3d render, cgi, plastic, "
    "blurry, out of focus food, lowres, deformed, distorted, disfigured, extra chopsticks, broken chopsticks, "
    "fused fingers, bad hands, ugly, oversaturated, muddy, grainy, jpeg artifacts, burnt, moldy, raw meat"
)


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
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dish", default="ramen", help="comma-separated: " + ",".join(sorted(DISHES)))
    ap.add_argument("--seeds", default="1-4")
    ap.add_argument("--steps", type=int, default=36)
    ap.add_argument("--cfg", type=float, default=5.0)
    args = ap.parse_args()

    pipe = StableDiffusionXLPipeline.from_single_file(args.model, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++"
    )
    # 8 GB laptop GPU: keep only the active sub-model on the GPU (the whole pipeline spills into shared memory).
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=True)

    os.makedirs(args.out, exist_ok=True)
    for dish, seed in [(d, s) for d in args.dish.split(",") for s in seed_list(args.seeds)]:
        prompt = f"{DISHES[dish]}, {STYLE}"
        gen = torch.Generator("cuda").manual_seed(seed)
        image = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE,
            width=1024,
            height=1024,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            generator=gen,
        ).images[0]
        path = os.path.join(args.out, f"{dish}_{seed}.png")
        image.save(path)
        print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
