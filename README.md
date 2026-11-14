# makefun-food-picture

An entry for make.fun's **food-picture-01** ("Make the best-looking food picture").

The entry is `main.ts` (Node 22, standard library only). It checks `entry.png`, a whole PNG or JPEG of at most
1 MiB and 1024x1024, and copies those exact bytes to `/output/entry.png`, so both grader runs write the same file.

## How the picture was made

- `render/generate_flux.py` runs **FLUX.1-schnell** (open weights, Apache-2.0) locally on an 8 GB RTX 4060 laptop
  GPU with Hugging Face `diffusers`: a 4-bit GGUF copy of the transformer, the text encoders streamed through the
  GPU layer by layer, fixed seeds, 4 steps. Prompts live in `render/prompts.py`. No paid service or API key.
- `render/generate.py` is the earlier SDXL (RealVisXL V5.0) generator; its shots came out flatter, so FLUX won.
- Candidates were compared by eye at full size and at 300 px; the chosen one went through `render/finish.py`,
  which writes `entry.png` within the gate's limits (a lossless PNG when it fits, otherwise a high-quality JPEG).

## Run it

```bash
mkdir -p out && MAKEFUN_OUTPUT_DIR=out node main.ts
```

## Licence

MIT, see [LICENSE](LICENSE).
