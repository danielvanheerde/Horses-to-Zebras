# Diffusion track: DDPM from scratch for horse → zebra

A diffusion project that sits alongside the team's GAN baselines. We build a  
diffusion model **by hand** (schedule, forward process, loss, samplers and
U-Net, with no `diffusers` in the core). We translate horses to zebras with
SDEdit and compare the results against CycleGAN / AttentionGAN on the same test
images.

We expect it to do worse than the GANs at 64×64 on ~2.4k images. The point is to understand *why* and explore the differences. 

## How translation works 

Train a model to remove noise from images of horses **and** zebras, telling it
which one each image is. To translate a horse, add noise until its texture is
gone but its pose and layout survive, then denoise it while telling the model
"zebra". 

## Layout

Everything for this track lives in `diffusion/`. Run modules from the repo
root with `python -m diffusion.<module>`.

```markdown
diffusion/
├── README.md            this file
├── device.py            cuda → mps → cpu selection, and ensuring random seeding
├── configs/             one YAML per experiment (model size, batch, steps, paths)
├── data/
│   └── prepare.py       data/horse2zebra/*.jpg → cached 64×64 uint8 tensors
├── core/                the diffusion maths, written out by hand
│   ├── schedule.py      beta schedule, alpha, alpha-bar (how much noise)
│   ├── forward.py       q(x_t | x_0): noise an image in one jump
│   ├── losses.py        noise-prediction MSE (DDPM labels this "L_simple")
│   ├── sampler.py       DDPM ancestral + DDIM, classifier-free guidance
│   └── sdedit.py        noise-to-strength-then-denoise translation
├── models/
│   └── unet.py          hand-written diffusion U-Net
├── train.py             training loop (EMA, checkpoints, sample grids)
├── translate.py         SDEdit grid: fixed test horses × strengths to visualise it learning
├── eval/                (later)
├── notebooks/           (later) data exploration
└── tests/               quick sanity checks for core/ and models/
```

Generated files are kept out of git: the cached dataset goes in `data/` at the
repo root and runs/checkpoints go in `outputs/diffusion/<run>/`. Both are
already gitignored.

## Setup

Use the team's shared conda environment (see the root README). This track only
adds `pyyaml` to `environment.yml`.

## Hardware

Device order is CUDA → Apple MPS → CPU, chosen automatically.

- **Apple Silicon (M-series):** runs on MPS in float32. Use the `*_mps.yaml`
configs, which have a smaller batch. Keep the laptop plugged in and run under
`caffeinate` so it does not sleep mid-run.
- **Rented NVIDIA GPU:** use the full-size configs, inside `tmux`.

