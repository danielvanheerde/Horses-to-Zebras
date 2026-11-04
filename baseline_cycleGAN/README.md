# Baseline CycleGAN

This folder runs the **original, unchanged CycleGAN** from `third_party/pytorch-CycleGAN-and-pix2pix` on the unpaired horse/zebra dataset.

## Train

```powershell
python baseline_cycleGAN/train.py
```

This trains the default CycleGAN for 200 epochs (100 at the initial learning rate, then 100 with linear decay). Model weights are saved in `baseline_cycleGAN/checkpoints/horse2zebra_baseline/`.

## Generate images

After training:

```powershell
python baseline_cycleGAN/test.py
```

Images are saved under `baseline_cycleGAN/results/horse2zebra_baseline/test_latest/images/`.

## What are these scripts doing?

- `train.py` calls the original third-party `train.py` with the dataset and output paths.
- `test.py` calls the original third-party `test.py` to generate images using saved weights.
- Neither file defines a new model. The model and losses live in `third_party/pytorch-CycleGAN-and-pix2pix/models/cycle_gan_model.py`.

**Setup:** Follow the [root README](../README.md#clone-and-set-up) for the
submodules and shared environment. This pinned CycleGAN selects CUDA GPU 0
when available and CPU otherwise. It does not accept the older `--gpu_ids` flag.
Extra command-line arguments are forwarded to the upstream script, for example:

```sh
python baseline_cycleGAN/train.py --batch_size 1 --num_threads 0
```
