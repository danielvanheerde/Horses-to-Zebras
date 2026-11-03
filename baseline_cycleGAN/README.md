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

**GPU:** The reference scripts use CUDA GPU 0 by default. If you don't have an NVIDIA GPU with a working CUDA PyTorch install, add `"--gpu_ids", "-1",` to the argument list in **both** Python files to run on CPU (much slower).
