# AttentionGAN (horse to zebra)

This uses the **authors' original AttentionGAN implementation**, not a new model written by our team.

Paper: Tang et al., *AttentionGAN: Unpaired Image-to-Image Translation Using Attention-Guided Generative Adversarial Networks*.

Code: https://github.com/Ha0Tang/AttentionGAN

## Setup

Follow the [root README](../README.md#clone-and-set-up) to initialize the pinned
submodules, apply `patches/attentiongan-pillow.patch`, and create the shared
Conda environment. Do not clone another copy or install upstream dependency pins.

## Train

The `data/horse2zebra` directory should contain `trainA`, `trainB`, `testA`, and `testB`.

```powershell
python baseline_attentionGAN/train.py
```

This follows the author's horse2zebra training script: batch size 4, 60 epochs, 256-pixel crops, and GPU 0 when CUDA is available (otherwise CPU). We disable Visdom popup and set data loading workers to 0 for simpler Windows use.

## Test

```powershell
python baseline_attentionGAN/test.py
```

Outputs: `baseline_attentionGAN/results/`

Weights: `baseline_attentionGAN/checkpoints/horse2zebra_attention/`

## How is it different from CycleGAN?

The attention-guided generator predicts attention maps to select where to change the image and where to preserve the input. This is a **separate architecture**, not an attention toggle in `cycle_gan_model.py`.

Source of the model's losses and forward pass:
`third_party/AttentionGAN/models/attention_gan_model.py`

Note: training has not been run or verified with your Windows/CUDA environment. This package only provides simple entry points matching the authors' published commands. Compare epoch counts and batch size before claiming a fair evaluation against the existing 200-epoch CycleGAN baseline.

Extra command-line arguments are forwarded to the upstream script, for example:

```sh
python baseline_attentionGAN/train.py --batch_size 1 --gpu_ids -1
```
