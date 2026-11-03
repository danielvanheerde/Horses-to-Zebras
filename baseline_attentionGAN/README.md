# AttentionGAN (horse to zebra)

This uses the **authors' original AttentionGAN implementation**, not a new model written by our team.

Paper: Tang et al., *AttentionGAN: Unpaired Image-to-Image Translation Using Attention-Guided Generative Adversarial Networks*.

Code: https://github.com/Ha0Tang/AttentionGAN

## 1. Add the original implementation

From the root of `Horses-to-Zebras`:

```powershell
git clone https://github.com/Ha0Tang/AttentionGAN.git third_party/AttentionGAN
```

## 2. Install the extra dependencies

Activate the existing `horse2zebra` Conda environment. Keep the CUDA-enabled PyTorch installation that is already working for your CycleGAN.

```powershell
conda activate horse2zebra
python -m pip install visdom dominate
```

The original AttentionGAN repository is older and lists `scipy==1.1.0`. **Do not install that old pin into Python 3.11**; it is not compatible with this newer environment. The older repository may require further compatibility fixes with modern PyTorch/torchvision.

## 3. Train

The `data/horse2zebra` directory should contain `trainA`, `trainB`, `testA`, and `testB`.

```powershell
python attentionGAN/train.py
```

This follows the author's horse2zebra training script: batch size 4, 60 epochs, 256-pixel crops, and GPU 0. We disable Visdom popup and set data loading workers to 0 for simpler Windows use.

## 4. Test

```powershell
python attentionGAN/test.py
```

Outputs: `attentionGAN/results/`

Weights: `attentionGAN/checkpoints/horse2zebra_attention/`

## How is it different from CycleGAN?

The attention-guided generator predicts attention maps to select where to change the image and where to preserve the input. This is a **separate architecture**, not an attention toggle in `cycle_gan_model.py`.

Source of the model's losses and forward pass:
`third_party/AttentionGAN/models/attention_gan_model.py`

Note: training has not been run or verified with your Windows/CUDA environment. This package only provides simple entry points matching the authors' published commands. Compare epoch counts and batch size before claiming a fair evaluation against the existing 200-epoch CycleGAN baseline.
