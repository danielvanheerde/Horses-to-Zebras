# Setup

From the repository root, using Conda:

```powershell
conda env create -f environment-win.yml  # On Mac, use environment-mac.yml
conda activate horse2zebra
python -m pip install "setuptools<80" wheel
python -m pip install -r requirements.txt --no-build-isolation
```

If `horse2zebra` already exists, **skip** the `conda env create` line.


Download the images, then run a model:

```powershell
python scripts/download_data.py
python baseline_cycleGAN/train.py
# or: python baseline_attentionGAN/train.py
```

**GPU:** Run `python -c "import torch; print(torch.cuda.is_available())"`. For NVIDIA training, it should print `True`; otherwise install CUDA-enabled PyTorch from https://pytorch.org/get-started/locally/.

**AttentionGAN:** Its old code uses the removed `scipy.misc.imresize`; replace that call with Pillow as described in the project notes. Installing dependencies alone won't fix it.
