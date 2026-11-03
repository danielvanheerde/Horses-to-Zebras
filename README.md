# Horses-to-Zebras

We use Unpaired Image-to-Image Translation to try and transform horses into zebras.

Links:

- [Spec](https://webcms3.cse.unsw.edu.au/files/987abcfb972ecb9ef1b561d32e865ddb6760413135e83d64c7bcf139d0732df9)
- [Rubrics etc](https://webcms3.cse.unsw.edu.au/COMP9444/26T3/resources/124889)
- [Dataset](https://www.tensorflow.org/datasets/catalog/cycle_gan)
- [pytorch-CycleGAN-and-pix2pix Github](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

## Environment and setup

The environment files include the project dependencies. On Windows with an
NVIDIA GPU, create and activate the environment with:

```sh
conda env create -f environment-win.yml
conda activate horse2zebra
```

On macOS, use `environment-mac.yml` instead. If the `horse2zebra` environment
already exists, update it with:

```sh
conda env update -n horse2zebra -f environment-win.yml --prune
```

Substitute `environment-mac.yml` on macOS, then activate the environment.

**GPU:** Run `python -c "import torch; print(torch.cuda.is_available())"`. For NVIDIA training, it should print `True`; otherwise install CUDA-enabled PyTorch from [here](https://pytorch.org/get-started/locally/).

**AttentionGAN:** Its old code uses the removed `scipy.misc.imresize`; replace that call with Pillow as described in the project notes. Installing dependencies alone won't fix it.

## Download the dataset

Run

```sh
python ./scripts/download_data.py
```

to download the dataset for the first time into `./data`.

## Train a model

```sh
python baseline_cycleGAN/train.py
# or
python baseline_attentionGAN/train.py
```
