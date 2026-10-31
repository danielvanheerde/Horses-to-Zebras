# Horses-to-Zebras

We use Unpaired Image-to-Image Translation to try and transform horses into zebras.

Links:

- [Spec](https://webcms3.cse.unsw.edu.au/files/987abcfb972ecb9ef1b561d32e865ddb6760413135e83d64c7bcf139d0732df9)
- [Rubrics etc](https://webcms3.cse.unsw.edu.au/COMP9444/26T3/resources/124889)
- [Dataset](https://www.tensorflow.org/datasets/catalog/cycle_gan)
- [pytorch-CycleGAN-and-pix2pix Github](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

## Environment

Install dependencies using conda by running

```sh
conda env create -f environment-win.yml
```

(or using `environment-mac.yml` on mac),

and then activate the environment by running

```sh
conda activate horse2zebra
```

## Download the dataset

Run

```sh
python ./scripts/download_data.py
```

to download the dataset for the first time into `./data`.
