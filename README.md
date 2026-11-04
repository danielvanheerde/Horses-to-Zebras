# Horses-to-Zebras

We use Unpaired Image-to-Image Translation to try and transform horses into zebras.

Links:

- [Spec](https://webcms3.cse.unsw.edu.au/files/987abcfb972ecb9ef1b561d32e865ddb6760413135e83d64c7bcf139d0732df9)
- [Rubrics etc](https://webcms3.cse.unsw.edu.au/COMP9444/26T3/resources/124889)
- [Dataset](https://www.tensorflow.org/datasets/catalog/cycle_gan)
- [pytorch-CycleGAN-and-pix2pix Github](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

## Clone and set up

Install Git and Conda, then clone the project with its external model implementations:

```sh
git clone --recurse-submodules https://github.com/danielvanheerde/Horses-to-Zebras.git
cd Horses-to-Zebras
```

If you already cloned the project, run this from its root to get submodules:

```sh
git submodule update --init --recursive
```

Run that same command after pulling or switching branches whenever the submodules change.
Avoid cloning the upstream projects manually.
The source URLs are in `.gitmodules`; `git submodule status` shows the revisions.

AttentionGAN needs a Pillow compatibility patch for its removed SciPy resize
function. Apply it once after the initial submodule checkout:

```sh
git -C third_party/AttentionGAN apply ../../patches/attentiongan-pillow.patch
```

## Environment

Run:

```sh
conda env create -f environment.yml
conda activate horse2zebra
```

For updates within this setup:

```sh
conda env update -n horse2zebra -f environment.yml --prune
```

Do not separately install the third-party `requirements.txt` files.

Check the installation:

```sh
python -m pip check
python -c "from PIL import Image; import torch, torchvision; print(torch.__version__, torchvision.__version__); print('CUDA:', torch.cuda.is_available())"
python baseline_cycleGAN/train.py --help
python baseline_attentionGAN/train.py --help
```

CUDA should be `True` for NVIDIA GPU training. The current wrappers use CPU
when CUDA is unavailable.

Use `from PIL import Image` before importing torch/torchvision
in new scripts and notebooks because of [this bug](https://github.com/conda-forge/pillow-feedstock/issues/170).

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

## Project layout

- `baseline_cycleGAN/` and `baseline_attentionGAN/`: entry points for the reference models.
- `third_party/`: external implementations included as Git submodules (see `.gitmodules`).
- `notebooks/`: exploratory work and the final demonstration notebook.
- `scripts/`: commands for downloading data, training and evaluation.
- `data/` and model checkpoint folders: local generated files, excluded from Git.

Future (inspo from other research repos):

- `src/`: reusable project code as it is extracted from notebooks.
- `configs/`: experiment settings.
- `experiments/`: small experiment records: settings, seed, code revision, and observations.
- `results/`: small summaries and selected figures for comparison.
- `docs/`: sources, experiment planning, and project notes.

Only small curated figures belong in `results/figures/`, generated image batches
should go under an ignored output directories.

## Working together with Git

Use a separate branch for each task and merge through a GitHub pull request.
Keep `main` usable by the rest of the team. Run these commands from the repository root.

Start a new task with a clean working tree (`git status` should show no pending changes):

```sh
git switch main
git pull --ff-only origin main
git switch -c yourname/short-task-description
```

Save and share your work on that branch:

```sh
git status
git add .
git commit -m "blah blah blah"
git push -u origin HEAD
```

For notebooks, clear large outputs before committing.

Open a pull request on GitHub with `main` as the base. Explain what changed,
how to run it, and what you checked. For environment changes, update the relevant environment
files and setup instructions in the same pull request. Avoid editing the same notebook
simultaneously as they aren't tracked well by git.
