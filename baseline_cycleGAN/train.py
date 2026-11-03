"""Train the original CycleGAN on the horse/zebra dataset."""
from pathlib import Path
import subprocess
import sys

# Find the project folders.
project = Path(__file__).resolve().parent.parent
cyclegan = project / "third_party" / "pytorch-CycleGAN-and-pix2pix"

# Start the original CycleGAN training script with our dataset.
subprocess.run([
    sys.executable, "train.py",
    "--dataroot", str(project / "data" / "horse2zebra"),
    "--name", "horse2zebra_baseline",
    "--model", "cycle_gan",
    "--dataset_mode", "unaligned",
    "--checkpoints_dir", str(project / "baseline_cycleGAN" / "checkpoints"),
], cwd=cyclegan, check=True)
