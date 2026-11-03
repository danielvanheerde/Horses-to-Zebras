"""Generate zebra images using the trained CycleGAN."""
from pathlib import Path
import subprocess
import sys

project = Path(__file__).resolve().parent.parent
cyclegan = project / "third_party" / "pytorch-CycleGAN-and-pix2pix"

subprocess.run([
    sys.executable, "test.py",
    "--dataroot", str(project / "data" / "horse2zebra"),
    "--name", "horse2zebra_baseline",
    "--model", "cycle_gan",
    "--dataset_mode", "unaligned",
    "--checkpoints_dir", str(project / "baseline_cycleGAN" / "checkpoints"),
    "--results_dir", str(project / "baseline_cycleGAN" / "results"),
    "--num_test", "50",
], cwd=cyclegan, check=True)
