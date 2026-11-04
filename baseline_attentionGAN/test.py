"""Generate test images using the trained AttentionGAN."""
from pathlib import Path
import subprocess
import sys
import torch

project = Path(__file__).resolve().parent.parent
attentiongan = project / "third_party" / "AttentionGAN"

subprocess.run([
    # Preload Pillow before upstream imports torch.
    sys.executable, "-c",
    "from PIL import Image; import runpy; runpy.run_path('test.py', run_name='__main__')",
    "--dataroot", str(project / "data" / "horse2zebra"),
    "--name", "horse2zebra_attention",
    "--model", "attention_gan",
    "--dataset_mode", "unaligned",
    "--checkpoints_dir", str(project / "baseline_attentionGAN" / "checkpoints"),
    "--results_dir", str(project / "baseline_attentionGAN" / "results"),
    "--norm", "instance",
    "--phase", "test",
    "--no_dropout",
    "--load_size", "256",
    "--crop_size", "256",
    "--batch_size", "1",
    "--gpu_ids", "0" if torch.cuda.is_available() else "-1",
    "--num_threads", "0",
    "--epoch", "latest",
    "--saveDisk",
] + sys.argv[1:], cwd=attentiongan, check=True)
