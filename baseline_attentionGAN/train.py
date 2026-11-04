"""Train the authors' AttentionGAN model on unpaired horses and zebras."""
from pathlib import Path
import subprocess
import sys
import torch

project = Path(__file__).resolve().parent.parent
attentiongan = project / "third_party" / "AttentionGAN"

subprocess.run([
    # Preload Pillow before upstream imports torch.
    sys.executable, "-c",
    "from PIL import Image; import runpy; runpy.run_path('train.py', run_name='__main__')",
    "--dataroot", str(project / "data" / "horse2zebra"),
    "--name", "horse2zebra_attention",
    "--model", "attention_gan",
    "--dataset_mode", "unaligned",
    "--checkpoints_dir", str(project / "baseline_attentionGAN" / "checkpoints"),
    "--pool_size", "50",
    "--no_dropout",
    "--norm", "instance",
    "--lambda_A", "10",
    "--lambda_B", "10",
    "--lambda_identity", "0.5",
    "--load_size", "286",
    "--crop_size", "256",
    "--batch_size", "4",
    "--niter", "60",
    "--niter_decay", "0",
    "--gpu_ids", "0" if torch.cuda.is_available() else "-1",
    "--num_threads", "0",
    "--display_id", "-1",
    "--print_freq", "100",
] + sys.argv[1:], cwd=attentiongan, check=True)
