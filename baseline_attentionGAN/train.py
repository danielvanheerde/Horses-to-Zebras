"""Train the authors' AttentionGAN model on unpaired horses and zebras."""
from pathlib import Path
import subprocess
import sys

project = Path(__file__).resolve().parent.parent
attentiongan = project / "third_party" / "AttentionGAN"

subprocess.run([
    sys.executable, "train.py",
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
    "--gpu_ids", "0",
    "--num_threads", "0",
    "--display_id", "-1",
    "--print_freq", "100",
], cwd=attentiongan, check=True)
