"""Generate test images using the trained AttentionGAN."""
from pathlib import Path
import subprocess
import sys

project = Path(__file__).resolve().parent.parent
attentiongan = project / "third_party" / "AttentionGAN"

subprocess.run([
    sys.executable, "test.py",
    "--dataroot", str(project / "data" / "horse2zebra"),
    "--name", "horse2zebra_attention",
    "--model", "attention_gan",
    "--dataset_mode", "unaligned",
    "--checkpoints_dir", str(project / "attentionGAN" / "checkpoints"),
    "--results_dir", str(project / "attentionGAN" / "results"),
    "--norm", "instance",
    "--phase", "test",
    "--no_dropout",
    "--load_size", "256",
    "--crop_size", "256",
    "--batch_size", "1",
    "--gpu_ids", "0",
    "--num_threads", "0",
    "--epoch", "latest",
    "--saveDisk",
], cwd=attentiongan, check=True)
