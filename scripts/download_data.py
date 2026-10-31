# Adapted from ../third_party/pytorch-CycleGAN-and-pix2pix/datasets/download_cyclegan_dataset.sh
# Used python to be platform agnostic.
from pathlib import Path
import shutil
import urllib.request
import zipfile

URL = "https://efrosgans.eecs.berkeley.edu/cyclegan/datasets/horse2zebra.zip"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def download():
    """
    Downloads and extracts the horse2zebra dataset to ./data/
    Returns 0 for success, 1 for error.
    """
    if DATA_DIR.exists() and not DATA_DIR.is_dir():
        print(f"Error: {DATA_DIR} is not a directory.")
        return 1

    if DATA_DIR.exists() and any(DATA_DIR.iterdir()):
        print(f"Error: {DATA_DIR} is not empty.")
        return 1

    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        zip_path = DATA_DIR / "horse2zebra.zip"

        print("Downloading Horse2Zebra dataset...")
        with urllib.request.urlopen(URL) as response:
            with zip_path.open("wb") as output:
                shutil.copyfileobj(response, output)

        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path) as zip:
            zip.extractall(DATA_DIR)

    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    print(f"Dataset downloaded to {DATA_DIR / 'horse2zebra'}")
    return 0

if __name__ == "__main__":
    download()
