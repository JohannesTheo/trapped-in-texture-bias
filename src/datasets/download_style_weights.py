#!python

"""
This script replaces download_models.sh from the stylize-datasets repository.
Download pytorch-AdaIN Style weights from: https://github.com/naoto0804/pytorch-AdaIN
since the links in: https://github.com/bethgelab/stylize-datasets are broke.
"""

import json
from pathlib import Path
from utils import download

# setup path variables
ROOT = (Path(__file__).parent / "../..").resolve()
MODEL_DIR  = ROOT / "ext/stylize-datasets/models"

# download pytorch-AdaIN Style weights from: https://github.com/naoto0804/pytorch-AdaIN (get ids from the gdrive links)
GOOGLE_DRIVE_URL  = "https://docs.google.com/uc?export=download"
VGG_NORMALISED    = MODEL_DIR / "vgg_normalised.pth"
VGG_NORMALISED_ID = "1EpkBA2K2eYILDSyPTt0fztz59UjAIpZU"
VGG_DECODER       = MODEL_DIR / "decoder.pth"
VGG_DECODER_ID    = "1bMfhMMwPeXnYSQI6cDWElSZxOxc6aVyr"

if not VGG_NORMALISED.exists(): download(GOOGLE_DRIVE_URL, str(VGG_NORMALISED), params = { 'id': VGG_NORMALISED_ID})
if not VGG_DECODER.exists():    download(GOOGLE_DRIVE_URL, str(VGG_DECODER),    params = { 'id': VGG_DECODER_ID})

print(f"Download Style Weights: ✓\n")
