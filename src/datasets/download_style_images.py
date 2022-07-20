#!python

"""
Download Style Images 
https://github.com/bethgelab/stylize-datasets
https://www.kaggle.com/c/painter-by-numbers/data train.zip (36.04 GB)
"""

import sys
import json
import argparse
from pathlib import Path
from utils import download
from zipfile import ZipFile
from tqdm import tqdm

# setup path variables
ROOT = Path(__file__).parent / "../.."
with open(ROOT / 'path_config.json') as f:
  config = json.load(f)

DATA_ROOT = Path(config["DATA_ROOT"])
if not DATA_ROOT.is_absolute():
  DATA_ROOT = (ROOT / DATA_ROOT).resolve()

STYLE_PATH = DATA_ROOT / "style_images"
STYLE_ZIP = STYLE_PATH / "train.zip"
STYLE_DIR = STYLE_ZIP.with_suffix('')
print(STYLE_PATH)
# ensure the style directory exists
STYLE_PATH.mkdir(parents=True, exist_ok=True)

# get STYLE_URL
parser = argparse.ArgumentParser()
parser.add_argument("STYLE_URL", help="You need to provide the download url which you can get from https://www.kaggle.com/c/painter-by-numbers/data?select=train.zip (requires a Kaggle account). To get the actual download url, you can click the download button, stop the download and fish the url from your browsers extended download menu. Note that you need to wrap it in string quotes and that the url will expire after some time.")
args = parser.parse_args()

# download style images
if not STYLE_ZIP.exists(): download(args.STYLE_URL, str(STYLE_ZIP))
if not STYLE_DIR.is_dir():
  print(f"Unzip: {STYLE_ZIP}")
  with ZipFile(STYLE_ZIP, 'r') as zf:
    for file in tqdm(iterable=zf.namelist(), total=len(zf.namelist())):
      zf.extract(member=file, path=STYLE_PATH)
print(f"Download Style Images: ✓\n")
