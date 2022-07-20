#!python

"""
Download COCO val2017 (same as minival 2014)
https://cocodataset.org/#download

DATASETS_ROOT is loaded from ./config.json
"""

import json
from pathlib import Path
from zipfile import ZipFile
from utils import download

# setup path variables
ROOT = Path(__file__).parent / "../.."
with open(ROOT / 'path_config.json') as f:
  config = json.load(f)

DATA_ROOT = Path(config["DATA_ROOT"])
if not DATA_ROOT.is_absolute():
  DATA_ROOT = (ROOT / DATA_ROOT).resolve()

COCO_PATH = DATA_ROOT / "coco"

# download information and sub dirs
IMG_URL = "http://images.cocodataset.org/zips/val2017.zip"
IMG_ZIP = COCO_PATH / "val2017.zip"
IMG_DIR = IMG_ZIP.with_suffix('')
ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
ANN_ZIP = COCO_PATH / "annotations.zip"
ANN_DIR = ANN_ZIP.with_suffix('')

# ensure the coco directory exists
COCO_PATH.mkdir(parents=True, exist_ok=True)

# download and unzip COCO Images
if not IMG_ZIP.exists(): download(IMG_URL, str(IMG_ZIP))
if not IMG_DIR.is_dir():
    print(f"Unzip: {IMG_ZIP}")
    with ZipFile(IMG_ZIP, 'r') as zf: zf.extractall(COCO_PATH)

# download and unzip COCO Annotations
if not ANN_ZIP.exists(): download(ANN_URL, str(ANN_ZIP))
if not ANN_DIR.is_dir():
    print(f"Unzip: {ANN_ZIP}")
    with ZipFile(ANN_ZIP, 'r')  as zf: zf.extractall(COCO_PATH)

print(f"Download COCO: ✓\n")
