"""
This script creates alpha blends in pixel space
of coco and a stylized coco version.
"""

import os
import json
import shutil
from pathlib import Path
from utils import download
from PIL import Image
from tqdm import tqdm

def alpha_blend(coco_dir, stylized_coco_dir, output_dir, alpha_values):

  # ensure that the images in coco and the stylized coco versions are the same
  coco_file_paths = [fp for fp in coco_dir.iterdir() if fp.suffix == ".jpg"]
  coco_file_names = sorted([fp.name for fp in coco_file_paths])
  print(f"\n         COCO found: {len(coco_file_paths)} images in: {coco_dir}")

  style_file_paths = [fp for fp in stylized_coco_dir.iterdir() if fp.suffix == ".jpg"]
  style_file_names = sorted([fp.name for fp in style_file_paths])
      
  assert coco_file_names == style_file_names, f"COCO files and Stylized COCO files are not the same in: {version}"
  print(f"STYLIZED_COCO found: {len(style_file_paths)} images in: {stylized_coco_dir}")
  print(f"Alpha values: {alpha_values}")

  # create directory structure for requested alpha values
  for alpha in alpha_values:
    new_dir = output_dir / str(alpha)
    new_dir.mkdir(parents=True, exist_ok=True)
    
  # blend coco and stylized coco images
  print(f"OUTPUT DIR: {output_dir}")  
  for FILE in tqdm(coco_file_names):

    original_coco_img = Image.open(coco_dir / FILE).convert('RGB') # ensure images are in RGB mode (some are L)
    stylized_coco_img = Image.open(stylized_coco_dir / FILE)

    for alpha in alpha_values:
      blend = Image.blend(original_coco_img, stylized_coco_img, alpha)
      blend.save(output_dir / str(alpha) / FILE)

if __name__ == '__main__':

  # setup path variables
  ROOT = Path(__file__).parent / "../.."
  with open(ROOT / 'path_config.json') as f:
    config = json.load(f)

  DATA_ROOT = Path(config["DATA_ROOT"])
  if not DATA_ROOT.is_absolute():
    DATA_ROOT = (ROOT / DATA_ROOT).resolve()

  COCO_PATH          = DATA_ROOT / "coco" / "val2017"
  for style_num in range(1,11):
    STYLIZED_COCO_PATH = DATA_ROOT / "stylized_coco" / str(style_num) / "feature_space" / "1.0"
    OUTPUT_PATH        = DATA_ROOT / "stylized_coco" / str(style_num) / "pixel_space"
    ALPHA_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # 0 = original coco images, 1.0 = fully stylized coco images

    alpha_blend(COCO_PATH, STYLIZED_COCO_PATH, OUTPUT_PATH, ALPHA_VALUES)
