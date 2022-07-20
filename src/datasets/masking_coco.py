"""
This script creates masked blends between 
the COCO and stylized COCO datasets.

Stylized COCO objects
Stylized COCO background
"""

import os
import json
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
import numpy as np

def masking_coco(coco_dir, stylized_coco_dir, out_dir_objects, out_dir_background, annotations):

  # assuming the following directory structure:
  # stylized_coco_dir/feature_space/
  #     [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 , 0.9, 1.0]
  #
  # stylized_coco_dir/pixel_space/
  #          [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 , 0.9]
  #
  #  will reproduce the same directory structure in
  # out_dir_objects/
  # out_dir_backgrounds/

  # ensure that every stylized coco alpha blend version contains the same images as coco
  coco_file_paths = [fp for fp in coco_dir.iterdir() if fp.suffix == ".jpg"]
  coco_file_names = sorted([fp.name for fp in coco_file_paths])
  print(f"\n         COCO found: {len(coco_file_paths)} images in: {coco_dir}")

  feature_space_paths = [x for x in (stylized_coco_dir / "feature_space").iterdir() if x.is_dir()]
  feature_space_paths.sort(key=lambda p: float(p.name))

  pixel_space_paths = [x for x in (stylized_coco_dir / "pixel_space").iterdir() if x.is_dir()]
  pixel_space_paths.sort(key=lambda p: float(p.name))

  for alpha in feature_space_paths + pixel_space_paths:
    file_paths = [fp for fp in alpha.iterdir() if fp.suffix == ".jpg"]
    file_names = sorted([fp.name for fp in file_paths])

    assert coco_file_names == file_names, f"COCO files and Stylized COCO files are not the same in: {alpha}"
    print(f"STYLIZED_COCO found: {len(file_paths)} images in: {alpha}")

  # create output directories and copy the annotation file from COCO
  print(f"\nCreate stylized coco objects:    {out_dir_objects}")
  print(f"Create stylized coco background: {out_dir_background}")

  (out_dir_objects    / "annotations").mkdir(parents=True, exist_ok=True)
  (out_dir_background / "annotations").mkdir(parents=True, exist_ok=True)
  shutil.copy(annotations, out_dir_objects    / "annotations")
  shutil.copy(annotations, out_dir_background / "annotations")

  for alpha in feature_space_paths:
    (out_dir_objects    / "feature_space" / alpha.name).mkdir(parents=True, exist_ok=True)
    (out_dir_background / "feature_space" / alpha.name).mkdir(parents=True, exist_ok=True)

  for alpha in pixel_space_paths:
    (out_dir_objects    / "pixel_space" / alpha.name).mkdir(parents=True, exist_ok=True)
    (out_dir_background / "pixel_space" / alpha.name).mkdir(parents=True, exist_ok=True)

  # load coco
  coco = COCO(annotations)
  coco_ids = list(coco.getImgIds())

  print("\nPrecompute the union mask of all objects per image.")
  union_masks = {}
  for ID in tqdm(coco_ids):
    file_name = coco.imgs[ID]['file_name']
    ann = coco.loadAnns(coco.getAnnIds(ID))
    masks = [coco.annToMask(a) for a in ann]

    if len(masks) == 0:
      I = Image.open(COCO_PATH / "val2017" / file_name)
      union_mask = np.zeros((I.height, I.width), dtype=np.uint8)
    else:
      union_mask = masks[0]
      for m in masks:
        union_mask = np.logical_or(union_mask, m).astype(np.uint8)

    union_masks[ID] = Image.fromarray(union_mask * 255,"L")
  
  # Masking COCO and Stylized COCO -> Objects and Background
  for ID in tqdm(coco_ids):
    file_name = coco.imgs[ID]['file_name']
    coco_img  = Image.open(coco_dir / file_name).convert('RGB')

    # feature space
    for alpha in feature_space_paths:
      style_img        = Image.open(alpha / file_name)
      object_blend     = Image.composite(style_img, coco_img, union_masks[ID])
      background_blend = Image.composite(coco_img, style_img, union_masks[ID])

      object_blend.save(out_dir_objects        / "feature_space" / alpha.name / file_name)
      background_blend.save(out_dir_background / "feature_space" / alpha.name / file_name)
    
    # pixel space
    for alpha in pixel_space_paths:
      style_img        = Image.open(alpha / file_name)
      object_blend     = Image.composite(style_img, coco_img, union_masks[ID])
      background_blend = Image.composite(coco_img, style_img, union_masks[ID])

      object_blend.save(out_dir_objects        / "pixel_space" / alpha.name / file_name)
      background_blend.save(out_dir_background / "pixel_space" / alpha.name / file_name)

if __name__ == '__main__':
  # setup path variables
  ROOT = Path(__file__).parent / "../.."
  with open(ROOT / 'path_config.json') as f:
    config = json.load(f)

  DATA_ROOT = Path(config["DATA_ROOT"])
  if not DATA_ROOT.is_absolute():
    DATA_ROOT = (ROOT / DATA_ROOT).resolve()

  COCO_PATH   = DATA_ROOT / "coco"
  COCO_IMG    = COCO_PATH / "val2017"
  ANNOTATIONS = COCO_PATH / "annotations/instances_val2017.json"
  
  for style_num in range(1,11):
    STYLIZED_COCO_PATH       = DATA_ROOT / "stylized_coco" / str(style_num)
    STYLIZED_COCO_OBJECTS    = DATA_ROOT / "stylized_coco_objects" / str(style_num)
    STYLIZED_COCO_BACKGROUND = DATA_ROOT / "stylized_coco_background" / str(style_num)

    masking_coco(COCO_IMG, STYLIZED_COCO_PATH, STYLIZED_COCO_OBJECTS, STYLIZED_COCO_BACKGROUND, ANNOTATIONS)
