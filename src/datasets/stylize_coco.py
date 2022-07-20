"""
This script creates stylized versions of COCO using
https://github.com/bethgelab/stylize-datasets
which is included as a git submodule.
"""

import os
import json
import shutil
from pathlib import Path
from utils import download
from PIL import Image
from tqdm import tqdm

# create the stylized versions
def stylize_coco(content_dir, style_dir, output_dir, content_style_map=None, alpha_value=1.0):

  output_dir = output_dir / str(alpha_value)
  if output_dir.exists():
    print(f"\nFound stylized COCO {output_dir} skipping.")
    return
  else:
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    print(f"\nCreating stylized COCO {output_dir}")

    # stylize coco    
    if content_style_map is None:
      print(f"Using random styles.\n")
      os.system(f"python ./stylize.py --content-dir {content_dir} --style-dir {style_dir} --output-dir {output_dir} --num-styles 1 --alpha {alpha_value}")
    else:
      print(f"Using content style map {content_style_map}\n")
      os.system(f"python ./stylize.py --content-dir {content_dir} --style-dir {style_dir} --output-dir {output_dir} --style-map {content_style_map} --alpha {alpha_value}")

    # resolve hickups
    SKIPPED_IMGS = output_dir / "skipped_imgs.txt"
    while SKIPPED_IMGS.exists():
      print(f"\nResolving hickups")
      TMP = OUT_PATH / "tmp"
      TMP.mkdir()
      with open(SKIPPED_IMGS, 'r') as skipped:
        for img in skipped:
          shutil.copy(img.rstrip(), TMP)
      os.system(f"python ./stylize.py --content-dir {TMP} --style-dir {style_dir} --output-dir {output_dir} --num-styles 1")
      shutil.move(str(SKIPPED_IMGS), str(TMP))
      shutil.rmtree(TMP)

    print("\nPostprocessing & cleaning")
    file_names     = os.listdir(output_dir)
    for f in tqdm(file_names):
        if not f.endswith(".jpg"): continue
        
        # 1. rename files to original COCO IDs for easier use later
        s = f.split("-") 
        coco_id = s[0] + ".jpg"
        os.rename(output_dir / f, output_dir / coco_id)
        
        # 2. resize styled image if not same as original (not always the case after styling)
        coco_img   = Image.open(content_dir / coco_id)
        styled_img = Image.open(output_dir  / coco_id)
        if (coco_img.size != styled_img.size):
            styled_img.resize(coco_img.size).save(output_dir  / coco_id)

  print(f"\nCreate Stylized COCO: ✓")

if __name__ == '__main__':
  
  # setup path variables
  ROOT = Path(__file__).parent / "../.."
  with open(ROOT / 'path_config.json') as f:
    config = json.load(f)

  DATA_ROOT = Path(config["DATA_ROOT"])
  if not DATA_ROOT.is_absolute():
    DATA_ROOT = (ROOT / DATA_ROOT).resolve()

  COCO_PATH  = DATA_ROOT / "coco"
  COCO_IMGS  = COCO_PATH / "val2017"
  COCO_ANNS  = COCO_PATH / "annotations" / "instances_val2017.json"
  STYLE_IMGS = DATA_ROOT / "style_images" / "train"
  STYLIZED_COCO_PATH = DATA_ROOT / "stylized_coco"
  STYLIZED_COCO_ANNS = STYLIZED_COCO_PATH / "annotations"

  STYLIZE_CODE = ROOT / "ext" / "stylize-datasets"

  # create stylized_coco directory
  STYLIZED_COCO_PATH.mkdir(parents=True, exist_ok=True)

  # copy annotation file from COCO
  STYLIZED_COCO_ANNS.mkdir(parents=True, exist_ok=True)
  shutil.copy(COCO_ANNS, STYLIZED_COCO_ANNS / COCO_ANNS.name )
  
  # stylize coco
  os.chdir(STYLIZE_CODE)

  for style_num in range(1,11):
    OUT_PATH = STYLIZED_COCO_PATH / str(style_num) / "feature_space"
    CONTENT_STYLE_MAP = DATA_ROOT / "coco_style_maps" / f"coco_style_map_{style_num}.json"

    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        stylize_coco(COCO_IMGS, STYLE_IMGS, OUT_PATH, CONTENT_STYLE_MAP, alpha_value=alpha)
