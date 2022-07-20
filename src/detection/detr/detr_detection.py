import os
import json
from pathlib import Path

if __name__ == '__main__':
    # setup path variables
    ROOT = (Path(__file__).parent / "../../..").resolve()
    with open(ROOT / 'path_config.json') as f:
        path_config = json.load(f)

    DATA_ROOT = Path(path_config["DATA_ROOT"])
    if not DATA_ROOT.is_absolute():
        DATA_ROOT = (ROOT / DATA_ROOT).resolve()

    COCO_PATH = DATA_ROOT / "coco"
    COCO_PANOPTIC_PATH = DATA_ROOT / "coco_panoptic"
    STYLIZED_COCO_PATH       = "stylized_coco"
    STYLIZED_COCO_OBJECTS    = "stylized_coco_objects"
    STYLIZED_COCO_BACKGROUND = "stylized_coco_background"
    OUTPUT_DIR = ROOT / "detections" / "detr"
    STYLE_VERSIONS = ["1"]

    # detr CLI eval calls: https://gist.github.com/szagoruyko/9c9ebb8455610958f7deaa27845d7918
    DETR_R50     = { 'name': 'detr_r50' ,     'cmd': f"python custom_main.py --batch_size 1 --no_aux_loss --eval --resume https://dl.fbaipublicfiles.com/detr/detr-r50-panoptic-00ce5173.pth --masks --dataset_file coco_panoptic --coco_path {COCO_PATH} --coco_panoptic_path {COCO_PANOPTIC_PATH} --output_dir"}
    DETR_R50_DC5 = { 'name': 'detr_r50_dc5' , 'cmd': f"python custom_main.py --dilation --batch_size 1 --no_aux_loss --eval --resume https://dl.fbaipublicfiles.com/detr/detr-r50-dc5-panoptic-da08f1b1.pth --masks --dataset_file coco_panoptic --coco_path {COCO_PATH} --coco_panoptic_path {COCO_PANOPTIC_PATH} --output_dir"}
    DETR_R101    = { 'name': 'detr_r101' ,    'cmd': f"python custom_main.py --backbone resnet101 --batch_size 1 --no_aux_loss --eval --resume https://dl.fbaipublicfiles.com/detr/detr-r101-panoptic-40021d53.pth --masks --dataset_file coco_panoptic --coco_path {COCO_PATH} --coco_panoptic_path {COCO_PANOPTIC_PATH} --output_dir"}

    for model in [DETR_R50, DETR_R50_DC5, DETR_R101]:

        # 1. COCO
        output_dir = OUTPUT_DIR / model['name'] / 'coco/val2017'
        output_dir.mkdir(parents=True, exist_ok=True)
        os.system(f"{model['cmd']} {output_dir}")

        # 2. STYLIZED DATASETS
        # We rename the original val2017 folder and create symlinks to the stylized versions so we don't have to rewrite the detr dataset code.
        os.rename(COCO_PATH / "val2017", COCO_PATH / "val2017_bak")

        for dataset in [STYLIZED_COCO_PATH, STYLIZED_COCO_OBJECTS, STYLIZED_COCO_BACKGROUND]:
            for style in STYLE_VERSIONS:

                # feature space blendings
                for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                    
                    # create symlink to the stylized images
                    os.symlink(DATA_ROOT / dataset / style / 'feature_space' / str(alpha), COCO_PATH / 'val2017')

                    # run the evaluation
                    output_dir = OUTPUT_DIR / model['name'] / dataset / style / "feature_space" / str(alpha)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    os.system(f"{model['cmd']} {output_dir}")
                    
                    # remove the symlink
                    os.remove(COCO_PATH / 'val2017')
                    
                # pixel space blendings
                for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:

                    # create symlink to the stylized images
                    os.symlink(DATA_ROOT / dataset / style / 'pixel_space' / str(alpha), COCO_PATH / 'val2017')

                    # run the evaluation
                    output_dir = OUTPUT_DIR / model['name'] / dataset / style / "pixel_space" / str(alpha)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    os.system(f"{model['cmd']} {output_dir}")

                    # remove the symlink
                    os.remove(COCO_PATH / 'val2017')

        # bring back the original coco val2017 images
        os.rename(COCO_PATH / "val2017_bak", COCO_PATH / "val2017")
