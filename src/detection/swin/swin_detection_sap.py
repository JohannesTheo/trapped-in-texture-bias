import os
import json
from pathlib import Path
import pickle
import shutil

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def eval_and_save(res_file, out_path, annotations):

    out_path.mkdir(parents=True, exist_ok=True)
    shutil.move(res_file, out_path)
    
    cocoGt=COCO(annotations)
    resFile= str(out_path / 'results.segm.json')
    cocoDt=cocoGt.loadRes(resFile)

    for iou_type in ["bbox", "segm"]:
        cocoEval = COCOeval(cocoGt,cocoDt, iou_type)
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        out_file = out_path / f"coco_eval_{iou_type}.pkl"
        with open(out_file, "wb") as f:
            pickle.dump(cocoEval.eval, f)

if __name__ == '__main__':
    # setup path variables
    ROOT = (Path(__file__).parent / "../../..").resolve()
    with open(ROOT / 'path_config.json') as f:
        path_config = json.load(f)

    DATA_ROOT = Path(path_config["DATA_ROOT"])
    if not DATA_ROOT.is_absolute():
        DATA_ROOT = (ROOT / DATA_ROOT).resolve()

    COCO_PATH   = DATA_ROOT / "coco"
    ANNOTATIONS = DATA_ROOT / "coco/annotations/instances_val2017.json"
    STYLIZED_COCO_PATH       = "sap_coco"
    STYLIZED_COCO_OBJECTS    = "sap_objects"
    STYLIZED_COCO_BACKGROUND = "sap_background"
    OUTPUT_DIR = ROOT / "detections" / "swin"

    # swin models: https://github.com/SwinTransformer/Swin-Transformer-Object-Detection
    MODELS = [
        ["mask_rcnn_swin-T_3x",         "mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py",  "v1.0.2/", "mask_rcnn_swin_tiny_patch4_window7.pth"],
        ["mask_rcnn_swin-S_3x",         "mask_rcnn_swin_small_patch4_window7_mstrain_480-800_adamw_3x_coco.py", "v1.0.2/", "mask_rcnn_swin_small_patch4_window7.pth"],
        ["cascade_mask_rcnn_swin-B_3x", "cascade_mask_rcnn_swin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py",  "v1.0.2/", "cascade_mask_rcnn_swin_base_patch4_window7.pth"]
    ]
    WEIGHT_URL_BASE = "https://github.com/SwinTransformer/storage/releases/download/"

    # switch to swin repository
    SWIN_ROOT = ROOT / 'ext/Swin-Transformer-Object-Detection'
    os.chdir(SWIN_ROOT)

    # create local link to dataset root
    os.symlink(DATA_ROOT, 'data')

    # run detection
    for model, cfg, release_version, weights in MODELS:

        # 1. Download weights if not existing
        if not Path(f"./{weights}").is_file():
            os.system(f'wget {WEIGHT_URL_BASE}{release_version}{weights}')

        # 2. COCO
        output_dir = OUTPUT_DIR / model / 'coco_sap'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        os.system(f"CUDA_VISIBLE_DEVICES=2,3 python tools/test.py configs/swin/{cfg} {weights} --format-only --eval-options jsonfile_prefix=results")
        os.remove("./results.bbox.json")
        eval_and_save('./results.segm.json', output_dir, ANNOTATIONS)

        # 2. STYLIZED DATASETS
        # We rename the original val2017 folder and create symlinks to the stylized versions so we don't have to rewrite the detr dataset code.
        os.rename(COCO_PATH / "val2017", COCO_PATH / "val2017_bak")

        for dataset in [STYLIZED_COCO_PATH, STYLIZED_COCO_OBJECTS, STYLIZED_COCO_BACKGROUND]:

                    
            # create symlink to the stylized images
            os.symlink(DATA_ROOT / dataset , COCO_PATH / 'val2017')

            # run the evaluation
            output_dir = OUTPUT_DIR / model / dataset 
            output_dir.mkdir(parents=True, exist_ok=True)
            os.system(f"CUDA_VISIBLE_DEVICES=2,3 python tools/test.py configs/swin/{cfg} {weights} --format-only --eval-options jsonfile_prefix=results")
            os.remove("./results.bbox.json")
            eval_and_save('./results.segm.json', output_dir, ANNOTATIONS)
                    
            # remove the symlink
            os.remove(COCO_PATH / 'val2017')
                    
        # bring back the original coco val2017 images
        os.rename(COCO_PATH / "val2017_bak", COCO_PATH / "val2017")

    # remove the data link
    os.remove('data')
