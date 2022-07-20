import os
import json
from pathlib import Path
import pickle
import shutil

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def eval_and_save(res_dir, out_path, annotations):

    out_path.mkdir(parents=True, exist_ok=True)
    shutil.move(res_dir + "bbox_detections.json", out_path)
    shutil.move(res_dir + "mask_detections.json", out_path)
    
    for iou_type, res_file in [['bbox', "bbox_detections.json"],['segm', "mask_detections.json"]]:
        cocoGt=COCO(annotations)
        resFile= str(out_path / res_file)
        cocoDt=cocoGt.loadRes(resFile)

        cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
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
    OUTPUT_DIR = ROOT / "detections" / "yolact"

    # Yolact models: https://github.com/dbolya/yolact (store them in yolact/weights)
    MODELS = [
        ["yolact-R50-FPN",       "yolact_resnet50_54_800000.pth"],
        ["yolact-R101-FPN",      "yolact_base_54_800000.pth"],
        ["yolact-plus-R50-FPN",  "yolact_plus_resnet50_54_800000.pth"],
        ["yolact-plus-R101-FPN", "yolact_plus_base_54_800000.pth"]
    ]

    # switch to solo repository
    YOLACT_ROOT = ROOT / 'ext/yolact'
    os.chdir(YOLACT_ROOT)

    # create local link to coco root
    os.chdir("./data")
    os.symlink(COCO_PATH, 'coco')
    os.chdir("..")

    # run detection
    for model, weights in MODELS:

         # 1. COCO val2017
        output_dir = OUTPUT_DIR / model / 'coco_sap/val2017'
        output_dir.mkdir(parents=True, exist_ok=True)

        os.symlink(COCO_PATH / 'val2017', COCO_PATH / 'images')
        os.system(f"MKL_SERVICE_FORCE_INTEL='True' CUDA_VISIBLE_DEVICES=3 python eval.py --trained_model=weights/{weights} --output_coco_json")
        eval_and_save(f'./results/', output_dir, ANNOTATIONS)
        shutil.rmtree(f"./results")
        os.remove(COCO_PATH / 'images')
        
        # 2. STYLIZED DATASETS (S & P Noise)
        for dataset in [STYLIZED_COCO_PATH, STYLIZED_COCO_OBJECTS, STYLIZED_COCO_BACKGROUND]:
                          
            # create symlink to the stylized images
            os.symlink(DATA_ROOT / dataset, COCO_PATH / 'images')

            # run the evaluation
            output_dir = OUTPUT_DIR / model / dataset
            output_dir.mkdir(parents=True, exist_ok=True)

            os.system(f"MKL_SERVICE_FORCE_INTEL='True' CUDA_VISIBLE_DEVICES=3 python eval.py --trained_model=weights/{weights} --output_coco_json")
            eval_and_save(f'./results/', output_dir, ANNOTATIONS)
            shutil.rmtree(f"./results")

            # remove the symlink
            os.remove(COCO_PATH / 'images')

    # remove the data link
    os.remove('data/coco')
