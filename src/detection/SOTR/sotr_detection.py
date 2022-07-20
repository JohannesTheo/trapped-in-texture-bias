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
    resFile= str(out_path / 'coco_instances_results.json')
    cocoDt=cocoGt.loadRes(resFile)

    cocoEval = COCOeval(cocoGt,cocoDt, "segm")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    out_file = out_path / f"coco_eval_segm.pkl"
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
    STYLIZED_COCO_PATH       = "stylized_coco"
    STYLIZED_COCO_OBJECTS    = "stylized_coco_objects"
    STYLIZED_COCO_BACKGROUND = "stylized_coco_background"
    OUTPUT_DIR = ROOT / "detections" / "sotr"
    STYLE_VERSIONS = ["1"]

    # SOTR models: https://github.com/easton-cau/SOTR (download weights manually and store them in SOTR Root), default output dirs are defined in configs
    MODELS = [
        ["sotr_R101",     "configs/SOTR/R101.yaml",      "SOTR_R101.pth",     "tools/output/SOTR_R101"],
        ["sotr_R101_dcn", "configs/SOTR/R_101_DCN.yaml", "SOTR_R101_DCN.pth", "output/SOTR_R101_DCN"]
    ]

    CUDA = 'CUDA_VISIBLE_DEVICES="1,2,3"'
    # switch to sotr repository
    SOTR_ROOT = ROOT / 'ext/SOTR'
    os.chdir(SOTR_ROOT)

    # create local link to coco root
    os.symlink(COCO_PATH, 'datasets/coco')

    # run detection
    for model, cfg, weights, tmp_output_dir in MODELS:

        # 1. COCO val2017
        output_dir = OUTPUT_DIR / model / 'coco/val2017'
        output_dir.mkdir(parents=True, exist_ok=True)

        os.system(f"{CUDA} python tools/train_net.py --config-file {cfg} --eval-only --num-gpus 3 MODEL.WEIGHTS ./{weights}")
        eval_and_save(f'{tmp_output_dir}/inference/coco_instances_results.json', output_dir, ANNOTATIONS)
        shutil.rmtree(tmp_output_dir)

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
                    output_dir = OUTPUT_DIR / model / dataset / style / "feature_space" / str(alpha)
                    output_dir.mkdir(parents=True, exist_ok=True)

                    os.system(f"{CUDA} python tools/train_net.py --config-file {cfg} --eval-only --num-gpus 3 MODEL.WEIGHTS ./{weights}")
                    eval_and_save(f'{tmp_output_dir}/inference/coco_instances_results.json', output_dir, ANNOTATIONS)
                    shutil.rmtree(tmp_output_dir)

                    # remove the symlink
                    os.remove(COCO_PATH / 'val2017')
                    
                # pixel space blendings
                for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:

                    # create symlink to the stylized images
                    os.symlink(DATA_ROOT / dataset / style / 'pixel_space' / str(alpha), COCO_PATH / 'val2017')

                    # run the evaluation
                    output_dir = OUTPUT_DIR / model / dataset / style / "pixel_space" / str(alpha)
                    output_dir.mkdir(parents=True, exist_ok=True)

                    os.system(f"{CUDA} python tools/train_net.py --config-file {cfg} --eval-only --num-gpus 3 MODEL.WEIGHTS ./{weights}")
                    eval_and_save(f'{tmp_output_dir}/inference/coco_instances_results.json', output_dir, ANNOTATIONS)
                    shutil.rmtree(tmp_output_dir)

                    # remove the symlink
                    os.remove(COCO_PATH / 'val2017')
        
        # bring back the original coco val2017 images
        os.rename(COCO_PATH / "val2017_bak", COCO_PATH / "val2017")

    # remove the data link
    os.remove('datasets/coco')
