
import os
import pickle
import shutil
import json
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def eval_and_save(res_file, out_path, annotations):

    out_path.mkdir(parents=True, exist_ok=True)
    shutil.move(res_file, out_path)
    if out_path.name == 'test2017':
        return
    
    # only use local evaluation for val2017
    cocoGt=COCO(annotations)
    resFile= str(out_path / 'coco_results.json')
    cocoDt=cocoGt.loadRes(resFile)
    cocoEval = COCOeval(cocoGt,cocoDt,"bbox")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    out_file = out_path / "coco_eval_bbox.pkl"
    with open(out_file, "wb") as f:
        pickle.dump(cocoEval.eval, f)

def create_image_list(img_dir, img_list_out):
    with open(img_list_out, 'w') as f:
        for img in os.listdir(img_dir):
            if img.endswith(".jpg"):
                f.write(f"{img_dir}/{img}\n")

if __name__ == '__main__':
    
    # Project ROOT to load config file
    ROOT = (Path(__file__).parent / "../../..").resolve()
    with open(ROOT / 'path_config.json') as f:
        path_config = json.load(f)

    DATA_ROOT = Path(path_config["DATA_ROOT"])
    if not DATA_ROOT.is_absolute():
        DATA_ROOT = (ROOT / DATA_ROOT).resolve()

    COCO_VAL  = "coco/val2017"
    COCO_TEST = "coco/test2017"
    ANNOTATIONS = DATA_ROOT / "coco/annotations/instances_val2017.json"
    STYLIZED_COCO            = "stylized_coco"
    STYLIZED_COCO_OBJECTS    = "stylized_coco_objects"
    STYLIZED_COCO_BACKGROUND = "stylized_coco_background"
    STYLE_VERSIONS = ["1"]
    OUTPUT_DIR = ROOT / "detections" / "darknet"

    DARKNET_ROOT = ROOT / 'ext/darknet'
    os.chdir(DARKNET_ROOT)
    
    # darknet config files    
    CFG_COCO_DATA     = DARKNET_ROOT / 'cfg/coco.data'
    CFG_COCO_DATA_BAK = DARKNET_ROOT / 'cfg/coco.data.bak'
    IMG_LIST          = DARKNET_ROOT / 'image_list.txt'

    # Save a copy of the cfg/coco.data file
    shutil.copy(CFG_COCO_DATA, CFG_COCO_DATA_BAK)
    
    # New cfg/coco.data file with adjusted validation path
    with open(DARKNET_ROOT / 'cfg/coco.data', 'w') as f:
        f.write("classes=80\n")
        f.write(f"valid={DARKNET_ROOT}/image_list.txt\n")
        f.write("names=data/coco.names\n")
        f.write("eval=coco\n")

    # NOTE: For the yolov4-csp-640 model you need to make a copy of the yolov4-csp.cfg and adjust the weight and height.
    MODELS = {
        'yolov4-p6'     : 'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-p6.weights',
        'yolov4-p5'     : 'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-p5.weights',
        'yolov4-csp'    : 'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-csp.weights',
        'yolov4-csp-640': 'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-csp.weights',
        'yolov4'        : 'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights'
        'yolov3'        : 'https://pjreddie.com/media/files/yolov3.weights'
    }

    for model, weights in MODELS.items():
        
        # Download model weights if not existing
        if not (DARKNET_ROOT / f"{'yolov4-csp' if model == 'yolov4-csp-640' else model}.weights").is_file():
            os.system(f'wget {weights}')
 
        # coco val2017
        for dataset in [COCO_VAL, COCO_TEST]:
            IMG_PATH = DATA_ROOT / dataset
            OUT_PATH = OUTPUT_DIR / model / dataset
            create_image_list(IMG_PATH, IMG_LIST)
            os.system(f"./darknet detector valid cfg/coco.data cfg/{model}.cfg {'yolov4-csp' if model == 'yolov4-csp-640' else model}.weights")
            eval_and_save('./results/coco_results.json', OUT_PATH, ANNOTATIONS)

        # stylized coco datasets
        for dataset in [STYLIZED_COCO, STYLIZED_COCO_OBJECTS, STYLIZED_COCO_BACKGROUND]:
            for style in STYLE_VERSIONS:

                # feature space blendings
                for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:

                    IMG_PATH = DATA_ROOT  / dataset / style / "feature_space" / str(alpha)
                    OUT_PATH = OUTPUT_DIR / model / dataset / style / "feature_space" / str(alpha)
                    create_image_list(IMG_PATH, IMG_LIST)
                    os.system(f"./darknet detector valid cfg/coco.data cfg/{model}.cfg {'yolov4-csp' if model == 'yolov4-csp-640' else model}.weights")
                    eval_and_save('./results/coco_results.json', OUT_PATH, ANNOTATIONS)

                # pixel space blendings
                for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                    IMG_PATH = DATA_ROOT  / dataset / style / "pixel_space" / str(alpha)
                    OUT_PATH = OUTPUT_DIR / model / dataset / style / "pixel_space" / str(alpha)
                    create_image_list(IMG_PATH, IMG_LIST)
                    os.system(f"./darknet detector valid cfg/coco.data cfg/{model}.cfg {'yolov4-csp' if model == 'yolov4-csp-640' else model}.weights")
                    eval_and_save('./results/coco_results.json', OUT_PATH, ANNOTATIONS)

    # restore the original cfg/coco.data file and remove the image_list.txt
    shutil.copy(CFG_COCO_DATA_BAK, CFG_COCO_DATA)
    os.remove(IMG_LIST)
