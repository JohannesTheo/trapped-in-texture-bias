import os
import time
import json
import pickle
from pathlib import Path
import multiprocessing as mp
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.mask import decode
from pycocotools.cocoeval import COCOeval

def derive_bbox(path_list, annotation_path, counter):

    for p in path_list:
        print(p)
        if os.path.isfile(str(p / "coco_eval_bbox.pkl")):
            with counter.get_lock(): counter.value += 1
            continue

        with open(str(p / "coco_instances_results.json"),'rb') as f:
            mask_detections = json.load(f)
            
        print('convert')    
        bbox_detections = []
        for d in mask_detections:
            m = decode(d['segmentation'])
            y_indices, x_indices = np.where(m==1)

            y_min,y_max,x_min,x_max = 0,0,0,0
            if len(y_indices) > 0:
                y_min, y_max, = min(y_indices), max(y_indices)
            if len(x_indices) > 0:
                x_min, x_max, = min(x_indices), max(x_indices)

            x,y,w,h = x_min, y_min, (x_max - x_min), (y_max - y_min)
            d['bbox'] = list(map(float,[x,y,w,h]))
            del d['segmentation']
            bbox_detections.append(d)
        
        # save converted bbox results
        with open(str(p / "coco_bbox_results.json"), 'w') as f:
            json.dump(bbox_detections, f)

        # coco eval
        print('eval')
        cocoGt=COCO(annotation_path)
        cocoDt=cocoGt.loadRes(bbox_detections)

        cocoEval = COCOeval(cocoGt,cocoDt, "bbox")
        print('eval2')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        with open(str(p / "coco_eval_bbox.pkl"), 'wb') as f:
            pickle.dump(cocoEval.eval, f)

        with counter.get_lock(): counter.value += 1
    return

if __name__ == '__main__':
    # setup path variables
    ROOT = (Path(__file__).parent / "../../..").resolve()
    with open(ROOT / 'path_config.json') as f:
        path_config = json.load(f)

    DATA_ROOT = Path(path_config["DATA_ROOT"])
    if not DATA_ROOT.is_absolute():
        DATA_ROOT = (ROOT / DATA_ROOT).resolve()

    ANNOTATIONS = DATA_ROOT / "coco/annotations/instances_val2017.json"
    print(ANNOTATIONS)
    STYLIZED_COCO            = "stylized_coco"
    STYLIZED_COCO_OBJECTS    = "stylized_coco_objects"
    STYLIZED_COCO_BACKGROUND = "stylized_coco_background"
    OUTPUT_DIR     = ROOT / "detections" / "sotr"
    STYLE_VERSIONS = ["1"]

    MODELS = ["sotr_R101",  "sotr_R101_dcn"]

    path_list = []
    for model in MODELS:
        path_list.append(OUTPUT_DIR / model / 'coco/val2017')

        for dataset in [STYLIZED_COCO, STYLIZED_COCO_OBJECTS, STYLIZED_COCO_BACKGROUND]:
            for style in STYLE_VERSIONS:
                # feature space blendings
                for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                    path_list.append(OUTPUT_DIR / model / dataset / style / 'feature_space' / str(alpha))
                # pixel space blendings
                for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                    path_list.append(OUTPUT_DIR / model / dataset / style / 'pixel_space' / str(alpha))

    num_cores  = 4 #mp.cpu_count() 
    path_split = np.array_split(path_list, num_cores)
    pbar       = tqdm(total=len(path_list), desc=f"datasets")
    counter    = mp.Value('i', 0) # a shared counter to see progress
    pool = [mp.Process(target=derive_bbox, args=(path_list, ANNOTATIONS, counter), daemon=True) for path_list in path_split]
    for p in pool:
        p.start()

    while True:        
        pbar.update(counter.value - pbar.n)
        time.sleep(1)
        if counter.value == len(path_list): break

    for p in pool:
        p.join()

