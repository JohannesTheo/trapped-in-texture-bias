'''
This script loads the full result of coco test2017 and creates the test-dev split based on image_info_test-dev2017.json

Note that this is only necessary if you didn't read the coco documentation and run the detection on the full test set
instead of the test-dev2017 subset.
'''
import json
from pathlib import Path
from tqdm import tqdm
import zipfile
from zipfile import ZipFile

if __name__ == '__main__':
    
    # Project ROOT to load config file
    ROOT = (Path(__file__).parent / "../../..").resolve()
    with open(ROOT / 'path_config.json') as f:
        path_config = json.load(f)

    DATA_ROOT = Path(path_config["DATA_ROOT"])
    if not DATA_ROOT.is_absolute():
        DATA_ROOT = (ROOT / DATA_ROOT).resolve()

    TEST_DEV_INFO      = DATA_ROOT / "coco/annotations/image_info_test-dev2017.json"
    DARKNET_DETECTIONS = ROOT / 'detections/darknet'

    with open(TEST_DEV_INFO, 'r') as f:
        test_dev_info = json.load(f)
    
    test_dev_imgs = test_dev_info['images']
    test_dev_ids  = [img['id'] for img in test_dev_imgs]

    for model in ['yolov4', 'yolov4-csp', 'yolov4-csp-640', 'yolov4-p5', 'yolov4-p6']:

        print("Loading results for:", model)
        with open(DARKNET_DETECTIONS / model / 'coco/test2017/coco_results.json', 'r') as f:
             coco_results  = json.load(f)

        test_dev2017 = []
        for d in tqdm(coco_results):
            if d['image_id'] in test_dev_ids:
                test_dev2017.append(d)
        
        TEST_DEV_JSON = DARKNET_DETECTIONS / model / 'coco/test2017' / f'detections_test-dev2017_{model}_results.json'
        TEST_DEV_ZIP  = DARKNET_DETECTIONS / model / 'coco/test2017' / f'detections_test-dev2017_{model}_results.zip'

        print("Saving:", TEST_DEV_JSON)
        with open(TEST_DEV_JSON, 'w') as f:
            json.dump(test_dev2017, f)

        print("Zipping to:", TEST_DEV_ZIP)
        with ZipFile(TEST_DEV_ZIP, 'w') as f:
            f.write(TEST_DEV_JSON, TEST_DEV_JSON.name, zipfile.ZIP_DEFLATED)

