import torch
from detectron2 import model_zoo, config
from detectron2.data import build_detection_test_loader, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.modeling import build_model
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.evaluation import COCOEvaluator
from detectron2.evaluation.coco_evaluation import _evaluate_box_proposals, _evaluate_predictions_on_coco
from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger
from detectron2.utils.file_io import PathManager

setup_logger() # enable logging if desired
from tqdm import tqdm
import os
import json
import itertools
import pickle
from pathlib import Path

class SaveAllCOCOEvaluator(COCOEvaluator):
    """
    A modified version of the detectron2 Coco Evaluator that does not summarize results.

    1. It saves all results from pycocotools COCOeval    -> _eval_predictions
    2. It saves all results from proposal box evaluation -> _eval_box_proposals
    """
    def _eval_predictions(self, predictions, img_ids=None):
        """
        Evaluate predictions. Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for COCO format ...")
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))
        tasks = self._tasks or self._tasks_from_predictions(coco_results)

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            dataset_id_to_contiguous_id = self._metadata.thing_dataset_id_to_contiguous_id
            all_contiguous_ids = list(dataset_id_to_contiguous_id.values())
            num_classes = len(all_contiguous_ids)
            assert min(all_contiguous_ids) == 0 and max(all_contiguous_ids) == num_classes - 1

            reverse_id_mapping = {v: k for k, v in dataset_id_to_contiguous_id.items()}
            for result in coco_results:
                category_id = result["category_id"]
                assert category_id < num_classes, (
                    f"A prediction has class={category_id}, "
                    f"but the dataset only has {num_classes} classes and "
                    f"predicted class id should be in [0, {num_classes - 1}]."
                )
                result["category_id"] = reverse_id_mapping[category_id]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info(
            "Evaluating predictions with {} COCO API...".format(
                "unofficial" if self._use_fast_impl else "official"
            )
        )
        for task in sorted(tasks):
            assert task in {"bbox", "segm", "keypoints"}, f"Got unknown task: {task}!"
            coco_eval = (
                _evaluate_predictions_on_coco(
                    self._coco_api,
                    coco_results,
                    task,
                    kpt_oks_sigmas=self._kpt_oks_sigmas,
                    use_fast_impl=self._use_fast_impl,
                    img_ids=img_ids,
                )
                if len(coco_results) > 0
                else None  # cocoapi does not handle empty results very well
            )
            # new code:
            if self._output_dir:
                file_path = os.path.join(self._output_dir, f"coco_eval_{task}.pkl")
                self._logger.info("Saving coco_eval to {}".format(file_path))
                with PathManager.open(file_path, "wb") as f:
                    pickle.dump(coco_eval.eval, f)

            # original code:
            #res = self._derive_coco_results(
            #    coco_eval, task, class_names=self._metadata.get("thing_classes")
            #)
            #self._results[task] = res
    
    def _eval_box_proposals(self, predictions):
        """
        Evaluate the box proposals in predictions.
        Fill self._results with the metrics for "box_proposals" task.
        """
        if self._output_dir:
            # Saving generated box proposals to file.
            # Predicted box_proposals are in XYXY_ABS mode.
            bbox_mode = BoxMode.XYXY_ABS.value
            ids, boxes, objectness_logits = [], [], []
            for prediction in predictions:
                ids.append(prediction["image_id"])
                boxes.append(prediction["proposals"].proposal_boxes.tensor.numpy())
                objectness_logits.append(prediction["proposals"].objectness_logits.numpy())

            proposal_data = {
                "boxes": boxes,
                "objectness_logits": objectness_logits,
                "ids": ids,
                "bbox_mode": bbox_mode,
            }
            with PathManager.open(os.path.join(self._output_dir, "box_proposals.pkl"), "wb") as f:
                pickle.dump(proposal_data, f)

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating bbox proposals ...")
        res = {}
        areas = {"all": "", "small": "s", "medium": "m", "large": "l"}
        for limit in [100, 1000]:
            for area, suffix in areas.items():
                stats = _evaluate_box_proposals(predictions, self._coco_api, area=area, limit=limit)
                ar = "AR{}@{:d}".format(suffix, limit)
                re = "RE{}@{:d}".format(suffix, limit)
                res[ar] = float(stats["ar"].item() * 100)
                res[re] = [float(r * 100) for r in stats["recalls"]]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, f"bbox_proposals_eval.json")
            self._logger.info("Saving proposals evaluation to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(res))
                f.flush()
        #self._logger.info("Proposal metrics: \n" + create_small_table(res))
        self._results["box_proposals"] = res

def inference_on_dataset(model, data_loader, proposal_outputs, instance_evaluator, proposal_evaluator):
    """
    A modified version of detectron2/evaluation/evaluator.py inference_on_dataset()

    1. It uses tqdm instead of pytorch logging
    2. It processes RPN 'proposals' in addition to ROI head 'instances'
        - manual postprocessing is applied to the proposals
        - a separate Evaluator is used for the proposals
    """
    instance_evaluator.reset()
    proposal_evaluator.reset()

    with torch.no_grad():
        for idx, inputs in enumerate(tqdm(data_loader)):
            
            # get ROI Head outputs -> 'instances'
            outputs = model(inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            # manual postprocessing for proposal outputs
            proposals = []
            for proposal, input in zip(proposal_outputs[idx], inputs):
                proposals.append({'proposals': detector_postprocess(proposal, input['height'], input['width'])})

            instance_evaluator.process(inputs, outputs)
            proposal_evaluator.process(inputs, proposals)

    instance_results = instance_evaluator.evaluate()
    proposal_results = proposal_evaluator.evaluate()
    
    return instance_results, proposal_results

# pytorch forward hook to get RPN proposals
def get_proposals(self, input, output):
    raw_proposals.append(output[0])

def detectron_detection(model_cfg, weights_url, data_cfg_list):

    # init the model
    if model_cfg.endswith(".yaml"):
        cfg = model_zoo.get_config(model_cfg)
        model = model_zoo.get(model_cfg, trained=True)
    elif model_cfg.endswith(".yaml_contrastive"):
        # these configs are not in the model zoo, we have to do everything manually 
        model_cfg = model_cfg.strip("_contrastive")
        cfg = config.get_cfg()
        cfg.merge_from_file(model_cfg)
        cfg.MODEL.WEIGHTS = weights_url
        cfg.MODEL.DEVICE = 'cpu' if not torch.cuda.is_available() else 'cuda' # SyncBatchNorm expected input tensor to be on GPU though
        model = build_model(cfg)
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    elif model_cfg.endswith(".py"):
        # new_baseline weights not in model_zoo (yet), we do it manually
        cfg = model_zoo.get_config(model_cfg)
        model = model_zoo.get(model_cfg, trained=False, device='cuda')
        DetectionCheckpointer(model).load(weights_url)
    else:
        raise Exception(f"ERROR: {model_cfg} not supported")

    model.eval() # set to test mode
    model.proposal_generator.register_forward_hook(get_proposals)

    global raw_proposals

    # run inference per dataset
    for data_cfg in data_cfg_list:
        dataset_name = data_cfg["name"]
        print(f"DATASET: {dataset_name}")
        annotations  = data_cfg["annotation"]
        image_dir    = data_cfg["image_dir"]
        output_dir   = data_cfg["output_dir"]

        raw_proposals  = [] # reset proposals

        # register the dataset (only once per session)
        if dataset_name not in DatasetCatalog:
            register_coco_instances(dataset_name, {}, annotations, image_dir)

        # dataset loader (loader applies the preprocessing defined in cfg):
        if model_cfg.endswith(".yaml"):
            data_loader = build_detection_test_loader(cfg, dataset_name)
        elif model_cfg.endswith(".py"):
            data_cfg = cfg['dataloader']['test']
            data_cfg['dataset']['names'] =  dataset_name
            data_loader = config.instantiate(data_cfg)

        # create evaluators for 'instances' and 'proposals'
        instance_evaluator = SaveAllCOCOEvaluator(dataset_name, tasks=("bbox", "segm"), output_dir=output_dir, use_fast_impl=False)
        proposal_evaluator = SaveAllCOCOEvaluator(dataset_name, tasks=("bbox"),         output_dir=output_dir)

        results = inference_on_dataset(model, data_loader, raw_proposals, instance_evaluator, proposal_evaluator)

        # we only keep the instance results in coco format to save disk space as we only run official coco eval anyways.
        os.remove(output_dir / "instances_predictions.pth")

        # free memory
        del data_loader, instance_evaluator, proposal_evaluator, results 

    # free gpu
    del model
    torch.cuda.empty_cache()

if __name__ == '__main__':
    # setup path variables
    ROOT = Path(__file__).parent / "../../.."
    with open(ROOT / 'path_config.json') as f:
        path_config = json.load(f)

    DATA_ROOT = Path(path_config["DATA_ROOT"])
    if not DATA_ROOT.is_absolute():
        DATA_ROOT = (ROOT / DATA_ROOT).resolve()

    COCO_PATH   = DATA_ROOT / "coco"
    COCO_IMG    = COCO_PATH / "val2017"
    ANNOTATIONS = COCO_PATH / "annotations/instances_val2017.json"

    STYLIZED_COCO_PATH       = "stylized_coco"
    STYLIZED_COCO_OBJECTS    = "stylized_coco_objects"
    STYLIZED_COCO_BACKGROUND = "stylized_coco_background"
    OUTPUT_DIR = ROOT / "detections" / "detectron"

    # standard baselines
    BASELINES = {
        "config_path": "COCO-InstanceSegmentation", 
        "mask_rcnn_R_50_C4_1x" :  None, # weights are in the config
        "mask_rcnn_R_50_DC5_1x":  None,
        "mask_rcnn_R_50_FPN_1x":  None,
        "mask_rcnn_R_50_C4_3x":   None,
        "mask_rcnn_R_50_DC5_3x":  None,
        "mask_rcnn_R_50_FPN_3x":  None,
        "mask_rcnn_R_101_C4_3x":  None,
        "mask_rcnn_R_101_DC5_3x": None,
        "mask_rcnn_R_101_FPN_3x": None,
        "mask_rcnn_X_101_32x8d_FPN_3x": None
    }
    # new baselines trained with large scale jitter augmentation (and longer)
    NEW_BASELINES = {
        "config_path": "new_baselines",
        "S3_PREFIX" :  "https://dl.fbaipublicfiles.com/detectron2/", # from detectron2 model_zoo.py
        "mask_rcnn_R_50_FPN_100ep_LSJ":  "42047764/model_final_bb69de.pkl",
        "mask_rcnn_R_50_FPN_200ep_LSJ":  "42047638/model_final_89a8d3.pkl",
        "mask_rcnn_R_50_FPN_400ep_LSJ":  "42019571/model_final_14d201.pkl",
        "mask_rcnn_R_101_FPN_100ep_LSJ": "42025812/model_final_4f7b58.pkl",
        "mask_rcnn_R_101_FPN_200ep_LSJ": "42131867/model_final_0bb7ae.pkl",
        "mask_rcnn_R_101_FPN_400ep_LSJ": "42073830/model_final_f96b26.pkl", 
        "mask_rcnn_regnetx_4gf_dds_FPN_100ep_LSJ": "42047771/model_final_b7fbab.pkl",
        "mask_rcnn_regnetx_4gf_dds_FPN_200ep_LSJ": "42132721/model_final_5d87c1.pkl",
        "mask_rcnn_regnetx_4gf_dds_FPN_400ep_LSJ": "42025447/model_final_f1362d.pkl",
        "mask_rcnn_regnety_4gf_dds_FPN_100ep_LSJ": "42047784/model_final_6ba57e.pkl",
        "mask_rcnn_regnety_4gf_dds_FPN_200ep_LSJ": "42047642/model_final_27b9c1.pkl",
        "mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ": "42045954/model_final_ef3a80.pkl"
    }
    OTHER_SETTINGS = {
            "config_path": "Misc",
            "cascade_mask_rcnn_R_50_FPN_1x": None, # weights are in the config
            "cascade_mask_rcnn_R_50_FPN_3x": None,
            "mask_rcnn_R_50_FPN_1x_dconv_c3-c5": None,
            "mask_rcnn_R_50_FPN_3x_dconv_c3-c5": None,
            "panoptic_fpn_R_101_dconv_cascade_gn_3x": None,
            "cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv": None
            }
    # sls models pre-trained with contrastive learning (from PyContrast: https://github.com/HobbitLong/PyContrast/blob/master/pycontrast/detection/README.md)
    # dict k,v is {model_name: config}, weights are model_name.pth
    PYCONTRAST = {
        "config_path":  ROOT / "ext" / "PyContrast/pycontrast/detection/configs",
        "weights_path": ROOT / "ext" / "PyContrast/pycontrast/detection/",
        "R_50_FPN_Random_1x":     "R_50_FPN_1x_rand.yaml",
        "R_50_FPN_Random_2x":     "R_50_FPN_2x_rand.yaml",
        "R_50_FPN_Random_6x":     "R_50_FPN_6x_rand.yaml",
        "R_50_FPN_Supervised_1x": "R_50_FPN_1x.yaml",
        "R_50_FPN_Supervised_2x": "R_50_FPN_2x.yaml",
        "R_50_FPN_Supervised_6x": "R_50_FPN_6x.yaml",
        "R_50_FPN_InstDis_1x":    "R_50_FPN_1x_infomin.yaml",
        "R_50_FPN_InstDis_2x":    "R_50_FPN_2x_infomin.yaml",
        "R_50_FPN_PIRL_1x":       "R_50_FPN_1x_infomin.yaml",
        "R_50_FPN_PIRL_2x":       "R_50_FPN_2x_infomin.yaml",
        "R_50_FPN_MoCo_v1_1x":    "R_50_FPN_1x_infomin.yaml",
        "R_50_FPN_MoCo_v1_2x":    "R_50_FPN_2x_infomin.yaml",
        "R_50_FPN_MoCo_v2_2x":    "R_50_FPN_2x_infomin.yaml",
        "R_50_FPN_InfoMin_1x":    "R_50_FPN_1x_infomin.yaml",
        "R_50_FPN_InfoMin_2x":    "R_50_FPN_2x_infomin.yaml",
        "R_50_FPN_InfoMin_6x":    "R_50_FPN_6x_infomin.yaml"
    }

    # model to use in this run
    MODELS = [
        #"mask_rcnn_R_50_C4_1x",
        #"mask_rcnn_R_50_DC5_1x",
        #"mask_rcnn_R_50_FPN_1x",
        #"mask_rcnn_R_50_C4_3x",
        #"mask_rcnn_R_50_DC5_3x",
        #"mask_rcnn_R_50_FPN_3x",
        #"mask_rcnn_R_101_C4_3x",
        #"mask_rcnn_R_101_DC5_3x",
        #"mask_rcnn_R_101_FPN_3x",
        #"mask_rcnn_X_101_32x8d_FPN_3x",
        #"mask_rcnn_R_50_FPN_100ep_LSJ",
        #"mask_rcnn_R_50_FPN_200ep_LSJ",
        #"mask_rcnn_R_50_FPN_400ep_LSJ",
        #"mask_rcnn_R_101_FPN_100ep_LSJ",
        #"mask_rcnn_R_101_FPN_200ep_LSJ",
        #"mask_rcnn_R_101_FPN_400ep_LSJ",
        #"mask_rcnn_regnetx_4gf_dds_FPN_100ep_LSJ",
        #"mask_rcnn_regnetx_4gf_dds_FPN_200ep_LSJ",
        #"mask_rcnn_regnetx_4gf_dds_FPN_400ep_LSJ",
        #"mask_rcnn_regnety_4gf_dds_FPN_100ep_LSJ",
        #"mask_rcnn_regnety_4gf_dds_FPN_200ep_LSJ",
        #"mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ",
        "cascade_mask_rcnn_R_50_FPN_1x",
        "cascade_mask_rcnn_R_50_FPN_3x",
        "mask_rcnn_R_50_FPN_1x_dconv_c3-c5",
        "mask_rcnn_R_50_FPN_3x_dconv_c3-c5",
        "panoptic_fpn_R_101_dconv_cascade_gn_3x",
        "cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv",
        #"R_50_FPN_Random_1x",
        #"R_50_FPN_Random_2x",
        #"R_50_FPN_Random_6x",
        #"R_50_FPN_Supervised_1x",
        #"R_50_FPN_Supervised_2x",
        #"R_50_FPN_Supervised_6x",
        #"R_50_FPN_InstDis_1x",
        #"R_50_FPN_InstDis_2x",
        #"R_50_FPN_PIRL_1x",
        #"R_50_FPN_PIRL_2x",
        #"R_50_FPN_MoCo_v1_1x",
        #"R_50_FPN_MoCo_v1_2x",
        #"R_50_FPN_MoCo_v2_2x",
        #"R_50_FPN_InfoMin_1x",
        #"R_50_FPN_InfoMin_2x",
        #"R_50_FPN_InfoMin_6x"
        ]

    STYLE_VERSIONS = ["1"]

    for model in MODELS:
        print(f"MODEL: {model}")
        if model in BASELINES:
            model_cfg   = f"{BASELINES['config_path']}/{model}.yaml"
            weights_url = None
        elif model in OTHER_SETTINGS:
            model_cfg   = f"{OTHER_SETTINGS['config_path']}/{model}.yaml"
            weights_url = None
        elif model in NEW_BASELINES:
            model_cfg   = f"{NEW_BASELINES['config_path']}/{model}.py"
            weights_url = f"{NEW_BASELINES['S3_PREFIX']}{NEW_BASELINES['config_path']}/{model}/{NEW_BASELINES[model]}"
        elif model in PYCONTRAST:
            model_cfg   = f"{PYCONTRAST['config_path']}/{PYCONTRAST[model]}_contrastive"
            weights_url = f"{PYCONTRAST['weights_path']}/{model}.pth"
        else:
            raise Exception(f"ERROR: {model} not in any config list")

        data_cfg_list = []
        data_cfg = {
            "name" : "coco_val2017",
            "annotation": ANNOTATIONS,
            "image_dir":  COCO_IMG,
            "output_dir": OUTPUT_DIR / model / "coco" / "val2017"
        }
        data_cfg_list.append(data_cfg)
        for dataset in [STYLIZED_COCO_PATH, STYLIZED_COCO_OBJECTS, STYLIZED_COCO_BACKGROUND]:
            for style in STYLE_VERSIONS:

                # feature space blendings
                for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                    data_cfg = {
                        "name" : f"{dataset}_{style}_fs_{alpha}",
                        "annotation": ANNOTATIONS,
                        "image_dir":  DATA_ROOT  / dataset / style / "feature_space" / str(alpha),
                        "output_dir": OUTPUT_DIR / model / dataset / style / "feature_space" / str(alpha)
                    }
                    data_cfg_list.append(data_cfg)

                # pixel space blendings
                for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                    data_cfg = {
                        "name" : f"{dataset}_{style}_ps_{alpha}",
                        "annotation": ANNOTATIONS,
                        "image_dir":  DATA_ROOT  / dataset / style / "pixel_space" / str(alpha),
                        "output_dir": OUTPUT_DIR / model / dataset / style / "pixel_space" / str(alpha)
                    }
                    data_cfg_list.append(data_cfg)

        detectron_detection(model_cfg, weights_url, data_cfg_list)
