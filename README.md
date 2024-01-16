# Trapped in texture bias? A large scale comparison of deep instance segmentation [ECCV 2022]

![stylized coco](./imgs/dataset_introduction.png)

This is the official code release for the paper *Trapped in texture bias? A large scale comparison of deep instance segmentation*, accepted at ECCV 2022.

## Overview

The code release consists of three parts:

1. An object-centric version of Stylized COCO
2. Detection and evaluation code for 61 models covering *Cascade* and *Mask R-CNN*, *Swin Transformer*, *BMask*, *YOLACT(++)*, *DETR*, *BCNet*, *SOTR* and *SOLOv2*.
3. Data analysis and visualization (to reproduce results and figures)

### Directory structure

- `./datasets` - Stylized COCO will be created in this directory and the detection code assumes it to be there (configurable in `path_config.json`).
- `./detections` - Detection and evaluation files will be saved in this directory (configurable in `path_config.json`). NOTE: running 61 models on 60 copies of COCO val2017 will result in ~1TB of uncompressed data. If you are interested in custom evaluation and additional analysis please reach out to us.
- `./plots` - Figures from the analysis will be saved here.
- `./ext` - A central point to install detection frameworks and other required dependencies (we will release detailed installation instructions over the next weeks).
- `./src` - Source code to create datasets, run detections, evaluation, analysis and visualization

## 1. An object-centric version of Stylized COCO

We use a [slightly modified version](https://github.com/JohannesTheo/stylize-datasets) of [stylize-datasets](https://github.com/bethgelab/stylize-datasets) to create Stylized COCO. It allows us to fix the choice of style images ([pull request pending](https://github.com/bethgelab/stylize-datasets/pull/18)). The random but fixed style maps can be found in `./datasets/coco_style_maps/`. We use `coco_style_map_1.json` in all of our experiments.

The code in `./src/datasets/` can then be used to create the blending sequences of Stylized COCO and the object-centric versions Stylized Objects and Stylized Background. Note that this will produced 60 copies of the COCO val2017 subset.

## 2. Detection and evaluation

To run the detection on Stylized COCO and the object-centric variants, every framework has to be installed in `./ext`. Please refer to the corresponding projects respectively. The code in `./src/detection` runs the actual detection and evaluation per framework on all 61 datasets (including the original val2017 subset).

Detection results and evaluation files will be saved in `./detections`. Note that we use the [official coco API](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools) for consistency (some projects such as detectron provide optimized but custom coco evaluation code)

## 3. Data analysis and visualization

The code for the analysis and visualization can be found in `./src/analysis`. This is the most *researchy* part of our code release. Please reach out if something is unclear.
