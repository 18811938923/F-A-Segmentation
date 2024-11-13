# remove the existing config folders, else the training files will append to the existing ones
import os
import shutil

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

from ferrite.CocoTrainer import CocoTrainer
import optuna

import torch
import logging
import json
from detectron2.data.datasets import register_coco_instances

register_coco_instances("train", {}, "../FeCMnAlCr/annotations/instances_train2014.json", "../FeCMnAlCr/train2014")
register_coco_instances("val", {}, "../FeCMnAlCr/annotations/instances_val2014.json", "../FeCMnAlCr/val2014")
torch.cuda.empty_cache()
cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("train",)
cfg.DATASETS.TEST = ("val",)
# cfg.INPUT.MIN_SIZE_TRAIN = (800,)
# cfg.INPUT.MIN_SIZE_TEST = (800,)
cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.WEIGHTS = "model_final.pth"  # Let training initialize from model zoo
cfg.SOLVER.BASE_LR = 0.001 # 0.0005 # pick a good LR
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.STEPS = []
cfg.SOLVER.GAMMA = 0.05
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
cfg.TEST.EVAL_PERIOD = 100
# Maximum number of detections to return per image during inference (default 100 is based on the limit established for the COCO dataset)
cfg.TEST.DETECTIONS_PER_IMAGE = 128 # controls the maximum number of objects to be detected. Set it to a larger number if test images may contain >100 objects.
cfg.SOLVER.MAX_ITER = 5000  #adjust up if val mAP is still rising, adjust down if overfit
cfg.OUTPUT_DIR = "./output_test/"
# train model
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# trainer = DefaultTrainer(cfg)
trainer = CocoTrainer(cfg)
trainer.resume_or_load(resume=True)
trainer.train()