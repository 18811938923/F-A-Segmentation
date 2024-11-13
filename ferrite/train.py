from detectron2.utils.logger import setup_logger

setup_logger()
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer

import os
import pickle

from utils import *

# config_file_path = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
# checkpoint_url = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"


'''
instance segmentation
'''
config_file_path = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
checkpoint_url = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"

output_dir = "./output_test11"

num_classes = 1
class_names = ["ferrite"]

device = "cuda"

train_dataset_name = "LP_train"
train_images_path = "../FeCMnAlCr/train2014"
train_json_annot_path = "../FeCMnAlCr/annotations/instances_train2014.json"

test_dataset_name = "LP_test"
test_images_path = "../FeCMnAlCr/val2014"
test_json_annot_path = "../FeCMnAlCr/annotations/instances_val2014.json"

cfg_save_path = "OD_cfg.pickle"


###########################################################
# 注册训练集
register_coco_instances("LP_train", {},train_json_annot_path, train_images_path)
MetadataCatalog.get("LP_train").set(thing_classes = class_names,
                                    evaluator_type = 'coco',
                                    json_file=train_json_annot_path,
                                    image_root=train_images_path)


# 注册测试集
register_coco_instances("LP_test", {}, test_json_annot_path, test_images_path)
MetadataCatalog.get("LP_test").set(thing_classes = class_names,
                                    evaluator_type = 'coco',
                                    json_file=test_json_annot_path,
                                    image_root=test_images_path)
# plot_samples(dataset_name=train_dataset_name, n=3)

#####################################################
def main():
    cfg = get_train_cfg(config_file_path, checkpoint_url, train_dataset_name, test_dataset_name, num_classes, device, output_dir)

    with open(cfg_save_path, 'wb') as f:
        pickle.dump(cfg, f, protocol=pickle.HIGHEST_PROTOCOL)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)

    trainer.train()

if __name__ == '__main__':
    main()
