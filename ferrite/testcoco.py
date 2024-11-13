import random

import cv2
import numpy as np

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances
from pycocotools import mask as maskUtils
register_coco_instances("train", {}, "../FeCMnAlCr/annotations/instances_train2014.json", "../FeCMnAlCr/train2014")
register_coco_instances("val", {}, "../FeCMnAlCr/annotations/instances_val2014.json", "../FeCMnAlCr/val2014")

train_metadata = MetadataCatalog.get("train")
dataset_dicts = DatasetCatalog.get("train")

def cv_show(img, name):
    cv2.imshow(name, img)
    cv2.waitKey()
    cv2.destroyAllWindows()


# 选择一个固定的样本索引
sample_index = 248  # 固定选择第一个样本
d = dataset_dicts[sample_index]

img = cv2.imread(d["file_name"])
visualizer = Visualizer(img[:, :, ::-1], metadata=train_metadata, scale=1)
vis = visualizer.draw_dataset_dict(d)
cv_show(vis.get_image()[:, :, ::-1], "1")

# 获取每个类别的掩码
masks = d['annotations']

# 初始化体积和晶粒尺寸存储字典
class_pixel_counts = {}
class_grain_sizes = {}

# 定义像素到微米的转换比例
micron_per_pixel = 0.02

# 计算每个类别的像素数和平均晶粒尺寸
for mask in masks:
    category_id = mask['category_id']
    segmentation = mask['segmentation']

    # 只处理铁素体类别的掩码
    if category_id != 0:  # 假设铁素体的类别ID为0
        continue

    # 转换为二进制掩码
    if isinstance(segmentation, list):
        # Polygon
        rles = maskUtils.frPyObjects(segmentation, img.shape[0], img.shape[1])
        rle = maskUtils.merge(rles)
        binary_mask = maskUtils.decode(rle)
    elif isinstance(segmentation['counts'], list):
        # Uncompressed RLE
        rle = maskUtils.frPyObjects([segmentation], img.shape[0], img.shape[1])
        binary_mask = maskUtils.decode(rle)
    else:
        # Compressed RLE
        binary_mask = maskUtils.decode(segmentation)

    # 如果类别不在字典中，初始化
    if category_id not in class_pixel_counts:
        class_pixel_counts[category_id] = 0
        class_grain_sizes[category_id] = []

    # 计算类别的像素数
    class_pixel_counts[category_id] += cv2.countNonZero(binary_mask)

    # 计算晶粒尺寸
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        grain_size_pixels = cv2.contourArea(contour)
        grain_size_microns = grain_size_pixels * (micron_per_pixel ** 2)
        equivalent_diameter = 2 * np.sqrt(grain_size_microns / np.pi)
        class_grain_sizes[category_id].append(equivalent_diameter)

# 计算总像素数
total_pixels = img.shape[0] * img.shape[1]

# 计算体积占比和平均晶粒尺寸
for category_id, pixel_count in class_pixel_counts.items():
    volume_ratio = pixel_count / total_pixels
    average_grain_size = sum(class_grain_sizes[category_id]) / len(class_grain_sizes[category_id])

    print(f"Category ID: {category_id}")
    print(f"Volume Ratio: {volume_ratio}")
    print(f"Average Grain Size (microns): {average_grain_size}")