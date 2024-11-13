import numpy as np

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2 import model_zoo

from detectron2.utils.visualizer import ColorMode
from scipy.ndimage import label
import random
import cv2
import matplotlib.pyplot as plt

def plot_samples(dataset_name, n=1):
    dataset_custom = DatasetCatalog.get(dataset_name)
    dataset_custom_metadata = MetadataCatalog.get(dataset_name)

    for s in random.sample(dataset_custom, n):
        img = cv2.imread(s["file_name"])
        v = Visualizer(img[:,:,::-1], metadata=dataset_custom_metadata, scale=0.5)
        v = v.draw_dataset_dict(s)
        plt.figure(figsize=(15,20))
        plt.imshow(v.get_image())
        plt.show()

def get_train_cfg(config_file_path, checkpoint_url, train_dataset_name, test_dataset_name, num_classes, device, output_dir):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file_path))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint_url)
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (test_dataset_name,)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("train",)
    cfg.DATASETS.TEST = ("val",)
    # cfg.INPUT.MIN_SIZE_TRAIN = (800,)
    # cfg.INPUT.MIN_SIZE_TEST = (800,)
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.BASE_LR = 0.001  # 0.0005 # pick a good LR
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.STEPS = []
    cfg.SOLVER.GAMMA = 0.05
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
    cfg.TEST.EVAL_PERIOD = 2000
    # Maximum number of detections to return per image during inference (default 100 is based on the limit established for the COCO dataset)
    cfg.TEST.DETECTIONS_PER_IMAGE = 128  # controls the maximum number of objects to be detected. Set it to a larger number if test images may contain >100 objects.
    cfg.SOLVER.MAX_ITER = 20000  # adjust up if val mAP is still rising, adjust down if overfit

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.DEVICE = device
    cfg.OUTPUT_DIR = output_dir
    return cfg

def cv_show(img, name):
    cv2.imshow(name, img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def on_Image(image_path, predictor):
    class_names = ["ferrite"]
    im = cv2.imread(image_path)
    outputs = predictor(im)

    # Visualize the results (optional)
    v = Visualizer(im[:, :, ::-1], metadata={'thing_classes': class_names}, scale=0.5, instance_mode=ColorMode.IMAGE_BW)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    result_image = v.get_image()[:, :, ::-1]

    # Calculate the area occupied by "ferrite"
    instances = outputs["instances"]
    masks = instances.pred_masks.cpu().numpy()
    classes = instances.pred_classes.cpu().numpy()

    # Filter out instances of "ferrite"
    ferrite_masks = masks[classes == 0]

    # Calculate the area of each ferrite mask
    ferrite_areas = [np.sum(mask) for mask in ferrite_masks]

    # Calculate the total area of the image
    image_height, image_width = im.shape[:2]
    image_area = image_height * image_width

    # Calculate the percentage of the image occupied by "ferrite"
    total_ferrite_area = sum(ferrite_areas)
    ferrite_percentage = (total_ferrite_area / image_area) * 100

    # Calculate the equivalent diameter for each ferrite grain in pixels
    equivalent_diameters_ferrite = [2 * np.sqrt(area / np.pi) for area in ferrite_areas]

    # Convert equivalent diameters from pixels to micrometers
    micrometers_per_pixel = 0.02
    equivalent_diameters_ferrite_micrometers = [d * micrometers_per_pixel for d in equivalent_diameters_ferrite]

    # Calculate the average equivalent diameter in micrometers for ferrite
    if equivalent_diameters_ferrite_micrometers:
        average_grain_size_ferrite_micrometers = np.mean(equivalent_diameters_ferrite_micrometers)
    else:
        average_grain_size_ferrite_micrometers = 0

    # Calculate the area occupied by "austenite" (background)
    combined_ferrite_mask = np.any(ferrite_masks, axis=0)
    austenite_mask = ~combined_ferrite_mask

    # Label connected regions in the austenite mask
    labeled_austenite, num_features = label(austenite_mask)
    austenite_areas = [np.sum(labeled_austenite == i) for i in range(1, num_features + 1)]

    # Calculate the percentage of the image occupied by "austenite"
    total_austenite_area = np.sum(austenite_mask)
    austenite_percentage = (total_austenite_area / image_area) * 100

    # Calculate the equivalent diameter for each austenite grain in pixels
    equivalent_diameters_austenite = [2 * np.sqrt(area / np.pi) for area in austenite_areas]

    # Convert equivalent diameters from pixels to micrometers
    equivalent_diameters_austenite_micrometers = [d * micrometers_per_pixel for d in equivalent_diameters_austenite]

    # Calculate the average equivalent diameter in micrometers for austenite
    if equivalent_diameters_austenite_micrometers:
        average_grain_size_austenite_micrometers = np.mean(equivalent_diameters_austenite_micrometers)
    else:
        average_grain_size_austenite_micrometers = 0

    print(f"The detected 'ferrite' occupies {ferrite_percentage:.2f}% of the image.")
    print(f"The average ferrite grain size is {average_grain_size_ferrite_micrometers:.2f} micrometers.")
    print(f"The detected 'austenite' occupies {austenite_percentage:.2f}% of the image.")
    print(f"The average austenite grain size is {average_grain_size_austenite_micrometers:.2f} micrometers.")
    # Optionally, save or display the result image
    cv2.imshow("Result", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def on_Video(videoPath, predictor):
    class_names = ["five", "four", "one", "three", "two"]
    cap = cv2.VideoCapture(videoPath)
    if (cap.isOpened() == False):
        print("Error opening file...")
        return

    (success, image) = cap.read()
    while success:
        predictions = predictor(image)
        v = Visualizer(image[:,:,::-1], metadata={'thing_classes':class_names}, scale=0.5 ,instance_mode = ColorMode.SEGMENTATION)
        output = v.draw_instance_predictions(predictions["instances"].to("cpu"))

        # cv2.imread("Reuslt", output.get_image()[:,:,::-1])
        # cv2.namedWindow("result", 0)
        # cv2.resizeWindow("result", 1200, 600)

        #调用电脑摄像头进行检测
        cv2.namedWindow("result", cv2.WINDOW_FREERATIO) # 设置输出框的大小，参数WINDOW_FREERATIO表示自适应大小
        cv2.imshow("result" , output.get_image()[:,:,::-1])

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        (success, image) = cap.read()

