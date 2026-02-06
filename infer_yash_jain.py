import os
import cv2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
from pathlib import Path

# ----------------------------
# Пути
# ----------------------------
created_dir = Path("./predictions")
created_dir.mkdir(parents=True, exist_ok=True)
DATASET_NAME = "ui_train"  # важно: чтобы подтянуть thing_classes
MODEL_PATH = Path("./models/output_ui_detectron2/model_final.pth")
INPUT_IMAGE = Path("./models/UI-Elements-Detection-Dataset/val/images/e_commerce_www.target.com_1729629595.png")
OUTPUT_IMAGE = Path("./predictions/e_commerce_www.target.com_1729629595.png")
THING_CLASSES = [
    "link",
    "button",
    "input",
    "select",
    "textarea",
    "label",
    "checkbox",
    "radio",
    "dropdown",
    "slider",
    "toggle",
    "menu_item",
    "clickable",
    "icon",
    "image",
    "text",
]

# ----------------------------
# Конфиг
# ----------------------------
cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
)
cfg.MODEL.WEIGHTS = str(MODEL_PATH)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(THING_CLASSES)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
cfg.MODEL.DEVICE = "cpu"  # mac

predictor = DefaultPredictor(cfg)

# ----------------------------
# Инференс
# ----------------------------
image = cv2.imread(str(INPUT_IMAGE))
outputs = predictor(image)

metadata = MetadataCatalog.get("ui_infer")
metadata.set(thing_classes=THING_CLASSES)

v = Visualizer(
    image[:, :, ::-1],
    metadata=metadata,
    scale=1.0,
    instance_mode=ColorMode.IMAGE,
)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

cv2.imwrite(str(OUTPUT_IMAGE), out.get_image()[:, :, ::-1])
print("Saved to", OUTPUT_IMAGE)