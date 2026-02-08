import os
import json
from pathlib import Path
from PIL import Image
import yaml
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2 import model_zoo

# ----------------------------
# Пути
# ----------------------------
DATASET_DIR = "./ui_dataset_generator/dataset"  # COCO датасет, который создал Курсор
OUTPUT_DIR = "./models/output_ui_detectron2_generated"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_JSON = os.path.join(DATASET_DIR, "train/train_coco.json")
VAL_JSON = os.path.join(DATASET_DIR, "val/val_coco.json")
IMG_DIR_TRAIN = os.path.join(DATASET_DIR, "train/images")
IMG_DIR_VAL = os.path.join(DATASET_DIR, "val/images")

# ----------------------------
# Категории из dataset.yaml
# ----------------------------
with open(os.path.join(DATASET_DIR, "dataset.yaml")) as f:
    ds_yaml = yaml.safe_load(f)
categories = list(ds_yaml["names"])  # ['button', 'checkbox', 'input', 'link', ...]

# ----------------------------
# Регистрация датасета
# ----------------------------
register_coco_instances("ui_train", {"thing_classes": categories}, TRAIN_JSON, IMG_DIR_TRAIN)
register_coco_instances("ui_val", {"thing_classes": categories}, VAL_JSON, IMG_DIR_VAL)
ui_metadata = MetadataCatalog.get("ui_train")
print("Registered UI Elements dataset with classes:", ui_metadata.thing_classes)

# ----------------------------
# Конфигурация Detectron2
# ----------------------------
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

# ВАЖНО: путь к готовой модели от yash_jain для дообучения
cfg.MODEL.WEIGHTS = "./models/output_ui_detectron2/model_final.pth"

cfg.DATASETS.TRAIN = ("ui_train",)
cfg.DATASETS.TEST = ("ui_val",)

cfg.DATALOADER.NUM_WORKERS = 0
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.00025

# Настройка дообучения — небольшое количество итераций
cfg.SOLVER.MAX_ITER = 1000   # для теста, потом можно увеличить
cfg.SOLVER.STEPS = []         # без уменьшения lr по шагам
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(categories)

cfg.OUTPUT_DIR = OUTPUT_DIR
cfg.MODEL.DEVICE = "cpu"  # или "cuda"

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Обучение
# ----------------------------
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=True)  # resume=True, чтобы дообучать
trainer.train()

# ----------------------------
# Валидация
# ----------------------------
evaluator = COCOEvaluator("ui_val", cfg, False, output_dir=OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, "ui_val")
metrics = inference_on_dataset(trainer.model, val_loader, evaluator)
print("Validation metrics:", metrics)

# ----------------------------
# Инференс на примере
# ----------------------------
predictor = DefaultPredictor(cfg)
import cv2
sample_img_path = os.path.join(IMG_DIR_VAL, os.listdir(IMG_DIR_VAL)[0])
sample_img = cv2.imread(sample_img_path)
outputs = predictor(sample_img)
print(outputs)
