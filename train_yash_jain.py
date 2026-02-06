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
MODELS_DIR = "./models"
DATASET_DIR = os.path.join(MODELS_DIR, "UI-Elements-Detection-Dataset")
OUTPUT_DIR = "./output_ui_detectron2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Пути к COCO JSON
# ----------------------------
TRAIN_JSON = os.path.join(DATASET_DIR, "train_coco.json")
VAL_JSON = os.path.join(DATASET_DIR, "val_coco.json")
IMG_DIR_TRAIN = os.path.join(DATASET_DIR, "train/images")
IMG_DIR_VAL = os.path.join(DATASET_DIR, "val/images")

# ----------------------------
# Конвертация YOLO аннотаций в COCO
# ----------------------------
def yolo_to_coco(yolo_folder, image_folder, categories):
    images = []
    annotations = []
    ann_id = 1
    for idx, img_file in enumerate(sorted(os.listdir(image_folder))):
        if not img_file.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        img_path = os.path.join(image_folder, img_file)
        w, h = Image.open(img_path).size
        images.append({
            "file_name": img_file,
            "height": h,
            "width": w,
            "id": idx + 1
        })

        label_file = os.path.join(yolo_folder, Path(img_file).stem + ".txt")
        if os.path.exists(label_file):
            with open(label_file) as f:
                for line in f:
                    cls, x_c, y_c, bw, bh = map(float, line.strip().split())
                    x = (x_c - bw / 2) * w
                    y = (y_c - bh / 2) * h
                    bw *= w
                    bh *= h
                    annotations.append({
                        "id": ann_id,
                        "image_id": idx + 1,
                        "category_id": int(cls),
                        "bbox": [x, y, bw, bh],
                        "area": bw * bh,
                        "iscrowd": 0
                    })
                    ann_id += 1
    coco_json = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i, "name": name} for i, name in enumerate(categories)]
    }
    return coco_json

# Получаем категории из dataset.yaml (names может быть dict: 0: "link", 1: "button", ...)
with open(os.path.join(DATASET_DIR, "dataset.yaml")) as f:
    ds_yaml = yaml.safe_load(f)
names_raw = ds_yaml["names"]
nc = ds_yaml.get("nc", len(names_raw))
# Список названий в порядке id 0, 1, ..., nc-1 (для COCO и thing_classes)
if isinstance(names_raw, dict):
    categories = [names_raw[i] for i in range(nc)]
else:
    categories = list(names_raw)[:nc]

# Конвертируем train и val
for split in ["train", "val"]:
    img_folder = os.path.join(DATASET_DIR, split, "images")
    yolo_folder = os.path.join(DATASET_DIR, split, "labels")
    coco = yolo_to_coco(yolo_folder, img_folder, categories)
    with open(os.path.join(DATASET_DIR, f"{split}_coco.json"), "w") as f:
        json.dump(coco, f)
    print(f"COCO JSON saved for {split}")

# ----------------------------
# Регистрация датасета в Detectron2 (thing_classes передаём в metadata сразу)
# ----------------------------
THING_CLASSES = categories
metadata_train = {"thing_classes": THING_CLASSES}
metadata_val = {"thing_classes": THING_CLASSES}

register_coco_instances("ui_train", metadata_train, TRAIN_JSON, IMG_DIR_TRAIN)
register_coco_instances("ui_val", metadata_val, VAL_JSON, IMG_DIR_VAL)

ui_metadata = MetadataCatalog.get("ui_train")
print("Registered UI Elements dataset with classes:", ui_metadata.thing_classes)

# ----------------------------
# Конфигурация Detectron2
# ----------------------------
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("ui_train",)
cfg.DATASETS.TEST = ("ui_val",)
# 0 — чтобы ошибка датасета показывалась в основном процессе (не терялась в воркере)
cfg.DATALOADER.NUM_WORKERS = 0
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 500
cfg.SOLVER.STEPS = []
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(THING_CLASSES)
cfg.OUTPUT_DIR = OUTPUT_DIR
cfg.MODEL.DEVICE = "cpu"  # или "cuda", если есть GPU
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Обучение
# ----------------------------
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# ----------------------------
# Валидация
# ----------------------------
evaluator = COCOEvaluator("ui_val", cfg, False, output_dir=OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, "ui_val")
metrics = inference_on_dataset(trainer.model, val_loader, evaluator)
print("Validation metrics:", metrics)

# ----------------------------
# Пример инференса
# ----------------------------
predictor = DefaultPredictor(cfg)

# Получаем путь к изображению для инференса
sample_img_path = os.path.join(IMG_DIR_VAL, os.listdir(IMG_DIR_VAL)[0])

# Загружаем само изображение как ndarray через OpenCV (тип: np.ndarray)
import cv2
sample_img = cv2.imread(sample_img_path)
if sample_img is None:
    raise FileNotFoundError(f"Could not read image at {sample_img_path}")

outputs = predictor(sample_img)
print(outputs)