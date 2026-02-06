import os
import copy
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.data import build_detection_train_loader
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils

DATASET_ROOT = "models/Segmentation-of-web-UI-elements-with-Detectron2/content"
OUTPUT_DIR = "./output_kvvb_aug"

# ---------------------------
# Регистрируем COCO датасеты
# ---------------------------
register_coco_instances(
    "kvvb_train",
    {},
    f"{DATASET_ROOT}/train/_annotations.coco.json",
    f"{DATASET_ROOT}/train",
)

register_coco_instances(
    "kvvb_valid",
    {},
    f"{DATASET_ROOT}/val/_annotations.coco.json",
    f"{DATASET_ROOT}/val",
)

# ---------------------------
# Конфиг Detectron2
# ---------------------------
cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
)
cfg.MODEL.DEVICE = "cpu"

cfg.DATASETS.TRAIN = ("kvvb_train",)
cfg.DATASETS.TEST = ("kvvb_valid",)

# ---------------------------
# DataLoader и аугментации
# ---------------------------
cfg.DATALOADER.NUM_WORKERS = 2

# -----------------------------------------
# Кастомный mapper с аугментациями
# -----------------------------------------
def custom_mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    aug_input = T.AugInput(image)

    transform_list = [
        T.RandomFlip(horizontal=True, vertical=False),
        T.RandomBrightness(0.9, 1.1),
        T.RandomContrast(0.9, 1.1),
        T.ResizeShortestEdge(short_edge_length=(800, 800), max_size=1333),
    ]

    transforms = [t.get_transform(image) for t in transform_list]
    for t in transforms:
        image = t.apply_image(image)

    image = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
    
    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[1:])
        for obj in dataset_dict.pop("annotations")
    ]
    dataset_dict["image"] = image
    dataset_dict["instances"] = utils.annotations_to_instances(annos, image.shape[1:])
    return dataset_dict

# -----------------------------------------
# Тренер с кастомным mapper
# -----------------------------------------
class TrainerWithAug(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)

# ---------------------------
# Solver / Hyperparams
# ---------------------------
cfg.SOLVER.IMS_PER_BATCH = 1  # количество изображений на шаг
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 1500     # можно увеличить для лучших результатов
cfg.SOLVER.STEPS = []

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256  # больше примеров на изображение
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 12            # количество классов

# ---------------------------
# Input image sizes (для мелких объектов)
# ---------------------------
cfg.INPUT.MIN_SIZE_TRAIN = (1024,)
cfg.INPUT.MAX_SIZE_TRAIN = 1333
cfg.INPUT.MIN_SIZE_TEST = 1024
cfg.INPUT.MAX_SIZE_TEST = 1333

cfg.OUTPUT_DIR = OUTPUT_DIR
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# ---------------------------
# Тренировка
# ---------------------------
def main():
    trainer = TrainerWithAug(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == "__main__":
    main()