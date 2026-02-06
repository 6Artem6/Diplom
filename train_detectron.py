from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog

DATASET_ROOT = "models/Segmentation-of-web-UI-elements-with-Detectron2/content"

register_coco_instances(
    "kvvb_train",
    {},
    f"{DATASET_ROOT}/train/_annotations.coco.json",
    f"{DATASET_ROOT}/train",
)

register_coco_instances(
    "kvvb_valid",
    {},
    f"{DATASET_ROOT}/valid/_annotations.coco.json",
    f"{DATASET_ROOT}/valid",
)

print("Datasets registered:", DatasetCatalog.list())

import json

with open(f"{DATASET_ROOT}/train/_annotations.coco.json") as f:
    coco = json.load(f)

print("Categories:")
for c in coco["categories"]:
    print(c["id"], c["name"])

print("Total classes:", len(coco["categories"]))
