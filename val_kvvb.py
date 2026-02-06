from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.DEVICE = "cpu"
cfg.DATASETS.TEST = ("kvvb_valid",)
cfg.DATALOADER.NUM_WORKERS = 0

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)

evaluator = COCOEvaluator(
    "kvvb_valid",
    cfg,
    False,
    output_dir=cfg.OUTPUT_DIR
)

val_loader = build_detection_test_loader(cfg, "kvvb_valid")

metrics = inference_on_dataset(trainer.model, val_loader, evaluator)
print(metrics)