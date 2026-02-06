"""
Inference для обученной модели Detectron2 (Faster R-CNN R50 FPN).
Только inference: без обучения, без валидации, без регистрации датасетов.

Использование:
  python inference_kvvb.py --image path/to/image.png
  python inference_kvvb.py --image path/to/image.png --weights output_kvvb/model_final.pth

Структура допускает вынос в FastAPI: импорт predict(image_path) -> list[dict].
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, List

# Фиксированный список классов (индекс 0..11). Совпадает с датасетом kvvb.
THING_CLASSES: tuple[str, ...] = (
    "contactsPhone",
    "contactsSubtitle",
    "contactsSendformButton",
    "contactsEmail",
    "contactsTitle",
    "contactsAddress",
    "contactsEmailFormInput",
    "contactsMessageFormInput",
    "contactsSocialButtons",
    "contactsNameFormInput",
    "contactsSubjectFormInput",
    "contactsDescription",
)
NUM_CLASSES = len(THING_CLASSES)

DEFAULT_WEIGHTS_DIR = "/app/models/output_kvvb"
DEFAULT_WEIGHTS_FILE = "model_final.pth"
SCORE_THRESH_TEST = 0.5


def _build_cfg(weights_path: str) -> Any:
    """Собрать конфиг для inference. Не регистрирует датасеты, не трогает train/val."""
    from detectron2.config import get_cfg
    from detectron2 import model_zoo

    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESH_TEST
    return cfg


def _get_predictor(weights_path: str) -> Any:
    """DefaultPredictor с загруженными весами. Один раз на процесс."""
    from detectron2.engine import DefaultPredictor

    cfg = _build_cfg(weights_path)
    return DefaultPredictor(cfg)


def predict(image_path: str, weights_path: str | None = None) -> List[dict[str, Any]]:
    """
    Inference по одному изображению. Возвращает список предсказаний.

    Args:
        image_path: путь к изображению
        weights_path: путь к model_final.pth; если None — output_kvvb/model_final.pth

    Returns:
        [{"class": "<class_name>", "bbox": [x1, y1, x2, y2], "score": float}, ...]
    """
    if weights_path is None:
        weights_path = os.path.join(DEFAULT_WEIGHTS_DIR, DEFAULT_WEIGHTS_FILE)
    if not Path(weights_path).exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    predictor = _get_predictor(weights_path)
    try:
        import cv2
        im = cv2.imread(image_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load image: {image_path}") from e
    if im is None:
        raise RuntimeError(f"cv2.imread returned None: {image_path}")

    outputs = predictor(im)
    instances = outputs.get("instances")
    if instances is None:
        return []

    instances = instances.to("cpu")
    pred_boxes = instances.pred_boxes
    pred_classes = instances.pred_classes
    scores = instances.scores

    result: List[dict[str, Any]] = []
    for k in range(len(pred_classes)):
        cls_idx = int(pred_classes[k])
        class_name = THING_CLASSES[cls_idx] if 0 <= cls_idx < NUM_CLASSES else f"class_{cls_idx}"
        box = pred_boxes[k].tensor[0]
        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        score = float(scores[k])
        result.append({
            "class": class_name,
            "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
            "score": round(score, 4),
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Detectron2 inference (Faster R-CNN, kvvb).")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument(
        "--weights",
        default=os.path.join(DEFAULT_WEIGHTS_DIR, DEFAULT_WEIGHTS_FILE),
        help=f"Path to model_final.pth (default: {DEFAULT_WEIGHTS_DIR}/{DEFAULT_WEIGHTS_FILE})",
    )
    parser.add_argument("--indent", type=int, default=2, help="JSON indent")
    args = parser.parse_args()

    out = predict(args.image, weights_path=args.weights)
    print(json.dumps(out, indent=args.indent))


if __name__ == "__main__":
    main()
