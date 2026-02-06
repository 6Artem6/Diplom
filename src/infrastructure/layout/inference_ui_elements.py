"""
Detectron2 inference: UI-Elements (Yash Jain / output_ui_detectron2).

Perception-слой: только bbox + класс + score в координатах изображения.
Один формат выхода (стабильный JSON), одна система координат, без рескейлов.
Визуализация — только для ручного контроля, в пайплайн не передаётся.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, List

# Классы датасета UI-Elements-Detection-Dataset (Yash Jain), порядок id 0..15
THING_CLASSES: tuple[str, ...] = (
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
)
NUM_CLASSES = len(THING_CLASSES)

# Пути к весам: сначала env, затем /app/models (Docker), затем локальный output_ui_detectron2
DEFAULT_WEIGHTS_ENV = "ATOMS_V2_UI_ELEMENTS_WEIGHTS"
DEFAULT_WEIGHTS_DIR_DOCKER = "/app/models/output_ui_detectron2"
DEFAULT_WEIGHTS_FILE = "model_final.pth"
SCORE_THRESH_TEST = 0.5


def _build_cfg(weights_path: str) -> Any:
    """Конфиг Faster R-CNN, NUM_CLASSES=16, без регистрации датасетов."""
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
    from detectron2.engine import DefaultPredictor
    cfg = _build_cfg(weights_path)
    return DefaultPredictor(cfg)


def _resolve_weights_path(weights_path: str | None) -> str | None:
    """Возвращает путь к model_final.pth: env → /app/models → ./output_ui_detectron2."""
    if weights_path and Path(weights_path).exists():
        return weights_path
    env_path = os.environ.get(DEFAULT_WEIGHTS_ENV, "").strip()
    if env_path and Path(env_path).exists():
        return env_path
    docker_path = os.path.join(DEFAULT_WEIGHTS_DIR_DOCKER, DEFAULT_WEIGHTS_FILE)
    if Path(docker_path).exists():
        return docker_path
    local_path = Path(__file__).resolve().parents[2] / "output_ui_detectron2" / DEFAULT_WEIGHTS_FILE
    if local_path.exists():
        return str(local_path)
    root = Path(__file__).resolve().parents[3]
    local_root = root / "output_ui_detectron2" / DEFAULT_WEIGHTS_FILE
    if local_root.exists():
        return str(local_root)
    return None


def predict_raw(image_path: str, weights_path: str | None = None) -> List[dict[str, Any]]:
    """
    Инференс: сырой выход модели {class, bbox [x1,y1,x2,y2], score}.
    bbox в пикселях изображения, без рескейла.
    """
    resolved = _resolve_weights_path(weights_path)
    if not resolved:
        raise FileNotFoundError(
            "UI-Elements weights not found. Set ATOMS_V2_UI_ELEMENTS_WEIGHTS or place model_final.pth in output_ui_detectron2/"
        )
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    predictor = _get_predictor(resolved)
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


def predict(image_path: str, weights_path: str | None = None) -> List[dict[str, Any]]:
    """
    Perception-слой: стабильный JSON-формат для пайплайна.

    Возвращает список элементов в едином формате RawUICVAtom-стиль:
    - id: ui_atom_1, ui_atom_2, ...
    - source: "detectron2"
    - type: имя класса модели (link, button, input, ...)
    - bbox: [x1, y1, x2, y2] в пикселях изображения, без рескейла
    - confidence: float [0,1]

    Координаты — единственная система (исходное изображение). Детекция отделена от OCR и семантики.
    """
    raw = predict_raw(image_path, weights_path=weights_path)
    out: List[dict[str, Any]] = []
    for i, r in enumerate(raw):
        bbox = r.get("bbox", [0, 0, 0, 0])
        if len(bbox) < 4:
            bbox = [0.0, 0.0, 0.0, 0.0]
        out.append({
            "id": f"ui_atom_{i + 1}",
            "source": "detectron2",
            "type": r.get("class", "unknown"),
            "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
            "confidence": float(r.get("score", 0)),
        })
    return out  # stable JSON format for pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detectron2 inference: UI-Elements (output_ui_detectron2). Stable JSON output."
    )
    parser.add_argument("--image", required=True, help="Path to screenshot")
    parser.add_argument("--weights", default=None, help="Path to model_final.pth")
    parser.add_argument("--indent", type=int, default=2, help="JSON indent")
    args = parser.parse_args()

    out = predict(args.image, weights_path=args.weights)
    print(json.dumps(out, indent=args.indent))


if __name__ == "__main__":
    main()
