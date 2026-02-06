"""
Vision-first UI: Detectron2 Mask R-CNN (web UI) — единственный источник границ. Bbox из маски.

Совместимо с kvyb/Segmentation-of-web-UI-elements-with-Detectron2 (Mask R-CNN, instance segmentation).
Порядок классов датасета kvyb фиксирован в KVYB_THING_CLASSES; маппинг в наши типы валидируется.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Порядок классов датасета kvyb (индекс 0..N-1). Должен совпадать с _annotations.coco.json.
# NUM_CLASSES = len(KVYB_THING_CLASSES). Валидация: все имена есть в TYPE_BY_CLASS_NAME.
KVYB_THING_CLASSES: Tuple[str, ...] = (
    "button",
    "input",
    "text field",
    "title",
    "navbar",
    "header",
    "card",
    "section",
    "list",
    "text",
    "paragraph",
    "image",
)
KVYB_NUM_CLASSES = len(KVYB_THING_CLASSES)

# Маппинг имя_класса (lowercase) → наш type. Все KVYB_THING_CLASSES должны быть покрыты.
TYPE_BY_CLASS_NAME: Dict[str, str] = {
    "button": "button",
    "input": "input",
    "text field": "input",
    "title": "header",
    "navbar": "header",
    "header": "header",
    "card": "card",
    "section": "section",
    "list": "section",
    "text": "text_region",
    "paragraph": "panel",
    "image": "card",
}
# Семантический приоритет при вложенности: меньше = выше приоритет как "контейнер" (родитель).
# button/input — листья (высокий номер), card/section — контейнеры (низкий).
TYPE_PARENT_PRIORITY: Dict[str, int] = {
    "card": 0,
    "section": 1,
    "panel": 2,
    "header": 3,
    "text_region": 4,
    "input": 5,
    "button": 6,
}


def _validate_kvyb_mapping() -> None:
    for name in KVYB_THING_CLASSES:
        key = name.strip().lower()
        if key not in TYPE_BY_CLASS_NAME:
            logger.warning("vision_ui: kvyb class %r not in TYPE_BY_CLASS_NAME, will map to panel", name)


def _bbox_from_mask(mask: Any) -> Tuple[int, int, int, int]:
    """Bounding box из маски (H,W): (x, y, w, h)."""
    import numpy as np
    if hasattr(mask, "numpy"):
        m = mask.numpy()
    else:
        m = np.asarray(mask)
    if m.ndim == 3:
        m = m.squeeze()
    rows = np.any(m, axis=1)
    cols = np.any(m, axis=0)
    if not np.any(rows) or not np.any(cols):
        return 0, 0, 0, 0
    y1, y2 = int(np.argmax(rows)), int(len(rows) - np.argmax(rows[::-1]))
    x1, x2 = int(np.argmax(cols)), int(len(cols) - np.argmax(cols[::-1]))
    return x1, y1, x2 - x1, y2 - y1


def _type_from_class_index(cls_idx: int) -> str:
    """Тип по индексу класса kvyb. Валидированный маппинг."""
    if 0 <= cls_idx < len(KVYB_THING_CLASSES):
        name = KVYB_THING_CLASSES[cls_idx].strip().lower()
        return TYPE_BY_CLASS_NAME.get(name, "panel")
    return "panel"


def run_vision_ui_regions(image_path: str) -> List[Dict[str, Any]]:
    """
    Единственный источник границ: Detectron2 Mask R-CNN (web UI). Bbox строится из маски.
    Возвращает VisualElement: {x, y, w, h, type, score, parent_region_id}. PubLayNet не используется.
    """
    _validate_kvyb_mapping()
    path = Path(image_path)
    if not path.exists():
        return []
    weights_path = os.environ.get("WEB_UI_DETECTRON2_WEIGHTS", "").strip()
    if not weights_path or not Path(weights_path).exists():
        logger.debug("vision_ui: WEB_UI_DETECTRON2_WEIGHTS not set or missing")
        return []
    num_classes = int(os.environ.get("WEB_UI_DETECTRON2_NUM_CLASSES", str(KVYB_NUM_CLASSES)))
    try:
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        from detectron2 import model_zoo
    except ImportError:
        logger.debug("vision_ui: detectron2 not available")
        return []
    try:
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = weights_path
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(os.environ.get("WEB_UI_DETECTRON2_THRESH", "0.5"))
        try:
            import torch
            cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            cfg.MODEL.DEVICE = "cpu"
        predictor = DefaultPredictor(cfg)
    except Exception as e:
        logger.warning("vision_ui: failed to load Mask R-CNN: %s", e)
        return []
    try:
        import cv2
        import numpy as np
        im = cv2.imread(str(path))
        if im is None:
            return []
        outputs = predictor(im)
        instances = outputs.get("instances")
        if instances is None:
            return []
        instances = instances.to("cpu")
        classes = instances.pred_classes
        scores = instances.scores
        pred_masks = getattr(instances, "pred_masks", None)
        pred_boxes = instances.pred_boxes
        regions: List[Dict[str, Any]] = []
        for k in range(len(classes)):
            if pred_masks is not None and k < pred_masks.shape[0]:
                x, y, w, h = _bbox_from_mask(pred_masks[k])
            else:
                box = pred_boxes[k].tensor[0]
                x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                x, y = int(x1), int(y1)
                w, h = int(x2 - x1), int(y2 - y1)
            if w <= 0 or h <= 0:
                continue
            cls_idx = int(classes[k])
            our_type = _type_from_class_index(cls_idx)
            score = float(scores[k])
            regions.append({
                "x": x, "y": y, "w": w, "h": h,
                "type": our_type,
                "score": score,
                "parent_region_id": None,
            })
        _assign_parent_by_containment_and_priority(regions)
        for r in regions:
            r["_detector_source"] = "vision_ui"
        logger.info("vision_ui: image_path=%s regions=%d", image_path, len(regions))
        return regions
    except Exception as e:
        logger.warning("vision_ui: inference failed: %s", e)
        return []


def _assign_parent_by_containment_and_priority(regions: List[Dict[str, Any]]) -> None:
    """Родитель = наименьший содержащий регион. При равной площади — семантический приоритет: card > button (button внутри card)."""
    for i, child in enumerate(regions):
        cx, cy, cw, ch = child["x"], child["y"], child["w"], child["h"]
        c_x2, c_y2 = cx + cw, cy + ch
        best_j: Optional[int] = None
        best_area = 0
        best_priority = 999
        for j, par in enumerate(regions):
            if j == i:
                continue
            px, py, pw, ph = par["x"], par["y"], par["w"], par["h"]
            p_x2, p_y2 = px + pw, py + ph
            if px > cx or py > cy or p_x2 < c_x2 or p_y2 < c_y2:
                continue
            area = pw * ph
            prio = TYPE_PARENT_PRIORITY.get(par.get("type", "panel"), 2)
            if best_j is None or area < best_area or (area == best_area and prio < best_priority):
                best_j = j
                best_area = area
                best_priority = prio
        child["parent_region_id"] = best_j
