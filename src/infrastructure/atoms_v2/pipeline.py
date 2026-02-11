"""
Atoms_v2 pipeline: Detectron2 (atoms only) → CV visual regions → assign atoms → OCR (parallel full-page + per-region) → merge & conflict resolution → logical UI.

ML — только атомы. CV — контейнеры. OCR — параллельно по всей странице и по регионам. Merge — текст не подавляется UI.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Merge: текст считается внутри UI-элемента, если доля площади бокса внутри bbox атома >= этого порога.
TEXT_INSIDE_ATOM_IOU_THRESHOLD = 0.5
# Шум: боксы с площадью меньше не учитываем; больше MAX_BLOCK_SCREEN_RATIO от площади изображения — отбрасываем.
MIN_TEXT_BOX_AREA_PX = 15
MAX_BLOCK_SCREEN_RATIO = 0.8

# Маппинг имён классов датасета kvvb → визуальный тип (сырая карта UI).
KVVB_CLASS_TO_ATOM_TYPE: Dict[str, str] = {
    "contactsSendformButton": "button",
    "contactsSocialButtons": "button",
    "contactsEmailFormInput": "input",
    "contactsMessageFormInput": "input",
    "contactsNameFormInput": "input",
    "contactsSubjectFormInput": "input",
    "contactsTitle": "title",
    "contactsSubtitle": "text_block",
    "contactsDescription": "text_block",
    "contactsPhone": "text_block",
    "contactsEmail": "text_block",
    "contactsAddress": "text_block",
}

# Input от Detectron2 не используется: фантомы не отфильтровать. Источник input — только InputCandidateRecovery (OCR + Phase B).
DETECTRON2_INPUT_IGNORED_TYPES = frozenset({"input"})
# Централизованный чёрный список: атомы с этими типами от DL никогда не попадают в atoms/recovery/semantic_validation.
DL_INPUT_BLACKLIST = frozenset({"input"})

# UI-Elements (Yash Jain / output_ui_detectron2): классы уже в нужном виде (link, button, input, ...).
# Маппинг только для совместимости с _build_logical_ui (select→dropdown, textarea→input).
UI_ELEMENTS_CLASS_TO_ATOM_TYPE: Dict[str, str] = {
    "link": "link",
    "button": "button",
    "input": "input",
    "select": "dropdown",
    "textarea": "input",
    "label": "label",
    "checkbox": "checkbox",
    "radio": "radio",
    "dropdown": "dropdown",
    "slider": "slider",
    "toggle": "toggle",
    "menu_item": "menu_item",
    "clickable": "button",
    "icon": "icon",
    "image": "image",
    "text": "text_block",
}

ATOM_IOU_THRESHOLD = 0.5
# Выбор модели: ATOMS_V2_DETECTION_MODEL=ui_elements | kvvb (по умолчанию ui_elements при наличии весов)
DETECTION_MODEL_ENV = "ATOMS_V2_DETECTION_MODEL"
# Две модели одновременно: real (Yash Jain) + synthetic (generated), затем merge + stabilize_atoms
USE_DUAL_DETECTOR_ENV = "ATOMS_V2_USE_DUAL_DETECTOR"
REAL_WEIGHTS_ENV = "ATOMS_V2_REAL_WEIGHTS"
SYNTHETIC_WEIGHTS_ENV = "ATOMS_V2_SYNTHETIC_WEIGHTS"
REGION_CONTAINMENT_MARGIN = 2

def filter_dl_input_atoms(atoms: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """
    Централизованный фильтр: удаляет атомы с type из DL_INPUT_BLACKLIST и source == 'detectron2'.
    Input от DL не участвует в recovery/semantic_validation. Возвращает (filtered_atoms, count_removed).
    """
    kept: List[Dict[str, Any]] = []
    removed = 0
    for a in atoms:
        if a.get("source") == "detectron2" and (a.get("type") or "") in DL_INPUT_BLACKLIST:
            removed += 1
            if removed == 1:
                logger.debug("dl_input_suppressed: atom id=%s type=%s bbox=%s", a.get("id"), a.get("type"), a.get("bbox"))
            continue
        kept.append(a)
    return kept, removed


# v3: в UI участвуют только атомы с semantic_lock (назначенная роль в semantic_validation)
def _atoms_participating_in_ui(atoms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Атомы, участвующие в UI: v3 — только semantic_lock; иначе исключены *_candidate, layout, semantic_valid=False."""
    if any(a.get("semantic_lock") is not None for a in atoms[:3]):
        return [a for a in atoms if a.get("semantic_lock")]
    return [
        a for a in atoms
        if not (a.get("type") or "").endswith("_candidate")
        and (a.get("type") or "") != "layout"
        and a.get("semantic_valid", True)
    ]


def _ui_elements_weights_available() -> bool:
    """Проверяет наличие весов UI-Elements (output_ui_detectron2)."""
    try:
        from src.infrastructure.layout.inference_ui_elements import _resolve_weights_path
        return _resolve_weights_path(None) is not None
    except Exception:
        return False


def _run_dual_detectron2_atoms(image_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Запускает обе модели: real (Yash Jain) и synthetic (generated).
    Возвращает (atoms_real, atoms_synthetic). У каждого атома source "real" или "synthetic".
    """
    path = Path(image_path)
    if not path.exists():
        return [], []
    try:
        from src.infrastructure.layout.inference_ui_elements import predict as ui_elements_predict
    except ImportError:
        import sys
        project_root = Path(__file__).resolve().parents[3]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from src.infrastructure.layout.inference_ui_elements import predict as ui_elements_predict

    project_root = Path(__file__).resolve().parents[3]
    real_weights = os.environ.get(REAL_WEIGHTS_ENV, "").strip() or str(project_root / "models" / "output_ui_detectron2" / "model_final.pth")
    synthetic_weights = os.environ.get(SYNTHETIC_WEIGHTS_ENV, "").strip() or str(project_root / "models" / "output_ui_detectron2_generated" / "model_final.pth")
    if not Path(real_weights).exists():
        real_weights = str(project_root / "models" / "output_ui_detectron2" / "model_final.pth")
    if not Path(synthetic_weights).exists():
        synthetic_weights = str(project_root / "models" / "output_ui_detectron2_generated" / "model_final.pth")

    atoms_real: List[Dict[str, Any]] = []
    atoms_synthetic: List[Dict[str, Any]] = []

    if Path(real_weights).exists():
        try:
            raw_real = ui_elements_predict(image_path, weights_path=real_weights)
            for i, a in enumerate(raw_real):
                t = a.get("type", "unknown")
                mapped = UI_ELEMENTS_CLASS_TO_ATOM_TYPE.get(t, t)
                if mapped in DETECTRON2_INPUT_IGNORED_TYPES:
                    continue
                atoms_real.append({
                    "id": a.get("id", f"real_{i + 1}"),
                    "source": "real",
                    "type": mapped,
                    "bbox": list(a.get("bbox", [0, 0, 0, 0])),
                    "confidence": float(a.get("confidence", 0)),
                })
        except Exception as e:
            logger.warning("atoms_v2 dual real model: %s", e)

    if Path(synthetic_weights).exists():
        try:
            raw_syn = ui_elements_predict(image_path, weights_path=synthetic_weights)
            for i, a in enumerate(raw_syn):
                t = a.get("type", "unknown")
                mapped = UI_ELEMENTS_CLASS_TO_ATOM_TYPE.get(t, t)
                if mapped in DETECTRON2_INPUT_IGNORED_TYPES:
                    continue
                atoms_synthetic.append({
                    "id": a.get("id", f"syn_{i + 1}"),
                    "source": "synthetic",
                    "type": mapped,
                    "bbox": list(a.get("bbox", [0, 0, 0, 0])),
                    "confidence": float(a.get("confidence", 0)),
                })
        except Exception as e:
            logger.warning("atoms_v2 dual synthetic model: %s", e)

    return atoms_real, atoms_synthetic


def _run_detectron2_atoms(image_path: str) -> List[Dict[str, Any]]:
    """
    Perception-слой: Detectron2 — только геометрия и тип. Стабильный JSON (id, source, type, bbox, confidence).

    Модель выбирается по ATOMS_V2_DETECTION_MODEL: ui_elements (output_ui_detectron2) или kvvb.
    Координаты в пикселях изображения, без рескейла. Детекция отделена от OCR и семантики.
    """
    path = Path(image_path)
    if not path.exists():
        return []
    model_choice = os.environ.get(DETECTION_MODEL_ENV, "").strip().lower()
    # По умолчанию: ui_elements, если веса есть; иначе kvvb
    use_ui_elements = model_choice == "ui_elements" or (
        model_choice != "kvvb"
        and _ui_elements_weights_available()
    )

    # UI-Elements (output_ui_detectron2): стабильный JSON уже возвращается из predict()
    if use_ui_elements:
        try:
            from src.infrastructure.layout.inference_ui_elements import predict as ui_elements_predict
        except ImportError:
            import sys
            project_root = Path(__file__).resolve().parents[3]
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            from src.infrastructure.layout.inference_ui_elements import predict as ui_elements_predict
        try:
            atoms = ui_elements_predict(image_path, weights_path=None)
            if not atoms:
                return []
            # Приводим type к единому виду; input от Detectron2 не добавляем (источник input — только InputCandidateRecovery)
            out: List[Dict[str, Any]] = []
            for a in atoms:
                raw_type = a.get("type", "unknown")
                t = UI_ELEMENTS_CLASS_TO_ATOM_TYPE.get(raw_type, raw_type)
                if t in DETECTRON2_INPUT_IGNORED_TYPES:
                    continue
                a["type"] = t
                out.append(a)
            return out
        except FileNotFoundError as e:
            logger.warning("atoms_v2: %s", e)
            return []

    # Kvvb (legacy)
    try:
        from src.infrastructure.layout.inference_kvvb import predict as kvvb_predict
    except ImportError:
        import sys
        project_root = Path(__file__).resolve().parents[3]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from src.infrastructure.layout.inference_kvvb import predict as kvvb_predict
    weights = os.environ.get("ATOMS_V2_WEIGHTS", "")
    if not weights:
        weights = os.path.join("/app/models/output_kvvb", "model_final.pth")
    if not Path(weights).exists():
        logger.warning("atoms_v2: weights not found %s", weights)
        return []
    raw = kvvb_predict(image_path, weights_path=weights)
    atoms = []
    for i, r in enumerate(raw):
        cls_name = r.get("class", "")
        atom_type = KVVB_CLASS_TO_ATOM_TYPE.get(cls_name, "text_block")
        if atom_type in DETECTRON2_INPUT_IGNORED_TYPES:
            continue
        bbox = r.get("bbox", [0, 0, 0, 0])
        if len(bbox) < 4:
            bbox = [0.0, 0.0, 0.0, 0.0]
        atoms.append({
            "id": f"ui_atom_{i + 1}",
            "source": "detectron2",
            "type": atom_type,
            "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
            "confidence": float(r.get("score", 0)),
        })
    return atoms


def _run_cv_visual_regions(image_path: str) -> List[Dict[str, Any]]:
    """CV: визуальные контейнеры (карточки, секции, панели). Выход: [{id, bbox, type: visual_region, confidence}, ...]."""
    path = Path(image_path)
    if not path.exists():
        return []
    try:
        import cv2
        import numpy as np
    except ImportError:
        return []
    img = cv2.imread(str(path))
    if img is None:
        return []
    h, w = img.shape[:2]
    area = w * h
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = area * 0.015
    max_area = area * 0.95
    min_side = 30
    regions: List[Dict[str, Any]] = []
    for i, c in enumerate(contours):
        a = cv2.contourArea(c)
        if a < min_area or a > max_area:
            continue
        x, y, rw, rh = cv2.boundingRect(c)
        if rw < min_side or rh < min_side:
            continue
        if a / (rw * rh) < 0.5:
            continue
        regions.append({
            "id": f"region_{i}",
            "bbox": [x, y, x + rw, y + rh],
            "type": "visual_region",
            "confidence": 0.9,
        })
    regions.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))
    return regions


def _iou_box_region(box: List[float], reg_bbox: List[float]) -> float:
    """IoU(box, reg). box/reg_bbox = [x1,y1,x2,y2]."""
    ix1 = max(box[0], reg_bbox[0])
    iy1 = max(box[1], reg_bbox[1])
    ix2 = min(box[2], reg_bbox[2])
    iy2 = min(box[3], reg_bbox[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    return inter / max(1e-9, box_area)


def _assign_atoms_to_regions(
    atoms: List[Dict[str, Any]],
    regions: List[Dict[str, Any]],
) -> Dict[str, Optional[str]]:
    """atom_id -> region_id. Внутри bbox или IoU > порога. Иначе None (root)."""
    out: Dict[str, Optional[str]] = {}
    for a in atoms:
        bid = a.get("id", "")
        bbox = a.get("bbox", [0, 0, 0, 0])
        best_rid: Optional[str] = None
        best_iou = 0.0
        for r in regions:
            rbbox = r.get("bbox", [0, 0, 0, 0])
            iou = _iou_box_region(bbox, rbbox)
            if iou >= ATOM_IOU_THRESHOLD and iou > best_iou:
                best_iou = iou
                best_rid = r.get("id")
        out[bid] = best_rid
    return out


def _assign_ocr_to_regions(
    ocr_boxes: List[Dict[str, Any]],
    regions: List[Dict[str, Any]],
) -> Dict[str, Optional[str]]:
    """ocr_box_id -> region_id по IoU (область для связи: OCR и атом только в одном регионе)."""
    out: Dict[str, Optional[str]] = {}
    for ob in ocr_boxes:
        oid = ob.get("id", "")
        obbox = ob.get("bbox", [0, 0, 0, 0])
        if len(obbox) < 4:
            out[oid] = None
            continue
        best_rid: Optional[str] = None
        best_iou = 0.0
        for r in regions:
            rbbox = r.get("bbox", [0, 0, 0, 0])
            iou = _iou_box_region(obbox, rbbox)
            if iou >= ATOM_IOU_THRESHOLD and iou > best_iou:
                best_iou = iou
                best_rid = r.get("id")
        out[oid] = best_rid
    return out


def _run_full_page_ocr(image_path: str) -> List[Dict[str, Any]]:
    """Полностраничный OCR: run_text_detect + run_ocr_boxes. Независим от Detectron2/регионов.
    Возвращает список [{x, y, w, h, text, confidence, box_index}, ...]."""
    path = Path(image_path)
    if not path.exists():
        return []
    try:
        from src.infrastructure.debug.services import run_text_detect, run_ocr_boxes
    except ImportError:
        return []
    raw_boxes = run_text_detect(image_path)
    if not raw_boxes:
        return []
    ocr_results = run_ocr_boxes(image_path, raw_boxes)
    out: List[Dict[str, Any]] = []
    for i, box in enumerate(raw_boxes):
        r = ocr_results[i] if i < len(ocr_results) else {}
        out.append({
            "x": box.get("x", 0),
            "y": box.get("y", 0),
            "w": box.get("w", 0),
            "h": box.get("h", 0),
            "text": (r.get("text") or "").strip(),
            "confidence": float(r.get("confidence", 0)),
            "box_index": i,
        })
    return out


def _iou_text_box_atom(box: Dict[str, Any], atom_bbox: List[float]) -> float:
    """Доля площади текстового бокса (x,y,w,h), попадающая внутрь bbox атома [x1,y1,x2,y2]. Возвращает intersection/area(box)."""
    x, y = box.get("x", 0), box.get("y", 0)
    w, h = box.get("w", 0), box.get("h", 0)
    if w <= 0 or h <= 0:
        return 0.0
    box_area = w * h
    if len(atom_bbox) < 4:
        return 0.0
    ax1, ay1, ax2, ay2 = atom_bbox[0], atom_bbox[1], atom_bbox[2], atom_bbox[3]
    ix1 = max(x, ax1)
    iy1 = max(y, ay1)
    ix2 = min(x + w, ax2)
    iy2 = min(y + h, ay2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    return inter / max(1e-9, box_area)


def _coverage_ocr_atom(ocr_bbox: List[float], atom_bbox: List[float]) -> float:
    """coverage = area(intersection(ocr_box, atom_bbox)) / area(ocr_box). Оба bbox [x1,y1,x2,y2]."""
    if len(ocr_bbox) < 4 or len(atom_bbox) < 4:
        return 0.0
    ox1, oy1, ox2, oy2 = ocr_bbox[0], ocr_bbox[1], ocr_bbox[2], ocr_bbox[3]
    box_area = (ox2 - ox1) * (oy2 - oy1)
    if box_area <= 0:
        return 0.0
    ix1 = max(ox1, atom_bbox[0])
    iy1 = max(oy1, atom_bbox[1])
    ix2 = min(ox2, atom_bbox[2])
    iy2 = min(oy2, atom_bbox[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    return inter / max(1e-9, box_area)


# Эвристика link_type по типу атома (без ML). Не финальная семантика.
ATOM_TYPE_TO_LINK_TYPE: Dict[str, str] = {
    "button": "label",
    "input": "label",
    "title": "label",
    "navbar": "label",
    "card": "content",
    "container": "content",
    "text_block": "content",
    "icon": "label",
    "dropdown": "label",
    "modal": "content",
    "checkbox": "label",
    "radio": "label",
    "unknown": "content",
    # UI-Elements (output_ui_detectron2)
    "link": "label",
    "inline_link_candidate": "content",
    "inline_text_candidate": "content",
    "layout_candidate": "content",
    "layout": "content",
    "container_candidate": "content",
    "label": "label",
    "select": "label",
    "textarea": "content",
    "slider": "label",
    "toggle": "label",
    "menu_item": "label",
    "clickable": "label",
    "image": "content",
}


def _link_ocr_to_atoms(
    ocr_boxes: List[Dict[str, Any]],
    atoms: List[Dict[str, Any]],
    threshold: float = 0.5,
    regions: Optional[List[Dict[str, Any]]] = None,
    atom_to_region: Optional[Dict[str, Optional[str]]] = None,
) -> List[Dict[str, Any]]:
    """
    Merge Layer v2: связывает OCR-боксы с UI-атомами геометрически в пределах области.

    coverage = intersection(ocr_box, atom_bbox) / area(ocr_box).
    Доминирующий атом: max coverage; при равенстве — меньшая площадь атома.
    Если заданы regions и atom_to_region — связь только внутри одного CV-региона (ограничение области для связи).
    atom_id = None, link_type = standalone если ни один атом не подошёл.
    """
    links: List[Dict[str, Any]] = []
    atom_by_id = {a.get("id", ""): a for a in atoms}
    use_region_filter = regions is not None and atom_to_region is not None
    ocr_to_region: Dict[str, Optional[str]] = _assign_ocr_to_regions(ocr_boxes, regions) if use_region_filter and regions else {}
    for ob in ocr_boxes:
        ocr_id = ob.get("id", "")
        ocr_bbox = ob.get("bbox", [0, 0, 0, 0])
        if len(ocr_bbox) < 4:
            links.append({
                "ocr_box_id": ocr_id,
                "atom_id": None,
                "link_type": "standalone",
                "coverage_ratio": 0.0,
            })
            continue
        ocr_region: Optional[str] = ocr_to_region.get(ocr_id) if use_region_filter else None
        best_atom_id: Optional[str] = None
        best_coverage = 0.0
        best_atom_area: Optional[float] = None
        for a in atoms:
            acid = a.get("id", "")
            if use_region_filter and atom_to_region.get(acid) != ocr_region:
                continue
            abbox = a.get("bbox", [0, 0, 0, 0])
            cov = _coverage_ocr_atom(ocr_bbox, abbox)
            if cov < threshold:
                continue
            atom_area = (abbox[2] - abbox[0]) * (abbox[3] - abbox[1]) if len(abbox) >= 4 else 0.0
            if cov > best_coverage or (cov == best_coverage and best_atom_area is not None and atom_area < best_atom_area):
                best_coverage = cov
                best_atom_id = acid
                best_atom_area = atom_area
        if best_atom_id is None:
            links.append({
                "ocr_box_id": ocr_id,
                "atom_id": None,
                "link_type": "standalone",
                "coverage_ratio": 0.0,
            })
        else:
            atom_type = atom_by_id.get(best_atom_id, {}).get("type", "unknown")
            link_type = ATOM_TYPE_TO_LINK_TYPE.get(atom_type, "content")
            links.append({
                "ocr_box_id": ocr_id,
                "atom_id": best_atom_id,
                "link_type": link_type,
                "coverage_ratio": round(best_coverage, 4),
            })
    return links


def _text_inside_ui_from_links(
    text_ui_links: List[Dict[str, Any]],
    raw_ocr_boxes: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Строит text_inside_ui (atom_id -> [{text, bbox, confidence}]) из text_ui_links и raw_ocr_boxes для _build_logical_ui. Атомы не модифицируются."""
    ocr_by_id = {b.get("id", ""): b for b in raw_ocr_boxes}
    text_inside_ui: Dict[str, List[Dict[str, Any]]] = {}
    for link in text_ui_links:
        acid = link.get("atom_id")
        if not acid:
            continue
        ocr_id = link.get("ocr_box_id", "")
        ob = ocr_by_id.get(ocr_id)
        if not ob:
            continue
        text_inside_ui.setdefault(acid, []).append({
            "text": ob.get("text", ""),
            "bbox": ob.get("bbox", [0, 0, 0, 0]),
            "confidence": ob.get("confidence", 0),
        })
    return text_inside_ui


def _merge_ocr_with_atoms(
    ocr_boxes_with_text: List[Dict[str, Any]],
    atoms: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Для каждого OCR-бокса определяем доминирующий атом по IoU (intersection/area(box)).
    Текст не удаляется из общего слоя — только дополнительно привязывается к атомам.
    Возвращает: atom_id -> [{text, bbox, confidence}, ...]."""
    text_inside_ui: Dict[str, List[Dict[str, Any]]] = {a.get("id", ""): [] for a in atoms}
    for ob in ocr_boxes_with_text:
        best_acid: Optional[str] = None
        best_iou = 0.0
        for a in atoms:
            acid = a.get("id", "")
            bbox = a.get("bbox", [0, 0, 0, 0])
            iou = _iou_text_box_atom(ob, bbox)
            if iou >= TEXT_INSIDE_ATOM_IOU_THRESHOLD and iou > best_iou:
                best_iou = iou
                best_acid = acid
        if best_acid:
            text_inside_ui.setdefault(best_acid, []).append({
                "text": ob.get("text", ""),
                "bbox": [ob.get("x", 0), ob.get("y", 0), ob.get("x", 0) + ob.get("w", 0), ob.get("y", 0) + ob.get("h", 0)],
                "confidence": ob.get("confidence", 0),
            })
    return text_inside_ui


def _filter_ocr_noise(
    boxes: List[Dict[str, Any]],
    image_w: int,
    image_h: int,
) -> List[Dict[str, Any]]:
    """Отбрасываем слишком мелкие боксы и блоки > MAX_BLOCK_SCREEN_RATIO площади изображения."""
    if not boxes:
        return []
    area_max = image_w * image_h * MAX_BLOCK_SCREEN_RATIO
    out: List[Dict[str, Any]] = []
    for b in boxes:
        w, h = b.get("w", 0), b.get("h", 0)
        a = w * h
        if a < MIN_TEXT_BOX_AREA_PX:
            continue
        if a > area_max:
            continue
        out.append(b)
    return out


def _apply_legacy_grouping(
    boxes_with_text: List[Dict[str, Any]],
) -> Tuple[List[List[Dict[str, Any]]], List[List[List[Dict[str, Any]]]], List[Dict[str, Any]]]:
    """Группировка текстовых боксов в строки и абзацы (как в improved-full-pipeline).
    boxes_with_text: [{x, y, w, h, text, confidence, ...}]. Добавляем region_id=-1, box_index.
    Возвращает (lines, paragraphs, independent_text_blocks). independent_text_blocks = список блоков {type, text, bbox}."""
    try:
        from src.infrastructure.layout.text_grouping import (
            group_text_boxes_into_lines,
            group_lines_into_paragraphs,
        )
    except ImportError:
        return [], [], []
    boxes_with_region = [
        {**b, "region_id": -1, "box_index": b.get("box_index", i)}
        for i, b in enumerate(boxes_with_text)
    ]
    lines = group_text_boxes_into_lines(boxes_with_region, -1)
    paragraphs = group_lines_into_paragraphs(lines)
    independent_blocks: List[Dict[str, Any]] = []
    for para in paragraphs:
        boxes_para = [bx for ln in para for bx in ln]
        if not boxes_para:
            continue
        xs = [bx.get("x", 0) for bx in boxes_para]
        ys = [bx.get("y", 0) for bx in boxes_para]
        x2s = [bx.get("x", 0) + bx.get("w", 0) for bx in boxes_para]
        y2s = [bx.get("y", 0) + bx.get("h", 0) for bx in boxes_para]
        texts = []
        for bx in boxes_para:
            idx = bx.get("box_index", -1)
            if 0 <= idx < len(boxes_with_text):
                texts.append(boxes_with_text[idx].get("text", ""))
        text_para = " ".join(t for t in texts if t).strip()
        block_type = "paragraph" if len(para) > 1 else "label"
        independent_blocks.append({
            "type": block_type,
            "text": text_para,
            "bbox": [min(xs), min(ys), max(x2s), max(y2s)],
            "lines_count": len(para),
        })
    return lines, paragraphs, independent_blocks


def _run_ocr_per_region(
    image_path: str,
    regions: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """OCR только внутри каждого региона. Crop → OCR → координаты в image space. region_id -> [{text, bbox, confidence}]."""
    from src.infrastructure.debug.services import run_text_detect_roi, run_ocr_boxes

    path = Path(image_path)
    if not path.exists():
        return {r["id"]: [] for r in regions}
    out: Dict[str, List[Dict[str, Any]]] = {}
    for r in regions:
        rid = r.get("id", "")
        bbox = r.get("bbox", [0, 0, 0, 0])
        if len(bbox) < 4:
            out[rid] = []
            continue
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        reg_dict = {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}
        boxes = run_text_detect_roi(image_path, reg_dict, scale=1.5)
        if not boxes:
            out[rid] = []
            continue
        ocr_results = run_ocr_boxes(image_path, boxes)
        texts: List[Dict[str, Any]] = []
        for j, box in enumerate(boxes):
            t = ocr_results[j].get("text", "").strip() if j < len(ocr_results) else ""
            c = ocr_results[j].get("confidence", 0.0) if j < len(ocr_results) else 0.0
            texts.append({
                "text": t,
                "bbox": [box.get("x", 0), box.get("y", 0), box.get("x", 0) + box.get("w", 0), box.get("y", 0) + box.get("h", 0)],
                "confidence": c,
            })
        out[rid] = texts
    return out


def _assign_full_page_ocr_to_regions(
    full_page_boxes: List[Dict[str, Any]],
    regions: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Распределяем полностраничные OCR-боксы по регионам по IoU (intersection/area(box)). region_id -> [{text, bbox, confidence}]."""
    out: Dict[str, List[Dict[str, Any]]] = {r["id"]: [] for r in regions}
    for ob in full_page_boxes:
        best_rid: Optional[str] = None
        best_iou = 0.0
        for r in regions:
            rid = r.get("id", "")
            rbbox = r.get("bbox", [0, 0, 0, 0])
            iou = _iou_text_box_atom(ob, rbbox)
            if iou >= TEXT_INSIDE_ATOM_IOU_THRESHOLD and iou > best_iou:
                best_iou = iou
                best_rid = rid
        if best_rid:
            out.setdefault(best_rid, []).append({
                "text": ob.get("text", ""),
                "bbox": [ob.get("x", 0), ob.get("y", 0), ob.get("x", 0) + ob.get("w", 0), ob.get("y", 0) + ob.get("h", 0)],
                "confidence": ob.get("confidence", 0),
            })
    return out


def _merge_text_blocks(
    region_texts: Dict[str, List[Dict[str, Any]]],
    regions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Объединение текстов в блоки по Y, X, parent_region. Выход: [{id, region_id, role, text, bbox}]."""
    blocks: List[Dict[str, Any]] = []
    block_id = 0
    for rid, texts in region_texts.items():
        if not texts:
            continue
        sorted_texts = sorted(texts, key=lambda t: (t["bbox"][1], t["bbox"][0]))
        for t in sorted_texts:
            blocks.append({
                "id": f"text_block_{block_id}",
                "region_id": rid,
                "role": "paragraph",
                "text": t.get("text", ""),
                "bbox": t.get("bbox", [0, 0, 0, 0]),
                "confidence": t.get("confidence", 0),
            })
            block_id += 1
    return blocks


def _build_logical_ui(
    atoms: List[Dict[str, Any]],
    regions: List[Dict[str, Any]],
    atom_to_region: Dict[str, Optional[str]],
    text_blocks: List[Dict[str, Any]],
    text_inside_ui: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    """Логическое объединение: card = region + title + text/button; form = ≥2 input + button; navbar = верхние кнопки.
    text_inside_ui: atom_id -> [{text, bbox, confidence}] — текст внутри UI-элемента (не подавляем, используем как label)."""
    result: List[Dict[str, Any]] = []
    region_by_id = {r["id"]: r for r in regions}
    atoms_by_region: Dict[str, List[Dict[str, Any]]] = {r["id"]: [] for r in regions}
    atoms_by_region["_root"] = []
    for a in atoms:
        rid = atom_to_region.get(a.get("id", "")) or "_root"
        atoms_by_region.setdefault(rid, []).append(a)
    texts_by_region: Dict[str, List[Dict[str, Any]]] = {}
    for tb in text_blocks:
        rid = tb.get("region_id", "")
        texts_by_region.setdefault(rid, []).append(tb)

    text_inside_ui = text_inside_ui or {}

    for r in regions:
        rid = r["id"]
        bbox = r.get("bbox", [0, 0, 0, 0])
        region_atoms = atoms_by_region.get(rid, [])
        region_texts = texts_by_region.get(rid, [])
        has_title = any(a.get("type") == "title" for a in region_atoms)
        has_button = any(a.get("type") == "button" for a in region_atoms)
        inputs = [a for a in region_atoms if a.get("type") == "input"]
        node_type = "panel"
        if len(inputs) >= 2 and has_button:
            node_type = "form"
        elif has_title and (region_texts or has_button):
            node_type = "card"
        children: List[Dict[str, Any]] = []
        for a in region_atoms:
            acid = a.get("id", "")
            label = ""
            if acid and text_inside_ui.get(acid):
                label = " ".join(t.get("text", "") for t in text_inside_ui[acid] if t.get("text")).strip()
            children.append({
                "type": a.get("type", "text_block"),
                "bbox": a.get("bbox", []),
                "confidence": a.get("confidence", 0),
                "label": label,
            })
        for tb in region_texts:
            children.append({
                "type": "text",
                "text": tb.get("text", ""),
                "bbox": tb.get("bbox", []),
            })
        result.append({
            "type": node_type,
            "bbox": bbox,
            "children": children,
        })

    root_atoms = atoms_by_region.get("_root", [])
    top_y = min((a["bbox"][1] for a in root_atoms), default=0)
    if root_atoms and any(a["bbox"][1] < top_y + 80 for a in root_atoms):
        nav_buttons = [a for a in root_atoms if a.get("type") == "button" and a["bbox"][1] < top_y + 100]
        if nav_buttons:
            result.insert(0, {
                "type": "navbar",
                "bbox": [0, 0, 9999, 80],
                "children": [
                    {
                        "type": "button",
                        "bbox": a.get("bbox", []),
                        "confidence": a.get("confidence", 0),
                        "label": " ".join(t.get("text", "") for t in text_inside_ui.get(a.get("id", ""), []) if t.get("text")).strip(),
                    }
                    for a in nav_buttons
                ],
            })
    return result


# Минимальная L2-разница (0–255) между преобладающим цветом кнопки и средним цветом экрана; иначе FP
MIN_BUTTON_SCREEN_COLOR_DISTANCE = 22.0
MIN_BUTTON_CROP_AREA_PX = 100  # не проверять цвет для слишком маленьких кропов


def _filter_buttons_by_dominant_color(image_path: str, atoms: List[Dict[str, Any]]) -> None:
    """
    Synthetic button допустим только если преобладающий цвет области кнопки отличается от среднего цвета экрана.
    Иначе — container_candidate (отбросить ложно положительные срабатывания). Real не трогаем.
    """
    path = Path(image_path)
    if not path.exists():
        return
    try:
        from PIL import Image
    except ImportError:
        return
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return
    w, h = img.size
    if w <= 0 or h <= 0:
        return
    # Средний цвет экрана (downscale для скорости)
    screen_sample = img.resize((min(64, w), min(64, h)), Image.Resampling.BOX)
    pixels_screen = list(screen_sample.getdata())
    if not pixels_screen:
        return
    n = len(pixels_screen)
    screen_r = sum(p[0] for p in pixels_screen) / n
    screen_g = sum(p[1] for p in pixels_screen) / n
    screen_b = sum(p[2] for p in pixels_screen) / n

    for a in atoms:
        if a.get("type") != "button" or a.get("source") != "synthetic_only":
            continue
        bbox = a.get("bbox", [0, 0, 0, 0])
        if len(bbox) < 4:
            continue
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            continue
        if (x2 - x1) * (y2 - y1) < MIN_BUTTON_CROP_AREA_PX:
            continue
        crop = img.crop((x1, y1, x2, y2))
        pixels = list(crop.getdata())
        if not pixels:
            continue
        # Преобладающий цвет области — медиана по каналам
        sp = sorted(p[0] for p in pixels)
        btn_r = sp[len(sp) // 2]
        sp = sorted(p[1] for p in pixels)
        btn_g = sp[len(sp) // 2]
        sp = sorted(p[2] for p in pixels)
        btn_b = sp[len(sp) // 2]
        dist = (btn_r - screen_r) ** 2 + (btn_g - screen_g) ** 2 + (btn_b - screen_b) ** 2
        dist = dist ** 0.5
        if dist < MIN_BUTTON_SCREEN_COLOR_DISTANCE:
            a["type"] = "container_candidate"
            a["confidence"] = min(1.0, (a.get("confidence", 0) or 0) * 0.3)
            logger.debug("filter_buttons_by_color: button color ~ screen -> container_candidate id=%s", a.get("id"))


def get_atoms_after_postprocess(
    image_path: str,
    parallel_ocr: bool = True,
    legacy_text_pipeline: bool = True,
) -> List[Dict[str, Any]]:
    """
    Запускает пайплайн до (включительно) postprocess и возвращает атомы.
    Те же bbox, что идут в semantic_validation. Для экспорта в .det2.json (teacher_dataset_builder_v2).
    """
    result = run_atoms_v2_pipeline(
        image_path,
        parallel_ocr=parallel_ocr,
        legacy_text_pipeline=legacy_text_pipeline,
        stop_after_postprocess=True,
    )
    return result.get("atoms", [])


def run_atoms_v2_pipeline(
    image_path: str,
    parallel_ocr: bool = True,
    legacy_text_pipeline: bool = True,
    stop_after_postprocess: bool = False,
    use_experimental_multilevel_v2: bool = False,
    experimental_v2_debug_dir: Optional[str] = None,
    use_form_container_first: bool = False,
    form_container_first_debug_dir: Optional[str] = None,
    demo_mode: bool = False,
) -> Dict[str, Any]:
    """
    Пайплайн atoms_v2: Detectron2 (atoms) + CV regions; OCR; merge & conflict resolution; legacy grouping.

    use_form_container_first: если True, после multilevel запускается run_form_container_first_inference
    (FormContainerDetector → FormInnerLayout → SlotDetector → FieldLocator → FormGraph). Все уровни только внутри FormContainer.bbox.
    form_container_first_debug_dir: каталог для container_bbox.png, rows.png, slots.png, slot_assignments.png.
    demo_mode: идеальная форма для отладки до BPG (контейнер по площади, одна строка = один input, demo_*.json и validate_demo_pipeline).
    """
    log: List[str] = []
    path = Path(image_path)
    if not path.exists():
        return {
            "unified_ui": [], "atoms": [], "regions": [], "atom_to_region": {},
            "raw_ocr_boxes": [], "text_ui_links": [], "text_blocks": [], "independent_text_blocks": [], "lines": [], "paragraphs": [],
            "text_inside_ui": {}, "log": ["Image not found"], "debug_image_path": None,
        }

    use_dual = os.environ.get(USE_DUAL_DETECTOR_ENV, "").strip().lower() in ("1", "true", "yes")
    atoms_real: List[Dict[str, Any]] = []
    atoms_synthetic: List[Dict[str, Any]] = []

    if use_dual:
        atoms_real, atoms_synthetic = _run_dual_detectron2_atoms(image_path)
        atoms = []
        log.append(f"atoms_real={len(atoms_real)} atoms_synthetic={len(atoms_synthetic)}")
    else:
        atoms = _run_detectron2_atoms(image_path)
        log.append(f"atoms={len(atoms)}")

    regions = _run_cv_visual_regions(image_path)
    log.append(f"regions={len(regions)}")

    atom_to_region: Dict[str, Optional[str]] = {}
    if not use_dual:
        atom_to_region = _assign_atoms_to_regions(atoms, regions)

    full_page_boxes: List[Dict[str, Any]] = []
    raw_ocr_boxes: List[Dict[str, Any]] = []
    text_ui_links: List[Dict[str, Any]] = []
    text_inside_ui: Dict[str, List[Dict[str, Any]]] = {a.get("id", ""): [] for a in atoms}
    region_texts: Dict[str, List[Dict[str, Any]]] = {r["id"]: [] for r in regions}
    lines: List[List[Dict[str, Any]]] = []
    paragraphs: List[List[List[Dict[str, Any]]]] = []
    independent_text_blocks: List[Dict[str, Any]] = []

    log.append(f"legacy_text_pipeline={str(legacy_text_pipeline).lower()}")

    if parallel_ocr:
        full_page_boxes = _run_full_page_ocr(image_path)
        log.append(f"full_page_ocr_boxes={len(full_page_boxes)}")
        try:
            from PIL import Image
            with Image.open(image_path) as im:
                img_w, img_h = im.width, im.height
        except Exception:
            img_w, img_h = 1200, 800
        full_page_boxes = _filter_ocr_noise(full_page_boxes, img_w, img_h)
        log.append(f"full_page_ocr_after_noise={len(full_page_boxes)}")
        raw_ocr_boxes = [
            {
                "id": f"ocr_{i + 1}",
                "source": "ocr",
                "bbox": [float(b["x"]), float(b["y"]), float(b["x"]) + float(b["w"]), float(b["y"]) + float(b["h"])],
                "text": (b.get("text") or "").strip(),
                "confidence": float(b.get("confidence", 0)),
            }
            for i, b in enumerate(full_page_boxes)
        ]
        log.append(f"raw_ocr_boxes={len(raw_ocr_boxes)}")
        if use_dual:
            from src.infrastructure.atoms_v2.synthetic_ocr_filter import filter_synthetic_atoms_by_ocr
            from src.infrastructure.atoms_v2.merge_stabilize import stabilize_atoms
            filter_synthetic_atoms_by_ocr(atoms_synthetic, raw_ocr_boxes, atoms_real, regions)
            log.append("synthetic_ocr_filter applied")
            atoms = stabilize_atoms(atoms_real, atoms_synthetic, raw_ocr_boxes, regions)
            _filter_buttons_by_dominant_color(image_path, atoms)
            log.append("filter_buttons_by_color applied")
            atom_to_region = _assign_atoms_to_regions(atoms, regions)
            log.append(f"atoms_after_merge_stabilize={len(atoms)}")
        text_ui_links = _link_ocr_to_atoms(raw_ocr_boxes, atoms, threshold=TEXT_INSIDE_ATOM_IOU_THRESHOLD)
        text_inside_ui = _text_inside_ui_from_links(text_ui_links, raw_ocr_boxes)
        log.append(f"text_ui_links={len(text_ui_links)}")
        region_texts = _assign_full_page_ocr_to_regions(full_page_boxes, regions)
        if legacy_text_pipeline and full_page_boxes:
            lines, paragraphs, independent_text_blocks = _apply_legacy_grouping(full_page_boxes)
            log.append(f"lines={len(lines)} paragraphs={sum(len(p) for p in paragraphs)} independent_blocks={len(independent_text_blocks)}")
    else:
        try:
            region_texts = _run_ocr_per_region(image_path, regions)
        except Exception as e:
            logger.warning("atoms_v2 OCR per region failed: %s", e)
        flat_for_merge: List[Dict[str, Any]] = []
        for rid, texts in region_texts.items():
            for t in texts:
                b = t.get("bbox", [0, 0, 0, 0])
                if len(b) >= 4:
                    flat_for_merge.append({
                        "x": b[0], "y": b[1], "w": b[2] - b[0], "h": b[3] - b[1],
                        "text": t.get("text", ""), "confidence": t.get("confidence", 0),
                    })
        if flat_for_merge:
            text_inside_ui = _merge_ocr_with_atoms(flat_for_merge, atoms)
        if use_dual and not atoms:
            from src.infrastructure.atoms_v2.merge_stabilize import stabilize_atoms
            atoms = stabilize_atoms(atoms_real, atoms_synthetic, [], regions)
            _filter_buttons_by_dominant_color(image_path, atoms)
            atom_to_region = _assign_atoms_to_regions(atoms, regions)
            log.append(f"atoms_after_merge_stabilize={len(atoms)}")

    text_blocks = _merge_text_blocks(region_texts, regions)
    log.append(f"text_blocks={len(text_blocks)}")

    # Централизованное подавление input от Detectron2: не участвуют в recovery/semantic_validation
    atoms, dl_input_killed = filter_dl_input_atoms(atoms)
    if dl_input_killed:
        log.append("dl_input_suppressed: %d" % dl_input_killed)
        logger.info("dl_input_suppressed: %d (DL input atoms removed, source=detectron2)", dl_input_killed)

    # Post-processing: стабилизация атомов (фильтр ложных link, synthetic button/input из OCR)
    from src.infrastructure.atoms_v2.postprocess import run_postprocess
    atoms, text_ui_links = run_postprocess(
        atoms,
        raw_ocr_boxes,
        text_ui_links,
        regions,
        independent_text_blocks=independent_text_blocks,
    )
    log.append(f"atoms_after_postprocess={len(atoms)}")

    if stop_after_postprocess:
        return {"atoms": atoms, "raw_ocr_boxes": raw_ocr_boxes, "regions": regions, "log": log}

    # Тема экрана (по среднему luminance): для тёмной темы повышаем доверие к input внутри form
    dark_theme = False
    try:
        import cv2
        img = cv2.imread(str(path))
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            dark_theme = float(gray.mean()) < 128.0
    except Exception:
        pass

    # FormStructureDetection: FORM-FIRST — ищем формы до input
    form_regions: Optional[List[Dict[str, Any]]] = None
    try:
        from src.infrastructure.atoms_v2.form_structure_detection import detect_form_regions
        form_regions = detect_form_regions(regions, atoms, raw_ocr_boxes)
        log.append("form_structure_detection: %d form_regions" % len(form_regions))
    except Exception as form_e:
        log.append("form_structure_detection skipped: %s" % form_e)
        form_regions = None

    # Многоуровневое ориентирование: Macro → Meso → Micro → SchemaValidator; затем fallback
    layout_inferred_atoms: List[Dict[str, Any]] = []
    if form_regions is not None and len(form_regions) > 0:
        try:
            from src.infrastructure.atoms_v2.multilevel_field_inference import run_multilevel_field_inference
            layout_inferred_atoms, ml_log = run_multilevel_field_inference(
                form_regions, atoms, raw_ocr_boxes, regions or [],
                image_path=str(path), dark_theme=dark_theme,
            )
            log.extend(ml_log)
            if layout_inferred_atoms:
                atoms = atoms + layout_inferred_atoms
                log.append("multilevel_field: added %d atoms (source=multilevel)" % len(layout_inferred_atoms))
        except Exception as ml_e:
            log.append("multilevel_field skipped: %s" % ml_e)
        if not layout_inferred_atoms:
            try:
                from src.infrastructure.atoms_v2.visual_first_field_inference import run_visual_first_field_inference
                layout_inferred_atoms, vf_log = run_visual_first_field_inference(
                    form_regions, atoms, raw_ocr_boxes,
                    image_path=str(path), dark_theme=dark_theme,
                )
                log.extend(vf_log)
                if layout_inferred_atoms:
                    atoms = atoms + layout_inferred_atoms
                    log.append("visual_first_field: added %d atoms (fallback)" % len(layout_inferred_atoms))
            except Exception as vf_e:
                log.append("visual_first_field skipped: %s" % vf_e)
        if not layout_inferred_atoms:
            try:
                from src.infrastructure.atoms_v2.card_field_layout_inference import run_card_field_layout_inference
                layout_inferred_atoms, layout_log = run_card_field_layout_inference(
                    form_regions, atoms, raw_ocr_boxes, regions
                )
                log.extend(layout_log)
                if layout_inferred_atoms:
                    atoms = atoms + layout_inferred_atoms
                    log.append("card_field_layout_inference: added %d atoms (fallback)" % len(layout_inferred_atoms))
            except Exception as layout_e:
                log.append("card_field_layout_inference skipped: %s" % layout_e)

    # InputCandidateRecovery: только если внутри card найдено 0 полей (Phase B, Canny, OCR seeds)
    if not form_regions or len(layout_inferred_atoms) == 0:
        try:
            from src.infrastructure.atoms_v2.input_candidate_recovery import run_input_candidate_recovery
            screen_size = None
            if regions:
                rbboxes = [r.get("bbox", [0, 0, 0, 0]) for r in regions if len((r.get("bbox") or [])) >= 4]
                if rbboxes:
                    screen_size = (max(b[2] for b in rbboxes), max(b[3] for b in rbboxes))
            atoms, recovery_log = run_input_candidate_recovery(
                atoms, raw_ocr_boxes, regions, screen_size, image_path=str(path),
                form_regions=form_regions, dark_theme=dark_theme,
            )
            log.extend(recovery_log)
        except Exception as rec_e:
            log.append("input_candidate_recovery skipped: %s" % rec_e)
    else:
        log.append("input_candidate_recovery: skip (layout inferred %d fields, use as primary)" % len(layout_inferred_atoms))

    # Experimental multilevel v2 (side-path): от вершины к листьям, без изменения baseline
    if use_experimental_multilevel_v2:
        try:
            from src.infrastructure.atoms_v2.experimental_v2 import run_experimental_multilevel_v2
            existing_ids = {a.get("id", "") for a in atoms if a.get("id")}
            exp_v2_atoms, exp_v2_log, _ = run_experimental_multilevel_v2(
                str(path),
                raw_ocr_boxes=raw_ocr_boxes,
                dark_theme=dark_theme,
                debug_output_dir=experimental_v2_debug_dir,
                existing_atom_ids=existing_ids,
            )
            log.extend(exp_v2_log)
            if exp_v2_atoms:
                atoms = atoms + exp_v2_atoms
                log.append("experimental_multilevel_v2: added %d atoms (side-path)" % len(exp_v2_atoms))
        except Exception as exp_e:
            log.append("experimental_multilevel_v2 skipped: %s" % exp_e)

    # Form Container First (ТЗ): FormContainerDetector → FormInnerLayout → SlotDetector → FieldLocator → FormGraph
    if use_form_container_first:
        try:
            from src.infrastructure.atoms_v2.experimental_v2.run_form_container_first_inference import run_form_container_first_inference
            existing_ids = {a.get("id", "") for a in atoms if a.get("id")}
            detectron_regions = None
            if regions:
                detectron_regions = [{"bbox": r.get("bbox")} for r in regions if len((r.get("bbox") or [])) >= 4]
            fcf_atoms, fcf_log, _ = run_form_container_first_inference(
                str(path),
                raw_ocr_boxes=raw_ocr_boxes,
                dark_theme=dark_theme,
                debug_output_dir=form_container_first_debug_dir,
                existing_atom_ids=existing_ids,
                detectron_regions=detectron_regions,
                demo_mode=demo_mode,
            )
            log.extend(fcf_log)
            if fcf_atoms:
                atoms = atoms + fcf_atoms
                log.append("form_container_first: added %d atoms (side-path)" % len(fcf_atoms))
        except Exception as fcf_e:
            log.append("form_container_first skipped: %s" % fcf_e)

    # v3: input_probe_prepass_v3 — подготовка OCR до semantic_validation без назначения ролей
    try:
        from src.infrastructure.atoms_v2.input_probe_prepass_v3 import input_probe_prepass_v3
        input_probe_prepass_v3(atoms, raw_ocr_boxes)
        log.append("input_probe_prepass_v3 applied (probe_ocr_text/len for input candidates)")
    except Exception as probe_e:
        log.append("input_probe_prepass_v3 skipped: %s" % probe_e)

    # --- Интеграция CatBoost v2 как soft-prior (атомы не удаляются до semantic_validation) ---
    # 1. postprocess (выше)
    # 2. input_probe_prepass_v3 (выше) — probe_ocr_text/len для кандидатов input
    # 3. build_ui_graph(all atoms) → extract_features
    # 3. run_catboost_priors → atom["priors"]["interactive_score"], atom["priors"]["role_probs"]; не фильтрует, ui_role не назначает
    # 4. group_atoms → atom_groups (row_bucket, area_bucket, aspect)
    # 5. semantic_validation(atom_groups) — rule-based + ML priors; propagation ролей в группе
    # 6. input_bbox_prepass — только type in (input, weak_input), после semantic_validation
    # 7. run_ui_graph_pipeline → final role assignment
    # 8. dedup (image_id, atom_id)
    # 9. dataset_builder (label_quality: semantic/weak/teacher)
    atom_groups: Dict[str, List[str]] = {}
    try:
        from src.infrastructure.ui_graph.build import build_ui_graph
        from src.infrastructure.ui_graph.features import extract_features
        from src.infrastructure.ui_graph.catboost_priors import run_catboost_priors
        from src.infrastructure.atoms_v2.group_atoms import group_atoms as group_atoms_fn
        graph_all = build_ui_graph(atoms, raw_ocr_boxes, regions)
        features_all = extract_features(graph_all)
        run_catboost_priors(atoms, features_all)
        atom_groups = group_atoms_fn(atoms)
        log.append(f"catboost_priors and group_atoms applied (atoms={len(atoms)}, groups={len(atom_groups)})")
    except Exception as priors_e:
        logger.debug("atoms_v2: catboost_priors/group_atoms skipped: %s", priors_e)

    # Семантическая валидация (v3): единственный слой, назначающий semantic_role и semantic_lock
    from src.infrastructure.atoms_v2.semantic_validation import run_semantic_validation
    semantic_log, semantic_stats = run_semantic_validation(
        atoms, raw_ocr_boxes, regions, require_effect=True, atom_groups=atom_groups,
    )
    log.extend(semantic_log)
    saved_by_anchor_ids = set(
        (semantic_stats.get("saved_by_anchor") or {}).get("input", [])
        + (semantic_stats.get("saved_by_anchor") or {}).get("button", [])
    )
    for a in atoms:
        if a.get("id") in saved_by_anchor_ids:
            a["saved_by_anchor"] = True

    # v3: atoms_for_interaction — только semantic_lock; layout/text/noise не участвуют в ui_graph
    atoms_for_interaction = [a for a in atoms if a.get("semantic_lock")]
    log.append("atoms_v3: atoms_for_interaction=%s (semantic_lock)" % len(atoms_for_interaction))

    # Уточнение границ input (label cut, snap to edges, width normalize) — как у кнопок
    try:
        from src.infrastructure.atoms_v2.input_bbox_refinement import refine_input_bbox_like_button
        input_types = ("input", "weak_input")
        input_atoms = [a for a in atoms if (a.get("type") or "").lower() in input_types and a.get("bbox") and len(a.get("bbox", [])) >= 4]
        other_bboxes = [a.get("bbox") for a in input_atoms if a.get("bbox") and len(a.get("bbox", [])) >= 4]
        refined_count = 0
        for a in input_atoms:
            bbox = a.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            others = [b for b in other_bboxes if b != bbox]
            try:
                import cv2
                img = cv2.imread(str(path))
                img_shape = img.shape[:2] if img is not None else None
            except Exception:
                img_shape = None
            new_bbox = refine_input_bbox_like_button(
                list(bbox),
                image_path=str(path),
                raw_ocr_boxes=raw_ocr_boxes,
                other_field_bboxes_in_card=others,
                img_shape=img_shape,
            )
            if new_bbox != bbox:
                a["bbox"] = new_bbox
                refined_count += 1
        if refined_count:
            log.append("input_bbox_refinement: %d input bboxes refined (label cut, edges, width)" % refined_count)
    except Exception as ref_e:
        logger.debug("atoms_v2: input_bbox_refinement skipped: %s", ref_e)

    # Input bbox prepass: только для атомов с interactive_valid и type in (input, weak_input)
    try:
        from src.infrastructure.ui_graph.catboost_priors import input_bbox_prepass
        input_bbox_prepass(atoms, raw_ocr_boxes)
    except Exception as prepass_e:
        logger.debug("atoms_v2: input_bbox_prepass skipped: %s", prepass_e)

    # UI-граф v3: только структура; работает ТОЛЬКО с atoms_for_interaction; не назначает роли
    try:
        from src.infrastructure.ui_graph.build import run_ui_graph_pipeline_v3
        from src.infrastructure.ui_graph.debug import debug_per_atom_log, debug_stats, debug_graph_summary
        if not atoms_for_interaction:
            final_atoms, ui_graph, features_by_atom, ui_log, ui_stats, role_predictions = [], None, {}, ["ui_graph_v3: no atoms_for_interaction"], {}, {}
        else:
            final_atoms, ui_graph, features_by_atom, ui_log, ui_stats, role_predictions = run_ui_graph_pipeline_v3(
                atoms_for_interaction, raw_ocr_boxes, regions,
            )
        log.extend(ui_log)
        if ui_graph is not None:
            log.extend(debug_graph_summary(ui_graph))
            log.extend(debug_per_atom_log(ui_graph, role_predictions, features_by_atom))
        ds = debug_stats(atoms, final_atoms, role_predictions)
        log.append(
            "ui_graph v3 stats: before=%s after=%s overrides=%s"
            % (ds["before_count"], ds["after_count"], ds.get("override_counts", {}))
        )
        image_id = path.stem
        # v3: роли назначены только в semantic_validation; ui_graph только копирует semantic_role → ui_role
        for a in final_atoms:
            a["image_id"] = image_id
        try:
            from src.infrastructure.ui_graph.catboost_predictor import deduplicate_atoms_by_image_atom
            final_atoms = deduplicate_atoms_by_image_atom(final_atoms, image_id_key="image_id")
            log.append("atoms_v2: dedup by (image_id, atom_id): %s atoms" % len(final_atoms))
        except Exception as dedup_e:
            logger.debug("atoms_v2: dedup skipped: %s", dedup_e)
        atoms = final_atoms
        try:
            from src.infrastructure.ui_graph.dataset_builder import collect_catboost_dataset
            ds_stats = collect_catboost_dataset(
                atoms,
                features_by_atom,
                image_id=image_id,
                output_path="datasets/ui_atoms_catboost.csv",
            )
            log.append(
                "dataset_builder: added_total=%s semantic=%s weak=%s skipped=%s"
                % (ds_stats["added_total"], ds_stats["added_semantic"], ds_stats["added_weak"], ds_stats["skipped"])
            )
        except Exception as ds_e:
            logger.warning("atoms_v2: dataset_builder failed: %s", ds_e)
    except Exception as e:
        logger.warning("atoms_v2: ui_graph failed, using atoms as-is: %s", e)
        # v3: при падении ui_graph всё равно выставляем ui_role = semantic_role у semantic_lock атомов
        for a in atoms:
            if a.get("semantic_lock"):
                a["ui_role"] = a.get("semantic_role") or a.get("type") or ""
                a["ui_role_confidence"] = 0.6
        atoms = [a for a in atoms if a.get("semantic_lock")]

    # v3: в UI участвуют только semantic_lock атомы
    atoms_for_ui = _atoms_participating_in_ui(atoms)
    log.append(f"atoms_participating_ui={len(atoms_for_ui)}")
    atom_to_region = _assign_atoms_to_regions(atoms_for_ui, regions)
    # Связь OCR↔атом только внутри одного CV-региона (ограничение области для связи)
    text_ui_links = _link_ocr_to_atoms(
        raw_ocr_boxes, atoms_for_ui, threshold=TEXT_INSIDE_ATOM_IOU_THRESHOLD,
        regions=regions, atom_to_region=atom_to_region,
    )
    text_inside_ui = _text_inside_ui_from_links(text_ui_links, raw_ocr_boxes)
    unified_ui = _build_logical_ui(atoms_for_ui, regions, atom_to_region, text_blocks, text_inside_ui)
    log.append(f"unified_ui_nodes={len(unified_ui)}")

    debug_image_path: Optional[str] = None
    try:
        from src.infrastructure.debug import save_debug_image_atoms_v2
        debug_image_path = save_debug_image_atoms_v2(
            image_path,
            regions,
            atoms,  # уже стабилизированный список (CV + synthetic)
            f"atoms_v2_{path.stem}.png",
            raw_ocr_boxes=raw_ocr_boxes,
            text_ui_links=text_ui_links,
            lines=lines,
            independent_text_blocks=independent_text_blocks,
        )
        if debug_image_path:
            log.append(f"debug_image={debug_image_path}")
    except Exception as e:
        logger.warning("atoms_v2: failed to save debug image: %s", e)

    return {
        "unified_ui": unified_ui,
        "atoms": atoms,
        "regions": regions,
        "atom_to_region": atom_to_region,
        "raw_ocr_boxes": raw_ocr_boxes,
        "text_ui_links": text_ui_links,
        "text_blocks": text_blocks,
        "independent_text_blocks": independent_text_blocks,
        "lines": lines,
        "paragraphs": paragraphs,
        "text_inside_ui": text_inside_ui,
        "log": log,
        "debug_image_path": debug_image_path,
    }
