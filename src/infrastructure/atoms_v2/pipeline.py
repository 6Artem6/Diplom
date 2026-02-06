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
# Разрешённые типы: button, card, input, navbar, container, icon, checkbox, radio, dropdown, modal, text_block, unknown.
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
ATOM_IOU_THRESHOLD = 0.5
REGION_CONTAINMENT_MARGIN = 2


def _run_detectron2_atoms(image_path: str) -> List[Dict[str, Any]]:
    """
    Сырая карта UI: Detectron2 — только геометрия и визуальный тип.

    Инвариант CV-слоя: допустимы только id, source="detectron2", type, bbox, confidence.
    Запрещены: text, role, paragraph, label, lines_count, region_id.
    """
    path = Path(image_path)
    if not path.exists():
        return []
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
    atoms: List[Dict[str, Any]] = []
    for i, r in enumerate(raw):
        cls_name = r.get("class", "")
        atom_type = KVVB_CLASS_TO_ATOM_TYPE.get(cls_name, "text_block")
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


def run_atoms_v2_pipeline(
    image_path: str,
    parallel_ocr: bool = True,
    legacy_text_pipeline: bool = True,
) -> Dict[str, Any]:
    """
    Пайплайн atoms_v2: Detectron2 (atoms) + CV regions; OCR параллельно по всей странице и/или по регионам;
    merge & conflict resolution (текст не подавляется UI); legacy grouping (строки/абзацы).

    parallel_ocr=True: полностраничный OCR независимо от Detectron2; текст внутри input/button тоже извлекается.
    legacy_text_pipeline=True: группировка в строки и абзацы, фильтр шума.
    improved-full-pipeline не вызывается и не меняется.
    """
    log: List[str] = []
    path = Path(image_path)
    if not path.exists():
        return {
            "unified_ui": [], "atoms": [], "regions": [], "atom_to_region": {},
            "raw_ocr_boxes": [], "text_blocks": [], "independent_text_blocks": [], "lines": [], "paragraphs": [],
            "text_inside_ui": {}, "log": ["Image not found"], "debug_image_path": None,
        }

    atoms = _run_detectron2_atoms(image_path)
    log.append(f"atoms={len(atoms)}")
    regions = _run_cv_visual_regions(image_path)
    log.append(f"regions={len(regions)}")

    atom_to_region = _assign_atoms_to_regions(atoms, regions)

    full_page_boxes: List[Dict[str, Any]] = []
    raw_ocr_boxes: List[Dict[str, Any]] = []
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
        text_inside_ui = _merge_ocr_with_atoms(full_page_boxes, atoms)
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

    text_blocks = _merge_text_blocks(region_texts, regions)
    log.append(f"text_blocks={len(text_blocks)}")

    unified_ui = _build_logical_ui(atoms, regions, atom_to_region, text_blocks, text_inside_ui)
    log.append(f"unified_ui_nodes={len(unified_ui)}")

    debug_image_path: Optional[str] = None
    try:
        from src.infrastructure.debug import save_debug_image_atoms_v2
        debug_image_path = save_debug_image_atoms_v2(
            image_path,
            regions,
            atoms,
            f"atoms_v2_{path.stem}.png",
            raw_ocr_boxes=raw_ocr_boxes,
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
        "text_blocks": text_blocks,
        "independent_text_blocks": independent_text_blocks,
        "lines": lines,
        "paragraphs": paragraphs,
        "text_inside_ui": text_inside_ui,
        "log": log,
        "debug_image_path": debug_image_path,
    }
