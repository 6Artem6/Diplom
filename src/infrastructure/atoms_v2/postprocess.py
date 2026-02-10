"""
Post-processing слой UI-атомов: только данные (bbox, типы, confidence, регионы).

CV-модель — генератор гипотез; истина формируется здесь:
- фильтрация ложных link (внутри параграфа, без контейнера);
- синтетические button/link из строк OCR (длинные кнопки);
- синтетические input по геометрии (пустая область + label рядом).

Не использует изображения. Не двигает/не растягивает bbox от CV.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# --- Пороги (только геометрия) ---
OCR_TO_REGION_IOU_THRESHOLD = 0.3
LINE_Y_THRESHOLD_PX = 12.0
LINE_X_GAP_MAX_PX = 35.0
MIN_OCR_BOXES_FOR_SYNTHETIC_LINE = 2
SYNTHETIC_OVERLAP_IOU_SKIP = 0.45
# Синтетическая кнопка не создаётся из длинной горизонтальной строки (label + поле — не кнопка)
SYNTHETIC_BTN_MAX_ASPECT = 8.0
SMALL_LINK_WIDTH_PX = 90.0
LINK_INSIDE_PARAGRAPH_AREA_RATIO = 0.85
INPUT_MIN_WIDTH_PX = 40.0
INPUT_MAX_HEIGHT_PX = 80.0
INPUT_LABEL_OFFSET_MAX_PX = 120.0
FORM_LIKE_MIN_OCR_IN_REGION = 2


def _bbox_area(bbox: List[float]) -> float:
    if len(bbox) < 4:
        return 0.0
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return max(0.0, w * h)


def _bbox_union(bboxes: List[List[float]]) -> List[float]:
    if not bboxes:
        return [0.0, 0.0, 0.0, 0.0]
    xs = [b[0] for b in bboxes if len(b) >= 4]
    ys = [b[1] for b in bboxes if len(b) >= 4]
    x2s = [b[2] for b in bboxes if len(b) >= 4]
    y2s = [b[3] for b in bboxes if len(b) >= 4]
    if not xs:
        return [0.0, 0.0, 0.0, 0.0]
    return [min(xs), min(ys), max(x2s), max(y2s)]


def _coverage_bbox_in_bbox(inner: List[float], outer: List[float]) -> float:
    """Доля площади inner, попадающая в outer. inner/outer [x1,y1,x2,y2]."""
    if len(inner) < 4 or len(outer) < 4:
        return 0.0
    area_inner = _bbox_area(inner)
    if area_inner <= 0:
        return 0.0
    ix1 = max(inner[0], outer[0])
    iy1 = max(inner[1], outer[1])
    ix2 = min(inner[2], outer[2])
    iy2 = min(inner[3], outer[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    return inter / area_inner


def _iou_bbox_bbox(a: List[float], b: List[float]) -> float:
    if len(a) < 4 or len(b) < 4:
        return 0.0
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = _bbox_area(a)
    area_b = _bbox_area(b)
    union = area_a + area_b - inter
    return inter / max(1e-9, union)


def _assign_ocr_to_regions(
    raw_ocr_boxes: List[Dict[str, Any]],
    regions: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """region_id -> список OCR-боксов, попадающих в регион (coverage >= порога)."""
    out: Dict[str, List[Dict[str, Any]]] = {r.get("id", ""): [] for r in regions}
    for ob in raw_ocr_boxes:
        obbox = ob.get("bbox", [0, 0, 0, 0])
        if len(obbox) < 4:
            continue
        best_rid: Optional[str] = None
        best_cov = 0.0
        for r in regions:
            rid = r.get("id", "")
            rbbox = r.get("bbox", [0, 0, 0, 0])
            cov = _coverage_bbox_in_bbox(obbox, rbbox)
            if cov >= OCR_TO_REGION_IOU_THRESHOLD and cov > best_cov:
                best_cov = cov
                best_rid = rid
        if best_rid:
            out.setdefault(best_rid, []).append(ob)
    return out


def _group_ocr_into_lines(
    ocr_boxes: List[Dict[str, Any]],
) -> List[List[Dict[str, Any]]]:
    """Группирует OCR-боксы в строки: одинаковый Y (в пределах порога), близко по X."""
    if not ocr_boxes:
        return []
    sorted_boxes = sorted(
        ocr_boxes,
        key=lambda b: (
            (b["bbox"][1] + b["bbox"][3]) / 2,
            b["bbox"][0],
        ),
    )
    lines: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = [sorted_boxes[0]]
    y_prev = (sorted_boxes[0]["bbox"][1] + sorted_boxes[0]["bbox"][3]) / 2
    x2_prev = sorted_boxes[0]["bbox"][2]

    for ob in sorted_boxes[1:]:
        bbox = ob.get("bbox", [0, 0, 0, 0])
        if len(bbox) < 4:
            continue
        y_cur = (bbox[1] + bbox[3]) / 2
        x1_cur = bbox[0]
        if abs(y_cur - y_prev) <= LINE_Y_THRESHOLD_PX and (x1_cur - x2_prev) <= LINE_X_GAP_MAX_PX:
            current.append(ob)
            x2_prev = bbox[2]
        else:
            if current:
                lines.append(current)
            current = [ob]
            y_prev = y_cur
            x2_prev = bbox[2]
    if current:
        lines.append(current)
    return lines


def _filter_false_links(
    atoms: List[Dict[str, Any]],
    independent_text_blocks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Для атомов типа link: если маленькая ширина и bbox полностью внутри параграфа —
    понижаем confidence и/или меняем тип на inline_link_candidate. Bbox не трогаем.
    """
    paragraph_bboxes = [
        blk["bbox"]
        for blk in independent_text_blocks
        if blk.get("type") == "paragraph" and len(blk.get("bbox", [])) >= 4
    ]
    if not paragraph_bboxes:
        return list(atoms)

    out: List[Dict[str, Any]] = []
    for a in atoms:
        if a.get("type") != "link":
            out.append(dict(a))
            continue
        bbox = a.get("bbox", [0, 0, 0, 0])
        if len(bbox) < 4:
            out.append(dict(a))
            continue
        width = bbox[2] - bbox[0]
        if width > SMALL_LINK_WIDTH_PX:
            out.append(dict(a))
            continue
        inside_paragraph = False
        for pbbox in paragraph_bboxes:
            cov = _coverage_bbox_in_bbox(bbox, pbbox)
            if cov >= LINK_INSIDE_PARAGRAPH_AREA_RATIO:
                inside_paragraph = True
                break
        if not inside_paragraph:
            out.append(dict(a))
            continue
        # Понижаем confidence и помечаем тип
        new_atom = dict(a)
        new_atom["confidence"] = min(1.0, (a.get("confidence", 0) or 0) * 0.5)
        new_atom["type"] = "inline_link_candidate"
        out.append(new_atom)
        logger.debug("False link downgraded: id=%s width=%.0f", a.get("id"), width)
    return out


def _synthetic_buttons_from_ocr(
    atoms: List[Dict[str, Any]],
    raw_ocr_boxes: List[Dict[str, Any]],
    regions: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Создаёт синтетические button/link из строк OCR в одном регионе (одна линия Y, близко X).
    Возвращает (список новых синтетических атомов, список новых text_ui_links для них).
    """
    ocr_by_region = _assign_ocr_to_regions(raw_ocr_boxes, regions)
    synthetic_atoms: List[Dict[str, Any]] = []
    new_links: List[Dict[str, Any]] = []
    synth_id = 0
    atom_types_for_overlap = ("button", "link")

    for rid, region_ocr in ocr_by_region.items():
        if len(region_ocr) < MIN_OCR_BOXES_FOR_SYNTHETIC_LINE:
            continue
        lines = _group_ocr_into_lines(region_ocr)
        for line in lines:
            if len(line) < MIN_OCR_BOXES_FOR_SYNTHETIC_LINE:
                continue
            line_bbox = _bbox_union([b["bbox"] for b in line])
            lw = line_bbox[2] - line_bbox[0]
            lh = line_bbox[3] - line_bbox[1]
            if lh <= 0 or lw / lh > SYNTHETIC_BTN_MAX_ASPECT:
                continue
            # Не дублируем: если уже есть CV-атом (button/link) с большим IoU — пропускаем
            skip = False
            for a in atoms:
                if a.get("type") not in atom_types_for_overlap:
                    continue
                abbox = a.get("bbox", [0, 0, 0, 0])
                if _iou_bbox_bbox(line_bbox, abbox) >= SYNTHETIC_OVERLAP_IOU_SKIP:
                    skip = True
                    break
            if skip:
                continue
            synth_id += 1
            aid = f"synthetic_btn_{synth_id}"
            synthetic_atoms.append({
                "id": aid,
                "source": "synthetic",
                "type": "button",
                "bbox": line_bbox,
                "confidence": 0.75,
            })
            for ob in line:
                new_links.append({
                    "ocr_box_id": ob.get("id", ""),
                    "atom_id": aid,
                    "link_type": "label",
                    "coverage_ratio": 1.0,
                })
    return synthetic_atoms, new_links


def _synthetic_inputs(
    atoms: List[Dict[str, Any]],
    raw_ocr_boxes: List[Dict[str, Any]],
    regions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Эвристика: в регионах с несколькими OCR-боксами ищем «пустые» прямоугольники
    (нет текста внутри), рядом с которыми есть label (OCR слева/сверху). Добавляем synthetic input.
    """
    ocr_by_region = _assign_ocr_to_regions(raw_ocr_boxes, regions)
    cv_input_ids = {a.get("id") for a in atoms if a.get("type") == "input"}
    synthetic: List[Dict[str, Any]] = []
    synth_id = 0

    for r in regions:
        rid = r.get("id", "")
        rbbox = r.get("bbox", [0, 0, 0, 0])
        if len(rbbox) < 4:
            continue
        region_ocr = ocr_by_region.get(rid, [])
        if len(region_ocr) < FORM_LIKE_MIN_OCR_IN_REGION:
            continue
        # Есть ли уже input в этом регионе от CV?
        region_atoms = [a for a in atoms if _coverage_bbox_in_bbox(a.get("bbox", [0,0,0,0]), rbbox) > 0.1]
        has_cv_input = any(a.get("type") == "input" for a in region_atoms)
        if has_cv_input:
            continue
        # Добавляем synthetic input только если в регионе есть форма (button), иначе — фантом «после label»
        has_button_in_region = any(a.get("type") == "button" for a in region_atoms)
        if not has_button_in_region:
            continue
        # Ищем короткие OCR как кандидаты в label (не абзац)
        for ob in region_ocr:
            obbox = ob.get("bbox", [0, 0, 0, 0])
            if len(obbox) < 4:
                continue
            text = (ob.get("text") or "").strip()
            if len(text) > 35:
                continue
            # Пустая зона справа от label: [ox2, oy1, ox2 + w, oy2], w в разумных пределах
            ox1, oy1, ox2, oy2 = obbox[0], obbox[1], obbox[2], obbox[3]
            h = oy2 - oy1
            if h > INPUT_MAX_HEIGHT_PX or h < 8:
                continue
            # Поле ввода справа: ограничиваем смещение
            gap_start_x = ox2
            gap_end_x = min(rbbox[2], gap_start_x + 400)
            if gap_end_x - gap_start_x < INPUT_MIN_WIDTH_PX:
                continue
            # Проверяем, что в [gap_start_x, oy1, gap_end_x, oy2] нет OCR (пустая зона)
            candidate_bbox = [gap_start_x, oy1, gap_end_x, oy2]
            has_text_inside = False
            for other in raw_ocr_boxes:
                othbox = other.get("bbox", [0, 0, 0, 0])
                if len(othbox) < 4:
                    continue
                if _iou_bbox_bbox(othbox, candidate_bbox) > 0.1:
                    has_text_inside = True
                    break
            if has_text_inside:
                continue
            synth_id += 1
            synthetic.append({
                "id": f"synthetic_input_{synth_id}",
                "source": "synthetic",
                "type": "input",
                "bbox": candidate_bbox,
                "confidence": 0.65,
            })
            break
    return synthetic


def run_postprocess(
    atoms: List[Dict[str, Any]],
    raw_ocr_boxes: List[Dict[str, Any]],
    text_ui_links: List[Dict[str, Any]],
    regions: List[Dict[str, Any]],
    independent_text_blocks: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Post-processing: стабилизированный список UI-атомов и обновлённые text_ui_links.

    1. Фильтр ложных link (внутри параграфа → понижение confidence, тип inline_link_candidate).
    2. Синтетические button из строк OCR (длинные кнопки).
    3. Синтетические input по геометрии (пустая область + label рядом).

    Входные atoms и bbox от CV не двигаются и не растягиваются.
    Возвращает (stabilized_atoms, updated_text_ui_links).
    """
    independent_text_blocks = independent_text_blocks or []
    # 1. Фильтр ложных ссылок (только confidence/type)
    stabilized = _filter_false_links(atoms, independent_text_blocks)
    # 2. Синтетические кнопки из OCR-строк
    syn_buttons, new_links_btn = _synthetic_buttons_from_ocr(stabilized, raw_ocr_boxes, regions)
    stabilized = stabilized + syn_buttons
    text_ui_links = list(text_ui_links) + new_links_btn
    # 3. Синтетические input
    syn_inputs = _synthetic_inputs(stabilized, raw_ocr_boxes, regions)
    stabilized = stabilized + syn_inputs
    if syn_buttons or syn_inputs:
        logger.info(
            "postprocess: +%d synthetic buttons, +%d synthetic inputs",
            len(syn_buttons),
            len(syn_inputs),
        )
    return stabilized, text_ui_links
