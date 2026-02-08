"""
Detector merge + post-processing: онтологические инварианты UI.

Инвариант (Правка №1, №3): source == "real" → тип НИКОГДА не понижается из-за OCR/конфликтов.
OCR и synthetic — вспомогательные сигналы; real типы не деградируют в *_candidate.

- merge: synthetic матчится с real по IoU; при совпадении — real bbox и тип. Synthetic button не побеждает real.
- Стабилизация: конфликты → приоритет ролей (input ∩ button → input проигрывает, Правка №5); to_downgrade исключает real.
- Инварианты по типам применяются только к synthetic/synthetic_only; real пропускаются.
- Confidence — вторичный сигнал. Bbox не меняются.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

MERGE_IOU_THRESHOLD = 0.6
MERGE_IOU_THRESHOLD_BUTTON = 0.3  # button–button: при IoU ≥ 0.3 всегда берём real bbox
INPUT_LABEL_OFFSET_PX = 40
INPUT_LABEL_OVERLAP_AXIS = 0.5
INPUT_OCR_INSIDE_COVERAGE = 0.1
INPUT_VS_INPUT_OVERLAP_RATIO = 0.0  # input пересекается с input > 0% → candidate
LINK_OCR_COVERAGE = 0.4
LINK_OCR_IOU_MIN = 0.5  # link допустим, если хотя бы один OCR имеет IoU(link_bbox, ocr_bbox) >= это
LINK_BBOX_VS_OCR_MAX = 2.0  # link bbox не более чем в 2× OCR внутри
BUTTON_BBOX_VS_OCR_MAX = 6.0  # button по контуру, не по тексту: допускаем bbox до 6× OCR (контур может быть больше текста)
BUTTON_VS_LINK_OVERLAP_RATIO = 0.0  # button пересекается с link > 0% → button теряет
INPUT_VS_BUTTON_OVERLAP_RATIO = 0.0  # input пересекается с button > 0% → input теряет
CONTAINMENT_COVERAGE = 0.5  # A "содержит" B если B покрыт A на ≥ 50%
BUTTON_REGION_AREA_RATIO = 0.2
BUTTON_ASPECT_RATIO_MAX = 25.0  # длинные кнопки допускаются (отображать снова)
BUTTON_ASPECT_RATIO_MIN = 0.2
ATOM_REGION_MIN_IOU = 0.15
# Pagination/filters: 2+ OCR на одной горизонтали, шаг по X равномерный → container_candidate
PAGINATION_MIN_OCR_IN_ROW = 2
PAGINATION_MAX_Y_VARIANCE_RATIO = 0.3  # OCR на одной линии (Y в пределах 30% высоты bbox)


def _bbox_area(bbox: List[float]) -> float:
    if len(bbox) < 4:
        return 0.0
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return max(0.0, w * h)


def _intersection_area(a: List[float], b: List[float]) -> float:
    if len(a) < 4 or len(b) < 4:
        return 0.0
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def _iou_bbox_bbox(a: List[float], b: List[float]) -> float:
    if len(a) < 4 or len(b) < 4:
        return 0.0
    inter = _intersection_area(a, b)
    area_a = _bbox_area(a)
    area_b = _bbox_area(b)
    union = area_a + area_b - inter
    return inter / max(1e-9, union)


def _coverage_bbox_in_bbox(inner: List[float], outer: List[float]) -> float:
    """Доля площади inner, попадающая в outer."""
    if len(inner) < 4 or len(outer) < 4:
        return 0.0
    area_inner = _bbox_area(inner)
    if area_inner <= 0:
        return 0.0
    inter = _intersection_area(inner, outer)
    return inter / area_inner


def _outer_contains_inner(outer: List[float], inner: List[float], min_coverage: float = CONTAINMENT_COVERAGE) -> bool:
    """outer «содержит» inner, если inner покрыт outer на ≥ min_coverage."""
    return _coverage_bbox_in_bbox(inner, outer) >= min_coverage


def _normalize_atom(a: Dict[str, Any], source: str) -> Dict[str, Any]:
    return {
        "id": a.get("id", ""),
        "source": source,
        "type": a.get("type", "unknown"),
        "bbox": list(a.get("bbox", [0, 0, 0, 0])),
        "confidence": float(a.get("confidence", 0)),
    }


def merge_atoms(
    atoms_real: List[Dict[str, Any]],
    atoms_synthetic: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Слияние: synthetic матчится с real по IoU ≥ порогу; при совпадении — real bbox и тип."""
    out: List[Dict[str, Any]] = []
    used_real: Set[int] = set()
    used_synthetic: Set[int] = set()

    for si, satom in enumerate(atoms_synthetic):
        sbbox = satom.get("bbox", [0, 0, 0, 0])
        if len(sbbox) < 4:
            continue
        stype = satom.get("type", "")
        best_ri: Optional[int] = None
        best_iou = 0.0
        for ri, ratom in enumerate(atoms_real):
            if ri in used_real:
                continue
            rbbox = ratom.get("bbox", [0, 0, 0, 0])
            if len(rbbox) < 4:
                continue
            rtype = ratom.get("type", "")
            iou = _iou_bbox_bbox(sbbox, rbbox)
            threshold = MERGE_IOU_THRESHOLD_BUTTON if (stype == "button" and rtype == "button") else MERGE_IOU_THRESHOLD
            if iou >= threshold and iou > best_iou:
                best_iou = iou
                best_ri = ri
        if best_ri is not None:
            ratom = atoms_real[best_ri]
            used_real.add(best_ri)
            used_synthetic.add(si)
            rtype = ratom.get("type", "unknown")
            stype = satom.get("type", "unknown")
            final_type = rtype if rtype != stype else rtype
            merged_conf = max(float(ratom.get("confidence", 0)), float(satom.get("confidence", 0)))
            out.append({
                "id": ratom.get("id", f"merged_r_{best_ri}"),
                "source": "real",
                "type": final_type,
                "bbox": list(ratom.get("bbox", [0, 0, 0, 0])),
                "confidence": merged_conf,
            })

    for ri, ratom in enumerate(atoms_real):
        if ri in used_real:
            continue
        out.append(_normalize_atom(ratom, "real"))

    for si, satom in enumerate(atoms_synthetic):
        if si in used_synthetic:
            continue
        a = _normalize_atom(satom, "synthetic_only")
        a["confidence"] = min(1.0, a["confidence"] * 0.5)
        out.append(a)

    return out


# --- OCR и геометрия ---


def _has_ocr_inside_bbox(
    atom_bbox: List[float],
    raw_ocr_boxes: List[Dict[str, Any]],
    min_coverage: float = INPUT_OCR_INSIDE_COVERAGE,
) -> bool:
    if len(atom_bbox) < 4:
        return False
    for ob in raw_ocr_boxes:
        obbox = ob.get("bbox", [0, 0, 0, 0])
        if len(obbox) < 4:
            continue
        if _coverage_bbox_in_bbox(obbox, atom_bbox) >= min_coverage:
            return True
    return False


def _ocr_area_inside_bbox(atom_bbox: List[float], raw_ocr_boxes: List[Dict[str, Any]]) -> float:
    """Сумма площадей пересечений OCR с atom_bbox (учёт двойного подсчёта не нужен для отношения)."""
    if len(atom_bbox) < 4:
        return 0.0
    total = 0.0
    for ob in raw_ocr_boxes:
        obbox = ob.get("bbox", [0, 0, 0, 0])
        if len(obbox) < 4:
            continue
        total += _intersection_area(obbox, atom_bbox)
    return total


def _has_aligned_label(
    atom_bbox: List[float],
    raw_ocr_boxes: List[Dict[str, Any]],
    max_offset_px: float = INPUT_LABEL_OFFSET_PX,
    min_axis_overlap: float = INPUT_LABEL_OVERLAP_AXIS,
) -> bool:
    """Label = OCR слева с overlap по Y ≥ 50% высоты OCR, или сверху с overlap по X ≥ 50% ширины OCR, зазор ≤ max_offset_px."""
    if len(atom_bbox) < 4:
        return False
    x1, y1, x2, y2 = atom_bbox[0], atom_bbox[1], atom_bbox[2], atom_bbox[3]
    for ob in raw_ocr_boxes:
        obbox = ob.get("bbox", [0, 0, 0, 0])
        if len(obbox) < 4:
            continue
        ox1, oy1, ox2, oy2 = obbox[0], obbox[1], obbox[2], obbox[3]
        ocr_h = oy2 - oy1
        ocr_w = ox2 - ox1
        if ocr_h <= 0 or ocr_w <= 0:
            continue
        if ox2 <= x1 and (x1 - ox2) <= max_offset_px:
            overlap_y = min(oy2, y2) - max(oy1, y1)
            if overlap_y >= min_axis_overlap * ocr_h:
                return True
        if oy2 <= y1 and (y1 - oy2) <= max_offset_px:
            overlap_x = min(ox2, x2) - max(ox1, x1)
            if overlap_x >= min_axis_overlap * ocr_w:
                return True
    return False


def _atom_region(atom_bbox: List[float], regions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if len(atom_bbox) < 4:
        return None
    best_r: Optional[Dict[str, Any]] = None
    best_iou = 0.0
    for r in regions:
        rbbox = r.get("bbox", [0, 0, 0, 0])
        if len(rbbox) < 4:
            continue
        iou = _iou_bbox_bbox(atom_bbox, rbbox)
        if iou > best_iou:
            best_iou = iou
            best_r = r
    return best_r


def _atom_overlaps_region(atom_bbox: List[float], regions: List[Dict[str, Any]], min_iou: float = ATOM_REGION_MIN_IOU) -> bool:
    r = _atom_region(atom_bbox, regions)
    if r is None:
        return False
    rbbox = r.get("bbox", [0, 0, 0, 0])
    return len(rbbox) >= 4 and _iou_bbox_bbox(atom_bbox, rbbox) >= min_iou


def _button_ocr_in_row(atom_bbox: List[float], raw_ocr_boxes: List[Dict[str, Any]]) -> bool:
    """В bbox 2+ OCR на одной горизонтальной линии с похожим шагом → группа controls (pagination/filters), не одна кнопка."""
    if len(atom_bbox) < 4:
        return False
    inside: List[List[float]] = []
    for ob in raw_ocr_boxes:
        obbox = ob.get("bbox", [0, 0, 0, 0])
        if len(obbox) < 4:
            continue
        if _coverage_bbox_in_bbox(obbox, atom_bbox) < 0.3:
            continue
        inside.append(obbox)
    if len(inside) < PAGINATION_MIN_OCR_IN_ROW:
        return False
    h = atom_bbox[3] - atom_bbox[1]
    if h <= 0:
        return False
    y_centers = [(b[1] + b[3]) / 2 for b in inside]
    y_var = max(y_centers) - min(y_centers) if y_centers else 0
    if y_var > PAGINATION_MAX_Y_VARIANCE_RATIO * h:
        return False
    return True


def _synthetic_only_structurally_confirmed(
    atom: Dict[str, Any],
    atoms: List[Dict[str, Any]],
    raw_ocr_boxes: List[Dict[str, Any]],
    regions: List[Dict[str, Any]],
) -> bool:
    if atom.get("source") != "synthetic_only":
        return True
    bbox = atom.get("bbox", [0, 0, 0, 0])
    t = atom.get("type", "")
    if _atom_overlaps_region(bbox, regions, ATOM_REGION_MIN_IOU):
        return True
    if t == "input":
        return _has_ocr_inside_bbox(bbox, raw_ocr_boxes) or _has_aligned_label(bbox, raw_ocr_boxes)
    if t == "button":
        return _has_ocr_inside_bbox(bbox, raw_ocr_boxes, min_coverage=0.15)
    if t == "link":
        ocr_inside = _ocr_area_inside_bbox(bbox, raw_ocr_boxes)
        return ocr_inside > 0 and _bbox_area(bbox) / max(ocr_inside, 1e-9) <= LINK_BBOX_VS_OCR_MAX
    return False


# --- Фаза 1: конфликты и вложенность → приоритет ролей (confidence не используется) ---


def _collect_intersecting_pairs(atoms: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    """Пары индексов (i, j), i < j, с пересечением bbox > 0."""
    pairs: List[Tuple[int, int]] = []
    semantic = ("input", "button", "link")
    for i in range(len(atoms)):
        if atoms[i].get("type") not in semantic:
            continue
        bbox_i = atoms[i].get("bbox", [0, 0, 0, 0])
        if len(bbox_i) < 4:
            continue
        for j in range(i + 1, len(atoms)):
            if atoms[j].get("type") not in semantic:
                continue
            bbox_j = atoms[j].get("bbox", [0, 0, 0, 0])
            if len(bbox_j) < 4:
                continue
            if _intersection_area(bbox_i, bbox_j) <= 0:
                continue
            pairs.append((i, j))
    return pairs


def _resolve_conflicts(atoms: List[Dict[str, Any]]) -> Set[int]:
    """
    Приоритет ролей (односторонний, Правка №5): input ∩ button → input всегда проигрывает.
    button ∩ link → button теряет; input ∩ link → input теряет. Link содержит button/input → link теряет.
    Возвращает индексы для понижения до candidate; real исключаются (Правка №1, №3).
    """
    to_downgrade: Set[int] = set()
    pairs = _collect_intersecting_pairs(atoms)
    for i, j in pairs:
        ti = atoms[i].get("type", "")
        tj = atoms[j].get("type", "")
        bi = atoms[i].get("bbox", [0, 0, 0, 0])
        bj = atoms[j].get("bbox", [0, 0, 0, 0])
        if len(bi) < 4 or len(bj) < 4:
            continue
        if ti == "input" and tj == "button":
            to_downgrade.add(i)
            continue
        if ti == "button" and tj == "input":
            to_downgrade.add(j)
            continue
        if ti == "button" and tj == "link":
            to_downgrade.add(i)
            continue
        if ti == "link" and tj == "button":
            to_downgrade.add(j)
            continue
        if ti == "input" and tj == "link":
            to_downgrade.add(i)
            continue
        if ti == "link" and tj == "input":
            to_downgrade.add(j)
            continue
        if ti == "link" and tj in ("button", "input"):
            if _outer_contains_inner(bi, bj):
                to_downgrade.add(i)
            continue
        if tj == "link" and ti in ("button", "input"):
            if _outer_contains_inner(bj, bi):
                to_downgrade.add(j)
            continue
    # Real-детекции никогда не понижаются: synthetic не может побеждать real (Правка №1, №3)
    return {i for i in to_downgrade if atoms[i].get("source") != "real"}


# --- Фаза 2: инварианты по типам ---


def _enforce_input_invariants(
    atoms: List[Dict[str, Any]],
    raw_ocr_boxes: List[Dict[str, Any]],
    regions: List[Dict[str, Any]],
    conflict_indices: Set[int],
) -> None:
    """Input: не может без label при source != real; не может пересекаться с button. Real input без CV region в окрестности — фантом «после label» → layout_candidate."""
    for i, a in enumerate(atoms):
        if a.get("type") != "input":
            continue
        bbox = a.get("bbox", [0, 0, 0, 0])
        if len(bbox) < 4:
            continue
        # Real input без CV region в окрестности (IoU ≥ 0.1) — фантом от модели «после label»
        if a.get("source") == "real":
            if not _atom_overlaps_region(bbox, regions, min_iou=0.1):
                a["type"] = "layout_candidate"
                a["confidence"] = min(1.0, (a.get("confidence", 0) or 0) * 0.3)
                logger.debug("invariant input (real, no CV region) -> layout_candidate: id=%s", a.get("id"))
            continue
        if i in conflict_indices:
            a["type"] = "layout_candidate"
            a["confidence"] = min(1.0, (a.get("confidence", 0) or 0) * 0.3)
            logger.debug("invariant input (conflict) -> layout_candidate: id=%s", a.get("id"))
            continue
        if not _has_ocr_inside_bbox(bbox, raw_ocr_boxes) and not _has_aligned_label(bbox, raw_ocr_boxes):
            a["type"] = "layout_candidate"
            a["confidence"] = min(1.0, (a.get("confidence", 0) or 0) * 0.3)
            logger.debug("invariant input (no label) -> layout_candidate: id=%s", a.get("id"))
            continue
        if _input_overlaps_other_input(a, atoms):
            a["type"] = "layout_candidate"
            a["confidence"] = min(1.0, (a.get("confidence", 0) or 0) * 0.3)
            logger.debug("invariant input (overlap input) -> layout_candidate: id=%s", a.get("id"))


def _input_overlaps_other_input(atom: Dict[str, Any], atoms: List[Dict[str, Any]], min_ratio: float = INPUT_VS_INPUT_OVERLAP_RATIO) -> bool:
    bbox = atom.get("bbox", [0, 0, 0, 0])
    if len(bbox) < 4:
        return False
    area_a = _bbox_area(bbox)
    if area_a <= 0:
        return False
    for other in atoms:
        if other is atom or other.get("type") != "input":
            continue
        obbox = other.get("bbox", [0, 0, 0, 0])
        if len(obbox) < 4:
            continue
        if _intersection_area(bbox, obbox) / area_a > min_ratio:
            return True
    return False


def _enforce_button_invariants(
    atoms: List[Dict[str, Any]],
    raw_ocr_boxes: List[Dict[str, Any]],
    regions: List[Dict[str, Any]],
    conflict_indices: Set[int],
) -> None:
    """Button: не может быть без OCR при source != real; не группа pagination. Real — тип никогда не понижается (Правка №1, №3)."""
    for i, a in enumerate(atoms):
        if a.get("type") != "button":
            continue
        if a.get("source") == "real":
            continue  # Real: никогда смена типа на *_candidate; synthetic только fallback
        bbox = a.get("bbox", [0, 0, 0, 0])
        if len(bbox) < 4:
            continue
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if h <= 0:
            continue
        aspect = w / h
        if aspect > BUTTON_ASPECT_RATIO_MAX or aspect < BUTTON_ASPECT_RATIO_MIN:
            a["type"] = "container_candidate"
            a["confidence"] = min(1.0, (a.get("confidence", 0) or 0) * 0.3)
            logger.debug("invariant button (aspect) -> container_candidate: id=%s", a.get("id"))
            continue
        if i in conflict_indices:
            a["type"] = "container_candidate"
            a["confidence"] = min(1.0, (a.get("confidence", 0) or 0) * 0.3)
            logger.debug("invariant button (conflict) -> container_candidate: id=%s", a.get("id"))
            continue
        if a.get("source") != "real" and not _has_ocr_inside_bbox(bbox, raw_ocr_boxes, min_coverage=0.15):
            a["type"] = "container_candidate"
            a["confidence"] = min(1.0, (a.get("confidence", 0) or 0) * 0.3)
            logger.debug("invariant button (no OCR) -> container_candidate: id=%s", a.get("id"))
            continue
        ocr_inside = _ocr_area_inside_bbox(bbox, raw_ocr_boxes)
        if ocr_inside > 0:
            ratio = _bbox_area(bbox) / ocr_inside
            if ratio > BUTTON_BBOX_VS_OCR_MAX:
                a["type"] = "container_candidate"
                a["confidence"] = min(1.0, (a.get("confidence", 0) or 0) * 0.3)
                logger.debug("invariant button (bbox >> OCR) -> container_candidate: id=%s", a.get("id"))
                continue
        if _button_ocr_in_row(bbox, raw_ocr_boxes):
            a["type"] = "container_candidate"
            a["confidence"] = min(1.0, (a.get("confidence", 0) or 0) * 0.3)
            logger.debug("invariant button (pagination row) -> container_candidate: id=%s", a.get("id"))
            continue
        region = _atom_region(bbox, regions)
        if region:
            rbbox = region.get("bbox", [0, 0, 0, 0])
            if len(rbbox) >= 4:
                r_area = _bbox_area(rbbox)
                if r_area > 0 and _bbox_area(bbox) / r_area > BUTTON_REGION_AREA_RATIO:
                    a["type"] = "container_candidate"
                    a["confidence"] = min(1.0, (a.get("confidence", 0) or 0) * 0.3)
                    logger.debug("invariant button (huge vs region) -> container_candidate: id=%s", a.get("id"))


def _enforce_link_invariants(
    atoms: List[Dict[str, Any]],
    raw_ocr_boxes: List[Dict[str, Any]],
    conflict_indices: Set[int],
) -> None:
    """Link: OCR-якорь; не может содержать button/input. Real — тип никогда не понижается (Правка №1)."""
    for i, a in enumerate(atoms):
        if a.get("type") != "link":
            continue
        if a.get("source") == "real":
            continue  # Real: никогда смена типа на *_candidate
        bbox = a.get("bbox", [0, 0, 0, 0])
        if len(bbox) < 4:
            continue
        if i in conflict_indices:
            a["type"] = "inline_text_candidate"
            a["confidence"] = min(1.0, (a.get("confidence", 0) or 0) * 0.3)
            logger.debug("invariant link (conflict) -> inline_text_candidate: id=%s", a.get("id"))
            continue
        ocr_inside = _ocr_area_inside_bbox(bbox, raw_ocr_boxes)
        link_area = _bbox_area(bbox)
        if link_area <= 0:
            continue
        if ocr_inside <= 0:
            a["type"] = "inline_text_candidate"
            a["confidence"] = min(1.0, (a.get("confidence", 0) or 0) * 0.3)
            logger.debug("invariant link (no OCR) -> inline_text_candidate: id=%s", a.get("id"))
            continue
        if link_area / ocr_inside > LINK_BBOX_VS_OCR_MAX:
            a["type"] = "inline_text_candidate"
            a["confidence"] = min(1.0, (a.get("confidence", 0) or 0) * 0.3)
            logger.debug("invariant link (bbox >> OCR) -> inline_text_candidate: id=%s", a.get("id"))
            continue
        best_cov = 0.0
        best_iou = 0.0
        for ob in raw_ocr_boxes:
            obbox = ob.get("bbox", [0, 0, 0, 0])
            if len(obbox) < 4:
                continue
            cov = _coverage_bbox_in_bbox(obbox, bbox)
            if cov > best_cov:
                best_cov = cov
            iou = _iou_bbox_bbox(bbox, obbox)
            if iou > best_iou:
                best_iou = iou
        # Link допустим только если OCR-якорь: coverage ≥ 40% площади OCR ИЛИ IoU(link, ocr) ≥ 0.5
        if best_iou < LINK_OCR_IOU_MIN and best_cov < LINK_OCR_COVERAGE:
            a["type"] = "inline_text_candidate"
            a["confidence"] = min(1.0, (a.get("confidence", 0) or 0) * 0.3)
            logger.debug("invariant link (no OCR anchor: cov<40%% or iou<0.5) -> inline_text_candidate: id=%s", a.get("id"))


def _enforce_synthetic_only_unconfirmed(
    atoms: List[Dict[str, Any]],
    raw_ocr_boxes: List[Dict[str, Any]],
    regions: List[Dict[str, Any]],
    conflict_indices: Set[int],
) -> None:
    """Synthetic_only без структурного подтверждения или в конфликте → candidate. В конфликте synthetic всегда деградирует."""
    for idx, a in enumerate(atoms):
        if a.get("source") != "synthetic_only":
            continue
        t = a.get("type", "")
        if t not in ("button", "input", "link"):
            continue
        if idx in conflict_indices:
            if t == "input":
                a["type"] = "layout_candidate"
            elif t == "button":
                a["type"] = "container_candidate"
            else:
                a["type"] = "inline_text_candidate"
            a["confidence"] = min(1.0, (a.get("confidence", 0) or 0) * 0.3)
            logger.debug("synthetic_only (conflict) -> %s: id=%s", a["type"], a.get("id"))
            continue
        if _synthetic_only_structurally_confirmed(a, atoms, raw_ocr_boxes, regions):
            continue
        if t == "input":
            a["type"] = "layout_candidate"
        elif t == "button":
            a["type"] = "container_candidate"
        else:
            a["type"] = "inline_text_candidate"
        a["confidence"] = min(1.0, (a.get("confidence", 0) or 0) * 0.3)
        logger.debug("synthetic_only (unconfirmed) -> %s: id=%s", a["type"], a.get("id"))


def stabilize_atoms(
    atoms_real: List[Dict[str, Any]],
    atoms_synthetic: List[Dict[str, Any]],
    raw_ocr_boxes: List[Dict[str, Any]],
    regions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Merge и стабилизация по онтологическим инвариантам.

    1. merge_atoms(real, synthetic).
    2. Конфликты и вложенность: пары с пересечением → приоритет ролей (input < button < link; link не содержит control; input не содержит button).
    3. Инварианты: input без label при !real → layout_candidate; button без OCR при !real и bbox >> OCR → container_candidate; link bbox > 2× OCR → inline_text_candidate; button = pagination row → container_candidate.
    4. Synthetic_only в конфликте или без подтверждения → candidate.

    Confidence — только вторично. Bbox не меняются.
    """
    merged = merge_atoms(atoms_real, atoms_synthetic)
    conflict_indices = _resolve_conflicts(merged)
    _enforce_input_invariants(merged, raw_ocr_boxes, regions, conflict_indices)
    _enforce_button_invariants(merged, raw_ocr_boxes, regions, conflict_indices)
    _enforce_link_invariants(merged, raw_ocr_boxes, conflict_indices)
    _enforce_synthetic_only_unconfirmed(merged, raw_ocr_boxes, regions, conflict_indices)
    return merged
