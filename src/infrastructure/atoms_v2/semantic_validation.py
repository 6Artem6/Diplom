"""
Семантический слой: anchor (сам объект похож на input/button) + context (усилитель уверенности).

Anchor — разрешение на существование; context — усилитель, не обязательное условие.
if not anchor -> layout
elif anchor and not context -> type остаётся, confidence *= 0.6, semantic_valid = True
else -> semantic_valid = True

Input anchor: aspect ∈ [2, 15], height ∈ [18, 60] px, bbox не перекрыт текстом > 30% площади.
Button anchor: aspect ∈ [1.5, 30], area >= MIN_BUTTON_AREA (цвет от фона уже проверен в pipeline).
Container_candidate — жёстко: ≥2 semantic_valid неперекрывающихся атома; button/input только → layout.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

LABEL_OFFSET_PX = 40
LABEL_OVERLAP_AXIS = 0.5
OCR_INSIDE_COVERAGE = 0.2
SAME_LINE_Y_PX = 18
NEARBY_INPUT_PX = 80
CONTAINMENT_COVERAGE = 0.5
MIN_ATOMS_IN_CONTAINER = 2
MIN_BUTTONS_IN_ACTION_GROUP = 2
NO_EFFECT_THRESHOLD = 0.9

# Input anchor: визуальный input без обязательного контекста (расширено для input без label/формы)
INPUT_ANCHOR_ASPECT_MIN = 1.5
INPUT_ANCHOR_ASPECT_MAX = 20.0
INPUT_ANCHOR_HEIGHT_MIN_PX = 16
INPUT_ANCHOR_HEIGHT_MAX_PX = 70
INPUT_ANCHOR_TEXT_COVERAGE_MAX = 0.3  # bbox не перекрыт текстом более чем на 30% площади

# Button anchor: кликабельная область (цвет от фона уже проверен в pipeline)
BUTTON_ANCHOR_ASPECT_MIN = 1.5
BUTTON_ANCHOR_ASPECT_MAX = 30.0
MIN_BUTTON_AREA_PX = 100
# Кнопка без явных границ: текст должен быть по центру или с action-словом; иначе — label/поле
BUTTON_ASPECT_LABEL_LIKE_MAX = 6.0  # aspect > 6 и текст слева → не кнопка
ACTION_WORDS = frozenset({"search", "submit", "save", "login", "send", "go", "ok", "apply", "cancel", "create", "add"})

CONTEXT_CONFIDENCE_FACTOR = 0.6  # anchor без context → confidence *= это

LAYOUT_TYPE = "layout"

# Роли, которые дают semantic_lock (участвуют в interaction graph). layout/text/noise — никогда.
SEMANTIC_LOCK_TYPES = frozenset({
    "button", "weak_button", "input", "weak_input", "link", "checkbox", "textarea", "container_candidate",
})
# Порог покрытия: bbox B считается «внутри» bbox A, если доля площади B внутри A >= этого значения
BUTTON_ABSORB_COVERAGE_MIN = 0.5


def _bbox_area(bbox: List[float]) -> float:
    if len(bbox) < 4:
        return 0.0
    return max(0.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))


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


def _coverage_bbox_in_bbox(inner: List[float], outer: List[float]) -> float:
    if len(inner) < 4 or len(outer) < 4:
        return 0.0
    area_inner = _bbox_area(inner)
    if area_inner <= 0:
        return 0.0
    inter = _intersection_area(inner, outer)
    return inter / area_inner


def _ocr_fraction_of_bbox(bbox: List[float], raw_ocr_boxes: List[Dict[str, Any]]) -> float:
    """Доля площади bbox, покрытая OCR (перекрытие). > INPUT_ANCHOR_TEXT_COVERAGE_MAX → не input (текстовый блок)."""
    area_b = _bbox_area(bbox)
    if area_b <= 0:
        return 0.0
    total = 0.0
    for ob in raw_ocr_boxes:
        obbox = ob.get("bbox", [0, 0, 0, 0])
        if len(obbox) < 4:
            continue
        total += _intersection_area(obbox, bbox)
    return total / area_b


def _input_anchor_ok(
    bbox: List[float],
    raw_ocr_boxes: List[Dict[str, Any]],
) -> bool:
    """Input может быть input без формы/соседей, если: aspect ∈ [1.5, 20], height ∈ [16, 70] px, bbox не перекрыт текстом > 30%."""
    if len(bbox) < 4:
        return False
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    if h <= 0:
        return False
    aspect = w / h
    if aspect < INPUT_ANCHOR_ASPECT_MIN or aspect > INPUT_ANCHOR_ASPECT_MAX:
        return False
    if h < INPUT_ANCHOR_HEIGHT_MIN_PX or h > INPUT_ANCHOR_HEIGHT_MAX_PX:
        return False
    if _ocr_fraction_of_bbox(bbox, raw_ocr_boxes) > INPUT_ANCHOR_TEXT_COVERAGE_MAX:
        return False
    return True


def _button_anchor_ok(bbox: List[float]) -> bool:
    """Button может быть button сам по себе: aspect ∈ [1.5, 30], area >= MIN_BUTTON_AREA. Цвет от фона уже проверен в pipeline."""
    if len(bbox) < 4:
        return False
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    if h <= 0 or w <= 0:
        return False
    aspect = w / h
    if aspect < BUTTON_ANCHOR_ASPECT_MIN or aspect > BUTTON_ANCHOR_ASPECT_MAX:
        return False
    area = _bbox_area(bbox)
    if area < MIN_BUTTON_AREA_PX:
        return False
    return True


def _has_action_word_in_bbox(bbox: List[float], raw_ocr_boxes: List[Dict[str, Any]]) -> bool:
    """Есть ли внутри bbox OCR-текст с action-словом (Save, Cancel, Search, ...)."""
    for ob in raw_ocr_boxes:
        obbox = ob.get("bbox", [0, 0, 0, 0])
        if len(obbox) < 4:
            continue
        if _intersection_area(obbox, bbox) / max(1e-9, _bbox_area(obbox)) < 0.3:
            continue
        text = (ob.get("text") or "").strip().lower()
        if any(w in text for w in ACTION_WORDS):
            return True
    return False


def _ocr_left_aligned_in_bbox(bbox: List[float], raw_ocr_boxes: List[Dict[str, Any]]) -> bool:
    """True если OCR внутри bbox смещён влево (центр текста левее центра bbox) — признак label, не кнопки."""
    if len(bbox) < 4:
        return False
    cx_bbox = (bbox[0] + bbox[2]) / 2
    for ob in raw_ocr_boxes:
        obbox = ob.get("bbox", [0, 0, 0, 0])
        if len(obbox) < 4:
            continue
        if _intersection_area(obbox, bbox) / max(1e-9, _bbox_area(obbox)) < 0.2:
            continue
        cx_ocr = (obbox[0] + obbox[2]) / 2
        if cx_ocr < cx_bbox - 20:
            return True
    return False


def _ocr_centered_in_bbox(bbox: List[float], raw_ocr_boxes: List[Dict[str, Any]], tolerance: float = 0.25) -> bool:
    """True если OCR внутри bbox центрирован по вертикали и горизонтали — признак кнопки, не input."""
    if len(bbox) < 4:
        return False
    cx_bbox = (bbox[0] + bbox[2]) / 2
    cy_bbox = (bbox[1] + bbox[3]) / 2
    half_w = (bbox[2] - bbox[0]) / 2
    half_h = (bbox[3] - bbox[1]) / 2
    for ob in raw_ocr_boxes:
        obbox = ob.get("bbox", [0, 0, 0, 0])
        if len(obbox) < 4:
            continue
        if _intersection_area(obbox, bbox) / max(1e-9, _bbox_area(obbox)) < 0.2:
            continue
        cx_ocr = (obbox[0] + obbox[2]) / 2
        cy_ocr = (obbox[1] + obbox[3]) / 2
        if abs(cx_ocr - cx_bbox) <= half_w * tolerance and abs(cy_ocr - cy_bbox) <= half_h * tolerance:
            return True
    return False


def _input_vs_button_score(
    bbox: List[float],
    raw_ocr_boxes: List[Dict[str, Any]],
) -> Tuple[float, float]:
    """
    Скоринг input vs button по визуальным/текстовым признакам (без изображения).
    Возвращает (input_score, button_score) в диапазоне 0..1.
    Button: текст по центру, action-слово. Input: текст слева, label сверху/слева, нет action.
    """
    input_score = 0.0
    button_score = 0.0
    if len(bbox) < 4:
        return (input_score, button_score)
    if _ocr_centered_in_bbox(bbox, raw_ocr_boxes):
        button_score += 0.4
    if _ocr_left_aligned_in_bbox(bbox, raw_ocr_boxes):
        input_score += 0.35
    if _has_action_word_in_bbox(bbox, raw_ocr_boxes):
        button_score += 0.35
    if _has_aligned_label(bbox, raw_ocr_boxes):
        input_score += 0.3
    # Нет action-слова при наличии текста — скорее placeholder (input)
    if not _has_action_word_in_bbox(bbox, raw_ocr_boxes) and _text_inside_atom(bbox, raw_ocr_boxes):
        input_score += 0.15
    return (min(1.0, input_score), min(1.0, button_score))


def _assign_atoms_to_regions(
    atoms: List[Dict[str, Any]],
    regions: List[Dict[str, Any]],
    min_iou: float = 0.15,
) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for a in atoms:
        aid = a.get("id", "")
        bbox = a.get("bbox", [0, 0, 0, 0])
        if len(bbox) < 4:
            out[aid] = None
            continue
        best_rid: Optional[str] = None
        best_iou = 0.0
        for r in regions:
            rbbox = r.get("bbox", [0, 0, 0, 0])
            if len(rbbox) < 4:
                continue
            inter = _intersection_area(bbox, rbbox)
            area_a = _bbox_area(bbox)
            area_r = _bbox_area(rbbox)
            iou = inter / max(1e-9, area_a + area_r - inter)
            if iou >= min_iou and iou > best_iou:
                best_iou = iou
                best_rid = r.get("id")
        out[aid] = best_rid
    return out


def _text_inside_atom(
    atom_bbox: List[float],
    raw_ocr_boxes: List[Dict[str, Any]],
    min_coverage: float = OCR_INSIDE_COVERAGE,
) -> List[str]:
    texts: List[str] = []
    for ob in raw_ocr_boxes:
        obbox = ob.get("bbox", [0, 0, 0, 0])
        if len(obbox) < 4:
            continue
        if _coverage_bbox_in_bbox(obbox, atom_bbox) >= min_coverage:
            t = (ob.get("text") or "").strip()
            if t:
                texts.append(t)
    return texts


def _has_aligned_label(
    atom_bbox: List[float],
    raw_ocr_boxes: List[Dict[str, Any]],
    max_offset_px: float = LABEL_OFFSET_PX,
    min_overlap: float = LABEL_OVERLAP_AXIS,
) -> bool:
    if len(atom_bbox) < 4:
        return False
    x1, y1, x2, y2 = atom_bbox[0], atom_bbox[1], atom_bbox[2], atom_bbox[3]
    for ob in raw_ocr_boxes:
        obbox = ob.get("bbox", [0, 0, 0, 0])
        if len(obbox) < 4:
            continue
        ox1, oy1, ox2, oy2 = obbox[0], obbox[1], obbox[2], obbox[3]
        ow, oh = ox2 - ox1, oy2 - oy1
        if oh <= 0 or ow <= 0:
            continue
        if ox2 <= x1 and (x1 - ox2) <= max_offset_px:
            overlap_y = min(oy2, y2) - max(oy1, y1)
            if overlap_y >= min_overlap * oh:
                return True
        if oy2 <= y1 and (y1 - oy2) <= max_offset_px:
            overlap_x = min(ox2, x2) - max(ox1, x1)
            if overlap_x >= min_overlap * ow:
                return True
    return False


def _same_line(bbox_a: List[float], bbox_b: List[float], max_y_px: float = SAME_LINE_Y_PX) -> bool:
    if len(bbox_a) < 4 or len(bbox_b) < 4:
        return False
    cy_a = (bbox_a[1] + bbox_a[3]) / 2
    cy_b = (bbox_b[1] + bbox_b[3]) / 2
    return abs(cy_a - cy_b) <= max_y_px


def _distance_bbox_px(bbox_a: List[float], bbox_b: List[float]) -> float:
    if len(bbox_a) < 4 or len(bbox_b) < 4:
        return float("inf")
    cx_a = (bbox_a[0] + bbox_a[2]) / 2
    cy_a = (bbox_a[1] + bbox_a[3]) / 2
    cx_b = (bbox_b[0] + bbox_b[2]) / 2
    cy_b = (bbox_b[1] + bbox_b[3]) / 2
    return ((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5


def _region_has_form_structure(
    region_id: Optional[str],
    atoms: List[Dict[str, Any]],
    atom_to_region: Dict[str, Optional[str]],
    semantic_valid_only: bool = False,
) -> bool:
    if not region_id:
        return False
    inputs_in_r = [
        a for a in atoms
        if a.get("type") == "input" and atom_to_region.get(a.get("id")) == region_id
        and (not semantic_valid_only or a.get("semantic_valid"))
    ]
    buttons_in_r = [
        a for a in atoms
        if a.get("type") == "button" and atom_to_region.get(a.get("id")) == region_id
        and (not semantic_valid_only or a.get("semantic_valid"))
    ]
    return len(inputs_in_r) >= 2 and len(buttons_in_r) >= 1


def _button_in_action_group(
    atom: Dict[str, Any],
    atoms: List[Dict[str, Any]],
    atom_to_region: Dict[str, Optional[str]],
) -> bool:
    bbox = atom.get("bbox", [0, 0, 0, 0])
    if len(bbox) < 4:
        return False
    rid = atom_to_region.get(atom.get("id"))
    same_line_buttons = [
        a for a in atoms
        if a.get("type") == "button" and a is not atom and a.get("semantic_valid") is not False
        and atom_to_region.get(a.get("id")) == rid
        and _same_line(bbox, a.get("bbox", [0, 0, 0, 0]))
    ]
    return len(same_line_buttons) >= MIN_BUTTONS_IN_ACTION_GROUP - 1


def _button_nearby_input(
    atom: Dict[str, Any],
    atoms: List[Dict[str, Any]],
    atom_to_region: Dict[str, Optional[str]],
    max_px: float = NEARBY_INPUT_PX,
) -> bool:
    bbox = atom.get("bbox", [0, 0, 0, 0])
    if len(bbox) < 4:
        return False
    rid = atom_to_region.get(atom.get("id"))
    for a in atoms:
        if a.get("type") != "input" or not a.get("semantic_valid"):
            continue
        if atom_to_region.get(a.get("id")) != rid:
            continue
        if _distance_bbox_px(bbox, a.get("bbox", [0, 0, 0, 0])) <= max_px:
            return True
    return False


def _prune_to_layout(
    a: Dict[str, Any],
    reason: str,
    semantic_log: List[str],
    pruned_ids: Optional[Dict[str, List[str]]] = None,
) -> None:
    old_type = a.get("type", "")
    aid = a.get("id", "")
    a["type"] = LAYOUT_TYPE
    a["semantic_valid"] = False
    a["interactive_valid"] = False
    semantic_log.append(f"{aid} | {old_type} -> {LAYOUT_TYPE} | {reason}")
    if pruned_ids is not None and old_type in ("input", "button"):
        pruned_ids.setdefault(old_type, []).append(aid)


def _keep_anchor_only(a: Dict[str, Any], reason: str, semantic_log: List[str]) -> None:
    """Anchor прошёл, context нет: оставляем тип, semantic_valid=True, confidence *= CONTEXT_CONFIDENCE_FACTOR."""
    a["semantic_valid"] = True
    a["confidence"] = min(1.0, (a.get("confidence", 0) or 0) * CONTEXT_CONFIDENCE_FACTOR)
    semantic_log.append(f"anchor_only: {a.get('id', '')} | {reason}")


def _count_by_type(atoms: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for a in atoms:
        t = a.get("type") or ""
        counts[t] = counts.get(t, 0) + 1
    return counts


# ML priors: порог для усиления слабого текста (модель может усилить, правила могут перебить)
PRIOR_INPUT_MIN = 0.55
PRIOR_BUTTON_MIN = 0.55
PRIOR_INTERACTIVE_MIN = 0.55

# Propagation: минимум интерактивных в группе и медианная длина текста
PROPAGATION_MIN_INTERACTIVE_IN_GROUP = 2
PROPAGATION_MEDIAN_TEXT_LENGTH_MIN = 2
# Icon/emoji: не резать кнопки с высоким prior (weak_input/button разрешены при prior >= порога)
ICON_INTERACTIVE_MIN = 0.6


def _phase1_interactive_gate(
    atoms: List[Dict[str, Any]],
    raw_ocr_boxes: List[Dict[str, Any]],
    atom_to_region: Dict[str, Optional[str]],
    semantic_log: List[str],
    pruned_ids: Dict[str, List[str]],
) -> None:
    """
    Фаза 1 — Interactive gate: только interactive_valid и interactive_source.
    Не назначаем role/type (кроме prune → layout). Не используем role_probs.
    """
    for a in atoms:
        t = (a.get("type") or "").strip()
        priors = a.get("priors") or {}
        interactive_score = float(priors.get("interactive_score", 0))

        if t == "input":
            bbox = a.get("bbox", [0, 0, 0, 0])
            if len(bbox) < 4:
                a["interactive_valid"] = False
                continue
            if _input_anchor_ok(bbox, raw_ocr_boxes):
                a["interactive_valid"] = True
                a["interactive_source"] = "anchor"
            elif interactive_score >= PRIOR_INTERACTIVE_MIN:
                a["interactive_valid"] = True
                a["interactive_source"] = "prior"
            else:
                a["interactive_valid"] = False
                _prune_to_layout(a, "input no anchor, low interactive_score", semantic_log, pruned_ids)

        elif t == "button":
            bbox = a.get("bbox", [0, 0, 0, 0])
            if len(bbox) < 4:
                a["interactive_valid"] = False
                continue
            if _button_anchor_ok(bbox):
                a["interactive_valid"] = True
                a["interactive_source"] = "anchor"
            elif interactive_score >= PRIOR_INTERACTIVE_MIN:
                a["interactive_valid"] = True
                a["interactive_source"] = "prior"
            else:
                a["interactive_valid"] = False
                _prune_to_layout(a, "button no anchor, low interactive_score", semantic_log, pruned_ids)

        elif t == "layout":
            a["interactive_valid"] = interactive_score >= PRIOR_INTERACTIVE_MIN
            a["interactive_source"] = "prior" if a["interactive_valid"] else None

        elif t in ("input_candidate", "textarea_candidate"):
            a["interactive_valid"] = True
            a["interactive_source"] = "recovery"

        elif t == "container_candidate":
            a["interactive_valid"] = False

        elif t in ("link", "text_block", "title"):
            a["interactive_valid"] = True
            a["interactive_source"] = "anchor"

        else:
            a["interactive_valid"] = False


def _validate_input_anchor_first(
    atoms: List[Dict[str, Any]],
    raw_ocr_boxes: List[Dict[str, Any]],
    atom_to_region: Dict[str, Optional[str]],
    semantic_log: List[str],
    saved_by_anchor: Dict[str, List[str]],
    pruned_ids: Dict[str, List[str]],
) -> None:
    """Фаза 2 — Role assignment для input и input_candidate. Только для interactive_valid. role_probs используются здесь."""
    for a in atoms:
        if a.get("type") not in ("input", "input_candidate", "textarea_candidate"):
            continue
        if not a.get("interactive_valid"):
            a["semantic_valid"] = False
            continue
        bbox = a.get("bbox", [0, 0, 0, 0])
        if len(bbox) < 4:
            a["semantic_valid"] = False
            continue
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        aspect = w / max(h, 1e-9)
        input_score, button_score = _input_vs_button_score(bbox, raw_ocr_boxes)
        if aspect < BUTTON_ASPECT_LABEL_LIKE_MAX and button_score > input_score and button_score >= 0.5:
            _prune_to_layout(
                a,
                "input_vs_button_score: button_score=%.2f > input_score=%.2f (button-like, not input)"
                % (button_score, input_score),
                semantic_log,
                pruned_ids,
            )
            continue
        priors = a.get("priors") or {}
        role_probs = priors.get("role_probs") or {}
        prior_input = role_probs.get("input", 0)
        interactive_score = float(priors.get("interactive_score", 0))
        high_prior = prior_input >= PRIOR_INPUT_MIN and interactive_score >= PRIOR_INTERACTIVE_MIN
        if not _input_anchor_ok(bbox, raw_ocr_boxes):
            probe_len = a.get("probe_ocr_len") or 0
            prior_with_probe = (
                interactive_score >= PRIOR_INTERACTIVE_MIN
                and prior_input >= PRIOR_INPUT_MIN
                and probe_len >= 1
            )
            phase_b_confirmed = a.get("source") == "input_candidate_recovery"
            if high_prior or prior_with_probe or phase_b_confirmed:
                a["type"] = "weak_input"
                a["semantic_valid"] = True
                a["confidence"] = CONTEXT_CONFIDENCE_FACTOR
                saved_by_anchor.setdefault("input", []).append(a.get("id", ""))
                reason = "prior/probe" if (high_prior or prior_with_probe) else "Phase B confirmed"
                semantic_log.append(
                    "weak_input (%s): %s | input no anchor" % (reason, a.get("id", ""))
                )
            else:
                _prune_to_layout(a, "input no anchor (aspect/height/text coverage)", semantic_log, pruned_ids)
            continue
        has_label = (
            bool(_text_inside_atom(bbox, raw_ocr_boxes))
            or _has_aligned_label(bbox, raw_ocr_boxes)
            or (a.get("probe_ocr_len") or 0) >= 1
        )
        rid = atom_to_region.get(a.get("id"))
        form_context = _region_has_form_structure(rid, atoms, atom_to_region, semantic_valid_only=False)
        has_context = has_label or form_context
        if has_context:
            a["semantic_valid"] = True
            if (a.get("type") or "") == "input_candidate":
                a["type"] = "input"
            elif (a.get("type") or "") == "textarea_candidate":
                a["type"] = "textarea"
            continue
        a["type"] = "weak_input"
        a["semantic_valid"] = True
        a["confidence"] = CONTEXT_CONFIDENCE_FACTOR
        aid = a.get("id", "")
        saved_by_anchor.setdefault("input", []).append(aid)
        semantic_log.append("weak_input: %s | input anchor only (no label/form)" % aid)


def _validate_button_anchor_first(
    atoms: List[Dict[str, Any]],
    raw_ocr_boxes: List[Dict[str, Any]],
    atom_to_region: Dict[str, Optional[str]],
    semantic_log: List[str],
    saved_by_anchor: Dict[str, List[str]],
    pruned_ids: Dict[str, List[str]],
) -> None:
    """Фаза 2 — Role assignment для button. Только для interactive_valid. role_probs используются здесь."""
    for a in atoms:
        if a.get("type") != "button":
            continue
        if not a.get("interactive_valid"):
            a["semantic_valid"] = False
            continue
        bbox = a.get("bbox", [0, 0, 0, 0])
        if len(bbox) < 4:
            a["semantic_valid"] = False
            continue
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        aspect = w / max(h, 1e-9)
        if a.get("source") == "synthetic" and aspect > BUTTON_ASPECT_LABEL_LIKE_MAX:
            if not _has_action_word_in_bbox(bbox, raw_ocr_boxes) or _ocr_left_aligned_in_bbox(bbox, raw_ocr_boxes):
                _prune_to_layout(a, "synthetic button: high aspect, text left-aligned (label-like)", semantic_log, pruned_ids)
                continue
        priors = a.get("priors") or {}
        role_probs = priors.get("role_probs") or {}
        prior_button = role_probs.get("button", 0)
        interactive_score = float(priors.get("interactive_score", 0))
        high_prior = prior_button >= PRIOR_BUTTON_MIN and interactive_score >= PRIOR_INTERACTIVE_MIN
        if not _button_anchor_ok(bbox):
            if high_prior:
                saved_by_anchor.setdefault("button", []).append(a.get("id", ""))
                _keep_anchor_only(a, "button no anchor, high prior", semantic_log)
            else:
                _prune_to_layout(a, "button no anchor (aspect/area)", semantic_log, pruned_ids)
            continue
        rid = atom_to_region.get(a.get("id"))
        form_ctx = _region_has_form_structure(rid, atoms, atom_to_region, semantic_valid_only=True)
        if form_ctx:
            a["semantic_valid"] = True
            continue
        if _button_nearby_input(a, atoms, atom_to_region):
            a["semantic_valid"] = True
            continue
        if _button_in_action_group(a, atoms, atom_to_region):
            a["semantic_valid"] = True
            continue
        aid = a.get("id", "")
        saved_by_anchor.setdefault("button", []).append(aid)
        _keep_anchor_only(a, "button anchor only (no action)", semantic_log)


def _container_semantic_valid_children(
    container_bbox: List[float],
    container_id: str,
    atoms: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Дети контейнера с semantic_valid=True, покрытие внутри контейнера >= CONTAINMENT_COVERAGE."""
    children: List[Dict[str, Any]] = []
    for a in atoms:
        if a.get("id") == container_id:
            continue
        if not a.get("semantic_valid"):
            continue
        if a.get("type") == LAYOUT_TYPE:
            continue
        abbox = a.get("bbox", [0, 0, 0, 0])
        if len(abbox) < 4:
            continue
        if _coverage_bbox_in_bbox(abbox, container_bbox) >= CONTAINMENT_COVERAGE:
            children.append(a)
    return children


def _children_pairwise_overlap(children: List[Dict[str, Any]]) -> bool:
    """True если хотя бы одна пара детей перекрывается (intersection > 0)."""
    bboxes = [a.get("bbox", [0, 0, 0, 0]) for a in children]
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            if _intersection_area(bboxes[i], bboxes[j]) > 0:
                return True
    return False


def _validate_container_hard(
    atoms: List[Dict[str, Any]],
    atom_to_region: Dict[str, Optional[str]],
    semantic_log: List[str],
) -> None:
    """Container_candidate валиден только если содержит ≥2 semantic_valid атома, не перекрывающихся. Иначе или перекрывает button/input без группировки → layout."""
    for a in atoms:
        if a.get("type") != "container_candidate":
            continue
        bbox = a.get("bbox", [0, 0, 0, 0])
        if len(bbox) < 4:
            a["semantic_valid"] = False
            continue
        cid = a.get("id")
        children = _container_semantic_valid_children(bbox, cid or "", atoms)
        if len(children) < MIN_ATOMS_IN_CONTAINER:
            _prune_to_layout(a, "container has <2 semantic_valid children", semantic_log)
            continue
        if _children_pairwise_overlap(children):
            _prune_to_layout(a, "container children overlap", semantic_log)
            continue
        a["semantic_valid"] = True


def _set_layout_passive(atoms: List[Dict[str, Any]]) -> None:
    """layout → semantic_valid=False. interactive_valid=False → semantic_valid=False. Остальные без явного semantic_valid → True."""
    for a in atoms:
        if a.get("type") == LAYOUT_TYPE:
            a["semantic_valid"] = False
        elif not a.get("interactive_valid"):
            a["semantic_valid"] = False
        elif a.get("semantic_valid") is None:
            a["semantic_valid"] = True


def _text_length_inside_atom(atom: Dict[str, Any], raw_ocr_boxes: List[Dict[str, Any]]) -> int:
    bbox = atom.get("bbox", [0, 0, 0, 0])
    if len(bbox) < 4:
        return 0
    texts = _text_inside_atom(bbox, raw_ocr_boxes)
    return sum(len((t or "").strip()) for t in texts)


def _median_text_length_in_group(
    group_atoms: List[Dict[str, Any]],
    raw_ocr_boxes: List[Dict[str, Any]],
) -> float:
    if not group_atoms:
        return 0.0
    lengths = [_text_length_inside_atom(a, raw_ocr_boxes) for a in group_atoms]
    lengths = [l for l in lengths if l is not None]
    if not lengths:
        return 0.0
    lengths.sort()
    n = len(lengths)
    return float(lengths[n // 2]) if n % 2 else (lengths[n // 2 - 1] + lengths[n // 2]) / 2.0


def _is_icon_or_emoji_dominant(atom: Dict[str, Any], raw_ocr_boxes: List[Dict[str, Any]]) -> bool:
    """Нет текста или суммарная длина текста внутри bbox ≤ 2 — считаем icon/emoji dominant."""
    total = _text_length_inside_atom(atom, raw_ocr_boxes)
    return total <= PROPAGATION_MEDIAN_TEXT_LENGTH_MIN


def _propagate_semantic_in_groups(
    atoms: List[Dict[str, Any]],
    raw_ocr_boxes: List[Dict[str, Any]],
    atom_groups: Optional[Dict[str, List[str]]] = None,
    semantic_log: Optional[List[str]] = None,
) -> None:
    """
    Propagation только если: в группе ≥2 interactive_valid с одной ролью,
    медианная длина текста группы ≥ порога, атом не icon_only/emoji_dominant.
    """
    if not atom_groups or not atoms:
        return
    aid_to_atom: Dict[str, Dict[str, Any]] = {a.get("id", ""): a for a in atoms if a.get("id")}
    log = semantic_log if semantic_log is not None else []
    valid_types = ("button", "input", "weak_input", "weak_button")
    for gid, aids in atom_groups.items():
        if len(aids) <= 1:
            continue
        group_atoms = [aid_to_atom[aid] for aid in aids if aid in aid_to_atom]
        if not group_atoms:
            continue
        strong_in_group = [
            a for a in group_atoms
            if a.get("interactive_valid") and a.get("semantic_valid") and (a.get("type") or "") in valid_types
        ]
        if len(strong_in_group) < PROPAGATION_MIN_INTERACTIVE_IN_GROUP:
            continue
        dominant_type = strong_in_group[0].get("type", "button")
        if not all((a.get("type") or "") == dominant_type for a in strong_in_group):
            continue
        median_len = _median_text_length_in_group(group_atoms, raw_ocr_boxes)
        if median_len < PROPAGATION_MEDIAN_TEXT_LENGTH_MIN:
            continue
        for a in group_atoms:
            if a.get("semantic_valid") and (a.get("type") or "") in valid_types:
                continue
            if not a.get("interactive_valid"):
                continue
            if _is_icon_or_emoji_dominant(a, raw_ocr_boxes) and (
                (a.get("priors") or {}).get("interactive_score", 0) < ICON_INTERACTIVE_MIN
            ):
                continue
            old_type = a.get("type", "")
            a["type"] = dominant_type
            a["semantic_valid"] = True
            a["confidence"] = max((a.get("confidence") or 0), CONTEXT_CONFIDENCE_FACTOR)
            log.append("group_propagate: %s | %s -> %s (group %s)" % (a.get("id", ""), old_type, dominant_type, gid))


def _set_semantic_role_and_lock(atoms: List[Dict[str, Any]]) -> None:
    """
    v3: semantic_validation — единственный источник ролей.
    semantic_role = итоговый type после валидации.
    semantic_lock = True только если роль назначена (тип из SEMANTIC_LOCK_TYPES и semantic_valid).
    layout/text/noise всегда semantic_lock=False.
    """
    for a in atoms:
        t = (a.get("type") or "").strip().lower()
        a["semantic_role"] = t or "layout"
        if t in SEMANTIC_LOCK_TYPES and a.get("semantic_valid"):
            a["semantic_lock"] = True
        else:
            a["semantic_lock"] = False


def _button_absorbs_link_text_inside(atoms: List[Dict[str, Any]], semantic_log: List[str]) -> None:
    """
    button поглощает любые link/text внутри своего bbox: у таких атомов semantic_lock=False,
    они не попадают в atoms_for_interaction.
    """
    for a in atoms:
        if (a.get("type") or "").lower() != "button" or not a.get("semantic_valid"):
            continue
        bbox_btn = a.get("bbox", [0, 0, 0, 0])
        if len(bbox_btn) < 4:
            continue
        for other in atoms:
            if other is a or other.get("id") == a.get("id"):
                continue
            t_other = (other.get("type") or "").lower()
            if t_other not in ("link", "text_block", "title", "weak_link"):
                continue
            bbox_other = other.get("bbox", [0, 0, 0, 0])
            if len(bbox_other) < 4:
                continue
            if _coverage_bbox_in_bbox(bbox_other, bbox_btn) >= BUTTON_ABSORB_COVERAGE_MIN:
                other["semantic_lock"] = False
                semantic_log.append("button_absorb: %s absorbs %s (link/text inside bbox)" % (a.get("id", ""), other.get("id", "")))


def run_semantic_validation(
    atoms: List[Dict[str, Any]],
    raw_ocr_boxes: List[Dict[str, Any]],
    regions: List[Dict[str, Any]],
    require_effect: bool = True,
    atom_groups: Optional[Dict[str, List[str]]] = None,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Anchor-first: anchor — разрешение на тип; context — усилитель. ML priors могут усилить слабый текст.
    Группы (atom_groups): если один в группе усилился семантически, это влияет на остальных.
    Модифицирует atoms in-place.
    Возвращает (log_lines, stats). stats: before, after, diff, saved_by_anchor {input: [id...], button: [id...]}.
    """
    semantic_log: List[str] = []
    saved_by_anchor: Dict[str, List[str]] = {}
    pruned_ids: Dict[str, List[str]] = {"input": [], "button": []}
    if not atoms:
        return semantic_log, {"before": {}, "after": {}, "diff": {}, "saved_by_anchor": saved_by_anchor}

    for a in atoms:
        a["semantic_valid"] = False
        a["interactive_valid"] = False

    before = _count_by_type(atoms)
    atom_to_region = _assign_atoms_to_regions(atoms, regions)

    _phase1_interactive_gate(atoms, raw_ocr_boxes, atom_to_region, semantic_log, pruned_ids)
    _validate_input_anchor_first(atoms, raw_ocr_boxes, atom_to_region, semantic_log, saved_by_anchor, pruned_ids)
    _validate_button_anchor_first(atoms, raw_ocr_boxes, atom_to_region, semantic_log, saved_by_anchor, pruned_ids)
    _validate_container_hard(atoms, atom_to_region, semantic_log)
    _set_layout_passive(atoms)
    _propagate_semantic_in_groups(atoms, raw_ocr_boxes, atom_groups, semantic_log)

    _set_semantic_role_and_lock(atoms)
    _button_absorbs_link_text_inside(atoms, semantic_log)

    after = _count_by_type(atoms)
    diff: Dict[str, int] = {}
    all_types = set(before) | set(after)
    for t in all_types:
        diff[t] = after.get(t, 0) - before.get(t, 0)

    stats = {
        "before": before,
        "after": after,
        "diff": diff,
        "saved_by_anchor": saved_by_anchor,
        "pruned_ids": pruned_ids,
    }
    n_input_saved = len(saved_by_anchor.get("input", []))
    n_button_saved = len(saved_by_anchor.get("button", []))
    n_pruned_input = len(pruned_ids.get("input", []))
    n_pruned_button = len(pruned_ids.get("button", []))
    log_lines: List[str] = [
        "semantic_validation before: input=%s button=%s container_candidate=%s layout=%s"
        % (before.get("input", 0), before.get("button", 0), before.get("container_candidate", 0), before.get(LAYOUT_TYPE, 0)),
        "semantic_validation after:  input=%s button=%s container_candidate=%s layout=%s"
        % (after.get("input", 0), after.get("button", 0), after.get("container_candidate", 0), after.get(LAYOUT_TYPE, 0)),
        "semantic_validation saved_by_anchor: input=%s button=%s" % (n_input_saved, n_button_saved),
        "semantic_validation pruned (suppressed as button/layout): input=%d button=%d" % (n_pruned_input, n_pruned_button),
    ]
    log_lines.extend(semantic_log)
    if semantic_log:
        log_lines.append("semantic_validation applied (anchor + context)")

    if after.get("button", 0) == 0 and before.get("button", 0) > 0:
        ids = pruned_ids.get("button", [])
        log_lines.append("semantic_anchor_failed: atom_id=%s (button_after=0)" % (",".join(ids) if ids else "—"))
        logger.warning("semantic_anchor_failed: button_after=0 (anchor did not preserve any button)")
    input_like_after = after.get("input", 0) + after.get("weak_input", 0)
    if input_like_after == 0 and before.get("input", 0) > 0:
        ids = pruned_ids.get("input", [])
        log_lines.append("semantic_anchor_failed: atom_id=%s (input_after=0)" % (",".join(ids) if ids else "—"))
        logger.warning("semantic_anchor_failed: input_after=0 (anchor did not preserve any input)")

    if require_effect:
        in_before = before.get("input", 0)
        in_after = after.get("input", 0) + after.get("weak_input", 0)
        cont_before = before.get("container_candidate", 0)
        cont_after = after.get("container_candidate", 0)
        in_barely = in_before > 0 and in_after >= NO_EFFECT_THRESHOLD * in_before
        cont_barely = cont_before > 0 and cont_after >= NO_EFFECT_THRESHOLD * cont_before
        # Require pruning only when there was something to prune (e.g. form with 1 container + 1 input is allowed)
        had_multiple = cont_before >= 2 or in_before >= 2
        if had_multiple and in_barely and cont_barely:
            raise AssertionError(
                "semantic_validation_no_effect: input and container_candidate barely changed "
                "(before input=%s container=%s, after input+weak_input=%s container=%s). Layer must prune."
                % (in_before, cont_before, in_after, cont_after)
            )

    for line in semantic_log:
        logger.debug("semantic_validation: %s", line)
    return log_lines, stats
