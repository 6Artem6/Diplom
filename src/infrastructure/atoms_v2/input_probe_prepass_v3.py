"""
v3: Подготовка OCR-сигнала до semantic_validation без назначения ролей и без фильтрации атомов.

Назначение: дать semantic_validation данные для input-анкора (probe_ocr_text, probe_ocr_len),
чтобы реальные поля ввода могли получить semantic_lock.

Не меняет: type, semantic_valid, semantic_lock.
Не удаляет атомы. Только дописывает atom["probe_ocr_text"] и atom["probe_ocr_len"].
"""

from __future__ import annotations

from typing import Any, Dict, List

# Кандидаты на input: aspect ≥ 3, height ∈ [24, 60], width ≥ MIN_INPUT_WIDTH, type ∈ (layout, control_group, container_candidate, input)
PROBE_INPUT_ASPECT_MIN = 3.0
PROBE_INPUT_HEIGHT_MIN_PX = 24
PROBE_INPUT_HEIGHT_MAX_PX = 60
MIN_INPUT_WIDTH = 60
PROBE_EXPAND_PX = 7
PROBE_OCR_COVERAGE_MIN = 0.2  # доля площади OCR внутри expanded bbox для учёта
PROBE_CANDIDATE_TYPES = ("layout", "control_group", "container_candidate", "input", "input_candidate", "textarea_candidate")


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


def _coverage_in_outer(inner: List[float], outer: List[float]) -> float:
    area_inner = _bbox_area(inner)
    if area_inner <= 0:
        return 0.0
    return _intersection_area(inner, outer) / area_inner


def input_probe_prepass_v3(
    atoms: List[Dict[str, Any]],
    raw_ocr_boxes: List[Dict[str, Any]],
    expand_px: int = PROBE_EXPAND_PX,
) -> None:
    """
    Для атомов-кандидатов на input (aspect ≥ 3, height ∈ [24, 60], width ≥ MIN_INPUT_WIDTH,
    type ∈ layout, control_group, container_candidate, input): временно расширяет bbox, собирает OCR,
    записывает atom["probe_ocr_text"] и atom["probe_ocr_len"]. Роли не назначает, атомы не удаляет.
    """
    for a in atoms:
        t = (a.get("type") or "").strip().lower()
        if t not in PROBE_CANDIDATE_TYPES:
            a["probe_ocr_text"] = ""
            a["probe_ocr_len"] = 0
            continue
        bbox = a.get("bbox", [0, 0, 0, 0])
        if len(bbox) < 4:
            a["probe_ocr_text"] = ""
            a["probe_ocr_len"] = 0
            continue
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if h <= 0:
            a["probe_ocr_text"] = ""
            a["probe_ocr_len"] = 0
            continue
        aspect = w / h
        if aspect < PROBE_INPUT_ASPECT_MIN:
            a["probe_ocr_text"] = ""
            a["probe_ocr_len"] = 0
            continue
        if h < PROBE_INPUT_HEIGHT_MIN_PX or h > PROBE_INPUT_HEIGHT_MAX_PX:
            a["probe_ocr_text"] = ""
            a["probe_ocr_len"] = 0
            continue
        if w < MIN_INPUT_WIDTH:
            a["probe_ocr_text"] = ""
            a["probe_ocr_len"] = 0
            continue
        x1 = max(0.0, bbox[0] - expand_px)
        y1 = max(0.0, bbox[1] - expand_px)
        x2 = bbox[2] + expand_px
        y2 = bbox[3] + expand_px
        expanded = [x1, y1, x2, y2]
        texts: List[str] = []
        for ob in raw_ocr_boxes:
            obbox = ob.get("bbox", [0, 0, 0, 0])
            if len(obbox) < 4:
                continue
            if _coverage_in_outer(obbox, expanded) >= PROBE_OCR_COVERAGE_MIN:
                txt = (ob.get("text") or "").strip()
                if txt:
                    texts.append(txt)
        probe_text = " ".join(texts)
        a["probe_ocr_text"] = probe_text
        a["probe_ocr_len"] = len(probe_text)
