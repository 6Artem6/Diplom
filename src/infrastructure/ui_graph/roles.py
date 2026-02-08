"""
Роли UI: ui_role ≠ atom.type. atom.type — CV-гипотеза, ui_role — итоговая семантика.

ui_role ∈ {button, input, link, control_group, pagination, form, noise}.
Правила для weak supervision (rule-generated labels) для обучения классификатора.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from src.infrastructure.ui_graph.graph import UIGraph, AtomNode


class UIRole(str, Enum):
    BUTTON = "button"
    INPUT = "input"
    LINK = "link"
    WEAK_BUTTON = "weak_button"
    WEAK_LINK = "weak_link"
    WEAK_INPUT = "weak_input"
    CONTROL_GROUP = "control_group"
    PAGINATION = "pagination"
    FORM = "form"
    NOISE = "noise"


# Anchor-диапазоны: если тип + геометрия в диапазоне — никогда не noise (расширено для input без label)
INPUT_ANCHOR_ASPECT = (1.5, 20.0)
INPUT_ANCHOR_HEIGHT = (16, 70)
BUTTON_ANCHOR_ASPECT = (1.5, 30.0)
BUTTON_ANCHOR_AREA_MIN = 100


def rule_based_role(
    atom: AtomNode,
    features: Dict[str, float],
    graph: UIGraph,
) -> UIRole:
    """
    Эвристическое присвоение роли по признакам (weak supervision).
    Используется для генерации меток при отсутствии обученного классификатора и для спорных случаев.
    """
    atype = (atom.type or "").lower()
    area = features.get("area", 0)
    aspect_ratio = features.get("aspect_ratio", 0)
    num_adjacent = int(features.get("num_adjacent", 0))
    num_aligned_row = int(features.get("num_aligned_row", 0))
    num_inputs_nearby = int(features.get("num_inputs_nearby", 0))
    num_buttons_nearby = int(features.get("num_buttons_nearby", 0))
    row_group_size = int(features.get("row_group_size", 1))
    has_label = features.get("has_label", 0) >= 0.5
    has_action_word = features.get("has_action_word", 0) >= 0.5
    bbox_coverage_ocr = features.get("bbox_coverage_ocr", 0)
    mixed_types_in_row = features.get("mixed_types_in_row", 0) >= 0.5
    uniform_spacing_score = features.get("uniform_spacing_score", 0)
    ocr_inside_count = int(features.get("ocr_inside_count", 0))
    ocr_inside_mean_conf = features.get("ocr_inside_mean_conf", 0)
    ocr_inside_text_len = int(features.get("ocr_inside_text_len", 0))

    # OCR внутри + anchor-форма = сильный сигнал кнопки/ссылки (даже без соседей)
    ocr_strong_inside = ocr_inside_count >= 1 and ocr_inside_mean_conf >= 0.8
    button_anchor_aspect = BUTTON_ANCHOR_ASPECT[0] <= aspect_ratio <= BUTTON_ANCHOR_ASPECT[1]
    button_anchor_area = area >= BUTTON_ANCHOR_AREA_MIN

    # Anchor-геометрия: button/input/link в anchor-диапазоне никогда не noise (weak_* или тип)
    h = (atom.bbox[3] - atom.bbox[1]) if len(atom.bbox) >= 4 else 0

    if atype == "link":
        if bbox_coverage_ocr > 0.5 and area > 100:
            return UIRole.LINK
        if ocr_strong_inside and button_anchor_area:
            return UIRole.LINK
        if area > 100 and bbox_coverage_ocr > 0.2:
            return UIRole.WEAK_LINK
        if bbox_coverage_ocr < 0.1 and num_adjacent == 0 and not ocr_strong_inside and area < 100:
            return UIRole.NOISE
        return UIRole.LINK

    if atype == "weak_input":
        if INPUT_ANCHOR_ASPECT[0] <= aspect_ratio <= INPUT_ANCHOR_ASPECT[1] and INPUT_ANCHOR_HEIGHT[0] <= h <= INPUT_ANCHOR_HEIGHT[1]:
            return UIRole.WEAK_INPUT
        return UIRole.NOISE

    if atype == "input":
        if bbox_coverage_ocr > 0.7 and area < 200 and not ocr_strong_inside:
            return UIRole.NOISE
        if ocr_strong_inside and INPUT_ANCHOR_ASPECT[0] <= aspect_ratio <= INPUT_ANCHOR_ASPECT[1] and INPUT_ANCHOR_HEIGHT[0] <= h <= INPUT_ANCHOR_HEIGHT[1]:
            return UIRole.INPUT
        if num_inputs_nearby >= 1 and num_buttons_nearby >= 1:
            return UIRole.FORM
        if has_label or num_inputs_nearby >= 1:
            return UIRole.INPUT
        if INPUT_ANCHOR_ASPECT[0] <= aspect_ratio <= INPUT_ANCHOR_ASPECT[1] and INPUT_ANCHOR_HEIGHT[0] <= h <= INPUT_ANCHOR_HEIGHT[1]:
            return UIRole.INPUT
        return UIRole.NOISE

    if atype == "button":
        if ocr_strong_inside and button_anchor_aspect and button_anchor_area:
            return UIRole.BUTTON
        if row_group_size >= 3 and uniform_spacing_score > 0.7 and num_inputs_nearby == 0:
            return UIRole.PAGINATION
        if num_buttons_nearby >= 2 and num_aligned_row >= 1:
            return UIRole.CONTROL_GROUP
        if num_inputs_nearby >= 1 and num_buttons_nearby >= 0:
            return UIRole.FORM
        if has_action_word or has_label:
            return UIRole.BUTTON
        if button_anchor_aspect and button_anchor_area:
            return UIRole.WEAK_BUTTON
        return UIRole.NOISE

    if atype in ("layout", "layout_candidate", "container_candidate"):
        if ocr_strong_inside and button_anchor_aspect and button_anchor_area and ocr_inside_text_len <= 20:
            return UIRole.WEAK_BUTTON
        if ocr_strong_inside and area > 100 and ocr_inside_text_len <= 30:
            return UIRole.WEAK_LINK
        if num_adjacent >= 2 or row_group_size >= 2:
            return UIRole.CONTROL_GROUP
        return UIRole.NOISE

    if atype in ("text_block", "title", "label"):
        if num_adjacent == 0 and bbox_coverage_ocr < 0.3 and not ocr_strong_inside:
            return UIRole.NOISE
        return UIRole.LINK

    return UIRole.NOISE
