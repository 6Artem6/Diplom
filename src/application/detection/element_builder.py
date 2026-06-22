"""
Build DetectedElement records from GUI blocks (detection pipeline output).
"""

from __future__ import annotations

from typing import Dict, List, Tuple
from uuid import UUID

from src.domain.models.detected_element import DetectedElement
from src.domain.models.gui_block import GUIBlock
from src.domain.models.view import View


_CLASS_PRIORITY = (
    "button",
    "input",
    "textarea",
    "checkbox",
    "radio",
    "dropdown",
    "badge",
    "header",
    "image",
    "text_block",
    "section",
    "card",
    "list-item",
)


def resolve_element_class(block: GUIBlock, layout_features: Dict[str, object]) -> str:
    label = layout_features.get("class_label")
    if label:
        return str(label)
    types = [t.lower() for t in (block.element_types or [])]
    for preferred in _CLASS_PRIORITY:
        if preferred in types:
            return preferred
    if types:
        return types[0]
    return "unknown"


def derive_role(class_name: str) -> str:
    c = class_name.lower()
    if "text_block" in c or c == "text":
        return "text"
    if "button" in c:
        return "action"
    if "input" in c or "textarea" in c:
        return "input"
    if "image" in c or "icon" in c:
        return "visual"
    if "badge" in c or "label" in c:
        return "label"
    if "header" in c:
        return "heading"
    if "card" in c or "section" in c:
        return "container"
    return "unknown"


def flatten_bbox(bb: dict) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in bb.items():
        if k == "container_bbox" or isinstance(v, dict):
            continue
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def build_detected_element(
    block: GUIBlock,
    view: View,
    element_id: UUID,
    layout_features: Dict[str, object],
    detection_confidence: float = 0.85,
) -> DetectedElement:
    class_name = resolve_element_class(block, layout_features)
    raw_ocr = (
        str(layout_features.get("raw_ocr") or block.ocr_text_raw or "").strip() or None
    )
    cleaned = (
        str(layout_features.get("cleaned_text") or block.ocr_text or "").strip() or None
    )
    noisy = bool(layout_features.get("noisy_ocr", block.ocr_noisy))
    embedding_input = layout_features.get("embedding_input_text")
    embedding_source = layout_features.get("embedding_source")
    return DetectedElement(
        element_id=element_id,
        view_id=view.id,
        screenshot_id=block.screenshot_id,
        bbox=flatten_bbox(block.bounding_box or {}),
        class_name=class_name,
        role=derive_role(class_name),
        text=cleaned,
        cleaned_text=cleaned,
        ocr_text=raw_ocr,
        noisy_ocr=noisy,
        embedding_input=str(embedding_input) if embedding_input else None,
        embedding_source=str(embedding_source) if embedding_source else None,
        confidence=detection_confidence,
        pipeline_stage="detection",
        manifestation_id=element_id,
    )


def assign_entity_ids_to_elements(
    detected_elements: List[DetectedElement],
    man_to_entity: Dict[UUID, UUID],
) -> List[DetectedElement]:
    updated: List[DetectedElement] = []
    for el in detected_elements:
        eid = man_to_entity.get(el.element_id)
        updated.append(
            el.model_copy(update={"entity_instance_id": eid}) if eid else el
        )
    return updated
