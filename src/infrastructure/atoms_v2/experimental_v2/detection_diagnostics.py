"""
Диагностический модуль для визуальной детекции.

Собирает статистику, выявляет проблемы, выводит детальные логи.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class DetectionDiagnostics:
    """Диагностика детекции для одной формы."""
    
    # Контейнер
    container_bbox: Optional[List[float]] = None
    container_area: float = 0.0
    image_area: float = 0.0
    coverage_percent: float = 0.0
    
    # Визуальные кандидаты
    total_visual_candidates: int = 0
    visual_by_source: Dict[str, int] = field(default_factory=dict)
    
    # После классификации
    total_after_classification: int = 0
    by_element_type: Dict[str, int] = field(default_factory=dict)
    
    # После suppression/filtering
    total_after_suppression: int = 0
    suppression_reasons: List[str] = field(default_factory=list)
    
    # Вложенности
    nested_elements: List[Tuple[int, int]] = field(default_factory=list)  # (parent_idx, child_idx)
    overlapping_pairs: List[Tuple[int, int, float]] = field(default_factory=list)  # (idx1, idx2, iou)
    
    # Итоговые атомы
    total_atoms: int = 0
    atoms_by_type: Dict[str, int] = field(default_factory=dict)
    
    # Причины отклонения
    rejection_reasons: List[str] = field(default_factory=list)
    
    # OCR статистика
    ocr_blocks: int = 0
    median_text_height: float = 0.0
    rows_without_text: int = 0


def compute_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """Вычисляет IoU двух bbox."""
    if len(bbox1) < 4 or len(bbox2) < 4:
        return 0.0
    
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    inter = (x2 - x1) * (y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    return inter / max(1, area1 + area2 - inter)


def bbox_contains(outer: List[float], inner: List[float], threshold: float = 0.9) -> bool:
    """Проверяет, содержит ли outer bbox inner bbox (с порогом)."""
    if len(outer) < 4 or len(inner) < 4:
        return False
    
    # Площадь inner внутри outer
    x1 = max(outer[0], inner[0])
    y1 = max(outer[1], inner[1])
    x2 = min(outer[2], inner[2])
    y2 = min(outer[3], inner[3])
    
    if x2 <= x1 or y2 <= y1:
        return False
    
    inter = (x2 - x1) * (y2 - y1)
    inner_area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    
    return inter / max(1, inner_area) >= threshold


def analyze_nesting(elements: List[Dict[str, Any]]) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int, float]]]:
    """
    Анализирует вложенность и пересечения элементов.
    
    Returns:
        nested: список (parent_idx, child_idx) где parent полностью содержит child
        overlapping: список (idx1, idx2, iou) где элементы пересекаются
    """
    nested = []
    overlapping = []
    
    for i, e1 in enumerate(elements):
        bbox1 = e1.get("bbox", [])
        if len(bbox1) < 4:
            continue
            
        for j, e2 in enumerate(elements):
            if i >= j:
                continue
            bbox2 = e2.get("bbox", [])
            if len(bbox2) < 4:
                continue
            
            # Проверяем вложенность
            if bbox_contains(bbox1, bbox2, 0.85):
                nested.append((i, j))
            elif bbox_contains(bbox2, bbox1, 0.85):
                nested.append((j, i))
            else:
                # Проверяем пересечение
                iou = compute_iou(bbox1, bbox2)
                if iou > 0.2:
                    overlapping.append((i, j, iou))
    
    return nested, overlapping


def diagnose_visual_candidates(
    elements: List[Dict[str, Any]],
    container_bbox: List[float],
) -> DetectionDiagnostics:
    """
    Собирает диагностику визуальных кандидатов.
    """
    diag = DetectionDiagnostics()
    
    if len(container_bbox) >= 4:
        diag.container_bbox = container_bbox
        diag.container_area = (container_bbox[2] - container_bbox[0]) * (container_bbox[3] - container_bbox[1])
    
    diag.total_visual_candidates = len(elements)
    
    # По источникам
    for e in elements:
        src = e.get("source", "unknown")
        diag.visual_by_source[src] = diag.visual_by_source.get(src, 0) + 1
        
        etype = e.get("element_type", "unknown")
        diag.by_element_type[etype] = diag.by_element_type.get(etype, 0) + 1
    
    # Вложенности и пересечения
    diag.nested_elements, diag.overlapping_pairs = analyze_nesting(elements)
    
    return diag


def apply_suppression_with_log(
    elements: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Применяет NMS с логированием причин удаления.
    
    Returns:
        filtered: отфильтрованные элементы
        reasons: список причин удаления
    """
    if not elements:
        return [], []
    
    # Сортируем по confidence (выше = лучше)
    sorted_elements = sorted(elements, key=lambda e: -e.get("confidence", 0.0))
    
    keep = []
    reasons = []
    
    for e in sorted_elements:
        bbox = e.get("bbox", [])
        if len(bbox) < 4:
            reasons.append(f"invalid bbox: {e.get('element_type', '?')}")
            continue
        
        should_keep = True
        for kept in keep:
            kept_bbox = kept.get("bbox", [])
            iou = compute_iou(bbox, kept_bbox)
            if iou > iou_threshold:
                should_keep = False
                reasons.append(
                    f"suppressed {e.get('element_type', '?')} (conf={e.get('confidence', 0):.2f}) "
                    f"by {kept.get('element_type', '?')} (conf={kept.get('confidence', 0):.2f}), iou={iou:.2f}"
                )
                break
        
        if should_keep:
            keep.append(e)
    
    return keep, reasons


def separate_containers_and_leaves(
    elements: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Разделяет элементы на контейнеры (содержащие другие элементы) и leaf.
    
    Если элемент A полностью содержит элемент B:
    - A считается container
    - B считается leaf
    """
    containers = []
    leaves = []
    container_indices = set()
    
    for i, e1 in enumerate(elements):
        bbox1 = e1.get("bbox", [])
        if len(bbox1) < 4:
            continue
        
        is_container = False
        for j, e2 in enumerate(elements):
            if i == j:
                continue
            bbox2 = e2.get("bbox", [])
            if len(bbox2) < 4:
                continue
            
            if bbox_contains(bbox1, bbox2, 0.8):
                is_container = True
                container_indices.add(i)
                break
        
        if is_container:
            e1_copy = e1.copy()
            e1_copy["is_container"] = True
            containers.append(e1_copy)
        else:
            leaves.append(e1)
    
    return containers, leaves


def log_diagnostics(diag: DetectionDiagnostics, form_name: str = "") -> None:
    """Выводит диагностику в лог."""
    prefix = f"[DIAG {form_name}]" if form_name else "[DIAG]"
    
    logger.info(f"{prefix} Container: bbox={diag.container_bbox}, area={diag.container_area:.0f}, coverage={diag.coverage_percent:.1f}%")
    logger.info(f"{prefix} Visual candidates: {diag.total_visual_candidates}")
    logger.info(f"{prefix}   By source: {diag.visual_by_source}")
    logger.info(f"{prefix}   By type: {diag.by_element_type}")
    logger.info(f"{prefix} Nested pairs: {len(diag.nested_elements)}")
    logger.info(f"{prefix} Overlapping pairs: {len(diag.overlapping_pairs)}")
    
    if diag.suppression_reasons:
        logger.info(f"{prefix} Suppression reasons ({len(diag.suppression_reasons)}):")
        for r in diag.suppression_reasons[:10]:
            logger.info(f"{prefix}   - {r}")
    
    if diag.rejection_reasons:
        logger.info(f"{prefix} REJECTION REASONS:")
        for r in diag.rejection_reasons:
            logger.warning(f"{prefix}   !!! {r}")
    
    logger.info(f"{prefix} Final atoms: {diag.total_atoms}, by type: {diag.atoms_by_type}")


def print_diagnostics(diag: DetectionDiagnostics, form_name: str = "") -> str:
    """Возвращает строку с диагностикой для вывода."""
    lines = []
    prefix = f"[{form_name}]" if form_name else ""
    
    lines.append(f"{prefix} === DETECTION DIAGNOSTICS ===")
    lines.append(f"{prefix} Container: bbox={diag.container_bbox}")
    lines.append(f"{prefix}   area={diag.container_area:.0f}, coverage={diag.coverage_percent:.1f}%")
    lines.append(f"{prefix} Visual candidates: {diag.total_visual_candidates}")
    lines.append(f"{prefix}   By source: {diag.visual_by_source}")
    lines.append(f"{prefix}   By type: {diag.by_element_type}")
    lines.append(f"{prefix} After classification: {diag.total_after_classification}")
    lines.append(f"{prefix} After suppression: {diag.total_after_suppression}")
    lines.append(f"{prefix} Nested pairs: {len(diag.nested_elements)}")
    
    if diag.overlapping_pairs:
        lines.append(f"{prefix} Overlapping pairs:")
        for i, j, iou in diag.overlapping_pairs[:5]:
            lines.append(f"{prefix}   [{i}] <-> [{j}] iou={iou:.2f}")
    
    if diag.suppression_reasons:
        lines.append(f"{prefix} Suppression ({len(diag.suppression_reasons)}):")
        for r in diag.suppression_reasons[:5]:
            lines.append(f"{prefix}   - {r}")
    
    if diag.rejection_reasons:
        lines.append(f"{prefix} !!! REJECTIONS:")
        for r in diag.rejection_reasons:
            lines.append(f"{prefix}   {r}")
    
    lines.append(f"{prefix} Final atoms: {diag.total_atoms}")
    lines.append(f"{prefix}   By type: {diag.atoms_by_type}")
    
    return "\n".join(lines)
