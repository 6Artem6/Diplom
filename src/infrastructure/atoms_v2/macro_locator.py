"""
MacroLocator — макроуровень (GPS): крупные блоки страницы.

Находит header, footer, navbar, карточки формы. Сохраняет связи «блок внутри блока».
Не изменяет существующие функции; использует detect_form_regions и regions как источник.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Доли высоты экрана для эвристики header/footer
HEADER_TOP_RATIO = 0.15
FOOTER_BOTTOM_RATIO = 0.12
NAVBAR_HEIGHT_RATIO = 0.08


@dataclass
class LargeBlock:
    """Крупный блок страницы (макроуровень)."""
    bbox: List[float]
    block_type: str  # "header" | "footer" | "navbar" | "form_card" | "section"
    parent_id: Optional[str] = None
    confidence: float = 0.8


def detect_large_blocks(
    image_path: Optional[str],
    regions: List[Dict[str, Any]],
    atoms: List[Dict[str, Any]],
    raw_ocr_boxes: List[Dict[str, Any]],
    form_regions: Optional[List[Dict[str, Any]]] = None,
) -> List[LargeBlock]:
    """
    Определяет крупные блоки: header, footer, navbar, карточки формы.
    Вход: скриншот (путь), CV regions, atoms, OCR. form_regions опционально (если уже есть).
    Выход: список LargeBlock с bbox и типом. Связи parent_id по пересечению.
    """
    blocks: List[LargeBlock] = []
    if not regions:
        return blocks

    # Размер страницы по регионам
    rbboxes = [r.get("bbox", [0, 0, 0, 0]) for r in regions if len((r.get("bbox") or [])) >= 4]
    if not rbboxes:
        return blocks
    page_w = max(b[2] for b in rbboxes)
    page_h = max(b[3] for b in rbboxes)
    if page_h <= 0 or page_w <= 0:
        return blocks

    # Form cards — из переданных form_regions или через detect_form_regions
    cards: List[Dict[str, Any]] = []
    if form_regions:
        cards = form_regions
    else:
        try:
            from src.infrastructure.atoms_v2.form_structure_detection import detect_form_regions
            cards = detect_form_regions(regions, atoms, raw_ocr_boxes)
        except Exception as e:
            logger.debug("macro_locator: detect_form_regions failed: %s", e)
    for i, card in enumerate(cards):
        bbox = card.get("bbox", [0, 0, 0, 0])
        if len(bbox) >= 4:
            bid = card.get("id") or ("form_card_%d" % i)
            blocks.append(LargeBlock(bbox=list(bbox), block_type="form_card", parent_id=None, confidence=0.85))

    # Header: верхняя полоса (эвристика по положению)
    header_h = page_h * HEADER_TOP_RATIO
    if header_h >= 20:
        blocks.append(LargeBlock(
            bbox=[0.0, 0.0, page_w, header_h],
            block_type="header",
            parent_id=None,
            confidence=0.5,
        ))

    # Footer: нижняя полоса
    footer_h = page_h * FOOTER_BOTTOM_RATIO
    if footer_h >= 20:
        blocks.append(LargeBlock(
            bbox=[0.0, page_h - footer_h, page_w, page_h],
            block_type="footer",
            parent_id=None,
            confidence=0.5,
        ))

    # Navbar: можно уточнить по регионам с type navbar, если есть
    for r in regions:
        if (r.get("type") or "").strip().lower() in ("navbar", "nav"):
            bbox = r.get("bbox", [0, 0, 0, 0])
            if len(bbox) >= 4:
                blocks.append(LargeBlock(bbox=list(bbox), block_type="navbar", parent_id=None, confidence=0.7))

    return blocks
