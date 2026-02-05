"""
Layout-first: OCR → LayoutLMv3 → layout blocks. Detectron2 не используется как источник layout.

Порядок: 1) OCR (слова/строки + bbox), 2) LayoutLMv3 (layout-блоки по токенам),
3) Иерархия по вложенности, без глобального page-frame.
Вход LayoutLMv3: изображение + OCR bbox + текст. Выход: блоки (header, section, list, text, button-like).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Маппинг label LayoutLMv3 → наш type (header, section, card, list, text_region, button)
LAYOUTLMV3_LABEL_TO_TYPE: Dict[str, str] = {
    "O": "text_region",
    "B-HEADER": "header",
    "I-HEADER": "header",
    "B-BODY": "panel",
    "I-BODY": "panel",
    "B-TITLE": "header",
    "I-TITLE": "header",
    "B-LIST": "section",
    "I-LIST": "section",
    "B-TABLE": "section",
    "I-TABLE": "section",
    "B-FIGURE": "card",
    "I-FIGURE": "card",
}


def run_ocr_first(
    image_path: str,
    run_text_detect: Any,
    run_ocr_boxes: Any,
) -> Tuple[List[Dict[str, Any]], List[str], int, int]:
    """
    Шаг 1: OCR по всему изображению. Детекция слов/строк, затем OCR по bbox.
    Возвращает (words_with_bbox, texts, img_w, img_h). words_with_bbox = [{"x","y","w","h"}, ...].
    """
    path = Path(image_path)
    if not path.exists():
        return [], [], 0, 0
    img_w, img_h = 0, 0
    try:
        from PIL import Image
        with Image.open(image_path) as im:
            img_w, img_h = im.width, im.height
    except Exception:
        pass
    boxes = run_text_detect(image_path)
    if not boxes:
        return [], [], img_w, img_h
    ocr_results = run_ocr_boxes(image_path, boxes)
    words: List[Dict[str, Any]] = []
    for i, b in enumerate(boxes):
        t = ocr_results[i].get("text", "").strip() if i < len(ocr_results) else ""
        words.append({
            "x": b.get("x", 0), "y": b.get("y", 0),
            "w": b.get("w", 0), "h": b.get("h", 0),
            "text": t,
        })
    words.sort(key=lambda w: (w.get("y", 0), w.get("x", 0)))
    texts = [w.get("text", "") for w in words]
    return words, texts, img_w, img_h


def layoutlmv3_predict(
    image_path: str,
    words: List[Dict[str, Any]],
    texts: List[str],
    img_w: int,
    img_h: int,
) -> List[str]:
    """
    LayoutLMv3 token classification: предсказание метки для каждого токена (word).
    Возвращает список label для каждого слова.
    Сейчас: эвристика по bbox (верх 15% = HEADER). Подключение microsoft/layoutlmv3-base
    или fine-tuned layout-NER — при наличии transformers и весов.
    """
    if not words or not texts:
        return []
    # Опционально: загрузка LayoutLMv3ForTokenClassification + LayoutLMv3Processor для реального inference.
    # Без fine-tuned layout-NER весов используем эвристику по геометрии.
    labels: List[str] = []
    for w in words:
        y = w.get("y", 0)
        if img_h > 0 and y < img_h * 0.15:
            labels.append("B-HEADER")
        else:
            labels.append("O")
    return labels


def tokens_to_blocks(
    words: List[Dict[str, Any]],
    labels: List[str],
    img_w: int,
    img_h: int,
) -> List[Dict[str, Any]]:
    """
    Группировка токенов с одинаковым type (B-HEADER/I-HEADER → один блок) в блоки (merge bbox).
    """
    if not words or len(labels) != len(words):
        return []
    blocks: List[Dict[str, Any]] = []
    i = 0
    while i < len(words):
        label = labels[i] if i < len(labels) else "O"
        typ = LAYOUTLMV3_LABEL_TO_TYPE.get(label, "text_region")
        xs, ys, x2s, y2s = [words[i]["x"]], [words[i]["y"]], [words[i]["x"] + words[i]["w"]], [words[i]["y"] + words[i]["h"]]
        j = i + 1
        while j < len(words):
            lj = labels[j] if j < len(labels) else "O"
            tj = LAYOUTLMV3_LABEL_TO_TYPE.get(lj, "text_region")
            if tj != typ:
                break
            xs.append(words[j]["x"])
            ys.append(words[j]["y"])
            x2s.append(words[j]["x"] + words[j]["w"])
            y2s.append(words[j]["y"] + words[j]["h"])
            j += 1
        x1, y1 = min(xs), min(ys)
        w = max(x2s) - x1
        h = max(y2s) - y1
        if w > 0 and h > 0 and (img_w <= 0 or w < img_w * 0.95) and (img_h <= 0 or h < img_h * 0.95):
            blocks.append({"x": x1, "y": y1, "w": w, "h": h, "type": typ, "parent_region_id": None})
        i = j
    return blocks


def build_hierarchy_no_page_frame(
    blocks: List[Dict[str, Any]],
    img_w: int,
    img_h: int,
) -> List[Dict[str, Any]]:
    """
    Назначение parent_region_id по вложенности. Запрет: регион с геометрией page-frame
    (широкая низкая полоса) не может быть родителем контента ниже.
    """
    if not blocks:
        return blocks
    WIDE = 0.8
    SHORT = 0.15
    for i, child in enumerate(blocks):
        cx, cy, cw, ch = child["x"], child["y"], child["w"], child["h"]
        cy_center = cy + ch // 2
        best_parent = None
        best_area = 0
        for j, par in enumerate(blocks):
            if j == i:
                continue
            px, py, pw, ph = par["x"], par["y"], par["w"], par["h"]
            if px > cx or py > cy or px + pw < cx + cw or py + ph < cy + ch:
                continue
            if img_w > 0 and img_h > 0:
                if pw >= WIDE * img_w and ph <= SHORT * img_h and cy_center > py + ph:
                    continue
            area = pw * ph
            if best_parent is None or area < best_area:
                best_parent = j
                best_area = area
        child["parent_region_id"] = best_parent
    return blocks


def run_layoutlmv3_regions(
    image_path: str,
    run_text_detect: Any,
    run_ocr_boxes: Any,
) -> List[Dict[str, Any]]:
    """
    Layout-first: OCR → LayoutLMv3 → layout blocks → иерархия.
    Возвращает список регионов в формате run_ui_regions: {x, y, w, h, type, parent_region_id}.
    При недоступности LayoutLMv3 возвращает [] (вызывающий делает fallback на DL/CV).
    """
    words, texts, img_w, img_h = run_ocr_first(image_path, run_text_detect, run_ocr_boxes)
    if not words:
        return []
    labels = layoutlmv3_predict(image_path, words, texts, img_w, img_h)
    if not labels:
        labels = ["O"] * len(words)
    blocks = tokens_to_blocks(words, labels, img_w, img_h)
    if not blocks:
        return []
    blocks = build_hierarchy_no_page_frame(blocks, img_w, img_h)
    for r in blocks:
        r["_detector_source"] = "layoutlmv3"
    logger.info("layoutlmv3: image_path=%s blocks=%d", image_path, len(blocks))
    return blocks
