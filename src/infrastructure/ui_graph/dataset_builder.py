"""
Сбор датасета для обучения CatBoost (UI Atom Classification).

Один атом = одна строка. Только бинарные классы: button, input.
Включение: ui_role ∈ {button, input} или weak_button/weak_input (weak samples), semantic_lock, confidence пороги.
Признаки — только из features_by_atom (числовые и булевы), без OCR-текста.
Сохраняет в CSV, совместимый с CatBoost.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_PATH = "datasets/ui_atoms_catboost.csv"
SEMANTIC_CONFIDENCE_MIN = 0.85
WEAK_CONFIDENCE_MIN = 0.6

# Числовые признаки (только из features_by_atom или bbox)
FEATURE_NUMERIC = [
    "aspect_ratio",
    "area",
    "bbox_width",
    "bbox_height",
    "bbox_coverage_ocr",
    "relative_size_to_region",
    "num_adjacent",
    "num_aligned_row",
    "num_aligned_col",
    "row_group_size",
    "column_group_size",
    "uniform_spacing_score",
    "region_density",
]
# Булевы (0/1)
FEATURE_BOOL = [
    "has_label",
    "has_action_word",
    "saved_by_anchor",
    "semantic_lock",
    "is_inside_region",
]
# Служебные (не для обучения)
SERVICE_COLUMNS = ["image_id", "atom_id", "source_stage"]
CSV_HEADER = ["label", "label_quality"] + FEATURE_NUMERIC + FEATURE_BOOL + SERVICE_COLUMNS


def _bbox_width_height(bbox: List[float]) -> tuple:
    if not bbox or len(bbox) < 4:
        return 0.0, 0.0
    return float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])


def _row_from_atom(
    atom: Dict[str, Any],
    features: Dict[str, float],
    label: str,
    label_quality: str,
    image_id: str,
) -> Dict[str, str]:
    """Одна строка датасета: label, label_quality, признаки, служебные поля."""
    bbox = atom.get("bbox") or []
    bw, bh = _bbox_width_height(bbox)
    row = {
        "label": label,
        "label_quality": label_quality,
        "aspect_ratio": str(features.get("aspect_ratio", 0)),
        "area": str(features.get("area", 0)),
        "bbox_width": str(bw),
        "bbox_height": str(bh),
        "bbox_coverage_ocr": str(features.get("bbox_coverage_ocr", 0)),
        "relative_size_to_region": str(features.get("relative_size_to_region", 0)),
        "num_adjacent": str(int(features.get("num_adjacent", 0))),
        "num_aligned_row": str(int(features.get("num_aligned_row", 0))),
        "num_aligned_col": str(int(features.get("num_aligned_col", 0))),
        "row_group_size": str(int(features.get("row_group_size", 1))),
        "column_group_size": str(int(features.get("column_group_size", 1))),
        "uniform_spacing_score": str(features.get("uniform_spacing_score", 0)),
        "region_density": str(features.get("region_density", 0)),
        "has_label": "1" if features.get("has_label", 0) >= 0.5 else "0",
        "has_action_word": "1" if features.get("has_action_word", 0) >= 0.5 else "0",
        "saved_by_anchor": "1" if atom.get("saved_by_anchor") else "0",
        "semantic_lock": "1" if atom.get("semantic_lock") else "0",
        "is_inside_region": "1" if features.get("is_inside_region", 0) >= 0.5 else "0",
        "image_id": image_id,
        "atom_id": str(atom.get("id", "")),
        "source_stage": label_quality,
    }
    return row


def collect_catboost_dataset(
    atoms: List[Dict[str, Any]],
    features_by_atom: Dict[str, Dict[str, float]],
    image_id: str,
    output_path: str = DEFAULT_OUTPUT_PATH,
) -> Dict[str, int]:
    """
    Собирает строки датасета из атомов и признаков. Один атом = одна строка.

    Включение (semantic-quality):
      ui_role ∈ {button, input}, semantic_lock, ui_role_confidence >= 0.85, type != "layout",
      атом не был понижен в semantic_validation (type не layout).
    Weak samples: ui_role in (weak_button, weak_input), ui_role_confidence >= 0.6, semantic_lock;
      weak_input → label="input", label_quality="weak".

    Если файл существует — append, иначе создаётся с header.
    Возвращает: {added_total, added_semantic, added_weak, skipped}.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    added_semantic = 0
    added_weak = 0
    skipped = 0
    rows: List[Dict[str, str]] = []

    for atom in atoms:
        aid = atom.get("id", "")
        atype = (atom.get("type") or "").lower()
        ui_role = (atom.get("ui_role") or "").lower()
        ui_conf = float(atom.get("ui_role_confidence") or 0)
        semantic_lock = bool(atom.get("semantic_lock", False))

        if atype == "layout":
            skipped += 1
            continue
        feats = features_by_atom.get(aid, {})
        if not feats:
            skipped += 1
            continue

        # Semantic-quality: button / input
        if ui_role in ("button", "input") and semantic_lock and ui_conf >= SEMANTIC_CONFIDENCE_MIN:
            rows.append(_row_from_atom(atom, feats, ui_role, "semantic", image_id))
            added_semantic += 1
            continue
        # Weak: weak_button
        if ui_role == "weak_button" and semantic_lock and ui_conf >= WEAK_CONFIDENCE_MIN:
            rows.append(_row_from_atom(atom, feats, "button", "weak", image_id))
            added_weak += 1
            continue
        # Weak: weak_input
        if ui_role == "weak_input" and semantic_lock and ui_conf >= WEAK_CONFIDENCE_MIN:
            rows.append(_row_from_atom(atom, feats, "input", "weak", image_id))
            added_weak += 1
            continue
        skipped += 1

    added_total = len(rows)
    if rows:
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)
    stats = {
        "added_total": added_total,
        "added_semantic": added_semantic,
        "added_weak": added_weak,
        "skipped": skipped,
    }
    logger.debug(
        "dataset_builder: added_total=%s semantic=%s weak=%s skipped=%s",
        added_total, added_semantic, added_weak, skipped,
    )
    return stats
