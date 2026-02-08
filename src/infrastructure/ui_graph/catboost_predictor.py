"""
Применение обученной CatBoost-модели (ui_atom_catboost.cbm) для предсказания ролей атомов.

После extract_features: нормализация признаков как при обучении, predict, присвоение ui_role
и semantic_lock по pred_conf (>= 0.6 → label + semantic_lock=True, иначе weak_* + semantic_lock=False).
Не заменяет rule-based классификатор — вызывается опционально после run_ui_graph_pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Порядок признаков как в train_ui_atom_catboost (CSV без label, label_quality, image_id, atom_id, source_stage)
FEATURE_ORDER = [
    "aspect_ratio", "area", "bbox_width", "bbox_height", "bbox_coverage_ocr",
    "relative_size_to_region", "num_adjacent", "num_aligned_row", "num_aligned_col",
    "row_group_size", "column_group_size", "uniform_spacing_score", "region_density",
    "has_label", "has_action_word", "saved_by_anchor", "semantic_lock", "is_inside_region",
]
NUMERIC_TO_LOG1P = ["area", "bbox_width", "bbox_height"]
ASPECT_RATIO_CLIP = (0.0, 30.0)
CONFIDENCE_THRESHOLD = 0.6
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent.parent.parent / "models" / "ui_atom_catboost.cbm"


def _normalize_features(feats: Dict[str, float]) -> Dict[str, float]:
    """Нормализация как при обучении: log1p(area, bbox_*), clip(aspect_ratio)."""
    import math
    out = dict(feats)
    for col in NUMERIC_TO_LOG1P:
        if col in out:
            try:
                v = float(out[col])
            except (TypeError, ValueError):
                v = 0.0
            out[col] = math.log1p(max(0.0, v))
    if "aspect_ratio" in out:
        try:
            v = float(out["aspect_ratio"])
        except (TypeError, ValueError):
            v = 0.0
        out["aspect_ratio"] = max(ASPECT_RATIO_CLIP[0], min(ASPECT_RATIO_CLIP[1], v))
    for k in FEATURE_ORDER:
        if k not in out:
            out[k] = 0.0
        else:
            try:
                out[k] = float(out[k])
            except (TypeError, ValueError):
                out[k] = 0.0
    return out


def _feature_vector(feats: Dict[str, float]) -> List[float]:
    """Вектор признаков в порядке FEATURE_ORDER."""
    return [float(feats.get(k, 0)) for k in FEATURE_ORDER]


def apply_catboost_roles(
    final_atoms: List[Dict[str, Any]],
    features_by_atom: Dict[str, Dict[str, float]],
    image_id: str,
    model_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Для каждого атома в final_atoms: собирает признаки (features_by_atom + bbox_width/height),
    нормализует, предсказывает CatBoost, присваивает ui_role и semantic_lock.
    Модифицирует final_atoms in-place, возвращает тот же список.
    Если модель не загружена — атомы не меняются.
    """
    path = (model_path or DEFAULT_MODEL_PATH)
    if isinstance(path, str):
        path = Path(path)
    if not path.exists():
        logger.debug("CatBoost model not found at %s, skipping apply_catboost_roles", path)
        return final_atoms

    try:
        from catboost import CatBoostClassifier
    except ImportError:
        logger.warning("catboost not installed, skipping apply_catboost_roles")
        return final_atoms

    try:
        model = CatBoostClassifier()
        model.load_model(str(path))
    except Exception as e:
        logger.warning("CatBoost model load failed %s: %s", path, e)
        return final_atoms

    for a in final_atoms:
        aid = a.get("id", "")
        bbox = a.get("bbox") or []
        feats = dict(features_by_atom.get(aid, {}))
        if len(bbox) >= 4:
            feats["bbox_width"] = float(bbox[2] - bbox[0])
            feats["bbox_height"] = float(bbox[3] - bbox[1])
        else:
            feats["bbox_width"] = 0.0
            feats["bbox_height"] = 0.0
        feats.setdefault("saved_by_anchor", 0.0)
        feats.setdefault("semantic_lock", 0.0)
        feats_norm = _normalize_features(feats)
        vec = _feature_vector(feats_norm)
        try:
            proba = model.predict_proba([vec])[0]
            proba_button = float(proba[1]) if len(proba) > 1 else float(proba[0])
        except Exception as e:
            logger.debug("CatBoost predict failed for atom %s: %s", aid, e)
            continue
        pred_conf = max(proba_button, 1.0 - proba_button)
        pred_label = "button" if proba_button >= 0.5 else "input"
        if pred_conf >= CONFIDENCE_THRESHOLD:
            a["ui_role"] = pred_label
            a["ui_role_confidence"] = pred_conf
            a["semantic_lock"] = True
        else:
            a["ui_role"] = "weak_button" if pred_label == "button" else "weak_input"
            a["ui_role_confidence"] = pred_conf
            a["semantic_lock"] = False
    return final_atoms


def deduplicate_atoms_by_image_atom(
    atoms: List[Dict[str, Any]],
    image_id_key: str = "image_id",
) -> List[Dict[str, Any]]:
    """Оставляет первое вхождение для каждой пары (image_id, atom_id)."""
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for a in atoms:
        iid = a.get(image_id_key, "")
        aid = a.get("id", "")
        key = (iid, aid)
        if key in seen:
            continue
        seen.add(key)
        out.append(a)
    return out
