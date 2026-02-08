"""
CatBoost как soft-prior: только вероятности, без удаления атомов.

Ни один атом не удаляется. Результаты сохраняются в atom["priors"]:
- atom["priors"]["interactive_score"] = max(role_probs.values())  — единственный сигнал до назначения роли
- atom["priors"]["role_probs"] = {"button": ..., "input": ..., ...}

До semantic_validation используется только interactive_score (вопрос: «похож ли bbox на UI-контрол»).
role_probs используются только после interactive_valid в фазе назначения ролей.
Запрещено назначать atom["ui_role"] и фильтровать атомы.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

FEATURE_ORDER = [
    "aspect_ratio", "area", "bbox_width", "bbox_height", "bbox_coverage_ocr",
    "relative_size_to_region", "num_adjacent", "num_aligned_row", "num_aligned_col",
    "row_group_size", "column_group_size", "uniform_spacing_score", "region_density",
    "has_label", "has_action_word", "saved_by_anchor", "semantic_lock", "is_inside_region",
]
NUMERIC_TO_LOG1P = ["area", "bbox_width", "bbox_height"]
ASPECT_RATIO_CLIP = (0.0, 30.0)

_MODELS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "models"
DEFAULT_CANDIDATE_MODEL_PATH = _MODELS_DIR / "ui_atom_candidate.cbm"
DEFAULT_ROLE_MODEL_PATH = _MODELS_DIR / "ui_atom_catboost.cbm"
# CatBoost v2: multi-class (button, checkbox, input, link, textarea), обучен на Detectron2 bbox (teacher_dataset_builder_v2)
DEFAULT_ROLE_MODEL_PATH_V2 = _MODELS_DIR / "ui_atom_role_catboost_v2.cbm"
# Порядок классов в v2 совпадает с train_ui_atom_catboost_v2 (sorted labels)
ROLE_CLASSES_V2 = ["button", "checkbox", "input", "link", "textarea"]

# Порог bbox_coverage_ocr для маркировки bbox_quality (грязный Detectron bbox vs tight teacher bbox)
BBOX_COVERAGE_TIGHT_THRESHOLD = 0.3


def _normalize_features(feats: Dict[str, float]) -> Dict[str, float]:
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
    return [float(feats.get(k, 0)) for k in FEATURE_ORDER]


def _load_model(path: Optional[Path]) -> Any:
    if path is None or (isinstance(path, str) and not path) or not Path(path).exists():
        return None
    try:
        from catboost import CatBoostClassifier
        model = CatBoostClassifier()
        model.load_model(str(path))
        return model
    except Exception as e:
        logger.debug("catboost_priors: failed to load model %s: %s", path, e)
        return None


def _atom_feats(
    atom: Dict[str, Any],
    features_by_atom: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    aid = atom.get("id", "")
    feats = dict(features_by_atom.get(aid, {}))
    bbox = atom.get("bbox") or []
    if len(bbox) >= 4:
        feats["bbox_width"] = float(bbox[2] - bbox[0])
        feats["bbox_height"] = float(bbox[3] - bbox[1])
    else:
        feats["bbox_width"] = 0.0
        feats["bbox_height"] = 0.0
    feats.setdefault("saved_by_anchor", 0.0)
    feats.setdefault("semantic_lock", 0.0)
    return feats


def _detect_role_model_type(role_model: Any, vec: List[float]) -> str:
    """Возвращает 'v2' если модель multi-class (5 классов), иначе 'v1' (binary)."""
    try:
        proba = role_model.predict_proba([vec])[0]
        n = len(proba)
        if n > 2:
            return "v2"
        return "v1"
    except Exception:
        return "v1"


def run_catboost_priors(
    atoms: List[Dict[str, Any]],
    features_by_atom: Dict[str, Dict[str, float]],
    candidate_model_path: Optional[Path] = None,
    role_model_path: Optional[Path] = None,
    role_model_path_v2: Optional[Path] = None,
) -> None:
    """
    Мягкие priors: для каждого атома записывает atom["priors"]["interactive_score"]
    и atom["priors"]["role_probs"]. Ничего не удаляет, ui_role не назначает.
    interactive_score = max(role_probs) — единственный сигнал до semantic_validation (фаза interactive gate).
    Сначала пробует CatBoost v2 (multi-class), при отсутствии — v1 (binary). Fallback: interactive_score = 0.5.
    """
    candidate_path = candidate_model_path or DEFAULT_CANDIDATE_MODEL_PATH
    role_path_v2 = role_model_path_v2 or DEFAULT_ROLE_MODEL_PATH_V2
    role_path_v1 = role_model_path or DEFAULT_ROLE_MODEL_PATH
    if isinstance(candidate_path, str):
        candidate_path = Path(candidate_path)
    if isinstance(role_path_v2, str):
        role_path_v2 = Path(role_path_v2)
    if isinstance(role_path_v1, str):
        role_path_v1 = Path(role_path_v1)
    candidate_model = _load_model(candidate_path)
    role_model = _load_model(role_path_v2)
    role_model_v1 = _load_model(role_path_v1) if role_model is None else None
    if role_model is None:
        role_model = role_model_v1
    if candidate_model is None and role_model is None:
        for a in atoms:
            p = a.setdefault("priors", {})
            p["role_probs"] = {"button": 0.5, "input": 0.5}
            p["interactive_score"] = 0.5
        logger.debug("catboost_priors: no models, priors set to 0.5")
        return

    role_model_type = "v1"
    if role_model is not None and atoms:
        first_feats = _atom_feats(atoms[0], features_by_atom)
        first_norm = _normalize_features(first_feats)
        first_vec = _feature_vector(first_norm)
        role_model_type = _detect_role_model_type(role_model, first_vec)
        if role_model_type == "v2":
            logger.debug("catboost_priors: using v2 multi-class model")
        else:
            logger.debug("catboost_priors: using binary role model")

    for a in atoms:
        priors: Dict[str, Any] = a.setdefault("priors", {})
        feats = _atom_feats(a, features_by_atom)
        bbox_coverage = float(feats.get("bbox_coverage_ocr", 0))
        a["bbox_quality"] = "tight" if bbox_coverage >= BBOX_COVERAGE_TIGHT_THRESHOLD else "noisy"
        feats_norm = _normalize_features(feats)
        vec = _feature_vector(feats_norm)
        try:
            if role_model is not None:
                proba = role_model.predict_proba([vec])[0]
                if role_model_type == "v2" and len(proba) >= len(ROLE_CLASSES_V2):
                    role_probs = dict(zip(ROLE_CLASSES_V2, (float(p) for p in proba[: len(ROLE_CLASSES_V2)])))
                else:
                    p_button = float(proba[1]) if len(proba) > 1 else float(proba[0])
                    p_input = 1.0 - p_button
                    role_probs = {"button": p_button, "input": p_input}
                priors["role_probs"] = role_probs
                priors["interactive_score"] = float(max(role_probs.values()))
            else:
                priors["role_probs"] = {"button": 0.5, "input": 0.5}
                priors["interactive_score"] = 0.5
        except Exception as e:
            logger.debug("catboost_priors role failed for %s: %s", a.get("id"), e)
            priors["role_probs"] = {"button": 0.5, "input": 0.5}
            priors["interactive_score"] = 0.5

        if candidate_model is not None:
            try:
                proba = candidate_model.predict_proba([vec])[0]
                priors["interactive_score"] = float(proba[1]) if len(proba) > 1 else float(proba[0])
            except Exception as e:
                logger.debug("catboost_priors candidate failed for %s: %s", a.get("id"), e)


def input_bbox_prepass(
    atoms: List[Dict[str, Any]],
    raw_ocr_boxes: List[Dict[str, Any]],
    expand_px: int = 8,
    merge_ocr_iou_threshold: float = 0.3,
) -> None:
    """
    Расширение bbox только для атомов с interactive_valid и type in (input, weak_input) (после semantic_validation).
    Запрещено расширять bbox для атомов без семантического подтверждения.
    Модифицирует atoms in-place.
    """
    input_types = ("input", "weak_input")
    for a in atoms:
        if not a.get("interactive_valid"):
            continue
        if (a.get("type") or "").lower() not in input_types:
            continue
        bbox = a.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        x1 = max(0.0, x1 - expand_px)
        y1 = max(0.0, y1 - expand_px)
        x2 = x2 + expand_px
        y2 = y2 + expand_px
        for ob in raw_ocr_boxes:
            ob_bbox = ob.get("bbox")
            if not ob_bbox or len(ob_bbox) < 4:
                continue
            ox1, oy1, ox2, oy2 = ob_bbox[0], ob_bbox[1], ob_bbox[2], ob_bbox[3]
            inter_w = max(0, min(x2, ox2) - max(x1, ox1))
            inter_h = max(0, min(y2, oy2) - max(y1, oy1))
            inter_area = inter_w * inter_h
            ob_area = (ox2 - ox1) * (oy2 - oy1)
            if ob_area <= 0:
                continue
            iou = inter_area / ob_area if ob_area else 0
            if iou >= merge_ocr_iou_threshold or (inter_area > 0 and ob_area <= 2 * inter_area):
                x1 = min(x1, ox1)
                y1 = min(y1, oy1)
                x2 = max(x2, ox2)
                y2 = max(y2, oy2)
        a["bbox"] = [x1, y1, x2, y2]
