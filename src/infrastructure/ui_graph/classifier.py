"""
Классификатор ролей: tabular ML (sklearn RandomForest) или rule-based fallback.

Вход: feature vector AtomNode. Выход: ui_role + confidence.
Обучение: rule-generated labels (weak supervision). Не end-to-end.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.infrastructure.ui_graph.graph import UIGraph, AtomNode
from src.infrastructure.ui_graph.roles import UIRole, rule_based_role
from src.infrastructure.ui_graph.features import extract_features

logger = logging.getLogger(__name__)

# Порядок признаков для sklearn (фиксированный порядок)
FEATURE_ORDER = [
    "aspect_ratio", "area", "bbox_coverage_ocr", "relative_size_to_region",
    "ocr_inside_count", "ocr_inside_mean_conf", "ocr_inside_text_len",
    "num_adjacent", "num_aligned_row", "num_aligned_col", "num_inputs_nearby", "num_buttons_nearby",
    "is_inside_region", "region_density",
    "has_label", "has_action_word", "text_length",
    "row_group_size", "column_group_size", "uniform_spacing_score", "mixed_types_in_row",
]


def _feature_vector(features: Dict[str, float]) -> List[float]:
    return [float(features.get(k, 0)) for k in FEATURE_ORDER]


def _rule_based_predict(
    graph: UIGraph,
    features_by_atom: Dict[str, Dict[str, float]],
) -> Dict[str, Tuple[UIRole, float]]:
    """Предсказание только по правилам. weak_button/weak_link/weak_input → confidence = 0.6."""
    result: Dict[str, Tuple[UIRole, float]] = {}
    for aid, atom in graph.atoms.items():
        feats = features_by_atom.get(aid, {})
        role = rule_based_role(atom, feats, graph)
        if role in (UIRole.WEAK_BUTTON, UIRole.WEAK_LINK, UIRole.WEAK_INPUT):
            conf = 0.6
        elif role != UIRole.NOISE:
            conf = 0.7
        else:
            conf = 0.5
        result[aid] = (role, conf)
    return result


class RoleClassifier:
    """
    Классификатор ui_role. При отсутствии обученной модели использует rule_based_role.
    """
    def __init__(self, model_path: Optional[Path] = None):
        self._model = None
        self._model_path = model_path
        self._feature_order = FEATURE_ORDER
        if model_path and model_path.exists():
            self._load_model()

    def _load_model(self) -> None:
        try:
            import joblib
            self._model = joblib.load(self._model_path)
            logger.info("ui_graph: loaded role classifier from %s", self._model_path)
        except Exception as e:
            logger.warning("ui_graph: could not load classifier %s: %s", self._model_path, e)
            self._model = None

    def fit(
        self,
        X: List[List[float]],
        y: List[str],
    ) -> None:
        """Обучает RandomForest на rule-generated или ручных метках. X — векторы по FEATURE_ORDER."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            self._model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            self._model.fit(X, y)
            if self._model_path:
                import joblib
                self._model_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(self._model, self._model_path)
                logger.info("ui_graph: saved role classifier to %s", self._model_path)
        except ImportError:
            logger.warning("ui_graph: sklearn not available, using rule-based only")

    def predict(
        self,
        graph: UIGraph,
        features_by_atom: Dict[str, Dict[str, float]],
    ) -> Dict[str, Tuple[UIRole, float]]:
        """
        Предсказание ролей. Если модель обучена — использует её; иначе rule_based.
        Возвращает atom_id -> (ui_role, confidence).
        """
        if self._model is None:
            return _rule_based_predict(graph, features_by_atom)
        out: Dict[str, Tuple[UIRole, float]] = {}
        try:
            classes = self._model.classes_
            for aid, atom in graph.atoms.items():
                feats = features_by_atom.get(aid, {})
                vec = _feature_vector(feats)
                pred = self._model.predict([vec])[0]
                proba = self._model.predict_proba([vec])[0]
                idx = list(classes).index(pred) if pred in classes else 0
                conf = float(proba[idx]) if idx < len(proba) else 0.5
                try:
                    role = UIRole(pred)
                except ValueError:
                    role = UIRole.NOISE
                out[aid] = (role, conf)
        except Exception as e:
            logger.warning("ui_graph: classifier failed, fallback to rules: %s", e)
            return _rule_based_predict(graph, features_by_atom)
        return out


def classify_roles(
    graph: UIGraph,
    features_by_atom: Optional[Dict[str, Dict[str, float]]] = None,
    classifier: Optional[RoleClassifier] = None,
) -> Dict[str, Tuple[UIRole, float]]:
    """
    Классификация ролей для всех атомов графа.
    features_by_atom — результат extract_features(graph); если None — вычисляется внутри.
    classifier — опционально обученная модель.
    """
    if features_by_atom is None:
        features_by_atom = extract_features(graph)
    clf = classifier or RoleClassifier()
    return clf.predict(graph, features_by_atom)
