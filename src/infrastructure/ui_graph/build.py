"""
Сборка UI-графа и применение ролей к атомам.

v3: semantic_validation — ЕДИНСТВЕННЫЙ слой, назначающий семантические роли.
ui_graph (v3) работает только с atoms_for_interaction (semantic_lock); не назначает ui_role, не повышает интерактивность.

Legacy: run_ui_graph_pipeline — classify_roles + apply_roles_to_atoms (не использовать в v3).
v3: run_ui_graph_pipeline_v3 — только структура; ui_role = semantic_role (read-only).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from src.infrastructure.ui_graph.graph import UIGraph, AtomNode, OCRNode, RegionNode
from src.infrastructure.ui_graph.edges import build_edges
from src.infrastructure.ui_graph.features import extract_features
from src.infrastructure.ui_graph.classifier import classify_roles
from src.infrastructure.ui_graph.roles import UIRole

logger = logging.getLogger(__name__)

NOISE_DROP_CONFIDENCE_THRESHOLD = 0.4
SEMANTIC_PROMOTION_OCR_CONF_MIN = 0.8
SEMANTIC_PROMOTION_OCR_TEXT_LEN_MAX = 20


def _normalize_bbox(item: Dict[str, Any]) -> List[float]:
    """Приводит bbox к [x1, y1, x2, y2] из bbox или x,y,w,h."""
    bbox = item.get("bbox")
    if bbox and len(bbox) >= 4:
        return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
    x, y = float(item.get("x", 0)), float(item.get("y", 0))
    w, h = float(item.get("w", 0)), float(item.get("h", 0))
    return [x, y, x + w, y + h]


def build_ui_graph(
    atoms: List[Dict[str, Any]],
    raw_ocr_boxes: List[Dict[str, Any]],
    regions: List[Dict[str, Any]],
) -> UIGraph:
    """
    Строит UI-граф из atoms, raw_ocr_boxes, regions.
    Не создаёт bbox, не меняет CV — только агрегирует узлы и строит рёбра.
    """
    graph = UIGraph()
    for a in atoms:
        aid = a.get("id", "")
        if not aid:
            continue
        bbox = _normalize_bbox(a)
        graph.add_atom(AtomNode(
            id=aid,
            type=a.get("type", "unknown"),
            bbox=bbox,
            confidence=float(a.get("confidence", 0)),
            source=a.get("source", "real"),
        ))
    for ob in raw_ocr_boxes:
        oid = ob.get("id", "")
        if not oid:
            continue
        bbox = _normalize_bbox(ob)
        graph.add_ocr(OCRNode(
            id=oid,
            text=(ob.get("text") or "").strip(),
            bbox=bbox,
            confidence=float(ob.get("confidence", 0)),
        ))
    for r in regions:
        rid = r.get("id", "")
        if not rid:
            continue
        bbox = _normalize_bbox(r)
        graph.add_region(RegionNode(
            id=rid,
            bbox=bbox,
            shape_type=r.get("shape_type", "rect"),
        ))
    build_edges(graph)
    return graph


def apply_roles_to_atoms(
    atoms: List[Dict[str, Any]],
    role_predictions: Dict[str, Tuple[UIRole, float]],
    log_override: bool = True,
) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, int]]:
    """
    Применяет ui_role к списку атомов.
    - bbox не меняется.
    - semantic_lock = semantic_valid or saved_by_anchor: при True не дропаем, не меняем type.
    - Дроп только если: ui_role == noise AND not semantic_lock AND atom.type == layout AND confidence < threshold.
    - Иначе ui_role=noise — низкоприоритетный элемент, остаётся в списке.
    stats: dropped_noise, blocked_by_semantic_lock, attempted_drop_but_locked, weak_roles_assigned.
    """
    stats: Dict[str, int] = {
        "dropped_noise": 0,
        "blocked_by_semantic_lock": 0,
        "attempted_drop_but_locked": 0,
        "weak_roles_assigned": 0,
    }
    log_lines: List[str] = []
    final: List[Dict[str, Any]] = []
    for a in atoms:
        aid = a.get("id", "")
        if not aid:
            final.append(a)
            continue
        pred = role_predictions.get(aid)
        semantic_lock = a.get("semantic_valid", False) or a.get("saved_by_anchor", False)
        if pred is None:
            final.append(a)
            continue
        role, conf = pred
        if role == UIRole.NOISE:
            if semantic_lock:
                stats["attempted_drop_but_locked"] = stats.get("attempted_drop_but_locked", 0) + 1
                stats["blocked_by_semantic_lock"] = stats.get("blocked_by_semantic_lock", 0) + 1
                log_lines.append(f"ui_graph blocked drop: atom_id={aid} | atom.type={a.get('type')} semantic_lock=True")
                a_copy = dict(a)
                a_copy["ui_role"] = UIRole.NOISE.value
                a_copy["ui_role_confidence"] = conf
                final.append(a_copy)
                continue
            atom_type = a.get("type", "")
            if atom_type == "layout" and conf < NOISE_DROP_CONFIDENCE_THRESHOLD:
                stats["dropped_noise"] = stats.get("dropped_noise", 0) + 1
                log_lines.append(f"ui_graph drop: atom_id={aid} | atom.type={atom_type} -> noise (layout, low conf)")
                continue
            a_copy = dict(a)
            a_copy["ui_role"] = UIRole.NOISE.value
            a_copy["ui_role_confidence"] = conf
            final.append(a_copy)
            continue
        if role in (UIRole.WEAK_BUTTON, UIRole.WEAK_LINK, UIRole.WEAK_INPUT):
            stats["weak_roles_assigned"] = stats.get("weak_roles_assigned", 0) + 1
        a_copy = dict(a)
        a_copy["ui_role"] = role.value
        a_copy["ui_role_confidence"] = conf
        atom_type = a.get("type", "")
        if role.value != atom_type and log_override and not semantic_lock:
            key = f"override_{atom_type}_to_{role.value}"
            stats[key] = stats.get(key, 0) + 1
            log_lines.append(f"ui_graph override: atom_id={aid} | atom.type={atom_type} -> ui_role={role.value}")
        elif role.value != atom_type and semantic_lock:
            stats["blocked_by_semantic_lock"] = stats.get("blocked_by_semantic_lock", 0) + 1
        final.append(a_copy)
    return final, log_lines, stats


def _semantic_promotion(
    final_atoms: List[Dict[str, Any]],
    graph: UIGraph,
    role_predictions: Dict[str, Tuple[UIRole, float]],
    features_by_atom: Dict[str, Dict[str, float]],
) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, int]]:
    """
    Semantic promotion: layout → button/link если semantic_lock + ui_role in (WEAK_BUTTON, WEAK_LINK)
    или OCR внутри bbox с conf ≥ 0.8 и длиной текста ≤ 20.
    Модифицирует final_atoms in-place, возвращает (final_atoms, log_lines, stats).
    """
    log_lines: List[str] = []
    stats: Dict[str, int] = {"semantic_promoted_from_layout": 0}
    for a in final_atoms:
        aid = a.get("id", "")
        atype = (a.get("type") or "").lower()
        if atype not in ("layout", "layout_candidate"):
            continue
        semantic_lock = a.get("semantic_valid", False) or a.get("saved_by_anchor", False)
        ui_role_val = a.get("ui_role")
        try:
            role = UIRole(ui_role_val) if ui_role_val else None
        except ValueError:
            role = None
        feats = features_by_atom.get(aid, {})
        ocr_count = int(feats.get("ocr_inside_count", 0))
        ocr_conf = feats.get("ocr_inside_mean_conf", 0)
        ocr_text_len = int(feats.get("ocr_inside_text_len", 0))
        ocr_strong = ocr_count >= 1 and ocr_conf >= SEMANTIC_PROMOTION_OCR_CONF_MIN and ocr_text_len <= SEMANTIC_PROMOTION_OCR_TEXT_LEN_MAX
        promote = False
        new_type = ""
        if semantic_lock and role in (UIRole.WEAK_BUTTON, UIRole.WEAK_LINK, UIRole.BUTTON, UIRole.LINK):
            promote = True
            new_type = "button" if role in (UIRole.WEAK_BUTTON, UIRole.BUTTON) else "link"
        elif ocr_strong:
            promote = True
            new_type = "button" if feats.get("aspect_ratio", 0) and 1.5 <= feats.get("aspect_ratio", 0) <= 30 else "link"
        if promote and new_type:
            a["type"] = new_type
            ui_conf = a.get("ui_role_confidence", 0) or 0
            a["confidence"] = max((a.get("confidence") or 0), ui_conf)
            stats["semantic_promoted_from_layout"] = stats.get("semantic_promoted_from_layout", 0) + 1
            log_lines.append(f"semantic_promoted_from_layout: atom_id={aid} | layout -> {new_type}")
    return final_atoms, log_lines, stats


def _check_semantic_visibility_invariant(
    final_atoms: List[Dict[str, Any]],
) -> List[str]:
    """Инвариант: не должно быть semantic_lock and type==layout and ui_role in (button, weak_button, input, weak_input). Лог semantic_visibility_bug."""
    log_lines: List[str] = []
    for a in final_atoms:
        if not (a.get("semantic_lock") and (a.get("type") or "").lower() == "layout"):
            continue
        ur = a.get("ui_role")
        if ur in ("button", "weak_button", "input", "weak_input"):
            log_lines.append(f"semantic_visibility_bug: atom_id={a.get('id')} | semantic_lock=True type=layout ui_role={ur}")
    return log_lines


def run_ui_graph_pipeline_v3(
    atoms: List[Dict[str, Any]],
    raw_ocr_boxes: List[Dict[str, Any]],
    regions: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], UIGraph, Dict[str, Dict[str, float]], List[str], Dict[str, int], Dict[str, Tuple[UIRole, float]]]:
    """
    v3: Работает ТОЛЬКО с atoms_for_interaction (все атомы должны иметь semantic_lock=True).
    Не назначает роли, не повышает интерактивность. ui_role = semantic_role (read-only).
    Строит граф для структуры (row/column/alignment); control_group только из >=2 semantic_lock с одной semantic_role.
    """
    assert all(a.get("semantic_lock") for a in atoms), "v3: ui_graph accepts only atoms with semantic_lock"
    log_lines: List[str] = ["ui_graph_v3: read-only semantics, %s atoms" % len(atoms)]
    stats: Dict[str, int] = {"v3_read_only": 1}
    graph = build_ui_graph(atoms, raw_ocr_boxes, regions)
    features_by_atom = extract_features(graph)
    role_predictions: Dict[str, Tuple[UIRole, float]] = {}
    for a in atoms:
        aid = a.get("id", "")
        sr = (a.get("semantic_role") or a.get("type") or "").strip().lower()
        a["ui_role"] = sr
        a["ui_role_confidence"] = 1.0 if a.get("semantic_valid") else 0.6
        try:
            role_predictions[aid] = (UIRole(sr), a["ui_role_confidence"])
        except ValueError:
            role_predictions[aid] = (UIRole.NOISE, 0.0)
    for line in log_lines:
        logger.debug("ui_graph: %s", line)
    return atoms, graph, features_by_atom, log_lines, stats, role_predictions


def run_ui_graph_pipeline(
    atoms: List[Dict[str, Any]],
    raw_ocr_boxes: List[Dict[str, Any]],
    regions: List[Dict[str, Any]],
    classifier: Optional[Any] = None,
) -> Tuple[List[Dict[str, Any]], UIGraph, Dict[str, Dict[str, float]], List[str], Dict[str, int], Dict[str, Tuple[UIRole, float]]]:
    """
    Legacy: build → extract_features → classify → apply_roles_to_atoms.
    В v3 pipeline не вызывать: использовать run_ui_graph_pipeline_v3 с atoms_for_interaction.
    """
    graph = build_ui_graph(atoms, raw_ocr_boxes, regions)
    features_by_atom = extract_features(graph)
    role_predictions = classify_roles(graph, features_by_atom, classifier)
    final_atoms, log_lines, stats = apply_roles_to_atoms(atoms, role_predictions, log_override=True)
    final_atoms, promo_log, promo_stats = _semantic_promotion(final_atoms, graph, role_predictions, features_by_atom)
    log_lines.extend(promo_log)
    for k, v in promo_stats.items():
        stats[k] = stats.get(k, 0) + v
    inv_log = _check_semantic_visibility_invariant(final_atoms)
    log_lines.extend(inv_log)
    for line in inv_log:
        logger.warning("ui_graph: %s", line)
    if len(final_atoms) == 0:
        log_lines.append("ui_graph_fallback_triggered: len(final_atoms)==0, returning atoms with confidence*=0.9, ui_role=None")
        logger.warning("ui_graph_fallback_triggered: len(final_atoms)==0")
        fallback: List[Dict[str, Any]] = []
        for a in atoms:
            ac = dict(a)
            ac["confidence"] = min(1.0, (a.get("confidence", 0) or 0) * 0.9)
            ac["ui_role"] = None
            ac["ui_role_confidence"] = 0.0
            fallback.append(ac)
        return fallback, graph, features_by_atom, log_lines, stats, role_predictions

    n_inputs = sum(1 for a in final_atoms if (a.get("ui_role") or "").lower() in ("input", "weak_input"))
    if n_inputs == 0:
        final_ids = {a.get("id") for a in final_atoms}
        re_added = 0
        for a in atoms:
            if (a.get("type") or "").lower() not in ("input", "weak_input"):
                continue
            if a.get("id") in final_ids:
                continue
            ac = dict(a)
            ac["ui_role"] = "weak_input"
            ac["ui_role_confidence"] = 0.6
            ac["semantic_lock"] = True
            final_atoms.append(ac)
            re_added += 1
        if re_added:
            log_lines.append("ui_graph_fallback_inputs: 0 inputs in final_atoms, re-added %s dropped input/weak_input as weak_input" % re_added)
            logger.warning("ui_graph_fallback_inputs: re-added %s atoms as weak_input", re_added)
            stats["fallback_inputs_re_added"] = re_added

    for line in log_lines:
        logger.debug("ui_graph: %s", line)
    return final_atoms, graph, features_by_atom, log_lines, stats, role_predictions
