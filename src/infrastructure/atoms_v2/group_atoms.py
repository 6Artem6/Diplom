"""
Группировка одинаковых элементов (текст, размер, aspect, y-alignment) до semantic_validation.

Группы передаются в semantic_validation как контекст: если один элемент группы
усилился семантически, это влияет на остальных (propagation внутри semantic_validation).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple

# Бакеты для группировки: строка (y), площадь, aspect ratio
ROW_BUCKET_PX = 25
AREA_BUCKET = 80.0
ASPECT_BUCKET_SCALE = 10


def group_atoms(
    atoms: List[Dict[str, Any]],
    row_bucket_px: int = ROW_BUCKET_PX,
    area_bucket: float = AREA_BUCKET,
    aspect_bucket_scale: int = ASPECT_BUCKET_SCALE,
) -> Dict[str, List[str]]:
    """
    Группирует атомы по (строка, площадь, aspect). Каждому атому присваивается atom["group_id"].
    Возвращает словарь group_id -> [atom_id, ...] для передачи в semantic_validation.
    """
    groups: Dict[Tuple[int, int, int], List[Dict[str, Any]]] = defaultdict(list)
    for a in atoms:
        bbox = a.get("bbox") or []
        if len(bbox) < 4:
            continue
        y_center = (bbox[1] + bbox[3]) / 2
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        ar = (bbox[2] - bbox[0]) / max(bbox[3] - bbox[1], 1e-9)
        row_key = int(y_center // row_bucket_px)
        area_key = int(area // area_bucket) if area_bucket > 0 else 0
        ar_key = int(ar * aspect_bucket_scale)
        key = (row_key, area_key, ar_key)
        groups[key].append(a)

    result: Dict[str, List[str]] = {}
    for (row_k, area_k, ar_k), group_atoms_list in groups.items():
        if len(group_atoms_list) <= 1:
            for a in group_atoms_list:
                a["group_id"] = ""
            continue
        gid = f"g_{row_k}_{area_k}_{ar_k}"
        ids = [a.get("id", "") for a in group_atoms_list if a.get("id")]
        for a in group_atoms_list:
            a["group_id"] = gid
        result[gid] = ids
    return result
