#!/usr/bin/env python3
"""
Teacher-датасет v2 для CatBoost: bbox ТОЛЬКО из Detectron2 (те же, что в проде), label из DOM.

Объект обучения = atom из Detectron2 (det2.json). DOM используется только для назначения label.
Признаки считаются тем же кодом, что в inference: build_ui_graph(det_atoms) + extract_features.
Никаких DOM bbox в признаках, никакого улучшения bbox в train.

Вход на один экран (image_id): {image_id}.png, {image_id}.det2.json, {image_id}.elements.json,
опционально {image_id}.ocr.json или --with-ocr.

det2.json: список атомов Detectron2 (после postprocess), формат:
  [{"id": "...", "bbox": [x1,y1,x2,y2], "score": float, "class": "button"|"input"|"layout"|...}]
  или "atom_id"/"confidence"/"type" вместо id/score/class.

Запуск:
  python scripts/teacher_dataset_builder_v2.py --screenshots-dir datasets/ui_screenshots_catboost --output datasets/ui_atoms_teacher_v2.csv --with-ocr
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

DEFAULT_OUTPUT = "datasets/ui_atoms_teacher_v2.csv"
LABEL_QUALITY_TEACHER = "teacher"
SOURCE_STAGE_V2 = "teacher_v2"

# Совместимо с dataset_builder и train_ui_atom_catboost; добавлен dom_id
CSV_HEADER = [
    "label", "label_quality",
    "aspect_ratio", "area", "bbox_width", "bbox_height", "bbox_coverage_ocr",
    "relative_size_to_region", "num_adjacent", "num_aligned_row", "num_aligned_col",
    "row_group_size", "column_group_size", "uniform_spacing_score", "region_density",
    "has_label", "has_action_word", "saved_by_anchor", "semantic_lock", "is_inside_region",
    "image_id", "atom_id", "dom_id", "source_stage",
]

# Матчинг det → dom
IOU_MIN = 0.5
COVERAGE_MIN = 0.6  # intersection_area / area(det)
MATCH_SCORE_IOU_WEIGHT = 0.7
MATCH_SCORE_COVERAGE_WEIGHT = 0.3


def _bbox_area(bbox: List[float]) -> float:
    if not bbox or len(bbox) < 4:
        return 0.0
    return max(0.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))


def _intersection_area(a: List[float], b: List[float]) -> float:
    if len(a) < 4 or len(b) < 4:
        return 0.0
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def _iou(bbox_a: List[float], bbox_b: List[float]) -> float:
    inter = _intersection_area(bbox_a, bbox_b)
    area_a = _bbox_area(bbox_a)
    area_b = _bbox_area(bbox_b)
    if area_a <= 0 or area_b <= 0:
        return 0.0
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _coverage(bbox_inner: List[float], bbox_outer: List[float]) -> float:
    """intersection_area / area(inner)."""
    area_inner = _bbox_area(bbox_inner)
    if area_inner <= 0:
        return 0.0
    return _intersection_area(bbox_inner, bbox_outer) / area_inner


def _collect_image_ids(screenshots_dir: Path) -> List[Tuple[Path, Path, Path, Optional[Path]]]:
    """
    Собирает (det2_path, elements_path, image_path, ocr_path) для каждого image_id,
    у которого есть .det2.json и .elements.json.
    """
    out: List[Tuple[Path, Path, Path, Optional[Path]]] = []
    if not screenshots_dir.is_dir():
        return out
    for det2_path in sorted(screenshots_dir.rglob("*.det2.json")):
        image_id = det2_path.stem.removesuffix(".det2")
        dir_path = det2_path.parent
        elements_path = dir_path / f"{image_id}.elements.json"
        image_path = dir_path / f"{image_id}.png"
        ocr_path = dir_path / f"{image_id}.ocr.json"
        if not elements_path.exists() or not image_path.exists():
            logger.debug("Skip %s: missing elements or png", det2_path)
            continue
        out.append((det2_path, elements_path, image_path, ocr_path if ocr_path.exists() else None))
    return out


def _load_det2(path: Path) -> List[Dict[str, Any]]:
    """Загружает det2.json. Ожидает список {id|atom_id, bbox, score|confidence, class|type}."""
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and "atoms" in data:
        items = data["atoms"]
    else:
        return []
    atoms: List[Dict[str, Any]] = []
    for i, r in enumerate(items):
        aid = r.get("id") or r.get("atom_id") or f"det_{i}"
        bbox = r.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        score = r.get("score") if "score" in r else r.get("confidence", 0)
        cls = r.get("class") or r.get("type") or "unknown"
        atoms.append({
            "id": str(aid),
            "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
            "confidence": float(score),
            "type": str(cls),
            "source": "det2",
        })
    return atoms


def _load_elements(path: Path) -> List[Dict[str, Any]]:
    """Загружает DOM elements (Playwright). Ожидает список {id, type, bbox}."""
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "elements" in data:
        return data["elements"]
    return []


def _load_ocr_json(path: Path) -> List[Dict[str, Any]]:
    """Загружает ocr.json: список {id, text, bbox [x1,y1,x2,y2], confidence} или {x,y,w,h,text,confidence}."""
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, list):
        return []
    out: List[Dict[str, Any]] = []
    for i, b in enumerate(data):
        bbox = b.get("bbox")
        if bbox and len(bbox) >= 4:
            out.append({
                "id": b.get("id") or f"ocr_{i}",
                "text": (b.get("text") or "").strip(),
                "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                "confidence": float(b.get("confidence", 0)),
            })
        else:
            x, y = float(b.get("x", 0)), float(b.get("y", 0))
            w, h = float(b.get("w", 0)), float(b.get("h", 0))
            out.append({
                "id": b.get("id") or f"ocr_{i}",
                "text": (b.get("text") or "").strip(),
                "bbox": [x, y, x + w, y + h],
                "confidence": float(b.get("confidence", 0)),
            })
    return out


def _run_ocr_service(image_path: Path, service_url: str, timeout: int = 60) -> List[Dict[str, Any]]:
    """OCR через paddleocr_service. Возвращает raw_ocr_boxes как в проде."""
    url = f"{service_url.rstrip('/')}/ocr"
    try:
        with open(image_path, "rb") as f:
            body = f.read()
    except OSError:
        return []
    try:
        import requests
        resp = requests.post(url, files={"image": (image_path.name, body, "image/png")}, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.debug("OCR service %s: %s", image_path.name, e)
        return []
    out = []
    for i, b in enumerate(data):
        x, y = int(b.get("x", 0)), int(b.get("y", 0))
        w, h = max(1, int(b.get("w", 0))), max(1, int(b.get("h", 0)))
        out.append({
            "id": f"ocr_{i}",
            "text": (b.get("text") or "").strip(),
            "bbox": [x, y, x + w, y + h],
            "confidence": float(b.get("confidence", 0)),
        })
    return out


def _match_det_to_dom(
    det_atom: Dict[str, Any],
    dom_elements: List[Dict[str, Any]],
) -> Optional[Tuple[Dict[str, Any], float]]:
    """
    Для одного det-атома находит лучший DOM по IoU >= IOU_MIN или coverage >= COVERAGE_MIN.
    Возвращает (dom_element, match_score) или None.
    match_score = IoU*0.7 + coverage*0.3 для выбора лучшего среди кандидатов.
    """
    det_bbox = det_atom.get("bbox") or []
    if len(det_bbox) < 4:
        return None
    area_det = _bbox_area(det_bbox)
    if area_det <= 0:
        return None
    candidates: List[Tuple[Dict[str, Any], float, float]] = []
    for dom in dom_elements:
        dom_bbox = dom.get("bbox")
        if not dom_bbox or len(dom_bbox) < 4:
            continue
        iou = _iou(det_bbox, dom_bbox)
        coverage = _coverage(det_bbox, dom_bbox)
        if iou >= IOU_MIN or coverage >= COVERAGE_MIN:
            score = iou * MATCH_SCORE_IOU_WEIGHT + coverage * MATCH_SCORE_COVERAGE_WEIGHT
            candidates.append((dom, score, iou))
    if not candidates:
        return None
    best = max(candidates, key=lambda x: x[1])
    return (best[0], best[1])


def _image_size(image_path: Path) -> Optional[Tuple[int, int]]:
    try:
        from PIL import Image
        with Image.open(image_path) as im:
            return (im.width, im.height)
    except Exception:
        return None


def _build_regions(image_width: int, image_height: int) -> List[Dict[str, Any]]:
    """Один регион на всю картинку (как в teacher v1 при отсутствии CV regions)."""
    return [{
        "id": "full",
        "bbox": [0.0, 0.0, float(image_width), float(image_height)],
        "shape_type": "rect",
    }]


def _row_from_atom(
    atom: Dict[str, Any],
    features: Dict[str, float],
    label: str,
    dom_id: str,
    image_id: str,
) -> Dict[str, str]:
    """Одна строка CSV: label, label_quality, признаки, image_id, atom_id, dom_id, source_stage. bbox не модифицируем."""
    bbox = atom.get("bbox") or []
    bw = (bbox[2] - bbox[0]) if len(bbox) >= 4 else 0.0
    bh = (bbox[3] - bbox[1]) if len(bbox) >= 4 else 0.0
    return {
        "label": label,
        "label_quality": LABEL_QUALITY_TEACHER,
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
        "saved_by_anchor": "0",
        "semantic_lock": "0",
        "is_inside_region": "1" if features.get("is_inside_region", 0) >= 0.5 else "0",
        "image_id": image_id,
        "atom_id": str(atom.get("id", "")),
        "dom_id": dom_id,
        "source_stage": SOURCE_STAGE_V2,
    }


def _existing_keys(csv_path: Path) -> Set[Tuple[str, str]]:
    """Возвращает множество (image_id, atom_id) из существующего CSV для дедупликации при append."""
    if not csv_path.exists():
        return set()
    keys: Set[Tuple[str, str]] = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "image_id" not in (reader.fieldnames or []):
            return keys
        for row in reader:
            iid = (row.get("image_id") or "").strip()
            aid = (row.get("atom_id") or "").strip()
            if iid and aid:
                keys.add((iid, aid))
    return keys


def run(
    screenshots_dir: Path,
    output_path: Path,
    with_ocr: bool = False,
    ocr_service_url: Optional[str] = None,
    append: bool = True,
) -> Dict[str, int]:
    """
    Сканирует screenshots_dir на пары .det2.json + .elements.json.
    Для каждого image_id: строит атомы из det2, build_ui_graph + extract_features,
    матчит det → dom, пишет строки только для сматченных атомов.
    Dedup по (image_id, atom_id): при append не дописывает дубликаты.
    Возвращает: total_atoms, matched, skipped_no_match, skipped_dup, images_processed, images_skipped.
    """
    from src.infrastructure.ui_graph.build import build_ui_graph
    from src.infrastructure.ui_graph.features import extract_features

    pairs = _collect_image_ids(screenshots_dir)
    if not pairs:
        logger.warning("No .det2.json + .elements.json pairs in %s", screenshots_dir)
        return {
            "total_atoms": 0, "matched": 0, "skipped_no_match": 0, "skipped_dup": 0,
            "images_processed": 0, "images_skipped": 0,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_keys: Set[Tuple[str, str]] = _existing_keys(output_path) if append else set()
    file_exists = output_path.exists()

    total_atoms = 0
    matched = 0
    skipped_no_match = 0
    skipped_dup = 0
    images_processed = 0
    images_skipped = 0

    for det2_path, elements_path, image_path, ocr_path in pairs:
        image_id = det2_path.stem.removesuffix(".det2")
        size = _image_size(image_path)
        if not size:
            logger.debug("Skip %s: could not read image size", image_id)
            images_skipped += 1
            continue

        det_atoms = _load_det2(det2_path)
        dom_elements = _load_elements(elements_path)
        if not det_atoms:
            logger.debug("Skip %s: no det atoms", image_id)
            images_skipped += 1
            continue
        if not dom_elements:
            logger.debug("Skip %s: no DOM elements", image_id)
            images_skipped += 1
            continue

        raw_ocr: List[Dict[str, Any]] = []
        if ocr_path is not None:
            raw_ocr = _load_ocr_json(ocr_path)
        elif with_ocr and ocr_service_url:
            raw_ocr = _run_ocr_service(image_path, ocr_service_url)

        img_w, img_h = size
        regions = _build_regions(img_w, img_h)
        graph = build_ui_graph(det_atoms, raw_ocr, regions)
        features_by_atom = extract_features(graph)

        rows: List[Dict[str, str]] = []
        for atom in det_atoms:
            total_atoms += 1
            match = _match_det_to_dom(atom, dom_elements)
            if match is None:
                skipped_no_match += 1
                continue
            dom_elem, _ = match
            label = (dom_elem.get("type") or "").strip()
            if not label:
                skipped_no_match += 1
                continue
            dom_id = str(dom_elem.get("id", ""))
            key = (image_id, str(atom.get("id", "")))
            if key in existing_keys:
                skipped_dup += 1
                continue
            existing_keys.add(key)
            feats = features_by_atom.get(atom.get("id", ""), {})
            row = _row_from_atom(atom, feats, label, dom_id, image_id)
            rows.append(row)
            matched += 1

        if rows:
            with open(output_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
                if not file_exists:
                    writer.writeheader()
                    file_exists = True
                writer.writerows(rows)
            images_processed += 1

    return {
        "total_atoms": total_atoms,
        "matched": matched,
        "skipped_no_match": skipped_no_match,
        "skipped_dup": skipped_dup,
        "images_processed": images_processed,
        "images_skipped": images_skipped,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build teacher v2 CSV: Detectron2 bbox + DOM labels, same features as prod.",
    )
    parser.add_argument(
        "--screenshots-dir",
        type=Path,
        default=Path("datasets/ui_screenshots_catboost"),
        help="Directory containing *.det2.json, *.elements.json, *.png (and optionally *.ocr.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT),
        help="Output CSV path (append if exists, dedup by image_id+atom_id)",
    )
    parser.add_argument(
        "--with-ocr",
        action="store_true",
        help="Run OCR via service when .ocr.json not present",
    )
    parser.add_argument(
        "--ocr-service-url",
        type=str,
        default=os.environ.get("PADDLE_OCR_SERVICE_URL", "").strip(),
        help="PaddleOCR service URL (e.g. http://localhost:8001)",
    )
    parser.add_argument(
        "--no-append",
        action="store_true",
        help="Overwrite output instead of appending (dedup still by image_id+atom_id within run)",
    )
    args = parser.parse_args()

    args.screenshots_dir = args.screenshots_dir.resolve()
    args.output = args.output.resolve()
    if args.no_append and args.output.exists():
        args.output.write_text("")  # truncate

    ocr_url = args.ocr_service_url if args.ocr_service_url else None
    if args.with_ocr and not ocr_url:
        logger.info("OCR: --with-ocr set but no ocr-service-url; use .ocr.json files or set --ocr-service-url")

    stats = run(
        args.screenshots_dir,
        args.output,
        with_ocr=args.with_ocr,
        ocr_service_url=ocr_url,
        append=not args.no_append,
    )

    logger.info(
        "teacher_dataset_builder_v2: total_atoms=%s matched=%s skipped_no_match=%s skipped_dup=%s "
        "images_processed=%s images_skipped=%s output=%s",
        stats["total_atoms"],
        stats["matched"],
        stats["skipped_no_match"],
        stats["skipped_dup"],
        stats["images_processed"],
        stats["images_skipped"],
        args.output,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
