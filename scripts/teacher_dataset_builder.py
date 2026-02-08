#!/usr/bin/env python3
"""
Сбор teacher-датасета для CatBoost из скриншотов генератора и DOM-координат Playwright.

Читает все *.elements.json (координаты элементов из DOM) и соответствующие скриншоты.
Для каждого элемента: извлекает признаки через extract_features (как features_by_atom),
label берётся из DOM (data-ui-type). Пишет CSV для CatBoost с image_id и atom_id.

OCR: при --with-ocr использует paddleocr_service (--ocr-service-url или PADDLE_OCR_SERVICE_URL),
чтобы не ставить PaddleOCR локально.

Запуск:
  python scripts/teacher_dataset_builder.py --screenshots-dir datasets/ui_screenshots_catboost --output datasets/ui_atoms_teacher_catboost.csv --with-ocr --ocr-service-url http://localhost:8001
  export PADDLE_OCR_SERVICE_URL=http://localhost:8001
  python scripts/teacher_dataset_builder.py --screenshots-dir datasets/ui_screenshots_catboost --output datasets/ui_atoms_teacher_catboost.csv --with-ocr
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Добавляем корень проекта в path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

DEFAULT_OUTPUT = "datasets/ui_atoms_teacher_catboost.csv"
LABEL_QUALITY_TEACHER = "teacher"

# Заголовок CSV (для дедупликации и записи)
CSV_HEADER = [
    "label", "label_quality",
    "aspect_ratio", "area", "bbox_width", "bbox_height", "bbox_coverage_ocr",
    "relative_size_to_region", "num_adjacent", "num_aligned_row", "num_aligned_col",
    "row_group_size", "column_group_size", "uniform_spacing_score", "region_density",
    "has_label", "has_action_word", "saved_by_anchor", "semantic_lock", "is_inside_region",
    "image_id", "atom_id", "source_stage",
]


def _signature_for_dedup(row: Dict[str, str]) -> str:
    """
    Подпись для дедупликации: label + округлённые/дискретизированные ключевые признаки.
    Исключаем image_id, atom_id, bbox_coverage_ocr (сильно зависит от OCR/картинки).
    Одинаковые по смыслу элементы (один тип, одна геометрия, один контекст) дают один ключ.
    """
    def _float(s: str, default: float = 0.0) -> float:
        try:
            return float(s)
        except (ValueError, TypeError):
            return default

    def _int(s: str, default: int = 0) -> int:
        try:
            return int(float(s))
        except (ValueError, TypeError):
            return default

    label = (row.get("label") or "").strip()
    ar = round(_float(row.get("aspect_ratio")), 2)
    area = _int(row.get("area"))
    w = _int(row.get("bbox_width"))
    h = _int(row.get("bbox_height"))
    n_adj = _int(row.get("num_adjacent"))
    n_row = _int(row.get("num_aligned_row"))
    n_col = _int(row.get("num_aligned_col"))
    row_grp = _int(row.get("row_group_size"))
    col_grp = _int(row.get("column_group_size"))
    has_lbl = (row.get("has_label") or "0").strip()
    has_act = (row.get("has_action_word") or "0").strip()
    rel_size = round(_float(row.get("relative_size_to_region")), 3)
    unif = round(_float(row.get("uniform_spacing_score")), 2)
    region_d = round(_float(row.get("region_density")), 2)
    return (
        f"{label}|{ar}|{area}|{w}|{h}|{n_adj}|{n_row}|{n_col}|{row_grp}|{col_grp}|"
        f"{has_lbl}|{has_act}|{rel_size}|{unif}|{region_d}"
    )


def _collect_elements_files(screenshots_dir: Path) -> List[Path]:
    """Собирает все *.elements.json в директории и поддиректориях."""
    out: List[Path] = []
    if not screenshots_dir.is_dir():
        return out
    for p in screenshots_dir.rglob("*.elements.json"):
        if p.is_file():
            out.append(p)
    return sorted(out)


def _load_elements(path: Path) -> List[Dict[str, Any]]:
    """Загружает элементы из JSON. Ожидает список { id, type, bbox: [x1,y1,x2,y2] }."""
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "elements" in data:
        return data["elements"]
    return []


def _image_size(image_path: Path) -> Optional[tuple]:
    """Возвращает (width, height) изображения или None."""
    try:
        from PIL import Image
        with Image.open(image_path) as im:
            return (im.width, im.height)
    except Exception as e:
        logger.debug("image size %s: %s", image_path, e)
        return None


def _run_ocr_via_service(image_path: Path, service_url: str, timeout: int = 60) -> List[Dict[str, Any]]:
    """
    OCR через paddleocr_service: POST изображения на /ocr, возвращает raw_ocr_boxes.
    Сервис возвращает [{x, y, w, h, text, confidence}]; конвертируем в {id, text, bbox, confidence}.
    """
    url = f"{service_url.rstrip('/')}/ocr"
    try:
        with open(image_path, "rb") as f:
            body = f.read()
    except OSError as e:
        logger.debug("OCR service: read image %s: %s", image_path, e)
        return []
    try:
        try:
            import requests
            resp = requests.post(
                url,
                files={"image": (image_path.name, body, "image/png")},
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except ImportError:
            import urllib.request
            boundary = "----FormBoundary"
            payload = (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="image"; filename="{image_path.name}"\r\n'
                "Content-Type: image/png\r\n\r\n"
            ).encode("utf-8") + body + f"\r\n--{boundary}--\r\n".encode("utf-8")
            req = urllib.request.Request(url, data=payload, method="POST")
            req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
        out = []
        for i, b in enumerate(data):
            x = int(b.get("x", 0))
            y = int(b.get("y", 0))
            w = max(1, int(b.get("w", 0)))
            h = max(1, int(b.get("h", 0)))
            out.append({
                "id": f"ocr_{i}",
                "text": (b.get("text") or "").strip(),
                "bbox": [x, y, x + w, y + h],
                "confidence": float(b.get("confidence", 0)),
            })
        return out
    except Exception as e:
        logger.warning("OCR service %s failed for %s: %s", url, image_path.name, e)
        return []


def _run_ocr_local(image_path: Path) -> List[Dict[str, Any]]:
    """Локальный OCR (run_text_detect + run_ocr_boxes). Требует PaddleOCR/Тesseract."""
    try:
        from src.infrastructure.debug.services import run_text_detect, run_ocr_boxes
        raw_boxes = run_text_detect(str(image_path))
        if not raw_boxes:
            return []
        results = run_ocr_boxes(str(image_path), raw_boxes)
        out = []
        for i, box in enumerate(raw_boxes):
            r = results[i] if i < len(results) else {}
            x, y = int(box.get("x", 0)), int(box.get("y", 0))
            w, h = int(box.get("w", 0)), int(box.get("h", 0))
            out.append({
                "id": f"ocr_{i}",
                "text": (r.get("text") or "").strip(),
                "bbox": [x, y, x + w, y + h],
                "confidence": float(r.get("confidence", 0)),
            })
        return out
    except Exception as e:
        logger.debug("OCR local %s: %s", image_path, e)
        return []


def build_graph_from_elements(
    elements: List[Dict[str, Any]],
    image_width: int,
    image_height: int,
    raw_ocr_boxes: Optional[List[Dict[str, Any]]] = None,
) -> Any:
    """Строит UIGraph из элементов, одного региона (вся картинка) и опционально OCR."""
    from src.infrastructure.ui_graph.build import build_ui_graph

    atoms = []
    for e in elements:
        eid = e.get("id") or ""
        etype = (e.get("type") or "").strip()
        bbox = e.get("bbox")
        if not eid or not etype or not bbox or len(bbox) < 4:
            continue
        atoms.append({
            "id": eid,
            "type": etype,
            "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
            "confidence": 1.0,
            "source": "teacher",
        })
    regions = [{
        "id": "full",
        "bbox": [0.0, 0.0, float(image_width), float(image_height)],
        "shape_type": "rect",
    }]
    ocr = raw_ocr_boxes or []
    return build_ui_graph(atoms, ocr, regions)


def _row_from_atom(
    atom: Dict[str, Any],
    features: Dict[str, float],
    label: str,
    label_quality: str,
    image_id: str,
) -> Dict[str, str]:
    """Одна строка CSV (совместимо с dataset_builder)."""
    bbox = atom.get("bbox") or []
    bw = (bbox[2] - bbox[0]) if len(bbox) >= 4 else 0.0
    bh = (bbox[3] - bbox[1]) if len(bbox) >= 4 else 0.0
    return {
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
        "saved_by_anchor": "0",
        "semantic_lock": "0",
        "is_inside_region": "1" if features.get("is_inside_region", 0) >= 0.5 else "0",
        "image_id": image_id,
        "atom_id": str(atom.get("id", "")),
        "source_stage": label_quality,
    }


def run(
    screenshots_dir: Path,
    output_path: Path,
    with_ocr: bool = False,
    ocr_service_url: Optional[str] = None,
    dedup: bool = False,
) -> Dict[str, int]:
    """
    Сканирует screenshots_dir на *.elements.json, для каждого строит граф и признаки,
    пишет строки в CSV. При dedup=True оставляет по одной строке на подпись (label + ключевые признаки).
    Возвращает stats: processed, skipped, images_processed, images_skipped, dedup_dropped.
    """
    from src.infrastructure.ui_graph.features import extract_features

    elements_files = _collect_elements_files(screenshots_dir)
    if not elements_files:
        logger.warning("No *.elements.json found in %s", screenshots_dir)
        return {"processed": 0, "skipped": 0, "images_processed": 0, "images_skipped": 0, "dedup_dropped": 0}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists()
    processed = 0
    skipped = 0
    images_processed = 0
    images_skipped = 0
    dedup_dropped = 0
    seen_signatures: set = set()

    import csv
    for el_path in elements_files:
        image_id = el_path.stem.removesuffix(".elements") if el_path.stem.endswith(".elements") else el_path.stem
        dir_path = el_path.parent
        image_path = dir_path / f"{image_id}.png"
        if not image_path.exists():
            logger.debug("Skip %s: no image %s", el_path, image_path.name)
            images_skipped += 1
            continue

        size = _image_size(image_path)
        if not size:
            logger.debug("Skip %s: could not read image size", el_path)
            images_skipped += 1
            continue
        img_w, img_h = size

        elements = _load_elements(el_path)
        if not elements:
            logger.debug("Skip %s: no elements", el_path)
            images_skipped += 1
            continue

        raw_ocr: List[Dict[str, Any]] = []
        if with_ocr:
            if ocr_service_url:
                raw_ocr = _run_ocr_via_service(image_path, ocr_service_url)
            else:
                raw_ocr = _run_ocr_local(image_path)
        graph = build_graph_from_elements(elements, img_w, img_h, raw_ocr_boxes=raw_ocr)
        features_by_atom = extract_features(graph)

        rows: List[Dict[str, str]] = []
        for e in elements:
            eid = e.get("id") or ""
            label = (e.get("type") or "").strip()
            if not label:
                skipped += 1
                continue
            feats = features_by_atom.get(eid, {})
            row = _row_from_atom(e, feats, label, LABEL_QUALITY_TEACHER, image_id)
            if dedup:
                sig = _signature_for_dedup(row)
                if sig in seen_signatures:
                    dedup_dropped += 1
                    processed += 1
                    continue
                seen_signatures.add(sig)
            rows.append(row)
            processed += 1

        if rows:
            with open(output_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
                if not file_exists:
                    writer.writeheader()
                    file_exists = True
                writer.writerows(rows)
            images_processed += 1

    return {
        "processed": processed,
        "skipped": skipped,
        "images_processed": images_processed,
        "images_skipped": images_skipped,
        "dedup_dropped": dedup_dropped,
    }


def run_dedup_only(input_path: Path, output_path: Path) -> Dict[str, int]:
    """
    Читает CSV teacher-датасета, дедуплицирует по подписи (label + ключевые признаки),
    пишет результат в output_path. Возвращает stats: total_read, written, dedup_dropped.
    """
    import csv
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    if not input_path.exists():
        logger.warning("Input file not found: %s", input_path)
        return {"total_read": 0, "written": 0, "dedup_dropped": 0}

    seen: set = set()
    written = 0
    dedup_dropped = 0
    rows_out: List[Dict[str, str]] = []

    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or list(CSV_HEADER)
        total_read = 0
        for row in reader:
            total_read += 1
            sig = _signature_for_dedup(row)
            if sig in seen:
                dedup_dropped += 1
                continue
            seen.add(sig)
            rows_out.append(row)
            written += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    return {"total_read": total_read, "written": written, "dedup_dropped": dedup_dropped}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build teacher CSV for CatBoost from Playwright elements JSON and screenshots.",
    )
    parser.add_argument(
        "--screenshots-dir",
        type=Path,
        default=Path("datasets/ui_screenshots_catboost"),
        help="Directory (or parent) containing *.elements.json and *.png",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT),
        help="Output CSV path (append if exists)",
    )
    parser.add_argument(
        "--with-ocr",
        action="store_true",
        help="Run OCR on each image for bbox_coverage_ocr, has_label, etc.",
    )
    parser.add_argument(
        "--ocr-service-url",
        type=str,
        default=os.environ.get("PADDLE_OCR_SERVICE_URL", "").strip(),
        help="PaddleOCR service URL (e.g. http://localhost:8001). Default: PADDLE_OCR_SERVICE_URL env.",
    )
    parser.add_argument(
        "--dedup",
        action="store_true",
        help="Дедупликация по подписи (label + ключевые признаки); оставлять по одной строке на тип элемента.",
    )
    parser.add_argument(
        "--dedup-only",
        type=str,
        metavar="INPUT_CSV",
        help="Только дедупликация: прочитать INPUT_CSV, дедуплицировать, записать в --output (или INPUT_CSV с суффиксом _dedup).",
    )
    args = parser.parse_args()

    if args.dedup_only:
        input_csv = Path(args.dedup_only).resolve()
        output_csv = input_csv.parent / (input_csv.stem + "_dedup.csv")
        stats = run_dedup_only(input_csv, output_csv)
        logger.info(
            "teacher_dataset_builder (dedup-only): total_read=%s written=%s dedup_dropped=%s output=%s",
            stats["total_read"],
            stats["written"],
            stats["dedup_dropped"],
            output_csv,
        )
        return 0

    args.screenshots_dir = args.screenshots_dir.resolve()
    args.output = args.output.resolve()
    ocr_url = args.ocr_service_url if args.ocr_service_url else None
    if args.with_ocr and not ocr_url:
        logger.info("OCR: using local (set --ocr-service-url or PADDLE_OCR_SERVICE_URL to use paddleocr_service)")
    elif args.with_ocr and ocr_url:
        logger.info("OCR: using paddleocr_service at %s", ocr_url)
    if args.dedup:
        logger.info("Dedup: enabled (by label + key features)")

    stats = run(
        args.screenshots_dir,
        args.output,
        with_ocr=args.with_ocr,
        ocr_service_url=ocr_url,
        dedup=args.dedup,
    )

    logger.info(
        "teacher_dataset_builder: processed=%s skipped=%s images_processed=%s images_skipped=%s dedup_dropped=%s output=%s",
        stats["processed"],
        stats["skipped"],
        stats["images_processed"],
        stats["images_skipped"],
        stats.get("dedup_dropped", 0),
        args.output,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
