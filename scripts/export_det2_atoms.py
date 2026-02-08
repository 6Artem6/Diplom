#!/usr/bin/env python3
"""
Экспорт атомов Detectron2 после postprocess в .det2.json для teacher_dataset_builder_v2.

Для каждого изображения в директории запускает тот же код, что продовый пайплайн (Detectron2 + postprocess),
и сохраняет атомы в {image_id}.det2.json. Формат совместим с teacher_dataset_builder_v2 (_load_det2).

Запуск:
  python scripts/export_det2_atoms.py --screenshots-dir datasets/ui_screenshots_catboost/phase1
  python scripts/export_det2_atoms.py --screenshots-dir datasets/ui_screenshots_catboost --limit 10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _collect_images(screenshots_dir: Path, limit: int | None) -> List[Path]:
    """Собирает пути к .png в директории (и поддиректориях)."""
    if not screenshots_dir.is_dir():
        return []
    images: List[Path] = []
    for p in sorted(screenshots_dir.rglob("*.png")):
        if p.is_file():
            images.append(p)
            if limit is not None and len(images) >= limit:
                break
    return images


def _atoms_to_det2_format(atoms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Приводит атомы к формату .det2.json: id, bbox, score, class (teacher_dataset_builder_v2)."""
    out: List[Dict[str, Any]] = []
    for a in atoms:
        aid = a.get("id", "")
        bbox = a.get("bbox")
        if not aid or not bbox or len(bbox) < 4:
            continue
        out.append({
            "id": aid,
            "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
            "score": float(a.get("confidence", 0)),
            "class": a.get("type", "unknown"),
        })
    return out


def run(
    screenshots_dir: Path,
    limit: int | None = None,
    parallel_ocr: bool = True,
    legacy_text_pipeline: bool = True,
) -> Dict[str, int]:
    """
    Для каждого .png в screenshots_dir запускает get_atoms_after_postprocess и пишет {image_id}.det2.json.
    Возвращает: processed, skipped, total_atoms.
    """
    from src.infrastructure.atoms_v2.pipeline import get_atoms_after_postprocess

    images = _collect_images(screenshots_dir, limit)
    if not images:
        logger.warning("No .png found in %s", screenshots_dir)
        return {"processed": 0, "skipped": 0, "total_atoms": 0}

    processed = 0
    skipped = 0
    total_atoms = 0

    for i, image_path in enumerate(images):
        image_id = image_path.stem
        det2_path = image_path.parent / f"{image_id}.det2.json"
        try:
            atoms = get_atoms_after_postprocess(
                str(image_path),
                parallel_ocr=parallel_ocr,
                legacy_text_pipeline=legacy_text_pipeline,
            )
        except Exception as e:
            logger.warning("export_det2: %s failed: %s", image_id, e)
            skipped += 1
            continue
        payload = _atoms_to_det2_format(atoms)
        total_atoms += len(payload)
        try:
            det2_path.write_text(json.dumps(payload, ensure_ascii=False, indent=0), encoding="utf-8")
        except OSError as e:
            logger.warning("export_det2: write %s failed: %s", det2_path, e)
            skipped += 1
            continue
        processed += 1
        if (i + 1) % 20 == 0:
            logger.info("export_det2: %s/%s images", i + 1, len(images))

    return {"processed": processed, "skipped": skipped, "total_atoms": total_atoms}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export Detectron2 atoms (after postprocess) to .det2.json for teacher_dataset_builder_v2.",
    )
    parser.add_argument(
        "--screenshots-dir",
        type=Path,
        default=Path("datasets/ui_screenshots_catboost"),
        help="Directory containing .png (and optionally .elements.json for later teacher v2)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of images to process (default: all)",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable full-page OCR (postprocess may add fewer synthetic atoms)",
    )
    parser.add_argument(
        "--no-legacy-grouping",
        action="store_true",
        help="Disable legacy text grouping (independent_text_blocks)",
    )
    args = parser.parse_args()

    args.screenshots_dir = args.screenshots_dir.resolve()
    stats = run(
        args.screenshots_dir,
        limit=args.limit,
        parallel_ocr=not args.no_ocr,
        legacy_text_pipeline=not args.no_legacy_grouping,
    )
    logger.info(
        "export_det2_atoms: processed=%s skipped=%s total_atoms=%s dir=%s",
        stats["processed"],
        stats["skipped"],
        stats["total_atoms"],
        args.screenshots_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
