#!/usr/bin/env python3
"""
Сборка COCO-датасета для Detectron2 из JSON-файлов Playwright (bbox + type).

Читает JSON-файлы с координатами UI-элементов, копирует скриншоты в dataset/train/images
и dataset/val/images, объединяет аннотации в train_coco.json и val_coco.json,
создаёт dataset.yaml с именами категорий.

Использование:
  --train-dir DIR   директория с JSON (и PNG) для train
  --val-dir DIR     директория с JSON (и PNG) для val
  либо --input-dir DIR --val-ratio 0.2  (один каталог, разбиение по ratio)
  -o DIR            корень датасета (по умолчанию dataset/)
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

COCO_KEYS = ("info", "licenses", "images", "annotations", "categories")


def collect_categories(json_paths: list[Path]) -> list[str]:
    """Собирает уникальные type из всех elements, сортирует по алфавиту."""
    types: set[str] = set()
    for p in json_paths:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Skip %s: %s", p, e)
            continue
        for el in data.get("elements", []):
            t = el.get("type")
            if t:
                types.add(t)
    return sorted(types)


def build_category_id_map(categories: list[str]) -> dict[str, int]:
    """type -> category_id (0, 1, 2, ...)."""
    return {name: i for i, name in enumerate(categories)}


def process_split(
    json_paths: list[Path],
    category_to_id: dict[str, int],
    out_images_dir: Path,
    out_coco_path: Path,
    split_name: str,
) -> tuple[int, int]:
    """
    Обрабатывает список JSON-файлов: копирует изображения, собирает COCO.
    Возвращает (num_images, num_annotations).
    """
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    image_id = 1
    annotation_id = 1

    out_images_dir.mkdir(parents=True, exist_ok=True)

    for jpath in json_paths:
        try:
            data = json.loads(jpath.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Skip %s: %s", jpath, e)
            continue

        image_filename = data.get("image")
        if not image_filename:
            logger.warning("No 'image' in %s", jpath)
            continue

        src_image = jpath.parent / image_filename
        if not src_image.exists():
            logger.warning("Image not found: %s", src_image)
            continue

        viewport = data.get("viewport", {})
        width = int(viewport.get("width", 1440))
        height = int(viewport.get("height", 900))

        out_filename = f"{image_id:06d}.png"
        dst_image = out_images_dir / out_filename
        try:
            shutil.copy2(src_image, dst_image)
        except Exception as e:
            logger.warning("Copy failed %s -> %s: %s", src_image, dst_image, e)
            continue

        images.append({
            "id": image_id,
            "file_name": out_filename,
            "width": width,
            "height": height,
        })

        for el in data.get("elements", []):
            bbox = el.get("bbox")
            type_name = el.get("type")
            if not bbox or len(bbox) < 4 or type_name not in category_to_id:
                continue
            x_min, y_min, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            if w <= 0 or h <= 0:
                continue
            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_to_id[type_name],
                "bbox": [x_min, y_min, w, h],
                "area": w * h,
                "iscrowd": 0,
            })
            annotation_id += 1

        image_id += 1

    names_sorted = sorted(category_to_id, key=category_to_id.get)
    coco = {
        "info": {"description": f"UI elements {split_name} set"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": category_to_id[n], "name": n} for n in names_sorted],
    }

    out_coco_path.parent.mkdir(parents=True, exist_ok=True)
    out_coco_path.write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")

    return len(images), len(annotations)


def main(
    train_dir: Path | None = None,
    val_dir: Path | None = None,
    input_dir: Path | None = None,
    val_ratio: float = 0.2,
    output_root: Path = Path("dataset"),
) -> int:
    if train_dir is not None and val_dir is not None:
        train_jsons = list(Path(train_dir).resolve().glob("*.json"))
        val_jsons = list(Path(val_dir).resolve().glob("*.json"))
    elif input_dir is not None:
        all_jsons = list(Path(input_dir).resolve().glob("*.json"))
        random.Random(42).shuffle(all_jsons)
        n_val = max(1, int(len(all_jsons) * val_ratio))
        val_jsons = all_jsons[:n_val]
        train_jsons = all_jsons[n_val:]
    else:
        logger.error("Specify either (--train-dir and --val-dir) or --input-dir")
        return 1

    if not train_jsons and not val_jsons:
        logger.error("No JSON files found")
        return 1

    all_jsons = train_jsons + val_jsons
    categories = collect_categories(all_jsons)
    if not categories:
        logger.error("No categories (types) found in JSON files")
        return 1

    category_to_id = build_category_id_map(categories)
    out = Path(output_root).resolve()
    train_images_dir = out / "train" / "images"
    val_images_dir = out / "val" / "images"
    train_coco = out / "train" / "train_coco.json"
    val_coco = out / "val" / "val_coco.json"

    n_train_im, n_train_ann = 0, 0
    if train_jsons:
        n_train_im, n_train_ann = process_split(
            train_jsons,
            category_to_id,
            train_images_dir,
            train_coco,
            "train",
        )

    n_val_im, n_val_ann = 0, 0
    if val_jsons:
        n_val_im, n_val_ann = process_split(
            val_jsons,
            category_to_id,
            val_images_dir,
            val_coco,
            "val",
        )

    yaml_path = out / "dataset.yaml"
    yaml_content = f"""# COCO dataset for Detectron2 (UI elements)
# Generated by build_coco_dataset.py

names: {categories}

train:
  images: train/images
  annotations: train/train_coco.json

val:
  images: val/images
  annotations: val/val_coco.json
"""
    yaml_path.write_text(yaml_content, encoding="utf-8")

    total_im = n_train_im + n_val_im
    total_ann = n_train_ann + n_val_ann
    print("COCO dataset created for train/val with {} images and {} annotations.".format(total_im, total_ann))
    logger.info("Train: %d images, %d annotations", n_train_im, n_train_ann)
    logger.info("Val: %d images, %d annotations", n_val_im, n_val_ann)
    logger.info("Categories: %s", categories)
    logger.info("Output: %s", out)

    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Build COCO dataset from Playwright JSON (bbox + type) for Detectron2"
    )
    ap.add_argument("--train-dir", type=Path, help="Directory with train JSON (and images)")
    ap.add_argument("--val-dir", type=Path, help="Directory with val JSON (and images)")
    ap.add_argument("--input-dir", type=Path, help="Single dir with all JSON; split by --val-ratio")
    ap.add_argument("--val-ratio", type=float, default=0.2, help="Fraction for val when using --input-dir (default: 0.2)")
    ap.add_argument("-o", "--output", type=Path, default=Path("dataset"), help="Dataset root (default: dataset/)")
    args = ap.parse_args()

    if args.train_dir and args.val_dir:
        sys.exit(main(train_dir=args.train_dir, val_dir=args.val_dir, output_root=args.output))
    if args.input_dir:
        sys.exit(main(input_dir=args.input_dir, val_ratio=args.val_ratio, output_root=args.output))
    logger.error("Use either (--train-dir and --val-dir) or --input-dir")
    sys.exit(1)
