#!/usr/bin/env python3
"""
Сравнение вывода пайплайна (OCR/CV bbox из лога) с ground truth.

Читает NDJSON-лог (DEBUG_BBOX_LOG), для каждого изображения загружает
соответствующий ground truth JSON и выводит отчёт: совпадения по bbox,
расхождения по типу, пропущенные и лишние элементы.

Использование (внутри контейнера или на хосте):
  python scripts/compare_ground_truth.py /app/debug/debug.log /app/data/demo_forms/ground_truth

Или после прогона пайплайна с DEBUG_BBOX_LOG=/app/debug/debug.log:
  python scripts/compare_ground_truth.py
  (по умолчанию: log_path=debug/debug.log, ground_truth_dir=data/demo_forms/ground_truth)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple


# Ожидаемое соответствие GT role -> pipeline element_type (CV)
GT_ROLE_TO_EXPECTED_TYPE = {
    "text": "input",       # input type="text"
    "email": "input",
    "password": "input",
    "number": "input",
    "textarea": "textarea",
    "button": "action",
    "submit": "action",
    "select": "select",
    "select-one": "select",
    "label": "label",
    "title": "label",           # заголовок формы
    "section_title": "label",
    "subsection_title": "label",
}


def _iou(bbox1: List[float], bbox2: List[float]) -> float:
    if len(bbox1) < 4 or len(bbox2) < 4:
        return 0.0
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    a2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    return inter / max(1e-9, a1 + a2 - inter)


def _center(bbox: List[float]) -> Tuple[float, float]:
    if len(bbox) < 4:
        return 0.0, 0.0
    return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2


def _load_log_by_image(log_path: str) -> Dict[str, Dict[str, Any]]:
    """Читает NDJSON лог, группирует по image_path. Для каждого образа берётся последняя запись cv."""
    by_image: Dict[str, Dict[str, Any]] = {}
    if not os.path.isfile(log_path):
        return by_image
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            src = rec.get("source")
            img = rec.get("image_path", "")
            if not img or src != "cv":
                continue
            # нормализуем ключ до имени файла без пути
            base = os.path.basename(img)
            stem = os.path.splitext(base)[0]
            by_image[stem] = {
                "image_path": img,
                "bboxes": rec.get("data", {}).get("bboxes", []),
            }
    return by_image


def _load_ground_truth(gt_dir: str, stem: str) -> Dict[str, Any] | None:
    path = os.path.join(gt_dir, stem + ".json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _gt_element_expected_type(el: Dict[str, Any]) -> str:
    role = (el.get("role") or "").lower()
    if role == "text" and el.get("tag") == "input":
        return "input"
    return GT_ROLE_TO_EXPECTED_TYPE.get(role, "unknown")


def _expand_bbox(bbox: List[float], margin_ratio: float = 0.10) -> List[float]:
    """Расширить bbox на ±margin_ratio по каждой стороне (для допуска при сравнении)."""
    if len(bbox) < 4:
        return list(bbox)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    mw = w * margin_ratio
    mh = h * margin_ratio
    return [
        bbox[0] - mw, bbox[1] - mh,
        bbox[2] + mw, bbox[3] + mh,
    ]


def compare(
    log_path: str,
    ground_truth_dir: str,
    iou_threshold: float = 0.2,
    bbox_tolerance_ratio: float = 0.10,
) -> None:
    by_image = _load_log_by_image(log_path)
    if not by_image:
        print("Лог пуст или не найден: %s" % log_path, file=sys.stderr)
        print("Запустите пайплайн с DEBUG_BBOX_LOG=%s" % log_path, file=sys.stderr)
        return

    all_type_ok = 0
    all_type_mismatch = 0
    all_gt_unmatched = 0
    all_cv_extra = 0

    for stem in sorted(by_image.keys()):
        gt = _load_ground_truth(ground_truth_dir, stem)
        if not gt:
            print("[%s] ground truth не найден, пропуск" % stem)
            continue

        cv_data = by_image[stem]
        cv_bboxes = cv_data.get("bboxes") or []
        gt_elements = gt.get("elements") or []

        # Ожидаемые типы для GT (title/label -> label, input type=text -> input, etc.)
        gt_with_type = []
        for el in gt_elements:
            exp = _gt_element_expected_type(el)
            gt_with_type.append({**el, "_expected_type": exp})

        used_cv = [False] * len(cv_bboxes)
        matches = []
        gt_unmatched = []

        for gt_el in gt_with_type:
            gb = gt_el.get("bbox")
            if not gb or len(gb) < 4:
                continue
            gb_expanded = _expand_bbox(gb, bbox_tolerance_ratio)
            best_j = -1
            best_iou = 0.0
            for j, cv_el in enumerate(cv_bboxes):
                if used_cv[j]:
                    continue
                cb = cv_el.get("bbox")
                if not cb or len(cb) < 4:
                    continue
                iou = _iou(gb_expanded, cb)
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_j = j
            if best_j >= 0:
                used_cv[best_j] = True
                cv_el = cv_bboxes[best_j]
                exp = gt_el["_expected_type"]
                actual = (cv_el.get("element_type") or "").lower()
                type_ok = exp == actual or (exp == "action" and actual == "action")
                matches.append({
                    "gt": gt_el,
                    "cv": cv_el,
                    "iou": best_iou,
                    "expected": exp,
                    "actual": actual,
                    "type_ok": type_ok,
                })
                if type_ok:
                    all_type_ok += 1
                else:
                    all_type_mismatch += 1
            else:
                gt_unmatched.append(gt_el)
                all_gt_unmatched += 1

        cv_extra = [cv_bboxes[j] for j in range(len(cv_bboxes)) if not used_cv[j]]
        all_cv_extra += len(cv_extra)

        # Краткий отчёт по изображению
        print("\n=== %s ===" % stem)
        print("  GT элементов: %d, CV элементов: %d" % (len(gt_with_type), len(cv_bboxes)))
        print("  Совпадений по bbox (IoU>=%.2f): %d" % (iou_threshold, len(matches)))
        type_mismatches = [m for m in matches if not m["type_ok"]]
        if type_mismatches:
            print("  Ошибки типа (%d):" % len(type_mismatches))
            for m in type_mismatches[:15]:
                gt_role = m["gt"].get("role") or m["gt"].get("tag") or "?"
                text = (m["gt"].get("text") or m["gt"].get("placeholder") or "")[:40]
                print("    GT %s %s -> ожидался %s, получен %s (IoU=%.2f)" % (
                    gt_role, text, m["expected"], m["actual"], m["iou"]))
            if len(type_mismatches) > 15:
                print("    ... и ещё %d" % (len(type_mismatches) - 15))
        if gt_unmatched:
            print("  Пропущено GT (%d):" % len(gt_unmatched))
            for u in gt_unmatched[:10]:
                print("    %s bbox=%s" % (u.get("role"), u.get("bbox")))
        if cv_extra:
            print("  Лишних CV (%d):" % len(cv_extra))
            for e in cv_extra[:10]:
                print("    %s bbox=%s" % (e.get("element_type"), e.get("bbox")))

    print("\n--- Итого ---")
    print("  Тип совпал: %d" % all_type_ok)
    print("  Тип не совпал: %d" % all_type_mismatch)
    print("  Пропущено GT: %d" % all_gt_unmatched)
    print("  Лишних CV: %d" % all_cv_extra)


def main() -> None:
    ap = argparse.ArgumentParser(description="Сравнение лога пайплайна с ground truth")
    ap.add_argument("log_path", nargs="?", default="debug/debug.log", help="Путь к NDJSON-логу (cv bbox)")
    ap.add_argument("ground_truth_dir", nargs="?", default="data/demo_forms/ground_truth",
                    help="Каталог с demo_form_XX.json")
    ap.add_argument("--iou", type=float, default=0.2, help="Порог IoU для сопоставления bbox (default 0.2)")
    ap.add_argument("--tolerance", type=float, default=0.10, help="Допуск по границам ± (0.10 = ±10%%, default)")
    args = ap.parse_args()
    compare(args.log_path, args.ground_truth_dir, args.iou, args.tolerance)


if __name__ == "__main__":
    main()
