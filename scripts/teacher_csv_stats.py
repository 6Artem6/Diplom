#!/usr/bin/env python3
"""
Статистика по teacher-CSV для CatBoost: числовые признаки, label, дубликаты, пропуски.
Рекомендации по нормализации и интеграции в teacher_dataset_builder.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Числовые колонки (из dataset_builder)
FEATURE_NUMERIC = [
    "aspect_ratio", "area", "bbox_width", "bbox_height", "bbox_coverage_ocr",
    "relative_size_to_region", "num_adjacent", "num_aligned_row", "num_aligned_col",
    "row_group_size", "column_group_size", "uniform_spacing_score", "region_density",
]
FEATURE_BOOL = ["has_label", "has_action_word", "saved_by_anchor", "semantic_lock", "is_inside_region"]


def safe_float(s: str):
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def run(csv_path: Path) -> None:
    path = csv_path.resolve()
    if not path.exists():
        print(f"File not found: {path}")
        return

    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for row in reader:
            rows.append(row)

    n = len(rows)
    if n == 0:
        print("CSV is empty.")
        return

    print("=" * 60)
    print("TEACHER CSV STATISTICS")
    print("=" * 60)
    print(f"File: {path}")
    print(f"Total rows: {n}")
    print()

    # 1) Числовые признаки: mean, min, max, std
    print("--- 1) Numeric features (mean, min, max, std) ---")
    all_numeric = FEATURE_NUMERIC + FEATURE_BOOL
    stats = {}
    for col in all_numeric:
        if col not in rows[0]:
            continue
        vals = []
        for r in rows:
            v = safe_float(r.get(col, ""))
            if v is not None:
                vals.append(v)
        if not vals:
            stats[col] = {"mean": None, "min": None, "max": None, "std": None, "count": 0, "missing": n}
            continue
        mean = sum(vals) / len(vals)
        min_v = min(vals)
        max_v = max(vals)
        variance = sum((x - mean) ** 2 for x in vals) / len(vals)
        std = variance ** 0.5
        missing = n - len(vals)
        stats[col] = {"mean": mean, "min": min_v, "max": max_v, "std": std, "count": len(vals), "missing": missing}
        print(f"  {col}: mean={mean:.4f} min={min_v:.4f} max={max_v:.4f} std={std:.4f} (valid={len(vals)} missing={missing})")
    print()

    # 2) Количество по label
    print("--- 2) Count by label ---")
    label_counts = Counter(r.get("label", "").strip() for r in rows)
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        pct = 100.0 * count / n
        print(f"  {label}: {count} ({pct:.1f}%)")
    print()

    # 3) Дубликаты по (image_id, atom_id)
    print("--- 3) Duplicates by (image_id, atom_id) ---")
    key_counts = Counter((r.get("image_id", ""), r.get("atom_id", "")) for r in rows)
    dup_keys = {k: c for k, c in key_counts.items() if c > 1}
    n_dup_pairs = len(dup_keys)
    n_dup_rows = sum(c - 1 for c in dup_keys.values())
    print(f"  Unique (image_id, atom_id): {len(key_counts)}")
    print(f"  Pairs with duplicates: {n_dup_pairs}")
    print(f"  Extra rows (duplicates): {n_dup_rows}")
    if dup_keys:
        for (iid, aid), c in sorted(dup_keys.items(), key=lambda x: -x[1])[:5]:
            print(f"    Example: image_id={iid} atom_id={aid} count={c}")
    print()

    # 4) Пропуски по колонкам
    print("--- 4) Missing values (%) ---")
    any_missing = False
    for col in fieldnames:
        empty = sum(1 for r in rows if (r.get(col) or "").strip() == "")
        pct = 100.0 * empty / n
        if pct > 0:
            print(f"  {col}: {empty} ({pct:.1f}%)")
            any_missing = True
    if not any_missing:
        print("  (no missing values)")
    print()

    # 5) Рекомендации
    print("=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)

    recs = []

    # Масштаб признаков
    for col in ["area", "bbox_width", "bbox_height"]:
        if col in stats and stats[col]["max"] is not None:
            mx = stats[col]["max"]
            if mx > 1000:
                recs.append(
                    f"- {col}: max={mx:.0f} — нормализовать (log1p или scale к [0,1]) в скрипте обучения/препроцессинга."
                )
    if stats.get("aspect_ratio", {}).get("max"):
        ar_max = stats["aspect_ratio"]["max"]
        if ar_max > 20:
            recs.append(
                f"- aspect_ratio: max={ar_max:.2f} — ограничить/клиповать (например до 30) или log1p в препроцессинге."
            )
    if stats.get("bbox_coverage_ocr", {}).get("std") is not None and stats["bbox_coverage_ocr"]["std"] < 0.01:
        recs.append(
            "- bbox_coverage_ocr: малая дисперсия — при отсутствии OCR много нулей; оставить как есть или бинарный признак ( > 0)."
        )

    # Дубликаты
    if n_dup_rows > 0:
        recs.append(
            f"- Дубликаты (image_id, atom_id): {n_dup_rows} лишних строк — при сборе не допускать повторную запись одного и того же (image_id, atom_id); в teacher_dataset_builder при записи проверять ключ или собирать ключи из уже записанного CSV при append."
        )
    else:
        recs.append("- Дубликатов по (image_id, atom_id) нет — OK.")

    # Дисбаланс классов
    if label_counts:
        majority = max(label_counts.values())
        minority = min(label_counts.values())
        if majority > 3 * minority:
            recs.append(
                f"- Дисбаланс классов: max={majority}, min={minority} — при обучении CatBoost использовать scale_pos_weight или class_weights / сэмплирование."
            )

    # Строки
    recs.append(
        "- Строки: не удалять по пропускам (пропусков по ключевым признакам нет); при обучении CatBoost пропуски в числовых можно заменить на 0 или медиану в препроцессинге."
    )
    recs.append(
        "- Нормализацию (log1p, scale) выполнять в скрипте обучения CatBoost (train_ui_atom_catboost.py) перед fit; teacher_dataset_builder оставить сырые значения для прозрачности и воспроизводимости."
    )

    for r in recs:
        print(r)
    print()
    print("Integration in script:")
    print("  1) teacher_dataset_builder.py: при --dedup учитывать уже только уникальные (image_id, atom_id) при append, если нужно избежать дубликатов при повторном запуске на тех же скриншотах.")
    print("  2) train_ui_atom_catboost.py: загрузить CSV → заменить пустые числа на 0 или медиану → применить log1p(area), log1p(bbox_width), log1p(bbox_height), clip(aspect_ratio, 0, 30) или scale → обучить CatBoost.")
    print()


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "datasets" / "ui_atoms_teacher_catboost.csv"
    run(path)
