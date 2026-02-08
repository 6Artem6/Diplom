#!/usr/bin/env python3
"""
Обучение CatBoost MultiClass для UI-ролей на teacher-датасете v2 (bbox из Detectron2).

Модель предназначена для использования как soft-prior (priors.role_probs) в основном пайплайне,
не заменяет rule-based логику. Не встраивается в пайплайн — только офлайн-обучение.

Вход: CSV от teacher_dataset_builder_v2 (label_quality=teacher, source_stage=teacher_v2).
Классы: button, input, link, textarea, checkbox (без layout, noise, weak_*).
Признаки и препроцессинг — те же, что в проде (catboost_priors).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

# Пути по умолчанию
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = _PROJECT_ROOT / "datasets" / "ui_atoms_teacher_catboost_v2_phase1.csv"
DEFAULT_OUTPUT = _PROJECT_ROOT / "models" / "ui_atom_role_catboost_v2.cbm"

# Фильтр данных: только teacher v2
LABEL_QUALITY_FILTER = "teacher"
SOURCE_STAGE_FILTER = "teacher_v2"
ALLOWED_LABELS = {"button", "input", "link", "textarea", "checkbox"}

# Признаки (совпадают с catboost_priors / prod)
FEATURE_COLUMNS = [
    "aspect_ratio", "area", "bbox_width", "bbox_height", "bbox_coverage_ocr",
    "relative_size_to_region", "num_adjacent", "num_aligned_row", "num_aligned_col",
    "row_group_size", "column_group_size", "uniform_spacing_score", "region_density",
    "has_label", "has_action_word", "saved_by_anchor", "semantic_lock", "is_inside_region",
]
NUMERIC_TO_LOG1P = ["area", "bbox_width", "bbox_height"]
ASPECT_RATIO_CLIP = (0.0, 30.0)
NON_FEATURE_COLUMNS = {"label", "label_quality", "image_id", "atom_id", "dom_id", "source_stage"}


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """log1p(area, bbox_*), clip(aspect_ratio), все признаки к числу, пропуски → 0."""
    X = df.copy()
    for col in NUMERIC_TO_LOG1P:
        if col in X.columns:
            X[col] = np.log1p(pd.to_numeric(X[col], errors="coerce").fillna(0).clip(lower=0))
    if "aspect_ratio" in X.columns:
        X["aspect_ratio"] = np.clip(
            pd.to_numeric(X["aspect_ratio"], errors="coerce").fillna(0),
            ASPECT_RATIO_CLIP[0],
            ASPECT_RATIO_CLIP[1],
        )
    for c in X.columns:
        if X[c].dtype in ("object", "string"):
            continue
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
    return X


def compute_class_weights(y: pd.Series) -> dict[str, float] | None:
    """Веса по частотам: total / (n_classes * count_i) для баланса (важно для input, checkbox, textarea)."""
    counts = y.value_counts()
    n_classes = len(counts)
    total = counts.sum()
    if total <= 0 or n_classes == 0:
        return None
    return {c: total / (n_classes * counts.get(c, 1)) for c in counts.index}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train CatBoost MultiClass for UI roles on teacher v2 dataset (Detectron2 bbox).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to CSV (e.g. datasets/ui_atoms_teacher_catboost_v2_phase1.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to save model (e.g. models/ui_atom_role_catboost_v2.cbm)",
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Disable log1p/clip preprocessing",
    )
    args = parser.parse_args()

    dataset_path = args.dataset.resolve()
    output_path = args.output.resolve()

    if not dataset_path.exists():
        print(f"Error: dataset not found: {dataset_path}", file=sys.stderr)
        return 1

    df = pd.read_csv(dataset_path)

    # Фильтр: только teacher v2 и разрешённые классы
    df = df[
        (df["label_quality"].astype(str).str.strip() == LABEL_QUALITY_FILTER)
        & (df["source_stage"].astype(str).str.strip() == SOURCE_STAGE_FILTER)
        & (df["label"].astype(str).str.strip().str.lower().isin(ALLOWED_LABELS))
    ].copy()

    if df.empty:
        print("Error: no rows after filter (label_quality=teacher, source_stage=teacher_v2, label in button/input/link/textarea/checkbox).", file=sys.stderr)
        return 1

    n_total = len(df)
    print(f"Dataset size after filter: {n_total}")

    # Распределение классов
    label_counts = df["label"].astype(str).str.strip().str.lower().value_counts().sort_index()
    print("Class distribution:")
    for label, count in label_counts.items():
        pct = 100.0 * count / n_total
        print(f"  {label}: {count} ({pct:.1f}%)")

    y = df["label"].astype(str).str.strip().str.lower()
    class_weights = compute_class_weights(y)

    # Признаки: только FEATURE_COLUMNS, присутствующие в CSV; остальные дропаем
    drop_cols = [c for c in df.columns if c in NON_FEATURE_COLUMNS]
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    if not feature_cols:
        print("Error: no feature columns found in CSV.", file=sys.stderr)
        return 1
    X = df.drop(columns=[c for c in df.columns if c in drop_cols])
    X = X.reindex(columns=feature_cols, fill_value=0)

    if not args.no_preprocess:
        X = preprocess_features(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function="MultiClass",
        eval_metric="Accuracy",
        verbose=50,
        random_seed=42,
        class_weights=class_weights,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )

    best_iteration = model.get_best_iteration()
    best_score = model.get_best_score()
    print(f"bestIteration: {best_iteration}")
    if best_score and "validation" in best_score:
        print(f"bestTest (validation) score: {best_score['validation'].get('Accuracy', best_score['validation'])}")
    else:
        print("bestTest: (see logs above)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output_path))
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
