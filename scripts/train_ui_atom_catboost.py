"""
Обучение CatBoost для классификации UI-атомов (button vs input и др.).

Поддерживает:
- datasets/ui_atoms_catboost.csv (semantic/weak из пайплайна),
- datasets/ui_atoms_teacher_catboost.csv (teacher из Playwright).
Препроцессинг: log1p для area/bbox_*, clip aspect_ratio, заполнение пропусков (см. teacher_csv_stats.py).
"""
import argparse
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

DATASET_PATH = "datasets/ui_atoms_catboost.csv"
TEACHER_DATASET_PATH = "datasets/ui_atoms_teacher_catboost.csv"
MODEL_PATH = "models/ui_atom_catboost.cbm"

# Признаки для нормализации (по рекомендациям teacher_csv_stats.py)
NUMERIC_TO_LOG1P = ["area", "bbox_width", "bbox_height"]
ASPECT_RATIO_CLIP = (0.0, 30.0)


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """Нормализация по рекомендациям: log1p(area, bbox_*), clip(aspect_ratio). Заполнение пропусков 0."""
    X = df.copy()
    for col in NUMERIC_TO_LOG1P:
        if col in X.columns:
            X[col] = np.log1p(pd.to_numeric(X[col], errors="coerce").fillna(0))
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


def main():
    parser = argparse.ArgumentParser(description="Train CatBoost for UI atom classification")
    parser.add_argument("--dataset", type=str, default=DATASET_PATH, help="Path to CSV (semantic/weak or teacher)")
    parser.add_argument("--teacher", action="store_true", help="Use teacher CSV and label_quality in (semantic, weak, teacher)")
    parser.add_argument("--no-preprocess", action="store_true", help="Skip log1p/clip preprocessing")
    parser.add_argument("--binary", action="store_true", default=True, help="Binary: button=1, else=0 (default True)")
    args = parser.parse_args()

    path = args.dataset
    df = pd.read_csv(path)

    # Фильтр по label_quality
    if args.teacher:
        df = df[df["label_quality"].isin(["semantic", "weak", "teacher"])]
    else:
        df = df[df["label_quality"] == "semantic"]

    if df.empty:
        raise SystemExit("No rows after filter. Use --teacher for teacher CSV or check label_quality.")

    # Target: бинарный button=1 или многокласс (label)
    if args.binary:
        y = (df["label"] == "button").astype(int)
    else:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(df["label"].astype(str))

    drop_cols = {"label", "label_quality", "image_id", "atom_id", "dom_id", "source_stage"}
    X = df.drop(columns=[c for c in df.columns if c in drop_cols])

    if not args.no_preprocess:
        X = preprocess_features(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        verbose=50,
        random_seed=42,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )

    model.save_model(MODEL_PATH)
    print("Saved:", MODEL_PATH)


if __name__ == "__main__":
    main()
