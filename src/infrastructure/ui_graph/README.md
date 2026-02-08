# UI-граф

Структурный слой между CV (atoms, regions, OCR) и финальной семантикой. Не создаёт bbox, не меняет CV, не заменяет модели.

## Архитектура

- **Узлы:** AtomNode (id, type, bbox, confidence, source), OCRNode (id, text, bbox), RegionNode (id, bbox, shape_type).
- **Рёбра:** CONTAINS (region→atom, atom→OCR), ADJACENT (atom↔atom), ALIGNED_ROW, ALIGNED_COL, LABELED_BY (atom↔OCR), PART_OF (atom→region).

Правила построения рёбер — в `edges.py`. Признаки — в `features.py`. Роли — в `roles.py` и `classifier.py`.

## Роли (ui_role ≠ atom.type)

- **atom.type** — CV-гипотеза (Detectron2).
- **ui_role** ∈ {button, input, link, control_group, pagination, form, noise} — итоговая семантика.

Классификатор: rule-based (weak supervision) или sklearn RandomForest при наличии обученной модели.

## Пайплайн

```
cv → atoms → regions → ocr
  → build_ui_graph()
  → extract_features()
  → classify_roles()
  → apply_roles_to_atoms()
  → final_atoms (noise отброшены)
```

- bbox не меняется.
- ui_role == noise → атом исключается из final_atoms.
- ui_role != atom.type → логируется override.

## Зависимости

- scikit-learn — опционально (для обученного классификатора); при отсутствии используется только rule-based.
