# Примеры запросов к API atoms_v2 и experimental v2

Базовый URL (локально): `http://localhost:8000/api/v1/debug` (или тот, на котором поднят сервис). В docker-compose порт 8001.

**Отладочные PNG в контейнере:** при `DEBUG_OUTPUT_DIR=/app/debug` (задано в docker-compose) артефакты experimental_v2 и form_container_first пишутся в монтированную директорию `./debug` на хосте: `./debug/experimental_v2/<run_...>/`, `./debug/form_container_first/<run_...>/`. Файлы можно открывать на хосте без копирования из контейнера.

---

## 1. Базовый запуск (без experimental v2)

Только основной пайплайн: Detectron2, OCR, merge, legacy grouping.

```bash
curl -X POST "http://localhost:8000/api/v1/debug/atoms-v2-pipeline?parallel_ocr=true&legacy_text_pipeline=true" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@/path/to/screenshot.png"
```

**Ответ:** JSON с полями `atoms`, `regions`, `raw_ocr_boxes`, `log`, `unified_ui`, `text_ui_links` и др. В `log` — строки вида `multilevel_field: added N atoms`, `experimental_multilevel_v2` не вызывается.

---

## 2. С включённым experimental multilevel v2 (без сохранения визуализаций)

Дополнительно выполняется пайплайн v2 (Level 0–5); найденные поля v2 добавляются в `atoms`. Визуализации по уровням не сохраняются.

```bash
curl -X POST "http://localhost:8000/api/v1/debug/atoms-v2-pipeline?parallel_ocr=true&legacy_text_pipeline=true&use_experimental_multilevel_v2=true" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@/path/to/screenshot.png"
```

**Что получаете:**
- В `atoms` могут появиться элементы с `"recovery_source": "experimental_multilevel_v2"` и `"evidence": { "source": "experimental_multilevel_v2", "slot_id": "...", "slot_role": "input_slot" }`.
- В `log` — строки вида `experimental_v2: level0 segments=N`, `experimental_v2: level2 rows=N layout=...`, `experimental_multilevel_v2: added N atoms (side-path)`.
- Поля `experimental_v2_debug_directory` и `experimental_v2_debug_files` в ответе будут `null` и `[]`.

---

## 3. Experimental v2 с сохранением визуализаций по уровням

Включён и experimental v2, и запись PNG в каталог. Имя каталога задаётся параметром `experimental_v2_debug_dir` (подкаталог под временной директорией сервера).

```bash
curl -X POST "http://localhost:8000/api/v1/debug/atoms-v2-pipeline?parallel_ocr=true&legacy_text_pipeline=true&use_experimental_multilevel_v2=true&experimental_v2_debug_dir=exp_v2_run_1" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@/path/to/screenshot.png"
```

**Что получаете:**
- То же, что в п. 2 (atoms + log).
- В ответе:
  - `experimental_v2_debug_directory` — полный путь к каталогу на сервере (например `/tmp/experimental_v2/exp_v2_run_1`).
  - `experimental_v2_debug_files` — список имён файлов, например:
    - `level0_page_orientation.png`
    - `level1_semantic_regions.png`
    - `level2_form_skeleton.png`
    - `level3_slots.png`
    - `level4_slot_bbox.png`
    - `level5_form_graph.png`
- Файлы лежат на сервере в указанном каталоге; скачать их по этому API нельзя (нужен отдельный endpoint или доступ к ФС сервера).

---

## 4. Form Container First (ТЗ) — только через API/контейнер

Дополнительно выполняется пайплайн Form Container First: FormContainerDetector → FormInnerLayout → SlotDetector → FieldLocator → FormGraph. Все уровни только внутри bbox контейнера формы.

**Без сохранения визуализаций:**
```bash
curl -X POST "http://localhost:8001/api/v1/debug/atoms-v2-pipeline?parallel_ocr=true&legacy_text_pipeline=false&use_form_container_first=true" \
  -H "accept: application/json" \
  -F "image=@/path/to/screenshot.png"
```

**С сохранением визуализаций** (container_bbox.png, rows.png, slots.png, slot_assignments.png, form_graph.png):
```bash
curl -X POST "http://localhost:8001/api/v1/debug/atoms-v2-pipeline?parallel_ocr=true&legacy_text_pipeline=false&use_form_container_first=true&form_container_first_debug_dir=run_01_single_field_light" \
  -F "image=@/path/to/screenshot.png"
```

**Ответ:** в `atoms` — элементы с `"recovery_source": "form_container_first"`; в `log` — строки `form_container_first: 1 container confidence=...`, `form_container_first: rows=N layout=...`, `form_container_first: filled=M/K`. Поля `form_container_first_debug_directory` и `form_container_first_debug_files` заполняются при заданном `form_container_first_debug_dir`.

---

## 5. Цикл по скриншотам: experimental v2 и Form Container First

Оба пути можно включать в одном запросе. Уникальный каталог на файл — через имя в параметре.

```bash
for f in data/crm_forms/screenshots_single_forms/*.png; do
  name=$(basename "$f" .png)
  curl -s -X POST "http://localhost:8001/api/v1/debug/atoms-v2-pipeline?parallel_ocr=true&legacy_text_pipeline=false&use_experimental_multilevel_v2=true&experimental_v2_debug_dir=run_${name}&use_form_container_first=true&form_container_first_debug_dir=run_${name}" \
    -F "image=@${f}" | jq '.log[-5:], .atoms | length'
done
```

---

## 6. Только сохранить визуализации, не запуская путь

Визуализации пишутся только при выполнении соответствующего пайплайна. Если флаг пути выключен, его debug_dir игнорируется.

---

## 7. Сводка параметров

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `parallel_ocr` | bool | true | Полностраничный OCR. |
| `legacy_text_pipeline` | bool | true | Группировка в строки/абзацы. |
| `use_experimental_multilevel_v2` | bool | false | Запустить experimental v2; атомы добавляются в `atoms`. |
| `experimental_v2_debug_dir` | string, optional | null | Подкаталог для level0–level5 PNG. База: `DEBUG_OUTPUT_DIR` (в контейнере обычно `/app/debug` → монтировано `./debug`) или `$TMP`. Итог: `DEBUG_OUTPUT_DIR/experimental_v2/<значение>` или `$TMP/experimental_v2/...`. |
| `use_form_container_first` | bool | false | Запустить Form Container First; атомы добавляются в `atoms`. |
| `form_container_first_debug_dir` | string, optional | null | Подкаталог для container_bbox.png, rows.png, slots.png, slot_assignments.png. База: `DEBUG_OUTPUT_DIR` или `$TMP`. Итог: `DEBUG_OUTPUT_DIR/form_container_first/<значение>` или `$TMP/form_container_first/...`. |

---

## 8. Пример ответа (фрагмент) с experimental v2

```json
{
  "atoms": [
    { "id": "exp_v2_abc123...", "type": "input_candidate", "bbox": [...], "confidence": 0.8,
      "recovery_source": "experimental_multilevel_v2",
      "evidence": { "source": "experimental_multilevel_v2", "slot_id": "slot_xyz", "slot_role": "input_slot" }
    }
  ],
  "log": [
    "experimental_v2: level0 segments=12",
    "experimental_v2: level2 rows=4 layout=vertical",
    "experimental_multilevel_v2: added 3 atoms (side-path)"
  ],
  "experimental_v2_debug_directory": "/tmp/experimental_v2/exp_v2_run_1",
  "experimental_v2_debug_files": [
    "level0_page_orientation.png",
    "level1_semantic_regions.png",
    "level2_form_skeleton.png",
    "level3_slots.png",
    "level4_slot_bbox.png",
    "level5_form_graph.png"
  ],
  "form_container_first_debug_directory": "/tmp/form_container_first/run_01_single_field_light",
  "form_container_first_debug_files": [
    "container_bbox.png",
    "rows.png",
    "slots.png",
    "slot_assignments.png",
    "form_graph.png"
  ]
}
```
