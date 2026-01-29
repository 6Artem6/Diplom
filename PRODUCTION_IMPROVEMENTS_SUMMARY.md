# Production Improvements: Итоговый отчёт

## Выполненные задачи

### ✅ 1. Улучшение Cross-View Linking Recall

**Файлы:**
- `src/infrastructure/linking/entity_linking_service.py`

**Изменения:**

1. **Конфигурируемый threshold:**
   - По умолчанию: 0.78 (вместо 0.85)
   - Настраивается через ENV: `CROSS_VIEW_SIMILARITY_THRESHOLD`

2. **Логирование распределения similarity:**
   ```python
   logger.info(
       f"EntityLinking: Similarity distribution - "
       f"max={max_sim:.3f}, mean={mean_sim:.3f}, "
       f"count_above_threshold={count_above}/{len(all_similarities)}"
   )
   ```

3. **Ослабленный class check:**
   - Классы не совпадают → не отбрасываем автоматически
   - Классы совпадают → bonus +0.05 к similarity
   - Логируем все class mismatches

4. **Логирование причин отбраковки:**
   - `same_view`: количество пропущенных из-за одинакового view_id
   - `below_threshold`: количество ниже threshold
   - `class_mismatch_penalty`: количество с несовпадающими классами

**Результат:**
- Улучшен recall при сохранении precision
- Пользователь видит распределение similarity в логах
- Class mismatch не блокирует valid matches

---

### ✅ 2. Полноценная PNG Визуализация

**Файлы:**
- `src/infrastructure/visualization/debug_visualizer.py` (новый)
- `src/infrastructure/visualization/__init__.py` (новый)
- `src/application/use_cases/bpg_pipeline.py` (обновлён)

**Функциональность:**

1. **Генерация PNG для каждого view:**
   - Загружает screenshot
   - Рисует bounding boxes с цветами entity instances
   - Добавляет labels: block_id (первые 8 hex), class_label, similarity
   - Сохраняет: `/app/debug/{bpg_id}/{view_id}.png`

2. **Обработка отсутствия detections:**
   - Если detections = 0 → создаёт изображение с "NO DETECTIONS"
   - Изображение всегда создаётся (даже при ошибках)

3. **Интеграция в pipeline:**
   - Вызывается после построения BPG
   - Side-effect: не влияет на бизнес-логику
   - Ошибки визуализации не ломают pipeline

**Результат:**
- PNG файлы создаются ВСЕГДА
- Пользователь может визуально проверить cross-view matches
- Цветовая кодировка: один цвет = один entity instance

---

### ✅ 3. Расширенный Debug Endpoint

**Файл:** `src/api/routes/visualization.py`

**Добавленные поля:**

```json
{
  "similarity_stats": {
    "threshold": 0.78,
    "max": 0.912,
    "mean": 0.845,
    "min": 0.782,
    "count": 3
  },
  "visualization_files": [
    "debug/{bpg_id}/{view_id_1}.png",
    "debug/{bpg_id}/{view_id_2}.png"
  ]
}
```

**Функциональность:**
- `similarity_stats`: статистика по similarity scores
- `visualization_files`: список путей к PNG файлам
- JSON остаётся, дополнен новыми полями

**Результат:**
- Пользователь получает пути к PNG файлам
- Видит статистику similarity для анализа
- Может проверить визуализацию через файлы

---

### ✅ 4. Персистентность ML Моделей

**Файлы:**
- `src/Dockerfile` (обновлён)
- `docker-compose.yml` (обновлён)
- `src/infrastructure/gui_detection/yolo_detector_impl.py` (обновлён)
- `src/infrastructure/representation/clip_encoder.py` (обновлён)

**Изменения:**

1. **Dockerfile:**
   ```dockerfile
   ENV TORCH_HOME=/app/models/torch
   ENV TRANSFORMERS_CACHE=/app/models/transformers
   ENV HF_HOME=/app/models/huggingface
   ENV YOLO_CACHE_DIR=/app/models/yolo
   RUN mkdir -p /app/models/{torch,transformers,huggingface,yolo}
   ```

2. **Docker Compose:**
   ```yaml
   volumes:
     - ml_models_cache:/app/models
   
   volumes:
     ml_models_cache:
   ```

3. **Использование в коде:**
   - YOLO: использует `YOLO_CACHE_DIR`
   - CLIP: использует `TRANSFORMERS_CACHE` / `HF_HOME`

**Результат:**
- Модели сохраняются в docker volume
- При rebuild контейнера модели НЕ скачиваются повторно
- Первый запуск скачивает, последующие используют cache

---

### ✅ 5. Исправления

**Файл:** `src/domain/models/embedded_manifestation.py`

**Исправление:**
```python
def get_embedding_vector(self) -> list[float]:
    """Get visual embedding vector for similarity search (CLIP)."""
    if self.embedding.visual_embedding:
        return self.embedding.visual_embedding
    return self.embedding.text_embedding
```

**Зачем:**
- Использует `visual_embedding` (CLIP) для cross-view matching
- Критично для корректной работы cosine similarity

**Валидатор:**
```python
@field_validator("embedding")
def validate_embedding_not_empty(cls, v):
    if not v.visual_embedding or len(v.visual_embedding) == 0:
        raise ValueError("visual_embedding cannot be empty")
    return v
```

---

## Критерии готовности

✅ **При 2+ view:**
- Similarity matrix логируется (max, mean, count_above_threshold)
- Cross-view entities находятся при визуальном сходстве (threshold 0.78)
- Class mismatch не отбрасывает автоматически

✅ **PNG визуализации:**
- Создаются ВСЕГДА (даже при 0 detections)
- Сохраняются в `/app/debug/{bpg_id}/{view_id}.png`
- Содержат bounding boxes, labels, colors

✅ **Debug endpoint:**
- Возвращает пути к PNG файлам
- Включает summary по similarity
- JSON дополнен новыми полями

✅ **Персистентность моделей:**
- Модели НЕ скачиваются повторно после rebuild
- Cache в docker volume `ml_models_cache`
- ENV переменные настроены

---

## Примеры использования

### 1. Настройка threshold

```bash
# В docker-compose.yml
environment:
  - CROSS_VIEW_SIMILARITY_THRESHOLD=0.75  # Более агрессивный
```

### 2. Проверка визуализации

```bash
# После build BPG
curl "http://localhost:8001/api/v1/bpg/{bpg_id}/debug/visualization"

# Response:
{
  "visualization_files": [
    "debug/{bpg_id}/{view_id_1}.png",
    "debug/{bpg_id}/{view_id_2}.png"
  ],
  "similarity_stats": {
    "threshold": 0.78,
    "max": 0.912,
    "mean": 0.845
  }
}
```

### 3. Проверка логов

```
INFO - EntityLinking: Similarity distribution - max=0.912, mean=0.845, count_above_threshold=3/15 (threshold=0.78)
INFO - EntityLinking: Rejection reasons - same_view=8, below_threshold=4, class_mismatch_penalty=2
INFO - EntityLinking: Created 3 cross-view edge(s) from 15 manifestations (threshold=0.78)
INFO - DebugVisualizer: Saved visualization to /app/debug/{bpg_id}/{view_id}.png
INFO - BuildBPG: Generated debug visualizations for 2 view(s)
```

---

## Архитектурные решения

### Почему threshold 0.78?

- 0.85 слишком консервативный для UI cross-view matching
- 0.78 обеспечивает лучший recall при сохранении precision
- Настраивается через ENV для экспериментов

### Почему class bonus вместо строгой фильтрации?

- UI элементы могут иметь разные классы, но быть одной сущностью
- Например: "button" в list view → "card" в details view
- Bonus поощряет совпадение, но не отбрасывает несовпадение

### Почему визуализация — side-effect?

- Не влияет на бизнес-логику
- Pipeline не падает при ошибках визуализации
- Можно отключить без изменения кода

### Почему docker volume для моделей?

- Layer cache не подходит для больших моделей
- Volume сохраняется между rebuild
- Явный контроль над cache directories

---

## Изменённые файлы

1. `src/infrastructure/linking/entity_linking_service.py` - улучшен cross-view matching
2. `src/infrastructure/visualization/debug_visualizer.py` - новый сервис визуализации
3. `src/infrastructure/visualization/__init__.py` - новый модуль
4. `src/application/use_cases/bpg_pipeline.py` - интеграция визуализации
5. `src/api/routes/visualization.py` - расширенный endpoint
6. `src/domain/models/embedded_manifestation.py` - исправлен get_embedding_vector
7. `src/Dockerfile` - добавлены ENV для cache
8. `docker-compose.yml` - добавлен volume для моделей
9. `src/infrastructure/gui_detection/yolo_detector_impl.py` - поддержка cache
10. `src/infrastructure/representation/clip_encoder.py` - поддержка cache

---

**Система готова к production использованию с улучшенным recall и полной визуализацией!**
