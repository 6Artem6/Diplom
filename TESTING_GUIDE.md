# Руководство по тестированию исправлений

## Быстрый тест

### 1. Запуск системы

```bash
docker-compose up --build bpg_service
```

### 2. Тест с двумя скриншотами

```bash
curl -X POST "http://localhost:8001/api/v1/bpg/build" \
  -H "Content-Type: application/json" \
  -d '{
    "screenshot_paths": [
      "/app/data/view_list.png",
      "/app/data/view_details.png"
    ]
  }'
```

### 3. Проверка логов

**Должно быть в логах:**

```
INFO - BuildBPG: Starting pipeline with 2 screenshot(s)
INFO - ChromaVectorStore: Cleared gui_blocks collection
INFO - ChromaVectorStore: Stored 2 embeddings (phase=within_view, views=1, ...)
INFO - ChromaVectorStore: Stored 4 embeddings (phase=cross_view, views=2, ...)
INFO - EntityLinking: Created 1 cross-view edge(s) from 4 manifestations
INFO - EntityLinking: Created cross-view edge ... (view=v1) → ... (view=v2), similarity=0.850
INFO - EntityLinking: Created EntityInstance ... with 4 manifestations across 2 view(s)
INFO - BuildBPG: 1 EntityInstance(s) appear in multiple views
```

**НЕ должно быть:**

```
ERROR - Non-empty lists are required for ['metadatas']
ERROR - Empty metadata at index ...
ERROR - CrossViewEdge cannot be created between manifestations with same view_id
```

### 4. Проверка визуализации

```bash
curl "http://localhost:8001/api/v1/bpg/{bpg_id}/debug/visualization"
```

**Ожидаемый результат:**

```json
{
  "cross_view_edges": [
    {
      "source_id": "...",
      "target_id": "...",
      "source_view_id": "v1",
      "target_view_id": "v2",
      "validation": "✅ Different views"
    }
  ],
  "entity_instances": [
    {
      "id": "...",
      "view_count": 2,
      "is_cross_view": true
    }
  ],
  "summary": {
    "total_cross_view_edges": 1,
    "cross_view_entities": 1,
    "validation_passed": true
  }
}
```

### 5. Runtime-запрос

```bash
curl "http://localhost:8001/api/v1/bpg/{bpg_id}/context?query=Product"
```

**Ожидаемый результат:**
- Возвращает subgraph с EntityInstances
- Cross-view edges присутствуют
- `view_count >= 2` для cross-view entities

---

## Критерии успеха

✅ **Нет ошибок metadatas:**
- Нет `Non-empty lists are required for ['metadatas']`
- Нет `Empty metadata at index ...`

✅ **Cross-view edges корректны:**
- Все edges имеют `source_view_id != target_view_id`
- В логах: `validation: "✅ Different views"`

✅ **Cross-view entities:**
- `EntityInstance.view_count >= 2` для cross-view entities
- В логах: `EntityInstance ... across 2 view(s)`

✅ **Детерминированность:**
- Повторный build с теми же данными даёт тот же результат
- Логи показывают очистку Chroma при каждом build
