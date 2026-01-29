# Архитектурный аудит: Резюме исправлений

## 1. АУДИТ (что было сломано)

### Критические проблемы:

1. **Разорванный data flow:**
   - Manifestations создавались с random `entity_instance_id`
   - Entity instances группировались по этим случайным ID
   - **Исправлено:** Правильный порядок: clustering → entity instances → manifestations

2. **BPG не runtime-артефакт:**
   - Endpoints возвращали 501 (Not Implemented)
   - Нет persistence
   - **Исправлено:** Добавлен storage, реализованы endpoints

3. **Cross_view edges терялись:**
   - Создавались, но не добавлялись в BPG
   - **Исправлено:** Добавлены в BPG модель

### Средние проблемы:

4. **Архитектурная магия:**
   - Неявные допущения в entity linking
   - **Исправлено:** Явные интерфейсы и контракты

---

## 2. ИСПРАВЛЕНИЯ

### 2.1. Исправлен data flow

**Было:**
```python
manifestations = create_manifestations(blocks)  # с random entity_instance_id
entity_instances = create_entity_instances(manifestations)  # группировка по random ID
```

**Стало:**
```python
manifestations = create_manifestations(blocks)  # временные
cross_view_edges = link_entities(manifestations, embeddings)  # clustering
entity_instances = create_entity_instances(manifestations, cross_view_edges)  # правильная группировка
```

### 2.2. Добавлен BPG storage

- `InMemoryBPGStore` для PoC
- `BPGStorage` interface для расширяемости
- BPG сохраняется после построения

### 2.3. Реализованы endpoints

- `POST /api/v1/bpg/build` — строит и сохраняет BPG
- `GET /api/v1/bpg/{id}` — возвращает BPG
- `GET /api/v1/bpg/{id}/context` — возвращает релевантный subgraph для LLM

### 2.4. Cross_view edges в BPG

- Добавлено поле `cross_view_edges: List[CrossViewEdge]` в `BusinessProcessGraph`
- Edges передаются в `build_bpg()` и сохраняются

---

## 3. ОБНОВЛЁННАЯ АРХИТЕКТУРА

### Data Flow (исправленный)

```
Screenshots + Clickstreams
  ↓
[Preprocessing] → ScreenshotData
  ↓
[GUI Detection] → GUIBlocks
  ↓
[Representation] → MultimodalEmbeddings
  ↓
[Entity Linking] → CrossViewEdges (clustering)
                  → EntityInstances (из connected components)
  ↓
[BPG Construction] → BusinessProcessGraph (с cross_view_edges)
  ↓
[Storage] → InMemoryBPGStore
  ↓
[Runtime Query] → LLM Agent запрашивает /bpg/{id}/context
```

### Runtime Usage

```
LLM Agent: "What actions are available for Product?"
  ↓
GET /api/v1/bpg/{id}/context?entity_type=Product&min_confidence=0.5
  ↓
BPGQueryService.find_relevant_context()
  ↓
Returns: {entity_types: [Product], actions: [click], ...}
  ↓
LLM Agent: Uses BPG context for decision making
```

---

## 4. СТРУКТУРА РЕПОЗИТОРИЯ

```
src/
├── domain/
│   ├── models/              # BPG models
│   └── interfaces/
│       ├── bpg_storage.py   # НОВЫЙ
│       ├── bpg_query.py     # НОВЫЙ
│       └── ...
│
├── infrastructure/
│   ├── storage/
│   │   ├── bpg_store.py      # НОВЫЙ: InMemoryBPGStore
│   │   ├── bpg_query.py     # НОВЫЙ: BPGQueryServiceImpl
│   │   └── chroma_store.py
│   └── ...
│
└── api/
    ├── routes/
    │   └── bpg.py            # ИСПРАВЛЕНО: реализованы endpoints
    └── dependencies.py       # ИСПРАВЛЕНО: добавлены storage/query
```

---

## 5. КЛЮЧЕВЫЕ DATA-МОДЕЛИ

### BusinessProcessGraph (обновлено)

```python
class BusinessProcessGraph(BaseModel):
    id: UUID  # НОВОЕ
    entity_types: List[EntityType]
    entity_instances: List[EntityInstance]
    actions: List[Action]
    patterns: List[PatternNode]
    rules: List[Rule]
    edges: List[Union[FunctionalEdge, TemporalEdge, ...]]
    cross_view_edges: List[CrossViewEdge]  # НОВОЕ
    created_at: datetime  # НОВОЕ
```

### BPGStorage Interface (новое)

```python
class BPGStorage(ABC):
    async def save(bpg: BusinessProcessGraph) -> UUID
    async def get(bpg_id: UUID) -> Optional[BusinessProcessGraph]
    async def list_all() -> List[UUID]
```

### BPGQueryService Interface (новое)

```python
class BPGQueryService(ABC):
    async def find_relevant_context(
        bpg_id: UUID,
        query: Optional[str],
        entity_type: Optional[str],
        min_confidence: float,
    ) -> BusinessProcessGraph
```

---

## 6. END-TO-END ПРИМЕР

### Input:
- `screenshot1.png` (каталог)
- `screenshot2.png` (страница продукта)

### Pipeline:

1. **Preprocessing:**
   - `ScreenshotData(screenshot_id="s1", ocr_text="Product Widget")`

2. **GUI Detection:**
   - `GUIBlock(id="b1", screenshot_id="s1", element_types=["text", "image"])`

3. **Representation:**
   - `MultimodalEmbedding(block_id="b1", visual_emb=[...], text_emb=[...])`

4. **Entity Linking:**
   - Clustering: `b1` и `b2` похожи → `CrossViewEdge(source="m1", target="m2")`
   - Connected components → `EntityInstance(id="e1")`

5. **BPG Construction:**
   - `BusinessProcessGraph(id="bpg1", cross_view_edges=[...])`

6. **Storage:**
   - `BPGStorage.save(bpg)` → `bpg_id="bpg1"`

7. **Runtime Query:**
   - `GET /api/v1/bpg/bpg1/context?query=Product`
   - Returns: `{entity_types: [Product], actions: [click], ...}`

---

## 7. DOCKER-COMPOSE + ЗАПУСК

### docker-compose.yml

```yaml
services:
  bpg_service:
    build:
      context: .
      dockerfile: src/Dockerfile
    ports:
      - "8001:8000"
    volumes:
      - ./src:/app/src
      - ./data:/app/data
```

### Запуск

```bash
# Build and start
docker-compose up --build bpg_service

# Test
curl -X POST "http://localhost:8001/api/v1/bpg/build" \
  -H "Content-Type: application/json" \
  -d '{"screenshot_paths": ["/app/data/screenshot1.png"]}'
```

---

## ИТОГИ

✅ **Исправлено:**
- Data flow работает end-to-end
- BPG является runtime-артефактом
- Cross_view edges сохраняются
- Endpoints реализованы
- Storage добавлен

✅ **Готово к использованию:**
- Минимально рабочая система
- Mock данные, но реальный pipeline
- LLM agent может запрашивать BPG context

⚠️ **Ограничения (для PoC допустимо):**
- In-memory storage (не persistent)
- Mock YOLO/CLIP/OCR
- Упрощённая entity linking (threshold-based)

---

## Следующие шаги

1. Интеграция реальных моделей (YOLO, CLIP, OCR)
2. Улучшение entity linking (HDBSCAN clustering)
3. Persistence (Neo4j/PostgreSQL)
4. Улучшение query service (semantic search)
