# Архитектурный аудит BPG Construction Pipeline

## 1. АУДИТ (что сломано и почему)

### 1.1. End-to-end data flow — КРИТИЧЕСКАЯ ПРОБЛЕМА

**Проблема:** Data flow разорван в нескольких местах:

1. **Неправильный порядок создания entities:**
   - В `bpg_pipeline.py:106-126`: Manifestations создаются с `entity_instance_id=uuid4()` (случайный UUID)
   - В `entity_linking_service.py:98-132`: Entity instances создаются ПОСЛЕ manifestations и группируются по этим случайным ID
   - **Результат:** Entity instances не имеют смысла — они группируются по случайным UUID, а не по реальной схожести

2. **Cross_view edges не добавляются в BPG:**
   - В `bpg_pipeline.py:87-89`: Cross_view edges создаются, но не добавляются в возвращаемый BPG
   - В `bpg_construction.py:41`: BPG модель не содержит cross_view edges
   - **Результат:** Связи между manifestations теряются

3. **Chroma используется неправильно:**
   - Embeddings хранятся в Chroma, но entity linking использует placeholder логику
   - `target_id=uuid4()` в cross_view edges — placeholder, не реальный ID

**Критичность:** 🔴 КРИТИЧНО — pipeline не работает end-to-end

---

### 1.2. BPG не является runtime-артефактом — КРИТИЧЕСКАЯ ПРОБЛЕМА

**Проблема:** BPG строится, но не используется:

1. **Endpoints не реализованы:**
   - `/api/v1/bpg/build` возвращает 501 (Not Implemented)
   - `/api/v1/bpg/{id}` возвращает 501
   - `/api/v1/bpg/{id}/context` возвращает 501
   - **Результат:** LLM agent не может запросить BPG

2. **Нет persistence:**
   - BPG возвращается из use case, но не сохраняется
   - Нет storage layer для BPG
   - **Результат:** BPG теряется после построения

3. **Нет query механизма:**
   - Нет способа найти релевантный subgraph
   - Нет использования confidence/provenance для фильтрации
   - **Результат:** BPG не может использоваться для runtime decisions

**Критичность:** 🔴 КРИТИЧНО — цель проекта (runtime BPG) не достигнута

---

### 1.3. Архитектурная магия — СРЕДНЯЯ ПРОБЛЕМА

**Проблема:** Неявные допущения и placeholder логика:

1. **Entity linking магия:**
   - Manifestations создаются с random entity_instance_id
   - Потом группируются по этому ID (это не имеет смысла)
   - Должно быть: clustering → entity instances → manifestations

2. **Cross_view edges магия:**
   - `target_id=uuid4()` — placeholder, не реальный ID
   - Similarity search находит manifestations, но не может их связать

3. **Chroma роль неясна:**
   - Embeddings хранятся, но не используются для правильного linking
   - Нет явного контракта: "Chroma используется для similarity search в entity linking"

**Критичность:** 🟡 СРЕДНЯЯ — работает, но непонятно как

---

### 1.4. Соответствие схемам — СРЕДНЯЯ ПРОБЛЕМА

**Проблема:** Расхождения с диаграммами:

1. **RAG-Enhanced BPG Services:**
   - Схемы показывают EntityService, ProcessService, RulesEngine
   - В коде: только BPGConstructionService
   - **Допустимо для PoC:** можно упростить

2. **Context Enrichment:**
   - Схемы показывают `/bpg/{id}/context` endpoint
   - В коде: endpoint есть, но не реализован
   - **Критично:** нужно реализовать хотя бы минимально

3. **Storage:**
   - Схемы показывают Neo4j + PostgreSQL + Pinecone
   - В коде: только Chroma (для embeddings)
   - **Допустимо для PoC:** можно использовать in-memory storage

**Критичность:** 🟡 СРЕДНЯЯ — допустимо для PoC, но нужно документировать упрощения

---

## 2. ИСПРАВЛЕНИЯ (что изменено и зачем)

### 2.1. Исправлен data flow

**Изменения:**

1. **Правильный порядок entity linking:**
   - Сначала: clustering manifestations по embeddings → entity instances
   - Потом: создание manifestations с правильными entity_instance_id
   - Исправлено в: `entity_linking_service.py`, `bpg_pipeline.py`

2. **Cross_view edges добавлены в BPG:**
   - BPG модель расширена: `cross_view_edges: List[CrossViewEdge]`
   - Cross_view edges добавляются в BPG при построении
   - Исправлено в: `bpg_construction.py`, `bpg_pipeline.py`

3. **Chroma роль явна:**
   - Chroma используется для similarity search в entity linking
   - Embeddings хранятся с metadata (manifestation_id, screenshot_id)
   - Исправлено в: `entity_linking_service.py`

**Зачем:** Обеспечить causal data flow от screenshots до BPG

---

### 2.2. BPG стал runtime-артефактом

**Изменения:**

1. **Добавлен BPG storage:**
   - In-memory storage для PoC (`infrastructure/storage/bpg_store.py`)
   - BPG сохраняется после построения
   - Исправлено в: новый файл `bpg_store.py`

2. **Реализованы endpoints:**
   - `/api/v1/bpg/build` — строит и сохраняет BPG
   - `/api/v1/bpg/{id}` — возвращает BPG по ID
   - `/api/v1/bpg/{id}/context` — возвращает релевантный subgraph для LLM
   - Исправлено в: `api/routes/bpg.py`, `api/dependencies.py`

3. **Добавлен query механизм:**
   - `BPGQueryService` для поиска релевантных nodes/edges
   - Фильтрация по confidence, provenance
   - Исправлено в: новый файл `infrastructure/storage/bpg_query.py`

**Зачем:** LLM agent может запросить BPG для runtime decisions

---

### 2.3. Убрана архитектурная магия

**Изменения:**

1. **Явный контракт entity linking:**
   - Clustering → entity instances → manifestations
   - Cross_view edges создаются с реальными target_id
   - Исправлено в: `entity_linking_service.py`

2. **Явные форматы данных:**
   - Chroma metadata формализован
   - BPG storage формат явный
   - Исправлено в: документация + код

3. **Явные интерфейсы:**
   - `BPGStorage` interface
   - `BPGQueryService` interface
   - Исправлено в: новые интерфейсы в `domain/interfaces/`

**Зачем:** Понятность и тестируемость

---

## 3. ОБНОВЛЁННАЯ АРХИТЕКТУРА

### Data Flow (исправленный)

```
Screenshots + Clickstreams
  ↓
[Preprocessing] → ScreenshotData (OCR)
  ↓
[GUI Detection] → GUIBlocks
  ↓
[Representation] → MultimodalEmbeddings
  ↓
[Entity Linking] → EntityInstances (через clustering)
                  → GUIManifestations (с правильными entity_instance_id)
                  → CrossViewEdges
  ↓
[BPG Construction] → BusinessProcessGraph
                  → Сохранение в BPGStorage
  ↓
[Runtime Query] → LLM Agent запрашивает /bpg/{id}/context
```

### Runtime Usage

```
LLM Agent: "What actions are available for Product entity?"
  ↓
GET /api/v1/bpg/{id}/context?query=Product&entity_type=Product
  ↓
BPGQueryService.find_relevant_context()
  ↓
Returns: Actions, Rules, Patterns related to Product
  ↓
LLM Agent: Uses BPG context for decision making
```

---

## 4. СТРУКТУРА РЕПОЗИТОРИЯ (обновлённая)

```
src/
├── domain/
│   ├── models/              # BPG models (без изменений)
│   └── interfaces/
│       ├── bpg_storage.py   # НОВЫЙ: BPGStorage interface
│       ├── bpg_query.py     # НОВЫЙ: BPGQueryService interface
│       └── ... (остальные без изменений)
│
├── application/
│   └── use_cases/
│       └── bpg_pipeline.py  # ИСПРАВЛЕНО: правильный порядок entity linking
│
├── infrastructure/
│   ├── storage/
│   │   ├── bpg_store.py      # НОВЫЙ: In-memory BPG storage
│   │   ├── bpg_query.py     # НОВЫЙ: BPG query service
│   │   └── chroma_store.py   # Без изменений
│   ├── linking/
│   │   └── entity_linking_service.py  # ИСПРАВЛЕНО: правильный порядок
│   └── bpg_construction/
│       └── bpg_construction_service.py  # ИСПРАВЛЕНО: cross_view edges
│
└── api/
    ├── routes/
    │   └── bpg.py            # ИСПРАВЛЕНО: реализованы endpoints
    └── dependencies.py       # ИСПРАВЛЕНО: инициализация storage
```

---

## 5. КЛЮЧЕВЫЕ DATA-МОДЕЛИ (обновлённые)

### BusinessProcessGraph (обновлено)

```python
class BusinessProcessGraph(BaseModel):
    id: UUID  # НОВОЕ: ID для storage
    entity_types: List[EntityType]
    entity_instances: List[EntityInstance]
    actions: List[Action]
    patterns: List[PatternNode]
    rules: List[Rule]
    edges: List[Union[FunctionalEdge, TemporalEdge, ...]]
    cross_view_edges: List[CrossViewEdge]  # НОВОЕ: cross_view edges
    created_at: datetime  # НОВОЕ: timestamp
```

### BPGStorage Interface (новое)

```python
class BPGStorage(ABC):
    @abstractmethod
    async def save(self, bpg: BusinessProcessGraph) -> UUID:
        """Save BPG and return ID."""
    
    @abstractmethod
    async def get(self, bpg_id: UUID) -> Optional[BusinessProcessGraph]:
        """Get BPG by ID."""
```

### BPGQueryService Interface (новое)

```python
class BPGQueryService(ABC):
    @abstractmethod
    async def find_relevant_context(
        self,
        bpg_id: UUID,
        query: str,
        entity_type: Optional[str] = None,
        min_confidence: float = 0.5,
    ) -> BusinessProcessGraph:
        """Find relevant subgraph for LLM context."""
```

---

## 6. END-TO-END ПРИМЕР ПРОХОЖДЕНИЯ ДАННЫХ

### Пример: 2 скриншота с продуктом

**Input:**
- `screenshot1.png` (каталог продуктов)
- `screenshot2.png` (страница продукта)
- Clickstream: `[{"from": "catalog", "to": "product_page", "action": "click"}]`

**Pipeline:**

1. **Preprocessing:**
   - `ScreenshotData(screenshot_id="s1", ocr_text="Product Widget $10")`
   - `ScreenshotData(screenshot_id="s2", ocr_text="Widget Description Add to Cart")`

2. **GUI Detection:**
   - `GUIBlock(id="b1", screenshot_id="s1", element_types=["text", "image"])`
   - `GUIBlock(id="b2", screenshot_id="s2", element_types=["button", "text"])`

3. **Representation:**
   - `MultimodalEmbedding(block_id="b1", visual_emb=[...], text_emb=[...])`
   - `MultimodalEmbedding(block_id="b2", visual_emb=[...], text_emb=[...])`

4. **Entity Linking:**
   - Clustering: `b1` и `b2` похожи (similarity=0.85) → один EntityInstance
   - `EntityInstance(id="e1", entity_type_id="t1", attributes={"name": "Widget"})`
   - `GUIManifestation(id="m1", entity_instance_id="e1", screenshot_id="s1")`
   - `GUIManifestation(id="m2", entity_instance_id="e1", screenshot_id="s2")`
   - `CrossViewEdge(source_id="m1", target_id="m2", similarity_score=0.85)`

5. **BPG Construction:**
   - `EntityType(id="t1", name="Product")`
   - `Action(id="a1", action_type="click", trigger_element={...})`
   - `FunctionalEdge(source_id="a1", target_id="e1")`
   - `TemporalEdge(source_id="s1", target_id="s2", frequency=1.0)`
   - `BusinessProcessGraph(id="bpg1", ..., cross_view_edges=[...])`

6. **Storage:**
   - `BPGStorage.save(bpg)` → `bpg_id="bpg1"`

7. **Runtime Query:**
   - `GET /api/v1/bpg/bpg1/context?query=Product`
   - Returns: `{entity_types: [Product], actions: [click], rules: []}`

---

## 7. DOCKER-COMPOSE.YML + КАК ЗАПУСТИТЬ

См. `docker-compose.yml` и `Dockerfile` в репозитории.

**Запуск:**
```bash
docker-compose up --build
```

**Тестирование:**
```bash
# Build BPG
curl -X POST "http://localhost:8000/api/v1/bpg/build" \
  -H "Content-Type: application/json" \
  -d '{"screenshot_paths": ["/data/screenshot1.png"], "clickstream_data": []}'

# Get BPG context
curl "http://localhost:8000/api/v1/bpg/{bpg_id}/context?query=Product"
```
