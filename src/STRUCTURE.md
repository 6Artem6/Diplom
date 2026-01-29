# Структура репозитория BPG Construction Pipeline

## Архитектурное решение

Архитектура следует принципам **Clean Architecture** с явным разделением слоёв:

1. **Domain Layer** (`src/domain/`) — чистая бизнес-логика, без зависимостей от инфраструктуры
2. **Application Layer** (`src/application/`) — use cases, оркестрирующие domain-сервисы
3. **Infrastructure Layer** (`src/infrastructure/`) — конкретные реализации (YOLO, CLIP, Chroma)
4. **API Layer** (`src/api/`) — FastAPI endpoints, только инфраструктурный слой

### Почему такая структура?

**Для domain-aware LLM:**
- LLM runtime будет запрашивать BPG через API
- Domain-модели (EntityType, EntityInstance, etc.) — это язык общения с LLM
- Чистое разделение позволяет менять инфраструктуру без изменения domain-логики

**Для runtime-использования BPG:**
- BPG хранится как явная структура данных
- LLM может запрашивать контекст через `/api/v1/bpg/{id}/context`
- Provenance и Confidence позволяют LLM оценивать достоверность фактов

---

## Структура директорий

```
src/
├── domain/                    # Domain Layer (чистая бизнес-логика)
│   ├── models/               # Pydantic-модели BPG
│   │   ├── bpg_models.py    # Nodes: EntityType, EntityInstance, etc.
│   │   ├── bpg_edges.py     # Edges: CrossView, Functional, etc.
│   │   └── provenance.py    # Provenance, Confidence
│   └── interfaces/          # Абстрактные интерфейсы
│       ├── preprocessing.py
│       ├── gui_detection.py
│       ├── representation.py
│       ├── linking.py
│       └── bpg_construction.py
│
├── application/              # Application Layer (use cases)
│   └── use_cases/
│       └── bpg_pipeline.py  # BuildBPGUseCase - оркестрация pipeline
│
├── infrastructure/          # Infrastructure Layer (реализации)
│   ├── preprocessing/       # OCR, загрузка скриншотов
│   ├── gui_detection/       # YOLO detection (заглушка)
│   ├── representation/      # CLIP/SentenceTransformer (заглушка)
│   ├── storage/             # Chroma vector store
│   ├── linking/             # Entity linking через Chroma
│   └── bpg_construction/    # Построение BPG
│
└── api/                     # API Layer (FastAPI)
    ├── main.py              # FastAPI app
    ├── routes/              # Endpoints
    │   └── bpg.py           # /api/v1/bpg/*
    └── dependencies.py      # Dependency injection
```

---

## Сопоставление модулей pipeline с директориями

### 1. Preprocessing
- **Domain**: `domain/interfaces/preprocessing.py` — интерфейс `PreprocessingService`
- **Infrastructure**: `infrastructure/preprocessing/preprocessing_service.py` — реализация с OCR placeholder
- **Архитектурное решение**: Разделение позволяет менять OCR (Tesseract → PaddleOCR) без изменения domain-логики

### 2. GUI Detection
- **Domain**: `domain/interfaces/gui_detection.py` — интерфейс `GUIDetectionService`, модель `GUIBlock`
- **Infrastructure**: `infrastructure/gui_detection/gui_detection_service.py` — реализация с YOLO placeholder
- **Архитектурное решение**: YOLO-модель может быть заменена без изменения use case

### 3. Representation
- **Domain**: `domain/interfaces/representation.py` — интерфейс `RepresentationService`, модель `MultimodalEmbedding`
- **Infrastructure**: `infrastructure/representation/representation_service.py` — реализация с CLIP/SentenceTransformer placeholder
- **Архитектурное решение**: Embedding-модели могут быть заменены (CLIP → другой vision model)

### 4. Entity Linking
- **Domain**: `domain/interfaces/linking.py` — интерфейс `EntityLinkingService`
- **Infrastructure**: `infrastructure/linking/entity_linking_service.py` — реализация через Chroma
- **Архитектурное решение**: Chroma используется для similarity search, но может быть заменён на FAISS/Pinecone

### 5. BPG Construction
- **Domain**: `domain/interfaces/bpg_construction.py` — интерфейс `BPGConstructionService`, модель `BusinessProcessGraph`
- **Infrastructure**: `infrastructure/bpg_construction/bpg_construction_service.py` — реализация с placeholder логикой
- **Архитектурное решение**: Граф строится из entities/actions, edges инферируются из clickstreams

---

## Ключевые Pydantic-модели

### Nodes (BPG_SCHEMA.md)

1. **EntityType** — бизнес-концепция (Product, Order)
2. **EntityInstance** — конкретный экземпляр (Product #123)
3. **GUIManifestation** — визуальное представление entity в GUI
4. **Action** — действие, доступное в GUI
5. **PatternNode** — паттерн workflow (Checkout Flow)
6. **Rule** — бизнес-правило/ограничение

### Edges

1. **CrossViewEdge** — связывает разные GUIManifestations одной EntityInstance
2. **FunctionalEdge** — action → result
3. **TemporalEdge** — временной порядок из clickstreams
4. **ConditionalEdge** — preconditions/postconditions
5. **CompositionalEdge** — иерархия (Order contains Items)
6. **RoleEdge** — семантическая роль UI-элемента

### Metadata

1. **Provenance** — источник факта (evidence_sources, inference_method)
2. **Confidence** — уверенность в факте (score, method, metadata)

---

## Skeleton Pipeline

### Use Case: BuildBPGUseCase

```python
# application/use_cases/bpg_pipeline.py

async def execute(request: BuildBPGRequest) -> BusinessProcessGraph:
    # 1. Preprocessing
    screenshots = await preprocessing.load_screenshots(...)
    
    # 2. GUI Detection
    blocks = await gui_detection.detect_gui_blocks(...)
    
    # 3. Representation
    embeddings = await representation.generate_embeddings(...)
    
    # 4. Entity Linking
    cross_view_edges = await linking.link_entities(...)
    entity_instances = await linking.create_entity_instances(...)
    
    # 5. BPG Construction
    bpg = await bpg_construction.build_bpg(...)
    
    return bpg
```

### Заглушки

- **YOLO**: `GUIDetectionServiceImpl` возвращает mock blocks
- **CLIP/SentenceTransformer**: `RepresentationServiceImpl` возвращает zero-embeddings
- **OCR**: `PreprocessingServiceImpl` возвращает пустой OCR text
- **BPG Construction**: `BPGConstructionServiceImpl` создаёт placeholder edges/patterns

---

## Следующие шаги

1. **Интеграция реальных моделей**:
   - Загрузить YOLO weights, реализовать `YOLODetector`
   - Интегрировать CLIP для visual embeddings
   - Интегрировать SentenceTransformer для text embeddings
   - Добавить OCR (Tesseract или PaddleOCR)

2. **Улучшение entity linking**:
   - Использовать HDBSCAN для clustering manifestations
   - Комбинировать visual + text embeddings для similarity
   - Добавить temporal evidence (same entity в последовательных скриншотах)

3. **BPG Construction логика**:
   - Использовать PM4Py для temporal pattern mining
   - LLM-assisted inference для EntityType naming
   - Rule inference из observed constraints

4. **Storage и persistence**:
   - Сохранять BPG в Neo4j или PostgreSQL
   - Индексировать BPG для быстрого context retrieval
   - Реализовать `/api/v1/bpg/{id}/context` endpoint

5. **LLM Runtime интеграция**:
   - Endpoint для context enrichment
   - Validation endpoint для action plans
   - Explainability endpoint (why is this action valid/invalid?)

---

## Ограничения и допущения

### Skeleton-код

- Нет реальных ML-моделей (YOLO, CLIP, OCR)
- Mock embeddings (zero-vectors)
- Placeholder entity linking (простой threshold)
- Нет persistence (BPG не сохраняется)

### Архитектурные допущения

- Chroma используется для vector storage (может быть заменён)
- FastAPI — только инфраструктурный слой (не domain-логика)
- Domain-модели — Pydantic (типизация, валидация)

### Исследовательские допущения

- GUI-only evidence (нет backend/API)
- Confidence и Provenance — first-class citizens
- Читаемость > оптимизация
