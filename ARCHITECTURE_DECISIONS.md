# Архитектурные решения BPG Construction Pipeline

## 1. Архитектурное решение

### Clean Architecture с явным разделением слоёв

Архитектура следует принципам **Clean Architecture** для обеспечения:

1. **Независимости domain-логики от инфраструктуры**
   - Domain-модели (EntityType, EntityInstance, etc.) не зависят от YOLO, CLIP, Chroma
   - Позволяет менять ML-модели без изменения бизнес-логики

2. **Тестируемости**
   - Domain-логика тестируется с mock-реализациями
   - Use cases тестируются независимо от инфраструктуры

3. **Расширяемости**
   - Новые inference methods добавляются через интерфейсы
   - Замена Chroma на FAISS/Pinecone не требует изменения domain-логики

### Почему это важно для domain-aware LLM?

**LLM runtime использует BPG как источник знаний:**
- LLM запрашивает контекст через API: `/api/v1/bpg/{id}/context`
- Domain-модели (EntityType, Action, Rule) — это язык общения с LLM
- Provenance позволяет LLM объяснять, откуда взялся факт
- Confidence позволяет LLM оценивать достоверность и принимать решения

**Пример runtime-использования:**
```
LLM Agent: "Могу ли я добавить товар в корзину?"
→ Query BPG: "What actions are available for Product entity?"
→ BPG returns: Action(AddToCart) with confidence=0.9, provenance=[screenshot_123]
→ LLM validates: "Yes, based on BPG, AddToCart action exists with high confidence"
```

---

## 2. Структура репозитория

```
src/
├── domain/                    # Чистая бизнес-логика
│   ├── models/               # Pydantic-модели BPG
│   │   ├── bpg_models.py     # Nodes
│   │   ├── bpg_edges.py      # Edges
│   │   └── provenance.py     # Provenance, Confidence
│   └── interfaces/           # Абстрактные интерфейсы
│       ├── preprocessing.py
│       ├── gui_detection.py
│       ├── representation.py
│       ├── linking.py
│       └── bpg_construction.py
│
├── application/              # Use cases (оркестрация)
│   └── use_cases/
│       └── bpg_pipeline.py   # BuildBPGUseCase
│
├── infrastructure/          # Конкретные реализации
│   ├── preprocessing/       # OCR (placeholder)
│   ├── gui_detection/       # YOLO (placeholder)
│   ├── representation/      # CLIP/SentenceTransformer (placeholder)
│   ├── storage/             # Chroma vector store
│   ├── linking/             # Entity linking
│   └── bpg_construction/    # BPG assembly
│
└── api/                     # FastAPI endpoints
    ├── main.py
    ├── routes/
    │   └── bpg.py
    └── dependencies.py      # Dependency injection
```

### Сопоставление модулей pipeline с директориями

| Pipeline модуль | Domain Interface | Infrastructure Implementation | Архитектурное решение |
|----------------|------------------|------------------------------|----------------------|
| **Preprocessing** | `domain/interfaces/preprocessing.py` | `infrastructure/preprocessing/preprocessing_service.py` | Разделение позволяет менять OCR (Tesseract → PaddleOCR) |
| **GUI Detection** | `domain/interfaces/gui_detection.py` | `infrastructure/gui_detection/gui_detection_service.py` | YOLO может быть заменён без изменения use case |
| **Representation** | `domain/interfaces/representation.py` | `infrastructure/representation/representation_service.py` | CLIP/SentenceTransformer могут быть заменены |
| **Entity Linking** | `domain/interfaces/linking.py` | `infrastructure/linking/entity_linking_service.py` | Chroma используется для similarity, но может быть заменён |
| **BPG Construction** | `domain/interfaces/bpg_construction.py` | `infrastructure/bpg_construction/bpg_construction_service.py` | Граф строится из entities/actions, edges инферируются |

---

## 3. Ключевые Pydantic-модели

### Nodes (из BPG_SCHEMA.md)

#### EntityType
```python
class EntityType(BaseModel):
    id: UUID
    name: str  # "Product", "Order"
    description: Optional[str]
    confidence: Confidence
    provenance: Provenance
```

**Domain rationale:** LLM agents нуждаются в понимании "какие типы сущностей существуют в домене"

#### EntityInstance
```python
class EntityInstance(BaseModel):
    id: UUID
    entity_type_id: UUID
    attributes: Dict[str, Any]  # {"id": "123", "name": "Widget"}
    confidence: Confidence
    provenance: Provenance
```

**Domain rationale:** LLM agents рассуждают о конкретных экземплярах (Order #123)

#### GUIManifestation
```python
class GUIManifestation(BaseModel):
    id: UUID
    entity_instance_id: UUID
    screenshot_id: str
    bounding_box: Dict[str, float]
    visual_embedding: Optional[List[float]]  # CLIP
    text_embedding: Optional[List[float]]    # SentenceTransformer
    layout_features: Dict[str, Any]
```

**Domain rationale:** Одна и та же entity (Product #123) появляется по-разному на разных экранах. Cross-view linking позволяет отслеживать entity через UI states.

#### Action
```python
class Action(BaseModel):
    id: UUID
    action_type: str  # "click", "submit"
    trigger_element: Dict[str, Any]
    confidence: Confidence
    provenance: Provenance
```

**Domain rationale:** LLM agents нуждаются в понимании "какие действия возможны?" для валидации планов.

### Edges

#### CrossViewEdge
```python
class CrossViewEdge(BPGEdge):
    similarity_score: Optional[float]  # Multimodal similarity
```

**Domain rationale:** Связывает разные GUIManifestations одной EntityInstance. LLM может запросить "какие другие views показывают эту же entity?"

#### FunctionalEdge
```python
class FunctionalEdge(BPGEdge):
    action_id: UUID
```

**Domain rationale:** Представляет action → result. LLM может запросить "что происходит, когда я делаю X?"

#### TemporalEdge
```python
class TemporalEdge(BPGEdge):
    frequency: Optional[float]
    temporal_order: Optional[int]
```

**Domain rationale:** Представляет частый или причинный порядок из clickstreams. LLM может запросить "что обычно происходит после этого view?"

### Metadata

#### Provenance
```python
class Provenance(BaseModel):
    evidence_sources: List[str]  # Screenshot IDs, session IDs
    inference_method: InferenceMethod  # HEURISTIC, ML_MODEL, LLM_ASSISTED
    timestamp: datetime
    metadata: Dict[str, Any]
```

**Domain rationale:** LLM agents нуждаются в объяснении "почему это ограничение существует?" для explainability.

#### Confidence
```python
class Confidence(BaseModel):
    score: float  # [0, 1]
    method: InferenceMethod
    metadata: Dict[str, Any]
```

**Domain rationale:** LLM agents нуждаются в оценке достоверности фактов для принятия решений. Runtime validation может отклонять low-confidence edges.

---

## 4. Skeleton Pipeline

### Use Case: BuildBPGUseCase

```python
async def execute(request: BuildBPGRequest) -> BusinessProcessGraph:
    # 1. Preprocessing: Load screenshots, run OCR
    screenshots = await self.preprocessing.load_screenshots(...)
    
    # 2. GUI Detection: Detect elements, group into blocks
    blocks = await self.gui_detection.detect_gui_blocks(...)
    
    # 3. Representation: Generate multimodal embeddings
    embeddings = await self.representation.generate_embeddings(...)
    
    # 4. Entity Linking: Link entities across views
    cross_view_edges = await self.linking.link_entities(...)
    entity_instances = await self.linking.create_entity_instances(...)
    
    # 5. BPG Construction: Build graph with edges
    bpg = await self.bpg_construction.build_bpg(...)
    
    return bpg
```

### Заглушки (skeleton)

- **YOLO**: `GUIDetectionServiceImpl` возвращает mock blocks (2-3 блока на скриншот)
- **CLIP/SentenceTransformer**: `RepresentationServiceImpl` возвращает zero-embeddings (512-dim visual, 384-dim text)
- **OCR**: `PreprocessingServiceImpl` возвращает пустой OCR text
- **Entity Linking**: `EntityLinkingServiceImpl` использует простой threshold-based similarity
- **BPG Construction**: `BPGConstructionServiceImpl` создаёт placeholder edges/patterns

### Контракты (интерфейсы)

Все сервисы реализуют явные интерфейсы из `domain/interfaces/`:
- Позволяет тестировать use case с mock-реализациями
- Позволяет менять инфраструктуру без изменения domain-логики
- Обеспечивает типизацию и контракты

---

## 5. Следующие шаги

### 1. Интеграция реальных моделей

- **YOLO**: Загрузить weights, реализовать `YOLODetector.detect()`
- **CLIP**: Интегрировать для visual embeddings
- **SentenceTransformer**: Интегрировать для text embeddings
- **OCR**: Добавить Tesseract или PaddleOCR

### 2. Улучшение entity linking

- Использовать **HDBSCAN** для clustering manifestations
- Комбинировать visual + text embeddings для similarity
- Добавить temporal evidence (same entity в последовательных скриншотах)

### 3. BPG Construction логика

- Использовать **PM4Py** для temporal pattern mining из clickstreams
- **LLM-assisted inference** для EntityType naming (GPT-4 для семантического анализа)
- Rule inference из observed constraints (например, "Cart пуст → Checkout недоступен")

### 4. Storage и persistence

- Сохранять BPG в **Neo4j** или **PostgreSQL** (не только Chroma)
- Индексировать BPG для быстрого context retrieval
- Реализовать `/api/v1/bpg/{id}/context` endpoint с graph traversal

### 5. LLM Runtime интеграция

- **Context enrichment endpoint**: LLM запрашивает релевантный subgraph
- **Validation endpoint**: LLM валидирует action plan против BPG constraints
- **Explainability endpoint**: LLM объясняет, почему действие валидно/невалидно

---

## Ограничения и допущения

### Skeleton-код

- ❌ Нет реальных ML-моделей (YOLO, CLIP, OCR)
- ❌ Mock embeddings (zero-vectors)
- ❌ Placeholder entity linking (простой threshold)
- ❌ Нет persistence (BPG не сохраняется)

### Архитектурные допущения

- ✅ Chroma используется для vector storage (может быть заменён на FAISS/Pinecone)
- ✅ FastAPI — только инфраструктурный слой (не domain-логика)
- ✅ Domain-модели — Pydantic (типизация, валидация)

### Исследовательские допущения

- ✅ GUI-only evidence (нет backend/API)
- ✅ Confidence и Provenance — first-class citizens
- ✅ Читаемость > оптимизация

---

## Критерий качества архитектуры

Любое архитектурное решение должно быть объяснимо так:

> **"Почему это помогает LLM-агенту понимать бизнес-логику через GUI и использовать BPG в runtime?"**

Если ответ на этот вопрос отсутствует — решение неверно.

### Примеры правильных решений

1. **Provenance в каждой модели**: LLM может объяснить, откуда взялся факт
2. **Confidence scores**: LLM может оценить достоверность и принять решение
3. **CrossViewEdge**: LLM может отслеживать entity через UI states
4. **FunctionalEdge**: LLM может понять причинно-следственные связи
5. **TemporalEdge**: LLM может использовать workflow patterns для планирования

### Примеры неправильных решений

1. ❌ Смешивание domain-логики с FastAPI endpoints
2. ❌ Hardcoded ML-модели в domain-моделях
3. ❌ Отсутствие Provenance (нет explainability)
4. ❌ Отсутствие Confidence (нет uncertainty handling)
