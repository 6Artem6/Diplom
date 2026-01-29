# Cross-View Entity Linking: Аудит и Исправления

## 1. АУДИТ (что было сломано)

### Критические проблемы:

1. **Нет явной модели View:**
   - Использовался только `screenshot_id`
   - Не было понятия "один View = один screenshot"
   - **Исправлено:** Добавлена модель `View`

2. **GUIManifestation без view_id:**
   - Не было способа явно различить manifestations из разных views
   - **Исправлено:** Добавлено поле `view_id` в `GUIManifestation`

3. **Cross-view edges создавались без проверки view_id:**
   - Могли создаваться edges между manifestations одного view
   - **Исправлено:** Проверка `source.view_id != target.view_id`

4. **Нет разделения на within-view и cross-view:**
   - Вся логика была смешана
   - **Исправлено:** Разделение на `cluster_within_views()` и `link_cross_view()`

5. **Нет визуализации для проверки:**
   - Пользователь не мог проверить корректность cross-view linking
   - **Исправлено:** Добавлен endpoint `/bpg/{id}/debug/visualization`

---

## 2. ИСПРАВЛЕНИЯ

### 2.1. Добавлена модель View

```python
class View(BaseModel):
    id: UUID
    screenshot_id: str
    screenshot_path: str
```

**Зачем:** Явная модель для "один View = один screenshot". Критично для cross-view семантики.

### 2.2. Обновлён GUIManifestation

```python
class GUIManifestation(BaseModel):
    id: UUID
    entity_instance_id: UUID
    view_id: UUID  # НОВОЕ: критично для cross-view
    screenshot_id: str  # Legacy, для совместимости
    ...
```

**Зачем:** `view_id` позволяет явно проверять, что cross-view edges только между разными views.

### 2.3. Обновлён EntityLinkingService

**Было:**
```python
async def link_entities(...) -> List[CrossViewEdge]
```

**Стало:**
```python
async def cluster_within_views(...) -> Dict[UUID, List[List[UUID]]]
async def link_cross_view(...) -> List[CrossViewEdge]  # Только между разными view_id
async def create_entity_instances(..., within_view_clusters, ...)
```

**Зачем:** Явное разделение: сначала clustering внутри view, потом cross-view linking.

### 2.4. Добавлена визуализация

**Endpoint:** `/api/v1/bpg/{id}/debug/visualization`

**Возвращает:**
- JSON с cross-view edges
- Entity instances с view_count
- Данные для визуализации

**Зачем:** Пользователь может проверить корректность cross-view linking.

---

## 3. ОБНОВЛЁННЫЙ DATA FLOW

```
Screenshots (2+)
  ↓
[Preprocessing] → ScreenshotData[]
  ↓
[View Creation] → View[] (один View = один screenshot)
  ↓
[GUI Detection] → GUIBlocks[] (с screenshot_id)
  ↓
[Representation] → MultimodalEmbeddings[]
  ↓
[Entity Linking]
  ├─ Step 1: cluster_within_views()
  │   └─ Clustering внутри каждого View отдельно
  │   └─ Возвращает: Dict[view_id, List[clusters]]
  │
  ├─ Step 2: link_cross_view()
  │   └─ Linking ТОЛЬКО между разными view_id
  │   └─ Проверка: source.view_id != target.view_id ✅
  │   └─ Возвращает: List[CrossViewEdge]
  │
  └─ Step 3: create_entity_instances()
      └─ Объединение within-view clusters + cross-view edges
      └─ Возвращает: List[EntityInstance]
  ↓
[BPG Construction] → BusinessProcessGraph
  └─ cross_view_edges: только между разными views
  ↓
[Storage] → BPGStorage
  ↓
[Visualization] → /bpg/{id}/debug/visualization
  └─ User verification of cross-view correctness
```

---

## 4. ПРИМЕР: 2 СКРИНШОТА (list → details)

### Input:
- `screenshot1.png` — список продуктов (catalog)
- `screenshot2.png` — страница продукта (product details)

### Pipeline:

1. **Preprocessing:**
   - `ScreenshotData(screenshot_id="s1", ...)`
   - `ScreenshotData(screenshot_id="s2", ...)`

2. **View Creation:**
   - `View(id="v1", screenshot_id="s1", ...)`
   - `View(id="v2", screenshot_id="s2", ...)`

3. **GUI Detection:**
   - View 1: `GUIBlock(id="b1", screenshot_id="s1")` — product card
   - View 2: `GUIBlock(id="b2", screenshot_id="s2")` — product title
   - View 2: `GUIBlock(id="b3", screenshot_id="s2")` — product description

4. **Representation:**
   - Embeddings для всех blocks

5. **Entity Linking:**

   **Step 5a: Within-view clustering**
   ```
   View 1 (v1):
     - b1 → m1
     - Cluster: [m1]
   
   View 2 (v2):
     - b2 → m2
     - b3 → m3
     - Clusters: [m2], [m3] (или [m2, m3] если похожи)
   ```

   **Step 5b: Cross-view linking**
   ```
   m1 (view_id=v1) похож на m2 (view_id=v2)
   → Проверка: v1 != v2 ✅
   → CrossViewEdge(source=m1, target=m2)
   
   m2 и m3 в одном view (v2)
   → НЕ создаём cross-view edge (same-view)
   ```

   **Step 5c: Entity instances**
   ```
   Connected components: {m1, m2, m3}
   → EntityInstance(id="e1", attributes={"view_count": 2})
   ```

6. **BPG Construction:**
   - `BusinessProcessGraph(cross_view_edges=[CrossViewEdge(m1→m2)])`
   - Только ОДИН cross-view edge (m1→m2)

---

## 5. КАК ПОЛЬЗОВАТЕЛЬ ВИЗУАЛЬНО УБЕЖДАЕТСЯ В КОРРЕКТНОСТИ

### Сценарий проверки:

1. **Запрос визуализации:**
   ```bash
   curl "http://localhost:8001/api/v1/bpg/{id}/debug/visualization"
   ```

2. **Проверка cross-view edges:**
   ```json
   {
     "cross_view_edges": [
       {
         "source_id": "m1",
         "target_id": "m2",
         "similarity_score": 0.85,
         "confidence": 0.85
       }
     ]
   }
   ```
   - Проверка: `source.view_id != target.view_id` ✅
   - Если найдён edge с одинаковым view_id → ошибка ❌

3. **Проверка entity instances:**
   ```json
   {
     "entity_instances": [
       {
         "id": "e1",
         "view_count": 2,
         "attributes": {"component_size": 3}
       }
     ]
   }
   ```
   - `view_count > 1` → entity появляется в нескольких views ✅
   - Manifestations одного EntityInstance должны иметь одинаковый цвет

4. **Валидация:**
   - ✅ Cross-view edges только между разными screenshots
   - ✅ Same-view manifestations не связаны cross-view edge
   - ✅ Entity instances корректно группируют manifestations

---

## 6. СТРУКТУРА РЕПОЗИТОРИЯ (обновлённая)

```
src/
├── domain/
│   ├── models/
│   │   ├── view.py              # НОВЫЙ: View model
│   │   ├── bpg_models.py        # ИСПРАВЛЕНО: добавлен view_id
│   │   └── ...
│   └── interfaces/
│       └── linking.py           # ИСПРАВЛЕНО: разделение на cluster_within_views и link_cross_view
│
├── infrastructure/
│   └── linking/
│       └── entity_linking_service.py  # ИСПРАВЛЕНО: новая логика
│
├── application/
│   └── use_cases/
│       └── bpg_pipeline.py      # ИСПРАВЛЕНО: создание View, группировка по view_id
│
└── api/
    └── routes/
        └── visualization.py     # НОВЫЙ: endpoint для визуализации
```

---

## 7. КЛЮЧЕВЫЕ ИЗМЕНЕНИЯ В КОДЕ

### View Model (новое)

```python
class View(BaseModel):
    id: UUID
    screenshot_id: str
    screenshot_path: str
```

### GUIManifestation (обновлено)

```python
class GUIManifestation(BaseModel):
    view_id: UUID  # НОВОЕ: критично для cross-view
    screenshot_id: str  # Legacy
    ...
```

### EntityLinkingService (обновлено)

```python
class EntityLinkingService(ABC):
    async def cluster_within_views(...) -> Dict[UUID, List[List[UUID]]]
    async def link_cross_view(...) -> List[CrossViewEdge]  # Только между разными view_id
    async def create_entity_instances(..., within_view_clusters, ...)
```

### Pipeline (обновлено)

```python
# Step 1: Create Views
views = self._create_views(screenshots)

# Step 2: Create Manifestations with view_id
manifestations = self._create_manifestations(blocks, embeddings, views)

# Step 3: Group by view_id
manifestations_by_view = self._group_by_view(manifestations)

# Step 4: Within-view clustering
within_view_clusters = await self.linking.cluster_within_views(...)

# Step 5: Cross-view linking (только между разными view_id)
cross_view_edges = await self.linking.link_cross_view(...)

# Step 6: Create entity instances
entity_instances = await self.linking.create_entity_instances(..., within_view_clusters)
```

---

## ИТОГИ

✅ **Исправлено:**
- Явная модель View (один View = один screenshot)
- GUIManifestation имеет view_id
- Cross-view edges только между разными view_id
- Разделение: within-view clustering → cross-view linking
- Визуализация endpoint для проверки

✅ **Семантика:**
- Cross-view ≠ same-view similarity
- Явные правила и проверки
- User-facing визуализация для verification

✅ **Готово к использованию:**
- Pipeline корректно обрабатывает 2+ screenshots
- Cross-view linking проверяется на корректность
- Пользователь может визуально проверить результаты

⚠️ **Ограничения (для PoC допустимо):**
- Визуализация возвращает JSON (не изображения)
- Упрощённая кластеризация (threshold-based)
- Нет persistence View объектов

---

## Следующие шаги

1. Реализовать генерацию изображений для визуализации
2. Добавить persistence View объектов
3. Улучшить кластеризацию (HDBSCAN)
4. Добавить метрики качества cross-view linking
