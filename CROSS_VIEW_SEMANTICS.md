# Cross-View Entity Linking: Семантика и Визуализация

## Критическая семантика

### Cross-view ≠ Same-view similarity

**Cross-view linking** означает связывание одной и той же entity **между разными views (скриншотами)**.

**Same-view similarity** — это кластеризация похожих manifestations **внутри одного view**. Это НЕ cross-view linking.

### Правила

1. **View = Screenshot**
   - Один View = один screenshot
   - `View.id` уникален для каждого screenshot

2. **GUIManifestation.view_id**
   - Каждый GUIManifestation имеет `view_id`
   - Manifestations с одинаковым `view_id` находятся в одном screenshot

3. **CrossViewEdge ограничения**
   - CrossViewEdge создаётся **ТОЛЬКО** если `source.view_id != target.view_id`
   - Если `view_id` одинаковый → это same-view similarity, не cross-view

4. **Pipeline порядок**
   - Сначала: clustering внутри каждого view отдельно
   - Потом: cross-view linking между разными views
   - Наконец: создание EntityInstances из clusters + cross-view edges

---

## Пример: 2 скриншота (list → details)

### Input:
- `screenshot1.png` — список продуктов (catalog view)
- `screenshot2.png` — страница продукта (product details view)

### Pipeline:

1. **Preprocessing:**
   - `ScreenshotData(screenshot_id="s1", ...)`
   - `ScreenshotData(screenshot_id="s2", ...)`

2. **View Creation:**
   - `View(id="v1", screenshot_id="s1", screenshot_path="screenshot1.png")`
   - `View(id="v2", screenshot_id="s2", screenshot_path="screenshot2.png")`

3. **GUI Detection:**
   - View 1: `GUIBlock(id="b1", screenshot_id="s1", ...)` — product card
   - View 2: `GUIBlock(id="b2", screenshot_id="s2", ...)` — product title
   - View 2: `GUIBlock(id="b3", screenshot_id="s2", ...)` — product description

4. **Representation:**
   - Embeddings для всех blocks

5. **Entity Linking:**

   **Step 5a: Within-view clustering**
   - View 1: `[b1]` → cluster `[m1]` (один manifestation)
   - View 2: `[b2, b3]` → cluster `[m2, m3]` (похожие блоки в одном view)

   **Step 5b: Cross-view linking**
   - `m1` (view 1) похож на `m2` (view 2) → `CrossViewEdge(source=m1, target=m2)`
   - Проверка: `m1.view_id != m2.view_id` ✅ (cross-view)
   - `m2` и `m3` в одном view → НЕ создаём cross-view edge (same-view)

   **Step 5c: Entity instances**
   - Connected components: `{m1, m2, m3}` → один `EntityInstance(id="e1")`
   - `e1.attributes.view_count = 2` (появляется в 2 views)

6. **BPG Construction:**
   - `BusinessProcessGraph(cross_view_edges=[CrossViewEdge(m1→m2)])`
   - Только ОДИН cross-view edge (m1→m2), не m2→m3 (same-view)

---

## Визуализация для проверки

### Endpoint: `/api/v1/bpg/{id}/debug/visualization`

**Возвращает:**
- JSON с данными для визуализации
- Cross-view edges с source/target view_id
- Entity instances с view_count

**Визуализация должна показывать:**

1. **Bounding boxes** на каждом screenshot:
   - Каждый GUIManifestation обведён прямоугольником
   - Подпись: `entity_instance_id`

2. **Цветовая кодировка:**
   - Manifestations одного EntityInstance — один цвет
   - Разные EntityInstances — разные цвета

3. **Cross-view связи:**
   - Линии между screenshots (если отображаются рядом)
   - Или список: "Entity e1: View 1 (m1) ↔ View 2 (m2)"

4. **Проверка корректности:**
   - ✅ Cross-view edge только между разными view_id
   - ✅ Same-view manifestations не связаны cross-view edge
   - ✅ Entity instances имеют правильный view_count

---

## Обновлённый Data Flow

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
  │
  ├─ Step 2: link_cross_view()
  │   └─ Linking ТОЛЬКО между разными view_id
  │   └─ Проверка: source.view_id != target.view_id
  │
  └─ Step 3: create_entity_instances()
      └─ Объединение clusters + cross-view edges
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

## Как пользователь визуально убеждается в корректности

### Сценарий проверки:

1. **Запрос визуализации:**
   ```bash
   curl "http://localhost:8001/api/v1/bpg/{id}/debug/visualization"
   ```

2. **Проверка cross-view edges:**
   - Список всех cross-view edges
   - Для каждого edge: `source.view_id != target.view_id` ✅
   - Если найдён edge с одинаковым view_id → ошибка ❌

3. **Проверка entity instances:**
   - Каждый EntityInstance имеет `view_count >= 1`
   - Если `view_count > 1` → entity появляется в нескольких views ✅
   - Manifestations одного EntityInstance имеют одинаковый цвет

4. **Визуальная проверка (если реализована генерация изображений):**
   - Screenshot 1: bounding boxes с entity_instance_id
   - Screenshot 2: bounding boxes с entity_instance_id
   - Линии между screenshots показывают cross-view связи
   - Цветовая кодировка: один цвет = один EntityInstance

5. **Валидация:**
   - ✅ Cross-view edges только между разными screenshots
   - ✅ Same-view manifestations не связаны cross-view edge
   - ✅ Entity instances корректно группируют manifestations

---

## Итоги

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
