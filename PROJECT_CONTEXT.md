## 1. Описание проекта (Project Description)

**Название (условное):**
**GUI-only Business Process Graph for Domain-aware LLM Agents**

**Краткое описание:**
Проект — исследовательский PoC/прототип (магистерская работа), цель которого — сделать LLM-агентов **осознанными в бизнес-логике информационных систем**, не через код/документацию, а **исключительно через GUI**.

Система автоматически анализирует:

* скриншоты интерфейсов,
* clickstream / session traces,

и строит **Business Process Graph (BPG)** — явную графовую модель доменной логики:

* сущности,
* действия,
* workflow-паттерны,
* ограничения (preconditions, postconditions),
* роли UI-элементов.

BPG используется **в runtime** для:

* enrichment промптов LLM,
* валидации планов действий,
* объяснимости,
* recovery / replanning.

**Ключевой принцип:**

> explicit domain knowledge (BPG) + adaptive LLM reasoning → robust, explainable automation

---

### Основные сущности системы

**1. Data layer**

* Скриншоты GUI
* OCR-тексты
* Clickstreams (последовательности действий)

**2. Processing pipeline**

* GUI element detection & grouping
* Multimodal embeddings (visual + text + layout)
* Cross-view entity linking (одна сущность — разные экраны)
* Temporal / functional edge induction
* LLM-assisted semantic confirmation

**3. Knowledge representation**

* Business Process Graph (Graph DB)

  * Nodes: EntityType, EntityInstance, GUIManifestation, Action, PatternNode, Rule
  * Edges: cross_view, relational, functional, temporal, conditional, role

**4. Runtime usage**

* Context selection (minimal relevant BPG subgraph)
* Prompt enrichment
* Constraint checking
* Explainability (why this step is valid)

---

### Технологический стек (ориентир, не догма)

* **Python** — основной язык PoC
* **OpenCV / Layout-based heuristics** — GUI segmentation
* **OCR**: Tesseract / PaddleOCR
* **Embeddings**: CLIP, SentenceTransformers
* **Clustering**: HDBSCAN
* **Graph DB**: Neo4j (или PostgreSQL + pgvector)
* **LLM**: GPT-4V / Claude-3 / локальная LLM (абстрагировано)
* **Process mining**: PM4Py (для clickstream patterns)

---

### Ограничения и принципы

* ❌ Нет доступа к backend / API / DOM / исходному коду
* ❌ Нет ручного описания бизнес-логики
* ✅ Только GUI + поведение пользователя
* ✅ Все выводы должны иметь provenance + confidence
* ✅ Архитектура модульная, исследовательская

---

### Цель текущего репозитория

* Реализовать **минимально работающий end-to-end pipeline**:

  1. Screenshots → GUI blocks
  2. GUI blocks → Cross-view entities
  3. Entities + actions → BPG
  4. BPG → prompt context

* Сделать код **читаемым, расширяемым, пригодным для экспериментов**, а не production.

> Это исследовательская система, а не RPA-фреймворк.
