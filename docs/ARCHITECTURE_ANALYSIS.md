# BPG Pipeline: Architecture Analysis & Refactor Proposal

**Context:** Pipeline builds a BPG from UI screenshots. Current backends (pix2struct, layoutlmv3) yield 1–2 large blocks per screenshot, weak hierarchy, weak cross-view linking — indicating a **systemic architectural issue**, not tuning.

**Goal:** Root-cause analysis, corrected architecture, concrete refactor plan, model recommendations.

---

## 1. Current Architecture Evaluation

### 1.1 Data Flow (as implemented)

```
Screenshot paths
    ↓
[Preprocessing]  load_screenshots → ScreenshotData(screenshot_id, image_path, ocr_text, metadata)
    ↓   ocr_text is empty in current skeleton (PreprocessingServiceImpl has no OCR)
ScreenshotData per image
    ↓
[GUIDetectionService]  detect_gui_blocks(screenshot_path, ocr_text)
    ↓
    ├─ yolo_clip:    YOLO bbox → GUIBlock; ocr_text only copied into block.ocr_text (still empty)
    ├─ pix2struct:   VLM prompt → raw text → regex/JSON parse → GUIBlock; ocr_text = fallback label
    └─ layoutlmv3:   pytesseract words → lines → regions → GUIBlock; ocr_text = _line_text(line)
    ↓
all_blocks = flatten_gui_blocks(blocks)   # nested → flat for downstream
    ↓
[RepresentationService]  generate_embeddings(blocks) → CLIP crop per bbox, text_emb from block.ocr_text
    ↓
[EntityLinking]  cluster_within_views, link_cross_view → entity instances, cross_view_edges
    ↓
[BPGConstruction]  build_bpg(entity_instances, actions, …)
```

### 1.2 Responsibility Separation — Assessment

| Concern | Intended owner | Actual owner | Verdict |
|--------|----------------|--------------|---------|
| **Visual layout detection** (where elements are) | Dedicated layout detector | **Pix2Struct**: VLM output → parse. **LayoutLMv3**: OCR word positions → layout | **Wrong** — layout is inferred from text/vision-language, not from a bbox-first detector |
| **OCR** (what text is inside regions) | Preprocessing or post-detection | **Preprocessing**: empty. **LayoutLMv3**: OCR is *the* source of layout (words = regions). **Pix2Struct**: no OCR, prompt-only | **Wrong** — OCR is either unused or used to *define* layout |
| **Semantic embedding** | Representation | Representation (CLIP + text). Input = blocks from detection | **OK** — representation only consumes blocks |
| **Hierarchy construction** | After stable blocks | **LayoutLMv3**: hierarchy = lines→cards inside detection. **Pix2Struct**: flat only | **Wrong** — hierarchy is baked into “detection” instead of derived from bbox containment |
| **Cross-view linking** | Linking | Linking (Chroma, view_id, cross-view only) | **OK** — semantics are clear |

**Summary:** Layout detection, OCR, and hierarchy are **coupled and misplaced**. Layout is not produced by a dedicated bbox-first detector; it is inferred from VLMs or from OCR word positions.

### 1.3 Architectural Anti-Patterns

1. **Vision–language models as layout detectors**  
   Pix2Struct is used to *generate* bboxes via prompts. It was trained for document QA/captioning, not for regressing UI element bounds. Output is free-form text; bboxes exist only after brittle parsing. **Anti-pattern:** “Ask a VLM where things are” instead of “run a detector that outputs boxes.”

2. **Prompt-based parsing where deterministic detection is required**  
   BPG needs stable, reproducible blocks for entity linking and runtime. Pix2Struct output is non-deterministic and often unparseable; fallback is one full-screen block. **Anti-pattern:** Production layout should not depend on prompt engineering + regex.

3. **OCR responsible for discovering layout**  
   In LayoutLMv3, bboxes come from `pytesseract.image_to_data` → word positions → lines → regions. So “layout” = “where text is.” Buttons without text, icons, dividers, images are invisible. **Anti-pattern:** “Layout = text layout” instead of “layout = all visible elements, then enrich with OCR inside those elements.”

4. **Detection interface overloaded with OCR + hierarchy**  
   `GUIDetectionService.detect_gui_blocks(screenshot_path, ocr_text)` mixes:
   - who decides layout (detector vs OCR vs VLM),
   - what goes into `ocr_text` (preprocessing vs detection-internal),
   - who builds hierarchy (detection vs a later stage).  
   So the same abstraction is used for “YOLO bboxes,” “VLM + parse,” and “OCR-driven regions,” which have fundamentally different contracts.

5. **Preprocessing OCR disconnected**  
   Preprocessing can return `ocr_text` (e.g. full-page) but today it’s empty. That text is passed into detection but:
   - YOLO only copies it into blocks,
   - Pix2Struct uses it as fallback label,
   - LayoutLMv3 ignores it and runs its own OCR.  
   So there is no single, clear “OCR stage” — it’s ad hoc per backend.

---

## 2. Why Current Models Fail for UI Layout

### 2.1 Pix2Struct (prompted vision-to-text)

- **Training objective:** Image → text (captioning, document QA, figure/table parsing). The model predicts tokens, not coordinates.
- **No bbox output:** Bboxes appear only if we prompt “output type, label, bbox” and then parse text. The model was not trained to produce such formats reliably.
- **Document vs UI:** Trained on documents (PDFs, figures, web pages as *documents*). UI screens are different: many small interactive elements, clear spatial structure, and need for precise per-element boxes. Document understanding favors high-level description, not fine-grained UI segmentation.
- **Result:** Unstable, often unparseable output → 1–2 blocks or fallback. Good OCR elsewhere does not help Pix2Struct, because it does not use OCR; it only uses the prompt and the image.

**Root cause:** Using a **generative VLM** for **discriminative layout detection**. Layout detection should be “given image → list of bboxes + types,” not “given image + prompt → text → try to parse bboxes.”

### 2.2 LayoutLMv3 “as” Primary Layout Detector

- **What LayoutLMv3 is:** A document understanding model (Bboxes + text → representation). It expects **already tokenized/boxed** text (e.g. from OCR). It is not an OCR engine and not a “raw image → bbox” detector.
- **How it’s used here:** The name is reused for a **custom pipeline**: pytesseract → words+bboxes → lines → regions → GUIBlocks. So “layout” is really “OCR-driven layout.”
- **Why good OCR still gives bad layout:**
  - **OCR returns words, not elements.** Buttons, icons, images, dividers have no words → no bbox. So we get “text regions” only.
  - **Lines/regions are heuristics.** `LINE_DY`, `REGION_GAP_Y` are fixed; one set cannot fit all UIs. We get coarse blobs (e.g. one “card” = many lines), not true widget boundaries.
  - **Type is guessed from text.** `_looks_like_button(line)` uses keywords. No visual signal, so icon-only buttons are missed and many text lines are mislabeled.
- **Result:** A few large “card” blocks, weak or no hierarchy, and weak cross-view links because many UI elements are never represented as blocks.

**Root cause:** **OCR is used to define layout.** Layout should be defined by a **visual layout detector**; OCR should only **fill text inside** those boxes.

### 2.3 Good OCR ≠ Good Layout

- OCR answers: “Where are the words and what do they say?”
- Layout detection answers: “Where are the UI elements (buttons, fields, cards, lists, images) and what type are they?”
- So:
  - **OCR** → word boxes + text.
  - **Layout** → element boxes + coarse type (and optionally hierarchy).
  - **Enrichment** → “for each layout box, run OCR inside it” → element-level labels.

If layout is *derived* from word positions, then everything without text disappears, and granularity is limited by line/region heuristics. Hence: good OCR can coexist with bad layout.

---

## 3. Corrected Architecture

### 3.1 Target Pipeline (text diagram)

```
Screenshot
    ↓
[1] Visual Layout Detector (deterministic, bbox-first)
    → UI blocks: bbox + coarse type (button | text | card | list | header | input | image | …)
    → No OCR inside this step; no prompts that produce layout.
    ↓
[2] OCR inside blocks (text enrichment)
    → For each block, run OCR on crop(image, block.bbox) → block.ocr_text
    → Optional: use preprocessing OCR only for fallback or full-page context.
    ↓
[3] Embeddings (CLIP / text)
    → Unchanged: representation takes blocks (+ ocr_text) and produces embeddings.
    ↓
[4] Hierarchy (optional, from geometry)
    → Parent/child from bbox containment or tree builder; children attached to existing blocks.
    ↓
[5] Cross-view entity linking + BPG construction
    → Unchanged: linking and BPG build on (block, embedding) and view_id.
```

### 3.2 Responsibility Split

| Layer | Responsibility | Not responsible for |
|-------|----------------|---------------------|
| **Visual layout detector** | Image → list of (bbox, coarse_type). Deterministic, model-based (e.g. YOLO, DETR, UI-specific detector). | OCR, hierarchy, semantics, linking. |
| **OCR enrichment** | For each block (or full image once), run OCR → text. Fill `block.ocr_text` (or equivalent). | Deciding where blocks are; creating layout. |
| **Representation** | Blocks → visual + text embeddings, layout features. | Detecting layout; building hierarchy. |
| **Hierarchy** | After blocks exist: compute parent/child from bbox containment or tree algorithm. | Discovering bboxes; OCR. |
| **Linking / BPG** | Cross-view links, entity instances, graph. | Layout; OCR; hierarchy construction. |

### 3.3 Corrected Architecture Diagram (ASCII)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         BPG CONSTRUCTION PIPELINE                        │
└─────────────────────────────────────────────────────────────────────────┘

  Screenshot (image)
         │
         ▼
  ┌──────────────────────────────────────┐
  │  [1] Visual Layout Detector            │
  │  • bbox-first, deterministic          │
  │  • Output: List[(bbox, type)]         │
  │  • No OCR, no prompt-derived layout   │
  └──────────────────────────────────────┘
         │
         ▼  raw_blocks (bbox + coarse type)
  ┌──────────────────────────────────────┐
  │  [2] OCR Enrichment                   │
  │  • crop(image, bbox) → OCR → label    │
  │  • Fills block.ocr_text per block     │
  └──────────────────────────────────────┘
         │
         ▼  blocks (bbox + type + ocr_text)
  ┌──────────────────────────────────────┐
  │  [3] Representation                   │
  │  • CLIP(block crop), text(ocr_text)  │
  │  • Layout features from bbox         │
  └──────────────────────────────────────┘
         │
         ▼  embeddings
  ┌──────────────────────────────────────┐
  │  [4] Hierarchy (optional)            │
  │  • bbox containment → parent/child  │
  │  • Attach children to blocks         │
  └──────────────────────────────────────┘
         │
         ▼  blocks (possibly with children)
  ┌──────────────────────────────────────┐
  │  [5] Entity Linking + BPG             │
  │  • flatten for linking if needed     │
  │  • cluster_within_views              │
  │  • link_cross_view → BPG              │
  └──────────────────────────────────────┘
```

---

## 4. Model Recommendations

### 4.1 Comparison

| Option | Suitability for UI layout | Production readiness | Integration effort | Expected bbox quality |
|--------|----------------------------|------------------------|--------------------|------------------------|
| **YOLOv8 (or YOLOv11) pre-trained / fine-tuned on UI** | **High** — trained to output bbox + class. UI datasets (RICO, Enrico, WebUI, MacPaw) give button/card/list/etc. | **High** — Ultralytics, ONNX, local, batch, no prompt. | **Low** — you already have `YOLODetectorImpl`; make it the default layout source. | **High** for trained classes; consistent, debuggable. |
| **Florence-2 / OmniParser** | **Medium–High** — can do region detection + Caption/OCR; prompt-based. | **Medium** — research-grade; API/checkpoint availability and format may change. | **Medium** — new integration; output format (bbox + type) must be parsed or wrapped. | **Medium–High** if prompts and parsing are stabilized; less deterministic than YOLO. |
| **Current Pix2Struct** | **Low** — document-oriented; bbox only via prompt parse. | **Low** for layout — too brittle for “production layout.” | Already integrated. | **Low** — 1–2 blocks or fallback. |
| **Current “LayoutLMv3” (OCR→lines→regions)** | **Low** — layout = text layout; no non-text elements. | **Low** for layout — good as OCR only. | Already integrated. | **Low** — few large blocks, text-only. |

### 4.2 Recommended Choices

1. **Primary layout detector (production):**  
   **YOLO-family model trained on UI datasets** (e.g. MacPaw yolov11l-ui-elements-detection, or RICO/Enrico-style).  
   - Deterministic, bbox-first, no prompt parsing.  
   - Fits “Visual Layout Detector” role.  
   - Keep existing `GUIDetectionServiceImpl` + YOLO as the default when “layout” is required.

2. **OCR (enrichment only):**  
   **Tesseract / pytesseract** (or PaddleOCR if you need better quality) — run **inside** each block’s crop.  
   - Do not use OCR to define layout.  
   - Preprocessing can still do full-page OCR for metadata/fallback; block-level OCR stays in “OCR enrichment” step.

3. **Optional “research” layout:**  
   **Florence-2 / OmniParser** behind an optional backend, with clear contract: “output must be list of (bbox, type),” and a thin adapter that fits your `GUIBlock` contract.  
   - Evaluated separately from production path.  
   - Do not use as sole production layout source until quality and stability are proven.

4. **Pix2Struct / current LayoutLMv3:**  
   - **Pix2Struct:** Deprecate as layout detector; use only for experiments or as optional caption/description, not as source of bboxes.  
   - **LayoutLMv3 pipeline:** Reuse OCR and line/region logic **only** inside an “OCR enrichment” service (e.g. “per-block OCR” or “full-page OCR + assign words to blocks by overlap”). Do not use it to produce layout.

---

## 5. Incremental Refactor Plan

### 5.1 Principles

- Introduce a **VisualLayoutDetector** abstraction; keep **GUIBlock**, **Representation**, and **Linking** as-is where possible.
- Prefer **add new path + switch** over “rewrite everything.”
- Leave room for multiple backends (YOLO primary; others optional).

### 5.2 Step-by-Step Plan

**Phase A — Clarify layout vs OCR (no new deps)**  
1. **Rename / split interfaces (conceptually):**
   - **VisualLayoutDetector**: `(image_path) → List[RawLayoutBox]` where `RawLayoutBox = (bbox, coarse_type)`.
   - **OCREnrichment** (new, optional for now): `(image_path, List[RawLayoutBox]) → List[RawLayoutBox]` with `ocr_text` filled per box; or keep enrichment inside pipeline as a loop “for each block, OCR crop.”
2. **Implement VisualLayoutDetector for YOLO:**
   - Add `VisualLayoutDetector` interface and `YOLOLayoutDetector` that wraps existing YOLO; output `List[RawLayoutBox]`.
   - In pipeline, when backend is `yolo_clip`, call `YOLOLayoutDetector` → build `GUIBlock` from `RawLayoutBox`; then run OCR enrichment (e.g. per-block tesseract) and set `block.ocr_text`.
3. **Preprocessing:**  
   - Add real OCR (e.g. pytesseract full-page) in Preprocessing so `screenshot.ocr_text` is available. Use it for fallback labels or global context; block-level text still from “OCR enrichment” where possible.

**Phase B — Pipeline uses layout-then-OCR**  
4. **Refactor pipeline order:**
   - Step 2a: **Layout** — `VisualLayoutDetector(image) → raw_blocks` (or current `detect_gui_blocks` for YOLO path that now uses this under the hood).
   - Step 2b: **OCR enrichment** — for each block, `ocr_service.extract_text(crop(image, block.bbox))` → `block.ocr_text`.  
   - Keep flattening and the rest as today.
5. **GUIDetectionService contract:**
   - For **yolo_clip**: implement as “YOLO layout + optional in-loop OCR enrichment,” so `detect_gui_blocks` continues to return `List[GUIBlock]` with bbox + type + ocr_text.  
   - For **pix2struct** / **layoutlmv3**: mark as legacy/experimental; doc that they are not “layout detectors” and may be removed or limited to “OCR-only / caption-only” roles later.

**Phase C — Hierarchy after layout**  
6. **Hierarchy as post-step:**
   - New module or function: `build_hierarchy(blocks: List[GUIBlock]) → List[GUIBlock]` that assigns `children` from bbox containment (or simple tree builder).  
   - Call it after layout + OCR enrichment, before or after flatten for linking (e.g. flatten only for embedding/linking; keep tree for debug/visualization).

**Phase D — Optional backends**  
7. **Florence-2 / OmniParser (optional):**
   - Implement `Florence2LayoutDetector` implementing `VisualLayoutDetector` if you adopt it; plug behind a new backend name (e.g. `florence2`) and compare to YOLO in metrics/logs.  
8. **Config:**  
   - `GUI_ANALYSIS_BACKEND=yolo_clip` → primary, layout from YOLO + OCR enrichment.  
   - `pix2struct` / `layoutlmv3` → experimental or deprecated for layout; doc and logs make this explicit.

### 5.3 What to Leave Unchanged (minimize impact)

- **GUIBlock** — keep id, screenshot_id, bounding_box, element_types, ocr_text, visual_features, children.
- **RepresentationService** — input remains `List[GUIBlock]`; no change to contract.
- **EntityLinkingService** — still receives blocks + embeddings + view_id; no change.
- **BPGConstruction** — unchanged.
- **flatten_gui_blocks** — keep; use when passing blocks to representation/linking.
- **Chroma, views, cross_view_edges** — unchanged.

### 5.4 New / Changed Abstractions (summary)

```
VisualLayoutDetector (new)
  └─ detect_layout(image_path: str) -> List[RawLayoutBox]
  └─ RawLayoutBox = (bbox, coarse_type: str)

YOLOLayoutDetector implements VisualLayoutDetector  (wraps existing YOLO)

Pipeline:
  layout_boxes = layout_detector.detect_layout(path)
  blocks = [gui_block_from(b) for b in layout_boxes]
  for b in blocks: b.ocr_text = ocr_enrichment.crop_and_ocr(path, b.bounding_box)
  (optional) blocks = build_hierarchy(blocks)
  all_blocks = flatten_gui_blocks(blocks)
  … rest unchanged (representation → linking → BPG)
```

---

## 6. Code Quality & Best Practices

### 6.1 What Is Solid and Should Stay

- **Domain interfaces** — `GUIDetectionService`, `RepresentationService`, `EntityLinkingService`, `PreprocessingService` are clear and allow swapping implementations.
- **BPG pipeline orchestration** — `BuildBPGUseCase.execute()` is readable; steps (preprocess → detect → represent → link → build) are explicit.
- **Linking semantics** — Cross-view only when `view_id` differ; Chroma usage with `phase` and `view_id` is clear.
- **GUIBlock + flatten_gui_blocks** — Model and flattening are appropriate for multi-backend and nested blocks.
- **Config** — `get_gui_analysis_backend()` and env-based backend choice are simple and debuggable.
- **1:1 block–embedding correspondence** — `EmbeddedManifestation` and checks in pipeline avoid silent skew.

### 6.2 What to Simplify or Remove

- **Pix2Struct / LayoutLMv3 as layout backends** — Treat as experimental or deprecated for *layout*; do not invest in further prompt/parsing tuning for production. Either remove or clearly document as “non-production, OCR/caption only.”
- **Single “GUIDetectionService” for both bbox-first and OCR-first** — Split responsibilities: layout detector (bbox + type) vs OCR enrichment (text inside boxes). Then `GUIDetectionServiceImpl` can be “YOLO layout + OCR enrichment” and stay the main path.
- **Preprocessing OCR unused or ad hoc** — Decide: either preprocessing does full-page OCR and passes it to enrichment/fallback, or it is skipped and all OCR lives in “OCR enrichment.” Avoid multiple, inconsistent OCR sources.

### 6.3 Implicit Assumptions to Make Explicit

- **Fallback behavior:** Today Pix2Struct/LayoutLMv3 fall back to one full-screen block. Document: “fallback = no usable layout; pipeline continues with one block per view,” and that linking will be weak. Prefer failing or warning loudly when layout detector returns too few blocks (e.g. &lt; N per screen).
- **Backend contract:** Document that “layout backends” must return bboxes from a **detection/segmentation model**, not from parsing VLM output or OCR. That makes YOLO the reference implementation.
- **OCR semantics:** Document who sets `block.ocr_text` — “OCR enrichment” for production path; legacy backends may set it from internal OCR or prompts. Reduces confusion when comparing pipelines.

### 6.4 Testability and Data Flow

- **Inject VisualLayoutDetector and OCR enrichment** in the pipeline so tests can plug mocks (e.g. fixed list of `RawLayoutBox`, fixed OCR per bbox).
- **Representation and linking** already receive lists; easy to unit-test with pre-built blocks and embeddings.
- **Clear boundaries** (layout → OCR → representation → hierarchy → linking) make it easier to add integration tests per stage and to benchmark “layout only” vs “layout + OCR” vs “full pipeline.”

---

## Summary

| Issue | Root cause | Direction |
|-------|------------|-----------|
| 1–2 blocks, weak hierarchy/linking | Layout comes from VLM parsing or OCR word clustering, not from a bbox-first detector | Use a **Visual Layout Detector** (YOLO primary); OCR only to fill text in blocks |
| Pix2Struct underperforms | Model is generative, trained for docs; bboxes only via brittle prompt parsing | Do not use as production layout source; optional caption/experiments only |
| LayoutLMv3 underperforms | “Layout” = OCR-driven lines/regions; no non-text elements | Use its OCR **inside** a proper layout (e.g. per-block OCR); do not use it to define layout |
| Coupling OCR/layout/hierarchy | One interface does layout + OCR + hierarchy per backend | Split: LayoutDetector → OCR enrichment → optional Hierarchy → existing Representation/Linking |

**Concrete next step:** Implement `VisualLayoutDetector` + `YOLOLayoutDetector`, add a focused “OCR enrichment” step after layout, and route the default `yolo_clip` path through “layout first, then OCR.” Keep GUIBlock, Representation, and Linking as-is. Document pix2struct/layoutlmv3 as non-production layout backends.

---

## Appendix: First Code Increment (Done)

To anchor the refactor without changing the pipeline yet:

1. **`src/domain/interfaces/visual_layout_detector.py`**  
   - `RawLayoutBox`: TypedDict with `bbox`, `coarse_type`.  
   - `VisualLayoutDetector`: abstract `async def detect_layout(image_path: str) -> List[RawLayoutBox]`.  
   - Exported from `src/domain/interfaces/__init__.py`.

2. **Second code increment (done):**  
   - **`src/infrastructure/gui_detection/yolo_layout_detector.py`** — `YOLOLayoutDetector(VisualLayoutDetector)` wraps `YOLODetector`, maps detections to `RawLayoutBox`.  
   - **`src/infrastructure/ocr/ocr_enrichment.py`** — `enrich_blocks_with_ocr(image_path, blocks)` runs pytesseract on each block crop and sets `block.ocr_text`.  
   - **`GUIDetectionServiceImpl`** — for the injected implementation (yolo_clip path): `layout_detector.detect_layout(path)` → `_raw_boxes_to_gui_blocks(...)` → `enrich_blocks_with_ocr(path, blocks)` → fallback `ocr_text` from preprocessing for empty blocks. Contract `detect_gui_blocks(screenshot_path, ocr_text)` unchanged.

3. **Possible next steps (Phase B/C):**  
   - Preprocessing: add real full-page OCR so `screenshot.ocr_text` is non-empty for fallback.  
   - Optional: `build_hierarchy(blocks)` from bbox containment, call after OCR enrichment.  
   - Document pix2struct/layoutlmv3 as experimental/non-production in config or TESTING.md.
