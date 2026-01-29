# Testing Guide: Real Cross-View Linking

## Prerequisites

1. Docker and Docker Compose installed
2. Two screenshot images under `data/`: `list.png`, `details.png` (or render from `data/list.html`, `data/details.html`)
3. **Offline mode:** YOLO and CLIP models must be present in the `ml_models_cache` volume before start. No auto-download at runtime.

## Offline models (required for container start)

The service fails at startup if models are missing. Pre-populate the cache once (with network), then run fully offline:

1. **UI YOLO** – place `ui-elements-detection.pt` in the YOLO cache:
   ```bash
   # Optional: run a one-off container with network to download into the volume
   docker run --rm -v diplom_ml_models_cache:/app/models -it python:3.13-slim bash -c "
     pip install huggingface_hub ultralytics -q &&
     python -c \"
from huggingface_hub import hf_hub_download
from pathlib import Path
Path('/app/models/yolo').mkdir(parents=True, exist_ok=True)
p = hf_hub_download(repo_id='MacPaw/yolov11l-ui-elements-detection', filename='ui-elements-detection.pt', local_dir='/app/models/yolo')
print('Downloaded:', p)
\"
   "
   ```

2. **CLIP** – place the `openai/clip-vit-base-patch32` model in the transformers cache (e.g. run once with `TRANSFORMERS_OFFLINE=0` and `TRANSFORMERS_CACHE=/app/models/transformers`, then keep the volume).

Env used by the app: `UI_YOLO_MODEL_PATH=/app/models/yolo/ui-elements-detection.pt`, `DETECTION_CONFIDENCE_THRESHOLD=0.15`, `CROSS_VIEW_SIMILARITY_THRESHOLD=0.78`.

## GUI analysis backends

Switch via `GUI_ANALYSIS_BACKEND` (env):

- **yolo_clip** (default) – YOLO UI detection + CLIP embeddings. Requires UI YOLO + CLIP in cache. Fail-fast if ≥2 views and 0 cross-view edges.
- **pix2struct** – `google/pix2struct-base` vision-to-structure; prompt-based inference, output parsed into GUIBlocks. Debug PNGs under `/app/debug/{bpg_id}/pix2struct/`. On 0 cross-view edges: WARNING only.
- **layoutlmv3** – OCR (pytesseract) → word-level bbox → GUIBlocks; CLIP still used for linking. Debug PNGs under `/app/debug/{bpg_id}/layoutlmv3/`. On 0 cross-view edges: WARNING only. Install pytesseract and tesseract for best results.

### Backend expectations (pix2struct / layoutlmv3)

- **GUIBlock:** Each block has `element_types` (button|text|card|list|header|input), `bounding_box` (x,y,width,height + x1,y1,x2,y2), `ocr_text` (label). Nested blocks use `children` (e.g. card → text + buttons).
- **Pipeline:** Roots returned by detection are flattened via `flatten_gui_blocks()` before representation and linking, so embeddings/linking see all nodes including children.
- **Overlay:** Debug PNGs draw every block (roots + children) with bbox and label `type [backend]`. Overlay should align visually with UI.
- **Fallback:** One full-screen block only when the backend truly returns nothing (Pix2Struct: 0 parsed blocks; LayoutLMv3: 0 OCR words even after psm 6 and psm 11).
- **Logs:** Backend name, block count, and parse/OCR stats (e.g. `Pix2StructDetection: backend=pix2struct, path=…, blocks=N`; `LayoutLMv3Detection: … words=N, lines=N, regions=N, blocks=N`).
- **LayoutLMv3:** pytesseract uses `--psm 6` (block of text); if no words, retries with `--psm 11` (sparse text). Words → lines → regions (cards); minimum bbox size enforced.

Linking diagnostics: `CROSS_VIEW_LOG_TOP_K` (top-K similarity values), `CROSS_VIEW_LOG_TOP_K_UNMATCHED` (top-K unmatched cross-view pairs below threshold). When no edges are created, logs include unmatched pairs and suggest checking overlay PNG or lowering the threshold.

## Quick Start

### 1. Build and Start Service

```bash
docker-compose up --build bpg_service
```

Service will be available at `http://localhost:8001`

### 2. Prepare Test Screenshots

Place your screenshots in the `data/` directory:
- `data/list.png` - List view (e.g., product list)
- `data/details.png` - Details view (e.g., product details)

### 3. Build BPG

```bash
curl -X POST "http://localhost:8001/api/v1/bpg/build" \
  -H "Content-Type: application/json" \
  -d '{
    "screenshot_paths": [
      "/app/data/list.png",
      "/app/data/details.png"
    ]
  }'
```

**Expected Response:**
```json
{
  "id": "...",
  "entity_types": [...],
  "entity_instances": [...],
  "cross_view_edges": [...],
  ...
}
```

### 4. Check Logs

**Expected Log Output (with UI YOLO + CLIP):**

```
INFO - YOLODetector: Loading UI model from /app/models/yolo/ui-elements-detection.pt (conf=0.15)
INFO - YOLODetector: UI model loaded successfully
INFO - BuildBPG: Starting pipeline with 2 screenshot(s)
INFO - GUIDetection: Detecting in /app/data/list.png (view_id=list)
INFO - YOLODetector: Found N detections (threshold=0.15) in /app/data/list.png
INFO - GUIDetection: Created N GUIBlock(s) from N detections
INFO - CLIPEncoder: Loading model ... from ... (offline)
INFO - Representation: Generating embeddings for ... blocks
INFO - EntityLinking: Top-10 cross-view similarities: [...]
INFO - EntityLinking: Created M cross-view edge(s) ... (threshold=0.78)
INFO - BuildBPG: Built BPG ... with ... entity instance(s), ... cross-view edge(s)
```

**Fail-fast:** If there are ≥2 views and 0 cross-view edges, the pipeline raises **ERROR** (HTTP 422) with diagnostics, e.g. `blocks_per_screenshot`, `similarity_threshold`.

### 5. Get BPG

```bash
curl "http://localhost:8001/api/v1/bpg/{bpg_id}"
```

### 6. Visualize BPG (JSON + PNG URLs)

```bash
curl "http://localhost:8001/api/v1/bpg/{bpg_id}/debug/visualization"
```

The response includes `visualization_files`: list of `{ "filename": "<view_id>.png", "url": "/api/v1/bpg/{bpg_id}/debug/image/<view_id>.png" }`. Open `http://localhost:8001{url}` in a browser to view each debug PNG.

**Expected Response (excerpt):**
```json
{
  "bpg_id": "...",
  "visualization_files": [
    { "filename": "...", "url": "/api/v1/bpg/.../debug/image/....png" }
  ],
  "cross_view_edges": [
    {
      "source_id": "...",
      "target_id": "...",
      "similarity_score": 0.912,
      "source_view_id": "...",
      "target_view_id": "...",
      "validation": "✅ Different views"
    }
  ],
  "entity_instances": [
    {
      "id": "...",
      "view_count": 2,
      "is_cross_view": true,
      "color": "rgb(255, 128, 0)"
    }
  ],
  "summary": {
    "total_cross_view_edges": 2,
    "cross_view_entities": 2,
    "validation_passed": true
  }
}
```

## Success Criteria

✅ **For two screenshots of different views:**

1. `entity_instances_count >= 1`
2. `cross_view_edges_count >= 1`
3. All cross-view edges have `validation: "✅ Different views"`
4. At least one entity instance has `view_count >= 2`
5. Logs show CLIP similarity scores >= 0.85

## Troubleshooting

### No Cross-View Matches Found

**Possible causes:**
1. Screenshots don't contain similar entities
2. Similarity threshold too high (default: 0.85)
3. CLIP embeddings not matching correctly
4. YOLO not detecting elements correctly

**Solutions:**
- Check logs for YOLO detections
- Verify screenshots contain similar UI elements
- Lower similarity threshold (modify `similarity_threshold` in `entity_linking_service.py`)
- Check CLIP embeddings are being generated

### YOLO Model Not Loading

**Error:**
```
ERROR - YOLODetector: Failed to load model: ...
```

**Solution:**
- Model will be downloaded automatically on first run
- Check internet connection
- Verify disk space

### CLIP Model Not Loading

**Error:**
```
ERROR - CLIPEncoder: Failed to load model: ...
```

**Solution:**
- Model will be downloaded automatically on first run
- Check internet connection
- Verify disk space (~500MB for ViT-B/32)

### Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
- Models run on CPU by default
- If using GPU, reduce batch size or use CPU

## Performance Notes

- **YOLO inference:** ~100-200ms per screenshot (CPU)
- **CLIP inference:** ~50-100ms per crop (CPU)
- **Total pipeline:** ~2-5 seconds for 2 screenshots

## Next Steps

1. Implement full PNG visualization with PIL
2. Add support for more GUI element classes
3. Optimize CLIP inference (batch processing)
4. Add GPU support for faster inference
