# BPG Construction Service

Research PoC for building Business Process Graph from GUI-only data.

## Quick Start

### Using Docker

```bash
# Build and start service
docker-compose up --build bpg_service

# Service will be available at http://localhost:8001
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run service
uvicorn src.api.main:app --reload --port 8000
```

## API Endpoints

### Build BPG

```bash
curl -X POST "http://localhost:8001/api/v1/bpg/build" \
  -H "Content-Type: application/json" \
  -d '{
    "screenshot_paths": ["/app/data/screenshot1.png"],
    "clickstream_data": []
  }'
```

### Get BPG

```bash
curl "http://localhost:8001/api/v1/bpg/{bpg_id}"
```

### Get BPG Context (for LLM)

```bash
curl "http://localhost:8001/api/v1/bpg/{bpg_id}/context?query=Product&entity_type=Product&min_confidence=0.5"
```

## Architecture

See `ARCHITECTURE_AUDIT.md` for detailed architecture documentation.

## Data Flow

1. Screenshots → Preprocessing → ScreenshotData
2. ScreenshotData → GUI Detection → GUIBlocks
3. GUIBlocks → Representation → MultimodalEmbeddings
4. Embeddings → Entity Linking → EntityInstances + CrossViewEdges
5. Entities + Actions → BPG Construction → BusinessProcessGraph
6. BPG → Storage → Runtime Query (for LLM agents)
