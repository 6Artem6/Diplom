#!/usr/bin/env bash
set -euo pipefail

echo "===> Model init started"

YOLO_PATH="/app/models/yolo/ui-elements-detection.pt"

# -----------------------
# FORCE ONLINE TEMP
# -----------------------
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export TORCH_OFFLINE=0

# -----------------------
# CLIP warmup
# -----------------------
echo "===> Loading CLIP..."
python3 - <<'PY'
from transformers import CLIPModel, CLIPProcessor

CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir="/app/models/transformers", local_files_only=False)
CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir="/app/models/transformers", local_files_only=False)

print("CLIP OK")
PY

# -----------------------
# YOLO download via HF
# -----------------------
echo "===> Ensuring YOLO UI model..."
python3 - <<PY
import os
from huggingface_hub import hf_hub_download

path = "/app/models/yolo/ui-elements-detection.pt"
os.makedirs(os.path.dirname(path), exist_ok=True)

if not os.path.exists(path):
    hf_hub_download(
        repo_id="MacPaw/yolov11l-ui-elements-detection",
        filename="ui-elements-detection.pt",
        local_dir="/app/models/yolo",
        local_files_only=False,
    )
    print("YOLO weights downloaded")
else:
    print("YOLO weights already exist")
PY

# -----------------------
# YOLO warmup
# -----------------------
echo "===> Loading YOLO..."
python3 - <<'PY'
from ultralytics import YOLO
YOLO("/app/models/yolo/ui-elements-detection.pt")
print("YOLO OK")
PY

# -----------------------
# Pix2Struct warmup
# -----------------------
echo "===> Loading Pix2Struct..."
python3 - <<'PY'
from transformers import AutoProcessor
from transformers.models.pix2struct import Pix2StructForConditionalGeneration

cache = "/app/models/transformers"
AutoProcessor.from_pretrained("google/pix2struct-base", cache_dir=cache, local_files_only=False)
Pix2StructForConditionalGeneration.from_pretrained(
    "google/pix2struct-base", cache_dir=cache, local_files_only=False
)
print("Pix2Struct OK")
PY

# -----------------------
# LayoutLMv3 warmup
# -----------------------
echo "===> Loading LayoutLMv3..."
python3 - <<'PY'
from transformers import AutoProcessor, AutoModel

cache = "/app/models/transformers"
AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", cache_dir=cache, local_files_only=False)
AutoModel.from_pretrained("microsoft/layoutlmv3-base", cache_dir=cache, local_files_only=False)
print("LayoutLMv3 OK")
PY

# -----------------------
# SWITCH TO OFFLINE
# -----------------------
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_OFFLINE=1
export YOLO_OFFLINE=1

echo "===> All models ready!"

# -----------------------
# START APP
# -----------------------
exec uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
