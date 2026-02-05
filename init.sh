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
export YOLO_OFFLINE=0
export PADDLEOCR_OFFLINE=0
export PADDLE_OFFLINE=0
export FLAGS_use_avx=0
export FLAGS_enable_mkldnn=0
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1

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
# Detectron2 warmup
# -----------------------
echo "===> Loading Detectron2..."
python3 - <<'PY'
# Pillow 10+: Image.LINEAR перенесён в Image.Resampling.LINEAR — detectron2 ожидает Image.LINEAR
from PIL import Image
if not hasattr(Image, "LINEAR"):
    resampling = getattr(Image, "Resampling", None)
    Image.LINEAR = getattr(resampling, "LINEAR", Image.BILINEAR) if resampling else Image.BILINEAR

import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import model_zoo

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
if not torch.cuda.is_available():
    cfg.MODEL.DEVICE = "cpu"
predictor = DefaultPredictor(cfg)
print("Detectron2 OK")
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
# LayoutLMv3 warmup (REAL)
# -----------------------
echo "===> Loading LayoutLMv3 (token classification)..."
python3 - <<'PY'
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import torch
from PIL import Image

cache = "/app/models/transformers"

processor = LayoutLMv3Processor.from_pretrained(
    "nielsr/layoutlmv3-finetuned-funsd",
    cache_dir=cache,
    apply_ocr=False
)

model = LayoutLMv3ForTokenClassification.from_pretrained(
    "nielsr/layoutlmv3-finetuned-funsd",
    cache_dir=cache
)

# dummy image
image = Image.new("RGB", (224, 224), "white")

# Делаем 1 токен, чтобы не было mismatch
text = ["Hello"]  
boxes = [[0, 0, 100, 50]]  # ровно один bbox → один токен

encoding = processor(
    images=image,
    text=text,
    boxes=boxes,
    return_tensors="pt"
)

with torch.no_grad():
    model(**encoding)

print("LayoutLMv3 token-classification OK")
PY

# -----------------------
# PaddleOCR 2.x warmup — only when DISABLE_PADDLEOCR != 1 (Linux amd64)
# On Mac/arm64 (QEMU) Paddle segfaults; set DISABLE_PADDLEOCR=1 and use OCR service or empty OCR.
# -----------------------
if [ "${DISABLE_PADDLEOCR:-0}" != "1" ]; then
  echo "===> Loading PaddleOCR..."
  python3 - <<'PY'
from paddleocr import PaddleOCR
import numpy as np

ocr = PaddleOCR(use_angle_cls=False, show_log=False, use_gpu=False)
dummy = np.zeros((100, 300, 3), dtype=np.uint8)
ocr.ocr(dummy, cls=False)
print("PaddleOCR OK")
PY
else
  echo "===> PaddleOCR skipped (DISABLE_PADDLEOCR=1)"
fi

# -----------------------
# SWITCH TO OFFLINE
# -----------------------
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_OFFLINE=1
export YOLO_OFFLINE=1
export PADDLEOCR_OFFLINE=1
export PADDLE_OFFLINE=1

echo "===> All models ready!"

# -----------------------
# START APP
# -----------------------
RETRIES=0
MAX_RETRIES=3

while [ $RETRIES -lt $MAX_RETRIES ]; do
    echo "===> Starting Uvicorn (attempt $((RETRIES+1))/$MAX_RETRIES)..."
    exec uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
    STATUS=$?
    if [ $STATUS -eq 0 ]; then
        break
    fi
    RETRIES=$((RETRIES+1))
    echo "WARNING: Uvicorn crashed or exited (status $STATUS). Restarting in 2 seconds..."
    sleep 2
done

if [ $RETRIES -eq $MAX_RETRIES ]; then
    echo "ERROR: Uvicorn failed $MAX_RETRIES times. Exiting."
    exit 1
fi
