FROM python:3.11-slim

WORKDIR /app

# --------------------
# System dependencies
# --------------------
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    fonts-dejavu \
    curl \
    git \
    tesseract-ocr \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# --------------------
# Environment
# --------------------
ENV TORCH_HOME=/app/models/torch
ENV TRANSFORMERS_CACHE=/app/models/transformers
ENV HF_HOME=/app/models/huggingface
ENV YOLO_CACHE_DIR=/app/models/yolo
ENV PADDLEOCR_HOME=/app/models/paddleocr
ENV PADDLE_HOME=/app/models/paddle

# offline включаем ПОТОМ
ENV HF_HUB_OFFLINE=0
ENV TRANSFORMERS_OFFLINE=0
ENV TORCH_OFFLINE=0
ENV PADDLEOCR_OFFLINE=0
ENV PADDLE_OFFLINE=0

# --------------------
# Directories
# --------------------
RUN mkdir -p \
    /app/models/torch \
    /app/models/transformers \
    /app/models/huggingface \
    /app/models/yolo \
    /app/models/paddleocr \
    /app/models/paddle \
    /app/debug

# --------------------
# Python deps
# --------------------
ARG DISABLE_PADDLEOCR=0
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# PaddleOCR only on Linux amd64; skip on Mac/arm64 (QEMU) to avoid SIGSEGV
COPY requirements-paddle.txt .
RUN if [ "$DISABLE_PADDLEOCR" != "1" ]; then pip install --no-cache-dir -r requirements-paddle.txt; fi

# Runtime: same flag so init.sh and app skip Paddle when built without it
ENV DISABLE_PADDLEOCR=${DISABLE_PADDLEOCR}

# --------------------
# Detectron2 manual install (not on PyPI)
# --------------------
RUN pip install --no-cache-dir \
    'git+https://github.com/facebookresearch/detectron2.git@v0.6'

# Переключение: REMOVE_PAGE_FRAME_REGIONS=1 — удалять верхний блок из списка (не OCR, не emit).
# Иначе — только помечать page_frame (OCR=yes, emit=no).
ENV REMOVE_PAGE_FRAME_REGIONS=1

# --------------------
# Init script
# --------------------
COPY init.sh /app/init.sh
RUN chmod +x /app/init.sh

# --------------------
# App source
# --------------------
COPY src/ ./src/

EXPOSE 8000

ENTRYPOINT ["/app/init.sh"]
