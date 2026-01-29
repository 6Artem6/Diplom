FROM python:3.12-slim

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

# offline включаем ПОТОМ
ENV HF_HUB_OFFLINE=0
ENV TRANSFORMERS_OFFLINE=0
ENV TORCH_OFFLINE=0

# --------------------
# Directories
# --------------------
RUN mkdir -p \
    /app/models/torch \
    /app/models/transformers \
    /app/models/huggingface \
    /app/models/yolo \
    /app/debug

# --------------------
# Python deps
# --------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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
