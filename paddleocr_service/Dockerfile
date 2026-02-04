# PaddleOCR standalone — Linux amd64 only. No Torch/YOLO/Transformers.
# Build: docker build --platform linux/amd64 -t paddleocr-service .
# Run:   docker run --platform linux/amd64 -p 8001:8000 paddleocr-service
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    fonts-dejavu \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8000

ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
