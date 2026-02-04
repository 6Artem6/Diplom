#!/usr/bin/env bash
# Минимальная сборка и запуск BPG. Тесты — вручную (см. docs/RUN_AND_TEST.md).
set -e
cd "$(dirname "$0")/.."

echo "===> Build"
docker compose build bpg_service

echo "===> Start BPG (port 8001)"
docker compose up -d bpg_service

echo "===> Wait ~90s for warmup (CLIP, YOLO, Pix2Struct, LayoutLM)..."
sleep 90

echo "===> Quick test: layout"
curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:8001/api/v1/debug/layout -F "image=@data/1.png" && echo " OK" || echo " FAIL"

echo ""
echo "BPG: http://localhost:8001/docs"
echo "Test: curl -X POST http://localhost:8001/api/v1/debug/full-pipeline -F \"image=@data/1.png\" | jq ."
