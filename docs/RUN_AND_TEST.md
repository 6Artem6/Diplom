# Сборка и тест (минимум)

Из **корня репозитория**.

## 1. Сборка

```bash
docker compose build
```

Первый раз может занять 5–15 минут (загрузка образов и моделей).

## 2. Запуск

```bash
docker compose up -d bpg_service
```

BPG-сервис поднимется на **http://localhost:8001**.  
После старта подожди 1–2 минуты (warmup: CLIP, YOLO, Pix2Struct, LayoutLM).

Опционально (если нужен OCR и есть Linux amd64 или готовность к возможному SIGSEGV под QEMU):

```bash
docker compose up -d bpg_service paddleocr_service
```

## 3. Проверка

**Layout (без OCR):**
```bash
curl -s -X POST http://localhost:8001/api/v1/debug/layout -F "image=@data/1.png" | jq .
```

**Full pipeline (layout + text detect + OCR → gui_blocks):**
```bash
curl -s -X POST http://localhost:8001/api/v1/debug/full-pipeline -F "image=@data/1.png" | jq .
```

**Документация API:** http://localhost:8001/docs

## 4. Логи

```bash
docker compose logs -f bpg_service
```

Остановка: `docker compose down`
