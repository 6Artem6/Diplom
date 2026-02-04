# PaddleOCR: разделение local-dev и prod

PaddlePaddle/PaddleOCR под QEMU (Mac arm64 → linux/amd64) даёт SIGSEGV. Это ограничение платформы, не баг кода.

## Режимы работы

| Режим        | Платформа     | DISABLE_PADDLEOCR | Paddle в образе | OCR в рантайме                    |
|-------------|---------------|-------------------|-----------------|-----------------------------------|
| **local-dev** | Mac (arm64)   | `1`               | не ставится     | пустой результат или remote URL   |
| **prod/ci**   | Linux amd64   | `0` (по умолчанию) | ставится        | in-process PaddleOCR               |

## Сборка и запуск

**Mac (без PaddleOCR):**
```bash
docker build -f src/Dockerfile --build-arg DISABLE_PADDLEOCR=1 -t bpg-app:mac .
docker run -e DISABLE_PADDLEOCR=1 -p 8000:8000 bpg-app:mac
```
В образе нет paddle; `init.sh` не делает Paddle warmup; приложение не импортирует paddleocr. OCR возвращает `[]`, в логах — «PaddleOCR skipped (DISABLE_PADDLEOCR=1)».

**Linux amd64 (с PaddleOCR):**
```bash
docker build -f src/Dockerfile -t bpg-app:linux .
docker run -p 8000:8000 bpg-app:linux
```
Paddle ставится, warmup выполняется, OCR работает в процессе.

## Standalone PaddleOCR (Linux amd64)

Отдельный сервис только для OCR: без YOLO, Torch, Transformers.

- **Каталог:** `paddleocr_service/`
- **Сборка:** `docker build --platform linux/amd64 -t paddleocr-service ./paddleocr_service`
- **Запуск:** `docker run --platform linux/amd64 -p 8001:8000 paddleocr-service`
- **API:** `POST /ocr` — тело: multipart, поле `image` (файл). Ответ: JSON `[{x, y, w, h, text, confidence}, ...]`

На Mac основной контейнер можно запускать с `DISABLE_PADDLEOCR=1` и при необходимости вызывать OCR через standalone (на реальной Linux amd64 машине или в CI):
```bash
docker run -e DISABLE_PADDLEOCR=1 -e PADDLE_OCR_SERVICE_URL=http://host.docker.internal:8001 -p 8000:8000 bpg-app:mac
```

## Версии (зафиксированы)

- `numpy>=1.24,<2`
- `opencv-python==4.6.0.66`
- `paddlepaddle==2.6.2`
- `paddleocr==2.7.3`
- Torch/ultralytics — из основного `requirements.txt`, совместимые между собой

Paddle-зависимости вынесены в `requirements-paddle.txt`; в основном образе они ставятся только при `DISABLE_PADDLEOCR != 1`.

## Graceful fallback в основном сервисе

- При `DISABLE_PADDLEOCR=1` и без `PADDLE_OCR_SERVICE_URL`: OCR не вызывается, возвращается `[]`, в лог пишется, что OCR пропущен осознанно.
- При `DISABLE_PADDLEOCR=1` и заданном `PADDLE_OCR_SERVICE_URL`: запрос уходит в standalone; при таймауте/ошибке — `[]` и предупреждение в лог, пайплайн не падает.
- При сбое in-process Paddle (ImportError/Exception): возврат `[]`, логирование, без падения.

Никаких хаков под SIGSEGV, QEMU или сборку Paddle из исходников не используется.
