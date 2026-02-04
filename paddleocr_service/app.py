"""
PaddleOCR standalone service — Linux amd64 only.
Input: image (file). Output: list of {x, y, w, h, text, confidence}.
No YOLO, Torch, Transformers. For use when main app runs with DISABLE_PADDLEOCR=1 (e.g. Mac).
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, List

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PaddleOCR Service",
    description="Dedicated PaddleOCR 2.x — image in, bbox+text out. Linux amd64 only.",
)

# Lazy init so startup doesn't crash if import fails
_ocr_engine: Any = None


def _get_ocr_engine() -> Any:
    global _ocr_engine
    if _ocr_engine is not None:
        return _ocr_engine
    try:
        from paddleocr import PaddleOCR
        _ocr_engine = PaddleOCR(use_angle_cls=False, show_log=False, use_gpu=False)
        logger.info("PaddleOCR engine initialized")
        return _ocr_engine
    except Exception as e:
        logger.error("PaddleOCR init failed: %s", e)
        raise HTTPException(status_code=503, detail="PaddleOCR not available")


class TextBoxOut(BaseModel):
    x: int
    y: int
    w: int
    h: int
    text: str
    confidence: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "paddleocr"}


@app.post("/ocr", response_model=List[TextBoxOut])
async def ocr(image: UploadFile = File(..., description="Image file")) -> List[TextBoxOut]:
    """Run PaddleOCR on image. Returns bbox + text + confidence."""
    engine = _get_ocr_engine()
    suffix = Path(image.filename or "img").suffix or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        path = Path(f.name)
    try:
        content = await image.read()
        path.write_bytes(content)
        img = np.array(Image.open(path).convert("RGB"))
        result = engine.ocr(img, cls=False)
        out: List[TextBoxOut] = []
        if result and result[0]:
            for line in result[0]:
                if not line or len(line) < 2:
                    continue
                box_points, (text, confidence) = line[0], line[1]
                x_coords = [p[0] for p in box_points]
                y_coords = [p[1] for p in box_points]
                x1, x2 = int(min(x_coords)), int(max(x_coords))
                y1, y2 = int(min(y_coords)), int(max(y_coords))
                out.append(
                    TextBoxOut(
                        x=x1,
                        y=y1,
                        w=max(1, x2 - x1),
                        h=max(1, y2 - y1),
                        text=text or "",
                        confidence=float(confidence),
                    )
                )
        return out
    except Exception as e:
        logger.exception("OCR failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if path.exists():
            path.unlink()
