"""
Pix2Struct-based GUI analysis backend.

Uses google/pix2struct-base with Pix2StructProcessor / AutoProcessor.
Prompt asks for structured output (JSON or line-per-element); parse into GUIBlocks.
Fallback to one full-image block only when model returns empty or parsing yields 0 blocks.
"""

from pathlib import Path
from typing import List, Any
import logging
import re
import os
import json

from PIL import Image
import torch

from src.domain.interfaces.gui_detection import GUIDetectionService
from src.domain.models.gui_block import GUIBlock

logger = logging.getLogger(__name__)

BACKEND_NAME = "pix2struct"

# Prefer JSON for structure; model may not follow exactly, so we also support line-based
UI_PROMPT_JSON = (
    "Output a JSON array of UI elements. Each object: {\"type\":\"button\"|\"text\"|\"card\"|\"list\"|\"header\"|\"input\", "
    "\"label\":\"text here\", \"bbox\":[x,y,width,height]} in image pixels. Only valid JSON array, no other text."
)
UI_PROMPT_LINES = (
    "List UI elements one per line: type (button|text|card|list|header|input), label, bbox x,y,width,height. Example: button Submit 10,20,80,30"
)

LINE_PATTERN = re.compile(
    r"(?P<type>button|text|card|list|header|input|item)\s*[:\-]?\s*"
    r"\"?(?P<label>[^\"]*?)\"?\s*[,\s]+"
    r"(?:bbox\s*)?\[?(?P<x>\d+)\s*[,]\s*(?P<y>\d+)\s*[,]\s*(?P<w>\d+)\s*[,]\s*(?P<h>\d+)\]?",
    re.IGNORECASE,
)

VALID_TYPES = frozenset({"button", "text", "card", "list", "header", "input", "item"})


def _normalize_bbox(
    x: int, y: int, w: int, h: int, img_w: int, img_h: int
) -> tuple[int, int, int, int]:
    """Clip bbox to image and ensure min size 1."""
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))
    return x, y, w, h


def _raw_to_blocks_json(
    raw: str,
    screenshot_id: str,
    screenshot_path: str,
    img_w: int,
    img_h: int,
    ocr_text: str,
) -> List[GUIBlock]:
    """Parse raw model output as JSON array of {type, label, bbox}. Returns [] on failure."""
    blocks: List[GUIBlock] = []
    # Try to extract JSON array
    s = raw.strip()
    start = s.find("[")
    if start == -1:
        logger.debug("Pix2StructDetection: No '[' in output, JSON parse skipped")
        return []
    end = s.rfind("]")
    if end == -1 or end <= start:
        logger.debug("Pix2StructDetection: No ']' in output, JSON parse skipped")
        return []
    try:
        arr = json.loads(s[start : end + 1])
    except json.JSONDecodeError as e:
        logger.debug("Pix2StructDetection: JSON parse error: %s", e)
        return []
    if not isinstance(arr, list):
        return []
    for i, item in enumerate(arr):
        if not isinstance(item, dict):
            continue
        t = (item.get("type") or "").strip().lower() or "text"
        if t not in VALID_TYPES:
            t = "text"
        label = (item.get("label") or item.get("text") or "").strip()
        bbox = item.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            continue
        try:
            x, y = int(bbox[0]), int(bbox[1])
            w, h = int(bbox[2]), int(bbox[3])
        except (TypeError, ValueError):
            continue
        x, y, w, h = _normalize_bbox(x, y, w, h, img_w, img_h)
        block = GUIBlock(
            id=f"{screenshot_id}_pix2struct_{i}",
            screenshot_id=screenshot_id,
            screenshot_path=screenshot_path,
            bounding_box={"x": x, "y": y, "width": w, "height": h, "x1": x, "y1": y, "x2": x + w, "y2": y + h},
            element_types=[t],
            ocr_text=label or ocr_text,
            visual_features=[],
        )
        blocks.append(block)
    return blocks


def _raw_to_blocks_lines(
    raw: str,
    screenshot_id: str,
    screenshot_path: str,
    img_w: int,
    img_h: int,
    ocr_text: str,
) -> List[GUIBlock]:
    """Parse line-based format: type label x,y,w,h."""
    blocks: List[GUIBlock] = []
    for i, m in enumerate(LINE_PATTERN.finditer(raw)):
        try:
            x, y, w, h_ = int(m.group("x")), int(m.group("y")), int(m.group("w")), int(m.group("h"))
        except (ValueError, IndexError):
            continue
        x, y, w, h_ = _normalize_bbox(x, y, w, h_, img_w, img_h)
        el_type = (m.group("type") or "text").lower()
        if el_type not in VALID_TYPES:
            el_type = "text"
        label = (m.group("label") or "").strip()
        block = GUIBlock(
            id=f"{screenshot_id}_pix2struct_{i}",
            screenshot_id=screenshot_id,
            screenshot_path=screenshot_path,
            bounding_box={"x": x, "y": y, "width": w, "height": h_, "x1": x, "y1": y, "x2": x + w, "y2": y + h_},
            element_types=[el_type],
            ocr_text=label or ocr_text,
            visual_features=[],
        )
        blocks.append(block)
    return blocks


class Pix2StructDetectionService(GUIDetectionService):
    """
    GUI detection via Pix2Struct vision-to-structure.
    Uses Pix2StructProcessor or AutoProcessor; JSON then line-based parsing.
    Fallback block only when model output is empty or parsing yields 0 blocks.
    """

    def __init__(
        self,
        model_name: str = "google/pix2struct-base",
        cache_dir: str | None = None,
        max_new_tokens: int = 512,
    ) -> None:
        cache = cache_dir or os.getenv("TRANSFORMERS_CACHE", "/app/models/transformers")
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        logger.info("Pix2StructDetection: Loading %s from %s", model_name, cache)
        try:
            from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
            self.processor = Pix2StructProcessor.from_pretrained(model_name, cache_dir=cache)
            self.model = Pix2StructForConditionalGeneration.from_pretrained(model_name, cache_dir=cache)
        except Exception:
            from transformers import AutoProcessor, Pix2StructForConditionalGeneration
            self.processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache, trust_remote_code=True)
            self.model = Pix2StructForConditionalGeneration.from_pretrained(model_name, cache_dir=cache)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info("Pix2StructDetection: backend=%s, device=%s", BACKEND_NAME, self.device)

    def _run_inference(self, image: Image.Image, prompt: str) -> str:
        """Run model with prompt; return decoded text. Tries text/header_text then image-only."""
        try:
            try:
                inputs = self.processor(
                    text=prompt,
                    images=image,
                    return_tensors="pt",
                    add_special_tokens=False,
                )
            except TypeError:
                try:
                    inputs = self.processor(
                        images=image,
                        header_text=prompt,
                        return_tensors="pt",
                    )
                except TypeError:
                    inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            text = self.processor.batch_decode(out, skip_special_tokens=True)
            return (text[0] or "").strip()
        except Exception as e:
            logger.warning("Pix2StructDetection: Inference failed: %s", e)
            return ""

    async def detect_gui_blocks(
        self,
        screenshot_path: str,
        ocr_text: str,
    ) -> List[GUIBlock]:
        """
        Run Pix2Struct with UI prompt; parse JSON first, then line-based.
        Fallback to one full-image block only when parsing yields 0 blocks.
        """
        path = Path(screenshot_path)
        if not path.exists():
            logger.warning("Pix2StructDetection: Screenshot not found %s", screenshot_path)
            return self._fallback_block(screenshot_path, ocr_text)

        screenshot_id = path.stem
        try:
            image = Image.open(path).convert("RGB")
            w, h = image.size
        except Exception as e:
            logger.warning("Pix2StructDetection: Failed to load image %s: %s", screenshot_path, e)
            return self._fallback_block(screenshot_path, ocr_text)

        raw = self._run_inference(image, UI_PROMPT_JSON)
        if not raw:
            raw = self._run_inference(image, UI_PROMPT_LINES)

        blocks = _raw_to_blocks_json(raw, screenshot_id, screenshot_path, w, h, ocr_text)
        if not blocks:
            blocks = _raw_to_blocks_lines(raw, screenshot_id, screenshot_path, w, h, ocr_text)

        if not blocks:
            logger.warning(
                "Pix2StructDetection: No blocks parsed (backend=%s, path=%s, raw_len=%d); using fallback",
                BACKEND_NAME, screenshot_path, len(raw),
            )
            return self._fallback_block(screenshot_path, ocr_text)

        logger.info(
            "Pix2StructDetection: backend=%s, path=%s, blocks=%d",
            BACKEND_NAME, screenshot_path, len(blocks),
        )
        return blocks

    def _fallback_block(self, screenshot_path: str, ocr_text: str) -> List[GUIBlock]:
        """One full-image block only when model/parse truly yields nothing."""
        path = Path(screenshot_path)
        sid = path.stem
        try:
            im = Image.open(path)
            w, h = im.size
        except Exception:
            w, h = 800, 600
        return [
            GUIBlock(
                id=f"{sid}_pix2struct_fallback",
                screenshot_id=sid,
                screenshot_path=screenshot_path,
                bounding_box={"x": 0, "y": 0, "width": w, "height": h, "x1": 0, "y1": 0, "x2": w, "y2": h},
                element_types=["screen"],
                ocr_text=ocr_text or "",
                visual_features=[],
            )
        ]
