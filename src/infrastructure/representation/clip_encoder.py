"""
CLIP Encoder Implementation

Uses transformers CLIP (ViT-B/32) for visual embeddings.
Loads ONLY from local cache_dir; no HuggingFace API at runtime.
"""

from typing import List
import logging
import os
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from .embedding_service import EmbeddingService
from src.domain.interfaces.gui_detection import GUIBlock

logger = logging.getLogger(__name__)


def _default_clip_cache_dir() -> str:
    return os.getenv("TRANSFORMERS_CACHE", os.getenv("HF_HOME", "/app/models/transformers"))


class CLIPEncoder(EmbeddingService):
    """
    CLIP encoder: one instance, bbox-crop embeddings, L2-normalized.

    Architecture rationale:
    - Single instance (singleton) — no repeated init
    - Loads only from cache_dir; local_files_only=True, no HF fallback
    - If model missing → clear error at init, no auto-download
    - Embedding only from bbox crops; L2-normalization for cosine similarity
    """

    def __init__(
        self,
        model_name: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        """
        Initialize CLIP encoder from local cache.

        Args:
            model_name: HuggingFace model id (default from CLIP_MODEL_NAME or openai/clip-vit-base-patch32)
            cache_dir: Dir where model is stored (default from TRANSFORMERS_CACHE or /app/models/transformers)
        """
        self.model_name = model_name or os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")
        self.cache_dir = Path(cache_dir or _default_clip_cache_dir())
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info("CLIPEncoder: Loading model %s from %s (offline)", self.model_name, self.cache_dir)
        try:
            self.model = CLIPModel.from_pretrained(
                self.model_name,
                cache_dir=str(self.cache_dir),
                local_files_only=True,
            )
            self.processor = CLIPProcessor.from_pretrained(
                self.model_name,
                cache_dir=str(self.cache_dir),
                local_files_only=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"CLIP model not found in {self.cache_dir}. "
                "Add model for offline use (e.g. run once with local_files_only=False and cache_dir set). "
                f"Original error: {e}"
            ) from e

        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info("CLIPEncoder: Model loaded on %s", self.device)

    async def embed_visual(self, block: GUIBlock) -> List[float]:
        """
        Generate visual embedding for GUI block (bbox crop only, L2-normalized).
        """
        path = block.screenshot_path or f"/app/data/{block.screenshot_id}.png"
        screenshot_path = Path(path)
        if not screenshot_path.exists():
            logger.warning("CLIPEncoder: Screenshot not found %s", block.screenshot_id)
            return [0.0] * 512

        try:
            # --- Crop bbox ---
            image = Image.open(screenshot_path).convert("RGB")
            w, h = image.size
            bbox = block.bounding_box
            if "x1" in bbox:
                x1 = max(0, min(int(bbox["x1"]), w))
                y1 = max(0, min(int(bbox["y1"]), h))
                x2 = max(0, min(int(bbox["x2"]), w))
                y2 = max(0, min(int(bbox["y2"]), h))
            else:
                x = int(bbox.get("x", 0))
                y = int(bbox.get("y", 0))
                x1 = max(0, min(x, w))
                y1 = max(0, min(y, h))
                x2 = max(0, min(x + int(bbox.get("width", 0)), w))
                y2 = max(0, min(y + int(bbox.get("height", 0)), h))

            crop = image.crop((x1, y1, x2, y2))

            # --- Processor ---
            inputs = self.processor(images=crop, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # --- Forward ---
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)

                # --- Обрабатываем все возможные варианты ---
                if isinstance(outputs, torch.Tensor):
                    feats = outputs
                elif hasattr(outputs, "image_embeds"):
                    feats = outputs.image_embeds
                elif hasattr(outputs, "pooler_output"):
                    # fallback на pooler_output, если image_embeds нет
                    feats = outputs.pooler_output
                else:
                    raise TypeError(
                        f"CLIPEncoder: Unexpected output type from get_image_features: {type(outputs)}"
                    )

                # L2-нормализация
                feats = feats / feats.norm(dim=-1, keepdim=True)
                return feats[0].cpu().tolist()

        except Exception as e:
            logger.error("CLIPEncoder: Failed to embed block %s: %s", block.id, e)
            return [0.0] * 512

    async def embed_text(self, text: str) -> List[float]:
        """Placeholder: text embeddings not used for cross-view matching."""
        return [0.0] * 384
