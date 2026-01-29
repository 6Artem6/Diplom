"""
Debug overlay for alternative GUI backends.

Draws bbox + type + source (pix2struct/layoutlmv3) on screenshot
and saves to /app/debug/{bpg_id}/{backend}/.
"""

from pathlib import Path
from uuid import UUID
import logging
from typing import List

from PIL import Image, ImageDraw, ImageFont

from src.domain.models.gui_block import GUIBlock, flatten_gui_blocks

logger = logging.getLogger(__name__)


def save_backend_debug_pngs(
    bpg_id: UUID,
    backend: str,
    screenshot_path: str,
    blocks: List[GUIBlock],
    base_dir: str = "/app/debug",
) -> Path | None:
    """
    Draw blocks on screenshot and save PNG with type and source labels.

    Args:
        bpg_id: BPG identifier
        backend: 'pix2struct' or 'layoutlmv3'
        screenshot_path: Path to source image
        blocks: Blocks for this screenshot (same screenshot_id)
        base_dir: Debug root (default /app/debug)

    Returns:
        Path to saved PNG, or None on failure
    """
    path = Path(screenshot_path)
    if not path.exists():
        logger.warning("debug_overlay: Screenshot not found %s", screenshot_path)
        return None
    out_dir = Path(base_dir) / str(bpg_id) / backend
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = path.stem
    out_path = out_dir / f"{stem}.png"
    to_draw = flatten_gui_blocks(blocks)

    try:
        image = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12
            )
        except Exception:
            font = ImageFont.load_default()

        for i, block in enumerate(to_draw):
            bbox = block.bounding_box
            x1 = bbox.get("x1", bbox.get("x", 0))
            y1 = bbox.get("y1", bbox.get("y", 0))
            x2 = bbox.get("x2", x1 + bbox.get("width", 0))
            y2 = bbox.get("y2", y1 + bbox.get("height", 0))
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            color = (0, 128, 255) if backend == "pix2struct" else (128, 0, 255)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            type_str = block.element_types[0] if block.element_types else "unknown"
            label = f"{type_str} [{backend}]"
            draw.text((x1, max(0, y1 - 14)), label, fill=color, font=font)

        image.save(out_path, "PNG")
        logger.info(
            "debug_overlay: Saved %s overlay for %s (%d blocks) -> %s",
            backend,
            stem,
            len(to_draw),
            out_path,
        )
        return out_path
    except Exception as e:
        logger.warning("debug_overlay: Failed to save %s overlay: %s", backend, e)
        return None
