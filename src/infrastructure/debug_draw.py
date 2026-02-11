"""
Отрисовка для дебаг-визуализаций: линии и текст видны и на светлом, и на тёмном фоне.

Идея: двойная обводка (сначала светлая, затем тёмная) и подпись с контуром
(белый текст с чёрным контуром или наоборот).
BGR для cv2.
"""

from __future__ import annotations

from typing import Any, Tuple

# BGR: чёрный и белый для контраста на любом фоне
DEBUG_BGR_BLACK = (0, 0, 0)
DEBUG_BGR_WHITE = (255, 255, 255)


def rectangle_visible(
    img: Any,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color_bgr: Tuple[int, int, int],
    thickness: int = 2,
) -> None:
    """Рисует прямоугольник с двойной обводкой: виден и на светлом, и на тёмном фоне."""
    import cv2
    # Сначала более толстая светлая обводка (гало), затем основная цветная
    halo = thickness + 2
    cv2.rectangle(img, pt1, pt2, DEBUG_BGR_WHITE, halo)
    cv2.rectangle(img, pt1, pt2, color_bgr, thickness)


def putText_visible(
    img: Any,
    text: str,
    org: Tuple[int, int],
    font: int,
    font_scale: float,
    text_color_bgr: Tuple[int, int, int],
    outline_color_bgr: Tuple[int, int, int],
    thickness: int = 1,
) -> None:
    """Рисует подпись с контуром: текст читаем и на светлом, и на тёмном фоне."""
    import cv2
    x, y = org
    # Контур: смещения по 8 направлениям (или 4 для скорости)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            cv2.putText(img, text, (x + dx, y + dy), font, font_scale, outline_color_bgr, thickness + 1)
    cv2.putText(img, text, (x, y), font, font_scale, text_color_bgr, thickness)


def line_visible(
    img: Any,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color_bgr: Tuple[int, int, int],
    thickness: int = 1,
) -> None:
    """Рисует линию с двойной обводкой: видна и на светлом, и на тёмном фоне."""
    import cv2
    halo = thickness + 2
    cv2.line(img, pt1, pt2, DEBUG_BGR_WHITE, halo)
    cv2.line(img, pt1, pt2, color_bgr, thickness)


# --- PIL (RGB) helpers для layout_debug / debug/services ---
DEBUG_RGB_BLACK = (0, 0, 0)
DEBUG_RGB_WHITE = (255, 255, 255)


def pil_rectangle_visible(
    draw: Any,
    xy: Tuple[int, int, int, int],
    outline_rgb: Tuple[int, int, int],
    width: int = 2,
) -> None:
    """PIL: прямоугольник с двойной обводкой (светлая гало + основная)."""
    x1, y1, x2, y2 = xy
    draw.rectangle([x1, y1, x2, y2], outline=DEBUG_RGB_WHITE, width=width + 2)
    draw.rectangle([x1, y1, x2, y2], outline=outline_rgb, width=width)


def pil_text_visible(
    draw: Any,
    xy: Tuple[int, int],
    text: str,
    font: Any,
    fill_rgb: Tuple[int, int, int] = DEBUG_RGB_WHITE,
    outline_rgb: Tuple[int, int, int] = DEBUG_RGB_BLACK,
) -> None:
    """PIL: подпись с контуром (читаема на любом фоне)."""
    x, y = xy
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            draw.text((x + dx, y + dy), text, font=font, fill=outline_rgb)
    draw.text((x, y), text, font=font, fill=fill_rgb)
