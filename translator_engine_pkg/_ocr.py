"""
_ocr.py — Phát hiện bong bóng hội thoại + tiện ích ngôn ngữ.
OCR đọc chữ nay dùng VLM (xem _vlm_ocr.py) — file này chỉ còn phần
detection cổ điển (contour-based bubble finder) và language helpers,
dùng chung cho cả bước OCR lẫn lọc kết quả theo ngôn ngữ nguồn.
"""

import re

from ._common_utils import (
    ZH_RE,
    JA_HIRAGANA_RE,
    JA_KATAKANA_RE,
    JA_KANJI_RE,
    JA_RE,
    EN_WORD_RE,
    contains_chinese,
    contains_japanese,
    contains_cjk,
)

_EN_RE = re.compile(r"[a-zA-Z]{2,}")

# Re-export for backward compatibility (used by __init__.py exports)
_ZH_RE = ZH_RE
_JA_RE = JA_RE


def _find_speech_bubbles(img) -> list[tuple]:
    """
    Phát hiện vùng bong bóng hội thoại trong ảnh manga/manhwa.
    Returns: list of (x1, y1, x2, y2)
    """
    import cv2

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_area = h * w
    ksize  = max(7, w // 80)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    found: list[tuple] = []

    for thr in (190, 160):
        _, thresh = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (img_area * 0.002 < area < img_area * 0.35):
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            if max(bw, bh) / max(min(bw, bh), 1) > 6:
                continue
            pad = max(4, int(min(bw, bh) * 0.08))
            candidate = (
                max(0, x - pad), max(0, y - pad),
                min(w, x + bw + pad), min(h, y + bh + pad),
            )
            if not any(_iou_rect(candidate, b) > 0.7 for b in found):
                found.append(candidate)

    for thr in (60, 80):
        _, thresh = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY_INV)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (img_area * 0.003 < area < img_area * 0.20):
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            if max(bw, bh) / max(min(bw, bh), 1) > 5:
                continue
            pad = max(4, int(min(bw, bh) * 0.05))
            candidate = (
                max(0, x - pad), max(0, y - pad),
                min(w, x + bw + pad), min(h, y + bh + pad),
            )
            if not any(_iou_rect(candidate, b) > 0.7 for b in found):
                found.append(candidate)

    return found


def _rect_of_bbox(bbox) -> tuple:
    """4-point OCR bbox → (x1, y1, x2, y2) float."""
    import numpy as np
    pts = np.array(bbox, dtype=np.float32)
    return (float(pts[:, 0].min()), float(pts[:, 1].min()),
            float(pts[:, 0].max()), float(pts[:, 1].max()))


def _iou_rect(a: tuple, b: tuple) -> float:
    """Intersection-over-Union của 2 rect (x1,y1,x2,y2)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _bubble_coverage(ocr: tuple, bubble: tuple) -> float:
    """Fraction of the OCR rect covered by the bubble rect."""
    ax1, ay1, ax2, ay2 = ocr
    bx1, by1, bx2, by2 = bubble
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    ocr_area = max(1, (ax2 - ax1) * (ay2 - ay1))
    return inter / ocr_area


def has_chinese(text: str) -> bool:
    return contains_chinese(text)


def has_japanese(text: str) -> bool:
    return contains_japanese(text)


def has_english(text: str) -> bool:
    return bool(_EN_RE.search(text))
