"""
_utils.py — Tiện ích dùng chung: watermark detection, bbox/rect helpers, LAMA check.
"""

import re

_URL_LIKE_RE = re.compile(
    r"\b(?:https?://)?(?:www\.)?[A-Za-z0-9\-_.]+\.(?:com|net|org|info|xyz|top|site|online|tv|cc)\b",
    re.IGNORECASE,
)

# ── Lama availability (lazy, singleton) ───────────────────────────────────────
_lama_available = None


def check_lama_available() -> bool:
    """Kiểm tra simple-lama-inpainting đã cài và có thể import."""
    global _lama_available
    if _lama_available is not None:
        return _lama_available
    try:
        import simple_lama_inpainting  # noqa: F401
        _lama_available = True
    except Exception:
        _lama_available = False
    return _lama_available


def _get_lama():
    import simple_lama_inpainting
    return simple_lama_inpainting.SimpleLama()


# ── Watermark helpers ─────────────────────────────────────────────────────────

def _looks_like_watermark(text: str) -> bool:
    """Kiểm tra watermark - chỉ dùng dấu hiệu rõ ràng để tránh bắt nhầm ô thoại."""
    if not text or len(text.strip()) < 3:
        return False

    # Xóa khoảng trắng và ký tự không phải ASCII thông thường để kiểm tra domain
    compact = re.sub(r'\s+', '', text.lower())
    compact_ascii = re.sub(r'[^a-z0-9\-_.:/]', '', compact)

    # Chỉ match URL/domain rõ ràng: phải có .com/.net/... hoặc chứa "acg" cụm liền mạch
    if ".com" in compact_ascii or compact_ascii.endswith("com"):
        return True
    if _URL_LIKE_RE.search(text):
        return True
    # "acg" chỉ match khi đứng độc lập hoặc là toàn bộ text, tránh bắt nhầm chữ thường
    if re.search(r'\bacg\b', compact_ascii) or compact_ascii == 'acg':
        return True

    return False


# ── Bbox / rect helpers ───────────────────────────────────────────────────────

def _bbox_xyxy(bbox) -> tuple[int, int, int, int]:
    """4-point polygon bbox → (x1, y1, x2, y2) int."""
    import numpy as np
    pts = np.array(bbox, dtype=np.int32)
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    return int(x1), int(y1), int(x2), int(y2)


def _union_bboxes(bboxes: list[list[list[int]]]) -> list[list[int]]:
    if not bboxes:
        return [[0, 0], [0, 0], [0, 0], [0, 0]]
    xs = []
    ys = []
    for bbox in bboxes:
        x1, y1, x2, y2 = _bbox_xyxy(bbox)
        xs.extend([x1, x2])
        ys.extend([y1, y2])
    return [[min(xs), min(ys)], [max(xs), min(ys)], [max(xs), max(ys)], [min(xs), max(ys)]]


def _rect_expand(rect: tuple[int, int, int, int], img_shape, pad: int = 24) -> tuple[int, int, int, int]:
    h, w = img_shape[:2]
    x1, y1, x2, y2 = rect
    return (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(w, x2 + pad),
        min(h, y2 + pad),
    )


def _rect_intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 <= bx1 or ay2 <= by1 or bx2 <= ax1 or by2 <= ay1)


def _expand_bbox(bbox, img_shape, pad: int = 24):
    h, w = img_shape[:2]
    x1, y1, x2, y2 = _bbox_xyxy(bbox)
    return [
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(w, x2 + pad),
        min(h, y2 + pad),
    ]
