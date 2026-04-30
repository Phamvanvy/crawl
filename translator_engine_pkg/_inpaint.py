"""
_inpaint.py — Xóa text khỏi ảnh bằng OpenCV inpaint hoặc LAMA.
"""

from ._utils import _bbox_xyxy, check_lama_available, _get_lama


def inpaint_region(img, bbox, method: str = "opencv", dilate_ksize: int = 5,
                   dilate_iters: int = 2, inpaint_radius: int = 6):
    """Xóa text khỏi vùng bbox bằng OpenCV hoặc LAMA."""
    import cv2
    import numpy as np
    h, w = img.shape[:2]
    pts  = np.array(bbox, dtype=np.int32)
    x1, y1, x2, y2 = _bbox_xyxy(bbox)
    x1i, y1i = max(0, int(x1)), max(0, int(y1))
    x2i, y2i = min(w, int(x2)), min(h, int(y2))

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    mask = cv2.dilate(mask, np.ones((dilate_ksize, dilate_ksize), np.uint8), iterations=dilate_iters)

    if method == "lama" and check_lama_available():
        try:
            lama = _get_lama()
            from PIL import Image as _Image
            pil_img  = _Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            pil_mask = _Image.fromarray(mask)
            out = lama(pil_img, pil_mask)
            return cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)
        except Exception:
            pass

    return cv2.inpaint(img, mask, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA)
