"""
_inpaint.py — Xóa text khỏi ảnh bằng OpenCV inpaint hoặc LAMA.
"""

from ._stroke_mask import build_combined_mask
from ._utils import _bbox_xyxy, check_lama_available, _get_lama


def inpaint_region(img, bbox, method: str = "opencv", dilate_ksize: int = 5,
                   dilate_iters: int = 2, inpaint_radius: int = 6,
                   mask_override=None):
    """Xóa text khỏi vùng bbox bằng OpenCV hoặc LAMA.

    mask_override: mask uint8 full ảnh (255=inpaint) dùng thay mask chữ nhật
    mặc định — ví dụ mask bám nét chữ từ build_stroke_mask().
    """
    import cv2
    import numpy as np
    h, w = img.shape[:2]
    pts  = np.array(bbox, dtype=np.int32)
    x1, y1, x2, y2 = _bbox_xyxy(bbox)
    x1i, y1i = max(0, int(x1)), max(0, int(y1))
    x2i, y2i = min(w, int(x2)), min(h, int(y2))

    if mask_override is not None:
        mask = mask_override
    else:
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


def inpaint_regions(img, regions, method: str = "opencv", on_log=None):
    """Xóa mọi vùng trong `regions` với số pass inpaint tối thiểu.

    regions: list of (bbox_4pts, dilate_ksize, dilate_iters, inpaint_radius, kind).
    Thay vì inpaint toàn ảnh cho TỪNG region (O(số region × diện tích ảnh)),
    gộp mask trước: LAMA chạy đúng 1 lần cho cả ảnh; OpenCV chạy 1 lần cho
    mỗi nhóm inpaint_radius (thực tế ≤3 nhóm: text/wm/logo) để giữ đúng
    radius theo loại vùng.
    """
    if not regions:
        return img

    import cv2
    import numpy as np

    log = on_log or (lambda msg: None)

    if method == "lama" and check_lama_available():
        try:
            mask, n_stroke, n_rect = build_combined_mask(
                img, [(b, dk, di, kind) for b, dk, di, _, kind in regions], on_log=log
            )
            log(f"  [INPAINT] mask: {n_stroke} stroke / {n_rect} rect (1 pass lama)")
            lama = _get_lama()
            from PIL import Image as _Image
            pil_img  = _Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            pil_mask = _Image.fromarray(mask)
            out = lama(pil_img, pil_mask)
            return cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)
        except Exception:
            pass

    by_radius: dict[int, list] = {}
    for bbox, dk, di, radius, kind in regions:
        by_radius.setdefault(radius, []).append((bbox, dk, di, kind))

    n_stroke = n_rect = 0
    for radius, group in by_radius.items():
        mask, ns, nr = build_combined_mask(img, group, on_log=log)
        n_stroke += ns
        n_rect += nr
        img = cv2.inpaint(img, mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)
    log(f"  [INPAINT] mask: {n_stroke} stroke / {n_rect} rect ({len(by_radius)} pass)")
    return img
