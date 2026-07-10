# -*- coding: utf-8 -*-
"""
_stroke_mask.py — Tạo mask bám NÉT CHỮ cho inpaint (thay vì tô đặc cả bbox).

Vấn đề: mask hình chữ nhật bắt lama phải "bịa" lại nguyên mảng lớn → xoá/mờ
cả nền bong bóng. MIT đẹp vì mask của nó chỉ ôm sát nét chữ (mask refinement).
Module này xấp xỉ điều đó bằng OpenCV thuần: tách pixel khác màu nền cục bộ
trong bbox, lọc component, dilate theo cỡ chữ (đúng công thức của MIT).

Mỗi bbox được chấm điểm mask_confidence từ 3 tín hiệu (nền đồng nhất, tỷ lệ
pixel mask, cấu trúc component); dưới ngưỡng thì trả None để caller fallback
về mask chữ nhật như cũ — không bao giờ tệ hơn hành vi hiện tại.

Tắt toàn bộ bằng env VLM_STROKE_MASK=0; log chi tiết từng bbox bằng
VLM_MASK_DEBUG=1.
"""

import os

# Ngưỡng cứng: ngoài khoảng này mask chắc chắn không phải chữ.
_HARD_MIN_RATIO = 0.02
_HARD_MAX_RATIO = 0.60
_CONF_THRESHOLD = 0.5


def stroke_mask_enabled() -> bool:
    return os.environ.get("VLM_STROKE_MASK", "1") != "0"


def _mask_debug() -> bool:
    return os.environ.get("VLM_MASK_DEBUG", "") == "1"


def build_stroke_mask(img_bgr, bbox, dilation_offset: int = 6, on_log=None):
    """
    Mask nét chữ (uint8 full ảnh, 255=inpaint) cho 1 bbox 4 điểm,
    hoặc None nếu không đủ tin cậy (caller nên fallback mask chữ nhật).
    """
    import cv2
    import numpy as np

    from ._utils import _bbox_xyxy

    log = on_log or (lambda msg: None)
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = _bbox_xyxy(bbox)
    # Pad theo cỡ bbox để ôm cả viền chữ/anti-alias sát mép.
    pad = int(np.clip(max(x2 - x1, y2 - y1) * 0.04, 4, 16))
    x1 = max(0, int(x1) - pad)
    y1 = max(0, int(y1) - pad)
    x2 = min(w, int(x2) + pad)
    y2 = min(h, int(y2) + pad)
    if x2 - x1 < 12 or y2 - y1 < 12:
        return None

    crop = img_bgr[y1:y2, x1:x2]
    ch = crop.shape[0]

    # Nền cục bộ = median từng kênh BGR của viền 2px (giữ màu để bubble
    # xanh/hồng vẫn tách chữ đúng, không quy hết về grayscale).
    ring = np.concatenate([
        crop[:2].reshape(-1, 3), crop[-2:].reshape(-1, 3),
        crop[:, :2].reshape(-1, 3), crop[:, -2:].reshape(-1, 3),
    ]).astype(np.float32)
    bg = np.median(ring, axis=0)
    ring_std = float(ring.std(axis=0).mean())

    # Khoảng cách màu tới nền (max qua 3 kênh) → Otsu. Không phụ thuộc cực
    # tính: nét lẫn viền chữ đều khác màu nền nên đều vào mask.
    diff = np.abs(crop.astype(np.float32) - bg).max(axis=2)
    diff = np.clip(diff, 0, 255).astype(np.uint8)
    _, th = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(th)
    keep = [
        lbl for lbl in range(1, num)
        if stats[lbl, cv2.CC_STAT_AREA] >= 6
        and stats[lbl, cv2.CC_STAT_HEIGHT] < 0.95 * ch
    ]
    if not keep:
        return None

    heights = stats[keep, cv2.CC_STAT_HEIGHT].astype(np.float64)
    areas = stats[keep, cv2.CC_STAT_AREA].astype(np.float64)
    stroke = (np.isin(labels, keep).astype(np.uint8)) * 255

    # Dilate thích ứng cỡ chữ — đúng công thức mask refinement của MIT
    # (tmp_repo/manga_translator/mask_refinement/text_mask_utils.py:175).
    text_size = float(np.clip(np.median(heights), 8, 60))
    d = max((int((text_size + dilation_offset) * 0.3) // 2) * 2 + 1, 3)
    stroke = cv2.dilate(stroke, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d)))

    ratio = cv2.countNonZero(stroke) / float(stroke.size)
    if not (_HARD_MIN_RATIO <= ratio <= _HARD_MAX_RATIO):
        return None

    # mask_confidence: kết hợp nhiều tín hiệu thay vì gate cứng từng cái,
    # để trang gradient/screentone nhẹ không bị loại oan.
    s_bg = 1.0 - min(ring_std / 60.0, 1.0)

    if ratio < 0.04:
        s_ratio = (ratio - 0.02) / 0.02
    elif ratio <= 0.35:
        s_ratio = 1.0
    else:
        s_ratio = (0.55 - ratio) / 0.20
    s_ratio = float(np.clip(s_ratio, 0.0, 1.0))

    # Chữ thật: nhiều component cỡ vừa, chiều cao tương đồng, không có blob
    # nào áp đảo (blob áp đảo = ôm nhầm mảng tranh).
    n_comp = len(keep)
    h_med = float(np.median(heights))
    height_cv = float(heights.std() / h_med) if h_med > 0 else 99.0
    dominant = float(areas.max() / areas.sum())
    s_comp = 1.0 if n_comp >= 3 else 0.35
    if height_cv >= 0.8:
        s_comp *= 0.6
    if dominant > 0.5:
        s_comp *= 0.5

    conf = 0.4 * s_bg + 0.3 * s_ratio + 0.3 * s_comp
    if _mask_debug():
        log(
            f"  [MASK] bbox=({x1},{y1},{x2},{y2}) ring_std={ring_std:.1f} "
            f"ratio={ratio:.3f} n_comp={n_comp} height_cv={height_cv:.2f} "
            f"dominant={dominant:.2f} | s_bg={s_bg:.2f} s_ratio={s_ratio:.2f} "
            f"s_comp={s_comp:.2f} conf={conf:.2f} -> "
            f"{'stroke' if conf >= _CONF_THRESHOLD else 'rect'}"
        )
    if conf < _CONF_THRESHOLD:
        return None

    full = np.zeros((h, w), dtype=np.uint8)
    full[y1:y2, x1:x2] = stroke
    return full


def build_combined_mask(img, regions, on_log=None):
    """
    regions: list of (bbox_4pts, dilate_ksize, dilate_iters, kind) → mask uint8
    gộp (255=inpaint). kind == "text" thử mask bám nét chữ (build_stroke_mask)
    để inpainter chỉ phải vá nét mảnh thay vì bịa lại cả bbox; không tin cậy
    thì fallback tô đặc bbox như cũ. kind == "wm" (watermark bán trong suốt
    phủ lên tranh) luôn tô đặc bbox.

    Trả về (mask, n_stroke, n_rect).
    """
    import cv2
    import numpy as np

    h, w = img.shape[:2]
    use_stroke = stroke_mask_enabled()
    n_stroke = n_rect = 0
    mask = np.zeros((h, w), dtype=np.uint8)
    for bbox, dilate_ksize, dilate_iters, kind in regions:
        region_mask = None
        if kind == "text" and use_stroke:
            region_mask = build_stroke_mask(img, bbox, on_log=on_log)
        if region_mask is None:
            pts = np.array(bbox, dtype=np.int32)
            region_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(region_mask, [pts], 255)
            region_mask = cv2.dilate(
                region_mask, np.ones((dilate_ksize, dilate_ksize), np.uint8), iterations=dilate_iters
            )
            n_rect += 1
        else:
            n_stroke += 1
        mask = cv2.bitwise_or(mask, region_mask)
    return mask, n_stroke, n_rect
