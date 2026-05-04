"""
_ocr.py — OCR engines, speech bubble detection, deduplication.
Phụ thuộc: paddleocr (ưu tiên), easyocr (fallback), cv2, numpy (lazy imports).
"""

import re
import threading

import numpy  # noqa – checked lazily inside functions

# ── Constants ─────────────────────────────────────────────────────────────────
_ML_THREAD_LIMIT = "2"
_ZH_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
_EN_RE = re.compile(r"[a-zA-Z]{2,}")

# ── Global OCR readers (lazy, keyed by lang+gpu) ──────────────────────────────
_paddle_readers: dict = {}
_paddle_lock    = threading.Lock()
_easyocr_readers: dict = {}
_easyocr_lock   = threading.Lock()


def _get_paddle_reader(gpu: bool = True, lang: str = "ch"):
    """
    PaddleOCR — hiệu quả hơn nhiều cho font bold/comic và vertical text.
    ✅ Điều chỉnh ngưỡng thấp hơn để bắt text mờ/nhiều màu tốt hơn.
    """
    key = (lang, gpu)
    with _paddle_lock:
        if key not in _paddle_readers:
            from paddleocr import PaddleOCR  # lazy
            _paddle_readers[key] = PaddleOCR(
                use_angle_cls=True,
                lang=lang,
                use_gpu=gpu,
                show_log=False,
                enable_mkldnn=False if not gpu else True,
                cpu_threads=int(_ML_THREAD_LIMIT),
                det_db_thresh=0.08,       # Giảm → bắt chữ SFX/nghệ thuật màu sắc
                det_db_box_thresh=0.2,    # Giảm → ít bỏ sót hơn
                det_db_unclip_ratio=2.2,  # Tăng → bao phủ viền glow/stroke của SFX
            )
    return _paddle_readers[key]


def _get_easyocr_reader(gpu: bool = True, src_lang: str = "zh"):
    langs = ["ch_sim", "en"] if src_lang == "zh" else ["en"]
    key = (tuple(langs), gpu)
    with _easyocr_lock:
        if key not in _easyocr_readers:
            import easyocr  # lazy
            _easyocr_readers[key] = easyocr.Reader(langs, gpu=gpu)
    return _easyocr_readers[key]


# Keep old name for backward compat
_reader_lock = _easyocr_lock


def _get_reader(gpu: bool = True):
    return _get_easyocr_reader(gpu)


def _preprocess_for_ocr(img_array):
    """
    Increase contrast and reduce noise background for CG/VN/scene text.
    ✅ Handles multi-color text well using LAB color space.
    """
    import cv2
    
    # LAB Color Space: Lightness channel keeps good multi-color text
    lab = cv2.cvtColor(img_array, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    
    # CLAHE on L channel with higher clipLimit for better contrast increase
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    
    return cv2.cvtColor(enhanced_l, cv2.COLOR_GRAY2BGR)


def _preprocess_light_text(img_array):
    """
    Special preprocessing for light/green colored text that standard OCR misses.
    Uses multi-channel approach to detect faint text across all color channels.
    """
    import cv2
    import numpy as np
    
    # Split into 3 color channels
    channels = cv2.split(img_array)
    
    # Process each channel separately with adaptive thresholding
    combined_mask = np.zeros_like(channels[0], dtype=np.uint8)
    
    for channel in channels:
        # C=-2: lower constant → accept more pixels as text (helps faint/light text)
        _, mask = cv2.adaptiveThreshold(
            channel,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            9,  # Smaller tile size for better local adaptation
            -2  # Negative C → lower effective threshold, catches light/faint text
        )
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    # Clean up noise with morphological operations
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    # Convert back to BGR for OCR
    enhanced_bgr = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
    
    return enhanced_bgr


def _preprocess_multi_color_text(img_array):
    """
    Enhanced preprocessing specifically for multi-color/green text.
    Combines LAB color space with multi-channel adaptive thresholding.
    """
    import cv2
    import numpy as np
    
    # Step 1: Lab color space processing (good for general multi-color)
    lab = cv2.cvtColor(img_array, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    
    # Apply CLAHE with aggressive settings for light text
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(6, 6))
    enhanced_l = clahe.apply(l_channel)
    
    # Step 2: Multi-channel processing (better for green/faint colors)
    channels = cv2.split(img_array)
    combined_mask = np.zeros_like(channels[0], dtype=np.uint8)
    
    for channel in channels:
        _, mask = cv2.adaptiveThreshold(
            channel,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            8,  # Smaller tile size
            -3  # Negative C → very aggressive faint text detection
        )
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    # Clean up noise
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    # Return multi-channel binary result (targeted at colored text)
    return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)


def _preprocess_colored_text(img_array):
    """
    HSV-based preprocessing specifically for colored text (cyan/teal/green).
    Targets bright saturated colors that LAB and adaptive threshold miss.
    Returns inverted binary image: text = dark on white (PaddleOCR-friendly).
    """
    import cv2
    import numpy as np

    hsv = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)

    # Cyan / teal text: H 80–105, high S, high V
    mask_cyan  = cv2.inRange(hsv, np.array([80, 100, 100]), np.array([105, 255, 255]))
    # Green text: H 40–80
    mask_green = cv2.inRange(hsv, np.array([40, 100, 100]), np.array([80,  255, 255]))
    # White/light text: low S, very high V
    mask_white = cv2.inRange(hsv, np.array([0,  0,   200]), np.array([180, 40,  255]))

    mask = cv2.bitwise_or(mask_cyan, mask_green)
    mask = cv2.bitwise_or(mask, mask_white)

    # Dilate slightly to fill gaps inside characters
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Invert: PaddleOCR reads dark text on light background best
    return cv2.cvtColor(cv2.bitwise_not(mask), cv2.COLOR_GRAY2BGR)

# ─────────────────────────────────────────────────────────────────────────────
def _run_multichannel_ocr(img_array, gpu: bool = True, paddle_lang: str = "ch") -> list[tuple]:
    """
    Multi-channel OCR for images with multi-color/green text.
    Process each color channel separately and combine results.
    """
    import cv2
    import numpy as np
    
    # Use the new preprocessing function for better light text detection
    try:
        enhanced = _preprocess_light_text(img_array)
        
        paddle = _get_paddle_reader(gpu, lang=paddle_lang)
        result = paddle.ocr(enhanced, cls=True)
        hits = [(line[0], line[1][0], float(line[1][1])) for line in result[0]] if result and result[0] else []
        
        return hits if len(hits) > 0 else []
    except Exception:
        pass
    
    return []


def _run_light_text_ocr(img_array, gpu: bool = True, paddle_lang: str = "ch") -> list[tuple]:
    """
    Special OCR for light/green colored text using enhanced preprocessing.
    This is the primary fix for Issue 2 - green text not being detected.
    """
    import cv2
    import numpy as np
    
    try:
        # Use multi-color preprocessing which combines both approaches
        enhanced = _preprocess_multi_color_text(img_array)
        
        paddle = _get_paddle_reader(gpu, lang=paddle_lang)
        result = paddle.ocr(enhanced, cls=True)
        hits = [(line[0], line[1][0], float(line[1][1])) for line in result[0]] if result and result[0] else []
        
        return hits
    except Exception:
        pass
    
    return []

# ─────────────────────────────────────────────────────────────────────────────


def _run_ocr_engine(img_array, gpu: bool = True, src_lang: str = "zh") -> list[tuple]:
    """
    Raw OCR engine on numpy array image.
    Priority PaddleOCR, fallback to EasyOCR.
    ✅ Get results from all steps (don't stop early) to catch SFX/art text.
    Returns list of (bbox_4pts, text, confidence).
    """
    paddle_lang = "ch" if src_lang == "zh" else "en"
    all_hits: list[tuple] = []

    try:
        paddle = _get_paddle_reader(gpu, lang=paddle_lang)

        # ── Step 1: PaddleOCR directly on original image ────────────────────────
        result = paddle.ocr(img_array, cls=True)
        if result and result[0]:
            all_hits.extend([(line[0], line[1][0], float(line[1][1])) for line in result[0]])

        # ── Step 2: LAB CLAHE — always run to catch multi-color/SFX ─────────
        try:
            processed = _preprocess_for_ocr(img_array)
            result2 = paddle.ocr(processed, cls=True)
            if result2 and result2[0]:
                all_hits.extend([(line[0], line[1][0], float(line[1][1])) for line in result2[0]])
        except Exception:
            pass

        # ── Step 3: Multi-channel — run when results are few ─────────────────
        if len(all_hits) < 5:
            try:
                multi_result = _run_multichannel_ocr(img_array, gpu, paddle_lang)
                all_hits.extend(multi_result)
            except Exception:
                pass

        # ── Step 4: Light text OCR — ALWAYS run to catch faint text ──────────
        try:
            light_result = _run_light_text_ocr(img_array, gpu, paddle_lang)
            if light_result:
                all_hits.extend(light_result)
        except Exception:
            pass

        # ── Step 5: HSV colored text — ALWAYS run to catch cyan/green text ────
        try:
            colored = _preprocess_colored_text(img_array)
            result5 = paddle.ocr(colored, cls=True)
            if result5 and result5[0]:
                all_hits.extend([(line[0], line[1][0], (line[1][1])) for line in result5[0]])
        except Exception:
            pass
        return all_hits  # _run_ocr will dedup (IoU) after
    except ImportError:
        pass
    except Exception:
        pass

    # EasyOCR fallback (last resort)
    reader = _get_easyocr_reader(gpu, src_lang=src_lang)
    raw = reader.readtext(
        img_array,
        min_size=10,
        text_threshold=0.5,
        low_text=0.3,
        contrast_ths=1.0,
        adjust_contrast=0.7,
        canvas_size=2560,
    )
    return [(b, t, c) for b, t, c in raw]


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


def _run_ocr(img_array, gpu: bool = True, src_lang: str = "zh") -> list[tuple]:
    """
    OCR orchestration: full-image OCR + bubble-crop OCR to increase recall.
    Returns list of (bbox_4pts, text, confidence).
    """
    import cv2
    import numpy as np

    all_results: list[tuple] = []
    all_results.extend(_run_ocr_engine(img_array, gpu, src_lang))

    try:
        for x1, y1, x2, y2 in _find_speech_bubbles(img_array):
            crop = img_array[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            ch, cw = crop.shape[:2]
            scale = 1.0
            if max(ch, cw) < 400:
                scale = 400 / max(ch, cw)
                crop = cv2.resize(crop, None, fx=scale, fy=scale,
                                  interpolation=cv2.INTER_LANCZOS4)
            for bbox, text, conf in _run_ocr_engine(crop, gpu, src_lang):
                pts = np.array(bbox, dtype=np.float32)
                pts /= scale
                pts[:, 0] += x1
                pts[:, 1] += y1
                all_results.append((pts.tolist(), text, conf))
    except Exception:
        pass

    if not all_results:
        return []

    rects = [_rect_of_bbox(b) for b, _t, _c in all_results]
    keep = [True] * len(all_results)
    for i in range(len(all_results)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(all_results)):
            if not keep[j]:
                continue
            if _iou_rect(rects[i], rects[j]) > 0.5:
                if all_results[i][2] >= all_results[j][2]:
                    keep[j] = False
                else:
                    keep[i] = False
                    break

    return [r for r, k in zip(all_results, keep) if k]


def has_chinese(text: str) -> bool:
    return bool(_ZH_RE.search(text))


def has_english(text: str) -> bool:
    return bool(_EN_RE.search(text))