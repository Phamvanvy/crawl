"""
translator_engine.py — Pipeline dịch ảnh: ZH → VI (hoặc ngôn ngữ bất kỳ)
Pipeline: PaddleOCR (hoặc EasyOCR fallback)  →  OpenCV inpaint  →  Ollama (Qwen)  →  Pillow render

Các thư viện nặng (cv2, numpy, PIL, paddleocr/easyocr) được import lazily —
web_app.py có thể import module này mà không báo lỗi khi chưa cài.
"""

import json
import re
import shutil
import textwrap
import threading
from pathlib import Path

import requests

OLLAMA_BASE = "http://localhost:11434"
IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_ZH_RE      = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
_EN_RE      = re.compile(r"[a-zA-Z]{2,}")

# ── Global OCR readers (lazy, keyed by lang+gpu) ──────────────────────────────
_paddle_readers: dict = {}
_paddle_lock    = threading.Lock()
_easyocr_readers: dict = {}
_easyocr_lock   = threading.Lock()

def _get_paddle_reader(gpu: bool = True, lang: str = "ch"):
    """PaddleOCR — hiệu quả hơn nhiều cho font bold/comic và vertical text."""
    key = (lang, gpu)
    with _paddle_lock:
        if key not in _paddle_readers:
            from paddleocr import PaddleOCR  # lazy
            _paddle_readers[key] = PaddleOCR(
                use_angle_cls=True,  # phát hiện text xoay/dọc
                lang=lang,
                use_gpu=gpu,
                show_log=False,
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

def _run_ocr_engine(img_array, gpu: bool = True, src_lang: str = "zh") -> list[tuple]:
    """
    Raw OCR engine trên numpy array ảnh.
    Ưu tiên PaddleOCR (tốt hơn cho manga/comic/vertical text).
    Fallback sang EasyOCR nếu PaddleOCR chưa cài.
    Trả về list of (bbox_4pts, text, confidence).
    """
    paddle_lang = "ch" if src_lang == "zh" else "en"
    # PaddleOCR
    try:
        paddle = _get_paddle_reader(gpu, lang=paddle_lang)
        result = paddle.ocr(img_array, cls=True)
        if result and result[0]:
            return [(line[0], line[1][0], float(line[1][1])) for line in result[0]]
        return []
    except ImportError:
        pass
    except Exception:
        pass
    # EasyOCR fallback
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
    Tìm các vùng sáng (nền trắng/xám nhạt) đủ lớn và không quá dài.
    Returns: list of (x1, y1, x2, y2)
    """
    import cv2
    import numpy as np

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Ngưỡng tìm vùng nền sáng (bong bóng hội thoại thường nền trắng/xám nhạt)
    _, thresh = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)

    # Đóng lỗ hổng bên trong bong bóng (text tối nằm trong vùng sáng)
    ksize = max(7, w // 80)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = h * w
    bubbles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Lọc theo diện tích: 0.2% – 35% ảnh
        if not (img_area * 0.002 < area < img_area * 0.35):
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        # Bỏ qua hình quá dài / quá rộng (không phải bubble)
        if max(bw, bh) / max(min(bw, bh), 1) > 6:
            continue
        pad = max(4, int(min(bw, bh) * 0.08))
        bubbles.append((
            max(0, x - pad), max(0, y - pad),
            min(w, x + bw + pad), min(h, y + bh + pad),
        ))
    return bubbles


def _rect_of_bbox(bbox) -> tuple:
    """4-point OCR bbox → (x1, y1, x2, y2) float."""
    import numpy as np
    pts = np.array(bbox, dtype=np.float32)
    return float(pts[:, 0].min()), float(pts[:, 1].min()), \
           float(pts[:, 0].max()), float(pts[:, 1].max())


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


def _run_ocr(img_array, gpu: bool = True, src_lang: str = "zh") -> list[tuple]:
    """
    OCR orchestration: full-image OCR + bubble-crop OCR để tăng recall.
    Trả về list of (bbox_4pts, text, confidence).
    """
    import cv2
    import numpy as np

    all_results: list[tuple] = []

    # 1. Full-image OCR
    all_results.extend(_run_ocr_engine(img_array, gpu, src_lang))

    # 2. Speech bubble detection → per-crop OCR
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

    # 3. Deduplicate: xoá bbox trùng lặp (IOU > 0.5), giữ confidence cao hơn
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


# ── Font helper ───────────────────────────────────────────────────────────────
_font_cache: dict = {}

def _load_font(font_path: str | None, size: int):
    """Lazy-load ImageFont, tránh import lỗi khi Pillow chưa cài."""
    from PIL import ImageFont  # lazy
    key = (font_path, size)
    if key in _font_cache:
        return _font_cache[key]
    candidates = []
    if font_path:
        candidates.append(font_path)
    # System fonts first — Arial & Segoe UI have full Vietnamese (Latin Extended Additional)
    # NotoSans from the latin-greek-cyrillic repo may lack Vietnamese glyphs
    candidates += [
        str(Path(__file__).parent / "fonts" / "BeVietnamPro-Regular.ttf"),
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        str(Path(__file__).parent / "fonts" / "NotoSans-Regular.ttf"),
        r"C:\Windows\Fonts\tahoma.ttf",
    ]
    for p in candidates:
        try:
            f = ImageFont.truetype(p, size)
            _font_cache[key] = f
            return f
        except Exception:
            continue
    return ImageFont.load_default()


# ── Ollama helpers ────────────────────────────────────────────────────────────

def check_ollama() -> dict:
    """Kiểm tra Ollama đang chạy và liệt kê models có sẵn."""
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        if r.ok:
            models = [m["name"] for m in r.json().get("models", [])]
            return {"ok": True, "models": models}
        return {"ok": False, "error": f"HTTP {r.status_code}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def translate_batch(texts: list[str], model: str, src_lang: str = "zh") -> list[str]:
    """Dịch batch texts qua Ollama API, trả về list cùng thứ tự."""
    if not texts:
        return []
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    lang_name = "tiếng Trung" if src_lang == "zh" else "tiếng Anh"
    prompt = (
        f"Dịch các đoạn text {lang_name} sau sang tiếng Việt.\n"
        "Đây là hội thoại trong manhwa/manga, giữ nguyên cảm xúc và sự ngắn gọn.\n"
        "Trả về ĐÚNG MỘT JSON array, không giải thích thêm. /no_think\n\n"
        f"Texts:\n{numbered}\n\n"
        "Kết quả (chỉ JSON array):"
    )
    try:
        resp = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=180,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
        # Strip Qwen3 thinking tags <think>...</think>
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        s = raw.find("[")
        e = raw.rfind("]") + 1
        if s >= 0 and e > s:
            parsed = json.loads(raw[s:e])
            if isinstance(parsed, list) and len(parsed) == len(texts):
                def _extract(t):
                    if isinstance(t, str):
                        return t
                    if isinstance(t, dict):
                        for k in ("text", "translation", "result", "output", "translated", "vi"):
                            if k in t and isinstance(t[k], str):
                                return t[k]
                        # lấy value đầu tiên là str
                        for v in t.values():
                            if isinstance(v, str):
                                return v
                    return str(t)
                return [_extract(t) for t in parsed]
    except Exception:
        pass
    return texts  # fallback: giữ nguyên text gốc


# ── Inpainting ────────────────────────────────────────────────────────────────

def _bbox_xyxy(bbox) -> tuple[int, int, int, int]:
    """EasyOCR 4-point polygon → (x1, y1, x2, y2)"""
    import numpy as np
    pts = np.array(bbox, dtype=np.int32)
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    return int(x1), int(y1), int(x2), int(y2)


def inpaint_region(img, bbox, method: str = "opencv"):
    """Xóa text khỏi vùng bbox bằng openCV hoặc LAMA."""
    import cv2
    import numpy as np
    h, w = img.shape[:2]
    pts  = np.array(bbox, dtype=np.int32)
    x1, y1, x2, y2 = _bbox_xyxy(bbox)
    x1i, y1i = max(0, int(x1)), max(0, int(y1))
    x2i, y2i = min(w, int(x2)), min(h, int(y2))

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)

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

    # Fallback: OpenCV inpaint
    return cv2.inpaint(img, mask, inpaintRadius=6, flags=cv2.INPAINT_TELEA)


# ── Text rendering ────────────────────────────────────────────────────────────

def render_text(
    img_pil,
    bbox,
    text: str,
    font_path: str | None,
):
    """
    Vẽ text vào vùng bbox:
    - Vẽ background box (màu lấy từ vùng đã inpaint)
    - Text căn giữa, màu tương phản, có viền shadow
    """
    from PIL import ImageDraw
    import numpy as np
    if not text.strip():
        return img_pil

    draw = ImageDraw.Draw(img_pil)
    x1, y1, x2, y2 = _bbox_xyxy(bbox)
    x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
    iw, ih = img_pil.size
    bw = max(x2i - x1i, 40)
    bh = max(y2i - y1i, 16)

    # Lấy màu nền từ vùng đã inpaint
    arr = np.array(img_pil.convert("RGB"))
    ry1, ry2 = max(0, y1i), min(ih, y2i)
    rx1, rx2 = max(0, x1i), min(iw, x2i)
    if ry2 > ry1 and rx2 > rx1:
        roi = arr[ry1:ry2, rx1:rx2]
        bg_rgb = tuple(int(v) for v in np.median(roi.reshape(-1, 3), axis=0))
    else:
        bg_rgb = (20, 20, 20)

    brightness   = sum(bg_rgb) / 3
    text_color   = (255, 255, 255) if brightness < 140 else (15, 15, 15)
    shadow_color = (0, 0, 0)       if brightness < 140 else (255, 255, 255)

    # Tìm cỡ chữ tốt nhất
    best_font  = None
    best_lines = [text]
    for size in range(min(bh, 28), 7, -1):
        font = _load_font(font_path, size)
        try:
            bb = draw.textbbox((0, 0), "M", font=font)
            cw = max(bb[2] - bb[0], 1)
        except Exception:
            cw = max(int(size * 0.55), 1)
        cpl   = max(1, int(bw * 0.95 / cw))
        lines = textwrap.wrap(text, width=cpl) or [text]
        try:
            bb = draw.textbbox((0, 0), "Ắp", font=font)
            lh = bb[3] - bb[1] + 2
        except Exception:
            lh = size + 2
        if lh * len(lines) <= bh + size:
            best_font  = font
            best_lines = lines
            break

    if best_font is None:
        best_font = _load_font(font_path, 9)

    try:
        bb = draw.textbbox((0, 0), "Ắp", font=best_font)
        lh = bb[3] - bb[1] + 2
    except Exception:
        lh = 11

    total_h   = lh * len(best_lines)
    actual_y2 = min(ih, max(y2i, y1i + total_h + 8))

    # Vẽ background box
    draw.rectangle([x1i - 3, y1i - 3, x2i + 3, actual_y2 + 3], fill=bg_rgb)

    # Vẽ text căn giữa
    ty = y1i + max(0, (actual_y2 - y1i - total_h) // 2)
    for line in best_lines:
        try:
            lb = draw.textbbox((0, 0), line, font=best_font)
            lw = lb[2] - lb[0]
        except Exception:
            lw = len(line) * lh // 2
        tx = x1i + max(0, (bw - lw) // 2)
        for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1), (0, -1), (0, 1), (-1, 0), (1, 0)):
            draw.text((tx + dx, ty + dy), line, font=best_font, fill=shadow_color)
        draw.text((tx, ty), line, font=best_font, fill=text_color)
        ty += lh

    return img_pil


def _group_nearby_regions(results: list, gap_px: int = 18) -> list:
    """
    Gom các OCR bbox gần nhau (cùng cột/dòng văn bản) thành một group.
    Trả về list of list of (bbox, text, conf).
    """
    import numpy as np
    if not results:
        return []

    def to_xyxy(bbox):
        pts = np.array(bbox, dtype=np.float32)
        return (float(pts[:, 0].min()), float(pts[:, 1].min()),
                float(pts[:, 0].max()), float(pts[:, 1].max()))

    enriched = [(to_xyxy(b), b, t, c) for b, t, c in results]
    enriched.sort(key=lambda x: (x[0][1], x[0][0]))  # top→bottom, left→right
    used   = [False] * len(enriched)
    groups = []

    for i in range(len(enriched)):
        if used[i]:
            continue
        group = [enriched[i]]
        used[i] = True
        changed = True
        while changed:
            changed = False
            gx1 = min(it[0][0] for it in group)
            gy1 = min(it[0][1] for it in group)
            gx2 = max(it[0][2] for it in group)
            gy2 = max(it[0][3] for it in group)
            for j in range(len(enriched)):
                if used[j]:
                    continue
                r2     = enriched[j][0]
                v_gap  = max(0.0, max(r2[1] - gy2, gy1 - r2[3]))
                h_span = max(r2[2], gx2) - min(r2[0], gx1)
                h_olap = min(r2[2], gx2) - max(r2[0], gx1)
                h_rat  = h_olap / h_span if h_span > 0 else 0.0
                if v_gap <= gap_px and h_rat > 0.15:
                    group.append(enriched[j])
                    used[j] = True
                    changed = True
        groups.append([(b, t, c) for _, b, t, c in group])

    return groups


# ── Main translator class ─────────────────────────────────────────────────────

class ImageTranslator:
    def __init__(
        self,
        model: str = "qwen3:8b",
        font_path: str | None = None,
        use_gpu: bool = True,
        src_lang: str = "zh",
        inpainter: str = "opencv",
        on_log=None,
        on_progress=None,
    ):
        self.model       = model
        self.font_path   = font_path
        self.use_gpu     = use_gpu
        self.src_lang    = src_lang if src_lang in ("zh", "en") else "zh"
        self.inpainter   = inpainter if inpainter in ("opencv", "lama") else "opencv"
        self.on_log      = on_log or print
        self.on_progress = on_progress or (lambda d, t: None)

    def _log(self, msg: str):
        self.on_log(msg)

    def process_image(self, src: Path, dst: Path) -> bool:
        """Xử lý một ảnh: OCR → inpaint → dịch → render → lưu."""
        try:
            import cv2
            from PIL import Image
            self._log(f"  [OCR] {src.name}")

            import numpy as np
            img_orig = cv2.imdecode(np.fromfile(str(src), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img_orig is None:
                raise ValueError("Không đọc được file ảnh")

            bubbles = _find_speech_bubbles(img_orig)
            self._log(f"  [OCR] Phát hiện {len(bubbles)} bong bóng để OCR crop")
            raw_results = _run_ocr(img_orig, gpu=self.use_gpu, src_lang=self.src_lang)

            self._log(f"  [OCR] Phát hiện {len(raw_results)} vùng thô (full+crop)")
            if self.src_lang == "zh":
                for b, t, c in raw_results:
                    zh_len = len([ch for ch in t if '\u4e00' <= ch <= '\u9fff' or '\u3400' <= ch <= '\u4dbf'])
                    flag = "✓" if has_chinese(t) and (c > 0.1 or zh_len >= 4) else "✗"
                    self._log(f"    {flag} conf={c:.2f} zh={has_chinese(t)} text={t!r:.50}")
                results = [
                    (b, t, c) for b, t, c in raw_results
                    if has_chinese(t) and (c > 0.1 or len([ch for ch in t if '\u4e00' <= ch <= '\u9fff' or '\u3400' <= ch <= '\u4dbf']) >= 4)
                ]
                if not results:
                    self._log(f"  [SKIP] Không có text Trung: {src.name}")
                    shutil.copy2(src, dst)
                    return True
            else:  # en
                for b, t, c in raw_results:
                    flag = "✓" if c > 0.3 and has_english(t) else "✗"
                    self._log(f"    {flag} conf={c:.2f} en={has_english(t)} text={t!r:.50}")
                results = [
                    (b, t, c) for b, t, c in raw_results
                    if c > 0.3 and has_english(t) and len(t.strip()) >= 2
                ]
                if not results:
                    self._log(f"  [SKIP] Không có text Anh: {src.name}")
                    shutil.copy2(src, dst)
                    return True

            src_lbl = "Trung" if self.src_lang == "zh" else "Anh"
            self._log(f"  [OCR] {len(results)} vùng text {src_lbl} (sau lọc)")

            # Gom các vùng gần nhau thành block
            groups = _group_nearby_regions(results)
            self._log(f"  [OCR] → {len(groups)} block")

            # Dịch tất cả texts cùng lúc (flatten → translate → split lại)
            all_texts = [t for grp in groups for _, t, _ in grp]
            self._log("  [TRANS] Đang dịch…")
            all_trans = translate_batch(all_texts, self.model, src_lang=self.src_lang)
            idx = 0
            group_trans: list[list[str]] = []
            for grp in groups:
                n = len(grp)
                group_trans.append(all_trans[idx:idx + n])
                idx += n

            # Inpaint tất cả bbox
            img = img_orig.copy()
            for grp in groups:
                for bbox, _, _ in grp:
                    img = inpaint_region(img, bbox, method=self.inpainter)

            # Render mỗi group thành một text block sạch
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            for grp, trans_list in zip(groups, group_trans):
                rects = [_bbox_xyxy(b) for b, _, _ in grp]
                gx1   = int(min(r[0] for r in rects))
                gy1   = int(min(r[1] for r in rects))
                gx2   = int(max(r[2] for r in rects))
                gy2   = int(max(r[3] for r in rects))
                full_text    = " ".join(t for t in trans_list if t.strip())
                merged_bbox  = [[gx1, gy1], [gx2, gy1], [gx2, gy2], [gx1, gy2]]
                img_pil = render_text(img_pil, merged_bbox, full_text, self.font_path)

            # Lưu — giữ nguyên định dạng
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.suffix.lower() in (".jpg", ".jpeg"):
                img_pil.save(str(dst), "JPEG", quality=95)
            else:
                img_pil.save(str(dst))

            self._log(f"  [OK] → {dst.name}")
            return True

        except Exception as exc:
            self._log(f"  [FAIL] {src.name}: {exc}")
            try:
                shutil.copy2(src, dst)
            except Exception:
                pass
            return False

    def process_folder(
        self,
        input_dir: str,
        output_dir: str,
        stop_event: threading.Event | None = None,
    ) -> tuple[int, int]:
        """Xử lý toàn bộ ảnh trong input_dir, lưu vào output_dir."""
        inp = Path(input_dir)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        images = sorted(f for f in inp.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        if not images:
            self._log("Không tìm thấy ảnh trong thư mục.")
            return 0, 0

        total = len(images)
        self._log(f"Tổng: {total} ảnh cần xử lý")
        ok = fail = 0

        for i, path in enumerate(images, 1):
            if stop_event and stop_event.is_set():
                self._log("⚠  Đã dừng theo yêu cầu.")
                break
            success = self.process_image(path, out / path.name)
            if success:
                ok += 1
            else:
                fail += 1
            self.on_progress(i, total)

        return ok, fail


# ── manga-image-translator backend ───────────────────────────────────────────

_MIT_INSTALL_HINT = (
    "Cần Python 3.11 và venv riêng.\n"
    "1. Cài Python 3.11 từ python.org\n"
    "2. py -3.11 -m venv mit_venv\n"
    "3. mit_venv\\Scripts\\pip install git+https://github.com/zyddnys/manga-image-translator.git"
)

def _find_mit_python() -> str | None:
    """
    Tìm Python có manga_translator đã cài.
    Ưu tiên: mit_venv trong thư mục project → fallback py.exe -3.11.
    Trả về đường dẫn python hoặc None nếu không tìm thấy.
    Dùng file-existence check thay vì subprocess import (tránh timeout vì
    torch/cv2 import rất chậm lần đầu, ~30-60s).
    """
    import subprocess

    def _has_manga_translator(python_exe: Path) -> bool:
        """Kiểm tra manga_translator bằng cách tìm file trong site-packages."""
        # Thử Windows (Scripts/) và Linux (bin/)
        for sp_root in [
            python_exe.parent.parent / "Lib" / "site-packages",
            python_exe.parent.parent / "lib" / "site-packages",
        ]:
            if (sp_root / "manga_translator" / "__init__.py").exists():
                return True
        # Fallback: hỏi Python chính xác site-packages (nhanh, không load torch)
        try:
            r = subprocess.run(
                [str(python_exe), "-c",
                 "import sysconfig; print(sysconfig.get_path('purelib'))"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0:
                sp = Path(r.stdout.strip())
                if (sp / "manga_translator" / "__init__.py").exists():
                    return True
        except Exception:
            pass
        return False

    candidates = [
        # mit_venv trong cùng thư mục project
        Path(__file__).parent / "mit_venv" / "Scripts" / "python.exe",
        Path(__file__).parent / "mit_venv" / "bin" / "python",
    ]
    for path in candidates:
        if path.exists() and _has_manga_translator(path):
            return str(path)

    # Thử py launcher (Windows py.exe)
    for py_flag in ["-3.11", "-3.10"]:
        try:
            r = subprocess.run(
                ["py", py_flag, "-c", "import sys; print(sys.executable)"],
                capture_output=True, text=True, timeout=8,
            )
            if r.returncode == 0:
                exe = Path(r.stdout.strip())
                if exe.exists() and _has_manga_translator(exe):
                    return str(exe)
        except Exception:
            pass
    return None


def check_mit() -> dict:
    """Kiểm tra manga-image-translator đã cài và tìm Python phù hợp."""
    exe = _find_mit_python()
    if exe:
        return {"ok": True, "version": "installed", "python": exe}
    return {"ok": False, "error": _MIT_INSTALL_HINT}


class MITImageTranslator:
    """
    Backend dùng manga-image-translator (github.com/zyddnys/manga-image-translator)
    cho chất lượng inpainting và render chữ tốt hơn.
    Gọi `python -m manga_translator translate …` qua subprocess.
    """

    def __init__(
        self,
        translator: str = "m2m100_big",
        target_lang: str = "VIN",
        use_gpu: bool = True,
        python_path: str | None = None,
        detector: str = "",
        inpainter: str = "lama_large",
        upscale_ratio: str = "",
        detection_size: str = "",
        mask_dilation_offset: str = "",
        unclip_ratio: str = "",
        font_size_offset: str = "",
        font_size_minimum: str = "",
        font_size_fixed: str = "",
        font_color: str = "",
        verbose: bool = False,
        skip_no_text: bool = False,
        overwrite: bool = False,
        on_log=None,
        on_progress=None,
    ):
        self.translator           = translator
        self.target_lang          = target_lang
        self.use_gpu              = use_gpu
        self.python_path          = python_path or _find_mit_python()
        self.detector             = detector
        self.inpainter            = inpainter
        self.upscale_ratio        = upscale_ratio
        self.detection_size       = detection_size
        self.mask_dilation_offset = mask_dilation_offset
        self.unclip_ratio         = unclip_ratio
        self.font_size_offset     = font_size_offset
        self.font_size_minimum    = font_size_minimum
        self.font_size_fixed      = font_size_fixed
        self.font_color           = font_color
        self.verbose              = verbose
        self.skip_no_text         = skip_no_text
        self.overwrite            = overwrite
        self.on_log               = on_log or print
        self.on_progress          = on_progress or (lambda d, t: None)

    def _log(self, msg: str):
        self.on_log(msg)

    def process_folder(
        self,
        input_dir: str,
        output_dir: str,
        stop_event: threading.Event | None = None,
    ) -> tuple[int, int]:
        import subprocess
        import time

        if not self.python_path:
            self._log(f"  [FAIL] Không tìm thấy Python có manga_translator.")
            self._log(f"  [FAIL] {_MIT_INSTALL_HINT}")
            return 0, 0

        inp = Path(input_dir)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        images = sorted(f for f in inp.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        if not images:
            self._log("Không tìm thấy ảnh trong thư mục.")
            return 0, 0

        total = len(images)
        self._log(f"Tổng: {total} ảnh — manga-image-translator")
        self._log(f"  Translator : {self.translator}  →  {self.target_lang}")
        self.on_progress(0, total)

        # Phiên bản mới dùng config file thay vì CLI flags cho translator/inpainter/...
        import json, tempfile, os
        cfg: dict = {
            "translator": {
                "translator": self.translator,
                "target_lang": self.target_lang,
            },
        }
        if self.inpainter:
            cfg["inpainter"] = {"inpainter": self.inpainter}
        if self.detector:
            cfg["detector"] = {"detector": self.detector}
        if self.detection_size:
            cfg.setdefault("detector", {})["detection_size"] = int(self.detection_size)
        if self.mask_dilation_offset:
            cfg["mask_dilation_offset"] = int(self.mask_dilation_offset)  # top-level Config field
        if self.unclip_ratio:
            cfg.setdefault("detector", {})["unclip_ratio"] = float(self.unclip_ratio)
        if self.upscale_ratio:
            cfg["upscale"] = {"upscale_ratio": int(self.upscale_ratio)}
        if self.font_size_offset:
            cfg.setdefault("render", {})["font_size_offset"] = int(self.font_size_offset)
        if self.font_size_minimum:
            cfg.setdefault("render", {})["font_size_minimum"] = int(self.font_size_minimum)
        if self.font_size_fixed:
            cfg.setdefault("render", {})["font_size"] = int(self.font_size_fixed)
        if self.font_color:
            cfg.setdefault("render", {})["font_color"] = self.font_color

        # Auto-inject gpt_config for custom_openai to improve translation quality
        if self.translator == "custom_openai":
            gpt_cfg = Path(__file__).parent / "gpt_config_vi.yaml"
            if gpt_cfg.exists():
                cfg.setdefault("translator", {})["gpt_config"] = str(gpt_cfg)
                self._log(f"  [GPT] Using custom gpt_config: {gpt_cfg.name}")

        tf = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump(cfg, tf, ensure_ascii=False)
        tf.close()
        cfg_path = tf.name

        cmd = [self.python_path, "-m", "manga_translator"]
        if self.use_gpu:
            cmd.append("--use-gpu")
        if self.verbose:
            cmd.append("--verbose")
        cmd += ["local", "-i", str(inp), "-o", str(out), "--config-file", cfg_path]
        if self.skip_no_text:
            cmd.append("--skip-no-text")
        if self.overwrite:
            cmd.append("--overwrite")

        self._log(f"  [CMD] {' '.join(str(c) for c in cmd)}")

        # Theo dõi thư mục output để cập nhật tiến trình
        _last: list[int] = [0]
        _stop_watch = threading.Event()

        def _watcher():
            while not _stop_watch.is_set():
                try:
                    cnt = sum(
                        1 for f in out.rglob("*")
                        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
                    )
                    if cnt != _last[0]:
                        _last[0] = cnt
                        self.on_progress(cnt, total)
                except Exception:
                    pass
                time.sleep(0.8)

        wt = threading.Thread(target=_watcher, daemon=True)
        wt.start()

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            for line in proc.stdout:
                line = line.rstrip("\n")
                if line:
                    self._log(f"  {line}")
                if stop_event and stop_event.is_set():
                    proc.terminate()
                    self._log("⚠  Đã dừng theo yêu cầu.")
                    break
            proc.wait()
        except Exception as exc:
            self._log(f"  [FAIL] Lỗi chạy manga_translator: {exc}")
        finally:
            _stop_watch.set()
            wt.join(timeout=2)
            try:
                os.unlink(cfg_path)
            except Exception:
                pass

        ok = sum(
            1 for f in out.rglob("*")
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS
        )
        fail = max(0, total - ok)
        self.on_progress(ok, total)
        self._log(f"  [OK] Kết quả: {ok} ảnh trong {out}")
        return ok, fail
