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
        "Trả về ĐÚNG MỘT JSON array, không giải thích thêm.\n\n"
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
        s = raw.find("[")
        e = raw.rfind("]") + 1
        if s >= 0 and e > s:
            parsed = json.loads(raw[s:e])
            if isinstance(parsed, list) and len(parsed) == len(texts):
                return [str(t) for t in parsed]
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


def inpaint_region(img, bbox):
    """Xóa text khỏi vùng bbox bằng cách phát hiện màu nền và fill."""
    import cv2
    import numpy as np
    h, w = img.shape[:2]
    pts  = np.array(bbox, dtype=np.int32)
    x1, y1, x2, y2 = _bbox_xyxy(bbox)
    x1e = max(0, x1 - 6)
    y1e = max(0, y1 - 6)
    x2e = min(w, x2 + 6)
    y2e = min(h, y2 + 6)

    roi = img[y1e:y2e, x1e:x2e]
    if roi.size == 0:
        return img

    # Lấy màu nền từ đường viền bbox
    top    = roi[0, :]
    bot    = roi[-1, :]
    left   = roi[:, 0]
    right  = roi[:, -1]
    border = np.concatenate([top, bot, left, right], axis=0)
    bg     = np.median(border, axis=0).astype(np.uint8)
    brightness = float(np.mean(bg))

    # Tạo mask phủ vùng text
    mask   = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    kernel = np.ones((3, 3), np.uint8)
    mask   = cv2.dilate(mask, kernel, iterations=3)

    if brightness > 160:
        # Nền sáng (speech bubble trắng) → fill màu nền trực tiếp, sạch hơn inpaint
        result = img.copy()
        result[mask > 0] = bg
        return result
    else:
        # Nền tối / phức tạp → dùng cv2.inpaint
        return cv2.inpaint(img, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)


# ── Text rendering ────────────────────────────────────────────────────────────

def render_text(
    img_pil,
    bbox,
    text: str,
    font_path: str | None,
):
    """Vẽ text đã dịch vào vùng bbox, tự chọn cỡ chữ và xuống dòng."""
    from PIL import ImageDraw  # lazy
    draw = ImageDraw.Draw(img_pil)
    x1, y1, x2, y2 = _bbox_xyxy(bbox)
    bw = max(x2 - x1, 10)
    bh = max(y2 - y1, 10)

    best_font  = None
    best_lines = [text]

    for size in range(min(bh, 26), 7, -1):
        font = _load_font(font_path, size)
        try:
            sample_bb = draw.textbbox((0, 0), "M", font=font)
            cw = max(sample_bb[2] - sample_bb[0], 1)
        except Exception:
            cw = size * 0.6
        cpl   = max(1, int(bw / cw))
        lines = textwrap.wrap(text, width=cpl) or [text]
        try:
            lh_bb = draw.textbbox((0, 0), "Ắp", font=font)
            lh    = lh_bb[3] - lh_bb[1] + 3
        except Exception:
            lh = size + 3
        if lh * len(lines) <= bh:
            best_font  = font
            best_lines = lines
            break

    if best_font is None:
        best_font = _load_font(font_path, 8)

    try:
        lh_bb = draw.textbbox((0, 0), "Ắp", font=best_font)
        lh    = lh_bb[3] - lh_bb[1] + 3
    except Exception:
        lh = 11

    total_h = lh * len(best_lines)
    ty      = y1 + max(0, (bh - total_h) // 2)

    for line in best_lines:
        try:
            lb = draw.textbbox((0, 0), line, font=best_font)
            lw = lb[2] - lb[0]
        except Exception:
            lw = len(line) * (lh // 2)
        tx = x1 + max(0, (bw - lw) // 2)
        # Viền trắng để dễ đọc trên mọi nền
        for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1), (0, -1), (0, 1), (-1, 0), (1, 0)):
            draw.text((tx + dx, ty + dy), line, font=best_font, fill=(255, 255, 255))
        draw.text((tx, ty), line, font=best_font, fill=(15, 15, 15))
        ty += lh

    return img_pil


# ── Main translator class ─────────────────────────────────────────────────────

class ImageTranslator:
    def __init__(
        self,
        model: str = "qwen2.5:7b",
        font_path: str | None = None,
        use_gpu: bool = True,
        src_lang: str = "zh",
        on_log=None,
        on_progress=None,
    ):
        self.model       = model
        self.font_path   = font_path
        self.use_gpu     = use_gpu
        self.src_lang    = src_lang if src_lang in ("zh", "en") else "zh"
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

            img_orig = cv2.imread(str(src))
            if img_orig is None:
                raise ValueError("Không đọc được file ảnh")

            bubbles = _find_speech_bubbles(img_orig)
            self._log(f"  [OCR] Phát hiện {len(bubbles)} bong bóng để OCR crop")
            raw_results = _run_ocr(img_orig, gpu=self.use_gpu, src_lang=self.src_lang)

            self._log(f"  [OCR] Phát hiện {len(raw_results)} vùng thô (full+crop)")
            if self.src_lang == "zh":
                for b, t, c in raw_results:
                    flag = "✓" if c > 0.1 and has_chinese(t) else "✗"
                    self._log(f"    {flag} conf={c:.2f} zh={has_chinese(t)} text={t!r:.50}")
                results = [
                    (b, t, c) for b, t, c in raw_results
                    if c > 0.1 and has_chinese(t)
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

            bboxes = [r[0] for r in results]
            texts  = [r[1] for r in results]
            self._log(f"  [OCR] {len(texts)} vùng text Trung (sau lọc)")

            # Translate
            self._log("  [TRANS] Đang dịch…")
            translations = translate_batch(texts, self.model, src_lang=self.src_lang)

            # Inpaint trên ảnh gốc (không phải ảnh đã preprocess)
            img = img_orig.copy()
            for bbox in bboxes:
                img = inpaint_region(img, bbox)

            # Render text Việt
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            for bbox, trans in zip(bboxes, translations):
                img_pil = render_text(img_pil, bbox, trans, self.font_path)

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
        translator: str = "m2m100",
        target_lang: str = "VIN",
        use_gpu: bool = True,
        python_path: str | None = None,
        detector: str = "",
        inpainter: str = "lama_large",
        upscale_ratio: str = "",
        detection_size: str = "",
        mask_dilation_offset: str = "",
        font_size_offset: str = "",
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
        self.font_size_offset     = font_size_offset
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
        if self.upscale_ratio:
            cfg["upscale"] = {"upscale_ratio": int(self.upscale_ratio)}
        if self.font_size_offset:
            cfg["render"] = {"font_size_offset": int(self.font_size_offset)}

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
