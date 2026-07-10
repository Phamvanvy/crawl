# -*- coding: utf-8 -*-
"""
_image_translator.py — ImageTranslator: OCR → inpaint → dịch → render.
"""

import shutil
import threading
from collections import deque
from pathlib import Path

from ._ocr import (
    _find_speech_bubbles, has_chinese, has_japanese, has_english, _bubble_coverage,
)
from ._vlm_ocr import _run_vlm_ocr
from ._translate import (
    translate_batch, post_process_translation,
    comprehensive_post_processing,
)
from ._common_utils import contains_chinese, contains_japanese, contains_cjk, JA_RE, save_image_compressed
from ._utils import (
    _bbox_xyxy, _union_bboxes, _rect_expand, _rect_intersects, _expand_bbox,
    _looks_like_watermark,
)
from ._inpaint import inpaint_regions
from ._mit_inpaint_bridge import inpaint_regions_lama_large
from ._render import render_text, _group_nearby_regions

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ── Ngưỡng lọc vùng OCR theo ngôn ngữ ────────────────────────────────────────
_MIN_CONF_CJK = 0.1   # zh/ja: nhận nếu conf vượt ngưỡng này...
_MIN_ZH_CHARS = 4     # ...hoặc có đủ số ký tự Hán (VLM luôn trả conf 0.99)
_MIN_JA_CHARS = 3
_MIN_CONF_EN  = 0.3
_MIN_EN_LEN   = 2

# ── Pad mở rộng vùng watermark (px) ──────────────────────────────────────────
_WM_NEIGHBOR_PAD = 80   # vùng quét bbox lân cận watermark để gộp xoá
_WM_UNION_PAD    = 90   # union mọi watermark khi inpaint
_WM_LOGO_PAD     = 64   # từng logo riêng lẻ

# ── Neo group text vào bong bóng ─────────────────────────────────────────────
_BUBBLE_COV_MIN = 0.4   # coverage tối thiểu để coi group thuộc bong bóng
_MAX_CHARS_DIV  = 8     # max_chars = bề rộng bong bóng // 8
_MAX_LINES_DIV  = 14    # max_lines = chiều cao bong bóng // 14
_MIN_MAX_CHARS  = 6

# ── Tham số inpaint theo loại vùng: (dilate_ksize, dilate_iters, radius) ────
_TEXT_INPAINT     = (5, 2, 6)
_WM_UNION_INPAINT = (9, 4, 12)
_WM_LOGO_INPAINT  = (9, 4, 10)

# ── Pad khung render khi không có bong bóng đủ tin cậy ──────────────────────
_RENDER_PAD_X_FRAC = 0.30
_RENDER_PAD_Y_FRAC = 0.40
_RENDER_PAD_MIN    = 8

# ── Dải ký tự để đếm chữ Hán/kana ────────────────────────────────────────────
_ZH_RANGES = (  # CJK Unified (U+4E00–9FFF) + Extension A (U+3400–4DBF)
    (chr(0x4E00), chr(0x9FFF)), (chr(0x3400), chr(0x4DBF)),
)
_JA_RANGES = (  # Hiragana+Katakana (U+3040–30FF) + CJK Unified (U+4E00–9FFF)
    (chr(0x3040), chr(0x30FF)), (chr(0x4E00), chr(0x9FFF)),
)

_LANG_LABELS = {"zh": "Trung", "ja": "Nhật", "en": "Anh"}


def _cjk_count(text: str, ranges) -> int:
    return sum(1 for ch in text if any(lo <= ch <= hi for lo, hi in ranges))


def _group_bbox(grp) -> tuple[int, int, int, int]:
    """Bbox (x1, y1, x2, y2) bao trọn mọi vùng OCR trong 1 group."""
    rects = [_bbox_xyxy(b) for b, _, _ in grp]
    gx1 = int(min(r[0] for r in rects))
    gy1 = int(min(r[1] for r in rects))
    gx2 = int(max(r[2] for r in rects))
    gy2 = int(max(r[3] for r in rects))
    return gx1, gy1, gx2, gy2


def _best_bubble(ocr_rect, bubbles):
    """Bong bóng phủ ocr_rect nhiều nhất → (bubble | None, coverage)."""
    best_bubble = None
    best_cov = 0.0
    for bubble in bubbles:
        cov = _bubble_coverage(ocr_rect, bubble)
        if cov > best_cov:
            best_cov = cov
            best_bubble = bubble
    return best_bubble, best_cov


class ImageTranslator:
    def __init__(
        self,
        model: str = "qwen3:8b",
        vlm_model: str = "qwen2.5vl:7b",
        font_path: str | None = None,
        use_gpu: bool = True,
        src_lang: str = "zh",
        inpainter: str = "opencv",
        mit_python_path: str | None = None,
        overwrite: bool = False,
        font_scale: float = 0.60,
        cpu_priority: str = "below_normal",
        translation_style: str = "modern",
        llm_base_url: str = "",
        llm_api_type: str = "ollama",
        image_quality: int = 95,
        on_log=None,
        on_progress=None,
    ):
        self.model        = model
        self.vlm_model     = vlm_model or "qwen2.5vl:7b"
        self.font_path    = font_path
        self.use_gpu      = use_gpu
        self.src_lang     = src_lang if src_lang in ("zh", "en", "ja") else "zh"
        self.inpainter    = inpainter if inpainter in ("opencv", "lama", "lama_large") else "opencv"
        self.mit_python_path = mit_python_path or None
        self.overwrite    = overwrite
        self.font_scale   = max(0.3, min(2.0, float(font_scale)))
        self.translation_style = translation_style if translation_style in ("modern", "wuxia", "school", "lightnovel") else "modern"
        self.llm_base_url = llm_base_url or ""
        self.llm_api_type = llm_api_type if llm_api_type in ("ollama", "openai_compat") else "ollama"
        self.cpu_priority = cpu_priority if cpu_priority in ("normal", "below_normal", "idle") else "below_normal"
        # Mức nén ảnh đầu ra: 40–100. <100 = giảm dung lượng (PNG → JPEG nén).
        try:
            self.image_quality = max(40, min(100, int(image_quality)))
        except (TypeError, ValueError):
            self.image_quality = 95
        self.on_log       = on_log or print
        self.on_progress  = on_progress or (lambda d, t: None)

        # Context history: lưu (original, translated) để duy trì nhất quán xưng hô
        # deque(maxlen=12) tự động giới hạn kích thước, không cần popleft() thủ công
        self.context_history: deque[tuple[str, str]] = deque(maxlen=12)

    def _log(self, msg: str):
        self.on_log(msg)

    def _update_context_history(self, original: str, translated: str) -> None:
        """Thêm cặp (original, translated) vào context_history."""
        if original.strip() and translated.strip():
            self.context_history.append((original.strip(), translated.strip()))

    def process_image(self, src: Path, dst: Path) -> bool:
        """Xử lý một ảnh: OCR → inpaint → dịch → render → lưu."""
        try:
            import cv2
            import numpy as np
            self._log(f"  [OCR] {src.name}")

            img_orig = cv2.imdecode(np.fromfile(str(src), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img_orig is None:
                raise ValueError("Không đọc được file ảnh")

            bubbles, raw_results = self._detect_and_ocr(img_orig)
            raw_results, watermark_boxes = self._filter_watermarks(raw_results, img_orig.shape)

            results = self._filter_by_language(raw_results, src.name)
            if not results:
                shutil.copy2(src, dst)
                return True

            src_lbl = _LANG_LABELS.get(self.src_lang, "Anh")
            self._log(f"  [OCR] {len(results)} vùng text {src_lbl} (sau lọc)")

            groups = _group_nearby_regions(results)
            self._log(f"  [OCR] → {len(groups)} block")

            all_texts = [t for grp in groups for _, t, _ in grp]
            constraints = self._build_constraints(groups, bubbles)
            all_trans = self._translate_groups(all_texts, constraints)

            idx = 0
            group_trans: list[list[str]] = []
            for grp in groups:
                n = len(grp)
                group_trans.append(all_trans[idx:idx + n])
                idx += n

            regions = self._build_regions(groups, watermark_boxes, img_orig.shape)
            img = self._inpaint_all(img_orig.copy(), regions)
            img_pil = self._render_groups(img, groups, group_trans, bubbles, img_orig.shape)

            written = save_image_compressed(img_pil, dst, self.image_quality)

            self._log(f"  [OK] → {written.name}")
            return True

        except Exception as exc:
            import traceback
            self._log(f"  [FAIL] {src.name}: {exc}")
            self._log(traceback.format_exc())
            try:
                shutil.copy2(src, dst)
            except OSError as copy_exc:
                self._log(f"  [FAIL] Không copy được ảnh gốc: {copy_exc}")
            return False

    def _detect_and_ocr(self, img_orig):
        """Detect bong bóng + VLM OCR 2 lượt (full-page + crop) → (bubbles, raw_results)."""
        bubbles = _find_speech_bubbles(img_orig)
        self._log(f"  [OCR] Phát hiện {len(bubbles)} bong bóng để OCR crop")
        raw_results = _run_vlm_ocr(
            img_orig,
            model=self.vlm_model,
            llm_base_url=self.llm_base_url,
            llm_api_type=self.llm_api_type,
            src_lang=self.src_lang,
            on_log=self._log,
            bubbles=bubbles,
        )
        self._log(f"  [OCR] VLM ({self.vlm_model}) đọc được {len(raw_results)} vùng thô (full+crop)")
        return bubbles, raw_results

    def _filter_watermarks(self, raw_results, img_shape):
        """Tách vùng watermark/logo khỏi kết quả OCR → (raw_results, watermark_boxes).

        Mở rộng vùng watermark để gom cả các bbox lân cận (chữ ký/URL thường
        vỡ thành nhiều mảnh nhỏ quanh logo).
        """
        watermark_boxes = [b for b, t, c in raw_results if _looks_like_watermark(t)]
        if not watermark_boxes:
            return raw_results, watermark_boxes
        watermark_region = _union_bboxes(watermark_boxes)
        expanded_rect = _rect_expand(_bbox_xyxy(watermark_region), img_shape, pad=_WM_NEIGHBOR_PAD)
        self._log(f"  [OCR] Phát hiện {len(watermark_boxes)} watermark/logo (xóa)")
        extra_boxes = []
        for b, t, c in raw_results:
            if b in watermark_boxes:
                continue
            rect = _bbox_xyxy(b)
            if _rect_intersects(rect, expanded_rect):
                extra_boxes.append(b)
        if extra_boxes:
            self._log(f"  [OCR] Mở rộng {len(extra_boxes)} bbox gần watermark")
            watermark_boxes.extend(extra_boxes)
        raw_results = [
            (b, t, c) for b, t, c in raw_results
            if b not in watermark_boxes and not _rect_intersects(_bbox_xyxy(b), expanded_rect)
        ]
        return raw_results, watermark_boxes

    def _filter_by_language(self, raw_results, src_name):
        """Lọc vùng OCR theo ngôn ngữ nguồn, log ✓/✗ từng vùng.

        Trả về [] khi không còn vùng nào (đã log [SKIP]) — caller copy ảnh gốc.
        """
        if self.src_lang == "zh":
            detect, key = has_chinese, "zh"
            flag_ok = keep = lambda t, c: has_chinese(t) and (
                c > _MIN_CONF_CJK or _cjk_count(t, _ZH_RANGES) >= _MIN_ZH_CHARS)
        elif self.src_lang == "ja":
            detect, key = has_japanese, "ja"
            flag_ok = keep = lambda t, c: has_japanese(t) and (
                c > _MIN_CONF_CJK or _cjk_count(t, _JA_RANGES) >= _MIN_JA_CHARS)
        else:
            detect, key = has_english, "en"
            flag_ok = lambda t, c: c > _MIN_CONF_EN and has_english(t)
            keep = lambda t, c: flag_ok(t, c) and len(t.strip()) >= _MIN_EN_LEN

        for b, t, c in raw_results:
            flag = "✓" if flag_ok(t, c) else "✗"
            self._log(f"    {flag} conf={c:.2f} {key}={detect(t)} text={t!r:.50}")
        results = [(b, t, c) for b, t, c in raw_results if keep(t, c)]
        if not results:
            src_lbl = _LANG_LABELS.get(self.src_lang, "Anh")
            self._log(f"  [SKIP] Không có text {src_lbl}: {src_name}")
        return results

    def _build_constraints(self, groups, bubbles):
        """max_chars/max_lines cho từng vùng, theo cỡ bong bóng bao quanh group."""
        constraints: list[dict] = []
        for grp in groups:
            bubble, cov = _best_bubble(_group_bbox(grp), bubbles)
            if bubble and cov > _BUBBLE_COV_MIN:
                bw = int(bubble[2] - bubble[0])
                bh = int(bubble[3] - bubble[1])
                max_chars = max(_MIN_MAX_CHARS, bw // _MAX_CHARS_DIV)
                max_lines = max(1, bh // _MAX_LINES_DIV)
                for _ in grp:
                    constraints.append({"max_chars": max_chars, "max_lines": max_lines})
            else:
                for _ in grp:
                    constraints.append({})
        return constraints

    def _translate_groups(self, all_texts, constraints):
        """Dịch batch + post-process + cập nhật context history → list bản dịch."""
        # ── CONTEXT-AWARE TRANSLATION ─────────────────────────────────────────
        # Truyền toàn bộ context_history vào translate_batch để inject vào prompt.
        # translate_batch lấy 5 entries gần nhất từ context_history để xây dựng
        # System Prompt ngữ cảnh, giúp model dịch nhất quán về xưng hô.
        self._log(f"  [TRANS] Đang dịch… (context={len(self.context_history)} entries)")
        all_trans = translate_batch(
            all_texts,
            self.model,
            src_lang=self.src_lang,
            context_history=list(self.context_history),
            constraints=constraints,
            style=self.translation_style,
            llm_base_url=self.llm_base_url,
            llm_api_type=self.llm_api_type,
        )

        # ── POST-PROCESSING + CONTEXT HISTORY UPDATE ──────────────────────────
        # Duyệt qua từng cặp (original, translated), áp dụng post-processing,
        # rồi cập nhật context_history để các câu sau kế thừa ngữ cảnh xưng hô.
        for i, (orig, trans) in enumerate(zip(all_texts, all_trans)):
            # Kiểm tra ký tự CJK (Hán/Nhật) sót lại
            if contains_cjk(trans):
                n_cjk = len(JA_RE.findall(trans))
                self._log(f"    ⚠  {n_cjk} ký tự CJK chưa dịch → post-process")
                trans = comprehensive_post_processing(trans, orig)
            # Áp dụng đại từ nhân xưng theo ngữ cảnh
            trans = post_process_translation(trans, orig, src_lang=self.src_lang)
            all_trans[i] = trans
            # Cập nhật context cho các câu tiếp theo trong ảnh này
            self._update_context_history(orig, trans)
        return all_trans

    def _build_regions(self, groups, watermark_boxes, img_shape):
        """Danh sách vùng cần xoá: (bbox, dilate_ksize, dilate_iters, radius, kind).

        Gộp text + watermark thành 1 danh sách để lama_large có thể xoá cả ảnh
        trong 1 lượt subprocess. kind "text" được thử mask bám nét chữ; "wm"
        luôn tô đặc bbox (watermark phủ lên tranh, không tách nét được).
        """
        regions: list[tuple] = []
        for grp in groups:
            for bbox, _, _ in grp:
                regions.append((bbox, *_TEXT_INPAINT, "text"))

        if watermark_boxes:
            wx1, wy1, wx2, wy2 = _bbox_xyxy(_union_bboxes(watermark_boxes))
            wx1, wy1, wx2, wy2 = _rect_expand((wx1, wy1, wx2, wy2), img_shape, pad=_WM_UNION_PAD)
            union_wm = [[wx1, wy1], [wx2, wy1], [wx2, wy2], [wx1, wy2]]
            regions.append((union_wm, *_WM_UNION_INPAINT, "wm"))
            for bbox in watermark_boxes:
                x1, y1, x2, y2 = _expand_bbox(bbox, img_shape, pad=_WM_LOGO_PAD)
                logo_bbox = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                regions.append((logo_bbox, *_WM_LOGO_INPAINT, "wm"))
        return regions

    def _inpaint_all(self, img, regions):
        """Xoá text/watermark: lama_large (subprocess 1 lần/ảnh) hoặc inpaint gộp mask."""
        if self.inpainter == "lama_large":
            result = inpaint_regions_lama_large(
                img, [(b, dk, di, kind) for b, dk, di, _, kind in regions],
                python_path=self.mit_python_path,
                use_gpu=self.use_gpu, on_log=self._log,
            )
            if result is not None:
                return result
            return inpaint_regions(img, regions, method="opencv", on_log=self._log)
        return inpaint_regions(img, regions, method=self.inpainter, on_log=self._log)

    def _render_groups(self, img, groups, group_trans, bubbles, img_shape):
        """Vẽ bản dịch lên ảnh đã inpaint; neo vào bong bóng nếu đủ coverage."""
        import cv2
        from PIL import Image

        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        for grp_idx, (grp, trans_list) in enumerate(zip(groups, group_trans)):
            gx1, gy1, gx2, gy2 = _group_bbox(grp)
            full_text = "\n".join(t for t in trans_list if t.strip())

            bubble, cov = _best_bubble((gx1, gy1, gx2, gy2), bubbles)
            if bubble and cov > _BUBBLE_COV_MIN:
                rx1, ry1, rx2, ry2 = bubble
            else:
                ih_img, iw_img = img_shape[:2]
                pad_x = max(_RENDER_PAD_MIN, int((gx2 - gx1) * _RENDER_PAD_X_FRAC))
                pad_y = max(_RENDER_PAD_MIN, int((gy2 - gy1) * _RENDER_PAD_Y_FRAC))
                rx1 = max(0,      gx1 - pad_x)
                ry1 = max(0,      gy1 - pad_y)
                rx2 = min(iw_img, gx2 + pad_x)
                ry2 = min(ih_img, gy2 + pad_y)

            merged_bbox = [[rx1, ry1], [rx2, ry1], [rx2, ry2], [rx1, ry2]]
            img_pil = render_text(img_pil, merged_bbox, full_text, self.font_path,
                                  strict_clip=(cov > _BUBBLE_COV_MIN),
                                  font_scale=self.font_scale,
                                  bbox_index=grp_idx)
        return img_pil

    def process_folder(
        self,
        input_dir: str,
        output_dir: str,
        stop_event: threading.Event | None = None,
        images_override: list | None = None,
    ) -> tuple[int, int, list[str]]:
        """Xử lý toàn bộ ảnh trong input_dir, lưu vào output_dir."""
        inp = Path(input_dir)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        if images_override is not None:
            images = [Path(p) for p in images_override if Path(p).suffix.lower() in IMAGE_EXTS]
        else:
            images = sorted(f for f in inp.iterdir() if f.suffix.lower() in IMAGE_EXTS)

        if not images:
            self._log("Không tìm thấy ảnh trong thư mục.")
            return 0, 0, []

        total = len(images)
        self._log(f"Tổng: {total} ảnh cần xử lý")
        ok = fail = 0
        failed_paths: list[str] = []

        for i, path in enumerate(images, 1):
            if stop_event and stop_event.is_set():
                self._log("⚠  Đã dừng theo yêu cầu.")
                break
            dst = out / path.name
            # Khi nén PNG→JPEG (quality<100) file đầu ra đổi đuôi .jpg → cũng tính là đã có.
            already = dst.exists() or (
                self.image_quality < 100
                and path.suffix.lower() not in (".jpg", ".jpeg", ".webp")
                and dst.with_suffix(".jpg").exists()
            )
            if not self.overwrite and already:
                self._log(f"  [SKIP] Đã có: {path.name}")
                ok += 1
                self.on_progress(i, total)
                continue
            success = self.process_image(path, dst)
            if success:
                ok += 1
            else:
                fail += 1
                failed_paths.append(str(path))
            self.on_progress(i, total)
        return ok, fail, failed_paths
