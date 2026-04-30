"""
_image_translator.py — ImageTranslator: OCR → inpaint → dịch → render.
"""

import shutil
import threading
from pathlib import Path

from ._ocr import (
    _run_ocr, _find_speech_bubbles, has_chinese, has_english, _bubble_coverage,
)
from ._utils import (
    _bbox_xyxy, _union_bboxes, _rect_expand, _rect_intersects, _expand_bbox,
    _looks_like_watermark,
)
from ._inpaint import inpaint_region
from ._translate import translate_batch
from ._render import render_text, _group_nearby_regions

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class ImageTranslator:
    def __init__(
        self,
        model: str = "qwen3:8b",
        font_path: str | None = None,
        use_gpu: bool = True,
        src_lang: str = "zh",
        inpainter: str = "opencv",
        overwrite: bool = False,
        font_scale: float = 0.60,
        cpu_priority: str = "below_normal",
        on_log=None,
        on_progress=None,
    ):
        self.model        = model
        self.font_path    = font_path
        self.use_gpu      = use_gpu
        self.src_lang     = src_lang if src_lang in ("zh", "en") else "zh"
        self.inpainter    = inpainter if inpainter in ("opencv", "lama") else "opencv"
        self.overwrite    = overwrite
        self.font_scale   = max(0.3, min(2.0, float(font_scale)))
        self.cpu_priority = cpu_priority if cpu_priority in ("normal", "below_normal", "idle") else "below_normal"
        self.on_log       = on_log or print
        self.on_progress  = on_progress or (lambda d, t: None)

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

            watermark_boxes = [b for b, t, c in raw_results if _looks_like_watermark(t)]
            if watermark_boxes:
                watermark_region = _union_bboxes(watermark_boxes)
                expanded_rect = _rect_expand(_bbox_xyxy(watermark_region), img_orig.shape, pad=80)
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
            else:
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

            groups = _group_nearby_regions(results)
            self._log(f"  [OCR] → {len(groups)} block")

            all_texts = [t for grp in groups for _, t, _ in grp]
            constraints: list[dict] = []
            for grp in groups:
                rects = [_bbox_xyxy(b) for b, _, _ in grp]
                gx1 = int(min(r[0] for r in rects))
                gy1 = int(min(r[1] for r in rects))
                gx2 = int(max(r[2] for r in rects))
                gy2 = int(max(r[3] for r in rects))
                ocr_rect = (gx1, gy1, gx2, gy2)
                best_bubble = None
                best_cov = 0.0
                for bubble in bubbles:
                    cov = _bubble_coverage(ocr_rect, bubble)
                    if cov > best_cov:
                        best_cov = cov
                        best_bubble = bubble
                if best_bubble and best_cov > 0.4:
                    bw = int(best_bubble[2] - best_bubble[0])
                    bh = int(best_bubble[3] - best_bubble[1])
                    max_chars = max(6, bw // 8)
                    max_lines = max(1, bh // 14)
                    for _ in grp:
                        constraints.append({"max_chars": max_chars, "max_lines": max_lines})
                else:
                    for _ in grp:
                        constraints.append({})

            self._log("  [TRANS] Đang dịch…")
            all_trans = translate_batch(
                all_texts, self.model, src_lang=self.src_lang, constraints=constraints,
            )
            idx = 0
            group_trans: list[list[str]] = []
            for grp in groups:
                n = len(grp)
                group_trans.append(all_trans[idx:idx + n])
                idx += n

            img = img_orig.copy()
            for grp in groups:
                for bbox, _, _ in grp:
                    img = inpaint_region(img, bbox, method=self.inpainter)

            if watermark_boxes:
                wx1, wy1, wx2, wy2 = _bbox_xyxy(_union_bboxes(watermark_boxes))
                wx1, wy1, wx2, wy2 = _rect_expand((wx1, wy1, wx2, wy2), img_orig.shape, pad=90)
                union_wm = [[wx1, wy1], [wx2, wy1], [wx2, wy2], [wx1, wy2]]
                img = inpaint_region(
                    img, union_wm, method=self.inpainter,
                    dilate_ksize=9, dilate_iters=4, inpaint_radius=12,
                )
                for bbox in watermark_boxes:
                    x1, y1, x2, y2 = _expand_bbox(bbox, img_orig.shape, pad=64)
                    logo_bbox = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    img = inpaint_region(
                        img, logo_bbox, method=self.inpainter,
                        dilate_ksize=9, dilate_iters=4, inpaint_radius=10,
                    )

            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            for grp, trans_list in zip(groups, group_trans):
                rects = [_bbox_xyxy(b) for b, _, _ in grp]
                gx1 = int(min(r[0] for r in rects))
                gy1 = int(min(r[1] for r in rects))
                gx2 = int(max(r[2] for r in rects))
                gy2 = int(max(r[3] for r in rects))
                full_text = "\n".join(t for t in trans_list if t.strip())

                ocr_rect    = (gx1, gy1, gx2, gy2)
                best_bubble = None
                best_cov    = 0.0
                for bubble in bubbles:
                    cov = _bubble_coverage(ocr_rect, bubble)
                    if cov > best_cov:
                        best_cov    = cov
                        best_bubble = bubble
                if best_bubble and best_cov > 0.4:
                    rx1, ry1, rx2, ry2 = best_bubble
                else:
                    ih_img, iw_img = img_orig.shape[:2]
                    pad_x = max(8, int((gx2 - gx1) * 0.30))
                    pad_y = max(8, int((gy2 - gy1) * 0.40))
                    rx1 = max(0,      gx1 - pad_x)
                    ry1 = max(0,      gy1 - pad_y)
                    rx2 = min(iw_img, gx2 + pad_x)
                    ry2 = min(ih_img, gy2 + pad_y)

                merged_bbox = [[rx1, ry1], [rx2, ry1], [rx2, ry2], [rx1, ry2]]
                img_pil = render_text(img_pil, merged_bbox, full_text, self.font_path,
                                      strict_clip=(best_cov > 0.4),
                                      font_scale=self.font_scale)

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
            if not self.overwrite and dst.exists():
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
