"""
_render.py — Font loading, text wrapping, và rendering text lên ảnh PIL.
"""

from pathlib import Path

from ._ocr import has_chinese, has_english, _ZH_RE
from ._utils import _bbox_xyxy

# ── Font cache ────────────────────────────────────────────────────────────────
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
    candidates += [
        str(Path(__file__).parent.parent / "fonts" / "BeVietnamPro-Regular.ttf"),
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyh.ttf",
        r"C:\Windows\Fonts\simhei.ttf",
        str(Path(__file__).parent.parent / "fonts" / "NotoSans-Regular.ttf"),
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


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _draw_text_with_shadow(draw, pos, text, font, text_color, shadow_color):
    tx, ty = pos
    draw.text((tx, ty), text, font=font, fill=text_color,
              stroke_width=1, stroke_fill=shadow_color)


def _wrap_text_px(draw, text: str, font, max_px: int, allow_hard_split: bool = False) -> list | None:
    """Word-wrap `text` so each line fits within `max_px` pixels."""
    def width_of(value: str) -> int:
        try:
            bb = draw.textbbox((0, 0), value, font=font)
            return bb[2] - bb[0]
        except Exception:
            return len(value) * max(getattr(font, "size", 10), 8)

    def split_long_word(word: str) -> list[str]:
        if width_of(word) <= max_px:
            return [word]
        parts: list[str] = []
        current = ""
        for ch in word:
            if width_of(current + ch) <= max_px or not current:
                current += ch
            else:
                parts.append(current)
                current = ch
        if current:
            parts.append(current)
        return parts or [word]

    if text is None:
        return [""]

    lines: list[str] = []
    for paragraph in text.splitlines():
        if not paragraph:
            lines.append("")
            continue
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current: list[str] = []
        for word in words:
            candidate = " ".join(current + [word]) if current else word
            if width_of(candidate) <= max_px:
                current.append(word)
                continue
            if not current:
                if not allow_hard_split:
                    return None
                lines.extend(split_long_word(word))
            else:
                lines.append(" ".join(current))
                if width_of(word) <= max_px:
                    current = [word]
                else:
                    if not allow_hard_split:
                        return None
                    lines.extend(split_long_word(word))
                    current = []
        if current:
            lines.append(" ".join(current))

    return lines or [""]


def _render_line_height_sample(text: str) -> str:
    if has_chinese(text):
        for ch in text:
            if '\u4e00' <= ch <= '\u9fff' or '\u3400' <= ch <= '\u4dbf':
                return ch
        return "測"
    if has_english(text):
        return "Ab"
    return "Ắp"


def render_text(img_pil, bbox, text: str, font_path: str | None,
                strict_clip: bool = False, font_scale: float = 1.0):
    """Vẽ text ngang vào vùng bbox với pixel-accurate word wrap."""
    from PIL import ImageDraw
    import numpy as np
    if not text.strip():
        return img_pil

    if img_pil.mode != "RGB":
        img_pil = img_pil.convert("RGB")

    draw = ImageDraw.Draw(img_pil)
    x1, y1, x2, y2 = _bbox_xyxy(bbox)
    x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
    iw, ih = img_pil.size
    bw = max(x2i - x1i, 40)
    bh = max(y2i - y1i, 16)

    arr = np.array(img_pil.convert("RGB"))
    ry1, ry2 = max(0, y1i), min(ih, y2i)
    rx1, rx2 = max(0, x1i), min(iw, x2i)
    if ry2 > ry1 and rx2 > rx1:
        roi = arr[ry1:ry2, rx1:rx2]
        bg_rgb = tuple(int(v) for v in np.median(roi.reshape(-1, 3), axis=0))
    else:
        bg_rgb = (20, 20, 20)

    brightness   = sum(bg_rgb) / 3
    text_color   = (255, 255, 255) if brightness < 140 else (0, 0, 0)
    shadow_color = (0, 0, 0)       if brightness < 140 else (255, 255, 255)

    pad     = max(4, int(min(bw, bh) * 0.07))
    inner_w = max(bw - pad * 2, 20)
    inner_h = max(bh - pad * 2, 12)
    max_start = max(min(inner_h // 4, 24), 8)
    if not strict_clip:
        max_start = min(max_start, max(8, inner_h // 5))
    max_start = max(6, int(max_start * font_scale))
    wrap_w  = inner_w if strict_clip else min(inner_w, 180)

    best_font  = None
    best_lines = [text]
    best_lh    = 11
    wrap_candidates = [
        wrap_w,
        max(10, int(wrap_w * 0.85)),
        max(10, int(wrap_w * 0.70)),
        max(10, int(wrap_w * 0.55)),
    ]

    sample_text = _render_line_height_sample(text)
    for size in range(max_start, 4, -1):
        font = _load_font(font_path, size)
        try:
            bb = draw.textbbox((0, 0), sample_text, font=font)
            lh = bb[3] - bb[1] + 3
        except Exception:
            lh = size + 3
        for current_wrap in wrap_candidates:
            lines = _wrap_text_px(draw, text, font, current_wrap, allow_hard_split=False)
            if lines is None:
                continue
            if (lh + 2) * len(lines) <= inner_h:
                best_font  = font
                best_lines = lines
                best_lh    = lh
                wrap_w = current_wrap
                break
        if best_font is not None:
            break

    if best_font is None:
        best_font  = _load_font(font_path, 6)
        best_lines = _wrap_text_px(draw, text, best_font, max(10, int(inner_w * 0.55)), allow_hard_split=True)
        if best_lines is None:
            best_lines = [text]
        try:
            bb = draw.textbbox((0, 0), sample_text, font=best_font)
            best_lh = bb[3] - bb[1] + 2
        except Exception:
            best_lh = 8

    line_spacing = max(2, best_lh // 5)
    max_lines = max(1, inner_h // (best_lh + line_spacing))
    if len(best_lines) > max_lines:
        best_lines = best_lines[:max_lines]
        if best_lines:
            tail = best_lines[-1].rstrip()
            best_lines[-1] = (tail + "...") if tail else "..."

    total_h   = len(best_lines) * (best_lh + line_spacing) - line_spacing
    actual_y2 = min(y2i, y1i + total_h + pad * 2)

    draw.rectangle([x1i - 3, y1i - 3, x2i + 3, actual_y2 + 3], fill=bg_rgb)

    ty = y1i + max(pad, (actual_y2 - y1i - total_h) // 2)
    for line in best_lines:
        try:
            lb = draw.textbbox((0, 0), line, font=best_font)
            lw = lb[2] - lb[0]
        except Exception:
            lw = len(line) * best_lh // 2
        tx = x1i + max(pad, (bw - lw) // 2)
        _draw_text_with_shadow(draw, (tx, ty), line, best_font, text_color, shadow_color)
        ty += best_lh + line_spacing

    return img_pil


def _group_nearby_regions(results: list, gap_px: int = 18) -> list:
    """Gom các OCR bbox gần nhau (cùng cột/dòng) thành một group."""
    import numpy as np
    if not results:
        return []

    def to_xyxy(bbox):
        pts = np.array(bbox, dtype=np.float32)
        return (float(pts[:, 0].min()), float(pts[:, 1].min()),
                float(pts[:, 0].max()), float(pts[:, 1].max()))

    enriched = [(to_xyxy(b), b, t, c) for b, t, c in results]
    enriched.sort(key=lambda x: (x[0][1], x[0][0]))
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
