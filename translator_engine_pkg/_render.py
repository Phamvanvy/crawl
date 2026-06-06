"""
_render.py — Font loading, text wrapping, và rendering text lên ảnh PIL.
"""

from pathlib import Path

from ._ocr import has_chinese, has_english, _ZH_RE
from ._utils import _bbox_xyxy

# ── Font cache ────────────────────────────────────────────────────────────────
_font_cache: dict = {}

# Dấu câu không bao giờ được đứng đầu dòng (lưu ý 7)
_PUNCT_HEAD = set(".,!?;:…—–»")

# ── Danh sách font Việt hóa trong thư mục fonts/ (lưu ý 2, 4: đa dạng font)
_FONTS_DIR = Path(__file__).parent.parent / "fonts"

def _list_vi_fonts() -> list[str]:
    """Quét .ttf/.otf trong fonts/ theo thứ tự ưu tiên: MTO trước, rồi các font khác."""
    if not _FONTS_DIR.exists():
        return []
    all_fonts = sorted(str(p) for p in _FONTS_DIR.glob("*.ttf")) + \
                sorted(str(p) for p in _FONTS_DIR.glob("*.otf"))
    # Ưu tiên MTO trước
    priority = [str(_FONTS_DIR / "MTO Astro City.ttf")]
    rest = [f for f in all_fonts if f not in priority and "BeVietnamPro" not in f and "NotoSans" not in f]
    fallback = [f for f in all_fonts if "BeVietnamPro" in f or "NotoSans" in f]
    result = [f for f in priority if Path(f).exists()] + rest + fallback
    return [f for f in result if Path(f).exists()]

_VI_FONTS: list[str] = _list_vi_fonts()


def _load_font(font_path: str | None, size: int):
    """Lazy-load ImageFont, ưu tiên MTO Astro City, bỏ font hệ thống mặc định (lưu ý 2)."""
    from PIL import ImageFont  # lazy
    key = (font_path, size)
    if key in _font_cache:
        return _font_cache[key]
    candidates = []
    if font_path:
        candidates.append(font_path)
    # Ưu tiên toàn bộ font Việt trong fonts/ trước khi fallback hệ thống
    candidates += _VI_FONTS
    candidates += [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
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


def _pick_font_path(bbox_index: int = 0) -> str | None:
    """Chọn font theo bbox_index để đa dạng font giữa các bong bóng (lưu ý 4)."""
    if not _VI_FONTS:
        return None
    return _VI_FONTS[bbox_index % len(_VI_FONTS)]


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _draw_text_with_shadow(draw, pos, text, font, text_color, shadow_color, stroke_width: int = 2):
    tx, ty = pos
    draw.text((tx, ty), text, font=font, fill=text_color,
              stroke_width=stroke_width, stroke_fill=shadow_color)


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

    # Lưu ý 7: Dấu câu không bao giờ đứng đầu dòng — merge vào dòng trước
    merged: list[str] = []
    for line in lines:
        stripped = line.lstrip()
        if merged and stripped and stripped[0] in _PUNCT_HEAD:
            merged[-1] = merged[-1].rstrip() + stripped
        else:
            merged.append(line)

    return merged or [""]


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
                strict_clip: bool = False, font_scale: float = 1.0,
                bbox_index: int = 0, text_color=None, stroke_color=None,
                font_px: int | None = None, rotate: float = 0.0):
    """Vẽ text ngang vào vùng bbox với pixel-accurate word wrap.

    Dùng crop-draw-paste để text không thể tràn ra ngoài bbox (lưu ý 5, 6).
    bbox_index: dùng để chọn font đa dạng giữa các bong bóng (lưu ý 4).
    text_color/stroke_color: (R,G,B) ghi đè màu chữ/viền (None = tự chọn theo nền).
    rotate: góc nghiêng chữ (độ, dương = ngược chiều kim đồng hồ). Khi != 0, chữ được
    vẽ lên lớp trong suốt rồi xoay quanh tâm bbox (có thể tràn nhẹ ra ngoài khung).
    """
    from PIL import ImageDraw, Image
    import numpy as np
    if not text.strip():
        return img_pil

    # Lưu ý 2, 4: Ưu tiên font Việt hóa, xoay vòng theo bbox_index nếu font_path không chỉ định
    if not font_path:
        font_path = _pick_font_path(bbox_index)

    if img_pil.mode != "RGB":
        img_pil = img_pil.convert("RGB")

    x1, y1, x2, y2 = _bbox_xyxy(bbox)
    x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
    iw, ih = img_pil.size
    # Clamp bbox vào giới hạn ảnh
    x1i = max(0, x1i); y1i = max(0, y1i)
    x2i = min(iw, x2i); y2i = min(ih, y2i)
    bw = max(x2i - x1i, 40)
    bh = max(y2i - y1i, 16)

    # Crop vùng bbox ra để lấy màu nền; chữ vẽ lên lớp RGBA trong suốt riêng để có
    # thể xoay nghiêng mà không kéo theo nền.
    region = img_pil.crop((x1i, y1i, x1i + bw, y1i + bh))
    layer  = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
    draw   = ImageDraw.Draw(layer)

    # Lấy màu nền từ region để chọn màu chữ tương phản
    arr = np.array(region)
    bg_rgb = tuple(int(v) for v in np.median(arr.reshape(-1, 3), axis=0))

    brightness   = sum(bg_rgb) / 3
    auto_text    = (255, 255, 255) if brightness < 140 else (0, 0, 0)
    auto_shadow  = (0, 0, 0)       if brightness < 140 else (255, 255, 255)
    # Override màu nếu được chỉ định (vd từ Font color trong cài đặt); None = tự chọn.
    text_color   = tuple(text_color) if text_color is not None else auto_text
    shadow_color = tuple(stroke_color) if stroke_color is not None else auto_shadow
    # Vùng tối: stroke dày hơn để chữ trắng nổi bật trên nền đen
    # Vùng sáng: stroke mỏng hơn để không làm loè nền trắng
    stroke_width = 7 if brightness < 80 else (5 if brightness < 140 else 2)

    pad     = max(4, int(min(bw, bh) * 0.07))
    inner_w = max(bw - pad * 2, 20)
    inner_h = max(bh - pad * 2, 12)

    # Khi strict_clip=False, bbox đã được expand ~30-40% so với OCR text thực tế.
    # Dùng 80% inner để tính font/wrap, cho phép chữ to hơn và fit bubble tốt hơn.
    if strict_clip:
        render_w = inner_w
        render_h = inner_h
        abs_max_font = 26
    else:
        render_w = max(20, int(inner_w * 0.80))
        render_h = max(12, int(inner_h * 0.80))
        abs_max_font = 22

    max_start = max(min(render_h // 4, abs_max_font), 6)
    max_start = max(6, int(max_start * font_scale))
    wrap_w = render_w

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
    # font_px: ép cỡ chữ cố định (vùng gõ tay chọn cỡ) — bỏ qua vòng tự canh.
    # Vẫn wrap theo bề rộng (hard-split) và clip bằng crop-paste nếu tràn.
    if font_px is not None and int(font_px) >= 6:
        best_font  = _load_font(font_path, int(font_px))
        best_lines = _wrap_text_px(draw, text, best_font, max(10, render_w), allow_hard_split=True) or [text]
        try:
            bb = draw.textbbox((0, 0), sample_text, font=best_font)
            best_lh = bb[3] - bb[1] + 3
        except Exception:
            best_lh = int(font_px) + 3
    else:
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
                if (lh + 2) * len(lines) <= render_h:
                    best_font  = font
                    best_lines = lines
                    best_lh    = lh
                    wrap_w = current_wrap
                    break
            if best_font is not None:
                break

    if best_font is None:
        best_font  = _load_font(font_path, 6)
        best_lines = _wrap_text_px(draw, text, best_font, max(10, int(render_w * 0.55)), allow_hard_split=True)
        if best_lines is None:
            best_lines = [text]
        try:
            bb = draw.textbbox((0, 0), sample_text, font=best_font)
            best_lh = bb[3] - bb[1] + 2
        except Exception:
            best_lh = 8

    line_spacing = max(2, best_lh // 5)
    max_lines = max(1, render_h // (best_lh + line_spacing))
    if len(best_lines) > max_lines:
        best_lines = best_lines[:max_lines]
        if best_lines:
            tail = best_lines[-1].rstrip()
            best_lines[-1] = (tail + "...") if tail else "..."

    total_h = len(best_lines) * (best_lh + line_spacing) - line_spacing

    # Lưu ý 1: Không bôi trắng — inpainter đã xử lý nền, chỉ vẽ chữ lên.
    # Lưu ý 5, 6: Căn giữa dọc + ngang trong bong bóng, dùng tọa độ tương đối so với region
    ty = max(pad, (bh - total_h) // 2)
    for line in best_lines:
        try:
            lb = draw.textbbox((0, 0), line, font=best_font)
            lw = lb[2] - lb[0]
        except Exception:
            lw = len(line) * best_lh // 2
        # Căn giữa ngang — tx là tọa độ trong region, không thể vượt quá bw
        tx = max(pad, (bw - lw) // 2)
        _draw_text_with_shadow(draw, (tx, ty), line, best_font, text_color, shadow_color, stroke_width)
        ty += best_lh + line_spacing

    # Paste lớp chữ (RGBA) trở lại ảnh gốc, dùng alpha làm mask.
    try:
        ang = float(rotate)
    except (TypeError, ValueError):
        ang = 0.0
    if abs(ang) > 0.01:
        # Xoay quanh tâm, expand để không cắt góc; căn lại vào tâm bbox (tràn nhẹ OK).
        rot = layer.rotate(ang, resample=Image.BICUBIC, expand=True)
        px = x1i + (bw - rot.width) // 2
        py = y1i + (bh - rot.height) // 2
        img_pil.paste(rot, (px, py), rot)
    else:
        img_pil.paste(layer, (x1i, y1i), layer)
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
