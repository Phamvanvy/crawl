"""
translator_engine_pkg/__init__.py

Khởi tạo ML thread limits trước tất cả các import nặng,
sau đó re-export toàn bộ public API để tương thích với:
    import translator_engine as te
    from translator_engine_pkg import *
"""

import os as _os_env

# Giới hạn thread của ML libs trước khi import bất cứ thứ gì
_ML_THREAD_LIMIT = "2"
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os_env.environ.setdefault(_k, _ML_THREAD_LIMIT)

# ── Re-exports ────────────────────────────────────────────────────────────────
from ._ocr import (
    _get_paddle_reader,
    _get_easyocr_reader,
    _get_reader,
    _reader_lock,
    _paddle_readers,
    _paddle_lock,
    _easyocr_readers,
    _easyocr_lock,
    _preprocess_for_ocr,
    _run_ocr_engine,
    _find_speech_bubbles,
    _rect_of_bbox,
    _iou_rect,
    _bubble_coverage,
    _run_ocr,
    has_chinese,
    has_english,
    _ZH_RE,
    _EN_RE,
)

from ._utils import (
    _URL_LIKE_RE,
    _looks_like_watermark,
    _bbox_xyxy,
    _union_bboxes,
    _rect_expand,
    _rect_intersects,
    _expand_bbox,
    check_lama_available,
    _get_lama,
)

from ._inpaint import inpaint_region

from ._translate import (
    OLLAMA_BASE,
    check_ollama,
    _normalize_newlines,
    _normalize_vietnamese,
    translate_batch,
)

from ._render import (
    _font_cache,
    _load_font,
    _draw_text_with_shadow,
    _wrap_text_px,
    _render_line_height_sample,
    render_text,
    _group_nearby_regions,
)

from ._image_translator import ImageTranslator, IMAGE_EXTS

from ._mit_backend import (
    _MIT_INSTALL_HINT,
    _find_mit_python,
    check_mit,
    MITImageTranslator,
)

__all__ = [
    # ocr
    "has_chinese", "has_english", "_run_ocr", "_run_ocr_engine",
    "_find_speech_bubbles", "_iou_rect", "_bubble_coverage",
    "_ZH_RE", "_EN_RE",
    # utils
    "_looks_like_watermark", "_bbox_xyxy", "_union_bboxes",
    "_rect_expand", "_rect_intersects", "_expand_bbox",
    "check_lama_available",
    # inpaint
    "inpaint_region",
    # translate
    "OLLAMA_BASE", "check_ollama", "translate_batch",
    "_normalize_newlines", "_normalize_vietnamese",
    # render
    "_load_font", "render_text", "_group_nearby_regions",
    # main classes
    "ImageTranslator", "IMAGE_EXTS",
    "MITImageTranslator", "check_mit", "_MIT_INSTALL_HINT",
]
