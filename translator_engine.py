"""
translator_engine.py — Compatibility shim.
Logic đã được tách vào translator_engine_pkg/ (các module nhỏ hơn 1k dòng):
  _ocr.py            — OCR engines, bubble detection
  _utils.py          — bbox/rect helpers, watermark, LAMA check
  _inpaint.py        — inpaint_region
  _translate.py      — Ollama helpers, translate_batch
  _render.py         — font loading, text rendering
  _image_translator.py — ImageTranslator class
  _mit_backend.py    — MITImageTranslator class

web_app.py tiếp tục dùng: import translator_engine as te
"""

# Re-export toàn bộ public API từ package (ML thread limits đặt trong __init__.py)
from translator_engine_pkg import *  # noqa: F401, F403
from translator_engine_pkg import (  # noqa: F401 — explicit for type-checkers
    OLLAMA_BASE, IMAGE_EXTS,
    has_chinese, has_english,
    check_ollama, check_lama_available, check_mit,
    translate_batch, inpaint_region, render_text,
    ImageTranslator, MITImageTranslator,
    _MIT_INSTALL_HINT, _find_mit_python,
)