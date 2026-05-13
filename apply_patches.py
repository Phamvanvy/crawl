"""
apply_patches.py — Apply custom patches to manga_translator after install.

Usage:
    python apply_patches.py

Run this once after:
    pip install git+https://github.com/zyddnys/manga-image-translator.git
    (or after re-creating mit_venv)

Patches applied:
  1. manga_translator/rendering/__init__.py
     - Use target_font_size (after offset) for word-wrap calculation instead of
       original region.font_size — prevents bbox from expanding sideways into
       character art when Vietnamese text is longer than Chinese source.
     - Re-enable boundary clipping so text never overflows past image edges.
  2. manga_translator/rendering/text_render.py
     - Guard against empty line_width_list in put_text_horizontal to prevent
       ValueError: max() arg is an empty sequence when rendering invisible
       characters like ZWJ (U+200D) used for watermark erasure.
  3. manga_translator/translators/custom_openai.py
     - Custom OpenAI-compatible translator with Vietnamese enforcement.
     - Watermark detection → ZWJ fallback for inpainting without rendering.
     - Retry up to 10 times when output is not Vietnamese.
"""

import json
import shutil
import sys
from pathlib import Path
from urllib.parse import urlparse, unquote

PATCHES_DIR = Path(__file__).parent / "patches"

def find_site_packages() -> Path | None:
    """Find manga_translator in mit_venv or current venv."""
    candidates = [
        Path(__file__).parent / "mit_venv" / "Lib" / "site-packages",
        Path(__file__).parent / "mit_venv" / "lib",
    ]
    # Also check current venv
    for p in sys.path:
        sp = Path(p)
        if sp.name == "site-packages" and (sp / "manga_translator").exists():
            candidates.insert(0, sp)

    def _detect_editable_source(sp: Path) -> Path | None:
        for dist in sp.glob("manga_image_translator-*.dist-info"):
            direct_url = dist / "direct_url.json"
            if not direct_url.exists():
                continue
            try:
                data = json.loads(direct_url.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not data.get("dir_info", {}).get("editable"):
                continue
            url = data.get("url", "")
            parsed = urlparse(url)
            if parsed.scheme != "file":
                continue
            path = unquote(parsed.path)
            if path.startswith("/") and len(path) > 3 and path[2] == ":":
                path = path.lstrip("/")
            src = Path(path)
            if (src / "manga_translator").exists():
                return src
        return None

    for sp in candidates:
        if (sp / "manga_translator").exists():
            return sp
        editable_src = _detect_editable_source(sp)
        if editable_src is not None:
            return editable_src
        # Linux layout: site-packages is inside lib/pythonX.Y/
        for child in sp.glob("python*/site-packages"):
            if (child / "manga_translator").exists():
                return child
            editable_src = _detect_editable_source(child)
            if editable_src is not None:
                return editable_src
    return None


def apply():
    sp = find_site_packages()
    if sp is None:
        print("[FAIL] Cannot find manga_translator in any venv. Install it first.")
        sys.exit(1)

    patches = [
        (
            PATCHES_DIR / "manga_translator_rendering_init.py",
            sp / "manga_translator" / "rendering" / "__init__.py",
        ),
        (
            PATCHES_DIR / "manga_translator_rendering_text_render.py",
            sp / "manga_translator" / "rendering" / "text_render.py",
        ),
        (
            PATCHES_DIR / "manga_translator_translators_custom_openai.py",
            sp / "manga_translator" / "translators" / "custom_openai.py",
        ),
    ]

    for src, dst in patches:
        if not src.exists():
            print(f"[SKIP] Patch source not found: {src}")
            continue
        if not dst.parent.exists():
            print(f"[SKIP] Target directory not found: {dst.parent}")
            continue
        shutil.copy2(src, dst)
        print(f"[OK]   {dst.relative_to(sp)}")

    print("\nAll patches applied.")


if __name__ == "__main__":
    apply()
