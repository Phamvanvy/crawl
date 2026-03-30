"""
setup_translator.py — Cài đặt tất cả dependencies cho chức năng dịch ảnh
Chạy một lần: python setup_translator.py
"""

import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


def pip(*args):
    print(f"    pip {' '.join(args)}")
    subprocess.run([sys.executable, "-m", "pip", *args, "--quiet"], check=False)


def section(n, total, title):
    print(f"\n[{n}/{total}]  {title}")
    print("─" * 52)


def check(label, ok, extra=""):
    mark = "✔" if ok else "✘"
    print(f"  {mark}  {label}" + (f"  ({extra})" if extra else ""))


def main():
    print("=" * 52)
    print("  Image Translator — Setup cho GTX 1660 SUPER")
    print("=" * 52)

    # ── 1. PyTorch CUDA 12.1 ──────────────────────────────────────────────
    section(1, 5, "PyTorch + CUDA 12.1")
    print("  Đang tải (~2 GB lần đầu)…")
    pip(
        "install", "torch", "torchvision",
        "--index-url", "https://download.pytorch.org/whl/cu121",
    )

    # ── 2. PaddleOCR ──────────────────────────────────────────────────────
    section(2, 6, "PaddleOCR  (OCR chính — tốt hơn cho manga/comic)")
    print("  Models OCR sẽ tải tự động lần chạy đầu tiên (~500 MB)")
    pip("install", "paddlepaddle")
    pip("install", "paddleocr")

    # ── 3. EasyOCR (fallback) ─────────────────────────────────────────────
    section(3, 6, "EasyOCR  (fallback nếu PaddleOCR lỗi)")
    pip("install", "easyocr")

    # ── 4. OpenCV ─────────────────────────────────────────────────────────
    section(4, 6, "OpenCV  (inpainting)")
    pip("install", "opencv-python")

    # ── 5. Font tiếng Việt ───────────────────────────────────────────────
    section(5, 6, "Font tiếng Việt (Be Vietnam Pro)")
    font_dir  = Path(__file__).parent / "fonts"
    font_dir.mkdir(exist_ok=True)
    font_path = font_dir / "BeVietnamPro-Regular.ttf"
    if font_path.exists():
        check("Font đã có", True, str(font_path))
    else:
        # Be Vietnam Pro — thiết kế chuyên cho tiếng Việt, hỗ trợ đầy đủ Latin Extended Additional
        urls = [
            "https://github.com/google/fonts/raw/main/ofl/bevietnampro/BeVietnamPro-Regular.ttf",
            "https://raw.githubusercontent.com/be-fonts/be-vietnam-pro/master/fonts/ttf/BeVietnamPro-Regular.ttf",
        ]
        downloaded = False
        for url in urls:
            try:
                print(f"  Đang tải font từ GitHub…")
                urllib.request.urlretrieve(url, font_path)
                check("BeVietnamPro-Regular.ttf", True, str(font_path))
                downloaded = True
                break
            except Exception as e:
                print(f"  Thử URL khác… ({e})")
        if not downloaded:
            print("  ⚠  Không tải được font tự động.")
            print("  → Sẽ dùng Arial / Segoe UI hệ thống (hỗ trợ tiếng Việt trên Windows)")
            print("  → Hoặc tự tải BeVietnamPro-Regular.ttf vào thư mục fonts/")

    # ── 5. Ollama ─────────────────────────────────────────────────────────
    section(6, 6, "Ollama + Qwen2.5:7b")
    if shutil.which("ollama"):
        check("Ollama đã cài", True)
        print("  Đang kéo model Qwen2.5:7b (~4.7 GB, chạy một lần)…")
        result = subprocess.run(["ollama", "pull", "qwen2.5:7b"])
        if result.returncode == 0:
            check("qwen2.5:7b sẵn sàng", True)
        else:
            check("Pull model thất bại", False, "chạy: ollama pull qwen2.5:7b")
    else:
        check("Ollama chưa cài", False)
        print()
        print("  Để cài Ollama:")
        print("  1.  Tải tại: https://ollama.com/download/windows")
        print("  2.  Cài và khởi động Ollama")
        print("  3.  Mở terminal mới, chạy:")
        print("        ollama pull qwen2.5:7b   (~4.7 GB)")
        print()
        print("  Model nhỏ hơn (nhanh hơn nhưng kém hơn):")
        print("        ollama pull qwen2.5:3b   (~2 GB)")

    # ── Kết quả ───────────────────────────────────────────────────────────
    print(f"\n{'='*52}")
    print("  Kết quả kiểm tra:")
    print("─" * 52)

    try:
        import torch
        cuda = torch.cuda.is_available()
        check("PyTorch", True, torch.__version__)
        check("CUDA", cuda, torch.cuda.get_device_name(0) if cuda else "GPU không nhận diện được")
    except ImportError:
        check("PyTorch", False, "cài lại: pip install torch --index-url https://download.pytorch.org/whl/cu121")

    try:
        import easyocr  # noqa: F401
        check("EasyOCR", True)
    except ImportError:
        check("EasyOCR", False, "cài lại: pip install easyocr")

    try:
        import cv2  # noqa: F401
        check("OpenCV", True, cv2.__version__)
    except ImportError:
        check("OpenCV", False, "cài lại: pip install opencv-python")

    check("Font", font_path.exists(), str(font_path) if font_path.exists() else "dùng Arial")

    ollama_ok = bool(shutil.which("ollama"))
    check("Ollama", ollama_ok, "chạy app rồi kiểm tra" if not ollama_ok else "")

    print("─" * 52)
    print("  → Khởi động app: python web_app.py")
    print("=" * 52)


if __name__ == "__main__":
    main()
