#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_hybrid.py — Hybrid manga translation pipeline
==================================================
Pass 1 : lama_large (toàn bộ ảnh — nhanh, sạch cho nền trắng / bubble)
Pass 2 : manga_sd_cn (chỉ ảnh phức tạp — shading, tóc, chân nhân vật)

Cách dùng:
    python run_hybrid.py -i "D:/manga/raw" -o "D:/manga/translated"
    python run_hybrid.py -i ./raw -o ./out --complex-pages 003.jpg,007.jpg,012.jpg
    python run_hybrid.py -i ./raw -o ./out --force-sd
    python run_hybrid.py -i ./raw -o ./out --skip-pass1 --complex-pages all.csv
    python run_hybrid.py -i ./raw -o ./out --checkpoint "D:/models/manga_mono.safetensors"
    python run_hybrid.py -i ./raw -o ./out --translator custom_openai --target-lang VIN

Yêu cầu:
    - manga-image-translator đã cài trong mit_venv hoặc Python 3.11 path
    - File config_lama.json và config_sd.json cùng thư mục script này
    - Để dùng manga_sd_cn: pip install diffusers accelerate transformers xformers
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# ─── Constants ───────────────────────────────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Thư mục gốc chứa script này
SCRIPT_DIR   = Path(__file__).parent
CONFIG_LAMA  = SCRIPT_DIR / "config_lama.json"
CONFIG_SD    = SCRIPT_DIR / "config_sd.json"

# Ngưỡng tự động phát hiện ảnh bị lama nhòe (xem _is_lama_bleed)
LAMA_BLEED_THRESHOLD = 0.015  # > 1.5% pixel bị ảnh hưởng → flag pass2

# ─── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("hybrid")


# ─── Utility: tìm Python có manga_translator ────────────────────────────────

def find_mit_python() -> Optional[str]:
    """Tìm Python executable có manga_translator đã cài."""
    candidates = [
        SCRIPT_DIR / "mit_venv" / "Scripts" / "python.exe",
        SCRIPT_DIR / "mit_venv" / "bin" / "python",
    ]
    for path in candidates:
        if path.exists():
            ok = _check_mt(str(path))
            if ok:
                return str(path)
    # Fallback: py -3.11 / py -3.10
    for flag in ["-3.11", "-3.10"]:
        try:
            r = subprocess.run(
                ["py", flag, "-c", "import sys; print(sys.executable)"],
                capture_output=True, text=True, timeout=8,
            )
            if r.returncode == 0:
                exe = Path(r.stdout.strip())
                if exe.exists() and _check_mt(str(exe)):
                    return str(exe)
        except Exception:
            pass
    return None


def _check_mt(python_exe: str) -> bool:
    """Kiểm tra python_exe có import được manga_translator."""
    try:
        r = subprocess.run(
            [python_exe, "-c",
             "import importlib.util; print(importlib.util.find_spec('manga_translator') is not None)"],
            capture_output=True, text=True, timeout=10,
        )
        return r.returncode == 0 and r.stdout.strip().lower().startswith("true")
    except Exception:
        return False


# ─── Utility: đọc VRAM ──────────────────────────────────────────────────────

def get_free_vram_gb(python_exe: str) -> float:
    """Truy vấn VRAM qua subprocess (dùng torch trong MIT venv)."""
    script = (
        "import torch; "
        "print(round((torch.cuda.get_device_properties(0).total_memory "
        "- torch.cuda.memory_allocated(0)) / 1024**3, 2) "
        "if torch.cuda.is_available() else 99)"
    )
    try:
        r = subprocess.run(
            [python_exe, "-c", script],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode == 0:
            return float(r.stdout.strip())
    except Exception:
        pass
    return 99.0  # Giả định đủ VRAM nếu không truy vấn được


# ─── Heuristic: phát hiện ảnh bị lama nhòe ──────────────────────────────────

def _is_lama_bleed(orig_path: Path, out_path: Path) -> bool:
    """
    Heuristic: so sánh pixel vùng trắng giữa ảnh gốc và ảnh đã xử lý.
    Nếu lama inject màu xám vào vùng vốn trắng → tỉ lệ pixel bị ảnh hưởng
    vượt LAMA_BLEED_THRESHOLD → cần pass2.

    Trả về False nếu không có thư viện PIL (bỏ qua heuristic).
    """
    try:
        from PIL import Image
        import numpy as np

        orig = np.array(Image.open(orig_path).convert("L"), dtype=np.float32)
        out  = np.array(Image.open(out_path).convert("L"),  dtype=np.float32)

        if orig.shape != out.shape:
            out = np.array(
                Image.open(out_path).convert("L").resize(
                    (orig.shape[1], orig.shape[0])
                ),
                dtype=np.float32,
            )

        # Vùng trắng trong ảnh gốc (> 235 gray)
        white_mask = orig > 235
        if white_mask.sum() < 1000:
            return False  # Ảnh quá tối, không áp dụng heuristic

        # Pixel trong vùng trắng mà output bị tối đi ≥ 40 (lama inject shading)
        bleed = (white_mask & ((orig - out) > 40))
        ratio = bleed.sum() / white_mask.sum()

        if ratio > LAMA_BLEED_THRESHOLD:
            log.debug(f"  lama_bleed={ratio:.3f} → flag pass2: {orig_path.name}")
            return True
        return False

    except Exception as e:
        log.debug(f"  Heuristic skip ({e})")
        return False


# ─── Đọc danh sách complex pages ─────────────────────────────────────────────

def parse_complex_pages(arg: str, input_dir: Path) -> set[str]:
    """
    Phân tích --complex-pages:
      - Chuỗi CSV: "003.jpg,007.jpg"  → set tên file
      - File .txt / .csv              → đọc từng dòng
    Trả về set tên file (chỉ basename, không có đường dẫn).
    """
    pages: set[str] = set()
    if not arg:
        return pages

    # Thử xem có phải path đến file không
    p = Path(arg)
    if p.is_file():
        for line in p.read_text(encoding="utf-8").splitlines():
            name = line.strip()
            if name:
                pages.add(Path(name).name)
        log.info(f"Đọc {len(pages)} complex pages từ file: {p}")
        return pages

    # Chuỗi CSV
    for name in arg.split(","):
        name = name.strip()
        if name:
            pages.add(Path(name).name)
    log.info(f"Complex pages (CSV): {len(pages)} ảnh")
    return pages


# ─── Chạy MIT CLI ─────────────────────────────────────────────────────────────

def run_mit(
    python_exe: str,
    input_dir: Path,
    output_dir: Path,
    config_path: Path,
    *,
    use_gpu: bool = True,
    overwrite: bool = False,
    verbose: bool = False,
    translator: str = "",
    target_lang: str = "",
    images: list[str] | None = None,
    extra_env: dict[str, str] | None = None,
    low_vram: bool = False,
    pass_name: str = "",
) -> int:
    """
    Gọi MIT CLI. Trả về returncode.
    images: danh sách tên file (basename) cần dịch trong input_dir,
            None = dịch toàn bộ thư mục.
    """
    cmd = [python_exe, "-m", "manga_translator"]
    if use_gpu:
        cmd.append("--use-gpu")
    if verbose:
        cmd.append("--verbose")
    if low_vram:
        cmd += ["--low-vram"]

    cmd += ["local", "-i", str(input_dir), "-o", str(output_dir),
            "--config-file", str(config_path)]

    if overwrite:
        cmd.append("--overwrite")

    # Nếu chỉ dịch một số ảnh cụ thể → truyền danh sách
    if images:
        # MIT hỗ trợ: --images file1.jpg,file2.jpg
        cmd += ["--images", ",".join(images)]

    # Translator override (nếu muốn thay đổi translator mà không sửa config)
    if translator:
        cmd += ["--translator", translator]
    if target_lang:
        cmd += ["--target-lang", target_lang]

    label = f"[{pass_name}] " if pass_name else ""
    log.info(f"{label}CMD: {' '.join(cmd)}")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"]       = "1"
    env["PYTHONUNBUFFERED"] = "1"
    if extra_env:
        env.update(extra_env)

    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    # Stream log ra stdout
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            log.info(f"  MIT | {line}")

    proc.wait()
    elapsed = time.time() - t0
    log.info(f"{label}Kết thúc — returncode={proc.returncode}, thời gian={elapsed:.0f}s")
    return proc.returncode


# ─── Main pipeline ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hybrid manga translation: lama (pass1) + manga_sd_cn (pass2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-i", "--input",    required=True,  help="Thư mục ảnh gốc")
    parser.add_argument("-o", "--output",   required=True,  help="Thư mục ảnh đã dịch")
    parser.add_argument(
        "--complex-pages",
        default="",
        help=(
            'Danh sách ảnh cần dùng manga_sd_cn (pass2). '
            'CSV: "003.jpg,007.jpg" hoặc path file .txt/.csv.'
        ),
    )
    parser.add_argument(
        "--force-sd",
        action="store_true",
        help="Bỏ qua pass1, chạy toàn bộ bằng manga_sd_cn",
    )
    parser.add_argument(
        "--skip-pass1",
        action="store_true",
        help="Bỏ qua pass1 (lama), chỉ chạy pass2 trên ảnh trong --complex-pages",
    )
    parser.add_argument(
        "--auto-detect-bleed",
        action="store_true",
        help=(
            "Tự động phát hiện ảnh bị lama nhòe sau pass1 (heuristic) "
            "→ thêm vào danh sách pass2. Cần PIL/numpy."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help=(
            "Checkpoint SD1.5 cho pass2. HuggingFace ID hoặc path local .safetensors. "
            "Ví dụ: 'stablediffusionapi/manga-diffusion' "
            "hoặc 'D:/models/MangaMix_v10.safetensors'. "
            "Mặc định: Meina/MeinaMix_V11."
        ),
    )
    parser.add_argument("--use-gpu",   action="store_true", default=True,  help="Dùng GPU (mặc định)")
    parser.add_argument("--no-gpu",    action="store_true",                 help="Tắt GPU, dùng CPU")
    parser.add_argument("--overwrite", action="store_true",                 help="Ghi đè ảnh đã dịch")
    parser.add_argument("--verbose",   action="store_true",                 help="Log chi tiết từ MIT")
    parser.add_argument(
        "--translator",
        default="",
        help="Translator MIT (ví dụ: custom_openai, m2m100_big, sugoi). Nếu để trống → dùng trong config.",
    )
    parser.add_argument(
        "--target-lang",
        default="",
        help="Ngôn ngữ đích (ví dụ: VIN, JPN, ENG). Nếu để trống → dùng trong config.",
    )
    parser.add_argument(
        "--python",
        default="",
        help="Đường dẫn Python có manga_translator. Auto-detect nếu để trống.",
    )
    args = parser.parse_args()

    # ── Validate ─────────────────────────────────────────────────────────────
    input_dir  = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    if not input_dir.is_dir():
        log.error(f"Thư mục input không tồn tại: {input_dir}")
        sys.exit(1)

    if not CONFIG_LAMA.exists():
        log.error(f"Không tìm thấy config_lama.json: {CONFIG_LAMA}")
        sys.exit(1)

    if not CONFIG_SD.exists():
        log.error(f"Không tìm thấy config_sd.json: {CONFIG_SD}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Tìm Python MIT ───────────────────────────────────────────────────────
    python_exe = args.python.strip() or find_mit_python()
    if not python_exe:
        log.error(
            "Không tìm thấy Python có manga_translator!\n"
            "Cài đặt: py -3.11 -m venv mit_venv && "
            "mit_venv\\Scripts\\pip install git+https://github.com/zyddnys/manga-image-translator.git"
        )
        sys.exit(1)

    log.info(f"MIT Python: {python_exe}")

    # ── GPU / VRAM ────────────────────────────────────────────────────────────
    use_gpu    = args.use_gpu and not args.no_gpu
    low_vram   = False
    free_vram  = 99.0

    if use_gpu:
        free_vram = get_free_vram_gb(python_exe)
        log.info(f"VRAM trống: {free_vram:.1f} GB")
        if free_vram < 6.0:
            low_vram = True
            log.warning(f"VRAM < 6GB → bật low_vram mode (--low-vram --half --batch-size 1)")

    # ── Liệt kê ảnh ──────────────────────────────────────────────────────────
    all_images = sorted(
        f for f in input_dir.iterdir()
        if f.suffix.lower() in IMAGE_EXTS
    )
    if not all_images:
        log.error(f"Không có ảnh trong {input_dir}")
        sys.exit(1)
    log.info(f"Tổng ảnh: {len(all_images)}")

    # ── Xác định complex pages ───────────────────────────────────────────────
    complex_names = parse_complex_pages(args.complex_pages, input_dir)

    # ── Pass 1: lama ──────────────────────────────────────────────────────────
    if not args.force_sd and not args.skip_pass1:
        log.info("=" * 60)
        log.info("PASS 1 — lama_large (toàn bộ ảnh)")
        log.info("=" * 60)

        rc = run_mit(
            python_exe, input_dir, output_dir, CONFIG_LAMA,
            use_gpu=use_gpu,
            overwrite=args.overwrite,
            verbose=args.verbose,
            translator=args.translator,
            target_lang=args.target_lang,
            low_vram=low_vram,
            pass_name="PASS1-lama",
        )
        if rc != 0:
            log.warning(f"Pass 1 kết thúc với code {rc} — tiếp tục pass 2 nếu có")

        # ── Auto-detect lama bleed sau pass1 ─────────────────────────────────
        if args.auto_detect_bleed:
            log.info("Kiểm tra lama bleed …")
            for img in all_images:
                out_img = output_dir / img.name
                if not out_img.exists():
                    continue
                if _is_lama_bleed(img, out_img):
                    complex_names.add(img.name)
                    log.info(f"  + Auto flag pass2: {img.name}")

    # ── Pass 2: manga_sd_cn ───────────────────────────────────────────────────
    if args.force_sd:
        # Dịch lại toàn bộ bằng SD
        pass2_images = [f.name for f in all_images]
    elif complex_names:
        # Chỉ dịch ảnh trong danh sách
        pass2_images = [
            f.name for f in all_images
            if f.name in complex_names
        ]
        # Cảnh báo nếu có tên trong danh sách mà không tìm thấy file
        missing = complex_names - {f.name for f in all_images}
        if missing:
            log.warning(f"Các ảnh trong --complex-pages không tồn tại: {', '.join(sorted(missing))}")
    else:
        pass2_images = []

    if pass2_images:
        log.info("=" * 60)
        log.info(f"PASS 2 — manga_sd_cn ({len(pass2_images)} ảnh)")
        log.info("=" * 60)
        if args.checkpoint:
            log.info(f"Checkpoint SD: {args.checkpoint}")

        # Env vars cho pass2
        extra_env: dict[str, str] = {}
        if args.checkpoint:
            extra_env["MIT_SD_CHECKPOINT"] = args.checkpoint

        # Pass 2 luôn dùng --overwrite để ghi đè kết quả pass1 trên ảnh phức tạp
        rc = run_mit(
            python_exe, input_dir, output_dir, CONFIG_SD,
            use_gpu=use_gpu,
            overwrite=True,             # luôn ghi đè kết quả pass1
            verbose=args.verbose,
            translator=args.translator,
            target_lang=args.target_lang,
            images=pass2_images,
            extra_env=extra_env,
            low_vram=low_vram,
            pass_name="PASS2-sd_cn",
        )
        if rc != 0:
            log.warning(f"Pass 2 kết thúc với code {rc}")
    else:
        log.info("Không có ảnh nào cần pass2 (manga_sd_cn).")
        log.info("Gợi ý: Dùng --complex-pages, --force-sd, hoặc --auto-detect-bleed")

    # ── Tổng kết ─────────────────────────────────────────────────────────────
    output_count = sum(
        1 for f in output_dir.iterdir()
        if f.suffix.lower() in IMAGE_EXTS
    )
    log.info("=" * 60)
    log.info(f"XONG. Output: {output_dir}")
    log.info(f"  Ảnh trong thư mục output : {output_count}")
    if pass2_images:
        log.info(f"  Pass2 (manga_sd_cn)      : {len(pass2_images)} ảnh")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
