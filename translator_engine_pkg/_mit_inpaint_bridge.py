# -*- coding: utf-8 -*-
"""
_mit_inpaint_bridge.py — Cầu nối backend "default" → inpainter lama_large của
manga-image-translator (chạy trong mit_venv, xem mit_inpaint_helper.py).

Khác với inpaint_region() (opencv/simple-lama, xử lý từng bbox in-process),
lama_large chạy qua subprocess venv riêng nên launch 1 lần/box sẽ rất chậm
(mỗi lần load lại model). Vì vậy module này GỘP mọi vùng cần xoá của 1 ảnh
thành 1 mask duy nhất rồi gọi subprocess ĐÚNG 1 LẦN cho cả ảnh.
"""

import subprocess
import tempfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_HELPER = _PROJECT_ROOT / "mit_inpaint_helper.py"


def _build_combined_mask(shape, regions):
    """regions: list of (bbox_4pts, dilate_ksize, dilate_iters) → mask uint8 gộp (255=inpaint)."""
    import cv2
    import numpy as np

    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for bbox, dilate_ksize, dilate_iters in regions:
        pts = np.array(bbox, dtype=np.int32)
        region_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(region_mask, [pts], 255)
        region_mask = cv2.dilate(
            region_mask, np.ones((dilate_ksize, dilate_ksize), np.uint8), iterations=dilate_iters
        )
        mask = cv2.bitwise_or(mask, region_mask)
    return mask


def inpaint_regions_lama_large(
    img,
    regions: list,
    python_path: str | None,
    use_gpu: bool = True,
    inpainting_size: int = 2048,
    precision: str = "bf16",
    on_log=None,
    timeout: int = 600,
):
    """
    Xoá mọi vùng trong `regions` (list of (bbox_4pts, dilate_ksize, dilate_iters))
    bằng lama_large của MIT, gọi subprocess đúng 1 lần cho cả ảnh.
    Trả về ảnh BGR đã inpaint, hoặc None nếu lỗi/không có mit_venv (caller nên
    fallback sang inpaint_region opencv khi None).
    """
    log = on_log or (lambda msg: None)
    if not python_path:
        log("  [INPAINT] lama_large: chưa tìm thấy mit_venv — fallback OpenCV.")
        return None
    if not _HELPER.exists():
        log(f"  [INPAINT] lama_large: thiếu {_HELPER.name} — fallback OpenCV.")
        return None
    if not regions:
        return img

    import cv2

    mask = _build_combined_mask(img.shape, regions)
    td = tempfile.mkdtemp(prefix="mit_inp_")
    try:
        ip = Path(td) / "in.png"
        mp = Path(td) / "mask.png"
        op = Path(td) / "out.png"
        cv2.imwrite(str(ip), img)
        cv2.imwrite(str(mp), mask)
        device = "cuda" if use_gpu else "cpu"
        r = subprocess.run(
            [python_path, str(_HELPER), str(ip), str(mp), str(op),
             "lama_large", str(inpainting_size), precision, device],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
        )
        if op.exists():
            out = cv2.imread(str(op))
            if out is not None:
                return out
        log(f"  [INPAINT] lama_large lỗi: {((r.stderr or '') + (r.stdout or ''))[-200:]}")
        return None
    except Exception as exc:
        log(f"  [INPAINT] lama_large lỗi: {exc}")
        return None
    finally:
        import shutil
        shutil.rmtree(td, ignore_errors=True)
