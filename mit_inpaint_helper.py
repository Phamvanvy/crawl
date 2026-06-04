"""mit_inpaint_helper.py — Inpaint an image with a precomputed mask using MIT's
inpainter (e.g. lama_large). Run with the mit_venv Python (manga_translator installed).

Usage:
    python mit_inpaint_helper.py <img> <mask> <out> [inpainter] [size] [precision] [device]

- <img>  : input image (any format readable by cv2)
- <mask> : single-channel mask, 255 = pixels to inpaint
- <out>  : output path (PNG recommended)
This bypasses detection/OCR entirely — it only runs the inpainting model on the
given mask, so it works on regions OCR can't read (e.g. stylized SFX).
"""
import sys
import asyncio

import cv2


def main():
    if len(sys.argv) < 4:
        print("ERR usage: img mask out [inpainter] [size] [precision] [device]")
        sys.exit(2)
    img_path, mask_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    inpainter = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] else "lama_large"
    size = int(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] else 2048
    precision = sys.argv[6] if len(sys.argv) > 6 and sys.argv[6] else "bf16"
    device = sys.argv[7] if len(sys.argv) > 7 and sys.argv[7] else "cpu"

    from manga_translator.inpainting import dispatch, prepare
    from manga_translator.config import Inpainter, InpainterConfig, InpaintPrecision

    img = cv2.imread(img_path)
    if img is None:
        print(f"ERR cannot read image: {img_path}")
        sys.exit(3)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"ERR cannot read mask: {mask_path}")
        sys.exit(3)

    try:
        key = Inpainter(inpainter)
    except ValueError:
        key = Inpainter.lama_large
    try:
        prec = InpaintPrecision(precision)
    except ValueError:
        prec = InpaintPrecision.bf16
    cfg = InpainterConfig(inpainter=key, inpainting_size=size, inpainting_precision=prec)

    async def run():
        await prepare(key, device)
        out = await dispatch(key, img_rgb, mask, cfg, size, device, False)
        cv2.imwrite(out_path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

    asyncio.run(run())
    print("OK")


if __name__ == "__main__":
    main()
