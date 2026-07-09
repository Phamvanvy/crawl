# -*- coding: utf-8 -*-
"""
_vlm_ocr.py — OCR bằng vision-language model (VLM, vd Qwen2.5-VL), thay cho
PaddleOCR/EasyOCR trong backend "OCR + Ollama". Không dịch ở bước này — chỉ
đọc verbatim text, việc dịch vẫn dùng translate_batch (_translate.py) như cũ.

2 lượt để tăng recall, giống cấu trúc _run_ocr cũ:
  1. Toàn ảnh: yêu cầu VLM tự phát hiện + đọc MỌI vùng chữ (JSON [{bbox,text}]).
     VLM có grounding tốt (Qwen2.5-VL) nên đảm nhiệm luôn vai trò "detector"
     mà PaddleOCR từng làm.
  2. Crop từng bong bóng (_find_speech_bubbles) — đọc lại riêng để vớt các
     bong bóng bị bước 1 bỏ sót/đọc thiếu (tương tự bubble-crop pass cũ).
Kết quả gộp + dedup theo IoU, trả về list[(bbox_4pts, text, confidence)] —
cùng shape với _run_ocr cũ để _image_translator.py không cần đổi logic lọc.
"""

import base64
import json
import re

import requests

from ._ocr import _find_speech_bubbles, _iou_rect, _rect_of_bbox

OLLAMA_BASE = "http://localhost:8080"

_PAGE_PROMPT = (
    "You are an OCR engine reading a manga/manhwa/manhua page image "
    "({w}x{h} pixels). Find every region containing readable text "
    "(speech bubbles, narration boxes, sound effects/SFX, stylized or "
    "hand-drawn lettering) and output a JSON array, nothing else:\n"
    '[{{"bbox": [x1, y1, x2, y2], "text": "..."}}, ...]\n'
    "Rules:\n"
    "- bbox is [x1, y1, x2, y2] in absolute pixel coordinates of this exact "
    "image (0,0 = top-left, x2<={w}, y2<={h}).\n"
    "- text is the verbatim text in that region, preserving line breaks with \\n.\n"
    "- Do NOT translate. Do NOT romanize. Do NOT transliterate.\n"
    "- Skip regions with no legible text (pure art/decoration).\n"
    "- Output ONLY the JSON array, no markdown fences, no commentary."
)

_CROP_PROMPT = (
    "You are an OCR engine. Output ONLY the exact text visible in this "
    "cropped comic panel image, preserving line breaks. Do not translate. "
    "Do not romanize. Do not describe the image or add quotes/labels. "
    "If there is no legible text, output nothing."
)


def _encode_jpeg_b64(img_array) -> str:
    import cv2
    ok, buf = cv2.imencode(".jpg", img_array, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise ValueError("Không encode được ảnh để gửi VLM")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _call_vlm(
    img_array,
    prompt: str,
    model: str,
    llm_base_url: str = "",
    llm_api_type: str = "ollama",
    timeout: int = 180,
) -> str:
    """Gửi 1 ảnh + prompt cho VLM, trả về raw text response."""
    b64 = _encode_jpeg_b64(img_array)
    base = (llm_base_url.rstrip("/") if llm_base_url else OLLAMA_BASE)

    if llm_api_type == "openai_compat":
        payload = {
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }],
            "stream": False,
            "temperature": 0,
        }
        resp = requests.post(f"{base}/v1/chat/completions", json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    else:
        payload = {
            "model": model,
            "prompt": prompt,
            "images": [b64],
            "stream": False,
            "options": {"temperature": 0},
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json; charset=utf-8"}
        resp = requests.post(f"{base}/api/generate", data=body, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("response", "") or ""


_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _parse_page_json(raw: str, w: int, h: int) -> list[tuple]:
    """Parse JSON [{bbox,text}] response → list of (bbox_4pts, text, confidence)."""
    match = _JSON_ARRAY_RE.search(raw)
    if not match:
        return []
    try:
        items = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return []

    results: list[tuple] = []
    if not isinstance(items, list):
        return results
    for item in items:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        bbox = item.get("bbox")
        if not text or not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        try:
            x1, y1, x2, y2 = (float(v) for v in bbox)
        except (TypeError, ValueError):
            continue
        x1, x2 = sorted((max(0, min(x1, w)), max(0, min(x2, w))))
        y1, y2 = sorted((max(0, min(y1, h)), max(0, min(y2, h))))
        if x2 - x1 < 3 or y2 - y1 < 3:
            continue
        quad = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        results.append((quad, text, 0.99))
    return results


def _run_vlm_ocr(
    img_array,
    model: str,
    llm_base_url: str = "",
    llm_api_type: str = "ollama",
    src_lang: str = "zh",
    on_log=None,
) -> list[tuple]:
    """
    OCR orchestration bằng VLM: toàn ảnh (detect + đọc) + crop từng bong bóng.
    Trả về list of (bbox_4pts, text, confidence) — cùng shape với _run_ocr cũ.
    """
    log = on_log or (lambda msg: None)
    h, w = img_array.shape[:2]
    all_results: list[tuple] = []

    try:
        raw = _call_vlm(img_array, _PAGE_PROMPT.format(w=w, h=h), model,
                         llm_base_url, llm_api_type)
        page_hits = _parse_page_json(raw, w, h)
        if not page_hits and raw.strip():
            log(f"  [VLM] Model trả lời nhưng không parse được JSON bbox — "
                f"model có hỗ trợ ảnh (vision) không? Response: {raw.strip()[:160]!r}")
        all_results.extend(page_hits)
    except Exception as exc:
        log(f"  [VLM] Lỗi gọi model '{model}' (toàn ảnh): {exc}")

    bubble_errors = 0
    try:
        bubbles = _find_speech_bubbles(img_array)
        for x1, y1, x2, y2 in bubbles:
            crop = img_array[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            try:
                text = _call_vlm(crop, _CROP_PROMPT, model, llm_base_url, llm_api_type,
                                  timeout=60).strip()
            except Exception as exc:
                bubble_errors += 1
                if bubble_errors == 1:
                    log(f"  [VLM] Lỗi gọi model '{model}' (crop bong bóng): {exc}")
                continue
            if not text:
                continue
            bbox = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            all_results.append((bbox, text, 0.99))
    except Exception:
        pass

    if not all_results:
        return []

    rects = [_rect_of_bbox(b) for b, _t, _c in all_results]
    keep = [True] * len(all_results)
    for i in range(len(all_results)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(all_results)):
            if not keep[j]:
                continue
            if _iou_rect(rects[i], rects[j]) > 0.5:
                # Ưu tiên text dài hơn (thường là bản đọc đầy đủ hơn khi conf bằng nhau)
                if len(all_results[i][1]) >= len(all_results[j][1]):
                    keep[j] = False
                else:
                    keep[i] = False
                    break

    return [r for r, k in zip(all_results, keep) if k]
