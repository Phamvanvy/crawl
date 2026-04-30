"""
_translate.py — Ollama helpers, text normalization, batch translation.
"""

import json
import re

import requests

OLLAMA_BASE = "http://localhost:11434"


def check_ollama() -> dict:
    """Kiểm tra Ollama đang chạy và liệt kê models có sẵn."""
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        if r.ok:
            models = [m["name"] for m in r.json().get("models", [])]
            return {"ok": True, "models": models}
        return {"ok": False, "error": f"HTTP {r.status_code}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _normalize_newlines(text: str) -> str:
    if not isinstance(text, str):
        return text
    text = text.replace('\\r\\n', '\n')
    text = text.replace('\\n', '\n')
    text = text.replace('\\r', '\n')
    text = text.replace('/r/n', '\n')
    text = text.replace('/n', '\n')
    text = text.replace('/r', '\n')

    lines = [line.rstrip() for line in text.splitlines()]
    if len(lines) > 1:
        merged: list[str] = []
        for line in lines:
            stripped = line.strip()
            if merged and stripped and len(stripped) <= 2 and not re.search(r'[!\?\.,:;\-]$', merged[-1]):
                merged[-1] += stripped
            else:
                merged.append(line)
        text = "\n".join(merged)

    return text


def _normalize_vietnamese(text: str) -> str:
    if not isinstance(text, str):
        return text
    replacements = {
        'thoi mien': 'thôi miên',
        'phan ngoai truyen': 'phần ngoại truyện',
        'su tra thu': 'sự trả thù',
        'chu': 'chủ',
        'đong': 'đông',
        'don': 'đơn',
    }
    normalized = text
    for wrong, right in replacements.items():
        normalized = re.sub(rf'\b{re.escape(wrong)}\b', right, normalized, flags=re.IGNORECASE)

    if re.search(r"\b(mẹ|mẹ ơi|ba|ba ơi|bố|cha)\b", normalized, flags=re.IGNORECASE):
        normalized = re.sub(r"\bEm\b", "Con", normalized)
        normalized = re.sub(r"\bem\b", "con", normalized)
        normalized = re.sub(r"\btôi\b", "con", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\bmình\b", "con", normalized, flags=re.IGNORECASE)

    normalized = re.sub(r"\b(bố|ba|mẹ|cha)\s+con\s+em\b", r"\1", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\b(bố|ba|mẹ|cha)\s+con\s+tôi\b", r"\1", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bcon\s+em\b", "con", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bcon\s+tôi\b", "con", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bcon\s+mình\b", "con", normalized, flags=re.IGNORECASE)

    return normalized


def translate_batch(texts: list[str], model: str, src_lang: str = "zh",
                    constraints: list[dict] | None = None) -> list[str]:
    """Dịch batch texts qua Ollama API, trả về list cùng thứ tự."""
    if not texts:
        return []
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    prompt = (
        "Dịch các đoạn text sau sang tiếng Việt. Các đoạn text có thể là tiếng Trung hoặc tiếng Anh.\n"
        "Đây là hội thoại trong manhwa/manga, giữ nguyên cảm xúc và sự ngắn gọn.\n"
        "QUAN TRỌNG: Bản dịch phải NGẮN GỌN nhất có thể vì text sẽ nằm trong bong bóng hội thoại nhỏ.\n"
        "Không chia ký tự. Nếu xuống dòng thì chỉ chia giữa từ/cụm từ, không chặn giữa các chữ cái.\n"
        "Dùng từ ngắn, tránh từ dài không cần thiết. Dùng \\n để xuống dòng nếu câu dài.\n"
        "Kết quả phải là tiếng Việt chuẩn có dấu đầy đủ, không để lại chữ không dấu.\n"
        "Không để lại phần tiếng Anh trong kết quả.\n"
        "Nếu trong câu có xưng Mẹ hoặc Ba, hãy dùng 'con' cho người nói, không dùng 'em' hoặc 'tôi'.\n"
        "Trả về ĐÚNG MỘT JSON array, không giải thích thêm. /no_think\n\n"
        f"Texts:\n{numbered}\n\n"
        "Kết quả (chỉ JSON array):"
    )
    try:
        payload = {"model": model, "prompt": prompt, "stream": False}
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json; charset=utf-8"}
        resp = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            data=body,
            headers=headers,
            timeout=180,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        s = raw.find("[")
        e = raw.rfind("]") + 1
        if s >= 0 and e > s:
            parsed = json.loads(raw[s:e])
            if isinstance(parsed, list) and len(parsed) == len(texts):
                def _extract(t):
                    if isinstance(t, str):
                        return _normalize_vietnamese(_normalize_newlines(t))
                    if isinstance(t, dict):
                        for k in ("text", "translation", "result", "output", "translated", "vi"):
                            if k in t and isinstance(t[k], str):
                                return _normalize_vietnamese(_normalize_newlines(t[k]))
                        for v in t.values():
                            if isinstance(v, str):
                                return _normalize_vietnamese(_normalize_newlines(v))
                    return _normalize_vietnamese(_normalize_newlines(str(t)))
                return [_extract(t) for t in parsed]
    except Exception:
        pass
    return texts
