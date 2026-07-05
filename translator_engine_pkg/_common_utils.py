"""
_common_utils.py — Shared utilities for translation engine.
Centralized to reduce code duplication across modules.
"""

import re
import time
import hashlib
import json
from pathlib import Path
from collections import deque
from typing import Optional

# ── Regex patterns ─────────────────────────────────────────────────────────────

# Chinese characters (CJK Unified Ideographs + Extension A)
ZH_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")

# Japanese Hiragana + Katakana + Kanji
JA_HIRAGANA_RE = re.compile(r"[\u3040-\u309f]")
JA_KATAKANA_RE = re.compile(r"[\u30a0-\u30ff]")
JA_KANJI_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
JA_RE = re.compile(r"[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff\u3400-\u4dbf]")

# English words (3+ letters)
EN_WORD_RE = re.compile(r"\b[a-zA-Z]{3,}\b")

# Vietnamese diacritics — presence confirms Vietnamese output
VI_DIACRITIC_RE = re.compile(
    r"[àáâãèéêìíòóôõùúăđơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]",
    re.IGNORECASE,
)

# Watermark domain pattern
_WATERMARK_DOMAIN_RE = re.compile(
    r"[a-z0-9][a-z0-9._-]{1,64}\.(?:com|net|org|info|xyz|top|site|online|tv|cc)\b",
    re.IGNORECASE,
)

# Watermark / brand names list
_WATERMARK_KEYWORDS = [
    "acg", "pixiv", "twitter", "fanbox", "patreon", "x.com", "18cg","https", "http", "www", "com", "net", "org", "pix",
]


# ── Helper functions ───────────────────────────────────────────────────────────

def contains_chinese(text: str) -> bool:
    """True if text contains Chinese characters."""
    return bool(ZH_RE.search(text)) if isinstance(text, str) else False


def contains_japanese(text: str) -> bool:
    """True if text contains Japanese characters (Hiragana, Katakana, or Kanji)."""
    if not isinstance(text, str):
        return False
    return bool(JA_HIRAGANA_RE.search(text) or JA_KATAKANA_RE.search(text) or JA_KANJI_RE.search(text))


def contains_cjk(text: str) -> bool:
    """True if text contains any CJK characters (Chinese or Japanese)."""
    if not isinstance(text, str):
        return False
    return bool(JA_RE.search(text))


def has_english(text: str) -> bool:
    """True if text contains English words."""
    return bool(EN_WORD_RE.search(text)) if isinstance(text, str) else False


def contains_watermark_text(text: str) -> bool:
    """True if text looks like a watermark/logo/brand name."""
    if not isinstance(text, str) or not text.strip():
        return False

    compact = re.sub(r"\s+", "", text.lower())
    compact = re.sub(r"[^a-z0-9._:/-]", "", compact)
    
    # Check for watermark keywords
    for keyword in _WATERMARK_KEYWORDS:
        if keyword in compact:
            return True
    
    # Check for domain patterns
    if ".com" in compact or compact.endswith("com"):
        return True
    if "www." in compact:
        return True
    if "http://" in compact or "https://" in compact:
        return True
    if bool(_WATERMARK_DOMAIN_RE.search(compact)):
        return True

    return False


def strip_generation_artifacts(text: str, preserve_segment_tokens: bool = False) -> str:
    """Remove LLM generation artifacts (thoughts, special tokens, etc.)."""
    if not isinstance(text, str) or not text:
        return text

    cleaned = re.sub(r"</think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"</\|\d+\|>?", "", cleaned)
    cleaned = re.sub(
        r"<\|(?:assistant|user|system|im_start|im_end|eot_id|end_of_text|endoftext|begin_of_text|bos|eos|pad|unk)\|>",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"</?(?:s|bos|eos|pad|unk)>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\[/?INST\]|<<SYS>>|<</SYS>>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"</(?=[^a-zA-Z]|$)", "", cleaned)
    if not preserve_segment_tokens:
        cleaned = re.sub(r"<\|\d+\|>?", "", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def clean_watermark_fragments(text: str, source: str = "") -> str:
    """Remove watermark/brand name fragments from translated text."""
    if not isinstance(text, str) or not text.strip():
        return text

    cleaned = strip_generation_artifacts(text)
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    
    if not lines:
        return ""

    filtered_lines = [line for line in lines if not contains_watermark_text(line)]
    if filtered_lines:
        return "\n".join(filtered_lines)

    if contains_watermark_text(text) or (source and contains_watermark_text(source)):
        return ""

    return cleaned


def normalize_newlines(text: str) -> str:
    """Normalize various newline representations to actual newlines."""
    if not isinstance(text, str):
        return text

    text = strip_generation_artifacts(text)
    text = text.replace('\\r\\n', '\n')
    text = text.replace('\\n', '\n')
    text = text.replace('\\r', '\n')
    text = text.replace('/r/n', '\n')
    text = text.replace('/n', '\n')
    text = text.replace('/r', '\n')

    # Merge single-char lines that don't end with punctuation
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


def normalize_vietnamese(text: str) -> str:
    """Fix common Vietnamese spelling errors without changing pronouns."""
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

    return normalized


def needs_vietnamese_retry(text: str) -> bool:
    """True if text is likely English/Chinese rather than Vietnamese."""
    cleaned = clean_watermark_fragments(strip_generation_artifacts(text))
    if not cleaned or len(cleaned.strip()) < 4:
        return False
    if contains_cjk(cleaned):
        return True  # Untranslated CJK
    if VI_DIACRITIC_RE.search(cleaned):
        return False  # Vietnamese diacritics present = Vietnamese
    en_words = EN_WORD_RE.findall(cleaned)
    return len(en_words) >= 2  # 2+ plain Latin words, no VI diacritics = English


def text_to_hash(text: str, model: str = "", src_lang: str = "") -> str:
    """Generate a hash for caching translations."""
    content = f"{text}|{model}|{src_lang}"
    return hashlib.md5(content.encode('utf-8')).hexdigest()


class TranslationMemory:
    """In-memory translation cache to avoid re-translating identical text.
    
    Usage:
        tm = TranslationMemory(max_size=10000)
        # Get cached translation
        result = tm.get("你好", model="qwen3:8b", src_lang="zh")
        # Store new translation
        tm.set("你好", "Xin chào", model="qwen3:8b", src_lang="zh")
    """
    
    def __init__(self, max_size: int = 10000):
        self._cache: dict[str, dict] = {}
        self._access_order: deque = deque(maxlen=max_size)
        self.max_size = max_size
    
    def get(self, text: str, model: str = "", src_lang: str = "") -> Optional[str]:
        """Get cached translation if exists."""
        key = text_to_hash(text, model, src_lang)
        entry = self._cache.get(key)
        if entry:
            # Move to end (most recently used)
            if key in [k for k in self._access_order]:
                self._access_order.remove(key)
                self._access_order.append(key)
            return entry.get("translation")
        return None
    
    def set(self, source: str, translation: str, model: str = "", src_lang: str = ""):
        """Store translation in cache."""
        if not source.strip() or not translation.strip():
            return
        
        key = text_to_hash(source, model, src_lang)
        
        # Evict if full
        if len(self._cache) >= self.max_size and key not in self._cache:
            self._evict()
        
        self._cache[key] = {
            "source": source,
            "translation": translation,
            "model": model,
            "src_lang": src_lang,
            "timestamp": time.time(),
        }
        self._access_order.append(key)
    
    def _evict(self):
        """Evict least recently used entry."""
        if self._access_order:
            oldest = next(iter(self._access_order))
            self._cache.pop(oldest, None)
    
    def clear(self):
        """Clear all cached translations."""
        self._cache.clear()
        self._access_order.clear()
    
    def __len__(self) -> int:
        return len(self._cache)
    
    def save(self, filepath: str):
        """Save cache to JSON file."""
        data = {
            "cache": self._cache,
            "access_order": list(self._access_order),
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, filepath: str):
        """Load cache from JSON file."""
        path = Path(filepath)
        if not path.exists():
            return
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._cache = data.get("cache", {})
            self._access_order = deque(data.get("access_order", []), maxlen=self.max_size)
        except (json.JSONDecodeError, IOError):
            pass


# ── Pronoun helper for relationship detection ─────────────────────────────────

_RELATIONSHIP_CONTEXTS = {
    "parent_child": [
        r'妈妈|爸爸|母亲|父亲|妈咪|老爸',   # Mother, Father
        r'零花钱|压岁钱|红包',              # Pocket money, Red envelope
        r'学费|补习班|考试',                # Tuition, Tutoring, Exams
        r'奶奶|爷爷|外婆|外公',             # Grandparents
    ],
    "romantic": [
        r'(亲爱的|宝贝)',                   # Dear, Baby
        r'男朋友|女朋友|未婚夫|未婚妻',     # Boyfriend, Girlfriend, Fiance
        r'老公|老婆|丈夫|妻子',             # Husband, Wife
        r'我爱你|我喜欢你|嫁给我',          # I love you, Marry me
    ],
    "school": [
        r'(同学|同班|同桌)',               # Classmate
        r'老师|教授|导师',                 # Teacher, Professor
        r'学长|学姐|学弟|学妹',            # Senior/Junior student
    ],
}

# Japanese honorifics patterns
_JAPANESE_HONORIFICS = {
    r'\b(さん)\b': "anh/cô/chị",
    r'\b(くん|君)\b': "cậu/em",
    r'\b(ちゃん)\b': "cậu/bé",
    r'\b(様)\b': "ngài/thưa",
    r'\b(先生)\b': "thầy/cô/bác sĩ",
    r'\b(先輩)\b': "huynh/tiền bối",
    r'\b(後輩)\b': "đệ/em",
    r'\b(ちゃん)\b': "cậu/bé",
}

# Japanese onomatopoeia mappings
_JAPANESE_ONOMATOPOEIA = {
    "ワクワク": "hồi hộp",
    "ドキドキ": "tim đập nhanh",
    "ドキドキ": "run rẩy",
    "ゴロゴロ": "lục cục",
    "ザワザワ": "xôn xao",
    "キュン": "nhói nhơ",
    "カンカン": "giận dữ",
    "ビクン": "giật mình",
    "シュッシュ": "xịt xẹt",
    "パチパチ": "lách cách",
}


def detect_relationship_context(text: str) -> Optional[str]:
    """Detect relationship context from text to choose appropriate pronouns."""
    if not isinstance(text, str):
        return None
    
    for relation, patterns in _RELATIONSHIP_CONTEXTS.items():
        for pattern in patterns:
            if re.search(pattern, text):
                return relation
    return None


def apply_relationship_pronouns(translated: str, relationship: Optional[str]) -> str:
    """Adjust pronouns based on detected relationship context."""
    if not translated or not translated.strip():
        return translated
    
    replacements = {
        "parent_child": [
            (r'\btôi\b', 'con'),
            (r'\bem\b', 'con'),
        ],
        "romantic": [],
        "school": [],
    }
    
    adjusted = translated
    if relationship and relationship in replacements:
        for pattern, replacement in replacements[relationship]:
            adjusted = re.sub(pattern, replacement, adjusted, flags=re.IGNORECASE)
    
    return adjusted


def normalize_japanese_honorific(text: str) -> str:
    """Normalize Japanese honorifics in translated text.
    
    Converts honorifics to appropriate Vietnamese equivalents.
    """
    if not isinstance(text, str):
        return text
    
    # Remove honorific suffixes from translated text (they shouldn't appear in Vietnamese)
    for pattern, _ in _JAPANESE_HONORIFICS.items():
        text = re.sub(pattern, "", text)
    
    return text


def normalize_japanese_onomatopoeia(text: str) -> str:
    """Convert Japanese onomatopoeia to Vietnamese equivalents."""
    if not isinstance(text, str):
        return text
    
    for jp, vn in _JAPANESE_ONOMATOPOEIA.items():
        text = text.replace(jp, vn)

    return text


# ── Lưu ảnh có nén (giảm dung lượng) ─────────────────────────────────────────

def _save_jpeg(img_pil, dst: Path, quality: int) -> None:
    """Lưu ảnh PIL thành JPEG nén. Chuyển sang RGB nếu cần (JPEG không có alpha)."""
    im = img_pil
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    im.save(str(dst), "JPEG", quality=quality, optimize=True, progressive=True)


def save_image_compressed(img_pil, dst, quality: int = 95) -> Path:
    """Lưu ảnh PIL với mức nén `quality` (1–100) để giảm dung lượng file.

    - quality >= 100 → giữ nguyên format gốc, lossless (PNG không nén mất dữ liệu,
      JPEG/WEBP lưu chất lượng 95 như trước).
    - quality <  100 → JPEG/WEBP nén theo `quality`; PNG/BMP/khác được chuyển sang
      JPEG (đổi đuôi `.jpg`) để giảm dung lượng mạnh nhất.

    Trả về `Path` thật sự đã ghi (có thể khác `dst` khi đổi đuôi PNG→JPEG).
    """
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    q = max(1, min(100, int(quality)))
    ext = dst.suffix.lower()

    if q >= 100:
        if ext in (".jpg", ".jpeg"):
            img_pil.save(str(dst), "JPEG", quality=95)
        elif ext == ".webp":
            img_pil.save(str(dst), "WEBP", quality=95)
        else:
            img_pil.save(str(dst))
        return dst

    if ext in (".jpg", ".jpeg"):
        _save_jpeg(img_pil, dst, q)
        return dst
    if ext == ".webp":
        img_pil.save(str(dst), "WEBP", quality=q, method=4)
        return dst

    # PNG / BMP / khác → JPEG nén (giảm dung lượng nhiều nhất cho ảnh không trong suốt).
    new_dst = dst.with_suffix(".jpg")
    _save_jpeg(img_pil, new_dst, q)
    return new_dst