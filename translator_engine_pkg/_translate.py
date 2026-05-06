"""
_translate.py — Ollama helpers, text normalization, batch translation with aggressive Vietnamese enforcement.
"""

import json
import re
import time

import requests

OLLAMA_BASE = "http://localhost:11434"

# ── Compiled regex patterns (must be at top — used by helper functions below) ─
CHINESE_RE = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf]')
_WATERMARK_DOMAIN_RE = re.compile(
    r"[a-z0-9][a-z0-9._-]{1,64}\.(?:com|net|org|info|xyz|top|site|online|tv|cc)\b",
    re.IGNORECASE,
)
# Vietnamese has unique diacritical marks — their presence confirms Vietnamese output
_VI_DIACRITIC_RE = re.compile(
    r"[àáâãèéêìíòóôõùúăđơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]",
    re.IGNORECASE,
)
# 3+ letter Latin words — 2+ with no VI diacritics = likely English
_EN_WORD_RE = re.compile(r"\b[a-zA-Z]{3,}\b")


def _contains_watermark_text(text: str) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False

    compact = re.sub(r"\s+", "", text.lower())
    compact = re.sub(r"[^a-z0-9._:/-]", "", compact)
    return (
        "acg" in compact
        or ".com" in compact
        or compact.endswith("com")
        or "www." in compact
        or "http://" in compact
        or "https://" in compact
        or bool(_WATERMARK_DOMAIN_RE.search(compact))
    )


def _strip_generation_artifacts(text: str, preserve_segment_tokens: bool = False) -> str:
    if not isinstance(text, str) or not text:
        return text

    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"</\|\d+\|>?", "", cleaned)  # </|3|> and </|3|
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
        cleaned = re.sub(r"<\|\d+\|>?", "", cleaned)  # <|3|> and <|3|
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _needs_vietnamese_retry(text: str) -> bool:
    """True if text is likely English/Chinese rather than Vietnamese."""
    cleaned = _clean_watermark_fragments(_strip_generation_artifacts(text))
    if not cleaned or len(cleaned.strip()) < 4:
        return False
    if CHINESE_RE.search(cleaned):
        return True  # Untranslated Chinese
    if _VI_DIACRITIC_RE.search(cleaned):
        return False  # Vietnamese diacritics present = Vietnamese
    en_words = _EN_WORD_RE.findall(cleaned)
    return len(en_words) >= 2  # 2+ plain Latin words, no VI diacritics = English


def _build_vietnamese_retry_prompt(prompt: str) -> str:
    return (
        "LẦN THỬ LẠI BẮT BUỘC:\n"
        "- Chỉ được trả về tiếng Việt tự nhiên.\n"
        "- Mỗi phần tử trong JSON array phải tương ứng đúng 1 câu nguồn theo đúng thứ tự.\n"
        "- Nếu phần tử là watermark/logo hoặc chứa ACG, com, .com, .net, .org thì trả về chuỗi rỗng \"\" ở đúng vị trí đó.\n"
        "- CẤM tiếng Anh, tiếng Trung, và token rác như </, </|3|>, <|assistant|>, </s>.\n\n"
        f"{prompt}"
    )


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
    text = _strip_generation_artifacts(text)
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
    """Chuẩn hóa tiếng Việt - chỉ sửa lỗi chính tả, KHÔNG thay đổi đại từ nhân xưng."""
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


def _detect_relationship_context(text: str) -> str | None:
    """Phát hiện ngữ cảnh quan hệ từ nội dung text để chọn đại từ nhân xưng phù hợp."""
    parent_child_patterns = [
        r'妈妈|爸爸|母亲|父亲|妈咪|老爸',  # Mẹ, Bố
        r'零花钱|压岁钱|红包',  # Tiền lì xì
        r'学费|补习班|考试',  # Học phí, lớp học thêm
        r'奶奶|爷爷|外婆|外公',  # Bà, Ông
    ]
    
    for pattern in parent_child_patterns:
        if re.search(pattern, text):
            return "parent_child"
    
    romantic_patterns = [
        r'(亲爱的|宝贝)',  # Yêu thương
        r'男朋友|女朋友|未婚夫|未婚妻',  # Bạn trai/gái
        r'老公|老婆|丈夫|妻子',  # Chồng, Vợ
        r'我爱你|我喜欢你|嫁给我',  # Tình yêu
    ]
    
    for pattern in romantic_patterns:
        if re.search(pattern, text):
            return "romantic"
    
    school_friend_patterns = [
        r'(同学|同班|同桌)',  # Bạn cùng lớp
        r'老师|教授|导师',  # Giáo viên
        r'学长|学姐|学弟|学妹',  # Cấp trên/cấp dưới trường học
    ]
    
    for pattern in school_friend_patterns:
        if re.search(pattern, text):
            return "school"
    
    return None


def _apply_relationship_pronouns(translated: str, relationship: str | None) -> str:
    """Adjust pronouns based on detected relationship context.
    
    This function applies pronoun adjustments AFTER translation to ensure consistency.
    It uses context_history from ImageTranslator to maintain pronoun consistency across sentences.
    """
    if not translated or not translated.strip():
        return translated
    
    # Pronoun adjustment rules based on relationship type
    replacements = {
        # Parent-child relationships (mother/father) - child speaks to parent
        "parent_child": [
            (r'\btôi\b', 'con'),  # Replace 'tôi' with 'con' when child speaks to parent
            (r'\bem\b',   'con'),   # Replace 'em' with 'con' when child speaks to parent
        ],
        # Romantic relationships - keep original pronouns (tôi/em are acceptable)
        "romantic": [],
        # School/friend relationships - neutral pronouns are acceptable as-is
        "school": [],
    }
    
    adjusted = translated
    
    if relationship and relationship in replacements:
        for pattern, replacement in replacements[relationship]:
            adjusted = re.sub(pattern, replacement, adjusted, flags=re.IGNORECASE)
    
    return adjusted


def _build_pronoun_context_prompt(context_history: list[tuple[str, str]]) -> str:
    """Build pronoun context prompt from translation history.
    
    Extracts pronoun usage patterns from previous translations to guide current translation.
    """
    if not context_history or len(context_history) < 2:
        return ""
    
    # Get recent exchanges (last 5-6 pairs)
    recent = list(context_history)[-6:]
    
    lines = []
    for orig, trans in recent:
        clean_orig = orig.strip()[:80] if len(orig) > 80 else orig.strip()
        clean_trans = trans.strip()[:120] if len(trans) > 120 else trans.strip()
        
        # Extract pronouns from translation
        pronouns_found = []
        for p in ['con', 'tôi', 'em', 'mẹ', 'ba', 'anh']:
            if re.search(rf'\b{p}\b', clean_trans, flags=re.IGNORECASE):
                pronouns_found.append(p)
        
        if pronouns_found:
            lines.append(f"Source: {clean_orig}")
            lines.append(f"Translation: {clean_trans}")
            lines.append(f"Pronouns used: {', '.join(pronouns_found)}")
    
    return "\n\n".join(lines)


def _fix_pronoun_patterns(translated: str, source: str = "") -> str:
    """Sửa lỗi đại từ nhân xưng phổ biến bằng regex (lớp bảo vệ cuối)."""
    if not translated or not translated.strip():
        return translated
    
    # Quy tắc xưng hô đặc thù cho truyện Trung Quốc/VN
    replacements = {
        r"cầu xin anh.*con\b": lambda m: "con cầu xin anh",  # Sửa lỗi "anh...con" → "con cầu xin anh"
    }
    
    for pattern, replacer in replacements.items():
        if callable(replacer):
            translated = re.sub(pattern, replacer, translated)
        else:
            translated = re.sub(pattern, replacer, translated)
    
    return translated


def _fix_teacher_gender(translated: str) -> str:
    """Sửa lỗi giới tính giáo viên (thầy → cô cho nhân vật nữ)."""
    if not translated or not translated.strip():
        return translated
    
    # Giữ nguyên để tránh sai - OCR fragment ngắn khó biết chắc
    return translated


def _clean_watermark_fragments(translated: str, source: str = "") -> str:
    """Xóa watermark/brand names chưa dịch (ACG, .com, v.v.)."""
    if not translated or not translated.strip():
        return translated

    translated = _strip_generation_artifacts(translated)
    lines = [line.strip() for line in translated.splitlines() if line.strip()]
    if not lines:
        return ""

    filtered_lines = [line for line in lines if not _contains_watermark_text(line)]
    if filtered_lines:
        return "\n".join(filtered_lines)

    if _contains_watermark_text(translated) or _contains_watermark_text(source):
        return ""

    return translated


def comprehensive_post_processing(translated: str, source: str = "") -> str:
    """Xử lý toàn diện cho kết quả dịch."""
    if not translated or not translated.strip():
        return translated
    translated = _strip_generation_artifacts(translated)
    
    # Bước 1: Sửa lỗi đại từ nhân xưng bằng regex (lớp bảo vệ cuối)
    translated = _fix_pronoun_patterns(translated, source)
    
    # Bước 2: Kiểm tra và sửa chữ Hán chưa dịch
    remaining_zh = CHINESE_RE.findall(translated)
    if remaining_zh:
        print(f"⚠️  Lỗi dịch: {len(remaining_zh)} ký tự Hán chưa được dịch trong: {translated[:100]}...")
    
    # Bước 3: Sửa lỗi giới tính giáo viên (nếu cần)
    translated = _fix_teacher_gender(translated)
    
    # Bước 4: Xóa watermark/brand fragments
    translated = _clean_watermark_fragments(translated, source)
    
    return translated


def aggressive_vietnamese_enforcement(text: str, source_text: str = "", max_retries: int = 15) -> str:
    """Áp dụng xử lý mạnh buộc dịch hoàn toàn sang tiếng Việt.
    
    Args:
        text: Văn bản cần dịch
        source_text: Văn bản gốc để so sánh (nếu có)
        max_retries: Số lần thử lại khi phát hiện chữ Hán chưa dịch
        
    Returns:
        Văn bản đã được xử lý, đảm bảo 100% tiếng Việt
    """
    if not text or not text.strip():
        return text
    
    # Bước 1: Kiểm tra xem còn chữ Hán không
    remaining_zh = CHINESE_RE.findall(text)
    
    retry_count = 0
    while remaining_zh and retry_count < max_retries:
        print(f"⚠️  CHỮ HÁN CHƯA DỊCH (thử {retry_count + 1}/{max_retries}): {text[:100]}...")
        
        if remaining_zh in source_text:
            print("⚠️  Lỗi OCR: Một số ký tự Trung chưa được nhận diện")
        
        # Bước 2: Áp dụng xử lý đại từ nhân xưng
        relationship = _detect_relationship_context(source_text)
        text = _apply_relationship_pronouns(text, relationship)
        
        # Bước 3: Loại bỏ watermark và brand names
        text = _clean_watermark_fragments(text, source_text)
        
        # Nếu vẫn còn chữ Hán sau xử lý, thử lại với prompt gắt hơn
        remaining_zh = CHINESE_RE.findall(text)
        retry_count += 1
    
    return text


def regex_based_chinese_fix(translated: str, source: str = "") -> str:
    """Sử dụng regex để sửa lỗi dịch chưa hoàn chỉnh."""
    if not translated or not translated.strip():
        return translated
    
    remaining_zh = CHINESE_RE.findall(translated)
    
    if not remaining_zh:
        return translated
    
    print(f"⚠️  Lỗi dịch: {len(remaining_zh)} ký tự Hán chưa được dịch trong: {translated[:100]}...")
    
    return translated


def post_process_translation(translated_text: str, source_text: str = "") -> str:
    """Xử lý hậu kỳ để cải thiện đại từ nhân xưng và đảm bảo dịch đầy đủ."""
    if not translated_text or not translated_text.strip():
        return translated_text
    translated_text = _strip_generation_artifacts(translated_text)
    if _contains_watermark_text(translated_text) or _contains_watermark_text(source_text):
        return ""
    
    remaining_zh = CHINESE_RE.findall(translated_text)
    
    if remaining_zh:
        print(f"⚠️  Cảnh báo: Phát hiện {len(remaining_zh)} ký tự Hán chưa dịch trong: {translated_text[:100]}...")
    
    relationship = _detect_relationship_context(source_text)
    translated_text = _apply_relationship_pronouns(translated_text, relationship)

    return _clean_watermark_fragments(translated_text, source_text)


def translate_batch(texts: list[str], model: str, src_lang: str = "zh",
                   context_history: list[tuple[str, str]] | None = None,
                   constraints: list[dict] | None = None,
                   force_vietnamese: bool = True,
                   max_retries: int = 3,
                   retry_delay: float = 2.0,
                   timeout: int = 600) -> list[str]:
    """Dịch batch texts qua Ollama API với prompt mạnh buộc dịch sang tiếng Việt.
    
    Args:
        texts: Danh sách văn bản cần dịch
        model: Tên model Ollama (mặc định: qwen3:8b)
        src_lang: Ngôn ngữ nguồn ("zh" = Trung, "en" = Anh)
        context_history: Lịch sử hội thoại gần nhất (~12 trao đổi) để duy trì ngữ cảnh xưng hô
        constraints: Giới hạn độ dài bong bóng (max_chars, max_lines)
        force_vietnamese: Kích hoạt chế độ ép buộc dịch 100% tiếng Việt
        
    Returns:
        Danh sách văn bản đã dịch
    """
    if not texts:
        return []
    
    # ── BUILD CONTEXT PROMPT FROM HISTORY ───────────────────────────────────────
    context_prompt = ""
    if context_history and len(context_history) > 0:
        # Lấy ~5 trao đổi gần nhất để tránh prompt quá dài
        recent_exchanges = list(context_history)[-5:]
        
        context_parts = []
        for orig, trans in recent_exchanges:
            clean_orig = orig.strip()
            clean_trans = trans.strip()
            if clean_orig and clean_trans:
                # Thêm marker để Ollama hiểu đây là ngữ cảnh trước đó
                context_parts.append(f"=== TRƯỚC ĐÓ ===\nOriginal: {clean_orig}\nTranslation: {clean_trans}")
        
        if context_parts:
            context_prompt = "\n\n".join(context_parts) + "\n\n=== KẾT THÚC NGỮ CẢNH ===\n\n"

    # ── BUILD TRANSLATION PROMPT: context first, then rules, then texts ──────────
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    
    # Phần 1: Context history (đặt TRƯỚC rules để model đọc ngữ cảnh trước)
    if context_prompt:
        full_prompt = (
            f"=== NGỮ CẢNH HỘI THOẠI TRƯỚC ĐÓ (tham chiếu xưng hô) ===\n\n"
            f"{context_prompt}\n"
            "=== KẾT THÚC NGỮ CẢNH ===\n"
            "Dựa vào lịch sử trên, hãy dịch nhất quán về xưng hô và ngữ cảnh.\n\n"
        )
    else:
        full_prompt = ""
    
    # Quy tắc dịch thuật - CỰC KỲ GẠT (Force Vietnamese)
    PROMPT_RULES = (
        "\n=== BẮT BUỘC: DỊCH TOÀN BỘ SANG TIẾNG VIỆT ===\n\n"
        "Bạn là một dịch giả chuyên nghiệp. Dịch MỌI nội dung sau sang tiếng Việt.\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "QUY TẮC BẮT BUỘC (TỐI THƯỢNG):\n"
        "1. KHÔNG bao giờ để lại chữ Hán, Anh, hay bất kỳ ngôn ngữ nào khác\n"
        "2. Tên riêng: Dịch âm Hán-Việt (佳佳→Giai Giai, 小明→Tiểu Minh, 美香→Mỹ Hương)\n"
        "3. Nếu text là watermark/logo hoặc chứa ACG, com, .com, .net, .org: trả về chuỗi rỗng \"\"\n"
        "4. Emoji và ký tự đặc biệt: Giữ nguyên\n"
        "5. CẤM tuyệt đối đầu ra tiếng Anh hoặc Trung Quốc\n"
        "6. CẤM xuất token hệ thống hoặc rác như </, </|3|>, <|assistant|>, <|user|>, </s>\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "XỬ LÝ ĐẠI TỪ NHÂN XƯNG (QUAN TRỌNG):\n"
        "\n"
        "A. Quan hệ cha mẹ - con cái:\n"
        "   - Con nói với cha/mẹ: dùng \"con\" cho bản thân, \"mẹ/cha\" hoặc \"ba/mẹ\" cho phụ huynh\n"
        "   - Cha/mẹ nói với con: dùng \"con\" cho con cái\n"
        "   - Ví dụ:\n"
        "     * 妈妈可以把下个月的零花钱提前给我吗？→ Mẹ có thể chuyển tiền lì xì tháng sau cho con trước được không?\n"
        "     * 我这个月的零花钱花没了。→ Tiền lì xì của con tháng này hết rồi.\n\n"
        "B. Quan hệ yêu đương/romantic:\n"
        "   - Người yêu gọi nhau: \"anh/em\", \"tôi/bạn\", hoặc tên riêng\n"
        "   - Không dùng \"con\" trừ khi là quan hệ cha mẹ con cái\n\n"
        "C. Quan hệ bạn bè/cùng trang lứa:\n"
        "   - Dùng \"em/tôi\" hoặc \"anh/em\" tùy độ tuổi và tính cách\n"
        "\n"
        "D. Quan hệ thầy trò/đại học:\n"
        "   - Học sinh gọi giáo viên: \"cô/giáo viên\", \"thầy/cử nhân\"\n"
        "   - Giáo viên gọi học sinh: \"em\" hoặc tên\n\n"
        "E. Quan hệ cấp trên - cấp dưới (công sở):\n"
        "   - Cấp dưới gọi cấp trên: \"anh/chị/boss\" hoặc chức danh\n"
        "   - Cấp trên gọi cấp dưới: \"em\", \"bạn\", hoặc tên\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "XỬ LÝ CỤM TỪ HÁN-VIỆT (QUAN TRỌNG):\n"
        "\n"
        "Các cụm từ Hán cần dịch sang âm Hán-Việt hoặc nghĩa tiếng Việt:\n"
        "- 出嫁 → xuất giá / lấy chồng\n"
        "- 迷倒 → mê mẩn / say đắm\n"
        "- 公子 → công tử / quý tử\n"
        "- 殿下 → thiên hạ / bệ hạ\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "PHONG CÁCH:\n"
        "- Ngắn gọn, phù hợp bong bóng hội thoại\n"
        "- Dùng từ đơn giản, tránh từ dài\n"
        "- KHÔNG thêm giải thích hay bình luận\n"
        "- Nếu câu quá dài: dùng \\n để ngắt dòng\n\n"
    )
    
    # Thêm yêu cầu force Vietnamese nếu được kích hoạt
    if force_vietnamese:
        PROMPT_RULES += (
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "LẶP LẠI TỰ ĐỘNG:\n"
            "Nếu phát hiện chữ Hán trong kết quả, hãy yêu cầu model dịch lại.\n"
            "Mục tiêu: 100% tiếng Việt, không sót ký tự nào.\n\n"
        )
    
    base_prompt = PROMPT_RULES + f"{full_prompt}DỊCH NGAY:\n\n{numbered}\n\n---KẾT QUẢ (chỉ JSON array):---"

    prompt_attempts = [base_prompt]
    if force_vietnamese:
        prompt_attempts.extend([_build_vietnamese_retry_prompt(base_prompt)] * 2)

    def _extract(t):
        if isinstance(t, str):
            return _clean_watermark_fragments(_normalize_vietnamese(_normalize_newlines(t)))
        if isinstance(t, dict):
            for k in ("text", "translation", "result", "output", "translated", "vi"):
                if k in t and isinstance(t[k], str):
                    return _clean_watermark_fragments(_normalize_vietnamese(_normalize_newlines(t[k])))
            for v in t.values():
                if isinstance(v, str):
                    return _clean_watermark_fragments(_normalize_vietnamese(_normalize_newlines(v)))
        return _clean_watermark_fragments(_normalize_vietnamese(_normalize_newlines(str(t))))

    # ── RETRY LOGIC WITH TIMEOUT AND BACKOFF ────────────────────────────────────
    for attempt_idx, prompt_attempt in enumerate(prompt_attempts):
        try:
            payload = {"model": model, "prompt": prompt_attempt, "stream": False}
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers = {"Content-Type": "application/json; charset=utf-8"}
            
            resp = requests.post(
                f"{OLLAMA_BASE}/api/generate",
                data=body,
                headers=headers,
                timeout=timeout,  # Use configurable timeout
            )
            
            if not resp.ok:
                error_msg = f"HTTP {resp.status_code}: {resp.text[:200]}"
                print(f"⚠️  API Error (attempt {attempt_idx + 1}/{len(prompt_attempts)}): {error_msg}")
                
                if resp.status_code in (502, 503, 504, 429) or "connection" in error_msg.lower():
                    wait_time = min(retry_delay * (attempt_idx + 1), 60)
                    print(f"⏳ Chờ {wait_time}s trước khi retry...")
                    time.sleep(wait_time)
                    continue
                
                return []
            
            resp.raise_for_status()
            raw = _strip_generation_artifacts(resp.json().get("response", "") or "")

            s = raw.find("[")
            e = raw.rfind("]") + 1
            if s >= 0 and e > s:
                parsed = json.loads(raw[s:e])
                if isinstance(parsed, list) and len(parsed) == len(texts):
                    extracted = [_extract(t) for t in parsed]
                    
                    if force_vietnamese and any(_needs_vietnamese_retry(t) for t in extracted):
                        print("⚠️  Kết quả không phải tiếng Việt, retry...")
                        continue
                    
                    return extracted
        except requests.exceptions.Timeout:
            print(f"⏱️  Request timeout (attempt {attempt_idx + 1}/{len(prompt_attempts)})")
            if attempt_idx < len(prompt_attempts) - 1:
                wait_time = min(retry_delay * (attempt_idx + 1), 60)
                print(f"⏳ Chờ {wait_time}s trước khi retry...")
                time.sleep(wait_time)
                continue
        except requests.exceptions.ConnectionError as ce:
            print(f"🔌 Connection error: {str(ce)[:200]}")
            if attempt_idx < len(prompt_attempts) - 1:
                wait_time = min(retry_delay * (attempt_idx + 1), 60)
                print(f"⏳ Chờ {wait_time}s trước khi retry...")
                time.sleep(wait_time)
                continue
        except Exception as e:
            error_msg = str(e)[:200]
            print(f"❌ Unexpected error (attempt {attempt_idx + 1}/{len(prompt_attempts)}): {error_msg}")
            if attempt_idx < len(prompt_attempts) - 1:
                wait_time = min(retry_delay * (attempt_idx + 1), 60)
                print(f"⏳ Chờ {wait_time}s trước khi retry...")
                time.sleep(wait_time)
                continue
    
    # Fallback: return original texts if translation fails completely
    print("⚠️  Tất cả attempts thất bại, trả về text gốc.")
    return texts