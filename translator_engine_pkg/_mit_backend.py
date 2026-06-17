"""
_mit_backend.py — manga-image-translator backend (MITImageTranslator).
"""

import json
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from ._translate import OLLAMA_BASE
from ._common_utils import save_image_compressed

_MIT_INSTALL_HINT = (
    "Cần Python 3.11 và venv riêng.\n"
    "1. Cài Python 3.11 từ python.org\n"
    "2. py -3.11 -m venv mit_venv\n"
    "3. mit_venv\\Scripts\\pip install git+https://github.com/zyddnys/manga-image-translator.git"
)

# Project root = parent of this package directory
_PROJECT_ROOT = Path(__file__).parent.parent

# Subfolder inside an input dir where the web editor stores manual text-region
# sidecars: <input_dir>/.manga_regions/<image_filename>.json
REGIONS_DIRNAME = ".manga_regions"


def resolve_story_root(inp: Path) -> Path:
    """Gốc bộ truyện = thư mục CHA của thư mục 'original' gần nhất trên đường dẫn
    (kể cả chính inp). Glossary/pre_dict của bộ đặt ở đây để dùng chung cho MỌI
    chương, CHỈ áp cho bộ đó.

    Ví dụ:
      F:\\Comic\\[..]\\original\\chap01  → F:\\Comic\\[..]      (bộ nhiều chương)
      F:\\Comic\\[..]\\original          → F:\\Comic\\[..]      (one-shot dịch cả original)
      <không có 'original' trên path>    → inp.parent          (như pre_dict cũ)
    """
    try:
        inp = Path(inp).resolve()
    except Exception:
        inp = Path(inp)
    for anc in (inp, *inp.parents):
        if anc.name.lower() == "original":
            return anc.parent
    return inp.parent


def _load_regions(regions_dir: Path, image_name: str) -> dict | None:
    """Đọc sidecar vùng thủ công → {"mode", "regions"} hoặc None nếu không có.
    Mỗi region: {x,y,w,h} và (tùy chọn) "text" — chữ Việt gõ tay (bỏ qua OCR/dịch)."""
    f = regions_dir / (image_name + ".json")
    if not f.is_file():
        return None
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return None
    regions = [r for r in (data.get("regions") or []) if isinstance(r, dict)]
    if not regions:
        return None
    mode = data.get("mode", "merge")
    if mode not in ("merge", "replace"):
        mode = "merge"
    try:
        mask_dilate = max(0, min(6, int(data.get("mask_dilate", 1))))
    except (TypeError, ValueError):
        mask_dilate = 1
    return {"mode": mode, "regions": regions, "mask_dilate": mask_dilate}

# ── Translation style templates (dùng cho MIT custom_openai) ─────────────────
_GPT_BASE_RULES = """\
[ROLE] Expert Vietnamese manga localizer. OUTPUT ONLY VIETNAMESE.

CORE RULES:
1. Vietnamese ONLY — no CJK, no English, no system tokens (</s>, </, </|3|>, <|assistant|>, <|user|>, etc.) — EXCEPT segment markers <|n|>.
2. NAMES → âm Hán-Việt CHUẨN cho TỪNG chữ, VIẾT HOA chữ đầu mỗi tiếng, KHÔNG đoán bừa, KHÔNG thêm/bớt chữ: 佳佳→Giai Giai, 陈伟→Trần Vỹ, 苑静梅→Viên Tĩnh Mai (梅=Mai, KHÔNG "Miễu"), 北京→Bắc Kinh.
2b. TU TIÊN / VÕ HIỆP (cảnh giới, công pháp, thuật) → PHIÊN ÂM Hán-Việt, GIỮ NGUYÊN, TUYỆT ĐỐI KHÔNG diễn giải/dịch nghĩa (NGOẠI LỆ của rule 12 — KHÔNG biến cảnh giới thành mô tả, vd KHÔNG dịch 筑基后期 thành "phá vỡ giới hạn"). Cảnh giới: 炼气/练气→Luyện Khí, 筑基→Trúc Cơ (CHỈ 筑基), 结丹/結丹→Kết Đan (KHÔNG phải Trúc Cơ, KHÔNG phải Kim Đan), 金丹→Kim Đan, 元婴→Nguyên Anh, 化神→Hóa Thần, 炼虚→Luyện Hư, 合体→Hợp Thể, 大乘→Đại Thừa, 渡劫→Độ Kiếp. Giai đoạn: 初期/前期→sơ kỳ, 中期→trung kỳ, 后期→hậu kỳ, 巅峰→đỉnh phong, 圆满→viên mãn (vd 筑基后期→Trúc Cơ hậu kỳ). Môn phái & chức danh (VIẾT HOA như tên riêng): 门→Môn, 宗→Tông, 派→Phái, 阁→Các (vd 妙音门→Diệu Âm Môn, 妙=Diệu KHÔNG "Âm"); 左使/右使→Tả Sứ/Hữu Sứ, 长老→Trưởng Lão, 掌门→Chưởng Môn, 门主→Môn Chủ, 宗主→Tông Chủ, 弟子→đệ tử, 夫人→phu nhân; 人称→thường được gọi là. Thuật ngữ: 修炼→tu luyện, 凡修→phàm tu, 魔修→ma tu, 凡间/凡尘→phàm gian/phàm trần, 仙→tiên, 妖→yêu, 魔→ma, 突破→đột phá, 功法→công pháp, 法术/之术/术→pháp thuật/thuật, 灵气→linh khí, 真气→chân khí, 音律→âm luật (音律攻击→âm công), 媚惑之术/媚术→mị hoặc thuật / mị thuật.
3. VOLUMES: 上→Phần 1, 中→Phần 2, 下→Phần 3 / Phần cuối.
4. WATERMARK/URL/logo/ACG/handle → return "" (empty, nothing else).
5. SFX & tiếng CƯỜI → Vietnamese (KHÔNG để nguyên chữ Hán): 嘭→Bùm, 哈哈哈→hà hà hà, 哼→hừ, 嘿嘿嘿→hắc hắc hắc, 咯咯→khúc khích, 嘻嘻→hì hì, 呵呵→ha ha, 嘤嘤→hu hu, 噗嗤→phì cười.
6. FORMAT: Start EVERY translated segment with its input marker <|n|>, then a TYPE TAG in square brackets, then the Vietnamese translation. Same number/order. No JSON/other markup.
6b. TYPE TAG (REQUIRED, right after <|n|>, exactly one): [speech]=lời nói trong bong bóng (MẶC ĐỊNH — phân vân thì chọn cái này) | [thought]=suy nghĩ/độc thoại nội tâm/cảm giác cơ thể của nhân vật (vd "Toàn thân nóng rực", "Có tác dụng chưa?") — kể cả khi nằm trong bong bóng | [moan]=rên/khoái cảm (haa,hừ,ah♡) | [shout]=hét to | [anger]=giận dữ/gắt gỏng/đe doạ (cáu nhưng không nhất thiết gào; nếu vừa giận vừa hét → ưu tiên [anger]) | [fear]=sợ hãi/run rẩy/lắp bắp ("K-không thể…","Đ-đừng lại gần!") | [narration]=CHỈ lời NGƯỜI KỂ thuật lại bối cảnh/diễn biến (giọng kể khách quan, thường ngôi thứ 3) — TUYỆT ĐỐI không dùng cho suy nghĩ, cảm giác hay lời nói của nhân vật | [sfx]=tượng thanh THUẦN (câu có CẢ tiếng cười/rên LẪN lời nói → phân loại theo LỜI NÓI [speech]/[thought], KHÔNG phải [sfx]/[moan]) | [title]=tiêu đề bộ truyện/chương & chữ TRANG TRÍ (thư pháp, con dấu, logo truyện — thường kèm số chương 一/二/（五）; OCR hay đọc sai thành chuỗi Hán vô nghĩa) → trả ĐÚNG "<|n|>[title]", KHÔNG dịch, KHÔNG thêm chữ nào (vùng sẽ bị xoá khỏi ảnh, không render). Ví dụ: <|1|>[thought] Đầu óc cứ bồng bềnh… | <|2|>[anger] Mày dám! | <|3|>[fear] X-xin… đừng… | <|4|>[narration] Ba ngày sau, tại học viện… | <|5|>[title]
6c. MỖI SEGMENT CHỈ DỊCH ĐÚNG NỘI DUNG CỦA CHÍNH NÓ: segment có vẻ bị cắt dở thì dịch đúng phần dở đó. TUYỆT ĐỐI KHÔNG gộp nhiều segment thành một, KHÔNG dồn bản dịch của segment sau lên slot trước — số lượng và thứ tự bản dịch phải khớp 1:1 với input.
7. LENGTH: Dịch ĐẦY ĐỦ nghĩa — KHÔNG lược bỏ, KHÔNG rút gọn nội dung (vd 我在凡修当魔 → "Ta tu ma nơi phàm tu", KHÔNG cắt thành "Ta tu ma"). Viết LIỀN MẠCH thành câu trọn vẹn — KHÔNG tự chèn \\n để xuống dòng (việc xếp chữ cho vừa bong bóng do renderer tự wrap theo bề rộng khung + cỡ font, chính xác hơn).
8. PUNCT (BẮT BUỘC — KHÔNG segment nào được thiếu dấu kết): MỌI segment PHẢI kết thúc bằng đúng MỘT dấu hợp ngữ cảnh — câu hỏi→"?", quát/hét/giận/ngạc nhiên→"!", bỏ lửng/ngập ngừng/rên/cảm xúc lấp lửng→"…", câu trần thuật trung tính→".". Trong cảnh kịch tính/cảm xúc ƯU TIÊN "…" hoặc "!" thay cho dấu "." khô khan (tự nhiên hơn và NHÌN RÕ hơn); chỉ dùng "." cho câu thật sự bình thản. Segment kết thúc mà THIẾU dấu là SAI — phải thêm dấu vào. Không bắt đầu dòng bằng . , ! ? LẮP BẮP/NGẬP NGỪNG: nguồn lặp chữ để diễn lắp bắp (你，你… / 不、不… / 我…我) → GIỮ nét lắp trong tiếng Việt bằng cách lặp âm đầu có gạch nối, vd 你，你怎么又那个了→"C-con, con sao lại 'cái đó' nữa rồi…", 不、不要→"Đ-đừng…". TUYỆT ĐỐI KHÔNG làm trơn thành câu thường (đừng bỏ phần lặp).
9. TONE: RAW — literal, preserve harsh/vulgar/blunt tone, no softening.
10. GLOSSARY: 老师→cô giáo | 催眠→thôi miên | The End→Hết | 小 (prefix)→Tiểu | 媚药→thuốc kích dục | 春药→thuốc kích dục | 迷药/麻药/麻醉药→thuốc mê | 魔头→ma đầu (KHÔNG dịch nghĩa khác)
11. FALLBACK: garbled OCR trong THOẠI → translate visible + [...]. NHƯNG nếu CẢ segment là chuỗi Hán vô nghĩa KHÔNG thành câu, kiểu tiêu đề/con dấu/chữ trang trí → trả "<|n|>[title]" (rule 6b), KHÔNG cố dịch bừa. Empty/URL input → return "".
12. NATURAL & IN-CONTEXT WORDING: Pick the word that fits the SCENE and the speaker, NOT the stiff dictionary gloss. Use everyday spoken Vietnamese; match register to the mood. In intimate/erotic scenes prefer direct colloquial verbs — 插入/放进去/塞进去 → "đút vào" / "cho vào" (NOT the clinical "chèn vào"); 摸→sờ, 舔→liếm, 抱→ôm. Keep names, pronouns and recurring terms consistent with the STORY CONTEXT block (recent pages) when one is provided above.
"""

_GPT_STYLE_MODERN = """\
PRONOUNS (Modern / Slice-of-life):
- Default: 我→anh/em/tôi (context), 你→bạn/em/anh (context).
- Parent-child: child="con", parent="mẹ/ba". NEVER "tôi" for child speaking to parent.
  * 妈妈可以给我吗？→ Mẹ có cho con không?
  * ƯU TIÊN CAO: nếu trong CÙNG TRANG (bất kỳ segment hay narration nào) có 妈妈/母亲/妈/娘 hoặc 爸爸/父亲/爸/爹 → đây là cảnh CHA MẸ–CON: 我/你 giữa hai người này LUÔN là con↔mẹ/ba (我→con khi người nói là con; 你→con khi nói VỚI con; mẹ tự xưng→mẹ, ba tự xưng→ba). TUYỆT ĐỐI KHÔNG dùng anh/em kiểu tình cảm, dù STORY CONTEXT các trang trước gợi ý anh/em.
- Romance: anh/em (casual: tôi/bạn).
- Teacher: "cô/thầy". Student: "em". Peer: "tôi/bạn".
- Ambiguous context → default to neutral "em/tôi".
"""

_GPT_STYLE_WUXIA = """\
PRONOUNS (Cổ trang / Wuxia) — STRICT:
- 我 ALWAYS → "ta" (self, any gender).
- 你 ALWAYS → "ngươi".
  * 你是谁？→ Ngươi là ai? | 我不会告诉你 → Ta sẽ không nói cho ngươi.
WUXIA VOCABULARY:
- 公子→công tử | 姑娘/小姑娘→cô nương/tiểu cô nương | 女侠→nữ hiệp
- 采花贼/淫贼→dâm tặc | 迷药→mê dược | 肏→chịch/đụ
- 把你奸了→ta sẽ cưỡng hiếp ngươi | 轮奸→cưỡng hiếp tập thể
- 嘿嘿嘿→hắc hắc hắc | 哈哈哈→hà hà hà | 哼→hừ
"""

_GPT_STYLE_SCHOOL = """\
PRONOUNS (Học đường / School):
- Student (我) → "em". Teacher (我) → "tôi/cô/thầy" (context).
- 你 (student→teacher) → "cô/thầy". 你 (peer) → "bạn/cậu".
- Teacher: "cô giáo" or "thầy" (match gender context). Student: "em".
- Class group: "chúng em / cả lớp". Romantic (age-gap): anh/em.
"""

_GPT_STYLE_LIGHTNOVEL = """\
PRONOUNS (Light Novel / Manga Nhật):
- 私/わたし (watashi) → "tôi" (neutral/formal) or "mình" (casual).
- 僕/ぼく (boku) → "mình" (casual male, soft) or "tôi".
- 俺/おれ (ore) → "tao" (brash/rough) or "tớ" (casual).
- 俺様 (ore-sama) → "ta" (arrogant self-address).
- あなた/君/きみ (anata/kimi) → "bạn" or "cậu" (peer). Context: "em" if romantic.
- お前/おまえ (omae) → "mày" (rough) or "cậu" (casual).
- 先生 (sensei) → "thầy" or "cô" (match gender). Never translate as "giáo viên".
- 先輩 (senpai) → "senpai" (keep as-is) or "đàn anh/đàn chị".
- 後輩 (kouhai) → "đàn em" or "hậu bối".
- 俺の嫁 / 嫁 → "vợ tao" / "vợ".
JAPANESE SFX → Vietnamese: バン→Bàng, ドン→Đùng, ズキズキ→Nhói nhói, キャー→Kyaaa, ドキドキ→Tim đập mạnh.
LIGHT NOVEL TONE:
- Preserve internal monologue style (italics in source → translate directly).
- Isekai/fantasy titles: 勇者→dũng sĩ, 魔王→ma vương, 転生→chuyển sinh, 異世界→dị giới.
- Honorifics: -san→"san" or drop, -kun→drop or "cậu", -chan→"chan" or drop, -sama→"sama" or "đại nhân".
"""

_GPT_STYLE_BLOCKS: dict[str, str] = {
    "modern":     _GPT_STYLE_MODERN,
    "wuxia":      _GPT_STYLE_WUXIA,
    "school":     _GPT_STYLE_SCHOOL,
    "lightnovel": _GPT_STYLE_LIGHTNOVEL,
}


def _find_mit_python() -> str | None:
    """
    Tìm Python có manga_translator đã cài.
    Ưu tiên: mit_venv trong thư mục project → fallback py.exe -3.11.
    """
    def _has_manga_translator(python_exe: Path) -> bool:
        try:
            r = subprocess.run(
                [str(python_exe), "-c",
                 "import importlib.util; print(importlib.util.find_spec('manga_translator') is not None)"],
                capture_output=True, text=True, timeout=10,
            )
            return r.returncode == 0 and r.stdout.strip().lower().startswith("true")
        except Exception:
            return False

    candidates = [
        _PROJECT_ROOT / "mit_venv" / "Scripts" / "python.exe",
        _PROJECT_ROOT / "mit_venv" / "bin" / "python",
    ]
    for path in candidates:
        if path.exists() and _has_manga_translator(path):
            return str(path)

    for py_flag in ["-3.11", "-3.10"]:
        try:
            r = subprocess.run(
                ["py", py_flag, "-c", "import sys; print(sys.executable)"],
                capture_output=True, text=True, timeout=8,
            )
            if r.returncode == 0:
                exe = Path(r.stdout.strip())
                if exe.exists() and _has_manga_translator(exe):
                    return str(exe)
        except Exception:
            pass
    return None


def _python_imports_ok(python_exe: Path, modules: list[str]) -> tuple[bool, str]:
    try:
        imports = "; ".join(f"import {m}" for m in modules)
        r = subprocess.run(
            [str(python_exe), "-c", f"{imports}; print('OK')"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
        stdout = (r.stdout or "").strip()
        stderr = (r.stderr or "").strip()
        if r.returncode == 0 and stdout.endswith("OK"):
            return True, ""
        if stderr:
            return False, stderr
        if stdout:
            return False, stdout
        return False, f"returncode={r.returncode}"
    except Exception as exc:
        return False, str(exc)


def check_mit() -> dict:
    """Kiểm tra manga-image-translator đã cài và tìm Python phù hợp."""
    exe = _find_mit_python()
    if not exe:
        return {"ok": False, "error": _MIT_INSTALL_HINT}

    ok, error = _python_imports_ok(Path(exe), ["manga_translator", "torch", "PIL"])
    if ok:
        return {"ok": True, "version": "installed", "python": exe}

    return {
        "ok": False,
        "python": exe,
        "error": (
            f"Đã tìm thấy Python: {exe}, nhưng thiếu dependency runtime: {error}\n"
            f"Cài thêm package thiếu trong venv đó, ví dụ:\n"
            f"{exe} -m pip install pillow torch --index-url https://download.pytorch.org/whl/cu124"
        ),
    }


class MITImageTranslator:
    """
    Backend dùng manga-image-translator.
    Gọi `python -m manga_translator translate …` qua subprocess.
    """

    def __init__(
        self,
        translator: str = "m2m100_big",
        target_lang: str = "VIN",
        use_gpu: bool = True,
        python_path: str | None = None,
        detector: str = "",
        inpainter: str = "lama_mpe",
        inpainting_size: str = "",
        inpainting_precision: str = "",
        ollama_model: str = "",
        custom_openai_api_base: str = "",
        custom_openai_api_key: str = "",
        upscale_ratio: str = "",
        upscaler: str = "",
        detection_size: str = "",
        mask_dilation_offset: str = "",
        unclip_ratio: str = "",
        box_threshold: str = "",
        text_threshold: str = "",
        det_invert: bool = False,
        det_gamma_correct: bool = False,
        det_rotate: bool = False,
        det_auto_rotate: bool = False,
        ocr_model: str = "",
        ocr_prob: str = "",
        font_size_offset: str = "",
        font_size_minimum: str = "",
        font_size_fixed: str = "",
        font_color: str = "",
        narrow_width_mult: str = "",
        narrow_font_cap: str = "",
        verbose: bool = False,
        skip_no_text: bool = False,
        overwrite: bool = False,
        cpu_priority: str = "below_normal",
        gpt_style: str = "",
        image_quality: int = 95,
        use_global_glossary: bool = False,
        on_log=None,
        on_progress=None,
    ):
        self.translator            = translator
        self.target_lang           = target_lang
        self.use_gpu               = use_gpu
        self.python_path           = python_path or _find_mit_python()
        self.detector              = detector
        self.inpainter             = inpainter
        self.inpainting_size       = inpainting_size
        self.inpainting_precision  = inpainting_precision
        self.ollama_model          = ollama_model
        self.custom_openai_api_base = custom_openai_api_base
        self.custom_openai_api_key  = custom_openai_api_key
        self.upscale_ratio         = upscale_ratio
        self.upscaler              = upscaler
        self.detection_size        = detection_size
        self.mask_dilation_offset  = mask_dilation_offset
        self.unclip_ratio          = unclip_ratio
        self.box_threshold         = box_threshold
        self.text_threshold        = text_threshold
        self.det_invert            = bool(det_invert)
        self.det_gamma_correct     = bool(det_gamma_correct)
        self.det_rotate            = bool(det_rotate)
        self.det_auto_rotate       = bool(det_auto_rotate)
        self.ocr_model             = ocr_model
        self.ocr_prob              = ocr_prob
        self.font_size_offset      = font_size_offset
        self.font_size_minimum     = font_size_minimum
        self.font_size_fixed       = font_size_fixed
        self.font_color            = font_color
        self.narrow_width_mult     = narrow_width_mult
        self.narrow_font_cap       = narrow_font_cap
        self.verbose               = verbose
        self.skip_no_text          = skip_no_text
        self.overwrite             = overwrite
        self.gpt_style             = gpt_style if gpt_style in _GPT_STYLE_BLOCKS else "modern"
        self.cpu_priority = cpu_priority if cpu_priority in ("normal", "below_normal", "idle") else "below_normal"
        # Mức nén ảnh đầu ra vùng thủ công: 40–100. <100 = giảm dung lượng (PNG → JPEG nén).
        try:
            self.image_quality = max(40, min(100, int(image_quality)))
        except (TypeError, ValueError):
            self.image_quality = 95
        # Áp glossary DÙNG CHUNG (global) cho lần dịch này: chỉ hợp lý với
        # custom_openai (cơ chế bơm prompt + enforce nằm ở translator đó).
        self.use_global_glossary = bool(use_global_glossary)
        self.on_log       = on_log or print
        self.on_progress  = on_progress or (lambda d, t: None)
        self._pre_dict_path: str | None = None
        self._glossary_path: str | None = None

    def _log(self, msg: str):
        self.on_log(msg)

    def process_folder(
        self,
        input_dir: str,
        output_dir: str,
        stop_event: threading.Event | None = None,
        images_override: list | None = None,
    ) -> tuple[int, int, list[str]]:
        from ._image_translator import IMAGE_EXTS

        if not self.python_path:
            self._log("  [FAIL] Không tìm thấy Python có manga_translator.")
            self._log(f"  [FAIL] {_MIT_INSTALL_HINT}")
            return 0, 0, []

        inp = Path(input_dir)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Gốc bộ truyện (cha của 'original') — nơi đặt pre_dict.txt & glossary.txt
        # dùng chung cho MỌI chương, CHỈ áp cho bộ này.
        story_root = resolve_story_root(inp)

        # Từ điển sửa NGUỒN per-truyện (MIT --pre-dict): áp lên text sau OCR,
        # TRƯỚC khi dịch — chốt các lỗi OCR cố hữu của bộ (vd 阿访→阿诚 khi OCR
        # đọc nhầm tên nhân vật). Tra theo thứ tự: thư mục ảnh → thư mục cha →
        # gốc bộ (để bộ nhiều chương dùng chung 1 file). Mỗi dòng: <regex_sai> <thay_bằng>.
        self._pre_dict_path = next(
            (str(c) for c in (inp / "pre_dict.txt", inp.parent / "pre_dict.txt",
                              story_root / "pre_dict.txt")
             if c.is_file()),
            None,
        )
        if self._pre_dict_path:
            self._log(f"  [PRE-DICT] Từ điển sửa nguồn của bộ: {self._pre_dict_path}")

        # Glossary per-truyện (tên/xưng hô/cảnh giới) — bộ nhớ BỀN, đặt ở gốc bộ,
        # custom_openai đọc qua env MIT_GLOSSARY_PATH: bơm vào prompt + enforce sau
        # dịch + TỰ HỌC (ghi thêm tên mới sau mỗi lần chạy). Đường dẫn cố định ở
        # gốc bộ kể cả khi chưa tồn tại — lần chạy đầu sẽ tạo file.
        self._glossary_path = None
        if self.translator == "custom_openai":
            existing = next(
                (str(c) for c in (inp / "glossary.txt", inp.parent / "glossary.txt",
                                  story_root / "glossary.txt")
                 if c.is_file()),
                None,
            )
            self._glossary_path = existing or str(story_root / "glossary.txt")
            self._log(f"  [GLOSSARY] Glossary bộ truyện (tự học): {self._glossary_path}")

        # images_override = chỉ dịch các ảnh đã chọn (list đường dẫn đầy đủ),
        # None/rỗng = dịch cả thư mục như cũ. Sidecar vùng thủ công vẫn đọc từ
        # thư mục gốc (inp) bất kể có staging hay không.
        if images_override:
            images = sorted(
                (Path(p) for p in images_override
                 if Path(p).suffix.lower() in IMAGE_EXTS and Path(p).is_file()),
                key=lambda p: p.name,
            )
        else:
            images = sorted(f for f in inp.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        if not images:
            self._log("Không tìm thấy ảnh trong thư mục.")
            return 0, 0, []

        total = len(images)
        self._log(f"Tổng: {total} ảnh — manga-image-translator")
        self._log(f"  Translator : {self.translator}  →  {self.target_lang}")
        self.on_progress(0, total)

        cfg: dict = {
            "translator": {
                "translator": self.translator,
                "target_lang": self.target_lang,
            },
        }
        inp_cfg: dict = {}
        if self.inpainter:
            inp_cfg["inpainter"] = self.inpainter
        if self.inpainting_size:
            inp_cfg["inpainting_size"] = int(self.inpainting_size)
        if self.inpainting_precision:
            inp_cfg["inpainting_precision"] = self.inpainting_precision
        if inp_cfg:
            cfg["inpainter"] = inp_cfg
        if self.detector:
            cfg["detector"] = {"detector": self.detector}
        if self.detection_size:
            cfg.setdefault("detector", {})["detection_size"] = int(self.detection_size)
        if self.mask_dilation_offset:
            cfg["mask_dilation_offset"] = int(self.mask_dilation_offset)
        if self.unclip_ratio:
            cfg.setdefault("detector", {})["unclip_ratio"] = float(self.unclip_ratio)
        if self.box_threshold:
            cfg.setdefault("detector", {})["box_threshold"] = float(self.box_threshold)
        if self.text_threshold:
            cfg.setdefault("detector", {})["text_threshold"] = float(self.text_threshold)
        if self.det_invert:
            cfg.setdefault("detector", {})["det_invert"] = True
        if self.det_gamma_correct:
            cfg.setdefault("detector", {})["det_gamma_correct"] = True
        if self.det_rotate:
            cfg.setdefault("detector", {})["det_rotate"] = True
        if self.det_auto_rotate:
            cfg.setdefault("detector", {})["det_auto_rotate"] = True
        if self.ocr_model:
            cfg.setdefault("ocr", {})["ocr"] = self.ocr_model
        if self.ocr_prob:
            # Lọc rác OCR: bỏ vùng chữ có prob < ngưỡng (MIT mặc định 0.2 → lọt rác
            # khi box/text threshold để thấp). Vd 0.5 loại ký tự lẻ đọc bậy.
            cfg.setdefault("ocr", {})["prob"] = float(self.ocr_prob)
            self._log(f"  [OCR] Lọc vùng prob < {float(self.ocr_prob):.2f} (loại OCR đọc bậy)")
        if self.upscale_ratio:
            # revert_upscaling=True: phóng to để detect/inpaint ở res cao rồi THU NHỎ
            # về kích thước gốc → nét hơn, đúng cỡ chữ, không để lại bản upscale nhòe.
            up_cfg: dict = {
                "upscale_ratio": int(self.upscale_ratio),
                "revert_upscaling": True,
            }
            # Mặc định MIT là esrgan (bôi nhòe halftone manga); ưu tiên waifu2x cho line-art.
            up_cfg["upscaler"] = self.upscaler or "waifu2x"
            cfg["upscale"] = up_cfg
            self._log(f"  [UPSCALE] ×{self.upscale_ratio} upscaler={up_cfg['upscaler']} revert=True")
        if self.font_size_offset:
            cfg.setdefault("render", {})["font_size_offset"] = int(self.font_size_offset)
        if self.font_size_minimum:
            min_val = int(self.font_size_minimum)
            if self.font_size_offset:
                ofs_val = int(self.font_size_offset)
                if ofs_val < -8:
                    adjusted = max(8, min_val + ofs_val // 2)
                    if adjusted < min_val:
                        cfg.setdefault("render", {})["font_size_minimum"] = adjusted
                        self._log(
                            f"  [RENDER] font_size_minimum {min_val}→{adjusted} "
                            f"(tránh triệt tiêu offset={ofs_val})"
                        )
                    else:
                        cfg.setdefault("render", {})["font_size_minimum"] = min_val
                else:
                    cfg.setdefault("render", {})["font_size_minimum"] = min_val
            else:
                cfg.setdefault("render", {})["font_size_minimum"] = min_val
        if self.font_size_fixed:
            cfg.setdefault("render", {})["font_size"] = int(self.font_size_fixed)
        if self.font_color:
            cfg.setdefault("render", {})["font_color"] = self.font_color

        # Font Việt hóa — ưu tiên MTO Astro City, fallback NotoSans
        _vi_font_priority = [
            "MTO Astro City.ttf",
            # "NotoSans-Regular.ttf",
            # "BeVietnamPro-Regular.ttf",
        ]
        for _fn in _vi_font_priority:
            _fp = _PROJECT_ROOT / "fonts" / _fn
            if _fp.exists():
                cfg.setdefault("render", {})["font_path"] = str(_fp)
                self._log(f"  [FONT] Dùng font Việt: {_fn}")
                break

        # Vietnamese auto-apply
        # Bỏ fixed font_size — để MIT tự scale theo từng bubble.
        # Chỉ áp offset nhẹ (-8) để bù cho ký tự Latin rộng hơn CJK (~1.3x).
        if self.target_lang in ("VIN", "vi") and not self.font_size_fixed:
            pass  # Không cố định font_size — MIT sẽ tự tính
        if self.target_lang in ("VIN", "vi") and not self.font_size_offset and not self.font_size_fixed:
            cfg.setdefault("render", {})["font_size_offset"] = 2
            self._log("  [RENDER] font_size_offset=+2 (bù chữ VI; fit-loop trong patch tự thu nếu tràn)")
        if self.target_lang in ("VIN", "vi") and not self.font_size_minimum:
            cfg.setdefault("render", {})["font_size_minimum"] = 16
        if self.target_lang in ("VIN", "vi") and not self.unclip_ratio:
            cfg.setdefault("detector", {})["unclip_ratio"] = 3.5
            self._log("  [DETECT] Auto unclip_ratio=3.5 (Latin/VI text rộng hơn CJK)")
        # Giảm mask_dilation_offset (MIT mặc định 20) → mask inpaint sát nét chữ hơn,
        # bớt ăn lan vào artwork (mảng trắng/mất nét ở SFX đè lên hình). Tăng lại nếu
        # còn sót viền chữ gốc; giảm thêm (vd 0) nếu vẫn phá hình.
        if self.target_lang in ("VIN", "vi") and not self.mask_dilation_offset:
            cfg["mask_dilation_offset"] = 6
            self._log("  [INPAINT] Auto mask_dilation_offset=6 (mask sát nét, đỡ phá hình quanh SFX)")
        # Căn giữa thoại trong khung (đỡ lệch khi vùng text gốc đặt lệch trong bong bóng).
        if self.target_lang in ("VIN", "vi"):
            cfg.setdefault("render", {}).setdefault("alignment", "center")

        # Auto-inject gpt_config for custom_openai
        if self.translator == "custom_openai":
            gpt_cfg = _PROJECT_ROOT / "gpt_config_vi.yaml"
            if gpt_cfg.exists():
                if self.gpt_style and self.gpt_style != "modern":
                    # Tạo temp config với style-specific template
                    style_block = _GPT_STYLE_BLOCKS.get(self.gpt_style, _GPT_STYLE_MODERN)
                    composed = _GPT_BASE_RULES + "\n" + style_block + "\nTranslate the following text into Vietnamese:\n"
                    indented = "\n".join("    " + line if line else "" for line in composed.splitlines())
                    # temperature/top_p chỉ là fallback "sáng tác" — sampling thật do
                    # custom_openai _DETERMINISTIC quyết định (mặc định greedy, ghi đè
                    # các giá trị này). Giữ khớp với gpt_config_vi.yaml.
                    yaml_content = (
                        f"ollama:\n"
                        f"  temperature: 0.1\n"
                        f"  top_p: 0.85\n"
                        f"  chat_system_template: |\n"
                        f"{indented}\n"
                    )
                    tf_style = tempfile.NamedTemporaryFile(
                        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
                    )
                    tf_style.write(yaml_content)
                    tf_style.close()
                    cfg.setdefault("translator", {})["gpt_config"] = tf_style.name
                    self._log(f"  [GPT] Style: {self.gpt_style} → temp config {tf_style.name}")
                else:
                    cfg.setdefault("translator", {})["gpt_config"] = str(gpt_cfg)
                    self._log(f"  [GPT] Using custom gpt_config: {gpt_cfg.name}")

        tf = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump(cfg, tf, ensure_ascii=False)
        tf.close()
        cfg_path = tf.name

        # ── Manual-region detection ────────────────────────────────────────
        # Images with hand-drawn boxes (saved by the web editor as
        # .manga_regions/<name>.json) are reprocessed individually after the
        # batch pass, with MIT_MANUAL_REGIONS set so the detection patch injects
        # the boxes. Coordinates are normalized 0..1 → resolution independent.
        manual_jobs: list[tuple[Path, dict]] = []
        regions_dir = inp / REGIONS_DIRNAME
        if regions_dir.is_dir():
            for img in images:
                data = _load_regions(regions_dir, img.name)
                if data:
                    manual_jobs.append((img, data))
        manual_cfg_path = None
        if manual_jobs:
            self._log(f"  [MANUAL] {len(manual_jobs)} ảnh có vùng thủ công — chèn box lên ảnh đã dịch (không dịch lại cả trang).")
            # Config cho box OCR: detector=none → CHỈ box vẽ tay; bỏ upscale để
            # giữ nguyên trang đã dịch (không upscale/nhoè lại).
            cfg_manual = json.loads(json.dumps(cfg))
            cfg_manual["detector"] = {"detector": "none"}
            cfg_manual.pop("upscale", None)
            tfm = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
            json.dump(cfg_manual, tfm, ensure_ascii=False)
            tfm.close()
            manual_cfg_path = tfm.name

        # Ảnh ở chế độ Replace KHÔNG chạy lượt batch tự động — chỉ dịch đúng các
        # box vẽ tay (ở pass 2, đặt lên ảnh gốc). Tránh dịch lại cả trang đã dịch.
        replace_names = {img.name for img, data in manual_jobs if data["mode"] == "replace"}
        pass1_images = [im for im in images if im.name not in replace_names]
        run_pass1 = bool(pass1_images)

        # MIT là CLI theo thư mục: khi tập ảnh cần batch ≠ toàn bộ thư mục (do chọn
        # subset hoặc loại ảnh Replace) → trỏ MIT vào thư mục staging tạm.
        staging: Path | None = None
        if run_pass1 and (images_override or replace_names):
            import shutil as _shutil
            staging = Path(tempfile.mkdtemp(prefix="mit_select_"))
            for im in pass1_images:
                try:
                    _shutil.copy2(im, staging / im.name)
                except Exception as exc:
                    self._log(f"  [SELECT] Bỏ qua {im.name}: {exc}")
            scan_inp = staging
        else:
            scan_inp = inp
        if images_override:
            self._log(f"  [SELECT] Chỉ dịch {len(images)} ảnh đã chọn.")
        if replace_names:
            self._log(f"  [MANUAL] {len(replace_names)} ảnh Replace — bỏ qua batch, chỉ dịch vùng vẽ tay.")

        # ── Pass 1: batch the (selected) images (automatic detection) ──────
        cmd = [self.python_path, "-m", "manga_translator"]
        if self.use_gpu:
            cmd.append("--use-gpu")
        if self.verbose:
            cmd.append("--verbose")
        if self._pre_dict_path:
            cmd += ["--pre-dict", self._pre_dict_path]
        cmd += ["local", "-i", str(scan_inp), "-o", str(out), "--config-file", cfg_path]
        if self.skip_no_text:
            cmd.append("--skip-no-text")
        if self.overwrite:
            cmd.append("--overwrite")

        if run_pass1:
            self._log(f"  [CMD] {' '.join(str(c) for c in cmd)}")

        _last: list[int] = [0]
        _stop_watch = threading.Event()

        def _watcher():
            while not _stop_watch.is_set():
                try:
                    cnt = sum(
                        1 for f in out.rglob("*")
                        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
                    )
                    if cnt != _last[0]:
                        _last[0] = cnt
                        self.on_progress(cnt, total)
                except Exception:
                    pass
                time.sleep(0.8)

        wt = threading.Thread(target=_watcher, daemon=True)
        wt.start()

        try:
            if run_pass1:
                self._spawn_mit(cmd, stop_event=stop_event)
            else:
                self._log("  [MANUAL] Không có ảnh dịch tự động — chỉ chạy vùng vẽ tay.")

            # ── Pass 2: chèn box vẽ tay lên ảnh đã dịch (detector=none) ─────
            if manual_jobs and not (stop_event and stop_event.is_set()):
                self._run_manual_jobs(manual_jobs, out, manual_cfg_path, stop_event)
        except Exception as exc:
            self._log(f"  [FAIL] Lỗi chạy manga_translator: {exc}")
        finally:
            _stop_watch.set()
            wt.join(timeout=2)
            try:
                os.unlink(cfg_path)
            except Exception:
                pass
            if manual_cfg_path:
                try:
                    os.unlink(manual_cfg_path)
                except Exception:
                    pass
            if staging is not None:
                import shutil as _shutil
                _shutil.rmtree(staging, ignore_errors=True)

        if images_override:
            sel_names = {im.name for im in images}
            ok = sum(
                1 for f in out.rglob("*")
                if f.is_file() and f.name in sel_names and f.suffix.lower() in IMAGE_EXTS
            )
        else:
            ok = sum(
                1 for f in out.rglob("*")
                if f.is_file() and f.suffix.lower() in IMAGE_EXTS
            )
        fail = max(0, total - ok)
        self.on_progress(ok, total)
        self._log(f"  [OK] Kết quả: {ok} ảnh trong {out}")
        return ok, fail, []

    def _spawn_mit(self, cmd, stop_event: threading.Event | None = None, extra_env: dict | None = None):
        """Run one `manga_translator` subprocess, stream its logs, and wait.
        extra_env is merged into the child environment (e.g. MIT_MANUAL_REGIONS)."""
        sub_env = os.environ.copy()
        sub_env["PYTHONIOENCODING"] = "utf-8"
        sub_env["PYTHONUTF8"] = "1"
        sub_env["PYTHONUNBUFFERED"] = "1"
        # Nới ngang vùng chữ DỌC HẸP (patch rendering đọc env này). Rỗng/≤1 = tắt.
        if str(self.narrow_width_mult or "").strip():
            try:
                _nw = float(self.narrow_width_mult)
            except (TypeError, ValueError):
                _nw = 0.0
            if _nw > 1.0:
                sub_env["MIT_NARROW_WIDTH_MULT"] = str(_nw)
                self._log(f"  [RENDER] Nới ngang vùng dọc hẹp ×{_nw:g}")
        # Trần cỡ chữ (px) cho vùng nới-ngang (patch rendering đọc env này). Rỗng/≤0 = tắt.
        if str(self.narrow_font_cap or "").strip():
            try:
                _fc = float(self.narrow_font_cap)
            except (TypeError, ValueError):
                _fc = 0.0
            if _fc > 0:
                sub_env["MIT_NARROW_FONT_CAP"] = str(int(_fc))
                self._log(f"  [RENDER] Trần cỡ chữ vùng nới-ngang ≤ {int(_fc)}px")
        if self.translator == "custom_openai" and self.ollama_model:
            sub_env["CUSTOM_OPENAI_MODEL"] = self.ollama_model
            self._log(f"  [ENV] CUSTOM_OPENAI_MODEL={self.ollama_model}")
        if self.translator == "custom_openai" and self.custom_openai_api_base:
            sub_env["CUSTOM_OPENAI_API_BASE"] = self.custom_openai_api_base
            self._log(f"  [ENV] CUSTOM_OPENAI_API_BASE={self.custom_openai_api_base}")
        if self.translator == "custom_openai" and self.custom_openai_api_key:
            sub_env["CUSTOM_OPENAI_API_KEY"] = self.custom_openai_api_key
            self._log("  [ENV] CUSTOM_OPENAI_API_KEY=***")
        if self.translator == "custom_openai" and getattr(self, "_glossary_path", None):
            sub_env["MIT_GLOSSARY_PATH"] = self._glossary_path
            self._log(f"  [ENV] MIT_GLOSSARY_PATH={self._glossary_path}")
        # Glossary DÙNG CHUNG (global, SQLite): khi user bật, truyền đường dẫn kho +
        # cờ để custom_openai hợp nhất mục đã duyệt vào prompt/enforce và đẩy mục tự
        # học lên kho (chờ duyệt). Đường dẫn kho lấy từ glossary_store (gốc repo).
        if self.translator == "custom_openai" and getattr(self, "use_global_glossary", False):
            try:
                import glossary_store as _gstore
                sub_env["MIT_GLOSSARY_DB"] = _gstore.db_path()
                sub_env["MIT_GLOSSARY_USE_GLOBAL"] = "1"
                self._log(f"  [ENV] MIT_GLOSSARY_DB={sub_env['MIT_GLOSSARY_DB']} (áp glossary chung)")
            except Exception as _e:
                self._log(f"  [GLOSSARY] Không nạp được kho global: {_e}")

        _thread_limit = "4" if self.cpu_priority == "normal" else "3" if self.cpu_priority == "below_normal" else "2"
        for _env_key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                         "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
            sub_env[_env_key] = _thread_limit
        if extra_env:
            sub_env.update(extra_env)

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=sub_env,
        )

        try:
            import psutil as _psutil
            _PRIORITY_MAP = {
                "normal":       _psutil.NORMAL_PRIORITY_CLASS,
                "below_normal": _psutil.BELOW_NORMAL_PRIORITY_CLASS,
                "idle":         _psutil.IDLE_PRIORITY_CLASS,
            }
            p = _psutil.Process(proc.pid)
            p.nice(_PRIORITY_MAP.get(self.cpu_priority, _psutil.BELOW_NORMAL_PRIORITY_CLASS))
            total_cores = _psutil.cpu_count(logical=True) or 4
            if total_cores > 4 and self.cpu_priority != "normal":
                p_cores = [c for c in range(2, min(total_cores, 12), 2)]
                e_cores = [c for c in range(12, min(total_cores, 16))]
                if self.cpu_priority == "idle" and e_cores:
                    ai_cores = e_cores
                else:
                    ai_cores = p_cores
                p.cpu_affinity(ai_cores)
                self._log(f"  [CPU] Affinity={ai_cores}, priority={self.cpu_priority} (PID={proc.pid})")
            else:
                self._log(f"  [CPU] Priority={self.cpu_priority} (PID={proc.pid}, all cores)")
        except Exception as _pe:
            self._log(f"  [CPU] Không set được affinity: {_pe}")

        _LOG_BATCH_INTERVAL = {"normal": 0.05, "below_normal": 0.1, "idle": 0.2}
        _flush_interval = _LOG_BATCH_INTERVAL.get(self.cpu_priority, 0.1)
        _log_buf: list[str] = []
        _last_flush = time.monotonic()
        for line in proc.stdout:
            line = line.rstrip("\n")
            if line:
                _log_buf.append(line)
            if stop_event and stop_event.is_set():
                proc.terminate()
                for _bl in _log_buf:
                    self._log(f"  {_bl}")
                self._log("⚠  Đã dừng theo yêu cầu.")
                _log_buf.clear()
                break
            _now = time.monotonic()
            if _now - _last_flush >= _flush_interval:
                for _bl in _log_buf:
                    self._log(f"  {_bl}")
                _log_buf.clear()
                _last_flush = _now
        for _bl in _log_buf:
            self._log(f"  {_bl}")
        proc.wait()
        return proc.returncode

    def _run_manual_jobs(self, jobs, out_dir: Path, manual_cfg_path: str, stop_event: threading.Event | None):
        """Pass 2: xử lý các vùng vẽ tay LÊN ảnh đã dịch ở pass 1 (merge) hoặc lên
        ảnh gốc (replace), KHÔNG dịch lại cả trang. Mỗi vùng có 3 loại:
          • inpaint_only → CHỈ xoá sạch cả khung (inpaint), không OCR/dịch/vẽ chữ.
            Dùng để dọn lại vùng vẫn còn chữ/nhiễu sót.
          • Có "text" (gõ tay) → xoá ĐÚNG NÉT chữ trong box (giữ halftone/hoạt cảnh)
            rồi vẽ thẳng chữ Việt; bỏ qua OCR/dịch.
          • Không "text" → đưa qua MIT (detector=none) để OCR+dịch+inpaint+render."""
        import shutil
        for img_path, data in jobs:
            if stop_event and stop_event.is_set():
                break
            mode = data.get("mode", "merge")
            regions = data.get("regions") or []
            inpaint_regions = [r for r in regions if r.get("inpaint_only")]
            _rest         = [r for r in regions if not r.get("inpaint_only")]
            # Vùng "Xoá phẳng" không gõ chữ cũng xử lý cục bộ (lấp màu nền) — KHÔNG
            # đưa qua MIT: lama inpaint cả khung sẽ xoá mất panel mà người dùng muốn giữ.
            ocr_regions   = [r for r in _rest if not str(r.get("text") or "").strip() and not r.get("erase_flat")]
            typed_regions = [r for r in _rest if str(r.get("text") or "").strip() or r.get("erase_flat")]
            # Vùng gõ tay + vùng chỉ-xoá đều xử lý cục bộ (đọc/ghi ảnh 1 lần).
            render_regions = typed_regions + inpaint_regions

            translated = out_dir / img_path.name
            on_translated = (mode != "replace" and translated.is_file())
            base = translated if on_translated else img_path
            self._log(
                f"  [MANUAL] {img_path.name} — {'trên ảnh đã dịch' if on_translated else 'trên ảnh gốc'} "
                f"({mode}): {len(ocr_regions)} box OCR, {len(typed_regions)} box gõ tay, "
                f"{len(inpaint_regions)} box chỉ xoá."
            )

            try:
                if ocr_regions:
                    # MIT đọc base (qua bản copy tạm) + chèn box OCR → ghi ra out/name
                    payload = json.dumps({"mode": mode, "regions": ocr_regions}, ensure_ascii=False)
                    self._mit_manual_pass(base, img_path.name, out_dir, manual_cfg_path, payload, stop_event)
                elif base != translated:
                    # Không có box OCR → đảm bảo out/name = base để vẽ/xoá vùng tay lên
                    shutil.copy2(base, translated)

                if render_regions and not (stop_event and stop_event.is_set()):
                    self._render_typed_regions(translated, render_regions, mask_dilate=data.get("mask_dilate", 1))
            except Exception as exc:
                self._log(f"  [MANUAL] Lỗi xử lý {img_path.name}: {exc}")

    def _mit_manual_pass(self, source: Path, out_name: str, out_dir: Path,
                         manual_cfg_path: str, payload: str, stop_event):
        """Chạy MIT 1 ảnh với box OCR vẽ tay (detector=none), ghi đè out_dir/out_name."""
        import shutil
        tmp_in = tempfile.mkdtemp(prefix="mit_manual_")
        try:
            shutil.copy2(source, Path(tmp_in) / out_name)
            cmd = [self.python_path, "-m", "manga_translator"]
            if self.use_gpu:
                cmd.append("--use-gpu")
            if self.verbose:
                cmd.append("--verbose")
            if getattr(self, "_pre_dict_path", None):
                cmd += ["--pre-dict", self._pre_dict_path]
            cmd += ["local", "-i", tmp_in, "-o", str(out_dir),
                    "--config-file", manual_cfg_path, "--overwrite"]
            self._spawn_mit(cmd, stop_event=stop_event,
                            extra_env={"MIT_MANUAL_REGIONS": payload})
        finally:
            shutil.rmtree(tmp_in, ignore_errors=True)

    def _mit_inpaint(self, img_bgr, mask):
        """Inpaint mask bằng inpainter của MIT (lama_large) qua helper chạy trong
        mit_venv — KHÔNG qua OCR nên xoá được cả SFX không đọc nổi, tái tạo halftone
        đẹp như bong bóng. Trả ảnh BGR đã inpaint, hoặc None nếu lỗi."""
        import subprocess
        import shutil
        import cv2
        if not self.python_path:
            return None
        helper = _PROJECT_ROOT / "mit_inpaint_helper.py"
        if not helper.exists():
            return None
        td = tempfile.mkdtemp(prefix="mit_inp_")
        try:
            ip = Path(td) / "in.png"; mp = Path(td) / "mask.png"; op = Path(td) / "out.png"
            cv2.imwrite(str(ip), img_bgr)
            cv2.imwrite(str(mp), mask)
            inpainter = self.inpainter or "lama_large"
            size = str(self.inpainting_size or 2048)
            prec = self.inpainting_precision or "bf16"
            device = "cuda" if self.use_gpu else "cpu"
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"; env["PYTHONUTF8"] = "1"
            self._log(f"  [MANUAL] Inpaint nét chữ bằng MIT {inpainter} (device={device})…")
            r = subprocess.run(
                [self.python_path, str(helper), str(ip), str(mp), str(op), inpainter, size, prec, device],
                capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=600,
            )
            if op.exists():
                out = cv2.imread(str(op))
                if out is not None:
                    return out
            self._log(f"  [MANUAL] MIT inpaint helper lỗi: {((r.stderr or '') + (r.stdout or ''))[-200:]}")
            return None
        except Exception as exc:
            self._log(f"  [MANUAL] MIT inpaint lỗi: {exc}")
            return None
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def _parse_font_color(self):
        """Parse self.font_color ('FG' hoặc 'FG:BG' hex, vd 'FFFFFF:000000') →
        (text_rgb|None, stroke_rgb|None). None = để render tự chọn theo nền."""
        raw = (self.font_color or "").strip().lstrip("#")
        if not raw:
            return None, None

        def _hx(s):
            s = s.strip().lstrip("#")
            if len(s) != 6:
                return None
            try:
                return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
            except ValueError:
                return None

        parts = raw.split(":")
        fg = _hx(parts[0]) if parts and parts[0] else None
        bg = _hx(parts[1]) if len(parts) > 1 and parts[1] else None
        return fg, bg

    @staticmethod
    def _ring_bg_color(roi):
        """Màu nền (BGR uint8) ước lượng từ VÀNH ngoài box (~12% mỗi cạnh, trung vị
        — robust với vài pixel chữ chạm mép). Dùng cho chế độ "Xoá phẳng": lấp cả
        khung bằng màu của CHÍNH panel thay vì inpaint (inpaint cả khung sẽ vẽ lại
        cảnh phía sau như không có panel)."""
        import numpy as np
        h, w = roi.shape[:2]
        m = max(1, int(round(min(h, w) * 0.12)))
        ring = np.ones((h, w), dtype=bool)
        if h > 2 * m and w > 2 * m:
            ring[m:h - m, m:w - m] = False
        return np.median(roi[ring].reshape(-1, 3), axis=0).astype(np.uint8)

    @staticmethod
    def _stroke_mask(roi):
        """Tách NÉT chữ khỏi nền bong bóng trong 1 box gõ tay → mask uint8 (255=nét).

        Auto-dịch xoá sạch hơn vì mask bám nét chữ (text detector + CRF). Box gõ tay
        KHÔNG qua detector nên trước đây dùng Otsu XÁM + 'lớp thiểu số' (nét = lớp ít
        pixel hơn) — hỏng với SFX màu/viền: chữ pink trên bong bóng sáng có độ sáng
        gần nhau → Otsu xám tách trượt, buộc người dùng bật 'Xoá khung' (xoá cả nền).

        Cách mới: ước lượng MÀU NỀN từ VÀNH ngoài box (nơi ít chữ nhất) rồi lấy pixel
        KHÁC XA nền theo khoảng cách màu Lab làm nét → bắt được chữ MÀU mà không cần
        giả định thiểu số. Đơn sắc tối-trên-sáng vẫn đúng (Lab distance ~ luminance).
        Có bảo hiểm: nếu nền ước lượng hỏng (nét ~cả box / gần rỗng) → fallback Otsu
        xám cũ, nên không hồi quy các trường hợp đang chạy tốt."""
        import cv2
        import numpy as np
        h, w = roi.shape[:2]
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
        # Nền = trung vị màu của vành ngoài (~12% mỗi cạnh) — robust với vài pixel chữ
        # chạm mép. Vành rỗng (box quá nhỏ) → lấy trung vị cả ROI.
        m = max(1, int(round(min(h, w) * 0.12)))
        ring = np.ones((h, w), dtype=bool)
        if h > 2 * m and w > 2 * m:
            ring[m:h - m, m:w - m] = False
        bg = np.median(lab[ring], axis=0)
        dist = np.linalg.norm(lab - bg[None, None, :], axis=2)
        dn = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _t, fg = cv2.threshold(dn, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        frac = float((fg > 0).mean())
        if frac > 0.6 or frac < 0.004:
            # Nền ước lượng không đáng tin → quay về Otsu xám + lớp thiểu số như cũ.
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _t2, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            dark_frac = float((th == 0).mean())
            fg = ((th == 0) if dark_frac <= 0.5 else (th == 255)).astype(np.uint8) * 255
        return fg

    def _render_typed_regions(self, image_path: Path, typed_regions: list, mask_dilate: int = 1):
        """Vẽ chữ Việt gõ tay lên ảnh + xoá các vùng "chỉ inpaint". Xoá theo MẶT NẠ
        NÉT CHỮ (chỉ những pixel nét SFX trong box) với vùng gõ tay, hoặc theo CẢ
        KHUNG với vùng inpaint_only, rồi inpaint (lama_large của MIT, fallback TELEA)
        — giữ lại halftone/hoạt cảnh xung quanh. mask_dilate = số vòng dãn mask quanh
        nét (0 = sát nét nhất, ít loang trắng; cao hơn = xoá rộng/sạch hơn).
        Vùng inpaint_only không vẽ chữ — chỉ làm sạch."""
        import cv2
        import numpy as np
        from PIL import Image
        from ._render import render_text

        # cv2.imread chết với path Unicode (vd thư mục tiếng Trung) trên Windows →
        # đọc qua np.fromfile + imdecode (giống _image_translator).
        try:
            img = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        except OSError:
            img = None
        if img is None:
            self._log(f"  [MANUAL] Không đọc được {image_path.name} để vẽ chữ tay.")
            return
        h, w = img.shape[:2]

        boxes = []
        for r in typed_regions:
            try:
                x = float(r["x"]); y = float(r["y"]); rw = float(r["w"]); rh = float(r["h"])
            except (KeyError, TypeError, ValueError):
                continue
            x0 = int(round(max(0.0, min(1.0, x)) * w)); y0 = int(round(max(0.0, min(1.0, y)) * h))
            x1 = int(round(max(0.0, min(1.0, x + rw)) * w)); y1 = int(round(max(0.0, min(1.0, y + rh)) * h))
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            # font (tên file trong fonts/) + cỡ chữ px do người dùng chọn cho vùng này.
            font_name = str(r.get("font") or "").strip().replace("\\", "/").split("/")[-1]
            try:
                font_px = int(r.get("font_size") or 0) or None
            except (TypeError, ValueError):
                font_px = None
            # Siết mask riêng cho vùng (None = dùng mặc định của ảnh).
            try:
                rdil = max(0, min(10, int(r["mask_dilate"])))
            except (KeyError, TypeError, ValueError):
                rdil = int(mask_dilate)
            # Góc nghiêng chữ (độ); chỉ áp cho vùng gõ tay.
            try:
                rot = float(r.get("rotate") or 0.0)
            except (TypeError, ValueError):
                rot = 0.0
            # erase_box: vùng gõ tay nhưng xoá CẢ KHUNG (như inpaint_only) thay vì
            # chỉ nét — dùng cho chữ SFX màu/viền mà Otsu tách không sạch.
            # erase_flat: lấp CẢ KHUNG bằng màu nền (không inpaint) — cho chữ trên
            # chat box/panel màu phẳng: inpaint cả khung xoá mất panel, xoá theo nét
            # thì sót vệt (chữ ≈ màu nền). flat_color '#rrggbb' = màu người dùng tự
            # chọn; None = auto (trung vị vành ngoài box).
            fcol = None
            fc = str(r.get("flat_color") or "").strip().lstrip("#")
            if len(fc) == 6:
                try:
                    fcol = (int(fc[4:6], 16), int(fc[2:4], 16), int(fc[0:2], 16))  # BGR
                except ValueError:
                    fcol = None
            boxes.append((x0, y0, x1, y1, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                          str(r.get("text") or ""), font_name, font_px,
                          bool(r.get("inpaint_only")), rdil, rot, bool(r.get("erase_box")),
                          bool(r.get("erase_flat")), fcol))
        if not boxes:
            return

        # 1) Mặt nạ xoá. Vùng gõ tay: tách NÉT chữ khỏi nền theo màu (_stroke_mask) →
        #    chỉ xoá nét đó, giữ nền/halftone (bắt được cả chữ SFX màu). Vùng
        #    inpaint_only / "Xoá khung": xoá CẢ KHUNG (người dùng chủ động dọn sạch).
        mask = np.zeros((h, w), dtype=np.uint8)
        for (x0, y0, x1, y1, _bbox, _text, _font, _fpx, _ipo, _rdil, _rot, _ebox, _eflat, _fcol) in boxes:
            roi = img[y0:y1, x0:x1]
            if roi.size == 0:
                continue
            if _eflat:
                # Xoá PHẲNG: lấp cả khung bằng màu người dùng chọn, hoặc màu nền của
                # chính panel (vành ngoài box) — KHÔNG inpaint, panel giữ nguyên.
                img[y0:y1, x0:x1] = np.array(_fcol, dtype=np.uint8) if _fcol else self._ring_bg_color(roi)
                continue
            if _ipo or _ebox:
                mask[y0:y1, x0:x1] = 255  # cả khung (chỉ-xoá, hoặc gõ tay bật "xoá cả khung")
                continue
            sm = self._stroke_mask(roi)
            # Dãn mask quanh nét: ít vòng = sát nét, đỡ "lỗ" lama phải đoán → bớt
            # loang sáng; nhiều vòng = xoá rộng/sạch hơn (siết mask riêng từng vùng).
            if _rdil > 0:
                sm = cv2.dilate(sm, np.ones((3, 3), np.uint8), iterations=int(_rdil))
            mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], sm)
        if mask.any():
            out = self._mit_inpaint(img, mask)
            if out is not None and out.shape[:2] == img.shape[:2]:
                img = out  # lama_large của MIT (tái tạo halftone đẹp)
            else:
                img = cv2.inpaint(img, mask, 4, cv2.INPAINT_TELEA)  # fallback

        # 2) Vẽ chữ Việt lên vùng đã xoá.
        fonts_dir = _PROJECT_ROOT / "fonts"
        default_fp = fonts_dir / "MTO Astro City.ttf"
        default_font = str(default_fp) if default_fp.exists() else None

        def _resolve_typed_font(name: str) -> str | None:
            """Tên font người dùng chọn → đường dẫn trong fonts/ (fallback Astro City)."""
            if name:
                cand = fonts_dir / name
                if cand.exists():
                    return str(cand)
            return default_font

        fg_col, bg_col = self._parse_font_color()
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        for i, (_x0, _y0, _x1, _y1, bbox, text, font_name, font_px, _ipo, _rdil, _rot, _ebox, _eflat, _fcol) in enumerate(boxes):
            if text.strip() and not _ipo:
                img_pil = render_text(img_pil, bbox, text, _resolve_typed_font(font_name),
                                      strict_clip=True, font_scale=1.0, bbox_index=i,
                                      text_color=fg_col, stroke_color=bg_col,
                                      font_px=font_px, rotate=_rot)

        written = save_image_compressed(img_pil, image_path, self.image_quality)
        # PNG→JPEG đổi đuôi → ghi đè in-place: xoá file gốc để không còn 2 bản (.png + .jpg).
        if written != image_path and image_path.exists():
            try:
                image_path.unlink()
            except OSError:
                pass
        _n_typed = sum(1 for b in boxes if b[5].strip() and not b[8])
        _n_ipo = sum(1 for b in boxes if b[8])
        self._log(f"  [MANUAL] Đã xử lý {_n_typed} vùng chữ tay, {_n_ipo} vùng chỉ xoá vào {written.name}.")
