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

_MIT_INSTALL_HINT = (
    "Cần Python 3.11 và venv riêng.\n"
    "1. Cài Python 3.11 từ python.org\n"
    "2. py -3.11 -m venv mit_venv\n"
    "3. mit_venv\\Scripts\\pip install git+https://github.com/zyddnys/manga-image-translator.git"
)

# Project root = parent of this package directory
_PROJECT_ROOT = Path(__file__).parent.parent

# ── Translation style templates (dùng cho MIT custom_openai) ─────────────────
_GPT_BASE_RULES = """\
[ROLE] Expert Vietnamese manga localizer. OUTPUT ONLY VIETNAMESE.

CORE RULES:
1. Vietnamese ONLY — no CJK, no English, no system tokens (</s>, </, </|3|>, <|assistant|>, <|user|>, etc.) — EXCEPT segment markers <|n|>.
2. NAMES → Sino-Vietnamese: 佳佳→Giai Giai, 陈伟→Trần Vỹ, 北京→Bắc Kinh.
3. VOLUMES: 上→Phần 1, 中→Phần 2, 下→Phần 3 / Phần cuối.
4. WATERMARK/URL/logo/ACG/handle → return "" (empty, nothing else).
5. SFX → Vietnamese: 嘭→Bùm, 哈哈哈→hà hà hà, 哼→hừ, 嘿嘿嘿→hắc hắc hắc.
6. FORMAT: Start EVERY translated segment with its input marker <|n|> (same number, same order). Content may span multiple lines for long text. No JSON/numbering/other markup.
7. LENGTH: Keep translations concise. For long text use \\n within the segment to break lines (≤4 words/sub-line).
8. PUNCT: every segment ends with . ! ? or … Never start a line with . , ! ?
9. TONE: RAW — literal, preserve harsh/vulgar/blunt tone, no softening.
10. GLOSSARY: 老师→cô giáo | 催眠→thôi miên | The End→Hết | 小 (prefix)→Tiểu
11. FALLBACK: garbled OCR → translate visible + [...]. Empty/URL input → return "".
12. NATURAL & IN-CONTEXT WORDING: Pick the word that fits the SCENE and the speaker, NOT the stiff dictionary gloss. Use everyday spoken Vietnamese; match register to the mood. In intimate/erotic scenes prefer direct colloquial verbs — 插入/放进去/塞进去 → "đút vào" / "cho vào" (NOT the clinical "chèn vào"); 摸→sờ, 舔→liếm, 抱→ôm. Keep names, pronouns and recurring terms consistent with the STORY CONTEXT block (recent pages) when one is provided above.
"""

_GPT_STYLE_MODERN = """\
PRONOUNS (Modern / Slice-of-life):
- Default: 我→anh/em/tôi (context), 你→bạn/em/anh (context).
- Parent-child: child="con", parent="mẹ/ba". NEVER "tôi" for child speaking to parent.
  * 妈妈可以给我吗？→ Mẹ có cho con không?
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
        font_size_offset: str = "",
        font_size_minimum: str = "",
        font_size_fixed: str = "",
        font_color: str = "",
        verbose: bool = False,
        skip_no_text: bool = False,
        overwrite: bool = False,
        cpu_priority: str = "below_normal",
        gpt_style: str = "",
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
        self.font_size_offset      = font_size_offset
        self.font_size_minimum     = font_size_minimum
        self.font_size_fixed       = font_size_fixed
        self.font_color            = font_color
        self.verbose               = verbose
        self.skip_no_text          = skip_no_text
        self.overwrite             = overwrite
        self.gpt_style             = gpt_style if gpt_style in _GPT_STYLE_BLOCKS else "modern"
        self.cpu_priority = cpu_priority if cpu_priority in ("normal", "below_normal", "idle") else "below_normal"
        self.on_log       = on_log or print
        self.on_progress  = on_progress or (lambda d, t: None)

    def _log(self, msg: str):
        self.on_log(msg)

    def process_folder(
        self,
        input_dir: str,
        output_dir: str,
        stop_event: threading.Event | None = None,
    ) -> tuple[int, int, list[str]]:
        from ._image_translator import IMAGE_EXTS

        if not self.python_path:
            self._log("  [FAIL] Không tìm thấy Python có manga_translator.")
            self._log(f"  [FAIL] {_MIT_INSTALL_HINT}")
            return 0, 0, []

        inp = Path(input_dir)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

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

        # Auto-inject gpt_config for custom_openai
        if self.translator == "custom_openai":
            gpt_cfg = _PROJECT_ROOT / "gpt_config_vi.yaml"
            if gpt_cfg.exists():
                if self.gpt_style and self.gpt_style != "modern":
                    # Tạo temp config với style-specific template
                    style_block = _GPT_STYLE_BLOCKS.get(self.gpt_style, _GPT_STYLE_MODERN)
                    composed = _GPT_BASE_RULES + "\n" + style_block + "\nTranslate the following text into Vietnamese:\n"
                    indented = "\n".join("    " + line if line else "" for line in composed.splitlines())
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

        cmd = [self.python_path, "-m", "manga_translator"]
        if self.use_gpu:
            cmd.append("--use-gpu")
        if self.verbose:
            cmd.append("--verbose")
        cmd += ["local", "-i", str(inp), "-o", str(out), "--config-file", cfg_path]
        if self.skip_no_text:
            cmd.append("--skip-no-text")
        if self.overwrite:
            cmd.append("--overwrite")

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
            sub_env = os.environ.copy()
            sub_env["PYTHONIOENCODING"] = "utf-8"
            sub_env["PYTHONUTF8"] = "1"
            sub_env["PYTHONUNBUFFERED"] = "1"
            if self.translator == "custom_openai" and self.ollama_model:
                sub_env["CUSTOM_OPENAI_MODEL"] = self.ollama_model
                self._log(f"  [ENV] CUSTOM_OPENAI_MODEL={self.ollama_model}")
            if self.translator == "custom_openai" and self.custom_openai_api_base:
                sub_env["CUSTOM_OPENAI_API_BASE"] = self.custom_openai_api_base
                self._log(f"  [ENV] CUSTOM_OPENAI_API_BASE={self.custom_openai_api_base}")
            if self.translator == "custom_openai" and self.custom_openai_api_key:
                sub_env["CUSTOM_OPENAI_API_KEY"] = self.custom_openai_api_key
                self._log("  [ENV] CUSTOM_OPENAI_API_KEY=***")

            _thread_limit = "4" if self.cpu_priority == "normal" else "3" if self.cpu_priority == "below_normal" else "2"
            for _env_key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
                sub_env[_env_key] = _thread_limit

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
        except Exception as exc:
            self._log(f"  [FAIL] Lỗi chạy manga_translator: {exc}")
        finally:
            _stop_watch.set()
            wt.join(timeout=2)
            try:
                os.unlink(cfg_path)
            except Exception:
                pass

        ok = sum(
            1 for f in out.rglob("*")
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS
        )
        fail = max(0, total - ok)
        self.on_progress(ok, total)
        self._log(f"  [OK] Kết quả: {ok} ảnh trong {out}")
        return ok, fail, []
