"""
Image Crawler Web UI – Giao diện web để crawl và tải ảnh từ web.
Chạy: python web_app.py  →  tự mở http://localhost:5000
"""

import json
import mimetypes
import os
import re
import string
import sys
import threading
from collections import deque

# Đảm bảo stdout dùng UTF-8 trên Windows (tránh UnicodeEncodeError)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

from crawler import crawl, retry_failed_downloads
import translator_engine as te

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # 1 MB — JSON payload only

# ── Trạng thái toàn cục (một phiên crawl tại một thời điểm) ──────────────────
_lock = threading.Lock()
_logs: deque = deque()          # Circular buffer sự kiện
_log_total: int = 0             # Absolute counter (không bao giờ reset bởi eviction)
_crawl_queue: deque = deque()
_state: dict = {
    "running": False,
    "done": 0,
    "total": 0,
    "success": 0,
    "failed": 0,
    "queue": 0,
    "status": "ready",          # ready | running | done | error | stopped
    "failed_items": [],        # list of {"url": str, "index": int}
    "crawl_folder":  "",
    "crawl_referer": "",
}
_stop_event = threading.Event()   # ← signal dừng crawl
MAX_LOG = 3000


def _push(entry: dict) -> None:
    """Thêm entry vào circular buffer (thread-safe, xoay vòng khi full)."""
    global _log_total
    with _lock:
        if len(_logs) >= MAX_LOG:
            _logs.popleft()  # bỏ entry cũ nhất
        _logs.append(entry)
        _log_total += 1


def _enqueue_job(job: dict) -> None:
    with _lock:
        _crawl_queue.append(job)
        _state["queue"] = len(_crawl_queue)


def _parse_urls(raw_url) -> list[str]:
    if isinstance(raw_url, list):
        urls = [str(u).strip() for u in raw_url if str(u).strip()]
    else:
        raw = str(raw_url or "")
        urls = [part.strip() for part in re.split(r'[\r\n,;]+', raw) if part.strip()]
    return [u for u in urls if u.startswith(("http://", "https://"))]


def _crawl_queue_worker() -> None:
    cumulative_done = 0
    try:
        while True:
            with _lock:
                if _stop_event.is_set() or not _crawl_queue:
                    break
                job = _crawl_queue.popleft()
                _state["queue"] = len(_crawl_queue)
            urls = job["urls"]
            folder = job["folder"]
            delay = job["delay"]
            workers = job["workers"]
            timeout = job["timeout"]
            retries = job["retries"]

            _push({"type": "log", "msg": "─" * 60})
            _push({"type": "log", "msg": f"▶ Bắt đầu job queue: {len(urls)} URL"})
            for url in urls:
                if _stop_event.is_set():
                    break
                _push({"type": "log", "msg": f"  ▶ URL: {url}"})

                job_total = 0
                def on_progress(done: int, total: int) -> None:
                    nonlocal job_total
                    job_total = total
                    with _lock:
                        _state["done"] = cumulative_done + done
                        _state["total"] = cumulative_done + total
                    _push({"type": "progress", "done": cumulative_done + done, "total": cumulative_done + total})

                ok, fail, failed_items = crawl(
                    url=url,
                    output_folder=folder,
                    delay=delay,
                    max_workers=workers,
                    timeout=timeout,
                    max_retries=retries,
                    on_progress=on_progress,
                    on_log=lambda msg: _push({"type": "log", "msg": str(msg)}),
                    stop_event=_stop_event,
                )

                cumulative_done += job_total
                with _lock:
                    _state["success"] += ok
                    _state["failed"] += fail
                    _state["failed_items"].extend(failed_items)
                _push({"type": "log", "msg": f"  ✔ Job URL hoàn tất: {ok} thành công, {fail} thất bại."})
                if _stop_event.is_set():
                    break
    except Exception as exc:
        _push({"type": "log", "msg": f"✘  Lỗi queue: {exc}"})
        with _lock:
            _state.update(running=False, status="error")
        return

    with _lock:
        _state.update(running=False, status="stopped" if _stop_event.is_set() else "done")
    _push({"type": "done", "success": _state["success"], "failed": _state["failed"]})
    _push({"type": "log", "msg": f"✔  Hoàn thành queue: {_state['success']} thành công, {_state['failed']} thất bại."})


@app.route("/api/start", methods=["POST"])
def api_start():
    with _lock:
        running = _state["running"]

    data = request.get_json(silent=True) or {}
    raw_urls = data.get("urls", data.get("url", ""))
    urls = _parse_urls(raw_urls)
    folder = str(data.get("folder", "output")).strip() or "output"
    try:
        delay = max(0.0, min(10.0, float(data.get("delay", 0.3))))
        workers = max(1, min(32, int(data.get("workers", 4))))
        timeout = max(5, min(300, int(data.get("timeout", 20))))
        retries = max(0, min(10, int(data.get("retries", 3))))
    except (TypeError, ValueError):
        return jsonify({"error": "Giá trị cấu hình không hợp lệ."}), 400

    if not urls:
        return jsonify({"error": "Phải nhập ít nhất một URL hợp lệ."}), 400

    job = {
        "urls": urls,
        "folder": folder,
        "delay": delay,
        "workers": workers,
        "timeout": timeout,
        "retries": retries,
    }

    if not running:
        _reset()
        _enqueue_job(job)
        _push({"type": "log", "msg": f"▶ Bắt đầu queue mới: {len(urls)} URL"})
        threading.Thread(target=_crawl_queue_worker, daemon=True).start()
        return jsonify({"ok": True, "queued": len(urls), "queue_length": len(_crawl_queue)})

    _enqueue_job(job)
    _push({"type": "log", "msg": f"➕ Đã xếp thêm {len(urls)} URL vào queue. Queue hiện tại: {len(_crawl_queue)}"})
    return jsonify({"ok": True, "queued": len(urls), "queue_length": len(_crawl_queue)})


def _reset() -> None:
    """Xóa log và reset state (gọi trước khi bắt đầu crawl mới)."""
    global _log_total
    _stop_event.clear()           # ← xóa flag dừng trước khi bắt đầu
    with _lock:
        _logs.clear()
        _log_total = 0
        _crawl_queue.clear()
        _state.update(
            running=True, done=0, total=0,
            success=0, failed=0, queue=0,
            status="running", failed_items=[],
        )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/api/fs")
def api_fs():
    """Liệt kê thư mục con tại `path` (dùng cho folder browser)."""
    req_path = request.args.get("path", "").strip()

    # Windows: liệt kê ổ đĩa khi không có path
    if not req_path:
        if os.name == "nt":
            drives = []
            for d in string.ascii_uppercase:
                p = f"{d}:\\"
                if os.path.exists(p):
                    drives.append({"name": p, "path": p})
            return jsonify({"path": "", "parent": None, "entries": drives})
        req_path = "/"

    try:
        p = Path(req_path).resolve()
        if not p.exists() or not p.is_dir():
            # Fallback: trả về danh sách ổ đĩa / root thay vì lỗi
            if os.name == "nt":
                drives = []
                for d in string.ascii_uppercase:
                    dp = f"{d}:\\"
                    if os.path.exists(dp):
                        drives.append({"name": dp, "path": dp})
                return jsonify({"path": "", "parent": None, "entries": drives})
            def _nk(x): return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', x.name)]
            return jsonify({"path": "/", "parent": None,
                            "entries": [{"name": c.name, "path": str(c)}
                                        for c in sorted(Path("/").iterdir(), key=_nk)
                                        if c.is_dir()]})

        def _natural_key(x):
            return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', x.name)]

        entries = []
        try:
            for child in sorted(p.iterdir(), key=_natural_key):
                if not child.is_dir():
                    continue
                if child.name.startswith("."):
                    continue
                try:
                    # Kiểm tra quyền truy cập
                    next(child.iterdir(), None)
                    entries.append({"name": child.name, "path": str(child)})
                except PermissionError:
                    pass
        except PermissionError:
            pass

        parent = str(p.parent) if str(p.parent) != str(p) else None
        return jsonify({"path": str(p), "parent": parent, "entries": entries})

    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/stop", methods=["POST"])
def api_stop():
    with _lock:
        if not _state["running"]:
            return jsonify({"ok": True})
        _state.update(running=False, status="stopped")

    _stop_event.set()             # ← báo hiệu dừng ngay lập tức
    _push({"type": "log",  "msg": "⚠  Đã dừng — các request hiện tại sẽ hoàn thành rồi dừng hẳn."})
    _push({"type": "stopped"})
    return jsonify({"ok": True})


@app.route("/api/failed")
def api_failed():
    """Trả về danh sách ảnh bị lỗi từ lần crawl gần nhất."""
    with _lock:
        items   = list(_state.get("failed_items", []))
        folder  = _state.get("crawl_folder",  "")
        referer = _state.get("crawl_referer", "")
    return jsonify({"items": items, "folder": folder, "referer": referer})


@app.route("/api/retry_failed", methods=["POST"])
def api_retry_failed():
    """Retry chỉ các ảnh bị lỗi, không crawl lại trang."""
    with _lock:
        if _state["running"]:
            return jsonify({"error": "Đang crawl rồi. Vui lòng dừng trước."}), 409

    data = request.get_json(silent=True) or {}
    try:
        delay   = max(0.0, min(10.0, float(data.get("delay",   0.3))))
        workers = max(1,   min(32,   int(  data.get("workers", 4  ))))
        timeout = max(5,   min(300,  int(  data.get("timeout", 20 ))))
        retries = max(0,   min(10,   int(  data.get("retries", 3  ))))
    except (TypeError, ValueError):
        return jsonify({"error": "Giá trị cấu hình không hợp lệ."}), 400

    with _lock:
        failed_items = list(_state.get("failed_items", []))
        folder       = _state.get("crawl_folder",  "output")
        referer      = _state.get("crawl_referer", "")

    if not failed_items:
        return jsonify({"error": "Không có ảnh lỗi để retry."}), 400

    _reset()

    def run() -> None:
        try:
            def on_log(msg: str) -> None:
                _push({"type": "log", "msg": str(msg)})

            def on_progress(done: int, total: int) -> None:
                with _lock:
                    _state["done"]  = done
                    _state["total"] = total
                _push({"type": "progress", "done": done, "total": total})

            _push({"type": "log", "msg": "─" * 60})
            _push({"type": "log", "msg": f"↺  Retry {len(failed_items)} ảnh lỗi…"})
            _push({"type": "log", "msg": f"Config: delay={delay}s  workers={workers}  timeout={timeout}s  retries={retries}"})

            ok, fail, new_failed = retry_failed_downloads(
                items=failed_items,
                default_referer=referer,
                default_output_folder=folder,
                delay=delay,
                max_workers=workers,
                timeout=timeout,
                max_retries=retries,
                on_progress=on_progress,
                on_log=on_log,
                stop_event=_stop_event,
            )

            _push({"type": "log", "msg": f"\n✔  Retry xong: {ok} thành công, {fail} thất bại."})
            _push({"type": "done", "success": ok, "failed": fail})
            with _lock:
                _state.update(running=False, success=ok, failed=fail, status="done")
                _state["failed_items"]  = new_failed
                _state["crawl_folder"]  = folder
                _state["crawl_referer"] = referer

        except Exception as exc:
            _push({"type": "log", "msg": f"✘  Lỗi: {exc}"})
            _push({"type": "error", "msg": str(exc)})
            with _lock:
                _state.update(running=False, status="error")

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/logs")
def api_logs():
    """Trả về các log mới từ vị trí `from` trở về sau, cùng state hiện tại."""
    try:
        since = max(0, int(request.args.get("from", 0)))
    except (TypeError, ValueError):
        since = 0

    with _lock:
        total  = _log_total
        offset = total - len(_logs)   # số entry đã bị xoay vòng bỏ đi
        start  = max(0, since - offset)
        batch  = list(_logs)[start:]
        state  = dict(_state)

    return jsonify({"logs": batch, "total": total, "state": state})


# ── Translation state ────────────────────────────────────────────────────────
_t_lock      = threading.Lock()
_t_logs: deque = deque()          # Circular buffer
_t_log_total: int = 0
_t_state: dict = {
    "running": False, "done": 0, "total": 0,
    "success": 0, "failed": 0, "status": "ready",
    "failed_images": [],   # list of src paths that failed
    "input_dir": "",
    "output_dir": "",
}
_t_stop = threading.Event()


def _find_font() -> str | None:
    # Ưu tiên MTO Astro City cho dịch tiếng Việt, sau đến các font khác
    candidates = [
        Path(__file__).parent / "fonts" / "MTO Astro City.ttf",
        Path(__file__).parent / "fonts" / "BeVietnamPro-Regular.ttf",
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
        Path(__file__).parent / "fonts" / "NotoSans-Regular.ttf",
        Path(r"C:\Windows\Fonts\tahoma.ttf"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def _t_push(entry: dict) -> None:
    global _t_log_total
    with _t_lock:
        if len(_t_logs) >= MAX_LOG:
            _t_logs.popleft()
        _t_logs.append(entry)
        _t_log_total += 1


def _t_reset() -> None:
    global _t_log_total
    with _t_lock:
        _t_logs.clear()
        _t_log_total = 0
        _t_state.update(running=True, done=0, total=0, success=0, failed=0, status="running",
                        failed_images=[])
    _t_stop.clear()


# ── Translate routes ──────────────────────────────────────────────────────────

@app.route("/api/translate/check")
def api_translate_check():
    """Kiểm tra dependencies: torch, paddleocr/easyocr, ollama."""
    result = {}
    # PyTorch / CUDA
    try:
        import torch
        result["torch"] = torch.__version__
        result["cuda"]  = torch.cuda.is_available()
        result["gpu"]   = torch.cuda.get_device_name(0) if result["cuda"] else None
    except ImportError:
        result["torch"] = None
        result["cuda"]  = False
        result["gpu"]   = None
    # PaddleOCR (preferred)
    try:
        import paddleocr  # noqa: F401
        result["paddleocr"] = True
    except ImportError:
        result["paddleocr"] = False
    # EasyOCR (fallback)
    try:
        import easyocr  # noqa: F401
        result["easyocr"] = True
    except ImportError:
        result["easyocr"] = False
    # OpenCV
    try:
        import cv2  # noqa: F401
        result["opencv"] = cv2.__version__
    except ImportError:
        result["opencv"] = None
    # Ollama
    ollama = te.check_ollama()
    result["ollama"]        = ollama.get("ok", False)
    result["ollama_models"] = ollama.get("models", [])
    result["ollama_error"]  = ollama.get("error")
    # Font
    result["font"] = _find_font()
    # manga-image-translator
    mit = te.check_mit()
    result["mit"]         = mit.get("ok", False)
    result["mit_version"] = mit.get("version")
    result["mit_python"]  = mit.get("python")
    result["mit_error"]   = mit.get("error")
    # LAMA inpainter
    result["lama"] = te.check_lama_available()
    return jsonify(result)


@app.route("/api/translate/start", methods=["POST"])
def api_translate_start():
    with _t_lock:
        if _t_state["running"]:
            return jsonify({"error": "Đang dịch rồi. Vui lòng dừng trước."}), 409

    data           = request.get_json(silent=True) or {}
    input_dir      = str(data.get("input_dir",      "")).strip()
    output_dir     = str(data.get("output_dir",     "")).strip()
    model          = str(data.get("model",      "qwen2.5:7b")).strip() or "qwen2.5:7b"
    use_gpu        = bool(data.get("use_gpu", True))
    src_lang       = str(data.get("src_lang", "zh")).strip().lower()
    backend        = str(data.get("backend",       "default")).strip().lower()
    mit_translator = str(data.get("mit_translator", "m2m100")).strip() or "m2m100"
    mit_target_lang   = str(data.get("mit_target_lang",  "VIN")).strip() or "VIN"
    mit_detector      = str(data.get("mit_detector",     "")).strip()
    mit_inpainter     = str(data.get("mit_inpainter",    "lama_large")).strip()
    mit_upscale       = str(data.get("mit_upscale",      "")).strip()
    mit_det_size      = str(data.get("mit_det_size",     "")).strip()
    mit_mask_dil      = str(data.get("mit_mask_dil",     "")).strip()
    mit_unclip         = str(data.get("mit_unclip",       "")).strip()
    mit_font_ofs      = str(data.get("mit_font_ofs",     "")).strip()
    mit_font_min      = str(data.get("mit_font_min",     "")).strip()
    mit_font_fixed    = str(data.get("mit_font_fixed",   "")).strip()
    mit_font_color    = str(data.get("mit_font_color",   "")).strip()
    mit_custom_api_base = str(data.get("mit_custom_api_base", "")).strip()
    mit_custom_api_key  = str(data.get("mit_custom_api_key",  "")).strip()
    mit_verbose       = bool(data.get("mit_verbose",     False))
    mit_skip_no_text  = bool(data.get("mit_skip_no_text",False))
    mit_overwrite     = bool(data.get("mit_overwrite",   False))
    overwrite         = bool(data.get("overwrite",       False))
    cpu_priority      = str(data.get("cpu_priority",     "below_normal")).strip()
    if cpu_priority not in ("normal", "below_normal", "idle"):
        cpu_priority = "below_normal"
    inpainter         = str(data.get("inpainter",        "opencv")).strip()
    try:
        font_scale = float(data.get("font_scale", 0.60))
        font_scale = max(0.3, min(2.0, font_scale))
    except (TypeError, ValueError):
        font_scale = 0.75
    if inpainter not in ("opencv", "lama"):
        inpainter = "opencv"
    if src_lang not in ("zh", "en"):
        src_lang = "zh"
    if backend not in ("default", "mit"):
        backend = "default"

    if not input_dir:
        return jsonify({"error": "Vui lòng chọn thư mục ảnh nguồn."}), 400
    if not Path(input_dir).is_dir():
        return jsonify({"error": f"Thư mục không tồn tại: {input_dir}"}), 400
    if not output_dir:
        output_dir = str(Path(input_dir).parent / (Path(input_dir).name + "_vi"))

    _t_reset()

    def run() -> None:
        try:
            def on_log(msg: str):
                _t_push({"type": "log", "msg": str(msg)})

            def on_progress(done: int, total: int):
                with _t_lock:
                    _t_state["done"]  = done
                    _t_state["total"] = total
                _t_push({"type": "progress", "done": done, "total": total})

            _t_push({"type": "log", "msg": "─" * 60})
            _t_push({"type": "log", "msg": f"Nguồn : {input_dir}"})
            _t_push({"type": "log", "msg": f"Xuất  : {output_dir}"})
            if backend == "mit":
                _t_push({"type": "log", "msg": f"Backend: manga-image-translator  translator={mit_translator}  →{mit_target_lang}  GPU={use_gpu}"})
                mit_check = te.check_mit()
                translator = te.MITImageTranslator(
                    translator=mit_translator,
                    target_lang=mit_target_lang,
                    use_gpu=use_gpu,
                    python_path=mit_check.get("python"),
                    detector=mit_detector,
                    inpainter=mit_inpainter,
                    ollama_model=model,
                    custom_openai_api_base=mit_custom_api_base,
                    custom_openai_api_key=mit_custom_api_key,
                    upscale_ratio=mit_upscale,
                    detection_size=mit_det_size,
                    mask_dilation_offset=mit_mask_dil,
                    unclip_ratio=mit_unclip,
                    font_size_offset=mit_font_ofs,
                    font_size_minimum=mit_font_min,
                    font_size_fixed=mit_font_fixed,
                    font_color=mit_font_color,
                    verbose=mit_verbose,
                    skip_no_text=mit_skip_no_text,
                    overwrite=mit_overwrite,
                    cpu_priority=cpu_priority,
                    on_log=on_log,
                    on_progress=on_progress,
                )
            else:
                _t_push({"type": "log", "msg": f"Model : {model}  GPU={use_gpu}  Inpainter={inpainter}"})
                _t_push({"type": "log", "msg": f"Lang  : {'ZH→VI' if src_lang == 'zh' else 'EN→VI'}"})
                translator = te.ImageTranslator(
                    model=model,
                    font_path=_find_font(),
                    use_gpu=use_gpu,
                    src_lang=src_lang,
                    inpainter=inpainter,
                    overwrite=overwrite,
                    font_scale=font_scale,
                    cpu_priority=cpu_priority,
                    on_log=on_log,
                    on_progress=on_progress,
                )
            ok, fail, failed_images = translator.process_folder(
                input_dir=input_dir,
                output_dir=output_dir,
                stop_event=_t_stop,
            )
            _t_push({"type": "log",  "msg": f"\n✔  Xong: {ok} OK, {fail} lỗi."})
            _t_push({"type": "done", "success": ok, "failed": fail, "output_dir": output_dir})
            with _t_lock:
                _t_state.update(running=False, success=ok, failed=fail, status="done",
                                 failed_images=failed_images,
                                 input_dir=input_dir, output_dir=output_dir)
        except Exception as exc:
            _t_push({"type": "log",   "msg": f"✘  Lỗi: {exc}"})
            _t_push({"type": "error", "msg": str(exc)})
            with _t_lock:
                _t_state.update(running=False, status="error")

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"ok": True, "output_dir": output_dir})


@app.route("/api/translate/stop", methods=["POST"])
def api_translate_stop():
    with _t_lock:
        if not _t_state["running"]:
            return jsonify({"ok": True})
        _t_state.update(running=False, status="stopped")
    _t_stop.set()
    _t_push({"type": "log", "msg": "⚠  Yêu cầu dừng đã gửi."})
    return jsonify({"ok": True})


@app.route("/api/translate/failed")
def api_translate_failed():
    """Trả về danh sách ảnh bị lỗi từ lần dịch gần nhất."""
    with _t_lock:
        return jsonify({
            "images":     list(_t_state.get("failed_images", [])),
            "input_dir":  _t_state.get("input_dir", ""),
            "output_dir": _t_state.get("output_dir", ""),
        })


@app.route("/api/translate/retry_failed", methods=["POST"])
def api_translate_retry_failed():
    """Retry chỉ các ảnh dịch bị lỗi."""
    with _t_lock:
        if _t_state["running"]:
            return jsonify({"error": "Đang dịch rồi. Vui lòng dừng trước."}), 409

    with _t_lock:
        failed_images = list(_t_state.get("failed_images", []))
        input_dir     = _t_state.get("input_dir", "")
        output_dir    = _t_state.get("output_dir", "")

    if not failed_images:
        return jsonify({"error": "Không có ảnh lỗi để retry."}), 400
    if not input_dir or not Path(input_dir).is_dir():
        return jsonify({"error": "Thư mục nguồn không hợp lệ."}), 400

    data        = request.get_json(silent=True) or {}
    model       = str(data.get("model",      "qwen2.5:7b")).strip() or "qwen2.5:7b"
    use_gpu     = bool(data.get("use_gpu", True))
    src_lang    = str(data.get("src_lang", "zh")).strip().lower()
    inpainter   = str(data.get("inpainter", "opencv")).strip()
    if inpainter not in ("opencv", "lama"):
        inpainter = "opencv"
    if src_lang not in ("zh", "en"):
        src_lang = "zh"

    _t_reset()

    def run() -> None:
        try:
            def on_log(msg: str):
                _t_push({"type": "log", "msg": str(msg)})

            def on_progress(done: int, total: int):
                with _t_lock:
                    _t_state["done"]  = done
                    _t_state["total"] = total
                _t_push({"type": "progress", "done": done, "total": total})

            _t_push({"type": "log", "msg": "─" * 60})
            _t_push({"type": "log", "msg": f"↺  Retry {len(failed_images)} ảnh dịch lỗi…"})

            translator = te.ImageTranslator(
                model=model,
                font_path=_find_font(),
                use_gpu=use_gpu,
                src_lang=src_lang,
                inpainter=inpainter,
                font_scale=font_scale,
                on_log=on_log,
                on_progress=on_progress,
            )
            ok, fail, new_failed = translator.process_folder(
                input_dir=input_dir,
                output_dir=output_dir,
                stop_event=_t_stop,
                images_override=failed_images,
            )
            _t_push({"type": "log",  "msg": f"\n✔  Retry xong: {ok} OK, {fail} lỗi."})
            _t_push({"type": "done", "success": ok, "failed": fail, "output_dir": output_dir})
            with _t_lock:
                _t_state.update(running=False, success=ok, failed=fail, status="done",
                                 failed_images=new_failed,
                                 input_dir=input_dir, output_dir=output_dir)
        except Exception as exc:
            _t_push({"type": "log",   "msg": f"✘  Lỗi: {exc}"})
            _t_push({"type": "error", "msg": str(exc)})
            with _t_lock:
                _t_state.update(running=False, status="error")

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"ok": True, "output_dir": output_dir})


@app.route("/api/viewer/delete", methods=["POST"])
def api_viewer_delete():
    """Xóa file ảnh khỏi đĩa (dùng trong viewer)."""
    data = request.get_json(silent=True) or {}
    path_str = str(data.get("path", "")).strip()
    if not path_str:
        return jsonify({"error": "Thiếu path"}), 400
    p = Path(path_str).resolve()
    if not p.is_file() or p.suffix.lower() not in te.IMAGE_EXTS:
        return jsonify({"error": "File không hợp lệ"}), 400
    try:
        p.unlink()
        return jsonify({"ok": True})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/translate/logs")
def api_translate_logs():
    try:
        since = max(0, int(request.args.get("from", 0)))
    except (TypeError, ValueError):
        since = 0
    with _t_lock:
        total  = _t_log_total
        offset = total - len(_t_logs)
        start  = max(0, since - offset)
        batch  = list(_t_logs)[start:]
        state  = dict(_t_state)
    return jsonify({"logs": batch, "total": total, "state": state})


@app.route("/api/translate/image")
def api_translate_image():
    """Phục vụ file ảnh từ đường dẫn local (dùng cho preview)."""
    path_str = request.args.get("path", "").strip()
    if not path_str:
        return jsonify({"error": "Thiếu path"}), 400
    p = Path(path_str).resolve()
    if not p.is_file() or p.suffix.lower() not in te.IMAGE_EXTS:
        return jsonify({"error": "File không hợp lệ"}), 400
    mime, _ = mimetypes.guess_type(str(p))
    resp = send_file(str(p), mimetype=mime or "image/jpeg")
    # Cho phép browser cache thumbnail preview (mtime-based rev từ client)
    resp.headers["Cache-Control"] = "public, max-age=3600, immutable"
    return resp


@app.route("/api/translate/preview")
def api_translate_preview():
    """Liệt kê các ảnh đã xử lý trong output_dir."""
    folder = request.args.get("folder", "").strip()
    if not folder:
        return jsonify({"images": [], "total": 0})
    p = Path(folder).resolve()
    if not p.is_dir():
        return jsonify({"images": [], "total": 0})
    try:
        limit = int(request.args.get("limit", 0))
    except (TypeError, ValueError):
        limit = 0

    all_files = sorted(
        (f for f in p.iterdir() if f.suffix.lower() in te.IMAGE_EXTS),
        key=lambda f: f.stat().st_mtime,
    )
    total = len(all_files)
    # limit=0 → trả về tất cả
    display = all_files if limit <= 0 else all_files[-limit:]
    images = [
        {"name": f.name, "path": str(f), "mtime": int(f.stat().st_mtime)}
        for f in display
    ]
    return jsonify({"images": images, "total": total})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import webbrowser
    port = int(os.environ.get("PORT", 5000))
    threading.Timer(1.2, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    print(f"\n  ✔  Web UI: http://127.0.0.1:{port}\n")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
