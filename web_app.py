"""
Image Crawler Web UI – Giao diện web để crawl và tải ảnh từ web.
Chạy: python web_app.py  →  tự mở http://localhost:5000
"""

import json
import mimetypes
import os
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

from crawler import crawl
import translator_engine as te

app = Flask(__name__)

# ── Trạng thái toàn cục (một phiên crawl tại một thời điểm) ──────────────────
_lock = threading.Lock()
_logs: deque = deque()          # Circular buffer sự kiện
_log_total: int = 0             # Absolute counter (không bao giờ reset bởi eviction)
_state: dict = {
    "running": False,
    "done": 0,
    "total": 0,
    "success": 0,
    "failed": 0,
    "status": "ready",          # ready | running | done | error | stopped
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


def _reset() -> None:
    """Xóa log và reset state (gọi trước khi bắt đầu crawl mới)."""
    global _log_total
    _stop_event.clear()           # ← xóa flag dừng trước khi bắt đầu
    with _lock:
        _logs.clear()
        _log_total = 0
        _state.update(
            running=True, done=0, total=0,
            success=0, failed=0, status="running",
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
            return jsonify({"path": "/", "parent": None,
                            "entries": [{"name": c.name, "path": str(c)}
                                        for c in sorted(Path("/").iterdir())
                                        if c.is_dir()]})

        entries = []
        try:
            for child in sorted(p.iterdir(), key=lambda x: x.name.lower()):
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


@app.route("/api/start", methods=["POST"])
def api_start():
    with _lock:
        if _state["running"]:
            return jsonify({"error": "Đang crawl rồi. Vui lòng dừng trước."}), 409

    data = request.get_json(silent=True) or {}
    url    = str(data.get("url",    "")).strip()
    folder = str(data.get("folder", "output")).strip() or "output"
    try:
        delay   = max(0.0, min(10.0, float(data.get("delay",   0.3))))
        workers = max(1,   min(32,   int(  data.get("workers", 4  ))))
        timeout = max(5,   min(300,  int(  data.get("timeout", 20 ))))
    except (TypeError, ValueError):
        return jsonify({"error": "Giá trị cấu hình không hợp lệ."}), 400

    if not url.startswith(("http://", "https://")):
        return jsonify({"error": "URL phải bắt đầu bằng http:// hoặc https://"}), 400

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
            _push({"type": "log", "msg": f"URL   : {url}"})
            _push({"type": "log", "msg": f"Folder: {folder}"})
            _push({"type": "log", "msg": f"Config: delay={delay}s  workers={workers}  timeout={timeout}s"})

            ok, fail = crawl(
                url=url,
                output_folder=folder,
                delay=delay,
                max_workers=workers,
                timeout=timeout,
                on_progress=on_progress,
                on_log=on_log,
                stop_event=_stop_event,   # ← truyền event vào
            )

            _push({"type": "log", "msg": f"\n✔  Hoàn thành: {ok} thành công, {fail} thất bại."})
            _push({"type": "done", "success": ok, "failed": fail})
            with _lock:
                _state.update(running=False, success=ok, failed=fail, status="done")

        except Exception as exc:
            _push({"type": "log", "msg": f"✘  Lỗi: {exc}"})
            _push({"type": "error", "msg": str(exc)})
            with _lock:
                _state.update(running=False, status="error")

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"ok": True})


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
}
_t_stop = threading.Event()


def _find_font() -> str | None:
    # System fonts first — Arial & Segoe UI have full Vietnamese (Latin Extended Additional)
    # NotoSans from the latin-greek-cyrillic repo may lack Vietnamese glyphs
    candidates = [
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
        _t_state.update(running=True, done=0, total=0, success=0, failed=0, status="running")
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
    mit_font_ofs      = str(data.get("mit_font_ofs",     "")).strip()
    mit_verbose       = bool(data.get("mit_verbose",     False))
    mit_skip_no_text  = bool(data.get("mit_skip_no_text",False))
    mit_overwrite     = bool(data.get("mit_overwrite",   False))
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
                    upscale_ratio=mit_upscale,
                    detection_size=mit_det_size,
                    mask_dilation_offset=mit_mask_dil,
                    font_size_offset=mit_font_ofs,
                    verbose=mit_verbose,
                    skip_no_text=mit_skip_no_text,
                    overwrite=mit_overwrite,
                    on_log=on_log,
                    on_progress=on_progress,
                )
            else:
                _t_push({"type": "log", "msg": f"Model : {model}  GPU={use_gpu}"})
                _t_push({"type": "log", "msg": f"Lang  : {'ZH→VI' if src_lang == 'zh' else 'EN→VI'}"})
                translator = te.ImageTranslator(
                    model=model,
                    font_path=_find_font(),
                    use_gpu=use_gpu,
                    src_lang=src_lang,
                    on_log=on_log,
                    on_progress=on_progress,
                )
            ok, fail = translator.process_folder(
                input_dir=input_dir,
                output_dir=output_dir,
                stop_event=_t_stop,
            )
            _t_push({"type": "log",  "msg": f"\n✔  Xong: {ok} OK, {fail} lỗi."})
            _t_push({"type": "done", "success": ok, "failed": fail, "output_dir": output_dir})
            with _t_lock:
                _t_state.update(running=False, success=ok, failed=fail, status="done")
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
    return send_file(str(p), mimetype=mime or "image/jpeg")


@app.route("/api/translate/preview")
def api_translate_preview():
    """Liệt kê các ảnh đã xử lý trong output_dir."""
    folder = request.args.get("folder", "").strip()
    if not folder:
        return jsonify({"images": []})
    p = Path(folder).resolve()
    if not p.is_dir():
        return jsonify({"images": []})
    images = [
        {"name": f.name, "path": str(f)}
        for f in sorted(p.iterdir())
        if f.suffix.lower() in te.IMAGE_EXTS
    ]
    return jsonify({"images": images})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import webbrowser
    port = int(os.environ.get("PORT", 5000))
    threading.Timer(1.2, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    print(f"\n  ✔  Web UI: http://127.0.0.1:{port}\n")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
