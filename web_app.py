"""
Image Crawler Web UI – Giao diện web để crawl và tải ảnh từ web.
Chạy: python web_app.py  →  tự mở http://localhost:5000
"""

import hashlib
import json
import mimetypes
import os
import re
import string
import sys
import tempfile
import threading
import time
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

# ── Nhật ký file ─────────────────────────────────────────────────────────────
# Mỗi journal (crawl / dịch) được mirror TOÀN BỘ ra một file riêng, ghi đè
# (truncate) khi bắt đầu action mới. Nhật ký trên UI chỉ giữ MAX_LOG dòng gần
# nhất, còn file luôn đầy đủ — để đọc lại / đưa AI phân tích & debug chi tiết.
_CRAWL_LOG_PATH       = Path(__file__).parent / "crawl_log.txt"
_TRANSLATION_LOG_PATH = Path(__file__).parent / "translation_log.txt"


class _RunLogFile:
    """Mirror nhật ký của MỘT action ra file, đè (truncate) nội dung lần trước.

    `reopen()` được gọi khi bắt đầu action mới nên nhật ký cũ bị xoá. Mọi dòng
    log được nối thêm và flush ngay để file luôn đầy đủ kể cả khi tiến trình bị
    dừng giữa chừng. Mọi lỗi I/O đều nuốt im lặng — ghi log không bao giờ làm
    hỏng việc crawl/dịch.
    """

    def __init__(self, path: Path):
        self._path = path
        self._lock = threading.Lock()
        self._f = None

    def reopen(self, header_lines: list[str] | None = None) -> None:
        with self._lock:
            if self._f is not None:
                try:
                    self._f.close()
                except Exception:
                    pass
                self._f = None
            try:
                self._f = open(self._path, "w", encoding="utf-8")
                if header_lines:
                    self._f.write("\n".join(header_lines) + "\n")
                self._f.flush()
            except Exception:
                self._f = None

    def write(self, msg: str) -> None:
        with self._lock:
            if self._f is None:
                return
            try:
                self._f.write(str(msg) + "\n")
                self._f.flush()
            except Exception:
                pass


_crawl_logfile = _RunLogFile(_CRAWL_LOG_PATH)
_t_logfile     = _RunLogFile(_TRANSLATION_LOG_PATH)


def _push(entry: dict) -> None:
    """Thêm entry vào circular buffer (thread-safe, xoay vòng khi full)."""
    global _log_total
    with _lock:
        if len(_logs) >= MAX_LOG:
            _logs.popleft()  # bỏ entry cũ nhất
        _logs.append(entry)
        _log_total += 1
        if entry.get("type") == "log":
            _crawl_logfile.write(entry.get("msg", ""))


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
            skip_existing = job.get("skip_existing", True)

            _push({"type": "log", "msg": "─" * 60})
            _push({"type": "log", "msg": f"▶ Bắt đầu job queue: {len(urls)} URL"})
            for url in urls:
                if _stop_event.is_set():
                    break
                _push({"type": "log", "msg": f"  ▶ URL: {url}"})
                # Lưu folder/referer hiện tại để /api/failed hiển thị + retry fallback.
                with _lock:
                    _state["crawl_folder"]  = folder
                    _state["crawl_referer"] = url

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
                    skip_existing=skip_existing,
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
        "skip_existing": bool(data.get("skip_existing", True)),
    }

    # Quyết định khởi động worker + claim `running` trong CÙNG một lock để tránh
    # race: hai request /api/start đồng thời không thể cùng spawn 2 worker.
    started = False
    with _lock:
        if not _state["running"]:
            _stop_event.clear()
            _reset_locked()          # set running=True + clear queue/logs
            _crawl_queue.append(job)
            _state["queue"] = len(_crawl_queue)
            queue_length = len(_crawl_queue)
            started = True
        else:
            _crawl_queue.append(job)
            _state["queue"] = len(_crawl_queue)
            queue_length = len(_crawl_queue)

    if started:
        _push({"type": "log", "msg": f"▶ Bắt đầu queue mới: {len(urls)} URL"})
        threading.Thread(target=_crawl_queue_worker, daemon=True).start()
    else:
        _push({"type": "log", "msg": f"➕ Đã xếp thêm {len(urls)} URL vào queue. Queue hiện tại: {queue_length}"})
    return jsonify({"ok": True, "queued": len(urls), "queue_length": queue_length})


def _reset_locked() -> None:
    """Reset log + state. Caller PHẢI đang giữ `_lock`."""
    global _log_total
    _logs.clear()
    _log_total = 0
    _crawl_queue.clear()
    _state.update(
        running=True, done=0, total=0,
        success=0, failed=0, queue=0,
        status="running", failed_items=[],
    )
    _crawl_logfile.reopen([
        "# Nhật ký crawl — đè mỗi action mới (start queue / retry ảnh lỗi).",
        "# Đọc file này để AI debug chi tiết — UI chỉ giữ số dòng có hạn.",
        f"# Thời điểm : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "#" + "=" * 70,
        "",
    ])


def _reset() -> None:
    """Xóa log và reset state (gọi trước khi bắt đầu crawl mới)."""
    _stop_event.clear()           # ← xóa flag dừng trước khi bắt đầu
    with _lock:
        _reset_locked()


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
    "queue": 0,            # số job dịch đang chờ trong hàng đợi
}
_t_stop = threading.Event()
_t_queue: deque = deque()   # hàng đợi job dịch (mỗi job là một callable không tham số)

def _find_font() -> str | None:
    # Ưu tiên MTO Astro City cho dịch tiếng Việt, sau đến các font khác
    candidates = [
        Path(__file__).parent / "fonts" / "MTO Astro City.ttf",
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
        if entry.get("type") == "log":
            _t_logfile.write(entry.get("msg", ""))


def _t_reset_locked() -> None:
    """Reset log + state dịch. Caller PHẢI đang giữ `_t_lock`."""
    global _t_log_total
    _t_logs.clear()
    _t_log_total = 0
    _t_queue.clear()
    _t_state.update(running=True, done=0, total=0, success=0, failed=0, status="running",
                    failed_images=[], queue=0)


def _t_enqueue_or_start(job_fn) -> tuple[bool, int]:
    """Xếp job dịch vào hàng đợi (giống cơ chế queue của crawl).

    Chưa có job nào chạy → reset state + spawn worker (job chạy ngay).
    Đang dịch (vd: tab khác đã bấm Dịch) → job xếp hàng, tự chạy khi job trước xong.
    Quyết định + claim `running` trong CÙNG một lock để 2 request đồng thời
    không thể cùng spawn 2 worker.
    Trả về (started, queue_length).
    """
    started = False
    with _t_lock:
        if not _t_state["running"]:
            _t_stop.clear()
            _t_reset_locked()        # set running=True + clear log/queue
            started = True
        _t_queue.append(job_fn)
        _t_state["queue"] = len(_t_queue)
        queue_length = len(_t_queue)
    if started:
        threading.Thread(target=_t_queue_worker, daemon=True).start()
    return started, queue_length


def _t_queue_worker() -> None:
    """Chạy tuần tự từng job dịch trong hàng đợi.

    Mỗi thời điểm chỉ 1 job (OCR/inpaint/LLM chiếm GPU/VRAM — chạy song song
    dễ tràn bộ nhớ). `running` giữ nguyên True cho tới khi hàng đợi cạn để
    các tab đang poll không tưởng nhầm là đã xong khi còn job chờ.
    """
    while True:
        with _t_lock:
            if _t_stop.is_set() or not _t_queue:
                break
            job = _t_queue.popleft()
            _t_state["queue"] = len(_t_queue)
            _t_state.update(done=0, total=0, status="running")
        try:
            job()
        except Exception as exc:
            _t_push({"type": "log", "msg": f"✘  Lỗi job dịch: {exc}"})
            with _t_lock:
                _t_state["status"] = "error"
    with _t_lock:
        status = _t_state.get("status", "done")
        if _t_stop.is_set():
            status = "stopped"
        elif status == "running":
            status = "done"
        _t_state.update(running=False, status=status)


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
    data           = request.get_json(silent=True) or {}
    input_dir      = str(data.get("input_dir",      "")).strip()
    output_dir     = str(data.get("output_dir",     "")).strip()
    sel_images     = data.get("images") or []   # tên ảnh đã tích (rỗng = cả thư mục)
    model          = str(data.get("model",      "qwen2.5:7b")).strip() or "qwen2.5:7b"
    use_gpu        = bool(data.get("use_gpu", True))
    src_lang       = str(data.get("src_lang", "zh")).strip().lower()
    backend        = str(data.get("backend",       "default")).strip().lower()
    mit_translator = str(data.get("mit_translator", "m2m100")).strip() or "m2m100"
    mit_target_lang   = str(data.get("mit_target_lang",  "VIN")).strip() or "VIN"
    mit_detector      = str(data.get("mit_detector",     "")).strip()
    mit_inpainter     = str(data.get("mit_inpainter",    "lama_mpe")).strip()
    mit_inpaint_size  = str(data.get("mit_inpaint_size", "")).strip()
    mit_inpaint_prec  = str(data.get("mit_inpaint_prec", "")).strip().lower()
    if mit_inpaint_prec not in ("fp32", "fp16", "bf16"):
        mit_inpaint_prec = ""
    mit_upscale       = str(data.get("mit_upscale",      "")).strip()
    mit_upscaler      = str(data.get("mit_upscaler",     "")).strip().lower()
    if mit_upscaler not in ("waifu2x", "esrgan", "4xultrasharp"):
        mit_upscaler = ""
    mit_det_size      = str(data.get("mit_det_size",     "")).strip()
    mit_mask_dil      = str(data.get("mit_mask_dil",     "")).strip()
    mit_unclip         = str(data.get("mit_unclip",       "")).strip()
    mit_box_thr       = str(data.get("mit_box_thr",      "")).strip()
    mit_text_thr      = str(data.get("mit_text_thr",     "")).strip()
    mit_det_invert    = bool(data.get("mit_det_invert",  False))
    mit_det_gamma     = bool(data.get("mit_det_gamma",   False))
    mit_det_rotate    = bool(data.get("mit_det_rotate",  False))
    mit_det_auto_rotate = bool(data.get("mit_det_auto_rotate", False))
    mit_ocr           = str(data.get("mit_ocr",          "")).strip()
    if mit_ocr not in ("32px", "48px", "48px_ctc", "mocr"):
        mit_ocr = ""
    mit_ocr_prob      = str(data.get("mit_ocr_prob",     "")).strip()
    mit_font_ofs      = str(data.get("mit_font_ofs",     "")).strip()
    mit_font_min      = str(data.get("mit_font_min",     "")).strip()
    mit_font_fixed    = str(data.get("mit_font_fixed",   "")).strip()
    mit_font_color    = str(data.get("mit_font_color",   "")).strip()
    mit_narrow_wide   = str(data.get("mit_narrow_wide",  "")).strip()
    mit_narrow_fcap   = str(data.get("mit_narrow_fcap",  "")).strip()
    mit_custom_api_base = str(data.get("mit_custom_api_base", "")).strip()
    mit_custom_api_key  = str(data.get("mit_custom_api_key",  "")).strip()
    mit_verbose       = bool(data.get("mit_verbose",     False))
    mit_skip_no_text  = bool(data.get("mit_skip_no_text",False))
    mit_overwrite     = bool(data.get("mit_overwrite",   False))
    overwrite         = bool(data.get("overwrite",       False))
    cpu_priority      = str(data.get("cpu_priority",     "below_normal")).strip()
    translation_style = str(data.get("translation_style", "modern")).strip().lower()
    llm_base_url      = str(data.get("llm_base_url", "")).strip()
    llm_api_type      = str(data.get("llm_api_type", "ollama")).strip().lower()
    if cpu_priority not in ("normal", "below_normal", "idle"):
        cpu_priority = "below_normal"
    if translation_style not in ("modern", "wuxia", "school", "lightnovel"):
        translation_style = "modern"
    if llm_api_type not in ("ollama", "openai_compat"):
        llm_api_type = "ollama"
    inpainter         = str(data.get("inpainter",        "opencv")).strip()
    try:
        font_scale = float(data.get("font_scale", 0.60))
        font_scale = max(0.3, min(2.0, font_scale))
    except (TypeError, ValueError):
        font_scale = 0.60
    try:
        image_quality = int(data.get("image_quality", 95))
        image_quality = max(40, min(100, image_quality))
    except (TypeError, ValueError):
        image_quality = 95
    if inpainter not in ("opencv", "lama"):
        inpainter = "opencv"
    if src_lang not in ("zh", "en", "ja"):
        src_lang = "zh"
    if backend not in ("default", "mit"):
        backend = "default"

    if not input_dir:
        return jsonify({"error": "Vui lòng chọn thư mục ảnh nguồn."}), 400
    if not Path(input_dir).is_dir():
        return jsonify({"error": f"Thư mục không tồn tại: {input_dir}"}), 400
    if not output_dir:
        output_dir = str(Path(input_dir).parent / (Path(input_dir).name + "_vi"))

    # Ảnh đã chọn → chỉ dịch các ảnh đó. Chỉ lấy basename (chống path traversal),
    # ghép với thư mục nguồn. Rỗng/không hợp lệ → None = dịch cả thư mục.
    images_override = None
    if isinstance(sel_images, list) and sel_images:
        _base = Path(input_dir)
        _picked = []
        for _n in sel_images:
            _fp = _base / Path(str(_n)).name
            if _fp.is_file() and _fp.suffix.lower() in te.IMAGE_EXTS:
                _picked.append(str(_fp))
        images_override = _picked or None

    def run() -> None:
        # Nhật ký file của lần dịch này (đè lần trước). Header ghi đủ cấu hình để
        # sau này đọc lại / nhờ AI cải thiện mà không cần copy tay từ console.
        # Mọi dòng đẩy vào Nhật ký UI đều được _t_push mirror tự động vào file.
        _t_logfile.reopen([
            "# Nhật ký dịch ảnh (translation log) — đè mỗi lần dịch mới.",
            "# Đọc file này để AI phân tích & cải thiện prompt/cấu hình.",
            f"# Thời điểm : {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"# Nguồn     : {input_dir}",
            f"# Xuất      : {output_dir}",
            f"# Backend   : {backend}",
            f"# Ảnh chọn  : {len(images_override) if images_override else 'toàn bộ thư mục'}",
            (f"# MIT       : translator={mit_translator} target={mit_target_lang} "
             f"style={translation_style} model={model} detector={mit_detector or 'auto'} "
             f"inpainter={mit_inpainter} ocr={mit_ocr or 'auto'} ocr_prob={mit_ocr_prob or '-'} "
             f"upscale={mit_upscale or '-'} det_size={mit_det_size or '-'} "
             f"mask_dil={mit_mask_dil or '-'} unclip={mit_unclip or '-'} "
             f"box_thr={mit_box_thr or '-'} text_thr={mit_text_thr or '-'} "
             f"font_ofs={mit_font_ofs or '-'} font_min={mit_font_min or '-'} "
             f"font_fixed={mit_font_fixed or '-'} font_color={mit_font_color or '-'} "
             f"narrow_wide={mit_narrow_wide or '-'} narrow_fcap={mit_narrow_fcap or '-'} "
             f"verbose={mit_verbose} gpu={use_gpu}")
            if backend == "mit"
            else (f"# Default   : lang={src_lang} model={model} inpainter={inpainter} "
                  f"style={translation_style} font_scale={font_scale} gpu={use_gpu}"),
            f"# image_quality={image_quality} cpu_priority={cpu_priority}",
            "#" + "=" * 70,
            "",
        ])
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
                if not mit_check.get("ok"):
                    _t_push({"type": "log", "msg": f"  [FAIL] {mit_check.get('error')}"})
                    _t_push({"type": "log", "msg": "  [FAIL] Không thể chạy manga-image-translator do thiếu dependency."})
                    return
                translator = te.MITImageTranslator(
                    translator=mit_translator,
                    target_lang=mit_target_lang,
                    use_gpu=use_gpu,
                    python_path=mit_check.get("python"),
                    detector=mit_detector,
                    inpainter=mit_inpainter,
                    inpainting_size=mit_inpaint_size,
                    inpainting_precision=mit_inpaint_prec,
                    ollama_model=model,
                    custom_openai_api_base=mit_custom_api_base,
                    custom_openai_api_key=mit_custom_api_key,
                    upscale_ratio=mit_upscale,
                    upscaler=mit_upscaler,
                    detection_size=mit_det_size,
                    mask_dilation_offset=mit_mask_dil,
                    unclip_ratio=mit_unclip,
                    box_threshold=mit_box_thr,
                    text_threshold=mit_text_thr,
                    det_invert=mit_det_invert,
                    det_gamma_correct=mit_det_gamma,
                    det_rotate=mit_det_rotate,
                    det_auto_rotate=mit_det_auto_rotate,
                    ocr_model=mit_ocr,
                    ocr_prob=mit_ocr_prob,
                    font_size_offset=mit_font_ofs,
                    font_size_minimum=mit_font_min,
                    font_size_fixed=mit_font_fixed,
                    font_color=mit_font_color,
                    narrow_width_mult=mit_narrow_wide,
                    narrow_font_cap=mit_narrow_fcap,
                    verbose=mit_verbose,
                    skip_no_text=mit_skip_no_text,
                    overwrite=mit_overwrite,
                    cpu_priority=cpu_priority,
                    gpt_style=translation_style,
                    image_quality=image_quality,
                    on_log=on_log,
                    on_progress=on_progress,
                )
            else:
                lang_label = "ZH→VI" if src_lang == "zh" else "JA→VI" if src_lang == "ja" else "EN→VI"
                _t_push({"type": "log", "msg": f"Backend: default  Lang={lang_label}  Model={model}  GPU={use_gpu}  Inpainter={inpainter}"})
                translator = te.ImageTranslator(
                    model=model,
                    font_path=_find_font(),
                    use_gpu=use_gpu,
                    src_lang=src_lang,
                    inpainter=inpainter,
                    overwrite=overwrite,
                    font_scale=font_scale,
                    cpu_priority=cpu_priority,
                    translation_style=translation_style,
                    llm_base_url=llm_base_url,
                    llm_api_type=llm_api_type,
                    image_quality=image_quality,
                    on_log=on_log,
                    on_progress=on_progress,
                )

            ok, fail, failed_images = translator.process_folder(
                input_dir=input_dir,
                output_dir=output_dir,
                stop_event=_t_stop,
                images_override=images_override,
            )
            _t_push({"type": "log",  "msg": f"\n✔  Xong: {ok} OK, {fail} lỗi."})
            _t_push({"type": "done", "success": ok, "failed": fail, "output_dir": output_dir})
            # KHÔNG đụng `running` ở đây — _t_queue_worker quản lý cờ đó để các
            # job còn chờ trong hàng đợi được chạy tiếp.
            with _t_lock:
                _t_state.update(success=ok, failed=fail, status="done",
                                 failed_images=failed_images,
                                 input_dir=input_dir, output_dir=output_dir)
        except Exception as exc:
            _t_push({"type": "log",   "msg": f"✘  Lỗi: {exc}"})
            _t_push({"type": "error", "msg": str(exc)})
            with _t_lock:
                _t_state.update(status="error")

    started, queue_length = _t_enqueue_or_start(run)
    if not started:
        _t_push({"type": "log",
                 "msg": f"⏳ Đang có job dịch chạy — đã xếp hàng job mới: {input_dir} "
                        f"(vị trí {queue_length}). Sẽ tự chạy khi job trước xong."})
    return jsonify({"ok": True, "queued": (not started), "queue_length": queue_length,
                    "output_dir": output_dir})


@app.route("/api/translate/stop", methods=["POST"])
def api_translate_stop():
    with _t_lock:
        if not _t_state["running"]:
            return jsonify({"ok": True})
        dropped = len(_t_queue)
        _t_queue.clear()
        _t_state.update(running=False, status="stopped", queue=0)
    _t_stop.set()
    msg = "⚠  Yêu cầu dừng đã gửi."
    if dropped:
        msg += f" Đã hủy {dropped} job đang chờ trong hàng đợi."
    _t_push({"type": "log", "msg": msg})
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
    """Retry chỉ các ảnh dịch bị lỗi (xếp hàng nếu đang có job dịch chạy)."""
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
    if src_lang not in ("zh", "en", "ja"):
        src_lang = "zh"
    cpu_priority = str(data.get("cpu_priority", "below_normal")).strip()
    if cpu_priority not in ("normal", "below_normal", "idle"):
        cpu_priority = "below_normal"
    try:
        font_scale = float(data.get("font_scale", 0.60))
        font_scale = max(0.3, min(2.0, font_scale))
    except (TypeError, ValueError):
        font_scale = 0.60
    try:
        image_quality = int(data.get("image_quality", 95))
        image_quality = max(40, min(100, image_quality))
    except (TypeError, ValueError):
        image_quality = 95

    def run() -> None:
        _t_logfile.reopen([
            "# Nhật ký dịch ảnh (retry ảnh lỗi) — đè mỗi lần dịch mới.",
            "# Đọc file này để AI phân tích & cải thiện prompt/cấu hình.",
            f"# Thời điểm : {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"# Nguồn     : {input_dir}",
            f"# Xuất      : {output_dir}",
            f"# Retry     : {len(failed_images)} ảnh lỗi",
            (f"# Default   : lang={src_lang} model={model} inpainter={inpainter} "
             f"font_scale={font_scale} gpu={use_gpu} image_quality={image_quality} "
             f"cpu_priority={cpu_priority}"),
            "#" + "=" * 70,
            "",
        ])
        try:
            def on_log(msg: str):
                _t_push({"type": "log", "msg": str(msg)})

            def on_progress(done: int, total: int):
                with _t_lock:
                    _t_state["done"]  = done
                    _t_state["total"] = total
                _t_push({"type": "progress", "done": done, "total": total})

            translator = te.ImageTranslator(
                model=model,
                font_path=_find_font(),
                use_gpu=use_gpu,
                src_lang=src_lang,
                inpainter=inpainter,
                overwrite=True,
                font_scale=font_scale,
                cpu_priority=cpu_priority,
                image_quality=image_quality,
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
            # KHÔNG đụng `running` — _t_queue_worker quản lý cờ đó.
            with _t_lock:
                _t_state.update(success=ok, failed=fail, status="done",
                                 failed_images=new_failed,
                                 input_dir=input_dir, output_dir=output_dir)
        except Exception as exc:
            _t_push({"type": "log",   "msg": f"✘  Lỗi: {exc}"})
            _t_push({"type": "error", "msg": str(exc)})
            with _t_lock:
                _t_state.update(status="error")

    started, queue_length = _t_enqueue_or_start(run)
    if not started:
        _t_push({"type": "log",
                 "msg": f"⏳ Đang có job dịch chạy — retry {len(failed_images)} ảnh lỗi đã xếp hàng "
                        f"(vị trí {queue_length}). Sẽ tự chạy khi job trước xong."})
    return jsonify({"ok": True, "queued": (not started), "queue_length": queue_length,
                    "output_dir": output_dir})


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


# Thư mục cache thumbnail (WebP) — tránh tải/decode lại ảnh full-res cho lưới preview.
_THUMB_DIR = Path(tempfile.gettempdir()) / "crawl_preview_thumbs"


def _make_thumbnail(p: Path, mtime: int, size: int) -> Path | None:
    """Tạo (hoặc lấy từ cache) thumbnail WebP cạnh dài ≤ size px.

    Cache trên đĩa theo (path, mtime, size) → chỉ sinh 1 lần / ảnh.
    Trả None nếu thiếu Pillow hoặc lỗi → caller fallback ảnh gốc.
    """
    try:
        from PIL import Image
    except Exception:
        return None
    key = hashlib.sha1(f"{p}|{mtime}|{size}".encode("utf-8")).hexdigest()
    out = _THUMB_DIR / f"{key}.webp"
    if out.is_file():
        return out
    try:
        _THUMB_DIR.mkdir(parents=True, exist_ok=True)
        resample = getattr(Image, "Resampling", Image).LANCZOS
        with Image.open(p) as im:
            im.draft("RGB", (size, size))  # tăng tốc decode JPEG
            im = im.convert("RGB")
            im.thumbnail((size, size), resample)
            im.save(out, "WEBP", quality=80, method=4)
        return out
    except Exception:
        return None


def _prune_thumb_cache(max_age_days: int = 7, max_files: int = 5000) -> None:
    """Best-effort: dọn thumbnail cache cũ để temp không phình vô hạn.

    Xóa file quá `max_age_days` ngày, và nếu vẫn quá `max_files` thì xóa thêm
    các file cũ nhất. Không bao giờ ném lỗi (chạy lúc khởi động)."""
    try:
        if not _THUMB_DIR.is_dir():
            return
        files = []
        for f in _THUMB_DIR.glob("*.webp"):
            try:
                files.append((f, f.stat().st_mtime))
            except OSError:
                pass
    except Exception:
        return
    files.sort(key=lambda e: e[1])  # cũ nhất trước
    now = time.time()
    cutoff = now - max_age_days * 86400
    excess = len(files) - max_files
    for idx, (f, mtime) in enumerate(files):
        if idx < excess or mtime < cutoff:
            try:
                f.unlink()
            except OSError:
                pass


@app.route("/api/translate/image")
def api_translate_image():
    """Phục vụ file ảnh từ đường dẫn local (dùng cho preview).

    ?thumb=<px> → trả thumbnail WebP nhỏ (dùng cho lưới preview). Không có
    tham số này → trả ảnh full-res (dùng cho lightbox khi mở to).
    """
    path_str = request.args.get("path", "").strip()
    if not path_str:
        return jsonify({"error": "Thiếu path"}), 400
    p = Path(path_str).resolve()
    if not p.is_file() or p.suffix.lower() not in te.IMAGE_EXTS:
        return jsonify({"error": "File không hợp lệ"}), 400

    thumb = request.args.get("thumb", "").strip()
    if thumb:
        try:
            size = max(64, min(512, int(thumb)))
        except (TypeError, ValueError):
            size = 256
        try:
            mtime = int(p.stat().st_mtime)
        except OSError:
            mtime = 0
        tpath = _make_thumbnail(p, mtime, size)
        if tpath is not None:
            resp = send_file(str(tpath), mimetype="image/webp")
            resp.headers["Cache-Control"] = "public, max-age=86400, immutable"
            return resp
        # Pillow lỗi/thiếu → fallback xuống phục vụ ảnh gốc bên dưới.

    mime, _ = mimetypes.guess_type(str(p))
    resp = send_file(str(p), mimetype=mime or "image/jpeg")
    # Cho phép browser cache (rev theo mtime từ client)
    resp.headers["Cache-Control"] = "public, max-age=3600, immutable"
    return resp


@app.route("/api/translate/source")
def api_translate_source():
    """Phục vụ ẢNH GỐC (chưa dịch) khớp với một ảnh đã dịch — dùng cho so sánh.

    ?dir=<thư mục nguồn>&name=<tên file ảnh đã dịch>. Khớp ĐÚNG tên trước; nếu
    không có (vd pipeline đổi đuôi .jpg→.png) thì khớp theo phần tên KHÔNG đuôi
    (stem) với bất kỳ đuôi ảnh nào. ?thumb=<px> trả thumbnail như /api/translate/image.
    """
    dir_str = request.args.get("dir", "").strip()
    name = request.args.get("name", "").strip()
    if not dir_str or not name:
        return jsonify({"error": "Thiếu dir/name"}), 400
    d = Path(dir_str).resolve()
    if not d.is_dir():
        return jsonify({"error": "Thư mục nguồn không tồn tại"}), 404
    name = Path(name).name  # chống path traversal: chỉ lấy tên file
    cand = d / name
    if not (cand.is_file() and cand.suffix.lower() in te.IMAGE_EXTS):
        cand = None
        stem = Path(name).stem.lower()
        for f in d.iterdir():
            if f.is_file() and f.suffix.lower() in te.IMAGE_EXTS and f.stem.lower() == stem:
                cand = f
                break
    if cand is None:
        return jsonify({"error": "Không tìm thấy ảnh gốc"}), 404

    thumb = request.args.get("thumb", "").strip()
    if thumb:
        try:
            size = max(64, min(512, int(thumb)))
        except (TypeError, ValueError):
            size = 256
        try:
            mtime = int(cand.stat().st_mtime)
        except OSError:
            mtime = 0
        tpath = _make_thumbnail(cand, mtime, size)
        if tpath is not None:
            resp = send_file(str(tpath), mimetype="image/webp")
            resp.headers["Cache-Control"] = "public, max-age=86400, immutable"
            return resp

    mime, _ = mimetypes.guess_type(str(cand))
    resp = send_file(str(cand), mimetype=mime or "image/jpeg")
    resp.headers["Cache-Control"] = "public, max-age=3600, immutable"
    return resp


def _glossary_file_for(input_dir: str) -> Path:
    """Đường dẫn glossary.txt của bộ truyện chứa input_dir. Ưu tiên file đã tồn
    tại ở thư mục ảnh / thư mục cha; mặc định là gốc bộ (cha của 'original')."""
    inp = Path(input_dir)
    story_root = te.resolve_story_root(inp)
    for c in (inp / "glossary.txt", inp.parent / "glossary.txt",
              story_root / "glossary.txt"):
        if c.is_file():
            return c
    return story_root / "glossary.txt"


def _glossary_counts(text: str) -> tuple[int, int]:
    """(số mục khoá-cứng '=>', số dòng @note) — để hiển thị nhanh trên UI."""
    terms = notes = 0
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("@note"):
            notes += 1
        elif "=>" in line:
            terms += 1
    return terms, notes


@app.route("/api/translate/glossary")
def api_translate_glossary_get():
    """Đọc glossary.txt của bộ truyện chứa ?input_dir=. Trả {path, content,
    exists, terms, notes}. Glossary tự học sau mỗi lần dịch; UI cho xem & sửa tay."""
    input_dir = request.args.get("input_dir", "").strip()
    if not input_dir:
        return jsonify({"error": "Thiếu input_dir"}), 400
    if not Path(input_dir).is_dir():
        return jsonify({"error": f"Thư mục không tồn tại: {input_dir}"}), 404
    gpath = _glossary_file_for(input_dir)
    content = ""
    exists = gpath.is_file()
    if exists:
        try:
            content = gpath.read_text(encoding="utf-8")
        except Exception as e:
            return jsonify({"error": f"Không đọc được glossary: {e}"}), 500
    terms, notes = _glossary_counts(content)
    return jsonify({"path": str(gpath), "content": content, "exists": exists,
                    "terms": terms, "notes": notes})


@app.route("/api/translate/glossary", methods=["POST"])
def api_translate_glossary_save():
    """Ghi (đè) glossary.txt do người dùng sửa tay. Body JSON: {input_dir, content}."""
    data = request.get_json(silent=True) or {}
    input_dir = str(data.get("input_dir", "")).strip()
    content = data.get("content", "")
    if not input_dir:
        return jsonify({"error": "Thiếu input_dir"}), 400
    if not isinstance(content, str):
        return jsonify({"error": "content phải là chuỗi"}), 400
    if not Path(input_dir).is_dir():
        return jsonify({"error": f"Thư mục không tồn tại: {input_dir}"}), 404
    gpath = _glossary_file_for(input_dir)
    try:
        gpath.parent.mkdir(parents=True, exist_ok=True)
        gpath.write_text(content, encoding="utf-8")
    except Exception as e:
        return jsonify({"error": f"Không ghi được glossary: {e}"}), 500
    terms, notes = _glossary_counts(content)
    return jsonify({"ok": True, "path": str(gpath), "terms": terms, "notes": notes})


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

    # stat() một lần / file rồi sort theo mtime (tránh stat lặp lại trong list comp).
    entries: list[tuple[Path, float]] = []
    for f in p.iterdir():
        if f.suffix.lower() not in te.IMAGE_EXTS:
            continue
        try:
            entries.append((f, f.stat().st_mtime))
        except OSError:
            continue
    total = len(entries)
    # limit>0 → chọn N ảnh mới nhất theo mtime trước khi sort tên
    if limit > 0:
        entries.sort(key=lambda e: e[1])
        entries = entries[-limit:]
    # sort hiển thị theo tên (natural order) thay vì date
    entries.sort(key=lambda e: _regions_natural_key(e[0].name))
    images = [
        {"name": f.name, "path": str(f), "mtime": int(mtime)}
        for f, mtime in entries
    ]
    return jsonify({"images": images, "total": total})


# ── Manual text regions (hand-drawn boxes) ──────────────────────────────────

def _regions_natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', name)]


@app.route("/api/regions/list")
def api_regions_list():
    """Liệt kê ảnh trong thư mục nguồn kèm số vùng thủ công đã lưu (badge)."""
    dir_str = request.args.get("dir", "").strip()
    if not dir_str:
        return jsonify({"error": "Thiếu dir"}), 400
    p = Path(dir_str).resolve()
    if not p.is_dir():
        return jsonify({"error": f"Thư mục không tồn tại: {dir_str}"}), 400

    counts: dict[str, int] = {}
    # mtime sidecar vùng thủ công → mốc "vừa chỉnh sửa" cho sort "Mới sửa".
    region_mtime: dict[str, float] = {}
    rdir = p / te.REGIONS_DIRNAME
    if rdir.is_dir():
        for jf in rdir.glob("*.json"):
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
                n = len([r for r in (data.get("regions") or []) if isinstance(r, dict)])
            except Exception:
                n = 0
            if n:
                counts[jf.stem] = n  # jf.stem of "002.jpg.json" == "002.jpg"
            try:
                region_mtime[jf.stem] = jf.stat().st_mtime
            except OSError:
                pass

    images = []
    for f in sorted((c for c in p.iterdir() if c.is_file()), key=lambda c: _regions_natural_key(c.name)):
        if f.suffix.lower() not in te.IMAGE_EXTS:
            continue
        try:
            src_mtime = f.stat().st_mtime
        except OSError:
            src_mtime = 0.0
        # "Mới sửa" = mốc gần nhất giữa ảnh nguồn và sidecar vùng thủ công đã lưu.
        mtime = max(src_mtime, region_mtime.get(f.name, 0.0))
        images.append({
            "name": f.name, "path": str(f),
            "count": counts.get(f.name, 0), "mtime": int(mtime),
        })
    return jsonify({"images": images})


@app.route("/api/regions/get")
def api_regions_get():
    """Đọc sidecar vùng thủ công của một ảnh."""
    dir_str = request.args.get("dir", "").strip()
    img = request.args.get("img", "").strip()
    empty = {"mode": "merge", "regions": [], "mask_dilate": 1}
    if not dir_str or not img:
        return jsonify(empty)
    f = Path(dir_str).resolve() / te.REGIONS_DIRNAME / (img + ".json")
    if not f.is_file():
        return jsonify(empty)
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        mode = data.get("mode", "merge")
        if mode not in ("merge", "replace"):
            mode = "merge"
        regions = [r for r in (data.get("regions") or []) if isinstance(r, dict)]
        try:
            mask_dilate = max(0, min(10, int(data.get("mask_dilate", 1))))
        except (TypeError, ValueError):
            mask_dilate = 1
        return jsonify({"mode": mode, "regions": regions, "mask_dilate": mask_dilate})
    except Exception:
        return jsonify(empty)


@app.route("/api/regions/save", methods=["POST"])
def api_regions_save():
    """Ghi sidecar vùng thủ công cho một ảnh (toạ độ chuẩn hoá 0..1).
    Danh sách rỗng → xoá sidecar."""
    data = request.get_json(silent=True) or {}
    dir_str = str(data.get("dir", "")).strip()
    img = str(data.get("img", "")).strip()
    mode = str(data.get("mode", "merge")).strip().lower()
    if mode not in ("merge", "replace"):
        mode = "merge"

    p = Path(dir_str).resolve()
    if not dir_str or not p.is_dir():
        return jsonify({"error": "Thư mục không hợp lệ."}), 400
    imgp = p / img
    if not img or not imgp.is_file() or imgp.suffix.lower() not in te.IMAGE_EXTS:
        return jsonify({"error": "Ảnh không hợp lệ."}), 400

    clean = []
    for r in (data.get("regions") or []):
        try:
            x = float(r["x"]); y = float(r["y"])
            w = float(r["w"]); h = float(r["h"])
        except (KeyError, TypeError, ValueError):
            continue
        # clamp to [0,1] and drop degenerate boxes
        x = max(0.0, min(1.0, x)); y = max(0.0, min(1.0, y))
        w = max(0.0, min(1.0 - x, w)); h = max(0.0, min(1.0 - y, h))
        if w < 0.002 or h < 0.002:
            continue
        box = {"x": round(x, 5), "y": round(y, 5), "w": round(w, 5), "h": round(h, 5)}
        # inpaint_only = chỉ xoá sạch cả khung (không OCR/dịch/vẽ chữ). Ưu tiên cao
        # nhất: bỏ qua text/font nếu có.
        # erase_flat = lấp CẢ KHUNG bằng màu nền (không inpaint) — cho chữ trên chat
        # box/panel màu phẳng. Áp cho cả vùng chỉ-xoá lẫn gõ tay. flat_color =
        # '#rrggbb' người dùng tự chọn; vắng = auto (trung vị vành ngoài box).
        if r.get("erase_flat"):
            box["erase_flat"] = True
            fc = str(r.get("flat_color") or "").strip().lstrip("#")
            if re.fullmatch(r"[0-9a-fA-F]{6}", fc):
                box["flat_color"] = "#" + fc.lower()
        if r.get("inpaint_only"):
            box["inpaint_only"] = True
            clean.append(box)
            continue
        # text = chữ Việt gõ tay (tùy chọn) → bỏ qua OCR/dịch, vẽ thẳng
        txt = str(r.get("text") or "").strip()
        if txt:
            box["text"] = txt[:2000]
        # erase_box = vùng gõ tay nhưng xoá CẢ KHUNG (thay vì chỉ nét) trước khi vẽ
        # chữ — cho chữ SFX màu/viền mà mask theo nét tách không sạch.
        if r.get("erase_box"):
            box["erase_box"] = True
        # font = tên file font trong fonts/ (chỉ basename, chống path traversal);
        # font_size = cỡ chữ px (6..200, 0/None = tự canh). Chỉ có ý nghĩa khi có text.
        font_name = str(r.get("font") or "").strip().replace("\\", "/").split("/")[-1]
        if font_name and font_name.lower().endswith((".ttf", ".otf")):
            box["font"] = font_name[:80]
        try:
            fsz = int(r.get("font_size") or 0)
            if 6 <= fsz <= 200:
                box["font_size"] = fsz
        except (TypeError, ValueError):
            pass
        # mask_dilate riêng cho vùng (0..10); vắng = dùng mặc định của ảnh.
        try:
            mdl = int(r["mask_dilate"])
            if 0 <= mdl <= 10:
                box["mask_dilate"] = mdl
        except (KeyError, TypeError, ValueError):
            pass
        # rotate = góc nghiêng chữ (độ, -180..180); 0/vắng = không nghiêng.
        try:
            rot = float(r.get("rotate") or 0)
            if -180 <= rot <= 180 and abs(rot) > 0.01:
                box["rotate"] = round(rot, 1)
        except (TypeError, ValueError):
            pass
        clean.append(box)

    try:
        mask_dilate = max(0, min(10, int(data.get("mask_dilate", 1))))
    except (TypeError, ValueError):
        mask_dilate = 1

    rdir = p / te.REGIONS_DIRNAME
    f = rdir / (img + ".json")
    if not clean:
        try:
            f.unlink()
        except Exception:
            pass
        return jsonify({"ok": True, "count": 0})
    try:
        rdir.mkdir(exist_ok=True)
        f.write_text(
            json.dumps({"image": img, "mode": mode, "mask_dilate": mask_dilate, "regions": clean}, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify({"ok": True, "count": len(clean)})


def _autotrim_bbox(arr, tol: int = 22, frac: float = 0.985, gap: int = 8):
    """Bbox (x0,y0,x1,y1) sau khi gọt các DẢI viền đồng màu ở 4 cạnh, XÉT TỪNG CẠNH
    ĐỘC LẬP. Mỗi cạnh dùng màu (trung vị) của chính mép đó làm chuẩn rồi gọt dần các
    hàng/cột vào trong.

    Một hàng/cột là viền khi ≥ `frac` pixel của nó nằm trong `tol` so với màu mép —
    KHÔNG đòi một màu tuyệt đối, nên bỏ qua được vài pixel lạ (nhiễu JPEG, răng cưa).
    Quét có DUNG SAI KHOẢNG TRỐNG `gap`: bỏ qua tối đa `gap` dòng-không-viền liên tiếp
    rồi cắt tới dòng-viền SÂU NHẤT, chỉ dừng hẳn khi gặp nội dung thật (chuỗi dòng khác
    màu dài hơn gap). Nhờ vậy một vệt mờ/chữ-ghost còn sót/đường kẻ nằm GIỮA dải viền
    không làm trim dừng sớm và để sót phần viền phía sau.
    Cũng xử lý được ảnh chỉ có viền trên/dưới (ảnh tràn hết chiều ngang).
    Trả None nếu không có dải viền nào để cắt."""
    import numpy as np
    a = arr.astype(np.int16)
    h, w = a.shape[:2]

    def solid(line, ref):
        # line: (n, ch). True nếu ĐA SỐ pixel của dòng trùng màu mép tham chiếu.
        d = np.abs(line - ref).max(axis=1)  # sai khác kênh-lớn-nhất của từng pixel
        return float((d <= tol).mean()) >= frac

    def trim(line_at, n, ref):
        # Quét i=0..n-1 từ mép vào; trả số dòng viền cần cắt = tới dòng-viền sâu nhất,
        # bỏ qua các đứt quãng ≤ gap. 0 nghĩa là mép này không có viền.
        last, run = -1, 0
        for i in range(n):
            if solid(line_at(i), ref):
                last, run = i, 0
            else:
                run += 1
                if run > gap:
                    break
        return last + 1

    top = trim(lambda i: a[i], h, np.median(a[0], axis=0))
    bot = trim(lambda i: a[h - 1 - i], h, np.median(a[-1], axis=0))
    left = trim(lambda i: a[:, i], w, np.median(a[:, 0], axis=0))
    right = trim(lambda i: a[:, w - 1 - i], w, np.median(a[:, -1], axis=0))
    x0, y0, x1, y1 = left, top, w - right, h - bot

    if (x0, y0, x1, y1) == (0, 0, w, h):
        return None  # không gọt được cạnh nào
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None  # gọt gần hết ảnh (ảnh đồng màu) — bỏ qua
    return x0, y0, x1, y1


@app.route("/api/images/crop", methods=["POST"])
def api_images_crop():
    """Crop hàng loạt ảnh trong thư mục — bỏ viền/khoảng trắng thừa.
    Hai chế độ: rect = khung chữ nhật chuẩn hoá 0..1 (vẽ trong editor, áp cho mọi
    ảnh bất kể kích thước), hoặc auto=true = tự dò viền đồng màu từng ảnh và cắt.
    names rỗng = cả thư mục; dst rỗng = <nguồn>_crop; dst trùng nguồn = ghi đè."""
    from PIL import Image
    import numpy as np
    data = request.get_json(silent=True) or {}
    src_str = str(data.get("src", "")).strip()
    src = Path(src_str)
    if not src_str or not src.is_dir():
        return jsonify({"error": "Thư mục nguồn không hợp lệ."}), 400
    dst_str = str(data.get("dst", "")).strip()
    dst = Path(dst_str) if dst_str else src.parent / (src.name + "_crop")
    in_place = dst.resolve() == src.resolve()

    auto = bool(data.get("auto"))
    rect = data.get("rect") or {}
    if not auto:
        try:
            rx = max(0.0, min(1.0, float(rect["x"]))); ry = max(0.0, min(1.0, float(rect["y"])))
            rw = max(0.0, min(1.0 - rx, float(rect["w"]))); rh = max(0.0, min(1.0 - ry, float(rect["h"])))
        except (KeyError, TypeError, ValueError):
            return jsonify({"error": "Thiếu khung crop (rect) hợp lệ."}), 400
        if rw < 0.01 or rh < 0.01:
            return jsonify({"error": "Khung crop quá nhỏ."}), 400

    # names = tên file (basename, chống path traversal); rỗng = mọi ảnh trong nguồn.
    names = [Path(str(n)).name for n in (data.get("names") or []) if str(n).strip()]
    if names:
        candidates = [src / n for n in names]
    else:
        candidates = sorted(f for f in src.iterdir() if f.is_file())
    files, ignored = [], []
    for f in candidates:
        if f.is_file() and f.suffix.lower() in te.IMAGE_EXTS:
            files.append(f)
        elif names:
            # Chỉ báo file bị loại khi người dùng CHỈ ĐỊNH ảnh này (không phải quét cả thư
            # mục) — nếu không sẽ nhiễu vì thư mục thường lẫn file không phải ảnh.
            if not f.exists():
                ignored.append(f"{f.name}: không tìm thấy file")
            elif not f.is_file():
                ignored.append(f"{f.name}: không phải file")
            else:
                ignored.append(f"{f.name}: đuôi {f.suffix or '(trống)'} không hỗ trợ "
                               f"(chỉ {', '.join(sorted(te.IMAGE_EXTS))})")
    mode_label = "auto-trim viền đồng màu" if auto else "theo khung vùng"
    print(f"  ✂  Crop ({mode_label}): {len(files)} ảnh từ {src} → {dst}", flush=True)
    for ig in ignored:
        print(f"  ⚠  Crop bỏ qua {ig}", flush=True)
    if not files:
        return jsonify({"error": "Không có ảnh hợp lệ để crop.",
                        "ignored": ignored[:20], "ignored_count": len(ignored)}), 400

    try:
        dst.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        print(f"  ⚠  Crop lỗi tạo thư mục đích {dst}: {exc}", flush=True)
        return jsonify({"error": f"Không tạo được thư mục đích: {exc}"}), 500

    cropped, skipped, failed = 0, [], []
    for f in files:
        try:
            with Image.open(f) as im:
                im.load()
                w, h = im.size
                if auto:
                    arr = np.asarray(im.convert("RGB"))
                    bbox = _autotrim_bbox(arr)
                    if bbox is None:
                        skipped.append(f"{f.name}: không có viền đồng màu để cắt")
                        continue
                    x0, y0, x1, y1 = bbox
                    print(f"  ·  Crop {f.name}: cắt trên {y0} dưới {h - y1} "
                          f"trái {x0} phải {w - x1}px", flush=True)
                else:
                    x0 = int(round(rx * w)); y0 = int(round(ry * h))
                    x1 = int(round((rx + rw) * w)); y1 = int(round((ry + rh) * h))
                if x1 - x0 < 8 or y1 - y0 < 8:
                    skipped.append(f"{f.name}: vùng giữ lại quá nhỏ ({x1 - x0}×{y1 - y0}px)")
                    continue
                out = im.crop((x0, y0, x1, y1))
                if out.mode in ("RGBA", "P") and f.suffix.lower() in (".jpg", ".jpeg"):
                    out = out.convert("RGB")
                save_kw = {"quality": 95} if f.suffix.lower() in (".jpg", ".jpeg", ".webp") else {}
                out.save(dst / f.name, **save_kw)
                cropped += 1
        except Exception as exc:
            msg = f"{f.name}: {type(exc).__name__}: {exc}"
            failed.append(msg)
            print(f"  ⚠  Crop lỗi {msg}", flush=True)
    for sk in skipped:
        print(f"  ·  Crop bỏ qua {sk}", flush=True)
    print(f"  ✔  Crop xong: {cropped} cắt · {len(skipped)} bỏ qua · "
          f"{len(failed)} lỗi · {len(ignored)} loại", flush=True)
    return jsonify({
        "ok": True, "cropped": cropped,
        "skipped": len(skipped), "skipped_detail": skipped[:20],
        "ignored": ignored[:20], "ignored_count": len(ignored),
        "failed": failed[:20], "fail_count": len(failed),
        "dst": str(dst), "in_place": in_place,
    })


@app.route("/api/images/compress", methods=["POST"])
def api_images_compress():
    """Giảm dung lượng ảnh hàng loạt — nén lại theo `quality` (40–100).
    PNG/BMP được chuyển sang JPEG nén (đuôi đổi .jpg). names rỗng = cả thư mục;
    dst rỗng = ghi đè TẠI CHỖ trong thư mục nguồn."""
    from PIL import Image
    data = request.get_json(silent=True) or {}
    src_str = str(data.get("src", "")).strip()
    src = Path(src_str)
    if not src_str or not src.is_dir():
        return jsonify({"error": "Thư mục nguồn không hợp lệ."}), 400
    dst_str = str(data.get("dst", "")).strip()
    dst = Path(dst_str) if dst_str else src
    in_place = dst.resolve() == src.resolve()
    try:
        quality = max(40, min(100, int(data.get("quality", 80))))
    except (TypeError, ValueError):
        quality = 80
    if quality >= 100:
        return jsonify({"error": "Chất lượng 100 = giữ nguyên gốc, không giảm dung lượng. "
                                 "Hãy chọn mức < 100."}), 400

    names = [Path(str(n)).name for n in (data.get("names") or []) if str(n).strip()]
    candidates = [src / n for n in names] if names else sorted(f for f in src.iterdir() if f.is_file())
    files, ignored = [], []
    for f in candidates:
        if f.is_file() and f.suffix.lower() in te.IMAGE_EXTS:
            files.append(f)
        elif names and not f.is_file():
            ignored.append(f"{f.name}: không tìm thấy file")
    print(f"  🗜  Nén ảnh (q={quality}): {len(files)} ảnh từ {src} → {dst}", flush=True)
    if not files:
        return jsonify({"error": "Không có ảnh hợp lệ để nén.",
                        "ignored": ignored[:20], "ignored_count": len(ignored)}), 400

    try:
        dst.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        return jsonify({"error": f"Không tạo được thư mục đích: {exc}"}), 500

    done, failed = 0, []
    bytes_before = bytes_after = 0
    for f in files:
        try:
            before = f.stat().st_size
            with Image.open(f) as im:
                im.load()
                written = te.save_image_compressed(im, dst / f.name, quality)
            # PNG→JPEG đổi đuôi + ghi đè tại chỗ → xoá file gốc để không nhân đôi.
            if in_place and written.resolve() != f.resolve() and f.exists():
                try:
                    f.unlink()
                except OSError:
                    pass
            after = written.stat().st_size
            bytes_before += before
            bytes_after += after
            done += 1
        except Exception as exc:
            msg = f"{f.name}: {type(exc).__name__}: {exc}"
            failed.append(msg)
            print(f"  ⚠  Nén lỗi {msg}", flush=True)
    saved = bytes_before - bytes_after
    pct = (saved / bytes_before * 100) if bytes_before else 0.0
    print(f"  ✔  Nén xong: {done} ảnh · {bytes_before/1048576:.1f}MB → "
          f"{bytes_after/1048576:.1f}MB (giảm {pct:.0f}%) · {len(failed)} lỗi", flush=True)
    return jsonify({
        "ok": True, "done": done,
        "failed": failed[:20], "fail_count": len(failed),
        "ignored": ignored[:20], "ignored_count": len(ignored),
        "quality": quality, "in_place": in_place, "dst": str(dst),
        "mb_before": round(bytes_before / 1048576, 2),
        "mb_after": round(bytes_after / 1048576, 2),
        "saved_pct": round(pct, 1),
    })


# ── Entry point ───────────────────────────────────────────────────────────────

def _auto_apply_mit_patches():
    """Best-effort: copy patches/ into the MIT venv on startup so edits to the
    patch files take effect without remembering to run apply_patches.py.
    Never crashes the web app — MIT not installed → skip silently."""
    try:
        import apply_patches
        if apply_patches.find_site_packages() is None:
            return  # MIT venv not installed yet — nothing to patch.
        apply_patches.apply()
    except SystemExit:
        pass  # apply() exits on "not found"; ignore so we never kill the app.
    except Exception as exc:
        print(f"  ⚠  Bỏ qua auto-apply patches MIT: {exc}")


if __name__ == "__main__":
    import webbrowser
    _auto_apply_mit_patches()
    _prune_thumb_cache()
    port = int(os.environ.get("PORT", 5000))
    threading.Timer(1.2, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    print(f"\n  ✔  Web UI: http://127.0.0.1:{port}\n")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
