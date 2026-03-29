"""
Image Crawler Web UI – Giao diện web để crawl và tải ảnh từ web.
Chạy: python web_app.py  →  tự mở http://localhost:5000
"""

import json
import os
import string
import threading
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from crawler import crawl

app = Flask(__name__)

# ── Trạng thái toàn cục (một phiên crawl tại một thời điểm) ──────────────────
_lock = threading.Lock()
_logs: list[dict] = []          # Buffer tất cả sự kiện
_state: dict = {
    "running": False,
    "done": 0,
    "total": 0,
    "success": 0,
    "failed": 0,
    "status": "ready",          # ready | running | done | error | stopped
}
MAX_LOG = 3000


def _push(entry: dict) -> None:
    """Thêm entry vào log buffer (thread-safe, giới hạn MAX_LOG)."""
    with _lock:
        if len(_logs) < MAX_LOG:
            _logs.append(entry)


def _reset() -> None:
    """Xóa log và reset state (gọi trước khi bắt đầu crawl mới)."""
    with _lock:
        _logs.clear()
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

    _push({"type": "log",  "msg": "⚠  Yêu cầu dừng đã gửi (các request đang chạy sẽ hoàn thành)."})
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
        batch = _logs[since:]
        total = len(_logs)
        state = dict(_state)

    return jsonify({"logs": batch, "total": total, "state": state})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import webbrowser
    port = int(os.environ.get("PORT", 5000))
    threading.Timer(1.2, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    print(f"\n  ✔  Web UI: http://127.0.0.1:{port}\n")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
