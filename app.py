"""
Image Crawler GUI - Giao diện đồ họa để crawl và tải ảnh từ web.
Yêu cầu: pip install -r requirements.txt
"""

import threading
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk

from crawler import crawl


class CrawlerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Crawler")
        self.resizable(True, True)
        self.minsize(620, 520)
        self._build_ui()
        self._crawling = False

    # ── Xây dựng giao diện ──────────────────────────────────────────────────

    def _build_ui(self):
        pad = {"padx": 10, "pady": 5}

        # URL
        url_frame = tk.LabelFrame(self, text="URL trang web", **pad)
        url_frame.pack(fill="x", **pad)
        self.url_var = tk.StringVar()
        tk.Entry(url_frame, textvariable=self.url_var, width=70).pack(
            fill="x", padx=6, pady=4
        )

        # Thư mục lưu
        folder_frame = tk.LabelFrame(self, text="Thư mục lưu ảnh", **pad)
        folder_frame.pack(fill="x", **pad)
        folder_inner = tk.Frame(folder_frame)
        folder_inner.pack(fill="x", padx=6, pady=4)
        self.folder_var = tk.StringVar(value="output")
        tk.Entry(folder_inner, textvariable=self.folder_var, width=60).pack(
            side="left", fill="x", expand=True
        )
        tk.Button(folder_inner, text="Chọn…", command=self._pick_folder).pack(
            side="left", padx=(4, 0)
        )

        # Tùy chọn nâng cao
        opts_frame = tk.LabelFrame(self, text="Tùy chọn", **pad)
        opts_frame.pack(fill="x", **pad)
        opts_inner = tk.Frame(opts_frame)
        opts_inner.pack(fill="x", padx=6, pady=4)

        tk.Label(opts_inner, text="Delay (giây):").grid(row=0, column=0, sticky="w")
        self.delay_var = tk.DoubleVar(value=0.3)
        tk.Spinbox(opts_inner, from_=0, to=5, increment=0.1,
                   textvariable=self.delay_var, width=6, format="%.1f").grid(
            row=0, column=1, padx=(4, 20), sticky="w"
        )

        tk.Label(opts_inner, text="Luồng tải:").grid(row=0, column=2, sticky="w")
        self.workers_var = tk.IntVar(value=4)
        tk.Spinbox(opts_inner, from_=1, to=16, textvariable=self.workers_var,
                   width=4).grid(row=0, column=3, padx=(4, 20), sticky="w")

        tk.Label(opts_inner, text="Timeout (giây):").grid(row=0, column=4, sticky="w")
        self.timeout_var = tk.IntVar(value=20)
        tk.Spinbox(opts_inner, from_=5, to=120, textvariable=self.timeout_var,
                   width=4).grid(row=0, column=5, padx=(4, 0), sticky="w")

        # Thanh tiến trình + nút
        ctrl_frame = tk.Frame(self)
        ctrl_frame.pack(fill="x", **pad)
        self.start_btn = tk.Button(
            ctrl_frame, text="▶  Bắt đầu crawl",
            bg="#2196F3", fg="white", font=("Arial", 10, "bold"),
            command=self._start
        )
        self.start_btn.pack(side="left")
        self.stop_btn = tk.Button(
            ctrl_frame, text="■  Dừng",
            bg="#f44336", fg="white", font=("Arial", 10, "bold"),
            command=self._stop, state="disabled"
        )
        self.stop_btn.pack(side="left", padx=(8, 0))
        self.status_lbl = tk.Label(ctrl_frame, text="Sẵn sàng.", anchor="w")
        self.status_lbl.pack(side="left", padx=(12, 0), fill="x", expand=True)

        self.progress = ttk.Progressbar(self, mode="determinate")
        self.progress.pack(fill="x", padx=10, pady=(0, 4))

        # Log
        log_frame = tk.LabelFrame(self, text="Nhật ký", **pad)
        log_frame.pack(fill="both", expand=True, **pad)
        self.log_box = scrolledtext.ScrolledText(
            log_frame, height=14, state="disabled",
            font=("Consolas", 9), bg="#1e1e1e", fg="#d4d4d4",
            insertbackground="white"
        )
        self.log_box.pack(fill="both", expand=True, padx=4, pady=4)

    # ── Xử lý sự kiện ──────────────────────────────────────────────────────

    def _pick_folder(self):
        folder = filedialog.askdirectory(title="Chọn thư mục lưu ảnh")
        if folder:
            self.folder_var.set(folder)

    def _log(self, msg: str):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _set_status(self, msg: str):
        self.status_lbl.config(text=msg)

    def _update_progress(self, done: int, total: int):
        if total:
            self.progress["maximum"] = total
            self.progress["value"] = done
            self._set_status(f"Đang tải… {done}/{total}")

    def _start(self):
        url = self.url_var.get().strip()
        folder = self.folder_var.get().strip()
        if not url:
            self._log("⚠  Vui lòng nhập URL.")
            return
        if not folder:
            self._log("⚠  Vui lòng chọn thư mục lưu ảnh.")
            return
        if not url.startswith(("http://", "https://")):
            self._log("⚠  URL phải bắt đầu bằng http:// hoặc https://")
            return

        self._crawling = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.progress["value"] = 0
        self._log(f"\n{'─'*60}")
        self._log(f"URL   : {url}")
        self._log(f"Folder: {folder}")

        def run():
            try:
                ok, fail = crawl(
                    url=url,
                    output_folder=folder,
                    delay=self.delay_var.get(),
                    max_workers=self.workers_var.get(),
                    timeout=self.timeout_var.get(),
                    on_progress=lambda d, t: self.after(0, self._update_progress, d, t),
                    on_log=lambda msg: self.after(0, self._log, msg),
                )
                self.after(
                    0, self._log,
                    f"\n✔  Hoàn thành: {ok} thành công, {fail} thất bại."
                )
                self.after(0, self._set_status, f"Xong. {ok} ảnh tải thành công.")
            except Exception as exc:
                self.after(0, self._log, f"\n✘  Lỗi: {exc}")
                self.after(0, self._set_status, "Lỗi.")
            finally:
                self._crawling = False
                self.after(0, self.start_btn.config, {"state": "normal"})
                self.after(0, self.stop_btn.config, {"state": "disabled"})

        threading.Thread(target=run, daemon=True).start()

    def _stop(self):
        # Ghi chú: dừng mềm — các request đang chạy sẽ hoàn thành tự nhiên
        self._crawling = False
        self._log("⚠  Yêu cầu dừng đã gửi (các request đang chạy sẽ hoàn thành).")
        self._set_status("Đang dừng…")


if __name__ == "__main__":
    app = CrawlerApp()
    app.mainloop()
