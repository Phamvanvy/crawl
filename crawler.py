"""
Image Crawler - Crawl và tải ảnh từ bất kỳ trang web nào.
Sử dụng: python crawler.py <url> <output_folder> [options]
"""

import json
import os
import re
import time
import hashlib
import argparse
import threading
from pathlib import Path
from urllib.parse import urljoin, urlparse, unquote

import requests
from bs4 import BeautifulSoup

# ── Cấu hình mặc định ──────────────────────────────────────────────────────────
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg", ".avif"}
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
}
# Các thuộc tính HTML chứa URL ảnh (bao gồm lazy-load)
SRC_ATTRS = ["src", "data-src", "data-lazy", "data-original", "data-url", "data-lazy-src"]


# ── Tiện ích ────────────────────────────────────────────────────────────────────

def is_image_url(url: str) -> bool:
    """Kiểm tra URL có trỏ đến file ảnh không."""
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in IMAGE_EXTENSIONS)


def sanitize_filename(name: str, max_len: int = 200) -> str:
    """Làm sạch tên file, loại bỏ ký tự không hợp lệ trên Windows."""
    name = unquote(name)
    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    name = name.strip(". ")
    return name[:max_len] if name else "image"


def url_to_filename(url: str, index: int) -> str:
    """Tạo tên file từ URL; nếu không rõ thì dùng hash + số thứ tự."""
    parsed = urlparse(url)
    basename = os.path.basename(parsed.path)
    if basename and "." in basename:
        return sanitize_filename(basename)
    # Không có tên rõ ràng → dùng hash ngắn
    short_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    return f"image_{index:04d}_{short_hash}.jpg"


def unique_path(folder: Path, filename: str) -> Path:
    """Tránh ghi đè file đã tồn tại bằng cách thêm hậu tố số."""
    target = folder / filename
    if not target.exists():
        return target
    stem, suffix = os.path.splitext(filename)
    counter = 1
    while True:
        new_name = f"{stem}_{counter}{suffix}"
        candidate = folder / new_name
        if not candidate.exists():
            return candidate
        counter += 1


# ── Phần tải ảnh ───────────────────────────────────────────────────────────────

class ImageDownloader:
    def __init__(self, output_folder: str, delay: float = 0.3,
                 max_workers: int = 4, timeout: int = 20,
                 on_progress=None, on_log=None,
                 stop_event: threading.Event | None = None):
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.delay = delay
        self.max_workers = max_workers
        self.timeout = timeout
        self.on_progress = on_progress or (lambda done, total: None)
        self.on_log = on_log or print
        self.stop_event = stop_event or threading.Event()
        self._lock = threading.Lock()
        self._claimed: set[Path] = set()  # paths đã được claim bởi threads khác
        self._semaphore = threading.Semaphore(max_workers)
        self.success = 0
        self.failed = 0

    def _get_session(self, referer: str) -> requests.Session:
        return make_session(referer=referer)

    def _claim_path(self, filename: str) -> Path:
        """Thread-safe: reserve unique dest path trước khi ghi."""
        stem, suffix = os.path.splitext(filename)
        with self._lock:
            candidate = self.output_folder / filename
            counter = 0
            while candidate.exists() or candidate in self._claimed:
                counter += 1
                candidate = self.output_folder / f"{stem}_{counter}{suffix}"
            self._claimed.add(candidate)
        return candidate

    def _download_one(self, url: str, index: int, total: int, referer: str):
        if self.stop_event.is_set():
            return
        with self._semaphore:
            if self.stop_event.is_set():
                return
            filename = url_to_filename(url, index)
            dest = self._claim_path(filename)  # thread-safe reservation
            try:
                session = self._get_session(referer)
                resp = session.get(url, timeout=self.timeout, stream=True)
                resp.raise_for_status()
                content_type = resp.headers.get("Content-Type", "")
                if "image" not in content_type and not is_image_url(url):
                    self.on_log(f"  [skip] Không phải ảnh: {url}")
                    return
                with open(dest, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                with self._lock:
                    self.success += 1
                    done = self.success + self.failed
                    self.on_progress(done, total)
                self.on_log(f"  [OK {index}/{total}] {dest.name}")
            except Exception as exc:
                with self._lock:
                    self.failed += 1
                    done = self.success + self.failed
                    self.on_progress(done, total)
                self.on_log(f"  [FAIL {index}/{total}] {url}  →  {exc}")
            time.sleep(self.delay)

    def download_all(self, image_urls: list[str], referer: str):
        total = len(image_urls)
        threads = []
        for i, url in enumerate(image_urls, start=1):
            if self.stop_event.is_set():
                break
            t = threading.Thread(
                target=self._download_one,
                args=(url, i, total, referer),
                daemon=True,
            )
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        return self.success, self.failed


# ── Phần cào trang ─────────────────────────────────────────────────────────────

def _extract_urls_from_js_text(text: str, base_url: str, seen: set, urls: list):
    """Trích xuất URL ảnh từ nội dung JS/JSON (hỗ trợ trang lazy-load qua script)."""
    # 1. Tìm JSON array kiểu: "page_url":["url1","url2",...] hoặc "images":[...]
    for pattern in [
        r'"(?:page_url|images?|imgs?|photo(?:s|_list)?|pic(?:s|_list)?|src_list)"\s*:\s*(\[.*?\])',
        r"'(?:page_url|images?|imgs?|photo(?:s|_list)?|pic(?:s|_list)?|src_list)'\s*:\s*(\[.*?\])",
    ]:
        for m in re.finditer(pattern, text, re.DOTALL | re.IGNORECASE):
            try:
                arr = json.loads(m.group(1))
                for item in arr:
                    if isinstance(item, str):
                        raw = item.strip()
                        if not raw:
                            continue
                        full = urljoin(base_url, raw)
                        if full not in seen and is_image_url(full):
                            seen.add(full)
                            urls.append(full)
            except (json.JSONDecodeError, ValueError):
                pass

    # 2. Tìm URL ảnh trực tiếp trong chuỗi JS (quoted strings)
    for m in re.finditer(r'["\']((https?://[^"\']+\.(?:jpg|jpeg|png|gif|webp|bmp|avif)))["\']',
                         text, re.IGNORECASE):
        full = m.group(1)
        if full not in seen:
            seen.add(full)
            urls.append(full)


def extract_image_urls(html: str, base_url: str, session: "requests.Session | None" = None) -> list[str]:
    """Trích xuất tất cả URL ảnh từ HTML, bao gồm lazy-load và JS data files."""
    soup = BeautifulSoup(html, "lxml")
    seen: set[str] = set()
    urls: list[str] = []

    # ── 1. Thẻ <img> và <source> ─────────────────────────────────────────────
    for tag in soup.find_all(["img", "source"]):
        for attr in SRC_ATTRS:
            raw = tag.get(attr, "").strip()
            if not raw or raw.startswith("data:"):
                continue
            full = urljoin(base_url, raw)
            if full not in seen and (is_image_url(full) or "image" in tag.get("type", "")):
                seen.add(full)
                urls.append(full)

    # ── 2. Thuộc tính srcset ─────────────────────────────────────────────────
    for tag in soup.find_all(["img", "source"]):
        srcset = tag.get("srcset", "")
        for part in srcset.split(","):
            raw = part.strip().split()[0] if part.strip() else ""
            if not raw or raw.startswith("data:"):
                continue
            full = urljoin(base_url, raw)
            if full not in seen and is_image_url(full):
                seen.add(full)
                urls.append(full)

    # ── 3. Nội dung inline <script> ─────────────────────────────────────────
    for script in soup.find_all("script", src=False):
        if script.string:
            _extract_urls_from_js_text(script.string, base_url, seen, urls)

    # ── 4. External <script src="..."> trên cùng domain (JS data files) ──────
    parsed_base = urlparse(base_url)
    if session is None:
        session = requests.Session()
        session.headers.update(DEFAULT_HEADERS)

    for script in soup.find_all("script", src=True):
        src = script.get("src", "").strip()
        if not src:
            continue
        full_src = urljoin(base_url, src)
        # Chỉ fetch script cùng domain
        if urlparse(full_src).netloc != parsed_base.netloc:
            continue
        try:
            r = session.get(full_src, timeout=15)
            r.raise_for_status()
            _extract_urls_from_js_text(r.text, base_url, seen, urls)
        except Exception:
            pass

    return urls


def make_session(referer: str = "") -> requests.Session:
    s = requests.Session()
    s.headers.update(DEFAULT_HEADERS)
    if referer:
        s.headers["Referer"] = referer
    return s


def fetch_page(url: str, timeout: int = 20,
               session: "requests.Session | None" = None) -> str:
    """Tải HTML của trang."""
    if session is None:
        session = make_session(referer=url)
    resp = session.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def crawl(url: str, output_folder: str, delay: float = 0.3,
          max_workers: int = 4, timeout: int = 20,
          on_progress=None, on_log=None,
          stop_event: threading.Event | None = None) -> tuple[int, int]:
    """
    Crawl ảnh từ `url` → lưu vào `output_folder`.
    Trả về (số ảnh thành công, số ảnh lỗi).
    """
    log = on_log or print
    session = make_session(referer=url)
    log(f"Đang tải trang: {url}")
    html = fetch_page(url, timeout=timeout, session=session)
    image_urls = extract_image_urls(html, url, session=session)

    if not image_urls:
        log("Không tìm thấy ảnh nào trên trang này.")
        return 0, 0

    log(f"Tìm thấy {len(image_urls)} ảnh. Bắt đầu tải xuống…")
    downloader = ImageDownloader(
        output_folder=output_folder,
        delay=delay,
        max_workers=max_workers,
        timeout=timeout,
        on_progress=on_progress,
        on_log=log,
        stop_event=stop_event,
    )
    return downloader.download_all(image_urls, referer=url)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Crawl ảnh từ một trang web và lưu vào folder chỉ định."
    )
    parser.add_argument("url", help="URL trang web cần crawl")
    parser.add_argument("output", help="Thư mục lưu ảnh (sẽ được tạo nếu chưa có)")
    parser.add_argument("--delay", type=float, default=0.3,
                        help="Thời gian chờ giữa các request (giây, mặc định 0.3)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Số luồng tải đồng thời (mặc định 4)")
    parser.add_argument("--timeout", type=int, default=20,
                        help="Timeout mỗi request (giây, mặc định 20)")
    args = parser.parse_args()

    ok, fail = crawl(
        url=args.url,
        output_folder=args.output,
        delay=args.delay,
        max_workers=args.workers,
        timeout=args.timeout,
    )
    print(f"\nHoàn thành: {ok} thành công, {fail} thất bại.")
    print(f"Ảnh được lưu tại: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
