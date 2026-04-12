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

try:
    import curl_cffi.requests as _curl_requests
    _CURL_CFFI_AVAILABLE = True
except ImportError:
    _CURL_CFFI_AVAILABLE = False

# ── Cấu hình mặc định ──────────────────────────────────────────────────────────
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg", ".avif"}
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Ch-Ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Connection": "keep-alive",
    "Cache-Control": "max-age=0",
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
                 max_retries: int = 3,
                 on_progress=None, on_log=None,
                 stop_event: threading.Event | None = None):
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.delay = delay
        self.max_workers = max_workers
        self.timeout = timeout
        self.max_retries = max(0, int(max_retries))
        self.failed_items: list[dict] = []  # {"url": str, "index": int}
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
            last_exc: Exception | None = None
            for attempt in range(self.max_retries + 1):
                if self.stop_event.is_set():
                    return
                if attempt > 0:
                    wait = min(2 ** attempt, 30)
                    self.on_log(f"  [RETRY {attempt}/{self.max_retries}] {dest.name}  (chờ {wait}s…)")
                    time.sleep(wait)
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
                    suffix = f" (retry {attempt}x)" if attempt > 0 else ""
                    self.on_log(f"  [OK {index}/{total}] {dest.name}{suffix}")
                    time.sleep(self.delay)
                    return
                except requests.exceptions.HTTPError as exc:
                    status_code = exc.response.status_code if exc.response is not None else 0
                    # 4xx errors (except 429 Too Many Requests) are permanent — stop retrying
                    if status_code != 429 and 400 <= status_code < 500:
                        last_exc = exc
                        break
                    last_exc = exc
                except Exception as exc:
                    last_exc = exc
            # All attempts exhausted
            with self._lock:
                self.failed += 1
                done = self.success + self.failed
                self.on_progress(done, total)
                self.failed_items.append({
                    "url": url,
                    "index": index,
                    "folder": str(self.output_folder),
                    "referer": referer,
                })
            self.on_log(f"  [FAIL {index}/{total}] {url}  →  {last_exc}")
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
        return self.success, self.failed, self.failed_items


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


def make_wnacg_session(base_url: str, timeout: int = 20):
    """Tạo session cho wnacg dùng curl_cffi (Chrome TLS fingerprint) nếu có,
    ngược lại fallback về requests thường với headers đầy đủ."""
    parsed = urlparse(base_url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    if _CURL_CFFI_AVAILABLE:
        s = _curl_requests.Session(impersonate="chrome124")
        # Warm-up homepage để lấy cookies
        try:
            s.get(origin, timeout=timeout)
        except Exception:
            pass
        s.headers["Referer"] = origin
        return s
    # Fallback: requests thường
    s = requests.Session()
    s.headers.update(DEFAULT_HEADERS)
    try:
        s.get(origin, timeout=timeout)
    except Exception:
        pass
    s.headers["Referer"] = origin
    s.headers["Sec-Fetch-Site"] = "same-origin"
    return s


def fetch_page(url: str, timeout: int = 20,
               session: "requests.Session | None" = None) -> str:
    """Tải HTML của trang."""
    if session is None:
        session = make_session(referer=url)
    resp = session.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def _crawl_single_page(url: str, output_folder: str, delay: float = 0.3,
                       max_workers: int = 4, timeout: int = 20,
                       max_retries: int = 3,
                       on_progress=None, on_log=None,
                       stop_event: threading.Event | None = None,
                       session=None) -> tuple[int, int, list[dict]]:
    """
    Crawl ảnh từ một trang `url` → lưu vào `output_folder`.
    Trả về (số ảnh thành công, số ảnh lỗi, danh sách URL lỗi).
    """
    log = on_log or print
    if session is None:
        session = make_session(referer=url)
    log(f"Đang tải trang: {url}")
    html = fetch_page(url, timeout=timeout, session=session)
    image_urls = extract_image_urls(html, url, session=session)

    if not image_urls:
        log("Không tìm thấy ảnh nào trên trang này.")
        return 0, 0, []

    log(f"Tìm thấy {len(image_urls)} ảnh. Bắt đầu tải xuống…")
    downloader = ImageDownloader(
        output_folder=output_folder,
        delay=delay,
        max_workers=max_workers,
        timeout=timeout,
        max_retries=max_retries,
        on_progress=on_progress,
        on_log=log,
        stop_event=stop_event,
    )
    return downloader.download_all(image_urls, referer=url)


def retry_failed_downloads(
        items: list[dict], default_referer: str, default_output_folder: str,
        delay: float = 0.3, max_workers: int = 4,
        timeout: int = 20, max_retries: int = 3,
        on_progress=None, on_log=None,
        stop_event: threading.Event | None = None,
) -> tuple[int, int, list[dict]]:
    """Tải lại các URL bị lỗi, nhóm theo folder (mỗi gallery vào đúng subfolder).
    Mỗi item phải có 'url'; tùy chọn có 'folder' và 'referer'.
    """
    log = on_log or print
    log(f"↺  Replay {len(items)} ảnh lỗi…")

    # Nhóm theo (folder, referer)
    groups: dict[tuple[str, str], list[str]] = {}
    for item in items:
        folder  = item.get("folder")  or default_output_folder
        referer = item.get("referer") or default_referer
        key = (folder, referer)
        groups.setdefault(key, []).append(item["url"])

    total_ok = total_fail = 0
    all_failed: list[dict] = []
    cumulative = 0

    for (folder, referer), urls in groups.items():
        log(f"  Folder: {folder}  ({len(urls)} ảnh)")

        def _prog(done: int, total: int, _base: int = cumulative) -> None:
            if on_progress:
                on_progress(_base + done, _base + total)

        downloader = ImageDownloader(
            output_folder=folder,
            delay=delay,
            max_workers=max_workers,
            timeout=timeout,
            max_retries=max_retries,
            on_progress=_prog,
            on_log=log,
            stop_event=stop_event,
        )
        ok, fail, failed = downloader.download_all(urls, referer=referer)
        total_ok   += ok
        total_fail += fail
        all_failed.extend(failed)
        cumulative += ok

    return total_ok, total_fail, all_failed


# ── wnacg multi-page search crawler ───────────────────────────────────────────

_WNACG_NETLOCS = {"wnacg.com", "wnacg.ru", "wnacg01.link"}


def _is_wnacg_url(url: str) -> bool:
    """True nếu URL thuộc domain wnacg."""
    return urlparse(url).netloc in _WNACG_NETLOCS


def _is_wnacg_search(url: str) -> bool:
    """True nếu URL là trang tìm kiếm/danh sách của wnacg."""
    p = urlparse(url)
    if p.netloc not in _WNACG_NETLOCS:
        return False
    # Khớp cả /search/ và /search/index.php
    return p.path.rstrip("/") in ("/search", "/search/index.php") or \
           p.path.startswith("/search/")


def _wnacg_slide_url(url: str) -> str | None:
    """Chuyển photos-index-aid-ID.html → photos-slide-aid-ID.html.
    Nếu đã là slide URL, trả về nguyên."""
    m = re.search(r'(https?://[^/]+)/photos-(?:index|slide)-aid-(\d+)\.html', url)
    if not m:
        return None
    return f"{m.group(1)}/photos-slide-aid-{m.group(2)}.html"


def _wnacg_gallery_links(html: str, base_url: str) -> list[tuple[str, str]]:
    """Trả về [(title, index_url), ...] từ trang tìm kiếm wnacg (theo thứ tự HTML)."""
    soup = BeautifulSoup(html, "lxml")
    aid_data: dict[str, tuple[str, str]] = {}   # aid → (title, url)

    for a in soup.find_all("a", href=True):
        m = re.search(r'/photos-index-aid-(\d+)\.html', a["href"])
        if not m:
            continue
        aid = m.group(1)
        full_url = urljoin(base_url, a["href"])

        # Lấy text, bỏ alt text của <img> lồng bên trong
        for img in a.find_all("img"):
            img.decompose()
        text = a.get_text(" ", strip=True)

        # Ưu tiên anchor có text dài hơn (title link vs thumbnail link)
        if aid not in aid_data or len(text) > len(aid_data[aid][0]):
            title = sanitize_filename(text) if text else f"gallery_{aid}"
            aid_data[aid] = (title or f"gallery_{aid}", full_url)

    return list(aid_data.values())


def _wnacg_max_page(html: str) -> int:
    """Tìm số trang lớn nhất từ thanh phân trang wnacg."""
    soup = BeautifulSoup(html, "lxml")
    max_p = 1
    for a in soup.find_all("a", href=True):
        m = re.search(r'[?&]p=(\d+)', a["href"])
        if m:
            max_p = max(max_p, int(m.group(1)))
    return max_p


def _titles_common_prefix(titles: list[str]) -> str:
    """Tìm tiền tố chung dài nhất của danh sách chuỗi, cắt tại ranh giới từ."""
    if not titles:
        return ""
    prefix = titles[0]
    for s in titles[1:]:
        # Rút ngắn prefix cho đến khi s bắt đầu bằng prefix
        while prefix and not s.startswith(prefix):
            prefix = prefix[:-1]
        if not prefix:
            return ""
    return prefix.rstrip()


def crawl_search_listing(
        url: str, output_folder: str, delay: float = 0.3,
        max_workers: int = 4, timeout: int = 20, max_retries: int = 3,
        on_progress=None, on_log=None,
        stop_event: threading.Event | None = None,
) -> tuple[int, int, list[dict]]:
    """Crawl tất cả gallery từ trang tìm kiếm wnacg (tự động duyệt nhiều trang).

    Thứ tự crawl:
    - Trang: từ trang cao nhất → trang 1
    - Trong mỗi trang: từ gallery cuối → gallery đầu (dưới lên trên)

    Cấu trúc thư mục output:
        output_folder / <tên series chung> / <phần riêng của mỗi gallery>
    Ví dụ:  D:\\Comic / [ryota tanaka] 逆轉 / 1-6
    """
    log = on_log or print
    stop = stop_event or threading.Event()
    log(f"Đang phân tích trang tìm kiếm: {url}")
    session = make_wnacg_session(url, timeout=timeout)
    first_html = fetch_page(url, timeout=timeout, session=session)
    detected_max = _wnacg_max_page(first_html)

    # Số trang cao nhất = max(trang trong URL, tìm thấy từ HTML)
    parsed_qs: dict[str, str] = {}
    for part in urlparse(url).query.split("&"):
        if "=" in part:
            k, v = part.split("=", 1)
            parsed_qs[k] = v
    url_page = int(parsed_qs["p"]) if parsed_qs.get("p", "").isdigit() else 1
    max_page = max(url_page, detected_max)

    log(f"Tổng số trang phát hiện: {max_page}.  Thu thập danh sách gallery…")

    def make_page_url(p: int) -> str:
        if "p=" in url:
            return re.sub(r'([?&]p=)\d+', lambda mo: mo.group(1) + str(p), url)
        sep = "&" if "?" in url else "?"
        return f"{url}{sep}p={p}"

    # ── Phase 1: thu thập tất cả gallery trên mọi trang (high → low) ─────────
    pages_galleries: list[list[tuple[str, str]]] = []  # [(title, idx_url), ...]
    page_htmls: dict[int, str] = {url_page: first_html}

    for pg in range(max_page, 0, -1):
        if stop.is_set():
            break
        try:
            html = page_htmls.get(pg) or fetch_page(make_page_url(pg), timeout=timeout, session=session)
        except Exception as exc:
            log(f"  [ERR] Không tải được trang {pg}: {exc}")
            pages_galleries.append([])
            continue
        galleries = _wnacg_gallery_links(html, make_page_url(pg))
        galleries.reverse()  # dưới → trên
        pages_galleries.append(galleries)

    # ── Phase 2: tính series folder từ tiền tố chung ─────────────────────────
    all_titles = [title for galleries in pages_galleries for title, _ in galleries]
    prefix = _titles_common_prefix(all_titles)
    series_name = sanitize_filename(prefix) if prefix else ""

    if series_name:
        series_folder = Path(output_folder) / series_name
        log(f"Series folder: {series_folder}")
    else:
        series_folder = Path(output_folder)

    if all_titles:
        log(f"Tổng cộng {len(all_titles)} gallery.  Bắt đầu tải từ trang {max_page} → 1")

    # ── Phase 3: crawl từng gallery ──────────────────────────────────────────
    total_ok = total_fail = 0
    all_failed: list[dict] = []
    cumulative_ok = 0
    page_num = max_page

    for galleries in pages_galleries:
        if stop.is_set():
            break

        log(f"\n{'─' * 60}")
        log(f"📄 Trang {page_num}/{max_page}  ({len(galleries)} gallery)")
        page_num -= 1

        for title, idx_url in galleries:
            if stop.is_set():
                break

            slide_url = _wnacg_slide_url(idx_url)
            if not slide_url:
                log(f"  [SKIP] Không chuyển được sang slide URL: {idx_url}")
                continue

            # Tên subfolder = phần riêng sau prefix chung
            sub = title[len(prefix):].strip() if prefix else title
            sub = sanitize_filename(sub) or title
            dest = str(series_folder / sub)

            log(f"\n  ▶ {sub}  ({title})")
            log(f"     {slide_url}")

            def _prog(done: int, total: int, _base: int = cumulative_ok) -> None:
                if on_progress:
                    on_progress(_base + done, _base + total)

            ok, fail, failed = _crawl_single_page(
                url=slide_url,
                output_folder=dest,
                delay=delay,
                max_workers=max_workers,
                timeout=timeout,
                max_retries=max_retries,
                on_progress=_prog,
                on_log=log,
                stop_event=stop,
                session=session,
            )
            total_ok += ok
            total_fail += fail
            all_failed.extend(failed)
            cumulative_ok += ok

    log(f"\n{'═' * 60}")
    log(f"✔  Hoàn thành toàn bộ: {total_ok} ảnh thành công, {total_fail} thất bại.")
    return total_ok, total_fail, all_failed


def crawl(url: str, output_folder: str, delay: float = 0.3,
          max_workers: int = 4, timeout: int = 20,
          max_retries: int = 3,
          on_progress=None, on_log=None,
          stop_event: threading.Event | None = None) -> tuple[int, int, list[dict]]:
    """
    Entry point chính: crawl ảnh từ `url` → lưu vào `output_folder`.
    Tự động nhận diện URL tìm kiếm wnacg và xử lý nhiều trang.
    Trả về (số ảnh thành công, số ảnh lỗi, danh sách URL lỗi).
    """
    if _is_wnacg_search(url):
        return crawl_search_listing(
            url=url, output_folder=output_folder, delay=delay,
            max_workers=max_workers, timeout=timeout, max_retries=max_retries,
            on_progress=on_progress, on_log=on_log, stop_event=stop_event,
        )
    # URL wnacg đơn lẻ (slide/index page) cũng cần curl_cffi
    _session = make_wnacg_session(url, timeout=timeout) if _is_wnacg_url(url) else None
    return _crawl_single_page(
        url=url, output_folder=output_folder, delay=delay,
        max_workers=max_workers, timeout=timeout, max_retries=max_retries,
        on_progress=on_progress, on_log=on_log, stop_event=stop_event,
        session=_session,
    )


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

    ok, fail, _ = crawl(
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
