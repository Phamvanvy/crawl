# 🕷️ Image Crawler & Manga Translator

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)

> **Image Crawler** — Công cụ tải ảnh từ web + dịch manga (Trung → Việt) với OCR và model AI.

Một ứng dụng web Python kết hợp 4 chức năng chính:
1. 🕷️ **Crawl ảnh từ web** — Tự động phát hiện và tải ảnh từ bất kỳ trang web nào
2. ⬇️ **Download hàng loạt** — Multi-thread download (4 luồng), retry với backoff, thread-safe file naming
3. 🌐 **Dịch manga** — Dịch text Trung/Anh → Việt trong ảnh với OCR + Ollama hoặc `manga-image-translator`
4. 👁️ **Xem ảnh** — Lightbox viewer trực tiếp trên trình duyệt

---

---

## Yêu cầu hệ thống

- Python **3.10–3.11** cho backend **manga-image-translator**, hoặc Python **3.12** nếu chỉ dùng OCR+Ollama
- Windows / Linux / macOS
- *(Tùy chọn)* GPU NVIDIA để tăng tốc OCR (~5×)

---

## Cài đặt nhanh

```bash
# 1. Tạo virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

# 2. Cài dependencies cơ bản
pip install -r requirements.txt

# 3. Chạy server
start.bat          # Windows (double-click hoặc chạy trong CMD)
# hoặc
python web_app.py  # nếu đã activate venv
```

> **Lưu ý Windows:** Luôn chạy qua `start.bat` hoặc activate venv trước (`d:\repos\crawl\.venv\Scripts\activate`) — nếu không, Python system sẽ không tìm thấy `cv2`, `flask`, v.v.

Trình duyệt sẽ tự mở tại `http://127.0.0.1:5000`.

---

## Tính năng

### 🕷 Crawl ảnh
- Nhập URL trang web → tự động phát hiện và tải tất cả ảnh
- Cấu hình: delay, số luồng tải đồng thời, timeout
- Nút **Dừng** — dừng ngay lập tức, không tải thêm ảnh mới
- Lưu preset cấu hình (Nhanh / An toàn / Nặng / Lười biếng)

### 🌐 Dịch ảnh (ZH → VI / EN → VI)

Hỗ trợ 2 backend:

| Backend | Chất lượng | Cài đặt |
|---------|-----------|---------|
| **OCR + Ollama** | Tốt | Cần EasyOCR/PaddleOCR + Ollama |
| **manga-image-translator** | Tốt hơn (inpaint + render chuyên dụng) | Cần Python 3.11 riêng |

- Xử lý hàng loạt cả thư mục
- Preview ảnh kết quả ngay trong trình duyệt

---

## Cài đặt tính năng dịch ảnh

## 🕷️ Crawl & Download Ảnh

### Tính năng chính
| # | Chức năng | Mô tả |
|-|-|-|
| 1 | **Trích xuất URL ảnh** | Tự động parse `<img>`, `<source>`, `srcset`, lazy-load (data-lazy, data-src), JSON trong script |
| 2 | **Download hàng loạt** | Multi-thread với 4 luồng, semaphore-controlled, retry max 3 lần với exponential backoff |
| 3 | **Thread-safe naming** | Lock + `_claimed` set để tránh file conflict khi nhiều threads cùng ghi |
| 4 | **Wnacg support** | Tự động crawl trang tìm kiếm, phát hiện series prefix cho folder structure |

### CLI Usage
```bash
python crawler.py <url> <output_folder> \
    --delay 0.3   # Thời gian delay giữa requests (giây)
    --workers 4   # Số luồng song song (mặc định: 4)
    --timeout 20  # Timeout mỗi request (giây, mặc định: 20)
```

### API Usage
```python
from crawler import crawl

ok, fail, failed = crawl(
    url="https://example.com/gallery",
    output_folder="./output",
    delay=0.3,
    max_workers=4,
    timeout=20,
)
# ok: số ảnh thành công, fail: số ảnh lỗi
```

### Structure
```
crawler.py (690 dòng)
├── ImageDownloader class          — Multi-thread download với semaphore
├── extract_image_urls()           — Parse HTML + JS cho lazy-load
├── _crawl_single_page()           — Crawl một trang đơn
└── crawl()                        — Entry point, tự động detect wnacg search
```

---

#### Bước 1 — Cài dependencies AI

```bash
python setup_translator.py
```

Script sẽ cài: **PyTorch (CUDA)**, **EasyOCR**, **OpenCV**, font **NotoSans**.

Hoặc cài thủ công:

```bash
# PyTorch với CUDA 12.4 (tương thích RTX 30/40/50 series)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# RTX 50 series (Blackwell): nếu có lỗi CUDA, dùng nightly:
# pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# OCR + xử lý ảnh
pip install easyocr opencv-python-headless
```

#### Bước 2 — Cài Ollama + model

1. Tải và cài Ollama từ [ollama.com](https://ollama.com)
2. Kéo model về:

```bash
ollama pull qwen2.5:7b       # ~4.7 GB, cân bằng tốt/nhanh
ollama pull qwen3:14b        # ~9 GB, chất lượng cao (cần 12+ GB VRAM)
# GPU 16 GB: nên dùng qwen3:14b hoặc qwen2.5:32b-q4
ollama pull qwen2.5:32b-q4_K_M  # ~20 GB RAM (có thể dùng với 16 GB VRAM + RAM swap)
```

3. Đảm bảo Ollama đang chạy (tray icon hoặc `ollama serve`)

---

## 🌐 Dịch Ảnh (Text Trung → Việt)

### 2 Backend hỗ trợ

| Backend | Chất lượng | Yêu cầu |
|---------|-----------|---------|
| **OCR + Ollama** | Tốt | EasyOCR/PaddleOCR + Ollama với model Qwen |
| **manga-image-translator** | Tốt hơn | Inpaint + render chuyên dụng, cần Python 3.11 riêng |

### OCR + Ollama Backend

Cài dependencies AI:
```bash
python setup_translator.py
```

Hoặc thủ công:
```bash
# PyTorch với CUDA 12.4 (RTX 30/40/50 series)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# OCR + xử lý ảnh
pip install easyocr opencv-python-headless
```

**Cài model Ollama:**
```bash
ollama pull qwen2.5:7b       # ~4.7 GB, cân bằng tốt/nhanh
ollama pull qwen3:14b        # ~9 GB, chất lượng cao (cần 12+ GB VRAM)
```

### manga-image-translator Backend

Cần **Python 3.10 hoặc 3.11**. Tạo venv riêng trong project:
```bash
py -3.11 -m venv mit_venv
mit_venv\Scripts\pip install git+https://github.com/zyddnys/manga-image-translator.git

# Áp dụng patches tùy chỉnh
python apply_patches.py
```

`apply_patches.py` sẽ copy các patch vào `manga_translator`:
- Fix Vietnamese text rendering (word-wrap với `target_font_size`)
- Guard empty `line_width_list` trong `text_render.py`
- Custom OpenAI translator với watermark detection → ZWJ fallback

---

manga-image-translator yêu cầu **Python 3.10 hoặc 3.11** (không tương thích Python 3.12+).  
Cần tạo một venv riêng bên cạnh venv chính:

```bat
REM 1. Cài Python 3.11 từ https://python.org (tích "Add to PATH")

REM 2. Tạo venv riêng tên "mit_venv" trong thư mục project
py -3.11 -m venv mit_venv

REM 3. Cài manga-image-translator vào venv đó
mit_venv\Scripts\pip install git+https://github.com/zyddnys/manga-image-translator.git
```

Ứng dụng sẽ tự phát hiện `mit_venv\Scripts\python.exe` khi bấm **🔍 Kiểm tra** trong tab Dịch ảnh.

> **Lưu ý:** `mit_venv` phải nằm trong cùng thư mục với `web_app.py`. Không cần thay đổi venv chính dùng để chạy `web_app.py`.

---

## ⚙️ Cấu hình & Patching

### Kiểm tra hệ thống

Trong tab **Dịch ảnh**, bấm **🔍 Kiểm tra** để xem trạng thái:

| Mục | Trạng thái | Hành động |
|-----|-----------|-----------|
| PyTorch | ❌ | `pip install torch` |
| CUDA | ❌ No GPU | CPU vẫn hoạt động, chậm hơn |
| EasyOCR | ❌ | `pip install easyocr` |
| Ollama | ❌ not running | Mở app Ollama |
| MIT (manga-image-translator) | ❌ | Xem hướng dẫn bên dưới |

---

### Patching manga_translator

Tự động áp dụng các patches tùy chỉnh:
```bash
python apply_patches.py
```

Các patch được apply tự động vào `manga_translator`:
1. **rendering/__init__.py** — Fix word-wrap với Vietnamese text dài
2. **rendering/text_render.py** — Guard empty line_width_list
3. **translators/custom_openai.py** — Custom translator với watermark detection

---

Trong tab **Dịch ảnh**, bấm **🔍 Kiểm tra** để xem trạng thái:

| Mục | Trạng thái | Hành động |
|-----|-----------|-----------|
| PyTorch | ❌ | `pip install torch` |
| CUDA | ❌ No GPU | CPU vẫn hoạt động, chậm hơn |
| EasyOCR | ❌ | `pip install easyocr` |
| Ollama | ❌ not running | Mở app Ollama |
| MIT | ❌ | Xem hướng dẫn cài `mit_venv` bên trên |

---

## Cấu trúc project

```
crawl/
├── web_app.py              — Flask server + tất cả API routes
├── crawler.py              — Logic crawl và tải ảnh
├── translator_engine.py    — Shim tương thích (import từ package bên dưới)
├── translator_engine_pkg/  — Package dịch ảnh (các module < 1000 dòng)
│   ├── __init__.py         — Khởi tạo, re-export toàn bộ public API
│   ├── _ocr.py             — PaddleOCR/EasyOCR, bubble detection
│   ├── _utils.py           — Helpers: bbox, watermark, LAMA check
│   ├── _inpaint.py         — inpaint_region (OpenCV / LAMA)
│   ├── _translate.py       — Ollama API helpers, translate_batch
│   ├── _render.py          — Font loading, text rendering
│   ├── _image_translator.py — ImageTranslator class (Ollama backend)
│   └── _mit_backend.py     — MITImageTranslator class (manga-image-translator)
├── gpt_config_vi.yaml      — Cấu hình custom_openai cho manga-image-translator
├── setup_translator.py     — Script cài AI dependencies
├── requirements.txt        — Dependencies cơ bản
├── start.bat               — Script khởi động nhanh (Windows)
├── templates/
│   ├── index.html          — Giao diện web (single-page app)
│   └── partials/
│       ├── _styles.html         — Aggregator CSS (include 3 sub-files)
│       ├── _styles_base.html    — CSS variables, reset, layout
│       ├── _styles_components.html — Buttons, progress, log panel
│       ├── _styles_app.html     — Lightbox, viewer, folder browser
│       ├── _scripts.html        — Aggregator JS (include 6 sub-files)
│       ├── _scripts_i18n.html   — i18n translations + applyLang
│       ├── _scripts_utils.html  — Presets, config, URL tag helpers
│       ├── _scripts_crawl.html  — Crawl tab logic + folder browser
│       ├── _scripts_translate.html — Translate tab logic
│       ├── _scripts_lightbox.html  — Lightbox viewer
│       ├── _scripts_viewer.html    — Image viewer tab
│       ├── _tab_crawl.html      — HTML tab crawl
│       ├── _tab_translate.html  — HTML tab dịch ảnh
│       ├── _tab_viewer.html     — HTML tab xem ảnh
│       └── _modals.html         — Modal dialogs
├── fonts/                  — Font tiếng Việt (BeVietnamPro)
└── models/                 — Model detector (PaddleOCR)
```

---

## 📖 Hướng dẫn sử dụng

### 1. Crawl ảnh từ web
1. Mở tab **🕷️ Crawl ảnh**
2. Nhập URL trang chứa ảnh
3. Chọn thư mục lưu (nút 📁)
4. Chọn preset hoặc tùy chỉnh delay/workers/timeout
5. Bấm **▶ Bắt đầu**

### 2. Dịch manga (Trung → Việt)
1. Mở tab **🌐 Dịch ảnh**
2. Bấm **🔍 Kiểm tra** — đảm bảo tất cả ✅
3. Chọn **Thư mục nguồn** (ảnh cần dịch)
4. Chọn **Backend**: OCR+Ollama hoặc manga-image-translator
5. Chọn model Ollama (nếu dùng OCR backend)
6. Bấm **▶ Bắt đầu dịch**

---

### Crawl ảnh

1. Mở tab **🕷 Crawl ảnh**
2. Nhập URL trang chứa ảnh
3. Chọn thư mục lưu (nút 📁)
4. Chọn preset hoặc tùy chỉnh cấu hình
5. Bấm **▶ Bắt đầu**

### Dịch ảnh

1. Mở tab **🌐 Dịch ảnh (ZH→VI)**
2. Bấm **🔍 Kiểm tra** — đảm bảo tất cả ✅
3. Chọn **Thư mục nguồn** (ảnh cần dịch)
4. Chọn **Thư mục xuất** (để trống → tự tạo `<tên>_vi/`)
5. Chọn model Ollama
6. Bấm **▶ Bắt đầu dịch**

---

## GPU & hiệu năng

| Cấu hình | OCR (mỗi ảnh) | Dịch (mỗi ảnh) |
|---------|--------------|----------------|
| **RTX 5060 Ti 16 GB (Blackwell)** | **<0.5s** | **~2–5s** |
| RTX 4070+ / 3080+ (12–16 GB) | ~0.5–1s | ~3–8s |
| GTX 1660 / RTX 3060 (6–8 GB) | ~1–2s | ~5–15s |
| CPU only | ~5–10s | ~5–15s |

Thắt cổ chai chính là Ollama — thời gian dịch phụ thuộc model và số vùng text.

**GPU 16 GB VRAM:** có thể chạy model tới **qwen3:14b** hoặc **qwen2.5:32b-q4** cho chất lượng dịch tốt nhất.

---

## Cấu trúc dự án

```
crawl/
├── web_app.py              — Flask server, routes API và giao diện
├── crawler.py             — Crawl trang web và tải ảnh
├── translator_engine.py   — Shim tương thích (import từ translator_engine_pkg)
├── translator_engine_pkg/ — Package dịch ảnh (các module < 1000 dòng)
│   ├── _ocr.py            — OCR engines, bubble detection
│   ├── _utils.py          — bbox/rect helpers, watermark, LAMA check
│   ├── _inpaint.py        — inpaint_region (OpenCV / LAMA)
│   ├── _translate.py      — Ollama API helpers, translate_batch
│   ├── _render.py         — Font loading, text rendering
│   ├── _image_translator.py — ImageTranslator (Ollama backend)
│   └── _mit_backend.py   — MITImageTranslator (manga-image-translator)
├── gpt_config_vi.yaml     — Cấu hình model, ngôn ngữ, inpainter
├── requirements.txt       — Dependencies
├── start.bat              — Script khởi động nhanh (Windows)
├── setup_translator.py    — Cài AI dependencies tự động
├── templates/             — HTML/JS/CSS giao diện web
│   └── partials/          — Các partial được include
│       ├── _styles*.html  — CSS phân theo nhóm (base/components/app)
│       └── _scripts*.html — JS phân theo tab (i18n/utils/crawl/translate/viewer)
├── fonts/                 — Font hỗ trợ tiếng Việt
└── models/                — Model detector (PaddleOCR)
```

---

## Roadmap

- [ ] Hỗ trợ dịch nhiều ngôn ngữ đầu ra (JP, KO, EN)
- [ ] Batch download + dịch từ danh sách URL
- [ ] Docker image
- [ ] Plugin trình duyệt (Chrome extension)
- [ ] LaMa inpainting mặc định khi GPU có đủ VRAM

---

## Đóng góp

Pull request luôn được chào đón! Để bắt đầu:

1. Fork repo
2. Tạo branch mới: `git checkout -b feature/ten-tinh-nang`
3. Commit thay đổi: `git commit -m "feat: mô tả ngắn"`
4. Push: `git push origin feature/ten-tinh-nang`
5. Mở Pull Request

> Xin hãy viết test (nếu có) và đảm bảo `python -m py_compile translator_engine_pkg/*.py` không lỗi.

---

## License

MIT
