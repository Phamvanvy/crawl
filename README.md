# 🕷️ Image Crawler & Manga Translator

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)

> **Image Crawler** — Công cụ tải ảnh từ web + dịch manga (Trung/Anh → Việt) với OCR và model AI.

Một ứng dụng web Python kết hợp 4 chức năng chính:
1. 🕷️ **Crawl ảnh từ web** — Tự động phát hiện và tải ảnh từ bất kỳ trang web nào
2. ⬇️ **Download hàng loạt** — Multi-thread download (4 luồng), retry với backoff, thread-safe file naming
3. 🌐 **Dịch manga** — Dịch text Trung/Anh → Việt trong ảnh với OCR + Ollama hoặc `manga-image-translator`
4. 👁️ **Xem ảnh** — Lightbox viewer trực tiếp trên trình duyệt

---

## Yêu cầu hệ thống

- Python **3.10–3.12** (khuyên dùng 3.12 cho web app)
- Python **3.11** riêng để chạy `manga-image-translator` (nếu dùng MIT backend)
- Windows / Linux / macOS
- *(Tuỳ chọn)* GPU NVIDIA — tăng tốc OCR ~5×, bắt buộc cho LAMA inpainting

---

## Cài đặt nhanh

```powershell
# 1. Tạo virtual environment cho web app
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

# 2. Cài dependencies cơ bản
pip install -r requirements.txt

# 3. Chạy server
python web_app.py
# hoặc double-click start.bat (Windows)
```

Trình duyệt sẽ tự mở tại `http://127.0.0.1:5000`.

---

## Cài đặt đầy đủ (tất cả tính năng)

### Bước 1 — Cài web app và OCR + Ollama backend

```powershell
# Tạo và kích hoạt venv (Python 3.12 hoặc 3.11)
python -m venv .venv
.venv\Scripts\activate

# Cài tất cả dependencies bao gồm PyTorch, OCR, Flask
python setup_translator.py
# Hoặc thủ công:
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install easyocr paddlepaddle paddleocr opencv-python
```

### Bước 2 — Cài Ollama

1. Tải và cài Ollama từ [ollama.com](https://ollama.com)
2. Kéo model về (chọn 1):

```powershell
ollama pull qwen2.5:7b          # ~4.7 GB, cân bằng tốt/nhanh
ollama pull qwen3:14b           # ~9 GB, chất lượng cao (cần 12+ GB VRAM)
ollama pull qwen2.5:32b-q4_K_M  # ~20 GB, chất lượng tốt nhất
```

3. Đảm bảo Ollama đang chạy (tray icon hoặc `ollama serve`)

### Bước 3 — Cài manga-image-translator backend (tuỳ chọn, chất lượng cao hơn)

Backend này yêu cầu **Python 3.11** riêng và phải cài vào `mit_venv` riêng biệt.

> ⚠️ **KHÔNG dùng `pip install git+https://...`** — lệnh đó bỏ sót toàn bộ subpackages (`rendering/`, `translators/`...).  
> Phải clone repo rồi cài editable (`-e`).

Chạy từng lệnh từ thư mục gốc project (`e:\repos\crawl`):

```powershell
# 1. Tải Python 3.11 từ https://python.org nếu chưa có (bật "Add to PATH")
py -3.11 -m venv mit_venv

# 2. Nâng cấp pip
.\mit_venv\Scripts\python.exe -m pip install --upgrade pip

# 3. Cài PyTorch CUDA 12.4 (RTX 30/40/50 series)
.\mit_venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# Không có GPU:
# .\mit_venv\Scripts\python.exe -m pip install torch torchvision

# 4. Clone source manga-image-translator (PHẢI dùng clone, không pip install git+)
git clone https://github.com/zyddnys/manga-image-translator.git tmp_repo

# 5. Cài dependencies từ repo
.\mit_venv\Scripts\python.exe -m pip install -r tmp_repo\requirements.txt

# 6. Cài package ở chế độ editable (bao gồm đầy đủ subpackages)
.\mit_venv\Scripts\python.exe -m pip install -e tmp_repo

# 7. Áp dụng các patch tùy chỉnh cho tiếng Việt
python apply_patches.py
```

**Kiểm tra sau khi cài:**
```powershell
.\mit_venv\Scripts\python.exe -c "import manga_translator; print('OK')"
# Kết quả mong đợi: OK
```

> **Lưu ý:** Mỗi lần cập nhật MIT, chạy lại `python apply_patches.py`.  
> `mit_venv` phải nằm trong cùng thư mục với `web_app.py`.

---

## Cài đặt một lệnh (one-liner PowerShell)

```powershell
py -3.11 -m venv mit_venv ; .\mit_venv\Scripts\python.exe -m pip install --upgrade pip ; .\mit_venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 ; git clone https://github.com/zyddnys/manga-image-translator.git tmp_repo ; .\mit_venv\Scripts\python.exe -m pip install -r tmp_repo\requirements.txt ; .\mit_venv\Scripts\python.exe -m pip install -e tmp_repo ; python apply_patches.py
```

---

## Patches tùy chỉnh (`apply_patches.py`)

`apply_patches.py` copy 3 file patch vào `manga_translator` bên trong `mit_venv`:

| File patch | Nội dung |
|---|---|
| `rendering/__init__.py` | Giảm font size thay vì mở rộng bbox khi text VI dài hơn TQ — ngăn tràn ra ngoài bong bóng |
| `rendering/text_render.py` | Guard empty `line_width_list` (tránh crash với ký tự ZWJ) |
| `translators/custom_openai.py` | Custom OpenAI translator, watermark detection → ZWJ fallback, retry 10 lần nếu output không phải Việt |

---

## Sử dụng

### 1. Crawl ảnh
1. Mở tab **🕷️ Crawl ảnh**
2. Nhập URL trang chứa ảnh
3. Chọn thư mục lưu (nút 📁)
4. Chọn preset hoặc tùy chỉnh delay/workers/timeout
5. Bấm **▶ Bắt đầu**

### 2. Dịch manga (Trung/Anh → Việt)
1. Mở tab **🌐 Dịch ảnh**
2. Bấm **🔍 Kiểm tra** — đảm bảo các mục cần thiết ✅
3. Chọn **Thư mục nguồn** (ảnh cần dịch)
4. Chọn **Backend**:
   - **OCR + Ollama**: cần EasyOCR/PaddleOCR + Ollama đang chạy
   - **manga-image-translator**: cần `mit_venv` đã cài (chất lượng tốt hơn)
5. Bấm **▶ Bắt đầu dịch**

### 3. Xem ảnh
1. Mở tab **👁️ Viewer**
2. Nhập đường dẫn thư mục
3. Click ảnh để mở lightbox

---

## Cấu trúc project

```
crawl/
├── web_app.py              — Flask server + tất cả API routes
├── crawler.py              — Logic crawl và tải ảnh
├── translator_engine.py    — Shim tương thích (import từ package bên dưới)
├── translator_engine_pkg/  — Package dịch ảnh
│   ├── _ocr.py             — PaddleOCR/EasyOCR, bubble detection
│   ├── _utils.py           — Helpers: bbox, watermark, LAMA check
│   ├── _inpaint.py         — inpaint_region (OpenCV / LAMA)
│   ├── _translate.py       — Ollama API helpers, translate_batch
│   ├── _render.py          — Font loading, text rendering
│   ├── _image_translator.py — ImageTranslator class (Ollama backend)
│   └── _mit_backend.py     — MITImageTranslator class (MIT backend)
├── patches/                — Patch files cho manga-image-translator
│   ├── manga_translator_rendering_init.py     — Fix font sizing VI
│   ├── manga_translator_rendering_text_render.py — Guard ZWJ crash
│   └── manga_translator_translators_custom_openai.py — Custom translator
├── apply_patches.py        — Script áp dụng patches vào mit_venv
├── gpt_config_vi.yaml      — Config custom_openai translator
├── setup_translator.py     — Cài AI dependencies tự động
├── requirements.txt        — Dependencies cơ bản (Flask, requests, ...)
├── start.bat               — Khởi động nhanh (Windows)
├── templates/              — Giao diện web (HTML/JS/CSS)
│   ├── index.html
│   └── partials/           — Partial templates theo tab/chức năng
├── fonts/                  — Font tiếng Việt (MTO, BeVietnamPro, ...)
└── models/                 — Model detector (PaddleOCR)
```

---

## GPU & hiệu năng

| Cấu hình | OCR (mỗi ảnh) | Dịch (mỗi ảnh) |
|---------|--------------|----------------|
| RTX 5060 Ti 16 GB | <0.5s | ~2–5s |
| RTX 4070+ / 3080+ (12–16 GB) | ~0.5–1s | ~3–8s |
| GTX 1660 / RTX 3060 (6–8 GB) | ~1–2s | ~5–15s |
| CPU only | ~5–10s | ~5–15s |

---

## License

MIT
