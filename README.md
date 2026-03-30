# Image Crawler & Translator

Ứng dụng web để **crawl ảnh** từ trang web và **dịch text Trung → Việt** trong ảnh (manhwa/manhua).

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

### Backend OCR + Ollama

#### Bước 1 — Cài dependencies AI

```bash
python setup_translator.py
```

Script sẽ cài: **PyTorch (CUDA)**, **EasyOCR**, **OpenCV**, font **NotoSans**.

Hoặc cài thủ công:

```bash
# PyTorch với CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# OCR + xử lý ảnh
pip install easyocr opencv-python-headless
```

#### Bước 2 — Cài Ollama + model

1. Tải và cài Ollama từ [ollama.com](https://ollama.com)
2. Kéo model về:

```bash
ollama pull qwen2.5:7b    # ~4.7 GB, chất lượng cao
# hoặc
ollama pull qwen2.5:3b    # ~2 GB, nhanh hơn
```

3. Đảm bảo Ollama đang chạy (tray icon hoặc `ollama serve`)

---

### Backend manga-image-translator

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

### Kiểm tra hệ thống

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
├── web_app.py            # Flask server + tất cả API routes
├── crawler.py            # Logic crawl và tải ảnh
├── translator_engine.py  # Pipeline OCR → inpaint → dịch → render
├── setup_translator.py   # Script cài AI dependencies
├── requirements.txt      # Dependencies cơ bản
├── templates/
│   └── index.html        # Giao diện web (single-page app)
├── fonts/                # Font NotoSans (tạo bởi setup_translator.py)
└── output_test/          # Thư mục output mẫu
```

---

## Sử dụng

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
| GPU (GTX 1660+) | ~1–2s | ~5–15s |
| CPU only | ~5–10s | ~5–15s |

Thắt cổ chai chính là Ollama — thời gian dịch phụ thuộc model và số vùng text.

---

## License

MIT
