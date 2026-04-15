Project Map
src/models/: Chứa các định dạng dữ liệu (Database schema).
src/services/: Nơi xử lý logic nghiệp vụ chính.
src/controllers/: Nơi tiếp nhận request và gọi service.
Data Flow
User -> Controller -> Service -> Model -> Database
Use code with caution.
# Project Map
- `crawler.py`: Chứa logic crawl trang web và tải ảnh.
- `translator_engine.py`: Xử lý pipeline OCR → inpaint → dịch → render ảnh.
- `web_app.py`: Flask server, routes API và giao diện web.
- `templates/`: Cấu trúc giao diện HTML/JS/CSS cho web app.
- `project_index.py`: Tạo index local và knowledge graph để truy vấn nội bộ.
- `repo_index_instructions.md`: Hướng dẫn dùng index và prompt cho AI.

# Data Flow
User -> Web UI -> Flask route -> business logic -> crawler/translator -> filesystem
Use code with caution.
