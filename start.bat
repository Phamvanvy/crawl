@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat

REM Ollama custom_openai cho manga-image-translator (chọn trong dropdown "Custom OpenAI / Ollama")
REM   qwen3-nothink   = 8b  (~5.2GB VRAM)  - nhanh
REM   qwen3-14b-nothink = 14b (~9.3GB VRAM) - thông minh hơn (khuyến nghị 16GB+)
set CUSTOM_OPENAI_API_BASE=http://localhost:11434/v1
set CUSTOM_OPENAI_MODEL=qwen3-14b-nothink

python web_app.py
pause
