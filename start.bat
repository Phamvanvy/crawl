@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python web_app.py
pause
