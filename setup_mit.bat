@echo off

cd /d "%~dp0"

title Cai dat manga-image-translator

color 0A

echo.

echo ============================================================

echo   CAI DAT manga-image-translator (MIT backend)

echo ============================================================

echo.

echo [1/4] Kiem tra Python 3.11...

py -3.11 --version >nul 2>&1

if errorlevel 1 goto nopy311


echo.

goto step2

:nopy311

color 0C

echo.

echo   [LOI] Khong tim thay Python 3.11!

echo.

echo   Tai Python 3.11 (Windows 64-bit installer):

echo   https://www.python.org/downloads/release/python-3119/

echo.

echo   QUAN TRONG khi cai dat:

echo   - BO TICH "Add Python 3.11 to PATH"

echo     (de khong ghi de Python 3.12 dang dung)

echo   - Giu nguyen cac o khac, chon Install Now

echo.

echo   Sau khi cai xong, dong cua so va double-click lai setup_mit.bat

echo.

pause

exit /b 1

:step2

echo [2/4] Tao virtual environment mit_venv...

if exist "mit_venv\Scripts\python.exe" (

    echo   mit_venv da ton tai, bo qua.

) else (

    py -3.11 -m venv mit_venv

    if errorlevel 1 goto errvenv

    echo   Da tao mit_venv thanh cong.

)

echo.

:step3

echo [3/4] Nang cap pip...

mit_venv\Scripts\python -m pip install --upgrade pip --quiet

echo   pip da cap nhat.

echo.

:step4

echo [4/4] Cai PyTorch CUDA 12.1 + manga-image-translator...

echo   Co the mat 10-30 phut, ~2-5 GB

echo   Binh thuong neu thay nhieu dong chu cuon - dang tai...

echo.

mit_venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet

if errorlevel 1 (

    echo   [CANH BAO] PyTorch CUDA that bai, cai CPU-only...

    mit_venv\Scripts\pip install torch torchvision --quiet

)

mit_venv\Scripts\pip install "git+https://github.com/zyddnys/manga-image-translator.git" --quiet

if errorlevel 1 goto errinstall

echo.

echo Kiem tra import...

mit_venv\Scripts\python -c "import manga_translator; print(\"  [OK] Import thanh cong!\")"

if errorlevel 1 goto errimport

color 0A

echo.

echo ============================================================

echo   CAI DAT HOAN TAT!

echo.

echo   Tiep theo:

echo   1. Chay start.bat de mo ung dung

echo   2. Tab Dich anh -^> bam Kiem tra

echo   3. O MIT phai hien mau xanh

echo ============================================================

echo.

pause

exit /b 0

:errvenv

color 0C

echo   [LOI] Khong tao duoc mit_venv!

echo.

pause

exit /b 1

:errinstall

color 0C

echo.

echo   [LOI] Cai that bai! Kiem tra mang va thu lai.

echo.

pause

exit /b 1

:errimport

color 0C

echo   [LOI] Import that bai. Thu chay lai file nay.

echo.

pause

exit /b 1

