@echo off
echo ================================================
echo  Handball AI Coach - Offline Model Setup
echo  Run this ONCE to download all AI models.
echo  After this, everything works without internet.
echo ================================================
echo.

cd /d "%~dp0"

echo [1/3] Checking Python...
python --version
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.10+
    pause
    exit /b 1
)

echo.
echo [2/3] Installing required packages...
pip install ultralytics boxmot opencv-python numpy --quiet

echo.
echo [3/3] Downloading AI models...
python scripts/setup_models.py --all

echo.
echo ================================================
echo  Setup complete! Run live_match.bat to start.
echo ================================================
pause
