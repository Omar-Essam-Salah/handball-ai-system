@echo off
title Handball Studio
color 0B
pushd "%~dp0"

set "PYEXE=.venv\Scripts\python.exe"
if not exist "%PYEXE%" set "PYEXE=python"

cls
echo  ================================================================
echo      HANDBALL STUDIO            (leave this window open)
echo  ================================================================
echo.
echo   - The match opens in your browser on THIS PC automatically.
echo   - On your PHONE (same WiFi): open the app and tap "Auto-find",
echo     or type the http://...:8000 address shown below.
echo.
echo   Starting the AI engine (first start takes ~15 seconds)...
echo  ----------------------------------------------------------------
echo.

REM free port 8000 from any previous run
for /f "tokens=5" %%P in ('netstat -ano ^| findstr "LISTENING" ^| findstr ":8000"') do taskkill /F /PID %%P >nul 2>&1

REM open the PC browser once the server is up
start "" cmd /c "timeout /t 15 /nobreak >nul & start "" http://localhost:8000"

"%PYEXE%" src\mobile_server.py --source handball.mp4 --port 8000

echo.
echo  [stopped]  press any key to close.
pause >nul
popd
