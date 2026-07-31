@echo off
title HANDBALL STUDIO
color 0B
pushd "%~dp0"

set "PYEXE=.venv\Scripts\python.exe"
if not exist "%PYEXE%" set "PYEXE=python"

cls
echo  ================================================================
echo         H A N D B A L L   S T U D I O      (final)
echo  ================================================================
echo.
echo    Full analysis program: live view, biomechanics, tactics,
echo    heat maps, shot/save reports, AI, and the phone app.
echo.
echo    - The dashboard opens in your browser on THIS PC.
echo    - On your PHONE (same WiFi): open the Handball app and tap
echo      Auto-find, or open the http://...:8000 address shown below.
echo.
echo    Starting the AI engine (about 15 seconds)...
echo  ----------------------------------------------------------------

REM free port 8000 from any previous run
for /f "tokens=5" %%P in ('netstat -ano ^| findstr "LISTENING" ^| findstr ":8000"') do taskkill /F /PID %%P >nul 2>&1

REM open the PC browser once the server is up
start "" cmd /c "timeout /t 15 /nobreak >nul & start "" http://localhost:8000"

"%PYEXE%" src\mobile_server.py --source handball.mp4 --port 8000

echo.
echo  [stopped]  press any key to close.
pause >nul
popd
