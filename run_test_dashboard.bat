@echo off
setlocal enabledelayedexpansion
title Handball AI - Test Video with Dashboard
color 0E
pushd "%~dp0"

set "PYEXE=.venv\Scripts\python.exe"
if not exist "%PYEXE%" (
    for /f "delims=" %%F in ('where python 2^>nul') do (
        if not defined PYEXE_FALLBACK set "PYEXE_FALLBACK=%%F"
    )
    if not defined PYEXE_FALLBACK (
        echo [ERROR] Python not found.
        pause
        popd
        exit /b 1
    )
    set "PYEXE=!PYEXE_FALLBACK!"
)

cls
echo  ================================================================
echo    HANDBALL AI  -  Test Video + Dashboard  (Video loop enabled)
echo  ================================================================
echo.

if "%~1"=="" (
    set "VIDEO=handball.mp4"
    if not exist "!VIDEO!" set "VIDEO="
    set /p "VIDEO=Video file (drag here or type) [!VIDEO!]: "
) else (
    set "VIDEO=%~1"
)

set "DEVICE=cuda:0"
set "MODEL=yolov8s-pose.pt"
if exist "yolov8s-pose.engine" set "MODEL=yolov8s-pose.engine"

echo [1/2] Starting Dashboard...
start "Handball Dashboard" cmd /k ""%PYEXE%" -m streamlit run dashboard\app.py --server.port 8501 --server.headless true"
timeout /t 5 /nobreak >nul
start "" http://localhost:8501

echo [2/2] Starting AI Pipeline with Video (infinite loop)...
start "Handball Pipeline" cmd /k "for /l %%i in (1,1,1000) do ( "%PYEXE%" src\pipeline.py --source "!VIDEO!" --device !DEVICE! --model "!MODEL!" --imgsz 640 --conf 0.40 --ball-conf 0.05 --infer-every 2 --display-every 2 --scale 0.7 --debug && echo [Pipeline] Video ended, restarting... )"

echo.
echo  ================================================================
echo   System running. Dashboard live at http://localhost:8501
echo   Video will loop automatically.
echo  ================================================================
pause
popd