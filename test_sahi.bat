@echo off
setlocal enabledelayedexpansion
title Handball AI - SAHI Mode (better ball + distant-player recall)
color 0A
pushd "%~dp0"

set "PYEXE=.venv\Scripts\python.exe"
if not exist "%PYEXE%" (
    for /f "delims=" %%F in ('where python 2^>nul') do (
        if not defined PYEXE_FALLBACK set "PYEXE_FALLBACK=%%F"
    )
    if not defined PYEXE_FALLBACK (
        echo [ERROR] Python not found.
        pause & popd & exit /b 1
    )
    set "PYEXE=!PYEXE_FALLBACK!"
)

cls
echo  ================================================================
echo    HANDBALL AI  -  SAHI MODE
echo  ================================================================
echo    * Sliced inference: better ball + distant-player detection
echo    * ~5x slower than base mode — use for offline analysis
echo    * infer-every=4 to keep effective FPS playable
echo  ================================================================
echo.

if "%~1"=="" (
    set "VIDEO=hand3.mp4"
    if not exist "!VIDEO!" set "VIDEO="
    set /p "VIDEO=Video file [!VIDEO!]: "
) else (
    set "VIDEO=%~1"
)

set "DEVICE=cuda:0"
set "MODEL=yolov8s-pose.pt"
if exist "yolov8s-pose.engine" set "MODEL=yolov8s-pose.engine"

"%PYEXE%" src\pipeline.py ^
    --source "!VIDEO!" ^
    --device !DEVICE! ^
    --model "!MODEL!" ^
    --imgsz 640 ^
    --conf 0.40 ^
    --ball-conf 0.05 ^
    --infer-every 4 ^
    --display-every 2 ^
    --scale 0.7 ^
    --debug ^
    --use-sahi

echo.
echo  ----------------------------------------------------------------
echo   Done. Press any key to close.
pause
popd
