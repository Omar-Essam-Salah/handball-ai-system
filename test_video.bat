@echo off
setlocal enabledelayedexpansion
title Handball AI - Smooth and Accurate Mode
color 0B
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
echo    HANDBALL AI  -  Smooth and High-Memory Tracking Mode
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

echo.
echo  Press Q in the video window to quit.
echo.

:: Phase-1 perf: infer-every 2 + ROI ball detector + velocity interp.
:: لو الـ FPS لسه قليل، جرّب --infer-every 3.
"%PYEXE%" src\pipeline.py ^
    --source "!VIDEO!" ^
    --device !DEVICE! ^
    --model "!MODEL!" ^
    --imgsz 640 ^
    --conf 0.40 ^
    --ball-conf 0.03 ^
    --infer-every 2 ^
    --display-every 1 ^
    --scale 0.7 ^
    --debug ^
    --no-server

echo.
echo  ----------------------------------------------------------------
echo   In-window keys:
echo     Q - quit       D - toggle debug overlay
echo     A - manual goal Home    B - manual goal Away
echo     R - recalibrate colors  F - flip sides
echo  ----------------------------------------------------------------

echo.
echo [Done] Press any key to close.
pause
popd