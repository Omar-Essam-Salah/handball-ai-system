@echo off
setlocal enabledelayedexpansion
title Handball AI - Court Calibration
color 0E
pushd "%~dp0"

REM ── locate Python (prefer project venv) ──────────────────────────────────
set "PYEXE=.venv\Scripts\python.exe"
if not exist "%PYEXE%" (
    for /f "delims=" %%F in ('where python 2^>nul') do (
        if not defined PYEXE_FALLBACK set "PYEXE_FALLBACK=%%F"
    )
    if not defined PYEXE_FALLBACK (
        echo [ERROR] Python not found. Install Python 3.10+ or create .venv first.
        pause
        popd
        exit /b 1
    )
    set "PYEXE=!PYEXE_FALLBACK!"
)

cls
echo  ================================================================
echo    HANDBALL AI  -  Pre-match Court Calibration
echo  ================================================================
echo.
echo  This tool grabs ONE frame from your source, lets you click the 4
echo  court corners (TL -^> TR -^> BR -^> BL), and saves the homography
echo  matrix to homography_matrix.npy so the live engine can render the
echo  bird's-eye minimap.
echo.
echo  Hotkeys:  Enter=save  U=undo  R=reset  ESC=cancel
echo.
echo  Examples:
echo    rtsp://admin:PASS@192.168.8.167:554/Streaming/Channels/102
echo    handball.mp4
echo.

set /p "SOURCE=Source (RTSP URL or video file): "
if "!SOURCE!"=="" (
    echo [ERROR] No source provided.
    pause
    popd
    exit /b 1
)

echo.
echo  Source: !SOURCE!
echo.
"%PYEXE%" calibrate_court.py --source "!SOURCE!"

echo.
pause
popd
