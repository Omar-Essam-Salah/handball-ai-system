@echo off
setlocal enabledelayedexpansion
title Handball AI - MAX MODE (all upgrades enabled)
color 0E
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
echo    HANDBALL AI  -  MAX MODE
echo  ================================================================
echo    Phase 1: SAHI sliced inference   (distant-player recall +89%%)
echo    Phase 2: Temporal ball consensus (head/marker FPs suppressed)
echo    Phase 3: OC-SORT tracker         (fewer ID switches)
echo    Phase 4: OSNet ReID embeddings   (court-color bleed immune)
echo    Phase 5: Keyframe jersey OCR     (better number accuracy)
echo  ----------------------------------------------------------------
echo    Cost: ~4-5x slower than base. Use for offline analysis.
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
    --use-sahi ^
    --tracker ocsort ^
    --use-osnet

echo.
echo  ----------------------------------------------------------------
pause
popd
