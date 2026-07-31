@echo off
setlocal enabledelayedexpansion
title Handball AI Coach - Live Match
color 0A
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
echo    HANDBALL AI COACH  -  Live Match
echo    (Original pipeline + Streamlit dashboard)
echo  ================================================================
echo.

REM ── read camera IP from config\camera.yaml (best-effort) ────────────────
set "CAM_IP=192.168.8.190"
if exist "config\camera.yaml" (
    for /f "tokens=2 delims=: " %%i in ('findstr /B /C:"ip:" config\camera.yaml 2^>nul') do (
        if not defined CAM_IP_FROM_CFG set "CAM_IP_FROM_CFG=%%i"
    )
    if defined CAM_IP_FROM_CFG set "CAM_IP=!CAM_IP_FROM_CFG!"
)

set /p "USER_IP=Camera IP [!CAM_IP!]: "
if not "!USER_IP!"=="" set "CAM_IP=!USER_IP!"

set "TEAM_A=Home"
set /p "USER_A=Team A name [!TEAM_A!]: "
if not "!USER_A!"=="" set "TEAM_A=!USER_A!"

set "TEAM_B=Away"
set /p "USER_B=Team B name [!TEAM_B!]: "
if not "!USER_B!"=="" set "TEAM_B=!USER_B!"

echo.
echo  ----------------------------------------------------------------
echo   Camera IP   : !CAM_IP!
echo   Teams       : !TEAM_A!  vs  !TEAM_B!
echo   Dashboard   : http://localhost:8501
echo  ----------------------------------------------------------------
echo  Tip: run calibrate.bat once at the venue before kickoff
echo       so the minimap shows correct foot positions.
echo.

REM ── camera reachability probe ───────────────────────────────────────────
echo Checking camera at !CAM_IP! ...
ping -n 1 -w 2000 !CAM_IP! >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Camera !CAM_IP! is NOT pinging. Press any key to try anyway, or close window to cancel.
    pause >nul
) else (
    echo [OK] Camera reachable.
)
echo.

echo [1/2] Starting AI Pipeline ...

start "Handball Pipeline" cmd /k ""%PYEXE%" src\pipeline.py --source !CAM_IP! --model yolov8s-pose.pt --device cuda:0 --infer-every 2 --imgsz 640 --scale 0.7 --display-every 1 --debug"

echo [2/2] Waiting 12 s for the model to load ...
timeout /t 12 /nobreak >nul

echo Starting Dashboard ...
start "Handball Dashboard" cmd /k ""%PYEXE%" -m streamlit run dashboard\app.py --server.port 8501 --server.headless true"

timeout /t 4 /nobreak >nul
start "" http://localhost:8501

echo.
echo  ================================================================
echo   System running.
echo.
echo   Pipeline window keys:
echo     R  - recalibrate team colors
echo     C  - re-detect court lines
echo     F  - flip attacking sides (use at half-time)
echo     A  - manual goal for !TEAM_A!
echo     B  - manual goal for !TEAM_B!
echo     D  - toggle debug overlay
echo     Q  - quit
echo.
echo   Debug mode is ON. Watch the pipeline terminal for:
echo     * SHOT / GOAL / POSSESSION events
echo     * scorer attribution
echo     * ReID enrollments and ball state
echo  ================================================================
echo.
pause
popd
