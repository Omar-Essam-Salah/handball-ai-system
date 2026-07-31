@echo off
setlocal enabledelayedexpansion
title Apply newest trained ball model
color 0A
pushd "%~dp0"

echo  ================================================================
echo    Searching for the newest trained ball model...
echo  ================================================================
echo.

set "LATEST="
set "LATEST_TIME=0"

REM Walk every training run folder and pick the one with the newest best.pt
for /f "delims=" %%D in ('dir /b /ad /o-d "runs\detect\training\runs\ball_*" 2^>nul') do (
    if exist "runs\detect\training\runs\%%D\weights\best.pt" (
        if not defined LATEST (
            set "LATEST=runs\detect\training\runs\%%D\weights\best.pt"
            echo Found: !LATEST!
            goto :found
        )
    )
)

:found
if not defined LATEST (
    echo [ERROR] No trained ball model found under runs\detect\training\runs\
    echo Did the training finish? Look for the message:
    echo     "Results saved to ..."
    pause
    popd
    exit /b 1
)

echo.
echo  ----------------------------------------------------------------
echo  Copying: !LATEST!
echo  To:      handball_ball.pt
echo  ----------------------------------------------------------------
copy /Y "!LATEST!" "handball_ball.pt"

if errorlevel 1 (
    echo [ERROR] Copy failed.
    pause
    popd
    exit /b 1
)

echo.
echo  ================================================================
echo    SUCCESS — new model active.
echo    Next:   test_video.bat hand3.mp4
echo  ================================================================
pause
popd
