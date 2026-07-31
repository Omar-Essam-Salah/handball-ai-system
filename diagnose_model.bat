@echo off
setlocal enabledelayedexpansion
title Diagnose ball model
color 0E
pushd "%~dp0"

echo  ================================================================
echo    DIAGNOSING BALL MODEL
echo  ================================================================
echo.

echo [1] handball_ball.pt status:
if exist "handball_ball.pt" (
    dir handball_ball.pt | findstr "handball_ball"
) else (
    echo   FILE NOT FOUND
)
echo.

echo [2] Latest training run:
for /f "delims=" %%D in ('dir /b /ad /o-d "runs\detect\training\runs\ball_*" 2^>nul') do (
    echo   %%D
    if exist "runs\detect\training\runs\%%D\weights\best.pt" (
        dir "runs\detect\training\runs\%%D\weights\best.pt" ^| findstr "best"
        if exist "runs\detect\training\runs\%%D\results.csv" (
            echo.
            echo   Last 3 lines of results.csv ^(epoch, train_loss, val_metrics^):
            powershell -NoProfile -Command "Get-Content 'runs\detect\training\runs\%%D\results.csv' | Select-Object -Last 3"
        )
    ) else (
        echo   ^[no best.pt in this run^]
    )
    goto :one_done
)
:one_done
echo.

echo [3] Training dataset size:
for /f %%N in ('dir /b "training\dataset\images\train\*.jpg" 2^>nul ^| find /c /v ""') do echo   train images: %%N
for /f %%N in ('dir /b "training\dataset\labels\train\*.txt" 2^>nul ^| find /c /v ""') do echo   train labels: %%N
echo.

echo [4] How many of those labels are user-corrected?
for /f %%N in ('dir /b "training\dataset\images\train\*_user.jpg" 2^>nul ^| find /c /v ""') do echo   user-corrected: %%N
echo.

echo [5] Quick functional test of the LOADED model:
.venv\Scripts\python.exe -c "import sys; sys.path.insert(0, 'src'); from detector import Detector; import os, time; d = Detector(device='cuda:0', conf=0.40, ball_conf=0.05, imgsz=640); mp = 'handball_ball.pt'; print('  weights mtime:', time.ctime(os.path.getmtime(mp)) if os.path.exists(mp) else 'NO FILE'); import cv2; cap = cv2.VideoCapture('hand3.mp4'); fps = cap.get(cv2.CAP_PROP_FPS) or 25; cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps*60)); ok, f = cap.read(); cap.release(); print('  frame read:', ok, f.shape if ok else None); dets = d.detect(f) if ok else []; balls = [x for x in dets if x.class_id==32]; persons=[x for x in dets if x.class_id==0]; print(f'  persons: {len(persons)}'); print(f'  balls:   {len(balls)}'); [print(f'    ball conf={b.confidence:.2f} box={int(b.x1)},{int(b.y1)}-{int(b.x2)},{int(b.y2)}') for b in balls[:5]]"

echo.
echo  ================================================================
pause
popd
