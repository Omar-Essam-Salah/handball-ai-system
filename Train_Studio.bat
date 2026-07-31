@echo off
title Handball - Training Studio
cd /d "%~dp0"
echo Launching Training Studio...
.venv\Scripts\python.exe src\train_studio.py
if errorlevel 1 (
  echo.
  echo [!] Training Studio exited with an error. See the message above.
  pause
)
