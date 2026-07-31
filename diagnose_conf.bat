@echo off
title Check ball-model confidence distribution
color 0E
pushd "%~dp0"

.venv\Scripts\python.exe scripts\diagnose_conf.py hand3.mp4

echo.
echo  ================================================================
pause
popd
