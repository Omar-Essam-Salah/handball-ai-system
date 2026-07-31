@echo off
setlocal
title Handball Studio
pushd "%~dp0"

REM Use python.exe (not pythonw) so any startup error is VISIBLE in this window.
set "PYEXE=.venv\Scripts\python.exe"
if not exist "%PYEXE%" set "PYEXE=python"

echo Starting Handball Studio...  (leave this window open)
"%PYEXE%" src\handball_studio.py
echo.
echo [Studio closed] If it closed immediately, the error is shown above.
pause >nul
popd
