@echo off
setlocal enabledelayedexpansion
title Handball Mobile Server
color 0B
pushd "%~dp0"

set "PYEXE=.venv\Scripts\python.exe"
if not exist "%PYEXE%" set "PYEXE=python"

cls
echo  ================================================================
echo    HANDBALL MOBILE  -  phone / tablet monitoring server
echo  ================================================================
echo.
echo  Open the URL shown below on your phone (same WiFi network).
echo  Then tap "Open" with a video name, or "Scan cameras".
echo.
set "SRC=%~1"
if "%SRC%"=="" set /p "SRC=Video file or RTSP to start now (blank = choose on phone): "

"%PYEXE%" src\mobile_server.py --source "%SRC%" --port 8000

echo.
echo [Server stopped] Press any key to close.
pause >nul
popd
