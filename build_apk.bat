@echo off
title Build Handball Studio APK
set "WS=C:\Users\dell\android-build"
for /f "delims=" %%J in ('dir /b /ad "%WS%\jdk\jdk-*"') do set "JAVA_HOME=%WS%\jdk\%%J"
set "ANDROID_HOME=%WS%\sdk"
set "ANDROID_SDK_ROOT=%WS%\sdk"
set "PATH=%JAVA_HOME%\bin;%PATH%"
pushd "%~dp0android"
call "%WS%\gradle\gradle-8.7\bin\gradle.bat" assembleDebug --no-daemon
popd
copy /Y "%~dp0android\app\build\outputs\apk\debug\app-debug.apk" "%~dp0HandballStudio.apk"
copy /Y "%~dp0HandballStudio.apk" "%~dp0webapp\HandballStudio.apk"
echo.
echo APK -> HandballStudio.apk
pause
