@echo off
setlocal EnableExtensions
chcp 65001 >nul 2>&1
title Prop EA Calendar SSL Fix

cd /d "%~dp0.."

echo ========================================
echo  Prop EA Calendar SSL Fix (VPS)
echo  Installs certifi and refreshes cache
echo ========================================
echo.

set "PY_CMD="
where python >nul 2>&1
if not errorlevel 1 set "PY_CMD=python"

if not defined PY_CMD (
    where py >nul 2>&1
    if not errorlevel 1 set "PY_CMD=py -3"
)

if not defined PY_CMD (
    echo [ERROR] Python not found in PATH.
    call :wait_close 15
    exit /b 1
)

echo [1/3] Installing certifi and requirements...
%PY_CMD% -m pip install -r "%~dp0..\requirements.txt"
if errorlevel 1 (
    echo [ERROR] pip install failed.
    call :wait_close 15
    exit /b 1
)
echo.

echo [2/3] Verifying CA bundle...
%PY_CMD% -c "import certifi; print('certifi CA:', certifi.where())"
if errorlevel 1 (
    echo [ERROR] certifi import failed.
    call :wait_close 15
    exit /b 1
)
echo.

echo [3/3] Fetching economic calendar...
%PY_CMD% calendar_service.py --once
if errorlevel 1 (
    echo.
    echo [ERROR] Calendar fetch still failed.
    echo         Copy cache\calendar.json from your dev PC if needed.
    call :wait_close 15
    exit /b 1
)

echo.
echo [OK] Calendar cache ready. Restart start_mt5_bridge.bat if the bridge is running.
call :wait_close 5
exit /b 0

:wait_close
set "WAIT_SEC=%~1"
if not defined WAIT_SEC set "WAIT_SEC=5"
echo [INFO] Closing this window in %WAIT_SEC% seconds...
timeout /t %WAIT_SEC% /nobreak >nul 2>&1
if errorlevel 1 ping 127.0.0.1 -n %WAIT_SEC% >nul
exit /b 0
