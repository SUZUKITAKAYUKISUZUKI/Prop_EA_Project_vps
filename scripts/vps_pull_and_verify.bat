@echo off
setlocal EnableExtensions
chcp 65001 >nul 2>&1
title Prop EA — VPS git pull + verify

cd /d "%~dp0\.."

echo ========================================
echo  VPS deploy verify
echo  Folder: %CD%
echo ========================================
echo.

where git >nul 2>&1
if errorlevel 1 (
    echo [ERROR] git not found in PATH
    pause
    exit /b 1
)

echo [1/4] git remote
git remote -v
echo.

echo [2/4] git fetch + pull
git fetch origin
git pull origin main
if errorlevel 1 (
    echo [ERROR] git pull failed — resolve conflicts or check network
    pause
    exit /b 1
)
echo.

echo [3/4] HEAD commit
git log -1 --format=%%h %%ci %%s
echo.

echo [4/4] Live code marker in main_platform.py
findstr /C:"LIVE_SETUP_MATCH_VERSION" main_platform.py
if errorlevel 1 (
    echo [FAIL] LIVE_SETUP_MATCH_VERSION not found — OLD main_platform.py
    echo        Expected: signal_v2 after 2026-06-19 deploy
    pause
    exit /b 1
)

where py >nul 2>&1
if not errorlevel 1 (
    echo.
    echo [optional] bridge smoke test...
    py -3 scripts\vps_bridge_smoke.py
)

echo.
echo [OK] Pull complete. Restart start_mt5_bridge.bat if Python was running.
pause
