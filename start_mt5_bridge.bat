@echo off
setlocal EnableExtensions
chcp 65001 >nul 2>&1
title Prop EA MT5 Bridge Launcher

cd /d "%~dp0"

echo ========================================
echo  Prop EA MT5 Bridge Server
echo  API:      http://127.0.0.1:8000/health
echo  Ollama + Calendar + LLM audit
echo ========================================
echo.

REM --- Resolve Python: prefer "python", fallback to Windows "py -3" launcher ---
set "PY_CMD="
where python >nul 2>&1
if not errorlevel 1 set "PY_CMD=python"

if not defined PY_CMD (
    where py >nul 2>&1
    if not errorlevel 1 set "PY_CMD=py -3"
)

if not defined PY_CMD (
    echo [ERROR] Python not found in PATH.
    echo         Install Python 3.10+ and enable "Add to PATH", or use the "py" launcher.
    echo         VPS tip: run "py -3 --version" in cmd to verify.
    call :wait_close 15
    exit /b 1
)

%PY_CMD% -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.10 or newer is required.
    %PY_CMD% --version 2>nul
    call :wait_close 15
    exit /b 1
)

%PY_CMD% -c "import fastapi, uvicorn" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    %PY_CMD% -m pip install -r "%~dp0requirements.txt"
    if errorlevel 1 (
        echo [ERROR] pip install failed.
        call :wait_close 15
        exit /b 1
    )
    echo.
)

echo [1/2] Checking port 8000...
%PY_CMD% "%~dp0bridge_preflight.py" --kill-stale
if errorlevel 1 (
    echo.
    echo [ERROR] Port 8000 is in use. Stop the other bridge process first.
    call :wait_close 15
    exit /b 1
)

REM Skip launch if integrated runtime is already healthy
%PY_CMD% -c "import json,urllib.request; d=json.loads(urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=2).read()); raise SystemExit(0 if 'calendar' in d and 'llm_auditor' in d else 1)" >nul 2>&1
if not errorlevel 1 (
    echo.
    echo [OK] Bridge runtime already running.
    echo      http://127.0.0.1:8000/health
    call :wait_close 3
    exit /b 0
)

echo [2/2] Starting MT5 Bridge API...
echo       http://127.0.0.1:8000/health
echo.

start "Prop EA MT5 Bridge Server" /D "%~dp0" cmd /k "%PY_CMD% -m uvicorn mt5_bridge:app --host 127.0.0.1 --port 8000 --log-level info"

echo [OK] Server started in a new window.
echo      Close that window or press Ctrl+C to stop.
call :wait_close 3
exit /b 0

:wait_close
REM Avoid Japanese on the same block as timeout (VPS codepage mojibake guard)
set "WAIT_SEC=%~1"
if not defined WAIT_SEC set "WAIT_SEC=5"
echo [INFO] Closing this window in %WAIT_SEC% seconds...
timeout /t %WAIT_SEC% /nobreak >nul 2>&1
if errorlevel 1 ping 127.0.0.1 -n %WAIT_SEC% >nul
exit /b 0
