@echo off
setlocal EnableExtensions
cd /d "%~dp0\.."
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0verify_vps_deploy_ready.ps1" %*
set "EC=%ERRORLEVEL%"
echo.
if "%EC%"=="0" (
    echo [OK] Ready for commit/push in Prop_EA_Project_vps
) else if "%EC%"=="2" (
    echo [WARN] Sync ran but no git changes — save dev files and sync again?
) else (
    echo [FAIL] Fix mismatches before push
)
pause
exit /b %EC%
