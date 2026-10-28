@echo off
REM ==============================================================================
REM Wan2GP Lockstep Dual Synchronizer for Windows
REM Updates BOTH local repository and Modal Cloud Worker at the exact same time
REM ==============================================================================

cd /d "%~dp0"
title Wan2GP Dual Sync (Local Repo + Modal Cloud)

echo ============================================================
echo   Wan2GP Dual Sync: Local Repository + Modal Cloud Worker   
echo ============================================================

REM Choose best available Python interpreter
if exist "%~dp0.venv\Scripts\python.exe" (
    set "PY_BIN=%~dp0.venv\Scripts\python.exe"
) else (
    where python >nul 2>nul
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Python not found. Please install Python 3.10+ and add to PATH.
        pause
        exit /b 1
    )
    set "PY_BIN=python"
)

"%PY_BIN%" sync_upstream_and_modal.py %*

echo.
pause
