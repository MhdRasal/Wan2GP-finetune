@echo off
REM ==============================================================================
REM Wan2GP Cloud Launcher for Windows
REM Zero local GPU requirements - runs in cloud client mode
REM ==============================================================================

cd /d "%~dp0"
title Wan2GP Cloud (Modal GPU)

echo ============================================================
echo           Starting Wan2GP in Cloud Mode (Modal GPU)         
echo ============================================================

REM Check Python interpreter (.venv or system python)
if exist "%~dp0.venv\Scripts\python.exe" (
    set "PY_BIN=%~dp0.venv\Scripts\python.exe"
) else (
    where python >nul 2>nul
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Python is not found. Please install Python 3.10+ from python.org and tick "Add to PATH".
        pause
        exit /b 1
    )
    set "PY_BIN=python"
)

REM Check if sync requested via argument
echo "%*" | findstr /I "\--sync" >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo [*] Synchronizing local repository and Modal cloud worker in lockstep...
    "%PY_BIN%" sync_upstream_and_modal.py
    echo.
) else (
    REM Check if AUTO_SYNC is set in .env
    findstr /C:"AUTO_SYNC=\"true\"" .env >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        echo [*] AUTO_SYNC active: Synchronizing local repo and cloud worker...
        "%PY_BIN%" sync_upstream_and_modal.py
        echo.
    )
)

REM Check if .env is configured with Modal (unless RunPod is selected)
findstr /C:"WAN2GP_DEFAULT_PROVIDER=\"runpod\"" .env >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo [*] Cloud Provider: RunPod Serverless (Beta Preview)
) else (
    findstr /C:"MODAL_ENDPOINT" .env >nul 2>nul
    if %ERRORLEVEL% neq 0 (
        echo.
        echo [!] Cloud Modal is not set up yet on your device.
        echo [*] Launching the 1-click Modal setup wizard now...
        echo.
        "%PY_BIN%" cloud\modal\setup_modal.py
        echo.
    )
)

echo [*] Launching Wan2GP Web UI...
"%PY_BIN%" wgp.py --cloud --open-browser %*

if %ERRORLEVEL% neq 0 (
    pause
)
