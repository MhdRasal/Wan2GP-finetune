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

REM Check Python
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not found. Please install Python 3.10+ from python.org and tick "Add to PATH".
    pause
    exit /b 1
)

REM Check if .env is configured with Modal
findstr /C:"MODAL_ENDPOINT" .env >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo.
    echo [!] Cloud Modal is not set up yet on your device.
    echo [*] Launching the 1-click Modal setup wizard now...
    echo.
    python cloud\modal\setup_modal.py
    echo.
)

echo [*] Launching Wan2GP Web UI...
python wgp.py --cloud --provider modal --open-browser %*

if %ERRORLEVEL% neq 0 (
    pause
)
