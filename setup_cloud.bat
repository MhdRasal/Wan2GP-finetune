@echo off
cd /d "%~dp0"
python cloud\modal\setup_modal.py %*
if %ERRORLEVEL% neq 0 (
    pause
)
