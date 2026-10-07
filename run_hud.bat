@echo off
title Industrial DIN Rail Assembly Verification HUD
cd /d "%~dp0"
echo ========================================================================
echo  STARTING INDUSTRIAL ASSEMBLY VERIFICATION HUD
echo  Camera Device: 1 (USB Phone Camera) with Device 0 fallback
echo ========================================================================
"C:\Users\melwi\Desktop\pen-assembly-checker\.venv\Scripts\python.exe" live_demo.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] HUD process exited with code %ERRORLEVEL%.
    pause
)
