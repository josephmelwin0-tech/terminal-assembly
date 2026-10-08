@echo off
title Industrial DIN Rail Assembly Verification HUD
cd /d "%~dp0"
echo ========================================================================
echo  STARTING INDUSTRIAL ASSEMBLY VERIFICATION HUD
echo  Webcam: Active (Press 'c' in HUD window anytime to cycle cameras)
echo  Controls: [q] Quit  |  [r] Reset Step  |  [c] Cycle Camera  |  [s] Save Snapshot
echo ========================================================================
"C:\Users\melwi\Desktop\pen-assembly-checker\.venv\Scripts\python.exe" live_demo.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] HUD process exited with code %ERRORLEVEL%.
    pause
)
