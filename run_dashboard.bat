@echo off
title Industrial DIN Rail Web Dashboard Server
cd /d "%~dp0"
echo ========================================================================
echo  STARTING INDUSTRIAL WEB DASHBOARD SERVER (http://127.0.0.1:8000)
echo ========================================================================
"C:\Users\melwi\Desktop\pen-assembly-checker\.venv\Scripts\python.exe" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
pause
