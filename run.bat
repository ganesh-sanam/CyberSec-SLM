@echo off
title CyberSec-SLM Server
echo ============================================================
echo           Starting CyberSec-SLM Application...
echo ============================================================
echo.
echo Opening browser at http://localhost:7860 in 3 seconds...
start "" http://localhost:7860
echo.
python app.py
pause
