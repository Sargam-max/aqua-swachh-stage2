@echo off
title Stop Aqua-Swachh Dashboard
echo ===============================================================
echo   Stopping Aqua-Swachh Server on Port 8000...
echo ===============================================================
echo.
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING BOUND"') do (
    taskkill /F /PID %%a >nul 2>&1
)
echo [OK] Port 8000 cleared. Server has been stopped.
ping -n 2 127.0.0.1 >nul
