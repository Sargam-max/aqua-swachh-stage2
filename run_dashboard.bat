@echo off
cd /d "%~dp0"
title Aqua-Swachh Smart Purifier AI Dashboard

echo ===============================================================
echo   AQUA-SWACHH: Smart Purifier ML Web Dashboard
echo   atomQUEST Track 2 - Team Aqua-Swachh (RGIPT)
echo ===============================================================
echo.

rem 1. Check if server is already active on port 8000
netstat -ano | findstr ":8000" | findstr "LISTENING" >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Aqua-Swachh server is already running on http://127.0.0.1:8000!
    echo [INFO] Opening web browser...
    start "" "http://127.0.0.1:8000"
    ping -n 2 127.0.0.1 >nul
    exit /b 0
)

rem 2. Clean up any stale zombie process on port 8000
echo [INFO] Preparing port 8000...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000" ^| findstr "BOUND"') do (
    taskkill /F /PID %%a >nul 2>&1
)

rem 3. Detect Python environment
set PYTHON_EXE=
if exist "%~dp0.venv\Scripts\python.exe" set PYTHON_EXE=%~dp0.venv\Scripts\python.exe
if "%PYTHON_EXE%"=="" if exist "C:\Users\chris\Downloads\aqua-swachh\.venv\Scripts\python.exe" set PYTHON_EXE=C:\Users\chris\Downloads\aqua-swachh\.venv\Scripts\python.exe
if "%PYTHON_EXE%"=="" if exist "c:\Users\chris\Documents\antigravity\vibrant-bardeen\.venv\Scripts\python.exe" set PYTHON_EXE=c:\Users\chris\Documents\antigravity\vibrant-bardeen\.venv\Scripts\python.exe
if "%PYTHON_EXE%"=="" set PYTHON_EXE=python

echo [OK] Using Python: %PYTHON_EXE%

rem 4. Ensure dependencies are intact
"%PYTHON_EXE%" -c "import multipart" >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Installing required dependency python-multipart...
    "%PYTHON_EXE%" -m pip install python-multipart -q
)

rem 5. Ensure model weights are present
if not exist "%~dp0model\saved_models\random_forest_filter_model.joblib" (
    echo [INFO] Generating dataset and training ML model...
    "%PYTHON_EXE%" -u "%~dp0data\generate_dataset.py"
    "%PYTHON_EXE%" -u -m model.train
)

rem 6. Schedule browser to open 2 seconds after server starts
echo [INFO] Launching browser at http://127.0.0.1:8000 ...
start "" /b cmd /c "ping -n 3 127.0.0.1 >nul & start http://127.0.0.1:8000"

echo.
echo ===============================================================
echo   SERVER IS LIVE ON: http://127.0.0.1:8000
echo.
echo   [IMPORTANT] KEEP THIS WINDOW OPEN!
echo   Closing this window will stop the dashboard server.
echo   You can minimize this window while using the application.
echo   To stop the server when finished, press Ctrl+C.
echo ===============================================================
echo.

"%PYTHON_EXE%" -m uvicorn web.app:app --host 0.0.0.0 --port 8000

echo.
echo Server has stopped.
pause
