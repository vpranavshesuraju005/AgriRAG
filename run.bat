@echo off
setlocal
color 0a
title AgriRAG Precision Farming Dashboard

:: Suppress TensorFlow Logs and oneDNN warnings to avoid terminal clutter
set "TF_CPP_MIN_LOG_LEVEL=2"
set "TF_ENABLE_ONEDNN_OPTS=0"

set "VENV_DIR=venv"

echo =======================================================
echo          AgriRAG: Precision Farming AI
echo =======================================================

:: Check if virtual environment exists, if not create it
if not exist "%VENV_DIR%" (
    echo [AgriRAG] Initializing system for the first time...
    python -m venv %VENV_DIR%
    if errorlevel 1 (
        echo [ERROR] Python not found or venv creation failed. Please ensure Python is installed and added to PATH.
        pause
        exit /b 1
    )
    echo [AgriRAG] Virtual environment created successfully.
)

:: Activate the environment
echo [AgriRAG] Activating AI Environment...
call %VENV_DIR%\Scripts\activate
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)

:: Install/Upgrade dependencies cleanly
echo [AgriRAG] Verifying dependencies and core AI modules...
python -m pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [WARNING] Minor errors occurred while resolving some dependencies. The dashboard will attempt to start anyway.
) else (
    echo [AgriRAG] Core dependencies are up to date.
)

:: Initialize MySQL Database and Tables
echo [AgriRAG] Verifying MySQL Connections...
python init_db.py

:: Handle Django Database Migrations
echo [AgriRAG] Synchronizing System Schemas...
python manage.py makemigrations
python manage.py migrate

:: Start the browser automatically
echo [AgriRAG] Launching Precision Farming Dashboard...
start http://127.0.0.1:8080/

:: Run the Django server
echo [AgriRAG] Server starting on port 8080...
python manage.py runserver 8080

pause
