@echo off
title Financial Transaction Fraud Risk & Anomaly Detection System
color 0A

echo ===============================================================================
echo   FINANCIAL TRANSACTION ANOMALY DETECTION AND FRAUD RISK PREDICTION SYSTEM
echo   GTU BE Computer Engineering Semester 7 - InfoLabz IT Services Pvt. Ltd.
echo ===============================================================================
echo.

:: Detect Python
set PYTHON_EXE=python
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    if exist "C:\Users\dell\AppData\Local\Programs\Python\Python311\python.exe" (
        set PYTHON_EXE="C:\Users\dell\AppData\Local\Programs\Python\Python311\python.exe"
    ) else (
        echo [ERROR] Python was not found on your system!
        echo Please ensure Python 3.11+ is installed and added to PATH.
        pause
        exit /b 1
    )
)

echo [INFO] Using Python: %PYTHON_EXE%
%PYTHON_EXE% --version
echo.

:: Step 1: Check requirements
echo [1/4] Checking dependencies...
%PYTHON_EXE% -m pip install -r requirements.txt --quiet
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install required dependencies.
    pause
    exit /b 1
)
echo [OK] Dependencies verified.
echo.

:: Step 2: Check dataset
echo [2/4] Verifying dataset...
if not exist "dataset\transactions.csv" (
    echo [INFO] Dataset not found. Generating PaySim benchmark dataset...
    %PYTHON_EXE% dataset/generate_dataset.py --rows 60000 --output dataset/transactions.csv
) else (
    echo [OK] Dataset found at dataset\transactions.csv
)
echo.

:: Step 3: Check models
echo [3/4] Checking trained machine learning models...
if not exist "models\fraud_model.pkl" (
    echo [INFO] Supervised fraud models not found. Running training pipeline...
    %PYTHON_EXE% src/train_classification.py
)
if not exist "models\anomaly_model.pkl" (
    echo [INFO] Anomaly detection model not found. Running Isolation Forest training...
    %PYTHON_EXE% src/train_anomaly_model.py
)
echo [OK] Model artifacts ready.
echo.

:: Step 4: Launch Streamlit Dashboard
echo [4/4] Launching Interactive Streamlit Dashboard...
echo Dashboard will open automatically in your default web browser.
echo Press Ctrl+C in this terminal window to stop the application.
echo ===============================================================================
echo.

%PYTHON_EXE% -m streamlit run dashboard/app.py

pause
