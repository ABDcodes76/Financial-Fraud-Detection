@echo off
title FINSEC AI - Push to GitHub
echo ========================================================
echo        FINSEC AI - Auto Push to GitHub
echo ========================================================
echo.

set PYTHON_EXE=python
if exist "C:\Users\dell\AppData\Local\Programs\Python\Python311\python.exe" (
    set PYTHON_EXE="C:\Users\dell\AppData\Local\Programs\Python\Python311\python.exe"
)

if "%~1"=="" (
    %PYTHON_EXE% sync_github.py
) else (
    %PYTHON_EXE% sync_github.py "%~1"
)

echo.
pause
