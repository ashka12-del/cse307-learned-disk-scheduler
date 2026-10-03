@echo off
setlocal
cd /d "%~dp0"
title CSE-307 Interactive Scheduler Demo

set "PYTHON_EXE="
if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=.venv\Scripts\python.exe"
if not defined PYTHON_EXE if exist "..\..\.venv\Scripts\python.exe" set "PYTHON_EXE=..\..\.venv\Scripts\python.exe"
if not defined PYTHON_EXE where py >nul 2>nul && set "PYTHON_EXE=py"
if not defined PYTHON_EXE where python >nul 2>nul && set "PYTHON_EXE=python"

if not defined PYTHON_EXE (
    echo Python was not found. Install Python 3.10 or newer, then try again.
    pause
    exit /b 1
)

echo Enter a queue such as:
echo 98 183 37 122 14 124 65 67
echo.
call %PYTHON_EXE% src\demo.py
echo.
pause

