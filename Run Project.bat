@echo off
setlocal
cd /d "%~dp0"
title CSE-307 Learned Disk Scheduler
set "MPLCONFIGDIR=%TEMP%\cse307-matplotlib"
if not exist "%MPLCONFIGDIR%" mkdir "%MPLCONFIGDIR%" >nul 2>nul

set "PYTHON_EXE="

if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=.venv\Scripts\python.exe"
if not defined PYTHON_EXE if exist "..\..\.venv\Scripts\python.exe" set "PYTHON_EXE=..\..\.venv\Scripts\python.exe"
if not defined PYTHON_EXE where py >nul 2>nul && set "PYTHON_EXE=py"
if not defined PYTHON_EXE where python >nul 2>nul && set "PYTHON_EXE=python"

if not defined PYTHON_EXE (
    echo Python was not found.
    echo Install Python 3.10 or newer from https://www.python.org/downloads/
    echo During installation, select "Add Python to PATH".
    echo Then double-click this file again.
    pause
    exit /b 1
)

echo Running scheduler tests...
call %PYTHON_EXE% -m pytest -q -p no:cacheprovider
if errorlevel 1 (
    echo.
    echo Tests failed. Review the messages above.
    pause
    exit /b 1
)

echo.
echo Running the complete experiment...
call %PYTHON_EXE% src\run_all.py
if errorlevel 1 (
    echo.
    echo The experiment failed. Review the messages above.
    pause
    exit /b 1
)

echo.
echo Experiment completed successfully.
echo Opening the results folder...
start "" "%CD%\results"
pause

