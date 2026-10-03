@echo off
setlocal
cd /d "%~dp0"
title CSE-307 Learned Disk Scheduler
set "MPLCONFIGDIR=%TEMP%\cse307-matplotlib"
if not exist "%MPLCONFIGDIR%" mkdir "%MPLCONFIGDIR%" >nul 2>nul

set "BASE_PYTHON="
if exist ".venv\Scripts\python.exe" goto environment_ready
if exist "..\..\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=..\..\.venv\Scripts\python.exe"
    goto dependencies_ready
)
where py >nul 2>nul && set "BASE_PYTHON=py"
if not defined BASE_PYTHON where python >nul 2>nul && set "BASE_PYTHON=python"

if not defined BASE_PYTHON (
    echo Python was not found.
    echo Install Python 3.10 or newer from https://www.python.org/downloads/
    echo During installation, select "Add Python to PATH".
    echo Then double-click this file again.
    pause
    exit /b 1
)

echo Creating the local Python environment...
call %BASE_PYTHON% -m venv .venv
if errorlevel 1 (
    echo Could not create .venv.
    pause
    exit /b 1
)

:environment_ready
set "PYTHON_EXE=.venv\Scripts\python.exe"

:dependencies_ready
call "%PYTHON_EXE%" -c "import numpy, pandas, sklearn, matplotlib, pytest" >nul 2>nul
if errorlevel 1 (
    echo Installing required packages. This may take a few minutes on the first run...
    call "%PYTHON_EXE%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Package installation failed. Check the internet connection and messages above.
        pause
        exit /b 1
    )
)

echo Running scheduler tests...
call "%PYTHON_EXE%" -m pytest -q -p no:cacheprovider
if errorlevel 1 (
    echo.
    echo Tests failed. Review the messages above.
    pause
    exit /b 1
)

echo.
echo Running the complete experiment...
call "%PYTHON_EXE%" src\run_all.py
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

