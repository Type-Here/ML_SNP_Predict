@echo off

:: Check if Conda is installed
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo Conda is not installed or not in PATH.
    echo Please install Conda and ensure it is available in the PATH.
    exit /b 1
)

:: Check if a Conda environment is active
if "%CONDA_DEFAULT_ENV%"=="" (
    echo No conda environment detected.
    echo Please activate the conda environment before running the application.
    echo Default conda env name: fia_env
    exit /b 1
) else (
    echo Conda environment detected: %CONDA_DEFAULT_ENV%
)

:: Check Python version and run the application
python --version >nul 2>&1
if %errorlevel%==0 (
    python UI\main.py
) else (
    python3 UI\main.py
)
