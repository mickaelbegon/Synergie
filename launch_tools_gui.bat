@echo off
setlocal

set "CONDA_EXE=C:\Users\micka\miniconda3\Scripts\conda.exe"
set "ENV_PYTHON=C:\Users\micka\miniconda3\envs\synergie-data\python.exe"
set "REPO_DIR=%~dp0"

if not exist "%ENV_PYTHON%" (
    echo Missing Python executable:
    echo   %ENV_PYTHON%
    echo.
    echo Recreate the synergie-data environment first.
    pause
    exit /b 1
)

pushd "%REPO_DIR%"
"%ENV_PYTHON%" tools_gui.py
set "EXIT_CODE=%ERRORLEVEL%"
popd

if not "%EXIT_CODE%"=="0" (
    echo.
    echo Synergie Tools stopped with exit code %EXIT_CODE%.
    pause
)

exit /b %EXIT_CODE%
