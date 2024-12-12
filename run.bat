@echo off
set CONDA_EXE=""
for /f "delims=" %%i in ('where conda.exe 2^>nul') do set CONDA_EXE=%%i
if "%CONDA_EXE%"=="" (
    echo Conda is not installed.
    python -m streamlit run main.py --server.maxUploadSize 10000
) else (
    echo Conda is installed at %CONDA_EXE%.
    set /p ENV_NAME=Please enter the Conda environment name to activate:
    echo Activating Conda environment %ENV_NAME%...
    call conda activate %ENV_NAME%
    streamlit run main.py --server.maxUploadSize 10000
)