::[Bat To Exe Converter]
::
::YAwzoRdxOk+EWAjk
::fBw5plQjdCyDJGyX8VAjFDZbXg2JM2WGIrof/eX+4f6UnmoUQMoqes/p36SBM9w3+ErqcKkFw3dblvQoCQ9dfQaUewYIoG1NuCqMNMj8
::YAwzuBVtJxjWCl3EqQJgSA==
::ZR4luwNxJguZRRnk
::Yhs/ulQjdF+5
::cxAkpRVqdFKZSjk=
::cBs/ulQjdF+5
::ZR41oxFsdFKZSDk=
::eBoioBt6dFKZSDk=
::cRo6pxp7LAbNWATEpCI=
::egkzugNsPRvcWATEpCI=
::dAsiuh18IRvcCxnZtBJQ
::cRYluBh/LU+EWAnk
::YxY4rhs+aU+JeA==
::cxY6rQJ7JhzQF1fEqQJQ
::ZQ05rAF9IBncCkqN+0xwdVs0
::ZQ05rAF9IAHYFVzEqQJQ
::eg0/rx1wNQPfEVWB+kM9LVsJDGQ=
::fBEirQZwNQPfEVWB+kM9LVsJDGQ=
::cRolqwZ3JBvQF1fEqQJQ
::dhA7uBVwLU+EWDk=
::YQ03rBFzNR3SWATElA==
::dhAmsQZ3MwfNWATElA==
::ZQ0/vhVqMQ3MEVWAtB9wSA==
::Zg8zqx1/OA3MEVWAtB9wSA==
::dhA7pRFwIByZRRnk
::Zh4grVQjdCyDJGyX8VAjFDZbXg2JM2WGIrof/eX+4f6UnmoUQMoqes/p36SBM9w3+ErqcKkFw3dblvQoCQ9dfQaUewYIu3tMuGGXecKEtm8=
::YB416Ek+ZG8=
::
::
::978f952a14a936cc963da21a135fa983
@echo off
set CONDA_EXE=
for %%i in (conda.exe) do set CONDA_EXE=%%~$PATH:i

if "%CONDA_EXE%"=="" (
    echo Conda is not installed.
    python -m streamlit run main.py --server.maxUploadSize 10000
) else (
    echo Conda is installed at %CONDA_EXE%.
    set /p ENV_NAME=Please enter the Conda environment name to activate: 
    conda activate %ENV_NAME%
    streamlit run main.py --server.maxUploadSize 10000
)