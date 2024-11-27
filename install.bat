@echo off
chcp 65001 >nul
REM 檢查是否已安裝 Python
WHERE pip >nul 2>&1
IF %ERRORLEVEL% NEQ 0 ECHO [101mPython未安裝，請先安裝Python...[0m && PAUSE && EXIT /B 1

REM 安裝CUDA Toolkit 11.8
WHERE nvcc >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
    ECHO [92mCUDA已安裝...[0m
    goto PIP
    )
IF NOT %ERRORLEVEL% EQU 0 (
    set /p "cuda=尚未安裝CUDA Toolkit，是否要安裝？(y/n):"
)
IF %cuda% EQU n goto PIP
IF %cuda% EQU N goto PIP

ECHO [92m開始安裝CUDA Toolkit 11.8...[0m
start /wait cuda_11.8.0_windows_network.exe

:PIP
set /p "pip=是否要安裝Python library？(y/n):"

IF %pip% EQU n PAUSE && EXIT /B 1
IF %pip% EQU N PAUSE && EXIT /B 1

ECHO [92m開始安裝library...[0m
REM 安裝具有 CUDA 支持的 PyTorch、TorchVision 和 Torchaudio
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

REM 安裝其他依賴
pip install -r requirements.txt

echo [92m安裝完成...[0m
PAUSE