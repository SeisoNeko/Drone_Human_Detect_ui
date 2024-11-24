@echo off

REM 安裝具有 CUDA 支持的 PyTorch、TorchVision 和 Torchaudio
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

REM 安裝其他依賴
pip install -r requirements.txt