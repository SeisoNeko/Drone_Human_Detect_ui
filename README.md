# 無人機辨識系統

本專案使用 RTDETR 模型進行無人機辨識，支持圖片和影片的上傳。

## 功能

- 上傳圖片或影片
- 使用 RTDETR 模型進行無人機辨識
- 顯示原始圖片或影片
- 顯示預測結果

## 安裝

1. 複製此儲存庫到本地端：

    ```bash
    git clone https://github.com/SeisoNeko/Drone_Human_Detect_ui
    ```

2. 進入專案目錄：

    ```bash
    cd Drone_Human_Detect_ui
    ```

3. 建立並啟動虛擬環境：

    ```bash
    conda create -n yourProjectName python=3.8  #3.8以上
    conda activate yourProjectName  
    ```

4. 安裝所需的套件：

    ```bash
    pip install -r requirements.txt
    ```

5. 安裝對應版本的pytorch
    請參考 https://pytorch.org/ 並選取適合本地裝置的版本進行安裝
    舉例: windows用戶並且使用cuda 11.8 請使用
    ```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

6. 放入model
    請致 rtdetr/weights/ 放入合適的model pth檔 
    並到 rtdetr\tools\infer.py 第256行 輸入model路徑

## 使用方法

1. 啟動 Streamlit 應用：

    ```bash
    streamlit run main.py
    ```

2. 或者 執行run.py
    ```bash
    python run.py
    ```

3. 在瀏覽器中打開 `http://localhost:8501`，上傳圖片或影片進行無人機辨識。

## 專案結構

- `main.py`：主應用程式碼
- `requirements.txt`：所需的 Python 套件
- `model/`：存放 RTDETR 模型的目錄
- `result/`：存放預測結果的目錄

## 依賴項目

- Python 3.8+
- Streamlit
- ultralytics
- moviepy
- PIL (Pillow)

## 貢獻

歡迎提交問題和請求合併。如果您想貢獻代碼，請先分叉此儲存庫，創建一個分支，提交您的更改，然後創建一個 Pull Request。

## 授權

此專案使用 MIT 授權。詳情請參閱 [LICENSE](LICENSE) 文件。"# Drone_Human_Detect_ui" 
"# Drone_Human_Detect_ui" 
