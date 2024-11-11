import os
import streamlit as st
from ultralytics import RTDETR
import tempfile
import atexit
import shutil
import cv2
from moviepy.editor import VideoFileClip

# 用於存儲臨時目錄的列表
temp_dirs = []

def cleanup_temp_dirs():
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

# 註冊清理函數
atexit.register(cleanup_temp_dirs)

def make_log(results, fps):
    frame = 1
    with open('log.txt', 'w') as f:
        for result in results:
            seconds = str(f"{frame / fps:.2f}")
            frame += 1
            if "persons" in result.verbose():
                f.write(seconds + 's: ' + result.verbose() + '\n')
            

def choose_place():                                             #之後要加入訓練好的各種模型
    place = st.selectbox('選擇場景', ['山地', '河邊', '海上'])
    if place == '山地':
        st.write('已選擇山地場景')
        return RTDETR("model/rtdetr-x.pt")
    elif place == '河邊':
        st.write('已選擇河邊場景')
        return RTDETR("model/rtdetr-x.pt")
    elif place == '海上':
        st.write('已選擇海上場景')
        return RTDETR("model/rtdetr-x.pt")

def main():
    st.title('無人機影像辨識系統')
    st.write('本系統使用 RTDETR 模型進行無人機影像辨識，支援圖片和影片格式。')
    data_file = st.file_uploader('上傳圖片/影像', type=['mp4', 'mov', 'avi', 'png', 'jpg', 'jpeg'])
    model = choose_place()
    if data_file is not None:
        file_name = data_file.name
        if 'processed' not in st.session_state:
            st.session_state.processed = {}
        # Process and save results
        if st.button('開始辨識') or file_name in st.session_state.processed:
            result = output(data_file, file_name, model)
            st.session_state.processed[file_name] = result

        if file_name in st.session_state.processed:
            st.write(f'File: {file_name}')
            st.write(st.session_state.processed[file_name])
    else:
        st.text('請上傳圖片或影片')


def output(data_file, file_name, model=None):
    global fps
    temp_dir = tempfile.mkdtemp()
    temp_dirs.append(temp_dir)
    temp_file_path = os.path.join(temp_dir, data_file.name)

    with open(temp_file_path, 'wb') as f:
        f.write(data_file.getbuffer())

    st.text(f"檔案已保存到: {temp_file_path}")

    file_extension = os.path.splitext(data_file.name)[1].lower()
    if file_extension in ['.mp4', '.mov', '.avi']:
        # 顯示上傳的影片
        st.text('原始影像')
        st.video(temp_file_path)
        fps = cv2.VideoCapture(temp_file_path).get(cv2.CAP_PROP_FPS)

    elif file_extension in ['.png', '.jpg', '.jpeg']:
        # 顯示上傳的圖片
        st.text('原始圖片')
        st.image(temp_file_path)
        fps = 1

    if not file_name in st.session_state.processed:
        results = inference(temp_file_path, model)

    # 顯示預測結果
    output_dir = "result/predict"
    st.text('預測結果')
    
    try :
        if file_extension in ['.mp4', '.mov', '.avi']:
            predict_path_avi = os.path.join(output_dir, os.path.splitext(file_name)[0] + '.avi')
            predict_path_mp4 = os.path.join(output_dir, os.path.splitext(file_name)[0] + '.mp4')
            
            # 將 AVI 轉換為 MP4
            clip = VideoFileClip(predict_path_avi)
            clip.write_videofile(predict_path_mp4)
            
            st.video(predict_path_mp4)
            with open(predict_path_mp4, 'rb') as f:
                st.download_button('下載預測影片', f, os.path.splitext(file_name)[0] + '.mp4')
            with open('log.txt', 'rb') as log:
                st.download_button('下載預測報告', log, os.path.splitext(file_name)[0] + '.txt')
        elif file_extension in ['.png', '.jpg', '.jpeg']:
            predict_path = os.path.join(output_dir, file_name)
            st.image(predict_path)
            with open(predict_path, 'rb') as f:
                st.download_button('下載預測照片', f, file_name)

        
    except:
        st.text('預測結果無法顯示')

def inference(temp_file_path, model=None):

    try:
        shutil.rmtree("./result")
    except:
        pass
    
    result = model(temp_file_path, show=True, save=True, project='result', name='predict')
    make_log(result, fps)
    return result

if __name__ == '__main__':
    main()