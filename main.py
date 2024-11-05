import os
import streamlit as st
from ultralytics import RTDETR
import tempfile
import atexit
import shutil
from moviepy.editor import VideoFileClip

# 用於存儲臨時目錄的列表
temp_dirs = []

def cleanup_temp_dirs():
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

# 註冊清理函數
atexit.register(cleanup_temp_dirs)

def main():
    st.title('無人機辨識系統')
    st.write('本系統使用 RTDETR 模型進行無人機辨識，支持圖片和影片的上傳。')
    data_file = st.file_uploader('上傳圖片/影像', type=['mp4', 'mov', 'avi', 'png', 'jpg', 'jpeg'])
    if data_file is not None:

        file_name = data_file.name
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

        elif file_extension in ['.png', '.jpg', '.jpeg']:
            # 顯示上傳的圖片
            st.text('原始圖片')
            st.image(temp_file_path)

        # 使用 RTDETR 模型進行預測
        model = RTDETR("model/rtdetr-x.pt")

        try:
            shutil.rmtree("./result")
        except:
            st.text("result 目錄不存在")
    
        result = model(temp_file_path, show=False, save=True, project='result', name='predict')

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
            elif file_extension in ['.png', '.jpg', '.jpeg']:
                predict_path = os.path.join(output_dir, file_name)
                
                st.image(predict_path)
        except:
            st.text('預測結果無法顯示')

if __name__ == '__main__':
    main()