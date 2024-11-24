import streamlit as st
import os
import torch
import torchvision.transforms as T
import numpy as np 
from PIL import Image
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import cv2
import zipfile
import shutil
from rtdetr.tools.infer import InitArgs, draw, initModel
from anomalyDET import anomaly_main

video_format = ['mp4', 'mov', 'avi', 'MP4', 'MOV', 'AVI']
image_format = ['png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG']

def main():
    st.title('無人機人員偵測系統')
    
    # upload the file
    uploaded_files = st.file_uploader('上傳圖片/影像', type=['mp4', 'mov', 'avi', 'png', 'jpg', 'jpeg'], accept_multiple_files=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if uploaded_files is not None:
        # 重置 session state 變數
        if 'last_uploaded_files' not in st.session_state:
            st.session_state.last_uploaded_files = []
        if 'detect_annotations' not in st.session_state:
            st.session_state.detect_annotations = {}
        if 'infer_correct' not in st.session_state:
            st.session_state.infer_correct = False

        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            file_extension = file_name.split('.')[-1]

            if file_name not in st.session_state.last_uploaded_files:
                st.session_state.detect_annotations[file_name] = None
                st.session_state.last_uploaded_files.append(file_name)

            upload_success = st.success(f"檔案 {file_name} 已成功上傳！")
            if file_extension in image_format:
                st.image(uploaded_file)
            elif file_extension in video_format:    
                st.video(uploaded_file)
            else:
                st.warning(f"檔案 {file_name} 格式不支援！")
                continue

            # create dir of to save the input file and inference outcome
            base_name = file_name.split('.')[0]
            os.makedirs(f"inputFile/{base_name}", exist_ok=True)
            output_path = f"outputFile/{base_name}"
            os.makedirs(output_path, exist_ok=True)

            # copy the video to inputFile
            save_path = f"inputFile/{base_name}/{file_name}"
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            fps = cv2.VideoCapture(save_path).get(cv2.CAP_PROP_FPS)
            print("fps: ", fps)
                
            # close the success message    
            upload_success.empty()
            
            save_success = st.success(f"檔案已儲存至 {save_path}")
            save_success.empty()
        
        # 顯示「開始推理」按鈕
        if st.button("開始推理所有檔案"):
            st.session_state.infer_correct = True

        # Start to inference if not already done
        if st.session_state.infer_correct:
            for uploaded_file in uploaded_files:
                file_name = uploaded_file.name
                if st.session_state.detect_annotations[file_name] is None:
                    base_name = file_name.split('.')[0]
                    file_extension = file_name.split('.')[-1]
                    save_path = f"inputFile/{base_name}/{file_name}"
                    output_path = f"outputFile/{base_name}"
                    args = InitArgs(save_path, True, output_path, device)
                    model = initModel(args)
                    st.session_state.detect_annotations[file_name] = infer(args, model, base_name)
                    if file_extension in video_format:
                        st.video(os.path.join(output_path, base_name+".mp4"))
                        log_path = make_log(st.session_state.detect_annotations[file_name], fps, base_name)
                        st.success(f"偵測結果已儲存至 {log_path}")

            # 提供下載壓縮檔案的按鈕
            zip_path = zip_output_files()
            with open(zip_path, "rb") as f:
                st.download_button('下載所有預測檔案', f, 'output_files.zip')
            
            # 提供下載log檔案的按鈕
            log_zip_path = zip_log_files()
            with open(log_zip_path, "rb") as f:
                st.download_button('下載所有log檔案', f, 'log_files.zip')

            # 清理掉臨時檔案
    st.button("清理臨時檔案", on_click=lambda: cleanup_files())

def zip_output_files():
    zip_path = "output_files.zip"
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for root, dirs, files in os.walk("outputFile"):
            for file in files:
                zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), "outputFile"))
    return zip_path

def zip_log_files():
    log_zip_path = "log_files.zip"
    with zipfile.ZipFile(log_zip_path, 'w') as zipf:
        for root, dirs, files in os.walk("log"):
            for file in files:
                zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), "log"))
    return log_zip_path

def cleanup_files():
    try:
        shutil.rmtree("inputFile", ignore_errors=True)
        shutil.rmtree("outputFile", ignore_errors=True)
        shutil.rmtree("log", ignore_errors=True)
        os.remove("output_files.zip")
        os.remove("log_files.zip")
        st.success("成功清理所有臨時檔案.")
        st.session_state.last_uploaded_files = []
        st.session_state.detect_annotations = {}
        st.session_state.infer_correct = False
    except OSError as e:
        st.error(f"清除檔案失敗: {e}")
    
# '''
# Infer function : 
#     parameters: 
#         args:  paramerters for model initialize, including the path for 
#                input and output file and type of data
#         model: use for inference
#     function:
#         1. Inference the data from user input 
#         2. The interrupt button to stop the inference
#     To do:
#         If the video is too long maybe we can crop the video and than start to inference
# '''
def infer(args, model, name, format = 'video'):
    if 'infer_correct' not in st.session_state:
        st.session_state.infer_correct = False

    def toggle_infer():
        st.session_state.infer_correct = not st.session_state.infer_correct
    button_placeholder = st.empty()

    if not st.session_state.infer_correct:
        button_placeholder.button("開始 Inference", on_click=toggle_infer)

    detect_annotation = []
    if st.session_state.infer_correct:
        cap = cv2.VideoCapture(args.imfile)
        
        # get the fps, w, h, if the input video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # set output video type .mp4
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        output_video = cv2.VideoWriter(os.path.join(args.outputdir, name+".mp4"), fourcc, fps, (width, height))
        
        # Initialize the progress bar with the total number of frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = st.progress(0)  # Progress bar initialized at 0%
        
        # Create a button to interrupt the inference
        interrupt_button = st.button('中斷推理', key=name)
        is_interrupted = False
        
        if not cap.isOpened():
            print("cap can not open")
            exit()
        # Diplay inference result in real time
        frame_placeholder = st.empty()
        while cap.isOpened():
            
            ret, frame = cap.read()
            if not ret:
                print("Frame end or can not read frame")
                break
            if interrupt_button:
                is_interrupted = True
                st.warning("推理已中斷！")
                break  # Break the loop to stop the inference
                
            # change the graph type from bgr to rgb 
            im_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            w, h = im_pil.size
            orig_size = torch.tensor([w, h])[None].to(args.device)
        
            # Resize the graph and change to tensor type to inference
            transforms = T.Compose([
                T.Resize((640, 640)),  
                T.ToTensor(),
            ])
            im_data = transforms(im_pil)[None].to(args.device)
                
            output = model(im_data, orig_size)
            labels, boxes, scores = output
                
            detect_frame, box_count = draw([im_pil], labels, boxes, scores, 0.5)
            
            frame_out = cv2.cvtColor(np.array(detect_frame), cv2.COLOR_RGB2BGR)
            # Display inference result
            frame_placeholder.image(frame_out, channels="BGR", use_container_width=True)
            output_video.write(frame_out)
            
            # Update the progress bar
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            progress = current_frame / total_frames
            progress_bar.progress(progress)  # Update progress bar
            print(f"Progress: {current_frame} / {total_frames}")

            # Collect the frame that is detected
            if  box_count > 0:
                detect_annotation.append(current_frame)

                
        # close all the windows
        cap.release()
        output_video.release()
        cv2.destroyAllWindows()
        if is_interrupted:
            st.info("推理過程已停止。")
        else:
            st.success("推理完成")
        return detect_annotation
         
def make_log(detect_annotation, fps, file_name):
    os.makedirs("log", exist_ok=True)
    log_path = os.path.join("log", file_name + '.txt')
    with open(log_path, 'w') as f:
        f.write("偵測到的人員在以下秒數：\n")
        for i in detect_annotation:
            f.write(f"{i/fps:.2f}秒\n")
            
    return log_path

if __name__ == '__main__':
    st.sidebar.title('選用預測方法')
    page = st.sidebar.selectbox("選擇預測方法", ("模型偵測系統", "異常偵測系統"))
    if page == "模型偵測系統":
        main()
    else:
        anomaly_main()
    
