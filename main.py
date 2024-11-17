import streamlit as st
import os
import torch
import torch.nn as nn 
import torchvision.transforms as T
from torch.cuda.amp import autocast
import numpy as np 
from PIL import Image, ImageDraw, ImageFont
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import cv2
import time
import argparse
from tqdm import tqdm
from rtdetr.tools.infer import InitArgs, draw, initModel

video_format = ['mp4', 'mov', 'avi', 'MP4', 'MOV', 'AVI']
image_format = ['png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG']

def main():
    st.title('無人機人員偵測系統')
    
    # upload the file
    uploaded_file = st.file_uploader('上傳圖片/影像', type=['mp4', 'mov', 'avi', 'png', 'jpg', 'jpeg'])

    try:
        file_extension = uploaded_file.name.split('.')[-1]
        print(file_extension)
    except:
        pass

    choose_device = st.selectbox("選擇預測使用裝置", ["CPU", "GPU"])
    if choose_device == "CPU":
        device = "cpu"
    else:
        device = "cuda:0"
    
    if uploaded_file is not None:
        
        # 重置 session state 變數
        if 'last_uploaded_file' not in st.session_state:
            st.session_state.last_uploaded_file = None
        if st.session_state.last_uploaded_file != uploaded_file.name:
            st.session_state.detect_annotation = None
            st.session_state.last_uploaded_file = uploaded_file.name
            st.session_state.infer_correct = False  # 重置 infer_correct 變數

        upload_success = st.success("檔案已成功上傳！")
        if file_extension in image_format:
            st.image(uploaded_file)
        elif file_extension in video_format:    
            st.video(uploaded_file)
        else:
            st.warning("檔案格式不支援！")
            return

        # create dir of to save the input file and inference outcome
        file_name = uploaded_file.name.split('.')[0]
        os.makedirs(f"inputFile/{file_name}", exist_ok=True)
        output_path = f"outputFile/{file_name}"
        os.makedirs(output_path, exist_ok=True)

        # copy the video to inputFile
        save_path = f"inputFile/{file_name}/{uploaded_file.name}"
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        fps = cv2.VideoCapture(save_path).get(cv2.CAP_PROP_FPS)
        print("fps: ",fps)
            
        # Create the parser for inference and init the model
        # The reason I put the initialize and model here \
        # is the time for inferecne will be shorter
        args = InitArgs(save_path, True, output_path, device)
        model = initModel(args)
        # close the success message    
        upload_success.empty()
        
        save_success = st.success(f"檔案已儲存至 {save_path}")
        save_success.empty()
        
        detect_annotation = None
        # Initialize session state for inference results
        if 'detect_annotation' not in st.session_state:
            st.session_state.detect_annotation = None
        
        # Start to inference if not already done
        if st.session_state.detect_annotation is None:
            st.session_state.detect_annotation = infer(args, model)
        
        detect_annotation = st.session_state.detect_annotation

        if detect_annotation is not None:
            output_path = f"outputFile/{file_name}/output.mp4"
            st.text("推理結果")
            st.video(output_path)
            with open(output_path, "rb") as f:
                st.download_button('下載預測檔案', f, os.path.splitext(file_name)[0] + '.mp4')
            if file_extension in video_format:
                make_log(detect_annotation, fps, file_name)
                with open(os.path.join("log", file_name + '.txt'), "rb") as f:
                    st.download_button('下載偵測時間', f, file_name+'.txt')
            
        else:
            pass
            
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
def infer(args, model):
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
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

        output_video = cv2.VideoWriter(os.path.join(args.outputdir,"output.mp4"), fourcc, fps, (width, height))
        
        # Initialize the progress bar with the total number of frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = st.progress(0)  # Progress bar initialized at 0%
        
        # Create a button to interrupt the inference
        interrupt_button = st.button('中斷推理')  
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
            
            
if __name__ == '__main__':
    main()
    
